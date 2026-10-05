"""域随机化 (Domain Randomization) 仪器 — Sim-to-Real 迁移第一道防线。

根据 contracts/dr_params/manufacturing_variance.yaml 的制造差异分布 + 物理参数分布,
在训练时随机采样仿真参数, 使策略对参数变化鲁棒, 缩小 Sim-to-Real 差距。

DR 参数类别:
  1. 动力学参数: 质量、惯量、摩擦、阻尼
  2. 执行器参数: 齿轮间隙、力矩常数、响应延迟
  3. 传感器噪声: 编码器量化、IMU 漂移、相机曝光
  4. 环境参数: 光照、纹理、桌面摩擦

用法:
  dr = DomainRandomizer(DRConfig())
  for episode in range(N):
      dr_params = dr.sample_params()     # 随机采样一组参数
      env.set_dr_params(dr_params)        # 应用到环境
      obs = env.reset()
      ...
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DRParamSpec:
    """单个 DR 参数规格。"""
    name: str
    distribution: str          # "uniform" | "normal" | "loguniform" | "choice"
    low: float = 0.0           # uniform/loguniform 下界
    high: float = 1.0          # uniform/loguniform 上界
    mean: float = 0.0          # normal 均值
    std: float = 1.0           # normal 标准差
    choices: Optional[list] = None  # choice 的候选值
    relative: bool = False     # 是否相对名义值 (乘性) 而非加性
    nominal: float = 1.0       # 名义值 (relative=True 时用)

    def sample(self) -> float:
        if self.distribution == "uniform":
            v = np.random.uniform(self.low, self.high)
        elif self.distribution == "loguniform":
            v = np.exp(np.random.uniform(np.log(self.low), np.log(self.high)))
        elif self.distribution == "normal":
            v = np.random.normal(self.mean, self.std)
        elif self.distribution == "choice":
            assert self.choices is not None
            v = random.choice(self.choices)
        else:
            raise ValueError(f"未知分布: {self.distribution}")

        if self.relative:
            return v * self.nominal
        return v


@dataclass
class DRConfig:
    """域随机化配置 — 控制随机化范围和类别开关。"""
    # 类别开关
    enable_dynamics: bool = True     # 动力学参数
    enable_actuator: bool = True     # 执行器参数
    enable_sensor: bool = True       # 传感器噪声
    enable_visual: bool = True       # 视觉外观
    enable_environment: bool = True  # 环境光照/纹理

    # 随机化强度 (0=关闭, 1=全量)
    dynamics_scale: float = 1.0
    actuator_scale: float = 1.0
    sensor_scale: float = 1.0
    visual_scale: float = 1.0
    environment_scale: float = 1.0

    # 增量模式: 随训练进度逐步增大随机化范围
    curriculum_enabled: bool = False
    curriculum_start: float = 0.1    # 初始强度
    curriculum_end: float = 1.0      # 最终强度
    curriculum_total_episodes: int = 1000

    # resetti con seed
    seed: Optional[int] = None

    # 自定义参数规格 (追加到默认集合)
    extra_specs: List[DRParamSpec] = field(default_factory=list)


class DomainRandomizer:
    """域随机化引擎 — 采样 + 应用 + 课程学习调度。

    与 contracts/dr_params/manufacturing_variance.yaml 对齐:
    制造差异驱动的 DR 分布直接从制造数据 (ICT/FCT/遥测) 更新。
    """

    # 默认 DR 参数规格 (基于典型 7-DOF 机械臂)
    DEFAULT_DYNAMICS_SPECS: List[Dict[str, Any]] = [
        # (名称, 分布, 低, 高, 相对模式, 名义值)  — 质量 ±20%
        dict(name="link_mass", distribution="uniform", low=0.8, high=1.2, relative=True, nominal=1.0),
        # 摩擦系数 ±50%
        dict(name="joint_friction", distribution="uniform", low=0.5, high=1.5, relative=True, nominal=0.01),
        # 关节阻尼 ±30%
        dict(name="joint_damping", distribution="uniform", low=0.7, high=1.3, relative=True, nominal=0.1),
        # 质心偏移 ±5mm
        dict(name="com_offset", distribution="uniform", low=0.0, high=0.005),
    ]

    DEFAULT_ACTUATOR_SPECS: List[Dict[str, Any]] = [
        # 力矩常数 ±10%
        dict(name="torque_constant", distribution="uniform", low=0.9, high=1.1, relative=True, nominal=1.0),
        # 齿轮间隙 ±50%
        dict(name="backlash", distribution="loguniform", low=1e-5, high=1e-3),
        # 响应延迟 0-5ms
        dict(name="actuator_delay_ms", distribution="uniform", low=0.0, high=5.0),
        # 电机电阻 ±15% (温度效应)
        dict(name="motor_resistance", distribution="uniform", low=0.85, high=1.15, relative=True, nominal=1.0),
    ]

    DEFAULT_SENSOR_SPECS: List[Dict[str, Any]] = [
        # 编码器量化噪声 ±0.5 分辨率
        dict(name="encoder_noise_deg", distribution="uniform", low=0.0, high=0.01),
        # IMU 加速度偏置 ±0.1 m/s²
        dict(name="imu_bias", distribution="uniform", low=-0.1, high=0.1),
        # 相机曝光 ±20%
        dict(name="exposure", distribution="uniform", low=0.8, high=1.2, relative=True, nominal=1.0),
        # CAN 通信延迟 0-2ms (对齐 dr_params manufacturing_variance)
        dict(name="can_latency_ms", distribution="uniform", low=0.0, high=2.0),
    ]

    DEFAULT_VISUAL_SPECS: List[Dict[str, Any]] = [
        # 光照强度 ±30%
        dict(name="light_intensity", distribution="uniform", low=0.7, high=1.3, relative=True, nominal=1.0),
        # 桌面纹理随机选择
        dict(name="table_texture", distribution="choice",
             choices=["wood_white", "wood_dark", "marble", "metal", "plastic_gray"]),
        # 背景颜色 ±10% RGB
        dict(name="bg_color_jitter", distribution="uniform", low=-0.1, high=0.1),
    ]

    DEFAULT_ENV_SPECS: List[Dict[str, Any]] = [
        # 桌面摩擦系数 ±40%
        dict(name="table_friction", distribution="uniform", low=0.6, high=1.4, relative=True, nominal=0.8),
        # 物体重力 ±2%
        dict(name="gravity", distribution="uniform", low=0.98, high=1.02, relative=True, nominal=9.81),
        # 物体位置 ±3cm
        dict(name="object_position_jitter", distribution="uniform", low=0.0, high=0.03),
    ]

    def __init__(self, config: DRConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)

        # 构建参数规格列表
        self._specs: Dict[str, DRParamSpec] = {}
        self._build_specs()

        # 课程学习状态
        self._episode_counter = 0
        # 最近一次 load_manufacturing_dr 实际应用条数 (0 = 未生效)
        self.last_load_applied = 0

    def _build_specs(self) -> None:
        """根据 config 开关构建参数规格集合。"""
        cfg = self.config
        spec_groups = [
            (cfg.enable_dynamics, cfg.dynamics_scale, self.DEFAULT_DYNAMICS_SPECS),
            (cfg.enable_actuator, cfg.actuator_scale, self.DEFAULT_ACTUATOR_SPECS),
            (cfg.enable_sensor, cfg.sensor_scale, self.DEFAULT_SENSOR_SPECS),
            (cfg.enable_visual, cfg.visual_scale, self.DEFAULT_VISUAL_SPECS),
            (cfg.enable_environment, cfg.environment_scale, self.DEFAULT_ENV_SPECS),
        ]

        for enabled, scale, defaults in spec_groups:
            if not enabled:
                continue
            for d in defaults:
                spec = DRParamSpec(**d)
                if scale != 1.0:
                    spec = self._scale_spec(spec, scale)
                self._specs[spec.name] = spec

        # 追加自定义规格
        for spec in cfg.extra_specs:
            self._specs[spec.name] = spec

    def _scale_spec(self, spec: DRParamSpec, scale: float) -> DRParamSpec:
        """缩放到随机化范围 (scale=0 → 退化为名义值)."""
        import copy
        s = copy.deepcopy(spec)
        if spec.distribution == "uniform":
            center = (spec.low + spec.high) / 2
            half_range = (spec.high - spec.low) / 2 * scale
            s.low = center - half_range
            s.high = center + half_range
        elif spec.distribution == "normal":
            s.std = spec.std * scale
        elif spec.distribution == "loguniform":
            log_mid = (np.log(spec.low) + np.log(spec.high)) / 2
            log_half = (np.log(spec.high) - np.log(spec.low)) / 2 * scale
            s.low = np.exp(log_mid - log_half)
            s.high = np.exp(log_mid + log_half)
        return s

    def sample_params(self) -> Dict[str, float]:
        """采样一组 DR 参数, 直接传给 env.set_dr_params()."""
        # 课程学习: 当前强度
        if self.config.curriculum_enabled:
            progress = min(self._episode_counter / max(self.config.curriculum_total_episodes, 1), 1.0)
            current_scale = self.config.curriculum_start + (
                self.config.curriculum_end - self.config.curriculum_start) * progress
            # 临时缩放所有参数的采样范围
            params = {}
            for name, spec in self._specs.items():
                base_spec = self._scale_spec(spec, current_scale)
                params[name] = base_spec.sample()
            self._episode_counter += 1
            return params
        else:
            return {name: spec.sample() for name, spec in self._specs.items()}

    def nominal_params(self) -> Dict[str, float]:
        """返回名义参数 (无随机化), 用于 baseline 对比。"""
        params = {}
        for name, spec in self._specs.items():
            if spec.distribution == "uniform":
                params[name] = (spec.low + spec.high) / 2
            elif spec.distribution == "normal":
                params[name] = spec.mean
            elif spec.distribution == "loguniform":
                params[name] = np.exp((np.log(spec.low) + np.log(spec.high)) / 2)
            elif spec.distribution == "choice":
                params[name] = spec.choices[len(spec.choices) // 2] if spec.choices else 0
            else:
                params[name] = spec.nominal
        return params

    def load_manufacturing_dr(self, path: str) -> int:
        """从制造差异分布 (contracts/dr_params/manufacturing_variance.yaml) 加载 DR 参数。

        该 yaml 由 EdalabAdapter.generate_dr_config 自动制造数据推断。

        v0.5: 原实现三处静默 `return` (无 PyYAML / 文件缺失 / 键缺失) 让调用方无法
        区分「没加载」与「加载了 0 项」, 飞轮闭环断了也没人知道。改为统一返回
        实际应用的参数个数, 并在 self.last_load_applied 留痕。
        """
        self.last_load_applied = 0
        try:
            import yaml
        except ImportError:
            return 0

        p = Path(path)
        if not p.exists():
            return 0

        with open(p, encoding="utf-8") as f:
            dr_data = yaml.safe_load(f)

        if not dr_data or not isinstance(dr_data, dict) or "dr_params" not in dr_data:
            return 0

        applied = 0
        for pname, pdata in dr_data["dr_params"].items():
            if not isinstance(pdata, dict):
                continue
            scale_factor = pdata.get("range_scale", 1.0)
            dist = pdata.get("distribution", "uniform")

            # v0.5: 制造侧新增的参数 (默认规格里没有) 也应当被采纳, 否则飞轮闭环
            # 产出的新参数永远进不了 DR 采样集合。
            if dist == "uniform" and ("low" not in pdata or "high" not in pdata):
                continue

            if dist == "normal":
                self._specs[pname] = DRParamSpec(
                    name=pname, distribution="normal",
                    mean=pdata.get("mean", 0.0), std=pdata.get("std", 1.0) * scale_factor,
                    relative=pdata.get("relative", False), nominal=pdata.get("nominal", 1.0))
            elif dist == "uniform":
                half = (pdata["high"] - pdata["low"]) / 2 * scale_factor
                mid = (pdata["high"] + pdata["low"]) / 2
                self._specs[pname] = DRParamSpec(
                    name=pname, distribution="uniform",
                    low=mid - half, high=mid + half,
                    relative=pdata.get("relative", False), nominal=pdata.get("nominal", 1.0))
            else:
                continue
            applied += 1

        self.last_load_applied = applied
        return applied

    @property
    def param_names(self) -> List[str]:
        return list(self._specs.keys())

    def __len__(self) -> int:
        return len(self._specs)
