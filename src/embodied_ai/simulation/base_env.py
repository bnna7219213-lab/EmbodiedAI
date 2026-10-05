"""仿真环境抽象基类 — 对齐 Gymnasium API, 统一仿真/真机接口。

设计原则:
  - 仿真和真机实现同一个接口, RL 训练代码无需关心底层
  - 支持多视角 RGB + 本体感受 + 触觉 (可扩展)
  - 自动记录 episode 数据 (用于行为克隆数据采集)

用法:
  env = MuJoCoEnv(config)  # 或 PyBulletEnv / RealRobotEnv
  obs = env.reset()
  while not done:
      action = policy(obs)
      obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EnvConfig:
    """仿真环境配置 (对齐 contracts/hw_requirements.yaml)。"""
    # 时间
    dt: float = 0.001              # 仿真步长 (1ms 对齐 FOC 控制)
    control_freq: int = 20         # 控制频率 (20Hz 对齐 VLA 输出)
    render_freq: int = 30          # 渲染频率 (30Hz)

    # 观测
    state_dim: int = 12            # 本体感受维度 (关节角度+速度)
    image_size: Tuple[int, int] = (224, 224)
    num_views: int = 1             # 相机视角数
    image_channels: int = 3        # RGB

    # 动作
    action_dim: int = 12           # 关节数
    action_horizon: int = 10       # 预测步数
    action_lower: Optional[np.ndarray] = None  # 关节下限 (rad)
    action_upper: Optional[np.ndarray] = None  # 关节上限 (rad)

    # 场景
    scene_name: str = "table_pick"  # 场景名称
    max_episode_steps: int = 500   # 单回合最大步数

    # 噪声 (Sim-to-Real 参数)
    obs_noise_std: float = 0.0     # 观测噪声标准差
    action_noise_std: float = 0.0  # 动作噪声标准差
    latency_ms: float = 0.0        # 执行延迟 (ms)

    # 随机化
    enable_dr: bool = False        # 是否启用域随机化
    dr_config_path: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """单步返回 (对齐 Gymnasium)。"""
    observation: Dict[str, Any]   # {"state": ..., "images": ..., "tactile": ...}
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]


class SimEnv(ABC):
    """仿真环境抽象基类。

    子类必须实现:
      - reset() → observation
      - step(action) → StepResult
      - close()
    可选实现:
      - render() → RGB array
      - set_dr_params(params: dict)
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self.step_count = 0
        self.episode_count = 0
        self._episode_data: List[Dict[str, Any]] = []

        # 动作边界
        if config.action_lower is None:
            config.action_lower = np.full(config.action_dim, -np.pi)
        if config.action_upper is None:
            config.action_upper = np.full(config.action_dim, np.pi)

        # 控制周期 (仿真步数/控制步)
        self._control_interval = max(1, int(1.0 / (config.dt * config.control_freq)))

    # ------------------------------------------------------------------ #
    # 抽象接口
    # ------------------------------------------------------------------ #
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """重置环境, 返回初始观测。"""

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """执行一步: action [action_dim] → StepResult。"""

    @abstractmethod
    def close(self) -> None:
        """关闭环境, 释放资源。"""

    # ------------------------------------------------------------------ #
    # 通用工具方法
    # ------------------------------------------------------------------ #
    def sample_action(self) -> np.ndarray:
        """随机合法动作 (冒烟/随机策略基线)。子类可覆盖为更贴合场景的采样。"""
        return np.random.uniform(self.config.action_lower, self.config.action_upper).astype(float)

    def reward_terms(self) -> Dict[str, float]:
        """可观测的奖励分解 (对齐可解释性要求)。默认返回空, 子类覆盖。"""
        return {}
    def _build_observation(self, state: np.ndarray,
                           images: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """构建标准观测字典 (含噪声注入)。"""
        obs = {"state": state.copy()}
        if images is not None:
            obs["images"] = images.copy()

        # 观测噪声
        if self.config.obs_noise_std > 0:
            obs["state"] = obs["state"] + np.random.normal(
                0, self.config.obs_noise_std, size=state.shape)
        return obs

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """动作裁剪 + 噪声注入 + 延迟模拟。"""
        # 裁剪到关节极限
        a = np.clip(action, self.config.action_lower, self.config.action_upper)
        # 执行器噪声
        if self.config.action_noise_std > 0:
            a = a + np.random.normal(0, self.config.action_noise_std, size=a.shape)
        return a

    def record_step(self, obs: Dict, action: np.ndarray, reward: float,
                    next_obs: Dict, done: bool) -> None:
        """记录一步到 episode buffer (用于 BC 数据采集)。"""
        self._episode_data.append({
            "obs": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()},
            "action": action.copy(),
            "reward": reward,
            "done": done,
            "step": self.step_count,
        })

    def export_episode(self, path: Optional[str] = None) -> Dict[str, Any]:
        """导出当前 episode 为训练格式 (对齐 plan.md 数据格式)。"""
        episode = {
            "length": len(self._episode_data),
            "total_reward": sum(d["reward"] for d in self._episode_data),
            "observations": [d["obs"] for d in self._episode_data],
            "actions": [d["action"] for d in self._episode_data],
            "success": self._episode_data[-1]["done"] if self._episode_data else False,
            "env_config": {
                "dt": self.config.dt,
                "scene": self.config.scene_name,
                "dr": self.config.enable_dr,
            },
        }
        if path:
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(episode, f, default=str, ensure_ascii=False)
        return episode

    def clear_episode(self) -> None:
        """清空当前 episode buffer。"""
        self._episode_data = []

    @property
    def action_space(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.config.action_lower, self.config.action_upper

    @property
    def obs_dim(self) -> int:
        return self.config.state_dim

    @property
    def action_dim(self) -> int:
        return self.config.action_dim
