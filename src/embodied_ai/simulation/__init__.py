"""仿真环境核心模块 — 参考环境 / MuJoCo / PyBullet + 域随机化 + Sim-to-Real。

提供具身智能体训练所需的仿真闭环:
  - SimEnv:            仿真环境抽象基类 (对齐 Gymnasium API)
  - ReferenceEnv:      ★ v0.5 纯 numpy 参考环境 (零外部依赖, 默认可用)
  - MuJoCoEnv:         MuJoCo 包装器 (速度快, 接触模型准; 需 mujoco)
  - PyBulletEnv:       PyBullet 包装器 (多体动力学, 易扩展; 需 pybullet)
  - DomainRandomizer:  域随机化引擎 (物理参数/视觉/执行器) + 课程学习
  - SimToRealPipeline: 迁移管线 (DR → 差距量化 → 诊断建议)
  - make_env / available_backends: 后端自动选择 (v0.5)

后端选择策略 (make_env):
  backend="auto"       → 优先 mujoco, 其次 pybullet, 都没有则回退 reference
  backend="reference"  → 强制纯 numpy 参考环境 (CI / 无 GPU / 无外部依赖)
  backend="mujoco"|"pybullet" → 强制指定, 缺失直接抛 ImportError (不静默降级)

Phase A 核心交付: 让仿真中的 RL 策略能迁移到真机。

依赖:
  - 强依赖: numpy
  - 可选: mujoco / pybullet (缺失时由 make_env 自动回退, 不阻断管线)
"""

from __future__ import annotations

from typing import List

from .base_env import EnvConfig, SimEnv, StepResult
from .domain_randomization import DRConfig, DRParamSpec, DomainRandomizer
from .reference_env import PointMassEnv, ReferenceEnv
from .sim_to_real import SimToRealPipeline, SimToRealResult

__all__ = [
    "SimEnv", "EnvConfig", "StepResult",
    "DomainRandomizer", "DRConfig", "DRParamSpec",
    "ReferenceEnv", "PointMassEnv",
    "SimToRealPipeline", "SimToRealResult",
    "make_env", "available_backends", "BACKENDS",
]

# 后端注册表: 名字 → (模块, 类名)。延迟导入避免可选依赖拖慢 import。
_BACKEND_REGISTRY = {
    "reference": (".reference_env", "ReferenceEnv"),
    "mujoco": (".mujoco_env", "MuJoCoEnv"),
    "pybullet": (".pybullet_env", "PyBulletEnv"),
}

# auto 模式下的优先级
_AUTO_PREFERENCE = ("mujoco", "pybullet", "reference")


def _probe(backend: str):
    """尝试导入后端类, 返回类对象; 不可用时返回 None。

    判定顺序: 模块可导入 → 类有 is_available() 时必须返回 True (MuJoCo/PyBullet
    的依赖在 __init__ 里才 import, 仅靠模块导入会误判为可用)。
    """
    module_path, cls_name = _BACKEND_REGISTRY[backend]
    try:
        import importlib
        mod = importlib.import_module(module_path, package=__name__)
        cls = getattr(mod, cls_name)
    except Exception:  # noqa: BLE001 — 模块缺失 / 依赖缺失都视为不可用
        return None
    check = getattr(cls, "is_available", None)
    if callable(check):
        try:
            if not check():
                return None
        except Exception:  # noqa: BLE001
            return None
    return cls


def available_backends() -> List[str]:
    """当前环境下真实可用的后端名列表 (reference 恒可用)。"""
    return [b for b in _BACKEND_REGISTRY if _probe(b) is not None]


BACKENDS = list(_BACKEND_REGISTRY)


def make_env(config: EnvConfig, backend: str = "auto", real: bool = False, **kwargs) -> SimEnv:
    """按后端名创建环境。

    Args:
        config:  EnvConfig
        backend: auto|reference|mujoco|pybullet
        real:    是否按"真机侧"语义构造 (目前 ReferenceEnv 支持, 用于 sim2real 演示)
        **kwargs: 透传给具体环境构造函数
    """
    if backend == "auto":
        for name in _AUTO_PREFERENCE:
            cls = _probe(name)
            if cls is not None:
                return _instantiate(cls, config, real, kwargs)
        raise RuntimeError("无可用仿真后端 (reference 应恒可用, 请检查安装)")

    if backend not in _BACKEND_REGISTRY:
        raise ValueError(f"未知后端: {backend}. 合法: auto|{ '|'.join(_BACKEND_REGISTRY) }")

    cls = _probe(backend)
    if cls is None:
        raise ImportError(
            f"后端 {backend} 不可用 (依赖缺失). 可用后端: {available_backends()}; "
            f"或用 backend='reference' 走零依赖参考环境")
    return _instantiate(cls, config, real, kwargs)


def _instantiate(cls, config: EnvConfig, real: bool, kwargs: dict) -> SimEnv:
    """构造环境: ReferenceEnv 接受 real 参数, 其余后端忽略。"""
    if cls is ReferenceEnv:
        return cls(config, real=real, **kwargs)
    return cls(config, **kwargs)
