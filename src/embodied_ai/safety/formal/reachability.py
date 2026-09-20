"""可达集分析 — 基于区间运算的安全包络验证。

核心思想: 给定当前状态 x(t) 和不确定性 Δ, 计算未来 T 秒所有可达状态 X_reachable(t+τ)。
若 X_reachable(t+τ) ∩ X_unsafe = ∅, 则安全; 否则触发 STO。

简化实现 (中心 + 半径区间):
  X_reachable(t+τ) = ẋ(τ) ⊕ Δ(τ)  (Minkowski 和)
  ẋ(τ) = x₀ + ∫₀ᵗ f(x,u) dτ  (中心轨迹, Euler 积分)
  Δ(τ) = Δ₀ + ∫₀ᵗ J(x,u) Δ dτ  (不确定度传播, Jacobian 积分)

检验: 对于每个可达集, 检查是否与以下集合相交:
  - 关节角度限制: θ ∈ [θ_min, θ_max]
  - 速度限制: ω ∈ [-ω_max, ω_max]
  - 力矩限制: τ ∈ [-τ_max, τ_max]
  - 障碍距离: d(x,d) > d_safe
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class IntervalSet:
    """仿射区间集合: center + radii (对角不确定性)。"""
    center: np.ndarray           # [n]
    radii: np.ndarray            # [n] (非负)

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.radii = np.asarray(self.radii, dtype=float)
        assert self.radii.shape == self.center.shape

    @property
    def lower(self) -> np.ndarray:
        return self.center - self.radii

    @property
    def upper(self) -> np.ndarray:
        return self.center + self.radii

    def intersects(self, other: "IntervalSet") -> bool:
        """两区间是否相交 (各维度有重叠)。"""
        return bool(np.all(self.lower <= other.upper) and np.all(self.upper >= other.lower))

    def subtract_radius(self, r: float) -> "IntervalSet":
        """收缩半径 (安全裕量)。"""
        return IntervalSet(self.center, np.maximum(self.radii - r, 0.0))

    def contains(self, point: np.ndarray) -> bool:
        """点是否在集合内。"""
        return bool(np.all(point >= self.lower) and np.all(point <= self.upper))

    def split_around(self, delta: np.ndarray) -> "IntervalSet":
        """从中心扩展 δ 后的集合。"""
        return IntervalSet(self.center, self.radii + np.abs(delta))


@dataclass
class SafetyEnvelope:
    """安全包络 — 定义关节/力矩/障碍的硬限制。"""
    name: str
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    # 可选: 障碍物距离约束 (仅当 lower_bound[0] 是障碍距离时启用)
    obstacle_distance_idx: Optional[int] = None
    safe_distance: float = 0.0


class ReachabilityAnalyzer:
    """区间可达集分析器。

    简化的刚体动力学前向积分:
      x' = f(x, u)  (Euler, dt 可配)
      Δ' = |J(x, u)| · Δ + ε_extr  (不确定度传播, J 为 Jacobian 上界估计)
    """

    def __init__(self, state_dim: int, dt: float = 0.001):
        self.state_dim = state_dim
        self.dt = dt

    def forward_reachable(
        self,
        x0: np.ndarray,
        delta0: np.ndarray,
        control: np.ndarray,
        dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
        jacobian_bound: Callable[[np.ndarray, np.ndarray], np.ndarray],
        horizon_steps: int,
        extra_uncertainty: float = 0.0,
    ) -> List[IntervalSet]:
        """前向传播 interval reachable set。

        Args:
            x0: 初始状态 [state_dim]
            delta0: 初始不确定度 [state_dim]
            control: 控制输入 (假设 horizon 内恒定, 简化)
            dynamics: f(x,u) → x_dot
            jacobian_bound: J(x,u) → |∂f/∂x| 上界矩阵 [state_dim, state_dim]
            horizon_steps: 前向步数
            extra_uncertainty: 额外不确定度累积常数
        Returns:
            各时间步的可达集列表 (含 t=0)
        """
        x = np.asarray(x0, dtype=float).copy()
        delta = np.asarray(delta0, dtype=float).copy()
        reachable = [IntervalSet(x.copy(), delta.copy())]
        for _ in range(horizon_steps):
            # 中心前向积分
            x_dot = dynamics(x, control)
            x = x + x_dot * self.dt
            # 不确定度传播: Δ' = (|J|·Δ + ε) · dt
            J = jacobian_bound(x, control)
            delta = (J @ delta + extra_uncertainty) * self.dt + delta
            delta = np.maximum(delta, 0.0)  # 保持非负
            reachable.append(IntervalSet(x.copy(), delta.copy()))
        return reachable

    def check_safety(
        self,
        reachable_sequence: List[IntervalSet],
        envelope: SafetyEnvelope,
    ) -> Tuple[bool, int, Optional[IntervalSet]]:
        """检查可达序列是否始终在安全包络内。

        Returns:
            (is_safe, violation_step, violation_set)
            is_safe = True 表示全部在包络内
            violation_step 是首次违反的时间步 (-1 表示无违反)
        """
        for step, rset in enumerate(reachable_sequence):
            if np.any(rset.lower < envelope.lower_bound) or np.any(rset.upper > envelope.upper_bound):
                return False, step, rset
        return True, -1, None

    def compute_min_stop_distance(
        self,
        x0: np.ndarray,
        delta0: np.ndarray,
        max_decel: float,
        dynamics: Callable,
        jacobian_bound: Callable,
        horizon_steps: int,
    ) -> float:
        """计算: 以 max_decel 开始减速, 到 stop 时质心移动的距离。

        使用 speed state index=0 作为速度 (简化假设: v 是第一个状态分量)。
        """
        v0 = x0[0]
        dist = 0.0
        v = v0
        for _ in range(horizon_steps):
            if v <= 0:
                break
            dist += v * self.dt
            v -= max_decel * self.dt
        return dist

    def stop_time_budget_met(
        self,
        x0: np.ndarray,
        delta0: np.ndarray,
        max_decel: float,
        dynamics: Callable,
        jacobian_bound: Callable,
        budget_steps: int,
    ) -> bool:
        """检查是否在 budget_steps 内停到零速度。"""
        v = x0[0] + delta0[0]  # 考虑最坏情况
        t_steps = 0
        while v > 0.01 and t_steps < budget_steps:
            v -= max_decel * self.dt
            t_steps += 1
        return v <= 0.01

    def check_stop_budget(self, current_speed: float, budget_steps: int, max_decel: float) -> tuple:
        """简化入口: 检查 speed 在 budget_steps 内能否停到零 (不需要完整动力学)。"""
        v = current_speed
        t_steps = 0
        while v > 0.01 and t_steps < budget_steps:
            v -= max_decel * self.dt
            t_steps += 1
        return v <= 0.01, max(current_speed * t_steps * self.dt, 0.0)
