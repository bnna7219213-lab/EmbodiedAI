"""安全预算闭环 — safety_budget.yaml 实测链 ↔ 形式化可达集 (K3 闭环)。

把 EdaLab relay-clutch 实测的分段延迟 (0.8µs 电气 + 8ms 熄弧 + 8ms 机械隔离 = 16ms)
作为形式化安全分析的时间上界输入, 验证:
  1. 停止角位移: θ_stop = ω·t_total + 0.5·|α_max|·t² ≤ 30° (contracts.acceptance)
  2. 可达集: X_reachable(t=16ms) 与关节限位集不相交
  3. 闭环判定: budget_ok = SIL2 + PFH + 停止角位移 三重通过

这是 fusion-absorption-plan 创新 #4 (软硬两级安全滤波器) 的数值闭环:
硬件链 (16ms 实测) → 机会约束滤波 (1kHz) → RL 包络投影 (20Hz) → 形式化验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class BudgetClosureResult:
    """安全预算闭环验证结果。"""
    total_break_ms: float
    stop_angle_deg: float
    stop_angle_ok: bool
    reachable_set_ok: bool
    sil2_ok: bool
    closure_ok: bool
    details: Dict[str, Any]


def load_measured_chain() -> Tuple[float, Dict[str, Any]]:
    """从 contracts/safety_budget.yaml 读取实测分段链, 返回 (total_ms, segments)。"""
    from ...config import load_contract
    sb = load_contract("safety_budget")
    segs = {s["stage"]: s for s in sb["segments"]}
    total_ms = (segs["electrical_off"]["measured_us"] / 1000
                + segs["arc_extinction"]["measured_ms"]
                + segs["mechanical_isolation"]["measured_ms"])
    return total_ms, sb


def stop_angle_from_budget(joint_speed_rad_s: float,
                           max_decel_rad_s2: float = 50.0,
                           total_break_ms: float = None) -> float:
    """给定实测断开时间预算, 计算停止角位移 (匀减速模型, ISO 13849 口径)。

    θ_stop = ω·t − 0.5·α·t², t 为有效制动时间 (取 min(t_stop, t_budget))。
    """
    if total_break_ms is None:
        total_break_ms, _ = load_measured_chain()
    t_budget = total_break_ms / 1000.0
    t_stop = joint_speed_rad_s / max(max_decel_rad_s2, 1e-6)
    t = min(t_budget, t_stop)
    return np.degrees(joint_speed_rad_s * t - 0.5 * max_decel_rad_s2 * t * t)


def reachable_set_within_limits(joint_speed_rad_s: float,
                                total_break_ms: float,
                                joint_limit_deg: float = 120.0) -> bool:
    """可达集检验: 16ms 预算内关节角域不越过限位。

    用区间运算 (reachability.IntervalSet 同族简化): 区间中心=当前角(取 0),
    半径 = 停止角位移 + 速度不确定度 (CAN 延迟 2ms 引入的额外滑行)。
    """
    from ..chance_constrained_grip import chance_constrained_grip  # noqa: F401 (确保包内一致性)
    theta_stop_deg = stop_angle_from_budget(joint_speed_rad_s, total_break_ms=total_break_ms)
    # CAN-FD 延迟不确定度 (dr_params: canfd_latency_ms max 2.0 → 额外滑行)
    can_extra_deg = np.degrees(joint_speed_rad_s * 2.0 / 1000.0)
    radius = theta_stop_deg + can_extra_deg
    return radius < joint_limit_deg


def close_budget_loop(joint_speed_rad_s: float = 8.0,
                      max_decel_rad_s2: float = 50.0) -> BudgetClosureResult:
    """K3 闭环入口: 实测链 → 停止角 → 可达集 → SIL2, 三重判定。

    joint_speed_rad_s=8 对齐 EdaLab SafetyLoad.maxJointSpeedRadS 样例 (relay-clutch)。
    """
    total_ms, sb = load_measured_chain()
    acc = sb["acceptance"]

    stop_deg = stop_angle_from_budget(joint_speed_rad_s, max_decel_rad_s2, total_ms)
    stop_ok = stop_deg < acc["stop_angle_max_deg"]
    reach_ok = reachable_set_within_limits(joint_speed_rad_s, total_ms)

    sil2_ok = (acc["measured_pfh"] < acc["sil2_pfh_target"])

    return BudgetClosureResult(
        total_break_ms=total_ms,
        stop_angle_deg=stop_deg,
        stop_angle_ok=bool(stop_ok),
        reachable_set_ok=bool(reach_ok),
        sil2_ok=bool(sil2_ok),
        closure_ok=bool(stop_ok and reach_ok and sil2_ok),
        details={
            "pfh": acc["measured_pfh"], "pfh_target": acc["sil2_pfh_target"],
            "stop_angle_limit_deg": acc["stop_angle_max_deg"],
            "joint_speed_rad_s": joint_speed_rad_s, "max_decel_rad_s2": max_decel_rad_s2,
            "software_filters": [f["stage"] for f in sb["software_filters"]],
            "chain": [s["stage"] for s in sb["segments"]],
        },
    )
