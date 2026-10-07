"""安全预算闭环 — safety_budget.yaml 实测链 ↔ 形式化可达集 (K3 闭环)。

把 EdaLab relay-clutch 实测的分段延迟 (0.8µs 电气 + 8ms 熄弧 + 8ms 机械隔离 = 16ms)
作为形式化安全分析的时间上界输入, 验证:
  1. 停止角位移: θ_stop = ω·t − 0.5·|α_max|·t² ≤ 30° (contracts.acceptance)
  2. 可达集: X_reachable(t=16ms) 与关节限位集不相交
  3. 闭环判定: budget_ok = SIL2 + PFH + 停止角位移 三重通过

P0 修复说明:
  原代码 L55 `t = min(t_budget, t_stop)` 把"停不下来"掩盖成 PASS —
  当 ω=8, α=50 时 t_stop=0.16s > t_budget=0.016s, 机器人**根本停不下来**,
  但代码用 t_budget 代入公式, 得到 6.97° < 30° 判 PASS。真实完全停止角
  ω²/2α = 0.64 rad = 36.7° > 30° 应判 FAIL。
  修复: t_stop > t_budget 时返回 +inf, 让上层按 inf > 30° 判 FAIL。
  并新增 stop_angle_ok 的诚实布尔字段, 报告是否真的能在预算内停下。
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
    # P0 修复: 显式报告"能否在预算内停下" (原代码 min 掩盖停不下来)
    stopped_within_budget: bool = True


def load_measured_chain() -> Tuple[float, Dict[str, Any]]:
    """从 contracts/safety_budget.yaml 读取分段链, 返回 (total_ms, segments)。

    P0 修复 (v1.1): safety_budget.yaml v1.1 把 `measured_*` 重命名为 `assumed_*`
    (诚实声明: 数值来自常量查表, 非 HIL 实测)。为向后兼容, 优先读 assumed_*,
    回退到 measured_*。
    """
    from ...config import load_contract
    sb = load_contract("safety_budget")
    segs = {s["stage"]: s for s in sb["segments"]}
    def _get(d: Dict, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return 0.0
    total_ms = (_get(segs["electrical_off"], "assumed_us", "measured_us") / 1000
                + _get(segs["arc_extinction"], "assumed_ms", "measured_ms")
                + _get(segs["mechanical_isolation"], "assumed_ms", "measured_ms"))
    return total_ms, sb


def stop_angle_from_budget(joint_speed_rad_s: float,
                           max_decel_rad_s2: float = 50.0,
                           total_break_ms: float = None) -> float:
    """给定实测断开时间预算, 计算停止角位移 (匀减速模型, ISO 13849 口径)。

    匀减速: θ_stop = ω·t − 0.5·α·t² (t ≤ t_stop=ω/α)
           完全停止角 = ω²/(2α) (t=t_stop 时)

    P0 修复: 若 t_stop > t_budget (预算内停不下来), 返回 +inf,
    让上层判定为 FAIL — 原代码用 min 掩盖此情况。
    """
    if total_break_ms is None:
        total_break_ms, _ = load_measured_chain()
    t_budget = total_break_ms / 1000.0
    t_stop = joint_speed_rad_s / max(max_decel_rad_s2, 1e-6)
    # P0 修复: 停不下来 → 用完全停止角 (不是 t_budget 内位移)
    if t_stop > t_budget:
        return float("inf")
    return np.degrees(joint_speed_rad_s * t_stop - 0.5 * max_decel_rad_s2 * t_stop * t_stop)


def reachable_set_within_limits(joint_speed_rad_s: float,
                                total_break_ms: float,
                                joint_limit_deg: float = 120.0) -> bool:
    """可达集检验: 16ms 预算内关节角域不越过限位。

    用区间运算 (reachability.IntervalSet 同族简化): 区间中心=当前角(取 0),
    半径 = 停止角位移 + 速度不确定度 (CAN 延迟 2ms 引入的额外滑行)。
    P0 修复: 若 stop_angle_from_budget 返回 inf (停不下来), 可达集判 FAIL。
    """
    from ..chance_constrained_grip import chance_constrained_grip  # noqa: F401 (确保包内一致性)
    theta_stop_deg = stop_angle_from_budget(joint_speed_rad_s, total_break_ms=total_break_ms)
    if not np.isfinite(theta_stop_deg):
        return False
    # CAN-FD 延迟不确定度 (dr_params: canfd_latency_ms max 2.0 → 额外滑行)
    can_extra_deg = np.degrees(joint_speed_rad_s * 2.0 / 1000.0)
    radius = theta_stop_deg + can_extra_deg
    return radius < joint_limit_deg


def close_budget_loop(joint_speed_rad_s: float = 8.0,
                      max_decel_rad_s2: float = 50.0) -> BudgetClosureResult:
    """K3 闭环入口: 实测链 → 停止角 → 可达集 → SIL2, 三重判定。

    joint_speed_rad_s=8 对齐 EdaLab SafetyLoad.maxJointSpeedRadS 样例 (relay-clutch)。

    P0 修复:
      - stop_angle_from_budget 在停不下来时返回 inf, stop_angle_ok 应判 False
      - 新增 stopped_within_budget 字段, 明确报告是否真能在预算内停下
      - SIL2 判定交叉校验: 若 PFH 超 SIL2 上限 → sil2_ok=False (原代码同 YAML 两常量比较)
    """
    total_ms, sb = load_measured_chain()
    acc = sb["acceptance"]

    t_budget_s = total_ms / 1000.0
    t_stop_s = joint_speed_rad_s / max(max_decel_rad_s2, 1e-6)
    stopped_within_budget = bool(t_stop_s <= t_budget_s)

    stop_deg = stop_angle_from_budget(joint_speed_rad_s, max_decel_rad_s2, total_ms)
    # P0 修复: inf > 任何有限阈值 → False (原代码 min 掩盖成 6.97° < 30° 判 PASS)
    stop_ok = bool(np.isfinite(stop_deg) and stop_deg < acc["stop_angle_max_deg"])
    reach_ok = reachable_set_within_limits(joint_speed_rad_s, total_ms)

    # SIL2 判定: PFH < sil2_pfh_target
    # P0 v1.1 修复: v1.1 把 measured_pfh 重命名为 assumed_pfh (诚实边界),
    # 但 sil2_ok 判定语义保持不变 (比较阈值), 关键是 provenance 标记。
    # sil2_release_ready 需 provenance_confidence="measured_hil" 才为真。
    pfh_key = "assumed_pfh" if "assumed_pfh" in acc else "measured_pfh"
    pfh_val = float(acc[pfh_key])
    sil2_ok = bool(pfh_val < acc["sil2_pfh_target"])
    # 诚实标记: sil2_ok 是否基于独立测量
    # v1.1 之前: False (同 YAML 常量, 循环自证)
    # v1.1 之后: False (假设 + 建模, 非 HIL)
    # 未来: provenance_confidence="measured_hil" 时可改 True
    sil2_cross_checked = False
    sil2_release_ready = False  # 需 HIL 后由 YAML metadata.sil2_release_ready 提升

    return BudgetClosureResult(
        total_break_ms=total_ms,
        stop_angle_deg=float(stop_deg),
        stop_angle_ok=stop_ok,
        reachable_set_ok=bool(reach_ok),
        sil2_ok=sil2_ok,
        closure_ok=bool(stop_ok and reach_ok and sil2_ok),
        stopped_within_budget=stopped_within_budget,
        details={
            "pfh": pfh_val, "pfh_key": pfh_key, "pfh_target": acc["sil2_pfh_target"],
            "stop_angle_limit_deg": acc["stop_angle_max_deg"],
            "joint_speed_rad_s": joint_speed_rad_s, "max_decel_rad_s2": max_decel_rad_s2,
            "t_budget_s": t_budget_s, "t_stop_s": t_stop_s,
            "stopped_within_budget": stopped_within_budget,
            "sil2_cross_checked": sil2_cross_checked,
            "sil2_release_ready": sil2_release_ready,
            "provenance_confidence": sb.get("metadata", {}).get("provenance_confidence", "assumed"),
            "software_filters": [f["stage"] for f in sb["software_filters"]],
            "chain": [s["stage"] for s in sb["segments"]],
        },
    )
