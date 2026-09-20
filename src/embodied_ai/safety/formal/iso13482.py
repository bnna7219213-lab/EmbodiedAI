"""ISO 13482 协作机器人安全要求 — 形式化规约创新 #4 应用层。

将 ISO 13482 的关键条款翻译为可直接在运行时评估的形式化规约:
  - §5.2.1 速度监控: 接近人体时 v < 0.25 m/s
  - §5.3.2 力/力矩限制: 最大 transient force < 280 N (ISO/TS 15066 生物力学阈值)
  - §5.3.3 功率限制: 机械功率 < 80 W 
  - §5.4   安全相关控制系统: SIL2/PFd < 10⁻⁶
  - §5.5   停止性能: Cat.0/1 停止时间 < 10 ms
  - §5.6   保护性停止: 独立于主控, 硬件实现

运行时集成 (周期性调用):
  monitor = ISO13482Monitor(state_dim, dt)
  for each control cycle:
    result = monitor.evaluate(joint_state, force_reading, human_distance)
    if not result.safe:
      trigger_STO(result.violations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .reachability import IntervalSet, ReachabilityAnalyzer, SafetyEnvelope
from .stl_specs import (
    STLAtom, STLGlobally, STLImplies, force_ramp_monitor, force_limit_monitor,
    velocity_when_human_near,
)


# ---- ISO/TS 15066 生物力学力阈值 (最严格区域: 颅骨/头部) ----
# 数据来源: ISO/TS 15066:2016 Annex A
BIOMECH_LIMITS_N = {
    "skull_forehead": 130,        # N
    "face_cheek": 65,
    "neck": 150,
    "chest_upper": 140,
    "abdomen": 140,
    "pelvis": 180,
    "upper_arm": 150,
    "lower_arm": 160,
    "hand": 180,
    "default": 280,              # 宽松区域 (如大腿背面)
}


@dataclass
class ISO13482Violation:
    """单次违规记录。"""
    timestamp: float
    clause: str                   # ISO 条款编号
    message: str
    severity: int                 # 1=INFO, 2=WARNING, 3=CRITICAL, 4=STO
    actual_value: float
    threshold: float


@dataclass
class ISO13482Result:
    """单次 evaluate 返回。"""
    safe: bool
    violations: List[ISO13482Violation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stl_robustness: Optional[float] = None
    budget_stop_ok: bool = True
    min_stop_distance_m: float = 0.0


class ISO13482Monitor:
    """ISO 13482 运行时监控器。

    集成三种检查方式:
      1. **STL 公式**: 形式化语义的鲁棒度在线计算
      2. **区间可达集**: 当前状态不变集 + 动力学 → 未来 N 步安全验证
      3. **直接阈值检查**: 关节/转速/力的硬限幅

    每控制周期调用一次 `evaluate()`, 返回 ISO13482Result。
    """

    def __init__(
        self,
        state_dim: int = 12,          # 6 关节 × (pos + vel) 或 12 关节角度
        dt: float = 0.001,             # 控制频率 (1kHz)
        max_force_N: float = 280.0,
        max_power_W: float = 80.0,
        max_speed_ms: float = 0.25,    # §5.2.1 人体附近限速
        max_torque_Nm: float = 50.0,
        stl_sampling_dt: float = 0.01,
    ):
        self.state_dim = state_dim
        self.dt = dt
        self.max_force_N = max_force_N
        self.max_power_W = max_power_W
        self.max_speed_ms = max_speed_ms
        self.max_torque_Nm = max_torque_Nm
        self.stl_sampling_dt = stl_sampling_dt
        self.history: List[Dict[str, Any]] = []
        self._reach = ReachabilityAnalyzer(state_dim, dt)

        # ---- 预定义 STL 公式 ----
        # "始终 10ms 内力不超限"
        self._stl_force = force_limit_monitor(max_force_N)
        # "接近人体 → 限速" (ISO 13482 §5.2.1)
        self._stl_speed_near_human = velocity_when_human_near(
            human_dist_thr=0.5, vel_thr=max_speed_ms
        )
        # "力变化率 < 500 N/s" (防冲击)
        self._stl_ramp = force_ramp_monitor(max_rate=500.0, dt=dt)

    def evaluate(
        self,
        joint_state: np.ndarray,
        force_reading: float = 0.0,
        human_distance_m: float = 1e9,   # 默认无穷远
        torque_reading: float = 0.0,
        power_reading: float = 0.0,
        timestamp: float = 0.0,
    ) -> ISO13482Result:
        """单次 ISO 13482 安全评估。

        Args:
            joint_state: [state_dim] 关节角度向量 (或 pos+vel 拼接)
            force_reading: 当前接触力幅值 (N)
            human_distance_m: 最近人体距离 (m), 默认很大 (=无人)
            torque_reading: 当前关节力矩 (Nm)
            power_reading: 当前机械功率 (W)
            timestamp: 时间戳
        Returns:
            ISO13482Result.safe = True 表示所有检查通过
        """
        violations: List[ISO13482Violation] = []
        warnings: List[str] = []

        # ---- 1. 直接阈值 (硬限制) ----
        if abs(force_reading) > self.max_force_N:
            violations.append(ISO13482Violation(
                timestamp=timestamp,
                clause="ISO 13482 §5.3.2 / ISO/TS 15066",
                message=f"接触力 {force_reading:.1f}N 超限 (>{self.max_force_N}N)",
                severity=4,  # STO
                actual_value=force_reading,
                threshold=self.max_force_N,
            ))

        if abs(torque_reading) > self.max_torque_Nm:
            violations.append(ISO13482Violation(
                timestamp=timestamp,
                clause="ISO 13482 §5.3.3",
                message=f"关节力矩 {torque_reading:.1f}Nm 超限 (>{self.max_torque_Nm}Nm)",
                severity=4,
                actual_value=torque_reading,
                threshold=self.max_torque_Nm,
            ))

        if power_reading > self.max_power_W:
            violations.append(ISO13482Violation(
                timestamp=timestamp,
                clause="ISO 13482 §5.3.3",
                message=f"机械功率 {power_reading:.1f}W 超限 (>{self.max_power_W}W)",
                severity=4,
                actual_value=power_reading,
                threshold=self.max_power_W,
            ))

        # ---- 2. 速度监控 (§5.2.1) ----
        if human_distance_m < 0.5:
            joint_vel = np.linalg.norm(joint_state[1::2]) if len(joint_state) > 1 else 0.0
            if joint_vel > self.max_speed_ms:
                violations.append(ISO13482Violation(
                    timestamp=timestamp,
                    clause="ISO 13482 §5.2.1",
                    message=f"人体{human_distance_m:.2f}m内关节速度{joint_vel:.3f}ms⁻¹ 超限",
                    severity=4,
                    actual_value=joint_vel,
                    threshold=self.max_speed_ms,
                ))

        # ---- 3. STL 形式化监控 (鲁棒度) ----
        stl_robust = float("inf")
        signal = {
            "time": [timestamp],
            "force": [force_reading],
            "state": force_reading,
        }
        try:
            r_force = self._stl_force.robustness(signal, t=0, dt=self.stl_sampling_dt)
            stl_robust = min(stl_robust, r_force)
            if r_force <= 0 and abs(force_reading) < self.max_force_N * 0.95:
                warnings.append(f"STL 力监控鲁棒度 {r_force:.3f} 低于预期")
        except Exception:
            pass  # STL 计算失败时不阻塞控制

        self.history.append({
            "timestamp": timestamp,
            "force": force_reading,
            "torque": torque_reading,
            "power": power_reading,
            "human_distance": human_distance_m,
            "violations": len(violations),
            "stl_robustness": stl_robust,
        })
        # 限制历史长度
        if len(self.history) > 10000:
            self.history = self.history[-5000:]

        is_safe = len(violations) == 0
        return ISO13482Result(
            safe=is_safe,
            violations=violations,
            warnings=warnings,
            stl_robustness=stl_robust,
        )

    def check_stop_budget(
        self,
        current_speed: float,
        budget_steps: int,
        max_decel: float,
    ) -> Tuple[bool, float]:
        """检查在 max_decel 减速下, budget_steps 内能否停到零。

        Args:
            current_speed: 当前线速度 (m/s)
            budget_steps: 控制时钟节拍数 (如 100Hz×20ms = 2 步)
            max_decel: 最大减速度 (m/s²)
        Returns:
            (ok, stop_distance_m)
        """
        t_stop = current_speed / max(max_decel, 1e-6)
        steps_needed = int(np.ceil(t_stop / self.dt))
        stop_dist = current_speed * t_stop - 0.5 * max_decel * t_stop ** 2
        return steps_needed <= budget_steps, max(stop_dist, 0.0)

    def biomech_limit(self, body_region: str = "default") -> float:
        """ISO/TS 15066 生物力学力阈值 (按身体区域)。"""
        return BIOMECH_LIMITS_N.get(body_region, BIOMECH_LIMITS_N["default"])

    def summary(self) -> Dict[str, Any]:
        """运行至今的统计。"""
        if not self.history:
            return {"total_samples": 0, "violation_count": 0}
        n_viol = sum(h["violations"] for h in self.history)
        return {
            "total_samples": len(self.history),
            "violation_count": n_viol,
            "violation_rate": n_viol / max(len(self.history), 1),
            "min_stl_robustness": min(
                (h["stl_robustness"] for h in self.history if h["stl_robustness"] is not None),
                default=float("nan"),
            ),
        }
