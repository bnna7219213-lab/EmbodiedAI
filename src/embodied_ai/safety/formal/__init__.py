"""形式化安全层 (创新 #4) — STL 规约 + 区间可达集 + ISO 13482 运行时监控。

导出:
  - STL: STLAtom, STLGlobally, STLFinally, STLImplies, STLAnd, STLOr, STLNot
  - Reachability: IntervalSet, SafetyEnvelope, ReachabilityAnalyzer
  - ISO 13482: ISO13482Monitor, BIOMECH_LIMITS_N, ISO13482Result, ISO13482Violation

用法:
  from embodied_ai.safety.formal import ISO13482Monitor, STLGlobally, STLAtom, IntervalSet

  monitor = ISO13482Monitor(state_dim=12, dt=0.001)
  result = monitor.evaluate(joint_state, force_reading=30.0, human_distance_m=0.3)
  if not result.safe:
    trigger_STO(result.violations)   # STO — Safe Torque Off (<1ms)
"""

from .iso13482 import (
    BIOMECH_LIMITS_N,
    ISO13482Monitor,
    ISO13482Result,
    ISO13482Violation,
)
from .reachability import (
    IntervalSet,
    ReachabilityAnalyzer,
    SafetyEnvelope,
)
from .stl_specs import (
    STLAtom,
    STLFinally,
    STLGlobally,
    STLImplies,
    force_limit_monitor,
    force_ramp_monitor,
    velocity_when_human_near,
)

__all__ = [
    "BIOMECH_LIMITS_N",
    "ISO13482Monitor",
    "ISO13482Result",
    "ISO13482Violation",
    "IntervalSet",
    "ReachabilityAnalyzer",
    "SafetyEnvelope",
    "STLAtom",
    "STLFinally",
    "STLGlobally",
    "STLImplies",
    "force_limit_monitor",
    "force_ramp_monitor",
    "velocity_when_human_near",
]
