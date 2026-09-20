"""横切安全层 S — 形式化 (STL + 可达集 + ISO 13482) + 概率性 (机会约束)。

K1: chance_constrained_grip (概率性, 不确定性量化)
K3: formal/ (形式化: STL 规约 + 区间可达集 + ISO 13482 运行时监控 — 创新 #4)
"""
from . import chance_constrained_grip
from .formal import (
    ISO13482Monitor,
    IntervalSet,
    ReachabilityAnalyzer,
    SafetyEnvelope,
    STLAtom,
    STLGlobally,
    STLFinally,
    STLImplies,
)

__all__ = [
    "chance_constrained_grip",
    "ISO13482Monitor",
    "IntervalSet",
    "ReachabilityAnalyzer",
    "SafetyEnvelope",
    "STLAtom",
    "STLGlobally",
    "STLFinally",
    "STLImplies",
]
