"""中枢决策栈 (C1-C5): 信号处理 / 感知 / 记忆 / 决策(nanoGPT·VLA) / 中枢编排。

layer_registry — 全部架构层 (L1-L7 + P1-P5 + S + D1-D5 + C1-C5) 机器可读单一真源。
center_brain  — 端到端链路编排 facade (不重复实现既有模块)。
spatial_reasoning — V2L 空间智能可解释论证。
"""
from . import layer_registry
from .layer_registry import LAYER_REGISTRY, LayerSpec, layers_by_stack, as_dict, validate_registry
from .spatial_reasoning import SpatialArgument, SpatialReasoner
from .center_brain import CenterBrain, CenterStepResult

__all__ = ["layer_registry", "LAYER_REGISTRY", "LayerSpec", "layers_by_stack",
           "as_dict", "validate_registry",
           "SpatialArgument", "SpatialReasoner",
           "CenterBrain", "CenterStepResult"]
