"""部署域 (吸收 NG12): 延迟审计 / 量化 / TensorRT 验证 / ARM 目标部署。

K2 交付: latency_audit (hw_requirements 契约预算自动 diff)。
v0.6: latency_audit 支持 budget_overrides — 由 platform.arm_budget_overrides()
      为无 NPU/GPU 的 Linux on ARM 目标给出保守放大后的分段预算。
K3+: 量化校准 / ONNX 导出 / 逐层精度验证。
"""
from .latency_audit import audit_latency, LatencyAuditResult, LatencySegment

__all__ = ["audit_latency", "LatencyAuditResult", "LatencySegment"]
