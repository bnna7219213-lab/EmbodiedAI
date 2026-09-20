"""部署适配 — TensorRT / INT8 量化 / 端到端延迟核验。

K3 占位 (接 future-exploration-plan Phase B 对接):
  - INT8/FP16 导出 (ONNX → TensorRT engine)
  - Jetson Orin 延迟基准 (vla_tiny 目标 20Hz / 单帧 < 50ms)
  - Quantization-Aware Training (QAT) Hook

注: 实际部署依赖 TensorRT / torch.onnx.export, 当前为包骨架占位。
"""

__all__: list = []
