"""L6 VLA 小模型 — Vision-Language-Action Nano (复用 nanoGPT 训练管线)。

K3 导出:
  VLAConfig — 超参数 dataclass (消融能力)
  VLAModel — 端到端 VLA 网络 (Vision + Transformer + Action)
  VLATrainer — 训练循环 (warmup + cosine + mixed precision + checkpoint)
  ActionTokenizer — 连续动作编解码 + VQ 离散化接口
  VisionEncoder — Tiny ViT (多视角 RGB → tokens)

设计原则:
  - ≤200M 总参数, Jetson Orin 20Hz 推理
  - 消融实验: VLAConfig.ablations() 提供 4 套对比配置
  - 安全关键: 关节角度 clip 到 joint_limits (确定性保障)
"""

from .action_tokenizer import ActionTokenizer
from .vla_config import VLAConfig
from .vla_model import VLAModel
from .vla_trainer import VLATrainer
from .vision_encoder import VisionEncoder

__all__ = [
    "ActionTokenizer",
    "VLAConfig",
    "VLAModel",
    "VLATrainer",
    "VisionEncoder",
]
