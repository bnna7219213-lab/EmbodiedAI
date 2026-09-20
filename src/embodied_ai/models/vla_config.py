"""VLA 模型配置 (K3: 复用 nanoGPT ModelConfig dataclass 模式)。

设计参照 nanoGPT llm.py 的 ModelConfig:
  - dataclass 封装超参, 支持同进程多配置 (消融实验)
  - JSON 序列化/反序列化 (可复现)
  - 设备 + seed + 工程化选项 (混合精度/梯度裁剪/KV-cache)
  - 附加 VLA 特有字段: 视觉编码器、动作词表、多模态融合
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class VLAConfig:
    """VLA (Vision-Language-Action) 小模型超参数。

    配置分 5 个域:
      1. 模型结构 (transformer backbone, 复用 nanoGPT 字段命名)
      2. 视觉编码器 (ViT 简化)
      3. 语言/状态融合
      4. 动作 tokenizer + 解码
      5. 训练工程化 (对齐 nanoGPT ModelConfig: seed/device/混合精度/ckpt)

    小模型目标: ≤200M 总参数, Jetson Orin 20Hz 推理。
    """

    # ============== 1. 模型结构 (Transformer backbone, 对齐 nanoGPT) ==============
    n_layer: int = 6                # Transformer Decoder 层数
    n_embd: int = 384               # 嵌入维度
    n_head: int = 6                 # 注意力头数
    n_kv_head: Optional[int] = None  # GQA KV 头数 (None = MHA = n_head)
    dropout: float = 0.0            # Dropout
    block_size: int = 256           # 上下文窗口长度 (visual+lang+state+action tokens)

    # ============== 2. 视觉编码器 (简化 ViT-like) ==============
    img_size: int = 224             # 输入图像尺寸
    patch_size: int = 16            # Patch 大小
    in_channels: int = 3            # RGB
    vision_embd: int = 384          # 视觉特征维度 (= n_embd, 直接对齐)
    num_vision_layers: int = 4      # Vision encoder 层数 (ViT-Tiny ~ 4 层)
    num_vision_heads: int = 6       # Vision attention heads
    num_views: int = 3              # 多视角摄像头数 (头顶/胸部/手端)

    # ============== 3. 语言/状态融合 ==============
    vocab_size: int = 256           # 语言词表大小 (子词 BPE 后, 小模型 256 足够)
    state_dim: int = 30             # 本体感受维度 (关节角 + 速度 + 力矩)
    state_embd: int = 384           # 状态投影维度 (= n_embd)
    fusion_type: str = "cross_attention"  # "cross_attention" / "concat" / "prepend"

    # ============== 4. 动作 Tokenizer ==============
    action_dim: int = 12            # 动作维度 (关节数)
    action_horizon: int = 10        # 预测时间步长 (Block/N-Block)
    vq_vocab_size: int = 256        # VQ-VAE 动作词表大小
    vq_embd: int = 384              # VQ embedding 维度
    vq_commitment_cost: float = 0.25  # VQ-VAE commitment loss 权重
    # 简化模式: 是否直接使用连续回归 (True=放弃 VQ 离散化)
    continuous_action: bool = True

    # ============== 5. 训练工程化 (对齐 nanoGPT ModelConfig) ==============
    batch_size: int = 16
    max_iters: int = 10000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    warmup_iters: int = 100
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_iters: int = 20

    # 对齐 nanoGPT M2 工程化选项
    device: str = "cuda"
    seed: int = 42
    deterministic: bool = True
    use_sdpa: bool = True
    use_mixed_precision: bool = True

    # 检查点/日志
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
    ablation_name: str = "vla_base"

    @property
    def num_patches(self) -> int:
        """每帧图像 patch 数"""
        return (self.img_size // self.patch_size) ** 2

    @property
    def vision_tokens_per_view(self) -> int:
        return self.num_patches

    @property
    def total_vision_tokens(self) -> int:
        return self.num_patches * self.num_views

    def estimate_params(self) -> dict:
        """估算各组件参数量 (粗略, 用于消融实验的参数量报告)."""
        # Vision encoder (简化 ViT)
        patch_dim = self.patch_size * self.patch_size * self.in_channels
        vision_params = (
            patch_dim * self.vision_embd  # patch projection
            + self.num_vision_layers * (
                4 * self.vision_embd ** 2  # self-attn Q/K/V/O
                + 2 * self.vision_embd * self.vision_embd * 4  # MLP (4x expand)
                + 2 * self.vision_embd * 2  # 2 layer norms
            )
        )
        # State projection
        state_params = self.state_dim * self.state_embd
        # Fusion transformer (decoder-only, per-layer)
        # self-attn + MLP + 2 layernorm
        head_dim = self.n_embd // self.n_head
        kv_dim = head_dim * (self.n_kv_head or self.n_head) * 2  # K+V
        attn_params = self.n_embd * (self.n_embd + kv_dim + self.n_embd)  # Q,K,V,O
        mlp_params = 2 * self.n_embd * (self.n_embd * 4)  # SwiGLU ~ 4x expand
        layer_params = attn_params + mlp_params + 2 * self.n_embd * 4  # layernorms
        transformer_params = self.n_layer * layer_params
        # Action head
        action_head_params = self.n_embd * self.action_dim * self.action_horizon
        # VQ codebook
        vq_params = self.vq_vocab_size * self.vq_embd
        total = vision_params + state_params + transformer_params + action_head_params + vq_params
        return {
            "vision_encoder": vision_params,
            "state_projection": state_params,
            "fusion_transformer": transformer_params,
            "action_head": action_head_params,
            "vq_codebook": vq_params,
            "total": total,
        }

    # ============== 配置持久化 (对齐 nanoGPT JSON 模式) ==============
    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VLAConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def ablations() -> dict:
        """消融配置集合 — 对比所用 (~4 个配置, 单因素变化)."""
        base = VLAConfig()
        return {
            "vla_tiny": VLAConfig(
                n_layer=4, n_embd=256, n_head=4,
                num_vision_layers=2, action_horizon=5,
                ablation_name="vla_tiny",
            ),
            "vla_base": base,
            "vla_large": VLAConfig(
                n_layer=8, n_embd=512, n_head=8,
                num_vision_layers=6, action_horizon=15,
                ablation_name="vla_large",
            ),
            "vla_no_vision": VLAConfig(
                num_views=0, block_size=128,
                ablation_name="vla_no_vision",
            ),
        }
