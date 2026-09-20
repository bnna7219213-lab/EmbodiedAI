"""Vision Encoder (简化 ViT-Tiny) — 多视角 RGB → 视觉 token 序列。

K3 策略: 极简 ViT (~10M 参数), 将每帧 224×264 RGB 拆为 196 个 patch_token, 
3 视角 → 504 个视觉 token, 投影到 n_embd 维度后直接输入 Fusion Transformer。
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class PatchEmbed(nn.Module):
        """图像 → patch 序列 (线性投影替代 ViT 原始 patchify + linear)."""
        def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=384):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.n_patches = (img_size // patch_size) ** 2
            self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, x):
            # x: [B, C, H, W] → [B, embed_dim, n_patch_per_dim, n_patch_per_dim] → [B, n_patches, embed_dim]
            x = self.proj(x)
            return x.flatten(2).transpose(1, 2)

    class VisionEncoder(nn.Module):
        """Tiny ViT — N 层 transformer encoder, 输出每 patch 的表征。

        参数量估算: 4 层 × (4×384² self-attn + 4×384×1536 MLP) ≈ 14M。
        """
        def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=384,
                     n_layers=4, n_heads=6, num_views=3):
            super().__init__()
            self.num_views = num_views
            self.n_patches = (img_size // patch_size) ** 2
            self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim) * 0.02)
            self.view_embed = nn.Embedding(num_views, embed_dim) if num_views > 0 else None
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True,
            )
            self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, frames: torch.Tensor, view_ids: torch.Tensor | None = None) -> torch.Tensor:
            """Args:
                frames: [B*V, C, H, W] 所有视角的图像拼在第 0 维
                view_ids: [B*V] 每帧对应视角 id (0..num_views-1), 可被 None 省略
            Returns:
                tokens: [B*V, n_patches, embed_dim]
            """
            x = self.patch_embed(frames) + self.pos_embed
            if self.view_embed is not None and view_ids is not None:
                x = x + self.view_embed(view_ids).unsqueeze(1)
            x = self.norm(self.layers(x))
            return x

else:
    class PatchEmbed:
        def __init__(self, **kw): raise RuntimeError("vision_encoder 需要 PyTorch")
    class VisionEncoder:
        def __init__(self, **kw): raise RuntimeError("vision_encoder 需要 PyTorch")
