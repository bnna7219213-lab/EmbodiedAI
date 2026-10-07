"""动作分词器 — 将连续动作序列编码为离散 token (VQ-VAE 简化) 或连续回归。

K3 小模型策略:
  - 默认使用 continuous_action=True: 经过 action head 直接回归关节角度
  - 保留 VQ-VAE 离散化接口 (动作词表对齐 nanoGPT 消融模式)
  - Action tokenizer 与 nanoGPT 的字符/BPE tokenizer 共享"encode/decode" 抽象

数据流:
  训练: action_continuous [B, T, action_dim] → encode → logits [B, T, action_dim]
  推理: state + vision + lang → decode → action_continuous [B, action_horizon, action_dim]
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # 空占位, 仅在无 torch 时 import 不报错
    class _Stub: ...
    nn = type("nn", (), {"Module": object, "Linear": _Stub, "LayerNorm": _Stub})  # type: ignore


if HAS_TORCH:

    class ActionTokenizer(nn.Module):
        """连续动作编码/解码器。

        支持两种模式:
          continuous (默认): 直接经 action head 从隐状态回归动作向量
          discrete (VQ-VAE): 动作代码本 + 残差量化, 自回归解码动作 token
        """

        def __init__(
            self,
            action_dim: int = 12,
            action_horizon: int = 10,
            n_embd: int = 384,
            continuous: bool = True,
            vq_vocab_size: int = 256,
            vq_commitment: float = 0.25,
            joint_limits: tuple = (-math.pi, math.pi),
        ):
            super().__init__()
            self.action_dim = action_dim
            self.action_horizon = action_horizon
            self.continuous = continuous
            self.joint_limits = joint_limits

            # action head: 从 transformer 隐状态预测动作
            self.action_head = nn.Sequential(
                nn.Linear(n_embd, n_embd * 2),
                nn.GELU(),
                nn.Linear(n_embd * 2, action_dim * action_horizon),
            )

            # VQ codebook (discrete 模式)
            self.use_vq = not continuous
            if self.use_vq:
                self.vq_vocab_size = vq_vocab_size
                self.vq_commitment = vq_commitment
                self.codebook = nn.Embedding(vq_vocab_size, action_dim)
                # 初始化 codebook 在 joint limits 范围内
                nn.init.uniform_(self.codebook.weight, *joint_limits)

        def encode(self, action: torch.Tensor) -> torch.Tensor:
            """连续动作 → 隐空间表示 (训练时用)。

            Args:
                action: [B, T, action_dim] 连续动作
            Returns:
                target: [B, T, action_dim * action_horizon] 展平目标
            """
            B, T, D = action.shape
            assert D == self.action_dim
            # 展平为 (B, T, action_dim * horizon) 目标
            return action.reshape(B, T, -1)

        def decode(self, hidden: torch.Tensor) -> torch.Tensor:
            """隐状态 → 动作预测 (推理时用)。

            Args:
                hidden: [B, n_embd] 末端隐状态
            Returns:
                action: [B, action_horizon, action_dim]
            """
            raw = self.action_head(hidden)  # [B, action_dim * horizon]
            action = raw.reshape(-1, self.action_horizon, self.action_dim)
            # clip 到关节限位
            return torch.clamp(action, self.joint_limits[0], self.joint_limits[1])

        def forward(self, hidden: torch.Tensor, target_action=None) -> tuple:
            """训练 forward: 返回预测 + loss。

            P0 修复: 原代码未 clamp 到 joint_limits, 而 docstring
            "训练/推理都钳位到 joint_limits" 是假宣称 (decode 有 clamp, forward 无)。
            修复: forward 也 clamp 到 joint_limits, 让安全宣称生效。
            """
            B = hidden.shape[0]
            logits = self.action_head(hidden)  # [B, action_dim * horizon]
            pred = logits.reshape(B, self.action_horizon, self.action_dim)
            # P0 修复: clamp 到 joint_limits (与 decode 一致)
            pred = torch.clamp(pred, self.joint_limits[0], self.joint_limits[1])

            loss = None
            if target_action is not None:
                target_flat = target_action.reshape(B, -1)
                pred_flat = pred.reshape(B, -1)
                loss = F.mse_loss(pred_flat, target_flat)

            return pred, loss

        def discretize(self, action: torch.Tensor) -> torch.Tensor:
            """连续动作 → 最近 codebook token (VQ, 可选)。"""
            if not self.use_vq:
                raise RuntimeError("discretize 仅在 VQ 模式下可用")
            # 最近邻
            flat = action.reshape(-1, self.action_dim)
            distances = (flat.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(-1)
            return distances.argmin(dim=-1)

        def codebook_lookup(self, tokens: torch.Tensor) -> torch.Tensor:
            """codebook token → 连续动作 (解码)。"""
            if not self.use_vq:
                raise RuntimeError("codebook_lookup 仅在 VQ 模式下可用")
            return self.codebook(tokens)

else:
    # 无 torch 时的桩
    class ActionTokenizer:
        def __init__(self, **kwargs):
            raise RuntimeError("action_tokenizer 需要 PyTorch")
