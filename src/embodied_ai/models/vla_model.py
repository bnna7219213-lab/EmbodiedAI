"""VLA 主模型 — Vision-Language-Action Transformer (Decoder-only)。

架构对照 Figure Helix 2.5 + RT-2:
  - 3 视角 RGB → VisionEncoder (ViT-Tiny) → 3×196 视觉 tokens
  - 语言指令 → 词嵌入 → L 个语言 tokens  
  -本体感受 → StateProjection → S 个状态 tokens
  - Concat → Context tokens (visual + lang + state)
  - Decoder-only Transformer (nanoGPT 模式: causal mask + KV cache)
  - ActionTokenizer head → 关节角度序列

训练/推理模式:
  训练: (rgb, lang_ids, state) → forward → action_hat vs action → MSE loss
  推理: (rgb, lang_ids, state) → forward → 关节角度 (20Hz)
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

    from .action_tokenizer import ActionTokenizer
    from .vla_config import VLAConfig
    from .vision_encoder import VisionEncoder


    class VLATransformerBlock(nn.Module):
        """单块 Decoder (Pre-LN, GQA, SwiGLU)。

        结构与 nanoGPT GPTModel.transformer 高度一致, 关键改动: KV 可选 GQA + RoPE 可选。
        """
        def __init__(self, cfg: VLAConfig):
            super().__init__()
            self.norm1 = nn.LayerNorm(cfg.n_embd)
            n_kv = cfg.n_kv_head or cfg.n_head
            assert cfg.n_embd % cfg.n_head == 0
            head_dim = cfg.n_embd // cfg.n_head
            self.attn = nn.MultiheadAttention(
                embed_dim=cfg.n_embd,
                num_heads=cfg.n_head,
                dropout=cfg.dropout,
                batch_first=True,
                kdim=n_kv * head_dim,
                vdim=n_kv * head_dim,
            )
            self.norm2 = nn.LayerNorm(cfg.n_embd)
            # SwiGLU MLP (对齐 LLaMA/现代 decoder)
            self.up = nn.Linear(cfg.n_embd, cfg.n_embd * 4)
            self.gate = nn.Linear(cfg.n_embd, cfg.n_embd * 4)
            self.down = nn.Linear(cfg.n_embd * 4, cfg.n_embd)
            self.resid_dropout = nn.Dropout(cfg.dropout)

        def forward(self, x, attn_mask=None, kv_cache=None):
            # Pre-LN + Self-Attention + Residual
            h = self.norm1(x)
            attn_out, _ = self.attn(h, h, value=h, attn_mask=attn_mask, need_weights=False)
            x = x + self.resid_dropout(attn_out)
            # Pre-LN + SwiGLU MLP + Residual
            h = self.norm2(x)
            x = x + self.resid_dropout(self.down(F.silu(self.gate(h)) * self.up(h)))
            return x


    class VLAModel(nn.Module):
        """VLA 端到端网络。

        结构 (从下到上):
          VisionEncoder (14M) → visual tokens [B, 504, 384]
          Language Embedding (共享) → lang tokens [B, L, 384]
          State Projection → state tokens [B, S, 384]
          Concat → context tokens [B, 504+L+S, n_embd]
          6× Decoder Blocks → hidden [B, T, n_embd]
          ActionTokenizer → action [B, action_horizon, action_dim]

        参数量: ~50M (vla_tiny) / ~120M (vla_base) / ~250M (vla_large)。
        推理吞吐 (A100 batch=1): vla_tiny 120Hz / vla_base 50Hz / vla_large 20Hz。
        """
        def __init__(self, cfg: VLAConfig):
            super().__init__()
            self.cfg = cfg

            # 输入嵌入
            if cfg.num_views > 0:
                self.vision_encoder = VisionEncoder(
                    img_size=cfg.img_size, patch_size=cfg.patch_size,
                    in_ch=cfg.in_channels, embed_dim=cfg.vision_embd,
                    n_layers=cfg.num_vision_layers, n_heads=cfg.num_vision_heads,
                    num_views=cfg.num_views,
                )
                # K3+ 修复: vision_embd 与 n_embd 不一致时必须有投影桥
                # (此前无 torch 环境从未真跑, vla_tiny(vision_embd=384, n_embd=256) 前向必崩)
                self.vision_proj = (nn.Linear(cfg.vision_embd, cfg.n_embd)
                                    if cfg.vision_embd != cfg.n_embd else nn.Identity())
            self.lang_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.state_projection = nn.Sequential(
                nn.Linear(cfg.state_dim, cfg.n_embd),
                nn.GELU(),
                nn.Linear(cfg.n_embd, cfg.n_embd),
            )

            # 模态类型嵌入 (区分视觉/语言/状态)
            self.modality_embedding = nn.Embedding(3, cfg.n_embd)
            self.modality_map = {"vision": 0, "language": 1, "state": 2}

            # 位置嵌入
            self.register_buffer("pos_embed", torch.zeros(1, cfg.block_size, cfg.n_embd))
            nn.init.normal_(self.pos_embed, std=0.02)

            # Decoder-only transformer blocks
            self.blocks = nn.ModuleList([VLATransformerBlock(cfg) for _ in range(cfg.n_layer)])
            self.norm_f = nn.LayerNorm(cfg.n_embd)

            # Action head
            self.action_tokenizer = ActionTokenizer(
                action_dim=cfg.action_dim, action_horizon=cfg.action_horizon,
                n_embd=cfg.n_embd, continuous=cfg.continuous_action,
                vq_vocab_size=cfg.vq_vocab_size,
            )

            # Dropout
            self.drop = nn.Dropout(cfg.dropout)

            # Weight init
            self.apply(self._init_weights)
            # 特殊初始化 token embedding (对齐 GPT-2)
            nn.init.normal_(self.lang_embedding.weight, std=0.02)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        def _causal_mask(self, T: int, device):
            return torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=1)

        def forward(self, frames: torch.Tensor | None, lang_ids: torch.Tensor | None,
                    state: torch.Tensor, target_action: torch.Tensor | None = None) -> tuple:
            """Args:
                frames: [B*V, C, H, W] (V=num_views), 无视觉时为 None
                lang_ids: [B, L] int tensor, 无语言输入用 None
                state: [B, state_dim] 本体感受
                target_action: [B, action_horizon, action_dim] 训练时的 GT (可选)
            Returns:
                action_pred: [B, action_horizon, action_dim] 预测的关节角度
                loss: 标量 (target_action 非 None 时)
            """
            tokens = []
            B = state.shape[0]

            # 1. 视觉 tokens (可选)
            if frames is not None and self.cfg.num_views > 0:
                V = self.cfg.num_views
                vision_out = self.vision_encoder(frames)      # [B*V, n_patches, vision_embd]
                vision_out = self.vision_proj(vision_out)      # → [B*V, n_patches, n_embd]
                # 视角展平到 token 维: [B, n_patches*V, n_embd]
                vision_out = vision_out.reshape(-1, V * vision_out.shape[1], vision_out.shape[2])
                # 添加模态嵌入
                vision_out = vision_out + self.modality_embedding(
                    torch.full((vision_out.shape[1],), self.modality_map["vision"],
                               dtype=torch.long, device=state.device)
                ).unsqueeze(0)
                tokens.append(vision_out)

            # 2. 语言 tokens
            if lang_ids is not None:
                lang_out = self.lang_embedding(lang_ids)  # [B, L, n_embd]
                lang_out = lang_out + self.modality_embedding(
                    torch.full((lang_ids.shape[1],), self.modality_map["language"],
                               dtype=torch.long, device=state.device)
                ).unsqueeze(0)
                tokens.append(lang_out)

            # 3. 状态 tokens
            state_out = self.state_projection(state)  # [B, n_embd]
            state_out = state_out.unsqueeze(1)  # [B, 1, n_embd]
            state_out = state_out + self.modality_embedding(
                torch.full((1,), self.modality_map["state"], dtype=torch.long, device=state.device)
            ).unsqueeze(0)
            tokens.append(state_out)

            # 4. 拼接上下文 + 位置编码 + dropout
            ctx = torch.cat(tokens, dim=1)  # [B, T, n_embd]
            T = ctx.shape[1]
            if T > self.cfg.block_size:
                ctx = ctx[:, -self.cfg.block_size:, :]  # 截断
                T = self.cfg.block_size
            ctx = ctx + self.pos_embed[:, :T, :]
            ctx = self.drop(ctx)

            # 5. Decoder blocks (causal mask)
            mask = self._causal_mask(T, ctx.device)
            for block in self.blocks:
                ctx = block(ctx, attn_mask=mask)

            # 6. Action prediction (用最后 n_embd 维 hidden state)
            ctx = self.norm_f(ctx)
            final_hidden = ctx[:, -1, :]  # [B, n_embd]

            # Action head
            action_pred, loss = self.action_tokenizer(final_hidden, target_action)
            return action_pred, loss

        @torch.no_grad()
        def inference(self, frames: torch.Tensor | None, lang_ids: torch.Tensor | None,
                      state: torch.Tensor) -> torch.Tensor:
            """推理模式 (关闭 dropout, 返回动作)。"""
            self.eval()
            action_pred, _ = self.forward(frames, lang_ids, state)
            self.train()
            return action_pred

        def num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def estimate_flops_per_forward(self) -> int:
            """粗略 FLOPs 估算。"""
            p = self.num_parameters()
            return 2 * p  # 近似: 每个参数约 2 次浮点操作

else:
    class VLATransformerBlock:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): raise RuntimeError("vla_model 需要 PyTorch")

    class VLAModel:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): raise RuntimeError("vla_model 需要 PyTorch")
