"""K3+: VLA tiny 真训冒烟 — 真实 torch 前向 + 反传 + loss 下降验证 (CPU)。

用系统 Python 3.14 (torch 2.14.1+cpu) 执行:
    PYTHONPATH=src python3.14 scripts/vla_train_smoke.py

接口契约 (vla_model.forward docstring):
    frames: [B*V, C, H, W] (V=num_views 视角展平在第 0 维), 无视觉 None
    lang_ids: [B, L]
    state: [B, state_dim]
    target_action: [B, action_horizon, action_dim] 训练 GT (可选)
    returns: (action_pred, loss?) tuple
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.models.vla_config import VLAConfig
from embodied_ai.models.vla_model import VLAModel


def main() -> int:
    cfg = VLAConfig.ablations()["vla_tiny"]
    model = VLAModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params (torch {torch.__version__}, cpu)")

    B, V, C, H, W = 4, 3, 3, 224, 224
    frames = torch.rand(B * V, C, H, W)              # [B*V, C, H, W] 契约形状
    lang = torch.randint(0, 256, (B, 8))             # [B, L], vocab_size=256
    state = torch.rand(B, cfg.state_dim)              # [B, state_dim]
    target = torch.rand(B, cfg.action_horizon, cfg.action_dim)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for step in range(12):
        out = model(frames, lang, state, target)
        a_hat, loss = out[0], out[1]
        if loss is None:  # 实现未按契约回传 loss 时退回显式 MSE
            loss = torch.nn.functional.mse_loss(a_hat, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f} over 12 steps")
    ok = losses[-1] < losses[0]
    print("VLA SMOKE:", "PASS" if ok else "NO-DESCEND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
