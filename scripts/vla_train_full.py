"""vla_base 持续化完整训练 — nanoGPT checkpoint/resume 模式 (Python 3.14 + torch 2.14.1)。

用法 (3.14 环境):
    PYTHONPATH=src python3.14 scripts/vla_train_full.py                    # 从头训练
    PYTHONPATH=src python3.14 scripts/vla_train_full.py --resume          # 从 latest 断点续训
    PYTHONPATH=src python3.14 scripts/vla_train_full.py --iters 60 --ckpt-interval 20

训练数据: 合成遥操作风格 batch (3 视角 RGB + 语言 ids + 本体状态 + 动作 GT)。
真数据接入点: get_batch() 替换为 dataset loader (遥测 D2 → 训练 D4, 见 bigdata.py)。
验收: loss 下降 + 周期 checkpoint 落盘 (原子写 + latest 副本) + --resume 断点衔接。
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.models.vla_config import VLAConfig
from embodied_ai.models.vla_model import VLAModel
from embodied_ai.models.vla_trainer import VLATrainer


def make_trainer(size: str, ckpt_dir: str, seed: int) -> VLATrainer:
    cfg = VLAConfig.ablations()[size]
    cfg.ckpt_dir = ckpt_dir
    cfg.seed = seed
    cfg.device = "cpu"              # CPU 训练环境 (3.14 无 CUDA)
    cfg.use_mixed_precision = False  # 无 GPU 时关混合精度
    torch.manual_seed(seed)
    model = VLAModel(cfg)
    return VLATrainer(cfg, model)


def synth_batch(cfg: VLAConfig, B: int = 2):
    """合成 batch (保留: 无数据集时的回退路径)。"""
    V = cfg.num_views
    frames = torch.rand(B * V, 3, cfg.img_size, cfg.img_size)
    lang = torch.randint(0, cfg.vocab_size, (B, 8))
    state = torch.rand(B, cfg.state_dim)
    target = torch.rand(B, cfg.action_horizon, cfg.action_dim)
    return frames, lang, state, target


def make_replay_batcher(dataset_path: str, cfg: VLAConfig, batch_size: int = 2):
    """真实 D4 数据集 → batch 生成器 (bigdata D2→D4 管线消费端)。

    返回 (train_batch_fn, val_batch_fn, provenance)。
    val 集接 estimate_loss 用 (best_val_loss/loss_log 生效的前提)。
    """
    from embodied_ai.models.replay_dataset import (build_pairs, load_telemetry_jsonl,
                                                   make_loaders)
    records, provenance = load_telemetry_jsonl(Path(dataset_path))
    split = build_pairs(records)
    if split.n_pairs < 8:
        raise ValueError(f"数据集对数不足 ({split.n_pairs} < 8), 无法 train/val 分割: {dataset_path}")
    tr_loader, va_loader = make_loaders(split.train, split.val, cfg=cfg, batch_size=batch_size)

    def train_batch():
        frames, lang, state, action = next(iter(tr_loader))
        # ReplayDataset 输出 [B,V,C,H,W] → 契约 [B*V,C,H,W]
        B, V = frames.shape[0], frames.shape[1]
        return frames.reshape(B * V, *frames.shape[2:]), lang, state, action

    val_batches = []

    def val_batch():
        nonlocal val_batches
        if not val_batches and va_loader is not None:
            val_batches = list(va_loader)
        if not val_batches:
            return train_batch()
        frames, lang, state, action = val_batches[0]
        B, V = frames.shape[0], frames.shape[1]
        return frames.reshape(B * V, *frames.shape[2:]), lang, state, action

    return train_batch, val_batch, provenance


def main() -> int:
    ap = argparse.ArgumentParser(description="vla_base 持续化训练 (nanoGPT resume 模式)")
    ap.add_argument("--size", default="vla_base", choices=list(VLAConfig.ablations().keys()))
    ap.add_argument("--iters", type=int, default=60, help="总步数 (从当前 iter 起算)")
    ap.add_argument("--ckpt-interval", type=int, default=20)
    ap.add_argument("--eval-interval", type=int, default=10)
    ap.add_argument("--ckpt-dir", default="experiments/vla_train")
    ap.add_argument("--resume", action="store_true", help="从 latest.pt 断点续训")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default=None,
                    help="D4 训练 jsonl (bigdata export_training_jsonl 产物); 缺省用 synth_batch")
    args = ap.parse_args()

    t_start = time.time()
    trainer = make_trainer(args.size, args.ckpt_dir, args.seed)
    cfg = trainer.cfg
    n_params = sum(p.numel() for p in trainer.model.parameters())

    train_batch = None
    val_batch = None
    if args.dataset:
        train_batch, val_batch, prov = make_replay_batcher(args.dataset, cfg)
        print(f"=== vla 训练 ({args.size}, {n_params/1e6:.2f}M params, torch {torch.__version__}) ===")
        print(f"[dataset] {args.dataset}")
        print(f"[dataset] 遥测字段: {prov.get('telemetry_fields', '?')} "
              f"(provenance={prov.get('telemetry_provenance', '?')}) | "
              f"合成占位: {prov.get('synthetic_fields', '?')}")
        print(f"[dataset] 生成方式: {prov.get('generation', '?')}")
        print(f"[dataset] 关节覆盖: {prov.get('joints', '?')} | 批次绑定: {prov.get('policy_binding', {}).get('batch_sn', '?')}")
    else:
        print(f"=== vla 训练 ({args.size}, {n_params/1e6:.2f}M params, torch {torch.__version__}) ===")
        print("[dataset] 未指定, 使用 synth_batch 合成回退")

    start_iter = 0
    if args.resume:
        latest = Path(args.ckpt_dir) / f"{cfg.ablation_name}_latest.pt"
        if latest.exists():
            info = trainer.load_checkpoint(str(latest))
            start_iter = info["iter"]
            print(f"[resume] 从 step {start_iter} 续训 (历史 loss {len(info['loss_log'])} 条, "
                  f"best_val={info['best_val_loss']:.4f})")
        else:
            print(f"[resume] 无可用检查点 ({latest}), 从头开始")

    end_iter = start_iter + args.iters
    print(f"计划: step {start_iter} → {end_iter}, checkpoint 间隔 {args.ckpt_interval}")

    losses = []
    for it in range(start_iter, end_iter):
        if train_batch is not None:
            frames, lang, state, target = train_batch()
        else:
            frames, lang, state, target = synth_batch(cfg)
        loss = trainer.train_step((frames, lang, state, target))
        losses.append(loss)
        if (it + 1) % args.eval_interval == 0:
            # val 集可用时记录真实 val loss (best_val_loss/loss_log 生效)
            if val_batch is not None:
                vf, vl, vs, vt = val_batch()
                with torch.no_grad():
                    _, vloss = trainer.model(vf, vl, vs, vt)
                trainer.loss_log.append({"iter": it + 1, "train": loss, "val": vloss.item()})
                if vloss.item() < trainer.best_val_loss:
                    trainer.best_val_loss = vloss.item()
            recent = losses[-args.eval_interval:]
            print(f"  step {it+1:4d} | loss {sum(recent)/len(recent):.4f} "
                  f"| 本段 {recent[0]:.4f}→{recent[-1]:.4f}"
                  + (f" | val {trainer.loss_log[-1]['val']:.4f} (best {trainer.best_val_loss:.4f})"
                     if trainer.loss_log else ""))
        if (it + 1) % args.ckpt_interval == 0 or it + 1 == end_iter:
            p = trainer.save_checkpoint()
            print(f"  [ckpt] step {it+1} → {p}")

    elapsed = time.time() - t_start
    print(f"=== 完成: {args.iters} 步, loss {losses[0]:.4f} → {losses[-1]:.4f}, "
          f"{elapsed:.1f}s ({elapsed/args.iters:.2f}s/step) ===")
    return 0 if losses[-1] < losses[0] else 1


if __name__ == "__main__":
    sys.exit(main())
