"""VLA Trainer — 端到端训练/评估/检查点系统。

复用 nanoGPT train.py 的 Trainer 选项:
  - 学习率 warmup + cosine decay
  - 梯度裁剪 + 混合精度 (bf16)
  - 周期性 eval loss 估计 + checkpoint save/load
  - ablation config 切换
  - 可复现 seed

数据流:
  for iter in range(max_iters):
    1. 取 batch (frames, lang, state, action_gt)
    2. forward → action_pred, loss
    3. backward → clip_grad_norm_ → optimizer.step
    4. eval_interval 步后估计验证 loss
    达 best_val 时存 checkpoint
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Optional

try:
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:

    from .action_tokenizer import ActionTokenizer
    from .vla_config import VLAConfig
    from .vla_model import VLAModel


    class VLATrainer:
        """VLA 训练器。

        参考 nanoGPT Trainer + eval.py 评测体系, 对齐字段名和工程化选项。
        """

        def __init__(self, cfg: VLAConfig, model: VLAModel,
                     train_dataset=None, val_dataset=None):
            self.cfg = cfg
            self.model = model.to(cfg.device)
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.optimizer = self._build_optimizer()
            self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_mixed_precision and torch.cuda.is_available())
            self.iter_num = 0
            self.best_val_loss = float("inf")
            self.loss_log: list[dict] = []

            # 随机种子 (对齐 nanoGPT set_seed)
            self._set_seed(cfg.seed, cfg.deterministic)

        def _set_seed(self, seed: int, deterministic: bool):
            import random
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        def _build_optimizer(self) -> AdamW:
            """AdamW + 分组 weight decay (bias/LayerNorm 不加)."""
            params = self.model.parameters()
            decay_params = [p for p in params if p.dim() >= 2]
            nodecay_params = [p for p in params if p.dim() < 2]
            return AdamW([
                {"params": decay_params, "weight_decay": self.cfg.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ], lr=self.cfg.learning_rate, betas=(self.cfg.beta1, self.cfg.beta2))

        def _lr_schedule(self, it: int) -> float:
            """Warmup + Cosine decay (对齐 nanoGPT)."""
            if it < self.cfg.warmup_iters:
                return self.cfg.max_lr * (it + 1) / self.cfg.warmup_iters
            if it > self.cfg.max_iters:
                return self.cfg.min_lr
            decay_ratio = (it - self.cfg.warmup_iters) / max(self.cfg.max_iters - self.cfg.warmup_iters, 1)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.cfg.min_lr + coeff * (self.cfg.max_lr - self.cfg.min_lr)

        @torch.no_grad()
        def estimate_loss(self, eval_iters: int | None = None) -> dict[str, float]:
            """对 train/val 采样若干 batch 估计平均 loss。"""
            iters = eval_iters or self.cfg.eval_iters
            out: dict[str, float] = {}
            self.model.eval()
            split_loader = {"train": self.train_dataset, "val": self.val_dataset}
            for split, ds in split_loader.items():
                if ds is None:
                    continue
                losses = []
                for _ in range(iters):
                    batch = ds.get_batch(self.cfg.batch_size) if hasattr(ds, "get_batch") else None
                    if batch is None:
                        break
                    frames, lang, state, action_gt = batch
                    frames = frames.to(self.cfg.device) if frames is not None else None
                    lang = lang.to(self.cfg.device) if lang is not None else None
                    state = state.to(self.cfg.device)
                    action_gt = action_gt.to(self.cfg.device)
                    with torch.amp.autocast("cuda", enabled=self.cfg.use_mixed_precision):
                        _, loss = self.model(frames, lang, state, action_gt)
                    losses.append(loss.item() if loss is not None else 0.0)
                out[split] = sum(losses) / max(len(losses), 1)
            self.model.train()
            return out

        def train_step(self, batch) -> float:
            """单步训练, 返回 loss 标量。"""
            frames, lang, state, action_gt = batch
            frames = frames.to(self.cfg.device) if frames is not None else None
            lang = lang.to(self.cfg.device) if lang is not None else None
            state = state.to(self.cfg.device)
            action_gt = action_gt.to(self.cfg.device)

            # LR schedule
            lr = self._lr_schedule(self.iter_num)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", enabled=self.cfg.use_mixed_precision):
                action_pred, loss = self.model(frames, lang, state, action_gt)

            loss_val = loss.item()
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.iter_num += 1
            return loss_val

        def train(self, get_batch_fn: Callable | None = None) -> dict:
            """完整训练循环。

            Args:
                get_batch_fn: () → (frames, lang, state, action_gt) batch tuple。
                              train/val 共用一个函数, 通过返回不同数据区分。
            Returns:
                dict: best_val_loss + final_iter + total_time_sec。
            """
            t_start = time.time()
            for it in range(self.cfg.max_iters):
                # Eval 检查点
                if it % self.cfg.eval_interval == 0 and it > 0:
                    losses = self.estimate_loss()
                    mean_val = losses.get("val", float("nan"))
                    mean_train = losses.get("train", float("nan"))
                    self.loss_log.append({"iter": it, "train": mean_train, "val": mean_val})
                    if mean_val < self.best_val_loss:
                        self.best_val_loss = mean_val
                        self.save_checkpoint(suffix="best")
                    print(f"[iter {it}] train={mean_train:.4f} val={mean_val:.4f}")

                # 训练 step — 要求 user 传入 get_batch_fn
                if get_batch_fn is not None:
                    batch = get_batch_fn("train") if callable(get_batch_fn) else None
                else:
                    batch = None
                if batch is None:
                    break
                loss = self.train_step(batch)

            t_elapsed = time.time() - t_start
            return {
                "best_val_loss": self.best_val_loss,
                "final_iter": self.iter_num,
                "total_time_sec": round(t_elapsed, 1),
            }

        def save_checkpoint(self, suffix: str = "last", path: str | None = None):
            """保存检查点 (模型权重 + 优化器状态 + iter 进度 + config)。"""
            if path is None:
                path = f"{self.cfg.ckpt_dir}/{self.cfg.ablation_name}_{suffix}.pt"
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iter": self.iter_num,
                "best_val_loss": self.best_val_loss,
                "config": self.cfg.to_json(),
                "loss_log": self.loss_log,
            }
            torch.save(ckpt, path)
            return path

        def load_checkpoint(self, path: str):
            """加载检查点 (恢复模型 + 优化器 + iter)。"""
            ckpt = torch.load(path, map_location=self.cfg.device)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.iter_num = ckpt["iter"]
            self.best_val_loss = ckpt["best_val_loss"]
            self.loss_log = ckpt.get("loss_log", [])
            return ckpt

else:
    class VLATrainer:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): raise RuntimeError("vla_trainer 需要 PyTorch")
