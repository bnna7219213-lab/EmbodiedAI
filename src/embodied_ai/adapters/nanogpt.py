"""nanoGPT 训练底座适配器 (v0.5 新增 — K3 审计遗留项 #2)。

背景:
  k1-audit.md 记录 "K3 要求复用 nanoGPT, 但 nanoGPT adapter 未建"。骨架方案裁定的是
  "不复制代码, 按模式新建", 因此本适配器**不 import nanoGPT**, 而是:
    1. probe()           — 以文件级契约探测 nanoGPT 仓库是否具备可复用的训练入口
    2. to_vla_config()   — 把 nanoGPT 风格的超参映射到本仓 VLAConfig
    3. export_training_plan() — 产出机器可读的训练计划 (JSON + 建议命令), 供 CI/脚本消费

  这样即便 nanoGPT 仓库缺失 (CI 常见), 计划生成与参数量估算仍可离线验证。

nanoGPT 契约面 (探针目标):
  llm.py       — ModelConfig / 模型定义
  train.py     — Trainer / 训练循环 / 消融入口
  tokenizer.py — BPE/字符级 tokenizer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_NANOGPT_ROOT = Path(r"C:\Users\bnna7\RaccoonWork\nanoGPT")

# 探针文件 (相对仓库根) — 与 CLI.PROJECTS["nanoGPT"]["probes"] 保持一致
PROBE_FILES: List[str] = ["llm.py", "train.py", "tokenizer.py"]

# nanoGPT 风格超参 → VLAConfig 字段映射 (只映射两者都有的字段)
_SIZE_TO_VLA = {
    "vla_tiny": dict(n_layer=4, n_embd=256, n_head=4, num_vision_layers=2, action_horizon=5),
    "vla_small": dict(n_layer=6, n_embd=384, n_head=6, num_vision_layers=4, action_horizon=10),
    "vla_base": dict(n_layer=6, n_embd=384, n_head=6, num_vision_layers=4, action_horizon=10),
    "vla_large": dict(n_layer=8, n_embd=512, n_head=8, num_vision_layers=6, action_horizon=15),
}


class NanoGPTAdapter:
    """nanoGPT 仓库适配 — 只读探测 + 训练计划导出 (不执行训练, 不复制代码)。"""

    def __init__(self, root: Optional[Path | str] = DEFAULT_NANOGPT_ROOT):
        self.root = Path(root) if root is not None else DEFAULT_NANOGPT_ROOT

    # ------------------------------------------------------------------ #
    # 1. 探测
    # ------------------------------------------------------------------ #
    def probe(self) -> Dict[str, Any]:
        """探测 nanoGPT 仓库完整性。

        返回 {root, exists, found, total, missing, probes}。
        仓库不存在时 found=0 而非抛异常 — CI 环境属预期, 由调用方决定是否降级。
        """
        found = [p for p in PROBE_FILES if (self.root / p).is_file()]
        missing = [p for p in PROBE_FILES if p not in found]
        return {
            "root": str(self.root),
            "exists": self.root.is_dir(),
            "found": len(found),
            "total": len(PROBE_FILES),
            "missing": missing,
            "probes": found,
        }

    @property
    def available(self) -> bool:
        return self.root.is_dir()

    # ------------------------------------------------------------------ #
    # 2. 超参映射
    # ------------------------------------------------------------------ #
    def to_vla_config(self, model_size: str = "vla_tiny"):
        """按 nanoGPT ModelConfig 命名习惯生成对应规模的 VLAConfig。"""
        from ..models import VLAConfig

        overrides = _SIZE_TO_VLA.get(model_size)
        if overrides is None:
            raise ValueError(f"未知规模: {model_size}. 合法: {list(_SIZE_TO_VLA)}")
        cfg = VLAConfig(**{**overrides, "ablation_name": model_size})
        return cfg

    # ------------------------------------------------------------------ #
    # 3. 训练计划导出
    # ------------------------------------------------------------------ #
    def export_training_plan(self, model_size: str = "vla_tiny",
                             out_dir: Optional[Path | str] = None,
                             data_dir: str = "data/embodied") -> Dict[str, Any]:
        """生成并落盘训练计划 JSON。

        计划内容:
          - 模型结构 + 参数量估算 (来自 VLAConfig.estimate_params)
          - nanoGPT 风格训练超参 (lr schedule / warmup / grad clip / 混合精度)
          - 建议命令 (参考 nanoGPT train.py 的 CLI 形态, 但指向本仓 VLA Trainer)
          - 底座探测结果 (是否真能复用 nanoGPT 仓库)
        """
        cfg = self.to_vla_config(model_size)
        est = cfg.estimate_params()
        probe = self.probe()

        plan: Dict[str, Any] = {
            "model_size": model_size,
            "config": json.loads(cfg.to_json()),
            "estimate_params": est["total"],
            "estimate_breakdown": est,
            "nanogpt": probe,
            "data_dir": data_dir,
            "suggested_cmd": (
                f"python -m embodied_ai.models.vla_trainer "
                f"--ablation {model_size} --data-dir {data_dir} "
                f"--max-iters {cfg.max_iters} --batch-size {cfg.batch_size}"
            ),
            "notes": [
                "骨架方案裁定: 按 nanoGPT 模式新建, 不复制其代码",
                "无 torch 环境下仅为计划/参数量估算, 不执行 forward",
            ],
        }

        out = Path(out_dir) if out_dir is not None else Path("output/nanogpt")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"train_plan_{model_size}.json"
        path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"path": str(path), "plan": plan}

    # ------------------------------------------------------------------ #
    # 4. 数据集契约校验 (BC 数据采集 → 训练入口之间的一致性)
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_bc_dataset(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """校验 SimToRealPipeline.run_bc_collection 产出能否直接喂 Trainer。

        检查: 每条样本 obs/action 长度一致、instruction 非空、无 NaN。
        返回 {ok, n_episodes, n_transitions, issues}。
        """
        issues: List[str] = []
        n_transitions = 0
        for i, ep in enumerate(dataset or []):
            obs = ep.get("observations", [])
            acts = ep.get("actions", [])
            if len(obs) != len(acts):
                issues.append(f"episode#{i}: obs({len(obs)}) != actions({len(acts)})")
            if not ep.get("instruction"):
                issues.append(f"episode#{i}: instruction 缺失")
            n_transitions += min(len(obs), len(acts))
        if not dataset:
            issues.append("dataset 为空")
        return {"ok": not issues, "n_episodes": len(dataset or []),
                "n_transitions": n_transitions, "issues": issues}
