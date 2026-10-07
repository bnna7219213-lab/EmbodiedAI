"""nanoGPT adapter — 模型训练底座桥接 (K3+)。

skeleton-plan §3.3 边界: "不复制代码，按模式新建（保持两仓库独立）"。
本 adapter 落地该边界的机器化:
  1. 结构映射: VLAConfig ↔ nanoGPT ModelConfig 的字段对照表 (训练超参/工程化选项对齐)
  2. 底座探测: nanoGPT 仓库存在性 + llm.py 结构 + checkpoints 状态 (真实扫描, 非硬编码)
  3. 语言词表: nanoGPT 训练好的 tokenizer 可作为 VLA language 嵌入初始化的语料来源
  4. Trainer 模式迁移检查: VLAConfig 的训练字段命名与 nanoGPT 对齐性审计

不做: 直接 import llm.py (跨仓库耦合) — 经文件系统/子进程只读交互。
"""

from __future__ import annotations

import json
from dataclasses import asdict as dc_asdict
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional

NANOGPT_ROOT = Path(r"C:\Users\bnna7\RaccoonWork\nanoGPT")

# VLAConfig 字段 → nanoGPT ModelConfig 字段的语义映射 (训练/工程化对齐审计用)
FIELD_MAPPING = {
    # 结构
    "n_layer": "n_layer",
    "n_embd": "n_embd",
    "n_head": "n_head",
    "block_size": "block_size",
    "vocab_size": "vocab_size",
    "dropout": "dropout",
    # 训练 (VLAConfig 训练字段 → nanoGPT Trainer 侧)
    "batch_size": "batch_size",
    "max_iters": "max_iters",
    "learning_rate": "learning_rate",
    "eval_interval": "eval_interval",
    "weight_decay": "weight_decay",
    "grad_clip": "grad_clip",
    "seed": "seed",
}

# nanoGPT 有而 VLA 侧暂无的工程化能力 (M2 训练工程化) — 待吸收清单
NANOGPT_ENGINEERING_FEATURES = [
    "use_sdpa", "use_mixed_precision", "autocast_enabled",
    "grad_scaler_enabled", "ce_chunk_size", "ckpt_interval", "resume_from",
]


class NanoGPTAdapter:
    """nanoGPT 训练底座 (llm.py) 的只读桥接。

    用法:
        ad = NanoGPTAdapter()
        ad.probe()                      # → {"exists": True, "checkpoints": [...], ...}
        ad.structural_diff(vla_cfg)     # → VLA ↔ nanoGPT 字段对齐报告
        ad.latest_checkpoint()          # → Path | None
    """

    def __init__(self, root: Path = None):
        self.root = Path(root) if root is not None else NANOGPT_ROOT

    # ------------------------------------------------------------------ #
    # 底座探测 (真实文件系统扫描; found/total/missing 供 CLI 摘要)
    # ------------------------------------------------------------------ #
    def probe(self) -> Dict[str, Any]:
        r: Dict[str, Any] = {"root": str(self.root), "exists": self.root.is_dir()}
        # 契约三件套 (CLI/自检摘要的 found/total 口径)
        wanted = {
            "llm.py": (self.root / "llm.py"),
            "train.py": (self.root / "train.py"),
            "tokenizer.py": (self.root / "tokenizer.py"),
        }
        # 可选资产: 存在与否不影响"底座是否可用"的判定, 单独列出
        optional = {
            "checkpoints/latest.pt": (self.root / "checkpoints" / "latest.pt"),
            "data/bin": (self.root / "data" / "bin"),
        }
        r["total"] = len(wanted)
        found = {k: p.exists() for k, p in wanted.items()}
        r["found"] = sum(found.values())
        r["missing"] = [k for k, ok in found.items() if not ok]
        r["optional_present"] = [k for k, p in optional.items() if p.exists()]
        r["optional_missing"] = [k for k, p in optional.items() if not p.exists()]
        if not r["exists"]:
            return r
        llm = self.root / "llm.py"
        r["llm_py"] = llm.exists()
        r["llm_size_kb"] = llm.stat().st_size // 1024 if llm.exists() else 0
        # 结构扫描: class/dataclass 定义
        if llm.exists():
            src = llm.read_text(encoding="utf-8", errors="replace")
            r["has_model_config"] = "class ModelConfig" in src
            r["has_gpt_model"] = "class GPTLanguageModel" in src
            r["has_trainer"] = "class Trainer" in src
        # checkpoints
        ckpt_dir = self.root / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.is_dir() else []
        r["checkpoints"] = [p.name for p in ckpts]
        r["latest_checkpoint"] = str(ckpt_dir / "latest.pt") if (ckpt_dir / "latest.pt").exists() else None
        # config.json (模型配置快照)
        cfg_json = ckpt_dir / "config.json"
        r["trained_config"] = json.loads(cfg_json.read_text(encoding="utf-8")) if cfg_json.exists() else None
        # tokenizer / 数据
        r["has_tokenizer"] = (self.root / "tokenizer.py").exists()
        r["bin_data"] = [p.name for p in (self.root / "data" / "bin").glob("*.bin")] \
            if (self.root / "data" / "bin").is_dir() else []
        return r

    # ------------------------------------------------------------------ #
    # 训练计划导出 (CLI 消费: export_training_plan → {path, plan})
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_vla_config(model_size: str):
        """消融档位名 → VLAConfig。未知档位抛 ValueError (不做静默回退)。

        静默回退到 vla_tiny 会让"写错档位名"变成"训练了一个小模型", 排查成本极高。
        """
        from ..models.vla_config import VLAConfig  # 延迟, 避免环
        abl = VLAConfig.ablations()
        if model_size not in abl:
            raise ValueError(f"未知消融档位: {model_size} "
                             f"(可用: {', '.join(sorted(abl))})")
        return abl[model_size]

    def export_training_plan(self, model_size: str = "vla_tiny",
                              out_dir: Path = Path("output/nanogpt")) -> Dict[str, Any]:
        """按 VLA 消融档位生成 nanoGPT 底座训练计划 (含参数量估算与建议命令)。"""
        cfg = self.to_vla_config(model_size)
        est = cfg.estimate_params()["total"]
        pack = self.to_nanogpt_training_pack(cfg)
        plan = {
            "model_size": model_size,
            "estimate_params": est,
            # 完整配置快照: 训练计划要能独立复现, 不能只有几个字段
            "config": dc_asdict(cfg),
            "suggested_cmd": (f"python llm.py --n-layer {pack['n_layer']} --n-embed {pack['n_embd']} "
                              f"--n-head {pack['n_head']} --iters {pack.get('max_iters', 2000)}"),
            "training_pack": pack,
            "probe_root": str(self.root),
        }
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"train_plan_{model_size}.json"
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"path": str(path), "plan": plan}

    def latest_checkpoint(self) -> Optional[Path]:
        ckdir = self.root / "checkpoints"
        if not ckdir.is_dir():
            return None
        latest = ckdir / "latest.pt"
        if latest.exists():
            return latest
        pts = sorted(ckdir.glob("*.pt"))
        return pts[-1] if pts else None

    # ------------------------------------------------------------------ #
    # 结构对齐审计: VLAConfig ↔ ModelConfig
    # ------------------------------------------------------------------ #
    def structural_diff(self, vla_cfg) -> Dict[str, Any]:
        """对照 VLAConfig 实例与 nanoGPT ModelConfig 的字段语义映射。"""
        # nanoGPT ModelConfig 字段名 (dataclass 定义实扫)
        llm = self.root / "llm.py"
        nanogpt_fields: set = set()
        if llm.exists():
            src = llm.read_text(encoding="utf-8", errors="replace")
            in_cfg = False
            for line in src.splitlines():
                if line.startswith("class ModelConfig"):
                    in_cfg = True
                    continue
                if in_cfg:
                    s = line.strip()
                    if s.startswith("def ") or (s and not s.startswith(("#", "@"))):
                        if line and not line[0].isspace():
                            in_cfg = False  # 类体结束
                        elif ":" in s and not s.startswith("#"):
                            name = s.split(":")[0].strip()
                            if name.isidentifier():
                                nanogpt_fields.add(name)
        vla_fields = {f.name for f in dc_fields(vla_cfg)}
        mapped_present = {v: k for k, v in FIELD_MAPPING.items() if v in nanogpt_fields}
        return {
            "vla_fields": sorted(vla_fields),
            "nanogpt_fields": sorted(nanogpt_fields),
            "shared_semantics": {k: v for k, v in FIELD_MAPPING.items() if k in vla_fields and v in nanogpt_fields},
            "vla_only": sorted(vla_fields - set(FIELD_MAPPING.keys())),
            "nanogpt_engineering_missing_in_vla": [f for f in NANOGPT_ENGINEERING_FEATURES if f in nanogpt_fields],
            "mapping_coverage": len([1 for k, v in FIELD_MAPPING.items() if k in vla_fields and v in nanogpt_fields])
                                 / max(len(FIELD_MAPPING), 1),
        }

    # ------------------------------------------------------------------ #
    # 训练配置迁移: VLA 消融 → nanoGPT 兼容的训练超参包
    # ------------------------------------------------------------------ #
    def to_nanogpt_training_pack(self, vla_cfg, note: str = "") -> Dict[str, Any]:
        """生成 nanoGPT Trainer 可消费的训练超参包 (字段名按 ModelConfig)。"""
        pack: Dict[str, Any] = {}
        for vla_name, ng_name in FIELD_MAPPING.items():
            if hasattr(vla_cfg, vla_name):
                pack[ng_name] = getattr(vla_cfg, vla_name)
        pack["_source"] = f"embodied_ai VLAConfig ({getattr(vla_cfg, 'ablation_name', 'unnamed')})"
        pack["_note"] = note or "VLA 语言骨干预训练超参 (经 nanoGPT 底座)"
        return pack

    def tokenizer_vocab(self) -> Optional[Dict[str, Any]]:
        """读取 nanoGPT 训练好的 tokenizer json (VLA language 嵌入语料来源)。"""
        for cand in ["data/bin/shakespeare_m6_tokenizer.json", "checkpoints/tokenizer.json"]:
            p = self.root / cand
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return None

    # ------------------------------------------------------------------ #
    # BC 数据集契约校验 (采集侧 → 训练侧的闸门)
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_bc_dataset(dataset: Any, require_instruction: bool = True) -> Dict[str, Any]:
        """校验行为克隆数据集是否符合 VLA/BC 训练管线的输入契约。

        契约 (对齐 SimToRealPipeline.run_bc_collection 的产出格式):
          - 顶层为非空 list, 每条是一个 episode dict;
          - 每条含 observations / actions, 且**长度相等** (否则无法监督);
          - 每个 action 必须是有限数值 (NaN/Inf 会让 loss 变 nan);
          - require_instruction 时 instruction 非空字符串。

        返回 {"ok": bool, "issues": [...], "n_episodes": int, "n_steps": int}。
        """
        issues: List[str] = []
        if not isinstance(dataset, (list, tuple)):
            return {"ok": False, "issues": ["数据集必须是 episode 列表"],
                    "n_episodes": 0, "n_steps": 0}
        if len(dataset) == 0:
            return {"ok": False, "issues": ["数据集为空"], "n_episodes": 0, "n_steps": 0}

        n_steps = 0
        for i, ep in enumerate(dataset):
            if not isinstance(ep, dict):
                issues.append(f"episode[{i}] 不是 dict")
                continue
            if "observations" not in ep:
                issues.append(f"episode[{i}] 缺 observations")
                continue
            if "actions" not in ep:
                issues.append(f"episode[{i}] 缺 actions")
                continue
            obs, acts = ep["observations"], ep["actions"]
            try:
                lo, la = len(obs), len(acts)
            except TypeError:
                issues.append(f"episode[{i}] observations/actions 不可计长")
                continue
            if lo != la:
                issues.append(f"episode[{i}] 观测/动作长度不等: {lo} vs {la}")
            n_steps += min(lo, la)
            for j, a in enumerate(acts):
                vals = a if isinstance(a, (list, tuple)) else [a]
                for v in vals:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        issues.append(f"episode[{i}].actions[{j}] 非数值: {v!r}")
                        continue
                    if not (fv == fv and abs(fv) != float("inf")):
                        issues.append(f"episode[{i}].actions[{j}] 非有限值: {v!r}")
            if require_instruction and not str(ep.get("instruction", "")).strip():
                issues.append(f"episode[{i}] instruction 为空")
            # 只报前若干条, 避免刷屏
            if len(issues) >= 20:
                issues.append("... (问题过多, 已截断)")
                break

        return {"ok": len(issues) == 0, "issues": issues,
                "n_episodes": len(dataset), "n_steps": n_steps}
