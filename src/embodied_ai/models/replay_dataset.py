"""重放数据集 — bigdata D2 遥测 → D4 训练张量 (真实 dataset loader)。

数据流 (K3++ 真数据接入, 替换 synth_batch):
  TelemetryRecord (D2: joint/电流/温度/位置/故障) jsonl
    → ReplayDataset (torch Dataset)
      → 特征-动作对: state = [电流, 温度, 位置] (遥测真实字段),
        action = 下一时刻 actual_pos (自回归监督: 从当前遥测预测下一步)
      → 模态诚实性 (provenance):
          rgb / lang 遥测中不存在 → 确定性零张量 + provenance="synthetic_zero"
          不冒充真实数据; 真视觉/语言接入后 provenance 变 "real"

时序分割: train/val 按时间前 80%/后 20% (不随机打乱, 防时序泄漏)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 实测 source 词汇表单一真源在数据接入层 —— 此前本模块与 bigdata 的导出器各写
# 一套 ("real"/"telemetry"/"measured" vs 无条件 real_fields), 于是同一份数据在
# 两处得到相反的 provenance 结论。
from ..adapters.bigdata import MEASURED_SOURCES

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# D2 遥测记录 → 特征向量的固定投影 (电流/温度/位置 3 字段 × 关节)
FEATURE_FIELDS = ("motor_current_a", "temperature_c", "actual_pos_deg")


def _record_to_feature(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    """单条遥测 → [3] 特征 (None 表示字段缺失不可用)。"""
    vals = [rec.get(f) for f in FEATURE_FIELDS]
    if any(v is None for v in vals):
        return None
    return np.asarray([float(v) for v in vals], dtype=np.float32)


def _records_to_matrix(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """同关节连续记录 → [T, 3] 矩阵; 帧级 provenance。"""
    feats, prov = [], []
    for rec in records:
        f = _record_to_feature(rec)
        if f is None:
            continue
        feats.append(f)
        prov.append(rec.get("source", "unknown"))
    return np.stack(feats) if feats else np.zeros((0, 3), dtype=np.float32), prov


@dataclass
class ReplaySplit:
    """train/val 划分结果。"""
    train: List[Tuple[np.ndarray, np.ndarray]]   # (state, action) 对
    val: List[Tuple[np.ndarray, np.ndarray]]
    provenance: Dict[str, Any]
    n_records: int
    n_pairs: int


def build_pairs(records: List[Dict[str, Any]], horizon: int = 1,
                val_ratio: float = 0.2) -> ReplaySplit:
    """遥测记录 → (state, action) 自回归对 + 时序分割。

    action = 未来 horizon 步的实际位置 (从遥测位置序列直接构成监督信号)。
    """
    # 按关节分组, 组内按时间排序
    by_joint: Dict[str, List[Dict]] = {}
    for r in records:
        by_joint.setdefault(r.get("joint_name", "unknown"), []).append(r)
    for j in by_joint:
        by_joint[j].sort(key=lambda r: r.get("timestamp_s", 0))

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    joint_counts: Dict[str, int] = {}
    for joint, recs in by_joint.items():
        mat, prov = _records_to_matrix(recs)
        T = len(mat)
        joint_counts[joint] = T
        # 自回归对: (x_t, pos_{t+horizon})
        pos = mat[:, FEATURE_FIELDS.index("actual_pos_deg")]
        for t in range(T - horizon):
            pairs.append((mat[t].copy(), np.asarray([pos[t + horizon]], dtype=np.float32)))

    # 时序分割: 全局前 80% train, 后 20% val (不打乱)
    n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 4 else 0
    split_idx = len(pairs) - n_val
    train, val = pairs[:split_idx], pairs[split_idx:]

    # P0 修复: real_fields 应按 source 判定, 而非"字段非 None"。
    # 原代码把 source="unknown" 或 source="synthetic" 的记录也计入 real_fields,
    # 让训练误以为有真实遥测。修复: 只统计 MEASURED_SOURCES 内的 source。
    real_field_set = set()
    for r in records:
        src = str(r.get("source", "")).lower()
        if src in MEASURED_SOURCES:
            for f in FEATURE_FIELDS:
                if r.get(f) is not None:
                    real_field_set.add(f)
    real_fields = sorted(real_field_set)
    synthetic_only_fields = [f for f in FEATURE_FIELDS if f not in real_fields]
    return ReplaySplit(
        train=train, val=val,
        provenance={
            "feature_fields": FEATURE_FIELDS,
            "real_fields": real_fields,
            "telemetry_provenance": "measured" if real_fields else "synthetic",
            "synthetic_only_fields": synthetic_only_fields,
            "synthetic_fields": ["rgb", "lang"],
            "note": ("rgb/lang 为确定性零张量占位 (遥测中不存在), provenance=synthetic_zero; "
                     "训练时应按 real_fields 判断真实数据覆盖度, synthetic_only_fields 为空才表示全真"),
            "joints": joint_counts,
        },
        n_records=len(records), n_pairs=len(pairs),
    )


if HAS_TORCH:

    class ReplayDataset(Dataset):
        """(state, action) 对的 torch Dataset — vla_train_full 真实 loader。

        模态扩展: state_vec → VLA 的 state 投影输入; rgb/lang 确定性零张量。
        action_dim 由 horizon 决定 (当前=1, 标量位置)。
        """

        def __init__(self, pairs: List[Tuple[np.ndarray, np.ndarray]],
                     cfg=None, img_size: int = 224):
            self.pairs = pairs
            self.cfg = cfg
            self.img_size = img_size
            self.n_views = getattr(cfg, "num_views", 3) if cfg else 3
            self.state_dim = getattr(cfg, "state_dim", 12) if cfg else 12
            self.action_dim = getattr(cfg, "action_dim", 6) if cfg else 6
            self.action_horizon = getattr(cfg, "action_horizon", 8) if cfg else 8
            self.lang_len = 8
            self.vocab = getattr(cfg, "vocab_size", 256) if cfg else 256

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, i: int):
            state_raw, action_raw = self.pairs[i]
            # state: 遥测 [3] → VLA state_dim (零填充, 前三维真实遥测)
            state = np.zeros(self.state_dim, dtype=np.float32)
            n = min(len(state_raw), self.state_dim)
            state[:n] = state_raw[:n]
            # action: [1] → [horizon, action_dim] (位置放第 0 维, 其余零)
            action = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
            action[0, 0] = action_raw[0]
            # rgb/lang: 确定性零张量 (provenance=synthetic_zero, 模态诚实)
            frames = np.zeros((self.n_views, 3, self.img_size, self.img_size), dtype=np.float32)
            lang = np.zeros(self.lang_len, dtype=np.int64)
            return (torch.from_numpy(frames), torch.from_numpy(lang),
                    torch.from_numpy(state), torch.from_numpy(action))

    def make_loaders(pairs_train, pairs_val, cfg=None, batch_size: int = 2,
                     workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        ds_tr = ReplayDataset(pairs_train, cfg)
        ds_va = ReplayDataset(pairs_val, cfg)
        tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=workers)
        va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=workers) \
            if len(ds_va) else None
        return tr, va

else:
    class ReplayDataset:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): raise RuntimeError("replay_dataset 需要 PyTorch")


def load_telemetry_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """读 D4 训练 jsonl (BigdataAdapter.export_training_jsonl 格式)。

    返回 (记录行, provenance 头); 无头行时 provenance 为空 dict (兼容裸 D2 dump)。
    """
    records: List[Dict[str, Any]] = []
    provenance: Dict[str, Any] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0 and "_schema" in obj:
                provenance = obj.get("_provenance", {})
                continue
            records.append(obj)
    return records, provenance
