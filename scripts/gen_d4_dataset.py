"""生成 D4 训练数据集 — 从合成遥测源 (演示端到端; 真机遥测替换 ingest 即同管线)。

流程: BigdataAdapter.ingest (D2) → export_training_jsonl (D4) → vla_train_full --dataset
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.adapters.bigdata import BigdataAdapter


def main() -> int:
    bd = BigdataAdapter(output_dir=Path("experiments/d4_dataset"))
    rng = np.random.default_rng(42)
    # 两个关节 × 60 时刻的模拟遥测 (电流/温度/位置连贯时序, 位置带趋势)
    for joint in ("shoulder_L", "elbow_L"):
        pos = 0.0
        for t in range(60):
            pos += rng.normal(0.05, 0.3)  # 位置随机游走 (自回归监督信号)
            bd.ingest_canfd_frame({
                "timestamp_s": float(t) * 0.05,
                "actual_current_a": float(10.0 + rng.normal(0, 1.2)),
                "temp_c": float(40.0 + t * 0.1 + rng.normal(0, 0.5)),
                "actual_pos_deg": float(np.clip(pos, -90, 90)),
                "fault_events": [],
            }, joint=joint, board_serial="SN-D4-001", source="replay_gen")
    out = bd.export_training_jsonl()
    print(f"D4 数据集: {out['path']}")
    print(f"记录数: {out['n_records']} (2 关节 × 60 时刻)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
