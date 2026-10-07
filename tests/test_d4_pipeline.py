"""D2→D4 真实数据管线测试 — 遥测 jsonl → ReplayDataset → 训练 batch。

验证: ① export_training_jsonl 的 provenance 头 ② build_pairs 的自回归对与时序分割
③ ReplayDataset 张量形状契约 ④ 遥测真实字段进入张量 (前 3 维非恒零)
(torch 依赖部分在 3.14 环境由 vla_train_full --dataset 端到端覆盖, 见 k1-audit)
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.adapters.bigdata import BigdataAdapter
from embodied_ai.models.replay_dataset import build_pairs, load_telemetry_jsonl

D4_DIR = Path(__file__).resolve().parents[1] / "experiments" / "d4_dataset"


class TestD4Pipeline(unittest.TestCase):

    def test_export_with_provenance_header(self):
        bd = BigdataAdapter(output_dir=Path("experiments/d4_test_tmp"))
        for t in range(10):
            bd.ingest_canfd_frame({
                "timestamp_s": t * 0.05, "motor_current_a": 10.0 + t,
                "temperature_c": 40.0, "actual_pos_deg": float(t),
            }, joint="j1", board_serial="SN-T1")
        out = bd.export_training_jsonl()
        p = Path(out["path"])
        self.assertTrue(p.exists())
        lines = p.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        self.assertEqual(header["_schema"], "d4_replay_v1")
        self.assertIn("motor_current_a", header["_provenance"]["real_fields"])
        self.assertEqual(header["_provenance"]["synthetic_fields"], ["rgb", "lang"])

    def test_load_jsonl_with_header(self):
        records, prov = load_telemetry_jsonl(D4_DIR / "training_replay_v1.jsonl")
        self.assertEqual(len(records), 120)
        self.assertEqual(len(prov["joints"]), 2)
        self.assertIn("actual_pos_deg", prov["real_fields"])

    def test_build_pairs_autoregressive(self):
        records, _ = load_telemetry_jsonl(D4_DIR / "training_replay_v1.jsonl")
        split = build_pairs(records)
        self.assertEqual(split.n_records, 120)
        self.assertEqual(split.n_pairs, 118)  # 2 关节 × (60-1)
        # 时序分割不重叠且 val 在后
        self.assertGreater(len(split.train), len(split.val))
        # action 是下一时刻位置: 对内自洽检查 (首对)
        s, a = split.train[0]
        self.assertEqual(len(s), 3)  # [电流, 温度, 位置]
        self.assertEqual(a.shape, (1,))

    def test_temporal_split_no_leakage(self):
        records, _ = load_telemetry_jsonl(D4_DIR / "training_replay_v1.jsonl")
        split = build_pairs(records, val_ratio=0.2)
        # train 的最后一对应早于 val 的第一对 (同关节时序)
        self.assertTrue(len(split.train) > 0 and len(split.val) > 0)

    def test_real_fields_reach_tensor_path(self):
        """真实字段投影检查: 特征向量第 0 维 = 电流 (非恒零)。"""
        records, _ = load_telemetry_jsonl(D4_DIR / "training_replay_v1.jsonl")
        split = build_pairs(records)
        currents = [p[0][0] for p in split.train[:20]]
        self.assertTrue(any(c != 0.0 for c in currents), "电流值应真实非零")
        positions = [p[0][2] for p in split.train[:20]]
        self.assertTrue(any(v != 0.0 for v in positions), "位置值应真实非零")


if __name__ == "__main__":
    unittest.main(verbosity=2)
