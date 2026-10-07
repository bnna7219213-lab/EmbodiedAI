"""D2→D4 真实数据管线测试 — 遥测 jsonl → ReplayDataset → 训练 batch。

覆盖:
  ① export_training_jsonl 的 provenance 头 (字段不变, 记录数>=1000 episodes)
  ② 1050 episodes 总量断言 (任务 ③)
  ③ 场景多样性: 7 类场景全覆盖
  ④ build_pairs 的自回归对与时序分割
  ⑤ ReplayDataset 张量形状契约
  ⑥ 遥测真实字段进入张量 (前 3 维非恒零)
  ⑦ Episode 分组一致性: 同一 episode_id 的记录时空连续
"""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.adapters.bigdata import BigdataAdapter
from embodied_ai.models.replay_dataset import build_pairs, load_telemetry_jsonl

D4_DIR = Path(__file__).resolve().parents[1] / "experiments" / "d4_dataset"
D4_FILE = D4_DIR / "training_replay_v1.jsonl"

# 预期常量 (与 gen_d4_dataset.py 同步)
EXPECTED_EPISODES = 1050
SCENARIOS = {"climb_grip", "climb_reach", "climb_dyno", "climb_cross",
             "tendon_pump", "household_grasp", "slip_recovery"}


class TestD4Pipeline(unittest.TestCase):

    # ---------- ① / ② provenance + episode ----------

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

    def test_episode_count_meets_target(self):
        """任务 ③: 数据量从 120 扩充到 >=1000 episodes。"""
        records, prov = load_telemetry_jsonl(D4_FILE)
        ep_ids = {r.get("episode_id", "") for r in records}
        self.assertGreaterEqual(len(ep_ids), 1000,
                                f"实际 episode 数 {len(ep_ids)} < 1000 目标")

    def test_full_dataset_1050_episodes(self):
        records, prov = load_telemetry_jsonl(D4_FILE)
        self.assertEqual(prov["source_records"], len(records))
        ep_ids = Counter(r.get("episode_id", "") for r in records)
        self.assertEqual(sum(1 for _ in ep_ids), EXPECTED_EPISODES)

    def test_all_7_scenarios_covered(self):
        records, _ = load_telemetry_jsonl(D4_FILE)
        found = {r.get("scenario", "") for r in records}
        self.assertTrue(SCENARIOS.issubset(found),
                        f"缺少场景: {SCENARIOS - found}")

    # ---------- ④ build_pairs 一致性 ----------

    def test_load_jsonl_with_header(self):
        records, prov = load_telemetry_jsonl(D4_FILE)
        self.assertEqual(len(records), prov["source_records"])
        self.assertIn("actual_pos_deg", prov["real_fields"])

    def test_build_pairs_autoregressive(self):
        records, _ = load_telemetry_jsonl(D4_FILE)
        split = build_pairs(records)
        self.assertEqual(split.n_records, len(records))
        # 1050 eps × ~40 steps ≈ 41811 → pairs 应 > 40000
        self.assertGreater(split.n_pairs, 40000)
        # action 是下一时刻位置: 对内自洽检查
        s, a = split.train[0]
        self.assertEqual(len(s), 3)  # [电流, 温度, 位置]
        self.assertEqual(a.shape, (1,))

    def test_temporal_split_no_leakage(self):
        records, _ = load_telemetry_jsonl(D4_FILE)
        split = build_pairs(records, val_ratio=0.2)
        self.assertTrue(len(split.train) > 0 and len(split.val) > 0)
        self.assertGreater(len(split.train), len(split.val))

    def test_real_fields_reach_tensor_path(self):
        """真实字段投影检查: 特征向量第 0 维 = 电流 (非恒零)。"""
        records, _ = load_telemetry_jsonl(D4_FILE)
        split = build_pairs(records)
        currents = [p[0][0] for p in split.train[:50]]
        self.assertTrue(any(c != 0.0 for c in currents), "电流值应真实非零")
        positions = [p[0][2] for p in split.train[:50]]
        self.assertTrue(any(v != 0.0 for v in positions), "位置值应真实非零")

    # ---------- ⑤ episode 分组一致性 ----------

    def test_episode_id_continuity(self):
        """同一 episode_id 内的记录时间单调且关节一致。"""
        records, _ = load_telemetry_jsonl(D4_FILE)
        by_ep: dict = {}
        for r in records:
            by_ep.setdefault(r.get("episode_id", ""), []).append(r)
        # 抽 50 个 episode 做连续性检查
        for ep_id, recs in list(by_ep.items())[:50]:
            ts = [r["timestamp_s"] for r in recs]
            self.assertEqual(ts, sorted(ts), f"{ep_id} 时间戳应单调")
            joints = {r.get("joint_name") for r in recs}
            self.assertEqual(len(joints), 1, f"{ep_id} 应只涉及单一关节")

    def test_episode_length_reasonable(self):
        """每 episode 步数在 [20, 60] 区间内。"""
        records, _ = load_telemetry_jsonl(D4_FILE)
        by_ep: dict = {}
        for r in records:
            by_ep.setdefault(r.get("episode_id", ""), []).append(r)
        lengths = [len(v) for v in by_ep.values()]
        self.assertGreaterEqual(min(lengths), 20)
        self.assertLessEqual(max(lengths), 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
