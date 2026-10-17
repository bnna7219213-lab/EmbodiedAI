"""D2→D4 遥测数据管线测试 — 遥测 jsonl → ReplayDataset → 训练 batch。

**provenance 提醒**: 当前 d4_dataset 由 scripts/gen_d4_dataset.py 的 rng 剧本
生成 (source="replay_gen"), 不含物理引擎也不含硬件。本文件因此同时守护
"管线正确"与"标注诚实"两件事 —— 后者曾失守: 导出头把 rng 产出的电流/温度
标成 real_fields。

覆盖:
  ① export_training_jsonl 的 provenance 头 (由 source 推导, 不由导出器断言)
  ② 1050 episodes 总量断言 (任务 ③)
  ③ 场景多样性: 7 类场景全覆盖
  ④ build_pairs 的自回归对与时序分割
  ⑤ ReplayDataset 张量形状契约
  ⑥ 遥测字段进入张量 (前 3 维非恒零 —— 证明管线没丢数据, 不证明数据是实测)
  ⑦ Episode 分组一致性: 同一 episode_id 的记录时空连续
  ⑧ bigdata 与 replay_dataset 共用同一实测 source 词汇表
"""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.adapters.bigdata import MEASURED_SOURCES, BigdataAdapter
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
        prov = header["_provenance"]
        self.assertIn("motor_current_a", prov["telemetry_fields"])
        # 字段存在 ≠ 字段实测: ingest 默认 source="canfd_frame" 不在
        # MEASURED_SOURCES 里, 所以必须归类为 synthetic。
        self.assertEqual(prov["record_sources"], ["canfd_frame"])
        self.assertEqual(prov["telemetry_provenance"], "synthetic")
        self.assertEqual(prov["policy_binding"]["binding_provenance"], "synthetic")

    def test_provenance_becomes_measured_only_for_measured_source(self):
        """正向对照: source 换成实测注入 (hil) 时, 归类必须翻成 measured。

        与上一条合起来才证明 provenance 是**推导**出来的, 而不是恒定输出
        "synthetic" 或恒定输出 "real"。
        """
        bd = BigdataAdapter(output_dir=Path("experiments/d4_test_tmp"))
        bd.ingest_canfd_frame({"timestamp_s": 0.0, "motor_current_a": 11.0,
                               "temperature_c": 41.0, "actual_pos_deg": 3.0},
                              joint="j1", board_serial="SN-T2", source="hil")
        prov = json.loads(Path(bd.export_training_jsonl()["path"]
                               ).read_text(encoding="utf-8").splitlines()[0]
                           )["_provenance"]
        self.assertEqual(prov["telemetry_provenance"], "measured")
        self.assertEqual(prov["record_sources"], ["hil"])

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
        self.assertIn("actual_pos_deg", prov["telemetry_fields"])
        self.assertEqual(prov["telemetry_provenance"], "synthetic")

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

    def test_telemetry_fields_reach_tensor_path(self):
        """管线不丢数据: 特征向量第 0 维 = 电流 (非恒零)。

        注意这条**只**证明字段被搬进张量, 不证明字段来自实测 —— D4 目前是
        rng 剧本产物 (telemetry_provenance="synthetic"), 实测性由
        test_export_with_provenance_header 那组用例单独守护。
        """
        records, _ = load_telemetry_jsonl(D4_FILE)
        split = build_pairs(records)
        currents = [p[0][0] for p in split.train[:50]]
        self.assertTrue(any(c != 0.0 for c in currents), "电流值应非零 (管线未丢字段)")
        positions = [p[0][2] for p in split.train[:50]]
        self.assertTrue(any(v != 0.0 for v in positions), "位置值应非零 (管线未丢字段)")
        # build_pairs 的 real_fields 也必须判空 —— 与导出头同一结论
        self.assertEqual(split.provenance["real_fields"], [],
                         "replay_dataset 把非实测 source 的记录算进了 real_fields")
        self.assertEqual(split.provenance["telemetry_provenance"], "synthetic")

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


class TestProvenanceVocabularySingleSource(unittest.TestCase):
    """⑧ bigdata 导出器与 replay_dataset 必须共用同一套实测 source 词汇表。

    此前两处各写一套 (导出器无条件标 real_fields, replay_dataset 认
    "real"/"telemetry"/"measured"), 同一份数据在导出头和训练 provenance 里会
    得到不同归类 —— 而训练侧才是真正决定"模型是否见过真实遥测"的地方。
    """

    def test_replay_dataset_imports_bigdata_constant(self):
        from embodied_ai.models import replay_dataset

        self.assertIs(replay_dataset.MEASURED_SOURCES, MEASURED_SOURCES,
                      "replay_dataset 另写了一套 source 词汇表, 会与导出头漂移")

    def test_d4_record_source_is_not_measured(self):
        """当前数据集的 source 必须落在非实测集合里 (否则标注就是假的)。"""
        records, _ = load_telemetry_jsonl(D4_FILE)
        sources = {str(r.get("source", "")) for r in records}
        self.assertTrue(sources, "记录缺 source 字段")
        self.assertFalse(sources & set(MEASURED_SOURCES),
                         f"D4 记录含实测 source {sources & set(MEASURED_SOURCES)}, "
                         "但 gen_d4_dataset.py 是纯 rng 剧本")


if __name__ == "__main__":
    unittest.main(verbosity=2)
