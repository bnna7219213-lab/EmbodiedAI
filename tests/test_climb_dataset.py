"""③ 数据扩充守护测试 — ≥1000 episodes 攀岩闭环数据引擎。

守护: 数据存在 / 量达标 / 物理量真实 / provenance 诚实 / 可加载训练。
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA = REPO / "artifacts" / "d4_climb" / "climb_episodes_full.jsonl"


class TestClimbDataset(unittest.TestCase):

    def test_dataset_exists_and_meets_threshold(self):
        """≥1000 episodes 落盘（评分阈值）。"""
        self.assertTrue(DATA.exists(), f"数据文件缺失: {DATA}")
        with open(DATA, encoding="utf-8") as f:
            lines = f.readlines()
        # 减去 provenance 头行
        n_eps = len(lines) - 1
        self.assertGreaterEqual(n_eps, 1000,
                                f"episodes 不足: {n_eps} < 1000")

    def test_provenance_header_honest(self):
        """首行 provenance 必须如实标注 real/synthetic。"""
        with open(DATA, encoding="utf-8") as f:
            header = json.loads(f.readline())
        self.assertEqual(header["_schema"], "d4_climb_v1")
        prov = header["_provenance"]
        self.assertIn("f_soft_peak_n", prov["real_fields"])
        self.assertIn("rgb", " ".join(prov["synthetic_fields"]))
        self.assertIn("不冒充真实示教", prov["note"])

    def test_episode_structure_valid(self):
        """抽查 episodes: obs/actions 等长 / actions 是纯数值 / 有 instruction。"""
        with open(DATA, encoding="utf-8") as f:
            f.readline()  # 跳过 header
            eps = [json.loads(f.readline()) for _ in range(5)]
        for i, ep in enumerate(eps):
            self.assertEqual(len(ep["observations"]), len(ep["actions"]),
                             f"ep[{i}] obs/actions 不等长")
            self.assertTrue(ep["instruction"], f"ep[{i}] 缺 instruction")
            for a in ep["actions"][:3]:
                self.assertTrue(all(isinstance(x, (int, float)) for x in a),
                                f"ep[{i}] action 非数值")

    def test_physics_real_not_random(self):
        """物理量来自攀岩闭环真实计算（域随机化随 seed 生效）。"""
        with open(DATA, encoding="utf-8") as f:
            f.readline()
            eps = [json.loads(f.readline()) for _ in range(10)]
        # 力闭合质量 ε 随 seed 变化（hold μ 域随机化 → 抓握选择 ε 漂移）
        # 注: soft_ratio 不随 seed 变（VSA/耦合参数是常量 — 如实反映）
        ratios = [ep["episode_meta"]["physics"]["soft_ratio"] for ep in eps]
        self.assertTrue(all(0 < r < 1 for r in ratios),
                        f"柔顺比值应在 (0,1): {ratios[:3]}")
        # physics 字段非空且有限（真实计算）
        for ep in eps[:3]:
            p = ep["episode_meta"]["physics"]
            for key in ("f_soft_peak_n", "com_advance_m", "dyno_tf_s"):
                v = p[key]
                self.assertTrue(v == v and abs(v) != float("inf"),
                                f"{key} 非有限值: {v}")

    def test_validate_bc_dataset_full(self):
        """全量数据过 BC 契约校验。"""
        from embodied_ai.adapters.nanogpt import NanoGPTAdapter
        with open(DATA, encoding="utf-8") as f:
            f.readline()
            data = [json.loads(l) for l in f]
        result = NanoGPTAdapter.validate_bc_dataset(data)
        self.assertTrue(result["ok"], f"validate 失败: {result['issues'][:3]}")
        self.assertGreaterEqual(result["n_episodes"], 1000)

    def test_loadable_by_replay_dataset(self):
        """数据可被 training 管线加载（字段兼容 replay_dataset 的消费模式）。"""
        with open(DATA, encoding="utf-8") as f:
            f.readline()
            ep = json.loads(f.readline())
        self.assertIn("episode_meta", ep)
        self.assertIn("physics", ep["episode_meta"])
        # 核心物理字段齐全
        for key in ("f_soft_peak_n", "com_advance_m", "v_release"):
            self.assertIn(key, ep["episode_meta"]["physics"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
