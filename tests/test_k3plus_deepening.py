"""K3+ 纵向深化测试 — VLA 投影桥 / nanoGPT adapter / 飞轮→控制环。

VLA 真训冒烟 (torch 3.14 环境) 在 scripts/vla_train_smoke.py 单独执行,
此处只测不依赖 torch 的部分 + 投影桥结构存在性。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestK3Plus_VisionProjection(unittest.TestCase):
    """K3+ 修复: vision_embd ≠ n_embd 时的投影桥 (vla_tiny 曾前向必崩)。"""

    def test_projection_declared(self):
        import inspect
        from embodied_ai.models import vla_model as vm
        src = inspect.getsource(vm)
        self.assertIn("vision_proj", src, "vlA_model 应含 vision_proj 投影桥")

    def test_tiny_config_dimension_mismatch_exists(self):
        from embodied_ai.models.vla_config import VLAConfig
        cfg = VLAConfig.ablations()["vla_tiny"]
        # 该 bug 的触发条件: tiny 的 vision_embd(默认384) ≠ n_embd(256)
        self.assertNotEqual(cfg.vision_embd, cfg.n_embd,
                            "tiny 应存在维度差 (投影桥的存在性前提)")


class TestK3Plus_NanoGPTAdapter(unittest.TestCase):
    """nanoGPT 训练底座桥接 (真实文件系统探测)。"""

    def setUp(self):
        from embodied_ai.adapters import NanoGPTAdapter
        self.ad = NanoGPTAdapter()
        self.probe = self.ad.probe()

    def test_probe_finds_base(self):
        self.assertTrue(self.probe["exists"], "nanoGPT 仓库应存在")
        self.assertTrue(self.probe["has_model_config"])
        self.assertTrue(self.probe["has_gpt_model"])

    def test_checkpoints_detected(self):
        self.assertIn("latest.pt", self.probe["checkpoints"])
        self.assertEqual(self.ad.latest_checkpoint().name, "latest.pt")

    def test_trained_config_snapshot(self):
        tc = self.probe["trained_config"]
        self.assertIsInstance(tc, dict)
        self.assertIn("n_layer", tc)

    def test_structural_mapping_full_coverage(self):
        from embodied_ai.models.vla_config import VLAConfig
        d = self.ad.structural_diff(VLAConfig.ablations()["vla_tiny"])
        self.assertEqual(d["mapping_coverage"], 1.0, "13 个语义字段应 100% 映射")
        self.assertIn("use_sdpa", d["nanogpt_engineering_missing_in_vla"],
                      "SDPA 工程化选项应出现在待吸收清单")

    def test_training_pack_fields(self):
        from embodied_ai.models.vla_config import VLAConfig
        pack = self.ad.to_nanogpt_training_pack(VLAConfig.ablations()["vla_tiny"])
        for ng_field in ("n_layer", "n_embd", "n_head", "block_size", "learning_rate"):
            self.assertIn(ng_field, pack)
        self.assertIn("vla_tiny", pack["_source"])

    def test_tokenizer_loadable(self):
        tok = self.ad.tokenizer_vocab()
        self.assertIsNotNone(tok, "nanoGPT tokenizer json 应可读")


class TestK3Plus_FlywheelControlLoop(unittest.TestCase):
    """飞轮 DR → make_controller → dr_override 生效链。"""

    def test_dr_overrides_reach_mcts(self):
        from embodied_ai.control.household_robot_system import make_controller
        dr = {"overrides": {"friction_coefficient": {"min": 0.0, "max": 1.5}}}
        ctrl = make_controller(dr_config=dr)
        base = make_controller()
        mu_dr = getattr(ctrl.task_planner.mcts, "_mu_override", None)
        mu_base = getattr(base.task_planner.mcts, "_mu_override", None)
        self.assertIsNotNone(mu_dr, "DR 控制器的 _mu_override 应被注入")
        self.assertIsNone(mu_base, "基线控制器不应有 _mu_override")
        self.assertAlmostEqual(mu_dr, 0.75)  # (0+1.5)/2

    def test_controller_step_with_dr(self):
        import numpy as np
        from embodied_ai.control.household_robot_system import make_controller
        ctrl = make_controller(dr_config={"overrides": {"friction_coefficient": {"min": 0, "max": 1.2}}})
        obs = {"ee_position": np.array([0.3, 0.0, 1.0]), "joint_positions": np.zeros(12)}
        a = ctrl.step(obs)
        self.assertIsNotNone(a)


if __name__ == "__main__":
    unittest.main(verbosity=2)
