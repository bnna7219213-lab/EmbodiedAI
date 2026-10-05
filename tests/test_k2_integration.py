"""K2/K3 集成测试 — B1-B4 补课后重写 (2026-09-20)。

变更说明: 原测试基于已废弃的旧 FLOW_CHAIN 口径 (edalab_stages Stage 名),
k1-audit.md 判定其为口径漂移的编码化。本版测试以 EdaCli.cmdEmbodiedAiFull
真实 11 步为唯一权威 (B2), 并加入 B1 live 模式真调冒烟。
"""
import sys
import unittest
from pathlib import Path

from embodied_ai.adapters.edalab import FLOW_CHAIN, EdalabAdapter
from embodied_ai.adapters.bigdata import BigdataAdapter
from embodied_ai.deploy import audit_latency
from embodied_ai.safety.formal import close_budget_loop


class TestK2A_RealFlowChain(unittest.TestCase):
    """B2: 真实 11 步口径 (EdaCli 权威)。"""

    def test_11_stages_match_edacli(self):
        """逐步与 EdaCli.kt cmdEmbodiedAiFull 对齐 (L551-581)。"""
        expected = ["forge_core", "mass_props", "structural_fem", "kinematics",
                    "thermal_sim", "pi_analysis", "motor_drive", "data_flywheel",
                    "relay_clutch", "em_coupling", "multi_physics"]
        self.assertEqual([s["name"] for s in FLOW_CHAIN], expected)

    def test_cli_names_match_edacli_dispatch(self):
        """CLI 名与 EdaCli when 分支逐字一致 (live 调用名)。"""
        cli_names = [s["cli"] for s in FLOW_CHAIN]
        self.assertEqual(cli_names, ["forge-core", "mass-props", "structural-fem", "kinematics",
                                     "thermal-sim", "pi-analysis", "motor-drive", "data-flywheel",
                                     "relay-clutch", "em-coupling", "multi-physics"])

    def test_step_numbers_sequential(self):
        self.assertEqual([s["step"] for s in FLOW_CHAIN], list(range(1, 12)))

    def test_forge_core_args_match_edacli(self):
        """forge-core 的调用参数与 EdaCli L551 抄录一致。"""
        fc = FLOW_CHAIN[0]
        self.assertEqual(fc["cli_args"], ["v1.0", "household_grasp", "20", "30", "25", "canfd", "5"])


class TestK2B_LiveMode(unittest.TestCase):
    """B1: subprocess 真调 Kotlin 引擎。"""

    def setUp(self):
        self.adapter = EdalabAdapter()
        self.jar = Path(r"C:\Users\bnna7\workspace\EdaLab\edalabcore\build\libs\edalabcore.jar")

    def test_dry_mode_default(self):
        r = self.adapter.run_stage("relay_clutch", {})
        self.assertEqual(r["mode"], "dry")
        self.assertIn(r["status"], ("READY", "ENGINE_MISSING"))

    def test_live_relay_clutch(self):
        """live 冒烟: relay-clutch 真调, 断言 Kotlin 引擎输出。"""
        if not self.jar.exists():
            self.skipTest("jar 未构建 (需 ./gradlew :edalabcore:jar)")
        r = self.adapter.run_stage("relay_clutch", {}, mode="live")
        self.assertEqual(r["status"], "LIVE_OK",
                         f"live 调用失败: {r.get('stderr', r.get('error', ''))[:200]}")
        self.assertIn("16.00ms", r["stdout"], "安全链 16ms 实测输出应出现在 stdout")
        self.assertIn("总 16.00ms", r["stdout"])
        # 风险行提取
        self.assertTrue(any("CRITICAL" in l for l in r["risk_lines"]),
                         "侵蚀 CRITICAL 告警应被提取")

    def test_live_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.run_stage("si_analysis", {}, mode="live")  # 旧口径名应被拒绝


class TestK2C_BigdataFlywheel(unittest.TestCase):
    """B3: 遥测 → DR → 训练配置 全链 (无人工)。"""

    def test_telemetry_to_training_config(self):
        bd = BigdataAdapter(output_dir=Path("output/dr_test"))
        # 模拟 HIL 遥测 30 条 (含故障事件)
        for i in range(30):
            bd.ingest_canfd_frame({
                "actual_current_a": 10.0 + (i % 5) * 0.8,
                "temp_c": 40.0 + (i % 8),
                "actual_pos_deg": float(i),
            }, joint="shoulder_L", board_serial="SN-001")
        bd.ingest_canfd_frame({"actual_current_a": 12.0, "temp_c": 55.0,
                               "fault_events": [{"fault_type": "OVERCURRENT"}]},
                              joint="shoulder_L", board_serial="SN-001")
        cfg = bd.flywheel_to_training(strategy_v="v2.0")
        self.assertEqual(cfg["replay_records"], 31)
        self.assertEqual(cfg["policy_binding"]["batch_sn"], "SN-001")
        self.assertTrue(Path(cfg["dr_yaml_path"]).exists(), "DR yaml 应已落盘")
        self.assertIn("ambient_temp", cfg["dr_overrides"], "温度 DR 应从遥测生成")
        self.assertIn("fault_inject_overcurrent", cfg["dr_overrides"], "故障注入 DR 应生成")

    def test_edalab_flywheel_output_ingest(self):
        bd = BigdataAdapter()
        n = bd.ingest_edalab_flywheel_output({
            "telemetry_records": [
                {"motor_current_a": 11.0, "temperature_c": 42.0},
                {"motor_current_a": 13.0, "temperature_c": 44.0},
            ]
        })
        self.assertEqual(n, 2)
        self.assertEqual(len(bd.records), 2)


class TestK2D_MCTS(unittest.TestCase):
    """B4: ClimbingMCTS 真实现验证。"""

    def test_mcts_returns_plan(self):
        from embodied_ai.planning.mcts_integration import ClimbingMCTS, ClimbingState
        import numpy as np
        s = ClimbingState(ee_position=np.array([0.0, 0.0, 1.0]),
                          joint_positions=np.zeros(6), stability_margin=0.5)
        plan = ClimbingMCTS(n_simulations=60).search(
            s, goal_fn=lambda st: st.ee_position[2] > 1.4)
        self.assertIsNotNone(plan)
        if plan:
            self.assertGreater(len(plan), 0)

    def test_dr_override_hook(self):
        from embodied_ai.planning.mcts_integration import ClimbingMCTS
        mcts = ClimbingMCTS()
        mcts.dr_override({"friction_coefficient": {"mean": 0.6, "std": 0.1}})  # 不应抛异常


class TestK3_BudgetClosure(unittest.TestCase):
    """K3: safety_budget 实测链 ↔ 形式化闭环。"""

    def test_close_budget_loop(self):
        r = close_budget_loop(joint_speed_rad_s=8.0)
        self.assertAlmostEqual(r.total_break_ms, 16.0, places=2)
        self.assertTrue(r.stop_angle_ok, f"停止角 {r.stop_angle_deg}° 应 < 30°")
        self.assertTrue(r.reachable_set_ok)
        self.assertTrue(r.sil2_ok)
        self.assertTrue(r.closure_ok)

    def test_stop_angle_scales_with_speed(self):
        from embodied_ai.safety.formal.budget_closure import stop_angle_from_budget
        slow = stop_angle_from_budget(2.0)
        fast = stop_angle_from_budget(8.0)
        self.assertGreater(fast, slow, "速度越大停止角越大")


class TestK3_DeployLatency(unittest.TestCase):
    """K3: deploy 延迟审计 (契约预算自动 diff)。"""

    def test_baseline_within_budget(self):
        a = audit_latency()
        self.assertTrue(a.ok)
        self.assertAlmostEqual(a.total_measured_ms, 29.0, places=1)

    def test_over_budget_blocks(self):
        a = audit_latency(measured={"model_inference": 60.0})  # 60ms > 15ms 预算
        self.assertFalse(a.ok)
        seg = next(s for s in a.segments if s.name == "model_inference")
        self.assertEqual(seg.verdict, "BLOCKING")


class TestK3_VLAStatic(unittest.TestCase):
    """VLA 静态验证 (无 torch 环境可跑)。"""

    def test_param_count_estimate(self):
        from embodied_ai.models.vla_config import VLAConfig
        cfg = VLAConfig.ablations()["vla_tiny"]
        self.assertGreater(cfg.n_layer, 0)
        base = VLAConfig()
        self.assertGreater(base.n_layer, cfg.n_layer, "tiny 应小于 base")

    def test_import_graceful(self):
        import embodied_ai.models.vla_model as vm
        # 无 torch 环境下 import 不崩 (HAS_TORCH=False 降级)
        self.assertTrue(hasattr(vm, "HAS_TORCH"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
