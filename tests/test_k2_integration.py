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
    """K3: safety_budget 实测链 ↔ 形式化闭环。

    P0 修复后: 原代码 `t=min(t_budget,t_stop)` 掩盖"停不下来", 假判 PASS。
    修复后诚实暴露: t_stop > t_budget 时返回 +inf, closure_ok=False。
    测试按新物理断言 — 不再验证"停不下来也能通过"的错误结论。
    """

    def test_close_budget_loop_honest_fail_when_cannot_stop(self):
        """8 rad/s 时 t_stop=0.16s >> t_budget=0.016s → 停不下来 → FAIL。"""
        r = close_budget_loop(joint_speed_rad_s=8.0)
        self.assertAlmostEqual(r.total_break_ms, 16.0, places=2)
        # 完全停止角 ω²/2α = 64/100 = 0.64 rad = 36.67° > 30° 阈值
        self.assertFalse(r.stopped_within_budget, "8rad/s 时 t_stop=0.16s > 0.016s, 应报停不下来")
        self.assertEqual(r.stop_angle_deg, float("inf"), "停不下来时应返回 +inf (原代码用 min 掩盖)")
        self.assertFalse(r.stop_angle_ok, "stop_angle=inf > 30° 应判 FAIL")
        self.assertFalse(r.reachable_set_ok, "停不下来时可达集超限位应 FAIL")
        self.assertFalse(r.closure_ok, "整体闭环应 FAIL")
        # SIL2 判定仍独立通过 (PFH < target)
        self.assertTrue(r.sil2_ok)

    def test_stop_angle_stops_within_budget_for_slow_speed(self):
        """低速 (0.5 rad/s, t_stop=0.01s < t_budget=0.016s) 时停止角有限且 < 30°。"""
        from embodied_ai.safety.formal.budget_closure import stop_angle_from_budget
        import numpy as np
        slow = stop_angle_from_budget(0.5)  # t_stop = 0.5/50 = 0.01s < 0.016s
        self.assertLess(slow, 30.0, "低速应在预算内停下, 停止角 < 30°")
        self.assertTrue(np.isfinite(slow), "预算内停下时停止角应为有限值")

    def test_stop_angle_returns_inf_when_cannot_stop(self):
        """t_stop > t_budget 时返回 +inf (诚实 FAIL, 不掩盖)。"""
        from embodied_ai.safety.formal.budget_closure import stop_angle_from_budget
        import numpy as np
        for speed in [2.0, 4.0, 8.0, 16.0]:  # 都 > α·t_budget = 50*0.016 = 0.8 rad/s
            v = stop_angle_from_budget(speed)
            self.assertEqual(v, float("inf"),
                             f"{speed} rad/s 时 t_stop={speed/50:.3f}s > 0.016s, 应返 inf")

    def test_stop_angle_monotonic_within_budget(self):
        """预算内停下时, 停止角随速度单调递增。"""
        from embodied_ai.safety.formal.budget_closure import stop_angle_from_budget
        # 都在预算内 (t_stop = v/50 ≤ 0.016 → v ≤ 0.8 rad/s)
        v1 = stop_angle_from_budget(0.2)
        v2 = stop_angle_from_budget(0.5)
        v3 = stop_angle_from_budget(0.8)
        self.assertLess(v1, v2, "速度越大停止角越大")
        self.assertLess(v2, v3, "速度越大停止角越大")


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
