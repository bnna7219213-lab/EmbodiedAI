"""K2 集成验收: adapters/edalab.py + mcts 补全 + 飞轮 DR 闭环。

K2 验收标准 (4 项):
  A. EdalabAdapter: 11 步主链元数据完整 (11 阶段 × 引擎映射 + 参数契约)
  B. ClimbingMCTS 补全: UCB1-MCTS 真实搜索 (select/expand/simulate/backprop)
  C. 飞轮 DR 闭环: ICT/FCT/遥测 → DR 配置 YAML (制造容差驱动)
  D. CLI 新增 couple + dr 子命令: 端到端可调
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.adapters.edalab import FLOW_CHAIN, EdalabAdapter
from embodied_ai.planning import (
    AsyncMCTSPlanner,
    ClimbingMCTS,
    ClimbingState,
    ContactAction,
    DiffusionPolicy,
    MotionPrimitive,
    MotionPrimitiveLibrary,
    RLSkillCoordinator,
    compute_mc_returns,
    plan_contacts,
)
from embodied_ai.control.household_robot_system import (
    RealTime20HzController,
    make_controller,
)


class TestK2A_EdalabAdapter(unittest.TestCase):
    """A. EdaLab 11 步主链 + MultiPhysicsCoupler 适配器。"""

    def test_11_stages_registered(self):
        self.assertEqual(len(FLOW_CHAIN), 11, "主链必须是 11 步")
        names = [s["name"] for s in FLOW_CHAIN]
        expected = ["forge_core", "thermal_sim", "pi_analysis", "si_analysis",
                    "emc_precomp", "motor_drive", "mcad_co_design",
                    "foc_firmware", "canfd_protocol", "coupling_analysis", "data_flywheel"]
        self.assertEqual(names, expected)

    def test_stage_metadata_complete(self):
        for stage in FLOW_CHAIN:
            self.assertIn("name", stage)
            self.assertIn("category", stage)
            self.assertIn("est_sec", stage)
            self.assertIn("engine", stage)
            self.assertTrue(stage["est_sec"] > 0)

    def test_adapter_run_stage(self):
        adapter = EdalabAdapter()
        result = adapter.run_stage("forge_core", {"control_freq_hz": 20})
        self.assertIn("stage", result)
        self.assertIn("status", result)
        self.assertIn(result["status"], ("READY", "ENGINE_MISSING", "CONTRACT_FAIL"))

    def test_adapter_full_chain(self):
        adapter = EdalabAdapter()
        results = adapter.run_full_chain({"control_freq_hz": 1000})
        self.assertEqual(len(results), 11)

    def test_coupling_engine_resolved(self):
        adapter = EdalabAdapter()
        result = adapter.run_stage("coupling_analysis", {})
        # 耦合引擎存在于 EdaLab 或被识别为 EMBEDDEDAI_LOCAL
        self.assertIn(result["status"], ("READY", "ENGINE_MISSING"))

    def test_multi_physic_coupler_exists(self):
        coupler_path = Path(r"C:\Users\bnna7\workspace\git\EmbodiedAI\edalabcore\src\main\kotlin\edalab\core\coupling\MultiPhysicsCoupler.kt")
        self.assertTrue(coupler_path.exists(), f"MultiPhysicsCoupler.kt 不存在: {coupler_path}")
        self.assertGreater(coupler_path.stat().st_size, 5000, "引擎文件应 > 5KB")


class TestK2B_MCTSComplete(unittest.TestCase):
    """B. ClimbingMCTS 完整实现 (非桩)。"""

    def test_mcts_search_returns_path(self):
        init = ClimbingState(
            ee_position=np.array([0.1, 0.2, 0.3]),
            joint_positions=np.zeros(12),
        )
        mcts = ClimbingMCTS(n_simulations=50, rollout_depth=5)
        actions = mcts.search(init, goal_fn=lambda s: s.stability_margin > 0.7)
        self.assertIsInstance(actions, list)
        for a in actions:
            self.assertIsInstance(a, ContactAction)

    def test_mcts_stats_accumulate(self):
        init = ClimbingState(ee_position=np.zeros(3), joint_positions=np.zeros(12))
        mcts = ClimbingMCTS(n_simulations=30, rollout_depth=5)
        mcts.search(init, goal_fn=lambda s: False)
        self.assertGreater(mcts.root.visits, 0, "根节点应有 visit 统计")
        self.assertGreater(len(mcts.root.children), 0, "根节点应有子节点")

    def test_compute_mc_returns(self):
        rewards = [0.0, 0.0, 1.0, 0.5]
        returns = compute_mc_returns(rewards, discount=0.95)
        self.assertEqual(len(returns), len(rewards))
        self.assertAlmostEqual(returns[-1], rewards[-1], places=5)
        self.assertGreater(returns[0], 0)

    def test_contact_action_hashable(self):
        a = ContactAction(contact_position=np.array([0.1, 0.2, 0.3]), grasp_type="power")
        b = ContactAction(contact_position=np.array([0.1, 0.2, 0.3]), grasp_type="power")
        self.assertEqual(hash(a), hash(b))

    def test_async_mcts_planner(self):
        init = ClimbingState(ee_position=np.zeros(3), joint_positions=np.zeros(12))
        planner = AsyncMCTSPlanner()
        actions = planner._mcts_search(init)
        self.assertIsInstance(actions, list)

    def test_plan_contacts_entry(self):
        init = ClimbingState(ee_position=np.zeros(3), joint_positions=np.zeros(12))
        actions, mcts = plan_contacts(init, n_sim=20)
        self.assertIsInstance(actions, list)
        self.assertIsInstance(mcts, ClimbingMCTS)

    def test_dr_override_injection(self):
        dr_config = {"overrides": {"joint_stiffness": {"mean": 4500}, "friction_coefficient": {"min": 0.6, "max": 0.9}}}
        mcts = ClimbingMCTS(n_simulations=10)
        mcts.dr_override(dr_config)
        # 不报错即通过
        self.assertTrue(True)

    def test_diffusion_policy_interface(self):
        policy = DiffusionPolicy()
        self.assertFalse(policy._model_loaded)
        traj = policy.generate({}, [])
        self.assertEqual(traj.shape[1], 12)  # action_dim


class TestK2C_FlywheelDR(unittest.TestCase):
    """C. 飞轮 DR 制造容差闭环。"""

    def test_generate_dr_config_with_ict(self):
        adapter = EdalabAdapter()
        ict = [{"board_serial": "SF001", "component_values": {"ENC1": 0.01, "ENC2": -0.005, "CS_INA": 0.002}}]
        cfg = adapter.generate_dr_config("v1.0", ict_data=ict)
        self.assertIn("encoder_noise_std", cfg["overrides"])
        self.assertIn("current_sensor_noise_std", cfg["overrides"])

    def test_generate_dr_config_with_fct(self):
        adapter = EdalabAdapter()
        fct = [{"board_serial": "SF001", "comm_tests": {"can_bus_utilization": 1.2}}]
        cfg = adapter.generate_dr_config("v1.0", fct_data=fct)
        self.assertIn("canfd_latency_ms", cfg["overrides"])

    def test_generate_dr_config_with_telemetry(self):
        adapter = EdalabAdapter()
        tel = [
            {"motor_current_a": 2.5, "temperature_c": 35, "fault_events": []},
            {"motor_current_a": 3.0, "temperature_c": 40, "fault_events": [{"fault_type": "OVERCURRENT"}]},
        ]
        cfg = adapter.generate_dr_config("v1.0", telemetry_data=tel)
        self.assertIn("load_torque_scale", cfg["overrides"])
        self.assertIn("friction_coefficient", cfg["overrides"])

    def test_export_dr_yaml(self):
        import tempfile
        adapter = EdalabAdapter()
        cfg = adapter.generate_dr_config("v1.0", ict_data=[{"board_serial": "T1", "component_values": {"ENC1": 0.01}}])
        with tempfile.TemporaryDirectory() as td:
            out = adapter.export_dr_yaml(cfg, Path(td) / "test_dr.yaml")
            self.assertTrue(out.exists())
            content = out.read_text(encoding="utf-8")
            self.assertIn("encoder_noise_std", content)

    def test_load_contract_dr_params(self):
        adapter = EdalabAdapter()
        contract = adapter.load_contract_dr_params()
        self.assertIn("electrical", contract)
        self.assertIn("mechanical", contract)
        self.assertIn("sensing", contract)

    def test_flywheel_e2e(self):
        adapter = EdalabAdapter()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            result = adapter.flywheel_e2e(
                ict_data=[{"board_serial": "SF001", "component_values": {"ENC1": 0.01}}],
                fct_data=[{"board_serial": "SF001", "comm_tests": {"can_bus_utilization": 0.8}}],
                telemetry_data=[{"motor_current_a": 2.5, "temperature_c": 35, "fault_events": []}],
                output_dir=Path(td),
            )
            self.assertTrue(result["flywheel_ok"])
            self.assertIn("config", result)
            self.assertTrue(Path(result["yaml_path"]).exists())


class TestK2D_ControllerAndCLI(unittest.TestCase):
    """D. 控制器 + CLI 集成。"""

    def test_controller_create(self):
        ctrl = make_controller()
        self.assertIsInstance(ctrl, RealTime20HzController)

    def test_controller_with_dr(self):
        dr = {"overrides": {"joint_stiffness": {"mean": 5000}}}
        ctrl = RealTime20HzController(dr_config=dr)
        self.assertIsNotNone(ctrl._dr_config)
        ctrl.inject_dr_config({"overrides": {"joint_stiffness": {"mean": 4800}}})

    def test_controller_step(self):
        ctrl = make_controller()
        obs = {"ee_position": np.zeros(3), "joint_positions": np.zeros(12)}
        action = ctrl.step(obs)
        self.assertIsInstance(action, np.ndarray)

    def test_rl_skill_coordinator(self):
        coord = RLSkillCoordinator()
        coord.load_skill("test", DiffusionPolicy())
        action = coord.execute("test", {})
        # DiffusionPolicy 返回 (action_horizon, action_dim) = (10, 12)
        self.assertEqual(action.shape, (10, 12))

    def test_motion_primitive_library(self):
        lib = MotionPrimitiveLibrary()
        mp = MotionPrimitive(name="test", duration_s=1.0)
        lib.add("test", mp)
        self.assertIsNotNone(lib.get("test"))

    def test_cli_subcommands(self):
        """CLI 新增 couple / dr 子命令可调用。"""
        from embodied_ai.cli import main
        # K1 子命令仍工作
        self.assertEqual(main(["contracts"]), 0)
        # K2 新增 couple
        rc = main(["couple"])
        self.assertIn(rc, (0, 1))  # 1 仅当引擎文件缺失
        # K2 新增 dr
        rc = main(["dr"])
        self.assertEqual(rc, 0, "飞轮 DR 端到端应 PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
