"""K6: v0.7 缺口补齐模块的单元测试 (柔性体 / 3D 全身 / 训练 / 硬件 / 认证 / L7)。

命名沿用既有约定 (test_k<N>_<主题>.py, unittest.TestCase)。
所有用例都用**小规模**参数, 保证整套 pytest 在本地与 CI 上数十秒级完成。
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embodied_ai.control.whole_body import (LegGeometry, SupportPolygon,  # noqa: E402
                                            WholeBody3D, WholeBodyConfig,
                                            foot_rectangle, leg_fk, leg_ik)
from embodied_ai.dynamics.soft_body import SoftBody, SoftBodyConfig  # noqa: E402
from embodied_ai.hardware.dexterous_hand import (GraspContact,  # noqa: E402
                                                 design_matrix, force_closure,
                                                 recommend,
                                                 underactuated_coupling)
from embodied_ai.hardware.foc_shutdown import (FOCConfig, FOCDrive,  # noqa: E402
                                               ShutdownChain, SVPWMModulator)
from embodied_ai.learning.lifelong import (LifelongConfig,  # noqa: E402
                                           LifelongLearner, compute_fisher,
                                           make_synthetic_tasks)
from embodied_ai.models.diffusion_policy import DiffusionPolicyModel  # noqa: E402
from embodied_ai.models.nn_core import MLP, Adam, grad_check, mse_loss  # noqa: E402
from embodied_ai.safety.certification import (architecture_allows,  # noqa: E402
                                              build_default_package, pfh_total,
                                              sil_from_pfh, sff)
from embodied_ai.simulation.real_robot_env import (MockTransport,  # noqa: E402
                                                   RealRobotConfig, RealRobotEnv,
                                                   crc8, decode_telemetry,
                                                   encode_joint_cmd)
from embodied_ai.simulation.base_env import EnvConfig  # noqa: E402


# =========================================================================== #
class TestK6A_NNCore(unittest.TestCase):
    """纯 numpy 训练内核: 反向传播必须与有限差分一致。"""

    def test_grad_check_relu(self):
        m = MLP([5, 8, 3], seed=0)
        r = grad_check(m, np.random.default_rng(1).standard_normal((4, 5)))
        self.assertTrue(r["ok"], r)
        self.assertLess(r["max_rel_err"], 1e-4)

    def test_grad_check_tanh(self):
        m = MLP([4, 6, 2], hidden_act="tanh", out_act="tanh", init="xavier", seed=2)
        r = grad_check(m, np.random.default_rng(3).standard_normal((3, 4)))
        self.assertTrue(r["ok"], r)

    def test_adam_updates_params_inplace(self):
        """优化器必须真的改到 params — 梯度引用被切断时 loss 会一动不动。"""
        m = MLP([2, 4, 1], seed=0)
        opt = Adam([(m.params, m.grads)], lr=1e-2)
        X = np.random.default_rng(0).standard_normal((16, 2))
        y = X @ np.array([[1.0], [-1.0]]) + 0.5
        before = mse_loss(m.forward(X), y)
        snapshot = [p.copy() for p in m.params]
        for _ in range(50):
            pred = m.forward(X)
            m.zero_grad()
            m.backward(2.0 * (pred - y) / y.size)
            opt.step()
        after = mse_loss(m.forward(X), y)
        self.assertLess(after, before)
        for a, b in zip(snapshot, m.params):
            self.assertFalse(np.allclose(a, b), "参数未被更新 (梯度引用断裂?)")

    def test_zero_grad_keeps_identity(self):
        m = MLP([3, 4, 1], seed=1)
        ids = [id(g) for g in m.grads]
        m.zero_grad()
        self.assertEqual([id(g) for g in m.grads], ids)


# =========================================================================== #
class TestK6B_SoftBody(unittest.TestCase):
    """G6: XPBD 柔性体。"""

    def test_hangs_and_settles(self):
        b = SoftBody(SoftBodyConfig(n=6, m=6, spacing=0.02))
        b.reset(height=0.2)
        for i in range(6):
            b.pin(i)
        for _ in range(int(1.0 / (1 / 200))):
            b.step(1 / 200.0)
        m = b.metrics()
        self.assertTrue(np.all(np.isfinite(b.pos)))
        self.assertLess(m.kinetic_energy, 2e-2)   # 6×6 网格; 10×10 时约 5e-3
        self.assertLess(m.max_stretch, 0.05)

    def test_grasp_lifts_com(self):
        b = SoftBody(SoftBodyConfig(n=6, m=6, spacing=0.02))
        b.reset(height=0.02)
        for _ in range(60):
            b.step(1 / 200.0)
        y0 = b.metrics().com[1]
        c = (3) * 6 + 3
        b.grab([c], target=b.pos[c].copy())
        tgt = b.pos[c].copy()
        for k in range(120):
            tgt[1] = b.pos[c][1] + 0.0005
            b.move_grasp(tgt)
            b.step(1 / 200.0)
        self.assertGreater(b.metrics().com[1], y0)

    def test_slip_only_when_overpowered(self):
        cfg = SoftBodyConfig(n=6, m=6, spacing=0.02)
        b = SoftBody(cfg)
        b.reset(height=0.02)
        for _ in range(60):
            b.step(1 / 200.0)
        c = 3 * 6 + 3
        b.grab([c])
        start = b.pos[c].copy()
        for k in range(60):
            b.move_grasp(start + np.array([0.0, 0.0005 * k, 0.0]))
            b.step(1 / 200.0)
        self.assertFalse(b.metrics().slip, "慢速提起不应打滑")


# =========================================================================== #
class TestK6C_WholeBody3D(unittest.TestCase):
    """G9: 3D 支撑多边形与腿部 IK。"""

    def test_ik_roundtrip(self):
        g = LegGeometry()
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = np.concatenate([rng.uniform(-0.4, 0.4, 3), [rng.uniform(0.1, 1.0)],
                                rng.uniform(-0.3, 0.3, 2)])
            p, R = leg_fk(q, g)
            q2 = leg_ik(p, R, g, foot_yaw=q[0])
            p2, R2 = leg_fk(q2, g)
            self.assertLess(np.linalg.norm(p2 - p), 1e-9)
            self.assertLess(np.linalg.norm(R2 - R), 1e-9)

    def test_ik_position_always_exact(self):
        """即使不给 foot_yaw, 位置解也必须精确 (偏差由 hip_roll 吸收)。"""
        g = LegGeometry()
        for yaw in (0.0, 0.3, -0.5):
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            p = np.array([0.1, -0.05, -0.55])
            q = leg_ik(p, R, g)
            p2, _ = leg_fk(q, g)
            self.assertLess(np.linalg.norm(p2 - p), 1e-6)

    def test_support_polygon_area_and_containment(self):
        poly = SupportPolygon(foot_rectangle((0.0, 0.0), 0.16, 0.14))
        self.assertGreater(poly.area, 0.0)
        ok, m = poly.contains(np.array([0.0, 0.0]))
        self.assertTrue(ok)
        self.assertGreater(m, 0.0)
        ok_far, _ = poly.contains(np.array([5.0, 5.0]))
        self.assertFalse(ok_far)

    def test_double_support_wider_than_single(self):
        wb = WholeBody3D(WholeBodyConfig(), LegGeometry())
        wb.init_gait()
        wb.support = "both"
        both = wb.support_polygon().area
        wb.support = "left"
        self.assertGreater(both, wb.support_polygon().area)

    def test_gait_advances_without_instability(self):
        wb = WholeBody3D(WholeBodyConfig(), LegGeometry())
        wb.init_gait()
        dt = 0.005
        for _ in range(2):
            for _ in range(int(wb.cfg.step_time / dt)):
                st = wb.step(dt, zmp_target=wb.plan_zmp(False))
                self.assertTrue(st.stable)
            wb.plan_step()
        self.assertGreater(wb.com[0], 0.1)


# =========================================================================== #
class TestK6D_Training(unittest.TestCase):
    """扩散 / BC / RL。"""

    def test_diffusion_loss_drops(self):
        m = DiffusionPolicyModel(obs_dim=4, action_dim=2, action_horizon=4,
                                 n_diffusion_steps=20, hidden=(32, 32), seed=0)
        X = np.random.default_rng(0).normal(size=(64, 4, 2))
        C = np.random.default_rng(1).normal(size=(64, 4))
        st = m.fit(X, C, iters=150, batch_size=16, eval_every=50)
        self.assertLess(st.final_loss, st.initial_loss)
        self.assertTrue(st.improved)

    def test_diffusion_sample_shape(self):
        m = DiffusionPolicyModel(obs_dim=4, action_dim=2, action_horizon=6,
                                 n_diffusion_steps=10, hidden=(32,), seed=0)
        s = m.sample(np.zeros(4), n_samples=3)
        self.assertEqual(s.shape, (3, 6, 2))
        s_d = m.sample(np.zeros(4), n_steps=3, deterministic=True, n_samples=1)
        self.assertEqual(s_d.shape, (1, 6, 2))

    def test_ppo_returns_improve(self):
        from embodied_ai.models.rl_algorithms import rl_demo
        r = rl_demo("ppo", iters=80)
        self.assertTrue(r["ok"], r)

    def test_sac_diagnostics_bounded(self):
        from embodied_ai.models.rl_algorithms import rl_demo
        r = rl_demo("sac", iters=800)
        d = r["diagnostics"]
        self.assertTrue(np.isfinite(d["q_loss"]))
        self.assertGreater(d["alpha"], 0.0)


# =========================================================================== #
class TestK6E_Hardware(unittest.TestCase):
    """FOC / SVPWM / 关机链 / 灵巧手 / CAN。"""

    def test_svpwm_volt_second_balance(self):
        sv = SVPWMModulator()
        rng = np.random.default_rng(0)
        mag = 0.5 * sv.v_dc / np.sqrt(3.0)
        for _ in range(30):
            th = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.05, 0.9) * mag
            d, diag = sv.duties(r * np.cos(th), r * np.sin(th))
            self.assertTrue(np.all((d >= 0) & (d <= 1)))
            self.assertLess(diag["recon_err"], 1e-6)

    def test_current_loop_tracks_at_locked_rotor(self):
        drv = FOCDrive(FOCConfig(), seed=0)
        drv.locked_rotor = True
        for _ in range(int(0.05 * drv.cfg.f_ctrl)):
            drv.step(10.0)
        self.assertLess(abs(drv.st.i_q - 10.0) / 10.0, 0.02)
        self.assertLess(abs(drv.st.i_d), 0.5)

    def test_overcurrent_latches(self):
        drv = FOCDrive(FOCConfig(), seed=1)
        for _ in range(50):
            drv.step(drv.cfg.i_max * 2.0)
            if drv.check_overcurrent():
                break
        self.assertTrue(drv.trip)
        drv.reset_trip()
        self.assertFalse(drv.trip)

    def test_shutdown_chain_budget(self):
        ch = ShutdownChain()
        ev = ch.evaluate()
        self.assertLessEqual(ev["electrical_off_us"], 1.0)
        self.assertLessEqual(ev["total_ms"], 20.0)
        self.assertTrue(ev["all_stages_ok"])

    def test_slow_comparator_rejected(self):
        ch = ShutdownChain()
        ch.stages[1].delay_s = 200e-6
        self.assertFalse(ch.evaluate()["electrical_ok"])

    def test_max_safe_speed_matches_stop_angle(self):
        ch = ShutdownChain()
        v = ch.max_safe_speed(angle_limit_deg=30.0)
        self.assertAlmostEqual(ch.stop_angle_deg(v), 30.0, places=3)

    def test_force_closure_tri_vs_cluster(self):
        mu = 0.6
        tri = [GraspContact(np.array([0.04 * np.cos(a), 0.04 * np.sin(a)]),
                            np.array([-np.cos(a), -np.sin(a)]), mu)
               for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]
        self.assertTrue(force_closure(tri)[0])
        cluster = [GraspContact(np.array([0.04, 0.0]), np.array([-1.0, 0.0]), mu),
                   GraspContact(np.array([0.04, 0.002]), np.array([-1.0, 0.0]), mu),
                   GraspContact(np.array([0.04, -0.002]), np.array([-1.0, 0.0]), mu)]
        self.assertFalse(force_closure(cluster)[0])

    def test_design_matrix_has_spread(self):
        rows = design_matrix()
        self.assertEqual(len(rows), 4)
        qs = [r["mean_quality"] for r in rows]
        self.assertGreater(max(qs), min(qs))
        self.assertTrue(recommend(rows)["recommended"])

    def test_underactuated_has_fewer_actuators(self):
        rows = design_matrix()
        for r in rows:
            self.assertLessEqual(r["n_actuators"], r["n_dof"])

    def test_can_crc_and_roundtrip(self):
        f = encode_joint_cmd(2, 180.0, 10.0, 20.0)
        self.assertEqual(f.data[7], crc8(bytes(f.data[:7])))
        self.assertRaises(ValueError, encode_joint_cmd, 9, 180.0, 0.0, 1.0)
        m = MockTransport(n_joints=6)
        m.open()
        m.send(encode_joint_cmd(0, 180.0, 0.0, 20.0))
        frame = m.recv(0.01)
        self.assertIsNotNone(frame)
        tel = decode_telemetry(frame)
        self.assertTrue(tel["crc_ok"])
        self.assertAlmostEqual(tel["pos_deg"], 180.0, delta=1.0)

    def test_real_env_safety_gate_blocks(self):
        env = RealRobotEnv(EnvConfig(scene_name="real", state_dim=6, action_dim=3,
                                     dt=0.005, max_episode_steps=30),
                           RealRobotConfig(transport="mock"))
        env.reset(seed=0)
        blocked = False
        for _ in range(6):
            r = env.step(np.array([50.0, 50.0, 50.0]))
            if r.info.get("sto"):
                blocked = True
                break
        env.close()
        self.assertTrue(blocked)


# =========================================================================== #
class TestK6F_CertificationLifelong(unittest.TestCase):
    """G10 认证 / G12 终身学习。"""

    def test_pfh_and_sil(self):
        pkg = build_default_package()
        self.assertLessEqual(pkg.pfh, 1e-6)
        self.assertGreaterEqual(sil_from_pfh(pkg.pfh), 2)
        self.assertGreaterEqual(sff(pkg.subsystems), 90.0)

    def test_architecture_constraint(self):
        pkg = build_default_package()
        self.assertTrue(architecture_allows(pkg.subsystems, 2))
        for s in pkg.subsystems:
            s.hft = 0
        self.assertFalse(architecture_allows(pkg.subsystems, 2))

    def test_rtm_coverage_complete(self):
        pkg = build_default_package()
        self.assertEqual(pkg.rtm_coverage, 1.0)
        self.assertEqual(pkg.hazard_coverage, 1.0)
        self.assertIn("危害分析", pkg.to_markdown())

    def test_zero_diagnostics_blows_pfh(self):
        pkg = build_default_package()
        for s in pkg.subsystems:
            s.dc_percent = 0.0
        self.assertGreater(pkg.pfh, 1e-6)

    def test_ewc_reduces_forgetting(self):
        tasks = make_synthetic_tasks(n=256, seed=0)
        base = LifelongLearner(4, 1, LifelongConfig(ewc_lambda=0.0,
                                                    iters_per_task=200), seed=0)
        mb = base.run_sequence(tasks, use_ewc=False, use_replay=False)
        ewc = LifelongLearner(4, 1, LifelongConfig(ewc_lambda=50.0,
                                                   iters_per_task=200), seed=0)
        me = ewc.run_sequence(tasks, use_ewc=True, use_replay=True)
        self.assertGreater(mb["task0"][-1], me["task0"][-1])

    def test_fisher_is_nonzero(self):
        tasks = make_synthetic_tasks(n=128, seed=0)
        net = MLP([4, 16, 1], seed=0)
        F = compute_fisher(net, tasks[0][0], tasks[0][1], max_samples=32)
        self.assertTrue(any(float(np.sum(f)) > 0 for f in F))


# =========================================================================== #
class TestK6G_Gui(unittest.TestCase):
    """L7 仪表盘: 路由与自检。"""

    def test_routes_registered(self):
        from embodied_ai.gui import DashboardState, routes
        table = routes(DashboardState())
        for p in ("/api/status", "/api/safety", "/api/sim", "/api/biped",
                  "/api/hand", "/api/platform", "/api/command"):
            self.assertIn(p, table)

    def test_status_lists_modules(self):
        from embodied_ai.gui import DashboardState, routes
        st = DashboardState()
        r = routes(st)["/api/status"]({})
        self.assertEqual(r["modules"].get("soft_body"), "ok")
        self.assertEqual(len(r["contracts"]), 3)

    def test_command_switches_mode(self):
        import json

        from embodied_ai.gui import DashboardState, routes
        st = DashboardState()
        table = routes(st)
        res = table["/api/command"]({"_body": [json.dumps({"cmd": "sto"}).encode()]})
        self.assertTrue(res["ok"])
        self.assertEqual(st.mode, "STO")

    def test_gui_demo_ok(self):
        from embodied_ai.gui import gui_demo
        r = gui_demo()
        self.assertTrue(r["ok"], r)


if __name__ == "__main__":
    unittest.main()
