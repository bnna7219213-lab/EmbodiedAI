"""仿真栈 / 运动控制 / 适配器测试 (v0.5 新增)。

覆盖此前**零测试**的模块:
  - src/embodied_ai/simulation/  (base_env / domain_randomization / sim_to_real /
                                   reference_env / make_env 后端选择)
  - src/embodied_ai/control/biped_locomotion.py
  - src/embodied_ai/adapters/nanogpt.py
  - src/embodied_ai/console.py  (GBK 编码崩溃回归)
  - src/embodied_ai/verification.py

设计约束 (可离线运行):
  全部走 reference 后端 / 临时目录, 不依赖 mujoco / pybullet / torch / EdaLab jar。
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from embodied_ai.simulation import (
    DRConfig,
    DRParamSpec,
    DomainRandomizer,
    EnvConfig,
    ReferenceEnv,
    SimEnv,
    SimToRealPipeline,
    available_backends,
    make_env,
)
from embodied_ai.control.biped_locomotion import (
    GaitGenerator,
    LIPM,
    LIPMParams,
    ZMPStabilityChecker,
    build_walking_plans,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


class TestK4A_BackendSelection(unittest.TestCase):
    """后端自动选择 — 无外部引擎时不得阻断管线。"""

    def test_reference_always_available(self):
        backs = available_backends()
        self.assertIn("reference", backs, "纯 numpy 参考环境必须恒可用")

    def test_auto_falls_back_without_crash(self):
        cfg = EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3)
        env = make_env(cfg, backend="auto")
        self.assertIsInstance(env, SimEnv)

    def test_explicit_reference(self):
        env = make_env(EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3),
                       backend="reference")
        self.assertIsInstance(env, ReferenceEnv)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            make_env(EnvConfig(), backend="isaac")

    def test_unavailable_backend_raises_import_error(self):
        from embodied_ai.simulation import available_backends as avail
        if "mujoco" not in avail():
            with self.assertRaises(ImportError):
                make_env(EnvConfig(), backend="mujoco")


class TestK4B_ReferenceEnv(unittest.TestCase):
    """参考环境协议: reset/step/DR/记录/导出。"""

    def setUp(self):
        self.cfg = EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3,
                             dt=0.01, max_episode_steps=50)
        self.env = make_env(self.cfg, backend="reference")

    def tearDown(self):
        self.env.close()

    def test_reset_returns_state(self):
        obs = self.env.reset(seed=0)
        self.assertIn("state", obs)
        self.assertEqual(obs["state"].shape, (6,))

    def test_action_clipped_to_limits(self):
        self.env.reset(seed=0)
        big = np.full(3, 1e6)
        res = self.env.step(big)
        self.assertLessEqual(np.abs(res.observation["state"][:3]).max(),
                             np.abs(self.cfg.action_upper).max() * 10)

    def test_episode_recorded(self):
        """v0.5 回归: 此前 env 从不调用 record_step, export_episode() 恒为空。"""
        self.env.reset(seed=0)
        for _ in range(10):
            self.env.step(self.env.sample_action())
        ep = self.env.export_episode()
        self.assertEqual(ep["length"], 10)
        self.assertEqual(len(ep["observations"]), 10)

    def test_dr_params_applied(self):
        self.env.reset(seed=0)
        self.env.set_dr_params({"link_mass": 2.0, "joint_friction": 1.5,
                                "actuator_delay_ms": 10.0, "gravity": 0.5})
        self.assertAlmostEqual(self.env.mass, 2.0)
        self.assertAlmostEqual(self.env.damping, 1.5)
        self.assertGreater(self.env.actuator_delay_steps, 0)

    def test_reward_decreases_with_distance(self):
        self.env.reset(seed=0)
        r0 = self.env.step(np.zeros(3)).reward
        r1 = self.env.step(np.zeros(3)).reward
        self.assertIsInstance(r0, float)
        self.assertIsInstance(r1, float)

    def test_export_episode_to_file(self):
        import tempfile
        self.env.reset(seed=1)
        self.env.step(self.env.sample_action())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ep.json"
            self.env.export_episode(str(p))
            self.assertTrue(p.exists())
            json.loads(p.read_text(encoding="utf-8"))


class TestK4C_DomainRandomization(unittest.TestCase):
    def test_sample_params_covers_categories(self):
        dr = DomainRandomizer(DRConfig(seed=1))
        params = dr.sample_params()
        self.assertIn("link_mass", params)
        self.assertIn("actuator_delay_ms", params)
        self.assertIn("table_friction", params)

    def test_nominal_params_deterministic(self):
        dr = DomainRandomizer(DRConfig(seed=2))
        a, b = dr.nominal_params(), dr.nominal_params()
        self.assertEqual(a, b)
        for k, v in a.items():
            # choice 类参数返回的是字符串 (纹理名等), 只校验数值型有限
            if isinstance(v, (int, float)):
                self.assertTrue(np.isfinite(v), f"{k} 名义值非有限")

    def test_curriculum_grows_range(self):
        """课程学习: 训练后期采样范围应大于前期。"""
        dr = DomainRandomizer(DRConfig(seed=3, curriculum_enabled=True,
                                       curriculum_start=0.1, curriculum_end=1.0,
                                       curriculum_total_episodes=10))
        early = [dr.sample_params()["link_mass"] for _ in range(2)]
        for _ in range(20):
            dr.sample_params()  # 推进课程
        late = [dr.sample_params()["link_mass"] for _ in range(20)]
        self.assertGreater(np.std(late), 0.0)
        self.assertLess(np.std(early), np.std(late))

    def test_scale_zero_collapses_to_nominal(self):
        dr = DomainRandomizer(DRConfig(seed=4, dynamics_scale=0.0))
        low, high = dr._specs["link_mass"].low, dr._specs["link_mass"].high
        self.assertAlmostEqual(low, high, places=9)

    def test_load_manufacturing_dr_returns_count(self):
        """v0.5 回归: 原实现三处静默 return, 无法区分"没加载"与"加载 0 项"。"""
        import tempfile
        dr = DomainRandomizer(DRConfig(seed=5))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dr.yaml"
            p.write_text(
                "dr_params:\n"
                "  link_mass:\n"
                "    distribution: uniform\n"
                "    low: 0.8\n"
                "    high: 1.2\n"
                "  brand_new_param:\n"
                "    distribution: uniform\n"
                "    low: 0.0\n"
                "    high: 2.0\n", encoding="utf-8")
            n = dr.load_manufacturing_dr(str(p))
            self.assertEqual(n, 2)
            self.assertIn("brand_new_param", dr.param_names)
            self.assertEqual(dr.last_load_applied, 2)

    def test_load_missing_file_returns_zero(self):
        dr = DomainRandomizer(DRConfig(seed=6))
        self.assertEqual(dr.load_manufacturing_dr("definitely/not/here.yaml"), 0)

    def test_spec_distributions(self):
        for dist, kw in (("uniform", {"low": 0.0, "high": 1.0}),
                         ("normal", {"mean": 0.0, "std": 1.0}),
                         ("loguniform", {"low": 1e-3, "high": 1.0}),
                         ("choice", {"choices": [1.0, 2.0]})):
            spec = DRParamSpec(name=dist, distribution=dist, **kw)
            v = spec.sample()
            self.assertTrue(np.isfinite(v) or isinstance(v, float))
        with self.assertRaises(ValueError):
            DRParamSpec(name="bad", distribution="gamma").sample()


class TestK4D_SimToReal(unittest.TestCase):
    class _Policy:
        def __init__(self, target):
            self.target = np.asarray(target, dtype=float)

        def predict(self, obs):
            s = obs["state"]
            n = len(self.target)
            return -2.0 * (s[:n] - self.target) - 0.3 * s[n:2 * n]

    def _cfg(self, noise):
        return EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3, dt=0.01,
                         max_episode_steps=300, obs_noise_std=noise,
                         action_noise_std=noise * 0.5)

    def test_pipeline_runs_end_to_end(self):
        sim = make_env(self._cfg(0.0), backend="reference")
        real = make_env(self._cfg(0.02), backend="reference", real=True)
        res = SimToRealPipeline(sim, real, self._Policy([0.4, 0.0, 0.0]),
                                DRConfig(seed=0, enable_visual=False,
                                         enable_environment=False)
                                ).run(sim_episodes=2, real_episodes=2,
                                      max_steps_per_episode=300)
        self.assertEqual(res.sim_episodes, 2)
        self.assertEqual(res.real_episodes, 2)
        self.assertTrue(res.gap_diagnosis)
        self.assertIn("Sim-to-Real 迁移报告", res.report())

    def test_report_is_ascii_safe(self):
        """v0.5 回归: report() 曾含 U+26A0/U+2022, GBK 控制台 print 即崩。"""
        rep = SimToRealPipeline.__new__(SimToRealPipeline)
        from embodied_ai.simulation.sim_to_real import SimToRealResult
        r = SimToRealResult(sim_episodes=1, real_episodes=1, avg_reward_real=-1.0,
                            success_gap=0.9, real_success_rate=0.0)
        text = r.report()
        text.encode("gbk")  # 不应抛 UnicodeEncodeError

    def test_sim_only_diagnosis(self):
        sim = make_env(self._cfg(0.0), backend="reference")
        res = SimToRealPipeline(sim, None, self._Policy([0.4, 0.0, 0.0]),
                                DRConfig(seed=0)).run(sim_episodes=1,
                                                      max_steps_per_episode=50)
        self.assertIn("未连接真机", res.gap_diagnosis)

    def test_bc_collection_shape(self):
        sim = make_env(self._cfg(0.0), backend="reference")
        pipe = SimToRealPipeline(sim, None, self._Policy([0.4, 0.0, 0.0]),
                                 DRConfig(seed=0))
        ds = pipe.run_bc_collection(n_episodes=2, max_steps=10)
        self.assertEqual(len(ds), 2)
        self.assertEqual(len(ds[0]["observations"]), 10)
        self.assertEqual(len(ds[0]["actions"]), 10)

    def test_bc_collection_zero_steps_no_crash(self):
        """v0.5 回归: max_steps<=0 时 next_result 未定义 → NameError。"""
        sim = make_env(self._cfg(0.0), backend="reference")
        pipe = SimToRealPipeline(sim, None, self._Policy([0.4, 0.0, 0.0]),
                                 DRConfig(seed=0))
        ds = pipe.run_bc_collection(n_episodes=1, max_steps=0)
        self.assertEqual(ds[0]["observations"], [])
        self.assertFalse(ds[0]["success"])


class TestK4E_Biped(unittest.TestCase):
    def test_lipm_omega(self):
        p = LIPMParams(g=9.81, z_c=0.8)
        self.assertAlmostEqual(p.omega, np.sqrt(9.81 / 0.8), places=9)

    def test_lipm_natural_dynamics_stays_put(self):
        """ZMP = CoM 投影时 CoM 不应移动 (平衡点)。"""
        lipm = LIPM(LIPMParams(dt=0.001))
        s = lipm.natural_dynamics(np.array([0.1, 0.0]))
        self.assertAlmostEqual(s[0], 0.1, places=6)

    def test_capture_point_formula(self):
        lipm = LIPM(LIPMParams())
        self.assertAlmostEqual(lipm.calc_capture_point(0.2, 0.3),
                               0.2 + 0.3 / lipm.p.omega, places=12)

    def test_gait_advances_and_stable(self):
        """v0.5 核心回归: 原实现开环积分发散到 -1.17e8 m。"""
        lipm = LIPM(LIPMParams(dt=0.005))
        plans = build_walking_plans(n_steps=4)
        gait = GaitGenerator(lipm, plans)
        checker = ZMPStabilityChecker(com_height=lipm.p.z_c, planar=True)
        dt = 0.005
        n = int(sum(p.step_time for p in plans) / dt)
        unstable = 0
        for _ in range(n):
            com, lf, rf = gait.update(dt)
            plan = plans[gait.active_step_index]
            support = lf if plan.support_foot == "left" else rf
            ok, _ = checker.is_stable(gait.get_zmp_instant(), support[:2])
            unstable += 0 if ok else 1
            self.assertTrue(np.isfinite(com).all(), "CoM 出现非有限值")
        self.assertEqual(unstable, 0, "不应有失稳帧")
        self.assertGreater(gait.get_zmp_instant()[0], 0.0, "CoM 未前进")

    def test_support_foot_matches_plan(self):
        """v0.5 回归: 支撑脚 y 曾硬编码, 与足步规划符号相反。"""
        lipm = LIPM(LIPMParams(dt=0.01))
        plans = build_walking_plans(n_steps=2)
        gait = GaitGenerator(lipm, plans)
        com, lf, rf = gait.update(0.01)
        # 首步 support_foot="left": 支撑脚 y 应与 plan.start_pos[1] 反号
        self.assertAlmostEqual(lf[1], -plans[0].start_pos[1], places=9)
        self.assertNotAlmostEqual(lf[1], rf[1], places=3)  # 两脚须分开

    def test_stability_checker_flags_far_zmp(self):
        checker = ZMPStabilityChecker(planar=True)
        ok, margin = checker.is_stable(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        self.assertTrue(ok)
        ok2, margin2 = checker.is_stable(np.array([0.5, 0.0]), np.array([0.0, 0.0]))
        self.assertFalse(ok2)
        self.assertLess(margin2, margin)

    def test_double_support_hull_widens_polygon(self):
        checker = ZMPStabilityChecker(planar=False)
        zmp = np.array([0.0, 0.0])
        _, single = checker.is_stable(zmp, np.array([0.0, 0.09]))
        _, hull = checker.is_stable(zmp, np.array([0.0, 0.09]),
                                    other_foot_center=np.array([0.0, -0.09]))
        self.assertGreater(hull, single)

    def test_recovery_step_uses_injected_height(self):
        slow = ZMPStabilityChecker(com_height=1.6)
        fast = ZMPStabilityChecker(com_height=0.4)
        a = slow.suggest_recovery_step(np.array([1.0]), np.array([0.0]))
        b = fast.suggest_recovery_step(np.array([1.0]), np.array([0.0]))
        self.assertNotAlmostEqual(a[0], b[0], places=4)  # 质心高度应影响恢复步


class TestK4F_NanoGPTAdapter(unittest.TestCase):
    def test_probe_structure(self):
        import tempfile
        from embodied_ai.adapters.nanogpt import NanoGPTAdapter
        with tempfile.TemporaryDirectory() as d:
            adapter = NanoGPTAdapter(root=d)
            # 空目录: 存在但无契约文件
            pr = adapter.probe()
            self.assertEqual(pr["found"], 0)
            self.assertEqual(pr["total"], 3)
            self.assertEqual(pr["missing"], ["llm.py", "train.py", "tokenizer.py"])
            # 造出契约文件后应被探测到
            for f in ("llm.py", "train.py", "tokenizer.py"):
                (Path(d) / f).write_text("# stub\n", encoding="utf-8")
            pr2 = adapter.probe()
            self.assertTrue(pr2["exists"])
            self.assertEqual(pr2["found"], 3)
            self.assertEqual(pr2["missing"], [])
        # 指向不存在的路径时 exists=False, 且不得抛异常
        self.assertFalse(NanoGPTAdapter(root="/definitely/not/here").probe()["exists"])

    def test_training_plan_export(self):
        import tempfile
        from embodied_ai.adapters.nanogpt import NanoGPTAdapter
        with tempfile.TemporaryDirectory() as d:
            out = NanoGPTAdapter(root=d).export_training_plan("vla_small", out_dir=d)
            self.assertTrue(Path(out["path"]).exists())
            self.assertGreater(out["plan"]["estimate_params"], 0)
            self.assertEqual(out["plan"]["config"]["ablation_name"], "vla_small")

    def test_unknown_size_rejected(self):
        from embodied_ai.adapters.nanogpt import NanoGPTAdapter
        with self.assertRaises(ValueError):
            NanoGPTAdapter().to_vla_config("vla_xxx")

    def test_validate_bc_dataset(self):
        from embodied_ai.adapters.nanogpt import NanoGPTAdapter
        good = [{"observations": [1, 2, 3], "actions": [1, 2, 3], "instruction": "pick"}]
        self.assertTrue(NanoGPTAdapter.validate_bc_dataset(good)["ok"])
        bad = [{"observations": [1, 2], "actions": [1], "instruction": ""}]
        r = NanoGPTAdapter.validate_bc_dataset(bad)
        self.assertFalse(r["ok"])
        self.assertTrue(r["issues"])


class TestK4G_ConsoleEncoding(unittest.TestCase):
    """v0.5 回归: Windows GBK 控制台下的 emoji 崩溃。"""

    def test_fallback_replaces_unencodable(self):
        from embodied_ai import console
        safe = console.s("\u2705 done", encoding="gbk")
        safe.encode("gbk")  # 不应抛异常
        self.assertNotIn("\u2705", safe)

    def test_ascii_passthrough(self):
        from embodied_ai import console
        self.assertEqual(console.s("plain ascii"), "plain ascii")

    def test_marks_degrade_under_gbk(self):
        """状态标记在 GBK 下必须降级为 ASCII, 且降级结果可编码。"""
        from embodied_ai import console
        expected = {"\u2705": "[OK]", "\u274c": "[FAIL]", "\u26a0": "[WARN]"}
        for uni, ascii_tag in expected.items():
            safe = console.s(uni, encoding="gbk")
            safe.encode("gbk")  # 不应抛 UnicodeEncodeError
            self.assertEqual(safe, ascii_tag)

    def test_cli_exit_codes_under_gbk(self):
        """子进程强制 GBK 输出编码, CLI 不得因编码崩。"""
        # 必须继承完整环境 (缺 SYSTEMROOT 等会让 torch/asyncio 加载失败)
        env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "gbk"}
        for cmd in ("contracts", "safety", "sim", "sim2real", "biped", "nanogpt"):
            with self.subTest(cmd=cmd):
                r = subprocess.run([sys.executable, "-m", "embodied_ai.cli", cmd],
                                   capture_output=True, cwd=str(REPO), env=env)
                self.assertEqual(r.returncode, 0,
                                 f"{cmd} 在 GBK 下失败:\n{r.stderr.decode('gbk', 'replace')}")


class TestK4H_Verification(unittest.TestCase):
    def test_run_all_returns_results(self):
        from embodied_ai import verification
        results = verification.run_all(skip_pytest=True)
        self.assertGreaterEqual(len(results), 6)
        for name, passed, detail in results:
            with self.subTest(check=name):
                self.assertTrue(passed, f"{name} 失败: {detail}")

    def test_summarize(self):
        from embodied_ai import verification
        s = verification.summarize([("a", True, "OK"), ("b", False, "x")])
        self.assertEqual(s["passed"], 1)
        self.assertEqual(s["failed"], ["b"])
        self.assertFalse(s["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
