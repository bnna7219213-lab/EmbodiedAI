"""K3 形式化安全层验收。

验收标准:
  A. STL 规约: STLGlobally/STLFinally/STLImplies/Atom 可构造 + robustness 计算有意义
  B. 区间可达集: IntervalSet + ReachabilityAnalyzer 前向传播 + 包络检验
  C. ISO 13482 监控: force/speed/torque/power 阈值检查 + 违规分级 + STO 触发
  D. VLA 模型骨架: VLAConfig 参数量估算 + 消融配置 + JSON 序列化 (无 torch 跳过网络前向)
"""
import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from embodied_ai.safety.formal import (
    IntervalSet,
    ReachabilityAnalyzer,
    SafetyEnvelope,
    STLAtom,
    STLFinally,
    STLGlobally,
    STLImplies,
    ISO13482Monitor,
    force_limit_monitor,
    velocity_when_human_near,
)
from embodied_ai.models import VLAConfig, ActionTokenizer, VLAModel


class TestK3A_STL(unittest.TestCase):
    def test_atom_robustness(self):
        atom = STLAtom("non_negative", lambda x: -x[0])  # x[0] >= 0 时可满足
        self.assertGreater(atom.robustness(np.array([5.0])), 0)
        self.assertLess(atom.robustness(np.array([-1.0])), 0)

    def test_globally_robustness(self):
        # "始终力 < 100"
        g = force_limit_monitor(max_force=100.0)
        # 构建信号 (单点时间序列)
        signal = {"time": [0.0], "force": [50.0], "state": [50.0]}
        r = g.robustness(signal, t=0, dt=0.01)
        self.assertIsInstance(r, float)

    def test_implies(self):
        # "接近 (<5) → 低速 (<2)"
        dist = STLAtom("dist_under", lambda x: float(x[0]) - 5.0)
        vel = STLAtom("vel_over", lambda x: float(x[1]) - 2.0)
        impl = STLImplies(dist, vel)
        # dist=3 (<5 成立), vel=1 (<2 成立) → 整体成立
        r = impl.robustness(np.array([3.0, 1.0, 0.0]))
        self.assertGreater(r, 0)

    def test_complex_formula(self):
        # G(force < 280) — 始终力不超 280N
        g = force_limit_monitor(max_force=280.0)
        # robustness > 0 (signal force=50)
        signal = {"time": [0.0], "force": [50.0], "state": [50.0]}
        r = g.robustness(signal, t=0, dt=0.01)
        self.assertGreater(r, 0)


class TestK3B_Reachability(unittest.TestCase):
    def test_intervalset_intersects(self):
        a = IntervalSet(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        b = IntervalSet(np.array([1.5, 1.5]), np.array([0.5, 0.5]))
        self.assertTrue(a.intersects(b))
        c = IntervalSet(np.array([5.0, 5.0]), np.array([0.1, 0.1]))
        self.assertFalse(a.intersects(c))

    def test_intervalset_contains(self):
        s = IntervalSet(np.array([0.0]), np.array([1.0]))
        self.assertTrue(s.contains(np.array([0.5])))
        self.assertFalse(s.contains(np.array([2.0])))

    def test_reachable_forward(self):
        # dx/dt = u (速度直接=控制)
        dynamics = lambda x, u: u
        # Jacobian 为零 (dx'/dx = 0)
        jac = lambda x, u: np.zeros((2, 2))
        analyzer = ReachabilityAnalyzer(state_dim=2, dt=0.01)
        reachable = analyzer.forward_reachable(
            x0=np.array([0.0, 0.0]),
            delta0=np.array([0.1, 0.1]),
            control=np.array([1.0, 0.5]),
            dynamics=dynamics,
            jacobian_bound=jac,
            horizon_steps=10,
            extra_uncertainty=0.01,
        )
        self.assertEqual(len(reachable), 11)  # 0..10
        # 中心前向: x0 + control * dt * steps
        np.testing.assert_allclose(reachable[10].center, np.array([0.1, 0.05]), atol=0.01)

    def test_safety_check(self):
        analyzer = ReachabilityAnalyzer(state_dim=2, dt=0.01)
        env = SafetyEnvelope(
            name="joint_limits",
            lower_bound=np.array([-3.0, -3.0]),
            upper_bound=np.array([3.0, 3.0]),
        )
        r_safe = IntervalSet(np.array([0.0, 0.0]), np.array([0.1, 0.1]))
        r_unsafe = IntervalSet(np.array([5.0, 5.0]), np.array([0.1, 0.1]))
        ok, step, _ = analyzer.check_safety([r_safe], env)
        self.assertTrue(ok)
        ok, step, _ = analyzer.check_safety([r_unsafe], env)
        self.assertFalse(ok)

    def test_stop_budget(self):
        analyzer = ReachabilityAnalyzer(state_dim=1, dt=0.001)
        # 0.25 m/s, 需减速度 > 0.25/0.02 = 12.5 m/s² 才能在 20ms (20步@1kHz) 内停下
        ok, dist = analyzer.check_stop_budget(0.25, budget_steps=20, max_decel=15.0)
        self.assertTrue(ok)
        self.assertGreater(dist, 0)


class TestK3C_ISO13482(unittest.TestCase):
    def setUp(self):
        self.monitor = ISO13482Monitor(
            state_dim=12, dt=0.001,
            max_force_N=280.0, max_speed_ms=0.25,
        )

    def test_safe_state(self):
        result = self.monitor.evaluate(
            joint_state=np.zeros(12),
            force_reading=50.0,
            human_distance_m=0.3,
            torque_reading=10.0,
            power_reading=5.0,
        )
        self.assertTrue(result.safe)

    def test_force_violation(self):
        result = self.monitor.evaluate(
            joint_state=np.zeros(12),
            force_reading=300.0,  # 超过 280
            human_distance_m=0.3,
            torque_reading=10.0,
            power_reading=5.0,
        )
        self.assertFalse(result.safe)
        self.assertTrue(any(v.severity == 4 for v in result.violations))

    def test_speed_violation_near_human(self):
        # 人体 0.3m 内 (thr=0.5), 关节速度过高
        state = np.zeros(12)
        state[1::2] = 0.5  # 所有速度 = 0.5 m/s (>0.25)
        result = self.monitor.evaluate(
            joint_state=state,
            force_reading=10.0,
            human_distance_m=0.3,
            torque_reading=5.0,
        )
        self.assertFalse(result.safe)

    def test_violations_have_iso_clause(self):
        result = self.monitor.evaluate(
            joint_state=np.zeros(12),
            force_reading=500.0,
            human_distance_m=0.1,
        )
        self.assertFalse(result.safe)
        for v in result.violations:
            self.assertIn("ISO 13482", v.clause)

    def test_biomech_limit(self):
        self.assertEqual(self.monitor.biomech_limit("skull_forehead"), 130)
        self.assertEqual(self.monitor.biomech_limit("nonexistent"), 280)

    def test_history_summary(self):
        for i in range(10):
            self.monitor.evaluate(np.zeros(12), force_reading=float(i))
        summary = self.monitor.summary()
        self.assertEqual(summary["total_samples"], 10)


class TestK3D_VLASkeleton(unittest.TestCase):
    def test_vla_config_defaults(self):
        cfg = VLAConfig()
        self.assertGreater(cfg.n_layer, 0)
        self.assertGreater(cfg.num_patches, 0)

    def test_param_estimation(self):
        cfg = VLAConfig()
        est = cfg.estimate_params()
        self.assertGreater(est["total"], 0)
        self.assertIn("vision_encoder", est)

    def test_ablations(self):
        ablations = VLAConfig.ablations()
        self.assertEqual(len(ablations), 4)
        self.assertIn("vla_tiny", ablations)
        self.assertIn("vla_base", ablations)

    def test_config_roundtrip_json(self):
        cfg = VLAConfig(ablation_name="test_k3")
        j = cfg.to_json()
        cfg2 = VLAConfig(**json.loads(j))
        self.assertEqual(cfg2.n_layer, cfg.n_layer)
        self.assertEqual(cfg2.ablation_name, "test_k3")

    def test_action_tokenizer_nontorch(self):
        """无 torch 时 ActionTokenizer 应 raise 明确错误 (不被静默吞)。"""
        try:
            import torch  # noqa: F401
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False
        if not HAS_TORCH:
            with self.assertRaises(RuntimeError):
                ActionTokenizer()


if __name__ == "__main__":
    unittest.main(verbosity=2)
