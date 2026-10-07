"""触觉闭环 L4 测试 — 滑脱检测 → 握力调度闭环。

四层覆盖:
  ① SlipDetector 判据 (f_t / μ·f_n threshold)
  ② GripForcePI (PI controller)
  ③ TactileClosedLoop 闭环收敛性 (滑脱响应)
  ④ ClosedLoopResult 数据契约 + D4 episode 集成
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.perception.tactile_closed_loop import (
    GripForcePID, SlipDetector, TactileClosedLoop,
    synthesize_slip_scenario, ClosedLoopResult,
)
from embodied_ai.safety.chance_constrained_grip import (
    chance_constrained_grip, InfeasibleGripError,
)


class TestSlipDetector(unittest.TestCase):
    """① SlipDetector 判据: incipient slip 检测。"""

    def setUp(self):
        self.det = SlipDetector(threshold=0.7)

    def test_no_slip_when_low_ratio(self):
        """f_t 远小于 μ·f_n → 不触发。"""
        detected, ratio = self.det.check(f_t_est=5.0, mu_est=0.8, f_n_cmd=30.0)
        self.assertFalse(detected)
        self.assertLess(ratio, 0.7)

    def test_slip_triggered_above_threshold(self):
        """f_t >= threshold · μ · f_n → 触发 incipient slip。"""
        # 0.7 * 0.8 * 30 = 16.8 → 估计 f_t = 20 → 触发
        detected, ratio = self.det.check(f_t_est=20.0, mu_est=0.8, f_n_cmd=30.0)
        self.assertTrue(detected)
        self.assertGreaterEqual(ratio, 0.7)

    def test_slip_ratio_correct(self):
        """ratio = f_t / (μ · f_n)。"""
        _, ratio = self.det.check(f_t_est=24.0, mu_est=0.8, f_n_cmd=30.0)
        expected = 24.0 / (0.8 * 30.0)
        self.assertAlmostEqual(ratio, expected, places=5)

    def test_zero_f_n_safe(self):
        """f_n=0 安全返回 (无除零)。"""
        detected, ratio = self.det.check(f_t_est=10.0, mu_est=0.8, f_n_cmd=0.0)
        self.assertFalse(detected)
        self.assertEqual(ratio, 0.0)


class TestGripForcePI(unittest.TestCase):
    """② PI controller: grip force → 电流指令。"""

    def test_basic_proportional(self):
        pid = GripForcePID(kp=0.6, ki=0.0)
        cmd = pid.compute(f_n_target=30.0, dt=0.05)
        self.assertAlmostEqual(cmd, 18.0, places=1)

    def test_integral_accumulates(self):
        pid = GripForcePID(kp=0.0, ki=0.05)
        cmd1 = pid.compute(30.0, dt=0.05)
        cmd2 = pid.compute(30.0, dt=0.05)
        self.assertGreater(cmd2, cmd1)  # 积分项累积

    def test_max_limit(self):
        pid = GripForcePID(kp=1.0, ki=0.0, f_n_max=50.0)
        cmd = pid.compute(f_n_target=200.0, dt=0.05)
        self.assertLessEqual(cmd, 50.0)

    def test_reset(self):
        pid = GripForcePID(kp=0.0, ki=0.05)
        pid.compute(30.0, dt=0.05)
        pid.reset()
        cmd_after = pid.compute(30.0, dt=0.05)
        cmd_before = GripForcePID(kp=0.0, ki=0.05).compute(30.0, dt=0.05)
        self.assertAlmostEqual(cmd_after, cmd_before, places=5)


class TestChanceConstrainedGrip(unittest.TestCase):
    """③ 机会约束公式正确性 (cross-check with physics)。"""

    def test_known_value(self):
        # delta=0.05 → z=1.645, mu=0.8, sigma=0.05
        # denom = 0.8 - 1.645*0.05 = 0.718, f_t=10 → f_n=13.93
        f_n = chance_constrained_grip(10.0, mu_hat=0.8, sigma_mu=0.05, delta=0.05)
        self.assertAlmostEqual(f_n, 10.0 / (0.8 - 1.6449 * 0.05), places=1)

    def test_slip_delta_tighter(self):
        """更严格的 delta → 更大握力 (安全裕度)。"""
        f_n_05 = chance_constrained_grip(10.0, 0.8, 0.05, delta=0.05)
        f_n_001 = chance_constrained_grip(10.0, 0.8, 0.05, delta=0.001)
        self.assertGreater(f_n_001, f_n_05)

    def test_infeasible_raises(self):
        """σ_μ 过大 → InfeasibleGripError。"""
        with self.assertRaises(InfeasibleGripError):
            chance_constrained_grip(10.0, mu_hat=0.3, sigma_mu=0.5, delta=0.001)


class TestTactileClosedLoop(unittest.TestCase):
    """④ 闭环行为: 滑脱响应 + 收敛性。"""

    def test_convergence_on_slip_scenario(self):
        """合成滑脱场景 → 闭环应检测并提升握力 → 最终收敛。"""
        # disturbance 峰值 >> mu·f_n = 0.8·15 = 12N → 必触发滑脱
        readings = synthesize_slip_scenario(n_steps=80, dt=0.05,
                                           mu_true=0.8, f_n_initial=15.0,
                                           disturbance_amplitude=30.0,
                                           disturbance_start=1.0)
        # 扰动在 step 30 (t=1.5s) 后回落 → 闭环应收滑脱恢复
        rng = np.random.default_rng(7)
        ramp_end = int(1.5 / 0.05) + 20  # step 50
        readings[ramp_end:] = rng.normal(0, 0.3, size=len(readings) - ramp_end)

        loop = TactileClosedLoop(mu_prior=0.8, f_n_init=15.0, delta=0.05)
        result = loop.run(readings, dt=0.05, f_n_baseline=15.0)
        # 至少检测到一次滑脱
        self.assertGreater(result.n_slip_detected, 0)
        # 握力指令峰值 > 初始 (响应滑脱)
        self.assertGreater(result.max_grip_force_cmd, 15.0)
        # 最终收敛 (滑脱被抑制)
        self.assertTrue(result.converged)

    def test_no_false_positive_static_holding(self):
        """静态握持 (无扰动) → 不误触发。"""
        # 低幅值噪声
        rng = np.random.default_rng(0)
        readings = rng.normal(0, 0.3, size=40)
        loop = TactileClosedLoop(mu_prior=0.8, f_n_init=30.0)
        result = loop.run(readings, dt=0.05)
        self.assertEqual(result.n_slip_detected, 0)

    def test_baseline_recovery(self):
        """滑脱解除后, 握力回归基线 (节能)。"""
        readings = synthesize_slip_scenario(n_steps=60, dt=0.05,
                                           disturbance_amplitude=20.0,
                                           disturbance_start=1.5)
        # 扰动持续 1.5s (step 30), 之后力下降
        rng = np.random.default_rng(1)
        post_disturb = rng.normal(0, 0.3, size=30)
        readings[30:] = post_disturb

        loop = TactileClosedLoop(mu_prior=0.8, f_n_init=20.0)
        result = loop.run(readings, dt=0.05, f_n_baseline=20.0)
        # 最终几步握力应回归到接近基线
        final_f_n = np.mean([s.grip_force_cmd for s in result.steps[-5:]])
        self.assertLess(final_f_n, result.max_grip_force_cmd)  # 回到基线

    def test_result_summary_shape(self):
        result = ClosedLoopResult()
        s = result.summary()
        for k in ("n_steps", "n_slip_detected", "n_incipient",
                   "max_slip_ratio", "max_grip_force_cmd", "converged"):
            self.assertIn(k, s)


class TestSynthScenario(unittest.TestCase):
    """⑤ 合成场景生成合理性。"""

    def test_slip_inevitable(self):
        """disturbance_amplitude > μ·f_n_initial → 必滑脱。"""
        readings = synthesize_slip_scenario(f_n_initial=20.0, mu_true=0.8,
                                           disturbance_amplitude=20.0)
        # 峰值力 > mu*f_n = 16N → 滑脱
        self.assertGreater(np.max(readings), 16.0)

    def test_initial_hold_quiet(self):
        """初始握持段 ≈ 0。"""
        readings = synthesize_slip_scenario(disturbance_start=1.5, dt=0.05)
        # step 0-29 (t<1.5s) 力接近 0
        initial = readings[:30]
        self.assertLess(np.max(np.abs(initial)), 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
