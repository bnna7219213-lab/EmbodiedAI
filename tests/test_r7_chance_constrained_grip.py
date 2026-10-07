"""R7 机会约束握力修复测试 (2026-10-07)。

验收要点 (对应评估 §6.1-2, plan.md S1.1):
  1. z 值精确: δ=0.05 → 1.6449; δ=0.001 → 3.0902 (旧线性近似得 1.408, 错 1.2x)
  2. InfeasibleGripError: denom<=0.05 时抛异常 (旧 max(denom, 0.05) 静默掩盖)
  3. grip_demo 可运行 (旧缺 numpy import 必 NameError), 匹配不确定度时滑脱率 ≤ δ
  4. 学习后过度自信 (低估 σ) 会触发滑脱率回升 — 保留 §6.6 原结论
  5. soft_contact_slip_early_warn: 摩擦不确定度过大触发 CRITICAL
"""
import math
import unittest

from embodied_ai.safety.chance_constrained_grip import (
    InfeasibleGripError,
    chance_constrained_grip,
    grip_demo,
    soft_contact_slip_early_warn,
)


class TestR7ZValue(unittest.TestCase):
    """R7-1: z 值必须用 Φ⁻¹(1-δ), 不能用线性近似 sqrt(2)*(1-2δ)."""

    def _z_1_minus_delta(self, delta):
        """直接用 statistics.NormalDist 计算真值 z, 反推 f_n."""
        from statistics import NormalDist
        return NormalDist().inv_cdf(1.0 - delta)

    def test_delta_05_gives_1_6449(self):
        """δ=0.05 → z=1.6449 (标准 1.6σ 置信)."""
        f_n = chance_constrained_grip(f_t_req=1.0, mu_hat=1.0, sigma_mu=0.1, delta=0.05)
        z_true = self._z_1_minus_delta(0.05)
        self.assertAlmostEqual(z_true, 1.6449, places=3)
        expected_fn = 1.0 / (1.0 - z_true * 0.1)
        self.assertAlmostEqual(f_n, expected_fn, places=4)
        # 反证: 旧线性近似 sqrt(2)*(1-0.1) = 1.273
        old_fn = 1.0 / (1.0 - math.sqrt(2) * (1 - 2 * 0.05) * 0.1)
        # 新值应比旧值大 (旧 z 更小 → denom 更大 → f_n 更小)
        self.assertGreater(f_n, old_fn,
                            msg=f"新 z={z_true:.4f} 应 > 旧 z=1.2728, 得 f_n 应更大")

    def test_delta_001_gives_3_0902(self):
        """δ=0.001 → z=3.0902 (评估 §6.1-2 直接引用此值)."""
        f_n = chance_constrained_grip(f_t_req=10.0, mu_hat=2.0, sigma_mu=0.1, delta=0.001)
        z_true = self._z_1_minus_delta(0.001)
        self.assertAlmostEqual(z_true, 3.0902, places=3)
        expected_fn = 10.0 / (2.0 - z_true * 0.1)
        self.assertAlmostEqual(f_n, expected_fn, places=4)
        # 反证: 旧 z=1.408 (差 2.2x), 会导致 denom 从 1.69 变成 1.86 → f_n 从 5.91 变 5.38
        old_fn = 10.0 / (2.0 - math.sqrt(2) * (1 - 2 * 0.001) * 0.1)
        # 新旧 f_n 差异 > 8% (旧公式系统性低估握力需求)
        self.assertGreater(f_n, old_fn, msg=f"新 z 大 → 需求握力应更大")
        # 相对差
        ratio = f_n / old_fn
        self.assertGreater(ratio, 1.05,
                            msg=f"新旧比值应 > 1.05, 实际 {ratio:.3f}")

    def test_delta_0001_gives_3_719(self):
        """δ=0.0001 → z=3.7190 (Python statistics 实测值)."""
        f_n = chance_constrained_grip(f_t_req=1.0, mu_hat=2.0, sigma_mu=0.1, delta=0.0001)
        z_true = self._z_1_minus_delta(0.0001)
        self.assertAlmostEqual(z_true, 3.7190, places=3)
        expected_fn = 1.0 / (2.0 - z_true * 0.1)
        self.assertAlmostEqual(f_n, expected_fn, places=4)
        # 旧近似 z=1.408 严重偏低 (差 2.6x)
        old_fn = 1.0 / (2.0 - math.sqrt(2) * (1 - 2 * 0.0001) * 0.1)
        self.assertGreater(f_n, old_fn * 1.05)


class TestR7Infeasible(unittest.TestCase):
    """R7-3: denom <= min_margin 时抛 InfeasibleGripError (旧 max(denom, 0.05) 掩盖)."""

    def test_huge_sigma_raises(self):
        """σ_μ 过大 → 机会约束无解."""
        with self.assertRaises(InfeasibleGripError):
            # denom = 0.5 - 1.6449*0.5 = -0.32, 严重不可行
            chance_constrained_grip(f_t_req=10.0, mu_hat=0.5, sigma_mu=0.5, delta=0.05)

    def test_tiny_mu_raises(self):
        """μ̂ 过小 → 机会约束无解."""
        with self.assertRaises(InfeasibleGripError):
            # denom = 0.2 - 1.6449*0.1 = 0.0355 < 0.05
            chance_constrained_grip(f_t_req=10.0, mu_hat=0.2, sigma_mu=0.1, delta=0.05)

    def test_extreme_delta_raises_when_sigma_large(self):
        """极小 δ + 大 σ → 无解."""
        with self.assertRaises(InfeasibleGripError):
            # denom = 1 - 3.0902*0.5 = -0.545
            chance_constrained_grip(f_t_req=10.0, mu_hat=1.0, sigma_mu=0.5, delta=0.001)

    def test_bad_delta_rejected(self):
        """δ ∉ (0,1) 拒绝."""
        with self.assertRaises(ValueError):
            chance_constrained_grip(f_t_req=10.0, mu_hat=1.0, sigma_mu=0.1, delta=0.0)
        with self.assertRaises(ValueError):
            chance_constrained_grip(f_t_req=10.0, mu_hat=1.0, sigma_mu=0.1, delta=1.0)
        with self.assertRaises(ValueError):
            chance_constrained_grip(f_t_req=10.0, mu_hat=1.0, sigma_mu=0.1, delta=-0.1)

    def test_negative_sigma_rejected(self):
        """σ < 0 拒绝."""
        with self.assertRaises(ValueError):
            chance_constrained_grip(f_t_req=10.0, mu_hat=1.0, sigma_mu=-0.1, delta=0.05)

    def test_old_behavior_would_return_wrong_value(self):
        """反证: 旧 max(denom, 0.05) 会静默返回 f_t/0.05=200N 的荒谬值。"""
        # 现在应该抛异常, 而不是给 200N
        with self.assertRaises(InfeasibleGripError):
            fn = chance_constrained_grip(f_t_req=10.0, mu_hat=0.5, sigma_mu=0.5, delta=0.05)
        # 若旧行为存在: 200N 明显过大 (攀岩场景 100N+ 就是致命值)


class TestR7GripDemo(unittest.TestCase):
    """R7-2: grip_demo 可运行 (旧缺 numpy import 必 NameError)."""

    def test_grip_demo_runs(self):
        """grip_demo 现在应成功返回 dict, 无 NameError."""
        result = grip_demo(mu_hat=0.85, sigma_mu=0.15, verbose=False)
        # 主验收: 匹配不确定度时滑脱率 ≤ δ + MC 抽样噪声 (0.5%)
        self.assertLessEqual(result["slip_cc"], 0.055,
                             f"匹配 σ 时滑脱率应 ≤5%, 实际 {result['slip_cc']:.4f}")
        # 学习后收敛 σ 应保持安全
        self.assertLessEqual(result["slip_learned_matched"], 0.055,
                             f"学习后 σ 也收敛时滑脱率应 ≤5%, 实际 {result['slip_learned_matched']:.4f}")
        # 过度自信 (低估 σ) 会触发滑脱率回升
        self.assertGreater(result["slip_learned_overconf"], result["slip_learned_matched"],
                            msg="过度自信时滑脱率应高于匹配情况")
        # 无裕度点估计滑脱率应远大于 δ
        self.assertGreater(result["slip_naive"], 0.055)
        # 主验收 flag
        self.assertTrue(result["ok"])

    def test_grip_demo_f_cc_uses_z_1_6449(self):
        """grip_demo 的 f_cc 值应使用 z=1.6449 (而非旧 1.414)."""
        result = grip_demo(mu_hat=0.85, sigma_mu=0.15, verbose=False)
        # f_t=30, mu=0.85, sigma=0.15, δ=0.05
        # z=1.6449: denom = 0.85 - 1.6449*0.15 = 0.6033, f_n = 30/0.6033 = 49.73
        expected_f = 30.0 / (0.85 - 1.6449 * 0.15)
        self.assertAlmostEqual(result["f_cc"], expected_f, places=1)
        # 反证: 旧 z=1.414 → f_n = 30/(0.85-0.212) = 47.0, 差 6%
        old_f = 30.0 / (0.85 - math.sqrt(2) * (1 - 2 * 0.05) * 0.15)
        self.assertGreater(result["f_cc"], old_f)


class TestR7EarlyWarn(unittest.TestCase):
    """R7-附加: soft_contact_slip_early_warn 提供分级告警."""

    def test_zero_delta_ft_is_info(self):
        r = soft_contact_slip_early_warn(delta_f_t=0.0, mu_hat=0.85, sigma_mu=0.15)
        self.assertTrue(r["ok"])
        self.assertEqual(r["warn_level"], "INFO")

    def test_large_delta_ft_normal(self):
        r = soft_contact_slip_early_warn(delta_f_t=5.0, mu_hat=0.85, sigma_mu=0.15)
        self.assertTrue(r["ok"])
        # denom = 0.85 - 1.6449*0.15 = 0.603, > 0.3 → INFO
        self.assertIn(r["warn_level"], ("INFO", "WARNING"))
        self.assertGreater(r["f_n_required"], 0)

    def test_uncertain_mu_critical(self):
        r = soft_contact_slip_early_warn(delta_f_t=5.0, mu_hat=0.3, sigma_mu=0.3)
        # denom = 0.3 - 1.6449*0.3 = -0.193, <= 0.05 → CRITICAL
        self.assertFalse(r["ok"])
        self.assertEqual(r["warn_level"], "CRITICAL")
        self.assertEqual(r["f_n_required"], float("inf"))


if __name__ == "__main__":
    unittest.main()
