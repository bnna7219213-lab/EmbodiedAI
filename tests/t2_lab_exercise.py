"""Lab 2 练习题 — chance_constrained_grip 正确性验证

来自 docs/18-practical-programming-labs.md Lab 2
本题：完成 test_exercise_for_you 中的计算并验证。

手算过程:
  f_t = 120 N  (dyno 峰值切向力)
  μ̂  = 0.85
  σ_μ = 0.08
  δ   = 0.001  (攀岩高安全)

  z = Phi^{-1}(1-0.001) = 3.0902
  denom = 0.85 - 3.0902 * 0.08 = 0.85 - 0.2472 = 0.6028
  f_n = 120 / 0.6028 = 199.1 N

答案: f_n ≈ 199.1 N
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.safety.chance_constrained_grip import (
    chance_constrained_grip, InfeasibleGripError, _z_1_minus_delta,
)
import pytest


class TestChanceConstrainedGripLab(unittest.TestCase):

    def test_z_value_delta_005(self):
        """δ=0.05 时 z 值应该是 1.6449"""
        z = _z_1_minus_delta(0.05)
        self.assertAlmostEqual(z, 1.6449, places=3)

    def test_z_value_delta_001(self):
        """δ=0.001 时 z 值应该是 3.0902 (攀岩高安全)"""
        z = _z_1_minus_delta(0.001)
        self.assertAlmostEqual(z, 3.0902, places=3)

    def test_infeasible_raises(self):
        """σ_μ 过大 (摩擦不确定度过高) → 机会约束无解"""
        with pytest.raises(InfeasibleGripError):
            chance_constrained_grip(10.0, mu_hat=0.3, sigma_mu=0.5, delta=0.001)

    def test_known_value_correct(self):
        """已知输入 → 已知输出: f_t=10N, μ̂=0.8, σ_μ=0.05, δ=0.05 → f_n=13.93"""
        result = chance_constrained_grip(10.0, mu_hat=0.8, sigma_mu=0.05, delta=0.05)
        self.assertAlmostEqual(result, 13.93, places=1)

    def test_safety_increases_with_stricter_delta(self):
        """更严格的 δ → 更大握力 (安全裕度更高)"""
        f_n_05 = chance_constrained_grip(10.0, 0.8, 0.05, delta=0.05)
        f_n_001 = chance_constrained_grip(10.0, 0.8, 0.05, delta=0.001)
        self.assertGreater(f_n_001, f_n_05)

    def test_exercise_for_you(self):
        """🎯 攀岩 dyno 场景: f_t=120N, μ̂=0.85, σ_μ=0.08, δ=0.001

        手算:
          z = 3.0902
          denom = 0.85 - 3.0902 × 0.08 = 0.85 - 0.2472 = 0.6028
          f_n = 120 / 0.6028 ≈ 199.1 N
        """
        result = chance_constrained_grip(120.0, 0.85, 0.08, delta=0.001)
        self.assertAlmostEqual(result, 199.1, delta=0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
