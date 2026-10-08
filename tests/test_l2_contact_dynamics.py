"""L2 接触动力学修复 — 全模块测试套件 (Gate 2 P1.4 + L2)。

覆盖子模块:
  • hunt_crossley_normal_function    HC阻尼模型
  • project_friction_cone            摩擦锥投影
  • kappa_broadphase / kappa_narrowphase  κ二阶段
  • ADMMContactSolver                ADMM ρ自适应求解器
  • convergence_check                dyno_capture_check解锁判据
  • make_defaults_from_contract      契约→默认接触参数
"""
import sys
import unittest
from pathlib import Path
from typing import List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embodied_ai.simulation.contact_dynamics import (
    ADMMConfig, ADMMContactSolver, ADMMResult,
    ContactPoint,
    hunt_crossley_normal_force,
    project_friction_cone,
    kappa_broadphase, kappa_narrowphase,
    convergence_check,
    make_defaults_from_contract,
)


# ══════════════════════════════════════════════════════════════════
# HC 阻尼模型测试
# ══════════════════════════════════════════════════════════════════

class TestHuntCrossley(unittest.TestCase):
    """Hunt-Crossley 法向力模型。"""

    def test_zero_penetration_zero_force(self):
        """零穿透 → 力 = 0 (Signorini 条件)。"""
        cp = ContactPoint(position=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
                          penetration=0.0)
        self.assertAlmostEqual(hunt_crossley_normal_force(cp), 0.0)

    def test_elastic_term_dominates_static(self):
        """静态压入 (速度=0): 力 = k · x^n (纯 Hertz 项)。"""
        cp = ContactPoint(position=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
                          penetration=0.001,  # 1 mm
                          velocity=np.zeros(3),  # 静态
                          stiffness_n_m=1e5, hertz_exponent=1.5)
        expected = 1e5 * (0.001 ** 1.5)  # = 1e5 · 3.16e-5 = 3.16 N
        self.assertAlmostEqual(hunt_crossley_normal_force(cp), expected, places=6)

    def test_damping_adds_during_compression(self):
        """压缩相 (v_n < 0): 阻尼正贡献 → 总力 > Hertz 弹性力。"""
        cp_static = ContactPoint(position=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
                                  penetration=0.001, velocity=np.zeros(3),
                                  stiffness_n_m=1e5, hertz_exponent=1.5, hc_damp_kappa=100.0)
        cp_compress = ContactPoint(position=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
                                    penetration=0.001, velocity=np.array([0.0, 0.0, -0.05]),
                                    stiffness_n_m=1e5, hertz_exponent=1.5, hc_damp_kappa=100.0)
        f_static = hunt_crossley_normal_force(cp_static)
        f_compress = hunt_crossley_normal_force(cp_compress)
        self.assertGreater(f_compress, f_static,
                           "压缩相阻尼应增加法向力")

    def test_higher_velocity_more_damping(self):
        """相同穿透下: 穿透速度越大 → 阻尼贡献越大。"""
        cp_slow = ContactPoint(position=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
                                penetration=0.001, velocity=np.array([0.0, 0.0, -0.01]),
                                stiffness_n_m=1e5, hertz_exponent=1.5, hc_damp_kappa=100.0)
        cp_fast = ContactPoint(position=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
                                penetration=0.001, velocity=np.array([0.0, 0.0, -0.1]),
                                stiffness_n_m=1e5, hertz_exponent=1.5, hc_damp_kappa=100.0)
        self.assertGreater(hunt_crossley_normal_force(cp_fast),
                           hunt_crossley_normal_force(cp_slow))

    def test_hertz_exponent_scales_nonlinearly(self):
        """指数 n=1.0 (平面) vs n=1.5 (球面): 球面增长更快 (x<1)。"""
        cp_plane = ContactPoint(position=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
                                penetration=0.001, velocity=np.zeros(3),
                                stiffness_n_m=1e5, hertz_exponent=1.0)
        cp_sphere = ContactPoint(position=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
                                 penetration=0.001, velocity=np.zeros(3),
                                 stiffness_n_m=1e5, hertz_exponent=1.5)
        # x=0.001 < 1 → x^1.5 < x^1.0
        self.assertLess(hunt_crossley_normal_force(cp_sphere),
                        hunt_crossley_normal_force(cp_plane))


# ══════════════════════════════════════════════════════════════════
# 摩擦锥投影测试
# ══════════════════════════════════════════════════════════════════

class TestFrictionConeProjection(unittest.TestCase):
    """摩擦锥投影 — ADMM 的核心子步骤。"""

    def test_inside_cone_unchanged(self):
        """锥内 (‖f_t‖ < μ·f_n): 力不变。"""
        fn, ft = project_friction_cone(10.0, np.array([2.0, 1.0]), mu=0.8)
        self.assertAlmostEqual(fn, 10.0)
        np.testing.assert_allclose(ft, [2.0, 1.0])

    def test_outside_cone_scaled_to_surface(self):
        """锥外 (‖f_t‖ > μ·f_n): 径向缩放到锥面。"""
        fn, ft = project_friction_cone(10.0, np.array([6.0, 6.0]), mu=0.5)
        # limit = μ·f_n = 5.0 → ‖ft‖ 应 = 5.0
        self.assertAlmostEqual(np.linalg.norm(ft), 5.0, places=6)
        self.assertAlmostEqual(fn, 10.0)

    def test_negative_normal_clamped(self):
        """f_n < 0: 钳到 0, 切向归零。"""
        fn, ft = project_friction_cone(-5.0, np.array([1.0, 2.0]), mu=0.5)
        self.assertAlmostEqual(fn, 0.0)
        np.testing.assert_allclose(ft, [0.0, 0.0])


# ══════════════════════════════════════════════════════════════════
# κ 二阶段接触检测
# ══════════════════════════════════════════════════════════════════

class TestKappaTwoPhase(unittest.TestCase):
    """κ 二阶段 — 广相机 + 窄相机。"""

    def test_broadphase_includes_near_contacts(self):
        """穿透 > -门限 → 宽门内。"""
        pens = np.array([0.001, 1e-6, -0.0005, -0.1])
        cfg = ADMMConfig(broadphase_radius_m=0.02, penetration_tol_m=1e-4)
        mask, count = kappa_broadphase(pens, cfg)
        self.assertEqual(count, 3)  # 前3个 > -(0.02 + 1e-4)

    def test_narrowphase_only_hard_contacts(self):
        """穿透 > penetration_tol → 硬接触 (窄门内)。"""
        pens = np.array([0.001, 5e-5, -1e-5, 1e-6])
        cfg = ADMMConfig(penetration_tol_m=1e-4)
        mask, idx = kappa_narrowphase(pens, cfg)
        np.testing.assert_array_equal(idx, [0])  # 仅 0.001 > 1e-4

    def test_no_hard_contacts(self):
        """全自由飞行 → 窄门空。"""
        pens = np.array([-0.01, -0.001, 1e-6, -1e-5])
        cfg = ADMMConfig(penetration_tol_m=1e-4)
        mask, idx = kappa_narrowphase(pens, cfg)
        self.assertEqual(len(idx), 0)


# ══════════════════════════════════════════════════════════════════
# ADMM 求解器 (核心)
# ══════════════════════════════════════════════════════════════════

def _make_contacts(n_c: int, penetration: float = 0.001,
                   vel: np.ndarray = None) -> List[ContactPoint]:
    """辅助: 构造均匀垂直壁接触点列。"""
    if vel is None:
        vel = np.array([0.0, 0.0, -0.01])  # 轻微压缩
    return [
        ContactPoint(
            position=np.array([0.1 * i, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            penetration=penetration,
            velocity=vel,
            mu=0.8, stiffness_n_m=1e5, hertz_exponent=1.5, hc_damp_kappa=50.0,
        )
        for i in range(n_c)
    ]


class TestADMMSolver(unittest.TestCase):
    """ADMM 接触力求解器 — ρ自适应 + 收敛判据。"""

    def test_empty_contacts(self):
        """无接触点 → 收敛且零力。"""
        solver = ADMMContactSolver(ADMMConfig(max_iter=100))
        result = solver.solve([])
        self.assertTrue(result.converged)
        self.assertEqual(result.n_iter, 0)

    def test_single_contact_converges(self):
        """单接触点 (1点 3DOF) 收敛。"""
        contacts = _make_contacts(1)
        solver = ADMMContactSolver(ADMMConfig(max_iter=200))
        result = solver.solve(contacts)
        self.assertTrue(result.converged, f"单接触未收敛: primal={result.primal_residual:.2e}")
        self.assertEqual(len(result.contact_forces), 1)
        self.assertGreater(result.normal_forces[0], 0.0, "法向力应 > 0 (抵抗穿透)")

    def test_four_contacts_converges(self):
        """四肢接触 (4点 12DOF) 收敛 — 攀岩典型。"""
        contacts = _make_contacts(4)
        solver = ADMMContactSolver(ADMMConfig(max_iter=400))
        result = solver.solve(contacts)
        self.assertTrue(result.converged, f"四肢接触未收敛: iter={result.n_iter}, "
                                          f"primal={result.primal_residual:.2e}")
        self.assertEqual(len(result.active_contacts), 4)

    def test_forces_friction_cone_compliant(self):
        """输出力在摩擦锥内。"""
        contacts = _make_contacts(4)
        solver = ADMMContactSolver(ADMMConfig(max_iter=400))
        result = solver.solve(contacts)
        for i, fn in enumerate(result.normal_forces):
            self.assertGreaterEqual(fn, -1e-6, f"fn[{i}] = {fn} < 0")
            ft_norm = np.linalg.norm(result.tangential_forces[i])
            limit = contacts[i].mu * max(fn, 0.0)
            self.assertLessEqual(ft_norm, limit + 1e-3,
                                 f"接触 {i}: ‖ft‖={ft_norm:.3f} > μ·fn={limit:.3f}")

    def test_adaptive_rho_within_bounds(self):
        """ρ自适应: 全程 ρ ∈ [ρ_min, ρ_max]。"""
        contacts = _make_contacts(4)
        cfg = ADMMConfig(rho_init=1.0, rho_min=1e-4, rho_max=1e4, adapt_period=5)
        solver = ADMMContactSolver(cfg)
        result = solver.solve(contacts)
        for rho in solver.rho_history:
            self.assertGreaterEqual(rho, cfg.rho_min)
            self.assertLessEqual(rho, cfg.rho_max)


# ══════════════════════════════════════════════════════════════════
# dyno_capture_check 解锁判据
# ══════════════════════════════════════════════════════════════════

class TestConvergenceCheck(unittest.TestCase):
    """convergence_check — dyno_capture_check 解锁入口。"""

    def test_converged_result_passes(self):
        """收敛 + 力合理 → True。"""
        contacts = _make_contacts(2)
        solver = ADMMContactSolver()
        result = solver.solve(contacts)
        ok, msg = convergence_check(result)
        self.assertTrue(ok, msg)

    def test_non_converged_fails(self):
        """未收敛 → False。"""
        result = ADMMResult(
            forces=np.zeros(3), contact_forces=[np.zeros(3)],
            normal_forces=[0.0], tangential_forces=[np.zeros(2)],
            n_iter=50, converged=False, primal_residual=1.0, dual_residual=1.0,
            rho_final=1.0, active_contacts=[0])
        ok, msg = convergence_check(result)
        self.assertFalse(ok)
        self.assertIn("未收敛", msg)

    def test_excessive_force_fails(self):
        """法向力 > 150N 判定失稳 (握持极限)。"""
        result = ADMMResult(
            forces=np.zeros(3), contact_forces=[np.array([0.0, 0.0, 200.0])],
            normal_forces=[200.0], tangential_forces=[np.zeros(2)],
            n_iter=30, converged=True, primal_residual=1e-6, dual_residual=1e-6,
            rho_final=1.0, active_contacts=[0])
        ok, msg = convergence_check(result)
        self.assertFalse(ok)
        self.assertIn("200.0N", msg)


# ══════════════════════════════════════════════════════════════════
# 契约→接触参数联动
# ══════════════════════════════════════════════════════════════════

class TestMakeDefaultsFromContract(unittest.TestCase):
    """make_defaults_from_contract — 制造 DR → 接触模型联动。"""

    _YAML_PATH = Path(__file__).resolve().parent.parent / "contracts" / "dr_params" / "manufacturing_variance.yaml"

    def test_yaml_loads(self):
        """契约文件存在且可解析。"""
        if not self._YAML_PATH.exists():
            self.skipTest(f"yaml 不存在: {self._YAML_PATH}")
        with open(self._YAML_PATH, encoding="utf-8") as f:
            contract = yaml.safe_load(f)
        self.assertIn("electrical", contract)
        self.assertIn("environment", contract)

    def test_defaults_from_contract(self):
        """make_defaults_from_contract: 契约 → ContactPoint 默认参数。"""
        if not self._YAML_PATH.exists():
            self.skipTest("yaml 不存在")
        with open(self._YAML_PATH, encoding="utf-8") as f:
            contract = yaml.safe_load(f)
        cp = make_defaults_from_contract(contract)
        self.assertGreater(cp.mu, 0.0)
        self.assertGreater(cp.stiffness_n_m, 0.0)
        self.assertIn(cp.hertz_exponent, (1.0, 1.5))
        self.assertGreater(cp.hc_damp_kappa, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
