"""Phase A 攀岩周期集成测试 — climb-soft-robot-plan §二 判定标准。

核心断言: 一个攀岩周期跑通, 每个物理量由对应模块真实计算 (无 mock);
A2 柔顺结论与 VSA-dyno 结论互证。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.simulation.climb_scene import (ClimbPhase, SoftFingertipCoupling,
                                                 generate_wall, run_climb_cycle,
                                                 soft_contact_peak_force,
                                                 fingertip_stiffness_from_soft_body)
from embodied_ai.dynamics.soft_body import SoftBody, SoftBodyConfig
from embodied_ai.simulation.wall_contact import (QuadContactState, WallContactConfig,
                                                 allocate_forces, limb_jacobian,
                                                 wall_frame, wall_stand_sequence)


class TestClimbCycle(unittest.TestCase):
    """A1+A2+A5: 完整周期集成。"""

    def test_full_cycle_reaches_climb_on(self):
        """六状态周期: hang→reach→swing→dyno→capture→climb_on 全部转移。"""
        r = run_climb_cycle(seed=42)
        self.assertEqual(r.phases_visited[-1], "climb_on")
        self.assertIn("dyno", r.phases_visited)
        self.assertIn("capture", r.phases_visited)

    def test_all_real_no_mock(self):
        r = run_climb_cycle(seed=42)
        self.assertTrue(r.all_real)
        # 各模块真实消费的痕迹: 抓握型来自 dexterous_hand 的类型映射
        self.assertTrue(r.grasp_choices)
        self.assertIn("pinch_close", set(r.grasp_choices.values()))
        # VSA 速度来自蓄能 (P1.3 live JSON), 非常数
        import math
        v_mag = math.hypot(*r.v_release)
        self.assertGreater(v_mag, 0.5)

    def test_com_advance_realistic(self):
        """双足推进: whole_body_3d 真实步态输出 (v0.7 已达 0.6m 量级)。"""
        r = run_climb_cycle(seed=42)
        self.assertGreater(r.com_advance_m, 0.3)
        self.assertLess(r.com_advance_m, 1.5)

    def test_soft_capture_below_rigid(self):
        """A2 核心物理结论: 柔顺分支峰值力显著低于刚性分支 (VSA-dyno 互证)。"""
        r = run_climb_cycle(seed=42)
        self.assertLess(r.f_soft_peak_n, r.f_rigid_peak_n)
        self.assertLess(r.soft_ratio, 0.8)   # 柔顺至少降 20%+

    def test_energy_scheduling_honest(self):
        """满速直弹超限被诚实记录 (物理结论不掩盖) + 调度后压线可行。"""
        r = run_climb_cycle(seed=42)
        self.assertTrue(r.full_speed_infeasible, "1.07m/s 满速直弹对 60N hold 必然超限")
        self.assertGreater(r.f_soft_full_speed_n, 100)
        # 调度后: 柔顺力压在 60N 限值附近 (±10%)
        self.assertAlmostEqual(r.f_soft_peak_n, 60.0, delta=6.0)
        self.assertTrue(r.capture_feasible)

    def test_flex_deformation_physical(self):
        """柔性预压形变量级: 凝胶指尖 2.67mm (1-10mm 文献范围)。"""
        r = run_climb_cycle(seed=42)
        self.assertGreater(r.delta_flex_m * 1000, 0.5)
        self.assertLess(r.delta_flex_m * 1000, 10.0)


class TestWallAndCoupling(unittest.TestCase):
    """墙面图谱 + 柔性耦合单元级。"""

    def test_wall_generation_domain_randomized(self):
        holds = generate_wall(seed=1)
        self.assertEqual(len(holds), 12)
        mus = [h.mu for h in holds]
        self.assertGreater(max(mus) - min(mus), 0.1, "μ 域随机化应产生差异")
        # 类型轮换覆盖三类
        types = {h.hold_type.value for h in holds}
        self.assertEqual(len(types), 3)

    def test_fingertip_stiffness_from_soft_body_config(self):
        """指尖刚度来自 soft_body 真实配置 (grasp_stiffness 3000 N/m)。"""
        tip = SoftBody(SoftBodyConfig(), seed=0)
        k = fingertip_stiffness_from_soft_body(tip)
        self.assertAlmostEqual(k, 3000.0, places=1)

    def test_soft_contact_monotone_in_speed(self):
        coup = SoftFingertipCoupling(k_struct_n_per_m=3000.0, f_preload_n=8.0,
                                     m_eff_kg=1.2, k_hc_n_per_m=10000.0)
        f1 = soft_contact_peak_force(coup, 0.3)["f_soft_coupled_peak_n"]
        f2 = soft_contact_peak_force(coup, 0.6)["f_soft_coupled_peak_n"]
        self.assertLess(f1, f2, "冲击速度越大柔顺峰值力越大")


class TestA3WallContact(unittest.TestCase):
    """A3: 斜壁多接触 — 4 肢雅可比 + 准静态力分配 + 蹲起换脚。"""

    def test_limb_jacobian_structure(self):
        """单肢雅可比: [I₃ | -[r]×] 结构与手算对照 (G1 资产公式)。"""
        com = np.zeros(3)
        c = np.array([0.3, 0.4, 0.1])
        J = limb_jacobian(c, com)
        self.assertEqual(J.shape, (3, 6))
        self.assertTrue(np.allclose(J[:, :3], np.eye(3)))
        r = c - com
        skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        self.assertTrue(np.allclose(J[:, 3:], -skew))

    def test_wall_frame_orthonormal(self):
        n, t = wall_frame(-10.0)
        self.assertAlmostEqual(np.dot(n, t), 0.0, places=12)
        self.assertAlmostEqual(np.linalg.norm(n), 1.0, places=12)
        # 仰角: 法向 y 分量应为正 (重力被法向分走 sin(10°))
        self.assertAlmostEqual(n[1], np.sin(np.radians(10.0)), places=3)

    def test_squat_stand_sequence_zero_unstable(self):
        """A3 验收: 斜壁蹲起-换脚全程 0 失稳 (摩擦锥/hold 载荷逐帧)。"""
        r = wall_stand_sequence()
        self.assertEqual(r["unstable_frames"], 0)
        self.assertTrue(r["ok"])
        self.assertEqual(r["n_frames"], 7)
        # 接触数序列: 4→4→3→4→3→4→4 (三肢帧存在 = 真正的换手脚)
        self.assertIn(3, [f["n_contacts"] for f in r["frames"]])

    def test_three_limb_hand_load_doubles(self):
        """物理正确性: 抬起一肢后手载荷应显著增大 (四肢体感验证)。"""
        cfg = WallContactConfig()
        st4 = QuadContactState(cfg, np.array([0.15, 0.3, 0]), np.array([-0.15, 0.3, 0]),
                               np.array([0.28, 1.1, 0]), np.array([-0.28, 1.1, 0]))
        st3 = QuadContactState(cfg, np.array([0.15, 0.3, 0]), np.array([-0.15, 0.3, 0]),
                               None, np.array([-0.28, 1.1, 0]))
        r4 = allocate_forces(st4, np.array([0, 0.55, 0]))
        r3 = allocate_forces(st3, np.array([0, 0.55, 0]))
        self.assertGreater(r3.max_hand_load_n, r4.max_hand_load_n * 1.3,
                           "三肢时手载荷应 > 四肢的 1.3 倍")

    def test_climb_cycle_consumes_wall_contact(self):
        """集成: climb_scene 周期真实消费 A3 斜壁段 (帧数+失稳数入结果)。"""
        r = run_climb_cycle(seed=42)
        self.assertEqual(r.wall_frames, 7)
        self.assertEqual(r.wall_unstable, 0)
        self.assertLess(r.wall_max_hand_n, 150.0, "手挂点载荷在 hold 极限内")


class TestA4DexterousGrasp(unittest.TestCase):
    """A4: 抓握选择 = dexterous_hand 力闭合质量驱动; 手接触 = 指分布模型。"""

    def test_grasp_choice_is_quality_driven(self):
        """选择由 ε 数值计算决定 (非 HoldType 硬映射): 每类 hold 的三候选 ε 有差异且选最大。"""
        r = run_climb_cycle(seed=42)
        self.assertTrue(r.grasp_quality_scores, "ε 计算结果应存在")
        for hid, scores in r.grasp_quality_scores.items():
            # 三类候选 ε 均被真实计算 (非零差异 → 非映射)
            self.assertEqual(len(scores), 3)
            vals = list(scores.values())
            self.assertGreater(max(vals) - min(vals), 1e-4,
                               f"hold {hid} 三候选 ε 应有数值差异")
            # 选择 = argmax ε
            self.assertEqual(r.grasp_choices[hid], max(scores, key=scores.get))

    def test_deep_power_wins_on_deep_pocket(self):
        """物理正确性: 深口袋 hold (seed 42 下 hold 2) 包络抓握 ε 最高。"""
        r = run_climb_cycle(seed=42)
        s2 = r.grasp_quality_scores[2]
        self.assertEqual(r.grasp_choices[2], "deep_power",
                         f"深口袋应选包络: {s2}")

    def test_fingertip_limit_enforced(self):
        """A4 双重约束: 手部总拉力经指分布后单指峰值 ≤ 60N Hertz 极限。"""
        r = run_climb_cycle(seed=42)
        self.assertLess(r.peak_finger_n, 60.0)
        self.assertTrue(r.fingertip_ok)
        # 三肢场景 (77.3N 总拉): 最弱指 42.3% 份额 → 32.7N
        self.assertAlmostEqual(r.peak_finger_n, 77.3 * 0.423, delta=1.0)

    def test_arm_approx_replaced(self):
        """ARM_APPROX 已关闭: 手接触为灵巧手指分布模型 (固连质量点已退役)。"""
        from embodied_ai.simulation import wall_contact as wc
        self.assertFalse(wc.ARM_APPROX, "A4 后 ARM_APPROX 应为 False")
        hc = wc.DexterousHandContact()
        dist = hc.distribute(77.3)
        self.assertEqual(len(dist), 5, "指分布应有 5 指 (F5)")
        self.assertGreater(max(dist.values()), min(dist.values()),
                           "欠驱动腱耦合下指力非均匀")

    def test_tendon_coupling_consumed(self):
        """指份额来自 underactuated_coupling 真实计算 (远端指承载最大)。"""
        from embodied_ai.simulation.wall_contact import DexterousHandContact
        import numpy as np
        hc = DexterousHandContact()
        self.assertTrue(np.all(np.diff(hc.finger_shares) > 0),
                       "腱耦合下各指份额递增 (远端最快承载最大)")
        self.assertAlmostEqual(hc.finger_shares.sum(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
