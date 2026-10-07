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


class TestP0PhysicsFixes(unittest.TestCase):
    """P0 缺陷修复验证: 打破 Phase A 的恒真断言, 让物理约束真可失败。

    背景: 原 wall_contact 的摩擦锥校验 f_side ≡ 0 恒等于 0, wrench_balance_err
    是恒等式, com/pos 从未被引用, 力矩从未测过 —— "0 失稳"结论无可信度。
    本测试类验证: 侧向载荷能让摩擦锥真失败, 力矩残差真被计算并暴露,
    极限姿态真能触发失稳。
    """

    def _quad_state(self, cfg):
        return QuadContactState(cfg,
                                np.array([0.15, 0.30, 0.0]), np.array([-0.15, 0.30, 0.0]),
                                np.array([0.28, 1.10, 0.0]), np.array([-0.28, 1.10, 0.0]))

    def test_lateral_load_triggers_friction_failure(self):
        """P0#1 修复: 注入侧向载荷 → 摩擦锥 margins 真非零, 超限即失败。"""
        # 无侧向载荷: margins ≈ 0 (构造使然, 但此时也不该失败)
        r0 = allocate_forces(self._quad_state(WallContactConfig()), np.array([0.0, 0.80, 0.0]))
        self.assertAlmostEqual(r0.friction_margins["foot_left"], 0.0, places=12)
        self.assertTrue(r0.ok)

        # 递增侧向载荷: margins 应单调上升 (真实物理), 到 40N 前后达到临界
        margins = []
        for n in (10.0, 20.0, 30.0, 40.0):
            r = allocate_forces(self._quad_state(WallContactConfig(lateral_load_n=n)),
                                np.array([0.0, 0.80, 0.0]))
            margins.append(r.friction_margins["foot_left"])
        self.assertLess(margins[0], margins[-1],
                        "margins 随侧向载荷单调上升 (打破恒真)")
        # μ_foot=0.9, fn≈17.05 → 极限 15.34N 侧向力对应 margin=1.0
        # 每只脚承担 lateral*0.7/2 → lateral=43.8N 时 margin=1.0
        self.assertLess(margins[-1], 1.0, "40N 侧向仍在脚摩擦极限内")

        # 超限侧向: margin > 1 → 摩擦锥失败 → ok=False
        r_fail = allocate_forces(self._quad_state(WallContactConfig(lateral_load_n=60.0)),
                                 np.array([0.0, 0.80, 0.0]))
        self.assertGreater(r_fail.friction_margins["foot_left"], 1.0)
        self.assertFalse(r_fail.ok, "60N 侧向载荷应触发摩擦锥失败")

    def test_torque_balance_actually_computed(self):
        """P0#2 修复: 力矩残差 = Σ(p−com)×f 真被计算, 且三肢姿态 > 四肢。"""
        cfg = WallContactConfig()
        # 四肢: 静态残差 ~10.7 N·m (原代码从未暴露)
        st4 = self._quad_state(cfg)
        r4 = allocate_forces(st4, np.array([0.0, 0.80, 0.0]))
        self.assertGreater(r4.torque_balance_err, 5.0,
                           "四肢静态力矩残差非零 (原代码完全未测)")

        # 三肢 (抬左手): 力矩残差应显著增大 (重心转移的物理体现)
        st3 = QuadContactState(cfg, np.array([0.15, 0.30, 0.0]), np.array([-0.15, 0.30, 0.0]),
                               None, np.array([-0.28, 1.10, 0.0]))
        r3 = allocate_forces(st3, np.array([0.0, 0.80, 0.0]))
        self.assertGreater(r3.torque_balance_err, r4.torque_balance_err,
                           "三肢体力矩残差 > 四肢 (重心转移有物理体现)")

    def test_limb_jacobian_actually_consumed(self):
        """P0#3 修复: limb_jacobian 被真正调用 (原代码零调用)。"""
        import inspect
        src = inspect.getsource(allocate_forces)
        self.assertIn("limb_jacobian(", src,
                      "allocate_forces 必须调用 limb_jacobian (力矩解)")
        # com 形参必须被实际引用 (原代码形参未使用)
        self.assertIn("com", src)

    def test_static_unstable_frames_honestly_reported(self):
        """P0 修复: 静态失稳帧数诚实暴露 (原代码假称 0 失稳)。"""
        r = wall_stand_sequence()
        # 动态容差下 (默认 40 N·m) 无失稳 —— 攀岩换脚是动态动作
        self.assertEqual(r["unstable_frames"], 0)
        self.assertTrue(r["ok"])
        # 但严格静态 (15 N·m) 下三肢姿态真的失稳, 必须诚实报告
        self.assertGreater(r["static_unstable_frames"], 0,
                           "静态失稳帧数应诚实暴露 (原代码假称 0)")
        # 峰值力矩 = 三肢换 hold 瞬间 (~33 N·m)
        self.assertGreater(r["peak_torque_nm"], 25.0)

    def test_strict_dynamic_competency_can_fail(self):
        """P0 修复: 力矩容差阈值真可失败 (不是永远通过的装饰性阈值)。"""
        r_strict = wall_stand_sequence(WallContactConfig(dynamic_competency_nm=10.0))
        self.assertGreater(r_strict["unstable_frames"], 0,
                           "严格容差下应有失稳帧 (阈值真可失败)")
        self.assertFalse(r_strict["ok"])

    def test_extreme_pose_forces_failure(self):
        """P0 修复: 单脚单手 + 额外负重 → 力矩爆表, 诚实判失败。"""
        cfg = WallContactConfig()
        st = QuadContactState(cfg, np.array([0.15, 0.30, 0.0]), None,
                              np.array([0.28, 1.10, 0.0]), None)
        r = allocate_forces(st, np.array([0.0, 0.80, 0.0]), extra_load_n=50.0)
        self.assertFalse(r.ok, "单脚单手持重应失稳 (原代码可能假通过)")
        self.assertGreater(r.torque_balance_err, 30.0,
                           "力矩远超动态补偿能力")

    def test_force_balance_residual_not_tautology(self):
        """P0 修复: 力平衡残差现在依赖外力一致性, 不是构造性恒等式。"""
        # 无外力: 构造保证 Σf = -total_w → err ≈ 0
        r0 = allocate_forces(self._quad_state(WallContactConfig()), np.array([0.0, 0.80, 0.0]))
        self.assertLess(r0.wrench_balance_err, 0.5,
                        "无外力时力平衡 (构造正确性)")
        # 力平衡现在真的验证了 Σf + total_w (total_w 包含 lateral_load_n)
        r_side = allocate_forces(self._quad_state(WallContactConfig(lateral_load_n=30.0)),
                                 np.array([0.0, 0.80, 0.0]))
        self.assertLess(r_side.wrench_balance_err, 0.5,
                        "有侧向外力时接触力正确反向平衡 (方向修复)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
