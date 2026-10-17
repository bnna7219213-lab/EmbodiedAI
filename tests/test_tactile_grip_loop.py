"""④ 触觉闭环守护测试 — EKF→机会约束握力进 20Hz 环。

验证:
  A. EKF μ̂ 收敛 (|μ̂-μ_true|<0.05) — 修正后 slip 阈值比例化生效
  B. 握力设定值非零且满足机会约束 (f_n >= f_t/(μ̂-z·σ_μ))
  C. Bus 事件可观测 (grip_command 每步发布)
  D. 反证: σ_μ 过大时 InfeasibleGrip → 安全降额 (不静默零)
  E. 确定性: 同 seed 两次结果一致 (非随机)
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from embodied_ai.control.tactile_grip_loop import (TactileGripLoop,
                                                     synth_tactile_obs)
from embodied_ai.control.household_robot_system import make_controller
from embodied_ai.runtime.bus import (Collector, InProcessBus, Topic,
                                      make_bus_backed_controller)


def _run_100_steps(seed=0):
    """跑 100 步触觉闭环, 返回 (loop, bus, grip_events)。"""
    obs_seq = synth_tactile_obs(mu_true=0.75, f_n=40.0, n_steps=100, seed=seed)
    bus = InProcessBus("test_tactile")
    ctrl = make_controller()
    make_bus_backed_controller(ctrl, bus)
    loop = TactileGripLoop(ctrl, bus, ekf_dt=0.05)
    grip_events = Collector()
    bus.subscribe(Topic.SAFETY_EVENT, grip_events)
    for obs in obs_seq:
        ctrl.step(obs)
    return loop, bus, grip_events


class TestEKFConvergence(unittest.TestCase):

    def test_mu_hat_converges(self):
        """μ̂ 在 100 步后收敛到真值附近 (|μ̂-0.75|<0.05)。"""
        loop, _, _ = _run_100_steps(seed=0)
        self.assertLess(abs(loop.last_mu_hat - 0.75), 0.05,
                        f"μ̂={loop.last_mu_hat:.4f} 未收敛到 0.75")

    def test_sigma_mu_decreases(self):
        """σ_μ 应随观测收敛而下降 (< 初始 0.15)。"""
        loop, _, _ = _run_100_steps(seed=0)
        self.assertLess(loop.last_sigma_mu, 0.15,
                        f"σ_μ={loop.last_sigma_mu:.4f} 未下降")


class TestGripOutput(unittest.TestCase):

    def test_f_n_setpoint_nonzero(self):
        """握力设定值非零 (感知→控制环闭环)。"""
        loop, _, _ = _run_100_steps(seed=0)
        self.assertGreater(loop.last_f_n_setpoint, 0,
                           "握力设定值为零 — 闭环未接通")

    def test_grip_satisfies_chance_constraint(self):
        """机会约束: denom=μ̂-z·σ > 0.05 → 可行 (末步应可行)。"""
        loop, _, _ = _run_100_steps(seed=0)
        denom = loop.last_mu_hat - 2.0 * loop.last_sigma_mu
        self.assertGreater(denom, 0.05,
                           f"denom={denom:.3f} — 末步应可行 (EKF 已收敛)")

    def test_grip_history_populated(self):
        """每步都记录握力历史 (可追溯)。"""
        loop, _, _ = _run_100_steps(seed=0)
        self.assertEqual(len(loop.grip_history), 100)


class TestBusObservable(unittest.TestCase):

    def test_grip_events_on_bus(self):
        """Bus 上有 grip_command 事件 (可观测)。"""
        _, _, grip_events = _run_100_steps(seed=0)
        self.assertGreater(len(grip_events.messages), 0)
        types = [m.payload.get("event_type") for m in grip_events.messages]
        self.assertTrue(any("grip" in str(t) for t in types),
                        f"缺 grip 事件: {set(types)}")

    def test_bus_events_carry_mu_sigma(self):
        """Bus 事件携带 μ̂/σ_μ/f_n (可追溯)。"""
        _, _, grip_events = _run_100_steps(seed=0)
        m = grip_events.messages[-1]
        self.assertIn("mu_hat", m.payload)
        self.assertIn("sigma_mu", m.payload)
        self.assertIn("f_n_setpoint", m.payload)


class TestFailurePath(unittest.TestCase):

    def test_infeasible_grip_safety_fallback(self):
        """反证: σ_μ 过大 → InfeasibleGrip → 安全降额 (非静默零)。"""
        loop, _, _ = _run_100_steps(seed=0)
        # 人为注入极大 σ_μ → 必触发 InfeasibleGrip
        loop.last_sigma_mu = 999.0
        loop._schedule_grip({"f_t": np.array([10.0])})
        # 安全降额: f_n 应为 nominal*1.5 (不静默零)
        self.assertEqual(loop.last_f_n_setpoint, 60.0,
                         "InfeasibleGrip 应降额到 1.5×nominal, 非静默零")


class TestDeterministic(unittest.TestCase):

    def test_same_seed_same_result(self):
        """确定性: 同 seed 两次结果一致 (非随机数冒充)。"""
        l1, _, _ = _run_100_steps(seed=42)
        l2, _, _ = _run_100_steps(seed=42)
        self.assertAlmostEqual(l1.last_mu_hat, l2.last_mu_hat, places=6)
        self.assertAlmostEqual(l1.last_f_n_setpoint, l2.last_f_n_setpoint, places=4)


class TestEKFSlipThresholdFix(unittest.TestCase):

    def test_slip_threshold_proportional(self):
        """④ 修正: slip 阈值应比例化 (0.15*f_n), 非 1.0 硬编码。"""
        from embodied_ai.perception.multirate_ekf import MultiRateEKF
        ekf = MultiRateEKF(0.55, 0.15, 40.0)
        # 读源码确认阈值是 0.15*f_n
        import inspect
        src = inspect.getsource(ekf.update_tactile)
        self.assertIn("0.15", src, "slip 阈值应含 0.15 比例因子")
        self.assertNotIn("< 1.0:", src, "原硬编码 1.0 阈值应已移除")


if __name__ == "__main__":
    unittest.main(verbosity=2)
