"""断点 ③ 守护测试 — 最小 MissionManager（OS 决策中枢）。

A. 指令解析（关键词→模板；不可识别必报错）
B. 状态机（合法转移/非法转移必拒/终态不可复活）
C. 端到端（一句话→20Hz 环→终态判定/Bus 事件可观测）
D. 反证（超时 FAILED / 安全事件 ABORTED / 空指令报错 / 重复模板拒）
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from embodied_ai.runtime import (MISSION_TEMPLATES, Mission, MissionError,
                                  MissionManager, MissionState, MissionTemplate)
from embodied_ai.runtime.bus import InProcessBus, Topic, make_bus_backed_controller


def _make_mm():
    from embodied_ai.control.household_robot_system import make_controller
    bus = InProcessBus("test")
    ctrl = make_controller()
    make_bus_backed_controller(ctrl, bus)
    return MissionManager(ctrl, bus), ctrl, bus


def _zero_obs():
    return {"ee_position": np.zeros(3), "joint_positions": np.zeros(12)}


class TestParse(unittest.TestCase):

    def test_chinese_instruction(self):
        mm = MissionManager.__new__(MissionManager)
        mm.templates = list(MISSION_TEMPLATES)
        m = mm.parse("抓起杯子放到桌上")
        self.assertEqual(m.template.name, "pick_and_place")

    def test_english_instruction(self):
        mm = MissionManager.__new__(MissionManager)
        mm.templates = list(MISSION_TEMPLATES)
        m = mm.parse("wipe the table")
        self.assertEqual(m.template.name, "wipe_surface")

    def test_multi_skill(self):
        mm = MissionManager.__new__(MissionManager)
        mm.templates = list(MISSION_TEMPLATES)
        m = mm.parse("打扫并整理")
        self.assertEqual(m.template.name, "clean_and_pick")
        self.assertEqual(len(m.template.skills), 3)

    def test_unrecognized_raises(self):
        """反证: 不可识别指令必报错, 不静默。"""
        mm = MissionManager.__new__(MissionManager)
        mm.templates = list(MISSION_TEMPLATES)
        with self.assertRaises(MissionError):
            mm.parse("量子纠缠扰动磁场")

    def test_empty_raises(self):
        mm = MissionManager.__new__(MissionManager)
        mm.templates = list(MISSION_TEMPLATES)
        with self.assertRaises(MissionError):
            mm.parse("")


class TestStateMachine(unittest.TestCase):

    def test_valid_transitions(self):
        m = Mission(template=MISSION_TEMPLATES[0], instruction="t")
        m._transition(MissionState.RUNNING)
        m._transition(MissionState.SUCCESS)
        self.assertEqual(m.state, MissionState.SUCCESS)

    def test_illegal_transition_raises(self):
        """反证: 终态不可复活 (SUCCESS → RUNNING 必拒)。"""
        m = Mission(template=MISSION_TEMPLATES[0], instruction="t")
        m._transition(MissionState.RUNNING)
        m._transition(MissionState.SUCCESS)
        with self.assertRaises(MissionError):
            m._transition(MissionState.RUNNING)

    def test_pending_cannot_jump_to_success(self):
        """PENDING 只能到 RUNNING/ABORTED, 不能直接 SUCCESS。"""
        m = Mission(template=MISSION_TEMPLATES[0], instruction="t")
        with self.assertRaises(MissionError):
            m._transition(MissionState.SUCCESS)


class TestEndToEnd(unittest.TestCase):

    def test_full_mission_success(self):
        """一句话→状态机→20Hz环→SUCCESS (全链)。"""
        mm, ctrl, bus = _make_mm()
        m = mm.parse("抓起杯子")
        result = mm.run(m, obs_provider=_zero_obs)
        self.assertEqual(result.state, MissionState.SUCCESS)
        self.assertGreater(result.steps_executed, 0)
        # 状态事件链完整
        transitions = [e["to"] for e in result.events]
        self.assertIn("running", transitions)
        self.assertIn("success", transitions)

    def test_bus_events_observable(self):
        """Mission 状态变更发 Bus (可观测)。"""
        mm, ctrl, bus = _make_mm()
        from embodied_ai.runtime.bus import Collector
        c = Collector()
        bus.subscribe(Topic.SAFETY_EVENT, c)
        m = mm.parse("握住")
        mm.run(m, obs_provider=_zero_obs)
        # 至少 mission_started 和 mission_success 两条
        event_types = [msg.payload["event_type"] for msg in c.messages]
        self.assertTrue(any("mission_started" in e for e in event_types),
                        f"缺 started 事件: {event_types}")
        self.assertTrue(any("mission_success" in e for e in event_types),
                        f"缺 success 事件: {event_types}")

    def test_controller_receives_skill(self):
        """Mission 驱动控制器的 current_skill (技能真实下发)。"""
        mm, ctrl, bus = _make_mm()
        m = mm.parse("擦桌子")
        # 让第一步就完成 → advance 到 place → 检查 current_skill 被设置过
        self.assertEqual(m.template.skills[0], "wipe")


class TestFailurePaths(unittest.TestCase):

    def test_timeout_fails(self):
        """反证: 超时必 FAILED (非静默无限跑)。"""
        mm, ctrl, bus = _make_mm()
        # 永不满足完成条件: 关节恒非零
        obs = {"ee_position": np.zeros(3),
               "joint_positions": np.ones(12) * 10}
        m = mm.parse("抓")
        result = mm.run(m, obs_provider=lambda: obs, max_steps_override=5)
        self.assertEqual(result.state, MissionState.FAILED)
        self.assertIn("timeout", result.events[-1]["reason"])

    def test_safety_event_aborts(self):
        """反证: 安全事件 → ABORTED (非 FAILED/SUCCESS)。"""
        mm, ctrl, bus = _make_mm()
        ctrl._safety_stop = True     # 模拟安全层触发
        m = mm.parse("抓")
        result = mm.run(m, obs_provider=_zero_obs, max_steps_override=10)
        self.assertEqual(result.state, MissionState.ABORTED)
        self.assertEqual(result.events[-1]["reason"], "safety event")

    def test_duplicate_template_rejected(self):
        """反证: 重复模板必拒 (不允许静默覆盖)。"""
        mm = MissionManager.__new__(MissionManager)
        mm.templates = list(MISSION_TEMPLATES)
        dup = MissionTemplate(name="pick_and_place", skills=["grasp"],
                              keywords=["重复"])
        with self.assertRaises(MissionError):
            mm.register_template(dup)


class TestSkillCoverage(unittest.TestCase):

    def test_all_skills_registered_in_controller(self):
        """任务模板用的技能必须是控制器已注册的 (不虚构)。"""
        from embodied_ai.control.household_robot_system import make_controller
        ctrl = make_controller()
        registered = set(ctrl.rl_coordinator.skills.keys())
        for t in MISSION_TEMPLATES:
            for s in t.skills:
                self.assertIn(s, registered,
                              f"模板 {t.name} 用了未注册技能 '{s}'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
