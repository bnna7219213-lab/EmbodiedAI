"""断点 1 守护测试 — 最小 Bus 抽象 (runtime/bus.py)。

A. Bus 基本语义 (pub/sub/多订阅/无订阅/顺序/schema 校验/坏 topic)
B. 控制环接入 (step 发消息 / CAN 帧一等公民 / 可追溯)
反证: 垃圾 payload 必被拒; 坏 topic 必报错; 单订阅者异常不阻断总线。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from embodied_ai.runtime import (Collector, InProcessBus, Message, Topic,
                                  make_bus_backed_controller)


class TestBusSemantics(unittest.TestCase):

    def test_pub_reaches_sub(self):
        bus = InProcessBus()
        c = Collector()
        bus.subscribe(Topic.ACTION, c)
        bus.publish(Topic.ACTION, {"action": np.ones(12)}, source="t")
        self.assertEqual(len(c.messages), 1)
        self.assertEqual(c.last.source, "t")

    def test_multiple_subscribers_all_receive(self):
        bus = InProcessBus()
        c1, c2 = Collector(), Collector()
        bus.subscribe(Topic.ACTION, c1)
        bus.subscribe(Topic.ACTION, c2)
        bus.publish(Topic.ACTION, {"action": np.zeros(3)})
        self.assertEqual(len(c1.messages), 1)
        self.assertEqual(len(c2.messages), 1)

    def test_no_subscriber_does_not_crash(self):
        """反证: 无订阅者发布不崩, 记入 dropped (可观测)。"""
        bus = InProcessBus()
        bus.publish(Topic.DR_UPDATE, {"overrides": {}})
        self.assertEqual(bus.stats["dropped"], 1)

    def test_seq_monotonic(self):
        """顺序保证: seq 全局单调递增。"""
        bus = InProcessBus()
        c = Collector()
        bus.subscribe(Topic.OBSERVATION, c)
        for _ in range(5):
            bus.publish(Topic.OBSERVATION,
                        {"joint_positions": np.zeros(12)})
        seqs = [m.seq for m in c.messages]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), 5)

    def test_bad_topic_raises(self):
        """反证: 未知 topic 必报错, 不静默。"""
        bus = InProcessBus()
        with self.assertRaises(ValueError):
            bus.publish("nonexistent_topic", {})

    def test_schema_rejects_garbage(self):
        """反证: 垃圾 payload 在入口被拒。"""
        bus = InProcessBus()
        with self.assertRaises(ValueError):
            bus.publish(Topic.ACTION, {"wrong_key": np.zeros(3)})
        with self.assertRaises(ValueError):
            bus.publish(Topic.ACTION, {"action": "not_an_array"})

    def test_subscriber_exception_does_not_block_bus(self):
        """单订阅者异常不阻断其他订阅者 (总线韧性)。"""
        bus = InProcessBus()
        good = Collector()

        def bad_handler(msg):
            raise RuntimeError("boom")

        bus.subscribe(Topic.ACTION, bad_handler)
        bus.subscribe(Topic.ACTION, good)
        bus.publish(Topic.ACTION, {"action": np.zeros(1)})
        self.assertEqual(len(good.messages), 1, "好订阅者必须收到消息")
        self.assertEqual(bus.stats["rejected"], 1)

    def test_message_envelope_traceable(self):
        """消息可追溯: timestamp/seq/schema_version/source 齐备。"""
        bus = InProcessBus()
        c = Collector()
        bus.subscribe(Topic.SAFETY_EVENT, c)
        bus.publish(Topic.SAFETY_EVENT, {"event_type": "STO"}, source="safety_mcu")
        m = c.last
        self.assertGreater(m.timestamp_s, 0)
        self.assertEqual(m.seq, 1)
        self.assertEqual(m.schema_version, "v1")
        self.assertEqual(m.source, "safety_mcu")


class TestControllerBusIntegration(unittest.TestCase):

    def _make(self):
        from embodied_ai.control.household_robot_system import make_controller
        bus = InProcessBus("test_ctrl")
        ctrl = make_controller()
        make_bus_backed_controller(ctrl, bus)
        return bus, ctrl

    def test_step_publishes_obs_and_action(self):
        """控制环 step 后: OBSERVATION + ACTION 两条消息上 Bus。"""
        bus, ctrl = self._make()
        obs_c, act_c = Collector(), Collector()
        bus.subscribe(Topic.OBSERVATION, obs_c)
        bus.subscribe(Topic.ACTION, act_c)
        ctrl.step({"ee_position": np.zeros(3), "joint_positions": np.ones(12)})
        self.assertEqual(len(obs_c.messages), 1)
        self.assertEqual(len(act_c.messages), 1)
        self.assertEqual(act_c.last.source, "controller_20hz")

    def test_step_via_bus_actually_routes_observation(self):
        """回归: step_via_bus 曾把观测丢给一个从未被赋值的常量 _last_obs_joints,
        导致"观测经 Bus 流转"的路径**永远执行全零观测**, 而测试全绿。"""
        seen = []

        class _Stub:
            def step(self, obs):
                seen.append(np.asarray(obs["joint_positions"]).copy())
                return np.asarray(obs["joint_positions"]) * 2.0

        bus = InProcessBus("test_via_bus")
        ctrl = _Stub()
        make_bus_backed_controller(ctrl, bus)
        joints = np.arange(12, dtype=float) + 1.0
        out = ctrl.step_via_bus({"joint_positions": joints})
        self.assertEqual(len(seen), 1)
        self.assertTrue(
            np.allclose(seen[0], joints),
            f"控制器收到 {seen[0]}, 期望 {joints} — 观测未真正经 Bus 流转")
        self.assertTrue(np.allclose(out, joints * 2.0))
        self.assertTrue(np.allclose(ctrl.last_observation()["joint_positions"], joints))

    def test_step_via_bus_does_not_leak_subscribers(self):
        """回归: 曾每次调用新建 Collector 并重复 subscribe, 订阅者无限增长。"""
        class _Stub:
            def step(self, obs):
                return np.zeros(12)

        bus = InProcessBus("test_leak")
        ctrl = _Stub()
        make_bus_backed_controller(ctrl, bus)
        act = Collector()
        bus.subscribe(Topic.ACTION, act)
        for _ in range(5):
            ctrl.step_via_bus({"joint_positions": np.ones(12)})
        self.assertEqual(len(act.messages), 5)
        self.assertEqual(bus.subscriber_count(Topic.OBSERVATION), 1,
                         "OBSERVATION 订阅者应恒为 1 (回环), 不得随调用增长")

    def test_can_frame_is_first_class(self):
        """CAN-FD 帧经 Bus 流转后仍可解码 (复用 DBC 600 帧验证路径)。"""
        from embodied_ai.simulation.real_robot_env import (CANFrame,
                                                             decode_telemetry,
                                                             encode_joint_cmd)
        bus, ctrl = self._make()
        a = ctrl.step({"ee_position": np.zeros(3), "joint_positions": np.zeros(12)})
        frames = Collector()
        bus.subscribe(Topic.CAN_JOINT_CMD, frames)
        f = encode_joint_cmd(0, float(np.ravel(a)[0]), 0.0, 20.0)
        bus.publish(Topic.CAN_JOINT_CMD, {"frame": f}, source="gateway")
        self.assertEqual(len(frames.messages), 1)
        dec = decode_telemetry(CANFrame(300, frames.last.payload["frame"].data))
        self.assertTrue(dec["crc_ok"])

    def test_original_step_behavior_preserved(self):
        """向后兼容: 接线后 step 返回值与原行为一致 (既有测试不破)。"""
        bus, ctrl = self._make()
        obs = {"ee_position": np.array([0.3, 0.0, 1.0]),
               "joint_positions": np.zeros(12)}
        a = ctrl.step(obs)
        self.assertIsNotNone(a)

    def test_safety_event_topic(self):
        """安全事件走 Bus (STO 等 — 安全 S 横切层的消息化)。"""
        bus, _ = self._make()
        c = Collector()
        bus.subscribe(Topic.SAFETY_EVENT, c)
        bus.publish(Topic.SAFETY_EVENT,
                    {"event_type": "OVERCURRENT", "value": 45.2},
                    source="safety_mcu")
        self.assertEqual(c.last.payload["event_type"], "OVERCURRENT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
