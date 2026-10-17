"""M4 认知中枢四要素 测试 (plan_v5 Sprint M)。

反证: 四要素完整性校验; 空记忆不崩; 多维查询; 五能力
(记录/分析/计算/规划/记忆); Bus 事件接入; 时间衰减召回。
"""
import numpy as np
import pytest

from embodied_ai.cognition import (
    CognitiveHub,
    EpisodicMemory,
    EventRecord,
    cognitive_demo,
)


def _ev(t, actor="robot", thing="approach", loc=(0.0, 0.0, 0.0), action=""):
    return EventRecord(t, np.array(loc), actor, thing, action)


class TestEventRecord:
    def test_four_elements(self):
        e = EventRecord(1.0, np.array([1.0, 2.0, 0.0]), "robot", "grasp",
                        "close", object_="cup", source="slam")
        t, loc, actor, thing = e.four_elements
        assert t == 1.0
        assert loc.shape == (3,)
        assert actor == "robot"
        assert thing == "grasp"
        assert e.object_ == "cup"

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError):
            EventRecord(-1.0, np.zeros(3), "a", "b")

    def test_empty_actor_raises(self):
        with pytest.raises(ValueError):
            EventRecord(0.0, np.zeros(3), "", "b")

    def test_empty_thing_raises(self):
        with pytest.raises(ValueError):
            EventRecord(0.0, np.zeros(3), "a", "")

    def test_location_must_be_3d(self):
        with pytest.raises(ValueError):
            EventRecord(0.0, np.zeros(2), "a", "b")


class TestEpisodicMemory:
    def test_record_and_query_all(self):
        mem = EpisodicMemory()
        mem.record(_ev(0.0))
        mem.record(_ev(1.0, actor="operator"))
        assert len(mem) == 2
        assert len(mem.query()) == 2

    def test_query_by_time_range(self):
        mem = EpisodicMemory()
        for t in (0.0, 5.0, 10.0):
            mem.record(_ev(t))
        assert len(mem.query(time_range=(4.0, 6.0))) == 1

    def test_query_by_actor_and_thing(self):
        mem = EpisodicMemory()
        mem.record(_ev(0.0, actor="robot", thing="grasp"))
        mem.record(_ev(1.0, actor="operator", thing="grasp"))
        mem.record(_ev(2.0, actor="robot", thing="place"))
        assert len(mem.query(actor="robot")) == 2
        assert len(mem.query(thing="grasp")) == 2
        assert len(mem.query(actor="robot", thing="grasp")) == 1

    def test_query_by_location_radius(self):
        mem = EpisodicMemory()
        mem.record(_ev(0.0, loc=(0.0, 0.0, 0.0)))
        mem.record(_ev(1.0, loc=(5.0, 0.0, 0.0)))
        near = mem.query(location_center=np.array([0.0, 0.0, 0.0]),
                         location_radius=1.0)
        assert len(near) == 1

    def test_query_location_half_specified_raises(self):
        """只给 center 不给 radius 曾静默跳过地点过滤 (99m 外事件照样返回)。"""
        mem = EpisodicMemory()
        mem.record(_ev(0.0, loc=(99.0, 99.0, 99.0)))
        with pytest.raises(ValueError, match="location_center"):
            mem.query(location_center=np.zeros(3))
        with pytest.raises(ValueError, match="location_center"):
            mem.query(location_radius=1.0)

    def test_empty_query_not_crash(self):
        assert EpisodicMemory().query() == []


class TestCognitiveHub:
    def test_five_capabilities(self):
        r = cognitive_demo()
        assert r["analysis"]["total"] == 3
        assert r["stats"]["event_rate_hz"] > 0
        assert r["plan_climb_hold"] == ["load"]
        assert r["recall_order"][0] == "handover"

    def test_record_and_analyze(self):
        hub = CognitiveHub()
        hub.record_event(_ev(0.0, thing="a"))
        hub.record_event(_ev(1.0, thing="b"))
        a = hub.analyze()
        assert a["total"] == 2
        assert a["sequence"] == ["a", "b"]
        assert a["time_span_s"] == pytest.approx(1.0)

    def test_analyze_empty(self):
        a = CognitiveHub().analyze()
        assert a["total"] == 0

    def test_compute_empty(self):
        assert CognitiveHub().compute() == {"event_rate_hz": 0.0}

    def test_compute_stats(self):
        hub = CognitiveHub()
        hub.record_event(_ev(0.0, loc=(0.0, 0.0, 0.0)))
        hub.record_event(_ev(2.0, loc=(2.0, 0.0, 0.0)))
        s = hub.compute()
        assert s["event_rate_hz"] == pytest.approx(0.5)
        assert s["centroid_x"] == pytest.approx(1.0)

    def test_plan_unknown_goal_raises(self):
        """未知目标曾返回 [], 与"前置条件已满足"完全不可区分。"""
        hub = CognitiveHub()
        with pytest.raises(KeyError, match="nonexistent"):
            hub.plan("nonexistent")
        # 已满足前置条件才是真正的空计划
        assert hub.plan("pick", available=["approach", "grasp"]) == []

    def test_plan_with_available(self):
        hub = CognitiveHub()
        assert hub.plan("pick", available=["approach", "grasp"]) == []
        assert hub.plan("pick", available=["approach"]) == ["grasp"]

    def test_remember_time_decay(self):
        hub = CognitiveHub()
        hub.record_event(_ev(0.0, thing="old"))
        hub.record_event(_ev(10.0, thing="new"))
        recall = hub.remember()
        assert [e.thing for e in recall] == ["new", "old"]
        # 显式 now_s 更大 → 顺序仍由衰减得分决定
        recall2 = hub.remember(now_s=100.0)
        assert recall2[0].thing == "new"

    def test_decay_rate_is_observable(self):
        """纯时间衰减是单调变换, 不改变次序 —— DECAY_RATE 曾是装饰参数
        (0.001 / 0.1 / 50.0 召回序完全相同)。得分必须随它变化才可验证。"""
        hub = CognitiveHub()
        hub.record_event(_ev(0.0, thing="old"))
        hub.record_event(_ev(10.0, thing="new"))
        scores = {}
        for rate in (0.01, 1.0):
            hub.DECAY_RATE = rate
            scores[rate] = dict(
                (e.thing, s) for s, e in hub.remember_scored(now_s=10.0))
        assert scores[0.01]["old"] == pytest.approx(np.exp(-0.01 * 10.0))
        assert scores[1.0]["old"] == pytest.approx(np.exp(-1.0 * 10.0))
        assert scores[0.01]["old"] > scores[1.0]["old"]
        # 最近事件得分恒为 1 (Δt=0)
        assert scores[0.01]["new"] == pytest.approx(1.0)
        # 次序不随 DECAY_RATE 改变 — 这正是必须暴露得分的原因
        assert [e.thing for _, e in hub.remember_scored(now_s=10.0)] == ["new", "old"]

    def test_remember_empty(self):
        assert CognitiveHub().remember() == []
        assert CognitiveHub().remember_scored() == []

    def test_bus_event_ingestion(self):
        from embodied_ai.runtime.bus import InProcessBus

        bus = InProcessBus()
        hub = CognitiveHub(bus=bus)
        rec = _ev(3.0, thing="bus_thing")
        bus.publish("event", {"record": rec}, source="test")
        assert len(hub.memory) == 1
        assert hub.memory.query()[0].thing == "bus_thing"

    def test_bus_bad_record_ignored(self):
        from embodied_ai.runtime.bus import InProcessBus

        bus = InProcessBus()
        hub = CognitiveHub(bus=bus)
        bus.publish("event", {"record": "not-an-EventRecord"}, source="test")
        assert len(hub.memory) == 0

    def test_slam_pose_reaches_cognition(self):
        """端到端: slam_demo(bus) → POSE_UPDATE → CognitiveHub.latest_pose。

        这是 plan_v5 宣称的"SLAM→PoseStore→认知中枢地点要素"数据流的**唯一**
        可执行证据 —— 此前 POSE_UPDATE 零发布者零订阅者, 该链路并不存在。
        """
        from embodied_ai.perception.slam import slam_demo
        from embodied_ai.runtime.bus import InProcessBus

        bus = InProcessBus("test_slam_cog")
        hub = CognitiveHub(bus=bus)
        r = slam_demo(bus=bus)
        assert r["snapshots"] == 50
        pose = hub.latest_pose("robot")
        assert pose is not None, "认知中枢没收到任何 SLAM 位姿"
        assert pose.shape == (3,)
        # 与 SLAM 自己的最后一条快照一致 (m): 沿 x 匀速 5s → x≈4.9
        assert pose[0] > 4.0, f"机器人 x 位姿 {pose[0]:.2f}m 与 50 步匀速不符"
        assert hub.latest_pose("ghost") is None
