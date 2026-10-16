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

    def test_plan_unknown_goal_empty(self):
        assert CognitiveHub().plan("nonexistent") == []

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

    def test_remember_empty(self):
        assert CognitiveHub().remember() == []

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
