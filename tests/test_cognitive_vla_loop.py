"""M6 VLA 双路径 × 认知中枢 端到端闭环 测试 (plan_v5 Sprint M)。

反证: 无语言必走 v2a / 有语言必走 vla; 事件四要素齐全且时间戳单调;
闭环跑到目标达成 (plan 返回空); 未知目标 KeyError 透传; act_fn 被调用;
地点要素取 SLAM 位姿, 缺位姿时诚实标注 assumed_zero。
"""
import numpy as np
import pytest

from embodied_ai.cognition import (
    CognitiveHub,
    CognitiveVLALoop,
    EventRecord,
    loop_demo,
)
from embodied_ai.models.vla_config import VLAConfig
from embodied_ai.models.vla_v2a import VLAPolicy


CFG = VLAConfig(modality="vla")


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    vision = rng.normal(size=(CFG.num_views, CFG.vision_embd))
    state = rng.normal(size=CFG.state_dim)
    lang = rng.normal(size=CFG.LANG_TOKENS)
    return vision, state, lang


class TestDecide:
    def test_no_language_uses_v2a(self):
        loop = CognitiveVLALoop(CognitiveHub(), cfg=CFG)
        vision, state, _ = _inputs()
        path, action = loop.decide(vision, state, None)
        assert path == "v2a"
        assert action.shape == (CFG.action_horizon, CFG.action_dim)

    def test_with_language_uses_vla(self):
        loop = CognitiveVLALoop(CognitiveHub(), cfg=CFG)
        vision, state, lang = _inputs()
        path, _ = loop.decide(vision, state, lang)
        assert path == "vla"

    def test_v2a_modality_rejects_language_transparent(self):
        """v2a 模态策略拒绝语言 — 闭环 decide 传语言必须透传 ValueError。"""
        cfg = VLAConfig(modality="v2a")
        loop = CognitiveVLALoop(CognitiveHub(), cfg=cfg)
        vision, state, lang = _inputs()
        with pytest.raises(ValueError):
            loop.decide(vision, state, lang)


class TestStep:
    def test_step_records_four_elements(self):
        hub = CognitiveHub()
        loop = CognitiveVLALoop(hub, cfg=CFG)
        vision, state, _ = _inputs()
        r = loop.step("grasp", vision, state, None, object_="cup")
        assert hub.memory.query()[0].thing == "grasp"
        t, loc, actor, thing = r.event.four_elements
        assert t > 0
        assert loc.shape == (3,)
        assert actor == "robot"
        assert thing == "grasp"
        assert r.event.object_ == "cup"
        assert r.event.source.startswith("v2a")

    def test_step_invokes_act_fn(self):
        calls = []
        hub = CognitiveHub()
        loop = CognitiveVLALoop(hub, cfg=CFG,
                                act_fn=lambda thing, a: calls.append(thing))
        vision, state, _ = _inputs()
        loop.step("approach", vision, state, None)
        assert calls == ["approach"]

    def test_timestamps_monotonic(self):
        hub = CognitiveHub()
        clock = [0.0]

        def _tick():
            clock[0] += 1.0
            return clock[0]

        loop = CognitiveVLALoop(hub, cfg=CFG, clock=_tick)
        vision, state, _ = _inputs()
        ts = [loop.step("a", vision, state, None).event.timestamp_s for _ in range(3)]
        assert ts == sorted(ts) and len(set(ts)) == 3

    def test_bad_clock_rejected(self):
        hub = CognitiveHub()
        loop = CognitiveVLALoop(hub, cfg=CFG, clock=lambda: 5.0)  # 恒定 → 非单调
        vision, state, _ = _inputs()
        loop.step("a", vision, state, None)
        with pytest.raises(ValueError):
            loop.step("b", vision, state, None)

    def test_location_from_slam_pose(self):
        from embodied_ai.runtime.bus import InProcessBus

        bus = InProcessBus()
        hub = CognitiveHub(bus=bus)
        bus.publish("pose_update",
                    {"pose": np.array([3.0, 4.0, 0.0]), "who": "robot",
                     "timestamp_s": 0.0}, source="slam")
        loop = CognitiveVLALoop(hub, cfg=CFG)
        vision, state, _ = _inputs()
        r = loop.step("grasp", vision, state, None)
        assert r.event.location_xyz == pytest.approx([3.0, 4.0, 0.0])
        assert "slam" in r.event.source

    def test_location_assumed_zero_when_no_pose(self):
        hub = CognitiveHub()
        loop = CognitiveVLALoop(hub, cfg=CFG)
        vision, state, _ = _inputs()
        r = loop.step("grasp", vision, state, None)
        assert np.allclose(r.event.location_xyz, 0.0)
        assert "assumed_zero" in r.event.source, "缺位姿必须诚实标注"


class TestRunUntilGoal:
    def test_v2a_loop_reaches_goal(self):
        hub = CognitiveHub()
        loop = CognitiveVLALoop(hub, cfg=CFG)
        vision, state, _ = _inputs()
        rep = loop.run_until_goal("climb_hold", vision, state, None)
        assert rep.goal_reached
        assert [s.thing for s in rep.steps] == ["approach", "grasp", "load"]
        assert rep.paths_used == {"v2a": 3}
        assert len(hub.memory) == 3
        assert rep.plan_remaining == []

    def test_vla_loop_reaches_goal(self):
        hub = CognitiveHub()
        loop = CognitiveVLALoop(hub, cfg=CFG)
        vision, state, lang = _inputs()
        rep = loop.run_until_goal("climb_hold", vision, state, lang)
        assert rep.goal_reached
        assert rep.paths_used == {"vla": 3}

    def test_unknown_goal_keyerror_passes_through(self):
        loop = CognitiveVLALoop(CognitiveHub(), cfg=CFG)
        vision, state, _ = _inputs()
        with pytest.raises(KeyError):
            loop.run_until_goal("nonexistent", vision, state, None)

    def test_max_steps_guard(self):
        """plan 缺口不动 (act_fn 不记录新 thing 也不影响 memory) → 必须抛 RuntimeError。"""
        hub = CognitiveHub()
        loop = CognitiveVLALoop(hub, cfg=CFG)
        vision, state, _ = _inputs()
        # max_steps=2 < 需要 3 步
        with pytest.raises(RuntimeError):
            loop.run_until_goal("climb_hold", vision, state, None, max_steps=2)

    def test_events_enter_memory_with_source(self):
        hub = CognitiveHub()
        loop = CognitiveVLALoop(hub, cfg=CFG)
        vision, state, lang = _inputs()
        rep = loop.run_until_goal("pick", vision, state, lang)
        assert rep.goal_reached
        things = [e.thing for e in hub.memory.query()]
        assert things == ["approach", "grasp"]
        assert all(e.source.startswith("vla") for e in hub.memory.query())


class TestLoopDemo:
    def test_demo_assertions(self):
        r = loop_demo()
        assert r["v2a"]["goal_reached"] and r["vla"]["goal_reached"]
        assert r["v2a"]["paths"] == {"v2a": 3}
        assert r["vla"]["paths"] == {"vla": 3}
        assert r["v2a"]["events"] == 3

    def test_demo_uses_real_policy_shapes(self):
        """闭环动作形状必须与 VLAPolicy 契约一致 (不漂移)。"""
        hub = CognitiveHub()
        p = VLAPolicy(CFG)
        loop = CognitiveVLALoop(hub, policy=p)
        vision, state, _ = _inputs()
        _, action = loop.decide(vision, state, None)
        assert action.shape == (CFG.action_horizon, CFG.action_dim)
