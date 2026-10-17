"""N2 中枢编排 CenterBrain 测试 (plan_v6 Sprint N)。

反证: POSE_UPDATE 真发布且 hub 收到 (B8 断链修复); 四要素事件齐全;
双路径选择正确; V2L 论证非空; act_fn 被调用; 无 Bus 不崩。
"""
import numpy as np
import pytest

from embodied_ai.center.center_brain import CenterBrain, CenterStepResult
from embodied_ai.cognition.cognitive_hub import CognitiveHub
from embodied_ai.models.vla_config import VLAConfig


CFG = VLAConfig(modality="vla")


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(CFG.num_views, CFG.vision_embd)),
            rng.normal(size=CFG.state_dim),
            rng.normal(size=CFG.LANG_TOKENS))


class TestPoseChain:
    def test_pose_update_reaches_hub(self):
        """B8 断链修复守护: POSE_UPDATE 发布 → hub.latest_pose() 收到。"""
        from embodied_ai.runtime.bus import InProcessBus

        bus = InProcessBus()
        hub = CognitiveHub(bus=bus)
        brain = CenterBrain(bus=bus, hub=hub)
        v, s, _ = _inputs()
        brain.step(0.0, None, 1.0, v, s, pose_xyz=np.array([1.5, 2.5, 0.0]))
        assert hub.latest_pose("robot") == pytest.approx([1.5, 2.5, 0.0])

    def test_pose_latest_wins(self):
        from embodied_ai.runtime.bus import InProcessBus

        bus = InProcessBus()
        hub = CognitiveHub(bus=bus)
        brain = CenterBrain(bus=bus, hub=hub)
        v, s, _ = _inputs()
        brain.step(0.0, None, 1.0, v, s, pose_xyz=np.array([1.0, 0.0, 0.0]))
        brain.step(1.0, None, 1.0, v, s, pose_xyz=np.array([2.0, 0.0, 0.0]))
        assert hub.latest_pose("robot") == pytest.approx([2.0, 0.0, 0.0])

    def test_event_location_from_pose(self):
        from embodied_ai.runtime.bus import InProcessBus

        bus = InProcessBus()
        hub = CognitiveHub(bus=bus)
        brain = CenterBrain(bus=bus, hub=hub)
        v, s, _ = _inputs()
        r = brain.step(0.0, None, 1.0, v, s,
                       pose_xyz=np.array([3.0, 4.0, 0.0]), thing="grasp")
        assert r.event.location_xyz == pytest.approx([3.0, 4.0, 0.0])
        assert "center" in r.event.source


class TestDecisionPath:
    def test_no_language_uses_v2a(self):
        brain = CenterBrain()
        v, s, _ = _inputs()
        r = brain.step(0.0, None, 1.0, v, s, lang_ids=None)
        assert r.decision_path == "v2a"

    def test_language_uses_vla(self):
        brain = CenterBrain()
        v, s, lang = _inputs()
        r = brain.step(0.0, None, 1.0, v, s, lang_ids=lang)
        assert r.decision_path == "vla"

    def test_action_shape(self):
        brain = CenterBrain()
        v, s, _ = _inputs()
        r = brain.step(0.0, None, 1.0, v, s)
        assert r.action.shape == (CFG.action_horizon, CFG.action_dim)


class TestSpatialArgument:
    def test_argument_present(self):
        brain = CenterBrain()
        v, s, _ = _inputs()
        r = brain.step(0.0, None, 1.0, v, s,
                       goal_vec=np.array([1.0, 0.0, 0.0]))
        assert len(r.spatial_argument.derivation_steps) > 0
        assert r.spatial_argument.conclusion


class TestActuationAndMemory:
    def test_act_fn_called(self):
        calls = []
        brain = CenterBrain(act_fn=lambda thing, a: calls.append(thing))
        v, s, _ = _inputs()
        brain.step(0.0, None, 1.0, v, s, thing="approach")
        assert calls == ["approach"]

    def test_events_via_bus_single_record(self):
        from embodied_ai.runtime.bus import InProcessBus

        bus = InProcessBus()
        hub = CognitiveHub(bus=bus)
        brain = CenterBrain(bus=bus, hub=hub)
        v, s, _ = _inputs()
        brain.step(0.0, None, 1.0, v, s, thing="a")
        assert len(hub.memory) == 1, "事件不得双重入账"

    def test_events_without_bus(self):
        hub = CognitiveHub()
        brain = CenterBrain(hub=hub)
        v, s, _ = _inputs()
        brain.step(0.0, None, 1.0, v, s, thing="a")
        assert len(hub.memory) == 1


class TestDemo:
    def test_run_demo(self):
        from embodied_ai.runtime.bus import InProcessBus

        bus = InProcessBus()
        hub = CognitiveHub(bus=bus)
        brain = CenterBrain(bus=bus, hub=hub)
        out = brain.run_demo()
        assert out["paths"] == ["v2a", "vla", "v2a"]
        assert out["events"] == 3
        assert out["pose_seen_by_hub"] == [2.0, 0.5, 0.0]
        assert all(n > 0 for n in out["argument_steps"])
