"""d3 真实硬件 E2E 闭环测试支架 测试 (plan_v6 后续)。

反证: 跳阶段必报错; S4 时延超预算必 FAIL (不是恒过); ESTOP 中断闭环;
每阶段产出可审计; mock 通道全链可跑到 DONE。
"""
import numpy as np
import pytest

from embodied_ai.center.center_brain import CenterBrain
from embodied_ai.cognition.cognitive_hub import CognitiveHub
from embodied_ai.hil.e2e_closed_loop import (
    PHASE_ORDER,
    E2EClosedLoopHarness,
    E2EPhase,
    MockHardwareChannel,
    e2e_demo,
)
from embodied_ai.runtime.bus import InProcessBus


def _harness(latency_ms=12.0, budget=50.0, with_brain=True):
    bus = InProcessBus()
    hub = CognitiveHub(bus=bus)
    brain = CenterBrain(bus=bus, hub=hub) if with_brain else None
    return E2EClosedLoopHarness(
        channel=MockHardwareChannel(latency_ms=latency_ms),
        center_brain=brain, latency_budget_ms=budget)


class TestPhaseGate:
    def test_phase_order_constant(self):
        assert PHASE_ORDER[0] is E2EPhase.S1_POWER_ON
        assert PHASE_ORDER[-1] is E2EPhase.DONE

    def test_skip_phase_rejected(self):
        h = _harness()
        with pytest.raises(PermissionError):
            h.run_phase(E2EPhase.S4_MOTION)   # 跳到 S4

    def test_out_of_order_rejected(self):
        h = _harness()
        h.run_phase(E2EPhase.S1_POWER_ON)
        with pytest.raises(PermissionError):
            h.run_phase(E2EPhase.S1_POWER_ON)   # 回退

    def test_sequential_advance(self):
        h = _harness()
        for phase in PHASE_ORDER[:-1]:
            rec = h.run_phase(phase)
            assert rec.passed, f"{phase.value} 未过: {rec.detail}"
        assert h.current_phase is E2EPhase.DONE


class TestRunAll:
    def test_full_chain_passes(self):
        h = _harness()
        report = h.run_all()
        assert report["all_pass"] is True
        assert report["reached"] == "done"
        assert report["n_phases_done"] == 5

    def test_each_phase_recorded(self):
        h = _harness()
        report = h.run_all()
        phases = [p["phase"] for p in report["phases"]]
        assert phases == [p.value for p in PHASE_ORDER[:-1]]
        assert all(p["detail"] for p in report["phases"])

    def test_without_brain_degrades_gracefully(self):
        h = _harness(with_brain=False)
        report = h.run_all()
        assert report["all_pass"] is True


class TestLatencyBudget:
    def test_latency_within_budget(self):
        h = _harness(latency_ms=12.0, budget=50.0)
        h.run_phase(E2EPhase.S1_POWER_ON)
        h.run_phase(E2EPhase.S2_PERCEPTION)
        h.run_phase(E2EPhase.S3_DECISION)
        rec = h.run_phase(E2EPhase.S4_MOTION)
        assert rec.passed
        assert rec.measurements["latency_worst_ms"] <= 50.0

    def test_over_budget_fails(self):
        """反证: 时延超预算必须 FAIL (判据不是恒过)。"""
        h = _harness(latency_ms=80.0, budget=50.0)
        h.run_phase(E2EPhase.S1_POWER_ON)
        h.run_phase(E2EPhase.S2_PERCEPTION)
        h.run_phase(E2EPhase.S3_DECISION)
        rec = h.run_phase(E2EPhase.S4_MOTION)
        assert not rec.passed, "80ms 时延不应过 50ms 预算"
        assert "超预算" in rec.detail

    def test_failed_phase_blocks_advance(self):
        h = _harness(latency_ms=80.0, budget=50.0)
        h.run_phase(E2EPhase.S1_POWER_ON)
        h.run_phase(E2EPhase.S2_PERCEPTION)
        h.run_phase(E2EPhase.S3_DECISION)
        h.run_phase(E2EPhase.S4_MOTION)   # FAIL
        assert h.current_phase is E2EPhase.S4_MOTION, "失败阶段不得推进"


class TestEstop:
    def test_estop_blocks_subsequent(self):
        h = _harness()
        h.run_phase(E2EPhase.S1_POWER_ON)
        h.trigger_estop()
        rec = h.run_phase(E2EPhase.S2_PERCEPTION)
        assert not rec.passed
        assert "ESTOP" in rec.detail

    def test_estop_blocks_motion_on_channel(self):
        ch = MockHardwareChannel()
        ch.estop()
        with pytest.raises(RuntimeError):
            ch.send_action(np.zeros(6))


class TestPerceptionChain:
    def test_pose_update_reaches_hub(self):
        h = _harness()
        h.run_phase(E2EPhase.S1_POWER_ON)
        rec = h.run_phase(E2EPhase.S2_PERCEPTION)
        assert rec.passed
        assert h.brain.hub.latest_pose("robot") is not None

    def test_decision_phase_shape(self):
        h = _harness()
        h.run_phase(E2EPhase.S1_POWER_ON)
        h.run_phase(E2EPhase.S2_PERCEPTION)
        rec = h.run_phase(E2EPhase.S3_DECISION)
        assert rec.passed
        assert "决策路径" in rec.detail
        assert "decision_path_v2a" in rec.measurements

    def test_task_events_recorded(self):
        h = _harness()
        for phase in (E2EPhase.S1_POWER_ON, E2EPhase.S2_PERCEPTION,
                      E2EPhase.S3_DECISION, E2EPhase.S4_MOTION):
            h.run_phase(phase)
        rec = h.run_phase(E2EPhase.S5_TASK)
        assert rec.passed
        assert rec.measurements["events_added"] == 1.0


class TestMockChannel:
    def test_telemetry_keys(self):
        ch = MockHardwareChannel(n_joints=4)
        t = ch.telemetry()
        assert len(t) == 4

    def test_latency_positive(self):
        ch = MockHardwareChannel()
        assert ch.send_action(np.zeros(6)) >= 0.0


class TestDemo:
    def test_demo_assertions(self):
        r = e2e_demo()
        assert r["all_pass"] is True
        assert r["reached"] == "done"
        assert r["n_phases_done"] == 5
        assert r["s4_latency_worst_ms"] <= 50.0
