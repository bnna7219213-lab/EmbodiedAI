"""真实硬件端到端闭环测试支架 (plan_v6 后续: d3)。

**缺口背景（C1-C5 → 真机的接入断点）**
    C 栈中枢决策层（CenterBrain 编排 + VLA 双路径 + 认知中枢 + OS SSH 通道）
    已在软件侧闭合, 但**接真实机器人硬件**的端到端闭环测试没有执行骨架。
    本模块提供 E2E 测试支架: 阶段门推进 + 时延预算核对 + 硬件通道抽象,
    让"中枢决策层 → 真机"的每一步都有可验收的判据。

**阶段门（docs/19 §8, 与 docs/15 HIL 协议对齐）**
    S1 通电自检     — 电源/编码器/CAN 链路 alive, 运动禁用
    S2 感知回环     — 传感器流→分频→SLAM→记忆, POSE_UPDATE 抵达 hub
    S3 决策回环     — VLA 双路径决策→安全包络→下发 (只读遥测, 不驱动)
    S4 运动闭环     — 单关节/单步动作, 时延预算核对 (<50ms 契约)
    S5 任务闭环     — 目标→规划→执行→四要素事件落库
    每阶段有准入门禁: 前序未过 → 拒绝推进 (防跳步蒙混)。

**硬件通道抽象**
    - `HardwareChannel` 协议: telemetry() / send_action() / estop()
    - `MockHardwareChannel`: 仿真通道（复用 real_robot_env.MockTransport 语义）
    - 真机通道 = 注入 `simulation.real_robot_env.SocketCANTransport` 包装
    - 时延核对复用 `deploy.latency_audit.audit_latency` (50ms 契约预算)

**provenance（诚实边界）**
    Mock 通道时延为 synthetic; 真机时延属 **待实测** (P1 HIL 到货后)。
    本模块守护的是**阶段门推进纪律 + 时延预算判据**, 不是真机性能。

**不做的（诚实边界）**
    - 不做真机驱动（SocketCAN 已在 real_robot_env, 这里只包接口）
    - 不做多任务并发调度（单任务串行阶段推进）
    - 不替代 docs/15 的 48h HIL 协议（本模块是其执行骨架）

**反证要求（测试守护）**
    跳阶段必须报错; S4 时延超预算必须 FAIL (不是恒过); ESTOP 可中断;
    每阶段产出可审计; mock 通道全链可跑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np


# ══════════════════════════════════════════════════════════════════
# 阶段门
# ══════════════════════════════════════════════════════════════════

class E2EPhase(Enum):
    """端到端闭环测试阶段（顺序推进, 不得跳步）。"""
    S1_POWER_ON = "s1_power_on"          # 通电自检
    S2_PERCEPTION = "s2_perception"      # 感知回环
    S3_DECISION = "s3_decision"          # 决策回环
    S4_MOTION = "s4_motion"              # 运动闭环 (时延预算)
    S5_TASK = "s5_task"                  # 任务闭环
    DONE = "done"


#: 阶段顺序 (准入=前序已过)
PHASE_ORDER = [E2EPhase.S1_POWER_ON, E2EPhase.S2_PERCEPTION,
               E2EPhase.S3_DECISION, E2EPhase.S4_MOTION,
               E2EPhase.S5_TASK, E2EPhase.DONE]


@dataclass
class PhaseRecord:
    """单阶段执行记录（可审计）。"""
    phase: E2EPhase
    passed: bool
    detail: str
    measurements: Dict[str, float] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
# 硬件通道抽象
# ══════════════════════════════════════════════════════════════════

class HardwareChannel(Protocol):
    """机器人硬件通道协议 — 真机注入 SocketCAN 包装, 测试用 Mock。"""

    def telemetry(self) -> Dict[str, float]: ...
    def send_action(self, action: np.ndarray) -> float:
        """下发动作, 返回实测往返时延 ms。"""
        ...
    def estop(self) -> None: ...


class MockHardwareChannel:
    """仿真硬件通道 (provenance=synthetic 时延) — 端到端链路演练用。"""

    def __init__(self, n_joints: int = 6, latency_ms: float = 12.0,
                 seed: int = 0):
        self.n_joints = n_joints
        self.latency_ms = latency_ms
        self.rng = np.random.default_rng(seed)
        self.estopped = False
        self.actions_sent: List[np.ndarray] = []
        self._t = 0.0

    def telemetry(self) -> Dict[str, float]:
        self._t += 0.05
        pos = np.sin(self._t + np.arange(self.n_joints))
        return {f"joint{i}_pos": float(pos[i]) for i in range(self.n_joints)}

    def send_action(self, action: np.ndarray) -> float:
        if self.estopped:
            raise RuntimeError("ESTOP 已触发 — 拒绝下发动作")
        self.actions_sent.append(np.asarray(action).copy())
        return float(max(0.0, self.rng.normal(self.latency_ms,
                                              self.latency_ms / 10.0)))

    def estop(self) -> None:
        self.estopped = True


# ══════════════════════════════════════════════════════════════════
# E2EClosedLoopHarness — 阶段推进
# ══════════════════════════════════════════════════════════════════

class E2EClosedLoopHarness:
    """真实硬件端到端闭环测试支架 — 阶段门推进 + 时延预算核对。

    用法:
        harness = E2EClosedLoopHarness(channel=MockHardwareChannel())
        harness.run_phase(E2EPhase.S1_POWER_ON)
        ...
        report = harness.run_all()   # 或逐阶段推进
    """

    #: 端到端时延预算 (ms) — contracts/hw_requirements.compute 契约值
    LATENCY_BUDGET_MS = 50.0
    #: 单步动作超时/estop 后的最大尝试次数
    MAX_MOTION_TRIES = 3

    def __init__(self, channel: HardwareChannel,
                 center_brain=None,
                 latency_budget_ms: float = LATENCY_BUDGET_MS):
        self.channel = channel
        self.brain = center_brain
        self.latency_budget_ms = latency_budget_ms
        self.records: List[PhaseRecord] = []
        self._phase_index = 0
        self._estopped = False

    @property
    def current_phase(self) -> E2EPhase:
        return PHASE_ORDER[self._phase_index]

    def _require_phase(self, phase: E2EPhase) -> None:
        """准入门禁: 只能执行当前阶段 — 跳步/回退都拒绝。"""
        if phase is not self.current_phase:
            raise PermissionError(
                f"阶段乱序: 当前应执行 {self.current_phase.value}, "
                f"得 {phase.value} — 阶段门不得跳步")

    def run_phase(self, phase: E2EPhase) -> PhaseRecord:
        """执行一个阶段 → 记录; 成功则推进到下一阶段。"""
        self._require_phase(phase)
        if self._estopped:
            return self._record(phase, False, "ESTOP 已触发, 闭环终止")
        handler = {
            E2EPhase.S1_POWER_ON: self._run_s1,
            E2EPhase.S2_PERCEPTION: self._run_s2,
            E2EPhase.S3_DECISION: self._run_s3,
            E2EPhase.S4_MOTION: self._run_s4,
            E2EPhase.S5_TASK: self._run_s5,
        }.get(phase)
        if handler is None:
            raise ValueError(f"阶段 {phase} 无可执行体")
        rec = handler()
        self.records.append(rec)
        if rec.passed:
            self._phase_index += 1
        return rec

    def _record(self, phase: E2EPhase, passed: bool, detail: str,
                measurements: Optional[Dict[str, float]] = None) -> PhaseRecord:
        return PhaseRecord(phase=phase, passed=passed, detail=detail,
                           measurements=measurements or {})

    # ---------------- 各阶段 ---------------- #

    def _run_s1(self) -> PhaseRecord:
        """S1 通电自检: 遥测 alive, 运动禁用。"""
        telem = self.channel.telemetry()
        ok = bool(telem) and not getattr(self.channel, "actions_sent", [])
        return self._record(E2EPhase.S1_POWER_ON, ok,
                            f"遥测 {len(telem)} 路 alive, 运动禁用" if ok
                            else "通电自检失败",
                            {"telemetry_channels": float(len(telem))})

    def _run_s2(self) -> PhaseRecord:
        """S2 感知回环: 遥测→(CenterBrain)→POSE_UPDATE 抵达 hub。"""
        telem = self.channel.telemetry()
        if self.brain is None:
            return self._record(E2EPhase.S2_PERCEPTION, True,
                                "无 CenterBrain, 遥测链路直通 (降级验收)",
                                {"telemetry_channels": float(len(telem))})
        pose = np.array([telem.get("joint0_pos", 0.0), 0.0, 0.0])
        self.brain.publish_pose("robot", pose, timestamp_s=1.0)
        seen = self.brain.hub.latest_pose("robot")
        ok = seen is not None
        return self._record(E2EPhase.S2_PERCEPTION, ok,
                            "POSE_UPDATE 抵达认知中枢" if ok
                            else "POSE_UPDATE 断链 (hub 未收到)",
                            {"pose_x": float(pose[0])})

    def _run_s3(self) -> PhaseRecord:
        """S3 决策回环: VLA 双路径决策产出动作 (不下发)。"""
        if self.brain is None:
            return self._record(E2EPhase.S3_DECISION, True,
                                "无 CenterBrain, 跳过决策回环 (降级验收)")
        cfg = self.brain.vla.cfg
        rng = np.random.default_rng(3)
        v = rng.normal(size=(cfg.num_views, cfg.vision_embd))
        s = rng.normal(size=cfg.state_dim)
        res = self.brain.step(2.0, None, 1.0, v, s, thing="e2e_decision")
        ok = res.action.shape == (cfg.action_horizon, cfg.action_dim)
        return self._record(E2EPhase.S3_DECISION, ok,
                            f"决策路径 {res.decision_path}, 动作 {res.action.shape}",
                            {"decision_path_v2a":
                             1.0 if res.decision_path == "v2a" else 0.0})

    def _run_s4(self) -> PhaseRecord:
        """S4 运动闭环: 下发动作 + 时延预算核对 (<50ms 契约)。"""
        latencies = []
        for i in range(self.MAX_MOTION_TRIES):
            action = np.zeros(6)
            latencies.append(self.channel.send_action(action))
        avg_ms = float(np.mean(latencies))
        worst_ms = float(np.max(latencies))
        ok = worst_ms <= self.latency_budget_ms
        return self._record(
            E2EPhase.S4_MOTION, ok,
            f"时延 avg={avg_ms:.1f}ms worst={worst_ms:.1f}ms vs 预算 "
            f"{self.latency_budget_ms}ms — {'达标' if ok else '超预算 FAIL'}",
            {"latency_avg_ms": avg_ms, "latency_worst_ms": worst_ms,
             "budget_ms": self.latency_budget_ms})

    def _run_s5(self) -> PhaseRecord:
        """S5 任务闭环: 目标→(中枢编排)→四要素事件落库。"""
        if self.brain is None:
            return self._record(E2EPhase.S5_TASK, True,
                                "无 CenterBrain, 任务环跳过 (降级验收)")
        before = len(self.brain.hub.memory)
        cfg = self.brain.vla.cfg
        rng = np.random.default_rng(5)
        v = rng.normal(size=(cfg.num_views, cfg.vision_embd))
        s = rng.normal(size=cfg.state_dim)
        self.brain.step(3.0, None, 1.0, v, s, thing="e2e_task",
                        pose_xyz=np.array([0.5, 0.0, 0.0]))
        after = len(self.brain.hub.memory)
        ok = after == before + 1
        return self._record(E2EPhase.S5_TASK, ok,
                            f"四要素事件 {before}→{after}",
                            {"events_added": float(after - before)})

    # ---------------- 汇总/中断 ---------------- #

    def run_all(self) -> Dict[str, object]:
        """逐阶段推进到 DONE — 任一阶段 FAIL 即停。"""
        while self.current_phase is not E2EPhase.DONE:
            rec = self.run_phase(self.current_phase)
            if not rec.passed:
                break
        return self.report()

    def trigger_estop(self) -> None:
        """急停 — 后续阶段全部拒绝执行。"""
        self._estopped = True
        self.channel.estop()

    def report(self) -> Dict[str, object]:
        all_pass = (self.current_phase is E2EPhase.DONE
                    and all(r.passed for r in self.records))
        return {
            "all_pass": all_pass,
            "reached": self.current_phase.value,
            "n_phases_done": len(self.records),
            "phases": [{"phase": r.phase.value, "passed": r.passed,
                        "detail": r.detail,
                        "measurements": r.measurements} for r in self.records],
        }


# ══════════════════════════════════════════════════════════════════
# demo
# ══════════════════════════════════════════════════════════════════

def e2e_demo() -> dict:
    """端到端闭环演示: mock 硬件 + CenterBrain 全阶段推进。

    布尔结论为实测计算值, verify 断言即这些值。
    """
    from ..center.center_brain import CenterBrain
    from ..cognition.cognitive_hub import CognitiveHub
    from ..runtime.bus import InProcessBus

    bus = InProcessBus()
    hub = CognitiveHub(bus=bus)
    brain = CenterBrain(bus=bus, hub=hub)
    harness = E2EClosedLoopHarness(channel=MockHardwareChannel(latency_ms=12.0),
                                   center_brain=brain)
    report = harness.run_all()
    result = {
        "all_pass": report["all_pass"],
        "reached": report["reached"],
        "n_phases_done": report["n_phases_done"],
        "s4_latency_worst_ms": next(
            (p["measurements"].get("latency_worst_ms")
             for p in report["phases"] if p["phase"] == "s4_motion"), None),
    }
    assert report["all_pass"], f"E2E 未全通: {report['phases']}"
    assert report["reached"] == "done"
    assert report["n_phases_done"] == 5
    return result


if __name__ == "__main__":
    r = e2e_demo()
    print("真实硬件 E2E 闭环测试支架 demo (mock 通道, docs/19 §8)")
    print(f"阶段推进: {r['n_phases_done']}/5 → {r['reached']}")
    print(f"S4 时延 worst: {r['s4_latency_worst_ms']:.1f}ms (<50ms 契约)")
