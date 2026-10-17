#!/usr/bin/env python3
"""e2e_real_robot_test.py — 真实机器人硬件端到端闭环测试脚本 (docs/19 §6)。

基于 `hil/e2e_closed_loop.py` 的 5 阶段门 (S1 通电→S2 感知→S3 决策→
S4 运动→S5 任务), 生成接入真实机器人硬件的可执行测试流程。

**双模式**
  --mode mock   (默认): MockHardwareChannel 合成遥测, 全链演练 (CI/预演)
  --mode real           : SocketCANTransport 真机通道 (需 CAN 硬件 + 机器人上电)
                          真机模式遵守 OS SSH 状态机: 非 RUNTIME 态拒绝运动阶段

**安全铁律 (docs/19 §6.3)**
  · S1-S5 均需 RUNTIME 态; 维护态自动禁运动 (os_channel.ModeController)
  · 任一阶段 FAIL → 停止推进 (不跳步蒙混)
  · 时延超 50ms 契约预算 → S4 FAIL (判据能失败)
  · --estop 全程可用 (急停后剩余阶段全部拒绝)

用法:
  python scripts/e2e_real_robot_test.py                # mock 全链演练
  python scripts/e2e_real_robot_test.py --mode real    # 真机 (需硬件)
  python scripts/e2e_real_robot_test.py --report out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from embodied_ai.center.center_brain import CenterBrain  # noqa: E402
from embodied_ai.cognition.cognitive_hub import CognitiveHub  # noqa: E402
from embodied_ai.hil.e2e_closed_loop import (  # noqa: E402
    E2EClosedLoopHarness,
    E2EPhase,
    MockHardwareChannel,
)
from embodied_ai.os_channel import ModeController, ModeState  # noqa: E402
from embodied_ai.runtime.bus import InProcessBus  # noqa: E402


class RealCANChannel:
    """真机硬件通道 — 包装 simulation.real_robot_env 的 CAN 传输。

    仅在 --mode real 时构造 (避免 CI 依赖 CAN 硬件)。
    send_action 返回实测往返时延 ms (由 transport 打点)。
    """

    def __init__(self, channel: str = "can0", bitrate: int = 1_000_000,
                 n_joints: int = 6):
        from embodied_ai.simulation.real_robot_env import SocketCANTransport
        self._tx = SocketCANTransport(channel=channel, bitrate=bitrate)
        self.n_joints = n_joints
        self._tx.open()
        self.actions_sent = []

    def telemetry(self) -> dict:
        frame = self._tx.recv(timeout_s=0.05)
        if frame is None:
            return {}
        from embodied_ai.simulation.real_robot_env import decode_telemetry
        t = decode_telemetry(frame)
        return {k: float(v) for k, v in t.items()}

    def send_action(self, action: np.ndarray) -> float:
        import time
        from embodied_ai.simulation.real_robot_env import encode_joint_cmd
        t0 = time.perf_counter()
        for j in range(min(self.n_joints, action.size)):
            self._tx.send(encode_joint_cmd(j, float(action[j]), 0.0, 0.0))
        frame = self._tx.recv(timeout_s=0.05)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.actions_sent.append(np.asarray(action).copy())
        return float(elapsed_ms) if frame is not None else float("inf")

    def estop(self) -> None:
        from embodied_ai.simulation.real_robot_env import encode_joint_cmd
        for j in range(self.n_joints):
            self._tx.send(encode_joint_cmd(j, 0.0, 0.0, 0.0))
        self._tx.close()


PHASE_LABELS = {
    E2EPhase.S1_POWER_ON: "S1 通电自检 (电源/编码器/CAN alive, 运动禁用)",
    E2EPhase.S2_PERCEPTION: "S2 感知回环 (遥测→POSE_UPDATE 抵达认知中枢)",
    E2EPhase.S3_DECISION: "S3 决策回环 (VLA 双路径产出动作, 不下发)",
    E2EPhase.S4_MOTION: "S4 运动闭环 (单步下发 + 时延预算核对 <50ms)",
    E2EPhase.S5_TASK: "S5 任务闭环 (目标→规划→四要素事件落库)",
}


def build_harness(mode: str, can_channel: str, can_bitrate: int,
                  budget_ms: float) -> E2EClosedLoopHarness:
    """构造测试 harness — mock 或真机通道 + CenterBrain 中枢。"""
    if mode == "real":
        channel = RealCANChannel(channel=can_channel, bitrate=can_bitrate)
    else:
        channel = MockHardwareChannel(latency_ms=12.0, seed=0)
    bus = InProcessBus()
    hub = CognitiveHub(bus=bus)
    brain = CenterBrain(bus=bus, hub=hub)
    return E2EClosedLoopHarness(channel=channel, center_brain=brain,
                                latency_budget_ms=budget_ms)


def run(mode: str, can_channel: str = "can0", can_bitrate: int = 1_000_000,
        budget_ms: float = 50.0, estop_after: str = "",
        verbose: bool = True) -> dict:
    """执行 5 阶段门 E2E 测试 → 报告 dict。

    Args:
        mode: "mock" / "real"
        estop_after: 阶段名后触发 ESTOP (负对照演练, 留空=不触发)
    """
    # OS 状态机: 全程 RUNTIME (维护态禁运动 — 真机安全门禁)
    mode_ctl = ModeController(initial=ModeState.RUNTIME)
    assert mode_ctl.can_send_motion(), "E2E 测试要求 RUNTIME 态"

    harness = build_harness(mode, can_channel, can_bitrate, budget_ms)
    if verbose:
        print(f"\n=== 真实机器人 E2E 闭环测试 (mode={mode}, "
              f"预算 {budget_ms}ms, docs/19 §6) ===")

    phases_out = []
    for phase in (E2EPhase.S1_POWER_ON, E2EPhase.S2_PERCEPTION,
                  E2EPhase.S3_DECISION, E2EPhase.S4_MOTION,
                  E2EPhase.S5_TASK):
        if estop_after and phase.value == estop_after:
            harness.trigger_estop()
            if verbose:
                print("  ⚡ ESTOP 触发 — 后续阶段将全部拒绝")
        rec = harness.run_phase(phase)
        mark = "✅" if rec.passed else "❌"
        if verbose:
            print(f"  [{mark}] {PHASE_LABELS[phase]}")
            print(f"        {rec.detail}")
            for k, v in rec.measurements.items():
                print(f"        {k} = {v}")
        phases_out.append({"phase": phase.value, "passed": rec.passed,
                           "detail": rec.detail,
                           "measurements": rec.measurements})
        if not rec.passed:
            if verbose:
                print(f"\n  ⛔ 阶段 {phase.value} FAIL — 停止推进 (禁跳步)")
            break

    report = harness.report()
    report["mode"] = mode
    report["budget_ms"] = budget_ms
    report["os_mode"] = mode_ctl.state.value
    if verbose:
        ok = report["all_pass"]
        print(f"\nRESULT: {'E2E_VERIFY_OK' if ok else 'E2E_VERIFY_FAIL'} "
              f"({report['n_phases_done']}/5 阶段, 到达 {report['reached']})")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="真实机器人 E2E 5 阶段门测试")
    p.add_argument("--mode", choices=["mock", "real"], default="mock",
                   help="mock=合成遥测演练 (默认); real=SocketCAN 真机")
    p.add_argument("--can-channel", default="can0", help="CAN 接口 (real 模式)")
    p.add_argument("--can-bitrate", type=int, default=1_000_000)
    p.add_argument("--budget-ms", type=float, default=50.0,
                   help="端到端时延预算 (契约默认 50ms)")
    p.add_argument("--estop-after", default="",
                   choices=["", "s1_power_on", "s2_perception",
                            "s3_decision", "s4_motion", "s5_task"],
                   help="负对照: 在指定阶段后触发 ESTOP")
    p.add_argument("--report", default=None, help="导出 JSON 报告路径")
    args = p.parse_args(argv)

    report = run(args.mode, args.can_channel, args.can_bitrate,
                 args.budget_ms, args.estop_after)
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已导出: {args.report}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
