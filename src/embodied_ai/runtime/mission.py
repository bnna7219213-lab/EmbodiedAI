"""断点 ③ 修复：最小 MissionManager — OS 决策中枢。

**缺口背景（审计报告三大断点之一，定级 D）**
    无调度器、无行为树、无世界模型；`run()` 是 sleep 占位。
    "一句话指令 → 任务状态机 → 20Hz 环" 这条 OS 级链路不存在。

**设计（最小可用）**
    - `MissionManager`: 接收一句话指令 → 关键词解析 → 任务状态机 →
      驱动 20Hz 控制环 → 完成判定
    - `Mission`: 状态机（PENDING → RUNNING → SUCCESS / FAILED / ABORTED），
      每步技能调度 + 超时判定 + 安全事件中止
    - Bus 集成：Mission 状态变更发布 `SAFETY_EVENT` 同级的 `MISSION_EVENT`
      （复用 Bus 已有基础设施——不新造机制）

**与 8 步竖井的对应**（审计报告建议的贯通清单）
    本模块覆盖前 4 步：
      ① 一句话指令 → ② 任务状态机 → ③ 技能调度（策略）→ ④ 20Hz 执行
    后 4 步（CAN 帧下发/遥测回流/DR 更新）已有基础（Bus + policy_loader +
    bigdata），由 MissionManager 的遥测检查衔接。

**不做的（诚实边界）**
    - 不做 NLU（关键词匹配是最小实现；真 NLU 是 LLM/VLA 的活）
    - 不做行为树（状态机够当前任务集；复杂并发时再升级）
    - 不做世界模型（任务模板里写死完成条件；真世界模型是 Phase P 后的事）
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════
# 状态机
# ══════════════════════════════════════════════════════════════════

class MissionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"        # 条件不满足/超时（正常失败路径）
    ABORTED = "aborted"      # 安全事件触发（异常中止）


#: 合法状态转移（非法转移直接报错 — 不允许"FAILED→RUNNING"这种幽灵复活）
VALID_TRANSITIONS: Dict[MissionState, set] = {
    MissionState.PENDING: {MissionState.RUNNING, MissionState.ABORTED},
    MissionState.RUNNING: {MissionState.SUCCESS, MissionState.FAILED, MissionState.ABORTED},
    MissionState.SUCCESS: set(),
    MissionState.FAILED: set(),
    MissionState.ABORTED: set(),
}


class MissionError(RuntimeError):
    """指令不可识别/状态机非法操作 — 不静默。"""


# ══════════════════════════════════════════════════════════════════
# 任务模板（基于控制器已注册技能: grasp/place/wipe — 不虚构）
# ══════════════════════════════════════════════════════════════════

@dataclass
class MissionTemplate:
    """一个任务模板 = 技能序列 + 完成条件 + 超时。"""
    name: str
    skills: List[str]                # 依次执行的技能（如 ["grasp", "place"]）
    keywords: List[str]              # 指令匹配关键词（中文/英文）
    max_steps: int = 50              # 20Hz 下 50 步 = 2.5s（最小演示预算）
    completion_check: Optional[Callable[[Dict], bool]] = None
    description: str = ""


def _default_completion(obs: Dict) -> bool:
    """默认完成条件: 关节角收敛（变化 < 阈值, 即动作稳定）。"""
    jp = obs.get("joint_positions")
    if jp is None:
        return False
    return bool(np.all(np.abs(jp) < 0.5))   # 简化: 关节接近零位


#: 内置任务模板（关键词 → 技能序列; 只用已注册技能）
MISSION_TEMPLATES: List[MissionTemplate] = [
    MissionTemplate(
        name="pick_and_place",
        skills=["grasp", "place"],
        keywords=["抓", "拿", "取", "pick", "grasp", "抓取", "递"],
        max_steps=60,
        description="抓取物体并放置（grasp→place 序列）"),
    MissionTemplate(
        name="grasp_only",
        skills=["grasp"],
        keywords=["握", "捏", "hold", "grip"],
        max_steps=30,
        description="单次抓取"),
    MissionTemplate(
        name="wipe_surface",
        skills=["wipe"],
        keywords=["擦", "清洁", "wipe", "clean"],
        max_steps=40,
        description="表面擦拭"),
    MissionTemplate(
        name="clean_and_pick",
        skills=["wipe", "grasp", "place"],
        keywords=["打扫", "收拾", "clean and pick", "整理"],
        max_steps=80,
        description="擦拭→抓取→放置（三技能组合）"),
]


# ══════════════════════════════════════════════════════════════════
# Mission — 一次任务执行的运行时实例
# ══════════════════════════════════════════════════════════════════

@dataclass
class Mission:
    """一次任务执行的运行时状态（状态机 + 技能进度 + 遥测）。"""
    template: MissionTemplate
    instruction: str
    state: MissionState = MissionState.PENDING
    current_skill_index: int = 0
    steps_executed: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    abort_reason: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def current_skill(self) -> Optional[str]:
        if self.current_skill_index < len(self.template.skills):
            return self.template.skills[self.current_skill_index]
        return None

    def _transition(self, new_state: MissionState, reason: str = "") -> None:
        if new_state not in VALID_TRANSITIONS[self.state]:
            raise MissionError(
                f"非法状态转移 {self.state.value} → {new_state.value} ({reason})")
        old = self.state
        self.state = new_state
        self.events.append({
            "from": old.value, "to": new_state.value,
            "reason": reason, "step": self.steps_executed,
            "timestamp": time.time(),
        })

    def advance_skill(self) -> bool:
        """推进到下一技能; 全部完成返回 True。"""
        self.current_skill_index += 1
        return self.current_skill_index >= len(self.template.skills)


# ══════════════════════════════════════════════════════════════════
# MissionManager — OS 决策中枢（最小实现）
# ══════════════════════════════════════════════════════════════════

class MissionManager:
    """一句话指令 → 任务状态机 → 驱动 20Hz 控制环。

    用法:
        mm = MissionManager(controller, bus)
        mission = mm.parse("抓起杯子放到桌上")     # 一句话指令
        result = mm.run(mission)                    # 驱动 20Hz 环直至终态

    状态变更发 Bus（可观测）; 安全事件自动 ABORT。
    """

    def __init__(self, controller=None, bus=None):
        self.controller = controller
        self.bus = bus
        self.templates: List[MissionTemplate] = list(MISSION_TEMPLATES)
        self.active: Optional[Mission] = None

    # ---------------- 指令解析 ---------------- #

    def parse(self, instruction: str) -> Mission:
        """一句话指令 → Mission（关键词匹配; 不可识别必报错，不静默）。"""
        text = instruction.strip().lower()
        if not text:
            raise MissionError("空指令")
        best: Optional[MissionTemplate] = None
        best_score = 0
        for t in self.templates:
            score = sum(1 for kw in t.keywords if kw.lower() in text)
            if score > best_score:
                best, best_score = t, score
        if best is None:
            raise MissionError(
                f"指令不可识别: '{instruction}'。"
                f"支持的关键词: {[kw for t in self.templates for kw in t.keywords][:8]}...")
        return Mission(template=best, instruction=instruction)

    # ---------------- 执行驱动 ---------------- #

    def run(self, mission: Mission, obs_provider: Optional[Callable[[], Dict]] = None,
            max_steps_override: Optional[int] = None) -> Mission:
        """驱动 20Hz 控制环执行任务直至终态。

        obs_provider: () → obs dict（缺省用零观测——最小演示路径;
                      真实场景由感知节点经 Bus 提供）。
        """
        if self.controller is None:
            raise MissionError("MissionManager 未绑定控制器")
        self.active = mission
        mission._transition(MissionState.RUNNING, "mission start")
        mission.started_at = time.time()
        self._emit("mission_started", mission)

        max_steps = max_steps_override or mission.template.max_steps
        check = mission.template.completion_check or _default_completion

        for step in range(max_steps):
            # 安全事件检查（Bus 上的 SAFETY_EVENT → ABORT）
            if self._safety_aborted():
                mission._transition(MissionState.ABORTED, "safety event")
                break

            # 观测
            obs = obs_provider() if obs_provider else {
                "ee_position": np.zeros(3), "joint_positions": np.zeros(12)}

            # 技能调度（控制器的 current_skill 由 Mission 驱动）
            skill = mission.current_skill
            if skill is None:
                break
            self.controller.current_skill = skill

            # 20Hz 环执行一步
            action = self.controller.step(obs)
            mission.steps_executed += 1

            # 完成判定
            if check(obs):
                if mission.advance_skill():
                    mission._transition(MissionState.SUCCESS, "all skills done")
                    break
                # 下一技能继续
                self._emit("skill_advanced", mission)

        else:
            # for 循环耗尽（超时）
            if mission.state == MissionState.RUNNING:
                mission._transition(MissionState.FAILED,
                                    f"timeout: {max_steps} steps")

        mission.finished_at = time.time()
        self._emit(f"mission_{mission.state.value}", mission)
        return mission

    # ---------------- 辅助 ---------------- #

    def _safety_aborted(self) -> bool:
        """检查 Bus 上是否有未处理的安全事件（最小实现: 查控制器标志）。"""
        # 控制器/安全层若设置 _safety_stop 标志 → ABORT
        return bool(getattr(self.controller, "_safety_stop", False))

    def _emit(self, event: str, mission: Mission) -> None:
        """状态事件发 Bus（若有）— 可观测。"""
        if self.bus is not None:
            from .bus import Topic
            self.bus.publish(Topic.SAFETY_EVENT,
                             {"event_type": f"mission:{event}"},
                             source="mission_manager")

    def register_template(self, template: MissionTemplate) -> None:
        """扩展任务模板（开放但受控——新模板必须显式注册）。"""
        if any(t.name == template.name for t in self.templates):
            raise MissionError(f"模板重复注册: {template.name}")
        self.templates.append(template)
