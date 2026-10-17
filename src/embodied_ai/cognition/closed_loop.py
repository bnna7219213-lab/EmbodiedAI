"""VLA 双路径 × 认知中枢 — 端到端闭环 (plan_v5 Sprint M: M6)。

**缺口背景（接线缺口）**
    M3 的 VLA 双路径（v+l→action / v→action）与 M4 的认知中枢四要素
    （记录/分析/计算/规划/记忆）此前是**两个孤岛**：VLA 输出动作后无人记录，
    认知中枢规划出的前置事件无人执行。本模块把二者焊成闭环——
    "感知 → VLA 决策 → 动作 → 事件记录(四要素) → 规划 → 下一步目标"。

**闭环数据流**
    目标 goal_thing
      → 认知中枢 plan() 给出尚缺前置事件序列
      → 逐步取出 step_thing
      → VLA 决策: 有语言指令走 v+l→action, 无语言走 v→action (纯视觉)
      → 动作执行 (仿真/真机接口), 位姿来自 SLAM (PoseStore / Bus POSE_UPDATE)
      → 认知中枢 record_event (四要素: 时间/地点/人物/事情) + Bus EVENT
      → 重新 plan(); 前置清空即目标达成

**双路径选择（用户口径: 等价双路径）**
    - 语言指令存在 → `infer_vla(vision, lang_ids, state)`（vision+language→action）
    - 语言指令缺省 → `infer_v2a(vision, state)`（vision-only→action, 纯视觉判断依据）
    选择由 `decide()` 统一封装, 路径名随事件记录 (`source="vla"`/`"v2a"`),
    事后可审计"这一步是靠语言还是纯视觉决定的"。

**provenance（诚实边界）**
    - 动作执行为**回调占位**（`act_fn` 由调用方注入: 仿真环境/真机接口）,
      本模块不含控制器。
    - VLA head 权重 provenance=synthetic（见 vla_v2a.py）; 闭环守护的是
      **链路完整性与四要素记录正确性**, 不是策略性能。
    - 位姿缺省时地点用零向量并记 `location_source="assumed_zero"`（诚实标注）。

**不做的（诚实边界）**
    - 不做控制器/力矩执行（control/ 已有, 经 act_fn 回调注入）
    - 不做学习式规划（plan() 是规则式, 见 cognitive_hub.py）
    - 不做并发多任务调度（单目标串行闭环; 并发属 mission.py）

**反证要求（测试守护）**
    无语言必须走 v2a 路径且事件 source 标注正确; 有语言走 vla 路径;
    闭环跑到目标达成后 plan() 返回空; 未知目标 KeyError 透传不吞;
    每步事件四要素齐全 (时间/地点/人物/事情) 且时间戳单调。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .cognitive_hub import CognitiveHub, EventRecord
from ..models.vla_config import VLAConfig
from ..models.vla_v2a import VLAPolicy


# ══════════════════════════════════════════════════════════════════
# StepResult — 单步决策结果
# ══════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    """闭环单步结果: 哪条路径 / 什么动作 / 记了什么事件。"""
    thing: str                          # 本步前置事件 (事情)
    path: str                           # "vla" (v+l) / "v2a" (v only)
    action: np.ndarray                  # [action_horizon, action_dim]
    event: EventRecord                  # 记入情景记忆的四要素事件


@dataclass
class LoopReport:
    """一次闭环运行报告。"""
    goal: str
    steps: List[StepResult] = field(default_factory=list)
    goal_reached: bool = False
    plan_remaining: List[str] = field(default_factory=list)

    @property
    def paths_used(self) -> Dict[str, int]:
        """本闭环用了哪些决策路径 (v2a / vla) 各多少步。"""
        out: Dict[str, int] = {}
        for s in self.steps:
            out[s.path] = out.get(s.path, 0) + 1
        return out


# ══════════════════════════════════════════════════════════════════
# CognitiveVLALoop — 端到端闭环
# ══════════════════════════════════════════════════════════════════

class CognitiveVLALoop:
    """VLA 双路径决策 × 认知中枢四要素 端到端闭环。

    接线关系:
      CognitiveHub (记录/规划/记忆) ←→ VLAPolicy (v+l / v 双路径)
      位姿来源: PoseStore 快照 / Bus POSE_UPDATE / 调用方直供
      动作执行: act_fn 回调注入 (仿真或真机)
    """

    def __init__(self, hub: CognitiveHub,
                 policy: Optional[VLAPolicy] = None,
                 cfg: Optional[VLAConfig] = None,
                 act_fn: Optional[Callable[[str, np.ndarray], None]] = None,
                 actor: str = "robot",
                 clock: Optional[Callable[[], float]] = None):
        self.hub = hub
        self.policy = policy if policy is not None else VLAPolicy(cfg)
        self.act_fn = act_fn
        self.actor = actor
        self._clock = clock
        self._t = 0.0                    # 内部时钟 (无外部 clock 时按步进 1s)
        self._last_t = -1.0              # 事件时间戳单调守护

    # ---------------- 单步决策 ---------------- #

    def decide(self, vision_feats: np.ndarray, state: np.ndarray,
               lang_ids: Optional[np.ndarray] = None) -> Tuple[str, np.ndarray]:
        """双路径选择: 有语言走 v+l→action, 无语言走 v→action。

        Returns:
            (path, action) — path ∈ {"vla", "v2a"}
        """
        if lang_ids is not None:
            return "vla", self.policy.infer_vla(vision_feats, lang_ids, state)
        return "v2a", self.policy.infer_v2a(vision_feats, state)

    def _now(self) -> float:
        if self._clock is not None:
            return float(self._clock())
        self._t += 1.0
        return self._t

    def _location(self, who: str = "robot") -> Tuple[np.ndarray, str]:
        """地点要素: 优先取 Bus/SLAM 位姿; 无则零向量 + 诚实标注。"""
        pose = self.hub.latest_pose(who)
        if pose is not None:
            return np.asarray(pose, dtype=np.float64).ravel(), "slam"
        return np.zeros(3), "assumed_zero"

    def step(self, thing: str, vision_feats: np.ndarray, state: np.ndarray,
             lang_ids: Optional[np.ndarray] = None,
             object_: Optional[str] = None) -> StepResult:
        """执行一个前置事件: VLA 决策 → act_fn → 记录四要素事件。"""
        path, action = self.decide(vision_feats, state, lang_ids)
        if self.act_fn is not None:
            self.act_fn(thing, action)
        t = self._now()
        if t <= self._last_t:
            raise ValueError(f"时间戳必须单调递增 (得 {t} <= {self._last_t})")
        self._last_t = t
        loc, loc_src = self._location()
        event = EventRecord(
            timestamp_s=t, location_xyz=loc, actor=self.actor,
            thing=thing, action=f"{path}_action", object_=object_,
            source=f"{path}|{loc_src}",
        )
        # 单一记录点: 有 Bus 时走消息路径 (hub 订阅 EVENT → 情景记忆), 无 Bus 时
        # 直接记录。两条都走会双重入账 (曾经 events=6 而非 3)。
        if self.hub.bus is not None:
            self.hub.bus.publish("event", {"record": event},
                                 source="cognitive_vla_loop")
        else:
            self.hub.record_event(event)
        return StepResult(thing=thing, path=path, action=action, event=event)

    # ---------------- 闭环运行 ---------------- #

    def run_until_goal(self, goal_thing: str,
                       vision_feats: np.ndarray,
                       state: np.ndarray,
                       lang_ids: Optional[np.ndarray] = None,
                       max_steps: int = 16) -> LoopReport:
        """闭环跑到目标达成: plan → 决策 → 记录 → 重 plan, 直到前置清空。

        Args:
            goal_thing: 目标事件 (须在 hub.PLAN_PREREQUISITES 中)
            lang_ids: 语言指令; None = 全程纯视觉 (v2a), 有值 = 全程 v+l (vla)
        Returns:
            LoopReport: 每步的路径/动作/事件 + 是否达成

        Raises:
            KeyError: 未知目标 —— 由 hub.plan() 透传 (不吞成"已达成")
            RuntimeError: max_steps 内未达成 (防失控循环)
        """
        report = LoopReport(goal=goal_thing)
        for _ in range(max_steps):
            remaining = self.hub.plan(goal_thing)   # KeyError 透传
            if not remaining:
                report.goal_reached = True
                report.plan_remaining = []
                return report
            thing = remaining[0]                   # 取最近前置事件执行
            report.steps.append(
                self.step(thing, vision_feats, state, lang_ids))
        report.plan_remaining = self.hub.plan(goal_thing)
        if not report.plan_remaining:
            report.goal_reached = True
            return report
        raise RuntimeError(
            f"闭环未在 max_steps={max_steps} 内达成 '{goal_thing}', "
            f"剩余前置: {report.plan_remaining}")


# ══════════════════════════════════════════════════════════════════
# demo — 双路径端到端闭环
# ══════════════════════════════════════════════════════════════════

def loop_demo() -> dict:
    """端到端闭环演示: 同一目标下 v+l 与 v-only 两条链路各自跑通。

    目标 climb_hold = [approach, grasp, load] 前置清空即达成;
    每步事件四要素齐全且进情景记忆。
    """
    from ..runtime.bus import InProcessBus

    cfg = VLAConfig(modality="vla")     # vla 模态下双路径都可用 (v2a 拒语言, 见 vla_v2a)
    rng = np.random.default_rng(3)
    vision = rng.normal(size=(cfg.num_views, cfg.vision_embd))
    state = rng.normal(size=cfg.state_dim)
    lang = rng.normal(size=cfg.LANG_TOKENS)

    reports = {}
    for label, lang_ids, t0 in (("v2a", None, 0.0), ("vla", lang, 100.0)):
        bus = InProcessBus()
        hub = CognitiveHub(bus=bus)
        # SLAM 位姿经 Bus POSE_UPDATE 送达 (地点要素来自 SLAM, 非零向量)
        bus.publish("pose_update",
                    {"pose": np.array([1.0, 2.0, 0.0]), "who": "robot",
                     "timestamp_s": t0},
                    source="slam")
        executed: List[str] = []
        clock = [t0]                    # 每链路独立时钟段, 步进 1s

        def _tick() -> float:
            clock[0] += 1.0
            return clock[0]

        loop = CognitiveVLALoop(
            hub, cfg=cfg,
            act_fn=lambda thing, action: executed.append(thing),
            clock=_tick,
        )
        rep = loop.run_until_goal("climb_hold", vision, state, lang_ids)
        reports[label] = {
            "goal_reached": rep.goal_reached,
            "steps": [s.thing for s in rep.steps],
            "paths": rep.paths_used,
            "events": len(hub.memory),
            "executed": executed,
            "loc_source": rep.steps[0].event.source if rep.steps else "",
        }

    result = {"v2a": reports["v2a"], "vla": reports["vla"]}
    # 反证: 两条链路都达成, 路径标注正确, 四要素事件数 = 步数, 动作真被执行
    assert result["v2a"]["goal_reached"] and result["vla"]["goal_reached"]
    assert result["v2a"]["steps"] == ["approach", "grasp", "load"]
    assert result["vla"]["steps"] == ["approach", "grasp", "load"]
    assert result["v2a"]["paths"] == {"v2a": 3}, f"v2a 链路路径标注错: {result['v2a']['paths']}"
    assert result["vla"]["paths"] == {"vla": 3}, f"vla 链路路径标注错: {result['vla']['paths']}"
    assert result["v2a"]["events"] == 3 and result["vla"]["events"] == 3
    assert result["v2a"]["executed"] == ["approach", "grasp", "load"]
    assert "slam" in result["v2a"]["loc_source"], "地点应来自 SLAM 位姿"
    return result


if __name__ == "__main__":
    r = loop_demo()
    print("VLA 双路径 × 认知中枢 端到端闭环 demo")
    for label in ("v2a", "vla"):
        d = r[label]
        name = "v→action (纯视觉)" if label == "v2a" else "v+l→action (视觉+语言)"
        print(f"  {name}: 达成={d['goal_reached']}, 步骤={d['steps']}, "
              f"路径计数={d['paths']}, 四要素事件={d['events']} 条")
