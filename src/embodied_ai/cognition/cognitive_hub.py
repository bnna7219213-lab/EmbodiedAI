"""认知中枢 — 四要素事件的记录/分析/计算/规划/记忆 (plan_v5 Sprint M: M4)。

**缺口背景（机器人端三大缺口之一）**
    机器人对**四要素（时间、地点、人物、事情）**进行记录、分析、计算、规划、
    记忆，形成认知中枢系统。项目此前有 mission.py（任务状态机）与 bus.py
    （消息流转），但没有"事件认知"层——四要素无处安放，无法回答
    "谁在何时何地做了什么"。

**设计（最小可用，不过度设计）**
    - `EventRecord`: 四要素事件记录 —
        时间 timestamp_s / 地点 location_xyz / 人物 actor / 事情 thing+action
    - `EpisodicMemory`: 情景记忆存储 + 多维查询（时间范围/地点半径/人物/事情）
    - `CognitiveHub` 五能力（对齐用户口径）:
        ① 记录 record_event —— 也订阅 Bus `EVENT` topic
        ② 分析 analyze —— 计数/频次/时序
        ③ 计算 compute —— 事件率/地点聚类等统计量
        ④ 规划 plan —— 目标事件 → 前置事件序列（规则式最小规划器）
        ⑤ 记忆 remember —— 时间衰减召回排序

**provenance（诚实边界）**
    规划器为**规则式最小实现**（先决条件表驱动），不是学习式规划；
    时间衰减率等参数为 assumed（合成验证）。

**不做的（诚实边界）**
    - 不做 LLM 语义理解 / 学习式规划（规划层已有 MCTS, 见 planning/）
    - 不做持久化存储（内存情景记忆; 落盘属后续）
    - 不做事件本体/知识图谱（过度设计）

**反证要求（测试守护）**
    空记忆查询不崩；坏四要素（缺时间/负时间戳）报错；地点过滤只给一半参数
    必须报错（不得静默不过滤）；plan 未知目标必须抛 KeyError（不得与"前置
    已满足"同样返回空）；remember 的衰减得分必须随 DECAY_RATE 变化。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════
# EventRecord — 四要素事件
# ══════════════════════════════════════════════════════════════════

@dataclass
class EventRecord:
    """四要素事件记录: 时间 / 地点 / 人物 / 事情。

    四要素映射:
      时间 = timestamp_s        地点 = location_xyz (3D 坐标, 来自 SLAM)
      人物 = actor              事情 = thing (对象) + action (动作)
    """
    timestamp_s: float                       # ① 时间
    location_xyz: np.ndarray                 # ② 地点 (3D 坐标)
    actor: str                               # ③ 人物 (执行者/智能体)
    thing: str                               # ④ 事情 (对象/主题)
    action: str = ""                         # ④ 事情 (动作描述)
    object_: Optional[str] = None            # 涉及的物体对象 (可选)
    source: str = ""                         # 来源标识 (如 "slam" / "operator")

    def __post_init__(self):
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s 必须 >= 0")
        if not self.actor:
            raise ValueError("actor (人物) 不能为空")
        if not self.thing:
            raise ValueError("thing (事情) 不能为空")
        loc = np.asarray(self.location_xyz, dtype=np.float64).ravel()
        if loc.size != 3:
            raise ValueError("location_xyz 必须是 3 维")
        self.location_xyz = loc

    @property
    def four_elements(self) -> Tuple[float, np.ndarray, str, str]:
        """(时间, 地点, 人物, 事情)。"""
        return self.timestamp_s, self.location_xyz, self.actor, self.thing


# ══════════════════════════════════════════════════════════════════
# EpisodicMemory — 情景记忆
# ══════════════════════════════════════════════════════════════════

class EpisodicMemory:
    """情景记忆存储 + 多维查询（记忆要素）。"""

    def __init__(self):
        self._events: List[EventRecord] = []

    def record(self, event: EventRecord) -> int:
        """记录一条事件, 返回索引。"""
        self._events.append(event)
        return len(self._events) - 1

    def query(self,
              time_range: Optional[Tuple[float, float]] = None,
              location_center: Optional[np.ndarray] = None,
              location_radius: Optional[float] = None,
              actor: Optional[str] = None,
              thing: Optional[str] = None) -> List[EventRecord]:
        """多维查询: 时间范围 / 地点半径 / 人物 / 事情（AND 组合）。

        Raises:
            ValueError: location_center 与 location_radius 只给了一个 —— 曾经
                这种情况静默跳过地点过滤 (99m 外的事件照样返回), 调用方以为
                过滤生效了。
        """
        if (location_center is None) != (location_radius is None):
            raise ValueError(
                "location_center 与 location_radius 必须同时给出或同时省略 "
                f"(得 center={location_center is not None}, "
                f"radius={location_radius is not None})")
        out = []
        for e in self._events:
            if time_range is not None and not (time_range[0] <= e.timestamp_s <= time_range[1]):
                continue
            if actor is not None and e.actor != actor:
                continue
            if thing is not None and e.thing != thing:
                continue
            if location_center is not None:
                c = np.asarray(location_center, dtype=np.float64).ravel()
                if np.linalg.norm(e.location_xyz - c) > location_radius:
                    continue
            out.append(e)
        return out

    def __len__(self) -> int:
        return len(self._events)


# ══════════════════════════════════════════════════════════════════
# CognitiveHub — 五能力
# ══════════════════════════════════════════════════════════════════

class CognitiveHub:
    """认知中枢: 对四要素记录、分析、计算、规划、记忆。

    对齐用户需求口径:
      "对四要素（时间、地点、人物、事情）进行记录、分析、计算、规划、记忆
       形成了认知中枢系统"。
    """

    #: 时间衰减率 (1/s, assumed — 记忆召回衰减参数)
    DECAY_RATE = 0.1

    def __init__(self, bus=None):
        self.memory = EpisodicMemory()
        self.bus = bus
        #: SLAM 经 Bus POSE_UPDATE 送来的最新位姿 (who → [x,y,z]) —— 四要素里
        #: "地点"的实时来源。此前 POSE_UPDATE 零订阅者, 该数据流并不存在。
        self._poses: Dict[str, np.ndarray] = {}
        self._pose_timestamp_s: Dict[str, float] = {}
        if bus is not None:
            bus.subscribe("event", self._on_bus_event)
            bus.subscribe("pose_update", self._on_bus_pose)

    def _on_bus_pose(self, msg) -> None:
        p = msg.payload.get("pose")
        if p is None:
            return
        who = str(msg.payload.get("who", "unknown"))
        arr = np.asarray(p, dtype=np.float64).ravel()
        if arr.size != 3:
            return
        self._poses[who] = arr
        self._pose_timestamp_s[who] = float(msg.payload.get("timestamp_s", 0.0))

    def latest_pose(self, who: str = "robot") -> Optional[np.ndarray]:
        """某个对象当前位姿 —— 来自 SLAM 的 POSE_UPDATE, 无数据返回 None。"""
        return self._poses.get(who)

    # ① 记录 #

    def record_event(self, event: EventRecord) -> int:
        return self.memory.record(event)

    def _on_bus_event(self, msg) -> None:
        """订阅 Bus `EVENT` topic — 消息层事件进入情景记忆。"""
        rec = msg.payload.get("record")
        if isinstance(rec, EventRecord):
            self.memory.record(rec)

    # ② 分析 #

    def analyze(self) -> Dict[str, object]:
        """计数/频次/时序分析（分析要素）。"""
        events = self.memory.query()
        if not events:
            return {"total": 0, "by_actor": {}, "by_thing": {},
                    "time_span_s": 0.0, "sequence": []}
        by_actor: Dict[str, int] = {}
        by_thing: Dict[str, int] = {}
        for e in events:
            by_actor[e.actor] = by_actor.get(e.actor, 0) + 1
            by_thing[e.thing] = by_thing.get(e.thing, 0) + 1
        ts = [e.timestamp_s for e in events]
        return {
            "total": len(events),
            "by_actor": by_actor,
            "by_thing": by_thing,
            "time_span_s": max(ts) - min(ts),
            "sequence": [e.thing for e in sorted(events, key=lambda x: x.timestamp_s)],
        }

    # ③ 计算 #

    def compute(self) -> Dict[str, float]:
        """统计量计算: 事件率 / 地点聚类均值（计算要素）。"""
        events = self.memory.query()
        if not events:
            return {"event_rate_hz": 0.0}
        ts = np.array([e.timestamp_s for e in events])
        locs = np.stack([e.location_xyz for e in events])
        span = float(ts.max() - ts.min()) if ts.size > 1 else 0.0
        rate = (len(events) - 1) / span if span > 0 else 0.0
        centroid = locs.mean(axis=0)
        return {
            "event_rate_hz": float(rate),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "location_spread_m": float(np.linalg.norm(locs - centroid, axis=1).mean()),
        }

    # ④ 规划 #

    #: 先决条件表 (规则式最小规划器): goal_thing → 需先发生的前置事件 thing 序列
    PLAN_PREREQUISITES: Dict[str, List[str]] = {
        "pick": ["approach", "grasp"],
        "place": ["pick", "move_to"],
        "climb_hold": ["approach", "grasp", "load"],
    }

    def plan(self, goal_thing: str,
             available: Optional[List[str]] = None) -> List[str]:
        """目标 → 尚缺的前置事件序列（规划要素, 规则式）。

        Args:
            goal_thing: 目标事件
            available: 已具备的前置事件（默认从情景记忆的 thing 集合取）
        Returns:
            需补充执行的事件序列; **空列表 = 前置条件已满足**

        Raises:
            KeyError: 目标未在先决条件表中。曾经未知目标也返回 [], 与
                "前置条件已满足" 完全不可区分 —— docstring 承诺的 "标注"
                并不存在, 调用方无法判断是该执行空计划还是目标根本不存在。
        """
        if goal_thing not in self.PLAN_PREREQUISITES:
            raise KeyError(
                f"未知规划目标 '{goal_thing}'; 已定义: "
                f"{sorted(self.PLAN_PREREQUISITES)}")
        if available is None:
            available = list({e.thing for e in self.memory.query()})
        have = set(available)
        return [step for step in self.PLAN_PREREQUISITES[goal_thing]
                if step not in have]

    # ⑤ 记忆 #

    def remember_scored(self, query_thing: Optional[str] = None,
                        now_s: Optional[float] = None
                        ) -> List[Tuple[float, EventRecord]]:
        """记忆召回 + 衰减得分 — 得分 = exp(-DECAY_RATE · (now - t))。

        纯时间衰减是单调变换, **不改变次序**（次序恒等于按时间倒序）。暴露
        得分是为了让 DECAY_RATE 成为可数值验证的量, 而不是一个对输出毫无
        影响的装饰参数（曾经改 DECAY_RATE 从 0.001 到 50.0, 召回序完全不变,
        "按时间衰减排序" 与 "按时间倒序" 无法区分）。
        """
        events = self.memory.query(thing=query_thing)
        if not events:
            return []
        t_now = (max(e.timestamp_s for e in events)
                 if now_s is None else now_s)
        scored = [(float(np.exp(-self.DECAY_RATE * (t_now - e.timestamp_s))), e)
                  for e in events]
        scored.sort(key=lambda se: se[0], reverse=True)
        return scored

    def remember(self, query_thing: Optional[str] = None,
                 now_s: Optional[float] = None) -> List[EventRecord]:
        """记忆召回 — 按时间衰减排序（记忆要素）。

        越近的事件得分越高; query_thing 过滤事情。需要得分本身请用
        `remember_scored()`。
        """
        return [e for _, e in self.remember_scored(query_thing, now_s)]


# ══════════════════════════════════════════════════════════════════
# demo — 四要素闭环
# ══════════════════════════════════════════════════════════════════

def cognitive_demo() -> dict:
    """四要素闭环演示: 记录 3 事件 → 分析/计算/规划/记忆。"""
    hub = CognitiveHub()
    hub.record_event(EventRecord(0.0, np.array([1.0, 0.0, 0.0]),
                                 "robot", "approach", "move_to_target"))
    hub.record_event(EventRecord(2.0, np.array([2.0, 0.5, 0.0]),
                                 "robot", "grasp", "close_fingers",
                                 object_="cup"))
    hub.record_event(EventRecord(5.0, np.array([2.0, 0.5, 0.0]),
                                 "operator", "handover", "give_cup",
                                 object_="cup"))

    analysis = hub.analyze()
    stats = hub.compute()
    plan = hub.plan("pick")                  # approach/grasp 已具备 → 应为空或缺啥补啥
    plan_full = hub.plan("climb_hold")       # load 缺失 → ["load"]
    recall = hub.remember()
    result = {
        "analysis": analysis,
        "stats": stats,
        "plan_pick": plan,
        "plan_climb_hold": plan_full,
        "recall_order": [e.thing for e in recall],
    }
    assert analysis["total"] == 3
    assert set(analysis["by_actor"]) == {"robot", "operator"}
    assert plan_full == ["load"], f"climb_hold 应只缺 load, 得 {plan_full}"
    # 记忆衰减: 最近事件 (5.0 handover) 应排最前
    assert result["recall_order"][0] == "handover"
    return result


if __name__ == "__main__":
    r = cognitive_demo()
    print("认知中枢 demo (四要素: 时间/地点/人物/事情)")
    print(f"分析: total={r['analysis']['total']}, actors={r['analysis']['by_actor']}")
    print(f"计算: 事件率={r['stats']['event_rate_hz']:.2f}Hz, "
          f"质心=({r['stats']['centroid_x']:.1f},{r['stats']['centroid_y']:.1f},"
          f"{r['stats']['centroid_z']:.1f})")
    print(f"规划: pick 需补 {r['plan_pick']}, climb_hold 需补 {r['plan_climb_hold']}")
    print(f"记忆召回序: {r['recall_order']} (时间衰减)")
