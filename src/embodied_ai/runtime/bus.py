"""断点 1 修复：最小运行时 Bus 抽象 — 层间消息化。

**缺口背景（审计报告三大断点之一）**
    层间协作靠 import 硬耦合；contracts/ 四件套只是静态 YAML/DBC，
    **没有运行时载体** —— "信号隐递控总线"只存在于文档，零代码。

**设计（最小可用，不过度设计）**
    - `Topic`: 枚举化的主题（观测/动作/遥测/安全事件/DR 更新）
    - `Message`: 带 timestamp/seq/schema_version/source 的信封（可追溯）
    - `InProcessBus`: 同步 pub/sub（20Hz 环内零延迟开销；跨进程留接口不实现）
    - CAN-FD 帧是 Bus 的一等公民（复用 DBC 编解码，600 帧已验证）

**不做的（诚实边界）**
    - 不做跨进程/跨机器（那是真实 ROS2 对接的事，Phase P 硬件侧）
    - 不做 QoS/持久化/重放（当前无此需求）
    - 不替换全部 import 调用 — 只把"该走消息"的边界（控制环 I/O、CAN、
      安全事件）挂上 Bus；内部纯函数调用保持直调（那是合理的耦合）

**反证要求（测试守护）**
    无订阅者发布不崩；坏 topic 报错；消息顺序保证；schema 校验拒绝垃圾。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


# ══════════════════════════════════════════════════════════════════
# Topic — 消息主题（对齐三栈: 控制L / 物理P / 数据D / 安全S）
# ══════════════════════════════════════════════════════════════════

class Topic(str, Enum):
    # 控制栈 L 层
    OBSERVATION = "observation"          # 传感器观测 (20Hz, 控制环输入)
    ACTION = "action"                    # 关节动作指令 (20Hz, 控制环输出)
    CAN_JOINT_CMD = "can_joint_cmd"      # CAN-FD 关节命令帧 (DBC BO_100-120)
    CAN_TELEMETRY = "can_telemetry"      # CAN-FD 遥测帧 (DBC BO_300)
    # 安全 S 横切
    SAFETY_EVENT = "safety_event"        # STO/过流/急停 (DBC BO_900)
    # 数据 D 栈
    DR_UPDATE = "dr_update"              # 域随机化参数更新 (飞轮回流)
    # 感知/认知栈 (plan_v5 Sprint M: 多传感器融合 + SLAM + 认知中枢)
    SENSOR_RAW = "sensor_raw"            # 分频处理后的传感器样本 (视频/微波/…)
    POSE_UPDATE = "pose_update"          # SLAM 位姿更新 (自身 + 对象, 带时间戳)
    EVENT = "event"                      # 认知中枢四要素事件 (时间/地点/人物/事情)


#: 每个 topic 的 payload schema（校验用; None = 任意）
_TOPIC_SCHEMAS: Dict[str, Dict[str, type]] = {
    Topic.OBSERVATION: {"joint_positions": np.ndarray},
    Topic.ACTION: {"action": np.ndarray},
    Topic.CAN_JOINT_CMD: {"frame": object},        # CANFrame
    Topic.CAN_TELEMETRY: {"frame": object},
    Topic.SAFETY_EVENT: {"event_type": str},
    Topic.DR_UPDATE: {"overrides": dict},
    # 感知/认知栈 (plan_v5 Sprint M)
    Topic.SENSOR_RAW: {"channel": str, "samples": np.ndarray},
    Topic.POSE_UPDATE: {"pose": np.ndarray},
    Topic.EVENT: {"record": object},           # cognition.EventRecord
}


# ══════════════════════════════════════════════════════════════════
# Message — 消息信封
# ══════════════════════════════════════════════════════════════════

@dataclass
class Message:
    """Bus 消息信封 — 每条消息可追溯（谁/何时/第几条/什么版本）。"""
    topic: Topic
    payload: Dict[str, Any]
    timestamp_s: float = field(default_factory=time.time)
    seq: int = 0                          # 由 Bus 填充（全局单调递增）
    source: str = ""                      # 发布者标识（如 "controller_20hz"）
    schema_version: str = "v1"

    def validate(self) -> None:
        """按 topic schema 校验 payload — 垃圾消息在入口被拒。"""
        schema = _TOPIC_SCHEMAS.get(self.topic)
        if schema is None:
            return
        for key, typ in schema.items():
            if key not in self.payload:
                raise ValueError(f"{self.topic}: payload 缺 '{key}'")
            v = self.payload[key]
            if typ is np.ndarray and not isinstance(v, np.ndarray):
                raise ValueError(f"{self.topic}.{key}: 期望 ndarray, 得 {type(v).__name__}")
            elif typ is not np.ndarray and not isinstance(v, typ):
                raise ValueError(f"{self.topic}.{key}: 期望 {typ.__name__}, 得 {type(v).__name__}")


# ══════════════════════════════════════════════════════════════════
# InProcessBus — 同步 pub/sub
# ══════════════════════════════════════════════════════════════════

class InProcessBus:
    """进程内同步消息总线（20Hz 环内零序列化开销）。

    语义保证:
      · publish: 同步逐个调用订阅者（顺序 = 订阅顺序）; 异常不中断其他订阅者
      · 无订阅者: 静默丢弃（记入 stats.dropped — 可观测, 不崩）
      · 顺序: 同一 topic 内 FIFO; seq 全局单调递增
      · 线程安全: 订阅/发布可跨线程（锁保护订阅表; 分发在锁外）
    """

    def __init__(self, name: str = "bus"):
        self.name = name
        self._subs: Dict[Topic, List[Callable[[Message], None]]] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self.stats: Dict[str, int] = {"published": 0, "delivered": 0,
                                      "dropped": 0, "rejected": 0}

    def subscribe(self, topic: Union[Topic, str],
                  handler: Callable[[Message], None]) -> None:
        """订阅主题。handler 收到完整 Message 信封。"""
        t = self._to_topic(topic)
        with self._lock:
            self._subs.setdefault(t, []).append(handler)

    def publish(self, topic: Union[Topic, str], payload: Dict[str, Any],
                source: str = "") -> Message:
        """发布消息 — 同步分发给全部订阅者; 返回已投递的 Message。"""
        t = self._to_topic(topic)
        msg = Message(topic=t, payload=payload, source=source)
        msg.validate()                     # 垃圾在入口被拒
        with self._lock:
            self._seq += 1
            msg.seq = self._seq
            handlers = list(self._subs.get(t, []))
        self.stats["published"] += 1
        if not handlers:
            self.stats["dropped"] += 1
            return msg
        for h in handlers:
            try:
                h(msg)
                self.stats["delivered"] += 1
            except Exception:              # noqa: BLE001 — 单订阅者异常不阻断总线
                self.stats["rejected"] += 1
        return msg

    @property
    def topics_active(self) -> List[str]:
        """有订阅者的主题（可观测）。"""
        with self._lock:
            return [t.value for t, hs in self._subs.items() if hs]

    def subscriber_count(self, topic: Union[Topic, str]) -> int:
        """某主题的订阅者数量 — 用来发现"每次调用重复 subscribe"的泄漏。"""
        t = self._to_topic(topic)
        with self._lock:
            return len(self._subs.get(t, []))

    @staticmethod
    def _to_topic(t: Union[Topic, str]) -> Topic:
        if isinstance(t, Topic):
            return t
        try:
            return Topic(t)
        except ValueError:
            raise ValueError(f"未知 topic '{t}'; 合法: {[x.value for x in Topic]}") from None


# ══════════════════════════════════════════════════════════════════
# 便捷: 收集器（测试/遥测用）
# ══════════════════════════════════════════════════════════════════

class Collector:
    """把收到的消息存进列表（测试断言 / 遥测记录用）。"""

    def __init__(self):
        self.messages: List[Message] = []

    def __call__(self, msg: Message) -> None:
        self.messages.append(msg)

    @property
    def last(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None


# ══════════════════════════════════════════════════════════════════
# 控制环接入 — BusBackedControllerMixin
# ══════════════════════════════════════════════════════════════════

def make_bus_backed_controller(controller, bus: InProcessBus,
                               source: str = "controller_20hz") -> None:
    """把 20Hz 控制器接到 Bus（就地改造, 不新建类 — 最小侵入）。

    接线后:
      · step(obs) 内部发布 OBSERVATION（含 joint_positions）
      · step 返回前发布 ACTION（含 action 向量）
      · 提供 step_via_bus(obs) — 从 Bus 上一次 OBSERVATION 消息取输入
        （模拟"观测来自消息层而非直调"的真实路径）

    原 step() 行为完全保留（向后兼容 — 既有 494 个测试不破）。
    """
    original_step = controller.step
    _last_obs = {"joint_positions": np.zeros(12)}

    def on_observation(msg: Message) -> None:
        _last_obs["joint_positions"] = msg.payload["joint_positions"]

    bus.subscribe(Topic.OBSERVATION, on_observation)

    def step_with_bus(obs):
        action = original_step(obs)
        bus.publish(Topic.OBSERVATION,
                    {"joint_positions": np.asarray(obs.get("joint_positions",
                                                           np.zeros(12)))},
                    source=source)
        bus.publish(Topic.ACTION, {"action": np.asarray(action)},
                    source=source)
        return action

    def step_via_bus(obs):
        """观测经 Bus 流转的版本: 发布 → on_observation 回环收到 → 再执行。

        publish 是同步分发, 所以上面的 on_observation 在 publish 返回前已写入
        _last_obs —— 这就是"感知节点发布、控制器订阅"的最小闭环。

        曾经这里每次调用新建 Collector 并重复 subscribe（订阅者无限增长）, 而
        真正喂给 original_step 的是一个从未被赋值的常量 _last_obs_joints,
        即该路径**永远执行全零观测**, 却被测试与 verify runtime_bus 放行。
        """
        bus.publish(Topic.OBSERVATION,
                    {"joint_positions": np.asarray(obs.get("joint_positions",
                                                           np.zeros(12)))},
                    source=source)
        action = original_step({"joint_positions": _last_obs["joint_positions"]})
        bus.publish(Topic.ACTION, {"action": np.asarray(action)},
                    source=source)
        return action

    controller.step = step_with_bus
    controller.bus = bus
    controller.step_via_bus = step_via_bus
    controller.last_observation = lambda: dict(_last_obs)
