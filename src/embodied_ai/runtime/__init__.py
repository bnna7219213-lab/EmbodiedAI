"""运行时层 — Bus 消息抽象 (断点 1 修复)。

InProcessBus: 同步 pub/sub (20Hz 环内零开销); CAN-FD 帧是一等消息。
跨进程/ROS2 对接不在本期范围 (Phase P 硬件侧)。
"""
from .bus import Collector, InProcessBus, Message, Topic, make_bus_backed_controller
from .mission import (MISSION_TEMPLATES, Mission, MissionError, MissionManager,
                      MissionState, MissionTemplate)

__all__ = ["Collector", "InProcessBus", "Message", "Topic",
           "make_bus_backed_controller",
           "MISSION_TEMPLATES", "Mission", "MissionError", "MissionManager",
           "MissionState", "MissionTemplate"]
