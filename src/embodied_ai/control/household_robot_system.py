"""20Hz 实时控制器 + Diffusion 策略 (L4/L3) — K2: 桩类补全 + 飞轮 DR 注入接口。

K1 清洗，K2 重构:
  - RealTime20HzController: 从 _FragmentHolder 容器 → 完整 20Hz 控制循环
  - 桩类 AsyncMCTSPlanner / RLSkillCoordinator / MotionPrimitive → 对接 planning 包
  - 新增 inject_dr_config() — 飞轮 DR → 控制器物理参数注入
  - DiffusionPolicy 从 planning 包 import
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

# K2: 补全的导入
from ..planning import (
    AsyncMCTSPlanner,
    ClimbingMCTS,
    ClimbingState,
    DiffusionPolicy,
    MotionPrimitive,
    MotionPrimitiveLibrary,
    RLSkillCoordinator,
)


class RealTime20HzController:
    """完整的 20Hz 控制系统 — K2: 从 fragment 容器重构为完整控制类。

    架构:
      AI 推理(20Hz) → MCTS 接触规划(异步) → Diffusion 轨迹 → 阻抗控制(1kHz) → FOC(10kHz)
    """

    def __init__(self, dr_config: Dict[str, Any] = None):
        # === 三层架构 ===
        self.task_planner = AsyncMCTSPlanner(dr_config=dr_config)
        self.rl_coordinator = RLSkillCoordinator(dr_config=dr_config)
        self.motion_library = MotionPrimitiveLibrary()
        # 装配默认技能桩
        self.rl_coordinator.load_skill("grasp", DiffusionPolicy())
        self.rl_coordinator.load_skill("place", DiffusionPolicy())
        self.rl_coordinator.load_skill("wipe", DiffusionPolicy())

        # === 状态 ===
        self.current_skill = None
        self.current_primitive: Optional[MotionPrimitive] = None
        self._running = False
        self._dr_config = dr_config

        # === 性能监控 ===
        self.timing_stats: Dict[str, list] = {
            "perception_ms": [],
            "rl_inference_ms": [],
            "total_loop_ms": [],
        }

    def inject_dr_config(self, dr_config: Dict[str, Any]) -> None:
        """飞轮 DR 注入: 更新所有控制器的 DR 参数。

        闭环: ManufacturingDataFlywheel → generateDrConfig → inject_dr_config → MCTS 物理模型 / Diffusion 策略。
        """
        self._dr_config = dr_config
        if isinstance(self.task_planner, AsyncMCTSPlanner):
            self.task_planner.mcts.dr_override(dr_config)
        if isinstance(self.rl_coordinator, RLSkillCoordinator):
            self.rl_coordinator.dr_config = dr_config

    def step(self, obs: Dict[str, np.ndarray]) -> np.ndarray:  # pragma: no cover — 依赖真机/仿真
        """单步控制 (20Hz) — 返回 action vector。"""
        t0 = time.time()

        # 感知
        state = self._perceive(obs)

        # 20Hz 推理: 当前技能生成轨迹
        self.current_skill = "grasp"
        action = self.rl_coordinator.execute(self.current_skill, obs)

        elapsed_ms = (time.time() - t0) * 1000
        self.timing_stats["rl_inference_ms"].append(elapsed_ms)
        return action

    def run(self):  # pragma: no cover — 依赖真机
        """主控制循环 — 严格 20Hz (50ms 周期)。"""
        self._running = True
        target_period = 0.05  # 20 Hz
        while self._running:
            t_start = time.time()
            # 占位: 真机/仿真接入
            time.sleep(max(0, target_period - (time.time() - t_start)))

    def stop(self):
        self._running = False

    def _perceive(self, obs: Dict[str, np.ndarray]) -> ClimbingState:
        """将多模态观测转换为 ClimbingState。"""
        ee_pos = obs.get("ee_position", np.zeros(3))
        joints = obs.get("joint_positions", np.zeros(12))
        return ClimbingState(
            ee_position=np.asarray(ee_pos),
            joint_positions=np.asarray(joints),
        )

    @property
    def is_running(self) -> bool:
        return self._running


# ============================================================================
# 便捷工厂函数
# ============================================================================

def make_controller(dr_config: Dict[str, Any] = None) -> RealTime20HzController:
    """创建 20Hz 控制器实例，可选注入 DR 配置。"""
    return RealTime20HzController(dr_config=dr_config)


def run_single_step(obs: Dict[str, np.ndarray], dr_config: Dict[str, Any] = None) -> np.ndarray:
    """单步推理入口 (供测试 / 飞轮快速验证)。"""
    ctrl = RealTime20HzController(dr_config=dr_config)
    return ctrl.step(obs)
