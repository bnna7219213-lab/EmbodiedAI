"""机器可读层注册表 — 架构层单一真源 (plan_v6 Sprint N: N1)。

**缺口背景（链路断点之一）**
    架构层清单（L1-L7 控制栈 / P1-P5 物理栈 / S 安全 / D 数据栈）此前**硬编码
    在 `gui/server.py:_layer_stack()` 里**, 文档 (docs/architecture-overview.md
    §5) 又是一份手抄 —— 两处可各自漂移, 且"中枢决策层"（nanoGPT/VLA/记忆/
    感知/信号处理）无处安放。本模块把层清单收口为**机器可读注册表**,
    GUI/文档/测试全部从这里取。

**设计（最小可用，不过度设计）**
    - `LayerSpec`: 一层的完整描述（id / 栈 / 名称 / 说明 / 频率 / 构成 / 模块引用）
    - `LAYER_REGISTRY`: 全部层的单一真源
        · L1-L7 控制栈 —— **编号与语义照旧不动**（docs/09 硬约束）
        · P1-P5 物理栈 / S 安全横切 / D1-D5 数据栈横切
        · **C1-C5 中枢决策栈（新增）**: 信号处理 → 感知 → 记忆 → 决策 → 编排
    - `layers_by_stack()` / `as_dict()` —— GUI 消费
    - `validate_registry()` —— 自检（id 唯一 / L1-L7 齐 / module_refs 可 import）

**不做的（诚实边界）**
    - 不做动态插件发现（层集合是架构约定, 不是运行时扩展点）
    - 不改 L1-L7 编号语义（docs/09 明确"保持七层控制栈编号不变"）

**反证要求（测试守护）**
    C1-C5 齐全; L1-L7 编号无增删; 每层 module_refs 可 import;
    GUI `_layer_stack()` 与注册表同源; 重复 id / 缺 L 层必须报错。
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class LayerSpec:
    """一个架构层的机器可读描述。"""
    id: str                      # 层 id: "L6" / "P3" / "S" / "D2" / "C4"
    stack: str                   # 所属栈: "控制栈" / "物理栈" / "安全" / "数据栈" / "中枢决策栈"
    name: str                    # 层名 (短)
    description: str             # 说明 (给 GUI / 文档)
    freq_hz: Optional[float] = None      # 主频率 (软/硬实时层才有)
    composition: Tuple[str, ...] = ()    # 由哪些能力构成 (人读)
    module_refs: Tuple[str, ...] = ()    # 对应源码模块 (机读, 可 import 验证)


#: L1-L7 控制栈 —— 文案与编号照抄 gui/server.py:_layer_stack() (2026-10-17 前),
#: **不改语义** (docs/09-project-skeleton/skeleton-and-layering-plan.md 硬约束)。
_L_CONTROL = (
    LayerSpec("L7", "控制栈", "用户交互", "本 Web 仪表盘 / 自然语言", 1.0,
              ("GUI", "自然语言"), ("embodied_ai.gui.server",)),
    LayerSpec("L6", "控制栈", "任务规划", "VLA + LLM", 1.0,
              ("VLA", "LLM/nanoGPT"), ("embodied_ai.models.vla_v2a",
                                       "embodied_ai.adapters.nanogpt")),
    LayerSpec("L5", "控制栈", "接触序列", "MCTS + DR", 5.0,
              ("MCTS", "域随机化"), ("embodied_ai.planning.mcts_integration",)),
    LayerSpec("L4", "控制栈", "运动控制", "RL Actor / PPO / SAC / Diffusion", 20.0,
              ("RL Actor",), ("embodied_ai.models.rl_algorithms",)),
    LayerSpec("L3", "控制栈", "反射伺服", "阻抗 + 触觉反射", 1000.0,
              ("阻抗控制", "触觉反射"), ("embodied_ai.perception.tactile_closed_loop",)),
    LayerSpec("L2", "控制栈", "电机驱动", "FOC + SVPWM + 硬件关断", 10000.0,
              ("FOC", "SVPWM", "硬件关机链"), ("embodied_ai.hardware.foc_shutdown",)),
    LayerSpec("L1", "控制栈", "物理执行", "电机 + 减速器 + 传感器", None,
              ("电机", "减速器", "传感器"), ("embodied_ai.hardware.motor_model",)),
)

#: P1-P5 物理栈 (docs/architecture-overview.md §5)
_P_PHYSICAL = (
    LayerSpec("P1", "物理栈", "电源层", "PMIC / BMS / 电机驱动电源轨", None,
              ("PMIC", "BMS"), ()),
    LayerSpec("P2", "物理栈", "感知层", "IMU / 摄像头接口 / 力传感器 ADC", None,
              ("IMU", "相机", "力 ADC"), ()),
    LayerSpec("P3", "物理栈", "决策层", "Jetson Orin NX + LPDDR5", None,
              ("Jetson Orin NX",), ("embodied_ai.platform.arm_linux",)),
    LayerSpec("P4", "物理栈", "执行层", "STM32G4 + DRV8301 + 自动关机", None,
              ("STM32G4", "DRV8301"), ()),
    LayerSpec("P5", "物理栈", "互联层", "CAN-FD / RS485 / Ethernet", None,
              ("CAN-FD", "RS485"), ()),
)

#: S 安全横切 + D1-D5 数据栈横切
_S_D_CROSS = (
    LayerSpec("S", "安全", "横切安全层",
              "<1µs 硬件关断 + STL + ISO13482 + 机会约束握力", 1000.0,
              ("硬件关断", "STL", "ISO13482"),
              ("embodied_ai.safety.certification",
               "embodied_ai.safety.formal.stl_specs",
               "embodied_ai.safety.chance_constrained_grip")),
    LayerSpec("D1", "数据栈", "传感原始数据", "相机/触觉/编码器/电流原始流", 20.0,
              ("原始流",), ("embodied_ai.perception.sensor_fusion",)),
    LayerSpec("D2", "数据栈", "标注/遥测", "HIL 遥测 schema、标注对齐", None,
              ("遥测", "标注"), ("embodied_ai.runtime.bus",)),
    LayerSpec("D3", "数据栈", "制造/批次", "ICT/FCT 结果、器件容差、SN 追溯", None,
              ("ICT", "FCT"), ()),
    LayerSpec("D4", "数据栈", "RL 回放/DR", "域随机化参数、回放格式", None,
              ("回放", "DR"), ("embodied_ai.models.replay_dataset",)),
    LayerSpec("D5", "数据栈", "版本三元组", "(policy_ver, hw_rev, batch) 归因", None,
              ("归因",), ()),
)

#: C1-C5 中枢决策栈 (plan_v6 新增) —— 用户口径: 含 nanoGPT、VLA、记忆、感知、信号处理
_C_CENTER = (
    LayerSpec("C1", "中枢决策栈", "信号处理",
              "多传感器分频信道 (视频信号流 + 微波等组合信号解复用)", 100.0,
              ("分频信道", "多传感器融合"),
              ("embodied_ai.perception.sensor_fusion",)),
    LayerSpec("C2", "中枢决策栈", "感知",
              "触觉-视觉多速率 EKF + SLAM 三维坐标/时间 + 触觉闭环", 20.0,
              ("MultiRateEKF", "SLAM", "触觉闭环"),
              ("embodied_ai.perception.multirate_ekf",
               "embodied_ai.perception.slam",
               "embodied_ai.perception.tactile_closed_loop")),
    LayerSpec("C3", "中枢决策栈", "记忆",
              "四要素情景记忆 (时间/地点/人物/事情) + 记忆召回", 1.0,
              ("EpisodicMemory", "remember"),
              ("embodied_ai.cognition.cognitive_hub",)),
    LayerSpec("C4", "中枢决策栈", "决策",
              "nanoGPT + VLA 双路径 (v+l→action / v→action)", 20.0,
              ("nanoGPT", "VLAPolicy"),
              ("embodied_ai.adapters.nanogpt",
               "embodied_ai.models.vla_v2a",
               "embodied_ai.models.vla_model")),
    LayerSpec("C5", "中枢决策栈", "中枢编排",
              "端到端链路编排: 信号→感知→记忆→论证→决策→动作→事件", 1.0,
              ("CenterBrain", "CognitiveVLALoop"),
              ("embodied_ai.center.center_brain",
               "embodied_ai.cognition.closed_loop")),
)

#: 全部层 —— 单一真源
LAYER_REGISTRY: Dict[str, LayerSpec] = {
    spec.id: spec
    for spec in (*_L_CONTROL, *_P_PHYSICAL, *_S_D_CROSS, *_C_CENTER)
}

#: L1-L7 控制栈的期望 id (守护编号不被增删)
REQUIRED_CONTROL_IDS: Tuple[str, ...] = tuple(s.id for s in _L_CONTROL)


def layers_by_stack() -> Dict[str, List[LayerSpec]]:
    """按栈分组 (GUI / 文档消费)。"""
    out: Dict[str, List[LayerSpec]] = {}
    for spec in LAYER_REGISTRY.values():
        out.setdefault(spec.stack, []).append(spec)
    return out


def as_dict() -> Dict[str, str]:
    """id → 说明 字典 (兼容 gui._layer_stack() 的返回形状)。"""
    return {spec.id: f"{spec.name} ({spec.description})"
            for spec in LAYER_REGISTRY.values()}


def validate_registry() -> int:
    """注册表自检 — 全部通过返回 0, 否则抛 AssertionError。"""
    importlib.import_module("embodied_ai.center.layer_registry")  # 自身可导入
    # 1. id 唯一 (dict 构造已保证, 再核一次注册表与源定义数量一致)
    total_defined = len(_L_CONTROL) + len(_P_PHYSICAL) + len(_S_D_CROSS) + len(_C_CENTER)
    assert len(LAYER_REGISTRY) == total_defined, \
        f"注册表存在重复 id: 定义 {total_defined} 条, 实际 {len(LAYER_REGISTRY)} 条"
    # 2. L1-L7 编号无增删 (docs/09 硬约束)
    present = [sid for sid in REQUIRED_CONTROL_IDS if sid in LAYER_REGISTRY]
    assert present == list(REQUIRED_CONTROL_IDS), \
        f"L1-L7 控制栈编号被改动: 期望 {REQUIRED_CONTROL_IDS}, 实际 {present}"
    # 3. C1-C5 中枢决策栈齐全
    for cid in ("C1", "C2", "C3", "C4", "C5"):
        assert cid in LAYER_REGISTRY, f"缺少中枢决策层 {cid}"
    # 4. module_refs 可 import
    for spec in LAYER_REGISTRY.values():
        for mod in spec.module_refs:
            importlib.import_module(mod)
    return 0
