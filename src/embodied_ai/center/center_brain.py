"""中枢编排 — 端到端链路 facade (plan_v6 Sprint N: N2)。

**缺口背景（"整条链路串连有缺点"）**
    C1 信号处理 / C2 感知 / C3 记忆 / C4 决策 / C5 编排 各模块已存在但互不相接:
    - CHANGELOG B8 实锤: `POSE_UPDATE` 零发布者/零订阅者 —— "SLAM→认知中枢
      地点要素"只存在于 docstring;
    - 触觉/化学生化 → 决策输入无通道;
    - V2L 空间论证与 VLA 决策各自为战。
    本模块是**纯编排 facade** —— 把既有模块串成一条链, 不重复实现任何能力。

**链路（固定顺序）**
    组合信号 → C1 分频 (SensorFusionHub.demux)
            → C2 感知 (触觉+EKF+化学生化融合 + SLAM 位姿)
            → POSE_UPDATE 发布 (修复 B8 断链: 发布者在此, hub 订阅读取)
            → C3 记忆 (EpisodicMemory 四要素事件)
            → V2L 空间论证 (SpatialReasoner, 可解释)
            → C4 决策 (VLAPolicy 双路径: 有语言 vla / 无语言 v2a)
            → 动作执行 (act_fn 回调)
            → 事件落库 (source="center")

**不做的（诚实边界）**
    - 不重复实现传感/SLAM/VLA/记忆 (一律引用既有模块, 注册表 module_refs 为证)
    - 不做控制器 (act_fn 回调注入)、不做学习式规划

**反证要求（测试守护）**
    POSE_UPDATE 真发布且 hub.latest_pose() 收到; 事件四要素齐全;
    语言缺省走 v2a / 有语言走 vla; 输入变则决策变。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..cognition.cognitive_hub import CognitiveHub, EventRecord
from ..models.vla_config import VLAConfig
from ..models.vla_v2a import VLAPolicy
from .spatial_reasoning import SpatialArgument, SpatialReasoner


@dataclass
class CenterStepResult:
    """中枢单步结果 — 决策 + 可解释论证 + 四要素事件 + 链路状态。"""
    action: np.ndarray                  # [action_horizon, action_dim]
    decision_path: str                  # "vla" (v+l) / "v2a" (v only)
    spatial_argument: SpatialArgument   # V2L 可解释论证
    event: EventRecord                  # 四要素事件
    channels: Dict[str, np.ndarray] = field(default_factory=dict)  # C1 分频结果
    state: Dict[str, Any] = field(default_factory=dict)            # 链路状态


class CenterBrain:
    """中枢编排 facade — 组合注入既有模块, 不自带能力实现。"""

    def __init__(self,
                 bus=None,
                 hub: Optional[CognitiveHub] = None,
                 fusion_hub=None,               # SensorFusionHub (C1)
                 ekf=None,                      # MultiRateEKF (C2)
                 chem_fusion=None,              # ChemTactileFusion (C2/N4)
                 vla: Optional[VLAPolicy] = None,
                 spatial: Optional[SpatialReasoner] = None,
                 act_fn: Optional[Callable[[str, np.ndarray], None]] = None,
                 actor: str = "robot"):
        self.bus = bus
        self.hub = hub if hub is not None else CognitiveHub(bus=bus)
        self.fusion = fusion_hub
        self.ekf = ekf
        self.chem = chem_fusion
        self.vla = vla if vla is not None else VLAPolicy(VLAConfig())
        self.spatial = spatial if spatial is not None else SpatialReasoner()
        self.act_fn = act_fn
        self.actor = actor
        self._pose_seq = 0

    # ---------------- POSE_UPDATE 发布 (修复 B8 断链) ---------------- #

    def publish_pose(self, who: str, pose_xyz: np.ndarray,
                     timestamp_s: float) -> None:
        """SLAM 位姿 → Bus POSE_UPDATE → 认知中枢地点要素。

        B8 断链修复: 此前 POSE_UPDATE 零发布者, hub.latest_pose() 永远 None,
        四要素的"地点"退化为零向量。本方法是**唯一合法发布口**。
        """
        if self.bus is None:
            return
        self._pose_seq += 1
        self.bus.publish("pose_update",
                         {"pose": np.asarray(pose_xyz, dtype=np.float64).ravel(),
                          "who": who, "timestamp_s": float(timestamp_s)},
                         source="center_brain")

    # ---------------- 单步链路 ---------------- #

    def step(self, timestamp_s: float,
             composite_signal: Optional[np.ndarray],
             fs: float,
             vision_feats: np.ndarray,
             state_feats: np.ndarray,
             lang_ids: Optional[np.ndarray] = None,
             goal_vec: Optional[np.ndarray] = None,
             thing: str = "step",
             pose_xyz: Optional[np.ndarray] = None,
             tactile_feats: Optional[np.ndarray] = None,
             chem_vec: Optional[np.ndarray] = None) -> CenterStepResult:
        """端到端单步: 分频→感知→POSE_UPDATE→论证→决策→动作→事件。"""
        # C1 信号处理 (可选: 有组合信号才分频)
        channels: Dict[str, np.ndarray] = {}
        if self.fusion is not None and composite_signal is not None:
            channels = self.fusion.channelizer.demux(composite_signal, fs)

        # C2 感知: 触觉 EKF + 化学生化融合 (可选)
        material = None
        if self.chem is not None and chem_vec is not None:
            material = self.chem.fuse(chem_vec, tactile_feats)
        ekf_state = None
        if self.ekf is not None:
            self.ekf.predict()
            ekf_state = self.ekf.x.copy()

        # POSE_UPDATE — B8 断链修复 (地点要素从此有真来源)
        pose = (np.zeros(3) if pose_xyz is None
                else np.asarray(pose_xyz, dtype=np.float64).ravel())
        self.publish_pose("robot", pose, timestamp_s)

        # V2L 空间论证 (可解释性)
        gvec = (np.zeros(3) if goal_vec is None
                else np.asarray(goal_vec, dtype=np.float64).ravel())
        argument = self.spatial.argue(vision_feats, gvec)

        # C4 决策 — 双路径 (有语言 vla / 无语言 v2a)
        if lang_ids is not None:
            path = "vla"
            action = self.vla.infer_vla(vision_feats, lang_ids, state_feats)
        else:
            path = "v2a"
            action = self.vla.infer_v2a(vision_feats, state_feats)

        # 动作执行 (回调注入)
        if self.act_fn is not None:
            self.act_fn(thing, action)

        # 四要素事件 (source="center|<path>|<v2l>")
        event = EventRecord(
            timestamp_s=timestamp_s, location_xyz=pose, actor=self.actor,
            thing=thing, action=f"{path}_via_center",
            source=f"center|{path}|v2l",
        )
        if self.bus is not None:
            self.bus.publish("event", {"record": event}, source="center_brain")
        else:
            self.hub.record_event(event)

        state = {
            "channels": sorted(channels),
            "ekf_mu": None if ekf_state is None else float(ekf_state[1]),
            "material": None if material is None else material.label,
            "pose_published": self.bus is not None,
        }
        return CenterStepResult(action=action, decision_path=path,
                                spatial_argument=argument, event=event,
                                channels=channels, state=state)

    # ---------------- demo ---------------- #

    def run_demo(self) -> dict:
        """端到端 smoke: 合成链路跑 3 步, 断言地点/路径/论证非空。"""
        cfg = self.vla.cfg
        rng = np.random.default_rng(11)
        vision = rng.normal(size=(cfg.num_views, cfg.vision_embd))
        state_feats = rng.normal(size=cfg.state_dim)
        lang = rng.normal(size=cfg.LANG_TOKENS)

        results = []
        for i, lang_ids in enumerate((None, lang, None)):
            results.append(self.step(
                timestamp_s=float(i), composite_signal=None, fs=1.0,
                vision_feats=vision, state_feats=state_feats,
                lang_ids=lang_ids, thing=f"step_{i}",
                pose_xyz=np.array([float(i), 0.5, 0.0]),
            ))
        paths = [r.decision_path for r in results]
        out = {
            "paths": paths,
            "events": len(self.hub.memory),
            "pose_seen_by_hub": (None if self.hub.latest_pose("robot") is None
                                 else self.hub.latest_pose("robot").tolist()),
            "argument_steps": [len(r.spatial_argument.derivation_steps)
                               for r in results],
        }
        assert paths == ["v2a", "vla", "v2a"], f"双路径选择错: {paths}"
        assert out["events"] == 3, f"四要素事件应 3 条, 得 {out['events']}"
        assert out["pose_seen_by_hub"] == [2.0, 0.5, 0.0], \
            f"POSE_UPDATE 断链未修复, hub 收到: {out['pose_seen_by_hub']}"
        assert all(n > 0 for n in out["argument_steps"]), "V2L 论证推演步骤为空"
        return out
