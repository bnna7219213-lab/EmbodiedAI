"""V2L 空间智能 — 可解释空间论证 (plan_v6 Sprint N: N5)。

**缺口背景（可解释性断点）**
    用户口径: "OS 层动作规划与可解释性由 Video-to-Language 的空间智能决定
    （想象空间思考，类比高二立体几何画辅助线、演算，能否产生决策）"。
    项目此前的 VLA 决策是端到端黑盒 —— 能出动作, 但说不出**为什么**。
    本模块补"空间论证"层: 从视觉特征构造辅助线式假设 → 几何推演 → 结论,
    结论与决策一并输出, 论证文本可进 GUI / 情景记忆 (source="v2l")。

**设计（最小可用，不过度设计）**
    - `SpatialArgument`: 论证结构 —— auxiliary_hypotheses（辅助线假设）
      / derivation_steps（推演步骤）/ conclusion（结论）/ language_caption
      （自然语言摘要）/ decision（决策向量）
    - `SpatialReasoner.argue(vision_feats, goal_vec)`: numpy 几何规则
      · 假设 = 感知特征各视角主方向 / 目标方位（"画辅助线"）
      · 推演 = 点积/夹角/距离计算（"演算"）
      · 结论 = 朝目标的方位判定 → 决策向量（"产生决策"）
    - `to_event()`: 论证 → 四要素事件 (source="v2l"), 喂 CognitiveHub

**provenance（诚实边界）**
    规则式几何推理（非学习式），caption 为模板化中文 —— **不是**语言模型输出。
    真实 V2L 语义理解属 待领域验证。

**不做的（诚实边界）**
    - 不做语言模型 caption 生成（模板化即可, 大模型属后续）
    - 不做三维重建/点云推理（几何规则按特征向量近似）

**反证要求（测试守护）**
    derivation_steps 非空; conclusion 与 decision 方向一致;
    输入变则结论/决策必变（防恒定输出假论证）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SpatialArgument:
    """空间论证结构 — 立体几何式: 辅助线 → 演算 → 结论。"""
    auxiliary_hypotheses: List[str] = field(default_factory=list)   # 辅助线/假设
    derivation_steps: List[str] = field(default_factory=list)       # 演算步骤
    conclusion: str = ""                                            # 结论
    language_caption: str = ""                                      # 自然语言摘要
    decision: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 决策向量

    def to_dict(self) -> dict:
        return {
            "auxiliary_hypotheses": list(self.auxiliary_hypotheses),
            "derivation_steps": list(self.derivation_steps),
            "conclusion": self.conclusion,
            "language_caption": self.language_caption,
            "decision": np.asarray(self.decision).tolist(),
        }


class SpatialReasoner:
    """V2L 空间论证器 — 视觉特征 + 目标 → 可解释论证 + 决策。

    想象空间思考的最小实现（规则式几何, 类比立体几何:
    作辅助线(假设) → 演算(点积/夹角) → 得结论 → 出决策)。
    """

    #: 方位判定阈值 (cos 角度, assumed — 规则参数)
    ALIGN_THRESHOLD = 0.5

    def argue(self, vision_feats: np.ndarray,
              goal_vec: np.ndarray) -> SpatialArgument:
        """构造空间论证。

        Args:
            vision_feats: 视觉特征 (展平后按视角分段, 每段代表一个观测方向)
            goal_vec: 目标方向向量 (3D)
        Returns:
            SpatialArgument: 辅助线 + 推演 + 结论 + 决策
        """
        v = np.asarray(vision_feats, dtype=np.float64).ravel()
        g = np.asarray(goal_vec, dtype=np.float64).ravel()
        if v.size == 0:
            raise ValueError("vision_feats 不能为空")
        if g.size != 3:
            raise ValueError("goal_vec 必须是 3 维")
        if not np.any(v):
            raise ValueError("vision_feats 全零 — 无视觉证据, 拒绝构造假论证")

        # ── 辅助线（假设）: 各视角主方向 + 目标方位 ──
        n_seg = 3
        seg_len = v.size // n_seg if v.size >= n_seg else 1
        hypotheses: List[str] = []
        seg_dirs = []
        for i in range(n_seg):
            seg = v[i * seg_len:(i + 1) * seg_len] if seg_len > 1 else v
            d = np.zeros(3)
            d[:min(3, seg.size)] = seg[:3]
            n = np.linalg.norm(d)
            d = d / n if n > 0 else np.zeros(3)
            seg_dirs.append(d)
            hypotheses.append(f"辅助线{i + 1}: 视角{i + 1}主方向 "
                              f"({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f})")
        g_norm = float(np.linalg.norm(g))
        g_dir = g / g_norm if g_norm > 0 else np.zeros(3)
        hypotheses.append(f"辅助线4: 目标方位 ({g_dir[0]:+.2f},{g_dir[1]:+.2f},"
                          f"{g_dir[2]:+.2f}), 距离 {g_norm:.2f}")

        # ── 演算（推演）: 逐视角与目标夹角 ──
        steps: List[str] = []
        aligns = []
        for i, d in enumerate(seg_dirs):
            cos = float(np.dot(d, g_dir))
            aligns.append(cos)
            steps.append(f"演算{i + 1}: cos(视角{i + 1}, 目标) = {cos:+.3f}")
        best = int(np.argmax(aligns)) if aligns else 0
        best_cos = aligns[best] if aligns else 0.0
        steps.append(f"演算{len(steps) + 1}: 最优视角 = {best + 1}, "
                     f"cos = {best_cos:+.3f} (阈值 {self.ALIGN_THRESHOLD:+.2f})")

        # ── 结论 + 决策 ──
        if best_cos >= self.ALIGN_THRESHOLD:
            conclusion = f"目标已对齐 (最优视角{best + 1}), 朝目标方向推进"
            decision = g_dir.copy()
        elif best_cos >= 0.0:
            conclusion = f"目标偏角中等 (cos={best_cos:+.2f}), 先转向再推进"
            decision = 0.5 * g_dir + 0.5 * seg_dirs[best]
        else:
            conclusion = f"目标在对侧 (cos={best_cos:+.2f}), 原地转向目标"
            decision = g_dir - seg_dirs[best]
        n_dec = float(np.linalg.norm(decision))
        if n_dec > 0:
            decision = decision / n_dec
        caption = (f"空间论证: {len(hypotheses)} 条辅助线 → {len(steps)} 步演算 → "
                   f"{conclusion}")
        return SpatialArgument(auxiliary_hypotheses=hypotheses,
                               derivation_steps=steps, conclusion=conclusion,
                               language_caption=caption, decision=decision)

    def to_event(self, argument: SpatialArgument, timestamp_s: float,
                 location_xyz: Optional[np.ndarray] = None,
                 actor: str = "robot") -> "EventRecord":
        """论证 → 四要素事件 (source="v2l"), 喂认知中枢。"""
        from ..cognition.cognitive_hub import EventRecord
        loc = (np.zeros(3) if location_xyz is None
               else np.asarray(location_xyz, dtype=np.float64).ravel())
        return EventRecord(
            timestamp_s=timestamp_s, location_xyz=loc, actor=actor,
            thing="spatial_reasoning", action=argument.conclusion,
            source="v2l",
        )
