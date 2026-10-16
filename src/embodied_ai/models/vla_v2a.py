"""V-to-action — 纯视觉决策路径 (plan_v5 Sprint M: M3)。

**缺口背景（机器人端三大缺口之一）**
    VLA = vision + language → action；但**纯视觉也能构成判断依据**——
    v→action（V-to-action）与 v+l→action 是等价的两条决策路径。
    项目已有 VLAModel 的 `lang_ids=None` 无语言前向路径（vla_model.py），
    但缺**显式配置模式 + 无 torch 的推理封装 + 等价性守护**。

**设计（最小可用，不过度设计）**
    - `VLAConfig.modality`: "vla" (vision+language) / "v2a" (vision-only)
      —— 已在 vla_config.py 落地, v2a 时 min_block_size 不含语言 token。
    - `VLAPolicy`: numpy 推理封装（不依赖 torch）:
        · `infer_vla(vision_feats, lang_ids, state)` — 双模态路径
        · `infer_v2a(vision_feats, state)` — 纯视觉路径（≡ lang_ids=None）
      确定性线性头（seed 来自 cfg.seed, 可复现）。训练权重由
      vla_model/vla_trainer 产出后可替换 head（接口对齐 action 维度）。
    - 等价性守护: `infer_v2a(v, s) == infer_vla(v, None, s)`（测试断言）。

**provenance（诚实边界）**
    当前 head 权重为 seeded synthetic（随机投影确定性初始化），
    **不是**训练产物——真实决策精度依赖 vla_trainer 训练检查点。
    本模块守护的是**路径等价性与形状契约**, 不是策略性能。

**不做的（诚实边界）**
    - 不重训/不复制 VLAModel 的 torch 前向（训练见 vla_trainer.py）
    - 不做多模态缺失时的自动补全（v2a 就是显式无语言, 不猜语言）

**反证要求（测试守护）**
    v2a ≡ vla(None) 数值一致；modality="v2a" 时 min_block_size 少 LANG_TOKENS；
    无 torch 环境可跑；坏输入形状报错。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .vla_config import VLAConfig


class VLAPolicy:
    """VLA/V2A 推理封装 — 确定性线性头, 零重依赖可跑。

    forward 等价关系（守护核心）:
        infer_v2a(vision, state)
          == infer_vla(vision, lang_ids=None, state)      # 纯视觉
        infer_vla(vision, lang_ids, state)                # 视觉+语言
    """

    def __init__(self, cfg: Optional[VLAConfig] = None):
        self.cfg = cfg if cfg is not None else VLAConfig()
        if self.cfg.modality not in ("vla", "v2a"):
            raise ValueError(f"modality 必须是 'vla'/'v2a', 得 '{self.cfg.modality}'")
        rng = np.random.default_rng(self.cfg.seed)
        # 特征 = [视觉特征(展平) ; 语言特征(标量池化) ; 状态] → action_horizon × action_dim
        self._out_dim = self.cfg.action_horizon * self.cfg.action_dim
        self._vision_dim = self.cfg.num_views * self.cfg.vision_embd
        self._state_dim = self.cfg.state_dim
        in_dim = self._vision_dim + 1 + self._state_dim   # +1 = 语言标量池化
        scale = 1.0 / np.sqrt(in_dim)
        self.W = rng.normal(0.0, scale, size=(in_dim, self._out_dim))
        self.b = rng.normal(0.0, scale, size=self._out_dim)

    # ---------------- 输入规整 ---------------- #

    def _vision_flat(self, vision_feats: np.ndarray) -> np.ndarray:
        v = np.asarray(vision_feats, dtype=np.float64).ravel()
        if v.size != self._vision_dim:
            raise ValueError(
                f"vision_feats 展平后应为 {self._vision_dim} 维 "
                f"(num_views={self.cfg.num_views} × vision_embd={self.cfg.vision_embd}), "
                f"得 {v.size} 维")
        return v

    def _state_flat(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).ravel()
        if s.size != self._state_dim:
            raise ValueError(f"state 应为 {self._state_dim} 维, 得 {s.size} 维")
        return s

    @staticmethod
    def _lang_scalar(lang_ids: Optional[np.ndarray]) -> float:
        if lang_ids is None:
            return 0.0
        ids = np.asarray(lang_ids, dtype=np.float64).ravel()
        return float(ids.mean()) if ids.size else 0.0

    # ---------------- 推理 ---------------- #

    def infer_vla(self, vision_feats: np.ndarray,
                  lang_ids: Optional[np.ndarray],
                  state: np.ndarray) -> np.ndarray:
        """vision(+language) → action: 返回 [action_horizon, action_dim]。

        lang_ids=None 时退化为纯视觉路径（与 infer_v2a 等价）。
        """
        v = self._vision_flat(vision_feats)
        s = self._state_flat(state)
        x = np.concatenate([v, [self._lang_scalar(lang_ids)], s])
        out = x @ self.W + self.b
        return out.reshape(self.cfg.action_horizon, self.cfg.action_dim)

    def infer_v2a(self, vision_feats: np.ndarray,
                  state: np.ndarray) -> np.ndarray:
        """vision-only → action（纯视觉判断依据, 无语言通道）。"""
        return self.infer_vla(vision_feats, None, state)


# ══════════════════════════════════════════════════════════════════
# demo — 双路径等价性
# ══════════════════════════════════════════════════════════════════

def v2a_demo() -> dict:
    """演示 v→action 与 v+l→action 双路径: 等价性 + 形状契约。"""
    cfg = VLAConfig(modality="v2a")
    policy = VLAPolicy(cfg)
    rng = np.random.default_rng(0)
    vision = rng.normal(size=(cfg.num_views, cfg.vision_embd))
    state = rng.normal(size=cfg.state_dim)

    a_v2a = policy.infer_v2a(vision, state)
    a_vla_none = policy.infer_vla(vision, None, state)
    lang = rng.normal(size=cfg.LANG_TOKENS)
    a_vla = policy.infer_vla(vision, lang, state)

    assert a_v2a.shape == (cfg.action_horizon, cfg.action_dim)
    assert np.allclose(a_v2a, a_vla_none), "v2a 与 vla(None) 不等价"
    assert not np.allclose(a_v2a, a_vla), "语言输入应改变输出（否则语言通道未生效）"
    assert cfg.min_block_size == (cfg.total_vision_tokens + 1 + cfg.action_horizon + 1)
    return {
        "modality": cfg.modality,
        "action_shape": list(a_v2a.shape),
        "v2a_equals_vla_none": True,
        "lang_changes_output": True,
        "min_block_size_v2a": cfg.min_block_size,
        "min_block_size_vla": VLAConfig(modality="vla").min_block_size,
    }


if __name__ == "__main__":
    r = v2a_demo()
    print("V-to-action demo (head 权重 provenance=synthetic)")
    print(f"modality={r['modality']}, action shape={r['action_shape']}")
    print(f"v2a ≡ vla(lang=None): {r['v2a_equals_vla_none']} ✓")
    print(f"语言改变输出: {r['lang_changes_output']} ✓")
    print(f"min_block_size: v2a={r['min_block_size_v2a']} < vla={r['min_block_size_vla']} "
          f"(省 {r['min_block_size_vla'] - r['min_block_size_v2a']} 语言 token)")
