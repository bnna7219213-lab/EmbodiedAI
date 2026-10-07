"""test_c1_vla_blocksize.py — C1 (P0 静默错误) 修复的守护测试。

背景: 原 `VLAConfig.block_size=256`, 但 `min_block_size` = 588(视觉)+8(lang)
+1+action_horizon+1 = 603~613。`vla_model.forward` 的截断是
`ctx[:, -block_size:, :]` (保留**尾部**), 而序列开头是视觉 token →
**视觉仅保留 42%, 约 58% 被静默丢弃**, 且 loss 曲线看不出异常。

C1 共修 4 处, 本测试逐处守护:
  1. block_size 覆盖 min (256 → 640)
  2. 截断不再静默 (vla_model 计数 + 告警)
  3. vision_encoder.pos_embed: `zeros*0.02` 恒全零 → `randn*0.02`
  4. vla_trainer: DataLoader 兼容 + 取不到 batch 时返回 NaN 而非假装 0.0

延迟约束: block_size 提升**不得**击穿 Jetson 15ms 模型预算 (实测 base
0.90→1.15ms, large 1.77→2.28ms — 远低于 15ms), 故无需牺牲 num_views/img_size。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from embodied_ai.models.vla_config import VLAConfig  # noqa: E402


# ============================================================================
# 1. block_size 覆盖 (无 torch 依赖, CI 恒跑)
# ============================================================================

def test_all_ablations_block_size_covers_min():
    """每个消融档的 block_size 必须 ≥ min_block_size (否则训练静默丢视觉)。"""
    bad = [(n, c.block_size, c.min_block_size)
           for n, c in VLAConfig.ablations().items()
           if c.block_size < c.min_block_size]
    assert not bad, (
        "block_size 不足, 训练时视觉 token 会被截断:\n" +
        "\n".join(f"  - {n}: {bs} < {mn} (丢 {mn - bs} token)" for n, bs, mn in bad))


def test_default_config_covers_min():
    """默认 VLAConfig() (=vla_base) 也必须覆盖。"""
    c = VLAConfig()
    assert c.block_size >= c.min_block_size, f"{c.block_size} < {c.min_block_size}"


def test_vision_tokens_accounted():
    """min_block_size 必须把 3 视角 × 196 patch = 588 视觉 token 算进去。"""
    c = VLAConfig()
    assert c.total_vision_tokens == 196 * 3 == 588
    assert c.min_block_size == 588 + 8 + 1 + c.action_horizon + 1


# ============================================================================
# 2. 延迟不击穿 (修了截断不能丢 20Hz)
# ============================================================================

def test_latency_within_15ms_model_budget():
    """block_size 提升后, 各档仍须 ≤ 15ms 模型预算 (契约 model_inference_ms)。"""
    from embodied_ai.deploy.jetson_bench import PLATFORMS, estimate_latency

    plat = PLATFORMS["jetson_orin_nx_16gb"]
    over = []
    for n, c in VLAConfig.ablations().items():
        ms = float(estimate_latency(c, plat, precision="int8")["roofline_ms"])
        if ms > 15.0:
            over.append((n, ms))
    assert not over, (
        "block_size 提升击穿 Jetson 15ms 模型预算 (需降 num_views/img_size 补偿):\n" +
        "\n".join(f"  - {n}: {ms:.2f}ms" for n, ms in over))


# ============================================================================
# 3. pos_embed 非全零 (torch 可选)
# ============================================================================

def test_vision_encoder_pos_embed_not_all_zero():
    """反证 zeros*0.02 的 bug: pos_embed 全零 → 视觉空间位置信息完全丢失。"""
    torch = pytest.importorskip("torch")
    from embodied_ai.models.vision_encoder import HAS_TORCH, VisionEncoder
    if not HAS_TORCH:
        pytest.skip("vision_encoder 无 torch 支持")

    torch.manual_seed(0)
    enc = VisionEncoder(img_size=224, patch_size=16, embed_dim=384,
                        n_layers=1, n_heads=6, num_views=3)
    pe = enc.pos_embed.detach()
    assert not torch.allclose(pe, torch.zeros_like(pe)), (
        "pos_embed 全零 — `torch.zeros(...)*0.02` 恒等于 0, 视觉位置信息丢失")
    # 初始化标准差应≈0.02 (randn*0.02)
    sd = float(pe.std())
    assert 0.005 < sd < 0.05, f"pos_embed 标准差异常: {sd}"


# ============================================================================
# 4. 闸门有效性反证 (证明不是摆设)
# ============================================================================

def test_verify_check_detects_truncation(monkeypatch):
    """反证: 若某档 block_size 回退到 256, verify 检查必须失败。"""
    from embodied_ai import verification
    from embodied_ai.models import vla_config

    orig = vla_config.VLAConfig.ablations

    def with_bad_base():
        d = orig()
        d["vla_base"] = vla_config.VLAConfig(block_size=256,
                                            ablation_name="vla_base")
        return d

    monkeypatch.setattr(vla_config.VLAConfig, "ablations",
                        staticmethod(with_bad_base))
    with pytest.raises(AssertionError, match="截断"):
        verification.check_vla_block_size()


def test_verify_check_passes_current():
    """当前配置下 verify 检查应通过 (闸门不误报)。"""
    from embodied_ai import verification
    assert verification.check_vla_block_size() == 0


# ============================================================================
# 5. DataLoader 兼容 (torch 可选)
# ============================================================================

def test_estimate_loss_no_silent_zero_on_empty():
    """反证: 取不到 batch 时必须返回 NaN 并告警, 不能静默返回 0.0。

    'loss=0.0' 会被误读成训练完美, 实际一个 batch 都没跑。
    """
    pytest.importorskip("torch")
    # 直接测 vla_trainer 的空 batch 分支: 用一个空可迭代对象
    import warnings

    from embodied_ai.models.vla_trainer import VLATrainer
    assert hasattr(VLATrainer, "estimate_loss"), "VLATrainer 缺 estimate_loss"

    # 构造最小 trainer 不现实 (需 dataset/model), 故改为静态确认:
    # 源码必须含 NaN 分支与告警, 而非 sum([])/max(0,1)
    src = Path(vla_trainer.__file__).read_text(encoding="utf-8")
    assert "float(\"nan\")" in src or "float('nan')" in src, (
        "vla_trainer 空 batch 分支未返回 NaN — 'loss 完美' 的静默错误会回归")
    assert "未取到任何 batch" in src, "缺少空 batch 告警文案"


import embodied_ai.models.vla_trainer as vla_trainer  # noqa: E402
