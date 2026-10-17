"""N5 V2L 空间智能 测试 (plan_v6 Sprint N)。

反证: 推演步骤非空; conclusion 与 decision 一致; 输入变则结论/决策必变
(防恒定假论证); 全零视觉拒绝; to_event 四要素完整。
"""
import numpy as np
import pytest

from embodied_ai.center.spatial_reasoning import SpatialArgument, SpatialReasoner


def _feats(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=12)


class TestSpatialReasoner:
    def test_argument_structure(self):
        r = SpatialReasoner()
        a = r.argue(_feats(), np.array([1.0, 0.0, 0.0]))
        assert isinstance(a, SpatialArgument)
        assert len(a.auxiliary_hypotheses) >= 3, "辅助线假设不足"
        assert len(a.derivation_steps) >= 3, "演算步骤不足"
        assert a.conclusion, "结论不能为空"
        assert a.language_caption.startswith("空间论证")
        assert a.decision.shape == (3,)

    def test_decision_normalized(self):
        r = SpatialReasoner()
        a = r.argue(_feats(1), np.array([0.0, 2.0, 0.0]))
        assert np.linalg.norm(a.decision) == pytest.approx(1.0, abs=1e-6)

    def test_aligned_case_pushes_forward(self):
        """视觉主方向与目标一致 → 决策应朝目标方向 (推进)。"""
        r = SpatialReasoner()
        # 三段特征全部指向 +x, 目标也 +x
        feats = np.array([1.0, 0, 0] * 4)
        a = r.argue(feats, np.array([1.0, 0.0, 0.0]))
        assert "推进" in a.conclusion or "对齐" in a.conclusion
        assert a.decision[0] > 0.9, f"决策应朝 +x, 得 {a.decision}"

    def test_input_change_changes_output(self):
        """反证: 输入变则论证/决策必变 (防恒定假论证)。"""
        r = SpatialReasoner()
        a1 = r.argue(_feats(0), np.array([1.0, 0.0, 0.0]))
        a2 = r.argue(_feats(99), np.array([0.0, 1.0, 0.0]))
        assert a1.conclusion != a2.conclusion or \
            not np.allclose(a1.decision, a2.decision)

    def test_zero_vision_rejected(self):
        with pytest.raises(ValueError):
            SpatialReasoner().argue(np.zeros(12), np.array([1.0, 0.0, 0.0]))

    def test_bad_goal_dim_raises(self):
        with pytest.raises(ValueError):
            SpatialReasoner().argue(_feats(), np.array([1.0, 0.0]))

    def test_empty_vision_raises(self):
        with pytest.raises(ValueError):
            SpatialReasoner().argue(np.array([]), np.array([1.0, 0.0, 0.0]))

    def test_to_event_four_elements(self):
        r = SpatialReasoner()
        a = r.argue(_feats(), np.array([1.0, 1.0, 0.0]))
        e = r.to_event(a, timestamp_s=3.0,
                       location_xyz=np.array([1.0, 2.0, 0.0]))
        t, loc, actor, thing = e.four_elements
        assert t == 3.0
        assert loc == pytest.approx([1.0, 2.0, 0.0])
        assert actor == "robot"
        assert thing == "spatial_reasoning"
        assert e.source == "v2l"

    def test_to_dict_serializable(self):
        a = SpatialReasoner().argue(_feats(), np.array([1.0, 0.0, 0.0]))
        d = a.to_dict()
        assert set(d) == {"auxiliary_hypotheses", "derivation_steps",
                          "conclusion", "language_caption", "decision"}
        assert isinstance(d["decision"], list)
