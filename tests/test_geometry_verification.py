"""d2 V2L 空间几何演算验证 测试 (plan_v6 后续)。

反证: 真值场景结论全对; 独立复算逐步核对; 变异单步必检出 (否则验证是摆设);
坏输入报错; 与 spatial_reasoning 输出格式兼容。
"""
import numpy as np
import pytest

from embodied_ai.center.geometry_verification import (
    TRUTH_CASES,
    GeometryVerifier,
    StepCheck,
    VerificationReport,
    geometry_verification_demo,
)
from embodied_ai.center.spatial_reasoning import SpatialReasoner


def _random_case(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=12), np.array([1.0, 0.5, 0.0])


class TestIndependentRecompute:
    def test_clean_argument_passes(self):
        feats, goal = _random_case(1)
        arg = SpatialReasoner().argue(feats, goal)
        rep = GeometryVerifier().verify(arg, feats, goal)
        assert rep.passed, rep.failures

    def test_step_checks_recomputed(self):
        feats, goal = _random_case(2)
        arg = SpatialReasoner().argue(feats, goal)
        rep = GeometryVerifier().verify(arg, feats, goal)
        assert len(rep.step_checks) == 3
        assert all(c.ok for c in rep.step_checks)
        # 报告精度 3 位小数 → 误差 < 半最小数位
        assert rep.summary()["max_step_error"] < GeometryVerifier.TOL

    def test_report_error_property(self):
        c = StepCheck(step_index=0, reported=0.5, recomputed=0.4, ok=False)
        assert c.error == pytest.approx(0.1)


class TestTruthCases:
    @pytest.mark.parametrize("name,feats,goal,expect_kw", TRUTH_CASES)
    def test_truth_case_conclusion(self, name, feats, goal, expect_kw):
        arg = SpatialReasoner().argue(feats, goal)
        assert expect_kw in arg.conclusion, \
            f"{name}: 结论『{arg.conclusion}』应含『{expect_kw}』"

    @pytest.mark.parametrize("name,feats,goal,expect_kw", TRUTH_CASES)
    def test_truth_case_verifies(self, name, feats, goal, expect_kw):
        arg = SpatialReasoner().argue(feats, goal)
        rep = GeometryVerifier().verify(arg, feats, goal)
        assert rep.passed, f"{name}: {rep.failures}"


class TestMutationNegativeControl:
    def test_mutation_detected(self):
        """变异负对照: 篡改单步 cos → 验证器必须检出 (否则验证是摆设)。"""
        feats, goal = _random_case(3)
        arg = SpatialReasoner().argue(feats, goal)
        mutated = GeometryVerifier.mutate(arg, step_index=0, delta=0.25)
        rep = GeometryVerifier().verify(mutated, feats, goal)
        assert not rep.passed, "变异论证必须被检出"
        assert not rep.step_checks[0].ok, "篡改的步骤必须复算不通过"

    def test_mutation_step1(self):
        feats, goal = _random_case(4)
        arg = SpatialReasoner().argue(feats, goal)
        mutated = GeometryVerifier.mutate(arg, step_index=1, delta=-0.3)
        assert not GeometryVerifier().verify(mutated, feats, goal).passed

    def test_mutation_only_touches_target_step(self):
        feats, goal = _random_case(5)
        arg = SpatialReasoner().argue(feats, goal)
        mutated = GeometryVerifier.mutate(arg, step_index=0, delta=0.1)
        assert mutated.derivation_steps[0] != arg.derivation_steps[0]
        assert mutated.derivation_steps[1] == arg.derivation_steps[1]
        assert mutated.conclusion == arg.conclusion


class TestBadInputs:
    def test_empty_steps_raises(self):
        from embodied_ai.center.spatial_reasoning import SpatialArgument
        arg = SpatialArgument()
        with pytest.raises(ValueError):
            GeometryVerifier().verify(arg, np.ones(12),
                                      np.array([1.0, 0.0, 0.0]))

    def test_bad_goal_dim_raises(self):
        feats, _ = _random_case()
        arg = SpatialReasoner().argue(feats, np.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError):
            GeometryVerifier().verify(arg, feats, np.array([1.0, 0.0]))

    def test_zero_goal_raises(self):
        feats, _ = _random_case()
        arg = SpatialReasoner().argue(feats, np.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError):
            GeometryVerifier().verify(arg, feats, np.zeros(3))

    def test_empty_vision_raises(self):
        with pytest.raises(ValueError):
            GeometryVerifier().verify(SpatialReasoner().argue(
                np.ones(12), np.array([1.0, 0.0, 0.0])),
                np.array([]), np.array([1.0, 0.0, 0.0]))


class TestDemo:
    def test_demo_assertions(self):
        r = geometry_verification_demo()
        assert r["all_truth_verify_pass"] is True
        assert r["all_truth_conclusions_right"] is True
        assert r["clean_pass"] is True
        assert r["mutation_detected"] is True
