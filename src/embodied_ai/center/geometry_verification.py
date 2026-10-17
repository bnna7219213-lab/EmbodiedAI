"""V2L 空间几何演算验证 — 独立复算 + 真值场景 + 变异负对照 (plan_v6 后续: d2)。

**缺口背景（可解释性验证断点）**
    spatial_reasoning.py 的 SpatialReasoner 输出"辅助线→演算→结论→决策",
    但论证的**数学正确性**没有独立验证 —— 推演步骤是自己写的, 自己算自己证
    (自证当验收)。本模块把空间几何演算**独立复算**:
    ① 逐步复算: 从论证文本提取的几何量, 用**独立公式**重算并逐步核对
    ② 真值场景: 已知解析解的几何场景（对齐/正交/反向）→ 结论必须正确
    ③ 变异负对照: 对论证做单步篡改 → 验证器必须检出 (否则验证是摆设)

**验证协议（docs/19 §7, 本模块是其代码化）**
    - cos(视角, 目标) 独立复算: 从 hypothesis 向量与 goal_vec 重算, 与
      derivation_steps 中报告值比对 (容差 1e-6)
    - decision 归一化 + 与结论方向一致性复算
    - 真值场景判据: 对齐(cos=1)→推进 / 正交(cos=0)→转向 / 反向(cos=-1)→掉头
    - 负对照: 篡改 derivation 里的 cos 数值 → verifier 必须报 mismatch

**provenance（诚实边界）**
    复算公式与 SpatialReasoner 同源同义（都是解析几何, 不是学习式）——
    独立性来自**重算与比对**（不共享中间量）, 不是来自不同算法。

**不做的（诚实边界）**
    - 不做证明论/形式化几何证明（数值复算足够守护本模块的正确性）
    - 不做三维重建验证（argue 输入是特征向量, 场景用解析合成）

**反证要求（测试守护）**
    真值场景结论全对; 变异单步必检出; 复算容差内通过;
    全零/坏输入报错; 与 spatial_reasoning 输出格式兼容。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .spatial_reasoning import SpatialArgument, SpatialReasoner


# ══════════════════════════════════════════════════════════════════
# 复算结果
# ══════════════════════════════════════════════════════════════════

@dataclass
class StepCheck:
    """单步演算复算结果。"""
    step_index: int
    reported: float                    # 论证里报告的 cos 值
    recomputed: float                  # 独立复算值
    ok: bool

    @property
    def error(self) -> float:
        return abs(self.reported - self.recomputed)


@dataclass
class VerificationReport:
    """空间几何演算验证报告。"""
    step_checks: List[StepCheck] = field(default_factory=list)
    conclusion_ok: bool = False
    decision_normalized: bool = False
    truth_case: str = ""
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "n_steps_checked": len(self.step_checks),
            "max_step_error": max((c.error for c in self.step_checks),
                                  default=0.0),
            "truth_case": self.truth_case,
        }


# ══════════════════════════════════════════════════════════════════
# GeometryVerifier — 独立复算验证器
# ══════════════════════════════════════════════════════════════════

class GeometryVerifier:
    """空间几何演算独立复算验证器。

    逐步复算 SpatialArgument.derivation_steps 中的 cos 值, 与独立公式
    结果比对; 校验 conclusion 与 decision 的方向一致性。
    """

    #: 复算容差 — 必须匹配论证**报告精度**: derivation_steps 里 cos 用
    #: `:.3f` 写出 (3 位小数), 故容差取半最小数位 5e-4。取 1e-6 会让一切
    #: 合法论证"复算失败"(曾经如此) —— 验证容差必须锚定报告精度。
    #: 变异负对照 delta=0.25 远大于此, 检出不受影响。
    TOL = 5e-4

    def verify(self, argument: SpatialArgument,
               vision_feats: np.ndarray,
               goal_vec: np.ndarray) -> VerificationReport:
        """独立复算 + 一致性校验 → VerificationReport。

        Raises:
            ValueError: 空推演步骤 / 坏输入。
        """
        v = np.asarray(vision_feats, dtype=np.float64).ravel()
        g = np.asarray(goal_vec, dtype=np.float64).ravel()
        if v.size == 0 or g.size != 3:
            raise ValueError("vision_feats 为空或 goal_vec 非 3 维")
        if not argument.derivation_steps:
            raise ValueError("derivation_steps 为空 — 无可复算的演算")
        report = VerificationReport()

        # ── 逐步复算: 重算每视角 cos, 与论证报告值比对 ──
        g_norm = float(np.linalg.norm(g))
        if g_norm == 0:
            raise ValueError("goal_vec 为零向量 — 无方向可复算")
        g_dir = g / g_norm
        n_seg = 3
        seg_len = v.size // n_seg if v.size >= n_seg else 1
        for i in range(n_seg):
            seg = v[i * seg_len:(i + 1) * seg_len] if seg_len > 1 else v
            d = np.zeros(3)
            d[:min(3, seg.size)] = seg[:3]
            nd = np.linalg.norm(d)
            d = d / nd if nd > 0 else np.zeros(3)
            recomputed = float(np.dot(d, g_dir))
            reported = self._extract_cos(argument.derivation_steps, i)
            if reported is None:
                report.failures.append(f"步骤{i + 1}: 无法从论证提取 cos 值")
                continue
            check = StepCheck(step_index=i, reported=reported,
                              recomputed=recomputed,
                              ok=abs(reported - recomputed) <= self.TOL)
            report.step_checks.append(check)
            if not check.ok:
                report.failures.append(
                    f"步骤{i + 1}: 报告 cos={reported:+.6f} ≠ 复算 "
                    f"{recomputed:+.6f} (误差 {check.error:.2e})")

        # ── decision 归一化 + 结论方向一致性 ──
        dec = np.asarray(argument.decision, dtype=np.float64).ravel()
        n_dec = float(np.linalg.norm(dec))
        report.decision_normalized = bool(abs(n_dec - 1.0) <= 1e-6) or \
            n_dec == 0.0
        if not report.decision_normalized:
            report.failures.append(f"decision 未归一化 |d|={n_dec:.6f}")

        # ── 结论语义与最优 cos 一致 ──
        # 判据: cos≥阈值 → 结论必须含"推进/对齐"; 否则必须含"转向"
        best_cos = max((c.recomputed for c in report.step_checks),
                       default=0.0)
        concl = argument.conclusion
        if best_cos >= SpatialReasoner.ALIGN_THRESHOLD:
            report.conclusion_ok = ("推进" in concl) or ("对齐" in concl)
            expected = "推进/对齐"
        else:
            report.conclusion_ok = "转向" in concl
            expected = "转向"
        if not report.conclusion_ok:
            report.failures.append(
                f"结论『{concl}』与最优 cos={best_cos:+.3f} 不一致 "
                f"(应含『{expected}』)")
        return report

    @staticmethod
    def _extract_cos(steps: List[str], seg_index: int) -> Optional[float]:
        """从 '演算N: cos(视角M, 目标) = +0.123' 提取报告值。"""
        prefix = f"演算{seg_index + 1}:"
        for s in steps:
            if s.startswith(prefix) and "cos" in s:
                try:
                    return float(s.split("=")[1].strip())
                except (IndexError, ValueError):
                    return None
        return None

    # ---------------- 变异负对照 ---------------- #

    @staticmethod
    def mutate(argument: SpatialArgument, step_index: int,
               delta: float = 0.1) -> SpatialArgument:
        """篡改某步 cos 数值 — 用于负对照 (验证器必须检出)。"""
        steps = list(argument.derivation_steps)
        target = f"演算{step_index + 1}:"
        for i, s in enumerate(steps):
            if s.startswith(target) and "cos" in s:
                val = float(s.split("=")[1].strip())
                steps[i] = s.split("=")[0] + f"= {val + delta:+.3f}"
                break
        return SpatialArgument(
            auxiliary_hypotheses=list(argument.auxiliary_hypotheses),
            derivation_steps=steps, conclusion=argument.conclusion,
            language_caption=argument.language_caption,
            decision=argument.decision.copy())


# ══════════════════════════════════════════════════════════════════
# 真值场景 + demo
# ══════════════════════════════════════════════════════════════════

#: 真值场景: (名称, 视觉三段方向, 目标方向, 期望结论关键字)
#: 注意分段口径: SpatialReasoner 把特征均分 3 段, 每段取前 3 维为方向 ——
#: 12 维特征按 4 元素/段切, 故用 tile([x,y,z,0], 3) 保证每段方向一致
#: (曾经用 [1,0,0]*4 导致段内错位, 场景失真)。
TRUTH_CASES = (
    ("aligned", np.tile([1.0, 0.0, 0.0, 0.0], 3),
     np.array([1.0, 0.0, 0.0]), "推进"),
    ("orthogonal", np.tile([1.0, 0.0, 0.0, 0.0], 3),
     np.array([0.0, 1.0, 0.0]), "转向"),
    ("opposite", np.tile([-1.0, 0.0, 0.0, 0.0], 3),
     np.array([1.0, 0.0, 0.0]), "转向"),
)


def geometry_verification_demo() -> dict:
    """验证演示: 真值场景 + 独立复算 + 变异负对照。

    全部布尔结论为实测计算值, verify 断言即这些值。
    """
    reasoner = SpatialReasoner()
    verifier = GeometryVerifier()
    truth_results = {}
    for name, feats, goal, expect_kw in TRUTH_CASES:
        arg = reasoner.argue(feats, goal)
        rep = verifier.verify(arg, feats, goal)
        conclusion_right = expect_kw in arg.conclusion
        truth_results[name] = {
            "conclusion": arg.conclusion,
            "expected_kw": expect_kw,
            "verify_passed": rep.passed,
            "conclusion_right": bool(conclusion_right),
        }

    # 变异负对照: 篡改步骤 0 的 cos → 验证器必须检出
    rng = np.random.default_rng(8)
    feats = rng.normal(size=12)
    goal = np.array([1.0, 0.5, 0.0])
    arg = reasoner.argue(feats, goal)
    clean_pass = verifier.verify(arg, feats, goal).passed
    mutated = GeometryVerifier.mutate(arg, step_index=0, delta=0.25)
    mutated_report = verifier.verify(mutated, feats, goal)
    mutation_detected = not mutated_report.passed

    result = {
        "truth_cases": truth_results,
        "clean_pass": clean_pass,
        "mutation_detected": mutation_detected,
        "all_truth_conclusions_right": all(
            t["conclusion_right"] for t in truth_results.values()),
        "all_truth_verify_pass": all(
            t["verify_passed"] for t in truth_results.values()),
    }
    assert result["all_truth_verify_pass"], f"真值场景复算失败: {truth_results}"
    assert result["all_truth_conclusions_right"], \
        f"真值场景结论错: {truth_results}"
    assert clean_pass, "未篡改论证应通过验证"
    assert mutation_detected, "变异论证必须被检出 (否则验证是摆设)"
    return result


if __name__ == "__main__":
    r = geometry_verification_demo()
    print("V2L 空间几何演算验证 demo (docs/19 §7)")
    for name, t in r["truth_cases"].items():
        print(f"  {name:12s} 结论『{t['conclusion']}』 期望含『"
              f"{t['expected_kw']}』 复算通过={t['verify_passed']} ✓")
    print(f"变异负对照检出: {r['mutation_detected']} ✓")
