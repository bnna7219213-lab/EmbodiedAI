"""机会约束握力调度 (横切安全层 S) — 拆自 climbing_dynamics_core.py (K1 拆分)。

公式: f_n >= |f_t| / (mu_hat - z_{1-delta} * sigma_mu)
其中 z_{1-delta} = Φ⁻¹(1-δ) 是标准正态分布的 (1-δ) 分位数。

R7 修复 (2026-10-07):
  1. numpy 补入导入 (原实现 `import numpy as np` 缺失, grip_demo 运行时必 NameError)
  2. z 值改用 statistics.NormalDist().inv_cdf(1-δ) — 原 `sqrt(2)*(1-2δ)` 线性近似
     只在 δ=0.05 时恰好 1.414 ≈ 正确 1.645; δ=0.001 时得 1.408 而正确值是 3.090
     (差 1.2x, 置信分位数全错, 直接导致握力计算系统性失真)
  3. `denom <= 0.05` 时抛 InfeasibleGripError — 原 `max(denom, 0.05)` 掩盖了不可行
     情形 (σ_μ 过大或 μ̂ 过小), 静默返回过低的 f_n 而实际物理上无解
  4. 显式暴露 InfeasibleGripError 类, 供上游 (climb_scene / wall_contact) 捕获
"""

from __future__ import annotations

from statistics import NormalDist


class InfeasibleGripError(ValueError):
    """机会约束无解: μ̂ - z_{1-δ} σ_μ <= min_margin。

    物理含义: 摩擦不确定度过大 (σ_μ 大或 δ 极小), 即使用全部可用握力也无法
    保证滑脱概率 < δ。调用方应:
      - 提高 δ (放宽约束, 允许更多滑脱风险)
      - 降低切向力 f_t (减小任务需求)
      - 更换摩擦更高的接触面 (换 hold / 增加胶带)
      - 触发安全停止 (攀岩场景: 立即释放握力, 由其他肢体承接)
    """


# 用 statistics 模块 (Python 3.8+ 内置) 的正态分布, 精确 inverse CDF
_NORMAL = NormalDist()


def _z_1_minus_delta(delta: float) -> float:
    """标准正态 Φ⁻¹(1-δ): δ=0.05 → 1.6449; δ=0.001 → 3.0902; δ=0.0001 → 3.891."""
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0,1), got {delta}")
    return _NORMAL.inv_cdf(1.0 - delta)


def chance_constrained_grip(f_t_req: float, mu_hat: float, sigma_mu: float,
                             delta: float = 0.05, min_margin: float = 0.05) -> float:
    """§4.5 机会约束握力: P(μ < f_t/f_n) <= δ → f_n >= f_t / (μ̂ - z_{1-δ} σ_μ)

    Parameters
    ----------
    f_t_req : float
        需求切向力 (N)
    mu_hat : float
        摩擦系数估计值
    sigma_mu : float
        摩擦系数标准差 (触觉不确定度)
    delta : float
        滑脱概率上界 (默认 5%, 攀岩高安全需求应用 0.001)
    min_margin : float
        denom 最小下界, 避免除零或过近临界点 (默认 0.05)

    Returns
    -------
    float
        需求法向力 f_n (N)

    Raises
    ------
    InfeasibleGripError
        denom = μ̂ - z σ_μ <= min_margin, 机会约束无解
    """
    if sigma_mu < 0:
        raise ValueError(f"sigma_mu must be >= 0, got {sigma_mu}")
    z = _z_1_minus_delta(delta)
    denom = mu_hat - z * sigma_mu
    if denom <= min_margin:
        raise InfeasibleGripError(
            f"机会约束无解: denom = μ̂({mu_hat:.4f}) - z(={z:.4f} for δ={delta:.4f}) * "
            f"σ_μ({sigma_mu:.4f}) = {denom:.4f} <= min_margin({min_margin}); "
            f"需提高 δ / 降低 f_t / 换更高摩擦接触面 / 触发安全停止"
        )
    return f_t_req / denom


def grip_demo(mu_hat: float = 0.85, sigma_mu: float = 0.15, verbose: bool = True) -> dict:
    """§6.6: Monte Carlo 滑脱率验证。

    R7 修复 (2026-10-07): 补 numpy import, z 用精确 Φ⁻¹, 保留原有对比结论。

    Returns
    -------
    dict
        {f_naive, slip_naive, f_cc, slip_cc, f_cc_learned, slip_learned_matched,
         slip_learned_overconf, delta, ok}
    """
    import numpy as np

    rng = np.random.default_rng(7)
    f_t_req = 30.0
    mu_true = mu_hat
    n_mc = 10000
    delta = 0.05

    def slip_rate(f_n: float, sigma_real: float) -> float:
        return float(np.mean(rng.normal(mu_true, sigma_real, n_mc) * f_n < f_t_req))

    f_naive = f_t_req / mu_hat
    f_cc = chance_constrained_grip(f_t_req, mu_hat, 0.15, delta=delta)
    f_cc_learned = chance_constrained_grip(f_t_req, mu_hat, 0.05, delta=delta)

    slip_naive = slip_rate(f_naive, 0.15)
    slip_cc = slip_rate(f_cc, 0.15)
    slip_learned_matched = slip_rate(f_cc_learned, 0.05)
    slip_learned_overconf = slip_rate(f_cc_learned, 0.15)

    if verbose:
        print("§6.6 机会约束握力 (Monte Carlo ×10000, δ=5%)")
        print(f"  [无裕度点估计]    f_n={f_naive:6.1f} N → 滑脱率 {slip_naive:.1%} (σ_real=0.15)")
        print(f"  [机会约束,未学习] f_n={f_cc:6.1f} N → 滑脱率 {slip_cc:.2%} (σ=0.15 匹配, 目标≤5%)")
        print(f"  [学习后 σ→0.05]   f_n={f_cc_learned:6.1f} N (握力降低 "
              f"{(1-f_cc_learned/f_cc)*100:.0f}%)")
        print(f"     真实 σ 也收敛到 0.05: 滑脱率 {slip_learned_matched:.2%} —— 安全保持")
        print(f"     真实 σ 仍 0.15 (过度自信): 滑脱率 {slip_learned_overconf:.1%} —— 风险回升,需保守切换")
        print(f"  → σ_μ 的诚实估计是握力调度的核心; 触觉学习使真 σ 与估计 σ 同步收敛才有双重收益")

    # 主验收: 匹配不确定度时滑脱率 ≤ δ (允许 0.5% 的 MC 抽样噪声)
    ok = slip_cc <= delta + 0.005 and slip_learned_matched <= delta + 0.005
    return {
        "f_naive": f_naive,
        "slip_naive": slip_naive,
        "f_cc": f_cc,
        "slip_cc": slip_cc,
        "f_cc_learned": f_cc_learned,
        "slip_learned_matched": slip_learned_matched,
        "slip_learned_overconf": slip_learned_overconf,
        "delta": delta,
        "ok": ok,
    }


# ============================================================
# §4.6 切向柔顺滑脱前兆 HMM（vs 静态摩擦圆判据）
# ============================================================
def soft_contact_slip_early_warn(delta_f_t: float, mu_hat: float,
                                  sigma_mu: float, delta: float = 0.05) -> dict:
    """切向力增量下, 检查剩余摩擦裕度是否触发机会约束滑脱前兆。

    Returns
    -------
    dict
        {ok: bool, warn_level: str, f_n_required: float, denom: float, reason: str}
    """
    if delta_f_t <= 0:
        return {"ok": True, "warn_level": "INFO", "f_n_required": 0.0,
                "denom": mu_hat, "reason": "无切向增量"}
    z = _z_1_minus_delta(delta)
    denom = mu_hat - z * sigma_mu
    if denom <= 0.05:
        return {"ok": False, "warn_level": "CRITICAL", "f_n_required": float("inf"),
                "denom": denom,
                "reason": f"摩擦不确定度过大, 机会约束无解 (denom={denom:.4f})"}
    f_n_required = delta_f_t / denom
    warn_level = "INFO" if denom > 0.3 else ("WARNING" if denom > 0.15 else "CRITICAL")
    return {"ok": True, "warn_level": warn_level, "f_n_required": f_n_required,
            "denom": denom, "reason": ""}


__all__ = [
    "InfeasibleGripError",
    "chance_constrained_grip",
    "grip_demo",
    "soft_contact_slip_early_warn",
]
