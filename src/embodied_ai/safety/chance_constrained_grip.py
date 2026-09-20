"""机会约束握力调度 (横切安全层 S) — 拆自 climbing_dynamics_core.py (K1 拆分, 逻辑未改)。

公式: f_n >= |f_t| / (mu_hat - z*sigma_mu)
"""
def chance_constrained_grip(f_t_req, mu_hat, sigma_mu, delta=0.05):
    """§4.5 机会约束握力: P(μ < f_t/f_n) <= δ → f_n >= f_t/(μ̂ - z_{1-δ} σ_μ)"""
    from math import erf, sqrt
    z = 1.6449 if delta == 0.05 else sqrt(2) * (1 - 2 * delta)  # 近似
    denom = mu_hat - z * sigma_mu
    return f_t_req / max(denom, 0.05)

def grip_demo(mu_hat, sigma_mu):
    """§6.6: Monte Carlo 滑脱率验证。
    语义: 控制器用机会约束握力 f_n = f_t/(μ̂ - z σ_assumed)。
    σ_assumed 是控制器持有的不确定度估计（新岩点 0.15 / EKF 学习后 0.05）。
    学习后控制器用更小的 σ_assumed → 更低握力 → 但同一物理 μ 波动下
    滑脱率会回升（"过度自信"风险）。真正的收益: 学习后若真实波动也收敛
    （同一岩点上连续抓握），则σ_real 也缩小 → 更低握力且保持安全。
    这里演示两点: (1) 机会约束在 σ 匹配时精确达标;
    (2) σ_assumed 高估或低估的敏感性。"""
    rng = np.random.default_rng(7)
    f_t_req = 30.0
    mu_true = mu_hat
    n_mc = 10000
    def slip_rate(f_n, sigma_real):
        return np.mean(rng.normal(mu_true, sigma_real, n_mc) * f_n < f_t_req)
    f_naive = f_t_req / mu_hat
    f_cc = chance_constrained_grip(f_t_req, mu_hat, 0.15)
    f_cc_learned = chance_constrained_grip(f_t_req, mu_hat, 0.05)
    print("§6.6 机会约束握力 (Monte Carlo ×10000)")
    print(f"  [无裕度点估计] f_n={f_naive:.0f} N → 滑脱率 {slip_rate(f_naive,0.15):.0%} (σ_real=0.15)")
    print(f"  [机会约束,未学习] f_n={f_cc:.0f} N → 滑脱率 {slip_rate(f_cc,0.15):.1%} (σ=0.15 匹配, 目标5%)")
    print(f"  [学习后 σ→0.05] f_n={f_cc_learned:.0f} N (握力降低 {(1-f_cc_learned/f_cc)*100:.0f}%)")
    print(f"     若真实 σ 也收敛到 0.05: 滑脱率 {slip_rate(f_cc_learned,0.05):.1%} —— 安全保持")
    print(f"     若真实 σ 仍 0.15(过度自信): 滑脱率 {slip_rate(f_cc_learned,0.15):.1%} —— 风险回升,需保守切换")
    print(f"  → σ_μ 的诚实估计是握力调度的核心; 触觉学习使真 σ 与估计 σ 同步收敛才有双重收益")

# ============================================================
# §4.6 切向柔顺滑脱前兆 HMM（vs 静态摩擦圆判据）
# ============================================================
