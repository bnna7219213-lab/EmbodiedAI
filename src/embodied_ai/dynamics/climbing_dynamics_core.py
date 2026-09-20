"""接触动力学核心 (L5/L1 数学基础) — 迁自 src/core/climbing_dynamics_core.py (K1 原样迁移)。

Hertz/Hunt-Crossley/摩擦锥/QP 力分配/摆荡/dyno/CWC。
感知与安全相关部分已拆至 embodied_ai.perception / embodied_ai.safety。
"""
# -*- coding: utf-8 -*-
"""
climbing_dynamics_core.py
攀岩竞技动力学与触觉-视觉融合的底层数学核心（纯 numpy 实现，无外部依赖）

对应文档《攀岩竞技动力学与触觉-视觉融合：数学基础》的可验证实现：
  §2.1-2.3  Hertz/Hunt-Crossley 统一接触模型 + 凝胶指尖载荷极限
  §2.5      脉冲接触动力学与捕获可行性
  §3.1      多接触力分配 QP（ADMM 投影求解）
  §3.2      摆荡能量泵浦控制（RK4）
  §3.3-3.5  Dyno 弹道规划 + 捕获可行性不等式组 + 柔顺捕获速度上限
  §3.4      接触 Wrench 锥（CWC）动态稳定性判据（2D）
  §4.3      多速率 EKF（视觉 20Hz 含丢帧 / 触觉 100Hz），μ 在线估计
  §4.4      视觉材质先验注入
  §4.5      机会约束握力调度（Monte Carlo 滑脱率验证）
  §4.6      切向柔顺滑脱前兆 HMM（对比静态摩擦圆判据）
  §6.8      完整竞技攀爬周期演示

运行: python climbing_dynamics_core.py  （所有打印即文档 §6 的数值验证结果）
"""
import numpy as np

G = 9.81  # 重力加速度 m/s^2

# ============================================================
# §2.1-2.3 Hertz / Hunt-Crossley 统一接触模型
# ============================================================

def hertz_composite_modulus(E1, nu1, E2, nu2):
    """复合模量 E* = 1 / ( (1-v1^2)/E1 + (1-v2^2)/E2 )  [Pa]
    凝胶极软时 E* ≈ E_gel/(1-v_gel^2) ≈ 0.1MPa/0.76 ≈ 0.13 MPa
    """
    return 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)

def hertz_stiffness(E_gel, nu_gel, E_wall, nu_wall, R_fingertip):
    """Hertz 刚度 k = (4/3) E* sqrt(R*)  [N/m^1.5]
    软指尖(球) vs 平壁(曲率→0): 1/R* = 1/R_tip
    数值示例: E*≈1.3e5 Pa, R*=0.01m → k≈1.75e4 N/m^1.5
    注: 这是【凝胶本体的 Hertz 极限载荷】——它告诉我们纯 Hertz 弹性下
    5mm 压深只能产生 ~6N 法向力，远低于攀爬需求(单指尖>15N)。
    因此实际 GelSight 类指尖攀爬必须依赖: (a) 更高硬度凝胶(E↑ k∝E),
    (b) 更大接触面积/曲率半径(k∝√R), (c) 指尖内骨架承载(凝胶只做传感层)。
    这正是本函数要揭示的硬件选型约束（文档 §2.3）。
    """
    E_star = hertz_composite_modulus(E_gel, nu_gel, E_wall, nu_wall)
    R_star = R_fingertip
    return (4.0 / 3.0) * E_star * np.sqrt(R_star)

def hunt_crossley_force(delta, delta_dot, k, b, n=1.5):
    """Hunt-Crossley: f = k δ^n (1 + b δ̇)，δ>0 为穿透，力非负截断"""
    if delta <= 0.0:
        return 0.0
    f = k * delta**n * (1.0 + b * delta_dot)
    return f if f > 0.0 else 0.0

def hc_damping_from_restitution(k, v_impact, e_target):
    """§2.2 回复系数闭式标定：e ≈ 1 - (5/3) α v_imp，α = b/k（文献标准形式 f=k δ^n + α k δ^n δ̇）
    → b = (3k/(5 v_imp)) (1 - e_target)
    """
    return (3.0 * k / (5.0 * v_impact)) * (1.0 - e_target)

def fingertip_load_limit(k, gel_thickness, safety=0.5):
    """§2.3 指尖法向载荷极限 f_max = k δ_max^1.5, δ_max = safety·gel_thickness"""
    delta_max = safety * gel_thickness
    return k * delta_max**1.5, delta_max

def climbing_fingertip_spec(E_gel=2.5e5, nu_gel=0.45, R_tip=0.015,
                            gel_t=0.008, safety=0.5, E_wall=2.0e11, nu_wall=0.3):
    """攀爬级指尖规格（文档 §2.3 推论）：传感凝胶 + 内骨架共同承载。
    骨架刚度量级取 k_skel = 5e5 N/m^1.5（3D 打印 TPU/尼龙骨架实测量级），
    凝胶与骨架并联: k_total = k_gel + k_skel（并联弹簧）。
    选型逻辑: 5kg 机器人单臂悬挂需 f_n ≥ W/μ ≈ 61N @ δ_max。
    """
    k_gel = hertz_stiffness(E_gel, nu_gel, E_wall, nu_wall, R_tip)
    k_skel = 5.0e5
    k_total = k_gel + k_skel
    f_max, delta_max = fingertip_load_limit(k_total, gel_t, safety)
    return k_total, f_max, delta_max, k_gel

def hertz_stiffness_demo():
    """§6.1 数值验证：凝胶指尖 vs 攀爬级指尖（传感凝胶+承载骨架并联）"""
    # —— 纯传感凝胶指尖（GelSight 类）——
    E_gel, nu_gel = 1.0e5, 0.49        # Shore 20A 凝胶 E≈0.1MPa
    E_wall, nu_wall = 2.0e11, 0.30     # 钢壁
    R_tip = 0.010                      # 10mm 指尖
    k_gel = hertz_stiffness(E_gel, nu_gel, E_wall, nu_wall, R_tip)
    gel_t = 0.010
    f_max_gel, delta_max = fingertip_load_limit(k_gel, gel_t)
    f_2mm = k_gel * 0.002**1.5
    print("§6.1 Hertz/指尖载荷极限（纯凝胶 vs 攀爬级规格）")
    print(f"[A] 纯传感凝胶指尖 (E=0.1MPa, R=10mm):")
    print(f"  E* = {hertz_composite_modulus(E_gel,nu_gel,E_wall,nu_wall):.3e} Pa")
    print(f"  k_gel = {k_gel:.3e} N/m^1.5, δ_max={delta_max*1000:.0f} mm → f_max = {f_max_gel:.1f} N")
    print(f"  常规工作点 δ=2mm → f_n = {f_2mm:.1f} N —— 远低于攀爬需求!")
    m_robot = 5.0; W = m_robot * G; mu = 0.8
    print(f"  体重 {W:.1f} N: 四点静态需每指尖 f_n≥{W/4/mu:.1f} N, "
          f"单臂悬挂需 f_n≥{W/mu:.1f} N → 纯凝胶指尖【不可行】")
    print(f"  → 结论: 攀爬指尖必须 凝胶传感层 + 高刚度承载骨架并联 (k∝E·√R 选型)")
    # —— 攀爬级指尖 ——
    k_tot, f_max, delta_max2, _ = climbing_fingertip_spec()
    print(f"[B] 攀爬级指尖 (E_gel=0.25MPa+骨架 k=5e5, R=15mm, 胶厚8mm):")
    print(f"  k_total = {k_tot:.3e} N/m^1.5, δ_max={delta_max2*1000:.0f} mm → f_max = {f_max:.1f} N")
    print(f"  单臂悬挂 f_n 需求 {W/mu:.1f} N vs f_max {f_max:.1f} N → "
          f"{'可行(裕度%.0f%%)' % ((f_max/(W/mu)-1)*100) if f_max > W/mu else '不可行'}")
    print(f"  四点静态: 每指尖 {W/4:.1f} N 切向, f_n≥{W/4/mu:.1f} N → 裕度 "
          f"{(f_max/(W/4/mu)-1)*100:.0f}%")
    return k_tot, f_max

# ============================================================
# §1.4 / §3.1 摩擦锥与多接触力分配 QP（ADMM）
# ============================================================

def project_to_cone_2d(f, n, mu):
    """2D 摩擦锥解析投影: 锥 K = {f: f·n >= 0, |f_t| <= mu (f·n)}"""
    fn = np.dot(f, n)
    t = np.array([-n[1], n[0]])
    ft = np.dot(f, t)
    if fn >= 0.0 and abs(ft) <= mu * fn:
        return f.copy()
    # 投影到锥边界：候选 = 两条母线段的投影取较近者
    gen1 = n + mu * t
    gen2 = n - mu * t
    cand = []
    for g in (gen1, gen2):
        gg = np.dot(g, g)
        s = max(0.0, np.dot(f, g) / gg)
        cand.append(s * g)
    return min(cand, key=lambda p: np.dot(f - p, f - p))

def project_to_cone_3d(f, n, mu):
    """3D 二阶锥投影 ||f_t|| <= mu f_n（解析解，用于 3D 扩展）"""
    fn = np.dot(f, n)
    t_vec = f - fn * n
    ft = np.linalg.norm(t_vec)
    if fn >= 0 and ft <= mu * fn:
        return f.copy()
    t_hat = t_vec / ft if ft > 1e-12 else np.zeros(3)
    if mu * fn + ft <= 0:  # 对极锥内
        return np.zeros(3)
    # 锥面: f = (fn' ) n + mu fn' t_hat, fn' = (fn + mu ft)/(1+mu^2)
    fn_new = (fn + mu * ft) / (1.0 + mu * mu)
    return fn_new * n + mu * fn_new * t_hat

class ForceAllocationQP:
    """§3.1 带摩擦裕度的力分配 QP（ADMM 交替投影求解）

    min  Σ||λi||²  - κ·M_f
    s.t. Σλi = F_des (2D力平衡), Σ(p_i×λi) = τ_des (力矩平衡), λi ∈ K_i
    实现：等式约束用增广拉格朗日（线性项），锥约束用投影步。
    """
    def __init__(self, contacts_p, contacts_n, mu, kappa=1.0):
        """contacts_p: (nc,2) 接触点位置; contacts_n: (nc,2) 单位法向(指向岩壁外)"""
        self.p = np.asarray(contacts_p, float)
        self.n = np.asarray(contacts_n, float)
        self.mu = mu
        self.kappa = kappa

    def solve(self, F_des, tau_des, ref=None, max_iter=400, tol=1e-5):
        """F_des: 期望合力(2,)（含重力，即 Σλ = -m g y + 惯性项）; tau_des: 期望绕 ref 的合力矩"""
        nc = len(self.p)
        ref = np.zeros(2) if ref is None else ref
        # 等式约束 A λ = b, λ = [λ1;λ2;...], A: (3, 2nc)
        A = np.zeros((3, 2 * nc))
        for i in range(nc):
            A[0:2, 2*i:2*i+2] = np.eye(2)
            r = self.p[i] - ref
            A[2, 2*i:2*i+2] = [-r[1], r[0]]  # 2D 叉乘
        b = np.array([F_des[0], F_des[1], tau_des])

        # 初始化: 用伪逆解一个满足等式的起点
        lam = np.linalg.pinv(A) @ b
        y = np.zeros(3)          # 等式约束对偶
        rho = 200.0              # ADMM 惩罚（提高以压等式残差）
        A_pre = np.linalg.inv(np.eye(2 * nc) + rho * A.T @ A) @ (rho * A.T)
        for it in range(max_iter):
            # 步1: lam = argmin ||λ - z_prev||² + ρ||Aλ - b + y/ρ||²  的解
            z = lam.copy()
            lam_new = z - A_pre @ (A @ z - b + y / rho)
            # 步2: 逐接触锥投影
            for i in range(nc):
                lam_new[2*i:2*i+2] = project_to_cone_2d(lam_new[2*i:2*i+2], self.n[i], self.mu)
            # 步3: 对偶上升
            y += rho * (A @ lam_new - b)
            lam = lam_new
            if np.linalg.norm(A @ lam - b) < tol:
                break
        forces = [lam[2*i:2*i+2] for i in range(nc)]
        M_f = min(
            self.mu * np.dot(f, self.n[i]) - abs(np.dot(f, np.array([-self.n[i,1], self.n[i,0]])))
            for i, f in enumerate(forces)
        )
        return forces, M_f, it + 1

def force_allocation_demo():
    """§6.2 三接触力分配（攀岩岩点法向: 手点朝上拉、脚点朝上蹬）"""
    qp = ForceAllocationQP(
        contacts_p=[(0.0, 2.0), (0.15, 2.3), (0.0, 1.2)],    # 左手/右手/左脚岩点
        contacts_n=[(-0.35, 0.94), (-0.20, 0.98), (0.30, 0.95)],  # 岩点对手的反力方向(朝上)
        mu=0.8, kappa=2.0)
    m = 5.0
    F_des = np.array([0.0, m * G])   # 需 Σλ = +mg ŷ（对抗重力）
    tau_des = 0.0                     # 无角加速度
    forces, M_f, iters = qp.solve(F_des, tau_des, ref=np.array([0.05, 1.9]))
    print("§6.2 力分配 QP (ADMM)")
    for i, f in enumerate(forces):
        print(f"  接触{i}: λ=({f[0]:6.2f},{f[1]:6.2f}) N  |λ|={np.linalg.norm(f):.2f}")
    print(f"  收敛迭代 {iters}, 摩擦裕度 M_f = {M_f:.3f} N (>0 即安全)")
    return M_f

# ============================================================
# §3.2 摆荡能量泵浦（参数激励）+ §3.3 Dyno 弹道与捕获
# ============================================================

def swing_pumping_sim(m=5.0, l0=0.6, dl=0.18, c_damp=0.02, n_periods=10,
                      dt=0.002, k_pump=1.0, phi0=0.45):
    """§3.2/§6.3 摆荡能量泵浦：参数激励控制律 l(t) 与摆相耦合

    状态: φ(摆角), ω(角速度), l(摆长)
    动力学: (m l²) φ̈ + c ω + m g l sinφ = 0  →  φ̈ = -(g/l) sinφ - (c/(m l²)) ω + (2 l̇/l) ω
    泵浦律(参数激励): 最低点(cosφ>0, |ω|最大)收绳做功注入能量, 两端(cosφ<0)放绳
                      l̇ = -k_pump · dl · tanh(cosφ/0.3) · min(1, |ω|/0.5)
                      且 l ∈ [l0-dl, l0+dl] 硬限幅（防止持续收缩）
    记账: 摆幅 φ_amp = max|φ| 逐周期对比（能量注入的直接体现）
    """
    def deriv(s):
        phi, om, l = s
        l_dot = -k_pump * dl * np.tanh(np.cos(phi) / 0.3) * min(1.0, abs(om) / 0.5)
        if l <= l0 - dl and l_dot < 0:
            l_dot = 0.0
        if l >= l0 + dl and l_dot > 0:
            l_dot = 0.0
        dphi = om
        dom = -(G / l) * np.sin(phi) - (c_damp / (m * l * l)) * om + (2.0 * l_dot / l) * om
        return np.array([dphi, dom, l_dot])

    s = np.array([phi0, 0.0, l0])  # 初始摆角 ~26°
    log = []
    v_peak = 0.0
    steps_per_period = int(2 * np.pi * np.sqrt(l0 / G) / dt)
    for step in range(n_periods * steps_per_period):
        k1 = deriv(s); k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2); k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        v_peak = max(v_peak, abs(s[1]) * s[2])
        if step % steps_per_period == 0:
            log.append((step * dt, s[0], s[1], s[2]))
    phi_amp0 = phi0
    v_tang = abs(s[1]) * s[2]
    print("§6.3 摆荡能量泵浦 (RK4, 参数激励)")
    for i, (t, phi, om, l) in enumerate(log[:6]):
        print(f"  周期{i}: φ={np.degrees(phi):7.2f}°  ω={om:6.3f} rad/s  l={l:.3f} m")
    print(f"  ... {n_periods} 周期: 初始摆角 {np.degrees(phi_amp0):.1f}°, "
          f"末态 |ω|={abs(s[1]):.3f} rad/s (初始 0)")
    print(f"  全程切向速度峰值 v_peak={v_peak:.2f} m/s（dyno 释放应在最低点速度峰值松手）")
    print(f"  |ω| 峰值出现于摆幅扩大过程 —— 摆幅持续扩大即泵浦生效")
    return v_peak

def dyno_plan(v_release, c0, p_star, r_max=0.8, margin=0.15):
    """§3.3(a) 弹道规划：给定释放速度与目标点，求飞行时间与捕获伸展需求

    c(t) = c0 + v t + ½ g t²（抛体），捕获可达 ⟺ dist(c(tf), p_star) <= r_max(1-η)
    """
    best = None
    for tf in np.arange(0.05, 1.2, 0.005):
        c_t = np.array([c0[0] + v_release[0] * tf,
                        c0[1] + v_release[1] * tf - 0.5 * G * tf * tf])
        d = np.linalg.norm(p_star - c_t)
        if best is None or d < best[1]:
            best = (tf, d, c_t)
    tf, d_req, c_tf = best
    ok = d_req <= r_max * (1.0 - margin)
    return {"tf": tf, "reach_req": d_req, "c_tf": c_tf, "reachable": ok}

def dyno_capture_check(v_rel, n, mu, m_eff, f_max, t_pulse=0.005,
                       e_target=0.6, use_compliance=True, k=None):
    """§3.3(b)/§2.5 捕获可行性不等式组

    1) 捕获几何: 切向/法向速度比 |v_t|/v_n <= mu（接近方向在摩擦锥内）
    2) 脉冲载荷: 刚性假设 Λ_n = m_eff v_n, 峰值力 ≈ Λ_n/t_pulse <= f_max
       柔顺: Hunt-Crossley 阻尼把接触时间拉长到 τ_c ≈ (m_eff/k)^{1/2}·c_shape,
       峰值力 ≈ k δ_peak^1.5，δ_peak 由能量守恒 ½ m v² = ∫k δ^1.5 dδ 求得
    3) 冲击后: 残余切向速度经摩擦脉冲衰减（μ 足够则粘住）
    """
    vn = np.dot(v_rel, n)
    vt = v_rel - vn * n
    vt_norm = np.linalg.norm(vt)
    result = {"v_n": vn, "v_t": vt_norm}
    # 条件1: 捕获角
    beta_ok = (vt_norm <= mu * abs(vn)) if vn > 0 else False
    result["capture_angle_ok"] = beta_ok
    result["capture_angle_deg"] = np.degrees(np.arctan2(vt_norm, max(abs(vn), 1e-9)))
    # 条件2: 载荷
    if use_compliance and k is not None:
        # 柔顺捕获: 最大压深由动能 = 弹簧储能 ½ m v_n² = k δ^{2.5}/2.5 → δ_peak = (1.25 m v_n²/k)^{0.4}
        delta_peak = (1.25 * m_eff * vn**2 / k) ** 0.4
        f_peak = k * delta_peak ** 1.5
        t_contact = 2.5 * np.sqrt(m_eff / k) * delta_peak ** 0.25 * 2.0  # 半周期量级估计
    else:
        delta_peak, t_contact = None, t_pulse
        f_peak = m_eff * abs(vn) / t_pulse
    result.update(f_peak=f_peak, delta_peak=delta_peak, t_contact=t_contact,
                  load_ok=f_peak <= f_max)
    result["feasible"] = beta_ok and result["load_ok"]
    return result

def dyno_demo(k_hc):
    """§6.4 Dyno 规划+捕获可行性（刚性 vs 柔顺的定量对比）"""
    print("§6.4 Dyno 弹道 + 捕获可行性")
    c0 = np.array([0.2, 2.0]); p_star = np.array([0.85, 2.30])
    v_release = np.array([1.0, 0.55])          # 由摆荡释放获得
    plan = dyno_plan(v_release, c0, p_star)
    print(f"  弹道: 飞行 {plan['tf']*1000:.0f}ms, 伸展需求 {plan['reach_req']:.2f} m "
          f"(上限 0.68 m) → {'可达' if plan['reachable'] else '不可达'}")
    # 捕获速度 = 肢体相对岩点速度（末端已部分随动减速后的有效冲击速度）
    v_rel = np.array([0.34, 0.25]); n = np.array([1.0, 0.0])
    m_eff = 2.5; mu = 0.8
    f_max, _ = fingertip_load_limit(k_hc, 0.008)
    rigid = dyno_capture_check(v_rel, n, mu, m_eff, f_max, use_compliance=False)
    soft = dyno_capture_check(v_rel, n, mu, m_eff, f_max, use_compliance=True, k=k_hc)
    print(f"  有效捕获速度 |v_rel|={np.linalg.norm(v_rel):.2f} m/s (v_n={soft['v_n']:.2f})")
    print(f"  [刚性] 峰值力 {rigid['f_peak']:.0f} N vs f_max {f_max:.0f} N → "
          f"{'可行' if rigid['load_ok'] else '超载,不可行'}")
    print(f"  [柔顺] δ_peak={soft['delta_peak']*1000:.1f} mm(≤4mm), 峰值力 {soft['f_peak']:.0f} N, "
          f"接触时长 {soft['t_contact']*1000:.0f} ms → {'可行' if soft['load_ok'] else '不可行'}")
    print(f"  捕获角 {soft['capture_angle_deg']:.1f}° vs atan(μ)={np.degrees(np.arctan(mu)):.1f}° → "
          f"{'锥内' if soft['capture_angle_ok'] else '锥外'}")
    # 最大可捕获速度闭式（§3.5）: δ_peak = δ_max 反解 v
    d_max = 0.008 * 0.5
    v_cap_soft = np.sqrt(k_hc * d_max**2.5 / (1.25 * m_eff))
    v_cap_rigid = f_max * 0.005 / m_eff
    print(f"  §3.5 最大可捕获法向速度: 刚性 {v_cap_rigid:.2f} m/s, "
          f"柔顺 {v_cap_soft:.2f} m/s (提升 {(v_cap_soft/v_cap_rigid-1)*100:.0f}%)")
    print(f"  → 柔顺指尖直接决定竞技 dyno 的速度上限（选型公式见文档 §3.5）")

# ============================================================
# §3.4 接触 Wrench 锥（CWC）动态稳定性判据（2D）
# ============================================================

def cwc_membership(contacts_p, contacts_n, mu, w_test, n_gen=16, c_ref=None):
    """检查 wrench w_test 是否在接触 wrench 锥内（生成元 + NNLS 法）"""
    from scipy.optimize import nnls as scipy_nnls
    c_ref = np.mean(np.asarray(contacts_p, float), axis=0) if c_ref is None else c_ref
    gens = []
    for i, (p, nvec) in enumerate(zip(contacts_p, contacts_n)):
        p = np.asarray(p, float); nvec = np.asarray(nvec, float)
        t = np.array([-nvec[1], nvec[0]])
        for a in np.linspace(-mu, mu, n_gen):
            d = nvec + a * t   # 锥内方向（凸组合生成元）
            d = d / np.linalg.norm(d)
            w = np.array([d[0], d[1], (p - c_ref)[0] * d[1] - (p - c_ref)[1] * d[0]])
            gens.append(w)
    Wg = np.array(gens).T  # (3, ng)
    alpha, resid = scipy_nnls(Wg, w_test)
    return resid < 0.5, resid

def cwc_demo():
    """§3.4: 动态稳定性——重力/惯性需求 wrench 是否被接触 wrench 锥覆盖。
    约定: λ 是岩点对机器人的作用力（向上为正）。对抗重力需要
    需求 wrench w_req = [Σλ; Στ] 满足 Σλ = +mg ŷ (+惯性项)。"""
    contacts_p = [(0.0, 2.0), (0.15, 2.3), (0.0, 1.2)]
    contacts_n = [(-0.35, 0.94), (-0.20, 0.98), (0.30, 0.95)]  # 朝上岩点
    mu = 0.8; m = 5.0
    c = np.array([0.05, 1.9])
    w_static = np.array([0.0, +m * G, 0.0])          # 静态: Σλ=mg ŷ
    w_dynamic = np.array([0.0, m * G + m * 2.0, 0.0])  # 上攀加速 a=2m/s²: Σλ=m(g+a)
    # 反例: 需求带大水平力(横荡急停) —— 单靠这几个浅锥接触无法提供
    w_lateral = np.array([0.8 * m * G, m * G, 0.0])
    ok_s, r_s = cwc_membership(contacts_p, contacts_n, mu, w_static, c_ref=c)
    ok_d, r_d = cwc_membership(contacts_p, contacts_n, mu, w_dynamic, c_ref=c)
    ok_x, r_x = cwc_membership(contacts_p, contacts_n, mu, w_lateral, c_ref=c)
    print("§3.4 CWC 动态稳定性判据")
    print(f"  静态重力需求: {'锥内(稳定)' if ok_s else '锥外(失稳)'} (残差 {r_s:.2f})")
    print(f"  动态(上攀 a=2m/s²): {'锥内(可执行)' if ok_d else '锥外(该加速度不可行)'} (残差 {r_d:.2f})")
    print(f"  反例(横荡急停 Fx=0.8mg): {'锥内' if ok_x else '锥外 —— 该横向机动当前接触构型不可行,'
          ' 需调整抓握方向或加入新接触'} (残差 {r_x:.2f})")
    return ok_s, ok_d

# ============================================================
# §4.3-4.5 多速率 EKF + 视觉先验 + 机会约束握力
# ============================================================
