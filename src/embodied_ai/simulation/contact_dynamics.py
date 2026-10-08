"""L2 接触动力学修复 — ADMM ρ自适应 + κ二阶段 + 收敛判据 + HC 阻尼。

解锁关系:
  L2 (本模块) → L3 MCTS 真实化 (物理正确的 rollout)
             → L4 尾触觉闭环训练 (基于 dyno_capture_check)
             → dyno_capture_check 需要 L2 的收敛判据判定接触稳定性

物理模型 (三层):
  • 法向: Signorini 互补条件 (非穿透 + 法向力非负 + 乘积为零)
  • 切向: 库伦摩擦锥 (‖f_t‖ ≤ μ·f_n, 滑移方向判据)
  • 阻尼: Hunt-Crossley (f_d = κ · x^n · ẋ, 非线性阻尼与穿透深度耦合)

数学基础:
  ADMM 求解接触 LCP:
    minimize  ½ f^T H f + f^T g     subject to  f ∈_friction_cone
    H = J · M^{-1} · J^T  (接触空间等效质量)
    g = J · M^{-1} · h · τ_ext + J · v_old  (外力 + 当前速度)

ADMM 变量分裂:
    f^{k+1} = prox_{ρ/2}(f - z^k + u^k)  (原始更新)
    z^{k+1} = proj_friction_cone(f^{u+1} + u^k)  (摩擦锥投影)
    u^{k+1} = u^k + f^{u+1} - z^{u+1}  (对偶更新)

ρ 自适应:
    若 primal/dual > μ = 10: ρ *= 2  (对偶残差大 → 增强原始正则)
    若 dual/primal > μ:     ρ /= 2  (原始残差大 → 放松)
    ρ ∈ [ρ_min, ρ_max] = [1e-4, 1e4]

收敛判据 (κ 二阶段后检查):
    primal_residual  = ‖f - z‖_∞  < abstol + rtol · max(‖f‖_∞, ‖z‖_∞)
    dual_residual    = ‖ρ·(z_prev - z)‖_∞ < abstol + rtol · ‖ρ·u‖_∞
    均满足 + iter < max_iter → converged

HC 阻尼模型 (软接触):
    f_n = max(0, k · x^n) + κ · x^n · ẋ
    k  = 接触刚度 (N/m^n)
    n  =  Hertz 指数 (球面 1.5, 平面 1.0, 软胶 1.0~1.3)
    κ  = 阻尼-刚度耦合系数 (s/m — 穿透速度依赖, 物理来源: 材料黏弹性)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class ContactPoint:
    """单接触点 — 物理参数 + 当前状态。"""
    position: np.ndarray               # (3,) 世界坐标
    normal: np.ndarray                 # (3,) 墙面/物体法向 (指向可压方向)
    penetration: float = 0.0           # 穿透深度 (m, > 0 表示穿透)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,) 接触点速度
    mu: float = 0.8                    # 摩擦系数
    # Hunt-Crossley 阻尼参数
    stiffness_n_m: float = 1e5        # 法向刚度 (N/m^n)
    hertz_exponent: float = 1.5        # Hertz 指数 (1.0~1.5)
    hc_damp_kappa: float = 50.0        # HC 阻尼耦合系数 (s/m)
    # 有效接触半径 (用于 Hertz 接触半径折算, 0 = 平面接触)
    contact_radius: float = 0.02


@dataclass
class ADMMConfig:
    """ADMM 求解器配置 — ρ 自适应 + 收敛判据。"""
    # 自适应 ρ 范围与阈值
    rho_init: float = 1.0              # 初始惩罚
    rho_min: float = 1e-4              # 下界
    rho_max: float = 1e4               # 上界
    adapt_threshold: float = 10.0      # primal/dual 比值阈值 μ
    adapt_period: int = 10             # 每隔几步做一次 ρ 自适应

    # 收敛判据
    max_iter: int = 400
    abstol: float = 1e-4               # 绝对容差
    rtol: float = 1e-3                 # 相对容差

    # κ 二阶段
    broadphase_radius_m: float = 0.02  # 广相机距离门 (m, > penetration threshold)
    penetration_tol_m: float = 1e-4    # 穿透阈值 (m)


@dataclass
class ADMMResult:
    """ADMM 求解产物 — 力 + 收敛态 + 诊断。"""
    forces: np.ndarray                 # (3·n_contacts,) 接触力向量 (法向+切向交替)
    contact_forces: List[np.ndarray]   # List[(3,)] 各接触点力向量
    normal_forces: List[float]         # 各接触点法向力 (N)
    tangential_forces: List[np.ndarray] # 各接触点切向力 (2D in tangent plane)
    n_iter: int
    converged: bool
    primal_residual: float
    dual_residual: float
    rho_final: float
    active_contacts: List[int]         # 接触中的索引 (摩擦锥激活)


# --------------------------------------------------------------------------- #
# Hunt-Crossley 阻尼模型 (软接触, 物理来源: 材料黏弹性)
# --------------------------------------------------------------------------- #

def hunt_crossley_normal_force(cp: ContactPoint) -> float:
    """HC 法向力 — 穿透深度的非线性函数 + 速度耦合阻尼。

    公式:
      f_n = max(0, k · x^n) + κ · x^n · ẋ         (ẋ < 0 压缩 → 阻尼正贡献)
      x  = penetration depth (≥ 0)
      ẋ  = penetration rate (v_normal · n, 压入为负)

    物理解:
      k·x^n    → Hertz 接触刚度 (指数 n = 1.5 球面, 1.0 平面)
      κ·x^ṅ·ẋ  → 材料黏弹性阻尼 (穿透越深 + 压入越快 → 阻尼越大)
                 阻尼只在压缩相 (ẋ≤0) 做正功; 反弹相阻尼减小 (不拉回来)
    """
    x = max(cp.penetration, 0.0)
    if x <= 1e-9:
        return 0.0
    # Hertz 弹性项
    f_elastic = cp.stiffness_n_m * (x ** cp.hertz_exponent)
    # 法向穿透速度 (沿法向, 压缩为负)
    v_n = float(np.dot(cp.velocity, cp.normal))
    # HC 阻尼项: 只在压缩时 (v_n ≤ 0, 即穿透率 > 0)
    rate_of_penetration = max(-v_n, 0.0)  # 压入速率正
    f_damp = cp.hc_damp_kappa * (x ** cp.hertz_exponent) * rate_of_penetration
    return f_elastic + f_damp


# --------------------------------------------------------------------------- #
# κ 二阶段接触求解 (κ = greek kappa, 论文惯例: κ-phase / two-phase)
# --------------------------------------------------------------------------- #

def kappa_broadphase(penetrations: np.ndarray,
                     kappa_config: ADMMConfig) -> Tuple[np.ndarray, np.ndarray]:
    """κ 阶段一 — 广相机检测: 哪些点在接触门限内。

    基于穿透距离门限 (broadphase_radius_m + penetration_tol_m),
    返回候选接触掩码与候选数量。
    """
    threshold = kappa_config.broadphase_radius_m + kappa_config.penetration_tol_m
    mask = penetrations > -threshold  # 穿透 > 负门限 = 在宽门内
    return mask, int(mask.sum())


def kappa_narrowphase(penetrations: np.ndarray,
                      kappa_config: ADMMConfig) -> Tuple[np.ndarray, np.ndarray]:
    """κ 阶段二 — 窄相机精检: 真正穿透的接触 (|x| < penetration_tol)。

    硬接触判据: penetration > penetration_tol_m
    输出: (穿透掩码, 穿透索引列表)
    """
    hard_mask = penetrations > kappa_config.penetration_tol_m
    idx = np.where(hard_mask)[0].astype(int)
    return hard_mask, idx


# --------------------------------------------------------------------------- #
# 摩擦锥投影 (ADMM 的关键子步骤)
# --------------------------------------------------------------------------- #

def project_friction_cone(f_n: float, f_t_vec: np.ndarray, mu: float) -> Tuple[float, np.ndarray]:
    """投影力向量到摩擦锥内。

  摩擦锥:  f_n ≥ 0  且  ‖f_t‖ ≤ μ · f_n

  操作:
    (1) f_n < 0 → 钳到 0, 同时切向归零
    (2) ‖f_t‖ > μ·f_n → 按比例缩放到锥面

    返回: (投影后法向力, 投影后切向力向量)
"""
    if f_n < 0.0:
        return 0.0, np.zeros_like(f_t_vec)
    ft_norm = np.linalg.norm(f_t_vec)
    limit = mu * f_n
    if ft_norm > limit:
        scale = limit / max(ft_norm, 1e-12)
        return f_n, f_t_vec * scale
    return f_n, f_t_vec


# --------------------------------------------------------------------------- #
# L2 核心: ADMM 接触力求解器 (ρ 自适应)
# --------------------------------------------------------------------------- #

class ADMMContactSolver:
    """ADMM-based friction contact force solver with adaptive penalty.

    参考: ADMM for contact simulation (Todoronv 2019, "Implicit-contact solver")
    核心迭代:
      f^{k+1}  = argmin  ½ f^T H f + f^T g  +  ρ/2‖f - z^k + u^k‖²
               = (H + ρI)^{-1} · (ρ(z^k - u^k) - g)
      z^{k+1}  = proj_friction_cone(f^{k+1} + u^k)
      u^{k+1}  = u^k + f^{k+1} - z^{k+1}
    """

    def __init__(self, config: Optional[ADMMConfig] = None):
        self.config = config or ADMMConfig()
        self._rho_log: List[float] = []

    @property
    def rho_history(self) -> List[float]:
        return list(self._rho_log)

    def solve(self,
              contacts: List[ContactPoint],
              mass_matrix_inv: Optional[np.ndarray] = None,
              ext_force: Optional[np.ndarray] = None,
              jacobian: Optional[np.ndarray] = None) -> ADMMResult:
        """完整求解流程:  κ 二阶段 → 装配 H/g → ADMM 迭代 → 收敛诊断。

        参数:
          contacts: 候选接触点
          mass_matrix_inv: M^{-1} (3n × 3n); 未提供则用单位矩阵 (各接触点等质量)
          ext_force: 外力向量 (3n,); 未提供则零
          jacobian: J (3n × 3n); 未提供则用单位矩阵 (点接触简化)

        返回: ADMMResult 含力 + 收敛态 + 诊断
        """
        cfg = self.config
        n_c = len(contacts)
        if n_c == 0:
            return ADMMResult(
                forces=np.zeros(0), contact_forces=[], normal_forces=[],
                tangential_forces=[], n_iter=0, converged=True,
                primal_residual=0.0, dual_residual=0.0, rho_final=cfg.rho_init,
                active_contacts=[])

        # ---- κ 二阶段 ----
        pens = np.array([c.penetration for c in contacts])
        broad_mask, _ = kappa_broadphase(pens, cfg)
        _, hard_idx = kappa_narrowphase(pens, cfg)

        if len(hard_idx) == 0:
            return ADMMResult(
                forces=np.zeros(3 * n_c), contact_forces=[np.zeros(3) for _ in range(n_c)],
                normal_forces=[0.0] * n_c, tangential_forces=[np.zeros(2) for _ in range(n_c)],
                n_iter=0, converged=True, primal_residual=0.0, dual_residual=0.0,
                rho_final=cfg.rho_init, active_contacts=[])

        # ---- 装配接触空间矩阵 ----
        # H = J · M^{-1} · J^T  (3n × 3n, 每个接触点 3 DOF)
        if jacobian is not None:
            J = jacobian
        else:
            J = np.eye(3 * n_c)  # 单位雅可比 (点接触简化)
        if mass_matrix_inv is not None:
            M_inv = mass_matrix_inv
        else:
            # 默认: 各接触点有效质量 0.1 kg (20kg 整机分配到 4 肢)
            m_eff = 0.1
            M_inv = np.eye(3 * n_c) / m_eff
        H = J @ M_inv @ J.T  # (3n, 3n) 接触空间等效质量

        # g = J · M^{-1} · h · τ_ext + J · v_contact (当前速度带来的项)
        if ext_force is not None:
            g = J @ M_inv @ ext_force
        else:
            g = np.zeros(3 * n_c)
        # 加入当前速度的贡献 (让 ADMM 力去驱动速度到 0)
        v_flat = np.concatenate([c.velocity for c in contacts])  # (3n,)
        g += J @ v_flat

        # ---- 每接触点基底 (n, t1, t2) 用于摩擦锥投影 ----
        bases = []  # 每个 contact 的 [normal_t, tangent1_t, tangent2_t]
        for c in contacts:
            n = c.normal / max(np.linalg.norm(c.normal), 1e-12)
            # 构造切平面基底
            if abs(n[0]) < 0.9:
                t1 = np.cross(n, np.array([1.0, 0.0, 0.0]))
            else:
                t1 = np.cross(n, np.array([0.0, 1.0, 0.0]))
            t1 /= max(np.linalg.norm(t1), 1e-12)
            t2 = np.cross(n, t1)
            t2 /= max(np.linalg.norm(t2), 1e-12)
            bases.append((n, t1, t2))

        # ---- ADMM 初始化 ----
        rho = cfg.rho_init
        dim = 3 * n_c
        f = np.zeros(dim)        # 原始变量 (接触空间力)
        z = np.zeros(dim)        # 辅助变量 (摩擦锥投影域)
        u = np.zeros(dim)        # 对偶变量 (scaled)
        self._rho_log = [rho]
        converged = False
        p_res = float('inf')
        d_res = float('inf')

        for it in range(cfg.max_iter):
            # ---- Step 1: f-update (原始步骤, 线性系统) ----
            H_rho = H + rho * np.eye(dim)
            rhs = rho * (z - u) - g
            f = np.linalg.solve(H_rho, rhs)  # 直接求解 (维度 ≤36, 无性能瓶颈)

            # ---- Step 2: z-update (逐接触点投影到摩擦锥) ----
            z_old = z.copy()
            f_proj = f + u  # 原始 + 对偶辅助量
            for ci in range(n_c):
                f_c = f_proj[3 * ci:3 * ci + 3]
                n_vec, t1_vec, t2_vec = bases[ci]
                fn = float(np.dot(f_c, n_vec))
                ft = np.array([np.dot(f_c, t1_vec), np.dot(f_c, t2_vec)])
                fn_p, ft_p = project_friction_cone(fn, ft, contacts[ci].mu)
                z[3 * ci:3 * ci + 3] = fn_p * n_vec + ft_p[0] * t1_vec + ft_p[1] * t2_vec

            # ---- Step 3: u-update (对偶变量) ----
            u = u + f - z

            # ---- 收敛判据检查 ----
            p_res = float(np.linalg.norm(f - z, ord=np.inf))
            d_res = float(np.linalg.norm(rho * (z_old - z), ord=np.inf))
            f_norm = max(float(np.linalg.norm(f, ord=np.inf)), 1e-12)
            z_norm = max(float(np.linalg.norm(z, ord=np.inf)), 1e-12)
            u_norm = max(float(np.linalg.norm(rho * u, ord=np.inf)), 1e-12)
            p_tol = cfg.abstol + cfg.rtol * max(f_norm, z_norm)
            d_tol = cfg.abstol + cfg.rtol * u_norm
            if p_res < p_tol and d_res < d_tol:
                converged = True
                break

            # ---- ρ 自适应 (每隔 adapt_period 步) ----
            if (it + 1) % cfg.adapt_period == 0:
                ratio = (p_res / max(d_res, 1e-12))
                if ratio > cfg.adapt_threshold:
                    rho = min(rho * 2.0, cfg.rho_max)
                    use_cholesky = self._update_cholesky(H, rho, L)
                elif (d_res / max(p_res, 1e-12)) > cfg.adapt_threshold:
                    rho = max(rho * 0.5, cfg.rho_min)
                    use_cholesky = self._update_cholesky(H, rho, L)
                self._rho_log.append(rho)

        # ---- 输出装配 ----
        active = hard_idx.tolist()
        contact_forces = [z[3 * i:3 * i + 3] for i in range(n_c)]
        normal_forces = [float(z[3 * i:3 * i + 3] @ bases[i][0]) for i in range(n_c)]
        tangential_forces = [
            np.array([float(z[3 * i:3 * i + 3] @ bases[i][1]),
                      float(z[3 * i:3 * i + 3] @ bases[i][2])])
            for i in range(n_c)
        ]

        return ADMMResult(
            forces=z, contact_forces=contact_forces, normal_forces=normal_forces,
            tangential_forces=tangential_forces, n_iter=it + 1, converged=converged,
            primal_residual=p_res, dual_residual=d_res, rho_final=rho,
            active_contacts=active)


# --------------------------------------------------------------------------- #
# 判据: 动态收敛检测 (解锁 dyno_capture_check)
# --------------------------------------------------------------------------- #

def convergence_check(result: ADMMResult) -> Tuple[bool, str]:
    """收敛判定 — 用于 dyno_capture_check 的"接触稳定性"判断。

    双判据 (严格):
    (1) ADMM 收敛 flag (primal + dual residuals 在容差内)
    (2) 最大法向力 ≤ 物理握持上限 (防止虚假大穿透产生不可信的大力)

    返回: (是否收敛 + 原因描述)
    """
    if not result.converged:
        return False, f"未收敛 (iter={result.n_iter}, primal={result.primal_residual:.2e})"
    if not result.active_contacts:
        return True, "无硬接触 — 自由飞行相"
    max_f = max(abs(f) for f in result.normal_forces) if result.normal_forces else 0.0
    # 折叠手极限 150N (与 wall_contact 一致)
    if max_f > 150.0:
        return False, f"法向力超握持极限 {max_f:.1f}N > 150N"
    return True, f"收敛 ({result.n_iter} iters, primal={result.primal_residual:.2e})"


# --------------------------------------------------------------------------- #
# 自适应参数 (DR 注入 → 跨批次一致)
# --------------------------------------------------------------------------- #

def make_defaults_from_contract(contract: dict) -> ContactPoint:
    """从 dr_params yaml 构造默认接触参数 (跨批次一致性)。"""
    env = contract.get("environment", {})
    mech = contract.get("mechanical", {})
    sens = contract.get("sensing", {})
    return ContactPoint(
        position=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
        mu=float(mu := sens.get("friction_coefficient", {}).get("mean", 0.8)),
        stiffness_n_m=float(mech.get("joint_stiffness", {}).get("mean", 1e5)),
        hertz_exponent=1.5 if env.get("wall_type") == "rock" else 1.0,
        hc_damp_kappa=float(mech.get("contact_damping_ns_m", {}).get("mean", 50.0)),
    )
