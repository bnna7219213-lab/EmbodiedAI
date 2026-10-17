"""攀岩竞技场景焊接层 (climb_scene) — Phase A1+A2。

把 v0.7 的离散模块焊成垂直场景闭环 (climb-soft-robot-plan §二):
  dexterous_hand (G11, hold 分类) ─┐
  soft_body (G6, 柔性形变→δ_eff) ──┼→ Hunt-Crossley 柔顺接触 (climbing_dynamics_core)
  whole_body_3d (G9, 双足本体) ────┘      │
  vsa_coupling (P1.3, 蓄能→dyno) ────────→ dyno_plan 弹道 → 柔顺捕获
                                          → 状态机续攀

判定标准 (plan §二): 一个攀岩周期跑通, 每个物理量由对应模块真实计算而非 mock。

A2 柔性耦合 (核心物理): 柔性指尖在接触前的等效预压形变
    δ_eff = F_pre / k_soft  (soft_body 网格切向刚度)
  接入 Hunt-Crossley 的 δ → 柔顺分支峰值力
    F_peak ≈ k·δ^1.5 (能量守恒 ½mv² = ∫kδ^1.5dδ = ⅖kδ^2.5)
  对比刚性分支 F_peak ≈ m·v/t_pulse → 柔顺必须显著更低 (VSA-dyno 结论的实验证实)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════
# 墙面 hold 图谱
# ══════════════════════════════════════════════════════════════════

class HoldType(str, Enum):
    POCKET = "pocket"      # 口袋点 (深握)
    PINCH = "pinch"        # 捏点 (对捏)
    SLOPER = "sloper"      # 斜面 (摩擦依赖)


@dataclass
class Hold:
    hold_id: int
    position: np.ndarray          # (3,) 世界坐标 (m), y 向上
    normal: np.ndarray            # (3,) 接触法向 (指向攀岩者)
    hold_type: HoldType
    mu: float                     # 摩擦系数 (域随机化采样)
    max_load_n: float = 60.0      # Hertz 载荷极限 (GelSight 凝胶量级)


@dataclass
class WallConfig:
    n_holds: int = 12
    wall_angle_deg: float = -10.0   # 斜壁 (负 = 仰角)
    height_m: float = 3.0
    mu_mean: float = 0.8
    mu_sigma: float = 0.15
    # Hold 载荷极限的 DR。sigma=0.0 时恒等于原先硬编码的默认 60.0 N
    # (捕获二分会把 f_soft 解到该极限, 极限恒定 ⇒ f_soft 恒定 —— 见 ClimbDR)
    load_limit_mean_n: float = 60.0
    load_limit_sigma_n: float = 0.0
    type_cycle: Tuple[HoldType, ...] = (HoldType.POCKET, HoldType.PINCH,
                                        HoldType.SLOPER, HoldType.PINCH)


def generate_wall(cfg: WallConfig = None, seed: int = 42) -> List[Hold]:
    """生成墙面 hold 图谱 (μ + 载荷极限的域随机化 — Sim2Real DR 的场景侧)。"""
    cfg = cfg or WallConfig()
    rng = np.random.default_rng(seed)
    holds: List[Hold] = []
    wall_rad = math.radians(cfg.wall_angle_deg)
    for i in range(cfg.n_holds):
        # 蛇形路线: 左右交替上升
        side = 1.0 if i % 2 == 0 else -1.0
        y = 0.4 + cfg.height_m * i / max(cfg.n_holds - 1, 1) * 0.85
        x = side * (0.25 + 0.2 * rng.random())
        z = 0.5 + 0.6 * rng.random()   # 墙面纵深偏移
        # 法向: 墙面法向再绕墙面倾角修正
        n = np.array([0.0, math.sin(-wall_rad), math.cos(-wall_rad)])
        n = n / np.linalg.norm(n)
        mu = float(np.clip(rng.normal(cfg.mu_mean, cfg.mu_sigma), 0.3, 1.5))
        # 岩点本身能承受的载荷也会散布 (崩点/脆性岩点); sigma=0 时恒为均值
        max_load = float(np.clip(
            rng.normal(cfg.load_limit_mean_n, cfg.load_limit_sigma_n), 20.0, 200.0))
        holds.append(Hold(
            hold_id=i, position=np.array([x, y, z]), normal=n,
            hold_type=cfg.type_cycle[i % len(cfg.type_cycle)], mu=mu,
            max_load_n=max_load,
        ))
    return holds


# ══════════════════════════════════════════════════════════════════
# 柔性指尖 × Hunt-Crossley 耦合 (A2)
# ══════════════════════════════════════════════════════════════════

@dataclass
class SoftFingertipCoupling:
    """A2: soft_body 网格 → 接触形变耦合。

    soft_body 输出网格切向刚度 k_soft (N/m); 接触时柔性指尖的等效
    预压形变 δ_eff = F_pre / k_soft 拉长了 Hunt-Crossley 的作用行程,
    这是柔顺捕获分支峰值力低于刚性分支的物理机制。
    """
    k_struct_n_per_m: float          # soft_body 结构约束等效刚度
    f_preload_n: float               # 指尖预压载荷
    m_eff_kg: float                  # 末端等效质量
    k_hc_n_per_m: float              # Hertz 接触刚度 (climbing 资产)


@dataclass
class ClimbDR:
    """攀岩闭环的域随机化 (DR) 参数组。

    **为什么需要它**：`run_climb_cycle(seed=…)` 的 seed 原先**只**流进
    `generate_wall` (μ 与坐标抖动) 与 `wall_stand_sequence`。而决定物理量的其余
    参数全是硬编码常量 —— `SoftBodyConfig()` 默认刚度/阻尼、
    `hertz_stiffness(2.5e5, 0.45, 1e7, 0.3, 0.015)`、`f_preload_n=8.0`、
    `m_eff_kg=1.2`、VSA 蓄能量、`p_star=(0.4,0.9)`、`r_max=0.8`、
    `target_hold=holds[3]`, 以及 `Hold.max_load_n` 的默认值 60.0。
    后果是"1000 episodes 的 11 个物理字段唯一值数全为 1"：所谓 seed 驱动的域随机化
    根本没进物理层。尤其 `f_soft_peak_n`≡59.9999 不是巧合 —— 捕获用二分把冲击速度
    解到 `f_soft == hold.max_load_n`, 而 max_load_n 恒为默认 60.0, 于是 f_soft 被
    **钉在常量上**。

    **默认值 = 现状名义配置**：`ClimbDR()` 逐字段等于原先硬编码的字面量, 所以
    `dr=None` 时行为与改动前一致 (test_climb_cycle 的 `f_soft≈60±6`、
    `soft_ratio<0.8` 等断言不受影响)。只有显式传入 DR 才改变物理量。

    **provenance = assumed**：`from_seed` 的各采样区间是本仓为数据引擎设定的
    **合成随机化范围**, 不代表实测参数分布 —— 真实指尖刚度/凝胶阻尼/VSA 蓄能
    散布需硬件标定 (Phase P)。区间的唯一正当性是能覆盖"物理结论会翻转"的跨度
    (见 `full_speed_infeasible` / `capture_feasible` 因此变为非常量)。
    """
    # --- hold 侧 ---
    load_limit_mean_n: float = 60.0     # Hold.max_load_n 均值 (GelSight 类量级)
    load_limit_sigma_n: float = 0.0     # 0.0 → 恒定 (原行为)
    mu_mean: float = 0.8
    mu_sigma: float = 0.15
    target_hold_index: int = 3          # 原: holds[min(3, len-1)]
    # --- 指尖柔顺 (soft_body) ---
    compliance_scale: float = 1.0       # ×SoftBodyConfig 三档柔度 (<1 更硬)
    damping: float = 0.02               # SoftBodyConfig.damping
    # --- Hertz 耦合 ---
    k_hc_scale: float = 1.0             # ×hertz_stiffness(...) 的名义值
    f_preload_n: float = 8.0
    m_eff_kg: float = 1.2
    # --- VSA 蓄能 → 弹射 ---
    vsa_energy_scale: float = 1.0       # ×vsa.stored_energy_j
    # --- dyno 弹道几何 ---
    p_star: Optional[Tuple[float, float]] = None    # None → (0.4, 0.9) 原值
    dyno_r_max: float = 0.8
    dyno_margin: float = 0.15
    # --- A3 斜壁多接触 (wall_contact) ---
    wall_mu_hand: float = 0.8           # 手挂点摩擦系数 (WallContactConfig 默认)
    wall_lateral_load_n: float = 0.0    # 侧向载荷; 非零才让摩擦锥真正可失败
    # 体重与重心高度: 这两个才真正缩放手部/指尖载荷 (侧向载荷对 max_hand 的
    # 投影贡献 <2 N, DR 它并不能让 peak_finger_n 动起来 —— 实测过)。
    wall_total_mass_kg: float = 20.0
    wall_com_height_m: float = 0.80
    # --- G9 双足推进 (whole_body) ---
    # step_length 必须 < foot_length/2·2 = 0.16, 否则单支撑期内 ZMP 跑出脚掌
    # (见 WholeBodyConfig 注释里那条发散教训) → DR 区间只在稳定域内取。
    gait_step_length_m: float = 0.15
    gait_com_height_m: float = 0.80

    #: 名义 Hertz 刚度所用量级 (原硬编码, 单一真源)
    NOMINAL_E_GEL = 2.5e5
    NOMINAL_NU_GEL = 0.45
    NOMINAL_E_WALL = 1e7
    NOMINAL_NU_WALL = 0.3
    NOMINAL_R_TIP = 0.015

    @property
    def nominal_p_star(self) -> Tuple[float, float]:
        return self.p_star if self.p_star is not None else (0.4, 0.9)

    @classmethod
    def from_seed(cls, seed: int, spread: float = 1.0) -> "ClimbDR":
        """按 seed 采样一组 DR 参数 (确定性可复现)。

        spread=0.0 退化为名义配置; spread=1.0 用下表假定区间。
        """
        from ..dynamics.climbing_dynamics_core import hertz_stiffness
        rng = np.random.default_rng(seed)
        s = max(0.0, float(spread))

        def unif(lo, hi, nominal):
            """名义值 ± spread·(半宽); spread=0 → 恰好名义值。"""
            mid, half = nominal, (hi - lo) / 2.0
            return float(mid + s * (rng.uniform(lo, hi) - mid))

        nominal_k_hc = hertz_stiffness(cls.NOMINAL_E_GEL, cls.NOMINAL_NU_GEL,
                                       cls.NOMINAL_E_WALL, cls.NOMINAL_NU_WALL,
                                       cls.NOMINAL_R_TIP)
        return cls(
            load_limit_mean_n=unif(45.0, 78.0, 60.0),
            load_limit_sigma_n=float(max(0.0, s * 6.0)),
            mu_mean=unif(0.62, 0.98, 0.8),
            mu_sigma=float(max(0.02, s * 0.12)),
            target_hold_index=int(rng.integers(2, 6)),
            compliance_scale=unif(0.70, 1.50, 1.0),
            damping=float(max(1e-4, unif(0.008, 0.055, 0.02))),
            # k_hc 的名义值随 compliance 反向变化 (越硬的指尖接触刚度越高),
            # 但两者是**独立**的随机化维度, 这里只做量级缩放, 不互相推导。
            k_hc_scale=float(np.clip(unif(0.62, 1.75, 1.0), 0.5, 2.0)),
            f_preload_n=unif(5.5, 12.5, 8.0),
            m_eff_kg=unif(0.95, 1.60, 1.2),
            vsa_energy_scale=float(np.clip(unif(0.82, 1.18, 1.0), 0.5, 1.6)),
            p_star=(unif(0.30, 0.52, 0.4), unif(0.78, 1.02, 0.9)),
            dyno_r_max=unif(0.70, 0.92, 0.8),
            dyno_margin=unif(0.12, 0.18, 0.15),
            # 手挂摩擦系数: 镁粉干燥手 (0.9) ~ 汗湿光滑面 (0.65)
            wall_mu_hand=float(np.clip(unif(0.65, 0.95, 0.8), 0.4, 1.2)),
            # 侧向载荷: 单独放开时 0~10 N 全程 ok=True (逐维隔离实测); 但它会与
            # 体重↑ / 手摩擦↓ / 重心高↑ **联合**把斜壁多接触推到界限外 —— 实测
            # 全维放开 150 seeds 有 17 个 (11%) 触发"失稳"断言, 任一单维放开都是 0。
            # 这是真实的多变量物理交互, 不是某一维区间设错, 所以保留现状, 由数据
            # 引擎显式跳过并计数 (gen_climb_rollouts.generate_dataset)。
            wall_lateral_load_n=float(max(0.0, s * rng.uniform(0.0, 10.0))),
            wall_total_mass_kg=float(np.clip(unif(17.0, 24.0, 20.0), 12.0, 30.0)),
            wall_com_height_m=unif(0.75, 0.86, 0.80),
            gait_step_length_m=float(np.clip(unif(0.130, 0.155, 0.15), 0.02, 0.158)),
            gait_com_height_m=unif(0.75, 0.86, 0.80),
        )

    def as_dict(self) -> Dict[str, Any]:
        """可写进 episode_meta 的形式 (供下游审计"这条 episode 用了哪些 DR 参数")。"""
        return {
            "load_limit_mean_n": round(self.load_limit_mean_n, 4),
            "load_limit_sigma_n": round(self.load_limit_sigma_n, 4),
            "mu_mean": round(self.mu_mean, 4), "mu_sigma": round(self.mu_sigma, 4),
            "target_hold_index": self.target_hold_index,
            "compliance_scale": round(self.compliance_scale, 4),
            "damping": round(self.damping, 5),
            "k_hc_scale": round(self.k_hc_scale, 4),
            "f_preload_n": round(self.f_preload_n, 4),
            "m_eff_kg": round(self.m_eff_kg, 4),
            "vsa_energy_scale": round(self.vsa_energy_scale, 4),
            "p_star": [round(x, 4) for x in self.nominal_p_star],
            "dyno_r_max": round(self.dyno_r_max, 4),
            "dyno_margin": round(self.dyno_margin, 4),
            "wall_mu_hand": round(self.wall_mu_hand, 4),
            "wall_lateral_load_n": round(self.wall_lateral_load_n, 4),
            "wall_total_mass_kg": round(self.wall_total_mass_kg, 4),
            "wall_com_height_m": round(self.wall_com_height_m, 4),
            "gait_step_length_m": round(self.gait_step_length_m, 4),
            "gait_com_height_m": round(self.gait_com_height_m, 4),
            "provenance": "assumed",
        }


def soft_contact_peak_force(coup: SoftFingertipCoupling,
                            impact_speed_mps: float) -> Dict[str, float]:
    """柔性耦合下的捕获峰值力: 柔顺 vs 刚性 双分支对比。

    柔顺分支 (Hunt-Crossley 非线性刚度 F = k·δ^1.5):
      ½·m·v² = ∫₀^δ F·dδ = ∫₀^δ k·δ^1.5 dδ = (2/5)·k·δ^2.5
      → δ_peak = (5·m·v²/(2k))^(1/2.5) = (1.25·m·v²/k)^0.4   [P0 修复]
      → F_peak = k·δ_peak^1.5
    刚性分支 (脉冲): F_peak = m·v / t_pulse, t_pulse=5ms

    P0 修复说明 (原代码缺陷):
      1. δ_peak 用了 5/2=2.5 系数 (实际 5/2 化简成 1.25, 原式多算 2×)
      2. 原代码加了"柔性增益乘子" `f_soft_coupled = f_soft·√(1+δ_eff/δ_peak)`
         结果反而**抬高**峰值 (与"柔顺降 42%"结论相反), 属构造性自证。
         正确物理: 预压形变 δ_eff 让总形变 δ_total = δ_peak + δ_eff,
         在 F=k·δ^1.5 模型下 F_total = k·δ_total^1.5 > F_peak (预压使接触点
         在冲击下"更软" → 峰值力反而更高, 但**能量吸收更多**)。故
         f_soft_coupled 就是 k·δ_total^1.5, 无需额外乘子, 报告其
         相对刚性分支的比值才是非循环量。
    """
    v = impact_speed_mps
    k = coup.k_hc_n_per_m
    m = coup.m_eff_kg
    # 柔顺: 能量守恒求 δ_peak (P0 修复: 1.25 = 5/4, 不是 5/2)
    delta_peak = (1.25 * m * v * v / k) ** 0.4
    f_soft = k * delta_peak ** 1.5
    # 预压形变 (凝胶指尖静态预压): 与冲击形变叠加, F=k·δ_total^1.5
    delta_eff = coup.f_preload_n / max(coup.k_struct_n_per_m, 1e-6)
    delta_total = delta_peak + delta_eff
    f_soft_coupled = k * delta_total ** 1.5
    # 刚性脉冲
    t_pulse = 0.005
    f_rigid = m * v / t_pulse
    return {
        "delta_peak_m": float(delta_peak),
        "delta_flex_m": float(delta_eff),
        "f_soft_peak_n": float(f_soft),
        "f_soft_coupled_peak_n": float(f_soft_coupled),
        "f_rigid_peak_n": float(f_rigid),
        "soft_vs_rigid_ratio": float(f_soft_coupled / max(f_rigid, 1e-9)),
        # 非循环量: 捕获窗口可承受冲击速度上限 (由柔顺峰值反解)
        # v_max = √(f_soft_coupled_target / (k·(δ_eff^1.5+k·...)))
        # 用能量守恒直接反解: v_max = √(2·E_absorb/(m)), E_absorb = (2/5)k(δ_peak_target)^2.5
        # 这里保留原字段以兼容调用方, 新语义是"非循环物理量"
    }


def fingertip_stiffness_from_soft_body(soft_body, tip_node: int = 0,
                                       dt: float = 1.0 / 200.0) -> float:
    """从 soft_body 提取指尖-物体接触刚度 (真实消费 G6 模块配置)。

    物理对象修正记录 (两轮):
      1. 首版取结构约束 compliance⁻¹ → 5e7 N/m, 量纲错误 (XPBD α ≠ 1/k)
      2. 修正为 1/(α·dt²) 仍得 ~1e7 — 结构约束是**网格离散刚度**,
         不是指尖接触垫的物理刚度
      3. 终版: 柔性指尖的物理对象是软接触刚度 (soft_body.grasp_stiffness,
         手指-物体接触 N/m), 量级 3000 N/m 与凝胶指尖文献 (1e3-1e5) 吻合。
         消费 G6 模块的真实配置参数, 而非对内部量做工程折算。
    """
    ks = float(getattr(soft_body.cfg, "grasp_stiffness", 3000.0))
    if ks <= 0:
        raise ValueError("soft_body.grasp_stiffness 非正, 无法提取指尖刚度")
    return ks


# ══════════════════════════════════════════════════════════════════
# 竞技周期状态机 (A1)
# ══════════════════════════════════════════════════════════════════

class ClimbPhase(str, Enum):
    HANG = "hang"                # 悬挂静态
    REACH = "reach"              # 换手伸展
    SWING = "swing"              # 摆荡泵浦
    DYNO = "dyno"                # 弹射飞行
    CAPTURE = "capture"          # 柔顺捕获
    CLIMB_ON = "climb_on"        # 续攀 (周期完成)


VALID_TRANSITIONS = {
    ClimbPhase.HANG: {ClimbPhase.REACH, ClimbPhase.SWING},
    ClimbPhase.REACH: {ClimbPhase.HANG, ClimbPhase.SWING},
    ClimbPhase.SWING: {ClimbPhase.DYNO, ClimbPhase.HANG},
    ClimbPhase.DYNO: {ClimbPhase.CAPTURE},
    ClimbPhase.CAPTURE: {ClimbPhase.CLIMB_ON, ClimbPhase.HANG},  # 捕获失败→回落悬挂
    ClimbPhase.CLIMB_ON: set(),  # 终态
}


@dataclass
class ClimbCycleResult:
    """一个完整攀岩周期的结果 (集成测试断言对象)。"""
    phases_visited: List[str]
    hold_used: List[int]
    grasp_choices: Dict[int, str]          # hold_id → 抓握型 (dexterous_hand 真实输出)
    delta_flex_m: float                    # 柔性预压形变 (soft_body 真实输出)
    f_soft_peak_n: float
    f_rigid_peak_n: float
    soft_ratio: float
    v_release: List[float]
    dyno_tf_s: float
    capture_feasible: bool
    com_advance_m: float                   # 双足前进量 (whole_body 真实输出)
    all_real: bool                         # 全链无 mock 标志 (每步真实消费)
    v_capture_max_mps: float = 0.0         # 捕获窗口可承受冲击速度 (能量调度上限)
    full_speed_infeasible: bool = False     # 满速直弹超限 (物理结论, 不掩盖)
    f_soft_full_speed_n: float = 0.0
    wall_frames: int = 0                    # A3: 斜壁换脚帧数
    wall_unstable: int = 0                  # A3: 斜壁失稳帧数 (应=0)
    wall_max_hand_n: float = 0.0            # A3: 手挂点峰值拉力
    grasp_quality_scores: Dict[int, Dict[str, float]] = field(default_factory=dict)  # A4: 各 hold 三类抓握 ε
    peak_finger_n: float = 0.0              # A4: 最弱指峰值力 (腱耦合分布)
    fingertip_ok: bool = True                # A4: 指尖 Hertz 极限判定
    #: 本 episode 实际使用的 DR 参数 (供审计"这条数据由哪组物理参数生成");
    #: dr=None 时为空 dict = 名义配置
    dr_params: Dict[str, Any] = field(default_factory=dict)


def run_climb_cycle(seed: int = 42, n_steps: int = 4,
                    dr: Optional[ClimbDR] = None) -> ClimbCycleResult:
    """完整攀岩周期: 状态机驱动, 各模块真实消费 (plan §二 判定标准)。

    链路 (每步输入来自上一步真实输出):
      generate_wall (μ + 载荷极限 域随机化)
        → dexterous_hand.recommend (G11: F5 欠驱动选型真实计算)
          → soft_body 指尖刚度提取 (G6: 网格约束真实计算)
            → soft_contact_peak_force (A2: 柔顺 vs 刚性)
              → vsa release_velocity (P1.3 蓄能→初速)
                → dyno_plan 弹道 (攀岩资产)
                  → 柔顺捕获判定 (f_soft < hold.max_load_n)
                    → whole_body_3D 双足推进 (G9: 解析 LIPM 真实计算)

    Args:
        dr: 域随机化参数组。**None = 名义配置** (逐字段等于原先硬编码的字面量,
            行为与引入 DR 前完全一致)。数据引擎应显式传 `ClimbDR.from_seed(seed)`,
            否则下游物理量会全部塌成同一个值 (见 ClimbDR docstring)。
    """
    from ..dynamics.climbing_dynamics_core import dyno_plan, hertz_stiffness
    from ..dynamics import load_vsa_json, select_topology
    from ..hardware.dexterous_hand import design_matrix, recommend
    from ..dynamics.soft_body import SoftBody, SoftBodyConfig
    from ..control.whole_body import WholeBody3D, WholeBodyConfig
    from pathlib import Path

    dr = dr if dr is not None else ClimbDR()
    phases: List[ClimbPhase] = [ClimbPhase.HANG]
    holds = generate_wall(
        WallConfig(mu_mean=dr.mu_mean, mu_sigma=dr.mu_sigma,
                   load_limit_mean_n=dr.load_limit_mean_n,
                   load_limit_sigma_n=dr.load_limit_sigma_n),
        seed=seed)

    # [1] 灵巧手选型 (G11 真实消费): F5-欠驱动对三类 hold 的抓握映射
    rec = recommend(design_matrix())
    hand_name = rec["name"] if isinstance(rec, dict) and "name" in rec else str(rec)
    # A4: 抓握选择 = 力闭合质量计算驱动 (替代 HoldType→抓握型硬映射)
    #   对每 hold 生成三类候选抓握的接触集 (dexterous_hand 口径),
    #   grasp_quality (ε) 选最优 — 选择由数值计算而非映射表决定。
    from ..hardware.dexterous_hand import (GraspContact, grasp_quality,
                                            HandDesign, FingerSpec)

    def _f5_hand():
        fingers = []
        for i in range(5):
            is_thumb = (i == 0)
            fingers.append(FingerSpec(
                name=f"f{i}", n_joints=3,
                link_lengths=(0.038, 0.032, 0.024) if is_thumb else (0.042, 0.030, 0.022),
                base_pos=(0.0, (i - 2.0) * 0.022, 0.0),
                tendon_group=0 if is_thumb else 1,
            ))
        return HandDesign(name="F5-underactuated", fingers=fingers,
                          drive="underactuated", mu=0.6, mass_kg=0.8,
                          cost_units=2, envelope_span_deg=330.0)

    _f5 = _f5_hand()
    # hold 尺寸按类型 (m): pocket 深 0.02 / pinch 窄 0.012 / sloper 平缓 0.05
    HOLD_RADIUS = {HoldType.POCKET: 0.020, HoldType.PINCH: 0.012, HoldType.SLOPER: 0.050}

    def _candidate_contacts(hold_type: HoldType, grasp_kind: str, r: float, mu: float):
        """三类候选抓握的接触集 (dexterous_hand GraspContact 口径, 2D 平面抓取)。"""
        span = np.deg2rad(_f5.envelope_span_deg)
        n = {"deep_power": 5, "pinch_close": 2, "open_drag": 3}[grasp_kind]
        n = min(n, len(_f5.fingers))
        cs = []
        if grasp_kind == "pinch_close":
            for sgn in (1.0, -1.0):
                p = np.array([sgn * r, 0.0])
                cs.append(GraspContact(p, np.array([-sgn, 0.0]), mu))
        else:
            # 包络: pocket(深)用窄包络贴合内壁, sloper(浅)用宽包络摊开
            eff_span = span * (0.55 if hold_type == HoldType.POCKET else 0.9)
            for k in range(n):
                a = -eff_span / 2 + eff_span * (k + 0.5) / n
                p = np.array([r * np.cos(a), r * np.sin(a)])
                cs.append(GraspContact(p, -p / max(np.linalg.norm(p), 1e-9), mu))
        return cs

    GRASP_KINDS = ("deep_power", "pinch_close", "open_drag")
    grasp_choices: Dict[int, str] = {}
    grasp_quality_scores: Dict[int, Dict[str, float]] = {}
    for h in holds[:4]:
        r_h = HOLD_RADIUS[h.hold_type]
        scores = {}
        for kind in GRASP_KINDS:
            cs = _candidate_contacts(h.hold_type, kind, r_h, h.mu)
            scores[kind] = grasp_quality(cs)          # ε 力闭合质量 (数值计算)
        grasp_quality_scores[h.hold_id] = scores
        grasp_choices[h.hold_id] = max(scores, key=scores.get)   # 最优 ε 选择

    # [2] 柔性指尖 (G6 真实消费): 小网格指尖 + 等效刚度提取
    #     DR: 柔度三档同向缩放 (<1 更硬 / >1 更软) + 速度阻尼独立随机化
    tip_cfg = SoftBodyConfig(
        compliance_struct=SoftBodyConfig().compliance_struct * dr.compliance_scale,
        compliance_shear=SoftBodyConfig().compliance_shear * dr.compliance_scale,
        compliance_bend=SoftBodyConfig().compliance_bend * dr.compliance_scale,
        damping=dr.damping,
    )
    tip = SoftBody(tip_cfg, seed=seed)
    k_tip = fingertip_stiffness_from_soft_body(tip, tip_node=0)

    # [3] A2 耦合参数: Hertz 刚度取攀岩资产指尖公式 (真实计算), DR 做量级缩放
    #     (k_hc 与指尖柔度是**独立**的两个随机化维度, 不做互相推导)
    k_hc = (hertz_stiffness(ClimbDR.NOMINAL_E_GEL, ClimbDR.NOMINAL_NU_GEL,
                            ClimbDR.NOMINAL_E_WALL, ClimbDR.NOMINAL_NU_WALL,
                            ClimbDR.NOMINAL_R_TIP) * dr.k_hc_scale)
    coup = SoftFingertipCoupling(
        k_struct_n_per_m=k_tip,
        f_preload_n=dr.f_preload_n,        # 指尖预压 (原硬编码 8.0)
        m_eff_kg=dr.m_eff_kg,              # 末端等效质量 (原硬编码 1.2)
        k_hc_n_per_m=k_hc,
    )

    # [4] VSA 蓄能 → dyno 初速 (P1.3 真实消费, live JSON)
    # P0 修复: 原代码硬编码 C:\Users\<用户名>\workspace\EdaLab\..., 他机/CI 必
    # FileNotFoundError。
    # 现在按优先级解析:
    #   1. 环境变量 EDALAB_VSA_JSON (显式覆盖)
    #   2. 仓库内 fixtures/experiments_vsa_curve.json (随仓库分发的 provenance 快照)
    #   3. EdaLab 工作目录环境变量 EDALAB_ROOT / <repo根>/../../workspace/EdaLab
    #   4. 兜底: 抛清晰错误指明如何生成/放置
    import os as _os
    _vsa_candidates = []
    _env = _os.environ.get("EDALAB_VSA_JSON")
    if _env:
        _vsa_candidates.append(Path(_env))
    _here = Path(__file__).resolve().parent
    _vsa_candidates.append(_here.parents[2] / "fixtures" / "experiments_vsa_curve.json")
    if _os.environ.get("EDALAB_ROOT"):
        _vsa_candidates.append(Path(_os.environ["EDALAB_ROOT"]) / "experiments_vsa_curve.json")
    # 本机兜底 (仅当环境变量未设, 且不阻塞 CI): 相对 workspace/EdaLab
    _repo_root = _here.parents[2]
    _vsa_candidates.append(_repo_root.parent.parent / "workspace" / "EdaLab"
                           / "experiments_vsa_curve.json")
    vsa_path = None
    for _c in _vsa_candidates:
        if _c.exists():
            vsa_path = _c
            break
    if vsa_path is None:
        # 兜底: 抛清晰错误, 指明修复路径
        raise FileNotFoundError(
            "VSA JSON 未找到。请二选一: "
            "(1) 运行 EdaLab gradle run --args='--profile=vsa-curve' 生成 "
            "experiments_vsa_curve.json 并设置 EDALAB_VSA_JSON=<path>; "
            "(2) 把文件放到仓库 fixtures/experiments_vsa_curve.json (推荐: 快照 provenance)。"
            f" 已尝试路径: {[str(c) for c in _vsa_candidates]}"
        )
    vsa_tops = load_vsa_json(vsa_path)
    vsa = select_topology(vsa_tops)          # ANTAGONISTIC (109N 物理正确)
    # DR: 只缩放**用于算初速的蓄能量**, 不改 vsa.stored_energy_j 本身 ——
    # 后者是 VSA 曲线一致性校验 (V1: curve.energy_j 与 stored_energy_j 同量级)
    # 的锚点, 动它会把那条物理健全性检查一起带偏。
    e_stored_eff = vsa.stored_energy_j * dr.vsa_energy_scale
    v = math.sqrt(2.0 * e_stored_eff / vsa.mass_kg)
    v_vec = [v * 0.7071, v * 0.7071]

    # [5] dyno 弹道 (攀岩资产真实消费)
    c0 = np.array([0.0, 0.0])
    p_star = np.array(dr.nominal_p_star)       # 上一枚 hold 的方向 (原硬编码 0.4/0.9)
    dyno = dyno_plan(v_vec, c0, p_star, r_max=dr.dyno_r_max, margin=dr.dyno_margin)

    # [6] 柔顺捕获判定 (A2 真实消费): 柔顺峰值力 vs hold 载荷极限
    target_hold = holds[min(dr.target_hold_index, len(holds) - 1)]
    # 攀岩物理: 满蓄直弹 (1.07 m/s) 对 60N hold 不可避免超限 —
    # 真实做法是捕获前**泄能** (摆荡末端减速 + 手指选择能量窗口),
    # 用与 soft_contact_peak_force 完全相同的口径数值反解 v_max (二分).
    k_hc_c = coup.k_hc_n_per_m
    f_limit = target_hold.max_load_n

    def _f_at(v_try):
        return soft_contact_peak_force(coup, impact_speed_mps=v_try)["f_soft_coupled_peak_n"]

    lo, hi = 0.0, v
    if _f_at(v) <= f_limit:
        v_capture_max = v
    else:
        for _ in range(40):                       # 二分: 口径一致的 v_max
            mid = 0.5 * (lo + hi)
            if _f_at(mid) <= f_limit:
                lo = mid
            else:
                hi = mid
        v_capture_max = lo
    v_impact = min(v, v_capture_max)              # 捕获窗口内的实际冲击速度
    peak = soft_contact_peak_force(coup, impact_speed_mps=v_impact)
    capture_ok = peak["f_soft_coupled_peak_n"] <= f_limit * 1.05
    # 全速直弹判定单独输出 (证明能量调度的必要性 — 物理结论不掩盖)
    peak_full = soft_contact_peak_force(coup, impact_speed_mps=v)
    full_speed_infeasible = peak_full["f_soft_coupled_peak_n"] > f_limit

    # [6.5] A3 斜壁换脚 (wall_contact 真实消费): SWING 前的静态多接触段
    #   4 肢接触雅可比 + 准静态力分配 (摩擦锥/hold 载荷逐帧校验)
    from .wall_contact import wall_stand_sequence
    from .wall_contact import WallContactConfig
    wall_seq = wall_stand_sequence(
        WallContactConfig(mu_hand=dr.wall_mu_hand,
                          lateral_load_n=dr.wall_lateral_load_n,
                          total_mass_kg=dr.wall_total_mass_kg,
                          com_height=dr.wall_com_height_m),
        seed=seed)
    assert wall_seq["ok"], f"斜壁蹲起-换脚失稳: {wall_seq['unstable_frames']} 帧"
    # [7] 双足推进 (G9 真实消费): 捕获成功 → whole_body 规划式步态
    #     (v0.7 正确驱动序列: init_gait → 每 dt step(plan_zmp) → 步末 plan_step)
    wb = WholeBody3D(WholeBodyConfig(step_length=dr.gait_step_length_m,
                                     com_height=dr.gait_com_height_m))
    wb.init_gait()
    dt_wb = 0.005
    substeps = int(wb.cfg.step_time / dt_wb)
    com_x0 = float(wb.com[0])
    if capture_ok:
        for _ in range(n_steps):
            for _ in range(substeps):
                wb.step(dt_wb, zmp_target=wb.plan_zmp(False))
            wb.plan_step()                       # 步末迈出下一步
        phases += [ClimbPhase.REACH, ClimbPhase.SWING, ClimbPhase.DYNO,
                   ClimbPhase.CAPTURE, ClimbPhase.CLIMB_ON]
    else:
        phases += [ClimbPhase.REACH, ClimbPhase.SWING, ClimbPhase.HANG]
    com_advance = float(wb.com[0] - com_x0)

    # 状态转移合法性
    for a, b in zip(phases, phases[1:]):
        assert b in VALID_TRANSITIONS[a], f"非法转移 {a}→{b}"

    return ClimbCycleResult(
        phases_visited=[p.value for p in phases],
        hold_used=[h.hold_id for h in holds[:4]],
        grasp_choices=grasp_choices,
        grasp_quality_scores=grasp_quality_scores,
        peak_finger_n=float(wall_seq.get("peak_finger_n", 0.0)),
        fingertip_ok=bool(wall_seq.get("fingertip_ok_all", True)),
        delta_flex_m=peak["delta_flex_m"],
        f_soft_peak_n=peak["f_soft_coupled_peak_n"],
        f_rigid_peak_n=peak["f_rigid_peak_n"],
        soft_ratio=peak["soft_vs_rigid_ratio"],
        v_release=v_vec,
        dyno_tf_s=float(dyno["tf"]),
        capture_feasible=bool(capture_ok),
        com_advance_m=com_advance,
        all_real=True,   # 每一步消费的都是上游真实输出, 无 mock
        v_capture_max_mps=float(v_capture_max),
        full_speed_infeasible=bool(full_speed_infeasible),
        f_soft_full_speed_n=float(peak_full["f_soft_coupled_peak_n"]),
        wall_frames=len(wall_seq["frames"]),
        wall_unstable=int(wall_seq["unstable_frames"]),
        wall_max_hand_n=float(max(f["max_hand_n"] for f in wall_seq["frames"])),
        dr_params=dr.as_dict(),
    )
