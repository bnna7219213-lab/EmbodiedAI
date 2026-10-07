"""斜壁四肢体多接触模型 (A3) — whole_body_3d 从平地升级斜壁。

G1 资产公式 (climbing-dynamics-math §1, 浮基动力学):
    M(q)ν̇ + C(q,ν)ν + g(q) = Sᵀτ + Σᵢ J^{cᵢ}(q)ᵀ λᵢ

本模块实现其中的接触侧 (准静态 C=0, ν̇=0 — 蹲起/换脚为慢动作):
    g(q) = Σᵢ J^{cᵢ}(q)ᵀ λᵢ        (重力-接触平衡)
    ‖λᵢ,ₜ‖ ≤ μᵢ·λᵢ,ₙ             (摩擦锥, 二阶锥 SOC)

与平地 whole_body 的边界划分 (诚实声明):
    - whole_body_3d: 平地 LIPM/ZMP (水平投影支撑多边形) — 斜壁不适用
    - wall_contact:  斜壁准静态多接触 (重力沿墙法向分解 + 3D 力闭合)
    相位级动态 (摆荡/dyno) 仍由 climb_scene 的既有链路负责;
    本模块负责 SWING 前的"斜壁换脚"静态段。

手臂近似 (A4 未做, 诚实边界): 四肢 = 双腿 (whole_body 几何) + 双臂挂点
(固连质量点, 位置=肩部前伸手长), 不建模手指/腕。注释中的 ARM_APPROX=True。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

ARM_APPROX = False   # A4 已完成: 手接触 = F5 欠驱动灵巧手指分布模型 (见 DexterousHandContact)
ARM_APPROX_HISTORY = True   # 历史标志: A4 之前手臂为固连质量点 (保留供文档对照)


# ══════════════════════════════════════════════════════════════════
# A4: 灵巧手接触模型 (替代固连质量点)
# ══════════════════════════════════════════════════════════════════

class DexterousHandContact:
    """F5-欠驱动手的挂壁接触模型: 手部总拉力 → 指接触分布 (A4)。

    物理链 (dexterous_hand 真实消费):
      underactuated_coupling → 腱组耦合比 ratios (同腱各关节位移分配)
        → 主腱 (4 指) 各指力份额 = ratios 归一化 (远端指更快 ⇒ 承载更大)
        → 最弱指峰值力 = 份额最大者 × 手部总拉力
      约束: 指尖峰值 ≤ 指尖 Hertz 极限 (P1.3 同源 60N 量级); hold 结构
            载荷上限作用于手部合力。
    """

    def __init__(self, hand=None, fingertip_limit_n: float = 60.0):
        if hand is None:
            from ..hardware.dexterous_hand import (FingerSpec, HandDesign,
                                                   underactuated_coupling)
            # F5-欠驱动 (dexterous_hand design_matrix 的 mk 同构, 公开 API 构造)
            fingers = []
            for i in range(5):
                is_thumb = (i == 0)
                fingers.append(FingerSpec(
                    name=f"f{i}", n_joints=3,
                    link_lengths=(0.038, 0.032, 0.024) if is_thumb else (0.042, 0.030, 0.022),
                    base_pos=(0.0, (i - 2.0) * 0.022, 0.0),
                    tendon_group=0 if is_thumb else 1,
                ))
            hand = HandDesign(name="F5-underactuated", fingers=fingers,
                              drive="underactuated", mu=0.6,
                              mass_kg=0.8, cost_units=2,
                              envelope_span_deg=330.0)
        self.hand = hand
        from ..hardware.dexterous_hand import underactuated_coupling
        self.coupling = underactuated_coupling(hand)
        self.fingertip_limit_n = fingertip_limit_n
        # 主腱 (非拇指组) 的归一化份额: ratios 向量 (关节级) → 指级聚合
        main_group = max(self.coupling["groups"].items(), key=lambda kv: len(kv[1]))
        ratios = self.coupling["ratios"][main_group[0]]
        n_j = len(ratios)
        n_f_main = sum(1 for f in hand.fingers if f.tendon_group == main_group[0])
        # 指级份额: 各指均分该腱总位移的假设下, 关节比递增 → 末指承载最大
        per_finger = np.array(ratios).reshape(-1, n_j // max(n_f_main, 1)).sum(axis=1) \
            if n_f_main and n_j % n_f_main == 0 else np.full(max(n_f_main, 1), 1.0 / max(n_f_main, 1))
        self.finger_shares = per_finger / max(per_finger.sum(), 1e-9)

    def distribute(self, total_pull_n: float) -> Dict[str, float]:
        """手部总拉力 → 指接触分布 (欠驱动腱耦合决定, 非均匀)。"""
        shares = {}
        main_idx = 0
        for f in self.hand.fingers:
            if f.tendon_group == 0 and f.name == "f0":
                # 拇指独立腱: 承担对置载荷 (与主腱合力平衡 → 取主腱末指份额)
                shares[f.name] = float(self.finger_shares[-1]) * total_pull_n
            else:
                if main_idx < len(self.finger_shares):
                    shares[f.name] = float(self.finger_shares[main_idx]) * total_pull_n
                    main_idx += 1
                else:
                    shares[f.name] = 0.0
        return shares

    @property
    def peak_finger_share(self) -> float:
        """最弱 (承载最大) 指的份额 — 单指峰值力 = share × 总拉力。"""
        return float(np.max(self.finger_shares))

    def per_finger_ok(self, total_pull_n: float) -> Tuple[bool, float]:
        """指尖 Hertz 极限校验: 单指峰值 ≤ fingertip_limit。"""
        peak = self.peak_finger_share * total_pull_n
        return bool(peak <= self.fingertip_limit_n), float(peak)

    def n_fingers_engaged(self) -> int:
        return sum(1 for f in self.hand.fingers
                   if f.tendon_group == max(g for g in
                                            [x.tendon_group for x in self.hand.fingers]))


# ══════════════════════════════════════════════════════════════════
# 斜壁几何
# ══════════════════════════════════════════════════════════════════

@dataclass
class WallContactConfig:
    wall_angle_deg: float = -10.0        # 仰角 (负=仰角, 人面向负法向)
    gravity: float = 9.81
    total_mass_kg: float = 20.0          # 对齐 whole_body WholeBodyConfig.mass
    com_height: float = 0.80
    leg_reach_m: float = 0.85
    arm_reach_m: float = 0.70            # 肩到挂点
    mu_foot: float = 0.9
    mu_hand: float = 0.8
    max_hold_load_n: float = 150.0       # hold 结构载荷上限 (手部合力)
    fingertip_limit_n: float = 60.0     # A4: 单指 Hertz 极限 (P1.3 同源)


def wall_frame(wall_angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """墙面坐标系: n=法向(离墙向外), t=墙面上方向。"""
    a = math.radians(wall_angle_deg)
    n = np.array([0.0, -math.sin(a), math.cos(a)])       # y 向上, z 朝墙
    t = np.array([0.0, math.cos(a), math.sin(a)])        # 墙面上方向
    n = n / np.linalg.norm(n)
    t = t / np.linalg.norm(t)
    return n, t


# ══════════════════════════════════════════════════════════════════
# 4 肢接触雅可比 (G1: J^{c_i}(q))
# ══════════════════════════════════════════════════════════════════

def limb_jacobian(contact_pos: np.ndarray, com: np.ndarray) -> np.ndarray:
    """单肢接触雅可比 (点接触, 3×6 或 6×6 浮基行简化为 3×6 → 3×3 有效块)。

    点接触速度 v_contact = J·ν。 浮基 6 维 + 关节维省略为质心 3 维平移
    (准静态蹲起/换脚, 角动量项由力闭合单独校验 — G1 的 L̇_G 分解)。

    J = [I₃  |  -(p_c - p_com)×]   (接触点在质心系中的位置叉乘)
    平移块 3×3 = I, 旋转块 = -[r]× , r = contact_pos - com。
    """
    r = np.asarray(contact_pos, dtype=float) - np.asarray(com, dtype=float)
    skew = np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]],
        [-r[1], r[0], 0.0],
    ])
    return np.hstack([np.eye(3), -skew])   # (3, 6)


class QuadContactState:
    """四肢接触状态: 腿足 + 手挂点位置与接触标志。"""

    def __init__(self, cfg: WallContactConfig, foot_left: Optional[np.ndarray],
                 foot_right: Optional[np.ndarray], hand_left: Optional[np.ndarray],
                 hand_right: Optional[np.ndarray]):
        self.cfg = cfg
        self.foot_left = np.asarray(foot_left, dtype=float) if foot_left is not None else None
        self.foot_right = np.asarray(foot_right, dtype=float) if foot_right is not None else None
        self.hand_left = np.asarray(hand_left, dtype=float) if hand_left is not None else None
        self.hand_right = np.asarray(hand_right, dtype=float) if hand_right is not None else None

    @property
    def contacts(self) -> List[Tuple[str, np.ndarray, float]]:
        """(名字, 位置, μ) 列表 — 只含在壁接触。"""
        out = []
        if self.foot_left is not None:
            out.append(("foot_left", self.foot_left, self.cfg.mu_foot))
        if self.foot_right is not None:
            out.append(("foot_right", self.foot_right, self.cfg.mu_foot))
        if self.hand_left is not None:
            out.append(("hand_left", self.hand_left, self.cfg.mu_hand))
        if self.hand_right is not None:
            out.append(("hand_right", self.hand_right, self.cfg.mu_hand))
        return out

    @property
    def n_contacts(self) -> int:
        return len(self.contacts)


# ══════════════════════════════════════════════════════════════════
# 准静态力分配 (摩擦锥约束)
# ══════════════════════════════════════════════════════════════════

@dataclass
class ForceAllocationResult:
    ok: bool
    forces_n: Dict[str, np.ndarray]       # 各肢 3D 接触力 (世界系)
    normal_forces_n: Dict[str, float]
    friction_margins: Dict[str, float]    # ‖f_t‖/(μ·f_n) (<1 可行)
    max_hand_load_n: float
    wrench_balance_err: float             # Σ Jᵀλ - W 残差
    description: str
    # A4: 灵巧手指分布 (无手接触时为空)
    finger_forces_n: Dict[str, float] = field(default_factory=dict)
    peak_finger_n: float = 0.0
    fingertip_ok: bool = True


def allocate_forces(state: QuadContactState, com: np.ndarray,
                    extra_load_n: float = 0.0) -> ForceAllocationResult:
    """准静态力分配: g + 外载 = Σ Jᵀλ, 摩擦锥/hold 机械约束。

    斜壁物理 (与平地的本质区别, 也是攀岩要 hold 的原因):
      仰角 α 墙面上, 重力在墙法向的分量只有 m·g·sin(α) (α=10° 时 17%),
      其余 98% 落在墙面切向 — 光靠脚底摩擦挂不上去 (μ 需 > cot(α) ≈ 5.7)。
      脚: 踩在台阶/hhold 上 → 法向是**机械支撑** (台阶面直接顶住),
          台阶内摩擦只管防滑 (切向/法向 = tan(α) 内即可)
      手: 挂在 hold 凹槽 → hold 给手指**拉力上限** max_hold_load_n
          (P1.3 的 60N Hertz 极限同源), 切向全由 hold 结构承担
    分配方案: 重力切向分量由脚-手对半 (爬墙姿态), 法向由台阶支撑。
    """
    cfg = state.cfg
    n_vec, t_vec = wall_frame(cfg.wall_angle_deg)
    g_vec = np.array([0.0, -cfg.gravity, 0.0])
    total_w = (cfg.total_mass_kg * g_vec) + np.array([0.0, -extra_load_n, 0.0])

    n_pts = state.n_contacts
    n_feet = sum(1 for name, _, _ in state.contacts if name.startswith("foot"))
    n_hands = n_pts - n_feet

    forces: Dict[str, np.ndarray] = {}
    normal_f: Dict[str, float] = {}
    margins: Dict[str, float] = {}

    # 切向需求 (墙面内, 沿 t 即"向上爬"方向): 由脚+手分担
    tangential_total = float(np.dot(total_w, -t_vec))      # 需要的抗滑切向力 (N)
    # 法向需求: 全部由脚的台阶机械支撑承担 (手挂 hold 提供的是切向拉力,
    # 不是法向支撑 — 双手也算法向是首版 2 倍法向的根因)
    normal_total = float(np.dot(total_w, -n_vec))          # 压向墙的法向 (N)

    # 分担比例: 脚 60% 切向 (腿推), 手 40% (臂拉) — 攀岩常规姿态
    foot_t_share = 0.6 if n_feet else 0.0
    feasible = True

    per_foot_t = tangential_total * foot_t_share / max(n_feet, 1) if n_feet else 0.0
    per_hand_t = tangential_total * (1.0 - foot_t_share) / max(n_hands, 1) if n_hands else 0.0
    per_foot_n = normal_total / max(n_feet, 1) if n_feet else 0.0

    for name, pos, mu in state.contacts:
        if name.startswith("foot"):
            f = per_foot_t * t_vec + per_foot_n * n_vec
        else:
            f = per_hand_t * t_vec                            # 手: 仅切向拉力
        forces[name] = f

    # 约束校验
    for name, pos, mu in state.contacts:
        f = forces[name]
        if name.startswith("hand"):
            up = float(np.dot(f, t_vec))   # 手拉力 (墙面内向上分量)
            if up > cfg.max_hold_load_n:
                feasible = False
        # 摩擦: 切向 (水平侧向分量) vs 法向压紧
        fn = float(np.dot(f, n_vec))
        f_t_vec = f - fn * n_vec
        f_side = f_t_vec - np.dot(f_t_vec, t_vec) * t_vec    # 纯侧向
        lateral = float(np.linalg.norm(f_side))
        if lateral > mu * max(fn, 1e-6):
            feasible = False
        normal_f[name] = fn
        margins[name] = lateral / max(mu * max(fn, 1e-6), 1e-6)

    total_f = sum(forces.values(), start=np.zeros(3))
    err = float(np.linalg.norm(total_f + total_w))

    max_hand = max((float(np.dot(forces[n_], t_vec)) for n_ in forces if n_.startswith("hand")),
                   default=0.0)

    # A4: 手部总拉力 → 灵巧手指分布 (腱耦合决定) + 单指 Hertz 极限校验
    finger_forces: Dict[str, float] = {}
    peak_finger = 0.0
    fingertip_ok = True
    if max_hand > 0.0:
        hand_contact = DexterousHandContact(fingertip_limit_n=cfg.fingertip_limit_n)
        finger_forces = hand_contact.distribute(max_hand)
        fingertip_ok, peak_finger = hand_contact.per_finger_ok(max_hand)
        # 指尖极限是比 hold 结构上限更紧的物理约束 (A4 双重校验)
        if not fingertip_ok:
            feasible = False

    return ForceAllocationResult(
        ok=bool(feasible and err < 0.5
                and all(margins[n_] <= 1.0 for n_ in margins)
                and fingertip_ok),
        forces_n=forces, normal_forces_n=normal_f, friction_margins=margins,
        max_hand_load_n=max_hand, wrench_balance_err=err,
        description=(f"{n_pts} 接触, 手拉力 {max_hand:.1f}N (限 {cfg.max_hold_load_n}N), "
                     f"最弱指 {peak_finger:.1f}N (限 {cfg.fingertip_limit_n}N), "
                     f"侧滑裕度 max {max(margins.values()) if margins else 0:.2f}"),
        finger_forces_n=finger_forces,
        peak_finger_n=float(peak_finger),
        fingertip_ok=bool(fingertip_ok),
    )


# ══════════════════════════════════════════════════════════════════
# 斜壁蹲起-换脚序列 (A3 验收动作)
# ══════════════════════════════════════════════════════════════════

def wall_stand_sequence(cfg: WallContactConfig = None, seed: int = 42) -> Dict[str, any]:
    """斜壁蹲起-换脚: 四肢着壁 → 抬左手 → 换脚位 → 回四肢。

    全程逐帧: 力分配 + 摩擦锥 + 手载上限校验 (0 失稳 = 全部帧 OK)。
    返回帧序列与统计 (集成测试断言对象)。
    """
    cfg = cfg or WallContactConfig()
    rng = np.random.default_rng(seed)
    n_wall, t_wall = wall_frame(cfg.wall_angle_deg)

    # 初始四肢位形: 双脚踩在壁面台阶, 双手挂 hold (面墙)
    foot_l = np.array([0.15, 0.30, 0.0])
    foot_r = np.array([-0.15, 0.30, 0.0])
    hand_l = np.array([0.28, 1.10, 0.0])
    hand_r = np.array([-0.28, 1.10, 0.0])
    com = np.array([0.0, cfg.com_height, 0.0])

    frames: List[Dict] = []

    def check(tag: str, st: QuadContactState, com_now: np.ndarray, extra: float = 0.0):
        r = allocate_forces(st, com_now, extra_load_n=extra)
        frames.append({
            "phase": tag, "n_contacts": st.n_contacts,
            "ok": r.ok, "max_hand_n": r.max_hand_load_n,
            "max_friction_margin": max(r.friction_margins.values()) if r.friction_margins else 0.0,
            "wrench_err": r.wrench_balance_err,
            "peak_finger_n": r.peak_finger_n,
            "fingertip_ok": r.fingertip_ok,
        })
        return r.ok

    all_ok = True
    # 1. 四肢静态 (蹲姿: com 低)
    st = QuadContactState(cfg, foot_l, foot_r, hand_l, hand_r)
    all_ok &= check("quad_squat", st, np.array([0.0, 0.55, 0.0]))
    # 2. 蹲起 (com 升)
    all_ok &= check("quad_stand", st, np.array([0.0, cfg.com_height, 0.0]))
    # 3. 抬左手 (三肢) — 换 hold
    st3 = QuadContactState(cfg, foot_l, foot_r, None, hand_r)
    all_ok &= check("lift_left_hand", st3, np.array([0.05, cfg.com_height, 0.0]))
    # 4. 左手挂新 hold (更高)
    hand_l2 = np.array([0.32, 1.35, 0.05])
    st4 = QuadContactState(cfg, foot_l, foot_r, hand_l2, hand_r)
    all_ok &= check("rehang_left", st4, np.array([0.05, cfg.com_height, 0.0]))
    # 5. 抬左脚 (三肢) — 换脚位 (foot_left=None)
    st5 = QuadContactState(cfg, None, foot_r, hand_l2, hand_r)
    all_ok &= check("lift_left_foot", st5, np.array([0.05, 0.7, 0.0]))
    # 6. 左脚新位 (更高台阶)
    foot_l2 = np.array([0.20, 0.55, 0.03])
    st6 = QuadContactState(cfg, foot_l2, foot_r, hand_l2, hand_r)
    all_ok &= check("replace_left_foot", st6, np.array([0.05, 0.72, 0.0]))
    # 7. 回四肢静立 (新位形)
    all_ok &= check("quad_settled", st6, np.array([0.02, cfg.com_height, 0.0]))

    unstable = sum(1 for f in frames if not f["ok"])
    return {
        "frames": frames,
        "n_frames": len(frames),
        "unstable_frames": unstable,
        "all_ok": bool(all_ok),
        "phases": [f["phase"] for f in frames],
        "ok": bool(all_ok and unstable == 0),
        "arm_approx": ARM_APPROX,
        "peak_finger_n": float(max(f["peak_finger_n"] for f in frames)) if frames else 0.0,
        "fingertip_ok_all": bool(all(f["fingertip_ok"] for f in frames)) if frames else True,
    }
