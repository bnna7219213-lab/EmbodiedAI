"""灵巧手设计决策 (G11) — v0.7 把"未做设计决策"变成可审计的选型结论。

缺口背景:
  评估报告: 「灵巧手设计 (G11) 完成度 0% — 未做设计决策: 3 指 vs 5 指、
  欠驱动 vs 全驱动」。这是全项目唯一一个 0% 的**硬件**条目, 而它直接决定
  P4 驱动板的通道数、CAN-FD 报文数、以及 L5 接触规划的动作空间维度。

方法 (不做"我觉得", 做可计算对比):
  1. 参数化手型: 手指数 × 每指关节数 × 驱动方式 (全驱动 / 腱驱欠驱动);
  2. 对**家庭任务集**逐项评估抓取可行性 — 力闭合 (force closure) 用摩擦锥
     wrench 凸包的数值判据, 而不是目测;
  3. 输出决策矩阵: 自由度(=执行器数) / 力闭合覆盖度 / 抓取质量 / 控制复杂度,
     并给出推荐 + 依据。

力闭合判据 (平面抓取, wrench = [fx, fy, τ]):
  接触 i 的法向 n_i、切向 t_i, 摩擦系数 μ ⇒ 摩擦力锥由两条边缘
    f_i± = n_i ± μ·t_i   (线性化)
  映射为 wrench: w = [f ; (r_i × f)]。
  力闭合 ⟺ ∀ 单位方向 d ∈ R³: max_i ⟨w_i, d⟩ > 0 —— 即 wrench 凸包把原点
  包在**内部**, 可抵抗任意外界 wrench。质量 ε = min_d max_i ⟨w_i,d⟩
  (Ferrari–Canny 式内接球半径的离散估计)。

欠驱动建模:
  单腱驱动多关节 ⇒ 关节角 q = A·s (s = 腱位移, A = 耦合比向量), 接触力在
  腱空间等分 (力矩 τ = Aᵀ·f_tendon)。因此欠驱动的**力分配不可独立控制** —
  这既是它的优势 (自适应包络, 无需力控) 也是短板 (无法做精细姿态调整)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["FingerSpec", "HandDesign", "GraspContact", "force_closure",
           "grasp_quality", "underactuated_coupling", "FAMILY_TASKS",
           "design_matrix", "recommend", "dexterous_hand_demo"]


# --------------------------------------------------------------------------- #
@dataclass
class FingerSpec:
    name: str
    n_joints: int = 3
    link_lengths: Sequence[float] = (0.040, 0.030, 0.022)
    # 基座位置 (手掌坐标系, x 前 / y 左 / z 上), 单位 m
    base_pos: Sequence[float] = (0.0, 0.0, 0.0)
    base_yaw: float = 0.0
    abduction: bool = False          # 是否有侧摆自由度
    actuated: bool = True            # 全驱动=每关节一个执行器
    tendon_group: int = 0            # 欠驱动时同一组 = 同一根腱


@dataclass
class HandDesign:
    name: str
    fingers: List[FingerSpec]
    palm_radius: float = 0.045
    mu: float = 0.6                  # 指尖摩擦系数 (硅胶包覆)
    drive: str = "full"              # "full" | "underactuated"
    mass_kg: float = 0.5
    cost_units: int = 0              # 相对成本 (执行器/腱数主导)
    # 手指能包络到的角度跨度 (度): 3 指 ≈ 240°, 5 指 ≈ 330° (加手掌支撑)。
    # 这个参数决定"接触点能分布多宽" — 是 3 指 vs 5 指唯一有物理意义的差别,
    # 只比"指数"是没有信息量的 (任务通常只需 3 个接触)。
    envelope_span_deg: float = 240.0

    @property
    def n_dof(self) -> int:
        return sum(f.n_joints + (1 if f.abduction else 0) for f in self.fingers)

    @property
    def n_actuators(self) -> int:
        if self.drive == "full":
            return self.n_dof
        return len({f.tendon_group for f in self.fingers}) + \
            sum(1 for f in self.fingers if f.abduction)

    @property
    def independence(self) -> float:
        """独立控制度 = 执行器数 / 自由度 (欠驱动 < 1)。"""
        return self.n_actuators / max(self.n_dof, 1)


@dataclass
class GraspContact:
    """一次抓取在物体表面的接触点 (平面抓取)。"""
    point: np.ndarray        # (2,) 接触点 (物体坐标系)
    normal: np.ndarray       # (2,) 指向物体内部的法向
    mu: float = 0.6

    def wrench_basis(self, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """摩擦锥两条边缘 → wrench [fx, fy, τ]。"""
        n = self.normal / max(np.linalg.norm(self.normal), 1e-9)
        t = np.array([-n[1], n[0]])
        out = []
        for d in (n + self.mu * t, n - self.mu * t):
            r = self.point - center
            tau = float(r[0] * d[1] - r[1] * d[0])
            out.append(np.array([d[0], d[1], tau]))
        return out[0], out[1]


# --------------------------------------------------------------------------- #
def force_closure(contacts: Sequence[GraspContact], center: np.ndarray = None,
                  n_dirs: int = 400, seed: int = 0) -> Tuple[bool, float]:
    """力闭合判定 + 质量 ε。

    ε = min_{|d|=1} max_i ⟨w_i, d⟩  (归一化到 wrench 基的平均模长)。
    ε > 0 ⇒ 力闭合。
    """
    if center is None:
        center = np.mean(np.array([c.point for c in contacts]), axis=0)
    W = []
    for c in contacts:
        a, b = c.wrench_basis(center)
        W.extend([a, b])
    W = np.array(W)
    if len(W) < 4:
        return False, 0.0

    rng = np.random.default_rng(seed)
    # 3D 单位方向 (wrench 空间): 用 Fibonacci 球采样 + 随机补充, 保证覆盖
    dirs = []
    n1 = max(n_dirs // 2, 8)
    for k in range(n1):
        z = 1.0 - 2.0 * (k + 0.5) / n1
        r = np.sqrt(max(1.0 - z * z, 0.0))
        phi = np.pi * (1 + np.sqrt(5)) * k
        dirs.append([r * np.cos(phi), r * np.sin(phi), z])
    for _ in range(n_dirs - n1):
        v = rng.standard_normal(3)
        dirs.append(v / np.linalg.norm(v))
    dirs = np.array(dirs)

    proj = W @ dirs.T                      # (2n, n_dirs)
    best = proj.max(axis=0)                # 每个方向的最优边缘
    eps = float(best.min())
    scale = float(np.mean(np.linalg.norm(W, axis=1)))
    return bool(eps > 1e-9), eps / max(scale, 1e-9)


def grasp_quality(contacts: Sequence[GraspContact], center: np.ndarray = None) -> float:
    ok, eps = force_closure(contacts, center)
    return eps


def underactuated_coupling(hand: HandDesign) -> Dict[str, np.ndarray]:
    """欠驱动的腱—关节耦合: 返回每个腱组的耦合比向量与关节→腱的映射。"""
    groups: Dict[int, List[int]] = {}
    idx = 0
    for f in hand.fingers:
        for j in range(f.n_joints):
            groups.setdefault(f.tendon_group, []).append(idx)
            idx += 1
        if f.abduction:
            idx += 1
    ratios = {}
    for g, js in groups.items():
        # 同一腱上各关节按 1:1 屈曲 (腱位移分配到各关节), 远端更快 ⇒ 递增比
        r = np.arange(1, len(js) + 1, dtype=float)
        ratios[g] = r / r.sum()
    return {"groups": groups, "ratios": ratios,
            "n_tendons": len(groups),
            "note": "欠驱动下同一腱组的关节力矩比固定 = ratios, 不可独立分配"}


# --------------------------------------------------------------------------- #
# 家庭任务集: 每项给出物体半径与所需接触数/抓取类型
FAMILY_TASKS = {
    "hold_mug":        dict(radius=0.040, n_contacts=3, label="握马克杯", need_power=True),
    "pinch_key":       dict(radius=0.008, n_contacts=2, label="捏钥匙", need_power=False),
    "grasp_towel":     dict(radius=0.060, n_contacts=3, label="抓毛巾(柔性)", need_power=True),
    "twist_cap":       dict(radius=0.017, n_contacts=3, label="拧瓶盖", need_power=False),
    "hold_plate":      dict(radius=0.110, n_contacts=4, label="端盘子", need_power=True),
    "pick_spoon":      dict(radius=0.012, n_contacts=2, label="取勺子", need_power=False),
    "hold_bottle":     dict(radius=0.035, n_contacts=3, label="握水瓶", need_power=True),
    "press_button":    dict(radius=0.006, n_contacts=1, label="按按钮", need_power=False),
}


def _contacts_for_task(task: dict, hand: HandDesign) -> List[GraspContact]:
    """按任务 + 手型生成接触点。

    包络抓取: 接触点均匀分布在 [−span/2, +span/2] 内, 而不是整圈 — 手指
    张不到的那一侧没有支撑, 这正是 3 指手抓大物体吃力的原因。
    捏取: 两指对置。
    """
    r = task["radius"]
    mu = hand.mu
    n = min(len(hand.fingers), task["n_contacts"])
    span = np.deg2rad(hand.envelope_span_deg)
    cs = []
    if task["need_power"] and n >= 3:
        for k in range(n):
            a = -span / 2 + span * (k + 0.5) / n
            p = np.array([r * np.cos(a), r * np.sin(a)])
            cs.append(GraspContact(p, -p / max(np.linalg.norm(p), 1e-9), mu))
    else:
        for sgn in (1.0, -1.0):
            p = np.array([sgn * r, 0.0])
            cs.append(GraspContact(p, np.array([-sgn, 0.0]), mu))
    return cs


def _size_penalty(hand: HandDesign, radius: float) -> float:
    """物体相对手掌越大越难包络 (手指张不开, 接触更接近指尖)。"""
    ratio = hand.palm_radius / max(radius, 1e-3)
    return float(np.clip(ratio, 0.35, 1.0))


def design_matrix() -> List[Dict]:
    """四个候选手型 × 家庭任务集的评估矩阵。"""
    def mk(name, n_fingers, drive, abduction=False, thumb_oppose=True):
        fingers = []
        for i in range(n_fingers):
            is_thumb = (i == 0 and thumb_oppose)
            fingers.append(FingerSpec(
                name=f"f{i}", n_joints=3,
                link_lengths=(0.042, 0.030, 0.022) if not is_thumb else (0.038, 0.032, 0.024),
                base_pos=(0.0, (i - (n_fingers - 1) / 2.0) * 0.022, 0.0),
                abduction=abduction and not is_thumb,
                tendon_group=(0 if is_thumb else 1),
            ))
        return HandDesign(name=name, fingers=fingers, drive=drive,
                          mass_kg=0.35 + 0.09 * n_fingers,
                          cost_units=(n_fingers * 3 if drive == "full" else 2),
                          envelope_span_deg=(330.0 if n_fingers >= 5 else 240.0))

    candidates = [
        mk("F3-underactuated", 3, "underactuated"),
        mk("F3-full", 3, "full"),
        mk("F5-underactuated", 5, "underactuated"),
        mk("F5-full", 5, "full"),
    ]
    rows = []
    for h in candidates:
        covered, qualities = [], []
        for key, task in FAMILY_TASKS.items():
            if task["n_contacts"] == 1:
                covered.append(1.0)          # 单指按压, 任何手型都行
                qualities.append(0.5)
                continue
            cs = _contacts_for_task(task, h)
            ok, eps = force_closure(cs)
            # 欠驱动对"需要精细姿态"的任务打折 (力分配不可独立控制)
            pen = 0.85 if (h.drive == "underactuated" and not task["need_power"]) else 1.0
            eps *= pen * _size_penalty(h, task["radius"])
            covered.append(1.0 if ok else 0.0)
            qualities.append(eps if ok else 0.0)
        rows.append({
            "design": h.name,
            "n_dof": h.n_dof,
            "n_actuators": h.n_actuators,
            "independence": round(h.independence, 3),
            "mass_kg": round(h.mass_kg, 3),
            "coverage": round(float(np.mean(covered)), 3),
            "mean_quality": round(float(np.mean(qualities)), 4),
            "min_quality": round(float(np.min(qualities)), 4),
        })
    return rows


def recommend(rows: List[Dict] = None) -> Dict:
    """加权推荐: 覆盖度优先, 其次质量, 再惩罚执行器数量 (成本/可靠性)。"""
    rows = rows or design_matrix()
    def score(r):
        return (0.45 * r["coverage"]
                + 0.30 * min(r["mean_quality"] / 0.25, 1.0)
                + 0.15 * min(r["min_quality"] / 0.10, 1.0)
                - 0.10 * (r["n_actuators"] / 15.0))
    best = max(rows, key=score)
    ranked = sorted(rows, key=score, reverse=True)
    return {"recommended": best["design"],
            "score": round(score(best), 4),
            "ranking": [r["design"] for r in ranked],
            "rationale": (
                f"{best['design']}: 覆盖度 {best['coverage']}, "
                f"平均抓取质量 {best['mean_quality']}, "
                f"执行器 {best['n_actuators']} 个 (独立度 {best['independence']})。"
                "欠驱动在包络抓取上接近全驱动, 但精细捏取与力分配受限;"
                "5 指提升覆盖度但执行器/成本显著上升。"
            )}


# --------------------------------------------------------------------------- #
def dexterous_hand_demo(seed: int = 0) -> Dict:
    """灵巧手选型自检: 力闭合判据正确性 + 决策矩阵 + 推荐。"""
    mu = 0.6
    # 1) 判据正确性: 三指 120° 均匀分布抓取圆盘 → 力闭合
    r = 0.04
    tri = [GraspContact(np.array([r * np.cos(a), r * np.sin(a)]),
                        np.array([-np.cos(a), -np.sin(a)]), mu)
           for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]
    ok_tri, eps_tri = force_closure(tri, seed=seed)

    # 2) 判据非恒真: 两指捏在对径但**共线法向**无法提供力矩 → 需检查
    oppose = [GraspContact(np.array([r, 0.0]), np.array([-1.0, 0.0]), mu),
              GraspContact(np.array([-r, 0.0]), np.array([1.0, 0.0]), mu)]
    ok_opp, eps_opp = force_closure(oppose, seed=seed)

    # 3) 极端反例: 所有接触点挤在一侧 → 不闭合
    cluster = [GraspContact(np.array([r * 1.0, 0.0]), np.array([-1.0, 0.0]), mu),
               GraspContact(np.array([r * 1.0, 0.002]), np.array([-1.0, 0.0]), mu),
               GraspContact(np.array([r * 1.0, -0.002]), np.array([-1.0, 0.0]), mu)]
    ok_clu, eps_clu = force_closure(cluster, seed=seed)

    # 4) 摩擦系数敏感性: μ 降到 0.05 时**抓取质量必须显著下降**
    #    (不能断言"不再力闭合": 只要有 μ>0, 切向力就能提供力矩, 仍闭合但很脆)
    tri_low = [GraspContact(c.point, c.normal, 0.05) for c in tri]
    ok_low, eps_low = force_closure(tri_low, seed=seed)
    mu_degrades = bool(eps_low < 0.5 * eps_tri)

    rows = design_matrix()
    rec = recommend(rows)
    coupling = underactuated_coupling(
        HandDesign(name="F5-underactuated",
                   fingers=[FingerSpec(f"f{i}", tendon_group=0 if i == 0 else 1)
                            for i in range(5)],
                   drive="underactuated"))

    return {
        "force_closure_tri": bool(ok_tri),
        "eps_tri": float(eps_tri),
        "force_closure_opposed": bool(ok_opp),
        "eps_opposed": float(eps_opp),
        "force_closure_cluster": bool(ok_clu),
        "eps_low_mu": float(eps_low),
        "mu_degrades_quality": mu_degrades,
        "n_candidates": len(rows),
        "n_tasks": len(FAMILY_TASKS),
        "recommended": rec["recommended"],
        "ranking": rec["ranking"],
        "n_tendons_underactuated": int(coupling["n_tendons"]),
        "matrix": rows,
        "ok": bool(ok_tri and ok_opp and (not ok_clu) and mu_degrades
                   and len(rows) == 4 and rec["recommended"]
                   and max(r["mean_quality"] for r in rows)
                   > 1.15 * min(r["mean_quality"] for r in rows)),
    }
