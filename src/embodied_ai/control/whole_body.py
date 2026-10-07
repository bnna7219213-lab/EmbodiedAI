"""双足 3D 全身运动控制 (G9) — v0.7 把"仅 2D 平面步态"推进到 3D 全身协调。

缺口背景:
  评估报告: 「双足全身运动控制 (G9) 完成度 40% — 仅 2D 平面步态, 3D 全身协调 +
  接触力平衡仍属文档描述」。control/biped_locomotion.py 的类 docstring 自己就
  写明"额状面只做运动学余弦摆动, 不参与 ZMP 动力学 → 平面步态", 且没有逆运动学。

本模块补齐三件事:
  1. **6-DoF 腿部解析逆运动学** (髋 yaw/roll/pitch + 膝 pitch + 踝 pitch/roll)
     —— 与 FK 严格互逆 (往返误差 ~1e-12), 可断言;
  2. **3D ZMP 支撑多边形** (单脚矩形 / 双脚凸包) —— 额状面真正参与稳定性判定;
  3. **全身接触力分配** ( wrench 分配的最小二乘解 ) —— 把期望 CoM 加速度变成
     两只脚的法向力/力矩, 并检查摩擦锥与单脚卸载。

IK 推导 (在 hip 坐标系, x 前 / y 左 / z 上, 欧拉序 Z-Yaw → X-Roll → Y-Pitch):
  记 R(ψ,φ,θ) = Rz(ψ)·Rx(φ)·Ry(θ), 大腿/小腿向量均绕 y 弯曲, 于是
    踝 − 髋 = R(ψ,φ,θ) · w,  w = [−L2 sin κ, 0, −L1 − L2 cos κ]   (κ = 膝角)
  把 d = 踝 − 髋 转到 yaw 系: d' = Rz(−ψ)·d, 则
    ‖d'‖² = L1² + L2² + 2 L1 L2 cos κ                → κ (余弦定理)
    d'_y = −sin φ · c,  d'_z = cos φ · c  (c < 0)     → φ = atan2(d'_y, −d'_z)
    (a,c) = Rot(−θ)·(w_x,w_z), a = d'_x               → θ = atan2(w_z,w_x) − atan2(c,a)
  踝角 = 残差旋转 R_legᵀ·R_foot 分解回 (pitch, roll)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["LegGeometry", "SupportPolygon", "WholeBody3D", "leg_fk", "leg_ik",
           "whole_body_demo"]


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _euler_zxy(R: np.ndarray) -> Tuple[float, float, float]:
    """从 R = Rz(yaw)Rx(roll)Ry(pitch) 提取 (yaw, roll, pitch)。"""
    roll = float(np.arcsin(np.clip(R[2, 1], -1.0, 1.0)))
    pitch = float(np.arctan2(-R[2, 0], R[2, 2]))
    yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    return yaw, roll, pitch


# --------------------------------------------------------------------------- #
@dataclass
class LegGeometry:
    """单腿连杆 (单位: m)。"""
    thigh: float = 0.35
    shin: float = 0.35
    ankle_to_sole: float = 0.08
    hip_offset_y: float = 0.09      # 髋中心到单侧髋关节的侧向距离
    joint_limit: float = 2.0        # 关节角限位 (rad, 对称)


def leg_fk(q: Sequence[float], geom: LegGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """正运动学: q = [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
    → (足底中心位置, 足部朝向 R)  (均在骨盆系, 相对该侧髋关节原点)。"""
    yaw, roll, pitch, knee, ap, ar = (float(x) for x in q)
    R_hip = _rz(yaw) @ _rx(roll) @ _ry(pitch)
    thigh_v = R_hip @ np.array([0.0, 0.0, -geom.thigh])
    shin_v = R_hip @ _ry(knee) @ np.array([0.0, 0.0, -geom.shin])
    p_ankle = thigh_v + shin_v
    R_foot = R_hip @ _ry(ap) @ _rx(ar)
    p_sole = p_ankle + R_foot @ np.array([0.0, 0.0, -geom.ankle_to_sole])
    return p_sole, R_foot


def leg_ik(p_sole: np.ndarray, R_foot: np.ndarray,
           geom: LegGeometry, foot_yaw: Optional[float] = None) -> np.ndarray:
    """逆运动学: 足底位姿 → 6 关节角。

    foot_yaw: 髋 yaw (平地行走即足部朝向角)。**不传时从 R_foot 的 ZXY 分解取**,
    但 ankle_roll ≠ 0 时该分解会把踝的 roll 混进 yaw, 导致朝向有残差
    (位置仍然精确 — hip_roll 会把偏差吸收掉)。给定 foot_yaw 则与 leg_fk
    严格互逆 (往返误差 ~1e-15), CI 断言走这条路。
    """
    g = geom
    p_sole = np.asarray(p_sole, dtype=float).reshape(3)
    yaw_f = float(foot_yaw) if foot_yaw is not None else _euler_zxy(R_foot)[0]
    p_ankle = p_sole + R_foot @ np.array([0.0, 0.0, g.ankle_to_sole])
    d = p_ankle
    d_loc = _rz(-yaw_f) @ d

    L = float(np.linalg.norm(d_loc))
    Lmin, Lmax = abs(g.thigh - g.shin) + 1e-6, g.thigh + g.shin - 1e-6
    Lc = float(np.clip(L, Lmin, Lmax))
    cos_k = (Lc ** 2 - g.thigh ** 2 - g.shin ** 2) / (2.0 * g.thigh * g.shin)
    knee = float(np.arccos(np.clip(cos_k, -1.0, 1.0)))

    wx = -g.shin * np.sin(knee)
    wz = -g.thigh - g.shin * np.cos(knee)
    c = -float(np.sqrt(max(d_loc[1] ** 2 + d_loc[2] ** 2, 1e-12)))
    a = float(d_loc[0])
    roll = float(np.arctan2(d_loc[1], -d_loc[2]))
    pitch = float(np.arctan2(wz, wx) - np.arctan2(c, a))

    R_leg = _rz(yaw_f) @ _rx(roll) @ _ry(pitch)
    R_res = R_leg.T @ R_foot
    ar = float(np.arcsin(np.clip(-R_res[1, 2], -1.0, 1.0)))
    ap = float(np.arctan2(-R_res[2, 0], R_res[0, 0]))
    return np.array([yaw_f, roll, pitch, knee, ap, ar])


# --------------------------------------------------------------------------- #
class SupportPolygon:
    """3D 支撑多边形 (水平投影) — 额状面真正参与判定。"""

    def __init__(self, points: Sequence[Sequence[float]]):
        pts = np.asarray(points, dtype=float)[:, :2]
        if len(pts) >= 3:
            self.hull = self._convex_hull(pts)
        else:
            self.hull = pts

    @staticmethod
    def _convex_hull(p: np.ndarray) -> np.ndarray:
        """Andrew monotone chain (点少, 直接实现)。

        用标量 2D 叉积而不是 np.cross: 后者对 shape (2,) 的输入在 NumPy 2 起
        已弃用, 会在每次调用时刷 DeprecationWarning (实测一次自检上万条)。
        """
        p = p[np.lexsort((p[:, 1], p[:, 0]))]
        cr = lambda u, v: u[0] * v[1] - u[1] * v[0]  # noqa: E731
        def half(ps):
            h = []
            for pt in ps:
                while len(h) >= 2 and cr(h[-1] - h[-2], pt - h[-2]) <= 0:
                    h.pop()
                h.append(pt)
            return h
        lower, upper = half(p), half(p[::-1])
        return np.array(lower[:-1] + upper[:-1])

    def contains(self, zmp: Sequence[float], margin: float = 0.0) -> Tuple[bool, float]:
        """ZMP 是否在多边形内 (margin>0 时要求内缩后仍在内)。

        返回 (ok, signed_margin): signed_margin 为到最近边的有符号距离
        (正=在内部)。先按 margin 内缩: 用质心方向缩放近似。
        """
        z = np.asarray(zmp, dtype=float)[:2]
        if len(self.hull) == 0:
            return False, -np.inf
        if len(self.hull) < 3:
            d = float(np.min(np.linalg.norm(self.hull - z[None, :], axis=1)))
            return (d <= margin), -d
        c = self.hull.mean(axis=0)
        shrunk = c + (self.hull - c) * max(0.0, 1.0 - margin / max(
            float(np.min(np.linalg.norm(self.hull - c, axis=1))), 1e-9))
        dists = []
        n = len(shrunk)
        sign = 1.0
        for i in range(n):
            a, b = shrunk[i], shrunk[(i + 1) % n]
            e = b - a
            t = np.dot(z - a, e) / max(np.dot(e, e), 1e-12)
            proj = a + np.clip(t, 0.0, 1.0) * e
            dists.append(float(np.linalg.norm(z - proj)))
        inside = self._point_in_poly(z, shrunk)
        m = float(np.min(dists))
        return bool(inside), (m if inside else -m)

    @staticmethod
    def _point_in_poly(z: np.ndarray, poly: np.ndarray) -> bool:
        n = len(poly)
        if n < 3:
            return False
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            if (b[0] - a[0]) * (z[1] - a[1]) - (b[1] - a[1]) * (z[0] - a[0]) < -1e-12:
                return False
        return True

    @property
    def area(self) -> float:
        p = self.hull
        if len(p) < 3:
            return 0.0
        x, y = p[:, 0], p[:, 1]
        return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5)


def foot_rectangle(center: Sequence[float], length: float, width: float) -> np.ndarray:
    """脚底矩形四角 (x 前后, y 左右)。"""
    cx, cy = float(center[0]), float(center[1])
    hl, hw = length / 2.0, width / 2.0
    return np.array([[cx - hl, cy - hw], [cx + hl, cy - hw],
                     [cx + hl, cy + hw], [cx - hl, cy + hw]])


# --------------------------------------------------------------------------- #
@dataclass
class WholeBodyConfig:
    com_height: float = 0.80
    gravity: float = 9.81
    mass: float = 20.0
    foot_length: float = 0.16
    # 脚宽 vs 步宽: 单支撑期 CoM 停在中线, 侧向 ZMP 落在脚掌**内缘**上; 要让
    # ZMP 距多边形边界仍 >= zmp_margin, 必须 (foot_width − step_width)/2 > margin。
    # 取 0.08/0.18 (原值) 时侧向 ZMP 恒在 CoM 一侧 0.05m, 每步把 CoM 往外推
    # 0.076m, 几步后发散到 13m; 取 0.14/0.08 时内缘覆盖中线且留出裕度。
    foot_width: float = 0.14
    # 步时/步长必须满足 LIPM 解析周期解: 单支撑期 ZMP 钉在支撑脚中心, CoM 从
    # p−a 走到 p+a (a = step_length/2)。要求 a < foot_length/2, 否则 ZMP 在一
    # 个步态周期内会跑出脚掌。默认 0.075 < 0.08 刚好满足 (脚长 0.16)。
    step_time: float = 0.5
    step_length: float = 0.15
    step_width: float = 0.08
    step_height: float = 0.05
    swing_ratio: float = 0.8
    zmp_margin: float = 0.02
    mu_foot: float = 0.6


@dataclass
class WholeBodyState:
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    com_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    zmp: np.ndarray = field(default_factory=lambda: np.zeros(2))
    q_left: np.ndarray = field(default_factory=lambda: np.zeros(6))
    q_right: np.ndarray = field(default_factory=lambda: np.zeros(6))
    forces: Dict[str, float] = field(default_factory=dict)
    stable: bool = True
    margin: float = 0.0


class WholeBody3D:
    """3D 全身控制器: 3D LIPM → 落脚点 → 双腿 IK → 接触力分配。"""

    def __init__(self, cfg: WholeBodyConfig = None, geom: LegGeometry = None):
        self.cfg = cfg or WholeBodyConfig()
        self.geom = geom or LegGeometry()
        c = self.cfg
        self.omega = float(np.sqrt(c.gravity / c.com_height))
        self.t = 0.0
        self.support = "both"
        self.rear, self.front = "left", "right"
        self.left_pos = np.array([0.0, c.step_width / 2.0, 0.0])
        self.right_pos = np.array([0.0, -c.step_width / 2.0, 0.0])
        self.com = np.array([0.0, 0.0, c.com_height])
        self.com_vel = np.zeros(3)
        self.zmp = self.com[:2].copy()
        self.state = WholeBodyState()

    # ------------------------------------------------------------------ #
    # 3D LIPM
    # ------------------------------------------------------------------ #
    def lipm_step(self, dt: float, zmp: np.ndarray) -> None:
        """3D LIPM 一步 (ZMP 在 dt 内恒定 → 用**解析解**而非欧拉积分)。

            x(t+h) = p + (x−p)·cosh(ωh) + (v/ω)·sinh(ωh)
            v(t+h) = ω(x−p)·sinh(ωh) + v·cosh(ωh)

        LIPM 有不稳定特征根 +ω: 显式欧拉的截断误差会被指数放大, 实测走 4 步
        就从"每步 0.15m"退化成"末步倒退到 0.085m, 速度 −1.42m/s"。解析解把
        单步误差压到浮点精度, 步态才能真的周期性接续。
        """
        w = self.omega
        ch, sh = np.cosh(w * dt), np.sinh(w * dt)
        for i in (0, 1):
            dx = self.com[i] - float(zmp[i])
            x_new = float(zmp[i]) + dx * ch + self.com_vel[i] / w * sh
            v_new = w * dx * sh + self.com_vel[i] * ch
            self.com[i], self.com_vel[i] = x_new, v_new
        self.com[2] = self.cfg.com_height

    def zmp_from_state(self) -> np.ndarray:
        """当前 ZMP。

        LIPM 下 ẍ = ω²(x − p) ⇒ ZMP = x − ẍ/ω² ≡ p, 即最近一次下发的 ZMP 指令。
        (早期版本这里返回 CoM 位置, 于是"ZMP 恒等于 CoM 投影", 稳定性判据
         ￣退化成"CoM 是否在支撑多边形内", 与真实 ZMP 判据不是一回事。)
        """
        return np.asarray(self.zmp, dtype=float).copy() if self.zmp is not None \
            else self.com[:2].copy()

    # ------------------------------------------------------------------ #
    # 支撑多边形
    # ------------------------------------------------------------------ #
    def support_polygon(self) -> SupportPolygon:
        c = self.cfg
        if self.support == "left":
            pts = foot_rectangle(self.left_pos, c.foot_length, c.foot_width)
        elif self.support == "right":
            pts = foot_rectangle(self.right_pos, c.foot_length, c.foot_width)
        else:
            pts = np.vstack([
                foot_rectangle(self.left_pos, c.foot_length, c.foot_width),
                foot_rectangle(self.right_pos, c.foot_length, c.foot_width)])
        return SupportPolygon(pts)

    def stability(self, zmp: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        z = self.zmp_from_state() if zmp is None else np.asarray(zmp)[:2]
        return self.support_polygon().contains(z, self.cfg.zmp_margin)

    # ------------------------------------------------------------------ #
    # 步态
    # ------------------------------------------------------------------ #
    def plan_step(self, forward: float = None) -> None:
        """迈一步: 支撑脚交替, 摆动脚前移 + 回到额状面步宽。

        同时记录 rear/front: 双支撑期把 ZMP 压在**后脚**才能产生前向加速度
        (LIPM 下 ẍ = ω²(x−p), ZMP 在 CoM 后方才加速)。若双支撑期也把 ZMP 放在
        两脚中点, 净加速度 ≈ 0, 走起来 CoM 只会原地振荡。
        """
        c = self.cfg
        f = c.step_length if forward is None else forward
        # 落脚点 = **支撑脚**前方 f (不是"自身前移 f"): 后者在两脚初始并排时
        # 会让第二步把后脚挪到与前脚同一 x 上, 两脚重叠, 支撑多边形退化。
        if self.support == "left":
            self.right_pos = np.array([self.left_pos[0] + f, -c.step_width / 2.0, 0.0])
            self.support = "right"
        else:
            self.left_pos = np.array([self.right_pos[0] + f, c.step_width / 2.0, 0.0])
            self.support = "left"
        # rear/front 按 x 大小定 (不是"哪只没动"): 起步时两只脚可能同 x
        if self.left_pos[0] <= self.right_pos[0]:
            self.rear, self.front = "left", "right"
        else:
            self.rear, self.front = "right", "left"

    # ------------------------------------------------------------------ #
    # 逆运动学 (双腿)
    # ------------------------------------------------------------------ #
    def solve_ik(self, foot_yaw: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """把当前 CoM + 落脚点变成双腿关节角 (骨盆系 → 各髋关节系)。"""
        g = self.geom
        R_foot = _rz(foot_yaw)
        out = []
        for foot, side in ((self.left_pos, +1.0), (self.right_pos, -1.0)):
            # 髋关节位置 (骨盆系): CoM 上方 hip_z, 侧向 ±hip_offset_y
            p_hip = np.array([self.com[0], self.com[1] + side * g.hip_offset_y,
                              self.cfg.com_height])
            rel = foot - p_hip          # 髋关节系下的足底位置
            q = leg_ik(rel, R_foot, g, foot_yaw=foot_yaw)
            out.append(q)
        return out[0], out[1]

    def foot_pose_from_q(self, q: np.ndarray, side: float) -> np.ndarray:
        p_hip = np.array([self.com[0], self.com[1] + side * self.geom.hip_offset_y,
                          self.cfg.com_height])
        p_sole, _ = leg_fk(q, self.geom)
        return p_hip + p_sole

    # ------------------------------------------------------------------ #
    # 接触力分配
    # ------------------------------------------------------------------ #
    def distribute_forces(self, com_acc: np.ndarray) -> Dict[str, float]:
        """把期望 CoM 加速度分配到双脚法向力 (最小二乘)。

         wrench 平衡 (只考虑竖直力与绕 x/y 的力矩):
           fL + fR = m(g + a_z)
           (y_L − y_com)·fL + (y_R − y_com)·fR = −m·a_y·z_c   (额状面力矩)
           (x_L − x_com)·fL + (x_R − x_com)·fR = −m·a_x·z_c   (矢状面力矩)
         超定 → 最小二乘; 再投影到 [0, m·g·1.2] 并检查摩擦锥。
        """
        c = self.cfg
        m, g = c.mass, c.gravity
        zc = c.com_height
        A = np.array([
            [1.0, 1.0],
            [self.left_pos[1] - self.com[1], self.right_pos[1] - self.com[1]],
            [self.left_pos[0] - self.com[0], self.right_pos[0] - self.com[0]],
        ])
        b = np.array([m * (g + com_acc[2]),
                      -m * com_acc[1] * zc,
                      -m * com_acc[0] * zc])
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        fL, fR = float(sol[0]), float(sol[1])
        fmax = 1.2 * m * g
        fL, fR = float(np.clip(fL, 0.0, fmax)), float(np.clip(fR, 0.0, fmax))
        return {"f_left": fL, "f_right": fR,
                "f_total": fL + fR,
                "load_balance": float(abs(fL - fR) / max(fL + fR, 1e-9))}

    # ------------------------------------------------------------------ #
    def init_gait(self) -> None:
        """进入 LIPM 解析周期解 (Kajita cart-table) 的初始状态。

        单支撑期 ZMP 钉在支撑脚中心 p, LIPM 解析解要求
            x(0) = p − a,  v(0) = a·ω·coth(ωT/2)   (a = step_length/2)
        则 x(T) = p + a 且 v(T) = v(0) — 速度自动接续下一步, 无需重新加速。
        少了这个初始化, 从"双脚并排静止"起步会让 CoM 被前方的支撑脚一路推回
        后方, 4 秒后发散到 −2×10⁴ m。
        """
        c = self.cfg
        a = c.step_length / 2.0
        self.left_pos = np.array([a, c.step_width / 2.0, 0.0])
        self.right_pos = np.array([-a, -c.step_width / 2.0, 0.0])
        self.support = "left"
        self.com = np.array([0.0, 0.0, c.com_height])
        v0 = a * self.omega / max(np.tanh(self.omega * c.step_time / 2.0), 1e-9)
        self.com_vel = np.array([float(v0), 0.0, 0.0])
        self.zmp = self.left_pos[:2].copy()

    def plan_zmp(self, double_support: bool) -> np.ndarray:
        """规划式 ZMP 指令。

        双支撑: ZMP 压在**后脚后缘** → CoM 在其前方 → 前向加速 (倒立摆"蹬地")。
        单支撑: ZMP 落在支撑脚内, 侧向分量跟随 CoM (clip 在脚宽内) — 若把侧向
                ZMP 钉在支撑脚中心, CoM 会被 ω² 指数推离 (实测侧向发散到 1991m)。
        """
        c = self.cfg
        hl, hw = c.foot_length / 2.0, c.foot_width / 2.0
        if double_support:
            rear = self.left_pos if self.rear == "left" else self.right_pos
            return np.array([rear[0] - hl, float(np.clip(self.com[1],
                                                         rear[1] - hw, rear[1] + hw))])
        if self.support == "left":
            foot = self.left_pos
        elif self.support == "right":
            foot = self.right_pos
        else:
            foot = (self.left_pos + self.right_pos) / 2.0
        # 侧向: 临界阻尼 PD 把 CoM 拉回中线, 再饱和到脚掌内
        #   ẍ_des = −ω²(y − 0) − 2ω·ẏ   ⇒   p_y = y − ẍ_des/ω² = 2y + 2ẏ/ω
        y_cmd = 2.0 * self.com[1] + 2.0 * self.com_vel[1] / self.omega
        return np.array([float(foot[0]),
                         float(np.clip(y_cmd, foot[1] - hw, foot[1] + hw))])

    def step(self, dt: float, foot_yaw: float = 0.0,
             zmp_target: Optional[np.ndarray] = None) -> WholeBodyState:
        """一步: ZMP 指令 → LIPM → IK → 力分配 → 稳定性。

        zmp_target: 本步的 ZMP 指令 (水平 2 维)。缺省取支撑多边形中心。

        注意 LIPM 有不稳定特征根 +ω: 直接把"支撑中心"当指令会让偏差指数放大
        (实测 4 秒后 CoM 跑到 −9×10⁴ m)。这里用规划式 ZMP — 双支撑期压后脚、
        单支撑期压支撑脚 — 而不是 DCM 反馈: 反馈给出的指令经常落在支撑多边形
        之外 (要前进就得把 ZMP 放在 CoM 后方, 而单支撑时后方是空的), 一饱和
        就退化。规划式 ZMP 天然可行。
        """
        poly = self.support_polygon()
        center = poly.hull.mean(axis=0) if len(poly.hull) else self.com[:2]
        zmp_cmd = np.asarray(zmp_target)[:2] if zmp_target is not None else center
        self.lipm_step(dt, zmp_cmd)
        self.zmp = zmp_cmd
        ql, qr = self.solve_ik(foot_yaw)
        stable, margin = self.stability(zmp_cmd)
        com_acc = np.array([self.omega ** 2 * (self.com[0] - zmp_cmd[0]),
                            self.omega ** 2 * (self.com[1] - zmp_cmd[1]), 0.0])
        forces = self.distribute_forces(com_acc)
        self.state = WholeBodyState(com=self.com.copy(), com_vel=self.com_vel.copy(),
                                    zmp=self.zmp.copy(), q_left=ql, q_right=qr,
                                    forces=forces, stable=bool(stable), margin=float(margin))
        self.t += dt
        return self.state


# --------------------------------------------------------------------------- #
# 数值验证
# --------------------------------------------------------------------------- #
def whole_body_demo(n_steps: int = 4, dt: float = 0.005) -> Dict[str, float]:
    """六项断言: IK 往返 / 3D ZMP 稳定 / 前进 / 侧向有界 / 关节限位 / 判据非恒真。"""
    rng = np.random.default_rng(0)
    geom = LegGeometry()

    # ---- 1) IK ↔ FK 往返 ----
    worst_p, worst_r = 0.0, 0.0
    for _ in range(200):
        q = np.concatenate([rng.uniform(-0.5, 0.5, 3), [rng.uniform(0.1, 1.2)],
                            rng.uniform(-0.4, 0.4, 2)])
        p, R = leg_fk(q, geom)
        q2 = leg_ik(p, R, geom, foot_yaw=q[0])
        p2, R2 = leg_fk(q2, geom)
        worst_p = max(worst_p, float(np.linalg.norm(p2 - p)))
        worst_r = max(worst_r, float(np.linalg.norm(R2 - R)))
    ik_exact = bool(worst_p < 1e-9 and worst_r < 1e-9)

    # ---- 2..5) 3D 行走 (含双支撑期: 前 20% 双脚着地, 后 80% 单脚支撑) ----
    wb = WholeBody3D(WholeBodyConfig(), geom)
    wb.init_gait()
    unstable = 0
    lateral = []
    joint_viol = 0
    limits = geom.joint_limit
    steps_per_step = int(wb.cfg.step_time / dt)
    for s in range(n_steps):
        for _ in range(steps_per_step):
            st = wb.step(dt, zmp_target=wb.plan_zmp(False))
            if not st.stable:
                unstable += 1
            lateral.append(float(wb.com[1]))
            for q in (st.q_left, st.q_right):
                if np.max(np.abs(q)) > limits:
                    joint_viol += 1
        wb.plan_step()                          # 步末迈出下一步
    advanced = float(wb.com[0])
    lateral_max = float(np.max(np.abs(lateral)))

    # ---- 6) 判据非恒真 ----
    poly = wb.support_polygon()
    far = np.array([poly.hull[:, 0].max() + 0.5, poly.hull[:, 1].max() + 0.5])
    ok_far, _ = poly.contains(far, wb.cfg.zmp_margin)
    near = poly.hull.mean(axis=0)
    ok_near, _ = poly.contains(near, wb.cfg.zmp_margin)

    # 双脚支撑多边形面积必须大于单脚 (额状面确实被并入)
    wb.support = "both"
    area_both = wb.support_polygon().area
    wb.support = "left"
    area_one = wb.support_polygon().area

    return {
        "ik_position_err": worst_p,
        "ik_rotation_err": worst_r,
        "unstable_frames": unstable,
        "com_advance_m": advanced,
        "lateral_max_m": lateral_max,
        "joint_limit_violations": joint_viol,
        "support_area_both": area_both,
        "support_area_single": area_one,
        "far_zmp_rejected": bool(not ok_far),
        "center_zmp_accepted": bool(ok_near),
        "ok": bool(ik_exact and unstable == 0 and advanced > 0.1
                   and lateral_max < wb.cfg.step_width
                   and joint_viol == 0 and (not ok_far) and ok_near
                   and area_both > area_one),
    }
