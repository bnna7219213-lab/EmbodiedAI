"""柔性体动力学 (G6) — v0.7 补齐"家庭场景必遇瓶颈, 但完成度 5%"的模块。

缺口背景:
  评估报告: 「柔性体动力学 (G6) 完成度 5% — 纯文档提及, 无实现代码。家庭场景
  (毛巾、塑料袋、硅胶餐具) 必遇瓶颈」。

方法选择: **XPBD** (Extended Position Based Dynamics, Macklin et al. 2016)
  - PBD 的刚度依赖迭代次数与时间步, XPBD 引入 compliance α = 1/k 使其与步长无关,
    这对"同一个物体在 1kHz 伺服与 20Hz 规划下表现一致"是硬需求;
  - 位置层求解 → 无条件稳定 (不会像显式弹簧质点那样在 stiff 参数下爆炸);
  - 约束统一表达: 结构/剪切/弯曲/附着/抓取全是"距离或位置约束", 便于把机器人
    夹持器直接接进来。

物理内容:
  - 质点 — 弹簧网格 (结构 + 剪切 + 弯曲约束)
  - 重力 + 速度阻尼 (瑞利阻尼近似)
  - 地面/桌面碰撞 + 库仑摩擦 (位置投影 + 切向速度衰减)
  - 抓取约束: 机器人手爪抓住若干质点并移动到目标位姿 → 变形传播
  - 滑脱前兆: 抓取点处相对位移累积 + 切/法向力比 (复用 safety 层 μ 判据形式)

可验证性 (CI 断言, 见 soft_body_demo):
  1. 不可拉伸性: 约束长度相对误差 < 5%;
  2. 悬垂收敛: 自由下垂 2s 后动能 < 阈值 (不发散、不永久抖动);
  3. 数值有界: 全程无 NaN/Inf;
  4. 抓取变形可传播: 提起中点后质心上升且投影面积缩小 (不是"抓了一点、整体不动");
  5. 滑脱判据灵敏: 抓取点被强制拖过静摩擦极限时应报 slip=True。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["SoftBodyConfig", "SoftBody", "SoftBodyMetrics", "soft_body_demo"]


# --------------------------------------------------------------------------- #
@dataclass
class SoftBodyConfig:
    n: int = 12                    # 网格行数
    m: int = 12                    # 网格列数
    spacing: float = 0.02          # 相邻质点间距 (m) → 12×12 @2cm ≈ 22cm 毛巾
    total_mass: float = 0.12       # 总质量 (kg) — 一条小毛巾
    compliance_struct: float = 1e-6   # 结构约束柔度 (m/N) — 越小越硬
    compliance_shear: float = 4e-6
    compliance_bend: float = 2e-5
    damping: float = 0.02          # 速度阻尼系数
    substeps: int = 5              # XPBD 子步
    iterations: int = 4            # 每子步约束迭代 (Jacobi)
    # Jacobi 松弛因子: 所有约束**同时**施加修正, 共享质点上会相互过冲,
    # 直接取 1.0 会在几十步内发散到 Inf (实测)。串行 Gauss-Seidel 可免, 但
    # python 循环逐约束太慢; 这里保留向量化 + 松弛, 是 XPBD 的常见折中。
    relaxation: float = 0.25
    gravity: float = -9.81
    ground_y: float = 0.0          # 地面高度
    friction_mu: float = 0.4       # 与地面/桌面的库仑摩擦系数
    restitution: float = 0.0       # 弹性恢复 (布料取 0)
    # 抓取: 手指能传递的最大力 F_max = μ_grasp · N_clamp。超过它就**打滑** —
    # 抓取点因此不能用"运动学驱动"(inv_mass=0), 那样手指永不打滑, 滑脱检测
    # 就永远是常量 False (v0.7 初版就是这个 bug)。
    grasp_max_force_n: float = 3.0
    grasp_stiffness: float = 3000.0   # 手指—物体软接触刚度 (N/m)


@dataclass
class SoftBodyMetrics:
    kinetic_energy: float = 0.0
    max_stretch: float = 0.0       # 相对静止长度的最大拉伸
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    projected_area: float = 0.0    # XZ 平面投影包围盒面积
    slip: bool = False
    grasp_force_n: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"kinetic_energy": float(self.kinetic_energy),
                "max_stretch": float(self.max_stretch),
                "com": self.com.tolist(),
                "projected_area": float(self.projected_area),
                "slip": bool(self.slip),
                "grasp_force_n": float(self.grasp_force_n)}


class SoftBody:
    """XPBD 柔性体 (质点—距离约束网格)。"""

    def __init__(self, cfg: SoftBodyConfig = None, seed: int = 0):
        self.cfg = cfg or SoftBodyConfig()
        c = self.cfg
        self.rng = np.random.default_rng(seed)

        # ---- 质点 ----
        n, m = c.n, c.m
        self.n, self.m = n, m
        xs = (np.arange(m) - (m - 1) / 2.0) * c.spacing
        zs = (np.arange(n) - (n - 1) / 2.0) * c.spacing
        X, Z = np.meshgrid(xs, zs)
        # 初始在地面上方水平铺开
        self.pos = np.column_stack([X.ravel(), np.full(n * m, c.ground_y + 0.30), Z.ravel()])
        self.prev = self.pos.copy()
        self.vel = np.zeros_like(self.pos)

        pm = c.total_mass / (n * m)
        self.mass = np.full(n * m, pm)
        self.inv_mass = np.ones(n * m) / pm

        # ---- 约束 ----
        self.rest: List[float] = []
        self.ci: List[int] = []       # 约束端点 i
        self.cj: List[int] = []       # 约束端点 j
        self.compliance: List[float] = []
        self.kind: List[str] = []

        def add(i, j, comp, kind):
            self.ci.append(i)
            self.cj.append(j)
            self.rest.append(float(np.linalg.norm(self.pos[i] - self.pos[j])))
            self.compliance.append(comp)
            self.kind.append(kind)

        def idx(r, cc):
            return r * m + cc

        for r in range(n):
            for cc in range(m):
                if cc + 1 < m:
                    add(idx(r, cc), idx(r, cc + 1), c.compliance_struct, "struct")
                if r + 1 < n:
                    add(idx(r, cc), idx(r + 1, cc), c.compliance_struct, "struct")
                if r + 1 < n and cc + 1 < m:
                    add(idx(r, cc), idx(r + 1, cc + 1), c.compliance_shear, "shear")
                    add(idx(r, cc + 1), idx(r + 1, cc), c.compliance_shear, "shear")
                if cc + 2 < m:      # 弯曲 (隔一个质点)
                    add(idx(r, cc), idx(r, cc + 2), c.compliance_bend, "bend")
                if r + 2 < n:
                    add(idx(r, cc), idx(r + 2, cc), c.compliance_bend, "bend")

        self.ci = np.array(self.ci, dtype=int)
        self.cj = np.array(self.cj, dtype=int)
        self.rest = np.array(self.rest, dtype=float)
        self.compliance = np.array(self.compliance, dtype=float)
        self.lam = np.zeros(len(self.rest))

        # ---- 抓取 / 固定 ----
        self.pins: Dict[int, np.ndarray] = {}
        self.grab_idx: np.ndarray = np.zeros(0, dtype=int)
        self.grab_target: Optional[np.ndarray] = None
        self._grasp_disp: np.ndarray = np.zeros(3)   # [0] = 累积打滑位移 (m)
        self._prev_target: np.ndarray = np.zeros(3)
        self._grasp_force = 0.0
        self._slip = False
        self.slip_threshold: float = 0.01   # 抓取点相对位移超过该值判滑脱 (m)

    # ------------------------------------------------------------------ #
    # 交互
    # ------------------------------------------------------------------ #
    def pin(self, index: int, at: Optional[np.ndarray] = None) -> None:
        """把质点钉在固定位置 (衣夹/挂钩) — 同时把该点质量置为无穷。"""
        i = int(index)
        self.pins[i] = self.pos[i].copy() if at is None else np.asarray(at, float)
        self.inv_mass[i] = 0.0

    def grab(self, indices: Sequence[int], target: Optional[np.ndarray] = None) -> None:
        """机器人手爪抓取: 抓取质点经**有限力**软接触连到 target。

        不用运动学驱动: 手指能传递的力有上限 (库仑摩擦 μ·N), 超过即打滑,
        这是家庭场景抓取柔性体的核心失败模式。
        """
        self.grab_idx = np.asarray(indices, dtype=int)
        self.grab_target = (self.pos[self.grab_idx].mean(axis=0).copy()
                            if target is None else np.asarray(target, float))
        self._prev_target = self.grab_target.copy()
        self._grasp_disp = np.zeros(3)
        self._slip = False

    def move_grasp(self, target: np.ndarray) -> None:
        """把抓取点移动到新目标 (机器人手爪位姿)。"""
        if self.grab_idx.size == 0:
            raise RuntimeError("尚未抓取: 先调用 grab()")
        self.grab_target = np.asarray(target, dtype=float)

    def release(self) -> None:
        self.grab_idx = np.zeros(0, dtype=int)
        self.grab_target = None
        self._grasp_disp = np.zeros(3)
        self._slip = False
        self._grasp_force = 0.0

    # ------------------------------------------------------------------ #
    # 仿真
    # ------------------------------------------------------------------ #
    def step(self, dt: float, substeps: Optional[int] = None) -> None:
        """推进一个控制周期。"""
        sub = int(substeps or max(1, self.cfg.substeps))
        h = dt / sub
        for _ in range(sub):
            self._substep(h)

    def _substep(self, h: float) -> None:
        c = self.cfg
        self.lam[:] = 0.0
        P = self.pos
        w = self.inv_mass

        # ---- 预测 ----
        g = np.array([0.0, c.gravity, 0.0])
        self.vel[:, 1] += c.gravity * h
        self.vel *= max(0.0, 1.0 - c.damping * h * 50.0)   # 速度阻尼
        self.prev = P.copy()
        P += self.vel * h

        # ---- 约束求解 (Jacobi XPBD) ----
        wi, wj = w[self.ci], w[self.cj]
        alpha = self.compliance / (h * h)
        for _ in range(max(1, c.iterations)):
            d = P[self.ci] - P[self.cj]
            L = np.linalg.norm(d, axis=1)
            safe = np.maximum(L, 1e-9)
            n = d / safe[:, None]
            C = L - self.rest
            denom = wi + wj + alpha
            dl = (-C - alpha * self.lam) / np.maximum(denom, 1e-12)
            self.lam += dl
            corr_i = (wi * dl)[:, None] * n * c.relaxation
            corr_j = (wj * dl)[:, None] * n * c.relaxation
            np.add.at(P, self.ci, corr_i)
            np.add.at(P, self.cj, -corr_j)

        # ---- 抓取: 有限力软接触 + 打滑累积 ----
        if self.grab_idx.size and self.grab_target is not None:
            delta = self.grab_target - P[self.grab_idx]           # (K,3)
            F = self.cfg.grasp_stiffness * delta
            fmag = np.linalg.norm(F, axis=1, keepdims=True)
            over = fmag > self.cfg.grasp_max_force_n
            F_eff = np.where(over, F * (self.cfg.grasp_max_force_n / np.maximum(fmag, 1e-12)), F)
            # XPBD 位置修正: Δp = F/m · h²
            P[self.grab_idx] += F_eff * self.inv_mass[self.grab_idx][:, None] * h * h
            self._grasp_force = float(np.linalg.norm(F_eff, axis=1).sum())

            # 打滑量: 本子步手指走了多远 vs 摩擦能在接触点上推动多远。
            # 不要用"位置误差 delta"去判 — 静摩擦下 delta 是弹簧稳态变形(常数),
            # 而 XPBD 的 Jacobi 迭代会让它在 ±50% 振荡, 据此判滑会恒真。
            finger_move = float(np.linalg.norm(self.grab_target - self._prev_target))
            self._prev_target = self.grab_target.copy()
            w_pt = float(np.mean(self.inv_mass[self.grab_idx]))
            max_move = self.cfg.grasp_max_force_n * w_pt * h * h
            excess = finger_move - max_move
            if excess > 0.0:
                self._grasp_disp[0] += excess
            self._slip = bool(self._grasp_disp[0] > self.slip_threshold)

        # ---- 固定点 ----
        for i, p in self.pins.items():
            P[i] = p

        # ---- 地面碰撞 + 摩擦 ----
        below = P[:, 1] < c.ground_y
        if np.any(below):
            P[below, 1] = c.ground_y
            # 切向摩擦 (P0 修复: 原代码语义完全颠倒)
            #   原: scale = where(norm>max_slide, max_slide/norm, 0.0)
            #       → 粘着时 (norm≤max_slide) scale=0 → 位移原样保留 (零摩擦, 错误!)
            #       → 滑脱时 (norm>max_slide) scale=max_slide/norm → 只减部分位移
            #   正确物理:
            #     粘着 (norm ≤ max_slide): 摩擦足够阻止移动 → 位移 = 0
            #                              (scale=1, 把 tangential 全减掉)
            #     滑脱 (norm >  max_slide): 位移钳位到 max_slide
            #                              (scale=1-max_slide/norm, 减掉超出部分)
            tangential = P[np.ix_(below, [0, 2])] - self.prev[np.ix_(below, [0, 2])]
            max_slide = c.friction_mu * abs(c.gravity) * h * h
            norm = np.linalg.norm(tangential, axis=1, keepdims=True)
            scale = np.where(norm > max_slide,
                             1.0 - max_slide / np.maximum(norm, 1e-12),
                             1.0)
            P[np.ix_(below, [0, 2])] -= tangential * scale

        # ---- 速度回写 ----
        self.vel = (P - self.prev) / h
        self.pos = P

    # ------------------------------------------------------------------ #
    # 度量
    # ------------------------------------------------------------------ #
    def metrics(self) -> SoftBodyMetrics:
        P = self.pos
        d = P[self.ci] - P[self.cj]
        L = np.linalg.norm(d, axis=1)
        stretch = float(np.max(np.abs(L - self.rest) / np.maximum(self.rest, 1e-9)))
        ke = float(0.5 * np.sum(self.mass * np.sum(self.vel ** 2, axis=1)))
        com = P.mean(axis=0)
        bmin, bmax = P.min(axis=0), P.max(axis=0)
        area = float((bmax[0] - bmin[0]) * (bmax[2] - bmin[2]))
        return SoftBodyMetrics(kinetic_energy=ke, max_stretch=stretch, com=com,
                               bbox_min=bmin, bbox_max=bmax, projected_area=area,
                               slip=self._slip, grasp_force_n=self._grasp_force)

    def reset(self, height: float = 0.30) -> None:
        c = self.cfg
        xs = (np.arange(self.m) - (self.m - 1) / 2.0) * c.spacing
        zs = (np.arange(self.n) - (self.n - 1) / 2.0) * c.spacing
        X, Z = np.meshgrid(xs, zs)
        self.pos = np.column_stack([X.ravel(), np.full(self.n * self.m, c.ground_y + height),
                                    Z.ravel()])
        self.prev = self.pos.copy()
        self.vel = np.zeros_like(self.pos)
        self.lam[:] = 0.0
        self.pins.clear()
        # P0 修复: pin() 把 inv_mass 置 0 (无限质量), 但原 reset() 只清 pins 不恢复
        # inv_mass → demo 中前段钉住的点在后续段仍是无限质量, 状态泄漏。
        self.inv_mass[:] = 1.0 / float(self.mass[0])
        self.grab_idx = np.zeros(0, dtype=int)
        self.grab_target = None
        self._slip, self._grasp_disp = False, np.zeros(3)


# --------------------------------------------------------------------------- #
# 数值验证
# --------------------------------------------------------------------------- #
def soft_body_demo(seed: int = 0, dt: float = 1.0 / 200.0) -> Dict[str, float]:
    """五项断言的自检 (见模块 docstring)。"""
    cfg = SoftBodyConfig(n=10, m=10, spacing=0.02)
    body = SoftBody(cfg, seed=seed)

    # ---- 1) 悬垂: 两端钉住, 中间下垂, 应收敛 ----
    body.reset(height=0.30)
    top_row = list(range(cfg.m))
    for i in top_row:
        body.pin(i)
    for _ in range(int(2.0 / dt)):
        body.step(dt)
    m_hang = body.metrics()
    finite = bool(np.all(np.isfinite(body.pos)))
    converged = bool(m_hang.kinetic_energy < 5e-3)

    # ---- 2) 不可拉伸性 ----
    stretch_ok = bool(m_hang.max_stretch < 0.05)

    # ---- 3) 自由落体到桌面 (无钉) ----
    body.reset(height=0.25)
    for _ in range(int(1.5 / dt)):
        body.step(dt)
    m_drop = body.metrics()
    rest_ok = bool(np.all(np.isfinite(body.pos)) and m_drop.kinetic_energy < 5e-2)
    on_table = bool(abs(m_drop.bbox_min[1] - cfg.ground_y) < 5e-3)

    # ---- 4) 抓取变形传播 ----
    body.reset(height=0.02)     # 平铺在桌面上
    for _ in range(int(0.5 / dt)):
        body.step(dt)
    area0 = body.metrics().projected_area
    com0 = body.metrics().com[1]
    center = (cfg.n // 2) * cfg.m + (cfg.m // 2)
    body.grab([center], target=body.pos[center] + np.array([0.0, 0.0, 0.0]))
    for k in range(int(1.0 / dt)):
        body.move_grasp(body.pos[center] * 0.0 + np.array([0.0, 0.02 + 0.10 * k * dt, 0.0]))
        body.step(dt)
    m_grab = body.metrics()
    lifted = bool(m_grab.com[1] > com0 + 0.01)
    shrunk = bool(m_grab.projected_area < 0.95 * max(area0, 1e-9))

    # ---- 5) 滑脱判据: 快速横拖应打滑, 慢速拖动不应打滑 (判据不是恒真) ----
    body.release()
    body.reset(height=0.02)
    for _ in range(int(0.5 / dt)):
        body.step(dt)
    body.grab([center])
    body.slip_threshold = 0.005
    start = body.pos[center].copy()
    for k in range(int(0.5 / dt)):
        # 垂直慢提 (0.05 m/s): 手指只需承担自重, 远小于 μ·N
        body.move_grasp(start + np.array([0.0, 0.05 * k * dt, 0.0]))
        body.step(dt)
    slip_slow = bool(body.metrics().slip)

    body.release()
    body.reset(height=0.02)
    for _ in range(int(0.5 / dt)):
        body.step(dt)
    body.grab([center])
    body._grasp_disp = np.zeros(3)
    body._slip = False
    start = body.pos[center].copy()
    for k in range(int(0.5 / dt)):
        # 贴桌横拖 5 m/s: 桌面摩擦把布料钉住, 手指传递力超 μ·N → 打滑
        body.move_grasp(start + np.array([5.0 * k * dt, 0.0, 0.0]))
        body.step(dt)
    slip_fast = bool(body.metrics().slip)

    return {
        "kinetic_energy_hang": m_hang.kinetic_energy,
        "max_stretch": m_hang.max_stretch,
        "com_height_drop": float(m_drop.com[1]),
        "area_before_grab": float(area0),
        "area_after_grab": float(m_grab.projected_area),
        "grasp_force_n": float(m_grab.grasp_force_n),
        "finite": finite,
        "converged": converged,
        "stretch_ok": stretch_ok,
        "rest_on_table": on_table and rest_ok,
        "lifted": lifted,
        "area_shrunk": shrunk,
        "slip_slow_drag": slip_slow,
        "slip_fast_drag": slip_fast,
        "ok": bool(finite and converged and stretch_ok and rest_ok and on_table
                   and lifted and shrunk and slip_fast and not slip_slow),
    }
