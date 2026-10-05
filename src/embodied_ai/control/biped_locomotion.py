"""双足运动控制 — ZMP + 线性倒立摆 (LIPM) 步态生成 (G9 缺口)。

全身运动控制的数学基础:
  1. 线性倒立摆模型 (LIPM): 预观控制 (Preview Control) 生成质心轨迹
  2. ZMP (零力矩点): 稳定性判据, 保证不摔倒
  3. 三次样条插值: 生成平滑足部轨迹
  4. 逆运动学: 足部位姿 → 关节角度

参考:
  - Harada et al. "An Analytical Method for Real-time Gait Generation"
  - Wieber "Trajectory Free Linear Model Predictive Control"
  - Ortubicalc "Bipedal Walking Control"

用法:
  l = LIPM(width=0.2, height=0.8, dt=0.001)
  gait = ZMPGaitGenerator(l, step_time=0.8, step_length=0.15)
  for t in range(10000):
      tau = gait.get_joint_command(t * 0.001)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class LIPMParams:
    """线性倒立摆模型参数。

    动力学方程: ẍ = (g/z_c) · x
    其中 g = 重力加速度, z_c = 质心恒定高度
    """
    g: float = 9.81              # 重力 (m/s²)
    z_c: float = 0.80            # 质心恒定高度 (m)
    dt: float = 0.001            # 仿真步长 (s)
    mass: float = 20.0           # 全身质量 (kg)

    # 派生参数
    @property
    def omega(self) -> float:
        """LIPM 特征频率。"""
        return np.sqrt(self.g / self.z_c)

    @property
    def cosh_omega_dt(self) -> float:
        return np.cosh(self.omega * self.dt)

    @property
    def sinh_omega_dt(self) -> float:
        return np.sinh(self.omega * self.dt)


class LIPM:
    """线性倒立摆模型 (Linear Inverted Pendulum Model)。

    在双足机器人步态生成中, 将全身简化为:
      - 点质量 m 在质心 CoM
      - 恒定高度 z_c 的无质量腿
      - 支撑脚为瞬时旋转点

    状态方程: ẍ = omega² * x
    离散后: x_{k+1} = A x_k + B u_k
    其中 u 是 ZMP 位置 (控制输入)。
    """

    def __init__(self, params: LIPMParams):
        self.p = params
        # v0.5: 删除了原实现里计算后从未使用的简化矩阵 self._A (死代码, 且量纲
        # 与能量守恒形式不一致, 易被误当作权威离散化)。权威离散化只保留 _Ad/_Bd。
        w = params.omega
        dt = params.dt

        self._Ad = np.array([
            [np.cosh(w * dt), (1 / w) * np.sinh(w * dt)],
            [w * np.sinh(w * dt), np.cosh(w * dt)]
        ])

        # 输入: ZMP 位置→ 影响质心加速度
        self._Bd = np.array([
            [1 - np.cosh(w * dt)],
            [-w * np.sinh(w * dt)]
        ])

    def step(self, state: np.ndarray, zmp_pos: float) -> np.ndarray:
        """单步前向: state [x, ẋ], zmp_pos 标量 → next_state。

        依据: ẍ = g/z_c * (x - x_zmp)
        """
        w = self.p.omega
        dt = self.p.dt
        zc = self.p.z_c

        x, x_dot = state[0], state[1]
        # 零阶保持: zmp_pos 在本周期内恒定
        # 解析解
        x_next = (x - zmp_pos) * np.cosh(w * dt) + (x_dot / w) * np.sinh(w * dt) + zmp_pos
        x_dot_next = (x - zmp_pos) * w * np.sinh(w * dt) + x_dot * np.cosh(w * dt)

        return np.array([x_next, x_dot_next])

    def compute_zmp_from_state(self, x: float, x_ddot: float) -> float:
        """从质心加速度反推 ZMP 位置。"""
        return x - (self.p.z_c / self.p.g) * x_ddot

    def natural_dynamics(self, state: np.ndarray) -> np.ndarray:
        """开环自然演化 (ZMP = CoM 投影), 返回下一步状态。"""
        return self.step(state, state[0])  # zmp = x (质心投影)

    def calc_capture_point(self, x: float, x_dot: float) -> float:
        """计算捕获点坐标。"""
        return x + x_dot / self.p.omega

    def calc_next_foot_placement(self, x: float, x_dot: float,
                                  step_time: float) -> float:
        """根据捕获点理论计算下一支撑脚位置。

        x_foot = capture_point * cosh(omega * step_time/2) / ...
        简化: x_foot ≈ (x + ẋ * T/2) / 2
        """
        cp = self.calc_capture_point(x, x_dot)
        # 加入反馈修正
        return cp * 0.5


@dataclass
class FootstepPlan:
    """单步步态参数。"""
    support_foot: str            # "left" | "right"
    # v0.5: 0.8s → 1.0s。0.8s 时余弦升程所需 ZMP 峰值偏移 ≈0.094m, 已超出默认脚长
    # 0.16m 的半长 0.08m (判据必失稳); 1.0s 时降到 ≈0.060m, 落在支撑多边形内。
    step_time: float = 1.0       # 步周期 (s)
    step_length: float = 0.15    # 步长 (m)
    step_width: float = 0.18     # 步宽 (m), 两脚之间
    step_height: float = 0.05    # 抬脚高度 (m)
    start_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    end_pos: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.0]))

    @property
    def swing_time(self) -> float:
        """摆动相时间 (≈ step_time * 0.8)。"""
        return self.step_time * 0.8

    @property
    def double_support_time(self) -> float:
        """双支撑相时间。"""
        return self.step_time * 0.2


class GaitGenerator:
    """步态生成器 — 足步规划 → 质心 (CoM) 轨迹 + 所需 ZMP + 足端轨迹。

    v0.5 重大修复 —— 原实现在数值上是**发散**的:
      原逻辑把 ZMP 参考轨迹 (tanh 斜坡) 直接喂给 LIPM 正动力学
      `ẍ = ω²(x − x_zmp)` 做**开环前向积分**。而 LIPM 的开环解含 e^{+ωt} 不稳定
      模态: ZMP 前置于 CoM 时 `x − x_zmp < 0`, CoM 反而向后加速, 误差按 e^{ωt}
      放大。实测默认参数下 6.4s 内 CoM 跑到 −1.17e8 m, 稳定性检查 100% 失稳,
      整个 G9 模块不可用。

    修复方案 —— **ZMP 钉在支撑脚中心的 LIPM 解析解 + 周期性边界条件**:
      1. ZMP 取支撑脚中心 (这是"ZMP 判据"定义上就满足稳定的唯一选择)
      2. 令单步内相对支撑脚的偏移 d = x − x_stance 走 LIPM 精确解
         d(τ) = d₀·cosh(ωτ) + (v₀/ω)·sinh(ωτ)
      3. 施加周期性 (d(0) = −L/2, d(T) = +L/2, ẋ(T) = ẋ(0)) 解出
         v₀ = ω·(L/2)·sinh(ωT)/(cosh(ωT) − 1)
         —— 两式联立恰好是 cosh²−1 = sinh², 即 LIPM 轨道能量守恒, 故步长 L 自由
      4. CoM 由解析式直接给出 (不做积分), 数值恒有界; 每步 advancing L

    已知简化 (写在明处, 不留暗坑):
      - 这是**平面 (planar) 步态**: 额状面只有运动学横向摆动, 不参与 ZMP 动力学。
        横向稳定在真实双足上由**落脚点** (捕获点) 而非 ZMP 控制 —— 见
        `LIPM.calc_capture_point` 与 `ZMPStabilityChecker.suggest_recovery_step`。
      - 因此 `get_zmp_instant()` 返回的是**指令 ZMP** (= 支撑脚中心), 用
        `ZMPStabilityChecker(planar=True)` 检查时只看矢状面脚长区间。
    """

    def __init__(self, lipm: LIPM, footstep_plans: List[FootstepPlan], start_y: float = 0.0):
        self.lipm = lipm
        self.plans = footstep_plans
        self._current_step: int = 0
        self._t_in_step: float = 0.0
        self._state = np.array([0.0, 0.0])     # 矢状面 [x, ẋ]
        self._state_y = np.array([start_y, 0.0])  # 额状面 [y, ẏ]
        self._start_y = float(start_y)
        self._zmp = np.array([0.0, start_y])   # 最近一帧的所需 ZMP
        self._active_step: int = 0             # 本帧**实际使用**的步序号 (跨越步进边界时仍指旧步)

    def reset(self) -> None:
        self._current_step = 0
        self._t_in_step = 0.0
        self._state = np.array([0.0, 0.0])
        self._state_y = np.array([self._start_y, 0.0])
        self._zmp = np.array([0.0, self._start_y])

    def update(self, delta_t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """更新一帧, 返回 (CoM_pos, left_foot_pos, right_foot_pos)。

        矢状面: ZMP 钉在支撑脚中心的 LIPM 解析解 + **周期性边界条件**
                d(0) = −L/2, d(T) = +L/2, ẋ(T) = ẋ(0)
                ⇒ v0 = ω·(L/2)·sinh(ωT)/(cosh(ωT)−1)   (LIPM 轨道能量守恒的必然解)
        额状面: 运动学横向摆动 (足部落点是横向的控制Authority, 见捕获点理论),
                不参与 ZMP 动力学 — 平面步态 (planar gait) 简化。
        """
        if self._current_step >= len(self.plans):
            # 序列结束: 保持最后位姿
            return (np.array([self._state[0], self._state_y[0], self.lipm.p.z_c]),
                    np.array([self._state[0], 0.09, 0.0]),
                    np.array([self._state[0], -0.09, 0.0]))

        plan = self.plans[self._current_step]
        self._active_step = self._current_step
        self._t_in_step += delta_t

        x, xd, y = self._com_trajectory(plan)

        self._state = np.array([x, xd])
        self._state_y = np.array([y, 0.0])
        # 所需 ZMP = 支撑脚中心 (由构造保证落在支撑多边形内)
        self._zmp = np.array([self._stance_x(plan), self._stance_y(plan)])

        com_pos = np.array([x, y, self.lipm.p.z_c])
        lf, rf = self._foot_trajectories(plan)

        if self._t_in_step >= plan.step_time:
            self._current_step += 1
            self._t_in_step = 0.0

        return com_pos, lf, rf

    # ------------------------------------------------------------------ #
    @staticmethod
    def _stance_x(plan: FootstepPlan) -> float:
        return float(plan.start_pos[0])

    @staticmethod
    def _stance_y(plan: FootstepPlan) -> float:
        return float(-plan.start_pos[1])

    def _com_trajectory(self, plan: FootstepPlan):
        """解析 CoM 轨迹 — 返回 (x, ẋ, y)。"""
        w = self.lipm.p.omega
        T = max(plan.step_time, 1e-6)
        tau = min(max(self._t_in_step, 0.0), T)

        half = (plan.end_pos[0] - plan.start_pos[0]) / 2.0
        cosh_wT, sinh_wT = np.cosh(w * T), np.sinh(w * T)
        v0 = w * half * sinh_wT / (cosh_wT - 1.0) if cosh_wT > 1.0 + 1e-12 else 0.0
        d0 = -half

        d = d0 * np.cosh(w * tau) + (v0 / w) * np.sinh(w * tau)
        v = d0 * w * np.sinh(w * tau) + v0 * np.cosh(w * tau)
        x = self._stance_x(plan) + d

        # 额状面: 余弦摆动 (运动学)
        u = min(tau / T, 1.0)
        p = 0.5 * (1.0 - np.cos(np.pi * u))
        prev_y = self._prev_support_y()
        y = prev_y + (self._stance_y(plan) - prev_y) * p

        return x, v, y

    def _prev_support_y(self) -> float:
        """上一步支撑脚的 y (首步取初始 y)。"""
        if self._current_step == 0:
            return self._start_y
        prev = self.plans[self._current_step - 1]
        return self._stance_y(prev)

    @property
    def current_step_index(self) -> int:
        """当前执行到第几步 (v0.5 新增, 供监测/测试读取内部进度)。"""
        return min(self._current_step, max(len(self.plans) - 1, 0))

    @property
    def active_step_index(self) -> int:
        """本帧实际使用的步序号。

        `current_step_index` 在跨越步进边界的那一帧已经指向**下一步**, 而返回的
        足端/ZMP 仍属于上一步 —— 用它在边界帧去取支撑脚会取到另一只脚, 产生
        虚假失稳 (实测 1600 帧里 6 个边界帧误报)。监测请用本属性。
        """
        return min(self._active_step, max(len(self.plans) - 1, 0))

    def _foot_trajectories(self, plan: FootstepPlan) -> Tuple[np.ndarray, np.ndarray]:
        """足端轨迹: 水平线性插值 + 抛物线抬脚。

        v0.5 修复: 支撑脚的横向 (y) 坐标曾硬编码 ±0.09, 与 build_walking_plans
        生成的 start_pos/end_pos 符号相反 (左支撑脚规划 y=-0.09, 绘制却用 +0.09),
        导致 ZMP 支撑多边形与实际足位错位、稳定性检查恒为失稳。改为一律回读 plan。
        """
        t = self._t_in_step / plan.step_time
        swing_ratio = plan.swing_time / plan.step_time
        phase = min(t / swing_ratio, 1.0) if swing_ratio > 0 else 1.0

        # 约定 (与 build_walking_plans 对齐): start_pos/end_pos 描述**摆动脚**,
        # 支撑脚落在另一侧 (y 取反) 并保持不动。
        sy = plan.start_pos[1]
        ey = plan.end_pos[1]
        support_y = -sy

        swing_x = plan.start_pos[0] + (plan.end_pos[0] - plan.start_pos[0]) * phase
        h_foot = 4 * plan.step_height * phase * (1 - phase)  # 抛物线抬脚

        if plan.support_foot == "left":
            lf = np.array([plan.start_pos[0], support_y, 0.0])   # 左支撑不动
            rf = np.array([swing_x, ey, h_foot])                 # 右摆动
        else:
            rf = np.array([plan.start_pos[0], support_y, 0.0])   # 右支撑不动
            lf = np.array([swing_x, ey, h_foot])                 # 左摆动

        return lf, rf

    def get_zmp_instant(self) -> np.ndarray:
        """当前**所需 ZMP** (2D) — 稳定性监测用。

        v0.5: 原实现返回 CoM 的 x (把 CoM 当 ZMP 用, 恒等于质心投影, 判据失真)。
        改为返回逆 LIPM 解出的所需 ZMP, 才能与支撑多边形做有意义的比较。
        """
        return self._zmp.copy()

    @property
    def com_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """当前 CoM 状态 (矢状面 [x, ẋ], 额状面 [y, ẏ])。"""
        return self._state.copy(), self._state_y.copy()


class ZMPStabilityChecker:
    """ZMP 稳定性检查器 — 实时判断是否即将摔倒。

    ZMP 必须在支撑多边形内 → 稳定。
    若 ZMP 距离边界 < margin → 触发保护步态。
    """

    def __init__(self, foot_length: float = 0.16, foot_width: float = 0.08,
                 com_height: float = 0.80, gravity: float = 9.81,
                 planar: bool = False):
        self.foot_length = foot_length
        self.foot_width = foot_width
        # v0.5: 恢复步原先硬编码 sqrt(9.81/0.8), 与 LIPMParams(z_c) 脱耦 —
        # 改了质心高度后恢复步仍按 0.8m 计算。改为可注入, 默认保持原数值。
        self.com_height = com_height
        self.gravity = gravity
        # v0.5: 平面步态 (planar) 下额状面由落脚点控制, 不参与 ZMP 判据。
        # 若按 3D 判据检查, 支撑脚 y=±step_width/2 而 ZMP y≈0, dy 恒大于 half_wid,
        # 会 100% 误报失稳 —— 这正是原实现 "失稳帧 100%" 的根因之一。
        self.planar = planar

    @property
    def omega(self) -> float:
        """LIPM 特征频率 (与 LIPMParams.omega 同式)。"""
        return float(np.sqrt(self.gravity / self.com_height))

    def is_stable(self, zmp: np.ndarray, support_foot_center: np.ndarray,
                  margin: float = 0.02, other_foot_center: np.ndarray = None
                  ) -> Tuple[bool, float]:
        """检查 ZMP 是否在支撑多边形内。

        Args:
            zmp:                 当前 ZMP (2D)
            support_foot_center: 支撑脚中心 (2D)
            margin:              触发保护步态的裕度阈值 (m)
            other_foot_center:   双支撑期另一只脚的中心 (给出时用两脚包围盒)

        Returns:
            (stable, safety_margin)  safety_margin 表示到边界的最小距离 (m)
        """
        half_len = self.foot_length / 2
        half_wid = self.foot_width / 2

        lo_x, hi_x = (support_foot_center[0] - half_len, support_foot_center[0] + half_len)
        lo_y, hi_y = (support_foot_center[1] - half_wid, support_foot_center[1] + half_wid)
        if other_foot_center is not None:
            lo_x = min(lo_x, other_foot_center[0] - half_len)
            hi_x = max(hi_x, other_foot_center[0] + half_len)
            lo_y = min(lo_y, other_foot_center[1] - half_wid)
            hi_y = max(hi_y, other_foot_center[1] + half_wid)

        margin_x = min(zmp[0] - lo_x, hi_x - zmp[0])
        if self.planar:
            min_margin = margin_x
        else:
            margin_y = min(zmp[1] - lo_y, hi_y - zmp[1])
            min_margin = min(margin_x, margin_y)

        return min_margin > margin, min_margin

    def suggest_recovery_step(self, com_vel: np.ndarray, zmp: np.ndarray) -> np.ndarray:
        """建议的恢复步位置 (捕获点理论): x_capture = x_zmp + v_com / omega。"""
        capture_x = zmp[0] + com_vel[0] / self.omega
        return np.array([capture_x, 0, 0])


def build_walking_plans(n_steps: int = 10, step_length: float = 0.15,
                        step_time: float = 1.0) -> List[FootstepPlan]:
    """构建直线行走的足步序列。"""
    plans: List[FootstepPlan] = []
    x_cursor = 0.0

    for i in range(n_steps):
        if i % 2 == 0:
            support = "left"
            start = np.array([x_cursor, -0.09])
            end = np.array([x_cursor + step_length, -0.09])
        else:
            support = "right"
            start = np.array([x_cursor, 0.09])
            end = np.array([x_cursor + step_length, 0.09])

        plans.append(FootstepPlan(
            support_foot=support,
            step_time=step_time,
            step_length=step_length,
            start_pos=start,
            end_pos=end,
        ))
        x_cursor += step_length

    # 结束步 (双脚并拢)
    plans.append(FootstepPlan(
        support_foot="left" if n_steps % 2 == 0 else "right",
        step_time=step_time * 2,
        step_length=0.0,
        start_pos=np.array([x_cursor, 0.09]) if n_steps % 2 == 0 else np.array([x_cursor, -0.09]),
        end_pos=np.array([x_cursor, 0.0]),
    ))

    return plans
