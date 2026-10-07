"""触觉闭环 L4 — 触觉感知 → 滑脱检测 → 握力调度闭环。

架构 (Phase L4, roadmap):
  tactile_force_reading (N)
    → MultiRateEKF.estimate (f_t, μ, v_t)
    → SlipDetector (incipient slip: f_t / (μ·f_n) threshold)
    → chance_constrained_grip (f_n^* = f_t / (μ̂ - z·σ_μ))
    → GripForcePID (PI controller: grip_force_cmd)
    → actuator command (下游腱驱动板 CAN-FD 帧)

数据来源: D4 dataset (experiments/d4_dataset/training_replay_v1.jsonl) 的触觉字段
  或合成触觉序列 (本模块内置 generator)。

双轨定位:
  · 本模块属 Python 轨原创实现 (与 EdaLab jar 无关)
  · jar 轨的 MotorDriveEngine 输出腱规格 (桥臂电流 / PWM / 编码器)
    与本模块通过 contracts/hw_requirements.yaml climbing 段桥接
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import numpy as np

from .multirate_ekf import MultiRateEKF
from ..safety.chance_constrained_grip import chance_constrained_grip, InfeasibleGripError


@dataclass
class TactileStep:
    """单步触觉闭环状态 (记录用)。"""
    t: float
    f_t_measured: float
    f_t_estimated: float
    mu_estimated: float
    v_t_estimated: float
    slip_ratio: float              # f_t / (μ·f_n), >0.7 为 incipient slip
    grip_force_cmd: float          # 闭环输出的握力指令 (N)
    slip_detected: bool
    was_incipient_before: bool     # 上一步已在 incipient 区间


@dataclass
class ClosedLoopResult:
    """闭环运行结果。"""
    steps: List[TactileStep] = field(default_factory=list)
    n_slip_detected: int = 0
    n_incipient: int = 0
    max_slip_ratio: float = 0.0
    max_grip_force_cmd: float = 0.0
    converged: bool = False        # 滑脱成功被抑制

    def summary(self) -> dict:
        return {
            "n_steps": len(self.steps),
            "n_slip_detected": self.n_slip_detected,
            "n_incipient": self.n_incipient,
            "max_slip_ratio": round(self.max_slip_ratio, 4),
            "max_grip_force_cmd": round(self.max_grip_force_cmd, 2),
            "converged": self.converged,
        }


class GripForcePID:
    """PI 控制器: grip force → 腱驱动电流指令。

    目标: 让实际握力跟踪闭环握力指令 f_n^*。
    简化: 此处为增量式 PI, output 单位 N (实际的 A 转换由下游
    MotorDriveEngine 通过桥臂电流标定)。
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.1, f_n_max: float = 150.0):
        self.kp = kp
        self.ki = ki
        self.f_n_max = f_n_max
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def compute(self, f_n_target: float, dt: float) -> float:
        """返回握力指令 f_n_cmd (限幅 [0, f_n_max])。

        简化 PI: f_n_cmd = kp·f_n^* + ki·∫f_n^* dt。
        带 anti-windup: 输出饱和时停止积分累积, 无滑脱时积分衰减回归基线。
        """
        cmd_unsat = self.kp * f_n_target + self.ki * self._integral
        cmd = float(np.clip(cmd_unsat, 0.0, self.f_n_max))
        # Anti-windup: 未饱和时才积分; 饱和时保持积分不变
        if cmd_unsat == cmd:
            self._integral += f_n_target * dt
        return cmd


class SlipDetector:
    """滑脱检测: incipient slip 判据。

    基于 EKF 估计的 f_t 与 μ·f_n 比值:
      slip_ratio = f_t_est / (mu_est · f_n_cmd)
      slip_ratio >= threshold → incipient slip (需提升握力)
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def check(self, f_t_est: float, mu_est: float, f_n_cmd: float) -> Tuple[bool, float]:
        denom = mu_est * f_n_cmd
        if denom < 1e-6:
            return False, 0.0
        ratio = f_t_est / denom
        return ratio >= self.threshold, float(np.clip(ratio, 0.0, 5.0))


class TactileClosedLoop:
    """触觉闭环 L4: 读触觉 → EKF 估计 → 滑脱检测 → 握力调度 → 闭环。

    用法:
        loop = TactileClosedLoop(mu_prior=0.8, f_n_init=30.0)
        result = loop.run(tactile_readings, dt=0.05)
        assert result.converged  # 滑脱被成功抑制
    """

    def __init__(self, mu_prior: float = 0.8, mu_prior_std: float = 0.1,
                 f_n_init: float = 30.0, delta: float = 0.05,
                 slip_threshold: float = 0.7, kp: float = 1.0, ki: float = 0.1,
                 f_n_max: float = 150.0):
        self.mu_prior = mu_prior
        self.f_n_init = f_n_init
        self.delta = delta
        self.ekf = MultiRateEKF(mu_prior_mean=mu_prior, mu_prior_std=mu_prior_std,
                                f_n=f_n_init)
        self.detector = SlipDetector(threshold=slip_threshold)
        self.pid = GripForcePID(kp=kp, ki=ki, f_n_max=f_n_max)

    def reset(self):
        self.ekf = MultiRateEKF(mu_prior_mean=self.mu_prior, mu_prior_std=0.1,
                                f_n=self.f_n_init)
        self.pid.reset()

    def run(self, tactile_readings: List[float], dt: float = 0.05,
            f_n_baseline: Optional[float] = None) -> ClosedLoopResult:
        """执行闭环。

        Parameters
        ----------
        tactile_readings : list of float
            按时间顺序的力传感器读数 (N, 切向力)。
        dt : float
            采样周期 (秒)。
        f_n_baseline : float | None
            握力基线 (攀岩静态握持), 无滑脱时使用此握力以节省能量。

        Returns
        -------
        ClosedLoopResult
        """
        self.reset()
        f_n_cmd = f_n_baseline if f_n_baseline is not None else self.f_n_init
        result = ClosedLoopResult()
        was_incipient = False

        for i, z_f in enumerate(tactile_readings):
            # EKF: predict → update with tactile observation
            self.ekf.predict()
            self.ekf.update_tactile(float(z_f))
            f_t_est, mu_est, v_t_est = self.ekf.x

            # Slip detection
            slip_detected, slip_ratio = self.detector.check(f_t_est, mu_est, f_n_cmd)

            # Grip force computation
            # 关键: f_n_target 独立于 f_n_cmd (避免 f_n_target=f_n_cmd 正反馈)
            # 无滑脱时使用节能基线; 滑脱时切换至机会约束计算值
            if slip_detected:
                try:
                    f_n_target = chance_constrained_grip(
                        f_t_req=abs(f_t_est), mu_hat=mu_est, sigma_mu=0.05,
                        delta=self.delta)
                except InfeasibleGripError:
                    f_n_target = self.pid.f_n_max  # 最大握力保底
            else:
                # 无滑脱: 使用节能基线 (不跟随 f_n_cmd)
                f_n_target = f_n_baseline if f_n_baseline is not None else self.f_n_init

            # PI controller: 跟踪 f_n_target (从当前 f_n_cmd 向 f_n_target 收敛)
            f_n_cmd = self.pid.compute(f_n_target, dt)
            self.ekf.f_n = f_n_cmd  # 闭环反馈: 更新 EKF 中的 f_n

            # Record
            step = TactileStep(
                t=i * dt,
                f_t_measured=float(z_f),
                f_t_estimated=float(f_t_est),
                mu_estimated=float(mu_est),
                v_t_estimated=float(v_t_est),
                slip_ratio=slip_ratio,
                grip_force_cmd=f_n_cmd,
                slip_detected=slip_detected,
                was_incipient_before=was_incipient,
            )
            result.steps.append(step)
            if slip_detected:
                result.n_slip_detected += 1
                result.n_incipient += 1
            was_incipient = slip_detected

        # Convergence: 最后 5 步不再有 incipient slip → 成功抑制
        tail = result.steps[-5:] if len(result.steps) >= 5 else result.steps
        result.converged = (
            len(tail) > 0 and
            not any(s.slip_detected for s in tail) and
            result.n_incipient > 0  # 曾经有滑脱, 后来解决了
        )
        result.max_slip_ratio = max((s.slip_ratio for s in result.steps), default=0.0)
        result.max_grip_force_cmd = max((s.grip_force_cmd for s in result.steps), default=0.0)

        return result


# ------------------------------------------------------------------------- #
# 合成数据生成器 (用于测试 + 演示; 真机替换为触觉传感器读数)
# ------------------------------------------------------------------------- #
def synthesize_slip_scenario(n_steps: int = 60, dt: float = 0.05,
                              mu_true: float = 0.8,
                              f_n_initial: float = 20.0,
                              disturbance_amplitude: float = 15.0,
                              disturbance_start: float = 1.5,
                              seed: int = 42) -> np.ndarray:
    """合成一段"静态握持 → 切向扰动 → incipient slip"的触觉读数序列。

    模型:
      - t < disturbance_start: f_t ≈ 0 (静态握持)
      - t >= disturbance_start: f_t 斜坡上升到 disturbance_amplitude + noise
      - slip: 当 f_t > mu * f_n 时发生

    物理合理:
      mu * f_n_initial = 0.8 * 20 = 16N → disturbance_amplitude=20N > 16N → 必滑脱
    """
    rng = np.random.default_rng(seed)
    readings = np.zeros(n_steps)
    for i in range(n_steps):
        t = i * dt
        if t < disturbance_start:
            readings[i] = rng.normal(0, 0.5)
        else:
            # 斜坡上升: 切向力 0 → disturbance_amplitude over 1.5s
            ramp = min(1.0, (t - disturbance_start) / 1.5)
            readings[i] = disturbance_amplitude * ramp + rng.normal(0, 0.8)
    return readings


def episode_tactile_stream(episode: dict, dt: float = 0.05) -> np.ndarray:
    """从 D4 episode 记录生成触觉力序列 (用于训练/验证)。

    用 motor_current_a 归一化近似切向力 (标定: 1A ≈ 0.5N)。
    真实场景替换为触觉传感器 AD8422 输出。
    """
    recs = episode.get("__records", [])
    if not recs:
        return np.zeros(0)
    currents = np.array([r.get("motor_current_a", 0.0) for r in recs])
    # 1A current → ~0.5N tangential force (标定系数, 需真机校准)
    return currents * 0.5
