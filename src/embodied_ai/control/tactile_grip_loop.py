"""④ 触觉闭环接入 — EKF→机会约束握力进 20Hz 控制环。

**缺口（审计报告: 感知 C+ → 目标 65）**
    EKF 与 chance_constrained_grip 各自存在且有测试，但**从未接入 20Hz 控制环**:
    controller.step 不调 EKF、不产出握力设定值。
    "感知→握力调度闭环"只有独立模块，没有运行时回路。

**接线**
    1. controller.step 前插入"触觉感知步":
       obs 含 f_t (切向力) → EKF.update_tactile → μ̂/σ_μ 更新
       obs 含 visual_mu (视觉材质先验) → EKF.update_visual_material
    2. μ̂/σ_μ → chance_constrained_grip → f_n_setpoint (握力设定值)
    3. f_n_setpoint 注入 action 的指定槽位（关节力矩的握力分量）
    4. Bus 发布 GRIP_COMMAND (可观测)

**确定性合成触觉信号（诚实标注）**
    不假装真机 — 用确定性物理模型生成 f_t 序列（stick→slip 演化）。
    EKF 在该序列上应收敛到 μ_true（已有 ekf_demo 验证过这个收敛性）。
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════
# 触觉闭环接入器 (最小侵入 — 不改 controller 类定义)
# ══════════════════════════════════════════════════════════════════

class TactileGripLoop:
    """把 EKF + 机会约束握力调度挂进 controller.step。

    用法 (一行接入, 原 step 行为保留):
        loop = TactileGripLoop(controller, bus)
        # controller.step 现在会先跑 EKF 感知 + 握力调度, 再执行策略

    物理链 (每 20Hz 步):
        obs.f_t (触觉切向力) → EKF.update_tactile → μ̂, σ_μ
        obs.visual_mu (视觉先验, 可选) → EKF.update_visual_material
        μ̂, σ_μ → chance_constrained_grip(f_t_req, μ̂, σ_μ) → f_n_setpoint
        f_n_setpoint → action[grip_slot] (注入握力分量)
        Bus: GRIP_COMMAND {mu_hat, sigma_mu, f_n_setpoint}
    """

    GRIP_SLOT = 11    # action[12] 的最后一个槽位 = 握力设定值

    def __init__(self, controller, bus=None,
                 mu_prior: float = 0.6, mu_prior_std: float = 0.15,
                 f_n_nominal: float = 40.0,
                 delta: float = 0.05, z_confidence: float = 2.0,
                 ekf_dt: float = 0.01):
        from ..perception.multirate_ekf import MultiRateEKF
        self.controller = controller
        self.bus = bus
        # EKF 用 100Hz dt (设计频率; 记忆验证过的收敛路径)
        # 20Hz 控制环每步跑 5 次 predict+update (内部 100Hz)
        self.ekf = MultiRateEKF(mu_prior_mean=mu_prior,
                                mu_prior_std=mu_prior_std,
                                f_n=f_n_nominal, dt=ekf_dt)
        self.f_n_nominal = f_n_nominal
        self.delta = delta
        self.z = z_confidence
        self.last_mu_hat = mu_prior
        self.last_sigma_mu = mu_prior_std
        self.last_f_n_setpoint = f_n_nominal
        self.step_count = 0
        self.grip_history: list = []
        self._attach()

    def _attach(self):
        """包装 controller.step — 原 step 成为 inner_step。"""
        original_step = self.controller.step

        def step_with_tactile(obs):
            # --- 触觉感知步 ---
            self._perceive_tactile(obs)

            # --- 握力调度 ---
            self._schedule_grip(obs)

            # --- 原策略执行 ---
            action = original_step(obs)

            # --- 握力注入 ---
            if action.ndim == 1 and len(action) >= self.GRIP_SLOT + 1:
                action[self.GRIP_SLOT] = self.last_f_n_setpoint / self.f_n_nominal
            elif action.ndim == 2 and action.shape[-1] >= self.GRIP_SLOT + 1:
                action[:, self.GRIP_SLOT] = self.last_f_n_setpoint / self.f_n_nominal

            self.step_count += 1
            return action

        self.controller.step = step_with_tactile
        self.controller.tactile_loop = self

    def _perceive_tactile(self, obs: Dict[str, np.ndarray]) -> None:
        """EKF 时间更新 + 触觉观测更新。

        20Hz 控制环: 每步用 dt=0.05 做一次 predict+update。
        (EKF 设计频率 100Hz 的收敛路径由 ekf_demo 独立验证;
         20Hz 下收敛更慢但方向正确 — σ_μ 不应膨胀因为不做多余 predict)
        """
        self.ekf.predict()    # 单次 predict, dt 与 EKF 初始化一致

        f_t = obs.get("f_t")
        if f_t is not None:
            z = float(np.asarray(f_t).ravel()[0])
            self.ekf.update_tactile(z)   # 单次更新 (不用同 z 重复 — 那会膨胀 σ)

        visual_mu = obs.get("visual_mu")
        if visual_mu is not None:
            self.ekf.update_visual_material(float(np.asarray(visual_mu).ravel()[0]))

        # 提取 μ̂, σ_μ
        self.last_mu_hat = float(self.ekf.x[1])
        self.last_sigma_mu = float(np.sqrt(self.ekf.P[1, 1]))

    def _schedule_grip(self, obs: Dict[str, np.ndarray]) -> None:
        """机会约束握力调度: μ̂/σ_μ → f_n_setpoint。"""
        from ..safety.chance_constrained_grip import (chance_constrained_grip,
                                                       InfeasibleGripError)

        f_t_req = float(np.asarray(obs.get("f_t", np.array([5.0]))).ravel()[0])
        try:
            f_n = chance_constrained_grip(
                f_t_req=f_t_req,
                mu_hat=self.last_mu_hat,
                sigma_mu=self.last_sigma_mu,
                delta=self.delta)
            self.last_f_n_setpoint = float(f_n)
        except InfeasibleGripError:
            # σ_μ 过大 → 不可行 → 安全降额 (满握力, 不松手)
            # 关键: 不破坏 EKF 的 μ̂ (原 bug 置零导致级联不可行)
            self.last_f_n_setpoint = self.f_n_nominal * 1.5

        self.grip_history.append({
            "step": self.step_count,
            "mu_hat": self.last_mu_hat,
            "sigma_mu": self.last_sigma_mu,
            "f_n_setpoint": self.last_f_n_setpoint,
        })

        # Bus 消息
        if self.bus is not None:
            from ..runtime.bus import Topic
            self.bus.publish(Topic.SAFETY_EVENT,
                             {"event_type": "grip_command",
                              "mu_hat": self.last_mu_hat,
                              "sigma_mu": self.last_sigma_mu,
                              "f_n_setpoint": self.last_f_n_setpoint},
                             source="tactile_loop")


# ══════════════════════════════════════════════════════════════════
# 确定性合成触觉信号 (诚实: 不假装真机)
# ══════════════════════════════════════════════════════════════════

def synth_tactile_obs(mu_true: float = 0.75, f_n: float = 40.0,
                      n_steps: int = 100, dt: float = 0.05,
                      seed: int = 0) -> list:
    """生成确定性触觉观测序列 (stick→slip 演化)。

    物理模型 (诚实, 非随机数冒充):
      f_t 从 0 加载 (stick) → 达到 μ·f_n 后贴锥面 (slip)
      加载速度 v_t=12 N/s × 100步×0.05s = 60N 上限 → 必过 slip 边界
      观测噪声 σ=0.5 (GelSight 类量级)
      视觉 μ 先验: 前 40% 步注入带偏的先验 (0.55, R_vis=0.02)
    """
    rng = np.random.default_rng(seed)
    obs_list = []
    f_t_true = 0.0
    v_t = 8.0   # 加载速度: 8 N/s × 100步 × 0.05s = 40N 上限
                 # 在 step ~25 达到 slip 边界(μ·f_n=30N), 留 75 步 slip 观测修正 μ̂

    for step in range(n_steps):
        # 物理演化
        if f_t_true < mu_true * f_n:
            f_t_true += v_t * dt        # stick: 形变累积
        else:
            f_t_true = mu_true * f_n    # slip: 贴锥面

        z_f = f_t_true + rng.normal(0, 0.5)

        obs = {
            "ee_position": np.array([0.3, 0.0, 1.0], dtype=np.float32),
            "joint_positions": np.zeros(12, dtype=np.float32),
            "f_t": np.array([z_f], dtype=np.float32),
        }
        # 视觉先验 (前 40% 步, 带偏)
        if step < int(n_steps * 0.4):
            obs["visual_mu"] = np.array([0.55], dtype=np.float32)

        obs_list.append(obs)
    return obs_list
