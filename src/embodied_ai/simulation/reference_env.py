"""参考仿真环境 — 零外部依赖 (纯 numpy) 的 SimEnv 实现 (v0.5 新增, G7 缺口补齐)。

背景:
  MuJoCoEnv / PyBulletEnv 分别依赖 mujoco / pybullet 两个重量级外部包。本机 (以及
  绝大多数 CI) 均未安装, 导致整个仿真栈在默认环境下**完全不可执行**: `reset/step/
  DR/Sim-to-Real` 一条链路都跑不通, 只能靠文档宣称。这是 G7 缺口修复后遗留的
  "不可验证" 缺口。

ReferenceEnv 的定位:
  - 一个受控的二阶点质量 (double integrator) + 目标趋近任务, 用半隐式欧拉积分;
  - 完整实现 SimEnv 协议 (reset/step/close/set_dr_params/record/export);
  - 支持 contracts 级别的观测噪声 / 动作噪声 / 延迟, 因此同一套代码就能表达
    "仿真侧" 与 "真机侧" 的差异, 让 SimToRealPipeline 可以**离线可复现**地跑通;
  - 不是物理精度承诺, 而是"管线可验证性"承诺 — 精度由 MuJoCoEnv 承担。

DR 参数映射 (与 domain_randomization 默认规格一致):
  link_mass -> 质量; joint_friction -> 粘性阻尼; gravity -> 重力项;
  actuator_delay_ms -> 动作延迟; encoder_noise_deg -> 观测噪声。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base_env import EnvConfig, SimEnv, StepResult

__all__ = ["ReferenceEnv", "PointMassEnv"]

# ReferenceEnv 的别名 (README/CLI 中以 point_mass 场景名引用)
PointMassEnv = None  # 下方赋值为 ReferenceEnv, 保持 `PointMassEnv` 可导入


class ReferenceEnv(SimEnv):
    """纯 numpy 参考环境 — 二阶点质量趋近目标 + 可切换的"真机退化"通道。"""

    def __init__(self, config: EnvConfig, real: bool = False):
        super().__init__(config)
        self.real = real
        self._rng = np.random.default_rng(0)

        # 可随机化的物理参数 (名义值)
        self.mass = 1.0
        self.damping = 0.6
        self.gravity = 0.0
        self.actuator_delay_steps = 0

        # 任务目标 (前 action_dim 维)
        self.target = np.zeros(config.action_dim)
        self.target[0] = 0.4

        self._state = np.zeros(config.state_dim)
        self._delay_queue: list = []
        self._last_obs: Optional[Dict[str, Any]] = None
        self._reset_target()

    # ------------------------------------------------------------------ #
    # 任务/初始状态
    # ------------------------------------------------------------------ #
    def _reset_target(self) -> None:
        d = min(self.config.action_dim, len(self.target))
        self.target[:] = 0.0
        if d > 0:
            self.target[0] = 0.4

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            np.random.seed(seed)
        self._state = np.zeros(self.config.state_dim)
        self._delay_queue = []
        self._last_obs = None
        self.step_count = 0
        self.clear_episode()
        return self._build_observation(self._state)

    # ------------------------------------------------------------------ #
    # 动力学
    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=float).reshape(-1)
        action = self._apply_action(action)

        # 执行器延迟 (真机/DR 注入): 以步为单位排队
        if self.actuator_delay_steps > 0:
            self._delay_queue.append(action)
            if len(self._delay_queue) > self.actuator_delay_steps:
                action = self._delay_queue.pop(0)
            else:
                action = np.zeros(self.config.action_dim)

        a = action[: self.config.action_dim]
        pos = self._state[: self.config.action_dim]
        vel = self._state[self.config.action_dim: 2 * self.config.action_dim]

        dt = self.config.dt
        # 半隐式欧拉: a = (F - c*v + g)/m
        acc = (a - self.damping * vel + self.gravity) / max(self.mass, 1e-6)
        vel_next = vel + acc * dt
        pos_next = pos + vel_next * dt

        n = self.config.action_dim
        self._state[:n] = pos_next
        self._state[n:2 * n] = vel_next

        self.step_count += 1
        obs = self._build_observation(self._state)

        reward, done, info = self._compute_reward(pos_next, vel_next, a)
        truncated = self.step_count >= self.config.max_episode_steps
        # v0.5: 原实现 (含 MuJoCoEnv) 从未调用 record_step, export_episode() 恒为空,
        # SimToRealPipeline.run_bc_collection 之外的 BC 数据链实际是断的。
        self.record_step(self._last_obs if self._last_obs is not None else obs,
                         a, reward, obs, done or truncated)
        self._last_obs = obs
        return StepResult(observation=obs, reward=reward, done=done,
                          truncated=truncated, info=info)

    def _compute_reward(self, pos: np.ndarray, vel: np.ndarray, action: np.ndarray):
        err = pos - self.target
        dist = float(np.linalg.norm(err))
        reward = -dist - 0.01 * float(np.dot(action, action)) - 0.02 * float(np.dot(vel, vel))
        success = dist < 0.05
        if success:
            reward += 10.0
        info = {"success": success, "distance": dist,
                "is_real": self.real, "step": self.step_count}
        return reward, success, info

    def close(self) -> None:
        self._delay_queue = []

    # ------------------------------------------------------------------ #
    # DR / 真机退化
    # ------------------------------------------------------------------ #
    def set_dr_params(self, params: Dict[str, float]) -> None:
        """把 DR 采样结果映射到参考环境的可随机化物理参数。"""
        self.mass = float(params.get("link_mass", self.mass))
        self.damping = float(params.get("joint_friction", self.damping))
        self.gravity = float(params.get("gravity", self.gravity))
        delay_ms = params.get("actuator_delay_ms")
        if delay_ms is not None:
            self.actuator_delay_steps = int(round(delay_ms / 1000.0 / self.config.dt))
        noise_deg = params.get("encoder_noise_deg")
        if noise_deg is not None:
            # 编码器噪声以度给出, 折算到观测噪声 (rad)
            self.config.obs_noise_std = max(self.config.obs_noise_std,
                                            float(np.deg2rad(noise_deg)))

    def sample_action(self) -> np.ndarray:
        """参考环境下的随机基线: 均匀采样, 幅度收敛到能驱动任务的量级。"""
        return np.random.uniform(-1.0, 1.0, size=self.config.action_dim)

    def reward_terms(self) -> Dict[str, float]:
        pos = self._state[: self.config.action_dim]
        vel = self._state[self.config.action_dim: 2 * self.config.action_dim]
        return {"distance": float(np.linalg.norm(pos - self.target)),
                "speed": float(np.linalg.norm(vel))}


PointMassEnv = ReferenceEnv
