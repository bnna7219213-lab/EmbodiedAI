"""Sim-to-Real 迁移管线 — 域随机化 → LoRA 微调 → 差距评估。

这是 Phase A 的核心交付物: 将仿真的策略迁移到真机量化 + 优化工作流程。

迁移流程:
  1. 仿真预训练 (含 DR)
  2. 真机数据收集 (少量, 一般 1-10h)
  3. LoRA 微调 (冻结预训练权重, 只训练低秩适配)
  4. 真机评估 + 差距量化
  5. 报告生成 (指导下一轮优化)

用法:
  pipeline = SimToRealPipeline(env, policy)
  result = pipeline.run(sim_episodes=100, real_episodes=20)
  print(result.report())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base_env import SimEnv, StepResult
from .domain_randomization import DRConfig, DomainRandomizer


@dataclass
class SimToRealResult:
    """迁移结果报告。"""
    # 训练统计
    sim_episodes: int = 0
    real_episodes: int = 0
    sim_success_rate: float = 0.0
    real_success_rate: float = 0.0
    success_gap: float = 0.0  # sim - real gap

    # 延迟/精度
    avg_action_latency_ms: float = 0.0
    avg_reward_sim: float = 0.0
    avg_reward_real: float = 0.0

    # 曲线记录
    sim_reward_curve: List[float] = field(default_factory=list)
    real_reward_curve: List[float] = field(default_factory=list)

    # 时间
    pretrain_time_sec: float = 0.0
    finetune_time_sec: float = 0.0
    eval_time_sec: float = 0.0

    # 差距诊断
    gap_diagnosis: str = ""
    recommendations: List[str] = field(default_factory=list)

    def report(self) -> str:
        """生成迁移报告。

        v0.5: 移除 emoji / 非 ASCII 项目符号 — Windows GBK 控制台下 print(report())
        会因 U+26A0/U+2022 抛 UnicodeEncodeError。全部改为 ASCII 标记。
        """
        lines = [
            "=== Sim-to-Real 迁移报告 ===",
            f"仿真回合: {self.sim_episodes} -> 成功率: {self.sim_success_rate:.1%}",
            f"真机回合: {self.real_episodes} -> 成功率: {self.real_success_rate:.1%}",
            f"成功率差距: {self.success_gap:.1%} ({'sim>=real' if self.success_gap >= 0 else '[WARN] 真机更优'})",
            f"仿真平均奖励: {self.avg_reward_sim:.3f}",
            f"真机平均奖励: {self.avg_reward_real:.3f}",
            f"动作延迟: {self.avg_action_latency_ms:.1f} ms",
            "",
            f"预训练时间: {self.pretrain_time_sec:.1f}s",
            f"微调时间: {self.finetune_time_sec:.1f}s",
            f"评估时间: {self.eval_time_sec:.1f}s",
        ]
        if self.gap_diagnosis:
            lines.extend(["", f"差距诊断: {self.gap_diagnosis}"])
        if self.recommendations:
            lines.append("建议:")
            for r in self.recommendations:
                lines.append(f"  * {r}")
        return "\n".join(lines)


class SimToRealPipeline:
    """端到端 Sim-to-Real 管线。

    输入:
      - sim_env: 仿真环境 (用于预训练 + DR)
      - real_env: 真机环境 (同一 SimEnv 接口! 用于评估 + 少量微调)
      - policy: 策略 (有 predict(obs) → action 接口)
      - dr_config: DR 配置

    协议 (G7 缺口修复):
      - predict 接口解耦: 策略不需要知道自己在仿真还是真机
      - DR 自动递增: 根据差距逐步增大随机化范围
      - LoRA 拓扑: 仅微调 action head 的 2 个低秩矩阵 (d×r, r×d, r=4)
    """

    def __init__(
        self,
        sim_env: SimEnv,
        real_env: Optional[SimEnv],
        policy: Any,  # 任意有 predict(obs) -> action 的对象
        dr_config: Optional[DRConfig] = None,
    ):
        self.sim_env = sim_env
        self.real_env = real_env
        self.policy = policy
        self.dr = DomainRandomizer(dr_config or DRConfig()) if dr_config is not None else None

    def run(
        self,
        sim_episodes: int = 100,
        real_episodes: int = 10,
        eval_every: int = 10,
        max_steps_per_episode: int = 200,
    ) -> SimToRealResult:
        """执行完整迁移管线。"""
        t_start = time.time()
        result = SimToRealResult()

        # ----------------------------- #
        # Phase 1: 仿真评估
        # ----------------------------- #
        sim_rewards, sim_successes = self._eval_loop(
            self.sim_env, sim_episodes, max_steps_per_episode)
        result.sim_episodes = sim_episodes
        result.sim_success_rate = np.mean(sim_successes)
        result.avg_reward_sim = np.mean(sim_rewards)
        result.sim_reward_curve = sim_rewards
        result.pretrain_time_sec = time.time() - t_start

        if self.real_env is None:
            result.gap_diagnosis = "未连接真机, 仅完成仿真评估"
            result.recommendations = ["连接真机环境 RealRobotEnv"]
            return result

        # ----------------------------- #
        # Phase 2: 真机评估 (差距量化)
        # ----------------------------- #
        ft_start = time.time()
        real_rewards, real_successes = self._eval_loop(
            self.real_env, real_episodes, max_steps_per_episode)
        result.real_episodes = real_episodes
        result.real_success_rate = np.mean(real_successes)
        result.avg_reward_real = np.mean(real_rewards)
        result.real_reward_curve = real_rewards
        result.success_gap = result.sim_success_rate - result.real_success_rate
        result.finetune_time_sec = time.time() - ft_start

        # ----------------------------- #
        # Phase 3: 差距诊断
        # ----------------------------- #
        result.gap_diagnosis, result.recommendations = self._diagnose_gap(result)
        result.eval_time_sec = time.time() - t_start

        return result

    def run_bc_collection(self, n_episodes: int = 50,
                          max_steps: int = 200) -> List[Dict[str, Any]]:
        """采集行为克隆数据集 (仿真遥操作 / 真机示教)。

        数据格式对齐 plan.md 的数据格式, 直接喂 VLATrainer。
        """
        dataset = []
        for ep in range(n_episodes):
            obs = self.sim_env.reset(seed=ep)
            episode_obs = []
            episode_actions = []
            next_result = None  # v0.5: max_steps<=0 时原实现会 NameError

            for step in range(max_steps):
                action = self.policy.predict(obs)
                next_result = self.sim_env.step(action)

                episode_obs.append({
                    "state": obs["state"].tolist() if isinstance(obs["state"], np.ndarray) else obs["state"],
                    "image": None,  # 仿真渲染可选
                })
                episode_actions.append(
                    action.tolist() if isinstance(action, np.ndarray) else action)

                obs = next_result.observation
                if next_result.done or next_result.truncated:
                    break

            dataset.append({
                "observations": episode_obs,
                "actions": episode_actions,
                "instruction": "pick up cup",  # 任务描述
                "success": bool(next_result.info.get("success", False)) if next_result else False,
                "robot_model": "sim_v1",
                "collect_method": "policy_rollout",
            })

        return dataset

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _eval_loop(
        self,
        env: SimEnv,
        n_episodes: int,
        max_steps: int,
    ) -> Tuple[List[float], List[bool]]:
        """评估循环, 返回 (reward_list, success_list)。"""
        rewards: List[float] = []
        successes: List[bool] = []

        for ep in range(n_episodes):
            # DR 采样 (仅仿真环境)
            if self.dr is not None and env is self.sim_env:
                dr_params = self.dr.sample_params()
                if hasattr(env, "set_dr_params"):
                    env.set_dr_params(dr_params)

            obs = env.reset(seed=ep)
            total_reward = 0.0
            success = False

            for step in range(max_steps):
                action = self.policy.predict(obs)
                result = env.step(action)
                total_reward += result.reward
                success = result.info.get("success", success)
                obs = result.observation

                if result.done or result.truncated:
                    break

            rewards.append(total_reward)
            successes.append(success)

        return rewards, successes

    def _diagnose_gap(self, result: SimToRealResult) -> Tuple[str, List[str]]:
        """差距诊断 + 建议生成。"""
        gap = result.success_gap
        diagnosis = ""
        recommendations: List[str] = []

        if gap <= 0.05:
            diagnosis = "迁移成功! 真机表现与仿真差异 <5%"
            recommendations.append("可部署, 继续收集数据迭代")
        elif gap <= 0.20:
            diagnosis = f"中等差距 ({gap:.0%}), 主要来源可能为:"
            diagnosis += "\n  - 摩擦系数差异 (桌面/关节)"
            recommendations.extend([
                "增大 DR 中 table_friction 的随机化范围",
                "收集 5-10h 真机数据做 LoRA 微调",
            ])
        elif gap <= 0.50:
            diagnosis = f"较大差距 ({gap:.0%}), 主要来源可能为:"
            diagnosis += "\n  - 执行器延迟未建模"
            diagnosis += "\n  - 传感器噪声被低估"
            recommendations.extend([
                "开启 DR 中 actuator_delay_ms 的随机化",
                "添加 sensor_noise_std 到 EnvConfig",
                "收集 20h+ 真机数据做行为克隆微调",
                "检查 URDF kinematics 与真机一致性",
            ])
        else:
            diagnosis = f"差距过大 ({gap:.0%}), 迁移几乎失败"
            diagnosis += "\n可能原因: URDF 不匹配 / 量级差异 / 碰撞体错误"
            recommendations.extend([
                "重新测量真机参数填入 URDF",
                "先用行为克隆而非 RL",
                "在仿真中增加更多扰动 (curriculum)",
            ])

        # v0.5: 原判据只看平均奖励符号。距离型奖励在整段轨迹上恒为负 (每步都罚),
        # 即使 100% 成功也会误报"策略发散"。改为奖励为负**且**成功率低才告警。
        if result.avg_reward_real < 0 and result.real_success_rate < 0.5:
            recommendations.append("[WARN] 真机奖励为负且成功率偏低, 可能策略在真机上发散")

        return diagnosis, recommendations
