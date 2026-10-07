"""策略梯度 RL: PPO 与 SAC — v0.7 补齐"RL 只有数学文档、没有可运行代码"的缺口。

缺口背景:
  评估报告称「RL 理论完整 (~90%)」但「无 PPO/SAC 真实训练管线」。docs/01-ai-model/
  reinforcement-learning.md 把 MDP → 贝尔曼 → 策略梯度 → PPO 推导得很完整, 而
  src/ 下**一行 RL 代码都没有** —— 数学与实现之间断链。

本模块把文档里的公式逐条落成可运行代码 (纯 numpy, 依赖 models/nn_core):

  PPO (Schulman et al. 2017)
    - GAE:        A_t = Σ_l (γλ)^l δ_{t+l},  δ_t = r_t + γV(s_{t+1}) - V(s_t)
    - 裁剪目标:    L = E[ min(ρ_t A_t, clip(ρ_t, 1-ε, 1+ε) A_t) ],  ρ = π/π_old
    - 价值/熵:    L = L^clip - c1·L^VF + c2·H
  SAC (Haarnoja et al. 2018)
    - 双 Q 最小值: y = r + γ( min_j Q̄_j(s',ã') - α log π(ã'|s') )
    - 策略目标:    J = E[ α log π(a|s) - min_j Q_j(s,a) ]   (重参数化 + tanh)
    - 温度自适应:  α ← α - η·( -log π - H_target ),  H_target = -dim(A)
    - tanh 修正:   log π(a|s) = log p(u|s) - Σ log(1 - tanh²(u) + eps)

可验证性 (CI 断言):
  训练后策略的平均回合回报必须**显著高于随机基线**, 且 PPO 的价值函数
  预测误差必须下降 (否则是优势估计写错, 最常见的静默 bug)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .nn_core import MLP, Adam, mse_grad, mse_loss

__all__ = ["PPOConfig", "PPOAgent", "SACConfig", "SACAgent", "ReplayBuffer",
           "RolloutBuffer", "rl_demo"]


def _soft_clip(x: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(x, -limit, limit)


# =========================================================================== #
# PPO
# =========================================================================== #
@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95            # GAE λ
    clip: float = 0.2            # ε
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    epochs: int = 4
    batch_size: int = 64
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None   # 早停阈值 (None = 不早停)


@dataclass
class RolloutBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    logps: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(self, o, a, lp, r, v, d) -> None:
        self.obs.append(np.asarray(o, dtype=float).reshape(-1))
        self.actions.append(np.asarray(a, dtype=float).reshape(-1))
        self.logps.append(float(lp))
        self.rewards.append(float(r))
        self.values.append(float(v))
        self.dones.append(bool(d))

    def clear(self) -> None:
        self.__init__()  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.rewards)

    def arrays(self) -> Dict[str, np.ndarray]:
        return {
            "obs": np.array(self.obs, dtype=float),
            "actions": np.array(self.actions, dtype=float),
            "logps": np.array(self.logps, dtype=float),
            "rewards": np.array(self.rewards, dtype=float),
            "values": np.array(self.values, dtype=float),
            "dones": np.array(self.dones, dtype=bool),
        }

    def compute_gae(self, gamma: float, lam: float,
                    last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """GAE(γ,λ) 优势 + TD(λ) 回报。"""
        r, v, d = self.rewards, self.values, self.dones
        n = len(r)
        adv = np.zeros(n)
        gae = 0.0
        for t in reversed(range(n)):
            next_v = last_value if t == n - 1 else v[t + 1]
            nonterm = 0.0 if (d[t] or (t == n - 1)) else 1.0
            delta = r[t] + gamma * next_v * nonterm - v[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + np.array(v)
        return adv, ret


class PPOAgent:
    """PPO (clip) — 高斯策略 + 价值网络。"""

    def __init__(self, obs_dim: int, act_dim: int, act_limit: float = 1.0,
                 hidden: Sequence[int] = (64, 64), seed: int = 0,
                 cfg: PPOConfig = None):
        self.cfg = cfg or PPOConfig()
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.act_limit = float(act_limit)
        self.rng = np.random.default_rng(seed)

        self.actor = MLP([obs_dim, *hidden, act_dim], hidden_act="relu",
                         out_act="none", seed=seed)
        self.critic = MLP([obs_dim, *hidden, 1], hidden_act="relu",
                          out_act="none", seed=seed + 1)
        self.log_std = np.full(act_dim, -0.5)
        self.log_std_grad = np.zeros(act_dim)
        self.actor_opt = Adam([(self.actor.params, self.actor.grads),
                               ([self.log_std], [self.log_std_grad])], lr=self.cfg.lr)
        self.critic_opt = Adam([(self.critic.params, self.critic.grads)], lr=self.cfg.lr)
        self._last_diagnostics: Dict[str, float] = {}

    # ---------------------------------------------------------------- #
    def _dist(self, obs_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.actor.forward(obs_b)
        std = np.exp(np.clip(self.log_std, -4.0, 1.0))
        return mean, std

    def act(self, obs: Any, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        s = obs["state"] if isinstance(obs, dict) else obs
        s = np.atleast_2d(np.asarray(s, dtype=float))[:, : self.obs_dim]
        mean, std = self._dist(s)
        if deterministic:
            a = mean[0]
        else:
            a = mean[0] + std * self.rng.standard_normal(self.act_dim)
        u = (a - mean[0]) / std
        logp = float(np.sum(-0.5 * (u ** 2) - np.log(std) - 0.5 * np.log(2 * np.pi)))
        v = float(self.critic.forward(s)[0, 0])
        return _soft_clip(a, self.act_limit), logp, v

    def predict(self, obs: Any) -> np.ndarray:
        """环境接口 (与 BC/Sim-to-Real 的 policy 协议一致)。"""
        return self.act(obs, deterministic=True)[0]

    def value(self, obs: Any) -> float:
        s = obs["state"] if isinstance(obs, dict) else obs
        s = np.atleast_2d(np.asarray(s, dtype=float))[:, : self.obs_dim]
        return float(self.critic.forward(s)[0, 0])

    # ---------------------------------------------------------------- #
    def update(self, buf: RolloutBuffer, last_value: float = 0.0) -> Dict[str, float]:
        d = buf.arrays()
        obs, act, old_logp = d["obs"], d["actions"], d["logps"]
        adv, ret = buf.compute_gae(self.cfg.gamma, self.cfg.lam, last_value)
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 优势标准化
        n = len(obs)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        cnt = 0
        for _ in range(self.cfg.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.cfg.batch_size):
                b = idx[start: start + self.cfg.batch_size]
                if len(b) == 0:
                    continue
                ob, ab = obs[b], act[b]
                adv_b, ret_b, oldp_b = adv[b], ret[b], old_logp[b]

                # ---- 策略梯度 ----
                mean, std = self._dist(ob)
                u = (ab - mean) / std
                logp = np.sum(-0.5 * u ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi), axis=-1)
                ratio = np.exp(logp - oldp_b)
                clipped = np.clip(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip)
                surr = -np.minimum(ratio * adv_b, clipped * adv_b)   # 取负 → 梯度上升
                ent = np.sum(0.5 * np.log(2 * np.pi * np.e * std ** 2), axis=-1)
                pol_loss = float(np.mean(surr)) - self.cfg.ent_coef * float(np.mean(ent))

                # dL/dlogp: 只保留"未被裁剪掉的"分支。
                # clip 分支 (ratio 落在 [1-ε,1+ε] 之外且方向不利) 的导数必须是 0 —
                # 早期版本这里填的是 clipped 而非 0, 饱和样本仍在推 ratio 继续外扩,
                # 表现为策略熵单调涨、回报不涨。
                take_min = (ratio * adv_b) <= (clipped * adv_b)
                dlogp = -(np.where(take_min, ratio, 0.0) * adv_b) / len(b)
                # ∂logp/∂mean = u/std ;  ∂logp/∂log_std = u² − 1
                dmean = dlogp[:, None] * u / std[None, :]
                dls = dlogp[:, None] * (u ** 2 - 1.0)
                self.actor.zero_grad()
                self.actor.backward(dmean)
                # 熵奖励: H = Σ(log σ + 0.5 log 2πe) ⇒ ∂H/∂log_std = 1,
                # loss 含 −c₂·H ⇒ 梯度项 −c₂ (与 dls 同为"每样本平均"尺度)
                self.log_std_grad.fill(0.0)
                self.log_std_grad += dls.sum(axis=0) - self.cfg.ent_coef
                self.actor.clip_grad_norm(self.cfg.max_grad_norm)
                self.actor_opt.step()

                # ---- 价值 ----
                vpred = self.critic.forward(ob)[:, 0]
                vloss = mse_loss(vpred, ret_b)
                self.critic.zero_grad()
                self.critic.backward(
                    np.column_stack([mse_grad(vpred[:, None], ret_b[:, None])[:, 0]]))
                self.critic.clip_grad_norm(self.cfg.max_grad_norm)
                self.critic_opt.step()

                stats["policy_loss"] += pol_loss
                stats["value_loss"] += float(vloss)
                stats["entropy"] += float(np.mean(ent))
                stats["approx_kl"] += float(np.mean(oldp_b - logp))
                stats["clip_frac"] += float(np.mean(np.abs(ratio - 1.0) > self.cfg.clip))
                cnt += 1
        if cnt:
            stats = {k: v / cnt for k, v in stats.items()}
        self._last_diagnostics = stats
        return stats

    # ---------------------------------------------------------------- #
    def train(self, env, iters: int = 30, horizon: int = 200,
              seed: int = 0) -> Dict[str, Any]:
        """在环境上训练, 返回回报曲线与诊断。"""
        returns: List[float] = []
        vf_errors: List[float] = []
        for it in range(iters):
            obs = env.reset(seed=seed + it)
            buf = RolloutBuffer()
            ep_ret, ep_len = 0.0, 0
            for t in range(horizon):
                a, logp, v = self.act(obs)
                res = env.step(a)
                done = bool(res.done or res.truncated)
                buf.add(obs["state"], a, logp, res.reward, v, done)
                ep_ret += res.reward
                ep_len += 1
                obs = res.observation
                if done:
                    break
            last_v = self.value(obs) if not (len(buf) and buf.dones[-1]) else 0.0
            stats = self.update(buf, last_value=last_v)
            returns.append(ep_ret)
            vf_errors.append(stats["value_loss"])
        return {"returns": returns, "value_loss": vf_errors,
                "initial_return": float(np.mean(returns[: max(1, len(returns) // 5)])),
                "final_return": float(np.mean(returns[-max(1, len(returns) // 5):])),
                "diagnostics": self._last_diagnostics}


# =========================================================================== #
# SAC
# =========================================================================== #
class ReplayBuffer:
    """环形经验回放。"""

    def __init__(self, obs_dim: int, act_dim: int, size: int = 50_000, seed: int = 0):
        self.obs = np.zeros((size, obs_dim), dtype=float)
        self.next_obs = np.zeros((size, obs_dim), dtype=float)
        self.acts = np.zeros((size, act_dim), dtype=float)
        self.rews = np.zeros(size, dtype=float)
        self.dones = np.zeros(size, dtype=bool)
        self.ptr, self.size, self.max = 0, 0, size
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, no, d) -> None:
        i = self.ptr
        self.obs[i] = np.asarray(o, dtype=float).reshape(-1)[: self.obs.shape[1]]
        self.next_obs[i] = np.asarray(no, dtype=float).reshape(-1)[: self.obs.shape[1]]
        self.acts[i] = np.asarray(a, dtype=float).reshape(-1)[: self.acts.shape[1]]
        self.rews[i] = float(r)
        self.dones[i] = bool(d)
        self.ptr = (i + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch: int) -> Dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=min(batch, self.size))
        return {"obs": self.obs[idx], "acts": self.acts[idx], "rews": self.rews[idx],
                "next_obs": self.next_obs[idx], "dones": self.dones[idx]}


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005          # 目标网络软更新系数
    lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    start_steps: int = 200      # 纯探索步数
    updates_per_step: int = 1
    hidden: Tuple[int, ...] = (64, 64)


class SACAgent:
    """SAC (自动温度版) — tanh 高斯 + 双 Q + 目标网络。"""

    def __init__(self, obs_dim: int, act_dim: int, act_limit: float = 1.0,
                 seed: int = 0, cfg: SACConfig = None):
        self.cfg = cfg or SACConfig()
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.act_limit = float(act_limit)
        self.rng = np.random.default_rng(seed)
        h = list(self.cfg.hidden)

        in_dim = obs_dim + act_dim
        self.q1 = MLP([in_dim, *h, 1], seed=seed)
        self.q2 = MLP([in_dim, *h, 1], seed=seed + 1)
        self.q1_t = MLP([in_dim, *h, 1], seed=seed)
        self.q2_t = MLP([in_dim, *h, 1], seed=seed + 1)
        self.q1_t.copy_from(self.q1)
        self.q2_t.copy_from(self.q2)

        self.actor = MLP([obs_dim, *h, act_dim * 2], seed=seed + 2)  # (mean, log_std)
        self.q_opt = Adam([(self.q1.params, self.q1.grads),
                           (self.q2.params, self.q2.grads)], lr=self.cfg.lr)
        self.pi_opt = Adam([(self.actor.params, self.actor.grads)], lr=self.cfg.lr)

        self.log_alpha = np.array([0.0])
        self.log_alpha_grad = np.zeros(1)
        self.alpha_opt = Adam([([self.log_alpha], [self.log_alpha_grad])],
                              lr=self.cfg.alpha_lr)
        self.target_entropy = -float(act_dim)
        self._diag: Dict[str, float] = {}

    # ---------------------------------------------------------------- #
    @property
    def alpha(self) -> float:
        return float(np.exp(np.clip(self.log_alpha[0], -10.0, 5.0)))

    def _sample_action(self, obs_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """重参数化采样。

        返回 (动作 a, log π(a|s), 反向所需中间量 dict)。
        中间量必须留存: 策略梯度要沿 u → (mean, log_std) 回传, 而 ε 是采样时
        抽的, 事后再抽会得到**不同的**样本, 梯度就成了噪声。
        """
        out = self.actor.forward(obs_b)
        mean, raw_ls = out[:, : self.act_dim], out[:, self.act_dim:]
        log_std = np.clip(raw_ls, -4.0, 1.0)
        std = np.exp(log_std)
        eps_n = self.rng.standard_normal(mean.shape)
        u = mean + std * eps_n
        a = np.tanh(u)
        logp = np.sum(-0.5 * eps_n ** 2 - log_std - 0.5 * np.log(2 * np.pi), axis=-1)
        # tanh Jacobian 修正: log π(a) = log p(u) - Σ log(1 - tanh²u + eps)
        logp = logp - np.sum(np.log(np.maximum(1.0 - a ** 2, 1e-6) + 1e-8), axis=-1)
        cache = {"u": u, "mean": mean, "std": std, "eps": eps_n,
                 "raw_ls": raw_ls, "a": a}
        return a, logp, cache

    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        s = obs["state"] if isinstance(obs, dict) else obs
        s = np.atleast_2d(np.asarray(s, dtype=float))[:, : self.obs_dim]
        out = self.actor.forward(s)
        mean = out[:, : self.act_dim]
        if deterministic:
            return np.tanh(mean)[0] * self.act_limit
        a, _, _ = self._sample_action(s)
        return a[0] * self.act_limit

    def predict(self, obs: Any) -> np.ndarray:
        return self.act(obs, deterministic=True)

    # ---------------------------------------------------------------- #
    def update(self, buf: ReplayBuffer) -> Dict[str, float]:
        if buf.size < self.cfg.batch_size:
            return {"q_loss": 0.0, "pi_loss": 0.0, "alpha": self.alpha}
        b = buf.sample(self.cfg.batch_size)
        ob, ab, rb, nob, db = b["obs"], b["acts"], b["rews"], b["next_obs"], b["dones"]

        # ---- 目标 Q ----
        an, logpn, _ = self._sample_action(nob)
        q1n = self.q1_t.forward(np.column_stack([nob, an]))[:, 0]
        q2n = self.q2_t.forward(np.column_stack([nob, an]))[:, 0]
        q_target = rb + self.cfg.gamma * (1.0 - db.astype(float)) * (
            np.minimum(q1n, q2n) - self.alpha * logpn)

        # ---- Q 更新 ----
        sa = np.column_stack([ob, ab])
        q1v = self.q1.forward(sa)[:, 0]
        q2v = self.q2.forward(sa)[:, 0]
        self.q1.zero_grad()
        self.q2.zero_grad()
        g1 = mse_grad(q1v[:, None], q_target[:, None])[:, 0]
        g2 = mse_grad(q2v[:, None], q_target[:, None])[:, 0]
        self.q1.backward(g1[:, None])
        self.q2.backward(g2[:, None])
        self.q_opt.step()
        q_loss = float(0.5 * (mse_loss(q1v, q_target) + mse_loss(q2v, q_target)))

        # ---- 策略更新: 最大化 J = E[α·log π(a|s) − min_j Q_j(s,a)] ----
        a_new, logp_new, cache = self._sample_action(ob)
        sa_new = np.column_stack([ob, a_new])
        q_min = np.minimum(self.q1.forward(sa_new)[:, 0], self.q2.forward(sa_new)[:, 0])
        pi_obj = float(np.mean(self.alpha * logp_new - q_min))

        # 重参数化下 log π 写成 ε 的函数 (ε 是采样噪声, 视作常数):
        #   log π = Σ[−½ε² − log σ − ½log2π] − Σ log(1 − tanh²u),  u = mean + σ·ε
        # 因此 ∂logπ/∂mean = 2a (只经 tanh 修正项), ∂logπ/∂logσ = −1 + 2a·σ·ε。
        # ← 常见错误: 把 −½ε² 也当作 u 的函数求导, 多出一项 −α·ε/σ, 它会把 log_std
        #   一路推向 −∞ (密度爆炸, 实测 mean_logπ 涨到 +34), 策略在几步内退化为确定。
        # J = Q − α·log π (最大化); 优化器做梯度下降 ⇒ 传入 grad = −∂J/∂param
        u, mean, std, eps_n = cache["u"], cache["mean"], cache["std"], cache["eps"]
        dq_da = self._dq_da(sa_new)
        dq_du = dq_da * (1.0 - a_new ** 2)          # ∂Q/∂u  (a = tanh u)
        dlogp_mean = 2.0 * a_new                    # ∂logπ/∂mean
        dlogp_logstd = -1.0 + 2.0 * a_new * std * eps_n   # ∂logπ/∂logσ
        grad_mean = -(dq_du - self.alpha * dlogp_mean) / len(ob)
        grad_logstd = -(dq_du * std * eps_n - self.alpha * dlogp_logstd) / len(ob)
        grad_logstd *= ((cache["raw_ls"] > -4.0) & (cache["raw_ls"] < 1.0))  # clip 区无梯度
        self.actor.zero_grad()
        self.actor.backward(np.column_stack([grad_mean, grad_logstd]))
        self.actor.clip_grad_norm(1.0)
        self.pi_opt.step()

        # ---- 温度自适应: min_α −α·(log π + H_target) ----
        # 熵高于目标 (logp + H_target < 0) → α 减小; 熵不足 → α 增大。
        self.log_alpha_grad.fill(0.0)
        self.log_alpha_grad[0] = -float(np.mean(logp_new + self.target_entropy))
        self.alpha_opt.step()

        # ---- 目标网络软更新 ----
        self.q1_t.copy_from(self.q1, self.cfg.tau)
        self.q2_t.copy_from(self.q2, self.cfg.tau)

        self._diag = {"q_loss": q_loss, "pi_objective": pi_obj, "alpha": self.alpha,
                      "mean_logp": float(np.mean(logp_new))}
        return self._diag

    def _dq_da(self, sa: np.ndarray) -> np.ndarray:
        """∂(min_j Q_j)/∂a — 解析反向 (取 min 分支), 再裁掉 obs 段。

        用 MLP.backward 得到输入梯度后取 [:, obs_dim:] 即可, 一次前向+一次反向;
        中心差分要 2·A 次前向 (A=动作维), 是这个训练循环里最贵的一步。
        副作用: 会往 q1/q2 的 grads 里累积梯度, 下一轮 update 开头已 zero_grad。
        """
        q1v = self.q1.forward(sa)[:, 0]
        q2v = self.q2.forward(sa)[:, 0]
        use_q1 = (q1v <= q2v)[:, None]
        self.q1.zero_grad()
        self.q1.forward(sa)
        g1 = self.q1.backward(np.ones((len(sa), 1)))
        self.q2.zero_grad()
        self.q2.forward(sa)
        g2 = self.q2.backward(np.ones((len(sa), 1)))
        return np.where(use_q1, g1[:, self.obs_dim:], g2[:, self.obs_dim:])

    # ---------------------------------------------------------------- #
    def train(self, env, total_steps: int = 3000, horizon: int = 200,
              seed: int = 0) -> Dict[str, Any]:
        buf = ReplayBuffer(self.obs_dim, self.act_dim, self.cfg.buffer_size, seed=seed)
        returns, ep_ret = [], 0.0
        obs = env.reset(seed=seed)
        for step in range(total_steps):
            if step < self.cfg.start_steps:
                a = np.random.uniform(-self.act_limit, self.act_limit, size=self.act_dim)
            else:
                a = self.act(obs)
            res = env.step(a)
            done = bool(res.done or res.truncated)
            buf.add(obs["state"], a, res.reward, res.observation["state"], done)
            ep_ret += res.reward
            obs = res.observation
            for _ in range(self.cfg.updates_per_step):
                self.update(buf)
            if done or (step + 1) % horizon == 0:
                returns.append(ep_ret)
                ep_ret = 0.0
                obs = env.reset(seed=seed + step)
        k = max(1, len(returns) // 5)
        return {"returns": returns,
                "initial_return": float(np.mean(returns[:k])) if returns else 0.0,
                "final_return": float(np.mean(returns[-k:])) if returns else 0.0,
                "diagnostics": dict(self._diag)}


# =========================================================================== #
# 自检
# =========================================================================== #
def _random_baseline(env, n_episodes: int = 5, horizon: int = 200) -> float:
    rets = []
    for ep in range(n_episodes):
        obs = env.reset(seed=5000 + ep)
        total = 0.0
        for _ in range(horizon):
            res = env.step(env.sample_action())
            total += res.reward
            obs = res.observation
            if res.done or res.truncated:
                break
        rets.append(total)
    return float(np.mean(rets))


def rl_demo(algo: str = "ppo", seed: int = 0, **kw) -> Dict[str, Any]:
    """在参考环境上训练并断言"比随机强"。返回可断言字典。"""
    from ..simulation import EnvConfig, make_env

    # 默认步数按"能稳定超过随机基线"标定: PPO ~120 次迭代 (≈2s),
    # SAC 是 off-policy + 单步更新, 需要约 12k 环境步 (≈12s) 才稳定见效。
    iters = int(kw.get("iters", 120 if algo == "ppo" else 12000))
    horizon = int(kw.get("horizon", 200))
    cfg = EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3, dt=0.01,
                    max_episode_steps=horizon)
    env = make_env(cfg, backend="reference")
    baseline = _random_baseline(env, n_episodes=3, horizon=horizon)

    if algo == "ppo":
        agent = PPOAgent(6, 3, act_limit=1.0, seed=seed)
        out = agent.train(env, iters=iters, horizon=horizon, seed=seed)
    elif algo == "sac":
        agent = SACAgent(6, 3, act_limit=1.0, seed=seed)
        out = agent.train(env, total_steps=iters, horizon=horizon, seed=seed)
    else:
        raise ValueError(f"未知算法: {algo} (支持 ppo / sac)")
    env.close()

    ok = out["final_return"] > out["initial_return"] and out["final_return"] > baseline
    out.update({"algo": algo, "random_return": baseline, "ok": bool(ok)})
    return out
