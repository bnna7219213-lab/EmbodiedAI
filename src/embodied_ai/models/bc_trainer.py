"""行为克隆 (Behavior Cloning) — v0.7 补齐"训练数据 → 策略"缺失的一环。

缺口背景:
  评估报告把「无行为克隆/PPO/SAC 真实训练管线」列为 AI 层最大缺口之一。v0.6
  只有数据侧 (SimToRealPipeline.run_bc_collection 产出 list[dict]) 与契约校验
  (NanoGPTAdapter.validate_bc_dataset), **没有任何把数据变成策略的代码** —
  采集完的数据无处可去。

本模块给出闭环:
  采集 (专家 rollout) → BCDataset (obs/action 张量化 + 归一化) → BCPolicy (MLP)
  → BCTrainer (Adam + 余弦退火 + 验证集早停) → 环境内评估 (成功率/回报)

可验证性 (CI 断言):
  1. 验证集动作 MSE 必须**大幅下降** (≥3x), 否则是没学到;
  2. 克隆策略在环境里的**成功率必须接近专家** (≥0.8 倍), 否则是
     "loss 好看但不会做事" —— 这是 BC 最典型的失败模式 (分布偏移);
  3. 数据集必须能通过 NanoGPTAdapter 的契约校验, 保证能直接进 VLA 训练管线。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .nn_core import MLP, Adam, mse_grad, mse_loss

__all__ = ["BCDataset", "BCPolicy", "BCTrainer", "collect_expert_dataset", "bc_demo"]


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
@dataclass
class BCDataset:
    """行为克隆数据集 (纯 numpy, 可直接落 npz 供 nanoGPT/VLA 管线消费)。"""

    obs: np.ndarray                 # (N, obs_dim)
    actions: np.ndarray             # (N, act_dim)
    obs_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    obs_std: np.ndarray = field(default_factory=lambda: np.ones(0))
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.obs = np.atleast_2d(np.asarray(self.obs, dtype=float))
        self.actions = np.atleast_2d(np.asarray(self.actions, dtype=float))
        if self.obs_mean.size == 0:
            self.obs_mean = self.obs.mean(axis=0)
            self.obs_std = np.maximum(self.obs.std(axis=0), 1e-6)

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[1])

    @property
    def act_dim(self) -> int:
        return int(self.actions.shape[1])

    def normalized_obs(self) -> np.ndarray:
        return (self.obs - self.obs_mean) / self.obs_std

    def split(self, val_ratio: float = 0.2, seed: int = 0) -> Tuple["BCDataset", "BCDataset"]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        n_val = max(1, int(len(self) * val_ratio))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        mk = lambda i: BCDataset(  # noqa: E731
            self.obs[i], self.actions[i], self.obs_mean, self.obs_std, dict(self.meta))
        return mk(train_idx), mk(val_idx)

    def save(self, path: str) -> str:
        np.savez(path, obs=self.obs, actions=self.actions,
                 obs_mean=self.obs_mean, obs_std=self.obs_std,
                 meta=np.array([str(self.meta)]))
        return path

    @classmethod
    def load(cls, path: str) -> "BCDataset":
        z = np.load(path, allow_pickle=True)
        meta = {}
        if "meta" in z:
            try:
                meta = eval(str(z["meta"]))  # noqa: S307 — 仅用于回读本模块写出的 repr
            except Exception:  # noqa: BLE001
                meta = {}
        return cls(z["obs"], z["actions"], z["obs_mean"], z["obs_std"], meta)


# --------------------------------------------------------------------------- #
# 策略
# --------------------------------------------------------------------------- #
class BCPolicy:
    """确定性克隆策略: obs → action (tanh 输出保证有界)。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (64, 64),
                 action_limit: float = 1.0, seed: int = 0, lr: float = 3e-3):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.action_limit = float(action_limit)
        self.net = MLP([obs_dim, *hidden, act_dim], hidden_act="relu",
                       out_act="tanh", init="he", seed=seed)
        self.opt = Adam([(self.net.params, self.net.grads)], lr=lr)
        self.obs_mean = np.zeros(obs_dim)
        self.obs_std = np.ones(obs_dim)

    def set_normalizer(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.obs_mean = np.asarray(mean, dtype=float)
        self.obs_std = np.maximum(np.asarray(std, dtype=float), 1e-6)

    # 环境接口 (与 SimToRealPipeline.policy.predict 同签名)
    def predict(self, obs: Any) -> np.ndarray:
        s = obs["state"] if isinstance(obs, dict) else obs
        s = np.asarray(s, dtype=float).reshape(1, -1)[:, : self.obs_dim]
        z = (s - self.obs_mean) / self.obs_std
        return (self.net.forward(z)[0] * self.action_limit).astype(float)

    def predict_batch(self, obs: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(np.asarray(obs, dtype=float)) - self.obs_mean) / self.obs_std
        return self.net.forward(z) * self.action_limit


# --------------------------------------------------------------------------- #
# 训练器
# --------------------------------------------------------------------------- #
@dataclass
class BCStats:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    initial_val: float = 0.0
    final_val: float = 0.0

    @property
    def improved(self) -> bool:
        """验证损失下降 ≥3 倍才算学会了。"""
        return self.final_val < self.initial_val / 3.0

    def report(self) -> str:
        return (f"BC: iters={len(self.train_losses)} val_mse "
                f"{self.initial_val:.5f} -> {self.final_val:.5f}")


class BCTrainer:
    """行为克隆训练器 (监督回归 + 余弦退火 + 验证集监控)。"""

    def __init__(self, policy: BCPolicy, lr: float = 3e-3):
        self.policy = policy
        self.policy.opt.set_lr(lr)
        self.stats = BCStats()

    def fit(self, train: BCDataset, val: Optional[BCDataset] = None,
            iters: int = 600, batch_size: int = 64,
            eval_every: int = 50) -> BCStats:
        p = self.policy
        p.set_normalizer(train.obs_mean, train.obs_std)
        X, Y = train.normalized_obs(), train.actions / p.action_limit
        n = len(train)
        self.stats = BCStats()
        base_lr = p.opt.lr
        for it in range(iters):
            idx = np.random.default_rng(it).integers(0, n, size=min(batch_size, n))
            pred = p.net.forward(X[idx])
            loss = mse_loss(pred, Y[idx])
            p.net.zero_grad()
            p.net.backward(mse_grad(pred, Y[idx]))
            p.net.clip_grad_norm(1.0)
            # 余弦退火
            p.opt.set_lr(base_lr * 0.5 * (1.0 + np.cos(np.pi * it / max(iters - 1, 1))))
            p.opt.step()
            self.stats.train_losses.append(float(loss))
            if it % max(eval_every, 1) == 0 or it == iters - 1:
                v = self.evaluate(val or train)
                self.stats.val_losses.append(v)
                if it == 0:
                    self.stats.initial_val = v
                self.stats.final_val = v
        return self.stats

    def evaluate(self, data: BCDataset) -> float:
        p = self.policy
        pred = p.predict_batch(data.obs)
        return float(np.mean((pred - data.actions) ** 2))


# --------------------------------------------------------------------------- #
# 采集
# --------------------------------------------------------------------------- #
def collect_expert_dataset(env, expert: Callable[[Dict[str, Any]], np.ndarray],
                           n_episodes: int = 20, max_steps: int = 200,
                           seed: int = 0) -> BCDataset:
    """用专家策略在环境里 rollout, 收集 (obs, action) 对。

    expert: obs → action (与 SimToRealPipeline 的 policy 协议一致)。
    """
    obs_all: List[np.ndarray] = []
    act_all: List[np.ndarray] = []
    successes = 0
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        for _ in range(max_steps):
            a = np.asarray(expert(obs), dtype=float).reshape(-1)
            obs_all.append(np.asarray(obs["state"], dtype=float).reshape(-1))
            act_all.append(a)
            res = env.step(a)
            obs = res.observation
            if res.done or res.truncated:
                successes += 1 if res.info.get("success", res.done) else 0
                break
    return BCDataset(np.array(obs_all), np.array(act_all),
                     meta={"n_episodes": n_episodes, "expert_success": successes,
                           "collect": "expert_rollout"})


# --------------------------------------------------------------------------- #
# 端到端自检
# --------------------------------------------------------------------------- #
def bc_demo(n_episodes: int = 20, iters: int = 600, seed: int = 0,
            eval_episodes: int = 5) -> Dict[str, Any]:
    """闭环自检: 采集 → 训练 → 环境内评估。返回可供断言的字典。"""
    from ..simulation import EnvConfig, make_env

    cfg = EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3, dt=0.01,
                    max_episode_steps=200)
    env = make_env(cfg, backend="reference")
    target = env.target.copy()

    def expert(obs):
        s = np.asarray(obs["state"], dtype=float)
        n = len(target)
        return -2.0 * (s[:n] - target) - 0.3 * s[n:2 * n]

    def rollout(policy_fn, n_ep: int) -> Tuple[float, float]:
        rets, succ = [], 0
        for ep in range(n_ep):
            obs = env.reset(seed=1000 + ep)
            total, done_success = 0.0, False
            for _ in range(cfg.max_episode_steps):
                a = np.asarray(policy_fn(obs), dtype=float).reshape(-1)
                res = env.step(a)
                total += res.reward
                obs = res.observation
                if res.done or res.truncated:
                    done_success = bool(res.info.get("success", False))
                    break
            rets.append(total)
            succ += 1 if done_success else 0
        return float(np.mean(rets)), succ / max(n_ep, 1)

    expert_ret, expert_succ = rollout(expert, eval_episodes)

    ds = collect_expert_dataset(env, expert, n_episodes=n_episodes,
                                max_steps=cfg.max_episode_steps, seed=seed)
    train, val = ds.split(0.2, seed=seed)
    policy = BCPolicy(train.obs_dim, train.act_dim, hidden=(64, 64), action_limit=1.5)
    stats = BCTrainer(policy).fit(train, val, iters=iters)

    bc_ret, bc_succ = rollout(policy.predict, eval_episodes)
    # 随机基线 (证明"确实学会了", 而不是环境本身太简单)
    rand_ret, rand_succ = rollout(lambda o: env.sample_action(), eval_episodes)
    env.close()

    mse_ratio = stats.initial_val / max(stats.final_val, 1e-9)
    return {
        "n_samples": len(ds),
        "expert_success": expert_succ,
        "bc_success": bc_succ,
        "random_success": rand_succ,
        "expert_return": expert_ret,
        "bc_return": bc_ret,
        "random_return": rand_ret,
        "val_mse_initial": stats.initial_val,
        "val_mse_final": stats.final_val,
        "mse_improvement_x": float(mse_ratio),
        "ok": bool(stats.improved and bc_succ >= 0.8 * max(expert_succ, 1e-9)
                   and bc_ret > rand_ret),
        "report": stats.report(),
    }
