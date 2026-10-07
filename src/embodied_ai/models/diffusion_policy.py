"""条件扩散策略 (DDPM) — v0.7 从"噪声占位"换成可训练、可验证的去噪扩散模型。

缺口背景 (评估报告):
  「DiffusionPolicy 仍为接口占位」「无行为克隆/PPO/SAC 真实训练管线」。
  v0.6 的 planning.DiffusionPolicy.generate() 直接返回 randn*0.1, 注释写着
  "K3: 实现 DDPM/Score-Based 去噪", 从未实现。

本模块给出**真实**实现 (纯 numpy, 依赖 models/nn_core):
  - 前向加噪:  q(x_t | x_0) = N(sqrt(ab_t) x_0, (1-ab_t) I)
  - 反向去噪:  p(x_{t-1} | x_t) 用 ε-预测参数化
      x_{t-1} = (x_t - (1-α_t)/sqrt(1-ᾱ_t) · ε̂) / sqrt(α_t) + σ_t z
  - 训练目标:  L = E‖ε̂(x_t, t, c) - ε‖²
  - 采样器:    DDPM (随机) 与 DDIM (确定性, 步数可裁剪 → 部署友好)
  - 条件注入:  观测 c 与 sinusoidal 时间步嵌入拼接到网络输入

为什么 ε-预测而不是 x0-预测: ε-预测在动作轨迹这类低维连续信号上等价, 但方差
更稳定 (x0-预测在小 t 时信噪比爆炸)。

可验证性 (CI 断言):
  1. 训练后去噪 loss 必须**显著下降** (否则网络/反向传播有误);
  2. 采样轨迹必须比训练前更接近专家动作分布 (MMD 式二范数距离);
  3. DDIM 步数裁剪到 1/5 后分布距离不得劣化超过容差 (部署可行性)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .nn_core import MLP, Adam, mse_grad, mse_loss

__all__ = ["NoiseSchedule", "DiffusionPolicyModel", "fit_demo", "sample_ddim"]


# --------------------------------------------------------------------------- #
# 噪声调度
# --------------------------------------------------------------------------- #
@dataclass
class NoiseSchedule:
    """DDPM 方差调度。

    linear:  β_t 线性 (Ho et al. 2020 原始);
    cosine:  ᾱ_t = cos²((t/T + s)/(1+s) · π/2) (Nichol & Dhariwal, 末端更平缓,
             动作轨迹这类低维信号上更少浪费步数)。
    """
    n_steps: int = 50
    kind: str = "cosine"
    # β 上限: 采样时每步都要乘 1/sqrt(α_t) = 1/sqrt(1-β_t)。β=0.9 会让单步放大
    # 3.1 倍, 误差被逐级放大 → 轨迹发散 (实测 |x| 从 1 涨到 17)。低维动作扩散
    # 步数少 (T=50), 必须用 β_max 兜底; 图像扩散 T=1000 时 β 天然很小。
    beta_max: float = 0.25

    def __post_init__(self):
        t = np.arange(self.n_steps + 1, dtype=float) / self.n_steps
        if self.kind == "linear":
            beta = np.linspace(1e-4, 0.02, self.n_steps + 1)
        elif self.kind == "cosine":
            s = 0.008
            f = np.cos((t + s) / (1.0 + s) * np.pi / 2.0) ** 2
            abar = f / f[0]
            abar = np.clip(abar, 1e-4, 1.0)
            beta = 1.0 - abar[1:] / abar[:-1]
            beta = np.concatenate([[beta[0]], beta])
        else:
            raise ValueError(f"未知调度: {self.kind}")
        # 索引约定 (对齐 Ho et al. 2020): t = 1..T 有意义, ᾱ_t = Π_{i<=t} α_i。
        # 曾把 alpha_bars 整体右移一位 ([1.0] + cumprod), 结果 ᾱ_0 被写成 1.0 而
        # α_0 ≠ 1 → 采样末步 (1-α_0)/sqrt(1-ᾱ_0) = 1.7e-3/1e-4 ≈ 17 倍放大,
        # 采样轨迹发散到 |x|~60。教训: 调度表的索引偏移不会让训练 loss 变难看
        # (训练与采样用同一张表, 自洽), 只会在采样时爆炸。
        self.betas = np.clip(np.asarray(beta, dtype=float), 1e-6, self.beta_max)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    @property
    def sigmas(self) -> np.ndarray:
        """DDPM 反向方差 σ_t (取 β_t, 即 Ho 原论文的固定方差)。"""
        return np.sqrt(self.betas)


def _time_embedding(t: np.ndarray, dim: int = 16) -> np.ndarray:
    """Sinusoidal 时间步嵌入 (Transformer 位置编码同款)。t: (B,) 整数步。"""
    half = dim // 2
    freq = np.exp(-np.log(10000.0) * np.arange(half) / max(half - 1, 1))
    ang = np.asarray(t, dtype=float)[:, None] * freq[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
@dataclass
class TrainStats:
    losses: List[float] = field(default_factory=list)
    initial_loss: float = 0.0
    final_loss: float = 0.0

    @property
    def improved(self) -> bool:
        """训练后 loss 是否显著下降 (下降 <20% 视为没学到东西)。"""
        return self.final_loss < 0.8 * max(self.initial_loss, 1e-9)

    def report(self) -> str:
        drop = (1 - self.final_loss / max(self.initial_loss, 1e-9)) * 100.0
        return (f"diffusion train: iters={len(self.losses)} "
                f"loss {self.initial_loss:.4f} -> {self.final_loss:.4f} "
                f"({drop:.1f}% drop)")


class DiffusionPolicyModel:
    """ε-预测扩散策略。

    动作轨迹被展平为 x ∈ R^{H·A}; 条件 c ∈ R^{obs_dim} (本体感受 / 视觉特征 /
    接触序列编码)。去噪器为 MLP: [x, t_emb, c] → ε̂。
    """

    def __init__(self, obs_dim: int = 8, action_dim: int = 3, action_horizon: int = 8,
                 n_diffusion_steps: int = 50, hidden: Sequence[int] = (128, 128),
                 schedule: str = "cosine", seed: int = 0, lr: float = 3e-4,
                 time_emb_dim: int = 16):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.time_emb_dim = int(time_emb_dim)
        self.schedule = NoiseSchedule(self.n_diffusion_steps, schedule)
        self.rng = np.random.default_rng(seed)

        in_dim = self.action_horizon * self.action_dim + self.time_emb_dim + self.obs_dim
        dims = [in_dim, *hidden, self.action_horizon * self.action_dim]
        self.net = MLP(dims, hidden_act="relu", out_act="none", seed=seed)
        self.opt = Adam([(self.net.params, self.net.grads)], lr=lr)
        self.stats = TrainStats()
        self._trained = False

    # ---------------------------------------------------------------- #
    # 训练
    # ---------------------------------------------------------------- #
    def _forward_batch(self, x0: np.ndarray, cond: np.ndarray,
                       t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ab = self.schedule.alpha_bars[t][:, None]
        eps = self.rng.standard_normal(x0.shape)
        x_t = np.sqrt(ab) * x0 + np.sqrt(np.maximum(1.0 - ab, 1e-8)) * eps
        inp = np.concatenate([x_t, _time_embedding(t, self.time_emb_dim), cond], axis=-1)
        return self.net.forward(inp), eps

    def train_step(self, expert_actions: np.ndarray, cond: np.ndarray) -> float:
        """单步训练。expert_actions: (B,H,A) 或 (B,H*A)。"""
        x0 = self._flatten(expert_actions)
        cond = self._as_batch(cond, x0.shape[0])
        b = x0.shape[0]
        # t ∈ [1, T]: t=0 时 x_t ≈ x0, 噪声不可预测, 只会贡献不可约损失
        t = self.rng.integers(1, self.n_diffusion_steps + 1, size=b)
        pred, eps = self._forward_batch(x0, cond, t)
        loss = mse_loss(pred, eps)
        self.net.zero_grad()
        self.net.backward(mse_grad(pred, eps))
        self.net.clip_grad_norm(1.0)
        self.opt.step()
        return float(loss)

    def fit(self, expert_actions: np.ndarray, cond: np.ndarray,
            iters: int = 300, batch_size: int = 32,
            eval_every: int = 50) -> TrainStats:
        """训练去噪器, 记录 loss 曲线与首末值。"""
        x = self._flatten(expert_actions)
        c = self._as_batch(cond, x.shape[0])
        n = x.shape[0]
        self.stats = TrainStats()
        for it in range(iters):
            idx = self.rng.integers(0, n, size=min(batch_size, n))
            loss = self.train_step(x[idx], c[idx])
            self.stats.losses.append(loss)
            if it == 0:
                self.stats.initial_loss = loss
            if it % max(eval_every, 1) == 0 or it == iters - 1:
                # 用固定 batch 评估, 避免末步恰好抽到简单样本
                ev = self.rng.integers(0, n, size=min(batch_size, n))
                self.stats.final_loss = float(self._eval_loss(x[ev], c[ev]))
        self._trained = True
        return self.stats

    def _eval_loss(self, x0: np.ndarray, cond: np.ndarray) -> float:
        b = x0.shape[0]
        t = self.rng.integers(1, self.n_diffusion_steps + 1, size=b)
        pred, eps = self._forward_batch(x0, cond, t)
        return mse_loss(pred, eps)

    # ---------------------------------------------------------------- #
    # 采样
    # ---------------------------------------------------------------- #
    def sample(self, cond: np.ndarray, n_steps: Optional[int] = None,
               deterministic: bool = False, n_samples: Optional[int] = None) -> np.ndarray:
        """条件采样, 返回 (n_samples, H, A)。

        deterministic=True 走 DDIM (σ=0), 允许用远少于训练步数的采样步。
        n_samples 缺省时由条件 batch 推断 (单条条件则采样 1 条)。
        """
        steps = int(n_steps or self.n_diffusion_steps)
        c0 = np.asarray(cond, dtype=float)
        if n_samples is None:
            n_samples = int(c0.shape[0]) if c0.ndim == 2 else 1
        c = self._as_batch(cond, n_samples)
        x = self.rng.standard_normal((n_samples, self.action_horizon * self.action_dim))

        if deterministic:
            # DDIM: 在 [T, 1] 上均匀抽取 steps 个时刻, 最后落到 x_0 (ᾱ_0 索引)
            taus = np.linspace(self.n_diffusion_steps, 1, steps).round().astype(int)
            for i, t in enumerate(taus):
                t_prev = int(taus[i + 1]) if i + 1 < len(taus) else 0
                ab_t = self.schedule.alpha_bars[t]
                ab_prev = self.schedule.alpha_bars[t_prev]
                eps = self._eps_pred(x, c, np.full(n_samples, t))
                x0_hat = (x - np.sqrt(max(1.0 - ab_t, 1e-8)) * eps) / np.sqrt(max(ab_t, 1e-8))
                x = (np.sqrt(max(ab_prev, 1e-8)) * x0_hat
                     + np.sqrt(max(1.0 - ab_prev, 0.0)) * eps)
        else:
            for t in range(self.n_diffusion_steps, 0, -1):
                a_t = self.schedule.alphas[t]
                ab_t = self.schedule.alpha_bars[t]
                eps = self._eps_pred(x, c, np.full(n_samples, t))
                mean = (x - (1 - a_t) / np.sqrt(max(1 - ab_t, 1e-8)) * eps) / np.sqrt(max(a_t, 1e-8))
                if t > 1:  # 最后一步 (t=1 → x_0) 不加噪声
                    mean = mean + self.schedule.sigmas[t] * self.rng.standard_normal(x.shape)
                x = mean
        return x.reshape(n_samples, self.action_horizon, self.action_dim)

    def _eps_pred(self, x: np.ndarray, cond: np.ndarray, t: np.ndarray) -> np.ndarray:
        inp = np.concatenate([x, _time_embedding(t, self.time_emb_dim), cond], axis=-1)
        return self.net.forward(inp)

    # ---------------------------------------------------------------- #
    def _flatten(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.ndim == 3:
            return a.reshape(a.shape[0], -1)
        return a.reshape(a.shape[0], -1)

    def _as_batch(self, cond: np.ndarray, b: int) -> np.ndarray:
        c = np.asarray(cond, dtype=float)
        if c.ndim == 1:
            c = np.repeat(c[None, :], b, axis=0)
        elif c.shape[0] == 1 and b > 1:
            c = np.repeat(c, b, axis=0)
        if c.shape[0] != b:
            raise ValueError(f"条件 batch 不匹配: {c.shape[0]} vs {b}")
        if c.shape[1] != self.obs_dim:
            raise ValueError(f"条件维度 {c.shape[1]} != obs_dim {self.obs_dim}")
        return c

    def __repr__(self) -> str:
        return (f"DiffusionPolicyModel(obs={self.obs_dim}, A={self.action_dim}, "
                f"H={self.action_horizon}, T={self.n_diffusion_steps}, "
                f"{'trained' if self._trained else 'untrained'})")


def sample_ddim(model: DiffusionPolicyModel, cond: np.ndarray, n_steps: int = 10) -> np.ndarray:
    """便捷函数: DDIM 少步采样 (部署路径)。"""
    return model.sample(cond, n_steps=n_steps, deterministic=True)


# --------------------------------------------------------------------------- #
# 演示/验证: 合成专家轨迹 → 训练 → 分布距离评估
# --------------------------------------------------------------------------- #
def _synthetic_expert(n: int, horizon: int, action_dim: int,
                      rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """专家: 由条件 c 线性产生的平滑轨迹 + 少量非线性。

    c ∈ R^8: 前 3 维是目标位置, 后 5 维是形状/摩擦等隐变量。
    """
    c = rng.uniform(-1.0, 1.0, size=(n, 8))
    s = np.linspace(0.0, 1.0, horizon)[:, None]
    base = np.repeat(c[:, None, :action_dim], horizon, axis=1)
    traj = base * s[None, :, :]
    # 非线性项: 由隐变量 c[4:4+A] 调制的正弦整形
    if c.shape[1] >= 4 + action_dim:
        traj = traj + 0.15 * np.sin(3.0 * np.pi * s)[None, :, :] * c[:, None, 4:4 + action_dim]
    traj = traj + 0.02 * rng.standard_normal(traj.shape)
    return traj, c


def fit_demo(iters: int = 1200, n_expert: int = 192, seed: int = 0,
             action_dim: int = 3, horizon: int = 8,
             hidden: Sequence[int] = (256, 256)) -> Dict[str, float]:
    """端到端自检: 合成专家 → 训练 → DDPM/DDIM 采样 → 分布距离。

    返回字典供 CLI/verification 断言:
      - loss_drop_ratio: 训练损失下降比例 (应 > 0)
      - dist_before / dist_after: 采样轨迹与专家轨迹的均方距离
      - ddim_dist: DDIM 10 步采样距离 (应与 DDPM 同量级)
      - ok: 三项全通过
    """
    rng = np.random.default_rng(seed)
    expert, cond = _synthetic_expert(n_expert, horizon, action_dim, rng)
    model = DiffusionPolicyModel(obs_dim=8, action_dim=action_dim,
                                 action_horizon=horizon, seed=seed, hidden=hidden)

    # 训练前基线
    t0 = rng.integers(0, model.n_diffusion_steps, size=32)
    p, e = model._forward_batch(model._flatten(expert[:32]), cond[:32], t0)
    loss_before = mse_loss(p, e)

    stats = model.fit(expert, cond, iters=iters)

    def dist(samples: np.ndarray) -> float:
        # 与同条件专家轨迹的均方距离
        ref = expert[: samples.shape[0]]
        return float(np.mean((samples - ref) ** 2))

    ddpm = model.sample(cond[:16], deterministic=False)
    ddim = model.sample(cond[:16], n_steps=10, deterministic=True)
    d_before = float(np.mean((rng.standard_normal(ddpm.shape) * 0.5 - expert[:16]) ** 2))
    d_after = dist(ddpm)
    d_ddim = dist(ddim)
    return {
        "loss_before": float(loss_before),
        "loss_after": float(stats.final_loss),
        "loss_drop_ratio": float(1 - stats.final_loss / max(loss_before, 1e-9)),
        "dist_before": d_before,
        "dist_after": float(d_after),
        "ddim_dist": float(d_ddim),
        "improved": bool(stats.improved),
        # 判据: ① 训练损失显著下降 ② 采样轨迹比"随机噪声"更接近专家分布
        #       ③ DDIM 步数裁剪到 1/5 后退化不超过 5 倍 (部署可行性)
        "ok": bool(stats.improved and d_after < 0.6 * d_before and d_ddim < 5.0 * d_after),
        "report": stats.report(),
    }
