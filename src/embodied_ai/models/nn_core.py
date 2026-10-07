"""极简纯 numpy 神经网络内核 — 让训练算法在无 PyTorch 环境下**真的能训练**。

为什么需要它 (v0.7 缺口修复):
  评估报告指出「AI 神经网络模型完成度 ~30%: 无实际训练数据流经, DiffusionPolicy
  仍为接口占位, 无行为克隆/PPO/SAC 真实训练管线」。根因之一是所有训练类模块都
  hard-depend torch, 于是:
    - 装了 torch  → 只能在本机跑, CI (无 torch) 上全部降级为桩, 训练分支永不执行;
    - 没装 torch  → VLA/BC/PPO/SAC 全部抛 RuntimeError, 算法正确性无人验证。
  决策: **算法正确性用零依赖 numpy 内核验证, 规模扩展交给 torch/VLA 管线**。
  本模块即为该内核: 全连接 MLP + 反向传播 + Adam/SGD, 数值可断言 (梯度有限差分校验)。

设计原则:
  - 只做 MLP (扩散策略/PPO/SAC/BC 全都是 MLP 或 MLP+卷积, 本阶段只需 MLP);
  - 参数与梯度用**同构 list** 管理, 与 torch 的 parameters() 语义对齐, 便于将来 1:1 移植;
  - forward 自动缓存反向所需中间量, backward 累加 (不清零, 由调用方 zero_grad());
  - 全部随机数来自 np.random.default_rng(seed), 可复现。

梯度约定: 所有 backward 接收**上游梯度** dL/dy (shape 与 forward 输出一致),
返回 dL/dx; 参数梯度累加进 self.grads。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MLP", "Adam", "SGD", "mse_loss", "mse_grad", "gaussian_log_prob",
    "grad_check", "he_init", "xavier_init",
]


# --------------------------------------------------------------------------- #
# 初始化
# --------------------------------------------------------------------------- #
def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He 初始化 (ReLU 族)。"""
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / max(fan_in, 1))


def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Glorot 初始化 (Tanh/线性)。"""
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(1.0 / max(fan_in, 1))


_ACT_FWD: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "silu": lambda z: z / (1.0 + np.exp(-z)),
    "none": lambda z: z,
    "linear": lambda z: z,
}

_ACT_BWD: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    # (z_or_y, dout) -> dout_local
    "relu": lambda z, d: d * (z > 0.0),
    "tanh": lambda y, d: d * (1.0 - y ** 2),
    "silu": lambda z, d: d * (1.0 / (1.0 + np.exp(-z))) * (1.0 + z * (1.0 - 1.0 / (1.0 + np.exp(-z)))),
    "none": lambda z, d: d,
    "linear": lambda z, d: d,
}


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
class MLP:
    """多层感知机 (batch 前向 + 手动反向)。

    参数布局: params = [W1, b1, W2, b2, ...], W[i] shape (d_in, d_out)。
    """

    def __init__(self, dims: Sequence[int], hidden_act: str = "relu",
                 out_act: Optional[str] = None, init: str = "he", seed: int = 0,
                 bias_last: bool = True):
        if len(dims) < 2:
            raise ValueError("dims 至少含输入/输出两段")
        self.dims: List[int] = [int(d) for d in dims]
        self.hidden_act = hidden_act
        self.out_act = out_act or "none"
        self.rng = np.random.default_rng(seed)

        self.params: List[np.ndarray] = []
        self.grads: List[np.ndarray] = []
        init_fn = he_init if init == "he" else xavier_init
        for i in range(len(self.dims) - 1):
            fan_in, fan_out = self.dims[i], self.dims[i + 1]
            self.params.append(init_fn(fan_in, fan_out, self.rng))
            is_last = (i == len(self.dims) - 2)
            self.params.append(np.zeros(fan_out) if (bias_last or not is_last) else np.zeros(0))
            self.grads.append(np.zeros((fan_in, fan_out)))
            self.grads.append(np.zeros(self.params[-1].shape))
        self.zero_grad()
        self._cache: List[Tuple[np.ndarray, np.ndarray, str]] = []
        self._x0: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- #
    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def zero_grad(self) -> None:
        """原地清零。

        必须是**原地**的: 优化器持有 (params, grads) 列表对的引用, 一旦这里
        重新绑定 self.grads, 优化器看到的就永远是构造时的旧 list (梯度恒为 0),
        表现为"loss 一动不动"。这是手写训练框架最经典的坑。
        """
        for g in self.grads:
            g.fill(0.0)

    def parameters(self) -> List[np.ndarray]:
        return self.params

    # ---------------------------------------------------------------- #
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向: dout = dL/dy (B, d_out)。累加参数梯度, 返回 dL/dx。"""
        dout = np.atleast_2d(np.asarray(dout, dtype=float))
        if not self._cache:
            raise RuntimeError("必须先 forward 再 backward")
        g = dout
        for i in reversed(range(len(self._cache))):
            z, y, act = self._cache[i]
            W = self.params[2 * i]
            # 激活
            g = _ACT_BWD[act](y if act == "tanh" else z, g)
            # 线性层 (第 0 层的输入即网络输入 _x0)
            h_in = self._cache[i - 1][1] if i > 0 else self._x0
            self.grads[2 * i] += h_in.T @ g
            if self.params[2 * i + 1].size:
                self.grads[2 * i + 1] += g.sum(axis=0)
            g = g @ W.T
        return g

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, d_in) 或 (d_in,) → y: 同 batch 维。缓存反向所需中间量。"""
        single = (x.ndim == 1)
        h = np.atleast_2d(np.asarray(x, dtype=float))
        self._x0 = h
        self._cache = []
        n_layers = len(self.dims) - 1
        for i in range(n_layers):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            z = h @ W + (b if b.size else 0.0)
            act = self.out_act if i == n_layers - 1 else self.hidden_act
            y = _ACT_FWD[act](z)
            self._cache.append((z, y, act))
            h = y
        return h[0] if single else h

    # ---------------------------------------------------------------- #
    def clip_grad_norm(self, max_norm: float) -> float:
        """全局梯度裁剪 (对齐 torch.nn.utils.clip_grad_norm_)。返回裁剪前范数。"""
        total = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in self.grads)))
        if max_norm > 0 and total > max_norm:
            scale = max_norm / (total + 1e-12)
            for g in self.grads:
                g *= scale  # 原地: 保持 list 身份, 理由同 zero_grad
        return total

    def state_dict(self) -> Dict[str, List[np.ndarray]]:
        return {"params": [p.copy() for p in self.params], "dims": np.array(self.dims)}

    def load_state_dict(self, sd: Dict[str, List[np.ndarray]]) -> None:
        """原地载入 (保持 params 列表身份, 优化器引用继续有效)。"""
        for i, p in enumerate(sd["params"]):
            if self.params[i].shape != p.shape:
                raise ValueError(f"形状不匹配: {self.params[i].shape} vs {p.shape}")
            self.params[i][...] = p
        self.zero_grad()

    def copy_from(self, other: "MLP", rho: float = 1.0) -> None:
        """Polyak/硬拷贝: params := rho*other + (1-rho)*self。"""
        for i, p in enumerate(other.params):
            self.params[i] = rho * p + (1.0 - rho) * self.params[i]


# --------------------------------------------------------------------------- #
# 优化器
# --------------------------------------------------------------------------- #
class Adam:
    """Adam (bias-corrected), 直接消费 (params, grads) 列表对。"""

    def __init__(self, params_grads: List[Tuple[List[np.ndarray], List[np.ndarray]]],
                 lr: float = 3e-4, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.pairs = params_grads
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m: List[List[np.ndarray]] = [[np.zeros_like(p) for p in pg[0]] for pg in self.pairs]
        self.v: List[List[np.ndarray]] = [[np.zeros_like(p) for p in pg[0]] for pg in self.pairs]

    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, (params, grads) in enumerate(self.pairs):
            if len(self.m[k]) != len(params):
                self.m[k] = [np.zeros_like(p) for p in params]
                self.v[k] = [np.zeros_like(p) for p in params]
            for i, (p, g) in enumerate(zip(params, grads)):
                g = g + self.wd * p if self.wd else g
                self.m[k][i] = self.b1 * self.m[k][i] + (1 - self.b1) * g
                self.v[k][i] = self.b2 * self.v[k][i] + (1 - self.b2) * (g * g)
                p -= self.lr * (self.m[k][i] / bc1) / (np.sqrt(self.v[k][i] / bc2) + self.eps)

    def zero_grad(self) -> None:
        for _, grads in self.pairs:
            for g in grads:
                g.fill(0.0)  # 原地: 保持与 MLP.grads 同一对象

    @property
    def lr_now(self) -> float:
        return self.lr

    def set_lr(self, lr: float) -> None:
        self.lr = lr


class SGD:
    """带动量的 SGD (轻量场景 / 对照实验)。"""

    def __init__(self, params_grads, lr: float = 1e-2, momentum: float = 0.9,
                 weight_decay: float = 0.0):
        self.pairs = params_grads
        self.lr = lr
        self.mu = momentum
        self.wd = weight_decay
        self.v: List[List[np.ndarray]] = [[np.zeros_like(p) for p in pg[0]] for pg in self.pairs]

    def step(self) -> None:
        for k, (params, grads) in enumerate(self.pairs):
            for i, (p, g) in enumerate(zip(params, grads)):
                g = g + self.wd * p if self.wd else g
                self.v[k][i] = self.mu * self.v[k][i] + g
                p -= self.lr * self.v[k][i]

    def zero_grad(self) -> None:
        for _, grads in self.pairs:
            for g in grads:
                g.fill(0.0)

    def set_lr(self, lr: float) -> None:
        self.lr = lr


# --------------------------------------------------------------------------- #
# 损失 / 概率工具
# --------------------------------------------------------------------------- #
def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    d = np.asarray(pred) - np.asarray(target)
    return float(np.mean(d * d))


def mse_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """dL/dpred, 与 mse_loss 的 mean 口径一致。"""
    d = np.asarray(pred) - np.asarray(target)
    return 2.0 * d / max(d.size, 1)


def gaussian_log_prob(x: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> np.ndarray:
    """各维独立高斯的 log p (sum over action dim)。x/mean: (B,A)。"""
    var = np.exp(2.0 * np.clip(log_std, -20.0, 2.0))
    return np.sum(-0.5 * ((x - mean) ** 2 / var + 2.0 * log_std + np.log(2.0 * np.pi)), axis=-1)


def gaussian_sample(mean: np.ndarray, log_std: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    return mean + np.exp(log_std) * rng.standard_normal(mean.shape)


# --------------------------------------------------------------------------- #
# 梯度自检 (数值验证: 本项目所有 numpy 模型的验收手段)
# --------------------------------------------------------------------------- #
def grad_check(net: MLP, x: np.ndarray, eps: float = 1e-5, tol: float = 1e-4) -> Dict[str, float]:
    """有限差分 vs 反向传播, 返回最大相对误差与是否通过。

    用于 CI: 反向传播一旦写错, 所有下游训练断言都会"看起来在学"实则学歪。
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    rng = np.random.default_rng(0)
    dout = rng.standard_normal((x.shape[0], net.dims[-1]))

    y = net.forward(x)
    net.zero_grad()
    net.backward(dout)
    ana = [g.copy() for g in net.grads]

    worst = 0.0
    for i, p in enumerate(net.params):
        if p.size == 0:
            continue
        num = np.zeros_like(p)
        flat_p, flat_num = p.ravel(), num.ravel()
        idx = np.argsort(np.abs(ana[i].ravel()))[::-1][: min(8, p.size)]
        flat_a = ana[i].ravel()
        for j in idx:
            old = flat_p[j]
            flat_p[j] = old + eps
            yp = net.forward(x)
            flat_p[j] = old - eps
            ym = net.forward(x)
            flat_p[j] = old
            num_j = float(np.sum((yp - ym) * dout) / (2 * eps))
            denom = max(abs(num_j), abs(flat_a[j]), 1e-6)
            worst = max(worst, abs(num_j - flat_a[j]) / denom)
    return {"max_rel_err": float(worst), "ok": bool(worst < tol)}
