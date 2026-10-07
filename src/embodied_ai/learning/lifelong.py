"""持续学习: EWC + 经验回放 + 遗忘度量 (G12)。

设计取舍:
  - 用 models/nn_core 的 MLP (零依赖), 因此遗忘实验在任何环境可复现;
  - Fisher 用**对角**近似 + 经验平均, 与原始 EWC 一致 (在线版会用 running
    average, 这里为了可断言用离线重算);
  - 惩罚项直接加进 grads 而不是加进 loss: 因为 nn_core 是手写反向, 把二次惩罚
    当成 loss 项再反向容易重复计时; 直接加梯度更清晰且数值一致。

关键坑 (v0.7 实测):
  - λ 不是越大越好: λ 过大时新任务根本学不动 (plasticity 塌了), 表现为"B 的
    误差和 A 一样烂"。验收必须**同时**看 forgetting 和 plasticity, 只看
    forgetting 会得出"λ 越大越好"的错误结论。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..models.nn_core import MLP, Adam, mse_grad, mse_loss

__all__ = ["LifelongConfig", "ReplayMemory", "EWCRegularizer", "LifelongLearner",
           "compute_fisher", "make_synthetic_tasks", "lifelong_demo"]


# --------------------------------------------------------------------------- #
@dataclass
class LifelongConfig:
    ewc_lambda: float = 50.0         # EWC 惩罚强度
    replay_ratio: float = 0.3        # 每批中来自旧任务的比例
    memory_per_task: int = 64        # 每个旧任务保留的样本数
    lr: float = 3e-3
    iters_per_task: int = 400
    batch_size: int = 32
    hidden: Sequence[int] = (64, 64)


class ReplayMemory:
    """每个任务保留一小撮样本 (核心集式回放, 不训练生成器)。"""

    def __init__(self, memory_per_task: int = 64, seed: int = 0):
        self.cap = int(memory_per_task)
        self.bank: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.rng = np.random.default_rng(seed)

    def add(self, task_id: int, X: np.ndarray, y: np.ndarray) -> None:
        idx = self.rng.permutation(len(X))[: self.cap]
        self.bank[int(task_id)] = (np.asarray(X)[idx], np.asarray(y)[idx])

    def sample(self, exclude: Optional[int] = None, n: int = 16,
               rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = rng or self.rng
        keys = [k for k in self.bank if k != exclude]
        if not keys:
            return np.zeros((0, 1)), np.zeros((0, 1))
        out_x, out_y = [], []
        per = max(1, n // len(keys))
        for k in keys:
            X, y = self.bank[k]
            idx = rng.integers(0, len(X), size=min(per, len(X)))
            out_x.append(X[idx])
            out_y.append(y[idx])
        return np.vstack(out_x), np.vstack(out_y)

    def __len__(self) -> int:
        return sum(len(v[0]) for v in self.bank.values())


def compute_fisher(net: MLP, X: np.ndarray, y: np.ndarray,
                   max_samples: int = 256) -> List[np.ndarray]:
    """对角 Fisher: F_i = E[(∂L/∂θ_i)²] (回归用 MSE 的梯度平方)。"""
    F = [np.zeros_like(p) for p in net.params]
    idx = np.random.default_rng(0).permutation(len(X))[: min(max_samples, len(X))]
    for i in idx:
        net.zero_grad()
        pred = net.forward(X[i:i + 1])
        net.backward(mse_grad(pred, y[i:i + 1]))
        for j, g in enumerate(net.grads):
            F[j] += g ** 2
    n = max(len(idx), 1)
    return [f / n for f in F]


@dataclass
class EWCRegularizer:
    """保存 θ* 与 Fisher, 施加 λ·F·(θ − θ*) 的梯度惩罚。"""
    fisher: List[np.ndarray] = field(default_factory=list)
    theta_star: List[np.ndarray] = field(default_factory=list)
    lambda_: float = 1.0

    def register(self, net: MLP, X: np.ndarray, y: np.ndarray) -> None:
        self.fisher = compute_fisher(net, X, y)
        self.theta_star = [p.copy() for p in net.params]

    def apply(self, net: MLP) -> float:
        """把惩罚**加进** grads (调用前必须已做完本任务的反向)。返回惩罚量级。"""
        if not self.fisher:
            return 0.0
        mag = 0.0
        for j, (Fj, th) in enumerate(zip(self.fisher, self.theta_star)):
            pen = self.lambda_ * Fj * (net.params[j] - th)
            net.grads[j] = net.grads[j] + pen
            mag += float(np.sum(pen ** 2))
        return float(np.sqrt(mag))


# --------------------------------------------------------------------------- #
class LifelongLearner:
    """顺序任务学习器: EWC + 回放, 带遗忘度量。"""

    def __init__(self, obs_dim: int, out_dim: int, cfg: LifelongConfig = None,
                 seed: int = 0):
        self.cfg = cfg or LifelongConfig()
        self.net = MLP([obs_dim, *self.cfg.hidden, out_dim],
                       hidden_act="relu", out_act="none", seed=seed)
        self.opt = Adam([(self.net.params, self.net.grads)], lr=self.cfg.lr)
        self.ewc = EWCRegularizer(lambda_=self.cfg.ewc_lambda)
        self.memory = ReplayMemory(self.cfg.memory_per_task, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.history: List[Dict] = []

    # ---------------------------------------------------------------- #
    def train_task(self, task_id: int, X: np.ndarray, y: np.ndarray,
                   use_ewc: bool = True, use_replay: bool = True) -> Dict:
        c = self.cfg
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_2d(np.asarray(y, dtype=float))
        n_replay = int(c.batch_size * c.replay_ratio)
        for it in range(c.iters_per_task):
            idx = self.rng.integers(0, len(X), size=min(c.batch_size, len(X)))
            Xb, yb = X[idx], y[idx]
            if use_replay and len(self.memory) > 0 and n_replay > 0:
                Xr, yr = self.memory.sample(exclude=task_id, n=n_replay, rng=self.rng)
                if len(Xr):
                    Xb = np.vstack([Xb, Xr])
                    yb = np.vstack([yb, yr])
            self.net.zero_grad()
            pred = self.net.forward(Xb)
            self.net.backward(mse_grad(pred, yb))
            if use_ewc:
                self.ewc.apply(self.net)
            self.net.clip_grad_norm(1.0)
            self.opt.step()
        # 学完登记 Fisher + 入回放库
        if use_ewc:
            self.ewc.register(self.net, X, y)
        self.memory.add(task_id, X, y)
        return {"task": task_id, "train_mse": float(mse_loss(self.net.forward(X), y))}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(mse_loss(self.net.forward(np.atleast_2d(X)),
                              np.atleast_2d(y)))

    # ---------------------------------------------------------------- #
    def run_sequence(self, tasks: Sequence[Tuple[np.ndarray, np.ndarray]],
                     use_ewc: bool = True, use_replay: bool = True
                     ) -> Dict[str, List[float]]:
        """顺序训练并在**每次学完后**重测所有任务 → 遗忘曲线。"""
        n = len(tasks)
        curve: Dict[str, List[float]] = {f"task{i}": [] for i in range(n)}
        for i, (X, y) in enumerate(tasks):
            self.train_task(i, X, y, use_ewc=use_ewc, use_replay=use_replay)
            for j in range(n):
                curve[f"task{j}"].append(self.evaluate(*tasks[j]))
        return curve


def forgetting_metrics(curve: Dict[str, List[float]]) -> Dict[str, float]:
    """从遗忘曲线算 forgetting / BWT /  plasticity。"""
    n = len(curve)
    forget = []
    for j in range(n):
        traj = curve[f"task{j}"]
        best = min(traj[j:n])                  # 学过之后的最好成绩
        final = traj[-1]
        forget.append(final - best)
    bwt = float(np.mean([curve[f"task{j}"][-1] - curve[f"task{j}"][j]
                         for j in range(n - 1)])) if n > 1 else 0.0
    return {"forgetting": [float(f) for f in forget],
            "mean_forgetting": float(np.mean(forget)),
            "backward_transfer": float(-bwt),
            "final_errors": [float(curve[f"task{j}"][-1]) for j in range(n)]}


# --------------------------------------------------------------------------- #
def make_synthetic_tasks(n: int = 512, dim: int = 4, seed: int = 0,
                         conflict: bool = True, shared_domain: bool = False):
    """两个顺序任务: 输入空间**分离**, 但映射符号**相反**。

    shared_domain=True 时两任务共用同一输入域 (y₂ = −y₁), 用于测**可塑性**:
    EWC 的 λ 过大时新任务根本学不动。分离域版本则用于测**遗忘**。

    为什么不让它们完全共用输入: 若 y₂ = −y₁ 且输入域相同, 那回放旧样本就会和
    新任务目标直接对撞, 混合训练学不出任何东西 — 这测的是"任务不可同时学",
    不是"遗忘"。输入分离后, 共享参数仍会被后一个任务改写 (遗忘真实存在),
    而回放不再自相矛盾, EWC/回放的收益才测得出来。
    """
    rng = np.random.default_rng(seed)
    if shared_domain:
        X = rng.uniform(-1.0, 1.0, size=(n, dim))
        y1 = np.sin(2 * np.pi * X[:, :1]) * 0.8 + 0.2 * X[:, 1:2]
        y2 = -y1 if conflict else np.cos(2 * np.pi * X[:, :1]) * 0.8
        return [(X, y1), (X, y2)]
    X1 = rng.uniform(-1.0, 0.0, size=(n, dim))
    X2 = rng.uniform(0.0, 1.0, size=(n, dim))
    y1 = np.sin(2 * np.pi * X1[:, :1]) * 0.8 + 0.2 * X1[:, 1:2]
    if conflict:
        y2 = -np.sin(2 * np.pi * X2[:, :1]) * 0.8 + 0.2 * X2[:, 1:2]
    else:
        y2 = np.cos(2 * np.pi * X2[:, :1]) * 0.8 + 0.2 * X2[:, 1:2]
    return [(X1, y1), (X2, y2)]


def lifelong_demo(seed: int = 0) -> Dict:
    """持续学习自检: 遗忘可观测 / EWC 有效 / λ 过大会压死可塑性。"""
    tasks = make_synthetic_tasks(seed=seed)
    c = LifelongConfig(iters_per_task=400)

    # 1) 基线: 顺序训练, 无保护
    base = LifelongLearner(4, 1, LifelongConfig(
        ewc_lambda=0.0, iters_per_task=c.iters_per_task), seed=seed)
    cv_base = base.run_sequence(tasks, use_ewc=False, use_replay=False)
    m_base = forgetting_metrics(cv_base)

    # 2) EWC + 回放
    ewc = LifelongLearner(4, 1, LifelongConfig(
        ewc_lambda=50.0, iters_per_task=c.iters_per_task), seed=seed)
    cv_ewc = ewc.run_sequence(tasks, use_ewc=True, use_replay=True)
    m_ewc = forgetting_metrics(cv_ewc)

    # 3) λ 过大: 旧任务守住了, 但新任务学不动 (判据非恒真)
    #    必须在**同域冲突**任务上测: 输入分离时惩罚只作用于旧域的参数,
    #    新域照学不误, 永远看不出可塑性损失。
    same = make_synthetic_tasks(seed=seed, shared_domain=True)
    loose = LifelongLearner(4, 1, LifelongConfig(
        ewc_lambda=0.0, iters_per_task=c.iters_per_task), seed=seed)
    m_loose = forgetting_metrics(loose.run_sequence(same, use_ewc=False,
                                                    use_replay=False))
    stiff = LifelongLearner(4, 1, LifelongConfig(
        ewc_lambda=5e4, iters_per_task=c.iters_per_task), seed=seed)
    m_stiff = forgetting_metrics(stiff.run_sequence(same, use_ewc=True,
                                                    use_replay=False))

    # 单任务上界 (学完 A 立即测 A)
    solo = LifelongLearner(4, 1, LifelongConfig(
        ewc_lambda=0.0, iters_per_task=c.iters_per_task), seed=seed)
    solo.train_task(0, *tasks[0], use_ewc=False, use_replay=False)
    err_a_solo = solo.evaluate(*tasks[0])

    forgetting_reduced = bool(m_ewc["mean_forgetting"]
                              < 0.5 * max(m_base["mean_forgetting"], 1e-9))
    stiff_loses_plasticity = bool(m_stiff["final_errors"][1]
                                  > 2.0 * max(m_loose["final_errors"][1], 1e-9))

    return {
        "baseline_forgetting": float(m_base["mean_forgetting"]),
        "ewc_forgetting": float(m_ewc["mean_forgetting"]),
        "baseline_task0_final": float(m_base["final_errors"][0]),
        "ewc_task0_final": float(m_ewc["final_errors"][0]),
        "task0_solo_error": float(err_a_solo),
        "stiff_task1_error": float(m_stiff["final_errors"][1]),
        "ewc_task1_error": float(m_ewc["final_errors"][1]),
        "forgetting_reduced": forgetting_reduced,
        "stiff_loses_plasticity": stiff_loses_plasticity,
        "memory_size": len(ewc.memory),
        "ok": bool(m_base["mean_forgetting"] > 1e-3          # 遗忘确实发生
                   and forgetting_reduced                    # EWC 有效
                   and stiff_loses_plasticity                # λ 过大有害
                   and m_ewc["final_errors"][1] < 0.2),      # 新任务仍学会
    }
