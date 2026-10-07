"""Signal Temporal Logic (STL) 规范 — 运行时监控 + 离线验证。

创新 #4 核心: 将安全需求用 STL 形式化描述, 生成运行时监控器。
STL 公式翻译:
  "始终[0,T] 力不超过阈值"        → G[0,T](force < F_max)
  "停止命令后10ms内到达零力矩"    → G(stop_cmd → F[0,10ms] torque==0)
  "接近人类时速度 < 0.25m/s"      → G(human_dist < 0.5m → velocity < 0.25)

运行时监控: 对于信号 f(t), 在任意时间 t 计算公式的满足度 (robustness):
  ρ(G[a,b] φ, t) = min_{t'∈[t+a,t+b]} ρ(φ, t')
  ρ(F[a,b] φ, t) = max_{t'∈[t+a,t+b]} ρ(φ, t')
  ρ(φ → ψ, t)    = max(-ρ(φ,t), ρ(ψ,t))
  ρ(X[a] φ, t)   = ρ(φ, t+a)

ρ > 0 → 公式在时间 t 成立; ρ ≤ 0 → 违反。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---- STL 公式类型 ----

@dataclass
class STLAtom:
    """原子命题: f(x) < 0 形式, 其中 f 是从状态空间到 R 的函数。"""
    name: str
    func: Callable[[np.ndarray], float]  # 公式 f(x), 当 f(x) < 0 时原子成立

    def robustness(self, x: np.ndarray) -> float:
        return -self.func(x)  # f(x) < 0 → ρ = -f(x); ρ > 0 成立


@dataclass
class STLOr:
    args: list
    def robustness(self, signal, t=0, dt=0.01) -> float:
        return max(_eval(self.args[0], signal, t, dt),
                   _eval(self.args[1], signal, t, dt))


@dataclass
class STLAnd:
    args: list
    def robustness(self, signal, t=0, dt=0.01) -> float:
        return min(_eval(self.args[0], signal, t, dt),
                   _eval(self.args[1], signal, t, dt))


@dataclass
class STLNot:
    arg: Any
    def robustness(self, signal, t=0, dt=0.01) -> float:
        return -_eval(self.arg, signal, t, dt)


@dataclass
class STLGlobally:
    """G[a,b] φ — 全局必然"""
    arg: Any
    a: float
    b: float
    def robustness(self, signal, t=0, dt=0.01) -> float:
        t_end = _last_t(signal, t, self.b, dt)
        vals = [_eval(self.arg, signal, t + s, dt)
                for s in _arange_steps(self.a, self.b, dt)
                if t + s <= t_end]
        return min(vals) if vals else 0.0


@dataclass
class STLFinally:
    """F[a,b] φ — 最终必然

    P0 修复: 空窗 (时间超出信号长度, vals 为空) 时返回 -inf,
    让上层按 |ρ| > 0 判 FAIL。原代码返 0.0 会被误判为"通过"。
    """
    arg: Any
    a: float
    b: float
    def robustness(self, signal, t=0, dt=0.01) -> float:
        vals = [_eval(self.arg, signal, t + s, dt)
                for s in _arange_steps(self.a, self.b, dt)
                if t + s <= _last_t(signal, t, self.b, dt)]
        return float(max(vals)) if vals else float(-np.inf)


@dataclass
class STLImplies:
    """φ → ψ"""
    left: Any
    right: Any
    def robustness(self, signal, t=0, dt=0.01) -> float:
        l = _eval(self.left, signal, t, dt)
        r = _eval(self.right, signal, t, dt)
        return max(-l, r)


@dataclass
class STLNext:
    """X[a] φ — 下一时刻 (延迟 a 秒后)。

    ρ(X[a] φ, t) = ρ(φ, t+a)
    文档 §robustness 早就写了该算子, 但代码缺失; v0.7 补齐。
    """
    arg: Any
    a: float = 0.0
    def robustness(self, signal, t=0, dt=0.01) -> float:
        return _eval(self.arg, signal, t + self.a, dt)


@dataclass
class STLUntil:
    """φ U[a,b] ψ — 直到: ψ 必须在 [t+a,t+b] 内某时刻成立, 且在此之前 φ 一直成立。

    ρ(φ U[a,b] ψ, t) = max_{t'∈[t+a,t+b]} ( min( ρ(ψ,t'),
                                                 min_{t''∈[t,t')} ρ(φ,t'') ) )
    """
    left: Any    # φ (必须持续成立)
    right: Any   # ψ (最终必须成立)
    a: float = 0.0
    b: float = 0.1
    def robustness(self, signal, t=0, dt=0.01) -> float:
        best = -np.inf
        t_end = _last_t(signal, t, self.b, dt)
        for tp in _arange_steps(self.a, self.b, dt):
            if t + tp > t_end:
                break
            r_psi = _eval(self.right, signal, t + tp, dt)
            # t'' ∈ [t, t')
            prefix = [_eval(self.left, signal, t + s, dt)
                      for s in _arange_steps(0.0, max(tp - dt, 0.0), dt)]
            val = min(r_psi, min(prefix) if prefix else r_psi)
            if val > best:
                best = val
        # P0 修复: 空窗 (无 tp 落入 [a,b] 或全超 t_end) 应返回 -inf,
        # 让上层判 FAIL。原代码 `if np.isfinite(best) else 0.0` 把 -inf 转 0.0,
        # "从未成立" 被误判为中性而非失败。
        return float(best)


_STL_COMPOSITE = (STLOr, STLAnd, STLGlobally, STLFinally, STLImplies, STLNot,
                  STLNext, STLUntil)


def _eval(formula, signal, t, dt) -> float:
    """对公式 + 信号 + 时间点求 robustness。"""
    if isinstance(formula, STLAtom):
        x = _interp(signal, t, dt)
        return formula.robustness(x)
    elif isinstance(formula, _STL_COMPOSITE):
        return formula.robustness(signal, t, dt)
    else:
        raise TypeError(f"未知 STL 公式类型: {type(formula)}")


def _interp(signal: dict | np.ndarray, t: float, dt: float) -> np.ndarray:
    """从采样信号中插值取 t 时刻状态。

    支持两种信号格式:
      - np.ndarray: 直接作为状态 (单时刻公式评估)
      - dict: {"time": [...], "force": [...], "state": [...]}

    P0 修复: 原 L169 用 `signal.get("force", [0.0] if not ts else [signal.get("force", [0.0])[idx]])`
    — `.get(k, default)` 的 default **总是**被求值, 会触发内层 `[0.0][idx]` (idx>0 时 IndexError),
    或"force 缺失但先访问 [0.0]" 的副作用。重写为显式分支。
    """
    if isinstance(signal, np.ndarray):
        return signal
    if "state" in signal and np.ndim(signal["state"]) >= 1:
        return np.asarray(signal["state"], dtype=float)
    ts = signal.get("time", [])
    idx = int(round(t / dt)) if dt > 0 else 0
    idx = max(0, min(idx, max(len(ts) - 1, 0)))
    # P0 修复: 显式分支, 避免 .get 默认值副作用求值
    force = signal.get("force")
    if force is None:
        return np.array([0.0])
    if np.ndim(force) >= 1:
        # force 是数组: 取第 idx 个采样点 (越界钳位)
        idx_clamped = max(0, min(idx, len(force) - 1)) if len(force) > 0 else 0
        return np.array([float(force[idx_clamped])])
    # force 是标量: 直接返回
    return np.array([float(force)])


def _last_t(signal, t, duration, dt):
    """时域上界。

    v0.7 修: 传 ndarray (单时刻状态) 时 `signal.get` 会抛 AttributeError —
    而 _interp 明确支持 ndarray 信号, 于是"支持两种信号格式"的承诺在 G/F/U
    三个时序算子上是假的。ndarray 没有时间轴 ⇒ 上界就是 t+duration。
    """
    if not isinstance(signal, dict):
        return t + duration
    ts = signal.get("time", [0.0])
    if ts is not None and len(ts) > 0:
        return min(t + duration, float(ts[-1]))
    return t + duration


def _arange_steps(a: float, b: float, dt: float):
    n = max(1, int((b - a) / dt) + 1)
    return [a + i * dt for i in range(n)]


# ---- 常用安全监控公式库 ----

def force_limit_monitor(max_force: float) -> STLGlobally:
    """始终[0,T] 力不超过 max_force。"""
    return STLGlobally(
        STLAtom("force_limit", lambda x: float(np.max(np.abs(x))) - max_force),
        a=0, b=0.1,  # 100ms 滑动窗口
    )


def velocity_when_human_near(human_dist_thr=0.5, vel_thr=0.25) -> STLGlobally:
    """接近人类 → 限速 (ISO 13482 速度约束)。"""
    # 人对距离 < human_dist_thr → 速度必须 < vel_thr
    dist = STLAtom("human_dist", lambda x: float(x[0]) - human_dist_thr if len(x) > 0 else -1.0)
    vel = STLAtom("velocity", lambda x: float(abs(x[1])) - vel_thr if len(x) > 1 else 0.0)
    return STLGlobally(STLImplies(dist, vel), a=0, b=0.1)


def eventually_stop(torque_key: str = "torque", t_timeout: float = 0.01) -> STLFinally:
    """停止信号后 t_timeout 内力矩必须归零 (STO 验证)。"""
    return STLFinally(
        STLAtom(f"{torque_key}_zero", lambda x: float(abs(x[0]))),
        a=0, b=t_timeout,
    )


def force_ramp_monitor(max_rate: float = 1000.0, dt: float = 0.001) -> STLGlobally:
    """力变化率不超过 max_rate N/s (平滑性约束, 防冲击)。"""
    return STLGlobally(
        STLAtom("force_rate", lambda x: abs(x[0]) - max_rate * dt),
        a=0, b=0.1,
    )


def release_only_after_stop(t_release: float = 0.016) -> STLUntil:
    """STO 之后才允许松开抱闸: (力矩保持为零) U[0,t_release] (抱闸已闭合)。

    这是 G10 认证里最容易被忽略的一条——若允许"关断后立即松闸", 机械隔离
    尚未建立时负载可能反向拖动关节。用 Until 表达: 在抱闸闭合成立之前,
    力矩必须持续为零。
    """
    torque_zero = STLAtom("torque_zero", lambda x: float(abs(x[0])) - 1e-3)
    brake_closed = STLAtom("brake_closed", lambda x: 0.5 - float(x[1]) if len(x) > 1 else -1.0)
    return STLUntil(torque_zero, brake_closed, a=0.0, b=t_release)
