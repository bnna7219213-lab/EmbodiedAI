"""G7 数字孪生: FFT 频谱 + 异常判别器 (双轨 Python 后端)。

**缺口背景 (G7)**
    Kotlin `DigitalTwinEngine` 只能做**阈值告警** (温度/功率超限即报) →
    是"仪表盘"而非"预测性维护": 无法发现早期故障 (轴承磨损/齿隙增大/Cogging
    异常在时域幅值上不显著, 但在**频谱**上有特征峰)。

**oracle 说明 (诚实标注)**
    jar 侧无独立 digital-twin CLI → **无 jar oracle**。本后端的对照基准:
      ① 合成已知故障信号 (注入确定频率/幅值) → 判别器必须识别出对应特征
      ② 反证: 健康信号 (纯噪声/单频正常) 不得报故障 (防"恒真判别器")
      ③ 阈值法作为对照基线的**局限**需被如实展示 (时域超限才报 → 漏检早期)
    "能识别已知注入"与"不误报"两条同时成立才算通过 — 单靠 ① 会写出恒真判别器。

**方法** (工程可验证, 非黑箱):
    1. 去均值 → 窗函数 (Hann) → `np.fft.rfft` 功率谱
    2. 特征频率检测: 基频 f0 及其谐波/边带 (轴承 BPFO/BPFI/BSF/FTF 通用公式)
    3. 异常判据: 特定频带能量占比超阈值 + 谱峰显著度 (peak/median)
    4. 与阈值基线对比: 展示"时域没超但频域有特征"的漏检场景
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════
# 频谱分析
# ══════════════════════════════════════════════════════════════════

@dataclass
class SpectrumResult:
    freqs_hz: np.ndarray
    power: np.ndarray
    peak_freq_hz: float
    peak_power: float
    peak_sharpness: float          # 峰/中位数 — 显著度 (尖锐特征 = 机械缺陷)
    band_energy_ratio: float       # 特征带能量 / 总能量


def power_spectrum(signal: np.ndarray, fs_hz: float = 1000.0,
                   detrend: bool = True) -> SpectrumResult:
    """单边功率谱 (Hann 窗 + rfft)。

    Args:
        signal: 时域样本 (1D)
        fs_hz:  采样率
    """
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    if n < 8:
        raise ValueError(f"样本太少 ({n}), 至少 8 点")
    if detrend:
        x = x - x.mean()
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    p = (np.abs(X) ** 2) / max(n, 1)
    f = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    # 忽略直流
    p, f = p[1:], f[1:]
    if p.size == 0:
        return SpectrumResult(f, p, 0.0, 0.0, 0.0, 0.0)
    idx = int(np.argmax(p))
    med = float(np.median(p)) or 1e-30
    return SpectrumResult(
        freqs_hz=f, power=p,
        peak_freq_hz=float(f[idx]), peak_power=float(p[idx]),
        peak_sharpness=float(p[idx] / med),
        band_energy_ratio=0.0,      # 由 band_ratio() 按需计算
    )


def band_ratio(freqs: np.ndarray, power: np.ndarray,
               f_lo: float, f_hi: float) -> float:
    """指定频带能量占总能量之比 (异常判据的输入)。"""
    tot = float(np.sum(power)) or 1e-30
    m = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(power[m]) / tot)


# ══════════════════════════════════════════════════════════════════
# 轴承/齿轮特征频率 (通用公式, 不依赖具体型号)
# ══════════════════════════════════════════════════════════════════

def bearing_fault_freqs(rpm: float, n_balls: int, ball_dia_mm: float,
                        pitch_dia_mm: float, contact_deg: float = 0.0
                        ) -> Dict[str, float]:
    """轴承四类故障特征频率 (Hz) — BPFO/BPFI/BSF/FTF 标准公式。

    这些频率在频谱上出现尖峰 = 对应部位早期缺陷 (阈值法在时域看不出来)。
    """
    fr = rpm / 60.0
    r = (ball_dia_mm / pitch_dia_mm) * np.cos(np.deg2rad(contact_deg))
    return {
        "fr_hz": float(fr),                                    # 转频
        "BPFO_hz": float((n_balls / 2.0) * fr * (1 - r)),      # 外圈
        "BPFI_hz": float((n_balls / 2.0) * fr * (1 + r)),      # 内圈
        "BSF_hz": float((pitch_dia_mm / (2 * ball_dia_mm)) * fr * (1 - r ** 2)),  # 滚珠
        "FTF_hz": float(0.5 * fr * (1 - r)),                   # 保持架
    }


# ══════════════════════════════════════════════════════════════════
# 异常判别器
# ══════════════════════════════════════════════════════════════════

@dataclass
class FaultSignature:
    """一类故障的频谱特征定义 (可证伪: 命中/未命中)。"""
    name: str
    band_hz: Tuple[float, float]
    min_band_ratio: float          # 该带能量占比下限
    min_sharpness: float           # 谱峰显著度下限


DEFAULT_SIGNATURES: List[FaultSignature] = [
    FaultSignature("bearing_outer_race", (0, 0), 0.15, 8.0),   # 带范围运行时按 BPFO 填
    FaultSignature("gear_mesh", (0, 0), 0.12, 6.0),
    FaultSignature("cogging_anomaly", (0, 0), 0.10, 5.0),
]


def detect_faults(signal: np.ndarray, fs_hz: float = 1000.0,
                  signatures: Optional[List[Tuple[str, Tuple[float, float], float, float]]] = None,
                  fault_freqs: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """频谱异常判别 — 返回命中/未命中的故障特征 (逐项可证伪)。

    signatures: [(name, (f_lo, f_hi), min_ratio, min_sharpness)]
                缺省时用 fault_freqs 的 BPFO/BPFI/齿轮啮合自动构造窄带。
    """
    sp = power_spectrum(signal, fs_hz)
    f, p = sp.freqs_hz, sp.power
    sigs = signatures
    if sigs is None:
        sigs = []
        ff = fault_freqs or {}
        # 窄带 (±5%) 围绕特征频率
        for key, label in (("BPFO_hz", "bearing_outer_race"),
                           ("BPFI_hz", "bearing_inner_race"),
                           ("BSF_hz", "bearing_ball")):
            if key in ff and ff[key] > 0:
                c = ff[key]
                sigs.append((label, (c * 0.95, c * 1.05), 0.08, 6.0))
        for key, label in (("gear_mesh_hz", "gear_mesh"),
                           ("cogging_hz", "cogging_anomaly")):
            if key in ff and ff[key] > 0:
                c = ff[key]
                sigs.append((label, (c * 0.95, c * 1.05), 0.08, 5.0))

    hits: List[Dict[str, Any]] = []
    for name, (lo, hi), min_ratio, min_sharp in sigs:
        ratio = band_ratio(f, p, lo, hi)
        # 该带内的显著度
        m = (f >= lo) & (f <= hi)
        sharp = float(np.max(p[m]) / (np.median(p) or 1e-30)) if np.any(m) else 0.0
        hit = (ratio >= min_ratio) and (sharp >= min_sharp)
        hits.append({"fault": name, "band_hz": [lo, hi], "band_ratio": ratio,
                     "sharpness": sharp, "hit": bool(hit),
                     "reason": (f"带能量占比 {ratio:.3f}≥{min_ratio} 且 "
                                f"谱峰显著度 {sharp:.1f}≥{min_sharp}"
                                if hit else
                                f"未达阈 (ratio {ratio:.3f}, sharp {sharp:.1f})")})

    return {
        "n_signatures": len(sigs),
        "n_hits": sum(1 for h in hits if h["hit"]),
        "hits": hits,
        "peak_freq_hz": sp.peak_freq_hz,
        "peak_sharpness": sp.peak_sharpness,
        "faulty": any(h["hit"] for h in hits),
    }


def threshold_baseline(signal: np.ndarray, limit: float) -> Dict[str, Any]:
    """阈值法基线 (jar 现有做法) — 用于展示其局限 (漏检早期故障)。"""
    x = np.asarray(signal, dtype=float).ravel()
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    return {"alarm": bool(peak > limit), "peak_abs": peak, "rms": rms,
            "limit": float(limit)}


# ══════════════════════════════════════════════════════════════════
# 合成故障信号 (用于可证伪验证 — 不是随机数)
# ══════════════════════════════════════════════════════════════════

def synth_vibration(fs_hz: float = 1000.0, duration_s: float = 1.0,
                    base_freq_hz: float = 25.0, noise: float = 0.02,
                    fault_freq_hz: float = 0.0, fault_amp: float = 0.0,
                    seed: int = 0) -> np.ndarray:
    """合成振动信号: 基频 + 噪声 (+ 可选注入故障特征频率)。

    fault_amp=0 → 健康信号; fault_amp>0 → 注入已知特征 (判别器应识别)。
    这是**确定性**信号 (固定 seed), 不是随机数冒充。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / fs_hz)
    x = 1.0 * np.sin(2 * np.pi * base_freq_hz * t)          # 正常运转基频
    x += 0.3 * np.sin(2 * np.pi * 2 * base_freq_hz * t)     # 二次谐波
    if fault_amp > 0 and fault_freq_hz > 0:
        x += fault_amp * np.sin(2 * np.pi * fault_freq_hz * t)   # 注入特征
    x += rng.normal(0, noise, t.size)                       # 本底噪声
    return x


# --------------------------------------------------------------------------- #
# 双轨注册: 能力名 "digital_twin" (G7)
# --------------------------------------------------------------------------- #
def _register() -> None:
    """G7 后端注册 (jar 侧无 digital-twin CLI → available_backends 只有 python)。"""
    from ..adapters.edalab import register_python_backend

    @register_python_backend("digital_twin")
    def digital_twin_backend(fs_hz: float = 1000.0, seed: int = 0) -> Dict[str, Any]:
        """G7: 频谱判别 vs 阈值基线 — 展示阈值法的漏检 (可证伪)。"""
        # 轴承特征频率 (关节模组典型: 1500rpm, 8 球)
        ff = bearing_fault_freqs(rpm=1500.0, n_balls=8, ball_dia_mm=5.0,
                                 pitch_dia_mm=22.0)
        ff["gear_mesh_hz"] = 1500.0 / 60.0 * 7 * 4      # 转频 × 减速比 × 齿数比
        ff["cogging_hz"] = 7 * 1500.0 / 60.0            # 极对数 × 转频

        # ① 健康信号: 不应报故障
        healthy = synth_vibration(fs_hz=fs_hz, fault_amp=0.0, seed=seed)
        r_healthy = detect_faults(healthy, fs_hz, fault_freqs=ff)

        # ② 注入故障: 必须识别出对应特征
        bpfo = ff["BPFO_hz"]
        faulty = synth_vibration(fs_hz=fs_hz, fault_freq_hz=bpfo,
                                 fault_amp=0.8, seed=seed)
        r_faulty = detect_faults(faulty, fs_hz, fault_freqs=ff)

        # ③ 阈值法基线: 相同信号, 阈值法是否漏检
        limit = 3.0
        b_healthy = threshold_baseline(healthy, limit)
        b_faulty = threshold_baseline(faulty, limit)

        # 判定: 健康不误报 + 故障被识别 (+ 阈值法漏检即证明 FFT 的价值)
        ok = (not r_healthy["faulty"]) and r_faulty["faulty"]
        return {
            "status": "OK" if ok else "FAIL",
            "ok": bool(ok),
            "fault_freqs": ff,
            "healthy_detected_fault": bool(r_healthy["faulty"]),
            "faulty_detected_fault": bool(r_faulty["faulty"]),
            "faulty_hits": [h["fault"] for h in r_faulty["hits"] if h["hit"]],
            "threshold_baseline": {
                "healthy_alarm": b_healthy["alarm"],
                "faulty_alarm": b_faulty["alarm"],
                "missed_early_fault": bool(not b_faulty["alarm"] and r_faulty["faulty"]),
            },
            "n_signatures": r_faulty["n_signatures"],
        }


_register()
