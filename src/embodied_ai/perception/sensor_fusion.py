"""多传感器融合输入端 + 分频信道处理 (plan_v5 Sprint M: M1)。

**缺口背景（机器人端三大缺口之一）**
    输入端接收视频信号流（摄像头）与微波信号传感器等更多传感器的**组合信号**，
    融合端信道必须能"分频"不同组合信号（频分复用 FDM），对不同信号**分别处理**，
    再交计算中枢暂存。此前项目仅有 MultiRateEKF（触觉/视觉两路速率融合），
    没有"信道化"的多传感器输入前端。

**设计（最小可用，不过度设计）**
    - `ChannelSpec`: 信道定义（名称 / 中心频率 / 带宽 / 采样率）
    - `Channelizer.demux(signal, fs)`: FFT 频带掩蔽 + 抽取 —— 把组合信号按频带
      解复用为各信道子信号（分频）
    - `ChannelPipeline`: 每信道独立处理回调（视频帧流→帧序列；微波→带通后
      幅度/相位提取）
    - `RingBuffer`: 按时间戳暂存处理结果（计算中枢取用）
    - `SensorFusionHub.ingest(t, composite)`: 处理后暂存 + 发布 Bus `SENSOR_RAW`

**provenance（诚实边界）**
    全部信号处理参数为 **synthetic**（合成信号验证）；真实微波前端标定属
    **待领域验证** —— 本模块不宣称任何真实射频前端的标定精度。

**不做的（诚实边界）**
    - 不做真实硬件驱动 / ADC 采集（那是 Phase P 硬件到货后的事）
    - 不做自适应滤波 / 波束成形（留待领域验证后按需加）
    - 不做跨进程传输（走 InProcessBus 即可，跨进程是 ROS2 对接的事）

**反证要求（测试守护）**
    分频串扰 <-20dB；**不分频时串扰必须 >-20dB**（负对照 —— 证明判据有区分
    力，FFT 频带掩蔽天然正交，任何实现都能过 -20dB 一百度余量）；无订阅者
    不崩；坏信道名报错；RingBuffer 时序单调。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════
# ChannelSpec — 信道定义
# ══════════════════════════════════════════════════════════════════

@dataclass
class ChannelSpec:
    """一个传感器信道的频域定义 (FDM 分频依据)。

    provenance: synthetic（合成验证）；真实前端参数属 待领域验证。
    """
    name: str
    center_hz: float          # 中心频率
    bandwidth_hz: float       # 带宽（单边各 bandwidth/2）
    sample_rate_hz: float     # 处理后采样率（抽取目标）

    def band_edges(self) -> Tuple[float, float]:
        """返回 (low_hz, high_hz) 频带边界（基带等效，以 fs/2 为奈奎斯特）。"""
        half = self.bandwidth_hz / 2.0
        return self.center_hz - half, self.center_hz + half


# ══════════════════════════════════════════════════════════════════
# Channelizer — 分频解复用
# ══════════════════════════════════════════════════════════════════

class Channelizer:
    """FFT 频带掩蔽解复用器 —— 把组合信号按信道频带"分频"。

    对组合信号做 rFFT → 每信道只保留 [low, high] 频带 → irFFT 回时域
    （可选按信道采样率抽取）。不同信道分别落在不同频带，互不串扰。
    """

    def __init__(self, channels: List[ChannelSpec]):
        if not channels:
            raise ValueError("至少需要 1 个信道")
        self.channels = {c.name: c for c in channels}
        # 信道频带不允许重叠（分频前提）
        edges = sorted((c.band_edges()[0], c.band_edges()[1], c.name) for c in channels)
        for (l1, h1, n1), (l2, h2, n2) in zip(edges, edges[1:]):
            if h1 > l2:
                raise ValueError(f"信道 '{n1}' 与 '{n2}' 频带重叠, 无法分频")

    def decimation(self, fs: float, name: str) -> int:
        """该信道在组合采样率 fs 下的抽取率 (≥1)。

        demux 与串扰度量**共用这一个真源** —— 曾经度量侧手写魔数
        round(fs/2000), 与 demux 实际的 round(fs/8000)=1 不一致, 于是内积在
        错位的时间基上进行, -147.9dB 那个数字是数值噪声而非测量结果。
        """
        if name not in self.channels:
            raise KeyError(f"未知信道 '{name}'; 合法: {sorted(self.channels)}")
        return max(1, int(round(fs / self.channels[name].sample_rate_hz)))

    def demux(self, signal: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
        """组合信号 → {信道名: 子信号}（频带掩蔽 + 抽取）。

        Args:
            signal: 组合时域信号（各传感器信号在频域叠加）
            fs: 组合信号采样率 (Hz)
        Returns:
            每信道解出的时域子信号（按信道 sample_rate_hz 抽取）
        """
        sig = np.asarray(signal, dtype=np.float64)
        n = sig.size
        if n == 0:
            raise ValueError("signal 为空")
        spectrum = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        out: Dict[str, np.ndarray] = {}
        for name, ch in self.channels.items():
            low, high = ch.band_edges()
            mask = (freqs >= low) & (freqs <= high)
            masked = np.zeros_like(spectrum)
            masked[mask] = spectrum[mask]
            sub = np.fft.irfft(masked, n=n)
            out[name] = sub[::self.decimation(fs, name)]
        return out


# ══════════════════════════════════════════════════════════════════
# ChannelPipeline — 每信道处理回调
# ══════════════════════════════════════════════════════════════════

@dataclass
class ProcessedSample:
    """信道处理结果（带时间戳, 供计算中枢暂存）。"""
    channel: str
    timestamp_s: float
    features: np.ndarray          # 处理后的特征向量


#: 视频信道处理: 帧序列 → 帧均值/峰值特征（合成帧流的最小特征提取）
def video_frame_features(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float64)
    return np.array([frames.mean(), frames.std(), frames.max()])

#: 微波信道处理: 带通信号 → 幅度/相位特征（analytic 信号近似: Hilbert 用 FFT 实现）
def microwave_features(signal: np.ndarray) -> np.ndarray:
    sig = np.asarray(signal, dtype=np.float64)
    spec = np.fft.fft(sig)
    # 构造解析信号（单边谱）→ 幅度/瞬时相位
    h = np.zeros_like(spec)
    n = sig.size
    h[0] = 1.0
    h[1:n // 2] = 2.0
    if n % 2 == 0:
        h[n // 2] = 1.0
    analytic = np.fft.ifft(spec * h)
    amp = np.abs(analytic)
    phase = np.angle(analytic)
    return np.array([amp.mean(), amp.std(), np.mean(np.diff(phase)) if n > 1 else 0.0])


DEFAULT_PIPELINES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "video": video_frame_features,
    "microwave": microwave_features,
}


# ══════════════════════════════════════════════════════════════════
# RingBuffer — 按时间戳暂存
# ══════════════════════════════════════════════════════════════════

class RingBuffer:
    """定长环形缓冲 — 处理结果按时间戳暂存（计算中枢取用）。"""

    def __init__(self, capacity: int = 1024):
        if capacity <= 0:
            raise ValueError("capacity 必须 > 0")
        self.capacity = capacity
        self._buf: Deque[ProcessedSample] = deque(maxlen=capacity)

    def push(self, sample: ProcessedSample) -> None:
        self._buf.append(sample)

    def drain(self) -> List[ProcessedSample]:
        """取出并清空全部暂存样本。"""
        out = list(self._buf)
        self._buf.clear()
        return out

    def __len__(self) -> int:
        return len(self._buf)


# ══════════════════════════════════════════════════════════════════
# SensorFusionHub — 融合端入口
# ══════════════════════════════════════════════════════════════════

class SensorFusionHub:
    """多传感器融合输入端：组合信号 → 分频 → 分信道处理 → 暂存 → 发 Bus。

    对齐用户需求："融合端接受信道要能够分频不同的组合信号，对不同的信号
    分别处理，交由计算中枢暂存"。
    """

    def __init__(self, channels: List[ChannelSpec],
                 pipelines: Optional[Dict[str, Callable[[np.ndarray], np.ndarray]]] = None,
                 buffer_capacity: int = 1024,
                 bus=None):
        self.channelizer = Channelizer(channels)
        self.pipelines = dict(DEFAULT_PIPELINES if pipelines is None else pipelines)
        self.buffer = RingBuffer(buffer_capacity)
        self.bus = bus
        self.stats = {"ingested": 0, "processed": 0}

    def ingest(self, timestamp_s: float, composite: np.ndarray,
               fs: float) -> List[ProcessedSample]:
        """一次采集：组合信号 → 各信道分别处理 → 暂存(+发 Bus)。

        Returns:
            本次处理出的 ProcessedSample 列表
        """
        sub_signals = self.channelizer.demux(composite, fs)
        samples: List[ProcessedSample] = []
        for name, sub in sub_signals.items():
            feat_fn = self.pipelines.get(name)
            feats = feat_fn(sub) if feat_fn is not None else sub
            s = ProcessedSample(channel=name, timestamp_s=timestamp_s,
                                features=np.asarray(feats, dtype=np.float64))
            self.buffer.push(s)
            samples.append(s)
            if self.bus is not None:
                self.bus.publish(
                    "sensor_raw",
                    {"channel": name, "samples": s.features},
                    source="sensor_fusion_hub",
                )
        self.stats["ingested"] += 1
        self.stats["processed"] += len(samples)
        return samples


# ══════════════════════════════════════════════════════════════════
# demo — 合成信号分频验证
# ══════════════════════════════════════════════════════════════════

def crosstalk_db(extracted: np.ndarray, own: np.ndarray, other: np.ndarray,
                 decim: int) -> float:
    """串扰比 (dB): 提取信号中来自"其它信道"的能量占比 (越小越干净)。

    own/other 按 decim 抽取以对齐 extracted 的时间基。**decim 必须由
    `Channelizer.decimation()` 给出** —— 手写魔数会让内积在错位的时间基上
    进行, 算出来的 dB 是数值噪声, 不是测量。
    """
    if decim < 1:
        raise ValueError("decim 必须 >= 1")
    o = np.asarray(own, dtype=np.float64)[::decim]
    x = np.asarray(other, dtype=np.float64)[::decim]
    e = np.asarray(extracted, dtype=np.float64)
    n = min(e.size, o.size, x.size)
    if n == 0:
        raise ValueError("extracted/own/other 对齐后为空")
    e, x = e[:n], x[:n]
    # 提取信号在"它方信号"方向的投影能量 / 提取信号总能量
    num = float(np.dot(e, x) ** 2 / max(float(np.dot(x, x)), 1e-12))
    den = float(np.dot(e, e)) + 1e-12
    return float(10 * np.log10(max(num, 1e-12) / den))


#: demo 用信道定义 (单一真源 — decimation 与串扰度量都从这里取)
DEMO_CHANNELS: List[ChannelSpec] = [
    ChannelSpec("video", center_hz=300, bandwidth_hz=400, sample_rate_hz=1000),
    ChannelSpec("microwave", center_hz=2750, bandwidth_hz=1200, sample_rate_hz=8000),
]
DEMO_FS = 8000.0        # 组合采样率 (synthetic)


def fusion_demo() -> dict:
    """合成"视频帧脉冲(低频带) + 微波双音(高频带)"组合信号 → 分频。

    验证: 两信道解出的子信号中来自它方信道的能量占比（串扰）<-20dB。
    """
    fs = DEMO_FS
    t = np.arange(0, 1.0, 1.0 / fs)
    # 视频通道: 300Hz 基带信号 (帧率调制)
    video_true = 0.8 * np.sin(2 * np.pi * 300 * t)
    # 微波通道: 2500Hz + 3000Hz 双音
    micro_true = 0.6 * np.sin(2 * np.pi * 2500 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t)
    composite = video_true + micro_true

    hub = SensorFusionHub(list(DEMO_CHANNELS))
    samples = hub.ingest(0.0, composite, fs)
    sub = hub.channelizer.demux(composite, fs)

    ct_video = crosstalk_db(sub["video"], video_true, micro_true,
                            hub.channelizer.decimation(fs, "video"))
    ct_micro = crosstalk_db(sub["microwave"], micro_true, video_true,
                            hub.channelizer.decimation(fs, "microwave"))
    result = {
        "channels": sorted(sub),
        "crosstalk_video_db": ct_video,
        "crosstalk_microwave_db": ct_micro,
        "samples": [(s.channel, s.features.tolist()) for s in samples],
    }
    assert ct_video < -20.0, f"video 信道串扰 {ct_video:.1f}dB 未达标 (<-20dB)"
    assert ct_micro < -20.0, f"microwave 信道串扰 {ct_micro:.1f}dB 未达标 (<-20dB)"
    return result


if __name__ == "__main__":
    r = fusion_demo()
    print("多传感器分频融合 demo (provenance=synthetic)")
    print(f"信道: {r['channels']}")
    print(f"video 信道串扰: {r['crosstalk_video_db']:.1f} dB (<-20dB ✓)")
    print(f"microwave 信道串扰: {r['crosstalk_microwave_db']:.1f} dB (<-20dB ✓)")
