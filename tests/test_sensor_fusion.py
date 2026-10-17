"""M1 多传感器融合 + 分频信道处理 测试 (plan_v5 Sprint M)。

反证: 分频串扰 <-20dB; 坏信道/空信号/重叠频带报错; RingBuffer 时序; Bus 发布。
"""
import numpy as np
import pytest

from embodied_ai.perception.sensor_fusion import (
    DEMO_CHANNELS,
    DEMO_FS,
    ChannelSpec,
    Channelizer,
    ProcessedSample,
    RingBuffer,
    SensorFusionHub,
    crosstalk_db,
    fusion_demo,
    microwave_features,
    video_frame_features,
)


FS = 8000.0


def _composite(t):
    video = 0.8 * np.sin(2 * np.pi * 300 * t)
    micro = 0.6 * np.sin(2 * np.pi * 2500 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t)
    return video + micro, video, micro


class TestChannelSpec:
    def test_band_edges(self):
        c = ChannelSpec("video", center_hz=500, bandwidth_hz=800, sample_rate_hz=1000)
        assert c.band_edges() == (100.0, 900.0)


class TestChannelizer:
    def test_demux_separates_channels(self):
        t = np.arange(0, 1.0, 1.0 / FS)
        comp, video, micro = _composite(t)
        specs = [
            ChannelSpec("video", 300, 400, 1000),
            ChannelSpec("microwave", 2750, 1200, 8000),
        ]
        ch = Channelizer(specs)
        sub = ch.demux(comp, FS)
        assert set(sub) == {"video", "microwave"}
        # 各自信道应与自身原信号相关性强 (时间基按抽取率对齐后比较)
        for spec, own in zip(specs, (video, micro)):
            decim = max(1, int(round(FS / spec.sample_rate_hz)))
            o = own[::decim]
            e = sub[spec.name]
            n = min(e.size, o.size)
            e, o = e[:n], o[:n]
            corr = np.dot(e, o) ** 2 / (np.dot(e, e) * np.dot(o, o) + 1e-12)
            assert corr > 0.5, f"{spec.name} 信道提取与原信号相关性过低: {corr:.3f}"

    def test_crosstalk_below_20db(self):
        r = fusion_demo()
        assert r["crosstalk_video_db"] < -20.0
        assert r["crosstalk_microwave_db"] < -20.0

    def test_crosstalk_metric_has_discriminating_power(self):
        """负对照: 不做分频时微波信道串扰必须 >-20dB。

        FFT 频带掩蔽对正交单音天然干净 (实测 -121/-154dB), <-20dB 的判据有
        一百多 dB 余量 —— 必须证明这个门槛真的能失败, 否则它守护不了任何东西。
        """
        t = np.arange(0, 1.0, 1.0 / FS)
        comp, _, micro = _composite(t)
        video = 0.8 * np.sin(2 * np.pi * 300 * t)
        raw = crosstalk_db(comp, micro, video, 1)     # 不分频, 直接用组合信号
        assert raw > -20.0, f"不分频串扰 {raw:.1f}dB — 判据已失去区分力"
        assert fusion_demo()["crosstalk_microwave_db"] < raw - 100

    def test_decimation_is_single_source_of_truth(self):
        """串扰度量与 demux 必须共用抽取率。

        曾经度量侧手写 round(fs/2000)=4, 而 demux 实际是 round(fs/8000)=1,
        内积在错位的时间基上进行 —— -147.9dB 那个数字是数值噪声而非测量。
        """
        ch = Channelizer([s for s in DEMO_CHANNELS])
        assert ch.decimation(DEMO_FS, "video") == 8
        assert ch.decimation(DEMO_FS, "microwave") == 1
        t = np.arange(0, 0.5, 1.0 / DEMO_FS)          # 4000 点
        sub = ch.demux(_composite(t)[0], DEMO_FS)
        assert sub["video"].size == 500               # 4000 / 8
        assert sub["microwave"].size == 4000          # 4000 / 1

    def test_decimation_unknown_channel_raises(self):
        ch = Channelizer([ChannelSpec("a", 100, 50, 200)])
        with pytest.raises(KeyError):
            ch.decimation(FS, "ghost")

    def test_crosstalk_bad_decim_raises(self):
        with pytest.raises(ValueError):
            crosstalk_db(np.ones(10), np.ones(10), np.ones(10), 0)

    def test_empty_signal_raises(self):
        ch = Channelizer([ChannelSpec("a", 100, 50, 200)])
        with pytest.raises(ValueError):
            ch.demux(np.array([]), FS)

    def test_no_channels_raises(self):
        with pytest.raises(ValueError):
            Channelizer([])

    def test_overlapping_bands_raise(self):
        with pytest.raises(ValueError):
            Channelizer([
                ChannelSpec("a", 500, 800, 1000),
                ChannelSpec("b", 700, 800, 1000),   # 与 a 频带重叠
            ])

    def test_demux_returns_every_declared_channel(self):
        ch = Channelizer([ChannelSpec("a", 100, 50, 200)])
        assert "a" in ch.demux(np.zeros(100), FS)


class TestPipelines:
    def test_video_features_shape(self):
        f = video_frame_features(np.ones((4, 4)))
        assert f.shape == (3,)
        assert f[0] == pytest.approx(1.0)

    def test_microwave_features_amplitude(self):
        t = np.arange(0, 0.1, 1.0 / 2000)
        sig = 2.0 * np.sin(2 * np.pi * 300 * t)
        f = microwave_features(sig)
        assert f.shape == (3,)
        assert f[0] == pytest.approx(2.0, rel=0.15)   # 幅度 ~2


class TestRingBuffer:
    def test_push_drain_capacity(self):
        rb = RingBuffer(capacity=2)
        rb.push(ProcessedSample("a", 0.0, np.zeros(2)))
        rb.push(ProcessedSample("a", 0.1, np.zeros(2)))
        rb.push(ProcessedSample("a", 0.2, np.zeros(2)))   # 挤掉最旧
        assert len(rb) == 2
        items = rb.drain()
        assert [s.timestamp_s for s in items] == [0.1, 0.2]
        assert len(rb) == 0

    def test_bad_capacity_raises(self):
        with pytest.raises(ValueError):
            RingBuffer(0)


class TestSensorFusionHub:
    def test_ingest_processes_each_channel(self):
        hub = SensorFusionHub([
            ChannelSpec("video", 300, 400, 1000),
            ChannelSpec("microwave", 2750, 1200, 8000),
        ])
        t = np.arange(0, 1.0, 1.0 / FS)
        comp, _, _ = _composite(t)
        samples = hub.ingest(1.5, comp, FS)
        assert [s.channel for s in samples] == ["video", "microwave"]
        assert all(s.timestamp_s == 1.5 for s in samples)
        assert len(hub.buffer) == 2
        assert hub.stats["processed"] == 2

    def test_ingest_publishes_to_bus(self):
        from embodied_ai.runtime.bus import Collector, InProcessBus

        bus = InProcessBus()
        c = Collector()
        bus.subscribe("sensor_raw", c)
        hub = SensorFusionHub(
            [ChannelSpec("video", 300, 400, 1000),
             ChannelSpec("microwave", 2750, 1200, 8000)],
            bus=bus,
        )
        t = np.arange(0, 0.25, 1.0 / FS)
        comp, _, _ = _composite(t)
        hub.ingest(0.0, comp, FS)
        assert len(c.messages) == 2
        assert {m.payload["channel"] for m in c.messages} == {"video", "microwave"}

    def test_no_bus_does_not_crash(self):
        hub = SensorFusionHub([ChannelSpec("video", 300, 400, 1000)])
        t = np.arange(0, 0.1, 1.0 / FS)
        hub.ingest(0.0, _composite(t)[0], FS)   # bus=None, 不崩
