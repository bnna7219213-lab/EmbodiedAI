"""d1 e-nose 真实标定方案 测试 (plan_v6 后续)。

反证: 已知模型数据能被恢复; 交叉补偿显著降串扰; 漂移超阈值必触发再标定;
残差超限必须 FAIL 门禁 (不是恒过); 未标定 compensate 必须报错。
"""
import numpy as np
import pytest

from embodied_ai.perception.chem_calibration import (
    ENOSE_CHANNELS,
    CalibrationPoint,
    CalibrationReport,
    ChannelCurve,
    DriftMonitor,
    EnoseCalibrator,
    calibration_demo,
    make_synthetic_calibration_set,
)


class TestCalibrationPoint:
    def test_shape(self):
        p = CalibrationPoint(concentrations=np.zeros(3),
                             responses=np.zeros(ENOSE_CHANNELS))
        assert p.responses.shape == (ENOSE_CHANNELS,)

    def test_bad_response_shape_raises(self):
        with pytest.raises(ValueError):
            CalibrationPoint(concentrations=np.zeros(3),
                             responses=np.zeros(2))


class TestEnoseCalibrator:
    def test_fit_requires_two_points(self):
        cal = EnoseCalibrator()
        pts, _ = make_synthetic_calibration_set(n_points=1)
        with pytest.raises(ValueError):
            cal.fit(pts)

    def test_underdetermined_rejected(self):
        """欠定/恰定系统会精确插值 (残差恒 0) → 必须拒绝 (防门禁恒过)。"""
        cal = EnoseCalibrator()
        # 3 气体 → 4 参数; 4 个点是恰定, 必须拒绝
        pts, _ = make_synthetic_calibration_set(n_points=4, n_gases=3)
        with pytest.raises(ValueError, match="欠定"):
            cal.fit(pts)

    def test_recovers_known_model(self):
        """已知 A_true 的合成数据 → 拟合恢复斜率/截距 (误差 < 5%)。"""
        pts, A_true = make_synthetic_calibration_set(n_points=8, seed=1)
        cal = EnoseCalibrator()
        cal.fit(pts)
        # 每通道主气体斜率应接近 A_true 对应元素
        for ch in range(ENOSE_CHANNELS):
            main = int(np.argmax(np.abs(A_true[ch])))
            slope = cal.A[ch, main]
            assert slope == pytest.approx(A_true[ch, main], rel=0.05), \
                f"ch{ch} 斜率恢复偏差: {slope} vs {A_true[ch, main]}"

    def test_cross_compensation_reduces_crosstalk(self):
        pts, A_true = make_synthetic_calibration_set(seed=2)
        cal = EnoseCalibrator()
        cal.fit(pts)
        # 单气体注入 → 补偿后非目标气体应接近 0
        conc = np.array([80.0, 0.0, 0.0])
        resp = A_true @ conc + np.full(ENOSE_CHANNELS, 0.3)
        est = cal.compensate(resp)
        assert abs(est[0] - 80.0) < 15.0, f"目标气体估计偏差: {est[0]}"
        assert np.abs(est[1:]).max() < 15.0, f"串扰未消除: {est}"

    def test_compensate_without_fit_raises(self):
        with pytest.raises(RuntimeError):
            EnoseCalibrator().compensate(np.zeros(ENOSE_CHANNELS))

    def test_batch_compensate(self):
        pts, _ = make_synthetic_calibration_set(seed=3)
        cal = EnoseCalibrator()
        cal.fit(pts)
        batch = np.stack([p.responses for p in pts])
        est = cal.compensate(batch)
        assert est.shape == (len(pts), 3)


class TestGate:
    def test_good_data_passes(self):
        pts, _ = make_synthetic_calibration_set(seed=4)
        report = EnoseCalibrator().fit(pts)
        assert report.passed, report.failures

    def test_noisy_data_fails_gate(self):
        """反证: 高噪声数据必须 FAIL 门禁 (门禁不是恒过)。"""
        pts, _ = make_synthetic_calibration_set(n_points=8, noise_std=25.0,
                                                seed=5)
        report = EnoseCalibrator().fit(pts)
        assert not report.passed, "高噪声标定不应过门禁"
        assert report.failures

    def test_summary_keys(self):
        pts, _ = make_synthetic_calibration_set(seed=6)
        s = EnoseCalibrator().fit(pts).summary()
        assert set(s) >= {"passed", "failures", "n_curves", "mean_r2",
                          "compensated_leakage"}


class TestDriftMonitor:
    def test_no_drift(self):
        m = DriftMonitor(np.full(ENOSE_CHANNELS, 0.3))
        assert not m.needs_recalibration(np.full(ENOSE_CHANNELS, 0.31))

    def test_drift_detected(self):
        m = DriftMonitor(np.full(ENOSE_CHANNELS, 0.3))
        assert m.needs_recalibration(np.full(ENOSE_CHANNELS, 0.4))

    def test_drift_value(self):
        m = DriftMonitor(np.full(ENOSE_CHANNELS, 1.0))
        assert m.drift(np.full(ENOSE_CHANNELS, 1.5)) == pytest.approx(0.5)

    def test_bad_baseline_shape_raises(self):
        with pytest.raises(ValueError):
            DriftMonitor(np.zeros(2))

    def test_bad_reading_shape_raises(self):
        m = DriftMonitor(np.full(ENOSE_CHANNELS, 0.3))
        with pytest.raises(ValueError):
            m.drift(np.zeros(2))


class TestDemo:
    def test_demo_assertions(self):
        r = calibration_demo()
        assert r["gate_passed"] is True
        assert r["drift_detected"] is True
        assert r["a_recovery_max_err"] < 0.15
