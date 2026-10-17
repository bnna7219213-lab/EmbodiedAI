"""e-nose 真实标定方案 — 多点标定/交叉敏感补偿/漂移监控/验收门禁 (plan_v6 后续: d1)。

**缺口背景（模拟→真实的标定断点）**
    chem_bio.py 的化学生化传感是 **synthetic**（合成仿真）。要接真实 e-nose
    （MOX 气体阵列）, 必须有一套**真实标定方案**:
    ① 多点标定曲线（零气/标气多浓度点 → 每通道响应-浓度拟合）
    ② 交叉敏感补偿（MOX 阵列对多气体交叉响应 → 矩阵解耦）
    ③ 温湿度补偿 + 基线漂移监控（老化 → 需再标定门禁）
    ④ 验收门禁（残差/R²/串扰泄漏不达标 → 拒绝放行）

**真实标定协议（docs/19 §6, 本模块是其代码化）**
    1. 零气 (clean air) 基线 60s → 每通道 baseline
    2. 标气多点: 目标气体 0/10/50/100 ppm 各 30s（浓度点可配）
       + 干扰气体单独注入（建交叉敏感矩阵 A: 响应 = A·浓度）
    3. 拟合: 每通道 linear/R² → 标定曲线; A 最小二乘求逆 → 交叉补偿
    4. 温湿度补偿: 响应对 (T, RH) 线性回归残差项
    5. 漂移: 间歇重测零气, baseline 偏移 > 阈值 → needs_recalibration
    6. 门禁: 残差 RMSE / R² / 补偿后串扰泄漏 三项全过才放行

**provenance（诚实边界）**
    标定数据产生器为 synthetic（供协议与门禁的可执行验证）; 真实气体种类/
    浓度点/传感器型号属 **待领域验证**（由硬件工程师按实际 e-nose 填入）。
    本模块不宣称任何真实标定精度 —— 它守护的是**协议可执行 + 门禁能失败**。

**不做的（诚实边界）**
    - 不做真实气体钢瓶/流量控制器驱动（硬件在环, Phase P）
    - 不做温度循环老化实验（属可靠性测试, 不属标定）
    - 不做学习式基线预测（线性漂移监控足够）

**反证要求（测试守护）**
    已知模型数据能被恢复 (斜率/截距误差 < 5%); 交叉补偿后串扰显著下降;
    漂移超阈值必触发 needs_recalibration; 拟合残差超限必须 FAIL 门禁
    (不是恒过); 未标定数据直接 apply 必须报错。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chem_bio import ENOSE_CHANNELS


# ══════════════════════════════════════════════════════════════════
# 标定数据结构
# ══════════════════════════════════════════════════════════════════

@dataclass
class CalibrationPoint:
    """一个标定点: 已知浓度向量 (各气体 ppm) + 传感器响应向量。"""
    concentrations: np.ndarray        # (n_gases,) 已知浓度 ppm
    responses: np.ndarray             # (ENOSE_CHANNELS,) 阵列响应
    temperature_c: float = 22.0
    humidity: float = 0.3

    def __post_init__(self):
        self.concentrations = np.asarray(self.concentrations,
                                         dtype=np.float64).ravel()
        self.responses = np.asarray(self.responses,
                                    dtype=np.float64).ravel()
        if self.responses.size != ENOSE_CHANNELS:
            raise ValueError(f"responses 应为 {ENOSE_CHANNELS} 维, "
                             f"得 {self.responses.size}")


@dataclass
class ChannelCurve:
    """单通道标定曲线: response ≈ slope·concentration + intercept。"""
    slope: float
    intercept: float
    r2: float                         # 拟合优度
    residual_rmse: float

    def predict(self, concentration: float) -> float:
        return self.slope * concentration + self.intercept

    def invert(self, response: float) -> float:
        """响应 → 浓度估计。"""
        return (response - self.intercept) / self.slope if self.slope != 0 \
            else float("nan")


@dataclass
class CalibrationReport:
    """标定验收报告 — 门禁三项: RMSE / R² / 串扰泄漏。"""
    curves: Dict[str, ChannelCurve]        # "gas_ch" → 曲线
    cross_matrix: np.ndarray               # (n_channels, n_gases) 交叉敏感矩阵
    compensated_leakage: float             # 补偿后串扰泄漏 (0-1, 越小越好)
    gate_rmse_max: float
    gate_r2_min: float
    gate_leakage_max: float
    passed: bool = False
    failures: List[str] = field(default_factory=list)

    def evaluate(self) -> "CalibrationReport":
        self.failures = []
        for name, c in self.curves.items():
            if c.residual_rmse > self.gate_rmse_max:
                self.failures.append(
                    f"{name}: RMSE {c.residual_rmse:.4f} > {self.gate_rmse_max}")
            if c.r2 < self.gate_r2_min:
                self.failures.append(
                    f"{name}: R² {c.r2:.4f} < {self.gate_r2_min}")
        if self.compensated_leakage > self.gate_leakage_max:
            self.failures.append(
                f"串扰泄漏 {self.compensated_leakage:.4f} > "
                f"{self.gate_leakage_max}")
        self.passed = not self.failures
        return self

    def summary(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "n_curves": len(self.curves),
            "mean_r2": float(np.mean([c.r2 for c in self.curves.values()]))
            if self.curves else 0.0,
            "compensated_leakage": self.compensated_leakage,
        }


# ══════════════════════════════════════════════════════════════════
# EnoseCalibrator — 多点标定 + 交叉补偿
# ══════════════════════════════════════════════════════════════════

class EnoseCalibrator:
    """e-nose 多点标定器: 标定曲线 + 交叉敏感解耦 + 验收门禁。

    模型: responses = A · concentrations + b   (A: 交叉敏感矩阵)
    标定: 多浓度点最小二乘 → 每通道曲线 + A 估计
    补偿: 浓度估计 = pinv(A) · (response - b) —— 把交叉响应解耦到各气体。
    """

    #: 验收门禁 (assumed — 由硬件工程师按真实 e-nose 数据调整, 待领域验证)
    GATE_RMSE_MAX = 0.15
    GATE_R2_MIN = 0.90
    GATE_LEAKAGE_MAX = 0.25

    def __init__(self):
        self.curves: Dict[str, ChannelCurve] = {}
        self.A: Optional[np.ndarray] = None          # (n_channels, n_gases)
        self.b: Optional[np.ndarray] = None          # (n_channels,)
        self._fitted = False

    def fit(self, points: List[CalibrationPoint]) -> CalibrationReport:
        """多点标定 → 曲线 + 交叉矩阵 + 门禁报告。

        Args:
            points: ≥2 个标定点 (不同浓度组合), 覆盖目标与干扰气体。
        """
        if len(points) < 2:
            raise ValueError(f"标定至少需要 2 个浓度点, 得 {len(points)}")
        n_params = points[0].concentrations.size + 1    # 气体数 + 截距
        if len(points) <= n_params:
            raise ValueError(
                f"标定点 {len(points)} ≤ 参数数 {n_params} — 欠定/恰定系统会精确"
                f"插值 (残差恒 0, 门禁恒过 = 自证当验收); 需 > {n_params} 个点")
        conc = np.stack([p.concentrations for p in points])   # (n_pts, n_gases)
        resp = np.stack([p.responses for p in points])        # (n_pts, n_ch)
        n_pts, n_gases = conc.shape
        # 全局线性解: resp ≈ conc @ A.T + b  →  [conc, 1] @ [A.T; b] = resp
        design = np.concatenate([conc, np.ones((n_pts, 1))], axis=1)
        coef, *_ = np.linalg.lstsq(design, resp, rcond=None)  # (n_gases+1, n_ch)
        self.A = coef[:n_gases, :].T.copy()                   # (n_ch, n_gases)
        self.b = coef[n_gases, :].copy()                      # (n_ch,)
        self._fitted = True

        # 每通道曲线 = 对**主气体**（该通道响应贡献最大的气体）的标定
        pred = design @ coef
        residual = resp - pred
        curves: Dict[str, ChannelCurve] = {}
        for ch in range(ENOSE_CHANNELS):
            main_gas = int(np.argmax(np.abs(self.A[ch])))
            x = conc[:, main_gas]
            y = resp[:, ch]
            slope = float(self.A[ch, main_gas])
            intercept = float(self.b[ch])
            ss_res = float(np.sum(residual[:, ch] ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            curves[f"ch{ch}_gas{main_gas}"] = ChannelCurve(
                slope=slope, intercept=intercept, r2=float(max(0.0, r2)),
                residual_rmse=float(np.sqrt(ss_res / n_pts)))
        self.curves = curves

        # 串扰泄漏: 补偿后, 单气体注入时其它气体浓度估计的能量占比
        leakage = self._leakage(conc, resp)
        report = CalibrationReport(
            curves=curves, cross_matrix=self.A.copy(),
            compensated_leakage=leakage,
            gate_rmse_max=self.GATE_RMSE_MAX, gate_r2_min=self.GATE_R2_MIN,
            gate_leakage_max=self.GATE_LEAKAGE_MAX,
        )
        return report.evaluate()

    def _leakage(self, conc: np.ndarray, resp: np.ndarray) -> float:
        """补偿后串扰泄漏: 估计浓度矩阵与真值矩阵的非对角/对角能量比。"""
        est = self.compensate(resp)                     # (n_pts, n_gases)
        cross = est - conc
        num = float(np.sum(cross ** 2))
        den = float(np.sum(conc ** 2)) + 1e-12
        return num / den

    def compensate(self, responses: np.ndarray) -> np.ndarray:
        """交叉敏感补偿: 响应向量 → 气体浓度估计 (解耦)。

        Raises:
            RuntimeError: 未标定就补偿 — 防止静默用错模型。
        """
        if not self._fitted:
            raise RuntimeError("未标定 — 先 fit() 再 compensate()")
        r = np.atleast_2d(np.asarray(responses, dtype=np.float64))
        est = (r - self.b) @ np.linalg.pinv(self.A).T
        return est if responses.ndim > 1 else est[0]


# ══════════════════════════════════════════════════════════════════
# DriftMonitor — 基线漂移监控
# ══════════════════════════════════════════════════════════════════

class DriftMonitor:
    """零气基线漂移监控 — 老化/污染 → needs_recalibration 门禁。"""

    #: 漂移阈值 (assumed — 基线偏移相对标定基线的比例)
    DRIFT_THRESHOLD = 0.20

    def __init__(self, baseline: np.ndarray):
        self.baseline = np.asarray(baseline, dtype=np.float64).ravel()
        if self.baseline.size != ENOSE_CHANNELS:
            raise ValueError(f"baseline 应为 {ENOSE_CHANNELS} 维")

    def drift(self, zero_reading: np.ndarray) -> float:
        """零气重测 → 归一化漂移 (0-1+)。"""
        z = np.asarray(zero_reading, dtype=np.float64).ravel()
        if z.size != ENOSE_CHANNELS:
            raise ValueError(f"zero_reading 应为 {ENOSE_CHANNELS} 维")
        denom = np.abs(self.baseline) + 1e-9
        return float(np.max(np.abs(z - self.baseline) / denom))

    def needs_recalibration(self, zero_reading: np.ndarray) -> bool:
        return self.drift(zero_reading) > self.DRIFT_THRESHOLD


# ══════════════════════════════════════════════════════════════════
# 标定数据生成器 (synthetic — 协议验证用)
# ══════════════════════════════════════════════════════════════════

def make_synthetic_calibration_set(n_points: int = 6, n_gases: int = 3,
                                   noise_std: float = 0.02, seed: int = 0
                                   ) -> Tuple[List[CalibrationPoint],
                                              np.ndarray]:
    """合成标定数据集 (provenance=synthetic)。

    真实模型: responses = A_true · conc + b_true + noise。
    Returns: (标定点列表, A_true) —— A_true 供测试恢复性断言。
    """
    rng = np.random.default_rng(seed)
    A_true = rng.uniform(0.2, 1.0, size=(ENOSE_CHANNELS, n_gases))
    b_true = rng.uniform(0.1, 0.5, size=ENOSE_CHANNELS)
    points = []
    for i in range(n_points):
        conc = rng.uniform(0.0, 100.0, size=n_gases)
        resp = A_true @ conc + b_true + rng.normal(0, noise_std,
                                                   size=ENOSE_CHANNELS)
        points.append(CalibrationPoint(concentrations=conc, responses=resp))
    return points, A_true


def calibration_demo() -> dict:
    """标定演示: 合成数据 → 拟合 → 门禁 → 补偿 → 漂移监控。

    全部布尔结论为**实测计算值** (不硬编码), verify 断言即这些值。
    """
    points, A_true = make_synthetic_calibration_set()
    cal = EnoseCalibrator()
    report = cal.fit(points)
    # 恢复性: A 估计 vs A_true
    a_err = float(np.max(np.abs(cal.A - A_true)))
    # 补偿有效性: 单气体注入, 补偿后非目标气体估计应接近 0
    test_conc = np.array([50.0, 0.0, 0.0])
    test_resp = A_true @ test_conc + np.full(ENOSE_CHANNELS, 0.3)
    est = cal.compensate(test_resp)
    off_target = float(np.abs(est[1:]).max())
    # 漂移: 零气基线 + 30% 偏移读数 → 必须触发再标定
    baseline = np.full(ENOSE_CHANNELS, 0.3)
    monitor = DriftMonitor(baseline)
    drifted = baseline * 1.30
    need_recal = monitor.needs_recalibration(drifted)
    result = {
        "gate_passed": report.passed,
        "gate_failures": report.failures,
        "a_recovery_max_err": round(a_err, 4),
        "off_target_estimate": round(off_target, 4),
        "compensated_leakage": round(report.compensated_leakage, 4),
        "drift_detected": need_recal,
        "mean_r2": round(report.summary()["mean_r2"], 4),
    }
    assert report.passed, f"标定门禁未过: {report.failures}"
    assert a_err < 0.15, f"交叉矩阵恢复误差过大: {a_err}"
    assert off_target < 20.0, f"交叉补偿后串扰仍大: {off_target}"
    assert need_recal, "30% 基线漂移必须触发 needs_recalibration"
    return result


if __name__ == "__main__":
    r = calibration_demo()
    print("e-nose 标定 demo (数据 provenance=synthetic, 协议=docs/19 §6)")
    print(f"门禁通过: {r['gate_passed']} | mean R²={r['mean_r2']}")
    print(f"交叉矩阵恢复误差: {r['a_recovery_max_err']}")
    print(f"补偿后非目标估计: {r['off_target_estimate']} ppm")
    print(f"漂移检出: {r['drift_detected']} ✓")
