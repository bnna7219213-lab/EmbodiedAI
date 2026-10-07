"""任务 B 双轨缺口测试 — G10 电机模型 / G7 频谱判别 / G2 FOC↔板级对齐。

每个缺口配**反证**: 证明守护真的在守, 而非"永远通过"。
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from embodied_ai.adapters.edalab import (EdalabAdapter, available_backends,
                                          resolve_backend)
from embodied_ai.hardware.foc_board_align import (cross_check, parse_board_facts)
from embodied_ai.hardware.motor_model import (MotorSpec, motor_spec_from_sources,
                                              verify_motor_spec)
from embodied_ai.hil.twin_fft import (band_ratio, bearing_fault_freqs,
                                      detect_faults, power_spectrum,
                                      synth_vibration, threshold_baseline)


# ══════════════════════════════════════════════════════════════════
# G10: 电机模型
# ══════════════════════════════════════════════════════════════════

class TestG10MotorModel(unittest.TestCase):

    def test_spec_from_real_sources(self):
        """参数来自真实来源 (FOCConfig + 契约), 非编造。"""
        s = motor_spec_from_sources()
        self.assertGreater(s.Rs_ohm, 0)
        self.assertGreater(s.psi_m_wb, 0)
        self.assertTrue(s.provenance.get("sources"), "缺溯源信息")
        self.assertTrue(any("FOCConfig" in x for x in s.provenance["sources"]))

    def test_ke_kt_physics_relation(self):
        """Ke = pp·psi, Kt = 1.5·pp·psi — 物理关系必须精确成立。"""
        s = MotorSpec(Rs_ohm=0.35, Ld_h=2.2e-4, Lq_h=2.2e-4, psi_m_wb=0.0125, pole_pairs=7)
        self.assertAlmostEqual(s.Ke_vs_per_rad, 7 * 0.0125, places=12)
        self.assertAlmostEqual(s.Kt_nm_per_a, 1.5 * 7 * 0.0125, places=12)

    def test_verify_passes(self):
        r = verify_motor_spec(motor_spec_from_sources())
        self.assertTrue(r["ok"], f"失败项: {r['failed']}")

    def test_refute_wrong_ke_is_caught(self):
        """反证: 故意写错 Ke → 校验必须 FAIL (否则校验形同虚设)。"""
        s = MotorSpec(Rs_ohm=0.35, Ld_h=2.2e-4, Lq_h=2.2e-4, psi_m_wb=0.0125,
                      pole_pairs=7, Ke_vs_per_rad=999.0)   # 故意错
        r = verify_motor_spec(s)
        self.assertFalse(r["ok"])
        self.assertIn("physics_Ke_polepairs_psi", r["failed"])

    def test_current_loop_gains_pole_zero_cancel(self):
        s = motor_spec_from_sources()
        g = s.current_loop_gains()
        self.assertAlmostEqual(g["kp_d"], s.Ld_h * g["omega_c_rad_s"], places=12)
        self.assertTrue(g["discrete_ok"], "ωc·T 必须 < 2 (离散稳定)")

    def test_backend_registered_and_available(self):
        self.assertIn("python", available_backends("motor_model"))

    def test_run_capability_reports_backend(self):
        r = EdalabAdapter().run_capability("motor_model", backend="python")
        self.assertEqual(r["backend"], "python")
        self.assertTrue(r["ok"])

    def test_forced_unavailable_backend_raises(self):
        """反证: 强制 jar 后端 (jar 无此能力) 必须抛错, 不静默回落。"""
        with self.assertRaises(ValueError):
            resolve_backend("motor_model", "jar")


# ══════════════════════════════════════════════════════════════════
# G7: 频谱判别
# ══════════════════════════════════════════════════════════════════

class TestG7TwinFFT(unittest.TestCase):

    def test_spectrum_finds_known_frequency(self):
        """注入 100Hz → 谱峰应在 100Hz 附近 (可证伪)。"""
        x = synth_vibration(fs_hz=1000.0, base_freq_hz=100.0, fault_amp=0.0, noise=1e-4)
        sp = power_spectrum(x, 1000.0)
        self.assertAlmostEqual(sp.peak_freq_hz, 100.0, delta=2.0)

    def test_bearing_freqs_formula(self):
        ff = bearing_fault_freqs(rpm=1500.0, n_balls=8, ball_dia_mm=5.0, pitch_dia_mm=22.0)
        fr = 1500.0 / 60.0
        self.assertAlmostEqual(ff["fr_hz"], fr, places=9)
        self.assertGreater(ff["BPFO_hz"], 0)
        self.assertLess(ff["BPFO_hz"], ff["BPFI_hz"])   # 外圈 < 内圈 (标准结论)

    def test_healthy_no_false_alarm(self):
        """反证 1: 健康信号不得报故障 (防恒真判别器)。"""
        x = synth_vibration(fs_hz=1000.0, fault_amp=0.0, seed=7)
        ff = bearing_fault_freqs(1500.0, 8, 5.0, 22.0)
        r = detect_faults(x, 1000.0, fault_freqs=ff)
        self.assertFalse(r["faulty"], f"健康信号误报: {[h for h in r['hits'] if h['hit']]}")

    def test_injected_fault_detected(self):
        """反证 2: 注入 BPFO 特征 → 必须识别 (防"永远不报")。"""
        ff = bearing_fault_freqs(1500.0, 8, 5.0, 22.0)
        x = synth_vibration(fs_hz=1000.0, fault_freq_hz=ff["BPFO_hz"],
                            fault_amp=0.8, seed=7)
        r = detect_faults(x, 1000.0, fault_freqs=ff)
        self.assertTrue(r["faulty"], "注入故障未被识别")
        self.assertIn("bearing_outer_race", [h["fault"] for h in r["hits"] if h["hit"]])

    def test_threshold_baseline_misses_early_fault(self):
        """阈值法 (jar 现有做法) 对早期故障漏检 — G7 的价值所在。"""
        ff = bearing_fault_freqs(1500.0, 8, 5.0, 22.0)
        x = synth_vibration(fs_hz=1000.0, fault_freq_hz=ff["BPFO_hz"],
                            fault_amp=0.8, seed=7)
        self.assertFalse(threshold_baseline(x, limit=3.0)["alarm"],
                         "该工况下阈值法应漏检 (时域幅值未超限)")
        self.assertTrue(detect_faults(x, 1000.0, fault_freqs=ff)["faulty"],
                        "频域应能识别")

    def test_deterministic_not_random(self):
        """确定性: 同 seed 两次结果一致 (非随机数冒充)。"""
        ff = bearing_fault_freqs(1500.0, 8, 5.0, 22.0)
        a = detect_faults(synth_vibration(seed=3, fault_freq_hz=ff["BPFO_hz"],
                                          fault_amp=0.8), 1000.0, fault_freqs=ff)
        b = detect_faults(synth_vibration(seed=3, fault_freq_hz=ff["BPFO_hz"],
                                          fault_amp=0.8), 1000.0, fault_freqs=ff)
        self.assertEqual(a["n_hits"], b["n_hits"])
        self.assertEqual([h["band_ratio"] for h in a["hits"]],
                         [h["band_ratio"] for h in b["hits"]])

    def test_backend_registered(self):
        self.assertIn("python", available_backends("digital_twin"))


# ══════════════════════════════════════════════════════════════════
# G2: FOC ↔ 板级对齐
# ══════════════════════════════════════════════════════════════════

REAL_BOARD_STDOUT = """=== SynapticForge 电机驱动板设计 (MotorDriveEngine POC) ===
规格名: MotorDrive, 桥臂 48.0V / 30.0A / 1440.0W
设计名称: SF-MDB-48V-30A-FOC_SENSORED
PCB尺寸: 69.28203mm × 55.425625mm
层数: 6
铜厚: 2.0oz
最高结温: 67.000015°C (裕量 82.999985°C)
需要散热片: false

关键器件:
  U1: STM32G474VET6 (LQFP100)
  U2: DRV8301 (60V, 2.3A gate drive) (QFN48)
  Q1-Q6: IRFS4410Z (100V/80A) (TO-220-7P)

固件特性: adc_current_sense, pwm_mcpwm, canfd_comm, foc_control, clarke_transform, park_transform, encoder_feedback, overcurrent_protection, thermal_protection, auto_shutdown_us1
"""


class TestG2BoardAlign(unittest.TestCase):

    def test_parse_real_stdout(self):
        """解析器吃的是 jar 真实输出格式 (非臆造)。"""
        b = parse_board_facts(REAL_BOARD_STDOUT)
        self.assertEqual(b.bus_voltage_v, 48.0)
        self.assertEqual(b.peak_current_a, 30.0)
        self.assertEqual(b.layers, 6)
        self.assertEqual(b.copper_oz, 2.0)
        self.assertIn("IRFS4410Z", b.mosfet)
        self.assertIn("foc_control", b.firmware_features)

    def test_cross_check_passes(self):
        from embodied_ai.hardware.foc_shutdown import FOCConfig
        b = parse_board_facts(REAL_BOARD_STDOUT)
        r = cross_check(b, motor_spec_from_sources(), FOCConfig())
        self.assertTrue(r["ok"], f"不匹配: {r['mismatches']}")

    def test_refute_voltage_mismatch_caught(self):
        """反证: 板级电压与 Python 侧不一致 → 必须进 mismatches。"""
        from embodied_ai.hardware.foc_shutdown import FOCConfig
        bad = REAL_BOARD_STDOUT.replace("48.0V / 30.0A", "24.0V / 30.0A")
        b = parse_board_facts(bad)
        r = cross_check(b, motor_spec_from_sources(), FOCConfig())
        self.assertFalse(r["ok"])
        self.assertIn("voltage_rating_aligned", [m["check"] for m in r["mismatches"]])

    def test_refute_current_mismatch_caught(self):
        from embodied_ai.hardware.foc_shutdown import FOCConfig
        bad = REAL_BOARD_STDOUT.replace("48.0V / 30.0A", "48.0V / 15.0A")
        r = cross_check(parse_board_facts(bad), motor_spec_from_sources(), FOCConfig())
        self.assertFalse(r["ok"])
        self.assertIn("current_rating_aligned", [m["check"] for m in r["mismatches"]])

    def test_foc_gains_self_consistent_after_fix(self):
        """G2 修正后: FOCConfig 增益与极点配置精确自洽 (原 0.014% 漂移)。"""
        from embodied_ai.hardware.foc_shutdown import FOCConfig
        fc = FOCConfig()
        s = motor_spec_from_sources()
        g = s.current_loop_gains()
        self.assertAlmostEqual(fc.kp_d, g["kp_d"], places=6)
        self.assertAlmostEqual(fc.ki_d, g["ki_d"], places=4)

    def test_backend_uses_real_jar_output(self):
        """集成: 后端默认 live 调 jar 取板级事实。"""
        r = EdalabAdapter().run_capability("foc_board_align", backend="python")
        self.assertTrue(r["ok"], f"不匹配: {r.get('mismatches')}")
        self.assertEqual(r["board"]["bus_voltage_v"], 48.0)
        self.assertIn("jar 输出板级规格但不含 kp/ki", r["note"])


class TestBackendRegistrationNoEmptyTable(unittest.TestCase):
    """回归: "只导入 adapters 看不到后端"的空表陷阱 (与 cli_registry 同类)。"""

    def test_all_backends_visible_after_adapter_import(self):
        for cap in ("hil", "motor_model", "digital_twin", "foc_board_align"):
            self.assertIn("python", available_backends(cap),
                          f"{cap} 后端不可见 (懒注册失效)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
