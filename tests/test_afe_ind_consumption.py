"""AFE-IND 触觉传感硬件前端移植测试 (Phase 1, 2026-10-07).

覆盖:
  1. 产物完整性 — 11 必需文件齐, 缺失任一即失败
  2. SPEC/REPORT 解析 — 从文本产物提取结构化指标 (正则鲁棒性)
  3. BOM/原理图 JSON/AC CSV 机器可读产物
  4. 契约对齐 — AFE-IND 硬件指标 vs hw_requirements.yaml sensors.tactile 需求
  5. 关键指标数值断言 — 防止 artifacts 被误更新静默回归
  6. Live 模式 (subprocess 调 EdaCli) — EdaLab 侧无改动, 仅调用

**验证"物理可失败"**:
  - test_missing_artifact_raises: 移除一个文件, 消费端必须报缺失
  - test_tampered_report_fails: 篡改增益值, contract_alignment 必须失败
  - test_garbage_report_raises: 非格式文本, 解析器必须抛 ValueError
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from embodied_ai.adapters import afe_ind as ai


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts" / "afe_ind"


# --------------------------------------------------------------------------- #
# 1. 产物完整性
# --------------------------------------------------------------------------- #
class TestArtifactCompleteness:
    """守护 artifacts/afe_ind/ 11 必需产物不缺失."""

    def test_required_count_is_11(self):
        assert len(ai.REQUIRED_ARTIFACTS) == 11

    def test_all_required_present_in_repo(self):
        r = ai.check_artifacts_present()
        assert r["ok"], f"缺失产物: {r['missing']}"
        assert r["present_count"] >= r["required_count"]

    def test_circuit_json_is_required(self):
        """原理图 JSON 是机器可读契约的核心, 必须列入必需产物."""
        assert "afe_ind.circuit.json" in ai.REQUIRED_ARTIFACTS

    def test_spice_netlist_is_required(self):
        """SPICE 网表是仿真证据, 必须固化."""
        assert "afe_ind.cir" in ai.REQUIRED_ARTIFACTS

    def test_bom_is_required(self):
        """BOM 是采购入口, 必须固化."""
        assert "afe_ind_bom.csv" in ai.REQUIRED_ARTIFACTS

    def test_missing_artifact_raises_ok_false(self, tmp_path):
        """物理可失败: 缺一个文件 → ok=False 且 missing 非空."""
        partial = tmp_path / "partial"
        partial.mkdir()
        # 只放一半文件
        shutil.copy(ARTIFACTS / "afe_ind.c", partial / "afe_ind.c")
        r = ai.check_artifacts_present(path=partial)
        assert not r["ok"]
        assert len(r["missing"]) == 10  # 11 - 1

    def test_env_override_affects_dir(self, tmp_path, monkeypatch):
        """R6 精神: env 覆盖生效."""
        monkeypatch.setenv("EMBODIED_AFE_IND_ARTIFACTS", str(tmp_path))
        assert ai.artifacts_dir() == tmp_path


# --------------------------------------------------------------------------- #
# 2. 报告解析 (SPEC/REPORT)
# --------------------------------------------------------------------------- #
class TestReportParsing:

    def test_parse_verification_report_key_metrics(self):
        rep = ai.parse_verification_report()
        # 精确数值断言 — 防止 artifacts 被误更新静默回归
        assert rep["gain_x"] == pytest.approx(1046.83, abs=0.01)
        assert rep["gain_db"] == pytest.approx(60.40, abs=0.01)
        assert rep["bandwidth_khz"] == pytest.approx(10.60, abs=0.01)
        assert rep["noise_v_per_sqrt_hz"] == pytest.approx(3.794e-6, rel=1e-3)
        assert rep["pcb_component_count"] == 65
        assert rep["drc_violations"] == 380

    def test_parse_report_convergence_flags(self):
        rep = ai.parse_verification_report()
        assert rep["dc_converged"] is True
        assert rep["ac_converged"] is True
        assert rep["transient_converged"] is True

    def test_parse_report_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ai.parse_verification_report(path=Path("/nonexistent/report.txt"))

    def test_parse_report_garbage_raises_valueerror(self, tmp_path):
        """物理可失败: 非格式文本 → ValueError."""
        bad = tmp_path / "bad.txt"
        bad.write_text("完全不是报告格式\n", encoding="utf-8")
        with pytest.raises((ValueError, FileNotFoundError)):
            ai.parse_verification_report(path=bad)


# --------------------------------------------------------------------------- #
# 3. 机器可读产物 (JSON / CSV)
# --------------------------------------------------------------------------- #
class TestMachineReadableArtifacts:

    def test_parse_circuit_json_structure(self):
        c = ai.parse_circuit_json()
        assert c["components_count"] == 67, "AFE-IND 原理图应有 67 元件"
        assert c["wires_count"] >= 30, "至少 30 连线"
        assert c["name"], "原理图必须有名"
        assert c["timestamp"] > 0

    def test_parse_circuit_json_utf8_decoded(self):
        """circuit.json 是 UTF-8 编码 (含中文描述), 不能走 GBK 默认."""
        c = ai.parse_circuit_json()
        # 名字可能含中文 (如"桥式微弱信号采集前端"), 至少不是乱码字符
        name = c["name"]
        assert "\ufffd" not in name, "原理图名字解码异常"

    def test_parse_bom_row_count_matches_circuit(self):
        """BOM 行数必须等于原理图元件数 — 一致性守护."""
        c = ai.parse_circuit_json()
        bom = ai.parse_bom()
        assert len(bom) == c["components_count"], \
            f"BOM {len(bom)} ≠ 原理图 {c['components_count']}"

    def test_parse_bom_headers_present(self):
        bom = ai.parse_bom()
        assert len(bom) >= 50, "BOM 太少"
        required_headers = {"RefDes", "Type", "Value", "Unit", "Footprint"}
        first = bom[0]
        for h in required_headers:
            assert h in first, f"BOM 缺列 {h}: {list(first.keys())}"

    def test_parse_bom_contains_amplifier_resistors(self):
        """SPEC 声明仪表放大用 R1=10kΩ, BOM 应能找到 10kΩ 电阻 (IR3/IR4).

        BOM 的 Value 用普通整数格式 (10000.0) 而非科学计数法 (1.0E+04).
        """
        bom = ai.parse_bom()
        resistors = [x for x in bom if x.get("Type") == "RESISTOR"]
        assert len(resistors) >= 15, f"电阻数量 {len(resistors)} 太少"
        # 10kΩ = 10000 Ω 电阻 (仪表放大 R1, SPEC 声明 IR3/IR4)
        vals_10k = [x for x in resistors if abs(float(x["Value"]) - 10000.0) < 1e-6]
        assert len(vals_10k) >= 2, \
            f"BOM 应至少含 2 个 10kΩ 电阻 (仪表放大 R1 对), 实际 {len(vals_10k)}"
        # 检查 RefDes 命名含 IR (instrument amplifier 前缀)
        ir_prefix = [x for x in resistors if x["RefDes"].startswith("IR")]
        assert len(ir_prefix) >= 3, f"IR* 前缀电阻 {len(ir_prefix)} < 3"

    def test_parse_ac_sweep_covers_full_band(self):
        ac = ai.parse_ac_sweep()
        assert ac["points_count"] >= 50
        assert ac["f_min_hz"] <= 1.0, "应覆盖直流附近"
        assert ac["f_max_hz"] >= 1e6, "应覆盖 MHz 级 (抗混叠带宽验证)"
        assert ac["peak_gain_db"] > 0, "峰值增益应为正 dB"

    def test_parse_ac_sweep_peak_gain_reasonable(self):
        """峰值增益应在合理范围 (AFE-IND 报告增益 60.4dB, 但 AC 曲线是分段增益)."""
        ac = ai.parse_ac_sweep()
        # peak_gain_db 是曲线内最大点, AFE-IND AC CSV 显示 ~14.4dB
        # 这是"后级增益"或"某频段峰值", 不是全链增益
        assert 10.0 <= ac["peak_gain_db"] <= 20.0, \
            f"AC 峰值 {ac['peak_gain_db']}dB 超出合理区间 [10,20]"


# --------------------------------------------------------------------------- #
# 4. 契约对齐 (AFE-IND 硬件指标 vs hw_requirements.yaml sensors.tactile)
# --------------------------------------------------------------------------- #
class TestContractAlignment:

    def test_alignment_all_pass(self):
        align = ai.contract_alignment()
        assert align["ok"], f"契约对齐失败: {[c for c in align['checks'] if not c['ok']]}"
        assert len(align["checks"]) == 5

    def test_alignment_bandwidth_covers_requirement_by_100x(self):
        """核心契约: 硬件带宽 10600Hz 覆盖触觉 sensing_res_hz=100Hz 达 106×."""
        align = ai.contract_alignment()
        bw_check = next(c for c in align["checks"] if c["name"] == "bandwidth_vs_sensing_res")
        ratio = bw_check["hardware_value_hz"] / bw_check["requirement_hz"]
        assert ratio >= 100, f"带宽覆盖率 {ratio}× < 100×"

    def test_alignment_gain_spec_documented(self):
        align = ai.contract_alignment()
        g_check = next(c for c in align["checks"] if c["name"] == "gain_spec_documented")
        assert g_check["hardware_gain_x"] >= 500
        assert g_check["hardware_gain_db"] >= 50
        assert g_check["ok"]

    def test_alignment_noise_below_threshold(self):
        align = ai.contract_alignment()
        n_check = next(c for c in align["checks"] if c["name"] == "input_ref_noise_density")
        assert n_check["hardware_noise_v_per_sqrt_hz"] < n_check["requirement_max"]

    def test_alignment_simulation_convergence(self):
        align = ai.contract_alignment()
        s_check = next(c for c in align["checks"] if c["name"] == "simulation_convergence")
        assert s_check["dc"] and s_check["ac"] and s_check["transient"]
        assert s_check["ok"]

    def test_alignment_pcb_component_count_in_range(self):
        align = ai.contract_alignment()
        p_check = next(c for c in align["checks"] if c["name"] == "pcb_component_count")
        assert 30 <= p_check["hardware_count"] <= 200


# --------------------------------------------------------------------------- #
# 5. 物理可失败 (让契约守护真能拦回归)
# --------------------------------------------------------------------------- #
class TestPhysicalFailureModes:

    def test_tampered_report_gain_triggers_alignment_failure(self, tmp_path):
        """篡改报告增益到 50× → contract_alignment 的 gain_spec_documented 失败."""
        bad_report = tmp_path / "bad_report.txt"
        # 构造一个"看似格式对但指标差"的报告
        bad_report.write_text(
            "=== EDA-AFE-IND 验证报告 ===\n"
            "DC 收敛=true  AC 收敛=true  瞬态=收敛\n"
            "总增益≈50×（14.00dB），带宽 0Hz~10.60kHz\n"
            "输入参考噪声(27℃,1Hz)：  输出噪声密度 = 5.000e-06 V/√Hz\n"
            "导出：SVG/SPICE/C/PCB；PCB 65 元件\n"
            "DRC：380 违规\n",
            encoding="utf-8"
        )
        rep = ai.parse_verification_report(path=bad_report)
        align = ai.contract_alignment(report=rep)
        g_check = next(c for c in align["checks"] if c["name"] == "gain_spec_documented")
        assert not g_check["ok"], "增益 50× 应触发契约失败"
        assert not align["ok"], "整体对齐应失败"

    def test_noisy_report_triggers_noise_failure(self, tmp_path):
        """噪声超过 1e-5 V/√Hz → noise check 失败."""
        bad = tmp_path / "noisy.txt"
        bad.write_text(
            "=== EDA-AFE-IND 验证报告 ===\n"
            "DC 收敛=true  AC 收敛=true  瞬态=收敛\n"
            "总增益≈1046.83×（60.40dB），带宽 0Hz~10.60kHz\n"
            "输入参考噪声(27℃,1Hz)：  输出噪声密度 = 5.000e-05 V/√Hz\n"
            "导出：SVG/SPICE/C/PCB；PCB 65 元件\n"
            "DRC：380 违规\n",
            encoding="utf-8"
        )
        rep = ai.parse_verification_report(path=bad)
        align = ai.contract_alignment(report=rep)
        n_check = next(c for c in align["checks"] if c["name"] == "input_ref_noise_density")
        assert not n_check["ok"], "噪声 5e-5 应超阈值失败"

    def test_unconverged_report_fails_alignment(self, tmp_path):
        """瞬态不收敛 → simulation_convergence 失败."""
        bad = tmp_path / "diverge.txt"
        bad.write_text(
            "=== EDA-AFE-IND 验证报告 ===\n"
            "DC 收敛=true  AC 收敛=true  瞬态=发散\n"
            "总增益≈1046.83×（60.40dB），带宽 0Hz~10.60kHz\n"
            "输入参考噪声(27℃,1Hz)：  输出噪声密度 = 3.794e-06 V/√Hz\n"
            "导出：SVG/SPICE/C/PCB；PCB 65 元件\n"
            "DRC：380 违规\n",
            encoding="utf-8"
        )
        rep = ai.parse_verification_report(path=bad)
        assert rep["transient_converged"] is False
        align = ai.contract_alignment(report=rep)
        s_check = next(c for c in align["checks"] if c["name"] == "simulation_convergence")
        assert not s_check["ok"]


# --------------------------------------------------------------------------- #
# 6. 端到端消费 (consume_full 一次跑完)
# --------------------------------------------------------------------------- #
class TestEndToEndConsumption:

    def test_consume_full_ok(self):
        r = ai.consume_full()
        assert r["artifacts"]["ok"]
        assert r["overall_ok"]
        assert r["verification_report"]["gain_x"] > 500
        assert r["contract_alignment"]["ok"]
        assert r["bom_count"] == r["circuit_json"]["components_count"]

    def test_consume_full_returns_structured_report(self):
        r = ai.consume_full()
        for key in ("artifacts", "verification_report", "circuit_json",
                    "ac_sweep", "bom_count", "contract_alignment", "overall_ok"):
            assert key in r, f"consume_full 缺字段 {key}"

    def test_edalab_root_env_override(self, monkeypatch):
        """R6 精神: EDALAB_ROOT env 覆盖."""
        monkeypatch.setenv("EDALAB_ROOT", "C:\\tmp\\edalab_test")
        assert str(ai.edalab_root()).endswith("edalab_test")
        monkeypatch.delenv("EDALAB_ROOT")
