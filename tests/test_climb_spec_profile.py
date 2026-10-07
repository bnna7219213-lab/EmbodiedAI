"""test_climb_spec_profile.py — B1: 任务档案参数化 (hw_requirements.yaml → forge-core / motor-drive).

B1 目标: 攀岩腱驱动关节板需求进 EdaLab 引擎 (C:\\Users\\bnna7\\workspace\\EdaLab),
11 步主链增加 spec 维度, 默认 household_grasp 行为不变 (向后兼容),
climbing 导出独立 EPS + FOC 板设计。

四层覆盖:
  1. 契约层: available_specs / build_spec_args — YAML 单一真相源, 数值不重复硬编码
  2. 兼容层: 不传 spec 时 CLI 参数与原 FLOW_CHAIN.cli_args 逐字一致
  3. 主链层: dry 11/11 READY + chain_green 判定语义
  4. 引擎层 (live): climbing spec 真实改变 EPS 与 FOC 板设计 (非标签化) —
     仅在 jar 存在时运行 (本机 EdaLab 构建产物), 否则 skip

真实接口 (src/embodied_ai/adapters/edalab.py, B2 口径):
  EdalabAdapter() 实例方法; run_stage(name, params, mode="dry"|"live", spec=...)
  结果字段: step / stage / status (READY|LIVE_OK|...) / spec / cli_args
  主链: run_full_chain(params, mode, spec) → chain_green(results, expect_steps)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from embodied_ai.adapters.edalab import (  # noqa: E402
    EdalabAdapter, FLOW_CHAIN, CLIMBING_SPEC, HOUSEHOLD_SPEC, EDALAB_JAR,
)
import embodied_ai.adapters.edalab as _edalab_mod  # noqa: E402


def _adapter() -> EdalabAdapter:
    return EdalabAdapter(timeout_sec=90)


def _stage(name: str, mode: str = "dry", spec: str = HOUSEHOLD_SPEC):
    return _adapter().run_stage(name, {}, mode=mode, spec=spec)


# ============================================================================
# 1. 契约层: spec 枚举 + 参数表
# ============================================================================

def test_available_specs_lists_household_and_climbing():
    specs = _edalab_mod.available_specs()
    assert specs[0] == HOUSEHOLD_SPEC == "household_grasp"   # 默认 profile 置顶
    assert CLIMBING_SPEC == "climbing" in specs


def test_available_specs_excludes_structural_yaml_sections():
    """结构性 yaml 节 (compute/motor_drive/sensors/...) 不是任务档案。

    B1 首个版本曾把全部顶层 key 当 spec 返回, 已修正为按 metadata.board_kind 过滤。
    """
    specs = _edalab_mod.available_specs()
    structural = {"metadata", "compute", "motor_drive", "sensors", "power",
                  "communication", "safety", "data_flywheel"}
    assert not (set(specs) & structural)
    assert len(specs) == 2


def test_household_build_spec_args_is_empty_fast_path():
    """household 是虚拟默认档案: 返回空表 → run_stage 回退 FLOW_CHAIN.cli_args。"""
    assert _edalab_mod.build_spec_args(HOUSEHOLD_SPEC) == {}


def test_household_fast_path_needs_no_yaml():
    """household 走早退路径, 不读 yaml → 契约文件缺失时默认行为仍可用。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        missing = Path(td) / "no_such.yaml"
        _orig = _edalab_mod._contracts_dir
        _edalab_mod._contracts_dir = lambda: missing.parent
        try:
            assert _edalab_mod.build_spec_args(HOUSEHOLD_SPEC) == {}
        finally:
            _edalab_mod._contracts_dir = _orig


def test_climbing_build_spec_args_contents():
    """climbing 档案参数表: [version, task, motionHz, TOPS, inferMs, bus,
    busMbps, nodes, bridgePeakA, tactileHz, encoderCh] + motor-drive variant."""
    args = _edalab_mod.build_spec_args(CLIMBING_SPEC)

    fc = args["forge_core"]
    assert fc[0] == "climbing_competition_v0.1"
    assert fc[1] == "climbing_tendon_drive"
    assert fc[2] == "20"        # 运动控制 20Hz (RL 层)
    assert fc[3] == "30"        # yaml 无 TOPS 字段 → FORGE_TOPS_DEFAULT
    assert fc[4] == "15"        # 推理 15ms (严于 household 25ms)
    assert fc[5] == "canfd"
    assert fc[6] == "5"         # 5Mbps
    assert fc[7] == "8"         # 8 伺服节点
    assert fc[8] == "19.2"      # 桥臂峰值 = 8ch × 4A × 0.6 占空
    assert fc[9] == "100"       # 触觉 100Hz
    assert fc[10] == "16"       # 编码器 16 通道 (8 伺服 × 双端)

    assert args["motor_drive"] == ["tendon"]   # 板种变体, 由 board_kind 含 tendon 判定
    assert set(args) == {"forge_core", "motor_drive"}   # 仅两个参数化阶段


def test_climbing_args_match_yaml_single_source_of_truth():
    """参数值必须与 yaml 字段逐项一致 —— 防 Kotlin/Python 侧漂移硬编码。"""
    req = _edalab_mod.load_hw_requirements()
    fc = _edalab_mod.build_spec_args(CLIMBING_SPEC)["forge_core"]

    c = req["climbing"]
    assert c["metadata"]["board_kind"] == "tendon_drive_joint_board"
    assert c["hand"]["tendon_groups_per_hand"] == 2       # group0 拇指 / group1 四指主腱

    md = c["motor_drive_tendon"]
    assert float(fc[8]) == float(md["bridge_peak_current_a"])
    assert int(fc[10]) == int(md["encoder_channels"])
    assert float(fc[4]) == float(req["compute"]["model_inference_ms"])
    assert fc[2] == str(req["compute"]["control_frequencies_hz"]["motion_control"])

    comm = c["communication"]
    assert fc[5] == comm["bus"].replace("_", "")   # yaml can_fd → ForgeCore 枚举 canfd
    assert int(fc[6]) == int(comm["data_bps"]) // 1_000_000
    assert int(fc[7]) == int(comm["node_count"])
    assert int(fc[9]) == int(req["sensors"]["tactile"]["frame_rate_hz"])


def test_flow_chain_shape_unchanged():
    """B1 只加 spec 维度, 不改 11 步链形。"""
    assert len(FLOW_CHAIN) == 11
    assert [s["name"] for s in FLOW_CHAIN][:3] == ["forge_core", "mass_props", "structural_fem"]
    assert FLOW_CHAIN[-1]["name"] == "multi_physics"
    assert [s["cli"] for s in FLOW_CHAIN][0] == "forge-core"   # 下划线 → 连字符映射


def test_build_spec_args_rejects_unknown_spec_gracefully():
    """未知档案 → 空表 (回退默认参数), 不抛异常打断主链。"""
    assert _edalab_mod.build_spec_args("nonexistent_profile") == {}


# ============================================================================
# 2. 兼容层: 不传 spec 时行为与 B1 前完全一致
# ============================================================================

def test_dry_default_uses_flow_chain_cli_args():
    result = _stage("forge_core")
    expected = FLOW_CHAIN[0]["cli_args"]
    assert result["cli_args"] == expected
    assert result["spec"] == HOUSEHOLD_SPEC       # 未传 spec → 记为默认
    assert result["status"] == "READY"
    assert result["stage"] == "forge_core"
    assert result["step"] == 1


def test_dry_household_explicit_equals_default():
    """显式 household_grasp == 隐式默认 (向后兼容的强断言, 忽略时间戳)."""
    def sig(r):
        return {k: v for k, v in r.items() if k not in ("started_at", "finished_at")}

    assert sig(_stage("forge_core", spec=HOUSEHOLD_SPEC)) == sig(_stage("forge_core"))


def test_dry_spec_override_changes_cli_args():
    result = _stage("forge_core", spec=CLIMBING_SPEC)
    override = _edalab_mod.build_spec_args(CLIMBING_SPEC)["forge_core"]
    assert result["cli_args"] == override
    assert result["spec"] == CLIMBING_SPEC
    assert result["cli_args"] != FLOW_CHAIN[0]["cli_args"]    # 确已切换
    assert result["status"] == "READY"


def test_dry_override_only_touches_parameterized_stages():
    """spec 覆盖不得波及其他 9 步 —— 它们用 EdaCli 内置样本输入, 与档案无关。"""
    md = _stage("motor_drive", spec=CLIMBING_SPEC)
    assert md["cli_args"] == ["tendon"]
    assert md["status"] == "READY"

    for name, expected in (("mass_props", []), ("kinematics", []),
                           ("thermal_sim", ["40", "40", "2", "40"]),
                           ("relay_clutch", [])):
        r = _stage(name, spec=CLIMBING_SPEC)
        assert r["cli_args"] == expected, f"污染: {name}"
        assert r["status"] == "READY"


def test_dry_stage_log_accumulates():
    ad = _adapter()
    ad.run_stage("forge_core", {}, mode="dry", spec=CLIMBING_SPEC)
    ad.run_stage("motor_drive", {}, mode="dry", spec=CLIMBING_SPEC)
    assert len(ad.stage_log) == 2          # property, 非方法


def test_dry_unknown_stage_raises():
    with pytest.raises(ValueError):
        _adapter().run_stage("not_a_stage", {}, mode="dry")


def test_dry_invalid_mode_raises():
    with pytest.raises(ValueError):
        _stage("forge_core", mode="dry-run")


# ============================================================================
# 3. 主链层: 11 步 dry + chain_green 判定
# ============================================================================

def test_full_chain_dry_climbing_all_ready():
    results = _adapter().run_full_chain({}, mode="dry", spec=CLIMBING_SPEC)
    assert len(results) == 11
    assert all(r["status"] == "READY" for r in results)
    assert all(r["spec"] == CLIMBING_SPEC for r in results)
    fc = next(r for r in results if r["stage"] == "forge_core")
    md = next(r for r in results if r["stage"] == "motor_drive")
    assert fc["cli_args"][8] == "19.2"
    assert md["cli_args"] == ["tendon"]
    ok, states = EdalabAdapter.chain_green(results)
    assert ok and all("=READY" in s for s in states)


def test_full_chain_dry_household_all_ready():
    results = _adapter().run_full_chain({}, mode="dry", spec=HOUSEHOLD_SPEC)
    assert len(results) == 11
    assert all(r["status"] == "READY" for r in results)


def test_full_chain_dry_default_backwards_compatible():
    """不传 spec 的主链 cli_args 序列必须与 FLOW_CHAIN 抄录值逐字一致。"""
    results = _adapter().run_full_chain({}, mode="dry")
    for r, step in zip(results, FLOW_CHAIN):
        assert r["cli_args"] == step["cli_args"], f"漂移: step {r['step']}"
        assert r["spec"] == HOUSEHOLD_SPEC


def test_chain_green_accepts_ready_and_live_ok():
    for st in ("READY", "LIVE_OK"):
        ok, states = EdalabAdapter.chain_green([{"step": i + 1, "stage": f"s{i}", "status": st}
                                                 for i in range(11)])
        assert ok
        assert len(states) == 11


@pytest.mark.parametrize("bad", ["CONTRACT_FAIL", "LIVE_FAIL", "LIVE_ERROR", "LIVE_TIMEOUT", "FAILED", "ENGINE_MISSING"])
def test_chain_green_rejects_fail_states(bad):
    ok, _ = EdalabAdapter.chain_green(
        [{"step": i + 1, "stage": f"s{i}", "status": "READY"} for i in range(10)]
        + [{"step": 11, "stage": "s10", "status": bad}])
    assert not ok


def test_chain_green_step_count_mismatch_not_green():
    ok, _ = EdalabAdapter.chain_green(
        [{"step": 1, "stage": "s", "status": "READY"}] * 9, expect_steps=11)
    assert not ok


def test_chain_green_empty_fails():
    ok, states = EdalabAdapter.chain_green([], expect_steps=11)
    assert not ok
    assert states == []


def test_chain_green_custom_expect_steps():
    ok, _ = EdalabAdapter.chain_green(
        [{"step": 1, "stage": "s1", "status": "READY"}] * 7, expect_steps=7)
    assert ok


# ============================================================================
# 4. 引擎层 (live): spec 真实改变 EPS 与 FOC 板设计
#    需要本机 EdaLab 构建产物 (EdaLab/edalabcore/build/libs/edalabcore.jar)
# ============================================================================

skip_no_jar = pytest.mark.skipif(
    not EDALAB_JAR.exists(),
    reason=f"本机无 EdaLab jar (需 cd EdaLab && ./gradlew :edalabcore:jar): {EDALAB_JAR}")


@skip_no_jar
def test_live_forge_core_climbing_produces_climbing_eps():
    """climbing 档案必须导出独立 EPS (非 household 标签替换)。

    断言依据 (2026-10-07 live 实测):
      EPS ID: SF-climbing_tendon_drive-climbing_competition_v0_1
      封装: 必需 (RELIABILITY 优先), 爬电距离: 3.0mm, 层数: 4, 传感链路: 7 项
      Components: 17 个 / BOM预估: $10313.002
      风险: 热裕量不足 3.35°C (known_gaps 记录的单桥模型失真)
    """
    r = _adapter().run_stage("forge_core", {}, mode="live", spec=CLIMBING_SPEC)
    assert r["status"] == "LIVE_OK", (r.get("error") or "")[-500:]
    text = r["stdout"]

    assert "策略档案: climbing" in text                     # 档案分流生效
    assert "SF-climbing_tendon_drive-climbing_competition_v0_1" in text   # EPS ID
    assert "封装: 必需" in text                             # RELIABILITY 优先 → IP54
    assert "3.0mm" in text                                  # 30V → 3.0mm (household 1.5mm)
    assert "层数: 4" in text                                # 19.2A < 20 → 4 层 (household 6)
    assert "7 项" in text                                   # 传感链路 7 项 (household 3)
    assert "17 个" in text                                  # BOM 17 元件 (household 12)
    assert "$10313.002" in text                             # BOM 成本差异 ~1.94x
    assert "热裕量不足" in text                             # known_gaps 记录的模型失真
    assert "SF-household" not in text                       # 无 household 串味


@skip_no_jar
def test_live_motor_drive_variant_dispatch():
    """variant='tendon' → 24V/19.2A/4 层 55.4mm; 默认 → 48V/30A/6 层 69.3mm。"""
    r = _adapter().run_stage("motor_drive", {}, mode="live", spec=CLIMBING_SPEC)
    assert r["status"] == "LIVE_OK", (r.get("error") or "")[-500:]
    tendon = r["stdout"]
    assert "板种: 腱驱动关节板" in tendon
    assert "规格名: TendonDrive-Climbing" in tendon
    assert "桥臂 24.0V / 19.2A / 460.8W" in tendon
    assert "SF-MDB-24V-19A-FOC_SENSORED" in tendon
    assert "55.425625mm × 44.3405mm" in tendon
    assert "层数: 4" in tendon and "2.0oz" in tendon
    assert "51.059208°C" in tendon                         # 通道级 4A → 结温低
    assert "需要散热片: false" in tendon

    r2 = _adapter().run_stage("motor_drive", {}, mode="live")
    assert r2["status"] == "LIVE_OK", (r2.get("error") or "")[-500:]
    default = r2["stdout"]
    assert "规格名: MotorDrive" in default
    assert "桥臂 48.0V / 30.0A / 1440.0W" in default
    assert "SF-MDB-48V-30A-FOC_SENSORED" in default
    assert "69.28203mm" in default and "层数: 6" in default
    assert "腱驱动" not in default


@skip_no_jar
def test_live_forge_core_household_unchanged():
    """household 默认参数 live 输出与 B1 前一致 (回归守护)。"""
    r = _adapter().run_stage("forge_core", {}, mode="live")
    assert r["status"] == "LIVE_OK", (r.get("error") or "")[-500:]
    text = r["stdout"]
    assert "SF-household_grasp-v1_0" in text
    assert "封装: 无" in text                       # 家用开放结构
    assert "1.5mm" in text                          # 12V → 1.5mm 爬电
    assert "层数: 6" in text                        # 30A ≥ 20 → 6 层
    assert "传感链路: 3 项" in text
    assert "12 个" in text and "$5318.6" in text    # BOM 12 元件
    assert "PWM" in text and "EMC" in text          # 家用侧风险是 EMC
    assert "climbing" not in text and "腱" not in text


@skip_no_jar
def test_live_specs_produce_distinct_designs():
    """两份 EPS 的 EPS-ID 与 BOM 必须不同 —— 证明参数化不是空转。"""
    def probe(spec):
        r = _adapter().run_stage("forge_core", {}, mode="live", spec=spec)
        assert r["status"] == "LIVE_OK", (r.get("error") or "")[-500:]
        lines = r["stdout"].splitlines()
        eps = next((l.strip() for l in lines if "EPS ID" in l), "")
        bom = next((l.strip() for l in lines if "BOM" in l), "")
        return eps, bom

    climbing = probe(CLIMBING_SPEC)
    household = probe(HOUSEHOLD_SPEC)
    assert climbing[0] != household[0]              # EPS-ID 不同
    assert climbing[1] != household[1]              # BOM 估算不同 (~1.94x)
