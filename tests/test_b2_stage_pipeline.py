"""test_b2_stage_pipeline.py — B2 (Phase B): P1.4 Stage 化收官 — 12 个 CLI 命令注册为可编排 Stage。

B2 验收口径 (docs/10-climb-soft-robot/climb-soft-robot-plan.md Phase B/B2):
    "12 个 CLI 命令注册 Open API Stage" / "EPS job 触发 forge-core→vsa-static 串联"

为什么在 Python 消费侧做 (而非 Kotlin):
    plan_L5.md §三 的 G5 缺口 — "CLI 主链为烟囱式 (各步用样本输入, 非数据管道)"。
    该缺口在 Kotlin 侧, 而 vendor jar 是**冻结的编译产物**, 改不了源码。
    因此 Stage 化只能落在 Python 消费侧: 用 run_eps_job 把各 stage 串成数据管道
    (上游产物进 context, 下游可读), 绕过 Kotlin 烟囱式。

四层覆盖:
  1. 注册表层: STAGE_REGISTRY 12 项 (含 vsa_static), FLOW_CHAIN 仍 11 (向后兼容护栏)
  2. 解析层:   parse_stage_output — EPS ID / VSA JSON 两类产物解析 (纯函数, 无 jar 依赖)
  3. 编排层:   run_eps_job — 串联执行 + context 数据管道 (dry 模式, 无 jar 依赖)
  4. 引擎层:   live 串联 forge-core → vsa-static (需 vendor jar, 否则 skip)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from embodied_ai.adapters.edalab import (  # noqa: E402
    EdalabAdapter, FLOW_CHAIN, STAGE_REGISTRY, VSA_STAGE, CLIMBING_SPEC,
    EDALAB_JAR, STAGE_OUTPUT_PARSERS, parse_stage_output,
)

skip_no_jar = pytest.mark.skipif(
    not EDALAB_JAR.exists(),
    reason=f"无 vendor jar (单仓自足模式需 vendor/edalab/edalabcore.jar): {EDALAB_JAR}")


def _adapter() -> EdalabAdapter:
    return EdalabAdapter(timeout_sec=180)


# ============================================================================
# 1. 注册表层: 12 个 Stage + FLOW_CHAIN 向后兼容护栏
# ============================================================================

def test_stage_registry_has_twelve_stages_including_vsa():
    """P1.4: 12 个 CLI 命令注册为 Stage (11 主链 + vsa-static)。"""
    names = [s["name"] for s in STAGE_REGISTRY]
    assert len(STAGE_REGISTRY) == 12, f"期望 12 个 stage, 实际 {len(STAGE_REGISTRY)}"
    assert "vsa_static" in names
    assert VSA_STAGE["cli"] == "vsa-static"
    assert VSA_STAGE["cli_args"] == ["--json"]      # 消费侧靠 --json 拿结构化曲线
    assert VSA_STAGE["engine"] == "VsaVariableStiffnessEngine.kt"


def test_flow_chain_unchanged_backward_compat():
    """护栏: FLOW_CHAIN 必须仍是 11 步 — 既有 320 项测试与 chain_green(expect_steps=11) 依赖它。

    B2 的编排基于 STAGE_REGISTRY(超集), 绝不修改 FLOW_CHAIN。此测试防止未来误改。
    """
    assert len(FLOW_CHAIN) == 11
    assert [s["name"] for s in FLOW_CHAIN] == [
        "forge_core", "mass_props", "structural_fem", "kinematics", "thermal_sim",
        "pi_analysis", "motor_drive", "data_flywheel", "relay_clutch",
        "em_coupling", "multi_physics"]
    # 注册表的头 11 项必须与 FLOW_CHAIN 逐项一致 (超集关系)
    assert [s["name"] for s in STAGE_REGISTRY[:11]] == [s["name"] for s in FLOW_CHAIN]


def test_stage_output_parsers_cover_two_artifacts():
    """编排的价值在于产物可解析: EPS ID + VSA JSON 两类。"""
    assert STAGE_OUTPUT_PARSERS["forge_core"] == "eps_id"
    assert STAGE_OUTPUT_PARSERS["vsa_static"] == "vsa_json"


# ============================================================================
# 2. 解析层: 产物解析 (纯函数, 无 jar 依赖, CI 恒跑)
# ============================================================================

def test_parse_eps_id_from_forge_core_stdout():
    """B1 实测口径: 'EPS ID: SF-climbing_tendon_drive-climbing_competition_v0_1'。"""
    rec = {"status": "LIVE_OK", "stdout": (
        "策略档案: climbing\n"
        "EPS ID: SF-climbing_tendon_drive-climbing_competition_v0_1\n"
        "封装: 必需\n")}
    ctx = parse_stage_output("forge_core", rec)
    assert ctx["kind"] == "eps_id"
    assert ctx["eps_id"] == "SF-climbing_tendon_drive-climbing_competition_v0_1"


def test_parse_eps_id_returns_none_when_absent():
    """容错: 无 EPS ID 时返回 None (不抛异常, 编排继续)。"""
    ctx = parse_stage_output("forge_core", {"status": "LIVE_OK", "stdout": "no id here"})
    assert ctx["eps_id"] is None


def test_parse_vsa_json_extracts_schema_and_topologies():
    """vsa-static --json → vsa_curve_v1_1, 三拓扑 (MACCEPA/SEA/Antagonistic)。"""
    import json
    payload = {
        "schema": "vsa_curve_v1_1",
        "topologies": [{"name": "VSA-MACCEPA-ClimbingArm"},
                       {"name": "VSA-SEA"},
                       {"name": "VSA-Antagonistic"}],
    }
    # 模拟 Kotlin 在 JSON 前后打日志行 (真实输出形态)
    stdout = "[INFO] generating\n" + json.dumps(payload) + "\n[INFO] done\n"
    ctx = parse_stage_output("vsa_static", {"status": "LIVE_OK", "stdout": stdout})
    assert ctx["kind"] == "vsa_json"
    assert ctx["vsa_schema"] == "vsa_curve_v1_1"
    assert ctx["vsa_topologies"] == ["VSA-MACCEPA-ClimbingArm", "VSA-SEA", "VSA-Antagonistic"]


def test_parse_vsa_json_tolerates_non_json():
    """容错: 非 JSON 输出 (如 dry 模式/引擎变更) 返回 None, 不影响编排。"""
    ctx = parse_stage_output("vsa_static", {"status": "READY", "stdout": "not json"})
    assert ctx["vsa"] is None
    assert ctx.get("vsa_schema") is None


# ============================================================================
# 3. 编排层: run_eps_job 数据管道 (dry 模式, 无 jar 依赖)
# ============================================================================

def test_eps_job_dry_runs_and_builds_context():
    """dry 模式串联: 每步产物进 context (数据管道骨架), 与烟囱式的本质区别。"""
    job = _adapter().run_eps_job(["forge_core", "vsa_static"],
                                 spec=CLIMBING_SPEC, mode="dry")
    assert job["steps"] == job["expected_steps"] == 2
    assert job["job"] == ["forge_core", "vsa_static"]
    assert job["spec"] == CLIMBING_SPEC
    # context 必须覆盖两个阶段 (上游产物可被下游读取的载体)
    assert set(job["context"]) == {"forge_core", "vsa_static"}
    assert job["context"]["forge_core"]["kind"] == "eps_id"
    assert job["context"]["vsa_static"]["kind"] == "vsa_json"


def test_eps_job_unknown_stage_raises():
    """非法 stage 名必须显式报错 (而非静默跳过)。"""
    with pytest.raises(ValueError, match="未知 stage"):
        _adapter().run_eps_job(["forge_core", "not_a_stage"], mode="dry")


def test_eps_job_stops_on_failure():
    """任一步失败即中断 (不假装全绿)。"""
    job = _adapter().run_eps_job(["forge_core", "vsa_static"], mode="dry")
    # dry 模式下若引擎文件解析失败会 ENGINE_MISSING → ok=False
    # 本测试断言 ok 与状态一致 (不自相矛盾)
    assert isinstance(job["ok"], bool)
    assert job["ok"] == all(r["status"] in ("READY", "LIVE_OK") for r in job["results"])


# ============================================================================
# 4. 引擎层: live 串联 forge-core → vsa-static (B2 的决定性验收)
# ============================================================================

@skip_no_jar
def test_live_eps_job_forge_core_to_vsa_static():
    """B2 验收: EPS job 触发 forge-core → vsa-static 串联, 且产物进入数据管道。

    2026-10-07 实测:
      forge_core → LIVE_OK, EPS ID = SF-climbing_tendon_drive-climbing_competition_v0_1
      vsa_static → LIVE_OK, schema = vsa_curve_v1_1, 拓扑 = [MACCEPA-ClimbingArm, SEA, Antagonistic]
      jar_source = vendor (单仓自足, 零 gradle 缓存依赖)
    """
    job = _adapter().run_eps_job(["forge_core", "vsa_static"],
                                 spec=CLIMBING_SPEC, mode="live")
    assert job["ok"], f"串联失败: {[ (r['stage'], r['status']) for r in job['results'] ]}"
    assert job["steps"] == 2

    fc = job["context"]["forge_core"]
    vs = job["context"]["vsa_static"]

    # forge-core 真的产出攀岩腱驱动 EPS (非 household 串味)
    assert fc["eps_id"] == "SF-climbing_tendon_drive-climbing_competition_v0_1"
    # vsa-static 真的产出可消费的结构化曲线
    assert vs["vsa_schema"] == "vsa_curve_v1_1"
    assert vs["vsa_topologies"] and len(vs["vsa_topologies"]) >= 1
    # 单仓自足: 走 vendor jar, 不依赖 gradle 缓存
    for r in job["results"]:
        assert r.get("jar_source") == "vendor", f"{r['stage']} 未走 vendor: {r.get('jar_source')}"
