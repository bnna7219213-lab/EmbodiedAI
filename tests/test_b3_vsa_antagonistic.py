"""test_b3_vsa_antagonistic.py — B3 (Phase B): VSA-ANTAGONISTIC 选型落地。

B3 定义 (docs/10-climb-soft-robot/climb-soft-robot-plan.md Phase B):
    内容: P1.3 结论 (攀岩指尖 109N 物理正确) 写入 B1 的电机/减速器参数
    验收: relay-clutch + vsa-static 联合 PASS

★ B3 的实质发现 (2026-10-07 实测, 本测试固化):
    契约原值 `vsa_antagonistic_peak_n: 109.0` 与 vsa-static 引擎实测
    **166.39N 不一致**。109 是 V-2 修复 (ANTAGONISTIC V 形公式改写为
    kMin + t*(kMax-kMin)) **之前**的过期口径; 且 docs/04 该表行的 stored(0.635J)
    疑混填了 SEA 的实测值 (SEA 实测 0.6379301)。契约已按引擎实测校准为 166.39,
    选型结论不变 (ANTAGONISTIC 仍是三拓扑峰值最小者: 166 < SEA 347 < MACCEPA 525)。

三层覆盖:
  1. 解析层:  relay-clutch 输出解析 (额定/传递扭矩 + 安全链时延) — 纯函数, 无 jar 依赖
  2. 校验层:  verify_b3_antagonistic 五判据, 含"旧值 109 应被拒绝"的反证
  3. 引擎层:  live 联合 relay-clutch + vsa-static, 五判据全绿 (需 vendor jar, 否则 skip)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from embodied_ai.adapters.edalab import (  # noqa: E402
    EdalabAdapter, CLIMBING_SPEC, EDALAB_JAR, parse_stage_output,
    verify_b3_antagonistic, load_hw_requirements,
)

skip_no_jar = pytest.mark.skipif(
    not EDALAB_JAR.exists(),
    reason=f"无 vendor jar (单仓自足模式需 vendor/edalab/edalabcore.jar): {EDALAB_JAR}")

# relay-clutch 实测输出形态 (2026-10-07, vendor jar, climbing 档案)
RELAY_CLUTCH_STDOUT = (
    "=== 具身智能体继电器/离合器触点动力学 (RelayClutchDynamicsEngine) ===\n"
    "继电器: SafetyShutdown-G4HAA48V (48.0V/30.0A AGSNO2)\n"
    "弹跳: 弹跳 8 次, 弹跳期 7.96ms, 稳定 8.95ms, 恢复系数 e=0.377\n"
    "离合器: JointBrake-FrictionPlate, 传递 6.38Nm (额定 12.0Nm, 53%), "
    "滑摩功 24.7J, 温升 0.1℃→峰值 40.1℃ -> PASS\n"
    "安全链路: 电气 0.8µs + 熄弧 8.0ms + 机械隔离 8.0ms = 总 16.00ms (目标 <20ms) -> PASS\n"
)


def _adapter() -> EdalabAdapter:
    return EdalabAdapter(timeout_sec=180)


# ============================================================================
# 1. 解析层: relay-clutch 物理量解析
# ============================================================================

def test_parse_clutch_extracts_torques_and_chain():
    """离合器的额定扭矩是 B3 '能否锁定 VSA 弹射峰值' 判据的分母。"""
    ctx = parse_stage_output("relay_clutch",
                             {"status": "LIVE_OK", "stdout": RELAY_CLUTCH_STDOUT})
    assert ctx["kind"] == "clutch"
    assert ctx["transmit_nm"] == pytest.approx(6.38)
    assert ctx["rated_nm"] == pytest.approx(12.0)
    assert ctx["safety_chain_ms"] == pytest.approx(16.00)
    assert ctx["clutch_verdict"] == "PASS"


def test_parse_clutch_tolerates_missing():
    """容错: 无输出时返回 None, 不抛异常 (编排继续)。"""
    ctx = parse_stage_output("relay_clutch", {"status": "READY", "stdout": ""})
    assert ctx["rated_nm"] is None
    assert ctx["clutch_verdict"] is None


# ============================================================================
# 2. 校验层: 五判据 (含防漂移反证)
# ============================================================================

def _synth_context(peak: float, rated: float = 12.0,
                   chain: float = 16.0, verdict: str = "PASS") -> dict:
    """构造 B3 校验用的 context (契约真实值 + 可调的引擎峰值)。"""
    return {
        "vsa_static": {"kind": "vsa_json", "vsa_peaks": {
            "ANTAGONISTIC": peak, "SEA": 346.5549, "MACCEPA": 525.2946}},
        "relay_clutch": {"kind": "clutch", "rated_nm": rated,
                         "safety_chain_ms": chain, "clutch_verdict": verdict},
    }


def test_b3_passes_with_calibrated_contract_value():
    """契约已校准为 166.39 (引擎实测) → 五判据应全绿。"""
    r = verify_b3_antagonistic(_synth_context(166.39))
    assert r["ok"], f"应通过: {r['checks']}"
    names = [c["check"] for c in r["checks"]]
    assert names == ["contract_matches_engine", "antagonistic_is_min_peak",
                     "clutch_holds_vsa_peak", "safety_chain_within_20ms",
                     "clutch_verdict_pass"]


def test_b3_rejects_stale_109_value():
    """反证 (B3 的核心守护): 旧值 109.0 必须被拒绝 — 防止契约回漂到过期口径。"""
    r = verify_b3_antagonistic(_synth_context(109.0))
    assert not r["ok"], "旧值 109 应被拒绝 (与校准后的契约 166.39 不一致)"
    bad = [c for c in r["checks"] if not c["ok"]]
    assert any(c["check"] == "contract_matches_engine" for c in bad)


def test_b3_detects_clutch_slip():
    """若峰值超出离合器额定 (经力臂换算) → 打滑, 必须 FAIL。"""
    # 166.39N × 0.022m = 3.66Nm; 额定压到 3.0Nm 即应判打滑
    r = verify_b3_antagonistic(_synth_context(166.39, rated=3.0))
    assert not r["ok"]
    assert any(c["check"] == "clutch_holds_vsa_peak" and not c["ok"]
               for c in r["checks"])


def test_b3_detects_safety_chain_overrun():
    """安全链 > 20ms → FAIL (契约预算)。"""
    r = verify_b3_antagonistic(_synth_context(166.39, chain=25.0))
    assert not r["ok"]
    assert any(c["check"] == "safety_chain_within_20ms" and not c["ok"]
               for c in r["checks"])


def test_contract_peak_is_calibrated_to_engine():
    """契约值必须已校准为 166.39 (不是过期的 109.0)。"""
    hw = load_hw_requirements()
    peak = hw["climbing"]["force_limits"]["vsa_antagonistic_peak_n"]
    assert peak == pytest.approx(166.39, abs=0.5), f"契约未校准: {peak}"
    # 力臂已显式化 (原散见 servo_peak_torque_nm 注释)
    assert hw["climbing"]["force_limits"]["fingertip_lever_arm_m"] == pytest.approx(0.022)


# ============================================================================
# 3. 引擎层: live 联合 relay-clutch + vsa-static (B3 决定性验收)
# ============================================================================

@skip_no_jar
def test_live_b3_relay_clutch_and_vsa_static_joint_pass():
    """B3 验收: relay-clutch + vsa-static 联合 PASS (物理量交叉, 非各自 LIVE_OK)。

    2026-10-07 实测 (vendor jar, climbing 档案):
      引擎 ANTAGONISTIC peak = 166.39064N  (契约已校准为 166.39)
      离合器额定 12.0Nm, 需求 = 166.39 × 0.022 = 3.66Nm ≤ 12.0 → 不打滑
      安全链 16.00ms < 20ms; 离合器 verdict=PASS
      三拓扑峰值: ANTAGONISTIC 166 < SEA 347 < MACCEPA 525 → 选型仍物理正确
    """
    job = _adapter().run_eps_job(["relay_clutch", "vsa_static"],
                                 spec=CLIMBING_SPEC, mode="live")
    assert job["ok"], f"串联失败: {[(r['stage'], r['status']) for r in job['results']]}"

    r = verify_b3_antagonistic(job["context"])
    assert r["ok"], f"B3 联合校验失败: {r['checks']}"

    v = r["values"]
    assert v["engine_peak_n"] == pytest.approx(166.39, abs=0.5)
    assert v["clutch_demand_nm"] <= v["clutch_rated_nm"], "离合器打滑"
    assert v["safety_chain_ms"] < 20.0
    assert v["clutch_verdict"] == "PASS"
    # 选型: ANTAGONISTIC 峰值最小 (最接近指尖 60N 极限)
    assert v["engine_peak_n"] == min(x for x in v["peaks"].values() if x is not None)
    # 单仓自足
    for res in job["results"]:
        assert res.get("jar_source") == "vendor"
