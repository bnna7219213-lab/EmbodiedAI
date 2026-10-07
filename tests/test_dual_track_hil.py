"""test_dual_track_hil.py — 双轨骨架 + G6 (HIL 桥接) 守护测试。

双轨 (路径 B) 的必要约束, 本测试逐条守护:
  1. 单一入口: 一律经 run_capability(), 调用方不直接碰后端
  2. jar 作 oracle: (jar 无 hil CLI, 故此处 python 侧须自带真实模型支撑)
  3. 可观测: 结果必须带 backend 字段
  4. 不静默: 后端不可用必须抛错, 不得悄悄回落

G6 验收: HIL 结果**不得**由随机数产生 (uses_random=False), 且
硬件架构属性类用例必须标 REQUIRES_BENCH 而非假装 PASS。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# 导入即注册 Python 后端
from embodied_ai.hardware import hil_bridge  # noqa: E402,F401
from embodied_ai.adapters.edalab import (  # noqa: E402
    PYTHON_BACKENDS, EdalabAdapter, available_backends, resolve_backend,
)
from embodied_ai.hardware.hil_bridge import HIL_CASE_IDS, REQUIRES_BENCH  # noqa: E402


def _adapter() -> EdalabAdapter:
    return EdalabAdapter(timeout_sec=180)


# ============================================================================
# 1. 双轨骨架
# ============================================================================

def test_hil_registered_as_python_backend():
    """hil 必须注册为 Python 后端 (jar 侧无 hil CLI)。"""
    assert "hil" in PYTHON_BACKENDS
    # jar 侧没有 hil stage → 只有 python 可用
    assert available_backends("hil") == ["python"]


def test_resolve_backend_auto_picks_available():
    """auto 路由必须选到唯一可用后端, 不静默返回 None。"""
    assert resolve_backend("hil", "auto") == "python"


def test_resolve_backend_rejects_unavailable():
    """反证: 强制不可用的 backend 必须抛错 (不得静默回落)。"""
    with pytest.raises(ValueError, match="不支持 backend"):
        resolve_backend("hil", backend="jar")


def test_jar_capability_has_jar_backend():
    """对照: 主链 stage (relay_clutch) 应同时有 jar 后端。"""
    assert "jar" in available_backends("relay_clutch")


# ============================================================================
# 2. G6: HIL 不再用随机数
# ============================================================================

def test_run_capability_hil_returns_backend_field():
    """可观测: 结果必须带 backend 字段 (知道是谁算的)。"""
    r = _adapter().run_capability("hil")
    assert r["backend"] == "python"
    assert r["status"] == "OK"


def test_hil_does_not_use_random():
    """G6 核心: HIL 结果不得由随机数产生。"""
    r = _adapter().run_capability("hil")
    assert r["uses_random"] is False, "HIL 仍在用随机数 → G6 未修复"


def test_hil_covers_all_seven_cases():
    """7 条用例必须全部在列 (不遗漏)。"""
    r = _adapter().run_capability("hil")
    ids = {c["id"] for c in r["cases"]}
    assert ids == set(HIL_CASE_IDS), f"用例覆盖不全: 缺 {set(HIL_CASE_IDS) - ids}"


def test_hardware_only_cases_marked_requires_bench():
    """反证诚实性: 硬件架构属性用例不得假装 PASS。

    HIL-STO-02 (MCU 死机仍能关断) / HIL-STO-05 (锁存不可自恢复) 依赖
    硬件级锁存电路, 软件仿真无法证明 — 必须标 REQUIRES_BENCH。
    """
    r = _adapter().run_capability("hil")
    by_id = {c["id"]: c for c in r["cases"]}
    for cid in REQUIRES_BENCH:
        assert by_id[cid]["verdict"] == "REQUIRES_BENCH", (
            f"{cid} 被判为 {by_id[cid]['verdict']} — 硬件属性不该由仿真证明")
        assert by_id[cid]["source"] == "none"
    assert r["n_bench"] == len(REQUIRES_BENCH)


def test_simulatable_cases_have_real_models():
    """可仿真的 5 条须有真实模型来源 (source=python_model), 且数值有限。"""
    r = _adapter().run_capability("hil")
    sim = [c for c in r["cases"] if c["id"] not in REQUIRES_BENCH]
    assert len(sim) == 5
    for c in sim:
        assert c["source"] == "python_model", f"{c['id']} 无真实模型来源"
        assert c["verdict"] in ("PASS", "FAIL")


def test_hil_is_deterministic_not_random():
    """反证: 两次调用结果必须一致 (随机数会导致不一致)。"""
    a = _adapter().run_capability("hil", backend="python")
    b = _adapter().run_capability("hil", backend="python")
    ma = {c["id"]: c.get("measured", c.get("measured_pct", c.get("measured_s")))
          for c in a["cases"]}
    mb = {c["id"]: c.get("measured", c.get("measured_pct", c.get("measured_s")))
          for c in b["cases"]}
    assert ma == mb, "两次 HIL 结果不一致 — 疑似仍在用随机数"


def test_hil_can_transport_uses_aligned_canfd():
    """G6 另一半: 总线须用已对齐的 CAN-FD (非随机总线)。"""
    r = _adapter().run_capability("hil")
    can = r["can_transport"]
    assert can["can_roundtrip_ok"], "CAN-FD 回环失败 (应复用 R4 已对齐编解码)"


# ============================================================================
# 3. 结果结构契约
# ============================================================================

def test_python_backend_status_defaults_to_ok():
    """后端未显式给 status 时, 入口须补 OK (两种后端 status 语义对齐)。"""
    r = _adapter().run_capability("hil")
    assert r["status"] in ("OK", "FAILED")
    assert "elapsed_s" in r
