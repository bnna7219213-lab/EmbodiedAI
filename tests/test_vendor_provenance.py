"""test_vendor_provenance.py — v4 漂移治理 L1 (指纹闸门) 的守护测试。

风险背景: vendor jar 是**冻结的编译产物**。原 EdaLab 源码若演进 (改引擎/改公式),
jar 不会自动更新 → 行为静默漂移, 且无人知晓。B3 已实证过一次:
契约 109.0 落后于引擎实测 166.39 (V-2 修复前口径)。

治理分两层:
  L1 指纹 (本测试): jar 被替换/重新构建 → sha256 变化 → 立刻暴露。
  L2 行为锚定: 契约锚定值 vs 引擎实测 (test_b3_vsa_antagonistic 等) —
     覆盖"源码已演进、jar 该更新了"的情形 (L1 只能发现 jar 变了)。

本测试的关键价值是**反证**: 篡改 hash / 声明缺失 jar 必须被抓到 —
证明闸门是真的在守, 而不是摆设。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from embodied_ai.adapters.edalab import (  # noqa: E402
    PROVENANCE_PATH, VENDOR_DIR, verify_vendor_provenance,
)


def _meta() -> dict:
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, meta: dict) -> Path:
    p = tmp_path / "PROVENANCE.json"
    p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return p


# ============================================================================
# 1. 正常态: 当前 vendor jar 与溯源记录一致
# ============================================================================

def test_provenance_file_exists_and_parseable():
    assert PROVENANCE_PATH.exists(), f"缺溯源元数据: {PROVENANCE_PATH}"
    meta = _meta()
    assert meta.get("schema") == "vendor_jar_provenance_v1"


def test_provenance_declares_both_required_jars():
    """两件缺一不可: core + Kotlin 运行时 (edalabcore 不自含运行时)。"""
    roles = {j["role"] for j in _meta()["jars"]}
    assert roles == {"core", "kotlin_runtime"}, f"声明的 role 不全: {roles}"


def test_verify_vendor_provenance_ok():
    r = verify_vendor_provenance()
    assert r["ok"], f"溯源校验失败: {[c for c in r['checks'] if not c['ok']]}"
    assert len(r["jars"]) == 2
    assert all(j["ok"] for j in r["jars"])


def test_provenance_sha_matches_actual_files():
    """记录里的 sha256 必须等于磁盘上 jar 的真实 sha256。"""
    import hashlib
    for jm in _meta()["jars"]:
        f = VENDOR_DIR / str(jm["file"])
        assert f.exists(), f"jar 缺失: {f}"
        actual = hashlib.sha256(f.read_bytes()).hexdigest()
        assert actual == jm["sha256"], (
            f"{jm['file']} 指纹漂移: 记录={jm['sha256'][:12]}… 实际={actual[:12]}… "
            f"(若 jar 已重新构建, 请按 PROVENANCE.json 的 update_procedure 同步更新)")
        assert len(f.read_bytes()) == jm["size_bytes"]


# ============================================================================
# 2. 反证: 闸门必须能抓到漂移 (证明不是摆设)
# ============================================================================

def test_tampered_sha256_is_detected(tmp_path):
    """反证 A: jar 被替换 (hash 变) 但 PROVENANCE 未更新 → 必须 FAIL。"""
    meta = _meta()
    meta["jars"][0]["sha256"] = "0" * 64      # 假装 jar 换成了别的构建
    r = verify_vendor_provenance(provenance_path=_write(tmp_path, meta))
    assert not r["ok"], "篡改 hash 竟未检出 — 闸门失效"
    assert any(c["check"].startswith("sha256[") and not c["ok"] for c in r["checks"])


def test_size_mismatch_is_detected(tmp_path):
    """反证 B: 文件大小与记录不符 → 必须 FAIL。"""
    meta = _meta()
    meta["jars"][0]["size_bytes"] = 12345
    r = verify_vendor_provenance(provenance_path=_write(tmp_path, meta))
    assert not r["ok"]
    assert any(c["check"].startswith("size[") and not c["ok"] for c in r["checks"])


def test_missing_declared_jar_is_detected(tmp_path):
    """反证 C: 声明了磁盘上不存在的 jar → 必须 FAIL。"""
    meta = _meta()
    meta["jars"].append({"file": "nonexistent-engine.jar", "role": "core",
                         "sha256": "0" * 64, "size_bytes": 1})
    r = verify_vendor_provenance(provenance_path=_write(tmp_path, meta))
    assert not r["ok"]
    assert any(c["check"].startswith("jar_exists[nonexistent") and not c["ok"]
               for c in r["checks"])


def test_missing_provenance_file_is_detected(tmp_path):
    """反证 D: PROVENANCE.json 本身缺失 → 必须 FAIL (不能静默通过)。"""
    r = verify_vendor_provenance(provenance_path=tmp_path / "nope.json")
    assert not r["ok"]
    assert any(c["check"] == "provenance_file_exists" and not c["ok"]
               for c in r["checks"])


def test_provenance_declares_update_procedure():
    """溯源元数据必须自带"jar 更新后怎么办"的流程 (否则治理不可执行)。"""
    meta = _meta()
    proc = ((meta.get("drift_policy") or {}).get("update_procedure") or [])
    assert len(proc) >= 3, "缺少 jar 更新流程说明"
    layers = ((meta.get("drift_policy") or {}).get("detection_layers") or [])
    assert len(layers) >= 2, "应至少声明 L1 指纹 + L2 行为锚定两层"
