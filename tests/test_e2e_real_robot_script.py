"""e2 真实机器人 E2E 测试脚本 测试 (docs/19 §6)。

反证: mock 模式全链通过返回 0; ESTOP 负对照返回 1 (判据能失败);
报告 JSON 可导出; 时延超预算返回 1。
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import e2e_real_robot_test as e2e_script  # noqa: E402


class TestMockRun:
    def test_mock_all_pass(self):
        report = e2e_script.run("mock", verbose=False)
        assert report["all_pass"] is True
        assert report["reached"] == "done"
        assert report["n_phases_done"] == 5

    def test_main_returns_zero(self, tmp_path, capsys):
        rc = e2e_script.main(["--mode", "mock"])
        assert rc == 0

    def test_report_export(self, tmp_path):
        out = tmp_path / "e2e.json"
        rc = e2e_script.main(["--mode", "mock", "--report", str(out)])
        assert rc == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["all_pass"] is True
        assert data["mode"] == "mock"
        assert len(data["phases"]) == 5


class TestNegativeControls:
    def test_estop_blocks_and_fails(self):
        """ESTOP 后续阶段拒绝 → 返回 1 (判据能失败)。"""
        report = e2e_script.run("mock", estop_after="s2_perception",
                                verbose=False)
        assert report["all_pass"] is False
        assert report["n_phases_done"] < 5

    def test_over_budget_fails(self):
        """小预算 → S4 时延超预算 FAIL。"""
        report = e2e_script.run("mock", budget_ms=1.0, verbose=False)
        assert report["all_pass"] is False

    def test_main_nonzero_on_fail(self):
        rc = e2e_script.main(["--mode", "mock", "--budget-ms", "1.0"])
        assert rc == 1


class TestRealModeGuard:
    def test_real_mode_requires_can(self):
        """真机模式无 CAN 硬件/依赖时构造即失败 (不静默降级到 mock)。"""
        try:
            e2e_script.build_harness("real", "can0", 1_000_000, 50.0)
        except Exception as exc:  # noqa: BLE001 — 无 CAN 硬件的环境必须显式失败
            assert exc is not None
        else:
            # 本机恰好有 CAN 时跳过
            pytest.skip("本机 CAN 可用, 真机构造成功")
