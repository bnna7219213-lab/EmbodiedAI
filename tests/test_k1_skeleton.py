"""契约与 CLI 测试 (K1 骨架) — unittest 兼容 (K2: 移除 pytest 硬依赖)。

覆盖: 契约加载校验、DBC 结构、安全链数值、CLI 子命令 (K1 三 + K2 二)、dynamics 模块。
"""
import subprocess
import sys
import unittest
from pathlib import Path

# v0.5: sys.path / 子进程环境统一由 tests/conftest.py 注入, 此处不再重复拼装。
from conftest import REPO, SRC, SUB_ENV

from embodied_ai.config import load_contract, validate_dbc

_SUB_ENV = SUB_ENV


class TestK1Contracts(unittest.TestCase):
    def test_hw_requirements_contract(self):
        hw = load_contract("hw_requirements")
        self.assertEqual(hw["motor_drive"]["bus_voltage_v"], 48)
        self.assertEqual(hw["motor_drive"]["peak_current_a"], 30)
        self.assertEqual(hw["motor_drive"]["protection"]["hardware_overcurrent_us"], 1)
        self.assertEqual(hw["compute"]["end_to_end_inference_budget_ms"], 50)

    def test_safety_budget_chain(self):
        sb = load_contract("safety_budget")
        segs = {s["stage"]: s for s in sb["segments"]}
        # P0 v1.1 修复: 字段名从 measured_* 改为 assumed_* (诚实边界)
        self.assertEqual(segs["electrical_off"]["assumed_us"], 0.8)
        total = (segs["electrical_off"]["assumed_us"] / 1000
                 + segs["arc_extinction"]["assumed_ms"]
                 + segs["mechanical_isolation"]["assumed_ms"])
        self.assertAlmostEqual(total, 16.0, places=1)  # 容差放宽到 0.1 (mini-YAML 解析精度)
        self.assertLess(total, sb["acceptance"]["total_break_max_ms"])
        # v1.1 新增: 诚实标记 SIL2 非发布级
        self.assertEqual(sb["metadata"]["provenance_confidence"], "assumed")
        self.assertFalse(sb["metadata"]["sil2_release_ready"])

    def test_safety_budget_provenance_honest(self):
        """v1.1 P0: 断言假设+建模数据被明确标记为 non-release (非 HIL 实测)。"""
        sb = load_contract("safety_budget")
        meta = sb["metadata"]
        self.assertIn(meta["provenance_confidence"],
                      ("assumed", "modeled", "modeled_hil", "measured_hil"))
        # 只有 measured_hil 才能发布 SIL2
        self.assertEqual(
            meta["sil2_release_ready"],
            meta["provenance_confidence"] == "measured_hil",
            "sil2_release_ready 必须与 provenance_confidence='measured_hil' 联动",
        )
        # acceptance 中必须区分假设 vs 实测 PFH
        acc = sb["acceptance"]
        self.assertIn("assumed_pfh", acc)
        # measured_pfh 若存在需与 assumed_pfh 相等 (未接入 HIL 前的诚实边界)
        if "measured_pfh" in acc:
            self.assertEqual(acc["measured_pfh"], acc["assumed_pfh"])

    def test_dr_params_chance_constraint_inputs(self):
        dr = load_contract("dr_params/manufacturing_variance")
        mu = dr["sensing"]["friction_coefficient"]
        self.assertGreater(mu["sigma_mu"], 0)
        self.assertLess(mu["sigma_mu"], mu["mu_hat"])
        self.assertEqual(dr["electrical"]["mosfet_rds_on"]["temp_coeff_percent_per_c"], 0.4)

    def test_dbc_structure(self):
        r = validate_dbc()
        # R4 (2026-10-07): DBC 6 关节全量遥测后消息数 9 → 14
        # (6 JointCmd + 1 Sync + 6 JointTelemetry + 1 SafetyEvent)
        self.assertEqual(r["message_count"], 14)
        self.assertTrue(r["required_ok"] and r["unique_ids"] and r["crc_coverage"])


class TestK1Modules(unittest.TestCase):
    def test_dynamics_imports(self):
        from embodied_ai.dynamics import climbing_dynamics_core as cdc
        k = cdc.hertz_stiffness(2.5e5, 0.45, 1e7, 0.3, 0.015)
        self.assertGreater(k, 0)

    def test_safety_module(self):
        from embodied_ai.safety import chance_constrained_grip as ccg
        fn = ccg.chance_constrained_grip(f_t_req=5.0, mu_hat=0.8, sigma_mu=0.15, delta=0.05)
        self.assertGreater(fn, 5.0)


class TestK1CLI(unittest.TestCase):
    def test_cli_commands(self):
        for cmd in [["audit"], ["contracts"], ["safety"], ["couple"], ["dr"],
                    ["sim"], ["sim2real"], ["biped"], ["nanogpt"], ["verify", "--no-pytest"]]:
            with self.subTest(cmd=cmd):
                r = subprocess.run(
                    [sys.executable, "-m", "embodied_ai.cli", *cmd],
                    capture_output=True, cwd=str(REPO), env=_SUB_ENV,
                    encoding="utf-8", errors="replace",
                )
                self.assertEqual(r.returncode, 0, f"{cmd} failed:\n{r.stdout}\n{r.stderr}")

    def test_cli_exit_nonzero_on_broken_contract(self):
        """契约不可解析时 safety 必须以非 0 退出 (退出码语义可用)。

        R-A S0.2 (2026-10-07) 修复: 原版靠"剥 PYTHONPATH 模拟包未安装"触发失败,
        但 editable install (`pip install -e .`) 让包始终可导入, 测试前提失效。
        现改为: 创建临时残缺 contracts 目录 (无 hw_requirements.yaml),
        用 EMBODIED_CONTRACTS_DIR 注入, 让 load_contract 抛 FileNotFoundError。
        """
        import tempfile
        with tempfile.TemporaryDirectory() as broken_dir:
            r = subprocess.run(
                [sys.executable, "-m", "embodied_ai.cli", "contracts"],
                capture_output=True, cwd=str(REPO),
                env={**SUB_ENV, "PYTHONPATH": str(REPO),
                      "EMBODIED_CONTRACTS_DIR": broken_dir},
                encoding="utf-8", errors="replace",
            )
            self.assertNotEqual(r.returncode, 0,
                                  f"残留契约目录应让 CLI 非 0 退出, 实得 stdout: {r.stdout[:300]}")


if __name__ == "__main__":
    unittest.main()
