"""契约与 CLI 测试 (K1 骨架) — unittest 兼容 (K2: 移除 pytest 硬依赖)。

覆盖: 契约加载校验、DBC 结构、安全链数值、CLI 子命令 (K1 三 + K2 二)、dynamics 模块。
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
os.environ.setdefault("PYTHONPATH", str(SRC))
sys.path.insert(0, str(SRC))

from embodied_ai.config import load_contract, validate_dbc

REPO = Path(__file__).resolve().parents[1]
_SUB_ENV = {**os.environ, "PYTHONPATH": str(SRC)}


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
        self.assertEqual(segs["electrical_off"]["measured_us"], 0.8)
        total = (segs["electrical_off"]["measured_us"] / 1000
                 + segs["arc_extinction"]["measured_ms"]
                 + segs["mechanical_isolation"]["measured_ms"])
        self.assertAlmostEqual(total, 16.0, places=1)  # 容差放宽到 0.1 (mini-YAML 解析精度)
        self.assertLess(total, sb["acceptance"]["total_break_max_ms"])

    def test_dr_params_chance_constraint_inputs(self):
        dr = load_contract("dr_params/manufacturing_variance")
        mu = dr["sensing"]["friction_coefficient"]
        self.assertGreater(mu["sigma_mu"], 0)
        self.assertLess(mu["sigma_mu"], mu["mu_hat"])
        self.assertEqual(dr["electrical"]["mosfet_rds_on"]["temp_coeff_percent_per_c"], 0.4)

    def test_dbc_structure(self):
        r = validate_dbc()
        self.assertEqual(r["message_count"], 9)
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
        for cmd in [["audit"], ["contracts"], ["safety"], ["couple"], ["dr"]]:
            with self.subTest(cmd=cmd):
                r = subprocess.run(
                    [sys.executable, "-m", "embodied_ai.cli", *cmd],
                    capture_output=True, text=True, cwd=str(REPO), env=_SUB_ENV,
                )
                self.assertEqual(r.returncode, 0, f"{cmd} failed:\n{r.stdout}\n{r.stderr}")


if __name__ == "__main__":
    unittest.main()
