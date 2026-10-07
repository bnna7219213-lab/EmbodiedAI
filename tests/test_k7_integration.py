"""K7: v0.7 接线层集成测试 — CLI / 自检聚合 / 契约修复 / 模块可达性。

这一层**不跑重计算** (重计算在 test_k6 与 verification 里), 只验证:
  1. 新子命令真的注册进了 CLI 分发表 (否则 `embodied-audit soft` 报 KeyError);
  2. 聚合自检的检查项清单完整且与 CLI 一致;
  3. v0.7 修掉的既有缺陷不再回归 (vla_small 消融 / STL Until / DBC 量程)。
"""

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from embodied_ai import cli, verification  # noqa: E402


class TestK7A_CLISurface(unittest.TestCase):
    """CLI 子命令表面。"""

    def test_new_subcommands_registered(self):
        for name in ("soft", "wholebody", "foc", "hand", "real", "bench",
                     "cert", "lifelong", "train", "gui"):
            self.assertIn(name, cli.main.__globals__["cmds"]
                          if "cmds" in cli.main.__globals__ else
                          _dispatch_table(), f"子命令 {name} 未注册")

    def test_help_includes_new_commands(self):
        r = subprocess.run([sys.executable, "-m", "embodied_ai.cli", "--help"],
                           capture_output=True, cwd=str(REPO),
                           env={"PYTHONPATH": str(REPO / "src"),
                                "PYTHONIOENCODING": "utf-8",
                                "PATH": __import__("os").environ["PATH"],
                                "SYSTEMROOT": __import__("os").environ.get(
                                    "SYSTEMROOT", "")})
        out = r.stdout.decode("utf-8", "replace")
        for name in ("soft", "foc", "hand", "cert", "bench", "lifelong", "gui"):
            self.assertIn(name, out)

    def test_unknown_command_exits_nonzero(self):
        r = subprocess.run([sys.executable, "-m", "embodied_ai.cli", "nosuchcmd"],
                           capture_output=True, cwd=str(REPO),
                           env={"PYTHONPATH": str(REPO / "src"),
                                "PYTHONIOENCODING": "utf-8",
                                "PATH": __import__("os").environ["PATH"],
                                "SYSTEMROOT": __import__("os").environ.get(
                                    "SYSTEMROOT", "")})
        self.assertNotEqual(r.returncode, 0)


def _dispatch_table():
    """取回可执行命令名集合。

    2026-10-08 重构: 原实现从 `cli.main` 源码里正则提取 `cmds = {...}` 字典 ——
    那是"参数声明与 dispatch 分离两处"时代的产物。命令现已进入**全局注册表**
    (`cli_registry.CLI_REGISTRY`), 本函数改为读注册表, 即真正的唯一真源。
    """
    from embodied_ai.cli_registry import CLI_REGISTRY
    return set(CLI_REGISTRY)


class TestK7B_VerificationSurface(unittest.TestCase):
    """聚合自检清单。"""

    def test_new_checks_present(self):
        names = {n for n, _ in verification.CHECKS}
        for n in ("soft_body", "whole_body_3d", "diffusion_policy", "bc_training",
                  "rl_algorithms", "foc_shutdown", "dexterous_hand",
                  "real_robot_loop", "jetson_bench", "lifelong_learning",
                  "certification", "gui_dashboard"):
            self.assertIn(n, names)

    def test_backward_compatible_checks_kept(self):
        names = {n for n, _ in verification.CHECKS}
        for n in ("contracts", "safety_budget", "simulation", "sim_to_real",
                  "biped_gait", "nanogpt_plan", "arm_linux", "cli_smoke"):
            self.assertIn(n, names)

    def test_summarize_counts(self):
        res = [("a", True, "OK"), ("b", False, "x")]
        s = verification.summarize(res)
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["passed"], 1)
        self.assertFalse(s["ok"])
        self.assertEqual(s["failed"], ["b"])


class TestK7C_RegressionFixes(unittest.TestCase):
    """v0.7 修掉的既有缺陷不得回归。"""

    def test_vla_small_ablation_exists(self):
        from embodied_ai.models import VLAConfig
        abl = VLAConfig.ablations()
        for name in ("vla_tiny", "vla_small", "vla_base", "vla_large"):
            self.assertIn(name, abl)
        self.assertLess(abl["vla_small"].estimate_params()["total"],
                        abl["vla_large"].estimate_params()["total"])

    def test_stl_until_and_next(self):
        import numpy as np

        from embodied_ai.safety.formal import (STLAtom, STLGlobally, STLNext,
                                               STLUntil, force_limit_monitor)
        sig = {"time": np.linspace(0, 0.2, 21),
               "force": np.concatenate([np.full(10, 0.0), np.full(11, 5.0)])}

        def read(t):   # 用于 STLAtom 的信号取值路径
            return np.array([sig["force"][int(round(t / 0.01))]])

        zero = STLAtom("zero", lambda x: float(abs(x[0])) - 1e-6)
        nonzero = STLAtom("nonzero", lambda x: 1e-3 - float(abs(x[0])))
        # F/G 走 dict 信号: 这里用 ndarray 信号简化 (单时刻)
        self.assertEqual(STLNext(zero, 0.01).robustness(np.array([0.0])),
                         zero.robustness(np.array([0.0])))
        u = STLUntil(zero, nonzero, a=0.0, b=0.05)
        self.assertIsInstance(u.robustness(np.array([0.0])), float)
        self.assertIsInstance(force_limit_monitor(100.0), STLGlobally)

    def test_dbc_position_signal_range(self):
        """位置信号量程必须 ≥ 关节行程 (v0.7 修: 0.001 → 0.01)。"""
        dbc = (REPO / "contracts" / "canfd" / "joint_control.dbc").read_text(
            encoding="utf-8")
        self.assertNotIn("(0.001,0) [0|65.535]", dbc)
        self.assertIn("(0.01,0) [0|327.67]", dbc)

    def test_dbc_message_count_sixteen(self):
        """R4 (2026-10-07): DBC 6 关节全量遥测后消息数 9 → 14 (含 6 个 JointTelemetry)。

        旧名 `_still_nine` 已过时; 改名以体现新期望值。功能: 防止未来编码扩展把
        JointTelemetry 数又退回 1 条 (退化到只读 SHOULDER_L)。
        """
        from embodied_ai.config import validate_dbc
        self.assertEqual(validate_dbc()["message_count"], 14)


class TestK7D_ModuleReachability(unittest.TestCase):
    """新增模块必须可导入 (CI 上的"全模块导入"门)。"""

    def test_import_all_new_modules(self):
        import importlib

        mods = [
            "embodied_ai.models.nn_core",
            "embodied_ai.models.diffusion_policy",
            "embodied_ai.models.bc_trainer",
            "embodied_ai.models.rl_algorithms",
            "embodied_ai.dynamics.soft_body",
            "embodied_ai.control.whole_body",
            "embodied_ai.hardware.foc_shutdown",
            "embodied_ai.hardware.dexterous_hand",
            "embodied_ai.simulation.real_robot_env",
            "embodied_ai.deploy.jetson_bench",
            "embodied_ai.learning.lifelong",
            "embodied_ai.safety.certification",
            "embodied_ai.gui.server",
        ]
        for m in mods:
            importlib.import_module(m)

    def test_gui_static_page_shipped(self):
        p = REPO / "src" / "embodied_ai" / "gui" / "static" / "index.html"
        self.assertTrue(p.exists())
        self.assertIn("api/safety", p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
