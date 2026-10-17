"""单仓自足实证 (selfcheck) 守护测试 — v0.8。

守护三件事:
  1. `run_selfcheck()` 在本仓返回 ok=True（5 项断言真的都成立, 不是声称）
  2. `selfcheck` 已注册进全局注册表（项目规矩: 新增调用方式必须注册, 否则落角落）
  3. 脚本薄壳与包内实现同源（避免"脚本一份、CLI 一份"两处维护）
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from embodied_ai.selfcheck import run_selfcheck


class TestSelfcheckLogic(unittest.TestCase):

    def test_selfcheck_passes_on_this_repo(self):
        """本仓应满足单仓自足 — 5 项断言全绿。"""
        res = run_selfcheck()
        self.assertTrue(res["ok"], f"未通过: {res['failed']}\n" +
                        "\n".join(f"  {c['check']}: {c['detail']}" for c in res["checks"] if not c["ok"]))

    def test_five_checks_present(self):
        """五项断言必须齐 (少一项 = 守护范围缩水)。"""
        res = run_selfcheck()
        names = [c["check"] for c in res["checks"]]
        self.assertEqual(len(names), 5, names)
        for kw in ("载体", "指纹", "运行时", "隔离", "可观测"):
            self.assertTrue(any(kw in n for n in names), f"缺少含『{kw}』的检查项: {names}")

    def test_failed_checks_carry_hint(self):
        """任何失败项都必须带修复指引（否则别人跑不起来时无从下手）。"""
        res = run_selfcheck()
        for c in res["checks"]:
            if not c["ok"]:
                self.assertTrue(c.get("hint"), f"失败项『{c['check']}』未提供修复指引")

    def test_vendor_dir_is_in_repo(self):
        """载体必须在本仓内（若指向外部目录, 单仓自足即为空话）。"""
        res = run_selfcheck()
        self.assertTrue(str(res["vendor_dir"]).startswith(str(REPO)),
                        f"vendor 目录不在本仓: {res['vendor_dir']}")


class TestSelfcheckRegistration(unittest.TestCase):
    """项目规矩: 新增调用方式必须注册（否则会落在角落无人顾及）。"""

    def test_registered_in_cli_registry(self):
        from embodied_ai.cli_registry import CLI_REGISTRY
        self.assertIn("selfcheck", CLI_REGISTRY,
                      "selfcheck 未注册进全局注册表 — 请用 register_command 注册")

    def test_catalog_lists_it(self):
        """catalog 能列出 = 注册成功（可发现性）。"""
        from embodied_ai.cli_registry import command_catalog
        self.assertIn("selfcheck", command_catalog())

    def test_has_examples(self):
        """注册项必须带示例（README 与用户都靠它）。"""
        from embodied_ai.cli_registry import CLI_REGISTRY
        self.assertTrue(CLI_REGISTRY["selfcheck"].examples,
                        "selfcheck 注册项缺 examples")

    def test_cli_json_mode_valid(self):
        """CLI --json 输出必须是合法 JSON 且 ok 为真（CI 可断言）。"""
        env = {"PYTHONPATH": str(SRC)}
        p = subprocess.run([sys.executable, "-m", "embodied_ai.cli", "selfcheck", "--json"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env={**__import__("os").environ, **env},
                           cwd=str(REPO), timeout=300)
        self.assertEqual(p.returncode, 0, f"退出码非 0:\n{p.stdout[-400:]}\n{p.stderr[-400:]}")
        data = json.loads(p.stdout)
        self.assertTrue(data["ok"], f"selfcheck 未通过: {data['failed']}")


class TestScriptThinShell(unittest.TestCase):
    """脚本薄壳与包内实现同源（单一真源 — 项目规矩）。"""

    def test_script_delegates_to_package(self):
        src = (REPO / "scripts" / "demo_selfcontained.py").read_text(encoding="utf-8")
        self.assertIn("from embodied_ai.selfcheck import", src,
                      "脚本应委托包内实现, 而不是自带一份逻辑")
        # 不应在脚本里重复实现断言逻辑
        self.assertNotIn("def run_selfcheck", src,
                         "脚本不应重复实现 run_selfcheck（会形成两处维护）")

    def test_script_runs(self):
        p = subprocess.run([sys.executable, str(REPO / "scripts" / "demo_selfcontained.py"),
                            "--json"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace",
                           env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
                           cwd=str(REPO), timeout=300)
        self.assertEqual(p.returncode, 0, f"脚本失败:\n{p.stdout[-400:]}\n{p.stderr[-400:]}")
        self.assertTrue(json.loads(p.stdout)["ok"])


class TestVerifyRefusesStrippedAsserts(unittest.TestCase):
    """verify 的全部 check 都以 assert 作判定, 而 `python -O` 会剥离 assert
    —— 那种情况下 run_all 恒返回全绿, 比 FAIL 更糟: 它给出假的 VERIFY_OK。
    护栏必须在 -O 下拒绝出结论。"""

    def _env(self):
        return {**os.environ, "PYTHONPATH": str(SRC)}

    def test_run_all_refuses_under_dash_o(self):
        code = ("from embodied_ai import verification;"
                "r = verification.run_all(skip_pytest=True);"
                "print(len(r), r[0][0], r[0][1])")
        p = subprocess.run([sys.executable, "-O", "-c", code],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=self._env(),
                           cwd=str(REPO), timeout=180)
        self.assertEqual(p.returncode, 0, p.stderr[-800:])
        n, name, passed = p.stdout.split()
        self.assertEqual(name, "asserts_stripped")
        self.assertEqual(passed, "False")
        self.assertEqual(n, "1", "-O 下不应执行任何 check (否则会假绿)")

    def test_run_all_executes_checks_without_dash_o(self):
        """对照组: 正常解释器下必须真的跑完全部 CHECKS 项。

        计数动态取 len(verification.CHECKS) —— 与 test_version_consistency 的
        README 口径守护同一真源, 新增 check 时不再需要手改这里。
        """
        from embodied_ai import verification as v
        code = ("from embodied_ai import verification;"
                "print(len(verification.run_all(skip_pytest=True)))")
        p = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=self._env(),
                           cwd=str(REPO), timeout=600)
        self.assertEqual(p.returncode, 0, p.stderr[-800:])
        # run_all 会打印各 check 的摘要, 计数在最后一行
        self.assertEqual(p.stdout.strip().splitlines()[-1],
                         str(len(v.CHECKS)))

    def test_verify_cli_exit_nonzero_under_dash_o(self):
        p = subprocess.run([sys.executable, "-O", "-m", "embodied_ai.cli",
                            "verify", "--no-pytest"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=self._env(),
                           cwd=str(REPO), timeout=300)
        self.assertEqual(p.returncode, 1, "-O 下 verify 必须非零退出")
        self.assertIn("VERIFY_FAIL", p.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
