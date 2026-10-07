"""v0.8: 单仓自足验证 — vendor jar 消费模式 (无外部 EdaLab / gradle 依赖)。

这是 "jar 入库 + 消费模式" 决策的回归守护:
  1. vendor/edalab/ 两件 jar 必须存在且可被解析
  2. EDALAB_HOME 指向不存在路径时, 仍能 LIVE_OK (证明不依赖外部 EdaLab)
  3. classpath 构造不依赖 gradle 缓存 (vendor 优先)
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from embodied_ai.adapters.edalab import (VENDOR_DIR, build_classpath,
                                          resolve_jar_paths)


class TestVendoredJars(unittest.TestCase):
    """jar 入库完整性。"""

    def test_vendor_jars_present(self):
        core = VENDOR_DIR / "edalabcore.jar"
        stdlib = sorted(VENDOR_DIR.glob("kotlin-stdlib-*.jar"))
        self.assertTrue(core.exists(), f"vendor core jar 缺失: {core}")
        self.assertTrue(stdlib, f"vendor kotlin-stdlib 缺失: {VENDOR_DIR}")
        # 体积合理性 (core ~1.9MB, stdlib ~1.7MB)
        self.assertGreater(core.stat().st_size, 1_000_000)
        self.assertGreater(stdlib[-1].stat().st_size, 1_000_000)

    def test_resolve_prefers_vendor(self):
        """解析优先级: vendor 优先于外部 EdaLab。"""
        p = resolve_jar_paths()
        self.assertEqual(p["source"], "vendor")
        self.assertTrue(str(p["core"]).startswith(str(REPO)),
                        f"core 应在本仓内: {p['core']}")

    def test_classpath_contains_vendor_first(self):
        cp = build_classpath()
        self.assertIsNotNone(cp)
        parts = cp.split(";")
        # 路径分隔符归一化后比较 (build_classpath 输出正斜杠)
        vendored = str(VENDOR_DIR).replace("\\", "/")
        self.assertTrue(parts[0].replace("\\", "/").startswith(vendored),
                        f"首个 classpath 条目应为 vendor jar: {parts[0]}")


class TestNoExternalDependency(unittest.TestCase):
    """隔离验证: 外部 EdaLab 不存在时仍可用 (单仓自足的决定性证据)。"""

    def test_live_with_missing_external_root(self):
        """EDALAB_HOME 指向不存在路径 → 仍 LIVE_OK (走 vendor)。"""
        env = {**os.environ, "EDALAB_HOME": "C:/nonexistent/edalab_for_test",
               "PYTHONPATH": str(SRC)}
        code = (
            "import os;"
            "os.environ['EDALAB_HOME']='C:/nonexistent/edalab_for_test';"
            "from embodied_ai.adapters.edalab import EdalabAdapter, resolve_jar_paths;"
            "p=resolve_jar_paths();"
            "assert p['source']=='vendor', p;"
            "r=EdalabAdapter().run_stage('relay_clutch', {}, mode='live');"
            "assert r['status']=='LIVE_OK', r.get('error','');"
            "assert '16.00ms' in r.get('stdout',''), '引擎输出缺失';"
            "print('ISOLATED_OK')"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              env=env, timeout=180, cwd=str(REPO))
        self.assertIn("ISOLATED_OK", proc.stdout,
                      f"隔离验证失败:\n{proc.stdout}\n{proc.stderr[-500:]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
