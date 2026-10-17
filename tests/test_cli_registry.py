"""test_cli_registry.py — CLI 全局注册表守护 (2026-10-08)。

重构动机: 此前命令的**参数声明**与 **dispatch 映射**分离在 cli.py 两处,
新增命令要改两处 → 极易只改一处 → **命令落在角落无人顾及**。

本测试守护注册表成为**唯一真源**:
  1. 注册表非空且每个命令元信息齐全 (help/分组/示例)
  2. `build_parser()` 启动时从注册表构建, 与注册表**一一对应 (无孤儿)**
  3. 新增命令若绕过注册表直接 add_parser → 会被检出
  4. 重复注册抛错 (防止静默覆盖)
  5. `catalog` 子命令可执行 (可观测手段)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import embodied_ai.cli as cli_mod  # noqa: E402  (导入即完成注册)
from embodied_ai.cli_registry import (  # noqa: E402
    CLI_REGISTRY, CommandSpec, build_parser, command_catalog, command_table,
    register_command,
)


# ============================================================================
# 1. 注册表基本完整性
# ============================================================================

def test_registry_not_empty():
    assert len(CLI_REGISTRY) >= 20, f"注册表命令过少: {len(CLI_REGISTRY)}"


def test_every_command_has_metadata():
    """每条命令必须有 help / 分组 / 至少一个示例 (否则读者无法采纳)。"""
    missing = []
    for name, spec in CLI_REGISTRY.items():
        if not spec.help:
            missing.append((name, "help"))
        if not spec.group or spec.group == "其他":
            missing.append((name, "group"))
        if not spec.examples:
            missing.append((name, "examples"))
    assert not missing, f"命令元信息不全: {missing}"


def test_examples_reference_real_command():
    """示例必须以 'embodied-audit <自身命令名>' 开头 (防文档误导)。"""
    for name, spec in CLI_REGISTRY.items():
        for ex in spec.examples:
            assert ex.startswith(f"embodied-audit {name}"), (
                f"{name} 的示例 {ex!r} 未指向自身命令")


def test_command_table_and_groups():
    tbl = command_table()
    assert len(tbl) == len(CLI_REGISTRY)
    names = {r["name"] for r in tbl}
    assert "verify" in names and "catalog" in names


# ============================================================================
# 2. 唯一真源: parser 与注册表一一对应 (无孤儿)
# ============================================================================

def test_parser_built_from_registry_has_no_orphan():
    """build_parser 产出的命令集合必须等于注册表 (双向)。"""
    ap = build_parser()
    sub = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub.choices) == set(CLI_REGISTRY), (
        "parser 与注册表不一致 — 有命令绕过注册表 (孤儿) 或注册表有未暴露命令")


def test_all_commands_have_executable_handler():
    for name, spec in CLI_REGISTRY.items():
        assert callable(spec.handler), f"{name} handler 不可调用"
        assert spec.handler.__module__.startswith("embodied_ai"), (
            f"{name} handler 不在本项目内: {spec.handler.__module__}")


def test_command_dispatch_works_via_registry():
    """main() 走注册表 dispatch — 用 catalog 做一次真实端到端 (无副作用)。"""
    rc = cli_mod.main(["catalog"])
    assert rc == 0


# ============================================================================
# 3. 反证: 重复注册必须抛错 (防止静默覆盖)
# ============================================================================

def test_duplicate_registration_raises():
    def dummy(ns):
        return 0
    with pytest.raises(ValueError, match="重复注册"):
        register_command("verify", "试图覆盖既有命令")(dummy)


def test_catalog_lists_all_and_mentions_registry():
    """catalog 必须列出全部命令, 并指明来源与新增约定 (可观测性)。"""
    text = command_catalog()
    for name in CLI_REGISTRY:
        assert name in text, f"catalog 漏了命令 {name}"
    assert "注册表" in text
    assert "register_command" in text


def test_duplicate_registration_did_not_corrupt_registry():
    """反证后注册表仍完好 (verify 的 handler 未被 dummy 替换)。"""
    spec = CLI_REGISTRY["verify"]
    assert spec.handler is cli_mod.cmd_verify


# --------------------------------------------------------------------------- #
# 最小 env 下的可导入性 (外部路径解析不得在 import 期抛错)
# --------------------------------------------------------------------------- #

def _minimal_env() -> dict:
    """只留跑起来必需的四项 —— 故意不含 HOME / USERPROFILE / HOMEPATH。"""
    import os
    return {k: v for k, v in {
        "PYTHONPATH": str(REPO / "src"),
        "PYTHONIOENCODING": "utf-8",
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    }.items() if v}


def test_package_imports_without_home_dir():
    """`Path.home()` 在无 HOME/USERPROFILE 时会 RuntimeError, 而这些外部仓库
    默认路径是**模块导入期**求值的 —— 一旦抛错整个包都 import 不了, 表现为
    CLI `--help` 空输出 + 退出码 1 (曾由 adapters/edalab.py 的 EDALAB_ROOT 触发)。
    """
    import subprocess

    code = ("import embodied_ai.paths as p;"
            "p.home();"
            "from embodied_ai.adapters.edalab import EDALAB_ROOT, gradle_cache_dir;"
            "from embodied_ai.adapters.nanogpt import NANOGPT_ROOT;"
            "import embodied_ai.cli;"
            "print('OK', bool(EDALAB_ROOT) and bool(gradle_cache_dir()) and bool(NANOGPT_ROOT))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       cwd=str(REPO), env=_minimal_env(), timeout=180)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-1200:]
    assert r.stdout.decode("utf-8", "replace").strip().startswith("OK True")


def test_cli_help_works_without_home_dir():
    """CLI 冒烟同样必须在最小 env 下可用 (test_k7_integration 依赖这一点)。"""
    import subprocess

    r = subprocess.run([sys.executable, "-m", "embodied_ai.cli", "--help"],
                       capture_output=True, cwd=str(REPO),
                       env=_minimal_env(), timeout=180)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-1200:]
    out = r.stdout.decode("utf-8", "replace")
    for name in ("verify", "catalog", "fusion-demo", "slam-demo", "cog-demo"):
        assert name in out, f"--help 缺命令 {name}"


def test_ext_root_env_var_wins_and_never_raises():
    from embodied_ai.paths import ext_root

    import os
    key = "EMBODIED_TEST_EXT_ROOT"
    os.environ[key] = str(REPO / "vendor")
    try:
        assert ext_root(key, "nope", "Nope") == REPO / "vendor"
    finally:
        del os.environ[key]
    # 无 env、无主目录线索时退到相对路径, 而不是抛 RuntimeError
    p = ext_root(key, "workspace", "EdaLab")
    assert p.parts[-2:] == ("workspace", "EdaLab")