"""CLI 全局注册表 — 命令的唯一真源 (2026-10-08)。

问题 (本次重构的动机): 此前 `cli.py` 里命令的**参数声明** (`add_parser` /
`add_argument`, L733-823) 与 **dispatch 映射** (`cmds` 字典, L825-833) 分离在
两处。新增命令要改两个地方, **很容易只改一处 → 命令落在角落无人顾及**,
也无法回答"到底有哪些命令可用 / 各自怎么用"。

本模块把三件事收进一处 (`CommandSpec`):
  1. 命令身份 (name / help / 分组 / 引入版本)
  2. 参数声明 (`add_args` 回调, 内含 argparse 调用)
  3. 执行入口 (handler)

并由此派生:
  - `build_parser()` — 启动时读注册表构建 argparse (**单一真源**, 不再散落)
  - `command_catalog()` — 生成可读命令目录 (CLI `catalog` 子命令 / 文档 / demo)

**约定**: 新增任何调用方式**必须**经 `register_command()` 注册, 不允许绕过注册表
直接 `sub.add_parser`。`tests/test_cli_registry.py` 会守护这条约定
(注册表与实际可执行命令必须一一对应, 无孤儿)。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["CommandSpec", "CLI_REGISTRY", "register_command",
           "build_parser", "command_catalog", "command_table", "groups"]


@dataclass
class CommandSpec:
    """一个 CLI 命令的完整定义 (身份 + 参数 + 入口 + 示例)。"""

    name: str
    help: str
    handler: Callable[[argparse.Namespace], int]
    group: str = "其他"
    since: str = "v0.7"
    description: str = ""
    add_args: Optional[Callable[[argparse.ArgumentParser], None]] = None
    examples: List[str] = field(default_factory=list)

    def attach(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """把本命令的参数声明挂到 parser 上 (由 build_parser 调用)。"""
        if self.add_args is not None:
            self.add_args(parser)
        return parser


class _LazyRegistry(dict):
    """注册表字典 — 首次访问时按需触发命令注册。

    动机 (测试抓到的陷阱): 命令注册实际发生在 `cli` 模块导入时 (`_register_all`)。
    若消费方只 `import cli_registry` 就查注册表, 会看到**空表**并误判"没有命令"
    —— 这正是"命令落在角落"的另一种形态。本子类让任何读取访问自动补齐注册,
    使"注册表是唯一真源"在**任何导入顺序**下都成立。
    """

    def _ensure(self) -> None:
        ensure_registered()

    def __contains__(self, key):            # type: ignore[override]
        self._ensure()
        return super().__contains__(key)

    def __getitem__(self, key):             # type: ignore[override]
        self._ensure()
        return super().__getitem__(key)

    def __iter__(self):                     # type: ignore[override]
        self._ensure()
        return super().__iter__()

    def __len__(self) -> int:
        self._ensure()
        return super().__len__()

    def get(self, key, default=None):       # type: ignore[override]
        self._ensure()
        return super().get(key, default)

    def keys(self):                         # type: ignore[override]
        self._ensure()
        return super().keys()

    def values(self):                       # type: ignore[override]
        self._ensure()
        return super().values()

    def items(self):                        # type: ignore[override]
        self._ensure()
        return super().items()


#: 全局注册表 — 命令名 → CommandSpec。**唯一真源**, 启动时由此构建 parser。
#: 访问自动触发注册 (_LazyRegistry), 因此不需要关心导入顺序。
CLI_REGISTRY: Dict[str, CommandSpec] = _LazyRegistry()

#: 重入保护: cli 自身导入期间会调用 register_command → 触发 ensure → 直接返回,
#: 避免"导入尚未完成的模块"造成死循环或空表。
_registration_attempted = False


def ensure_registered() -> None:
    """确保注册表已填充 (按需触发 `cli` 模块导入)。

    正常路径: `cli` 导入时 `_register_all()` 已注册全部命令 → 本函数是空操作。
    仅导入 `cli_registry` 的消费方: 本函数补一次导入, 使注册表可用 (消除空表陷阱)。

    **必须排除 `python -m embodied_ai.cli` 的情形**: 此时 cli.py 作为 `__main__`
    正在执行, 其 `_register_all()` 会完成注册; 若此处再 import `embodied_ai.cli`,
    会**二次执行** cli.py → 重复注册 → `register_command` 抛 ValueError。
    (此缺陷由 tests/test_selfcheck.py 的 CLI 子进程测试抓到。)
    """
    global _registration_attempted
    if dict.__len__(CLI_REGISTRY) > 0:
        return
    if _registration_attempted:
        return

    import sys as _sys
    # cli 正在作为 __main__ 执行 → 注册由其自身完成, 本函数直接返回
    main_mod = _sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)
    if main_file:
        try:
            from pathlib import Path as _P
            if _P(main_file).resolve() == (_P(__file__).resolve().parent / "cli.py"):
                return
        except (ValueError, OSError):
            pass

    _registration_attempted = True
    import importlib
    if "embodied_ai.cli" not in _sys.modules:
        importlib.import_module("embodied_ai.cli")


def register_command(name: str, help: str, *, group: str = "其他",
                     since: str = "v0.7", description: str = "",
                     add_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
                     examples: Optional[List[str]] = None) -> Callable:
    """注册一个 CLI 命令 (装饰器)。handler 必须是 `(Namespace) -> int`。

    新增命令**必须**走这里 —— 这是"不落在角落"的机制保证。
    """
    def deco(fn: Callable[[argparse.Namespace], int]) -> Callable:
        if name in CLI_REGISTRY:
            raise ValueError(f"命令 {name} 重复注册 (注册表是唯一真源, 不允许覆盖)")
        CLI_REGISTRY[name] = CommandSpec(
            name=name, help=help, handler=fn, group=group, since=since,
            description=description or (fn.__doc__ or "").strip(),
            add_args=add_args, examples=list(examples or []),
        )
        return fn
    return deco


def build_parser(prog: str = "embodied-audit",
                 description: str = "EmbodiedAI 统一 CLI") -> argparse.ArgumentParser:
    """**从注册表**构建 argparse —— 启动时读全局注册表, 不再逐条 add_parser。"""
    ap = argparse.ArgumentParser(prog=prog, description=description)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in sorted(CLI_REGISTRY):
        spec = CLI_REGISTRY[name]
        p = sub.add_parser(name, help=spec.help)
        spec.attach(p)
    return ap


def command_table() -> List[Dict[str, Any]]:
    """结构化命令清单 (供测试 / 文档生成 / 校验)。"""
    return [
        {
            "name": s.name, "help": s.help, "group": s.group, "since": s.since,
            "examples": list(s.examples), "has_args": s.add_args is not None,
        }
        for s in sorted(CLI_REGISTRY.values(), key=lambda x: (x.group, x.name))
    ]


def groups() -> List[str]:
    """按分组返回命令名 (目录展示用)。"""
    g: Dict[str, List[str]] = {}
    for s in sorted(CLI_REGISTRY.values(), key=lambda x: (x.group, x.name)):
        g.setdefault(s.group, []).append(s.name)
    return g


def command_catalog() -> str:
    """生成可读的命令目录 (分组 + 说明 + 示例), 供 `catalog` 子命令与文档使用。"""
    lines: List[str] = ["EmbodiedAI CLI 命令目录 (来源: 全局注册表 CLI_REGISTRY)", ""]
    gmap: Dict[str, List[CommandSpec]] = {}
    for s in CLI_REGISTRY.values():
        gmap.setdefault(s.group, []).append(s)
    for g in sorted(gmap):
        lines.append(f"[{g}]")
        for s in sorted(gmap[g], key=lambda x: x.name):
            lines.append(f"  embodied-audit {s.name:<12s} {s.help}")
            for ex in s.examples:
                lines.append(f"      例: {ex}")
        lines.append("")
    lines.append(f"共 {len(CLI_REGISTRY)} 个命令, {len(gmap)} 个分组。")
    lines.append("新增命令请经 register_command() 注册 (保证不落在角落)。")
    return "\n".join(lines)