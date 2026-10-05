"""终端安全输出层 — 消除 Windows GBK 控制台下的 UnicodeEncodeError。

背景 (v0.5 修复):
  Windows 中文环境默认 stdout 编码为 cp936/gbk, 无法编码 emoji (U+2705/U+274C) 与
  部分符号 (U+26A0/U+2022)。此前 cli.py 直接 print(f"... {'PASS ✅'}") 在
  `embodied-audit audit/safety/couple/dr` 四个子命令上必然抛
  `UnicodeEncodeError: 'gbk' codec can't encode character`, 导致退出码非 0、CI 误红。

策略 (两道防线):
  1. 主动: `ensure_utf8()` 尽力把 stdout/stderr 切成 UTF-8 (errors="replace")。
  2. 被动: 所有面向人的文本经由 `s()` 做「目标编码可编码性」降级 — 不可编码字符
     替换为 ASCII 近义串 (✅→[OK], ❌→[FAIL], ⚠→[WARN], →→->), 保证任何终端都不崩。

状态标记一律使用 `ok()` / `fail()` / `warn()` 返回的 ASCII 安全串, 禁止硬编码 emoji。
"""
from __future__ import annotations

import sys
from typing import Optional

__all__ = [
    "ensure_utf8",
    "stream_encoding",
    "s",
    "ok",
    "fail",
    "warn",
    "emit",
    "banner",
]

# 不可编码时的 ASCII 替换表 (Unicode → ASCII)
_FALLBACK_MAP = {
    "\u2705": "[OK]",
    "\u274c": "[FAIL]",
    "\u26a0": "[WARN]",
    "\u26a0\ufe0f": "[WARN]",
    "\u2192": "->",
    "\u2190": "<-",
    "\u2022": "*",
    "\u2013": "-",
    "\u2014": "-",
    "\u00b1": "+/-",
    "\u2212": "-",
    "\u03bc": "u",
    "\u00b5": "u",
    "\u03c3": "sigma",
    "\u2206": "delta",
}


def ensure_utf8() -> bool:
    """尽力把标准流切成 UTF-8。失败不影响调用方。返回是否成功。"""
    done = False
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
            done = True
        except Exception:  # noqa: BLE001 — 流被重定向/已是 UTF-8 都可能失败
            continue
    return done


def stream_encoding() -> str:
    """当前 stdout 编码 (不可判定则回退 utf-8)。"""
    enc = getattr(sys.stdout, "encoding", None)
    return enc or "utf-8"


def s(text: str, encoding: Optional[str] = None) -> str:
    """把文本降级为「当前终端可编码」的安全串。

    - 先按目标编码整体试编码; 成功则原样返回。
    - 失败则逐字符替换 (见 _FALLBACK_MAP), 仍不可编码的字符丢弃。
    """
    enc = encoding or stream_encoding()
    try:
        text.encode(enc)
        return text
    except (UnicodeEncodeError, LookupError):
        pass

    out = []
    for ch in text:
        if ch in _FALLBACK_MAP:
            out.append(_FALLBACK_MAP[ch])
            continue
        try:
            ch.encode(enc)
            out.append(ch)
        except (UnicodeEncodeError, LookupError):
            continue
    return "".join(out)


def ok() -> str:
    return s("\u2705")


def fail() -> str:
    return s("\u274c")


def warn() -> str:
    return s("\u26a0")


def emit(*parts, sep: str = " ") -> None:
    """安全 print: 自动降级后再输出, 且刷新以避免子进程交错。"""
    line = sep.join(str(p) for p in parts)
    print(s(line), flush=True)


def banner(title: str) -> None:
    emit("")
    emit("===", title, "===")
