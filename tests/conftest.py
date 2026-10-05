"""pytest 全局夹具 — 统一 sys.path 与子进程环境 (v0.5 新增)。

此前每个测试文件各自 `sys.path.insert(0, <src>)` 并手工拼 PYTHONPATH, 三份重复且
脆弱 (从仓库外目录跑 pytest 时相对路径解析会变)。这里统一注入一次:
  - sys.path: src/ 置顶, 让 `import embodied_ai` 始终指向仓库内的包
  - SUB_ENV:  供 subprocess 调用 CLI/Kotlin 时使用 (含 PYTHONIOENCODING=utf-8,
              避免 Windows GBK 控制台下子进程输出乱码)
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 子进程环境: PATH 必须继承, 否则找不到 python/java
SUB_ENV = {
    **os.environ,
    "PYTHONPATH": str(SRC),
    "PYTHONIOENCODING": "utf-8",
}

REPO = REPO_ROOT  # 便捷别名

# 供测试直接 import 使用
__all__ = ["REPO_ROOT", "REPO", "SRC", "SUB_ENV"]
