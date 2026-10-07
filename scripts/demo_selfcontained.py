"""单仓自足实证 demo — v4「jar 入库 + 消费」的可运行证明（薄壳）。

**核心逻辑在 `src/embodied_ai/selfcheck.py`**，本脚本只做入口转发 —— 避免
"脚本一份、CLI 一份"两处维护（项目规矩：单一真源）。

**为什么要有这个 demo**
    "单仓自足"不能只写在 README 里 —— 新加入的人（或三个月后的自己）必须能
    **一条命令跑出证据**。本 demo 就是那个证据：逐项断言，任何一项不成立即失败。

**它证明什么**（5 项，每项都有可证伪的断言）
    1. 载体齐全     vendor/edalab 两件 jar 存在（core 不自含 Kotlin 运行时的配套关系）
    2. 指纹一致     jar 的 sha256/size 与 PROVENANCE.json 一致（未被偷换/静默替换）
    3. 运行时可用    仅需 JRE（不需要 JDK / gradle / 外部 EdaLab 目录）
    4. 隔离可跑     EDALAB_HOME + GRADLE_USER_HOME 指向**不存在**路径时仍真实执行引擎
    5. 后端可观测    调用结果带 jar_source（能回答"这次跑的是哪份产物"）

**用法**
    python scripts/demo_selfcontained.py            # 人读：逐项 ✓/✗ + 失败修复指引
    python scripts/demo_selfcontained.py --json     # 机读：JSON（CI 断言 exit code）

    等价 CLI（已注册进全局注册表，见 `catalog`）:
    embodied-audit selfcheck [--json]

**退出码**：0 = 全部通过；1 = 有项失败（输出指明哪项 + 怎么修，不静默）
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from embodied_ai.selfcheck import cli_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli_main())
