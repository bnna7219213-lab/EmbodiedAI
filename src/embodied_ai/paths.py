"""外部资产路径解析 — 单一真源, 且**永不抛错**。

背景 (为什么需要这个模块):
    仓库里多处要给"外部仓库"找一个默认探针路径 (EdaLab / nanoGPT / ECAD-MCAD /
    SynapticForge)。此前这些位置写死成 `C:\\Users\\<用户名>\\...`, 既把个人机器
    布局连用户名一起提交进公开仓库, 又让任何换机器的探测必然失败。

    改成 `Path.home()` 推导后暴露出第二个坑: **`Path.home()` 在 HOME /
    USERPROFILE 都被剥掉的环境下会 `RuntimeError`** (隔离 CI、最小 env 的
    子进程冒烟测试就是这种场景)。而这些常量在**模块导入期**求值, 一抛错整个
    `embodied_ai` 包都 import 不了 —— tests/test_k7_integration.py 的 CLI
    `--help` 冒烟正是用最小 env 起子进程, 会因此全红。

    本模块把这两件事一次解决: env 覆盖优先, 主目录推导兜底, 兜底失败时退到
    相对路径 (这些默认值**本来就只是猜测**, 猜不中≪ 崩溃)。
"""
from __future__ import annotations

import os
from pathlib import Path

#: 主目录不可知时的退路。用相对路径而非抛错: 这些默认值**本来就只是猜测**。
_FALLBACK_HOME = Path(".")


def home() -> Path:
    """用户主目录; 无法确定时返回相对占位路径而不是抛 RuntimeError。"""
    try:
        return Path.home()
    except RuntimeError:
        return _FALLBACK_HOME


def ext_root(env_var: str, *relative: str) -> Path:
    """外部仓库默认路径: `env_var` 覆盖优先, 否则按主目录下的常见布局猜测。

    返回的路径**不保证存在** —— 调用方按"缺失即降级"处理 (PROJECTS 探针缺失
    不计为 PARTIAL; EdaLab 能力一律以入库的 vendor jar 为先)。
    """
    v = os.environ.get(env_var)
    return Path(v) if v else home().joinpath(*relative)
