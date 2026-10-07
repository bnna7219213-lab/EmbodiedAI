"""硬件层 (物理栈 P1-P5 的可执行模型)。

v0.7 新增。定位: 把 docs/04-hardware 里的电路/时序设计从"文档描述"变成
**可数值验证、可导出 HIL 测试用例**的代码, 让"硬件到达之前"也能把控制律、
时序预算与验收判据钉死。
"""
from .foc_shutdown import (FOCConfig, FOCDrive, ShutdownChain, SVPWMModulator,
                           foc_shutdown_demo)
# 双轨 (Path B): 导入即注册 Python 侧后端 (hil → PYTHON_BACKENDS)。
# 必须在此导入, 否则只有显式 import hil_bridge 时后端才可用,
# 运行时 (CLI / verify) 会查不到 hil 的 python 后端。
from . import hil_bridge as _hil_bridge  # noqa: F401

__all__ = ["FOCConfig", "FOCDrive", "ShutdownChain", "SVPWMModulator", "foc_shutdown_demo",
           "hil_bridge"]
