"""HIL / 数字孪生域。

G7: 频谱异常判别 (FFT + 轴承/齿轮特征频率) —— 替代 jar 的阈值告警。
导入即注册 `digital_twin` python 后端 (双轨机制)。
"""
from . import twin_fft as _twin_fft  # noqa: F401  (导入即注册 digital_twin 后端)

__all__ = ["twin_fft"]
