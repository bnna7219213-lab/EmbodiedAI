"""目标平台能力层 — Linux on ARM (aarch64) 的探测 / 就绪评估 / 部署产物生成。

背景 (v0.6):
  docs/04-hardware/embedded-linux-arm.md 详细写了 ARM 裸机 → UBOOT → 根文件系统 →
  内核调试的全栈知识, 但仓库里**没有任何一行代码**消费它: 目标板能不能跑、
  CAN 在不在、是不是 PREEMPT_RT、有没有 NPU、isolcpus 配没配 —— 全靠人肉登板看。
  于是"Jetson Orin 20Hz 推理"这条硬指标在 CI 与交付前都没有可判定的闸门。

  本包把这件事变成**可离线判定**的能力:
    1. probe_host()          — 读 /proc + /sys + /dev 形成目标画像 (也可对快照离线跑)
    2. evaluate_readiness()  — 对照 contracts/hw_requirements.yaml 逐项判定
                               (INFO / WARNING / BLOCKING, 不可判定的项标 N/A 而非假失败)
    3. arm_budget_overrides()— 无 NPU/GPU 时对 50ms 端到端预算做保守缩放
    4. render_systemd_units()— 按探测到的 isolcpus 生成 CPUAffinity 单元文件
    5. `embodied-audit platform` — CLI 入口, 退出码即结论

可测性: 所有输入都走文件系统, 因此可以用 tmp 目录造一份"目标板快照"在
x86 Windows 开发机上完整验证 (见 tests/test_k5_arm_linux.py)。
"""

from .arm_linux import (
    ArmCheck,
    ArmLatencyProfile,
    ArmLinuxTarget,
    ArmRequirements,
    KNOWN_BOARDS,
    arm_budget_overrides,
    arm_latency_profile,
    evaluate_readiness,
    probe_host,
    render_systemd_units,
    report_lines,
    summarize,
    target_to_dict,
    to_json,
)

__all__ = [
    "ArmCheck", "ArmLinuxTarget", "ArmRequirements", "ArmLatencyProfile",
    "KNOWN_BOARDS", "probe_host", "evaluate_readiness", "summarize",
    "report_lines", "target_to_dict", "to_json", "arm_latency_profile",
    "arm_budget_overrides", "render_systemd_units",
]
