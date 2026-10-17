"""机器人 Linux OS SSH — 双通道维护接口 (plan_v6 Sprint N: N3)。

**缺口背景（OS 通道断点）**
    机器人跑 Linux on ARM（Jetson / WSL）。需要一条 **SSH 维护通道**:
    远程 debug、修历史 bug、编写/推送出厂初始程序（机器人认知本体+程序）。
    用户明确: SSH **不用于遥控** —— 遥控是实时 runtime 与机器人协作、
    互相完成下一步规划动作; SSH 是让机器人**暂时停下来**做维护。

**双通道语义（核心区分, docstring 必须写清）**
    - `RUNTIME` 通道: 实时协作规划（人机互相配合完成下一步动作）, 20Hz 控制环内
    - `MAINTENANCE` 通道: SSH 远程维护（停机修 bug / 装程序 / 拉日志）, **非遥控**

**模式状态机（维护态停机流转）**
    RUNTIME → PAUSED → MAINTENANCE → PROVISIONED → RUNTIME
    铁律: **仅 RUNTIME 允许运动指令**; MAINTENANCE/PAUSED/PROVISIONED 下
    `can_send_motion()` 必须 False（维护通道绝不遥控）。

**设计（最小可用，不过度设计）**
    - `SshCommandBuilder`: 纯函数生成 ssh/scp/出厂脚本命令行
    - `CommandExecutor` 协议 + `DryRunExecutor`（默认, 只记录不执行）
      + `SubprocessExecutor`（真跑系统 ssh/scp — 不依赖 paramiko）
    - `OsChannelAdmin`: 维护工作流封装（进维护态/拉日志/装程序/出厂烧录/回运行态）

**不做的（诚实边界）**
    - 不做 paramiko 真实 SSH 连接管理（系统 ssh/scp 足够, 零第三方依赖）
    - 不做遥控遥操作（那是 runtime 通道的事, 见 cognition/closed_loop.py）
    - 不做密钥管理/known_hosts 策略（属运维配置）

**反证要求（测试守护）**
    状态机全转换表（合法边+非法边）; MAINTENANCE 下拒收运动指令;
    dry-run 命令序列含 ssh/scp token; provisioning 序列完整;
    未进维护态就 push/pull 必须报错。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, Tuple


# ══════════════════════════════════════════════════════════════════
# 通道与模式
# ══════════════════════════════════════════════════════════════════

class ChannelKind(Enum):
    """双通道: RUNTIME=实时协作规划(非遥控命令) / MAINTENANCE=SSH 停机维护。"""
    RUNTIME = "runtime"
    MAINTENANCE = "maintenance"


class ModeState(Enum):
    """机器人 OS 模式状态机。"""
    RUNTIME = "runtime"          # 正常运行, 人机实时协作规划
    PAUSED = "paused"            # 暂停 (维护前的安全过渡)
    MAINTENANCE = "maintenance"  # SSH 维护态 (停机修 bug / 装程序)
    PROVISIONED = "provisioned"  # 出厂程序已烧录, 待回运行态


#: 合法转换表 (全边列举; 未列出的边一律非法)
ALLOWED_TRANSITIONS = {
    ModeState.RUNTIME: {ModeState.PAUSED},
    ModeState.PAUSED: {ModeState.MAINTENANCE, ModeState.RUNTIME},
    ModeState.MAINTENANCE: {ModeState.PROVISIONED, ModeState.PAUSED},
    ModeState.PROVISIONED: {ModeState.RUNTIME},
}


class ModeController:
    """模式状态机 — 维护态停机流转 + 运动指令门禁。"""

    def __init__(self, initial: ModeState = ModeState.RUNTIME):
        self.state = initial
        self.history: List[ModeState] = [initial]

    def transition(self, target: ModeState) -> bool:
        """尝试转换; 非法转换抛 ValueError (不静默失败)。"""
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(
                f"非法模式转换 {self.state.value} → {target.value}; "
                f"合法目标: {[s.value for s in ALLOWED_TRANSITIONS[self.state]]}")
        self.state = target
        self.history.append(target)
        return True

    def can_send_motion(self) -> bool:
        """运动指令门禁 — 仅 RUNTIME 允许 (维护通道绝不遥控)。"""
        return self.state is ModeState.RUNTIME

    def require_motion_allowed(self) -> None:
        if not self.can_send_motion():
            raise PermissionError(
                f"模式 {self.state.value} 下禁止运动指令 "
                f"(维护/暂停态只做维护, 不遥控)")


# ══════════════════════════════════════════════════════════════════
# 命令构建 + 执行器
# ══════════════════════════════════════════════════════════════════

class SshCommandBuilder:
    """ssh/scp/出厂脚本命令行生成（纯函数, 不执行）。"""

    def __init__(self, user: str = "robot", ssh_opts: Optional[List[str]] = None):
        self.user = user
        self.ssh_opts = ssh_opts if ssh_opts is not None else ["-o", "BatchMode=yes"]

    def build_ssh_cmd(self, host: str, argv: List[str]) -> List[str]:
        """远程执行命令: ssh [opts] user@host -- argv…"""
        return ["ssh", *self.ssh_opts, f"{self.user}@{host}", "--", *argv]

    def build_scp_cmd(self, local: str, remote: str, host: str,
                      to_remote: bool = True) -> List[str]:
        """文件传输: scp [opts] local user@host:remote (或反向)。"""
        remote_ref = f"{self.user}@{host}:{remote}"
        src, dst = (local, remote_ref) if to_remote else (remote_ref, local)
        return ["scp", *self.ssh_opts, src, dst]

    def factory_program_script(self, program_name: str) -> str:
        """出厂初始程序脚本 — 机器人认知本体 + 启动程序清单。

        这是"机器人出厂时要跑的第一段程序": 建立认知本体目录、
        写入启动清单、挂上 systemd 用户服务（内容为模板, 待领域验证）。
        """
        return (
            "#!/bin/sh\n"
            "# 出厂初始程序 — 认知本体 + 启动清单 (provenance=待领域验证)\n"
            "set -e\n"
            f"NAME={program_name}\n"
            "mkdir -p /opt/embodied/cognition\n"
            "cat > /opt/embodied/cognition/manifest.txt <<'EOF'\n"
            "1. center_brain (中枢编排)\n"
            "2. cognitive_hub (四要素记忆)\n"
            "3. tactile_closed_loop (触觉闭环)\n"
            "EOF\n"
            "echo \"$NAME provisioned at $(date -Iseconds)\" "
            "> /opt/embodied/cognition/PROVISIONED\n"
            "echo PROVISION_OK\n"
        )


@dataclass
class ExecResult:
    """执行结果。"""
    argv: List[str]
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class CommandExecutor(Protocol):
    """执行器协议 — 可注入 (默认 DryRun, 真机用 Subprocess)。"""
    def run(self, argv: List[str]) -> ExecResult: ...


class DryRunExecutor:
    """默认执行器 — 只记录命令序列, 不真执行 (测试/CI/预演)。"""

    def __init__(self):
        self.recorded: List[List[str]] = []

    def run(self, argv: List[str]) -> ExecResult:
        self.recorded.append(list(argv))
        return ExecResult(argv=list(argv), returncode=0,
                          stdout="DRY_RUN", stderr="")


class SubprocessExecutor:
    """真执行器 — 调系统 ssh/scp (零第三方依赖)。"""

    def run(self, argv: List[str]) -> ExecResult:
        p = subprocess.run(argv, capture_output=True, text=True)
        return ExecResult(argv=list(argv), returncode=p.returncode,
                          stdout=p.stdout, stderr=p.stderr)


# ══════════════════════════════════════════════════════════════════
# OsChannelAdmin — 维护工作流
# ══════════════════════════════════════════════════════════════════

class OsChannelAdmin:
    """SSH 维护通道工作流 — 全部经注入 executor (默认 dry-run)。

    典型工作流:
        admin.enter_maintenance("fix historical bug")
        admin.pull_debug_logs("/var/log/embodied/")
        admin.push_program("dist/new_feature.py", "/opt/embodied/")
        admin.provision_factory_program("cognition_v2")
        admin.return_to_runtime()
    """

    def __init__(self, host: str,
                 executor: Optional[CommandExecutor] = None,
                 builder: Optional[SshCommandBuilder] = None,
                 mode: Optional[ModeController] = None):
        self.host = host
        self.executor = executor if executor is not None else DryRunExecutor()
        self.builder = builder if builder is not None else SshCommandBuilder()
        self.mode = mode if mode is not None else ModeController()

    def _require_maintenance(self) -> None:
        if self.mode.state not in (ModeState.MAINTENANCE, ModeState.PROVISIONED):
            raise PermissionError(
                f"维护操作要求 MAINTENANCE/PROVISIONED 态, 当前 "
                f"{self.mode.state.value} — 先 enter_maintenance()")

    # ---------------- 流转 ---------------- #

    def enter_maintenance(self, reason: str) -> ExecResult:
        """RUNTIME→PAUSED→MAINTENANCE: 停机进入 SSH 维护态。"""
        if not reason:
            raise ValueError("必须给出维护原因 (审计)")
        if self.mode.state is ModeState.RUNTIME:
            self.mode.transition(ModeState.PAUSED)
        self.mode.transition(ModeState.MAINTENANCE)
        return self.executor.run(self.builder.build_ssh_cmd(
            self.host, ["systemctl", "--user", "stop", "embodied-runtime",
                        f"# reason: {reason}"]))

    def return_to_runtime(self) -> ExecResult:
        """PROVISIONED/MAINTENANCE/PAUSED → RUNTIME。"""
        if self.mode.state is ModeState.MAINTENANCE:
            self.mode.transition(ModeState.PAUSED)
        elif self.mode.state is ModeState.PROVISIONED:
            self.mode.transition(ModeState.RUNTIME)
            return self.executor.run(self.builder.build_ssh_cmd(
                self.host, ["systemctl", "--user", "start", "embodied-runtime"]))
        self.mode.transition(ModeState.RUNTIME)
        return self.executor.run(self.builder.build_ssh_cmd(
            self.host, ["systemctl", "--user", "start", "embodied-runtime"]))

    # ---------------- 维护操作 ---------------- #

    def pull_debug_logs(self, remote_dir: str,
                        local_dir: str = "./logs") -> ExecResult:
        """拉取远程 debug 日志 (修历史 bug 用)。"""
        self._require_maintenance()
        return self.executor.run(self.builder.build_scp_cmd(
            local_dir, remote_dir, self.host, to_remote=False))

    def push_program(self, local_path: str, remote_path: str) -> ExecResult:
        """推送新功能/修复程序到机器人。"""
        self._require_maintenance()
        return self.executor.run(self.builder.build_scp_cmd(
            local_path, remote_path, self.host, to_remote=True))

    def provision_factory_program(self, program_name: str = "cognition"
                                  ) -> ExecResult:
        """烧录出厂初始程序 (认知本体+启动清单) → PROVISIONED。

        脚本先落本地临时文件再 scp 推送 (scp 的本地端必须是真实文件)。
        """
        import os
        import tempfile

        self._require_maintenance()
        script = self.builder.factory_program_script(program_name)
        fd, local_script = tempfile.mkstemp(prefix="factory_provision_",
                                            suffix=".sh")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(script)
            # 1) 推脚本  2) 远程执行  3) 转 PROVISIONED
            push = self.executor.run(self.builder.build_scp_cmd(
                local_script, "/tmp/factory_provision.sh", self.host,
                to_remote=True))
        finally:
            os.unlink(local_script)
        self.executor.run(self.builder.build_ssh_cmd(
            self.host, ["sh", "/tmp/factory_provision.sh"]))
        if self.mode.state is ModeState.MAINTENANCE:
            self.mode.transition(ModeState.PROVISIONED)
        return push

    # ---------------- 可观测 ---------------- #

    @property
    def channel(self) -> ChannelKind:
        """当前活跃通道: RUNTIME 态=协作规划通道, 否则=SSH 维护通道。"""
        return (ChannelKind.RUNTIME if self.mode.state is ModeState.RUNTIME
                else ChannelKind.MAINTENANCE)

    def status(self) -> dict:
        return {
            "host": self.host,
            "mode": self.mode.state.value,
            "channel": self.channel.value,
            "can_send_motion": self.mode.can_send_motion(),
            "executed": [list(a) for a in getattr(self.executor, "recorded", [])],
        }
