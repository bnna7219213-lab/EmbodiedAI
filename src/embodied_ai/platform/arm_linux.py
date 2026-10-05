"""Linux on ARM (aarch64) 目标能力探测与就绪评估 (v0.6 新增)。

设计要点
--------
1. **纯文件输入**: 只读 /proc、/sys、/dev、/etc。因此可以对"目标板快照目录"
   离线评估 —— 在 x86 Windows 开发机 / CI 上也能完整验证本模块。
2. **架构从 /proc/cpuinfo 推断**: ARM Linux 的 cpuinfo 有 `CPU architecture: 8`
   (ARMv8/aarch64) 或 `7` (ARMv7)。这样连 arch 也是文件驱动的, 无需 override。
   读不到时回退 platform.machine()。
3. **不可判定 ≠ 失败**: 缺少 /proc/cpuinfo (非 Linux 主机) 时, NEON / RT 内核 /
   cpufreq 等项标记为 applicable=False (N/A), 而不是假 FAIL。此前很多"跨平台
   探测"脚本在 Windows 上全项爆红, 就是这个没做。
4. **需求来自契约**: ArmRequirements.from_contract() 从
   contracts/hw_requirements.yaml 取 target_board / 内存 / CAN / GPU 预处理需求,
   避免又一份硬编码。

与既有能力的关系
----------------
  deploy.latency_audit  → 50ms 端到端预算闸门
  本模块 arm_budget_overrides → 无加速器时给出保守缩放后的预算, 喂回同一闸门
"""

from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
AARCH64 = {"aarch64", "arm64", "aarch64_be"}
ARM32 = {"armv7l", "armv8l", "armv6l", "arm"}

# 已知目标板 (与 contracts/hw_requirements.compute.target_board 对齐)
# memory_gb: 标称内存 (用于就绪评估基准); accel: 该板应提供的加速器名
KNOWN_BOARDS: Dict[str, Dict[str, Any]] = {
    "jetson_orin_nx_16gb": {
        "vendor": "nvidia", "memory_gb": 16.0, "accel": "nvidia-gpu",
        "match": ["orin nx", "jetson orin nx", "p3767"],
    },
    "jetson_orin_nano": {
        "vendor": "nvidia", "memory_gb": 8.0, "accel": "nvidia-gpu",
        "match": ["orin nano", "jetson orin nano", "p3768"],
    },
    "jetson_agx_orin": {
        "vendor": "nvidia", "memory_gb": 32.0, "accel": "nvidia-gpu",
        "match": ["agx orin", "jetson agx orin", "p3737"],
    },
    "rk3588": {
        "vendor": "rockchip", "memory_gb": 8.0, "accel": "rockchip-vpu",
        "match": ["rk3588", "rockchip rk3588"],
    },
    "imx8mp": {
        "vendor": "nxp", "memory_gb": 4.0, "accel": "drm",
        "match": ["imx8mp", "i.mx8m plus"],
    },
}

# ARMv8 基线 SIMD (aarch64 必带 asimd; 部分内核只列 neon)
_SIMD_FEATURES = ("asimd", "neon")


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
@dataclass
class ArmLinuxTarget:
    """一台 (真实或快照) Linux on ARM 目标机的画像。"""

    arch: str = ""
    is_linux: bool = True
    is_arm: bool = False
    is_aarch64: bool = False

    kernel_release: str = ""
    soc_model: str = ""
    vendor: str = "unknown"
    board_profile: str = ""

    cpu_count: int = 0
    cpu_features: Optional[List[str]] = None     # None = 无法判定 (无 cpuinfo)
    accelerators: List[str] = field(default_factory=list)

    rt_kernel: Optional[bool] = None             # None = 无法判定
    governor: str = ""
    isolated_cpus: List[int] = field(default_factory=list)
    mem_total_gb: Optional[float] = None
    thermal_c: Optional[float] = None
    can_ifaces: List[str] = field(default_factory=list)

    # 探测来源 (便于报告里说明"这是快照还是真机")
    source_root: str = "/"

    def has_feature(self, feature: str) -> bool:
        return bool(self.cpu_features) and feature in self.cpu_features

    @property
    def has_accelerator(self) -> bool:
        return bool(self.accelerators)


@dataclass
class ArmRequirements:
    """就绪评估的期望值 — 默认与 contracts/hw_requirements.yaml 一致。"""

    arch: str = "aarch64"
    board: str = "jetson_orin_nx_16gb"
    memory_gb: float = 16.0
    can_required: bool = True
    accel_required: bool = True
    rt_recommended: bool = True
    end_to_end_budget_ms: float = 50.0

    @classmethod
    def from_contract(cls) -> "ArmRequirements":
        """从契约取需求 (唯一真源), 契约缺失时回退默认值。"""
        try:
            from ..config import load_contract
            hw = load_contract("hw_requirements") or {}
        except Exception:  # noqa: BLE001
            return cls()

        compute = hw.get("compute", {}) or {}
        board = compute.get("target_board") or cls.board

        # 板名里的 "16gb" / "8gb" 就是标称内存
        mem = cls.memory_gb
        m = re.search(r"(\d+)\s*gb", str(board), re.IGNORECASE)
        if m:
            mem = float(m.group(1))

        accel_required = bool((compute.get("preprocessing") or {})
                              .get("gpu_preprocess_required", True))
        bus = str((hw.get("communication") or {}).get("bus", "can_fd")).lower()

        return cls(
            arch=cls.arch,
            board=str(board),
            memory_gb=mem,
            can_required=bus.startswith("can"),
            accel_required=accel_required,
            rt_recommended=cls.rt_recommended,
            end_to_end_budget_ms=float(compute.get("end_to_end_inference_budget_ms", 50.0)),
        )


@dataclass
class ArmCheck:
    """单项就绪检查。

    ok=True  → verdict INFO
    ok=False → verdict = severity (WARNING / BLOCKING)
    applicable=False → verdict N/A (输入不可得, 不作判定)
    """

    name: str
    ok: bool
    severity: str = "WARNING"      # 失败时的严重度
    detail: str = ""
    applicable: bool = True

    @property
    def verdict(self) -> str:
        if not self.applicable:
            return "N/A"
        return "INFO" if self.ok else self.severity


@dataclass
class ArmLatencyProfile:
    """ARM 目标上的延迟预算缩放建议。"""

    model_inference_scale: float = 1.0
    sensor_preprocess_scale: float = 1.0
    accelerator: str = ""
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 低层读取工具
# --------------------------------------------------------------------------- #
def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _infer_arch(cpuinfo: Optional[str], fallback: str) -> str:
    """从 /proc/cpuinfo 推断架构 (ARM Linux 专有字段), 读不到用 fallback。"""
    if cpuinfo:
        m = re.search(r"^CPU architecture\s*:\s*(\S+)", cpuinfo, re.MULTILINE)
        if m:
            v = m.group(1).strip()
            if v in ("8", "aarch64"):
                return "aarch64"
            return "armv7l"
        if "aarch64" in cpuinfo:
            return "aarch64"
    return fallback


def _parse_features(cpuinfo: Optional[str]) -> Optional[List[str]]:
    if not cpuinfo:
        return None
    feats: List[str] = []
    for line in cpuinfo.splitlines():
        if line.lower().startswith("features"):
            feats.extend(tok.strip() for tok in line.split(":", 1)[1].split())
    return feats or None


def _parse_isolcpus(cmdline: Optional[str]) -> List[int]:
    if not cmdline:
        return []
    m = re.search(r"isolcpus=([^\s]+)", cmdline)
    if not m:
        return []
    cpus: List[int] = []
    for part in m.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                lo, hi = (int(x) for x in part.split("-", 1))
                cpus.extend(range(lo, hi + 1))
            except ValueError:
                continue
        else:
            try:
                cpus.append(int(part))
            except ValueError:
                continue
    return sorted(set(cpus))


def _detect_board(model: str) -> str:
    low = (model or "").lower()
    for name, info in KNOWN_BOARDS.items():
        for pat in info["match"]:
            if pat in low:
                return name
    return ""


def _detect_vendor(model: str, etc_root: Path) -> str:
    if _exists(etc_root / "nv_tegra_release"):
        return "nvidia"
    low = (model or "").lower()
    if "rockchip" in low or re.search(r"\brk\d{4}", low):
        return "rockchip"
    if "imx" in low or "nxp" in low:
        return "nxp"
    for name, info in KNOWN_BOARDS.items():
        if _detect_board(model) == name:
            return info["vendor"]
    return "unknown"


def _detect_accelerators(dev_root: Path, etc_root: Path, vendor: str) -> List[str]:
    acc: List[str] = []
    if _exists(etc_root / "nv_tegra_release") or any(dev_root.glob("nvidia*")):
        acc.append("nvidia-gpu")
    if _exists(dev_root / "mpp_service"):
        acc.append("rockchip-vpu")
    if _exists(dev_root / "dri") or any(dev_root.glob("dri")):
        acc.append("drm")
    if any(dev_root.glob("video*")):
        acc.append("v4l2")
    return acc


def _detect_can(sysfs: Path, procfs: Path) -> List[str]:
    """CAN 接口: /sys/class/net/<if>/uevent 里 DEVTYPE=can (无 sysfs 时退 /proc/net/dev)。"""
    nets = sysfs / "class" / "net"
    found: List[str] = []
    try:
        for entry in sorted(nets.iterdir()):
            uevent = _read(entry / "uevent") or ""
            if "DEVTYPE=can" in uevent:
                found.append(entry.name)
    except OSError:
        pass
    if found:
        return found
    dev = _read(procfs / "net" / "dev") or ""
    return [line.split(":")[0].strip()
            for line in dev.splitlines()[2:]
            if line.split(":")[0].strip().startswith("can")]


def _detect_thermal(sysfs: Path) -> Optional[float]:
    base = sysfs / "class" / "thermal"
    try:
        zones = sorted(p for p in base.iterdir() if p.name.startswith("thermal_zone"))
    except OSError:
        return None
    temps = []
    for z in zones:
        raw = _read(z / "temp")
        if raw:
            try:
                temps.append(float(raw.strip()) / 1000.0)  # milli-°C
            except ValueError:
                continue
    return max(temps) if temps else None


# --------------------------------------------------------------------------- #
# 探测
# --------------------------------------------------------------------------- #
def probe_host(root: str = "/",
               sysfs: Optional[str] = None,
               procfs: Optional[str] = None,
               devfs: Optional[str] = None,
               etcfs: Optional[str] = None) -> ArmLinuxTarget:
    """探测一台 Linux on ARM 目标机 (或它的文件系统快照)。

    Args:
        root:   快照根; 内部默认 root/sys、root/proc、root/dev、root/etc
        sysfs/procfs/devfs/etcfs: 单独覆盖 (真机探测时传 "/sys" 等)
    """
    r = Path(root)
    sys_root = Path(sysfs) if sysfs else r / "sys"
    proc_root = Path(procfs) if procfs else r / "proc"
    dev_root = Path(devfs) if devfs else r / "dev"
    etc_root = Path(etcfs) if etcfs else r / "etc"

    cpuinfo = _read(proc_root / "cpuinfo")
    arch = _infer_arch(cpuinfo, platform.machine().lower())
    is_arm = arch in AARCH64 or arch in ARM32
    is_aarch64 = arch in AARCH64

    version_file = _read(proc_root / "version")
    osrelease_file = _read(proc_root / "sys" / "kernel" / "osrelease")
    version = version_file or ""
    # 只有在确实读到了 Linux 的 /proc 证据时才采信平台接口的值
    linux_evidence = version_file is not None or osrelease_file is not None
    osrelease = (osrelease_file or "").strip()
    if not osrelease:
        m = re.search(r"Linux version (\S+)", version)
        osrelease = m.group(1) if m else (platform.release() if linux_evidence else "")

    # 板级: 设备树 model (Jetson / RK3588 都在这里)
    model = ""
    for cand in (sys_root / "firmware" / "devicetree" / "base" / "model",
                 sys_root / "firmware" / "devicetree" / "model",
                 proc_root / "device-tree" / "model"):
        txt = _read(cand)
        if txt:
            model = txt.strip().strip("\x00")
            break

    vendor = _detect_vendor(model, etc_root)
    board = _detect_board(model)

    # RT 内核: PREEMPT_RT 会导出 /sys/kernel/realtime, 或版本串含 -rtN / PREEMPT_RT
    rt: Optional[bool] = None
    if _exists(sys_root / "kernel" / "realtime") or "PREEMPT_RT" in version:
        rt = True
    elif linux_evidence:
        rt = bool(re.search(r"-\s*rt\d+", osrelease))

    governor = ""
    g = _read(sys_root / "devices" / "system" / "cpu" / "cpu0" / "cpufreq" / "scaling_governor")
    if g:
        governor = g.strip()

    mem: Optional[float] = None
    meminfo = _read(proc_root / "meminfo") or ""
    m = re.search(r"MemTotal:\s+(\d+)\s*kB", meminfo)
    if m:
        mem = int(m.group(1)) / (1024 * 1024)

    # CPU 数: 优先数 /sys/devices/system/cpu/cpuN 目录 (online 文件是范围串, 不能直接拆分计数)
    cpu_count = os.cpu_count() or 0
    try:
        n = len([p for p in (sys_root / "devices" / "system" / "cpu").iterdir()
                 if p.is_dir() and re.fullmatch(r"cpu\d+", p.name)])
        if n:
            cpu_count = n
    except OSError:
        pass

    return ArmLinuxTarget(
        arch=arch,
        # 能读到 /proc/cpuinfo ⇒ 这是一份 Linux 快照; 否则看宿主 OS
        is_linux=(cpuinfo is not None) or os.name == "posix",
        is_arm=is_arm,
        is_aarch64=is_aarch64,
        kernel_release=osrelease,
        soc_model=model,
        vendor=vendor,
        board_profile=board,
        cpu_count=cpu_count,
        cpu_features=_parse_features(cpuinfo),
        accelerators=_detect_accelerators(dev_root, etc_root, vendor),
        rt_kernel=rt,
        governor=governor,
        isolated_cpus=_parse_isolcpus(_read(proc_root / "cmdline")),
        mem_total_gb=mem,
        thermal_c=_detect_thermal(sys_root),
        can_ifaces=_detect_can(sys_root, proc_root),
        source_root=str(r),
    )


# --------------------------------------------------------------------------- #
# 就绪评估
# --------------------------------------------------------------------------- #
def evaluate_readiness(target: ArmLinuxTarget,
                       requirements: Optional[ArmRequirements] = None,
                       arch_strict: bool = False,
                       as_target: bool = True) -> List[ArmCheck]:
    """逐项判定目标机是否满足契约需求。

    as_target=False 表示"只是看看这台机器的信息, 不把它当部署目标": 此时所有
    依赖目标板硬件/内核配置的检查都标 N/A。否则一台开发机 (或 CI runner) 会因为
    "没有 CAN / 没有 NPU / 不是 RT 内核" 被判 BLOCKING —— 这是**正确**的结论,
    但对"我在笔记本上跑一下看看"是噪声。所以判定与否必须由调用方显式选择,
    而不是隐式猜。
    """
    req = requirements or ArmRequirements.from_contract()
    dev_host = not target.is_arm
    checks: List[ArmCheck] = []

    # 1) 架构 (即便只是"看看信息", 架构也可直接判定)
    arch_ok = (target.arch in AARCH64) if req.arch == "aarch64" else (target.arch == req.arch)
    checks.append(ArmCheck(
        "arch", arch_ok, "BLOCKING",
        f"要求 {req.arch}, 实测 {target.arch or '(未知)'}",
        applicable=as_target or arch_strict or arch_ok,
    ))

    # 2) 板卡识别
    if target.board_profile:
        checks.append(ArmCheck(
            "board", target.board_profile == req.board, "WARNING",
            f"识别为 {target.board_profile} (契约目标 {req.board})",
            applicable=as_target,
        ))
    else:
        checks.append(ArmCheck(
            "board", False, "WARNING",
            f"未识别板卡 (设备树 model='{target.soc_model or 'N/A'}'); 延迟预算需重新标定",
            applicable=as_target and bool(target.soc_model),
        ))

    # 3) SIMD (ARMv8 基线)
    simd_ok = any(target.has_feature(f) for f in _SIMD_FEATURES)
    checks.append(ArmCheck(
        "simd_neon", simd_ok, "BLOCKING",
        f"CPU features: {', '.join(target.cpu_features[:8]) if target.cpu_features else '(无)'}",
        applicable=as_target and target.cpu_features is not None,
    ))

    # 4) 内存
    if target.mem_total_gb is None:
        checks.append(ArmCheck("memory", False, "WARNING", "无法读取 /proc/meminfo", applicable=False))
    else:
        ok = target.mem_total_gb >= req.memory_gb * 0.75
        checks.append(ArmCheck(
            "memory", ok,
            "BLOCKING" if target.mem_total_gb < req.memory_gb * 0.5 else "WARNING",
            f"{target.mem_total_gb:.1f} GB / 契约 {req.memory_gb:.1f} GB",
            applicable=as_target,
        ))

    # 5) CAN (关节通信唯一通道)
    if req.can_required:
        checks.append(ArmCheck(
            "can_interface", bool(target.can_ifaces), "BLOCKING",
            f"CAN 接口: {', '.join(target.can_ifaces) or '(无)'}",
            applicable=as_target and target.is_linux
            and (bool(target.can_ifaces) or _probeable(target)),
        ))

    # 6) 加速器 (GelSight 921KB/帧 预处理要求 GPU/NPU)
    if req.accel_required:
        checks.append(ArmCheck(
            "accelerator", target.has_accelerator, "WARNING",
            f"加速器: {', '.join(target.accelerators) or '(仅 CPU)'}",
            applicable=as_target and target.is_linux,
        ))

    # 7) PREEMPT_RT
    checks.append(ArmCheck(
        "preempt_rt", bool(target.rt_kernel), "WARNING",
        f"kernel={target.kernel_release or '?'} RT={target.rt_kernel}",
        applicable=as_target and target.rt_kernel is not None,
    ))

    # 8) cpufreq governor (调频会引入抖动)
    checks.append(ArmCheck(
        "cpu_governor", target.governor == "performance", "WARNING",
        f"governor={target.governor or '(未知)'}",
        applicable=as_target and bool(target.governor),
    ))

    # 9) isolcpus (大核专用于控制)
    checks.append(ArmCheck(
        "isolated_cpus", bool(target.isolated_cpus), "WARNING",
        f"isolcpus={target.isolated_cpus or '(未设置)'}",
        applicable=as_target and target.is_linux,
    ))

    # 10) 温度
    if target.thermal_c is None:
        checks.append(ArmCheck("thermal", False, "WARNING", "无 thermal_zone", applicable=False))
    else:
        checks.append(ArmCheck(
            "thermal", target.thermal_c < 85.0, "WARNING",
            f"{target.thermal_c:.1f} C (阈值 85 C)",
            applicable=as_target,
        ))

    return checks


def _probeable(target: ArmLinuxTarget) -> bool:
    """是否具备"可以判 CAN 不存在"的依据 (至少读到了 sysfs 或 procfs)。"""
    return bool(target.can_ifaces) or bool(target.cpu_features) or target.mem_total_gb is not None


def summarize(target: ArmLinuxTarget, checks: List[ArmCheck]) -> Dict[str, Any]:
    blocking = [c.name for c in checks if c.applicable and not c.ok and c.severity == "BLOCKING"]
    warnings = [c.name for c in checks if c.applicable and not c.ok and c.severity == "WARNING"]
    na = [c.name for c in checks if not c.applicable]
    info = [c.name for c in checks if c.applicable and c.ok]
    return {
        "ok": not blocking,
        "blocking": blocking, "warnings": warnings, "na": na, "info": info,
        "n_checks": len(checks),
        "target": target_to_dict(target),
    }


def target_to_dict(target: ArmLinuxTarget) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for k, v in target.__dict__.items():
        d[k] = list(v) if isinstance(v, list) else v
    return d


def report_lines(target: ArmLinuxTarget, checks: List[ArmCheck]) -> List[str]:
    """人类可读报告 (ASCII 标记, 经 console 层安全降级)。"""
    lines = [f"目标: {target.soc_model or '(未知板卡)'}",
             f"  arch            : {target.arch}"
             f"{'  (aarch64)' if target.is_aarch64 else ''}",
             f"  kernel          : {target.kernel_release or '?'}",
             f"  vendor/board    : {target.vendor} / {target.board_profile or '(未识别)'}",
             f"  cpu             : {target.cpu_count} 核, governor={target.governor or '?'}",
             f"  isolcpus        : {target.isolated_cpus or '(未设置)'}",
             f"  features        : {', '.join(target.cpu_features[:8]) if target.cpu_features else '(无)'}",
             f"  accelerators    : {', '.join(target.accelerators) or '(仅 CPU)'}",
             f"  memory          : "
             f"{f'{target.mem_total_gb:.1f} GB' if target.mem_total_gb is not None else '(未知)'}",
             f"  thermal         : "
             f"{f'{target.thermal_c:.1f} C' if target.thermal_c is not None else '(未知)'}",
             f"  CAN             : {', '.join(target.can_ifaces) or '(无)'}",
             "",
             f"{'check':18s} {'verdict':9s} detail"]
    for c in checks:
        lines.append(f"{c.name:18s} {c.verdict:9s} {c.detail}")
    s = summarize(target, checks)
    lines.append("")
    lines.append(f"汇总: BLOCKING={s['blocking'] or '(无)'}  WARNING={s['warnings'] or '(无)'}  "
                 f"N/A={s['na'] or '(无)'}")
    return lines


def to_json(target: ArmLinuxTarget, checks: List[ArmCheck]) -> str:
    return json.dumps({
        "target": target_to_dict(target),
        "checks": [{"name": c.name, "ok": c.ok, "severity": c.severity,
                    "verdict": c.verdict, "detail": c.detail,
                    "applicable": c.applicable} for c in checks],
        "summary": summarize(target, checks),
    }, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 延迟预算: ARM 目标上的保守缩放
# --------------------------------------------------------------------------- #
def arm_latency_profile(target: ArmLinuxTarget) -> ArmLatencyProfile:
    """按加速器有无给出预算缩放。

    契约 (hw_requirements) 的 15ms 模型推理 / 2ms 传感器预处理预算是**带 GPU/NPU**
    的前提 (preprocessing.gpu_preprocess_required=true)。纯 CPU 的 ARM 板上这个前提
    不成立, 保守放大后再喂给 deploy.audit_latency, 免得把"跑不动"误判成"通过"。

    缩放系数为**保守占位值**, 必须在目标板上用实测替换 —— 这点写在 notes 里,
    不做隐式承诺。
    """
    if target.has_accelerator:
        return ArmLatencyProfile(
            1.0, 1.0, ",".join(target.accelerators),
            [f"检测到加速器 ({', '.join(target.accelerators)}): 沿用契约预算"])
    return ArmLatencyProfile(
        3.0, 1.5, "",
        ["未检测到 NPU/GPU: 模型推理预算 ×3.0, 传感器预处理 ×1.5 (保守占位)",
         "缩放系数为估计值, 必须在目标板用 audit_latency(measured=...) 实测替换",
         "GelSight 921KB/帧 @20Hz 的 GPU 预处理在无加速器时通常是首要瓶颈"])


def arm_budget_overrides(target: ArmLinuxTarget,
                         base_budgets: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """生成喂给 deploy.audit_latency(budget_overrides=...) 的预算字典。"""
    if base_budgets is None:
        from ..deploy.latency_audit import DOC_BASELINE_MS
        base_budgets = dict(DOC_BASELINE_MS)
        try:
            from ..config import load_contract
            hw = load_contract("hw_requirements")
            base_budgets["model_inference"] = hw["compute"].get(
                "model_inference_ms", base_budgets["model_inference"])
        except Exception:  # noqa: BLE001
            pass
    p = arm_latency_profile(target)
    out = dict(base_budgets)
    if "model_inference" in out:
        out["model_inference"] = out["model_inference"] * p.model_inference_scale
    if "sensor_acquisition" in out:
        out["sensor_acquisition"] = out["sensor_acquisition"] * p.sensor_preprocess_scale
    return out


# --------------------------------------------------------------------------- #
# 部署产物: systemd 单元 (CPUAffinity 来自 isolcpus 探测)
# --------------------------------------------------------------------------- #
_UNITS = {
    "robot-control": "主控制环路 (L4 20Hz + CAN-FD 下发)",
    "vla-inference": "VLA 模型推理服务 (端到端 <50ms)",
    "safety-monitor": "安全链监控守护 (CAT_0 急停)",
}


def render_systemd_units(target: ArmLinuxTarget, out_dir: str,
                         exec_prefix: str = "/usr/bin") -> List[str]:
    """按探测结果渲染 systemd 单元文件, 返回写出的路径列表。

    CPUAffinity 直接用探测到的 isolcpus —— 文档里写的是 isolcpus=4,5,6,7,
    但不同板卡核数不同, 硬编码会造成"配了亲和性却指向不存在的核"。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    affinity = " ".join(str(c) for c in target.isolated_cpus)

    written: List[str] = []
    for name, desc in _UNITS.items():
        body = [
            "[Unit]",
            f"Description=EmbodiedAI {name} - {desc}",
            "After=network.target",
            "",
            "[Service]",
            "Type=simple",
            f"ExecStart={exec_prefix}/embodied-{name}",
            "Restart=always",
            "RestartSec=1",
        ]
        if affinity:
            body.append(f"CPUAffinity={affinity}")
        body += [
            "Nice=-10",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ]
        p = out / f"{name}.service"
        p.write_text("\n".join(body), encoding="utf-8")
        written.append(str(p))
    return written
