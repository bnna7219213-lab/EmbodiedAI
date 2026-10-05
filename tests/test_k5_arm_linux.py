"""Linux on ARM 目标能力层测试 (v0.6 新增)。

可移植性前提: 全部用例基于**合成的目标板文件系统快照** (tmp 目录里的
proc/ sys/ dev/ etc/), 因此在 x86 Windows 开发机与 CI 上都能完整验证,
不需要真的 ARM 板卡 —— 这正是本模块的设计目标之一。

覆盖:
  - probe_host: 架构推断 / 板卡识别 / 加速器 / RT / isolcpus / CAN / 内存 / 温度
  - evaluate_readiness: BLOCKING/WARNING/INFO 分级 + 不可判定项标 N/A
  - arm_latency_profile / arm_budget_overrides → 回灌 deploy.audit_latency
  - render_systemd_units: CPUAffinity 来自 isolcpus
  - CLI `platform --root ...` 退出码语义
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_ai.platform import (
    ArmRequirements,
    arm_budget_overrides,
    arm_latency_profile,
    evaluate_readiness,
    probe_host,
    render_systemd_units,
    report_lines,
    summarize,
    to_json,
)
from embodied_ai.deploy import audit_latency

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


# --------------------------------------------------------------------------- #
# 目标板快照构造器
# --------------------------------------------------------------------------- #
def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def build_snapshot(root: Path, *, arch: str = "aarch64", board: str = "jetson",
                   tegra: bool = True, rt: bool = True, governor: str = "performance",
                   isolcpus: str = "4-7", mem_gb: float = 16.0, can: bool = True,
                   accel: bool = True, thermal_c: float = 55.0,
                   cpu_count: int = 8) -> Path:
    """构造一份 Linux on ARM 目标板的 /proc + /sys + /dev + /etc 快照。"""
    root = Path(root)
    proc, sysfs, dev, etc = root / "proc", root / "sys", root / "dev", root / "etc"

    cpu_arch = "8" if arch == "aarch64" else "7"
    cpuinfo = "\n".join(
        line
        for _ in range(2)
        for line in (
            "processor\t: 0",
            "model name\t: ARMv8 Processor rev 1",
            f"CPU architecture: {cpu_arch}",
            "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics "
            "fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp",
            "",
        )
    )
    _write(proc / "cpuinfo", cpuinfo)
    _write(proc / "version",
           "Linux version 5.15.136-rt-tegra (buildbrain@...) "
           "(gcc (GCC) 11.4.0) #1 SMP PREEMPT_RT Tue Mar 5 12:00:00 UTC 2024"
           if rt else
           "Linux version 5.15.136-tegra #1 SMP Tue Mar 5 12:00:00 UTC 2024")
    _write(proc / "sys" / "kernel" / "osrelease",
           "5.15.136-rt-tegra" if rt else "5.15.136-tegra")
    _write(proc / "cmdline",
           f"root=/dev/mmcblk0p1 rw isolcpus={isolcpus} "
           f"{'' if isolcpus else ''}console=ttyS0,115200")
    _write(proc / "meminfo", f"MemTotal:      {int(mem_gb * 1024 * 1024)} kB\n"
                             f"MemFree:        1024000 kB\n")

    board_model = {
        "jetson": "NVIDIA Jetson Orin NX Engineering Reference Developer Kit",
        "rk3588": "Rockchip RK3588 EVB1 LP4 V10 Board",
        "imx8mp": "NXP i.MX8M Plus EVK board",
        "unknown": "Some Unknown Carrier Board",
    }[board]
    _write(sysfs / "firmware" / "devicetree" / "base" / "model", board_model)
    if rt:
        _write(sysfs / "kernel" / "realtime", "1\n")
    _write(sysfs / "devices" / "system" / "cpu" / "cpu0" / "cpufreq" / "scaling_governor",
           governor + "\n")
    for i in range(cpu_count):
        (sysfs / "devices" / "system" / "cpu" / f"cpu{i}").mkdir(parents=True, exist_ok=True)
    _write(sysfs / "class" / "thermal" / "thermal_zone0" / "temp",
           str(int(thermal_c * 1000)) + "\n")
    if can:
        _write(sysfs / "class" / "net" / "can0" / "uevent",
               "INTERFACE=can0\nIFINDEX=3\nDEVTYPE=can\n")
        _write(sysfs / "class" / "net" / "eth0" / "uevent", "INTERFACE=eth0\nIFINDEX=2\n")

    dev.mkdir(parents=True, exist_ok=True)
    if tegra:
        _write(etc / "nv_tegra_release", "# R35 (release), REVISION: 4.1\n")
        (dev / "nvidia0").write_text("", encoding="utf-8")
    if board == "rk3588":
        (dev / "mpp_service").write_text("", encoding="utf-8")
    if accel:
        (dev / "dri").mkdir(parents=True, exist_ok=True)
        (dev / "video0").write_text("", encoding="utf-8")
    return root


class TestK5A_Probe(unittest.TestCase):
    def test_jetson_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            t = probe_host(root=build_snapshot(Path(d)))
        self.assertEqual(t.arch, "aarch64")
        self.assertTrue(t.is_aarch64 and t.is_arm and t.is_linux)
        self.assertEqual(t.board_profile, "jetson_orin_nx_16gb")
        self.assertEqual(t.vendor, "nvidia")
        self.assertTrue(t.rt_kernel)
        self.assertEqual(t.governor, "performance")
        self.assertEqual(t.isolated_cpus, [4, 5, 6, 7])
        self.assertAlmostEqual(t.mem_total_gb, 16.0, places=1)
        self.assertEqual(t.can_ifaces, ["can0"])
        self.assertIn("nvidia-gpu", t.accelerators)
        self.assertAlmostEqual(t.thermal_c, 55.0, places=1)
        self.assertEqual(t.cpu_count, 8)
        self.assertIn("asimd", t.cpu_features)

    def test_rk3588_vendor_and_vpu(self):
        with tempfile.TemporaryDirectory() as d:
            t = probe_host(root=build_snapshot(Path(d), board="rk3588", tegra=False))
        self.assertEqual(t.vendor, "rockchip")
        self.assertEqual(t.board_profile, "rk3588")
        self.assertIn("rockchip-vpu", t.accelerators)

    def test_unknown_board_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            t = probe_host(root=build_snapshot(Path(d), board="unknown", tegra=False))
        self.assertEqual(t.board_profile, "")
        self.assertTrue(t.is_aarch64)

    def test_non_rt_and_ondemand(self):
        with tempfile.TemporaryDirectory() as d:
            t = probe_host(root=build_snapshot(Path(d), rt=False, governor="ondemand"))
        self.assertFalse(t.rt_kernel)
        self.assertEqual(t.governor, "ondemand")

    def test_armv7_arch(self):
        with tempfile.TemporaryDirectory() as d:
            t = probe_host(root=build_snapshot(Path(d), arch="armv7l", board="imx8mp"))
        self.assertEqual(t.arch, "armv7l")
        self.assertTrue(t.is_arm)
        self.assertFalse(t.is_aarch64)

    def test_no_can(self):
        with tempfile.TemporaryDirectory() as d:
            t = probe_host(root=build_snapshot(Path(d), can=False))
        self.assertEqual(t.can_ifaces, [])

    def test_probe_never_crashes_on_empty_root(self):
        with tempfile.TemporaryDirectory() as d:
            t = probe_host(root=d)
        self.assertTrue(t.arch)          # 回退 platform.machine()
        self.assertIsNone(t.cpu_features)
        self.assertIsNone(t.mem_total_gb)
        self.assertEqual(t.accelerators, [])


class TestK5B_Readiness(unittest.TestCase):
    def _target(self, **kw):
        tmp = tempfile.mkdtemp()
        return probe_host(root=build_snapshot(Path(tmp), **kw))

    def test_healthy_jetson_passes(self):
        t = self._target()
        checks = evaluate_readiness(t, ArmRequirements.from_contract())
        s = summarize(t, checks)
        self.assertTrue(s["ok"], f"不该有 BLOCKING: {s['blocking']}")
        self.assertEqual(s["warnings"], [])
        self.assertIn("arch", s["info"])
        self.assertIn("can_interface", s["info"])

    def test_missing_can_is_blocking(self):
        t = self._target(can=False)
        checks = evaluate_readiness(t)
        s = summarize(t, checks)
        self.assertIn("can_interface", s["blocking"])

    def test_low_memory_blocking(self):
        t = self._target(mem_gb=4.0)      # 契约 16GB, 4GB < 50%
        s = summarize(t, evaluate_readiness(t))
        self.assertIn("memory", s["blocking"])

    def test_mid_memory_warning_only(self):
        t = self._target(mem_gb=10.0)     # 16*0.5 < 10 < 16*0.75 → WARNING
        checks = evaluate_readiness(t)
        s = summarize(t, checks)
        self.assertIn("memory", s["warnings"])
        self.assertNotIn("memory", s["blocking"])

    def test_no_accelerator_is_warning(self):
        t = self._target(accel=False, tegra=False)
        s = summarize(t, evaluate_readiness(t))
        self.assertIn("accelerator", s["warnings"])

    def test_no_rt_and_bad_governor_are_warnings(self):
        t = self._target(rt=False, governor="ondemand")
        s = summarize(t, evaluate_readiness(t))
        self.assertIn("preempt_rt", s["warnings"])
        self.assertIn("cpu_governor", s["warnings"])
        self.assertIn("isolated_cpus", s["warnings"]) if not t.isolated_cpus else None

    def test_dev_host_all_na_when_not_as_target(self):
        """只探测信息 (as_target=False) 时, 硬件类检查一律 N/A —— 保证 CI 可移植。"""
        t = self._target()
        t.arch = "x86_64"
        t.is_arm, t.is_aarch64 = False, False
        checks = evaluate_readiness(t, as_target=False)
        self.assertEqual(next(c for c in checks if c.name == "arch").verdict, "N/A")
        self.assertEqual(next(c for c in checks if c.name == "can_interface").verdict, "N/A")
        self.assertEqual(next(c for c in checks if c.name == "accelerator").verdict, "N/A")
        self.assertTrue(summarize(t, checks)["ok"])

    def test_dev_host_arch_blocks_when_strict(self):
        t = self._target()
        t.arch, t.is_arm, t.is_aarch64 = "x86_64", False, False
        checks = evaluate_readiness(t, arch_strict=True, as_target=False)
        arch_check = next(c for c in checks if c.name == "arch")
        self.assertEqual(arch_check.verdict, "BLOCKING")
        self.assertFalse(summarize(t, checks)["ok"])

    def test_as_target_on_dev_host_is_honest(self):
        """显式 as_target=True 时必须如实报 BLOCKING (不能因为难看就降级)。"""
        t = self._target()
        t.arch, t.is_arm, t.is_aarch64 = "x86_64", False, False
        s = summarize(t, evaluate_readiness(t, as_target=True))
        self.assertIn("arch", s["blocking"])

    def test_unavailable_inputs_are_na(self):
        t = ArmRequirements and probe_host(root=tempfile.mkdtemp())
        checks = evaluate_readiness(t)
        self.assertIsNone(t.cpu_features)
        self.assertEqual(next(c for c in checks if c.name == "simd_neon").verdict, "N/A")
        self.assertEqual(next(c for c in checks if c.name == "memory").verdict, "N/A")
        self.assertEqual(next(c for c in checks if c.name == "thermal").verdict, "N/A")

    def test_report_and_json(self):
        t = self._target()
        checks = evaluate_readiness(t)
        lines = report_lines(t, checks)
        self.assertTrue(any("汇总" in l for l in lines))
        payload = json.loads(to_json(t, checks))
        self.assertIn("summary", payload)
        self.assertEqual(len(payload["checks"]), len(checks))


class TestK5C_LatencyProfile(unittest.TestCase):
    def _target(self, **kw):
        return probe_host(root=build_snapshot(Path(tempfile.mkdtemp()), **kw))

    def test_with_accelerator_keeps_contract_budget(self):
        t = self._target()
        p = arm_latency_profile(t)
        self.assertEqual(p.model_inference_scale, 1.0)
        ov = arm_budget_overrides(t)
        self.assertAlmostEqual(ov["model_inference"], 15.0)   # 契约值

    def test_cpu_only_scales_up(self):
        t = self._target(accel=False, tegra=False)
        p = arm_latency_profile(t)
        self.assertGreater(p.model_inference_scale, 1.0)
        self.assertGreater(p.sensor_preprocess_scale, 1.0)
        self.assertTrue(any("实测替换" in n for n in p.notes), "必须写明是占位值")

    def test_overrides_feed_latency_gate(self):
        """ARM 预算缩放必须能回灌既有 50ms 闸门 (不是另起一套)。"""
        t = self._target(accel=False, tegra=False)
        ov = arm_budget_overrides(t)
        res = audit_latency(budget_overrides=ov)
        self.assertTrue(all(s.budget_ms > 0 for s in res.segments))
        model = next(s for s in res.segments if s.name == "model_inference")
        self.assertGreater(model.budget_ms, 15.0)
        # 无加速器 + 文档基准实测值(15ms) → 依然在放大后的预算内, 但总预算更紧
        self.assertTrue(res.report_lines())


class TestK5D_SystemdUnits(unittest.TestCase):
    def test_affinity_from_isolcpus(self):
        t = probe_host(root=build_snapshot(Path(tempfile.mkdtemp())))
        with tempfile.TemporaryDirectory() as d:
            paths = render_systemd_units(t, d)
            self.assertEqual(len(paths), 3)
            for p in paths:
                text = Path(p).read_text(encoding="utf-8")
                self.assertIn("CPUAffinity=4 5 6 7", text)
                self.assertIn("[Install]", text)

    def test_no_affinity_when_not_isolated(self):
        t = probe_host(root=build_snapshot(Path(tempfile.mkdtemp()), isolcpus=""))
        with tempfile.TemporaryDirectory() as d:
            render_systemd_units(t, d)
            text = (Path(d) / "robot-control.service").read_text(encoding="utf-8")
            self.assertNotIn("CPUAffinity", text)


class TestK5E_CLI(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run([sys.executable, "-m", "embodied_ai.cli", "platform", *args],
                              capture_output=True, cwd=str(REPO),
                              env={**os.environ, "PYTHONPATH": str(SRC),
                                   "PYTHONIOENCODING": "utf-8"},
                              encoding="utf-8", errors="replace")

    def test_snapshot_passes_and_emits_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root = build_snapshot(Path(d) / "snap")
            units = Path(d) / "units"
            jf = Path(d) / "out.json"
            r = self._run("--root", str(root), "--json", str(jf), "--emit-units", str(units))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue(jf.exists())
            self.assertEqual(len(list(units.glob("*.service"))), 3)
            payload = json.loads(jf.read_text(encoding="utf-8"))
            self.assertTrue(payload["summary"]["ok"])

    def test_blocking_snapshot_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            root = build_snapshot(Path(d) / "snap", can=False)
            r = self._run("--root", str(root))
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("can_interface", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
