"""聚合自检 — `embodied-audit verify` 与 CI / scripts/verify 的共享实现 (v0.5 新增)。

设计原则:
  1. **可聚合**: 每个检查项返回 (name, passed, detail), 异常转为 FAIL 而非崩溃。
  2. **可在无外部依赖环境运行**: 仿真类检查走 reference 后端, EdaLab live 检查在
     jar 缺失时自动降级为 dry (不作为硬性失败)。
  3. **退出码语义**: 全部 PASS → 0, 任一 FAIL → 1。供 scripts/verify.* 与
     .github/workflows/ci.yml 直接消费, 不需要解析文本。

检查项分组:
  契约四件套 / 安全预算 / 仿真栈 / Sim-to-Real / 双足步态 / nanoGPT 计划 / pytest 套件
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, List, Tuple

VERSION = "0.6.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


# --------------------------------------------------------------------------- #
# 单项检查
# --------------------------------------------------------------------------- #
def check_contracts() -> int:
    from .config import load_contract, validate_dbc

    for name in ("hw_requirements", "safety_budget", "dr_params/manufacturing_variance"):
        load_contract(name)
    dbc = validate_dbc()
    ok = dbc["required_ok"] and dbc["unique_ids"] and dbc["crc_coverage"]
    assert dbc["message_count"] == 9, f"DBC 消息数应为 9, 实际 {dbc['message_count']}"
    return 0 if ok else 1


def check_safety_budget() -> int:
    from .config import load_contract
    from .safety.formal import close_budget_loop

    sb = load_contract("safety_budget")
    total_ms = sum(
        (s["measured_us"] / 1000 if "budget_us" in s else s["measured_ms"])
        for s in sb["segments"]
    )
    assert total_ms < sb["acceptance"]["total_break_max_ms"], f"安全链 {total_ms}ms 超预算"
    r = close_budget_loop(joint_speed_rad_s=8.0)
    assert r.closure_ok, "形式化安全闭环未通过"
    return 0


def check_simulation() -> int:
    from .simulation import DRConfig, DomainRandomizer, EnvConfig, make_env

    env = make_env(EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3,
                             max_episode_steps=20), backend="reference")
    dr = DomainRandomizer(DRConfig(seed=0, enable_visual=False, enable_environment=False))
    env.set_dr_params(dr.sample_params())
    obs = env.reset(seed=0)
    assert "state" in obs, "观测缺少 state"
    for _ in range(20):
        res = env.step(env.sample_action())
        if res.done or res.truncated:
            break
    assert len(env.export_episode()["observations"]) > 0, "episode 未记录"
    env.close()
    return 0


def check_sim_to_real() -> int:
    from .simulation import DRConfig, EnvConfig, SimToRealPipeline, make_env

    class _P:
        def __init__(self, target):
            self.target = target

        def predict(self, obs):
            s = obs["state"]
            n = len(self.target)
            return -2.0 * (s[:n] - self.target) - 0.3 * s[n:2 * n]

    target = [0.4, 0.0, 0.0]
    cfg = lambda noise: EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3, dt=0.01,
                                  max_episode_steps=300, obs_noise_std=noise,
                                  action_noise_std=noise * 0.5)
    sim = make_env(cfg(0.0), backend="reference")
    real = make_env(cfg(0.02), backend="reference", real=True)
    res = SimToRealPipeline(sim, real, _P(target),
                            DRConfig(seed=0, enable_visual=False, enable_environment=False)
                            ).run(sim_episodes=2, real_episodes=2, max_steps_per_episode=300)
    assert res.real_episodes == 2, "真机评估未执行"
    assert res.gap_diagnosis, "差距诊断为空"
    assert res.report(), "报告生成失败"
    return 0


def check_biped() -> int:
    import numpy as np

    from .control.biped_locomotion import (GaitGenerator, LIPM, LIPMParams,
                                           ZMPStabilityChecker, build_walking_plans)

    lipm = LIPM(LIPMParams())
    plans = build_walking_plans(n_steps=4)
    gait = GaitGenerator(lipm, plans)
    checker = ZMPStabilityChecker(com_height=lipm.p.z_c, planar=True)
    dt = 0.005
    n = int(sum(p.step_time for p in plans) / dt)
    unstable = 0
    for _ in range(n):
        com, lf, rf = gait.update(dt)
        plan = plans[gait.active_step_index]
        support = lf if plan.support_foot == "left" else rf
        stable, _ = checker.is_stable(gait.get_zmp_instant(), support[:2])
        unstable += 0 if stable else 1
    com_x = gait.get_zmp_instant()[0]
    assert np.isfinite(com_x) and com_x > 0.0, f"CoM 未前进或数值发散: {com_x}"
    assert unstable == 0, f"存在 {unstable} 帧失稳"
    cp = lipm.calc_capture_point(0.2, 0.3)
    assert abs(cp - (0.2 + 0.3 / lipm.p.omega)) < 1e-9, "捕获点公式不一致"
    rec = checker.suggest_recovery_step(np.array([0.5]), np.array([0.1]))
    assert rec[0] > 0.1, "恢复步方向错误"
    # 受扰后应能判出失稳 (判据不是恒真)
    ok, _ = checker.is_stable(np.array([0.5, 0.0]), np.array([0.0, 0.0]))
    assert not ok, "远离支撑脚的 ZMP 应判为失稳"
    return 0


def check_nanogpt_plan() -> int:
    from .adapters.nanogpt import NanoGPTAdapter

    adapter = NanoGPTAdapter()
    probe = adapter.probe()
    plan = adapter.export_training_plan(model_size="vla_tiny", out_dir="output/verify_nanogpt")
    assert plan["plan"]["estimate_params"] > 0, "参数量估算非正"
    assert Path(plan["path"]).exists(), "训练计划未落盘"
    # 仓库缺失属环境预期, 仅要求探测结构完整
    assert probe["total"] == 3, "探针清单应含 3 个文件"
    return 0


def check_arm_linux() -> int:
    """Linux on ARM 能力层 (v0.6) — 用合成快照验证, 不依赖真板卡。

    三组断言:
      1. 探测: Jetson Orin NX 快照应被识别为 aarch64 / nvidia / PREEMPT_RT / CAN
      2. 判定: 缺 CAN → BLOCKING; 开发机非 ARM 且非 strict → 架构 N/A (不假红)
      3. 闭环: 无加速器时的预算缩放必须能回灌 deploy.audit_latency
    """
    import tempfile
    from pathlib import Path as _P

    from .platform import (ArmRequirements, arm_budget_overrides, evaluate_readiness,
                           probe_host, summarize)

    def snap(**kw):
        tmp = tempfile.mkdtemp()
        root = _P(tmp) / "snap"
        _write_snapshot(root, **kw)
        return probe_host(root=str(root))

    t = snap()
    assert t.is_aarch64, f"未识别 aarch64: {t.arch}"
    assert t.vendor == "nvidia" and t.board_profile == "jetson_orin_nx_16gb", t.board_profile
    assert t.rt_kernel and t.can_ifaces == ["can0"], (t.rt_kernel, t.can_ifaces)

    s_ok = summarize(t, evaluate_readiness(t, ArmRequirements.from_contract()))
    assert s_ok["ok"], f"健康目标板不该有 BLOCKING: {s_ok['blocking']}"

    s_bad = summarize(snap(can=False), evaluate_readiness(snap(can=False)))
    assert "can_interface" in s_bad["blocking"], "缺 CAN 必须 BLOCKING"

    dev = snap()
    dev.arch, dev.is_arm, dev.is_aarch64 = "x86_64", False, False
    arch_check = next(c for c in evaluate_readiness(dev, as_target=False) if c.name == "arch")
    assert arch_check.verdict == "N/A", "开发机只探测信息时, 架构检查不应假红"
    # 但显式声明"这就是部署目标"时必须如实报 BLOCKING
    strict = summarize(dev, evaluate_readiness(dev, as_target=True))
    assert "arch" in strict["blocking"], "as_target=True 时不能因难看而降级"

    from .deploy import audit_latency
    res = audit_latency(budget_overrides=arm_budget_overrides(snap(accel=False, tegra=False)))
    model = next(x for x in res.segments if x.name == "model_inference")
    assert model.budget_ms > 15.0, "无加速器时推理预算必须放大"
    return 0


def _write_snapshot(root, *, arch="aarch64", board="jetson", tegra=True, rt=True,
                    governor="performance", isolcpus="4-7", mem_gb=16.0, can=True,
                    accel=True, thermal_c=55.0, cpu_count=8) -> None:
    """构造目标板文件系统快照 (与 tests/test_k5_arm_linux.py 的构造器同构)。"""
    root = Path(root)
    proc, sysfs, dev, etc = root / "proc", root / "sys", root / "dev", root / "etc"

    def w(p, text):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    w(proc / "cpuinfo", "\n".join((
        "processor\t: 0",
        "model name\t: ARMv8 Processor rev 1",
        f"CPU architecture: {'8' if arch == 'aarch64' else '7'}",
        "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics asimddp",
        "")))
    w(proc / "version", "Linux version 5.15.136-rt-tegra #1 SMP PREEMPT_RT" if rt
      else "Linux version 5.15.136-tegra #1 SMP")
    w(proc / "cmdline", f"root=/dev/mmcblk0p1 rw isolcpus={isolcpus} console=ttyS0,115200")
    w(proc / "meminfo", f"MemTotal:      {int(mem_gb * 1024 * 1024)} kB\n")
    w(sysfs / "firmware" / "devicetree" / "base" / "model", {
        "jetson": "NVIDIA Jetson Orin NX Engineering Reference Developer Kit",
        "rk3588": "Rockchip RK3588 EVB1 LP4 V10 Board",
        "unknown": "Some Unknown Carrier Board",
    }[board])
    if rt:
        w(sysfs / "kernel" / "realtime", "1\n")
    w(sysfs / "devices" / "system" / "cpu" / "cpu0" / "cpufreq" / "scaling_governor", governor)
    for i in range(cpu_count):
        (sysfs / "devices" / "system" / "cpu" / f"cpu{i}").mkdir(parents=True, exist_ok=True)
    w(sysfs / "class" / "thermal" / "thermal_zone0" / "temp", str(int(thermal_c * 1000)))
    if can:
        w(sysfs / "class" / "net" / "can0" / "uevent", "INTERFACE=can0\nDEVTYPE=can\n")
    dev.mkdir(parents=True, exist_ok=True)
    if tegra:
        w(etc / "nv_tegra_release", "# R35 (release), REVISION: 4.1\n")
        (dev / "nvidia0").write_text("", encoding="utf-8")
    if board == "rk3588":
        (dev / "mpp_service").write_text("", encoding="utf-8")
    if accel:
        (dev / "dri").mkdir(parents=True, exist_ok=True)


def check_cli_smoke() -> int:
    """CLI 子命令冒烟 — 覆盖 v0.5 修复的 GBK 输出崩溃回归。"""
    import os

    env = {**os.environ, "PYTHONPATH": str(SRC_ROOT), "PYTHONIOENCODING": "utf-8"}
    # audit 依赖兄弟仓库存在, 不作为硬失败; 其余子命令必须 0 退出
    for cmd in ("contracts", "safety", "sim", "sim2real", "biped", "nanogpt", "dr",
                "platform"):
        r = subprocess.run([sys.executable, "-m", "embodied_ai.cli", cmd],
                           capture_output=True, cwd=str(REPO_ROOT), env=env)
        assert r.returncode == 0, (
            f"CLI {cmd} 退出码 {r.returncode}\n"
            f"--- stdout ---\n{r.stdout.decode('utf-8', 'replace')}\n"
            f"--- stderr ---\n{r.stderr.decode('utf-8', 'replace')}")
    return 0


def check_pytest() -> int:
    """跑测试套件。

    必须继承**完整**环境: 只给 PATH/PYTHONPATH 会让 torch/asyncio 在 Windows 上
    因缺少 SYSTEMROOT 等变量而加载失败 (OSError 10106), 表现为假失败。
    """
    env = {**os.environ, "PYTHONPATH": str(SRC_ROOT), "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                       capture_output=True, cwd=str(REPO_ROOT), env=env)
    out = r.stdout.decode("utf-8", "replace")
    err = r.stderr.decode("utf-8", "replace")
    assert r.returncode == 0, (out + "\n" + err)[-3000:]
    return 0


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #
CHECKS: List[Tuple[str, Callable[[], int]]] = [
    ("contracts", check_contracts),
    ("safety_budget", check_safety_budget),
    ("simulation", check_simulation),
    ("sim_to_real", check_sim_to_real),
    ("biped_gait", check_biped),
    ("nanogpt_plan", check_nanogpt_plan),
    ("arm_linux", check_arm_linux),
    ("cli_smoke", check_cli_smoke),
]


def run_all(skip_pytest: bool = False) -> List[Tuple[str, bool, str]]:
    """执行全部检查项, 返回 [(name, passed, detail)]。"""
    results: List[Tuple[str, bool, str]] = []
    for name, fn in CHECKS:
        try:
            rc = fn()
            results.append((name, rc == 0, "OK" if rc == 0 else f"exit={rc}"))
        except Exception as e:  # noqa: BLE001 — 单项异常不应终止整体自检
            results.append((name, False, f"{type(e).__name__}: {e}"))

    if not skip_pytest:
        try:
            rc = check_pytest()
            results.append(("pytest", rc == 0, "OK" if rc == 0 else f"exit={rc}"))
        except Exception as e:  # noqa: BLE001
            results.append(("pytest", False, f"{type(e).__name__}: {e}"))
    return results


def summarize(results: List[Tuple[str, bool, str]]) -> Dict[str, Any]:
    n_pass = sum(1 for _, p, _ in results if p)
    return {"total": len(results), "passed": n_pass,
            "failed": [n for n, p, _ in results if not p],
            "ok": n_pass == len(results)}
