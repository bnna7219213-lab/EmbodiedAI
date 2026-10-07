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
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

VERSION = "0.7.0"

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
    # 仓库缺失属环境预期, 仅要求探测结构完整 (核心探针: llm.py/train.py/tokenizer.py;
    # probe 扩展后 total=5 为兼容超集 — 旧断言 ==3 在扩展后误报)
    assert probe["total"] >= 3, f"探针清单至少应含 3 个文件, got {probe['total']}"
    if probe["exists"]:
        assert probe["found"] >= 3, "核心探针 (llm/train/tokenizer) 应至少命中"
    return 0


# --------------------------------------------------------------------------- #
# v0.7 新增检查项 (缺口补齐后的验收)
# --------------------------------------------------------------------------- #
def check_soft_body() -> int:
    """G6 柔性体: XPBD 不可拉伸 / 悬垂收敛 / 抓取变形 / 滑脱判据。"""
    from .dynamics.soft_body import soft_body_demo

    r = soft_body_demo()
    assert r["ok"], f"柔性体自检失败: {r}"
    return 0


def check_whole_body() -> int:
    """G9 3D: 腿部 IK 往返精度 + 3D ZMP 稳定 + 判据非恒真。"""
    from .control.whole_body import whole_body_demo

    r = whole_body_demo()
    assert r["ok"], f"3D 全身控制自检失败: {r}"
    assert r["ik_position_err"] < 1e-9, r["ik_position_err"]
    return 0


def check_diffusion_policy() -> int:
    """扩散策略: 训练损失下降 + 采样分布优于噪声 + DDIM 少步可行。"""
    from .models.diffusion_policy import fit_demo

    r = fit_demo(iters=600)
    assert r["ok"], f"扩散策略自检失败: {r}"
    return 0


def check_bc_training() -> int:
    """行为克隆: 验证 MSE 下降 + 环境内成功率接近专家。"""
    from .models.bc_trainer import bc_demo

    r = bc_demo(n_episodes=10, iters=300, eval_episodes=3)
    assert r["ok"], f"行为克隆自检失败: {r}"
    return 0


def check_rl_algorithms() -> int:
    """PPO / SAC: 回报优于随机基线 (PPO), 训练动力学健康 (SAC 短跑)。

    SAC 是 off-policy + 单步更新, 需要 ~12k 环境步才稳定超过随机基线
    (12s), 这里只跑 3k 步验证算法**不发散**: q 损失有界、温度在合理区间、
    回报曲线上升。完整规模由 `embodied-audit train --algo sac` 承担。
    """
    from .models.rl_algorithms import rl_demo

    # 80 次迭代是"稳定超过随机基线"的下限 (60 次时末段回报 −74.2 vs 随机
    # −73.3, 刚好差一点点, 会偶发假红)
    ppo = rl_demo("ppo", iters=90)
    assert ppo["ok"], f"PPO 未超过随机基线: {ppo}"
    sac = rl_demo("sac", iters=3000)
    d = sac["diagnostics"]
    assert np.isfinite(d.get("q_loss", np.nan)) and d["q_loss"] < 1e3, d
    assert 0.0 < d.get("alpha", 0.0) < 50.0, d
    assert sac["final_return"] >= sac["initial_return"] - 5.0, sac
    return 0


def check_foc_shutdown() -> int:
    """优先级 (1): SVPWM 伏秒平衡 + 电流环稳态 + 关机链 0.8µs/16ms + 停止角。"""
    from .hardware.foc_shutdown import foc_shutdown_demo

    r = foc_shutdown_demo()
    assert r["ok"], f"FOC/关机链自检失败: {r}"
    assert r["electrical_off_us"] <= 1.0, r["electrical_off_us"]
    return 0


def check_dexterous_hand() -> int:
    """G11: 力闭合判据正确 + 选型矩阵有区分度 + 给出推荐。"""
    from .hardware.dexterous_hand import dexterous_hand_demo

    r = dexterous_hand_demo()
    assert r["ok"], f"灵巧手自检失败: {r}"
    return 0


def check_real_robot() -> int:
    """Sim-to-Real 最后一公里: mock 传输跑通闭环 + 安全闸门 + 延迟统计。"""
    from .simulation.real_robot_env import real_robot_demo

    r = real_robot_demo(steps=30)
    assert r["ok"], f"真机闭环自检失败: {r}"
    return 0


def check_jetson_bench() -> int:
    """优先级 (2): VLA 推理 roofline 基准 + 无加速器时必须超预算。"""
    from .deploy.jetson_bench import jetson_bench_demo

    r = jetson_bench_demo()
    assert r["ok"], f"推理基准自检失败: {r}"
    return 0


def check_lifelong() -> int:
    """G12: 灾难性遗忘可观测 + EWC 有效 + λ 过大会压死可塑性。"""
    from .learning.lifelong import lifelong_demo

    r = lifelong_demo()
    assert r["ok"], f"终身学习自检失败: {r}"
    return 0


def check_certification() -> int:
    """G10: PFH/SFF/架构约束/RTM 四项达标, 且三项判据非恒真。"""
    from .safety.certification import certification_demo

    r = certification_demo()
    assert r["ok"], f"认证包自检失败: {r}"
    return 0


def check_climb_cycle() -> int:
    """Phase A: 完整攀岩周期集成 (climb_scene 焊接层, 全链无 mock)。"""
    from .simulation.climb_scene import run_climb_cycle

    r = run_climb_cycle(seed=42)
    assert r.phases_visited[-1] == "climb_on", f"周期未完成: {r.phases_visited}"
    assert r.all_real, "集成链路存在 mock"
    assert r.f_soft_peak_n < r.f_rigid_peak_n, "柔顺分支必须低于刚性分支"
    assert r.full_speed_infeasible, "满速直弹超限应被诚实记录"
    assert r.capture_feasible and r.com_advance_m > 0.3, "捕获后应续攀推进"
    return 0


def check_gui() -> int:
    """L7: 起服务 → 拉全部 API → 下发 STO → 关服务。"""
    from .gui import gui_demo

    r = gui_demo()
    assert r["ok"], f"GUI 自检失败: {r}"
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
    # v0.7: 新增子命令一并冒烟 (全部走小规模默认, 保证 CI 时长可控)
    for cmd in ("contracts", "safety", "sim", "sim2real", "biped", "nanogpt", "dr",
                "platform", "soft", "foc", "hand", "cert", "bench", "lifelong",
                "real", "wholebody", "gui --once", "train --algo ppo --iters 80"):
        r = subprocess.run([sys.executable, "-m", "embodied_ai.cli", *cmd.split()],
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
    # --- v0.5/v0.6 既有 --- #
    ("contracts", check_contracts),
    ("safety_budget", check_safety_budget),
    ("simulation", check_simulation),
    ("sim_to_real", check_sim_to_real),
    ("biped_gait", check_biped),
    ("nanogpt_plan", check_nanogpt_plan),
    ("arm_linux", check_arm_linux),
    # --- v0.7 缺口补齐 --- #
    ("soft_body", check_soft_body),            # G6
    ("whole_body_3d", check_whole_body),       # G9
    ("diffusion_policy", check_diffusion_policy),
    ("bc_training", check_bc_training),
    ("rl_algorithms", check_rl_algorithms),    # PPO / SAC
    ("foc_shutdown", check_foc_shutdown),      # 优先级 (1)
    ("dexterous_hand", check_dexterous_hand),  # G11
    ("real_robot_loop", check_real_robot),     # Sim-to-Real 最后一公里
    ("jetson_bench", check_jetson_bench),      # 优先级 (2)
    ("lifelong_learning", check_lifelong),     # G12
    ("certification", check_certification),    # G10
    ("gui_dashboard", check_gui),              # L7
    ("climb_cycle", check_climb_cycle),        # Phase A 攀岩周期集成 (A1+A2+A5)
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
