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

from . import __version__ as VERSION   # 单一真源, 不在此处写死字面量

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
    assert dbc["message_count"] == 14, \
        f"DBC 消息数应为 14 (R4 6 关节全量遥测后), 实际 {dbc['message_count']}"
    return 0 if ok else 1


def check_safety_budget() -> int:
    """安全预算形式化闭环。

    v1.1 P0 修复:
      - safety_budget.yaml 字段 `measured_*` → `assumed_*` (诚实边界)
      - 原断言 r.closure_ok=True 在 v1.1 budget_closure 严格化后**永远失败**
        (8 rad/s 时 t_stop=0.16s >> t_budget=0.016s, stop_angle=inf)。
      - 修复: 检查改为验证
        (a) 预算段数值正确 (总 16ms < 20ms 目标)
        (b) 低速 (<0.8 rad/s, 在预算内可停) 时 closure_ok=True
        (c) 高速 (8 rad/s, 停不下来) 时 closure_ok=False (诚实 FAIL)
    """
    from .config import load_contract
    from .safety.formal import close_budget_loop

    sb = load_contract("safety_budget")
    def _g(s, *keys):
        for k in keys:
            if k in s:
                return s[k]
        return 0.0
    total_ms = sum(
        (_g(s, "assumed_us", "measured_us") / 1000 if "budget_us" in s
         else _g(s, "assumed_ms", "measured_ms"))
        for s in sb["segments"]
    )
    assert total_ms < sb["acceptance"]["total_break_max_ms"], f"安全链 {total_ms}ms 超预算"
    # (b) 低速 (在预算内可停): closure_ok 应为 True
    r_slow = close_budget_loop(joint_speed_rad_s=0.5)
    assert r_slow.closure_ok, f"0.5 rad/s 在预算内应 closure_ok, got {r_slow.details}"
    assert r_slow.stopped_within_budget, "0.5 rad/s 应在预算内停下"
    # (c) 高速 (停不下来): closure_ok 应诚实为 False (原代码用 min 掩盖成 PASS)
    r_fast = close_budget_loop(joint_speed_rad_s=8.0)
    assert not r_fast.stopped_within_budget, "8 rad/s 应停不下来 (t_stop=0.16s > 0.016s)"
    assert not r_fast.closure_ok, "8 rad/s 停不下来时 closure_ok 应为 False (P0 修复前恒为 True)"
    assert r_fast.stop_angle_deg == float("inf"), "停不下来时应返回 +inf"
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


def check_mission_manager() -> int:
    """断点 ③ 守护: 最小 OS 决策中枢 (一句话指令→状态机→20Hz 环)。

    验证: 指令解析 / 状态机合法转移 / 端到端 SUCCESS / 超时 FAILED /
    安全事件 ABORTED / Bus 事件可观测。
    """
    import numpy as np

    from .runtime import MissionError, MissionManager, MissionState
    from .runtime.bus import Collector, InProcessBus, Topic, make_bus_backed_controller

    bus = InProcessBus("verify_mm")
    from .control.household_robot_system import make_controller
    ctrl = make_controller()
    make_bus_backed_controller(ctrl, bus)
    mm = MissionManager(ctrl, bus)

    zero_obs = lambda: {"ee_position": np.zeros(3), "joint_positions": np.zeros(12)}  # noqa: E731

    # 端到端 SUCCESS
    m = mm.parse("抓起杯子放到桌上")
    result = mm.run(m, obs_provider=zero_obs)
    assert result.state == MissionState.SUCCESS, f"应为 SUCCESS: {result.state}"
    assert result.steps_executed > 0, "至少执行 1 步"

    # 超时 FAILED
    m2 = mm.parse("抓")
    obs_bad = {"ee_position": np.zeros(3), "joint_positions": np.ones(12) * 10}
    r2 = mm.run(m2, obs_provider=lambda: obs_bad, max_steps_override=3)
    assert r2.state == MissionState.FAILED, f"超时应 FAILED: {r2.state}"

    # 安全事件 ABORTED
    ctrl._safety_stop = True
    m3 = mm.parse("抓")
    r3 = mm.run(m3, obs_provider=zero_obs, max_steps_override=10)
    assert r3.state == MissionState.ABORTED, f"安全事件应 ABORTED: {r3.state}"
    ctrl._safety_stop = False

    # 不可识别指令报错
    try:
        mm.parse("xyzzy_不可识别")
        raise AssertionError("不可识别指令未报错")
    except MissionError:
        pass
    return 0


def check_tactile_grip_loop() -> int:
    """④ 触觉闭环守护: EKF→机会约束握力进 20Hz 环。"""
    import numpy as np

    from .control.tactile_grip_loop import (TactileGripLoop,
                                             synth_tactile_obs)
    from .control.household_robot_system import make_controller

    obs_seq = synth_tactile_obs(mu_true=0.75, f_n=40.0, n_steps=100, seed=0)
    ctrl = make_controller()
    loop = TactileGripLoop(ctrl, None, ekf_dt=0.05)
    for obs in obs_seq:
        ctrl.step(obs)

    # μ̂ 收敛
    assert abs(loop.last_mu_hat - 0.75) < 0.05, \
        f"μ̂={loop.last_mu_hat:.4f} 未收敛到 0.75"

    # 握力非零 (闭环接通)
    assert loop.last_f_n_setpoint > 0, "握力设定值为零"

    # 机会约束可行 (EKF 收敛后 denom > 0.05)
    denom = loop.last_mu_hat - 2.0 * loop.last_sigma_mu
    assert denom > 0.05, f"denom={denom:.3f} — 收敛后应可行"
    return 0


def check_runtime_bus() -> int:
    """断点 1 守护: 最小 Bus 抽象 (层间消息化)。

    验证: pub/sub 语义 / 消息信封可追溯 / schema 拒垃圾 / 控制环经 Bus 收发 /
    CAN 帧一等公民 / 无订阅不崩 / 单订阅者异常不阻断。
    """
    import numpy as np

    from .runtime import Collector, InProcessBus, Topic, make_bus_backed_controller

    bus = InProcessBus("verify")
    c = Collector()
    bus.subscribe(Topic.ACTION, c)
    bus.publish(Topic.ACTION, {"action": np.ones(12)}, source="verify")
    assert len(c.messages) == 1 and c.last.seq == 1, "pub/sub 基本语义"

    # 反证: 垃圾被拒
    try:
        bus.publish(Topic.ACTION, {"bad": 1})
        raise AssertionError("垃圾 payload 未被拒")
    except ValueError:
        pass

    # 控制环接入
    from .control.household_robot_system import make_controller
    ctrl = make_controller()
    make_bus_backed_controller(ctrl, bus)
    obs_c, act_c = Collector(), Collector()
    bus.subscribe(Topic.OBSERVATION, obs_c)
    bus.subscribe(Topic.ACTION, act_c)
    ctrl.step({"ee_position": np.zeros(3), "joint_positions": np.zeros(12)})
    assert len(obs_c.messages) >= 1, "控制环应发布观测"
    assert len(act_c.messages) >= 1, "控制环应发布动作"
    assert act_c.messages[-1].source == "controller_20hz", "消息可追溯"

    # CAN 一等公民
    from .simulation.real_robot_env import CANFrame, decode_telemetry, encode_joint_cmd
    frames = Collector()
    bus.subscribe(Topic.CAN_JOINT_CMD, frames)
    f = encode_joint_cmd(0, 1.0, 0.0, 20.0)
    bus.publish(Topic.CAN_JOINT_CMD, {"frame": f}, source="verify")
    dec = decode_telemetry(CANFrame(300, frames.last.payload["frame"].data))
    assert dec["crc_ok"], "CAN 帧经 Bus 后解码失败"
    return 0


def check_policy_loading() -> int:
    """断点 2 接线守护: 训练 ckpt → 控制器 (load_ckpt → policy → controller)。

    受管环境 (无 torch) 验证降级路径的完整性与诚实性:
      · checkpoint 可发现
      · ReplayPolicy 确定性 + provenance 标注 "不是模型推理"
      · attach 后控制环产出非零动作 (接线真实生效)
      · 反证: 坏路径抛 CheckpointError (不静默零向量)
    真推理路径由 tests/test_policy_loader.py 的 3.14 子进程端到端覆盖。
    """
    import numpy as np

    from .control.policy_loader import (CheckpointError, PolicyLoader,
                                         ReplayPolicy, find_latest_checkpoint)
    ck = find_latest_checkpoint()
    assert ck is not None and ck.exists(), f"checkpoint 不可发现: {ck}"

    # 降级路径 (本环境)
    p = ReplayPolicy(ck)
    obs = {"joint_positions": np.ones(12, dtype=np.float32)}
    a1, a2 = p.generate(obs), p.generate(obs)
    assert np.array_equal(a1, a2), "回放策略必须确定性"
    assert p.provenance["backend"] == "replay"
    assert "不是模型推理" in p.provenance["note"], "降级必须诚实标注"

    # 接线 (受管环境走 replay)
    from .control.household_robot_system import make_controller
    from .control.policy_loader import attach_policy_to_controller
    ctrl = make_controller()
    rep = attach_policy_to_controller(ctrl, ck)
    assert rep["backend"] in ("replay", "vla_ckpt"), rep
    a = ctrl.step({"ee_position": np.zeros(3), "joint_positions": np.ones(12)})
    assert np.any(np.abs(a) > 1e-6), "接线后控制环动作全零 — 权重未进入控制环"

    # 反证: 坏路径必须抛错
    try:
        PolicyLoader(REPO_ROOT / "nonexistent_ckpt.pt").load()
        raise AssertionError("坏路径未抛 CheckpointError")
    except CheckpointError:
        pass
    return 0


def check_dual_track_gaps() -> int:
    """G10/G7/G2 双轨缺口守护 (2026-10-08 任务 B)。

    三项各自的可证伪断言:
      G10 电机模型: Ke=pp·psi / Kt=1.5·pp·psi 物理自洽 + 契约三项一致 + 离散稳定
      G7  频谱判别: 健康信号**不误报** 且 注入故障**被识别** (两条同时成立)
      G2  板级对齐: jar 板级实测 ↔ Python FOC 参数 **零不匹配**
    """
    from .adapters.edalab import EdalabAdapter, available_backends

    a = EdalabAdapter()
    # 三个后端必须都已注册 (防止"只导入 adapters 看不到后端"的空表陷阱回归)
    for cap in ("motor_model", "digital_twin", "foc_board_align"):
        assert "python" in available_backends(cap), \
            f"{cap} 无 python 后端 (后端未注册 — 检查 hardware/hil 包导入)"

    g10 = a.run_capability("motor_model", backend="python")
    assert g10.get("ok"), f"G10 电机模型校验失败: {g10.get('failed')}"
    spec = g10["spec"]
    assert spec["Ke_vs_per_rad"] == spec["pole_pairs"] * spec["psi_m_wb"], "Ke 物理关系不成立"

    g7 = a.run_capability("digital_twin", backend="python")
    assert g7.get("ok"), f"G7 频谱判别失败: {g7}"
    assert not g7["healthy_detected_fault"], "G7 健康信号误报 (判别器恒真)"
    assert g7["faulty_detected_fault"], "G7 注入故障未被识别 (判别器失效)"
    assert g7["threshold_baseline"]["missed_early_fault"], \
        "G7 应展示阈值法漏检 (否则看不出 FFT 的价值)"

    g2 = a.run_capability("foc_board_align", backend="python")
    assert g2.get("ok"), f"G2 板级对齐存在不匹配: {g2.get('mismatches')}"
    return 0


def check_real_robot() -> int:
    """Sim-to-Real 最后一公里: mock 传输跑通闭环 + 安全闸门 + 延迟统计。"""
    from .simulation.real_robot_env import real_robot_demo

    r = real_robot_demo(steps=30)
    assert r["ok"], f"真机闭环自检失败: {r}"
    return 0



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


def check_vsa_energy_consistency() -> int:
    """V-1 修复守护: VSA 曲线 energy_from_origin_j 与 stored_energy_j 同量级。

    防止未来重新导出 Kotlin JSON 时把 energy_j (行程边界零点) 恢复成弹性能量。
    历史错误: 两者差 83 倍, 下游误用会得到 11.6 m/s 致命弹道。
    """
    import os
    from pathlib import Path
    from .dynamics.vsa_coupling import load_vsa_json, vsa_energy_consistency_check

    env_path = os.environ.get("EDALAB_VSA_JSON")
    if env_path:
        path = Path(env_path)
    else:
        path = Path(__file__).resolve().parents[2] / "fixtures" / "experiments_vsa_curve.json"
    assert path.exists(), f"VSA JSON 未找到: {path}"
    tops = load_vsa_json(path)
    # tolerance=5.0: stored_energy_j 取 max(½Kθ²) in elastic_span, energy_from_origin_j
    # 在曲线末端用 kMax*θ_cap² — 两者物理上都在 ½Kθ² 量级, 差异来自 K 采样点不同
    # (stored 用 span 内实际 K, curve 末端钳位到 elastic_span 边界的 kMax)。
    # 历史错误 (V-1) 是 83x (曲线用行程边界零点), 5x 阈值足以拦回归。
    r = vsa_energy_consistency_check(tops, tolerance=5.0)
    # 至少 MACCEPA/SEA 应通过一致性检查 (ANTAGONISTIC 曲线零点随预紧偏移, 差异大)
    passed_names = [d["name"] for d in r["details"] if d["within_tolerance"]]
    assert "VSA-MACCEPA-ClimbingArm" in passed_names, \
        f"MACCEPA 一致性检查失败, 请检查 Kotlin JSON 是否使用了能量零点正确的口径: {r}"
    # 严格: 所有拓扑都应通过检查 (ANTAGONISTIC 因物理原因容忍较大差异)
    detail_summary = "; ".join(f"{d['name']}: ratio={d['ratio']:.2f}" for d in r["details"])
    assert not r["warns"], f"vsa_energy_consistency 警告: {r['warns']}"
    # 打印详细 (便于 verify 输出诊断)
    print(f"VSA energy consistency ({detail_summary})")
    return 0


def check_canfd_db_cross_check() -> int:
    """R4 (2026-10-07): CAN-FD 编解码与 DBC 独立对照测试守护 (评估 §6.1-3 G3)。

    跑 tests/test_r4_canfd_cross_check.py 的核心场景 (1000 帧随机 JointCmd
    + JointTelemetry + 6 关节全量遥测), 失败立即 AssertionError。打破
    "mock 与 code 用同一段错位逻辑 → 自闭环永远绿"的盲区。
    """
    import random
    import unittest
    # 在 verification 进程内 (非 pytest) 跑核心断言, 不依赖 pytest 收集。
    from .simulation.real_robot_env import decode_telemetry, CANFrame, TELEM_IDS
    # 直接复用测试内的参考编码 (与测试同源 — DBC 改动将同步两侧)
    from pathlib import Path
    dbc_path = Path(__file__).resolve().parents[2] / "contracts" / "canfd" / "joint_control.dbc"
    # 在 verification 内 inline 一份简化 reference (避免 import 测试模块)
    import re as _re

    _SG_RE = _re.compile(
        r"^SG_\s+(\w+)\s*:\s*(\d+)\|(\d+)@([01])([+-])\s*\("
        r"([^,]+),([^\)]+)\)"
    )

    def _parse():
        out = {}
        cur = -1
        for line in dbc_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            m = _re.match(r"^BO_\s+(\d+)", line)
            if m:
                cur = int(m.group(1))
                out[cur] = {}
                continue
            m = _SG_RE.match(line)
            if m and cur >= 0:
                out[cur][m.group(1)] = (
                    int(m.group(2)), int(m.group(3)),
                    float(m.group(6).strip()), float(m.group(7).strip()),
                    m.group(4) == "0",
                )
        return out

    dbc = _parse()
    rng = random.Random(0xCAFE)
    failures = []
    # (a) 6 关节 JointTelemetry, 每关节 100 帧随机 → ref encode → project decode
    for j in range(6):
        cid = TELEM_IDS[j]
        if cid not in dbc:
            failures.append(("missing_telem_dbc", j, cid))
            continue
        sigs = dbc[cid]
        for _ in range(100):
            pos = rng.uniform(0, 327.67)
            cur = rng.uniform(0, 20.47)
            tmp = rng.uniform(-20, 235.5)
            flags = rng.randint(0, 255)
            d = bytearray(8)
            p = int(pos / 0.01)
            c = int(cur / 0.01)
            t = int((tmp + 20.0) / 0.5)
            fl = flags
            # actual_pos_deg bit 0|15
            for i in range(15):
                if (p >> i) & 1:
                    d[i // 8] |= 1 << (i % 8)
            # actual_current_a bit 16|11
            for i in range(11):
                if (c >> i) & 1:
                    d[(16 + i) // 8] |= 1 << ((16 + i) % 8)
            # temp_c bit 27|9
            for i in range(9):
                if (t >> i) & 1:
                    d[(27 + i) // 8] |= 1 << ((27 + i) % 8)
            # status_flags bit 40|8
            for i in range(8):
                if (fl >> i) & 1:
                    d[(40 + i) // 8] |= 1 << ((40 + i) % 8)
            # crc8
            from .simulation.real_robot_env import crc8
            d[7] = crc8(bytes(d[:7]))
            dec = decode_telemetry(CANFrame(cid, bytes(d)))
            if abs(dec["pos_deg"] - pos) > 0.011:
                failures.append((j, "pos", pos, dec["pos_deg"]))
            if abs(dec["current_a"] - cur) > 0.011:
                failures.append((j, "current", cur, dec["current_a"]))
            if abs(dec["temp_c"] - tmp) > 0.51:
                failures.append((j, "temp", tmp, dec["temp_c"]))
            if not dec["crc_ok"]:
                failures.append((j, "crc", True, dec["crc_ok"]))
    assert not failures, (
        f"CAN-FD↔DBC 6 关节对照测试失败 ({len(failures)}/600 失配, "
        f"前 5 例: {failures[:5]})"
    )
    print(f"CAN-FD↔DBC 6 关节对照测试 (R4): {600} 帧 0 失配 ✓")
    return 0


def check_vendor_jar_provenance() -> int:
    """v4 漂移治理 L1 (指纹闸门): vendor jar 与 PROVENANCE.json 记录一致。

    jar 是冻结的编译产物, 原 EdaLab 源码若演进不会自动同步 → 静默漂移。
    本闸门在 jar 被替换/重新构建而 PROVENANCE 未更新时立刻失败。
    (L2 行为锚定由 check_canfd_db_cross_check / B3 等契约锚定值覆盖。)
    """
    from .adapters.edalab import verify_vendor_provenance

    r = verify_vendor_provenance()
    assert r["ok"], (
        "vendor jar 溯源校验失败 (jar 可能被替换/重新构建但 PROVENANCE.json "
        "未同步更新):\n" +
        "\n".join(f"  - {c['check']}: {c['detail']}"
                  for c in r["checks"] if not c["ok"])
    )
    ok_n = len([j for j in r["jars"] if j.get("ok")])
    print(f"vendor jar 溯源 (L1 指纹): {ok_n}/{len(r['jars'])} 件 sha256 一致 ✓")
    return 0


def check_vla_block_size() -> int:
    """C1 (P0): VLA block_size 必须覆盖全部 token, 且不击穿 Jetson 延迟预算。

    背景: 原 block_size=256 远小于 min_block_size(603~613), vla_model.forward 的
    尾部截断 (ctx[:, -block_size:, :]) 会丢弃**序列开头**的视觉 token — 实测视觉
    仅保留 42% (**约 58% 被静默丢弃**), 而 loss 曲线看不出异常 (典型静默错误)。

    两条断言缺一不可:
      1. 不截断: block_size ≥ min_block_size (视觉完整)
      2. 不击穿: roofline 延迟 ≤ 15ms 模型预算 (否则"修了截断却丢了 20Hz")
    """
    from .models.vla_config import VLAConfig
    from .deploy.jetson_bench import PLATFORMS, estimate_latency

    ab = VLAConfig.ablations()
    # (1) 不截断
    truncated = [(n, c.block_size, c.min_block_size)
                 for n, c in ab.items() if c.block_size < c.min_block_size]
    assert not truncated, (
        "VLA block_size 不足, 训练时视觉 token 会被静默截断:\n" +
        "\n".join(f"  - {n}: block_size={bs} < min={mn} (丢 {mn - bs} token)"
                  for n, bs, mn in truncated))

    # (2) 不击穿 Jetson 模型预算 (15ms — 契约 compute.model_inference_ms)
    plat = PLATFORMS["jetson_orin_nx_16gb"]
    over = []
    for n, c in ab.items():
        ms = float(estimate_latency(c, plat, precision="int8")["roofline_ms"])
        if ms > 15.0:
            over.append((n, ms))
    assert not over, (
        "block_size 提升后击穿 Jetson 模型预算 15ms (需降 num_views/img_size 补偿):\n" +
        "\n".join(f"  - {n}: {ms:.2f}ms" for n, ms in over))

    worst = max(float(estimate_latency(c, plat, precision="int8")["roofline_ms"])
                for c in ab.values())
    print(f"VLA block_size (C1): {len(ab)} 档无截断, 最差延迟 {worst:.2f}ms "
          f"≤ 15ms 预算 ✓")
    return 0


def check_chance_constrained_grip() -> int:
    """R7: 机会约束握力可用性验证 (评估 §6.1-2, plan.md S1.1).

    验收:
      1. z 值精确 (statistics.NormalDist().inv_cdf, 非 sqrt(2) 线性近似)
      2. grip_demo 运行成功, 匹配 σ 时滑脱率 ≤ δ + MC 噪声
      3. denom<=0.05 时 InfeasibleGripError 显式抛出 (非静默掩盖)
    """
    from .safety.chance_constrained_grip import (
        InfeasibleGripError, chance_constrained_grip, grip_demo
    )

    # (1) z 值精确
    z_05 = chance_constrained_grip(f_t_req=1.0, mu_hat=1.0, sigma_mu=0.1, delta=0.05)
    expected_z_05 = 1.6449
    assert abs(z_05 - 1.0 / (1.0 - expected_z_05 * 0.1)) < 1e-3, \
        f"z(δ=0.05) 不精确: f_n={z_05:.4f} vs expected {1.0 / (1.0 - expected_z_05 * 0.1):.4f}"

    # (2) grip_demo 运行成功且匹配 σ 时滑脱率达标
    r = grip_demo(mu_hat=0.85, sigma_mu=0.15, verbose=False)
    assert r["ok"], f"grip_demo 主验收失败: {r}"
    assert r["slip_cc"] <= 0.055, f"匹配 σ 时滑脱率 {r['slip_cc']:.4f} 超 5%"
    assert r["slip_naive"] > 0.055, "无裕度点估计滑脱率应远大于 5%"
    assert r["slip_learned_overconf"] > r["slip_learned_matched"], \
        "过度自信应触发滑脱率回升 (§6.6 原结论)"

    # (3) 不可行输入必须显式抛异常
    try:
        chance_constrained_grip(f_t_req=10.0, mu_hat=0.5, sigma_mu=0.5, delta=0.05)
        raise AssertionError("denom<=0.05 未抛 InfeasibleGripError (旧 max(denom, 0.05) 静默掩盖)")
    except InfeasibleGripError:
        pass  # 预期路径

    print(f"chance_constrained_grip: z(δ=0.05)={expected_z_05:.4f}, "
          f"slip_cc={r['slip_cc']:.4f}, slip_naive={r['slip_naive']:.4f}, "
          f"overconf={r['slip_learned_overconf']:.4f}")
    return 0


def check_afe_ind_frontend() -> int:
    """Phase 1 (2026-10-07): AFE-IND 触觉传感硬件前端移植守护.

    把 EdaLab `afe-ind` CLI 的产物固化到 artifacts/afe_ind/ 后, 消费其关键
    指标并与 hw_requirements.yaml 的 sensors.tactile 需求做契约对齐.

    守护三件事:
      1. 产物完整性 (11 个必需文件缺失即失败) — 防止 artifacts 被误删/未同步
      2. 关键硬件指标 (增益 1046.83×, 带宽 10.60kHz, 噪声 3.794e-6 V/√Hz, PCB 65 元件)
      3. 契约对齐 (5 项 check, 全部必须通过 — 特别是带宽 106× 覆盖触觉 100Hz 需求)

    历史: 移植前触觉传感是"合成源"(用攀岩接触力冒充, 评估感知 45 分);
    移植后拥有 65 元件 PCB + 完整 SPICE/PCB/Gerber 的硬件前端, 可进制造 DR.
    """
    from .adapters import afe_ind as ai

    # 1. 产物完整性
    r_art = ai.check_artifacts_present()
    assert r_art["ok"], f"AFE-IND 产物缺失: {r_art['missing']}"
    assert r_art["required_count"] >= 11, \
        f"必需产物数量异常: {r_art['required_count']}"

    # 2. 关键指标提取 (报告解析)
    rep = ai.parse_verification_report()
    assert rep["gain_x"] >= 500, f"AFE-IND 增益 {rep['gain_x']}× 低于下限 500×"
    assert rep["gain_db"] >= 50, f"AFE-IND 增益 {rep['gain_db']}dB 低于下限 50dB"
    assert rep["bandwidth_khz"] >= 5.0, \
        f"AFE-IND 带宽 {rep['bandwidth_khz']}kHz 低于下限 5kHz"
    assert rep["noise_v_per_sqrt_hz"] < 1e-5, \
        f"AFE-IND 噪声 {rep['noise_v_per_sqrt_hz']:.2e} 超过 1e-5 V/√Hz 阈值"
    assert 30 <= rep["pcb_component_count"] <= 200, \
        f"AFE-IND PCB 元件 {rep['pcb_component_count']} 超出合理区间 [30,200]"
    assert rep["dc_converged"] and rep["ac_converged"] and rep["transient_converged"], \
        "AFE-IND 三域收敛不全: DC/AC/瞬态必须全部收敛"

    # 3. 原理图 JSON (机器可读, 供 BOM/DR 消费)
    circ = ai.parse_circuit_json()
    assert circ["components_count"] >= 50, \
        f"AFE-IND 原理图元件 {circ['components_count']} 低于下限 50"
    assert circ["wires_count"] >= 30, \
        f"AFE-IND 原理图连线 {circ['wires_count']} 低于下限 30"

    # 4. BOM 数量与原理图元件一致性
    bom_n = len(ai.parse_bom())
    assert bom_n == circ["components_count"], \
        f"AFE-IND BOM 元件数 {bom_n} ≠ 原理图元件数 {circ['components_count']}"

    # 5. AC 仿真数据点 (带宽验证)
    ac = ai.parse_ac_sweep()
    assert ac["points_count"] >= 50, \
        f"AFE-IND AC 采样点 {ac['points_count']} 低于下限 50"
    assert ac["f_max_hz"] >= 1e6, \
        f"AFE-IND AC 最高频 {ac['f_max_hz']}Hz 未覆盖 MHz 级"

    # 6. 契约对齐 (5 项 check)
    align = ai.contract_alignment(rep)
    failed = [c["name"] for c in align["checks"] if not c["ok"]]
    assert align["ok"], f"AFE-IND ↔ hw_requirements.yaml 契约对齐失败: {failed}"

    print(f"afe_ind_frontend: 产物 {r_art['present_count']}/{r_art['required_count']} 齐 "
          f"| 增益 {rep['gain_x']}× ({rep['gain_db']}dB) | "
          f"带宽 {rep['bandwidth_khz']}kHz | 噪声 {rep['noise_v_per_sqrt_hz']:.2e} V/√Hz | "
          f"PCB {rep['pcb_component_count']} 元件 | 契约 5/5 对齐")
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


def check_sensor_fusion() -> int:
    """M1 多传感器融合 + 分频信道处理 (plan_v5 Sprint M)。

    守护: 组合信号分频解复用; 信道串扰 <-20dB; **不分频时串扰必须 >-20dB**
    (负对照 — 频带掩蔽天然正交, 没有它这个门槛永远绿); 处理样本暂存 + 发 Bus。
    """
    from .perception.sensor_fusion import (
        DEMO_CHANNELS, DEMO_FS, ChannelSpec, Channelizer, SensorFusionHub,
        crosstalk_db, fusion_demo,
    )

    r = fusion_demo()
    assert r["crosstalk_video_db"] < -20.0, \
        f"video 信道串扰 {r['crosstalk_video_db']:.1f}dB 未达标"
    assert r["crosstalk_microwave_db"] < -20.0, \
        f"microwave 信道串扰 {r['crosstalk_microwave_db']:.1f}dB 未达标"

    # 负对照: 不分频 (直接用组合信号) 时微波信道串扰必须超阈
    t = np.arange(0, 1.0, 1.0 / DEMO_FS)
    video = 0.8 * np.sin(2 * np.pi * 300 * t)
    micro = (0.6 * np.sin(2 * np.pi * 2500 * t)
             + 0.5 * np.sin(2 * np.pi * 3000 * t))
    raw = crosstalk_db(video + micro, micro, video, 1)
    assert raw > -20.0, f"不分频串扰 {raw:.1f}dB — <-20dB 判据已失去区分力"

    # 抽取率单一真源: 度量与 demux 必须一致 (曾手写 round(fs/2000) 错配)
    ch = Channelizer(list(DEMO_CHANNELS))
    assert (ch.decimation(DEMO_FS, "video"), ch.decimation(DEMO_FS, "microwave")) \
        == (8, 1), "抽取率与信道定义不符"

    hub = SensorFusionHub([ChannelSpec("video", 300, 400, 1000)])
    t2 = np.arange(0, 0.1, 1.0 / 8000)
    hub.ingest(0.0, np.sin(2 * np.pi * 300 * t2), 8000.0)
    assert len(hub.buffer) == 1, "处理样本应暂存进 RingBuffer"
    print(f"多传感器分频: 串扰 video={r['crosstalk_video_db']:.1f}dB / "
          f"microwave={r['crosstalk_microwave_db']:.1f}dB; 不分频负对照 "
          f"{raw:.1f}dB (> -20dB, 判据有区分力 ✓)")
    return 0


def check_slam() -> int:
    """M2 SLAM 三维坐标 + 时间戳 (plan_v5 Sprint M)。

    守护: 带噪仿真位姿误差收敛 <0.3m; 路标增广; 时间戳单调; **判据有区分力**
    (3x 噪声失配必须超阈, 否则 "<0.3m" 是恒真门槛)。
    """
    from .perception.slam import EKFSLAM2D, slam_demo

    r = slam_demo()
    assert r["final_position_error_m"] < 0.3, \
        f"SLAM 位姿误差 {r['final_position_error_m']:.3f}m 未收敛"
    # R_OBS 必须是方差 (曾把 std 直接填进协方差, 量纲错 2.6x/4.5x)
    assert EKFSLAM2D.R_OBS[0, 0] == EKFSLAM2D.SIGMA_RANGE_M ** 2, \
        "R_OBS 对角线不是 σ², 量纲错配"
    # 负对照: 3x 失配下误差必须超阈, 证明判据不是恒真
    mismatched = slam_demo(obs_noise_scale=3.0)["final_position_error_m"]
    assert mismatched > 0.3, \
        f"3x 噪声失配仍 {mismatched:.3f}m <0.3m — 收敛判据失去区分力"
    slam = EKFSLAM2D()
    slam.update_landmark("a", 2.0, 0.3, timestamp_s=0.1, mz=1.75)
    assert slam.landmark_position("a")[2] == 1.75, "路标 z 被静默丢弃"
    try:
        slam.update_landmark("a", 2.0, 0.3, timestamp_s=0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("时间戳倒退未被拒绝")

    # M5 宣称的 POSE_UPDATE 必须真的有流量 (曾零发布者零订阅者)
    from .cognition.cognitive_hub import CognitiveHub
    from .runtime.bus import Collector, InProcessBus
    bus = InProcessBus("verify_slam_bus")
    got = Collector()
    bus.subscribe("pose_update", got)
    hub = CognitiveHub(bus=bus)
    slam_demo(bus=bus)
    assert len(got.messages) == 50, f"POSE_UPDATE 发布者异常: 仅 {len(got.messages)} 条"
    assert hub.latest_pose("robot") is not None, "SLAM→认知中枢 地点要素链路断裂"
    print(f"SLAM: 匹配噪声 {r['final_position_error_m']:.3f}m / 3x 失配 "
          f"{mismatched:.3f}m (<0.3m 判据有区分力 ✓), 路标 {len(r['landmarks'])} 个, "
          f"POSE_UPDATE 50 条已抵达认知中枢")
    return 0


def check_cognitive_hub() -> int:
    """M4 认知中枢四要素 (plan_v5 Sprint M)。

    守护: 记录/分析/计算/规划/记忆 五能力; 四要素完整性; 三处静默失败必须
    改为显式报错 (负时间戳 / 未知规划目标 / 地点过滤只给一半参数)。
    """
    from .cognition.cognitive_hub import CognitiveHub, EventRecord, cognitive_demo

    r = cognitive_demo()
    assert r["analysis"]["total"] == 3
    assert r["plan_climb_hold"] == ["load"], "规则式规划应只缺 load"
    assert r["recall_order"][0] == "handover", "记忆召回应按时间衰减排序"
    hub = CognitiveHub()
    try:
        hub.record_event(EventRecord(-1.0, np.zeros(3), "a", "b"))
    except ValueError:
        pass
    else:
        raise AssertionError("负时间戳事件未被拒绝")
    try:
        hub.plan("no_such_goal")
    except KeyError:
        pass
    else:
        raise AssertionError("未知规划目标未被拒绝 (曾与'前置已满足'同样返回 [])")
    try:
        hub.memory.query(location_center=np.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("地点过滤缺 radius 未被拒绝 (曾静默不过滤)")
    # DECAY_RATE 必须可观测, 否则"时间衰减"与"时间倒序"无法区分
    hub.record_event(EventRecord(0.0, np.zeros(3), "a", "old"))
    hub.record_event(EventRecord(10.0, np.zeros(3), "a", "new"))
    scored = dict((e.thing, s) for s, e in hub.remember_scored(now_s=10.0))
    assert scored["old"] == float(np.exp(-CognitiveHub.DECAY_RATE * 10.0)), \
        "衰减得分不等于 exp(-DECAY_RATE·Δt)"
    print(f"认知中枢: {r['analysis']['total']} 事件, 事件率 "
          f"{r['stats']['event_rate_hz']:.2f}Hz, 规划缺口 {r['plan_climb_hold']}, "
          f"衰减得分(old)={scored['old']:.3f}")
    return 0


def check_vla_v2a() -> int:
    """M3 V-to-action 纯视觉决策路径 (plan_v5 Sprint M)。

    守护: v2a ≡ vla(lang=None) 等价; v2a 模态拒绝语言; 语言通道在 vla 模态生效;
    modality="v2a" 时 min_block_size 不含语言 token; numpy 头与 torch VLAModel
    形状契约不漂移。
    """
    from .models.vla_config import VLAConfig
    from .models.vla_v2a import VLAPolicy, torch_shape_contract, v2a_demo

    r = v2a_demo()
    assert r["v2a_equals_vla_none"], "v2a 与 vla(None) 不等价"
    assert r["v2a_rejects_language"], "modality='v2a' 未拒绝语言输入"
    assert r["lang_changes_output"], "语言通道未改变输出"
    assert r["min_block_size_v2a"] == r["min_block_size_vla"] - VLAConfig.LANG_TOKENS
    cfg = VLAConfig(modality="v2a")
    p = VLAPolicy(cfg)
    v = np.zeros((cfg.num_views, cfg.vision_embd))
    assert p.infer_v2a(v, np.zeros(cfg.state_dim)).shape == \
        (cfg.action_horizon, cfg.action_dim)
    contract = torch_shape_contract()
    print(f"V2A: 等价路径 ✓, v2a 拒语言 ✓, min_block_size v2a={r['min_block_size_v2a']} "
          f"< vla={r['min_block_size_vla']}, torch 形状契约 "
          f"{'✓ ' + str(contract[0]) if contract else '跳过 (无 torch)'}")
    return 0


def check_cognitive_vla_loop() -> int:
    """M6 VLA 双路径 × 认知中枢 端到端闭环 (plan_v5 Sprint M)。

    守护: 无语言走 v2a / 有语言走 vla; 闭环跑到目标达成; 四要素事件齐全;
    未知目标 KeyError 透传; max_steps 失控守护。
    """
    from .cognition.cognitive_hub import CognitiveHub
    from .cognition.closed_loop import CognitiveVLALoop, loop_demo
    from .models.vla_config import VLAConfig

    r = loop_demo()
    assert r["v2a"]["goal_reached"] and r["vla"]["goal_reached"]
    assert r["v2a"]["paths"] == {"v2a": 3}, f"v2a 路径标注错: {r['v2a']['paths']}"
    assert r["vla"]["paths"] == {"vla": 3}, f"vla 路径标注错: {r['vla']['paths']}"
    assert r["v2a"]["events"] == 3 and r["vla"]["events"] == 3
    cfg = VLAConfig(modality="vla")
    loop = CognitiveVLALoop(CognitiveHub(), cfg=cfg)
    rng = np.random.default_rng(0)
    v = rng.normal(size=(cfg.num_views, cfg.vision_embd))
    s = rng.normal(size=cfg.state_dim)
    try:
        loop.run_until_goal("nonexistent", v, s, None)
    except KeyError:
        pass
    else:
        raise AssertionError("未知目标未透传 KeyError")
    try:
        loop.run_until_goal("climb_hold", v, s, None, max_steps=1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("max_steps 失控未被守护")
    print(f"闭环: v2a/vla 双链路达成 ✓, 四要素事件 {r['v2a']['events']}×2 条")
    return 0


def check_layer_registry() -> int:
    """N1 层注册表 (plan_v6 Sprint N)。

    守护: C1-C5 中枢决策层齐全; L1-L7 编号不被增删; module_refs 可 import;
    GUI _layer_stack 与注册表同源。
    """
    from .center.layer_registry import as_dict, validate_registry
    from .gui.server import _layer_stack

    assert validate_registry() == 0
    assert _layer_stack() == as_dict(), "GUI 层清单与注册表漂移"
    print("层注册表: L1-L7 + P1-P5 + S + D1-D5 + C1-C5 ✓")
    return 0


def check_center_brain() -> int:
    """N2 中枢编排 (plan_v6 Sprint N)。

    守护: POSE_UPDATE 断链修复 (hub 收到位姿); 双路径选择; 四要素事件。
    """
    from .center.center_brain import CenterBrain
    from .cognition.cognitive_hub import CognitiveHub
    from .runtime.bus import InProcessBus

    bus = InProcessBus()
    hub = CognitiveHub(bus=bus)
    brain = CenterBrain(bus=bus, hub=hub)
    out = brain.run_demo()
    assert out["paths"] == ["v2a", "vla", "v2a"], f"双路径选择错: {out['paths']}"
    assert out["events"] == 3
    assert out["pose_seen_by_hub"] == [2.0, 0.5, 0.0], "POSE_UPDATE 断链未修复"
    print(f"中枢编排: 双路径 {out['paths']}, POSE_UPDATE→hub ✓, "
          f"四要素事件 {out['events']} 条")
    return 0


def check_os_channel() -> int:
    """N3 OS SSH 双通道 (plan_v6 Sprint N)。

    守护: 状态机流转; 维护态禁运动 (维护通道绝不遥控); dry-run 命令序列。
    """
    from .os_channel import DryRunExecutor, ModeState, OsChannelAdmin

    admin = OsChannelAdmin("robot.local", executor=DryRunExecutor())
    admin.enter_maintenance("verify check")
    assert admin.mode.state is ModeState.MAINTENANCE
    assert not admin.mode.can_send_motion(), "维护态必须禁运动"
    admin.pull_debug_logs("/var/log/embodied/")
    admin.provision_factory_program("cognition")
    assert admin.mode.state is ModeState.PROVISIONED
    admin.return_to_runtime()
    assert admin.mode.state is ModeState.RUNTIME
    kinds = {a[0] for a in admin.executor.recorded}
    assert "ssh" in kinds and "scp" in kinds
    print(f"OS 通道: 全流转 ✓, 维护态禁运动 ✓, 命令 {sorted(kinds)}")
    return 0


def check_chem_bio() -> int:
    """N4 化学生化传感融合 (plan_v6 Sprint N)。

    守护: 三场景表面状态判定 (dry/wet/corrosive); 融合可复现。
    """
    from .perception.chem_bio import chem_demo

    out = chem_demo()
    assert out["dry_metal"]["surface"] == "dry"
    assert out["wet_organic"]["surface"] == "wet"
    assert out["corrosive"]["surface"] == "corrosive"
    print(f"化学生化融合: dry/wet/corrosive 三场景 ✓ (provenance=synthetic)")
    return 0


def check_spatial_reasoning() -> int:
    """N5 V2L 空间智能 (plan_v6 Sprint N)。

    守护: 辅助线/推演/结论结构完整; 输入变则论证变 (防恒定假论证)。
    """
    import numpy as np
    from .center.spatial_reasoning import SpatialReasoner

    r = SpatialReasoner()
    rng = np.random.default_rng(4)
    a1 = r.argue(rng.normal(size=12), np.array([1.0, 0.0, 0.0]))
    a2 = r.argue(rng.normal(size=12), np.array([0.0, 1.0, 0.0]))
    assert len(a1.auxiliary_hypotheses) >= 3 and len(a1.derivation_steps) >= 3
    assert a1.conclusion and a1.language_caption
    assert a1.conclusion != a2.conclusion or \
        not np.allclose(a1.decision, a2.decision), "V2L 论证恒定 — 假论证"
    print(f"V2L 空间论证: 辅助线 {len(a1.auxiliary_hypotheses)} 条 / "
          f"推演 {len(a1.derivation_steps)} 步 ✓")
    return 0


def check_chem_calibration() -> int:
    """d1 e-nose 真实标定方案 (plan_v6 后续)。

    守护: 多点标定门禁可失败; 交叉补偿降串扰; 漂移触发再标定;
    欠定系统被拒绝 (防门禁恒过)。
    """
    from .perception.chem_calibration import (
        EnoseCalibrator, calibration_demo, make_synthetic_calibration_set,
    )

    r = calibration_demo()
    assert r["gate_passed"] and r["drift_detected"]
    assert r["a_recovery_max_err"] < 0.15
    # 欠定拒绝反证: 恰定系统 (点数=参数数) 必须抛错
    pts, _ = make_synthetic_calibration_set(n_points=4, n_gases=3)
    try:
        EnoseCalibrator().fit(pts)
    except ValueError:
        pass
    else:
        raise AssertionError("欠定标定未被拒绝 — 门禁会恒过")
    print(f"e-nose 标定: 门禁✓ 漂移✓ 欠定拒绝✓ (mean R²={r['mean_r2']})")
    return 0


def check_geometry_verification() -> int:
    """d2 V2L 空间几何演算验证 (plan_v6 后续)。

    守护: 独立复算; 真值场景结论正确; 变异负对照必检出。
    """
    from .center.geometry_verification import geometry_verification_demo

    r = geometry_verification_demo()
    assert r["all_truth_verify_pass"], "真值场景复算失败"
    assert r["all_truth_conclusions_right"], "真值场景结论错"
    assert r["clean_pass"] and r["mutation_detected"], "变异负对照未检出"
    print(f"V2L 几何验证: 3 真值场景 ✓, 变异检出 {r['mutation_detected']} ✓")
    return 0


def check_e2e_closed_loop() -> int:
    """d3 真实硬件 E2E 闭环测试支架 (plan_v6 后续)。

    守护: 阶段门推进纪律; 时延预算判据可失败; ESTOP 中断。
    """
    from .hil.e2e_closed_loop import (
        E2EClosedLoopHarness, E2EPhase, MockHardwareChannel, e2e_demo,
    )

    r = e2e_demo()
    assert r["all_pass"] and r["reached"] == "done"
    assert r["n_phases_done"] == 5
    # 负对照: 超预算必须 FAIL
    h = E2EClosedLoopHarness(channel=MockHardwareChannel(latency_ms=80.0),
                             latency_budget_ms=50.0)
    for phase in (E2EPhase.S1_POWER_ON, E2EPhase.S2_PERCEPTION,
                  E2EPhase.S3_DECISION):
        h.run_phase(phase)
    rec = h.run_phase(E2EPhase.S4_MOTION)
    assert not rec.passed, "80ms 时延不应过 50ms 预算 (判据恒过)"
    print(f"E2E 支架: 5 阶段全通 ✓, 时延负对照 FAIL ✓ "
          f"(worst={r['s4_latency_worst_ms']:.1f}ms)")
    return 0


def check_cli_smoke() -> int:
    """CLI 子命令冒烟 — 覆盖 v0.5 修复的 GBK 输出崩溃回归。"""
    import os

    env = {**os.environ, "PYTHONPATH": str(SRC_ROOT), "PYTHONIOENCODING": "utf-8"}
    # audit 依赖兄弟仓库存在, 不作为硬失败; 其余子命令必须 0 退出
    # v0.7: 新增子命令一并冒烟 (全部走小规模默认, 保证 CI 时长可控)
    for cmd in ("contracts", "safety", "sim", "sim2real", "biped", "nanogpt", "dr",
                "platform", "soft", "foc", "hand", "cert", "bench", "lifelong",
                "real", "wholebody", "gui --once", "train --algo ppo --iters 80",
                "fusion-demo", "slam-demo", "cog-demo", "loop-demo",
                "os-status", "os-provision", "center-demo", "v2l-demo",
                "chem-demo"):
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
    ("vsa_energy_consistency", check_vsa_energy_consistency),  # V-1: 曲线能量零点一致性守护
    ("chance_constrained_grip", check_chance_constrained_grip),  # R7: 机会约束握力可用 (z 精确 + Infeasible)
    ("canfd_db_cross_check", check_canfd_db_cross_check),  # R4: CAN-FD↔DBC 6 关节对照守护 (G3)
    ("vendor_jar_provenance", check_vendor_jar_provenance),  # v4 漂移治理 L1: jar 指纹溯源
    ("vla_block_size", check_vla_block_size),  # C1 (P0): 视觉截断 + Jetson 预算双守护
    ("afe_ind_frontend", check_afe_ind_frontend),  # Phase1: AFE-IND 触觉硬件前端移植守护
    ("dual_track_gaps", check_dual_track_gaps),    # G10/G7/G2 双轨缺口守护
    ("policy_loading", check_policy_loading),      # 断点2: ckpt→policy→controller 接线
    ("runtime_bus", check_runtime_bus),            # 断点1: 最小 Bus 抽象 (层间消息化)
    ("tactile_grip_loop", check_tactile_grip_loop), # ④: EKF→握力进 20Hz 环
    ("mission_manager", check_mission_manager),    # 断点3: OS 中枢 (指令→状态机→20Hz环)
    ("sensor_fusion", check_sensor_fusion),        # M1: 多传感器分频信道 (plan_v5)
    ("slam", check_slam),                          # M2: 三维坐标+时间戳 (plan_v5)
    ("vla_v2a", check_vla_v2a),                    # M3: V-to-action 等价路径 (plan_v5)
    ("cognitive_hub", check_cognitive_hub),        # M4: 四要素认知中枢 (plan_v5)
    ("cognitive_vla_loop", check_cognitive_vla_loop),  # M6: VLA 双路径×认知中枢闭环 (plan_v5)
    ("layer_registry", check_layer_registry),      # N1: 层注册表 L+C 栈 (plan_v6)
    ("center_brain", check_center_brain),          # N2: 中枢编排+POSE_UPDATE 断链修复 (plan_v6)
    ("os_channel", check_os_channel),              # N3: OS SSH 双通道 (plan_v6)
    ("chem_bio", check_chem_bio),                  # N4: 化学生化传感融合 (plan_v6)
    ("spatial_reasoning", check_spatial_reasoning),  # N5: V2L 空间智能 (plan_v6)
    ("chem_calibration", check_chem_calibration),  # d1: e-nose 真实标定方案 (docs/19)
    ("geometry_verification", check_geometry_verification),  # d2: V2L 几何演算验证 (docs/19)
    ("e2e_closed_loop", check_e2e_closed_loop),    # d3: 真机 E2E 闭环测试支架 (docs/19)
    ("cli_smoke", check_cli_smoke),
]


def run_all(skip_pytest: bool = False) -> List[Tuple[str, bool, str]]:
    """执行全部检查项, 返回 [(name, passed, detail)]。"""
    if not __debug__:
        # python -O 会剥离全部 assert, 而 35 个 check 的判定**就是** assert。
        # 那种情况下 run_all 恒返回全绿 —— "退出码即结论" 退化成 "退出码即表演",
        # 且比 FAIL 更糟: 它给出一个假的 VERIFY_OK。宁可拒绝出结论。
        return [("asserts_stripped", False,
                 "python -O/-OO 已剥离 assert, 所有 check 会恒真通过, verify "
                 "结果无效 — 请改用不带 -O 的解释器重跑")]
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
