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
    ("vsa_energy_consistency", check_vsa_energy_consistency),  # V-1: 曲线能量零点一致性守护
    ("chance_constrained_grip", check_chance_constrained_grip),  # R7: 机会约束握力可用 (z 精确 + Infeasible)
    ("canfd_db_cross_check", check_canfd_db_cross_check),  # R4: CAN-FD↔DBC 6 关节对照守护 (G3)
    ("vendor_jar_provenance", check_vendor_jar_provenance),  # v4 漂移治理 L1: jar 指纹溯源
    ("vla_block_size", check_vla_block_size),  # C1 (P0): 视觉截断 + Jetson 预算双守护
    ("afe_ind_frontend", check_afe_ind_frontend),  # Phase1: AFE-IND 触觉硬件前端移植守护
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
