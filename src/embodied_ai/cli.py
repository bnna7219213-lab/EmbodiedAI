"""统一 CLI — embodied-audit 子命令总入口。

子命令分组:
  契约/资产:  audit / contracts / safety / couple / dr
  仿真迁移:  sim / sim2real / biped          (v0.5 新增)
  聚合自检:  verify                          (v0.5 新增, 一键全量, 供 CI 与 scripts/)

v0.5 变更:
  - 全部输出经 embodied_ai.console 安全层, 消除 Windows GBK UnicodeEncodeError
    (此前 audit/safety/couple/dr 四个子命令在该环境下必然崩溃, 退出码非 0)
  - 新增 sim / sim2real / biped / verify 四个子命令, 让仿真与运动控制栈可被自动化调用
  - verify 聚合全部检查项, 输出 PASS/FAIL 逐项清单并以退出码表达结果 (CI 友好)
"""
import argparse
import subprocess
import sys
from pathlib import Path

from . import console
from .adapters.edalab import FLOW_CHAIN, EdalabAdapter
from .config import load_contract, validate_dbc
from .deploy import audit_latency
from .platform import (ArmRequirements, arm_budget_overrides, evaluate_readiness,
                       probe_host, render_systemd_units, report_lines, summarize,
                       to_json)
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]

# 五项目角色 (skeleton-and-layering-plan §3.5 终版裁定)
PROJECTS = {
    "EmbodiedAI": {
        "root": REPO_ROOT,
        "role": "造什么 — 算法/数学/安全/数据 (Python 包, 本仓库)",
        "probes": ["src/embodied_ai/dynamics/climbing_dynamics_core.py",
                   "src/embodied_ai/planning/mcts_integration.py",
                   "src/embodied_ai/control/household_robot_system.py",
                   "contracts/hw_requirements.yaml"],
    },
    "EdaLab": {
        "root": Path(r"C:\Users\bnna7\workspace\EdaLab"),
        "role": "怎么造硬件 — 36 引擎 + 11 步主链 (Kotlin)",
        "probes": ["edalabcore/src/main/kotlin/edalab/core/coupling/MultiPhysicsCoupler.kt",
                   "edalabcore/src/main/kotlin/edalab/core/physics/RelayClutchDynamicsEngine.kt",
                   "edalabcore/src/main/kotlin/edalab/core/physics/ElectromechanicalCouplingEngine.kt"],
        "engine_dirs": ["design", "physics", "simulation", "safety", "bms", "hil", "ai", "mfg", "coupling", "pcb"],
    },
    "ECAD-MCAD": {
        "root": Path(r"C:\Users\bnna7\workspace\ECAD-MCAD"),
        "role": "生命周期方法论 — NG1-22 + P0-P9 (文档)",
        "probes": ["最终计划.md", "全链条技术缺口补充.md", "硬件全生命周期衔接方案.md"],
    },
    "SynapticForge": {
        "root": Path(r"C:\Users\bnna7\workspace\SynapticForge"),
        "role": "商业化融合层 — L4 ForgeCore 编排 + 飞轮 (文档)",
        "probes": ["README.md", "docs/02-tech-architecture/00-architecture-overview.md"],
    },
    "nanoGPT": {
        "root": Path(r"C:\Users\bnna7\RaccoonWork\nanoGPT"),
        "role": "模型训练底座 — tokenizer/Trainer/消融体系",
        "probes": ["llm.py", "train.py", "tokenizer.py"],
    },
}

CONTRACT_NAMES = ["hw_requirements", "safety_budget", "dr_params/manufacturing_variance"]


def _count_kt_engines(root: Path, dirs: list) -> int:
    base = root / "edalabcore/src/main/kotlin/edalab/core"
    n = 0
    for d in dirs:
        p = base / d
        if p.is_dir():
            n += len(list(p.glob("*.kt")))
    return n


# --------------------------------------------------------------------------- #
# 契约 / 资产
# --------------------------------------------------------------------------- #
def cmd_audit(args):
    console.banner("EmbodiedAI 五项目资产状态汇总 (三栈: 控制L/物理P/数据D/安全S横切)")
    all_ok = True
    for name, info in PROJECTS.items():
        root = info["root"]
        exists = root.is_dir()
        found = sum(1 for p in info["probes"] if (root / p).exists())
        status = "OK" if exists and found == len(info["probes"]) else ("PARTIAL" if exists else "MISSING")
        if status != "OK":
            all_ok = False
        console.emit(f"[{status:7s}] {name:13s} {info['role']}")
        console.emit(f"           root={root}")
        console.emit(f"           probes {found}/{len(info['probes'])}")
        if "engine_dirs" in info and exists:
            n = _count_kt_engines(root, info["engine_dirs"])
            console.emit(f"           Kotlin 引擎文件 ({'+'.join(info['engine_dirs'])}): {n}")
        console.emit("")

    console.emit("--- 契约四件套 ---")
    try:
        for name in CONTRACT_NAMES:
            load_contract(name)
        dbc = validate_dbc()
        dbc_ok = dbc["required_ok"] and dbc["unique_ids"] and dbc["crc_coverage"]
        mark = console.ok() if dbc_ok else console.fail()
        console.emit(f"[{mark}] contracts/ 四件套 (DBC {dbc['message_count']} 消息, "
                     f"CRC {'全覆盖' if dbc['crc_coverage'] else '缺失'})")
        if not dbc_ok:
            all_ok = False
    except Exception as e:  # noqa: BLE001
        console.emit(f"[{console.fail()}] 契约校验异常: {e}")
        all_ok = False

    console.emit("")
    console.emit("汇总:", "ALL OK " + console.ok() if all_ok else "存在缺口 " + console.warn() + " (见上)")
    return 0 if all_ok else 1


def cmd_contracts(args):
    console.banner("契约四件套校验")
    all_ok = True
    for name in CONTRACT_NAMES:
        try:
            d = load_contract(name)
            console.emit(f"[{console.ok()}] {name}.yaml — 顶层键: {', '.join(list(d.keys())[:6])}")
        except Exception as e:  # noqa: BLE001
            console.emit(f"[{console.fail()}] {name}: {e}")
            all_ok = False
    dbc = validate_dbc()
    dbc_ok = dbc["required_ok"] and dbc["unique_ids"] and dbc["crc_coverage"]
    console.emit(f"[{console.ok() if dbc_ok else console.fail()}] canfd/joint_control.dbc — "
                 f"{dbc['message_count']} 消息, ID唯一={dbc['unique_ids']}, CRC覆盖={dbc['crc_coverage']}")
    return 0 if (all_ok and dbc_ok) else 1


def cmd_safety(args):
    console.banner("安全链分段延迟预算核对 (safety_budget.yaml <-> EdaLab relay-clutch 实测)")
    sb = load_contract("safety_budget")
    total_ms = 0.0
    for seg in sb["segments"]:
        if "budget_us" in seg:
            console.emit(f"  {seg['stage']:24s} {seg['measured_us']} us  ({seg['owner']})")
            total_ms += seg["measured_us"] / 1000
        else:
            console.emit(f"  {seg['stage']:24s} {seg['measured_ms']} ms  ({seg['owner']})")
            total_ms += seg["measured_ms"]
    acc = sb["acceptance"]
    ok = total_ms < acc["total_break_max_ms"]
    console.emit(f"\n  合计 {total_ms:.2f} ms < {acc['total_break_max_ms']} ms 目标 -> "
                 f"{'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    sil2 = acc["measured_pfh"] < acc["sil2_pfh_target"]
    console.emit(f"  SIL2: PFH {acc['measured_pfh']:.1e} < {acc['sil2_pfh_target']:.1e} -> "
                 f"{'PASS' if sil2 else 'FAIL'}")
    sw = sb.get("software_filters", [])
    console.emit(f"  软件滤波: {', '.join(f['stage'] for f in sw)}")
    return 0 if (ok and sil2) else 1


def cmd_couple(args):
    """K2: EdaLab 11 步主链 (真实口径) + live 冒烟选项。"""
    console.banner("EmbodiedAI x EdaLab 11 步主链 (真实口径, EdaCli 权威)")
    adapter = EdalabAdapter(verbose=False)
    console.emit("\n[1] 11 步主链 (step / cli / engine):")
    all_ok = True
    for stage in FLOW_CHAIN:
        result = adapter.run_stage(stage["name"], {})
        ok = result["status"] == "READY"
        if not ok and result["status"] != "ENGINE_MISSING":
            all_ok = False
        icon = "OK" if result["status"] in ("READY", "ENGINE_MISSING") else "FAIL"
        console.emit(f"  [{icon:11s}] step {stage['step']:2d} {stage['name']:16s} cli={stage['cli']:14s} "
                     f"engine={stage['engine']}")
    coupler_path = adapter.root / "edalabcore/src/main/kotlin/edalab/core/coupling/MultiPhysicsCoupler.kt"
    console.emit(f"\n[2] MultiPhysicsCoupler (三域耦合引擎): "
                 f"{'存在 ' + console.ok() if coupler_path.exists() else '未找到 ' + console.fail()}")
    if getattr(args, "live", False):
        console.emit("\n[3] live 真调冒烟 (relay-clutch):")
        r = adapter.run_stage("relay_clutch", {}, mode="live")
        console.emit(f"  status={r['status']}")
        for line in r.get("risk_lines", [])[:3]:
            console.emit(f"  {line}")
        all_ok = all_ok and r["status"] == "LIVE_OK"
    console.emit(f"\n11 步主链: {'ALL OK ' + console.ok() if all_ok else 'FAIL ' + console.fail()}")
    return 0 if all_ok else 1


def cmd_dr(args):
    """K2: 飞轮 DR 制造容差闭环 (ICT/FCT/遥测 -> DR YAML)。"""
    console.banner("飞轮 DR 制造容差闭环 (创新 #1)")
    adapter = EdalabAdapter()
    ict = [{"board_serial": "SF001", "component_values": {"ENC1": 0.01, "CS_INA": 0.002}}]
    fct = [{"board_serial": "SF001", "comm_tests": {"can_bus_utilization": 0.8}}]
    tel = [{"motor_current_a": 2.5, "temperature_c": 35, "fault_events": []}]
    result = adapter.flywheel_e2e(ict_data=ict, fct_data=fct, telemetry_data=tel, output_dir=Path("output/dr"))
    cfg = result["config"]
    console.emit(f"  配置版本:     {cfg['config_version']}")
    console.emit(f"  DR 参数数:   {len(cfg['overrides'])}")
    console.emit(f"  YAML 输出:   {result['yaml_path']}")
    console.emit(f"  契约覆盖:    {len(result['variance_diff'].get('within_spec', []))} 项")
    ok = result["flywheel_ok"] and bool(result["yaml_path"])
    console.emit(f"\n飞轮闭环: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# v0.5 新增: 仿真 / 迁移 / 运动控制
# --------------------------------------------------------------------------- #
def cmd_sim(args):
    """仿真栈冒烟 — 无需 mujoco/pybullet 亦可跑通 (参考环境回退)。"""
    console.banner("仿真栈冒烟 (G7 缺口: SimEnv / DR / 后端自动选择)")
    from .simulation import DRConfig, DomainRandomizer, EnvConfig, available_backends, make_env

    backends = available_backends()
    console.emit(f"  可用后端: {', '.join(backends) if backends else '(无)'}")
    console.emit(f"  选定后端: {args.backend}")

    env = make_env(EnvConfig(scene_name=args.scene, state_dim=6, action_dim=3,
                             max_episode_steps=args.steps), backend=args.backend)
    console.emit(f"  实际后端: {type(env).__name__}")

    dr = DomainRandomizer(DRConfig(seed=args.seed)) if args.dr else None
    if dr is not None:
        params = dr.sample_params()
        if hasattr(env, "set_dr_params"):
            env.set_dr_params(params)
        console.emit(f"  DR 参数: {len(params)} 项 (示例: "
                     + ", ".join(f"{k}={v:.4g}" for k, v in list(params.items())[:3]) + ")")

    obs = env.reset(seed=args.seed)
    total_reward = 0.0
    steps = 0
    for _ in range(args.steps):
        action = env.sample_action()
        res = env.step(action)
        total_reward += res.reward
        steps += 1
        if res.done or res.truncated:
            break
    console.emit(f"  观测维度: {env.obs_dim}, 动作维度: {env.action_dim}")
    console.emit(f"  步数: {steps}, 累计奖励: {total_reward:.3f}")
    console.emit(f"  episode 记录: {len(env.export_episode()['observations'])} 条 (导出就绪)")
    env.close()
    console.emit(f"\n仿真栈: PASS {console.ok()}")
    return 0


def cmd_sim2real(args):
    """Sim-to-Real 差距诊断 — 以参考环境为 sim/real 两端做可复现演示。"""
    console.banner("Sim-to-Real 迁移管线 (DR -> 差距量化 -> 建议)")
    from .simulation import DRConfig, EnvConfig, SimToRealPipeline, make_env

    class _Policy:
        """PD 追踪策略: 满足 predict(obs) -> action 协议。"""

        def __init__(self, target, kp=2.0, kd=0.3):
            self.target = target
            self.kp, self.kd = kp, kd

        def predict(self, obs):
            state = obs["state"]
            d = state[: len(self.target)] - self.target
            return -self.kp * d - self.kd * state[len(self.target):]

    target = [0.4, 0.0, 0.0]
    # dt 默认 1ms (对齐 FOC), 但 PD 追踪的收敛需 ~3s; 演示用 10ms × steps 帧
    sim_cfg = EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3, dt=args.dt,
                        max_episode_steps=args.steps)
    real_cfg = EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3, dt=args.dt,
                         max_episode_steps=args.steps,
                         obs_noise_std=args.noise, action_noise_std=args.noise * 0.5)

    sim_env = make_env(sim_cfg, backend=args.backend)
    real_env = make_env(real_cfg, backend=args.backend, real=True)

    result = SimToRealPipeline(
        sim_env=sim_env, real_env=real_env, policy=_Policy(target),
        dr_config=DRConfig(seed=args.seed, enable_visual=False, enable_environment=False),
    ).run(sim_episodes=args.episodes, real_episodes=args.episodes,
          max_steps_per_episode=args.steps)

    console.emit(result.report())
    ok = abs(result.success_gap) <= 0.5
    console.emit(f"\n迁移管线: {'PASS ' + console.ok() if ok else '差距过大 ' + console.warn()}")
    return 0 if ok else 1


def cmd_biped(args):
    """双足步态生成 + ZMP 稳定性检查 (G9 缺口)。"""
    console.banner("双足运动控制 (LIPM + ZMP + 捕获点)")
    import numpy as np

    from .control.biped_locomotion import (GaitGenerator, LIPM, LIPMParams,
                                           ZMPStabilityChecker, build_walking_plans)

    lipm = LIPM(LIPMParams(z_c=args.com_height, dt=args.dt))
    plans = build_walking_plans(n_steps=args.steps, step_length=args.step_length, step_time=args.step_time)
    gait = GaitGenerator(lipm, plans)
    # 平面步态: 额状面由落脚点控制, ZMP 判据只看矢状面脚长区间
    checker = ZMPStabilityChecker(foot_length=args.foot_length, foot_width=args.foot_width,
                                  com_height=args.com_height, planar=True)

    dt = args.dt
    n_frames = int(sum(p.step_time for p in plans) / dt)
    unstable_frames = 0
    min_margin = float("inf")
    com_final = 0.0
    max_leg_excursion = 0.0  # CoM 相对支撑脚的最大偏移 (腿摆角约束)

    for _ in range(n_frames):
        com, lf, rf = gait.update(dt)
        com_final = com[0]
        zmp = gait.get_zmp_instant()
        plan = plans[gait.active_step_index]
        support = lf if plan.support_foot == "left" else rf
        stable, margin = checker.is_stable(zmp, support[:2], margin=args.margin)
        min_margin = min(min_margin, margin)
        max_leg_excursion = max(max_leg_excursion, abs(com[0] - support[0]))
        if not stable:
            unstable_frames += 1

    total = max(n_frames, 1)
    console.emit(f"  步数: {args.steps}, 步周期: {args.step_time}s, 步长: {args.step_length}m")
    console.emit(f"  特征频率 omega: {lipm.p.omega:.3f} rad/s")
    console.emit(f"  末端 CoM x: {com_final:.3f} m (期望 ~{args.steps * args.step_length:.3f} m)")
    console.emit(f"  ZMP 最小裕度: {min_margin:.4f} m (阈值 {args.margin} m)")
    console.emit(f"  失稳帧占比: {unstable_frames}/{total} = {unstable_frames / total:.1%}")
    console.emit(f"  CoM 相对支撑脚最大偏移: {max_leg_excursion:.3f} m "
                 f"(腿摆角 {np.degrees(np.arctan2(max_leg_excursion, args.com_height)):.1f} deg)")
    capture = lipm.calc_capture_point(com_final, 0.0)
    console.emit(f"  捕获点: {capture:.3f} m")

    # 验收: CoM 确实前进 + 全程稳定 + 数值有界
    ok = (com_final > 0.0 and unstable_frames == 0
          and np.isfinite(com_final) and min_margin > 0.0)
    console.emit(f"\n双足步态: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# v0.5 新增: 一键自检
# --------------------------------------------------------------------------- #
def _run_check(name: str, func) -> tuple:
    """执行单个检查项, 返回 (name, passed, detail)。异常视为失败而非崩溃。"""
    try:
        rc = func()
        return name, rc == 0, f"exit={rc}"
    except Exception as e:  # noqa: BLE001
        return name, False, f"{type(e).__name__}: {e}"


def cmd_platform(args):
    """v0.6: Linux on ARM 目标能力探测 + 就绪评估 (可离线对快照跑)。"""
    console.banner("Linux on ARM 目标能力探测")

    if args.root:
        target = probe_host(root=args.root)
        console.emit(f"  快照根: {args.root}")
    else:
        target = probe_host()
        console.emit(f"  探测对象: 当前主机 (arch={target.arch})")

    req = ArmRequirements.from_contract()
    # 只有显式 --root (目标板快照) 或 --as-target 才做就绪判定;
    # 否则只是"看看这台机器的信息", 硬件类检查标 N/A (开发机 / CI 上的正确行为)。
    as_target = bool(args.root) or args.as_target
    checks = evaluate_readiness(target, req, arch_strict=args.strict, as_target=as_target)
    for line in report_lines(target, checks):
        console.emit(line)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_json(target, checks), encoding="utf-8")
        console.emit(f"\n  JSON 落盘: {out}")

    if args.emit_units:
        paths = render_systemd_units(target, args.emit_units)
        console.emit(f"  systemd 单元: {len(paths)} 个 → {args.emit_units}")
        for p in paths:
            console.emit(f"    {p}")

    # ARM 目标上的预算缩放 → 直接喂回既有 50ms 闸门
    overrides = arm_budget_overrides(target)
    if overrides:
        res = audit_latency(budget_overrides=overrides)
        console.emit("")
        console.emit("  端到端预算 (按目标能力缩放后):")
        for line in res.report_lines():
            console.emit("   ", line)

    s = summarize(target, checks)
    console.emit("")
    if not target.is_arm and not args.strict:
        console.emit(f"  {console.warn()} 当前主机非 ARM: 架构类检查按 N/A 处理 "
                     f"(用 --root 载入目标板快照, 或 --strict 强制判定)")
    console.emit(f"\nLinux on ARM 就绪: {'PASS ' + console.ok() if s['ok'] else 'FAIL ' + console.fail()}")
    return 0 if s["ok"] else 1


# --------------------------------------------------------------------------- #
# v0.7 新增: 缺口补齐后的子命令
# --------------------------------------------------------------------------- #
def _emit_result(title: str, res: Dict[str, Any]) -> int:
    """统一输出: 关键数值 + PASS/FAIL。"""
    for k, v in res.items():
        if k in ("ok", "report", "matrix", "ranking", "advice", "gap_diagnosis",
                 "stages", "transport", "layers", "modules", "contracts", "events"):
            continue
        console.emit(f"    {k:34s} {v}")
    ok = bool(res.get("ok"))
    console.emit(f"\n{title}: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    return 0 if ok else 1


def cmd_soft(args):
    """G6 柔性体动力学 (XPBD)。"""
    console.banner("G6 柔性体动力学 (XPBD 布料/毛巾)")
    from .dynamics.soft_body import soft_body_demo

    return _emit_result("柔性体", soft_body_demo())


def cmd_wholebody(args):
    """G9 双足 3D 全身控制。"""
    console.banner("G9 双足 3D 全身控制 (3D ZMP + 6-DoF IK)")
    from .control.whole_body import whole_body_demo

    r = whole_body_demo(n_steps=args.steps)
    console.emit(f"    IK 往返位置误差              {r['ik_position_err']:.3e}")
    console.emit(f"    IK 往返旋转误差              {r['ik_rotation_err']:.3e}")
    console.emit(f"    CoM 前进                     {r['com_advance_m']} m")
    console.emit(f"    失稳帧                       {r['unstable_frames']}")
    console.emit(f"    双脚/单脚支撑面积            "
                 f"{r['support_area_both']:.4f} / {r['support_area_single']:.4f} m²")
    ok = bool(r["ok"])
    console.emit(f"\n3D 全身控制: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    return 0 if ok else 1


def cmd_foc(args):
    """优先级 (1): FOC + SVPWM + 自动关机链。"""
    console.banner("FOC 驱动 + SVPWM + 硬件自动关机链")
    from .hardware.foc_shutdown import ShutdownChain, foc_shutdown_demo

    r = foc_shutdown_demo()
    console.emit(f"    SVPWM 伏秒平衡误差           {r['svpwm_recon_err']:.3e} V")
    console.emit(f"    电流环阶跃稳态误差           {r['current_step_ss_error_pct']:.4f} %")
    console.emit(f"    电气关断                     {r['electrical_off_us']:.3f} us "
                 f"(预算 1.0 us)")
    console.emit(f"    总断开                       {r['total_break_ms']:.4f} ms "
                 f"(预算 20 ms)")
    console.emit(f"    停止角 @3rad/s               {r['stop_angle_deg']:.2f} deg")
    console.emit(f"    30° 限额下的最大关节速度      "
                 f"{r['max_safe_joint_speed_rad_s']:.3f} rad/s")
    console.emit(f"    HIL 测试用例                 {r['n_hil_cases']} 条")
    ok = bool(r["ok"])
    console.emit(f"\nFOC/关机链: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    return 0 if ok else 1


def cmd_hand(args):
    """G11 灵巧手选型。"""
    console.banner("G11 灵巧手设计决策")
    from .hardware.dexterous_hand import dexterous_hand_demo

    r = dexterous_hand_demo()
    console.emit(f"    {'方案':18s} {'自由度':>6s} {'执行器':>7s} {'覆盖':>6s} "
                 f"{'质量':>8s} {'kg':>6s}")
    for m in r["matrix"]:
        console.emit(f"    {m['design']:18s} {m['n_dof']:6d} {m['n_actuators']:7d} "
                     f"{m['coverage']:6.2f} {m['mean_quality']:8.4f} {m['mass_kg']:6.2f}")
    console.emit(f"\n  推荐: {r['recommended']}")
    console.emit(f"  排序: {' > '.join(r['ranking'])}")
    ok = bool(r["ok"])
    console.emit(f"\n灵巧手选型: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    return 0 if ok else 1


def cmd_real(args):
    """真机闭环 (Sim-to-Real 最后一公里)。"""
    console.banner("真机闭环 (Sim-to-Real)")
    from .simulation.real_robot_env import real_robot_demo

    r = real_robot_demo(steps=args.steps)
    console.emit(f"    传输层                       {args.transport}")
    console.emit(f"    真机回合评估                 {r['real_episodes_evaluated']}")
    console.emit(f"    平均环回延迟                 {r['avg_latency_ms']:.3f} ms "
                 f"({r['latency_samples']} 样本)")
    console.emit(f"    安全闸门拦截超限动作         {r['safety_gate_blocked']}")
    console.emit(f"    CRC8 往返                    {r['crc_roundtrip_ok']}")
    console.emit(f"    差距诊断                     {r['gap_diagnosis']}")
    ok = bool(r["ok"])
    console.emit(f"\n真机闭环: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    return 0 if ok else 1


def cmd_bench(args):
    """优先级 (2): VLA 推理延迟基准。"""
    console.banner("VLA 推理延迟基准 (roofline)")
    from .deploy.jetson_bench import PLATFORMS, bench_vla

    r = bench_vla(args.size, args.platform, args.precision)
    for line in r.report().splitlines():
        console.emit("   ", line)
    console.emit(f"\n  受限类型: {r.bound}")
    for a in r.advice:
        console.emit(f"   - {a}")
    ok = bool(r.ok)
    console.emit(f"\n推理基准: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    return 0 if ok else 1


def cmd_cert(args):
    """G10 安全认证工程化。"""
    console.banner("G10 安全认证证据包")
    from .safety.certification import build_default_package, certification_demo

    r = certification_demo()
    console.emit(f"    PFH                          {r['pfh_per_hour']:.3e} /h "
                 f"(目标 {1e-6:.0e})")
    console.emit(f"    SFF / 架构约束               {r['sff_percent']:.2f}% / "
                 f"{r['architecture_ok']}")
    console.emit(f"    可达 SIL                     {r['achieved_sil']}")
    console.emit(f"    RTM 覆盖 / 危害覆盖          {r['rtm_coverage']:.2f} / "
                 f"{r['hazard_coverage']:.2f}")
    console.emit(f"    结论                         {r['verdict']}")
    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_default_package().to_markdown(), encoding="utf-8")
        console.emit(f"\n  证据包 Markdown 落盘: {out}")
    ok = bool(r["ok"])
    console.emit(f"\n认证包: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    return 0 if ok else 1


def cmd_lifelong(args):
    """G12 终身学习。"""
    console.banner("G12 终身学习 (EWC + 回放)")
    from .learning.lifelong import lifelong_demo

    r = lifelong_demo()
    console.emit(f"    基线遗忘 (无保护)            {r['baseline_forgetting']:.4f}")
    console.emit(f"    EWC+回放遗忘                 {r['ewc_forgetting']:.4f}")
    console.emit(f"    任务0 最终误差 基线/EWC      "
                 f"{r['baseline_task0_final']:.4f} / {r['ewc_task0_final']:.4f}")
    console.emit(f"    可塑性: λ 过大时新任务误差   {r['stiff_task1_error']:.4f} "
                 f"(正常 {r['ewc_task1_error']:.4f})")
    ok = bool(r["ok"])
    console.emit(f"\n终身学习: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    return 0 if ok else 1


def cmd_train(args):
    """训练管线自检: BC / PPO / SAC / 扩散策略。"""
    # 各算法"能稳定超过基线"所需的步数量级差 2 个数量级, 统一默认值必然让其中
    # 几个假 FAIL (PPO 40 次迭代学不完, 扩散 300 次去噪还没收敛)。
    _TRAIN_DEFAULT_ITERS = {"bc": 600, "ppo": 120, "sac": 12000, "diffusion": 1200}
    iters = int(args.iters) if args.iters else _TRAIN_DEFAULT_ITERS[args.algo]
    console.banner(f"训练管线自检 ({args.algo}, iters={iters})")
    if args.algo == "bc":
        from .models.bc_trainer import bc_demo

        r = bc_demo(n_episodes=args.episodes, iters=iters)
        console.emit(f"    样本数                       {r['n_samples']}")
        console.emit(f"    成功率 专家/克隆/随机        "
                     f"{r['expert_success']:.2f} / {r['bc_success']:.2f} / "
                     f"{r['random_success']:.2f}")
        console.emit(f"    回报   专家/克隆/随机        "
                     f"{r['expert_return']:.2f} / {r['bc_return']:.2f} / "
                     f"{r['random_return']:.2f}")
        console.emit(f"    验证 MSE 改善                {r['mse_improvement_x']:.1f}x")
    elif args.algo == "diffusion":
        from .models.diffusion_policy import fit_demo

        r = fit_demo(iters=iters)
        console.emit(f"    去噪损失                     "
                     f"{r['loss_before']:.4f} -> {r['loss_after']:.4f}")
        console.emit(f"    采样轨迹 vs 专家 距离        {r['dist_after']:.4f} "
                     f"(随机基线 {r['dist_before']:.4f})")
        console.emit(f"    DDIM 10 步距离               {r['ddim_dist']:.4f}")
    else:
        from .models.rl_algorithms import rl_demo

        r = rl_demo(args.algo, iters=iters)
        console.emit(f"    回报 初/末                   "
                     f"{r['initial_return']:.2f} -> {r['final_return']:.2f}")
        console.emit(f"    随机基线                     {r['random_return']:.2f}")
        console.emit(f"    诊断                         {r['diagnostics']}")
    ok = bool(r["ok"])
    console.emit(f"\n{args.algo} 训练: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
    return 0 if ok else 1


def cmd_gui(args):
    """L7 仪表盘: --once 只自检, 否则常驻服务。"""
    console.banner("L7 运行台 (零依赖 Web 仪表盘)")
    from .gui import gui_demo
    from .gui.server import make_server

    if args.once:
        r = gui_demo(port=args.port)
        console.emit(f"    静态页                       {r['static_html_bytes']} bytes")
        console.emit(f"    模块可达                     {r['modules_ok']}/{r['modules_total']}")
        console.emit(f"    安全链 / 认证                {r['chain_total_ms']} ms / "
                     f"{r['cert_verdict']}")
        console.emit(f"    3D 步态前进 / 失稳           {r['biped_advance_m']} m / "
                     f"{r['biped_unstable']}")
        console.emit(f"    灵巧手推荐                   {r['hand_recommended']}")
        console.emit(f"    STO 指令 → mode              {r['mode_after_sto']}")
        ok = bool(r["ok"])
        console.emit(f"\n仪表盘: {'PASS ' + console.ok() if ok else 'FAIL ' + console.fail()}")
        return 0 if ok else 1

    srv, st = make_server(port=args.port)
    host, port = srv.server_address
    console.emit(f"  监听 http://{host}:{port}/  (Ctrl-C 退出)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        console.emit("  已停止")
    finally:
        srv.server_close()
    return 0


def cmd_verify(args):
    """聚合自检 — 契约 / 安全 / 仿真 / 迁移 / 步态 / 测试套件, 退出码表达总结果。"""
    from . import verification

    console.banner(f"EmbodiedAI 全量自检 (v{verification.VERSION})")
    results = verification.run_all(skip_pytest=getattr(args, "no_pytest", False))
    for name, passed, detail in results:
        mark = console.ok() if passed else console.fail()
        console.emit(f"  [{mark}] {name:22s} {detail}")
    n_pass = sum(1 for _, p, _ in results if p)
    console.emit("")
    console.emit(f"汇总: {n_pass}/{len(results)} PASS")
    all_ok = n_pass == len(results)
    console.emit("RESULT:", "VERIFY_OK " + console.ok() if all_ok else "VERIFY_FAIL " + console.fail())
    return 0 if all_ok else 1


def cmd_nanogpt(args):
    """nanoGPT 训练底座适配器 — 探测 / 训练计划导出 (K3 遗留补全)。"""
    console.banner("nanoGPT 训练底座适配")
    from .adapters.nanogpt import NanoGPTAdapter

    adapter = NanoGPTAdapter(root=args.root)
    probe = adapter.probe()
    console.emit(f"  root: {probe['root']}  存在: {probe['exists']}")
    console.emit(f"  探测文件: {probe['found']}/{probe['total']} "
                 f"缺失: {', '.join(probe['missing']) or '(无)'}")
    plan = adapter.export_training_plan(model_size=args.size, out_dir=args.out)
    console.emit(f"  训练计划: {plan['path']}")
    console.emit(f"  预计参数量: {plan['plan']['estimate_params']:,}")
    console.emit(f"  建议命令: {plan['plan']['suggested_cmd']}")
    # nanoGPT 仓库是**外部**资产 (默认路径为本机绝对路径), CI/其他机器上缺失属预期,
    # 不作为失败 — 交付物是"训练计划", 探测结果只作提示。
    if not (probe["exists"] and probe["found"] > 0):
        console.emit(f"  {console.warn()} nanoGPT 仓库不在默认路径 (外部资产, 非阻塞): {probe['root']}")
    console.emit(f"\nnanoGPT 适配: PASS {console.ok()}")
    return 0


def main(argv=None):
    console.ensure_utf8()
    ap = argparse.ArgumentParser(prog="embodied-audit", description="EmbodiedAI 统一 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("audit", help="五项目资产状态汇总")
    sub.add_parser("contracts", help="四件套契约校验")
    sub.add_parser("safety", help="安全链预算核对")
    p_couple = sub.add_parser("couple", help="K2: EdaLab 11 步主链 (真实口径) + live 冒烟选项")
    p_couple.add_argument("--live", action="store_true", help="真调 Kotlin 引擎冒烟 (relay-clutch)")
    sub.add_parser("dr", help="K2: 飞轮 DR 制造容差闭环")

    p_sim = sub.add_parser("sim", help="仿真栈冒烟 (自动选择可用后端)")
    p_sim.add_argument("--scene", default="point_mass", help="场景名 (默认 point_mass 参考环境)")
    p_sim.add_argument("--backend", default="auto", help="auto|reference|mujoco|pybullet")
    p_sim.add_argument("--steps", type=int, default=20)
    p_sim.add_argument("--seed", type=int, default=0)
    p_sim.add_argument("--dr", action="store_true", help="启用域随机化采样")

    p_s2r = sub.add_parser("sim2real", help="Sim-to-Real 差距诊断")
    p_s2r.add_argument("--backend", default="auto")
    p_s2r.add_argument("--episodes", type=int, default=3)
    p_s2r.add_argument("--steps", type=int, default=300)
    p_s2r.add_argument("--dt", type=float, default=0.01, help="仿真步长 (s)")
    p_s2r.add_argument("--noise", type=float, default=0.02, help="真机侧观测/动作噪声")
    p_s2r.add_argument("--seed", type=int, default=0)

    p_biped = sub.add_parser("biped", help="双足步态生成 + ZMP 稳定性检查")
    p_biped.add_argument("--steps", type=int, default=6, help="行走步数")
    p_biped.add_argument("--step-length", type=float, default=0.15)
    p_biped.add_argument("--step-time", type=float, default=1.0)
    p_biped.add_argument("--com-height", type=float, default=0.80)
    p_biped.add_argument("--dt", type=float, default=0.005)
    p_biped.add_argument("--foot-length", type=float, default=0.16)
    p_biped.add_argument("--foot-width", type=float, default=0.08)
    p_biped.add_argument("--margin", type=float, default=0.02)

    p_ng = sub.add_parser("nanogpt", help="nanoGPT 训练底座适配 (探测 + 训练计划)")
    p_ng.add_argument("--root", default=None, help="nanoGPT 仓库根 (默认内置路径)")
    p_ng.add_argument("--size", default="vla_tiny", choices=["vla_tiny", "vla_small", "vla_base", "vla_large"])
    p_ng.add_argument("--out", default="output/nanogpt", help="训练计划输出目录")

    p_plat = sub.add_parser("platform", help="Linux on ARM 目标能力探测 + 就绪评估 (v0.6)")
    p_plat.add_argument("--root", default=None,
                        help="目标板文件系统快照根 (含 proc/ sys/ dev/ etc/); 缺省探测当前主机")
    p_plat.add_argument("--strict", action="store_true",
                        help="强制判定架构 (开发机上非 ARM 时不再标 N/A)")
    p_plat.add_argument("--as-target", action="store_true",
                        help="把当前主机当作部署目标做完整就绪判定 (缺省只探测信息)")
    p_plat.add_argument("--json", default=None, help="把探测+检查结果写成 JSON")
    p_plat.add_argument("--emit-units", default=None,
                        help="按探测到的 isolcpus 渲染 systemd 单元到该目录")

    p_verify = sub.add_parser("verify", help="一键全量自检 (CI / scripts 入口)")
    p_verify.add_argument("--no-pytest", action="store_true", help="跳过 pytest 套件")

    # ---- v0.7 ---- #
    p_soft = sub.add_parser("soft", help="G6 柔性体动力学 (XPBD) 自检")
    p_wb = sub.add_parser("wholebody", help="G9 双足 3D 全身控制 (3D ZMP + IK)")
    p_wb.add_argument("--steps", type=int, default=4)

    p_foc = sub.add_parser("foc", help="FOC + SVPWM + 硬件自动关机链时序验证")
    p_hand = sub.add_parser("hand", help="G11 灵巧手设计决策矩阵")

    p_real = sub.add_parser("real", help="真机闭环 (Sim-to-Real 最后一公里)")
    p_real.add_argument("--transport", default="mock",
                        choices=["mock", "socketcan", "serial"])
    p_real.add_argument("--steps", type=int, default=60)

    p_bench = sub.add_parser("bench", help="VLA 推理延迟基准 (roofline)")
    p_bench.add_argument("--size", default="vla_tiny",
                         choices=["vla_tiny", "vla_small", "vla_base", "vla_large"])
    p_bench.add_argument("--platform", default="jetson_orin_nx_16gb",
                         choices=["jetson_orin_nx_16gb", "jetson_orin_nano",
                                  "jetson_agx_orin", "rk3588", "generic_arm_cpu"])
    p_bench.add_argument("--precision", default="int8", choices=["int8", "fp16", "fp32"])

    p_cert = sub.add_parser("cert", help="G10 安全认证证据包 (SIL/PFH/RTM)")
    p_cert.add_argument("--md", default=None, help="导出 Markdown 证据包到该路径")

    p_ll = sub.add_parser("lifelong", help="G12 终身学习 (EWC + 回放)")

    p_train = sub.add_parser("train", help="训练管线自检 (BC/PPO/SAC/扩散)")
    p_train.add_argument("--algo", default="bc",
                         choices=["bc", "ppo", "sac", "diffusion"])
    p_train.add_argument("--iters", type=int, default=None,
                         help="省略时按算法取已标定的默认规模")
    p_train.add_argument("--episodes", type=int, default=10)

    p_gui = sub.add_parser("gui", help="L7 运行台 (零依赖 Web 仪表盘)")
    p_gui.add_argument("--port", type=int, default=8765)
    p_gui.add_argument("--once", action="store_true",
                       help="只跑自检 (起服务→拉 API→关服务), 不常驻")

    ns = ap.parse_args(argv)
    cmds = {"audit": cmd_audit, "contracts": cmd_contracts, "safety": cmd_safety,
            "couple": cmd_couple, "dr": cmd_dr, "sim": cmd_sim, "sim2real": cmd_sim2real,
            "biped": cmd_biped, "nanogpt": cmd_nanogpt, "platform": cmd_platform,
            "verify": cmd_verify,
            # v0.7
            "soft": cmd_soft, "wholebody": cmd_wholebody, "foc": cmd_foc,
            "hand": cmd_hand, "real": cmd_real, "bench": cmd_bench,
            "cert": cmd_cert, "lifelong": cmd_lifelong, "train": cmd_train,
            "gui": cmd_gui}
    return cmds[ns.cmd](ns)


if __name__ == "__main__":
    sys.exit(main())
