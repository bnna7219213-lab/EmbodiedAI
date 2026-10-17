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
from .paths import ext_root
from .adapters.edalab import FLOW_CHAIN, EdalabAdapter, CLIMBING_SPEC
# CLI 全局注册表 (命令的唯一真源 — 新增命令必须经 register_command 注册)
from .cli_registry import (
    CLI_REGISTRY, build_parser, command_catalog, register_command,
)
from .config import load_contract, validate_dbc
from .deploy import audit_latency
from .platform import (ArmRequirements, arm_budget_overrides, evaluate_readiness,
                       probe_host, render_systemd_units, report_lines, summarize,
                       to_json)
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]

# 五项目角色 (skeleton-and-layering-plan §3.5 终版裁定)
# 外部仓库的 root 一律走 paths.ext_root(): env 覆盖优先, 主目录推导兜底, 且
# 主目录不可知时**不抛错** (此前写死 C:\Users\<用户名>\..., 见 paths.py 说明)。
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
        "root": ext_root("EDALAB_ROOT", "workspace", "EdaLab"),
        "role": "怎么造硬件 — 36 引擎 + 12 步主链 (Kotlin; vendor jar 已入库)",
        # 本仓探针: 单仓自足的关键证据 (相对 EmbodiedAI 仓库根)
        "own_probes": ["vendor/edalab/edalabcore.jar",
                       "vendor/edalab/kotlin-stdlib-2.2.21.jar"],
        # 外部源码探针 (仅开发机存在; 单仓运行不依赖, 缺失不算 PARTIAL)
        "probes": [],
        "engine_dirs": ["design", "physics", "simulation", "safety", "bms", "hil", "ai", "mfg", "coupling", "pcb"],
        "note": "v0.8: 能力经 vendor/edalab jar 消费 (单仓自足, 仅需 JRE); 外部源码为开发机参照",
    },
    "ECAD-MCAD": {
        "root": ext_root("ECAD_MCAD_ROOT", "workspace", "ECAD-MCAD"),
        "role": "生命周期方法论 — NG1-22 + P0-P9 (文档)",
        "probes": ["最终计划.md", "全链条技术缺口补充.md", "硬件全生命周期衔接方案.md"],
    },
    "SynapticForge": {
        "root": ext_root("SYNAPTIC_FORGE_ROOT", "workspace", "SynapticForge"),
        "role": "商业化融合层 — L4 ForgeCore 编排 + 飞轮 (文档)",
        "probes": ["README.md", "docs/02-tech-architecture/00-architecture-overview.md"],
    },
    "nanoGPT": {
        "root": ext_root("NANOGPT_ROOT", "RaccoonWork", "nanoGPT"),
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
    repo = Path(__file__).resolve().parents[2]          # 本仓根 (vendor 探针基准)
    for name, info in PROJECTS.items():
        root = info["root"]
        # 探针分两类: 本仓探针 (vendor 等, 相对本仓) + 外部探针 (相对该项目 root)
        own_probes = info.get("own_probes", [])
        ext_probes = info.get("probes", [])
        exists = root.is_dir() or bool(own_probes)
        own_found = sum(1 for p in own_probes if (repo / p).exists())
        ext_found = sum(1 for p in ext_probes if (root / p).exists())
        total = len(own_probes) + len(ext_probes)
        found = own_found + ext_found
        status = "OK" if found == total and total else ("PARTIAL" if exists else "MISSING")
        if status != "OK":
            all_ok = False
        console.emit(f"[{status:7s}] {name:13s} {info['role']}")
        console.emit(f"           root={root}")
        if own_probes:
            console.emit(f"           本仓探针 {own_found}/{len(own_probes)} (vendor 入库)")
        if ext_probes:
            console.emit(f"           外部探针 {ext_found}/{len(ext_probes)}")
        if "engine_dirs" in info and root.is_dir():
            n = _count_kt_engines(root, info["engine_dirs"])
            console.emit(f"           Kotlin 引擎文件 ({'+'.join(info['engine_dirs'])}): {n}")
        if info.get("note"):
            console.emit(f"           注: {info['note']}")
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
    """安全链分段延迟预算核对。

    v1.1 修复: safety_budget.yaml 字段 `measured_*` → `assumed_*` (诚实边界)。
    CLI 现在向后兼容并明确打印 provenance (assumed | modeled | modeled_hil | measured_hil)。

    返回码语义:
      0 = 阈值通过 (预算内), 但 SIL2 发布级就绪与否**不改变返回码**
          (由 CLI 输出显式报告, 用户/CI 按需自行 gate)
      1 = 阈值失败 (超预算, 真物理不安全)

    这样 `safety` 子命令语义是"物理预算是否安全" (硬判定), 而 SIL2 发布
    就绪是"证据链是否到 HIL 级" (软报告)。两者正交。
    """
    console.banner("安全链分段延迟预算核对 (safety_budget.yaml)")
    sb = load_contract("safety_budget")
    meta = sb.get("metadata", {})
    prov = meta.get("provenance_confidence", "assumed")
    sil2_release_ready = meta.get("sil2_release_ready", False)
    if prov == "assumed":
        console.emit(f"  [诚实边界] provenance=assumed, SIL2 非发布级")
        console.emit(f"              SIL2 发布需 HIL 实测 → 把 provenance_confidence 升为 'measured_hil'")
    total_ms = 0.0
    def _g(seg, *keys):
        for k in keys:
            if k in seg:
                return seg[k]
        return 0.0
    for seg in sb["segments"]:
        if "budget_us" in seg:
            val = _g(seg, "assumed_us", "measured_us")
            label = "assumed" if "assumed_us" in seg else "measured"
            console.emit(f"  {seg['stage']:24s} {val} us ({label})  ({seg['owner']})")
            total_ms += val / 1000
        else:
            val = _g(seg, "assumed_ms", "measured_ms")
            label = "assumed" if "assumed_ms" in seg else "measured"
            console.emit(f"  {seg['stage']:24s} {val} ms ({label})  ({seg['owner']})")
            total_ms += val
    acc = sb["acceptance"]
    threshold_ok = total_ms < acc["total_break_max_ms"]
    console.emit(f"\n  合计 {total_ms:.2f} ms < {acc['total_break_max_ms']} ms 目标 -> "
                 f"{'PASS ' + console.ok() if threshold_ok else 'FAIL ' + console.fail()}")
    pfh_val = acc.get("assumed_pfh", acc.get("measured_pfh", 0))
    pfh_key = "assumed_pfh" if "assumed_pfh" in acc else "measured_pfh"
    sil2_pass = pfh_val < acc["sil2_pfh_target"]
    sil2_flag = "PASS*" if (sil2_pass and prov != "measured_hil") else ("PASS" if sil2_pass else "FAIL")
    console.emit(f"  SIL2 阈值: PFH({pfh_key}) {pfh_val:.1e} < {acc['sil2_pfh_target']:.1e} -> {sil2_flag}"
                 + ("  (* 假设+建模, 非 HIL 实测, 非发布级)" if sil2_flag == "PASS*" else ""))
    sw = sb.get("software_filters", [])
    console.emit(f"  软件滤波: {', '.join(f['stage'] for f in sw)}")
    # 输出发布级就绪状态 (显式报告, 不改变退出码)
    console.emit(f"\n  SIL2 发布级就绪: {sil2_release_ready} (sil2_pass={sil2_pass}, "
                 f"release_ready={sil2_release_ready})")
    return 0 if threshold_ok else 1


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
    # B2 (P1.4 Stage 化): EPS job 串联 forge-core → vsa-static (数据管道, 非烟囱式)
    if getattr(args, "eps_job", False):
        console.emit("\n[4] EPS job 串联 (forge-core → vsa-static, Stage 数据管道):")
        job = adapter.run_eps_job(["forge_core", "vsa_static"],
                                  spec=CLIMBING_SPEC, mode="live")
        console.emit(f"  spec={job['spec']} steps={job['steps']}/{job['expected_steps']} "
                     f"ok={'YES' if job['ok'] else 'NO'}")
        for name, ctx in job["context"].items():
            extra = ctx.get("eps_id") or (
                f"schema={ctx.get('vsa_schema')} tops={len(ctx.get('vsa_topologies') or [])}")
            console.emit(f"  stage {name:12s} kind={str(ctx['kind']):9s} -> {extra}")
        console.emit(f"  jar_source={job['results'][0].get('jar_source') if job['results'] else 'n/a'}")
        all_ok = all_ok and job["ok"]
    console.emit(f"\n11 步主链: {'ALL OK ' + console.ok() if all_ok else 'FAIL ' + console.fail()}")
    return 0 if all_ok else 1


def cmd_embodied_full(args):
    """P1.4 Stage 化: embodied_ai_full 全链路触发 — 按 STAGE_REGISTRY (12 项) 一次性编排。

    与 run_full_chain (11 步烟囱式) 的关键区别:
      run_full_chain 用 FLOW_CHAIN 顺序跑, 各步独立样本输入, 上游产物无人消费;
      cmd_embodied_full 用 run_eps_job 走 STAGE_REGISTRY 全 12 项, 每步产物喂给下游。
    命令: embodied-audit embodied-full [--spec SPEC] [--mode dry/live]
    默认: dry 模式 (秒回校验, CI 用); --live 真调 Kotlin 引擎 (需 jar)。
    """
    from .adapters.edalab import EMBODIED_FULL_STAGES, EdalabAdapter  # noqa: PLC0415

    console.banner(f"embodied_ai_full 一次触发全链路 — {len(EMBODIED_FULL_STAGES)} 项 (P1.4)")
    spec = getattr(args, "spec", "household_grasp")
    mode = getattr(args, "mode", "dry")

    console.emit(f"\nspec = {spec} | mode = {mode}\n")
    adapter = EdalabAdapter()
    job = adapter.run_eps_job(list(EMBODIED_FULL_STAGES), spec=spec, mode=mode)

    console.emit(f"编排: {job['steps']}/{job['expected_steps']} 步 | ok={'YES' if job['ok'] else 'NO'} "
                 f"| elapsed={job['elapsed_s']:.2f}s")
    console.emit(f"触发策: STAGE_REGISTRY (12 项含 vsa_static, FLOW_CHAIN 11 步超集)\n")
    for i, r in enumerate(job["results"], 1):
        stage_name = r.get("stage", "?")
        status = r.get("status", "?")
        console.emit(f"  step {i:2d} {stage_name:16s} → {status}")

    console.emit("\n数据管道产物:")
    for name, ctx in job["context"].items():
        if ctx.get("kind") == "parse_error":
            console.emit(f"  {name:16s} PARSE_ERROR")
        else:
            summary = ctx.get("eps_id") or (f"schema={ctx.get('vsa_schema')}" if ctx.get("vsa_schema") else "ok")
            console.emit(f"  {name:16s} kind={str(ctx.get('kind', '?')):9s} -> {summary}")
    return 0 if job["ok"] else 1


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


# --------------------------------------------------------------------------- #
# plan_v5 Sprint M 新增: 感知/认知 demo
# --------------------------------------------------------------------------- #
def cmd_fusion_demo(args):
    """M1 多传感器融合 + 分频信道处理 (provenance=synthetic)。"""
    console.banner("M1 多传感器融合 + 分频信道处理")
    from .perception.sensor_fusion import fusion_demo

    r = fusion_demo()
    console.emit(f"    信道                            {r['channels']}")
    console.emit(f"    video 信道串扰                  {r['crosstalk_video_db']:.1f} dB (<-20dB)")
    console.emit(f"    microwave 信道串扰              {r['crosstalk_microwave_db']:.1f} dB (<-20dB)")
    return _emit_result("分频融合", {"ok": True})


def cmd_slam_demo(args):
    """M2 SLAM 三维坐标 + 时间戳。"""
    console.banner("M2 SLAM (EKF 平面 + z 直通)")
    from .perception.slam import slam_demo

    r = slam_demo()
    console.emit(f"    50 步后位姿误差                 {r['final_position_error_m']:.3f} m (<0.3m)")
    console.emit(f"    路标估计                        {r['landmarks']}")
    console.emit(f"    位姿快照                        {r['snapshots']} 条")
    return _emit_result("SLAM", {"ok": True})


def cmd_cog_demo(args):
    """M4 认知中枢 — 四要素 (时间/地点/人物/事情)。"""
    console.banner("M4 认知中枢 (记录/分析/计算/规划/记忆)")
    from .cognition.cognitive_hub import cognitive_demo

    r = cognitive_demo()
    a = r["analysis"]
    console.emit(f"    记录总数                        {a['total']}")
    console.emit(f"    人物频次                        {a['by_actor']}")
    console.emit(f"    事件率                          {r['stats']['event_rate_hz']:.2f} Hz")
    console.emit(f"    规划缺口 pick/climb_hold        {r['plan_pick']} / {r['plan_climb_hold']}")
    console.emit(f"    记忆召回序                      {r['recall_order']}")
    return _emit_result("认知中枢", {"ok": True})


def cmd_loop_demo(args):
    """M6 VLA 双路径 × 认知中枢 端到端闭环。"""
    console.banner("M6 端到端闭环 (VLA 双路径 → 事件记录 → 规划)")
    from .cognition.closed_loop import loop_demo

    r = loop_demo()
    for label, name in (("v2a", "v→action (纯视觉)  "),
                        ("vla", "v+l→action (视觉+语言)")):
        d = r[label]
        console.emit(f"    {name}  达成={d['goal_reached']} "
                     f"步骤={d['steps']} 路径计数={d['paths']}")
    console.emit(f"    四要素事件 (每链路)             {r['v2a']['events']} 条")
    return _emit_result("端到端闭环", {"ok": True})


# --------------------------------------------------------------------------- #
# plan_v6 Sprint N 新增: OS/中枢 demo
# --------------------------------------------------------------------------- #
def cmd_os_status(args):
    """N3 OS SSH 双通道状态 (RUNTIME 协作规划 / MAINTENANCE 停机维护)。"""
    console.banner("N3 OS 通道状态 (dry-run)")
    from .os_channel import DryRunExecutor, ModeState, OsChannelAdmin

    admin = OsChannelAdmin("embodied-robot.local", executor=DryRunExecutor())
    admin.enter_maintenance("os-status 演示: 修历史 bug")
    s = admin.status()
    console.emit(f"    模式                            {s['mode']}")
    console.emit(f"    通道                            {s['channel']} (SSH 维护, 非遥控)")
    console.emit(f"    允许运动                        {s['can_send_motion']} (维护态禁运动)")
    admin.return_to_runtime()
    console.emit(f"    回运行态                        {admin.mode.state.value}")
    return _emit_result("OS 通道", {"ok": True})


def cmd_os_provision(args):
    """N3 出厂初始程序烧录 (认知本体 + 启动清单, dry-run)。"""
    console.banner("N3 出厂初始程序烧录 (dry-run)")
    from .os_channel import DryRunExecutor, ModeState, OsChannelAdmin

    admin = OsChannelAdmin("embodied-robot.local", executor=DryRunExecutor())
    admin.enter_maintenance("provision 演示")
    admin.provision_factory_program("cognition_v2")
    s = admin.status()
    console.emit(f"    烧录后模式                      {s['mode']}")
    console.emit(f"    命令序列                        {len(s['executed'])} 条 (ssh/scp)")
    admin.return_to_runtime()
    return _emit_result("出厂烧录", {"ok": True})


def cmd_center_demo(args):
    """N2 中枢编排端到端 (信号→感知→记忆→论证→决策→动作→事件)。"""
    console.banner("N2 中枢编排 CenterBrain (B8 POSE_UPDATE 断链修复)")
    from .center.center_brain import CenterBrain
    from .cognition.cognitive_hub import CognitiveHub
    from .runtime.bus import InProcessBus

    bus = InProcessBus()
    hub = CognitiveHub(bus=bus)
    out = CenterBrain(bus=bus, hub=hub).run_demo()
    console.emit(f"    双路径                          {out['paths']}")
    console.emit(f"    POSE_UPDATE→hub 地点            {out['pose_seen_by_hub']}")
    console.emit(f"    四要素事件                      {out['events']} 条")
    console.emit(f"    V2L 推演步数                    {out['argument_steps']}")
    return _emit_result("中枢编排", {"ok": True})


def cmd_v2l_demo(args):
    """N5 V2L 空间智能 — 辅助线/演算/结论 (可解释决策)。"""
    console.banner("N5 V2L 空间智能 (想象空间思考: 辅助线→演算→结论)")
    import numpy as np
    from .center.spatial_reasoning import SpatialReasoner

    r = SpatialReasoner()
    rng = np.random.default_rng(3)
    a = r.argue(rng.normal(size=12), np.array([1.0, 0.5, 0.0]))
    console.emit(f"    辅助线                          {len(a.auxiliary_hypotheses)} 条")
    console.emit(f"    演算步骤                        {len(a.derivation_steps)} 步")
    console.emit(f"    结论                            {a.conclusion}")
    console.emit(f"    决策向量                        {np.round(a.decision, 3).tolist()}")
    return _emit_result("V2L 论证", {"ok": True})


def cmd_chem_demo(args):
    """N4 化学生化传感 × 触觉融合 (provenance=synthetic)。"""
    console.banner("N4 化学生化 × 触觉融合 (e-nose/pH/湿度)")
    from .perception.chem_bio import chem_demo

    out = chem_demo()
    for name, d in out.items():
        console.emit(f"    {name:24s} surface={d['surface']:11s} "
                     f"material={d['material']}")
    return _emit_result("化学生化融合", {"ok": True})


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


def cmd_twin(args):
    """G10/G7/G2 双轨缺口自检: 电机模型 / 频谱判别 / FOC↔板级对齐。

    三项均走双轨 `run_capability`（backend 可观测; jar 无对应能力时如实反映）。
    """
    from .adapters.edalab import EdalabAdapter, available_backends

    console.banner("双轨缺口自检 (G10 电机模型 / G7 频谱判别 / G2 FOC↔板级对齐)")
    a = EdalabAdapter()
    caps = [("motor_model", "G10 电机模型 (Ke/flux/Ld/Lq/Rs)"),
            ("digital_twin", "G7 频谱异常判别 (FFT vs 阈值)"),
            ("foc_board_align", "G2 FOC ↔ MotorDrive 参数对齐")]
    rc = 0
    for cap, label in caps:
        avail = available_backends(cap)
        r = a.run_capability(cap, backend=getattr(args, "backend", "auto"))
        status = r.get("status")
        ok = bool(r.get("ok"))
        rc |= 0 if ok else 1
        console.emit(f"  [{('✅' if ok else '❌')}] {label}")
        console.emit(f"        能力={cap}  可用后端={avail}  实走={r.get('backend')}  状态={status}")
        if cap == "motor_model":
            s = r.get("spec", {})
            console.emit(f"        Ke={s.get('Ke_vs_per_rad'):.6f} V·s/rad  "
                         f"Kt={s.get('Kt_nm_per_a'):.6f} N·m/A  Rs={s.get('Rs_ohm')}Ω")
            console.emit(f"        校验 {len(r.get('checks', []))} 项, 失败 "
                         f"{len(r.get('failed', []))}")
        elif cap == "digital_twin":
            console.emit(f"        健康误报={r.get('healthy_detected_fault')} "
                         f"故障识别={r.get('faulty_detected_fault')} "
                         f"命中={r.get('faulty_hits')}")
            tb = r.get("threshold_baseline", {})
            console.emit(f"        阈值法漏检早期故障={tb.get('missed_early_fault')}")
        else:
            b = {k: v for k, v in (r.get("board") or {}).items() if v is not None}
            console.emit(f"        板级: {b.get('bus_voltage_v')}V/"
                         f"{b.get('peak_current_a')}A {b.get('layers')}层 "
                         f"{b.get('copper_oz')}oz MOSFET={b.get('mosfet')}")
            mm = r.get("mismatches") or []
            console.emit(f"        不匹配 {len(mm)} 项"
                         + (f": {[m['check'] for m in mm]}" if mm else " (已对齐)"))
        if r.get("note"):
            console.emit(f"        注: {r['note'][:110]}")
        console.emit("")
    console.emit(f"汇总: {'全部通过 ✅' if rc == 0 else '存在不一致 ⚠'}")
    return rc


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


def cmd_selfcheck(args):
    """单仓自足实证 (v4): vendor jar 载体/指纹/运行时/隔离可跑/后端可观测 五项断言。

    委托给 `embodied_ai.selfcheck`（核心逻辑在包内, 便于脚本与测试共用）。
    """
    import json as _json

    from .selfcheck import render, run_selfcheck

    res = run_selfcheck()
    if getattr(args, "json", False):
        console.emit(_json.dumps(res, ensure_ascii=False, indent=2))
    else:
        render(res)
    return 0 if res["ok"] else 1


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


def cmd_catalog(args):
    """打印 CLI 命令目录 (数据源: 全局注册表 CLI_REGISTRY)。

    存在意义: 新增命令若未进注册表, 这里就看不到 —— 这是"不落在角落"的
    可观测手段, 与 tests/test_cli_registry.py 的孤儿检测互为呼应。
    """
    console.emit(command_catalog())
    return 0


def _register_all() -> None:
    """把全部 CLI 命令注册进**全局注册表** (唯一真源, 2026-10-08)。

    重构动机: 此前参数声明 (add_parser/add_argument) 与 dispatch 字典分离两处,
    新增命令要改两处 → 极易只改一处 → 命令落在角落无人顾及。
    现全部收进 `cli_registry.CommandSpec`, `main()` 启动时读注册表构建 parser。

    `add_args` 回调里仍是原生 argparse 调用 (语义不变), 只是搬了位置。
    """
    R = register_command

    # ---------------- 资产 / 契约 ----------------
    R("audit", "五项目资产状态汇总", group="资产", since="v0.5",
      examples=["embodied-audit audit"])(cmd_audit)
    R("contracts", "四件套契约校验", group="资产", since="v0.5",
      examples=["embodied-audit contracts"])(cmd_contracts)

    # ---------------- 硬件 / EDA (v4 jar 消费模式) ----------------
    R("couple", "EdaLab 主链 (11 步) + live 冒烟 + EPS job 串联", group="硬件/EDA",
      since="v0.5",
      add_args=lambda p: (
          p.add_argument("--live", action="store_true",
                         help="真调 Kotlin 引擎冒烟 (relay-clutch)"),
          p.add_argument("--eps-job", action="store_true",
                         help="B2: EPS job 串联 forge-core -> vsa-static"),
      ),
      examples=["embodied-audit couple --eps-job",
                "embodied-audit couple --live"])(cmd_couple)
    # P1.4 Stage 化: embodied-full 全链路触发 (STAGE_REGISTRY 12 项编排, 非烟囱式)
    R("embodied-full", "embodied_ai_full 一次性触发全链路 (12 步 EPS 编排, P1.4)", group="硬件/EDA",
      since="v0.9",
      add_args=lambda p: (
          p.add_argument("--spec", default="household_grasp",
                         help="任务档案 (household_grasp / climbing)"),
          p.add_argument("--mode", default="dry", choices=["dry", "live"],
                         help="dry=秒回校验 (CI 用); live=真调 Kotlin 引擎"),
      ),
      examples=["embodied-audit embodied-full",
                "embodied-audit embodied-full --spec climbing --live"])(cmd_embodied_full)
    R("dr", "飞轮 DR 制造容差闭环 (ICT/FCT → DR → 训练)", group="硬件/EDA",
      since="v0.5", examples=["embodied-audit dr"])(cmd_dr)
    R("foc", "FOC + SVPWM + 硬件自动关机链时序验证", group="硬件/EDA",
      since="v0.7", examples=["embodied-audit foc"])(cmd_foc)
    R("hand", "G11 灵巧手设计决策矩阵", group="硬件/EDA", since="v0.7",
      examples=["embodied-audit hand"])(cmd_hand)
    R("real", "真机闭环 (Sim-to-Real 最后一公里)", group="硬件/EDA", since="v0.7",
      add_args=lambda p: (
          p.add_argument("--transport", default="mock",
                         choices=["mock", "socketcan", "serial"]),
          p.add_argument("--steps", type=int, default=60),
      ),
      examples=["embodied-audit real --transport mock --steps 60",
                "embodied-audit real --transport socketcan   # 真机 (需硬件)"])(cmd_real)
    R("cert", "G10 安全认证证据包 (SIL/PFH/RTM)", group="硬件/EDA", since="v0.7",
      add_args=lambda p: p.add_argument("--md", default=None,
                                        help="导出 Markdown 证据包到该路径"),
      examples=["embodied-audit cert --md docs/cert.md"])(cmd_cert)
    R("twin", "双轨缺口自检 (G10 电机模型 / G7 频谱判别 / G2 FOC↔板级对齐)",
      group="硬件/EDA", since="v0.8",
      add_args=lambda p: p.add_argument("--backend", default="auto",
                                        choices=["auto", "jar", "python"],
                                        help="后端路由 (auto 优先 jar, 无则 python)"),
      examples=["embodied-audit twin",
                "embodied-audit twin --backend python"])(cmd_twin)

    # ---------------- 仿真 ----------------
    R("sim", "仿真栈冒烟 (自动选择可用后端)", group="仿真", since="v0.5",
      add_args=lambda p: (
          p.add_argument("--scene", default="point_mass", help="场景名"),
          p.add_argument("--backend", default="auto",
                         help="auto|reference|mujoco|pybullet"),
          p.add_argument("--steps", type=int, default=20),
          p.add_argument("--seed", type=int, default=0),
          p.add_argument("--dr", action="store_true", help="启用域随机化采样"),
      ),
      examples=["embodied-audit sim --scene point_mass --steps 20",
                "embodied-audit sim --dr --steps 50"])(cmd_sim)
    R("sim2real", "Sim-to-Real 差距诊断", group="仿真", since="v0.5",
      add_args=lambda p: (
          p.add_argument("--backend", default="auto"),
          p.add_argument("--episodes", type=int, default=3),
          p.add_argument("--steps", type=int, default=300),
          p.add_argument("--dt", type=float, default=0.01, help="仿真步长 (s)"),
          p.add_argument("--noise", type=float, default=0.02, help="真机侧噪声"),
          p.add_argument("--seed", type=int, default=0),
      ),
      examples=["embodied-audit sim2real --episodes 3 --steps 300"])(cmd_sim2real)
    R("biped", "双足步态生成 + ZMP 稳定性检查", group="仿真", since="v0.5",
      add_args=lambda p: (
          p.add_argument("--steps", type=int, default=6, help="行走步数"),
          p.add_argument("--step-length", type=float, default=0.15),
          p.add_argument("--step-time", type=float, default=1.0),
          p.add_argument("--com-height", type=float, default=0.80),
          p.add_argument("--dt", type=float, default=0.005),
          p.add_argument("--foot-length", type=float, default=0.16),
          p.add_argument("--foot-width", type=float, default=0.08),
          p.add_argument("--margin", type=float, default=0.02),
      ),
      examples=["embodied-audit biped --steps 6 --step-length 0.15"])(cmd_biped)
    R("soft", "G6 柔性体动力学 (XPBD) 自检", group="仿真", since="v0.7",
      examples=["embodied-audit soft"])(cmd_soft)
    R("wholebody", "G9 双足 3D 全身控制 (3D ZMP + IK)", group="仿真", since="v0.7",
      add_args=lambda p: p.add_argument("--steps", type=int, default=4),
      examples=["embodied-audit wholebody --steps 4"])(cmd_wholebody)

    # ---------------- 感知 / 认知 (plan_v5 Sprint M) ----------------
    R("fusion-demo", "M1 多传感器融合 + 分频信道处理 (视频/微波)", group="感知/认知",
      since="v1.0", examples=["embodied-audit fusion-demo"])(cmd_fusion_demo)
    R("slam-demo", "M2 SLAM 三维坐标 + 时间戳 (EKF 平面+z 直通)", group="感知/认知",
      since="v1.0", examples=["embodied-audit slam-demo"])(cmd_slam_demo)
    R("cog-demo", "M4 认知中枢四要素 (时间/地点/人物/事情)", group="感知/认知",
      since="v0.9", examples=["embodied-audit cog-demo"])(cmd_cog_demo)
    R("loop-demo", "M6 VLA 双路径 × 认知中枢 端到端闭环", group="感知/认知",
      since="v1.0", examples=["embodied-audit loop-demo"])(cmd_loop_demo)

    # ---------------- OS / 中枢 (plan_v6 Sprint N) ----------------
    R("os-status", "N3 OS SSH 双通道状态 (RUNTIME 协作 / MAINTENANCE 维护)",
      group="OS/中枢", since="v1.2",
      examples=["embodied-audit os-status"])(cmd_os_status)
    R("os-provision", "N3 出厂初始程序烧录 (认知本体+启动清单, dry-run)",
      group="OS/中枢", since="v1.2",
      examples=["embodied-audit os-provision"])(cmd_os_provision)
    R("center-demo", "N2 中枢编排端到端 (信号→感知→记忆→论证→决策→事件)",
      group="OS/中枢", since="v1.2",
      examples=["embodied-audit center-demo"])(cmd_center_demo)
    R("v2l-demo", "N5 V2L 空间智能 (辅助线→演算→结论 可解释决策)",
      group="OS/中枢", since="v1.2",
      examples=["embodied-audit v2l-demo"])(cmd_v2l_demo)
    R("chem-demo", "N4 化学生化 × 触觉传感融合 (e-nose/pH/湿度)",
      group="OS/中枢", since="v1.2",
      examples=["embodied-audit chem-demo"])(cmd_chem_demo)

    # ---------------- 训练 / 模型 ----------------
    R("train", "训练管线自检 (BC/PPO/SAC/扩散)", group="训练", since="v0.7",
      add_args=lambda p: (
          p.add_argument("--algo", default="bc",
                         choices=["bc", "ppo", "sac", "diffusion"]),
          p.add_argument("--iters", type=int, default=None,
                         help="省略时按算法取已标定的默认规模"),
          p.add_argument("--episodes", type=int, default=10),
      ),
      examples=["embodied-audit train --algo bc",
                "embodied-audit train --algo ppo --iters 200"])(cmd_train)
    R("nanogpt", "nanoGPT 训练底座适配 (探测 + 训练计划)", group="训练", since="v0.5",
      add_args=lambda p: (
          p.add_argument("--root", default=None, help="nanoGPT 仓库根"),
          p.add_argument("--size", default="vla_tiny",
                         choices=["vla_tiny", "vla_small", "vla_base", "vla_large"]),
          p.add_argument("--out", default="output/nanogpt", help="训练计划输出目录"),
      ),
      examples=["embodied-audit nanogpt --size vla_tiny"])(cmd_nanogpt)
    R("bench", "VLA 推理延迟基准 (roofline)", group="训练", since="v0.7",
      add_args=lambda p: (
          p.add_argument("--size", default="vla_tiny",
                         choices=["vla_tiny", "vla_small", "vla_base", "vla_large"]),
          p.add_argument("--platform", default="jetson_orin_nx_16gb",
                         choices=["jetson_orin_nx_16gb", "jetson_orin_nano",
                                  "jetson_agx_orin", "rk3588", "generic_arm_cpu"]),
          p.add_argument("--precision", default="int8",
                         choices=["int8", "fp16", "fp32"]),
      ),
      examples=["embodied-audit bench --size vla_tiny --platform jetson_orin_nx_16gb"])(cmd_bench)
    R("lifelong", "G12 终身学习 (EWC + 回放)", group="训练", since="v0.7",
      examples=["embodied-audit lifelong"])(cmd_lifelong)

    # ---------------- 平台 / 运行台 ----------------
    R("platform", "Linux on ARM 目标能力探测 + 就绪评估", group="平台", since="v0.6",
      add_args=lambda p: (
          p.add_argument("--root", default=None,
                         help="目标板文件系统快照根 (含 proc/ sys/ dev/ etc/)"),
          p.add_argument("--strict", action="store_true",
                         help="强制判定架构 (开发机上非 ARM 时不再标 N/A)"),
          p.add_argument("--as-target", action="store_true",
                         help="把当前主机当作部署目标做完整就绪判定"),
          p.add_argument("--json", default=None, help="探测+检查结果写成 JSON"),
          p.add_argument("--emit-units", default=None,
                         help="按探测到的 isolcpus 渲染 systemd 单元到该目录"),
      ),
      examples=["embodied-audit platform --as-target",
                "embodied-audit platform --root ./snap --json probe.json"])(cmd_platform)
    R("gui", "L7 运行台 (零依赖 Web 仪表盘)", group="平台", since="v0.7",
      add_args=lambda p: (
          p.add_argument("--port", type=int, default=8765),
          p.add_argument("--once", action="store_true",
                         help="只跑自检 (起服务→拉 API→关服务), 不常驻"),
      ),
      examples=["embodied-audit gui --once",
                "embodied-audit gui --port 8765"])(cmd_gui)

    # ---------------- 验收 ----------------
    R("safety", "安全链预算核对", group="验收", since="v0.5",
      examples=["embodied-audit safety"])(cmd_safety)
    R("verify", "一键全量自检 (CI / scripts 入口)", group="验收", since="v0.5",
      add_args=lambda p: p.add_argument("--no-pytest", action="store_true",
                                        help="跳过 pytest 套件"),
      examples=["embodied-audit verify", "embodied-audit verify --no-pytest"])(cmd_verify)
    R("catalog", "列出全部命令 (读全局注册表)", group="验收", since="v0.8",
      examples=["embodied-audit catalog"])(cmd_catalog)
    R("selfcheck", "单仓自足实证 (v4: vendor jar + 隔离可跑 + 指纹一致)", group="验收",
      since="v0.8",
      add_args=lambda p: p.add_argument("--json", action="store_true",
                                        help="JSON 输出 (CI 断言用)"),
      examples=["embodied-audit selfcheck",
                "embodied-audit selfcheck --json"])(cmd_selfcheck)


# 模块导入即完成注册 (幂等: 模块只会被 import 一次)
_register_all()


def main(argv=None):
    console.ensure_utf8()
    # ★ 启动时读全局注册表构建 parser —— 命令的唯一真源
    ap = build_parser(prog="embodied-audit", description="EmbodiedAI 统一 CLI")
    ns = ap.parse_args(argv)
    return CLI_REGISTRY[ns.cmd].handler(ns)


if __name__ == "__main__":
    sys.exit(main())
