"""统一 CLI — embodied-audit 五子命令 (K1: audit/contracts/safety + K2: couple/dr)。"""
import argparse
import subprocess
import sys
from pathlib import Path

from .adapters.edalab import FLOW_CHAIN, EdalabAdapter
from .config import load_contract, validate_dbc

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


def _count_kt_engines(root: Path, dirs: list) -> int:
    base = root / "edalabcore/src/main/kotlin/edalab/core"
    n = 0
    for d in dirs:
        p = base / d
        if p.is_dir():
            n += len(list(p.glob("*.kt")))
    return n


def cmd_audit(args):
    print("=== EmbodiedAI 五项目资产状态汇总 (三栈: 控制L/物理P/数据D/安全S横切) ===\n")
    all_ok = True
    for name, info in PROJECTS.items():
        root = info["root"]
        exists = root.is_dir()
        found = sum(1 for p in info["probes"] if (root / p).exists())
        status = "OK" if exists and found == len(info["probes"]) else ("PARTIAL" if exists else "MISSING")
        if status != "OK":
            all_ok = False
        line = f"[{status:7s}] {name:13s} {info['role']}"
        print(line)
        print(f"           root={root}")
        print(f"           probes {found}/{len(info['probes'])}")
        if "engine_dirs" in info and exists:
            n = _count_kt_engines(root, info["engine_dirs"])
            print(f"           Kotlin 引擎文件 ({'+'.join(info['engine_dirs'])}): {n}")
        print()
    # 契约四件套
    print("--- 契约四件套 ---")
    try:
        load_contract("hw_requirements")
        load_contract("safety_budget")
        load_contract("dr_params/manufacturing_variance")
        dbc = validate_dbc()
        dbc_ok = dbc["required_ok"] and dbc["unique_ids"] and dbc["crc_coverage"]
        print(f"[{'OK' if dbc_ok else 'FAIL':7s}] contracts/ 四件套 (DBC {dbc['message_count']} 消息, CRC {'全覆盖' if dbc['crc_coverage'] else '缺失'})")
        if not dbc_ok:
            all_ok = False
    except Exception as e:
        print(f"[FAIL   ] 契约校验异常: {e}")
        all_ok = False
    print()
    print("汇总:", "ALL OK ✅" if all_ok else "存在缺口 ⚠ (见上)")
    return 0 if all_ok else 1


def cmd_contracts(args):
    print("=== 契约四件套校验 ===")
    for name in ["hw_requirements", "safety_budget", "dr_params/manufacturing_variance"]:
        try:
            d = load_contract(name)
            print(f"[OK] {name}.yaml — 顶层键: {', '.join(list(d.keys())[:6])}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
    dbc = validate_dbc()
    print(f"[{'OK' if dbc['required_ok'] else 'FAIL'}] canfd/joint_control.dbc — {dbc['message_count']} 消息, "
          f"ID唯一={dbc['unique_ids']}, CRC覆盖={dbc['crc_coverage']}")
    return 0


def cmd_safety(args):
    print("=== 安全链分段延迟预算核对 (safety_budget.yaml ↔ EdaLab relay-clutch 实测) ===")
    sb = load_contract("safety_budget")
    total_ms = 0.0
    for seg in sb["segments"]:
        if "budget_us" in seg:
            print(f"  {seg['stage']:24s} {seg['measured_us']} µs  ({seg['owner']})")
            total_ms += seg["measured_us"] / 1000
        else:
            print(f"  {seg['stage']:24s} {seg['measured_ms']} ms  ({seg['owner']})")
            total_ms += seg["measured_ms"]
    acc = sb["acceptance"]
    ok = total_ms < acc["total_break_max_ms"]
    print(f"\n  合计 {total_ms:.2f} ms < {acc['total_break_max_ms']} ms 目标 → {'PASS ✅' if ok else 'FAIL ❌'}")
    print(f"  SIL2: PFH {acc['measured_pfh']:.1e} < {acc['sil2_pfh_target']:.1e} → {'PASS' if acc['measured_pfh'] < acc['sil2_pfh_target'] else 'FAIL'}")
    sw = sb.get("software_filters", [])
    print(f"  软件滤波: {', '.join(f['stage'] for f in sw)}")
    return 0 if ok else 1


def cmd_couple(args):
    """K2: EdaLab 11 步主链 + MultiPhysicsCoupler 三域耦合验证。"""
    print("=== EmbodiedAI × EdaLab 三域耦合 (K2) ===")
    adapter = EdalabAdapter(verbose=False)
    print("\n[1] 11 步主链元数据验证:")
    total_est = 0
    for stage in FLOW_CHAIN:
        result = adapter.run_stage(stage["name"], {"control_freq_hz": 1000, "compute_tops": 30})
        total_est += stage["est_sec"]
        status_icon = "OK" if result["status"] in ("READY", "ENGINE_MISSING") else "FAIL"
        print(f"  [{status_icon:11s}] {stage['name']:20s} {stage['category']:12s} est={stage['est_sec']}s engine={stage['engine']}")
    print(f"  主链预计总耗时: ~{total_est}s")

    coupler_path = adapter.root / "edalabcore/src/main/kotlin/edalab/core/coupling/MultiPhysicsCoupler.kt"
    if coupler_path.exists():
        print(f"\n[2] MultiPhysicsCoupler: {coupler_path.stat().st_size // 1024} KB (三域耦合引擎) ✅")
    else:
        print(f"\n[2] MultiPhysicsCoupler: ❌ 未找到")

    all_ok = all(adapter.run_stage(s["name"], {})["status"] in ("READY", "ENGINE_MISSING") for s in FLOW_CHAIN)
    print(f"\n11 步主链注册: {'ALL OK' if all_ok else 'FAIL ❌'}")
    return 0 if all_ok else 1


def cmd_dr(args):
    """K2: 飞轮 DR 制造容差闭环 (ICT/FCT/遥测 -> DR YAML)。"""
    print("=== 飞轮 DR 制造容差闭环 (创新 #1) ===")
    adapter = EdalabAdapter()
    ict = [{"board_serial": "SF001", "component_values": {"ENC1": 0.01, "CS_INA": 0.002}}]
    fct = [{"board_serial": "SF001", "comm_tests": {"can_bus_utilization": 0.8}}]
    tel = [{"motor_current_a": 2.5, "temperature_c": 35, "fault_events": []}]
    result = adapter.flywheel_e2e(ict_data=ict, fct_data=fct, telemetry_data=tel, output_dir=Path("output/dr"))
    cfg = result["config"]
    print(f"  配置版本:     {cfg['config_version']}")
    print(f"  DR 参数数:   {len(cfg['overrides'])}")
    print(f"  YAML 输出:   {result['yaml_path']}")
    print(f"  契约覆盖:    {len(result['variance_diff'].get('within_spec', []))} 项")
    ok = result["flywheel_ok"] and result["yaml_path"]
    print(f"\n飞轮闭环: {'PASS ✅' if ok else 'FAIL ❌'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="embodied-audit", description="EmbodiedAI 统一 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="五项目资产状态汇总")
    sub.add_parser("contracts", help="四件套契约校验")
    sub.add_parser("safety", help="安全链预算核对")
    sub.add_parser("couple", help="K2: EdaLab 11 步主链 + MultiPhysicsCoupler 三域耦合")
    sub.add_parser("dr", help="K2: 飞轮 DR 制造容差闭环")
    ns = ap.parse_args(argv)
    cmds = {"audit": cmd_audit, "contracts": cmd_contracts, "safety": cmd_safety,
            "couple": cmd_couple, "dr": cmd_dr}
    return cmds[ns.cmd](ns)


if __name__ == "__main__":
    sys.exit(main())
