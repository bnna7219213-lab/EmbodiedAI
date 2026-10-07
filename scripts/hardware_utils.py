"""scripts/hardware_utils.py — 制造/调试辅助工具集

提供 HIL 协议和制造指南中引用的命令行工具:
  - motor_identify: 电机参数辨识 (Rs/Ld/Lq/ψ)
  - hil_import: 导入实测 CSV/JSON
  - hil_gap_analyze: 对比仿真预言 vs 实测
  - hil_failure_log: 记录调试失败
  - weld_pre_check: 焊接前电气检查清单
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

REPO = Path(__file__).resolve().parents[1]
CONTRACTS = REPO / "contracts"
HIL_DIR = REPO / "experiments" / "hil_measurements"
FAIL_DIR = REPO / "experiments" / "hil_failures"


# ------------------------------------------------------------------------- #
# D4 数据集辅助: 按 scenario 统计力/电流
# ------------------------------------------------------------------------- #
def d4_analyze() -> Dict[str, Any]:
    """分析 D4 训练数据集的 scenario 分布和力统计。"""
    d4_file = REPO / "experiments" / "d4_dataset" / "training_replay_v1.jsonl"
    if not d4_file.exists():
        return {"error": f"{d4_file} 不存在, 请先运行 python scripts/gen_d4_dataset.py"}

    from collections import defaultdict
    scenario_currents: Dict[str, List[float]] = defaultdict(list)
    scenario_episodes: Dict[str, set] = defaultdict(set)
    n_total = 0

    with open(d4_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0 and "_schema" in obj:
                continue
            sc = obj.get("scenario", "unknown")
            scenario_currents[sc].append(obj.get("motor_current_a", 0.0))
            if "episode_id" in obj:
                scenario_episodes[sc].add(obj["episode_id"])
            n_total += 1

    stats = {}
    for sc in sorted(scenario_currents.keys()):
        currents = scenario_currents[sc]
        forces = [c * 0.5 for c in currents]  # 1A ≈ 0.5N
        stats[sc] = {
            "n_frames": len(currents),
            "n_episodes": len(scenario_episodes.get(sc, set())),
            "avg_current_a": sum(currents) / len(currents),
            "avg_force_n": sum(forces) / len(forces),
            "max_force_n": max(forces),
        }
    stats["__total_frames__"] = n_total
    return stats


# ------------------------------------------------------------------------- #
# 电机参数辨识: 从 DC 阶跃响应拟合 Rs, Ld, Lq
# ------------------------------------------------------------------------- #
def motor_identify_dc(v_dc: float, i_measured: float) -> Dict[str, float]:
    """直流压降法测 Rs: Rs = V / I (电机堵转)。"""
    if i_measured < 0.01:
        return {"error": "电流过小, 无法辨识"}
    rs = v_dc / i_measured
    return {
        "Rs_ohm": round(rs, 6),
        "method": "dc_voltage_drop",
        "input": {"V": v_dc, "I": i_measured},
    }


def motor_identify_lcr(freq_hz: float, z_magnitude: float, z_phase_deg: float) -> Dict[str, float]:
    """LCR 表法测电感: |Z|=sqrt(Rs²+(ωL)²), φ=atan(ωL/Rs)。"""
    import math
    omega = 2 * math.pi * freq_hz
    z = z_magnitude * math.cos(math.radians(z_phase_deg))  # 实部 ≈ Rs
    x = z_magnitude * math.sin(math.radians(z_phase_deg))  # 虚部 = ωL
    l = x / omega if omega > 0 else 0.0
    rs = z
    return {
        "Rs_ohm": round(rs, 6),
        "L_h": round(l, 9),
        "method": "lcr_meter",
        "input": {"freq_hz": freq_hz, "Z_mag": z_magnitude, "Z_phase": z_phase_deg},
    }


def motor_identify_bemf(rpm: float, v_line_v: float, pole_pairs: int) -> Dict[str, float]:
    """反电动势法测磁链: Ke = V_ll / (2π·rpm/60·pp), ψ = Ke/pp。"""
    import math
    omega_elec = 2 * math.pi * (rpm / 60.0) * pole_pairs
    ke = v_line_v / omega_elec if omega_elec > 0 else 0.0
    psi_m = ke / pole_pairs if pole_pairs > 0 else 0.0
    return {
        "Ke_vs_per_rad": round(ke, 6),
        "psi_m_wb": round(psi_m, 8),
        "method": "back_emf",
        "input": {"rpm": rpm, "V_line": v_line_v, "pp": pole_pairs},
    }


# ------------------------------------------------------------------------- #
# HIL 数据导入 + 分析
# ------------------------------------------------------------------------- #
def hil_import_csv(csv_path: Path, expected_col: str = "current_a") -> Dict[str, Any]:
    """导入示波器 CSV (格式: time_s, current_a)。"""
    times, currents = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row.get("time_s", row.get("Time", 0))))
            currents.append(float(row.get(expected_col, row.get("CH1", 0))))

    dt = (times[-1] - times[0]) / len(times) if len(times) > 1 else 0.0
    # 找阶跃响应参数
    i_final = sum(currents[-10:]) / 10 if len(currents) >= 10 else currents[-1]
    i_max = max(currents)
    overshoot = (i_max - i_final) / i_final * 100 if i_final > 0 else 0

    # 上升时间 (10% → 90%):
    t_rise = None
    i_10 = 0.1 * i_final
    i_90 = 0.9 * i_final
    for i, (t, c) in enumerate(zip(times, currents)):
        if c >= i_10 and t_rise is None:
            t_rise_10 = t
        if c >= i_90 and t_rise is not None:
            t_rise = t - t_rise_10
            break

    return {
        "path": str(csv_path),
        "n_samples": len(times),
        "dt_s": round(dt, 6),
        "i_final_a": round(i_final, 4),
        "i_max_a": round(i_max, 4),
        "overshoot_pct": round(overshoot, 2),
        "rise_time_ms": round(t_rise * 1000, 2) if t_rise else None,
    }


def hil_gap_analyze(expected_ms: Optional[float], actual_ms: Optional[float],
                     label: str = "step_response") -> Dict[str, Any]:
    """对比仿真预言 vs 实测。"""
    if expected_ms is None or actual_ms is None:
        return {"error": "需要 expected_ms 和 actual_ms 两个参数"}
    gap = actual_ms - expected_ms
    gap_pct = gap / expected_ms * 100 if expected_ms > 0 else 0
    if abs(gap_pct) < 5:
        status = "PASS"
    elif abs(gap_pct) < 20:
        status = "WARN — 偏差可接受, 建议排查"
    else:
        status = "FAIL — 偏差过大, 需归因"

    notes = []
    if abs(gap_pct) > 20:
        notes.append("检查 Rs/Ld 是否实测值与代码值一致")
        notes.append("检查 PWM 死区时间是否设置正确")
        notes.append("检查编码器角度是否零位对齐")

    return {
        "label": label,
        "expected_ms": expected_ms,
        "actual_ms": actual_ms,
        "gap_ms": round(gap, 3),
        "gap_pct": round(gap_pct, 2),
        "status": status,
        "notes": notes,
    }


# ------------------------------------------------------------------------- #
# 失败记录
# ------------------------------------------------------------------------- #
def hil_failure_log(stage: str, symptom: str, expected: str, actual: str,
                     root_cause: str = "", evidence: str = "") -> Path:
    """记录一次调试失败。输出 JSON 文件。"""
    FAIL_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    record = {
        "timestamp": ts,
        "stage": stage,
        "symptom": symptom,
        "expected": expected,
        "actual": actual,
        "root_cause": root_cause,
        "evidence": evidence,
    }
    out_path = FAIL_DIR / f"{ts}_{stage.replace(' ', '_')}.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ------------------------------------------------------------------------- #
# 焊接前电气检查清单
# ------------------------------------------------------------------------- #
def weld_pre_check(print_only: bool = True) -> List[Dict[str, Any]]:
    """焊接前检查清单 (B1 板)。"""
    checks = [
        {"item": "PCB 5V LDO 输出", "tool": "万用表", "target": "5.0 ±0.1 V", "pass": None},
        {"item": "PCB 3.3V LDO 输出", "tool": "万用表", "target": "3.3 ±0.1 V", "pass": None},
        {"item": "电源对地阻抗 (>10 kΩ)", "tool": "万用表 200kΩ", "target": ">10 kΩ", "pass": None},
        {"item": "USB-C 供电不稳?", "tool": "万用表 + 示波器", "target": "无振荡", "pass": None},
        {"item": "编码器 5V 输出", "tool": "万用表", "target": "5.0 ±0.1 V", "pass": None},
        {"item": "CAN 120 Ω 终端电阻", "tool": "万用表", "target": "≈60 Ω (2端并联)", "pass": None},
        {"item": "栅极驱动 PWM 静态", "tool": "万用表", "target": "0V (电机静止)", "pass": None},
        {"item": "电流传感零点 (<50 mV)", "tool": "万用表", "target": "<50 mV", "pass": None},
    ]
    return checks


# ------------------------------------------------------------------------- #
# CLI 入口
# ------------------------------------------------------------------------- #
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hardware utilities for HIL/manufacturing")
    sub = parser.add_subparsers(dest="cmd")

    # d4-analyze
    sub.add_parser("d4-analyze", help="分析 D4 数据集 scenario 分布")

    # motor-identify
    p_id = sub.add_parser("motor-identify", help="电机参数辨识")
    p_id.add_argument("--method", choices=["dc", "lcr", "bemf"], required=True)
    p_id.add_argument("--v-dc", type=float, help="DC 电压 (dc法)")
    p_id.add_argument("--i-meas", type=float, help="实测电流 (dc法)")
    p_id.add_argument("--freq", type=float, help="LCR 频率 (lcr法)")
    p_id.add_argument("--z-mag", type=float, help="阻抗幅值 (lcr法)")
    p_id.add_argument("--z-phase", type=float, help="阻抗相位 (lcr法)")
    p_id.add_argument("--rpm", type=float, help="转速 (bemf法)")
    p_id.add_argument("--v-line", type=float, help="线电压 (bemf法)")
    p_id.add_argument("--pp", type=int, help="极对数 (bemf法)")

    # hil-import-csv
    p_csv = sub.add_parser("hil-import-csv", help="导入示波器 CSV")
    p_csv.add_argument("--csv", type=Path, required=True)
    p_csv.add_argument("--col", default="current_a")

    # hil-gap-analyze
    p_gap = sub.add_parser("hil-gap-analyze", help="对比实测 vs 预言")
    p_gap.add_argument("--expected-ms", type=float)
    p_gap.add_argument("--actual-ms", type=float)
    p_gap.add_argument("--label", default="step_response")

    # hil-failure-log
    p_fail = sub.add_parser("hil-failure-log", help="记录调试失败")
    p_fail.add_argument("--stage", required=True)
    p_fail.add_argument("--symptom", required=True)
    p_fail.add_argument("--expected", required=True)
    p_fail.add_argument("--actual", required=True)
    p_fail.add_argument("--root-cause", default="")
    p_fail.add_argument("--evidence", default="")

    # weld-pre-check
    sub.add_parser("weld-pre-check", help="焊接前电气检查清单")

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return 0

    if args.cmd == "d4-analyze":
        stats = d4_analyze()
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    elif args.cmd == "motor-identify":
        if args.method == "dc":
            r = motor_identify_dc(args.v_dc, args.i_meas)
        elif args.method == "lcr":
            r = motor_identify_lcr(args.freq, args.z_mag, args.z_phase)
        elif args.method == "bemf":
            r = motor_identify_bemf(args.rpm, args.v_line, args.pp)
        else:
            r = {"error": "未知方法"}
        print(json.dumps(r, ensure_ascii=False, indent=2))

    elif args.cmd == "hil-import-csv":
        r = hil_import_csv(args.csv, args.col)
        print(json.dumps(r, ensure_ascii=False, indent=2))

    elif args.cmd == "hil-gap-analyze":
        r = hil_gap_analyze(args.expected_ms, args.actual_ms, args.label)
        print(json.dumps(r, ensure_ascii=False, indent=2))

    elif args.cmd == "hil-failure-log":
        p = hil_failure_log(args.stage, args.symptom, args.expected, args.actual,
                             args.root_cause, args.evidence)
        print(f"失败记录已保存: {p}")

    elif args.cmd == "weld-pre-check":
        checks = weld_pre_check()
        for i, c in enumerate(checks, 1):
            print(f"  [{i}] {c['item']}: 用{c['tool']}测, 目标={c['target']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
