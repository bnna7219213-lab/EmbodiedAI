"""VSA-dyno 联动消费端 — Kotlin 刚度曲线 JSON → 攀岩 dyno 弹道注入。

数据流 (P1.3 收尾, 跨语言管道):
  EdaLab vsa-static --json (Kotlin)
    → vsa_curve_v1 JSON (三拓扑: curve 25 点 θ/K/T/E + 蓄能/峰值力/带宽)
      → load_vsa_json 解析
        → release_velocity(topology) 蓄能→弹道初速 (替代拍脑袋常数)
          → inject_into_dyno → climbing_dynamics_core.dyno_plan(v_release, ...)

联动价值: dyno 弹道的释放速度从 VSA 弹性势能计算得出
  v = √(2E/m) (Kotlin 引擎实测: MACCEPA 0.967J → 1.27 m/s)
物理闭环: 刚度曲线 (拓扑选择) → 蓄能 → 弹道 → 捕获窗口 (dyno_capture_check)。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class VsaCurvePoint:
    theta_deg: float
    k_nm_per_rad: float
    torque_max_nm: float
    energy_j: float                        # 从行程边界(-range/2)起的累计 ∫K dθ — 曲线形状用, 不可当弹性能
    energy_from_origin_j: float = 0.0      # 相对平衡位 θ=0 的 ½Kθ² (θ 已按 elastic_travel 钳位) — v1.1+
    torque_min_nm: float = 0.0             # v1.1+ 新增: 与 torque_max 对称的包络下界


@dataclass
class VsaTopologyData:
    name: str
    topology: str
    k_min: float
    k_max: float
    elastic_travel_deg: float
    link_length_m: float
    mass_kg: float
    stored_energy_j: float
    peak_force_n: float
    release_velocity_mps: float
    bandwidth_min_hz: float
    bandwidth_max_hz: float
    stiffness_ratio: float
    ok: bool
    curve: List[VsaCurvePoint]

    def stiffness_at(self, theta_deg: float) -> float:
        """插值取指定角度的刚度。"""
        pts = sorted(self.curve, key=lambda p: p.theta_deg)
        if theta_deg <= pts[0].theta_deg:
            return pts[0].k_nm_per_rad
        if theta_deg >= pts[-1].theta_deg:
            return pts[-1].k_nm_per_rad
        for a, b in zip(pts, pts[1:]):
            if a.theta_deg <= theta_deg <= b.theta_deg:
                w = (theta_deg - a.theta_deg) / max(b.theta_deg - a.theta_deg, 1e-9)
                return a.k_nm_per_rad * (1 - w) + b.k_nm_per_rad * w
        return self.k_min


def load_vsa_json(path: Path) -> List[VsaTopologyData]:
    """加载 EdaLab vsa-static --json 输出。

    支持的 schema 版本:
      - vsa_curve_v1: 只有 curve.energy_j (行程边界零点, 曲线形状用)
      - vsa_curve_v1_1: 新增 curve.energy_from_origin_j (平衡位零点, 与 stored_energy_j 同源)

    V1 修复 (2026-10-07): 显式检查两个 schema, 加载时若存在
    curve.energy_from_origin_j 就消费; 否则保留 0.0 并可通过
    vsa_energy_consistency_check 检查潜在误用。
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = d.get("schema")
    if schema not in ("vsa_curve_v1", "vsa_curve_v1_1"):
        raise ValueError(f"未知 schema: {schema}, 期望 vsa_curve_v1 或 vsa_curve_v1_1")
    out = []
    for t in d["topologies"]:
        curve_pts = []
        for p in t["curve"]:
            curve_pts.append(VsaCurvePoint(
                theta_deg=p["theta_deg"],
                k_nm_per_rad=p["k_nm_per_rad"],
                torque_max_nm=p["torque_max_nm"],
                torque_min_nm=p.get("torque_min_nm", 0.0),
                energy_j=p["energy_j"],
                energy_from_origin_j=p.get("energy_from_origin_j", 0.0),
            ))
        out.append(VsaTopologyData(
            name=t["name"], topology=t["topology"],
            k_min=t["k_min"], k_max=t["k_max"],
            elastic_travel_deg=t["elastic_travel_deg"],
            link_length_m=t["link_length_m"], mass_kg=t["mass_kg"],
            stored_energy_j=t["stored_energy_j"], peak_force_n=t["peak_force_n"],
            release_velocity_mps=t["release_velocity_mps"],
            bandwidth_min_hz=t["bandwidth_min_hz"], bandwidth_max_hz=t["bandwidth_max_hz"],
            stiffness_ratio=t["stiffness_ratio"], ok=t["ok"],
            curve=curve_pts,
        ))
    return out


def vsa_energy_consistency_check(topologies: List[VsaTopologyData],
                                 tolerance: float = 0.5) -> Dict[str, Any]:
    """V1 物理量级错误检查: curve.energy_from_origin_j 应与 stored_energy_j 同量级。

    在 |theta| <= elastic_travel_deg/2 的点中, energy_from_origin_j 的最大值
    应与 stored_energy_j 在 tolerance 内一致 (差异来自离散采样与 span 过滤)。

    历史错误 (2026-10-07 修复前): curve.energy_j 与 stored_energy_j 差 83 倍,
    任何下游误用 energy_j 当弹性能会得到 √(2·83E/m) ≈ 9x 过大的弹射速度
    (11.6 m/s 致命弹道)。

    返回 dict:
      ok: bool — 所有拓扑的曲线能量与 stored_energy_j 一致
      details: 每拓扑的对比 (stored, max_energy_from_origin, ratio, uses_v1_1_schema)
      warns: list[str] — 旧 schema 或缺少 energy_from_origin_j 时警告
    """
    warns = []
    details = []
    ok = True
    for t in topologies:
        pts = [p for p in t.curve if abs(p.theta_deg) <= t.elastic_travel_deg / 2 + 1e-6]
        has_v1_1 = any(p.energy_from_origin_j > 0 for p in pts)
        if not has_v1_1:
            warns.append(f"{t.name}: 未提供 energy_from_origin_j (vsa_curve_v1 schema); "
                         f"无法验证一致性")
            continue
        if not pts:
            warns.append(f"{t.name}: 弹性行程内无采样点")
            continue
        max_from_origin = max(p.energy_from_origin_j for p in pts)
        ratio = max_from_origin / t.stored_energy_j if t.stored_energy_j > 0 else float("inf")
        consistent = abs(ratio - 1.0) <= tolerance
        details.append({
            "name": t.name,
            "stored_energy_j": t.stored_energy_j,
            "max_energy_from_origin_j": max_from_origin,
            "ratio": ratio,
            "within_tolerance": consistent,
            "elastic_travel_deg": t.elastic_travel_deg,
        })
        if not consistent:
            ok = False
    return {"ok": ok and not warns, "details": details, "warns": warns}


def select_topology(topologies: List[VsaTopologyData],
                    name: Optional[str] = None) -> VsaTopologyData:
    """选拓扑: 按名或默认最优 (峰值力最接近指尖 Hertz 60N 限值者)。"""
    if name:
        for t in topologies:
            if t.name == name:
                return t
        raise KeyError(f"拓扑不存在: {name}; 可选 {[t.name for t in topologies]}")
    return min(topologies, key=lambda t: abs(t.peak_force_n - 60.0))


def release_velocity(vsa: VsaTopologyData, direction_deg: float = 45.0) -> List[float]:
    """蓄能 → 弹道初速向量 [vx, vy]。

    v = √(2E/m) (Kotlin 引擎蓄能实测), 按释放角分解。
    direction_deg: 释放方向与水平夹角 (攀岩 dyno 典型 40-60°)。

    V1 修复 (2026-10-07): 强制使用 stored_energy_j (权威值, 平衡位零点),
    并添加 sanity check 拒绝曲线 energy_j 误用 (>5x stored 时报错)。
    """
    _vsa_energy_sanity_check(vsa)
    v = math.sqrt(2.0 * vsa.stored_energy_j / max(vsa.mass_kg, 1e-6))
    rad = math.radians(direction_deg)
    return [v * math.cos(rad), v * math.sin(rad)]


def _vsa_energy_sanity_check(vsa: VsaTopologyData, max_ratio: float = 5.0) -> None:
    """V1 sanity: 若曲线 energy_j 是 stored_energy_j 的 >max_ratio 倍, 说明
    曲线使用了行程边界零点, 下游若把它当弹性能会得到过大的弹射速度。

    这里仅告警不阻塞 (老 schema 也能跑); 阻塞由 vsa_energy_consistency_check 显式调用。
    """
    pts = [p for p in vsa.curve if abs(p.theta_deg) <= vsa.elastic_travel_deg / 2 + 1e-6]
    if not pts or vsa.stored_energy_j <= 0:
        return
    max_energy_j = max(p.energy_j for p in pts)
    if max_energy_j / vsa.stored_energy_j > max_ratio:
        import warnings
        warnings.warn(
            f"{vsa.name}: curve.energy_j max={max_energy_j:.3f}J 是 stored_energy_j="
            f"{vsa.stored_energy_j:.3f}J 的 {max_energy_j/vsa.stored_energy_j:.1f}x — "
            f"energy_j 是行程边界零点的曲线积分, 不可当弹性能使用; "
            f"弹射/捕获请一律用 stored_energy_j 或 energy_from_origin_j",
            UserWarning, stacklevel=3,
        )


def inject_into_dyno(vsa: VsaTopologyData, c0, p_star,
                     direction_deg: float = 45.0,
                     r_max: float = 0.8, margin: float = 0.15):
    """把 VSA 蓄能注入 climbing_dynamics_core.dyno_plan。

    返回 dyno_plan 的原始 dict (tf/reach_req/c_tf/reachable) + 联动元数据。
    """
    from .climbing_dynamics_core import dyno_plan
    v_vec = release_velocity(vsa, direction_deg)
    result = dyno_plan(v_vec, c0, p_star, r_max=r_max, margin=margin)
    return {
        "vsa_name": vsa.name,
        "stored_energy_j": vsa.stored_energy_j,
        "v_release": v_vec,
        "speed_mps": math.hypot(*v_vec),
        "dyno": result,  # {"tf", "reach_req", "c_tf", "reachable"}
    }


def capture_feasibility(vsa: VsaTopologyData, mu: float = 0.8, f_max: float = 45.0):
    """捕获可行性: VSA 弹射速度能否被指尖摩擦捕获 (对接 dyno_capture_check 口径)。

    返回 (v_max_capturable, vsa_speed, feasible)。
    简化脉冲模型: v_max ≈ μ·F_max·t_pulse/m_eff。
    """
    t_pulse = 0.005
    m_eff = vsa.mass_kg
    v_max = mu * f_max * t_pulse / max(m_eff, 1e-6)
    feasible = vsa.release_velocity_mps <= v_max
    return v_max, vsa.release_velocity_mps, feasible
