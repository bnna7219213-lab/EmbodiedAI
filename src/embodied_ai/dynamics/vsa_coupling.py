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
    energy_j: float


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
    """加载 EdaLab vsa-static --json 输出 (vsa_curve_v1 schema)。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("schema") != "vsa_curve_v1":
        raise ValueError(f"未知 schema: {d.get('schema')}, 期望 vsa_curve_v1")
    out = []
    for t in d["topologies"]:
        out.append(VsaTopologyData(
            name=t["name"], topology=t["topology"],
            k_min=t["k_min"], k_max=t["k_max"],
            elastic_travel_deg=t["elastic_travel_deg"],
            link_length_m=t["link_length_m"], mass_kg=t["mass_kg"],
            stored_energy_j=t["stored_energy_j"], peak_force_n=t["peak_force_n"],
            release_velocity_mps=t["release_velocity_mps"],
            bandwidth_min_hz=t["bandwidth_min_hz"], bandwidth_max_hz=t["bandwidth_max_hz"],
            stiffness_ratio=t["stiffness_ratio"], ok=t["ok"],
            curve=[VsaCurvePoint(p["theta_deg"], p["k_nm_per_rad"],
                                 p["torque_max_nm"], p["energy_j"]) for p in t["curve"]],
        ))
    return out


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
    """
    v = math.sqrt(2.0 * vsa.stored_energy_j / max(vsa.mass_kg, 1e-6))
    rad = math.radians(direction_deg)
    return [v * math.cos(rad), v * math.sin(rad)]


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
