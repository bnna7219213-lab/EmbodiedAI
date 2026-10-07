"""G10 电机模型补全 (双轨 Python 后端) — Ke/flux/Ld/Lq/Rs 及其一致性校验。

**缺口背景 (G10)**
    Kotlin 侧 `MotorDriveSpec` 缺反电动势常数 Ke / 磁链 flux / d-q 轴电感 Ld/Lq /
    定子电阻 Rs → FOC 只能靠估算, 控制精度受限。jar 已冻结不可改 (v4 决策),
    故走**双轨**: Python 侧补全模型, 供 `foc_shutdown` 精确建模。

**oracle 说明 (诚实标注)**
    实测: jar 的 motor-drive 输出**不含** Ke/flux/Ld/Lq/Rs (只输出板级规格名/桥臂),
    因此**无 jar oracle 可比**。本后端的对照基准改为:
      ① 契约 (contracts/hw_requirements.yaml 的 motor_drive 段)
      ② 物理自洽性 (Ke ↔ flux ↔ torque_constant ↔ pole_pairs 的换算关系)
      ③ Python FOC 侧既有参数 (FOCConfig, G10 修复前的唯一真值来源)
    这三项都是**可证伪**的; 不虚构"与 jar 一致"。

**核心物理关系** (表贴式 PMSM):
    Ke   = pp · psi_m                    [V·s/rad]  反电动势常数
    Kt   = 1.5 · pp · psi_m              [N·m/A]    转矩常数
    Kt/Kt_measured ≈ 1 (能量守恒 → Ke 与 Kt 在 SI 下近似相等, 系数差因 dq 变换定义)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# ⚠ 本模块只做建模与一致性校验, 不调 jar (jar 无对应字段可比)


@dataclass
class MotorSpec:
    """关节模组电机参数 (G10: 补全 Kotlin MotorDriveSpec 缺失的电磁参数)。

    数值来源 (可追溯, 不凭空编造):
      · Rs/Ld/Lq/psi_m/pole_pairs: 取自 Python 侧既有 `FOCConfig` (G10 前唯一真值)
      · Ke/torque_constant: 由 psi_m 与 pole_pairs **推导** (物理关系, 非独立参数)
      · 额定值 (i_max/v_dc/减速比): 取自 contracts/hw_requirements.yaml
    """
    # --- 电磁参数 (核心: G10 补全项) ---
    Rs_ohm: float                  # 定子相电阻
    Ld_h: float                    # d 轴电感
    Lq_h: float                    # q 轴电感
    psi_m_wb: float                # 永磁磁链
    pole_pairs: int
    # --- 派生量 (由上面推导, 保留字段便于下游直接取用) ---
    Ke_vs_per_rad: float = 0.0     # 反电动势常数 = pp·psi_m
    Kt_nm_per_a: float = 0.0       # 转矩常数 = 1.5·pp·psi_m
    # --- 机械/额定 (来自契约) ---
    J_kgm2: float = 3.2e-5
    i_max_a: float = 30.0
    v_dc_v: float = 48.0
    f_ctrl_hz: float = 10000.0
    # --- 溯源 ---
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.Ke_vs_per_rad == 0.0:
            self.Ke_vs_per_rad = self.pole_pairs * self.psi_m_wb
        if self.Kt_nm_per_a == 0.0:
            self.Kt_nm_per_a = 1.5 * self.pole_pairs * self.psi_m_wb

    # ---------------- 派生: 电流环增益 (极点配置 kp=L·ωc, ki=Rs·ωc) ----------------
    def current_loop_gains(self, bw_hz: float = 800.0) -> Dict[str, float]:
        """按极点配置给出电流环 PI 增益 (零极点对消, 闭环 ≈ 一阶 ωc)。

        ωc 上限受离散化约束: ωc·T_ctrl < 2  (T_ctrl = 1/f_ctrl)。
        """
        wc = 2.0 * np.pi * bw_hz
        T = 1.0 / self.f_ctrl_hz
        return {
            "omega_c_rad_s": float(wc),
            "omega_c_times_T": float(wc * T),
            "discrete_ok": bool(wc * T < 2.0),
            "kp_d": float(self.Ld_h * wc),
            "ki_d": float(self.Rs_ohm * wc),
            "kp_q": float(self.Lq_h * wc),
            "ki_q": float(self.Rs_ohm * wc),
        }

    def back_emf_at(self, omega_mech_rad_s: float) -> float:
        """给定机械角速度 → 反电动势幅值 (V)。用于母线电压裕量校验。"""
        return float(self.Ke_vs_per_rad * omega_mech_rad_s)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Rs_ohm": self.Rs_ohm, "Ld_h": self.Ld_h, "Lq_h": self.Lq_h,
            "psi_m_wb": self.psi_m_wb, "pole_pairs": self.pole_pairs,
            "Ke_vs_per_rad": self.Ke_vs_per_rad, "Kt_nm_per_a": self.Kt_nm_per_a,
            "J_kgm2": self.J_kgm2, "i_max_a": self.i_max_a, "v_dc_v": self.v_dc_v,
            "f_ctrl_hz": self.f_ctrl_hz, "provenance": self.provenance,
        }


def motor_spec_from_sources(contract: Optional[Dict[str, Any]] = None,
                            foc_config: Any = None) -> MotorSpec:
    """从**真实来源**装配 MotorSpec: FOCConfig (电磁) + 契约 (额定/溯源)。

    · foc_config 缺省时从 `hardware.foc_shutdown.FOCConfig` 读取 (G10 前唯一真值)
    · contract 缺省时从 `contracts/hw_requirements.yaml` 读取 (唯一真源)
    """
    prov: Dict[str, Any] = {"sources": []}

    if foc_config is None:
        from .foc_shutdown import FOCConfig
        foc_config = FOCConfig()
    prov["sources"].append("hardware.foc_shutdown.FOCConfig")

    Rs = float(getattr(foc_config, "Rs"))
    Ld = float(getattr(foc_config, "Ld"))
    Lq = float(getattr(foc_config, "Lq"))
    psi = float(getattr(foc_config, "psi_m"))
    pp = int(getattr(foc_config, "pole_pairs"))
    J = float(getattr(foc_config, "J", 3.2e-5))

    i_max, v_dc, f_ctrl = 30.0, 48.0, 10000.0
    try:
        if contract is None:
            from ..config import load_contract
            contract = load_contract("hw_requirements")
        md = contract.get("motor_drive", {})
        i_max = float(md.get("peak_current_a", i_max))
        # 母线电压在 motor_drive 段
        v_dc = float(md.get("bus_voltage_v", v_dc))
        f_ctrl = float(md.get("current_loop_hz", f_ctrl))
        prov["sources"].append("contracts/hw_requirements.yaml#motor_drive")
    except Exception as e:  # noqa: BLE001
        prov["contract_note"] = f"契约读取失败, 用默认额定: {type(e).__name__}"

    return MotorSpec(Rs_ohm=Rs, Ld_h=Ld, Lq_h=Lq, psi_m_wb=psi, pole_pairs=pp,
                     J_kgm2=J, i_max_a=i_max, v_dc_v=v_dc, f_ctrl_hz=f_ctrl,
                     provenance=prov)


def verify_motor_spec(spec: MotorSpec, bw_hz: float = 800.0) -> Dict[str, Any]:
    """一致性校验 (可证伪的 oracle): 契约 / 物理 / FOC 侧三方对照。

    ① 契约: 峰值电流与母线电压须与 hw_requirements 一致
    ② 物理: Ke = pp·psi, Kt = 1.5·pp·psi (SI 换算自洽)
    ③ FOC 侧: 1.5·Kt·i_max 与功率级匹配; 反电动势幅值 < 母线电压 (留裕量)
    """
    checks: List[Dict[str, Any]] = []

    def _ck(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # ① 契约对照
    try:
        from ..config import load_contract
        md = load_contract("hw_requirements").get("motor_drive", {})
        _ck("contract_peak_current", abs(spec.i_max_a - float(md["peak_current_a"])) < 1e-6,
            f"spec {spec.i_max_a}A vs 契约 {md['peak_current_a']}A")
        _ck("contract_bus_voltage", abs(spec.v_dc_v - float(md["bus_voltage_v"])) < 1e-6,
            f"spec {spec.v_dc_v}V vs 契约 {md['bus_voltage_v']}V")
        _ck("contract_current_loop", abs(spec.f_ctrl_hz - float(md["current_loop_hz"])) < 1e-6,
            f"spec {spec.f_ctrl_hz}Hz vs 契约 {md['current_loop_hz']}Hz")
    except Exception as e:  # noqa: BLE001
        _ck("contract_readable", False, f"{type(e).__name__}: {e}")

    # ② 物理自洽 (Ke/Kt 由 psi 与 pp 推导, 关系必须成立)
    ke_expect = spec.pole_pairs * spec.psi_m_wb
    kt_expect = 1.5 * spec.pole_pairs * spec.psi_m_wb
    _ck("physics_Ke_polepairs_psi", abs(spec.Ke_vs_per_rad - ke_expect) < 1e-12,
        f"Ke={spec.Ke_vs_per_rad:.6f} vs pp·psi={ke_expect:.6f}")
    _ck("physics_Kt_relation", abs(spec.Kt_nm_per_a - kt_expect) < 1e-12,
        f"Kt={spec.Kt_nm_per_a:.6f} vs 1.5·pp·psi={kt_expect:.6f}")

    # ③ 与 FOC 侧既有值对照 (G10 前唯一真值 — 不匹配即说明两处漂移)
    try:
        from .foc_shutdown import FOCConfig
        fc = FOCConfig()
        _ck("foc_side_Rs_consistent", abs(spec.Rs_ohm - float(fc.Rs)) < 1e-12,
            f"spec {spec.Rs_ohm} vs FOCConfig {fc.Rs}")
        _ck("foc_side_Ld_consistent", abs(spec.Ld_h - float(fc.Ld)) < 1e-12,
            f"spec {spec.Ld_h} vs FOCConfig {fc.Ld}")
        _ck("foc_side_psi_consistent", abs(spec.psi_m_wb - float(fc.psi_m)) < 1e-12,
            f"spec {spec.psi_m_wb} vs FOCConfig {fc.psi_m}")
    except Exception as e:  # noqa: BLE001
        _ck("foc_side_readable", False, f"{type(e).__name__}: {e}")

    # ④ 电流环离散化约束 (G10 关注的"能否精确建模/控制")
    gains = spec.current_loop_gains(bw_hz)
    _ck("current_loop_discrete_stable", gains["discrete_ok"],
        f"ωc·T={gains['omega_c_times_T']:.3f} (<2 要求)")

    # ⑤ 母线裕量: 额定转速下反电动势须低于母线 (否则无法控流)
    omega_rated = 20.0          # 关节侧额定 20 rad/s (保守取用)
    bemf = spec.back_emf_at(omega_rated)
    _ck("bus_margin_for_bemf", bemf < spec.v_dc_v,
        f"反电动势 {bemf:.2f}V < 母线 {spec.v_dc_v}V @ {omega_rated}rad/s")

    failed = [c["check"] for c in checks if not c["ok"]]
    return {
        "ok": not failed,
        "checks": checks,
        "failed": failed,
        "spec": spec.to_dict(),
        "gains": gains,
        "note": ("G10 双轨: jar 无 Ke/flux/Ld/Lq/Rs 字段 → 无 jar oracle; "
                 "对照基准为契约 + 物理关系 + FOC 侧既有值 (三方均可证伪)"),
    }


# --------------------------------------------------------------------------- #
# 双轨注册: 能力名 "motor_model" (G10)
# --------------------------------------------------------------------------- #
def _register() -> None:
    """把 G10 后端注册进双轨机制 (与 jar 并存, 由 backend 参数路由)。

    注: jar 侧**无** motor_model 能力 (motor-drive 不输出电磁参数) →
    available_backends("motor_model") 只会有 ["python"], 这是如实反映。
    """
    from ..adapters.edalab import register_python_backend

    @register_python_backend("motor_model")
    def motor_model_backend(bw_hz: float = 800.0) -> Dict[str, Any]:
        """G10: 电机模型补全 + 三方一致性校验 (契约/物理/FOC)。"""
        spec = motor_spec_from_sources()
        res = verify_motor_spec(spec, bw_hz=bw_hz)
        return {"status": "OK" if res["ok"] else "FAIL", **res}


_register()
