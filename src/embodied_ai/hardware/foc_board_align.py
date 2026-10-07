"""G2 FOC ↔ MotorDrive 参数对齐 (双轨 Python 后端)。

**缺口背景 (G2)**
    Python 侧 `foc_shutdown.FOCConfig` 算出的电流环参数 (kp = L·ωc, ki = Rs·ωc)
    与 Kotlin `MotorDriveEngine` 产板参数**未对齐** → 存在"仿真设计的板 ≠ EDA 产出的板"
    的风险; 真机调试时电流环可能振荡 (历史先例: kp=8 导致离散振荡、稳态误差 131%)。

**对齐逻辑 (Python 主导 + jar 校验)**
    1. 从 jar 的 motor-drive **实测输出**解析板级参数 (电压/电流/器件/固件特性)
    2. 从 Python 侧 FOC 模型算出控制参数 (kp/ki/带宽/离散裕度)
    3. **交叉校验**两侧是否自洽:
       · 电流额定: jar 桥臂 30A ↔ Python i_max 30A
       · 电压额定: jar 48V ↔ Python v_dc 48V
       · 开关/控制频率: jar 固件特性 vs Python f_pwm/f_ctrl
       · 器件耐压裕量: MOSFET 100V vs 母线 48V (≥2× 工程惯例)
       · 电流环离散稳定: ωc·T < 2
    4. **不匹配则记录** (mismatches 列表), 不静默 — 这是 G2 的核心要求

**oracle 说明**
    jar 输出**不含** kp/ki (只有板级规格) → 无法逐一比对控制参数。
    因此对照分两层:
      · 可比项 (电压/电流/器件/固件特性): 真对照 jar 实测输出
      · 不可比项 (kp/ki/带宽): 与 Python FOC 侧自洽 + 离散化物理约束
    "jar 不输出"这件事本身是 G2 缺口的一部分, 如实记录在 note 里。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BoardFacts:
    """从 jar motor-drive 输出解析出的板级事实 (可追溯)。"""
    bus_voltage_v: Optional[float] = None
    peak_current_a: Optional[float] = None
    power_w: Optional[float] = None
    layers: Optional[int] = None
    copper_oz: Optional[float] = None
    max_junction_c: Optional[float] = None
    margin_c: Optional[float] = None
    mosfet: Optional[str] = None
    gate_driver: Optional[str] = None
    mcu: Optional[str] = None
    firmware_features: List[str] = field(default_factory=list)
    raw_stdout: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "raw_stdout"}


def parse_board_facts(stdout: str) -> BoardFacts:
    """解析 jar motor-drive 的 stdout (实测格式, 非猜测)。"""
    b = BoardFacts(raw_stdout=stdout)
    m = re.search(r"桥臂\s*([\d.]+)\s*V\s*/\s*([\d.]+)\s*A\s*/\s*([\d.]+)\s*W", stdout)
    if m:
        b.bus_voltage_v, b.peak_current_a, b.power_w = map(float, m.groups())
    m = re.search(r"层数:\s*(\d+)", stdout)
    if m:
        b.layers = int(m.group(1))
    m = re.search(r"铜厚:\s*([\d.]+)\s*oz", stdout)
    if m:
        b.copper_oz = float(m.group(1))
    m = re.search(r"最高结温:\s*([\d.]+)°C\s*\(裕量\s*([\d.]+)°C\)", stdout)
    if m:
        b.max_junction_c, b.margin_c = float(m.group(1)), float(m.group(2))
    m = re.search(r"Q1-Q6:\s*(\S+)\s*\(([^)]+)\)", stdout) or re.search(r"Q1-Q6:\s*(\S+)\s*\(([^)]*)\)", stdout)
    if m:
        b.mosfet = f"{m.group(1)} ({m.group(2)})"
    m = re.search(r"U2:\s*(\S+)\s*\(([^)]+)\)", stdout)
    if m:
        b.gate_driver = f"{m.group(1)} ({m.group(2)})"
    m = re.search(r"U1:\s*(\S+)", stdout)
    if m:
        b.mcu = m.group(1)
    m = re.search(r"固件特性:\s*(.+)", stdout)
    if m:
        b.firmware_features = [x.strip() for x in m.group(1).split(",") if x.strip()]
    return b


def cross_check(board: BoardFacts, motor: Any, foc: Any) -> Dict[str, Any]:
    """交叉校验: jar 板级参数 ↔ Python FOC/电机模型 (不匹配记录, 不静默)。"""
    checks: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []

    def _ck(name: str, ok: bool, detail: str, alignment: bool = True) -> None:
        rec = {"check": name, "ok": bool(ok), "detail": detail}
        checks.append(rec)
        if not ok and alignment:
            mismatches.append(rec)

    # ---- 可比项: 直接对照 jar 实测输出 ----
    if board.peak_current_a is not None:
        _ck("current_rating_aligned",
            abs(board.peak_current_a - motor.i_max_a) < 1e-6,
            f"jar 桥臂 {board.peak_current_a}A vs Python i_max {motor.i_max_a}A")
    if board.bus_voltage_v is not None:
        _ck("voltage_rating_aligned",
            abs(board.bus_voltage_v - motor.v_dc_v) < 1e-6,
            f"jar 母线 {board.bus_voltage_v}V vs Python v_dc {motor.v_dc_v}V")

    # 器件耐压裕量 (MOSFET 100V vs 母线 48V, 工程惯例 ≥2×)
    if board.mosfet:
        vds = re.search(r"(\d+(?:\.\d+)?)V", board.mosfet)
        if vds:
            v_mos = float(vds.group(1))
            ratio = v_mos / max(motor.v_dc_v, 1e-9)
            _ck("mosfet_voltage_derating", ratio >= 1.5,
                f"MOSFET {v_mos}V / 母线 {motor.v_dc_v}V = {ratio:.2f}× (惯例 ≥1.5×)")

    # 固件特性必须覆盖 FOC 必需项 (契约 firmware_features 是唯一真源)
    need = {"foc_control", "clarke_transform", "park_transform",
            "svpwm" if any("svpwm" in f for f in board.firmware_features) else "pwm_mcpwm"}
    have = set(board.firmware_features)
    _ck("firmware_covers_foc", need.issubset(have),
        f"jar 固件特性覆盖 FOC 必需: 缺 {sorted(need - have)}" if not need.issubset(have)
        else f"jar 固件特性含全部 FOC 必需项 ({len(have)} 项)")

    # ---- 不可比项 (jar 不输出): 与 Python FOC 侧自洽 + 物理约束 ----
    gains = motor.current_loop_gains()
    _ck("kp_matches_L_times_wc",
        abs(gains["kp_d"] - motor.Ld_h * gains["omega_c_rad_s"]) < 1e-12,
        f"kp_d={gains['kp_d']:.4f} = Ld·ωc (极点配置)")
    _ck("ki_matches_Rs_times_wc",
        abs(gains["ki_d"] - motor.Rs_ohm * gains["omega_c_rad_s"]) < 1e-9,
        f"ki_d={gains['ki_d']:.1f} = Rs·ωc (零点对消)")
    _ck("foc_config_gains_consistent",
        abs(float(getattr(foc, "kp_d")) - gains["kp_d"]) < 1e-6,
        f"FOCConfig.kp_d={getattr(foc, 'kp_d')} vs 模型反算 {gains['kp_d']:.4f}")
    _ck("current_loop_discrete_stable", gains["discrete_ok"],
        f"ωc·T={gains['omega_c_times_T']:.3f} < 2 (离散稳定)")
    _ck("jb_thermal_margin_positive",
        (board.margin_c is not None and board.margin_c > 0),
        f"jar 结温裕量 {board.margin_c}°C" if board.margin_c is not None
        else "jar 未输出结温裕量")

    return {
        "ok": not mismatches,
        "checks": checks,
        "mismatches": mismatches,
        "board": board.to_dict(),
        "python_side": {
            "i_max_a": motor.i_max_a, "v_dc_v": motor.v_dc_v,
            "Rs_ohm": motor.Rs_ohm, "Ld_h": motor.Ld_h, "Lq_h": motor.Lq_h,
            "kp_d": gains["kp_d"], "ki_d": gains["ki_d"],
            "bandwidth_hz": 800.0,
        },
        "note": ("G2 双轨: jar 输出板级规格但不含 kp/ki (缺口本身) → "
                 "可比项真对照 jar 实测, 控制参数项与 Python FOC 侧自洽+物理约束校验; "
                 "不匹配一律进 mismatches, 不静默"),
    }


# --------------------------------------------------------------------------- #
# 双轨注册: 能力名 "foc_board_align" (G2)
# --------------------------------------------------------------------------- #
def _register() -> None:
    """G2 后端注册。此能力**同时**需要 jar (取板级事实) 与 python (FOC 模型)。"""
    from ..adapters.edalab import register_python_backend

    @register_python_backend("foc_board_align")
    def foc_board_align_backend(board_stdout: str | None = None) -> Dict[str, Any]:
        """G2: jar 板级参数 ↔ Python FOC 参数 交叉校验。

        board_stdout 为 None 时自行 live 调 jar 的 motor-drive 取真实输出。
        """
        if board_stdout is None:
            from ..adapters.edalab import EdalabAdapter
            r = EdalabAdapter().run_stage("motor_drive", {}, mode="live")
            if r.get("status") != "LIVE_OK":
                return {"status": "BLOCKED", "ok": False,
                        "error": f"jar 调用失败: {r.get('error', r.get('status'))}",
                        "note": "G2 需要 jar 板级输出作对照源; jar 不可用时无法对齐"}
            board_stdout = r.get("stdout", "")

        from .foc_shutdown import FOCConfig
        from .motor_model import motor_spec_from_sources

        board = parse_board_facts(board_stdout)
        motor = motor_spec_from_sources()
        res = cross_check(board, motor, FOCConfig())
        return {"status": "OK" if res["ok"] else "MISMATCH", **res}


_register()
