"""HIL 桥接 (G6 双轨修复, 2026-10-08) — Python 侧编排 + 真实物理模型。

G6 缺口: Kotlin 侧 `HILBridge` **用随机数**代替真实电路仿真 (MNA) 与总线
(CAN-FD)。后果: **HIL 测试是假的**, 无法验证真实硬件行为 —— 这正是安全体系
"无 HIL 真证据"、评分停在 78 分的根因。

双轨修法 (路径 B) — **不重写 MNA**, 而是:
  - Python 侧做**编排**: 执行 `foc_shutdown.hil_testcases()` 的 7 条用例,
    用 Python 侧**已有的真实物理模型**给出判定 —— SVPWM 伏秒平衡、FOC 电流环
    阶跃 (堵转标定)、关机链时序、停止角反解。这些模型本身带量化验收
    (伏秒平衡 5e-15 量级、电流环极点配置)。
  - 电路级求解仍可由 jar 的 `sim` 提供 (如需更深电路仿真), Python 侧负责
    调用与判定 —— 双轨本意是"Python 补编排、jar 提供求解", 不重复造轮子。

★ 诚实边界: 7 条中 **2 条 (HIL-STO-02 / HIL-STO-05) 是硬件架构属性**
  (硬件级锁存与关断不依赖 MCU 软件), 软件仿真**无法证明** — 标记
  `REQUIRES_BENCH` (需台架实测), **不用随机数假装通过**。
  这比"7 条全绿"更可信: 前者知道自己证明了什么, 后者什么都证明不了。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..adapters.edalab import register_python_backend
from .foc_shutdown import FOCDrive, ShutdownChain, SVPWMModulator

__all__ = ["hil_backend", "REQUIRES_BENCH", "HIL_CASE_IDS"]

# 需台架实测的用例 (硬件架构属性, 软件仿真无法证明)
REQUIRES_BENCH = {"HIL-STO-02", "HIL-STO-05"}
HIL_CASE_IDS = ["HIL-STO-01", "HIL-STO-02", "HIL-STO-03", "HIL-STO-04",
                "HIL-STO-05", "HIL-FOC-01", "HIL-FOC-02"]


def _svpwm_balance(seed: int = 0) -> Dict[str, Any]:
    """HIL-FOC-02: SVPWM 伏秒平衡 (真实调制器, 非随机数)。"""
    sv = SVPWMModulator()
    rng = np.random.default_rng(seed)
    worst = 0.0
    mag = 0.5 * sv.v_dc / np.sqrt(3.0)      # 线性调制区上限
    for _ in range(200):
        th = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.05, 0.95) * mag
        _, diag = sv.duties(r * np.cos(th), r * np.sin(th))
        worst = max(worst, float(diag["recon_err"]))
    return {"measured": worst, "expect": 1e-6, "pass": bool(worst < 1e-6)}


def _current_step_error(seed: int = 0) -> Dict[str, Any]:
    """HIL-FOC-01: q 轴电流阶跃稳态误差 (堵转标定 — 见 foc_shutdown 注释)。

    必须堵转: 自由转子会被反电动势拖成"物理正确但不是跟踪误差"的假象。
    """
    drv = FOCDrive(seed=seed)
    drv.locked_rotor = True
    i_ref = 10.0
    dt = 1.0 / drv.cfg.f_ctrl
    for _ in range(int(0.05 / dt)):
        drv.step(i_ref, load_torque=0.0)
    err_pct = abs(drv.st.i_q - i_ref) / i_ref * 100.0
    return {"measured_pct": float(err_pct), "expect_pct": 2.0,
            "pass": bool(err_pct < 2.0 and np.isfinite(drv.st.i_q))}


def _shutdown_timing() -> Dict[str, Any]:
    """HIL-STO-01 / HIL-STO-03: 关机链电气段与总断开时延 (真实时序模型)。

    注: 这些延迟当前仍是 `assumed_*` 设计值 (契约诚实标注), 需 HIL 台架
    实测后回填 `safety_budget.yaml` 的 measured_* —— 但**判据本身是真实的**
    (不再用随机数生成一个"看起来通过"的时延)。
    """
    chain = ShutdownChain()
    ev = chain.evaluate()
    if hasattr(chain, "electrical_only_s"):
        elec = float(chain.electrical_only_s())
    else:
        elec = float(sum(s.delay_s for s in chain.stages[:2]))
    total = float(sum(s.delay_s for s in chain.stages))
    return {
        "electrical_s": elec, "expect_electrical_s": 1.0e-6,
        "electrical_pass": bool(elec < 1.0e-6),
        "total_s": total, "expect_total_s": 20.0e-3,
        "total_pass": bool(total < 20.0e-3),
        "all_stages_ok": bool(ev.get("all_stages_ok", False)),
    }


def _stop_angle() -> Dict[str, Any]:
    """HIL-STO-04: 停止角 (由关节速度真实反解)。"""
    chain = ShutdownChain()
    ang = float(chain.stop_angle_deg(joint_speed_rad_s=3.0))
    return {"measured_deg": ang, "expect_deg": 30.0, "pass": bool(ang <= 30.0)}


def _can_transport_probe() -> Dict[str, Any]:
    """HIL 测量通道: 用已对齐的 CAN-FD 承载遥测 (替代随机总线)。

    真实 HIL 中上述测量值经 CAN-FD 遥测帧回传。此处复用 R4 已对齐的
    encode_joint_cmd / decode_telemetry (600 帧 0 失配已验) 验证通道可用。
    """
    from ..simulation.real_robot_env import (
        CANFrame, decode_telemetry, encode_joint_cmd,
    )
    f = encode_joint_cmd(0, 180.0, 0.0, 20.0)
    dec = decode_telemetry(CANFrame(300, f.data))
    return {"can_roundtrip_ok": bool(dec["crc_ok"]),
            "decoded_pos_deg": float(dec["pos_deg"])}


@register_python_backend("hil")
def hil_backend(seed: int = 0) -> Dict[str, Any]:
    """G6: HIL 桥接后端 — 7 条用例, 真实模型判定, **无随机数**。

    返回 status / n_cases / n_pass / n_bench / cases / can_transport /
    uses_random(False = G6 已修复)。
    """
    cases: List[Dict[str, Any]] = []

    # --- 5 条: 有真实模型支撑, 可仿真判定 ---
    sv = _svpwm_balance(seed)
    cases.append({"id": "HIL-FOC-02", "source": "python_model",
                  "verdict": "PASS" if sv["pass"] else "FAIL", **sv})

    ce = _current_step_error(seed)
    cases.append({"id": "HIL-FOC-01", "source": "python_model",
                  "verdict": "PASS" if ce["pass"] else "FAIL", **ce})

    st = _shutdown_timing()
    cases.append({"id": "HIL-STO-01", "source": "python_model",
                  "verdict": "PASS" if st["electrical_pass"] else "FAIL",
                  "measured_s": st["electrical_s"],
                  "expect_s": st["expect_electrical_s"],
                  "pass": st["electrical_pass"]})
    cases.append({"id": "HIL-STO-03", "source": "python_model",
                  "verdict": "PASS" if st["total_pass"] else "FAIL",
                  "measured_s": st["total_s"],
                  "expect_s": st["expect_total_s"],
                  "pass": st["total_pass"]})

    sa = _stop_angle()
    cases.append({"id": "HIL-STO-04", "source": "python_model",
                  "verdict": "PASS" if sa["pass"] else "FAIL", **sa})

    # --- 2 条: 硬件架构属性, 需台架实测 (不用随机数冒充通过) ---
    for cid in sorted(REQUIRES_BENCH):
        cases.append({
            "id": cid, "source": "none", "verdict": "REQUIRES_BENCH",
            "pass": None,
            "reason": "硬件架构属性 (硬件级锁存/关断不依赖 MCU 软件); "
                      "软件仿真无法证明, 需台架实测后回填",
        })

    can = _can_transport_probe()
    n_pass = sum(1 for c in cases if c["verdict"] == "PASS")
    n_fail = sum(1 for c in cases if c["verdict"] == "FAIL")
    n_bench = sum(1 for c in cases if c["verdict"] == "REQUIRES_BENCH")

    return {
        "status": "OK",
        "n_cases": len(cases), "n_pass": n_pass, "n_fail": n_fail,
        "n_bench": n_bench,
        "cases": cases,
        "can_transport": can,
        # G6 修复的核心标志: 结果不再由随机数产生
        "uses_random": False,
    }
