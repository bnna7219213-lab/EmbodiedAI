#!/usr/bin/env python3
"""export_b1_package.py — B1 攀岩腱驱动关节板：导出板级文件包（原理图/PCB/BOM/Gerber）+ 打样清单。

架构定位 (双轨消费):
  · jar 轨: EDA 能力 (PcbExporter/SpiceNetlist/SvgRenderer/CDriverGen) 以
    vendor/edalab/*.jar 二进制形态提供 — 这些是计算密集 + 版图算法型能力,
    EdaLab 源码不入库, PROVENANCE.json 保障版本可追溯 + sha256 漂移治理。
  · python 轨: G6/G10/G7/G2 等 jar 冻结论不了的缺口 (HIL/安全认证/可靠性),
    在 src/embodied_ai/hardware/ 下原创实现, @register_python_backend() 注册,
    与 jar 走不同路由 — 这部分代码公开, 与 EdaLab 无关。

  此脚本走 jar 轨 (EDA 引擎)。运行时仅需 JRE, 无需 JDK/gradle/外部 EdaLab。
  单仓自足验证: python -m embodied_ai.cli selfcheck

流程:
  1. 从 hw_requirements.yaml climbing.motor_drive_tendon 段读取 B1 设计参数
  2. 构造完整电路 JSON (基于 MotorDriveEngine selectComponents 器件清单 + 腱驱动拓扑)
  3. 调用 EdaLab CLI 子命令:
     - pcb     → PCB/SVG/JSON/DXF/Gerber (自动布局 + DRC, jar 轨)
     - render  → 原理图 SVG (jar 轨)
     - spice   → SPICE 网表 (.cir, jar 轨)
     - cgen    → C 驱动骨架 (jar 轨)
  4. 生成 BOM CSV (位号/型号/封装/数量/替代料)
  5. 生成打样清单 HTML (板规格 + 文件清单 + 注意事项)

用法:
  python scripts/export_b1_package.py [--out-dir path]
"""

from __future__ import annotations

import csv
import html
import json
import os
import string
import subprocess
import sys
import time
from pathlib import Path

# ---- 路径常量 ----
REPO = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO / "vendor" / "edalab"
EDALAB_JAR = VENDOR_DIR / "edalabcore.jar"
KOTLIN_STDLIB = VENDOR_DIR / "kotlin-stdlib-2.2.21.jar"
CONTRACTS = REPO / "contracts" / "hw_requirements.yaml"
DEFAULT_OUT = REPO / "artifacts" / "b1_tendon_drive_package"
EDALAB_MAIN = "edalab.core.cli.EdaCli"


def build_classpath() -> str:
    """构造 EdaLab jar 的classpath — 双轨架构中的 jar 轨。

    定位: 本项目消费 EDA 引擎 (PcbExporter/SpiceNetlist/SvgRenderer/CDriverGen)
    通过 vendor/edalab/*.jar 编译产物。这些 EDA 能力是计算密集型 + 版图算法型,
    以二进制jar 形态提供 (EdaLab 源码不入库, PROVENANCE.json 保障版本可追溯)。

    双轨的 Python 侧 (G6/G10/G7/G2 等 jar 冻结论不了的缺口, 如 HIL/安全认证/)
    在 src/embodied_ai/hardware/ 下原创实现, 经 @register_python_backend() 注册,
    与 jar 走不同路径 — 本函数仅构造 jar 轨的 classpath。

    单仓自足: clone + JRE 即可, 无需外部 EdaLab 目录/gradle/JDK。
    验证: python -m embodied_ai.cli selfcheck
    """
    parts = [str(EDALAB_JAR).replace("\\", "/")]
    if KOTLIN_STDLIB.exists():
        parts.append(str(KOTLIN_STDLIB).replace("\\", "/"))

    return ";".join(parts)


def run_edalab(cmd: list[str], cwd: Path = REPO) -> subprocess.CompletedProcess:
    """通过 vendor jar 双轨调用 EdaLab EDA 引擎。

    双轨架构:
      · jar 轨 (本路径): EdaCli 全局注册表中的 EDA 子命令
        (pcb / spice / render / cgen) — 算法密集 + 版图约束, 由 Kotlin 实现。
      · python 轨: @register_python_backend() 注册的 G6/G10/G7/G2 缺口
        (如 hil_bridge / safety/certification) — EmbodiedAI 原创, 正常入库。

    定位: 此函数实现"消费侧的物理层"—— python 字节流
      → java -cp vendor/edalab/* EdaCli <subcommand> → stdout/stderr
    不区分具体能力归属 (jar or python), 由上层按 backend 路由。

    JVM: -Dfile.encoding/-Dstdout.encoding/-Dstderr.encoding=UTF-8 (JDK17+),
    老JDK忽略, 由 _decode_output 兜底。
    cwd=REPO: 输出一律绝对路径, 不依赖外部环境。
    """
    cp = build_classpath()
    java = _resolve_java()
    full_cmd = [java,
                "-Dfile.encoding=UTF-8",
                "-Dstdout.encoding=UTF-8",
                "-Dstderr.encoding=UTF-8",
                "-cp", cp, EDALAB_MAIN] + cmd
    result = subprocess.run(full_cmd, capture_output=True, cwd=str(cwd), timeout=120)
    # 手动解码 — 传 text=True 會讓 Python 猜編碼, 在 Windows 中文環境下 GBK 會誤判
    result.stdout = _decode_output(result.stdout)
    result.stderr = _decode_output(result.stderr)
    return result


def _decode_output(data: bytes) -> str:
    """Windows 中文环境 JVM 输出解码: UTF-8 → GBK/GB18030 回退链。"""
    if not data:
        return ""
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            text = data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        if "�" not in text:
            return text
    return data.decode("utf-8", errors="replace")


def _resolve_java() -> str:
    """解析 java 路径: PATH → JAVA_HOME → EMBODIED_JAVA → 报错。"""
    import shutil
    which = shutil.which("java")
    if which:
        return which
    jh = os.environ.get("JAVA_HOME")
    if jh:
        cand = Path(jh) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if cand.exists():
            return str(cand)
    fb = os.environ.get("EMBODIED_JAVA")
    if fb and Path(fb).exists():
        return str(Path(fb))
    raise FileNotFoundError(
        "java 不可用: 请安装 JRE 或设置 JAVA_HOME / EMBODIED_JAVA "
        "(单仓自足仅需 JRE, 无需 JDK/gradle)")


# ============================================================================
# 1. 电路 JSON 构建
# ============================================================================

def make_pin(pid: str, name: str, dx: float, dy: float) -> dict:
    return {"id": pid, "name": name, "dx": dx, "dy": dy}


def make_component(instance_id: str, ref_des: str, comp_type: str,
                   name: str, value: float, unit: str, footprint: str,
                   pins: list[dict], lib_id: str = "", x: float = 0, y: float = 0,
                   description: str = "", rating: float = 0) -> dict:
    return {
        "instanceId": instance_id,
        "refDes": ref_des,
        "x": x,
        "y": y,
        "rotation": 0,
        "spec": {
            "id": lib_id or f"lib_custom_{ref_des.lower()}",
            "type": comp_type,
            "name": name,
            "value": value,
            "unit": unit,
            "rating": rating,
            "footprint": footprint,
            "description": description,
            "custom": True,
            "params": {},
            "pins": pins,
        },
        "subcircuitId": "",
        "portIndex": -1,
    }


def make_wire(wire_id: str, from_comp: str, from_pin: str,
              to_comp: str, to_pin: str) -> dict:
    return {
        "id": wire_id,
        "fromComp": from_comp,
        "fromPin": from_pin,
        "toComp": to_comp,
        "toPin": to_pin,
    }


def build_b1_circuit() -> dict:
    """构建 B1 攀岩腱驱动关节板电路 JSON。

    器件清单对齐 MotorDriveEngine::createTendonDriveSpec + selectComponents:
      · 24V 桥臂 / 19.2A 峰值 / 460.8W / 4 层 / FOC_SENSORED / 双编码器
    """
    comps: list[dict] = []
    wires: list[dict] = []
    inst = 0
    y_pos = 60

    def nxt_y(step: float = 80) -> float:
        nonlocal y_pos
        y_pos += step
        return y_pos

    def add_comp(ref_des: str, comp_type: str, name: str, value: float,
                 unit: str, footprint: str, pins: list[dict],
                 x: float = 60, desc: str = "", rating: float = 0) -> str:
        nonlocal inst
        inst += 1
        iid = f"inst_{inst}"
        comps.append(make_component(iid, ref_des, comp_type, name, value,
                                    unit, footprint, pins, x=x, y=nxt_y(0),
                                    description=desc, rating=rating))
        return iid

    # --- 电源输入连接器 ---
    j_pow = add_comp("J_PWR", "CONNECTOR", "电源输入 24V", 24, "V", "TE_MATELOK_2P",
                     [make_pin("pin1", "24V", -30, 0), make_pin("pin2", "GND", 30, 0)],
                     x=60, desc="24V 桥臂供电输入 (>=19.2A)", rating=24)

    # --- 24V 输入滤波电容 (bulk) ---
    c_bulk1 = add_comp("CBULK1", "CAPACITOR", "电解电容 470µF/35V", 4.7e-4, "F", "RADIAL_10x16",
                       [make_pin("a", "+", -30, 0), make_pin("b", "-", 30, 0)],
                       desc="24V 输入储能", rating=35)
    c_bulk2 = add_comp("CBULK2", "CAPACITOR", "MLCC 100nF/50V", 1e-7, "F", "0805",
                       [make_pin("a", "", -30, 0), make_pin("b", "", 30, 0)],
                       desc="24V 高频去耦", rating=50)

    # --- 栅极驱动 U2 (DRV8301) ---
    u_gate = add_comp("U_GATE", "IC", "栅极驱动 DRV8301", 0, "", "QFN48",
                      [make_pin("gh_a", "", -40, -20), make_pin("gl_a", "", -40, -10),
                       make_pin("gh_b", "", -40, 0), make_pin("gl_b", "", -40, 10),
                       make_pin("gh_c", "", -40, 20), make_pin("gl_c", "", -40, 30),
                       make_pin("inha", "", 40, -20), make_pin("inha", "", 40, -10),
                       make_pin("inhb", "", 40, 0), make_pin("inhb_l", "", 40, 10),
                       make_pin("inhc", "", 40, 20), make_pin("inhc_l", "", 40, 30),
                       make_pin("vdd", "", 0, -30), make_pin("gnd", "", 0, 30),
                       make_pin("so1", "", -40, 40), make_pin("so2", "", -40, 50),
                       make_pin("en_gate", "", 40, 40), make_pin("dc_cal", "", 40, 50)],
                      desc="60V 半桥栅极驱动 2.3A", rating=60)

    # --- MOSFET 三相桥 Q1-Q6 ---
    q1 = add_comp("Q1", "MOSFET", "MOSFET IRFS4410Z (A相上)", 0, "", "TO-220-7P",
                  [make_pin("g", "", -30, -10), make_pin("d", "", -30, 0),
                   make_pin("s", "", -30, 10), make_pin("d2", "", 30, 0)],
                  desc="100V/80A N-MOSFET A相上桥臂", rating=100)
    q2 = add_comp("Q2", "MOSFET", "MOSFET IRFS4410Z (A相下)", 0, "", "TO-220-7P",
                  [make_pin("g", "", -30, -10), make_pin("d", "", -30, 0),
                   make_pin("s", "", -30, 10), make_pin("d2", "", 30, 0)],
                  desc="100V/80A N-MOSFET A相下桥臂", rating=100)
    q3 = add_comp("Q3", "MOSFET", "MOSFET IRFS4410Z (B相上)", 0, "", "TO-220-7P",
                  [make_pin("g", "", -30, -10), make_pin("d", "", -30, 0),
                   make_pin("s", "", -30, 10), make_pin("d2", "", 30, 0)],
                  desc="100V/80A N-MOSFET B相上桥臂", rating=100)
    q4 = add_comp("Q4", "MOSFET", "MOSFET IRFS4410Z (B相下)", 0, "", "TO-220-7P",
                  [make_pin("g", "", -30, -10), make_pin("d", "", -30, 0),
                   make_pin("s", "", -30, 10), make_pin("d2", "", 30, 0)],
                  desc="100V/80A N-MOSFET B相下桥臂", rating=100)
    q5 = add_comp("Q5", "MOSFET", "MOSFET IRFS4410Z (C相上)", 0, "", "TO-220-7P",
                  [make_pin("g", "", -30, -10), make_pin("d", "", -30, 0),
                   make_pin("s", "", -30, 10), make_pin("d2", "", 30, 0)],
                  desc="100V/80A N-MOSFET C相上桥臂", rating=100)
    q6 = add_comp("Q6", "MOSFET", "MOSFET IRFS4410Z (C相下)", 0, "", "TO-220-7P",
                  [make_pin("g", "", -30, -10), make_pin("d", "", -30, 0),
                   make_pin("s", "", -30, 10), make_pin("d2", "", 30, 0)],
                  desc="100V/80A N-MOSFET C相下桥臂", rating=100)

    # --- 电流检测 U3 (INA240) x3 ---
    u_cs1 = add_comp("U_CS1", "IC", "电流检测 INA240A (A相)", 0, "", "MSOP10",
                     [make_pin("in+", "", -30, -10), make_pin("in-", "", -30, 10),
                      make_pin("out", "", 30, 0), make_pin("vcc", "", 0, -20),
                      make_pin("gnd", "", 0, 20)], desc="双向电流检测 A相")
    u_cs2 = add_comp("U_CS2", "IC", "电流检测 INA240A (B相)", 0, "", "MSOP10",
                     [make_pin("in+", "", -30, -10), make_pin("in-", "", -30, 10),
                      make_pin("out", "", 30, 0), make_pin("vcc", "", 0, -20),
                      make_pin("gnd", "", 0, 20)], desc="双向电流检测 B相")

    # --- 主控 U1 (STM32G474VET6 LQFP100) ---
    u_mcu = add_comp("U_MCU", "IC", "主控 STM32G474VET6", 0, "", "LQFP100",
                     [make_pin("pa0", "", -50, -30), make_pin("pa1", "", -50, -20),
                      make_pin("pa2", "", -50, -10), make_pin("pa3", "", -50, 0),
                      make_pin("pa4", "", -50, 10), make_pin("pa5", "", -50, 20),
                      make_pin("pa6", "", -50, 30), make_pin("pa7", "", -50, 40),
                      make_pin("pb6", "", 50, -30), make_pin("pb7", "", 50, -20),
                      make_pin("pb8_can_rx", "", 50, -10), make_pin("pb9_can_tx", "", 50, 0),
                      make_pin("vdd", "", 0, -40), make_pin("vss", "", 0, 40),
                      make_pin("nrst", "", 0, -50), make_pin("boot0", "", 0, 50)],
                     desc="Cortex-M4 170MHz FOC 电机控制 MCU", rating=3.3)

    # --- CAN-FD 收发器 U_CAN (TCAN1043A-Q1) ---
    u_can = add_comp("U_CAN", "IC", "CAN-FD 收发器 TCAN1043A-Q1", 0, "", "SOIC14",
                     [make_pin("canh", "", -30, -15), make_pin("canl", "", -30, 15),
                      make_pin("txd", "", 30, -15), make_pin("rxd", "", 30, 15),
                      make_pin("vcc", "", 0, -25), make_pin("gnd", "", 0, 25)],
                     desc="CAN-FD 5Mbps 车规收发器")

    # --- 编码器 U4 (AS5048A) ---
    u_enc = add_comp("U_ENC", "IC", "磁性编码器 AS5048A", 0, "", "TSSOP14",
                     [make_pin("mosi", "", -30, -20), make_pin("miso", "", -30, -10),
                      make_pin("sclk", "", -30, 0), make_pin("csn", "", -30, 10),
                      make_pin("vdd", "", 30, -15), make_pin("gnd", "", 30, 15),
                      make_pin("a", "", -30, 20), make_pin("b", "", -30, 30)],
                     desc="14-bit 绝对位置磁编码器", rating=3.3)

    # --- 腱张力检测 U_LC (AD8422 仪表放大器 ×8 腱组通道) ---
    for ch in range(8):
        add_comp(f"U_LC{ch}",
                 "IC", f"仪表放大器 AD8422 (张力检测 CH{ch})", 0, "", "SOIC8",
                 [make_pin("in+", "", -30, -10), make_pin("in-", "", -30, 10),
                  make_pin("out", "", 30, 0), make_pin("vcc", "", 0, -20),
                  make_pin("gnd", "", 0, 20), make_pin("ref", "", 30, 20)],
                 desc=f"腱张力应变片仪表放大 CH{ch}")

    # --- 电机相输出连接器 ---
    j_mot = add_comp("J_MOT", "CONNECTOR", "电机相 UVW 输出", 0, "", "JST_B3P-VH",
                     [make_pin("u", "U", -20, -15), make_pin("v", "V", -20, 0),
                      make_pin("w", "W", -20, 15)],
                     desc="电机三相 UVW 输出 (绑腱轮)")

    # --- CAN 连接器 ---
    j_can = add_comp("J_CAN", "CONNECTOR", "CAN-FD 总线", 0, "", "JST_B4P-VH",
                     [make_pin("canh", "", -20, -15), make_pin("canl", "", -20, 0),
                      make_pin("vcc", "", -20, 15), make_pin("gnd", "", -20, 30)],
                     desc="CAN-FD 5Mbps 总线接口")

    # --- 编码器连接器 ---
    j_enc = add_comp("J_ENC", "CONNECTOR", "编码器接口 (电机/腱双端)", 0, "", "JST_B6P-VH",
                     [make_pin("a", "", -20, -25), make_pin("b", "", -20, -15),
                      make_pin("z", "", -20, -5), make_pin("vcc", "", -20, 5),
                      make_pin("gnd", "", -20, 15), make_pin("data", "", -20, 25)],
                     desc="19-bit 双端绝对编码器接口")

    # --- GND ---
    gnd = add_comp("GND1", "GROUND", "系统地", 0, "V", "-",
                   [make_pin("gnd", "GND", 0, 0)], desc="系统地平面")

    # --- 关键导线 ---
    wire_n = [0]
    def w(fc, fp, tc, tp):
        wire_n[0] += 1
        wires.append(make_wire(f"wire_{wire_n[0]}", fc, fp, tc, tp))

    # 24V 网
    w(j_pow, "pin1", c_bulk1, "a")
    w(j_pow, "pin1", c_bulk2, "a")
    w(c_bulk1, "a", q1, "d")
    w(c_bulk1, "a", q3, "d")
    w(c_bulk1, "a", q5, "d")
    # 栅极驱动到MOSFET
    w(u_gate, "gh_a", q1, "g")
    w(u_gate, "gl_a", q2, "g")
    w(u_gate, "gh_b", q3, "g")
    w(u_gate, "gl_b", q4, "g")
    w(u_gate, "gh_c", q5, "g")
    w(u_gate, "gl_c", q6, "g")
    # 电机相 (A/B/C输出点)
    w(q1, "s", q2, "d"); w(q1, "s", j_mot, "u")
    w(q3, "s", q4, "d"); w(q3, "s", j_mot, "v")
    w(q5, "s", q6, "d"); w(q5, "s", j_mot, "w")
    # 电流检测
    w(q2, "s", u_cs1, "in+")
    w(q4, "s", u_cs2, "in+")
    # MCU到栅极驱动
    w(u_mcu, "pa6", u_gate, "inha"); w(u_mcu, "pa7", u_gate, "inha")
    w(u_mcu, "pa0", u_gate, "inhb"); w(u_mcu, "pa1", u_gate, "inhb_l")
    w(u_mcu, "pa2", u_gate, "inhc"); w(u_mcu, "pa3", u_gate, "inhc_l")
    # MCU到CAN
    w(u_mcu, "pb8_can_rx", u_can, "rxd")
    w(u_mcu, "pb9_can_tx", u_can, "txd")
    # CAN 到总线
    w(u_can, "canh", j_can, "canh")
    w(u_can, "canl", j_can, "canl")
    # MCU到编码器
    w(u_mcu, "pa4", u_enc, "csn")
    w(u_mcu, "pa5", u_enc, "sclk")
    w(u_enc, "a", j_enc, "a")
    w(u_enc, "b", j_enc, "b")
    # 电流检测输出到MCU ADC
    w(u_cs1, "out", u_mcu, "pa0")
    w(u_cs2, "out", u_mcu, "pa1")
    # GND — gnd 变量即为 GND1 的 instanceId
    gnd_map = {
        j_pow: "pin2", c_bulk1: "b", c_bulk2: "b", u_gate: "gnd",
        u_cs1: "gnd", u_cs2: "gnd", u_mcu: "vss", u_can: "gnd",
        u_enc: "gnd", j_can: "gnd", j_enc: "gnd",
    }
    for comp_iid, pin_name in gnd_map.items():
        w(comp_iid, pin_name, gnd, "gnd")

    return {
        "version": 2,
        "name": "B1 攀岩腱驱动关节板 (TendonDrive-Climbing)",
        "timestamp": int(time.time() * 1000),
        "components": comps,
        "wires": wires,
    }


# ============================================================================
# 2. BOM 生成
# ============================================================================

def generate_bom_csv(out_dir: Path) -> Path:
    """生成 BOM CSV (位号/型号/封装/数量/描述/替代料)。"""
    rows = [
        ["位号", "型号", "封装", "数量", "描述", "替代料"],
        ["U_MCU", "STM32G474VET6", "LQFP100", "1", "主控 Cortex-M4 170MHz FOC MCU", "STM32H743 / RA8M1G"],
        ["U_GATE", "DRV8301", "QFN48", "1", "60V 半桥栅极驱动 2.3A", "DRV8353RS / L6235"],
        ["Q1~Q6", "IRFS4410Z", "TO-220-7P", "6", "100V/80A N-MOSFET 三相桥臂", "IRFS7530 / AUIRFS8409"],
        ["U_CS1, U_CS2", "INA240AIDGSR", "MSOP10", "2", "双向电流检测放大器", "INA282 / MAX480"],
        ["U_CAN", "TCAN1043A-Q1", "SOIC14", "1", "CAN-FD 5Mbps 车规收发器", "MCP2562FD / TJA1043"],
        ["U_ENC", "AS5048A", "TSSOP14", "1", "14-bit 绝对磁编码器", "TLI5012B / AS5047P"],
        ["U_LC0~U_LC7", "AD8422", "SOIC8", "8", "仪表放大器 (腱应变片放大)", "AD620 / INA333"],
        ["CBULK1", "470µF/35V 电解", "RADIAL_10x16", "1", "24V 输入储能", "Nichicon UHE"],
        ["CBULK2", "100nF/50V MLCC", "0805", "1", "24V 高频去耦", "GRM21BR71H104KA01L"],
        ["J_PWR", "MATELOK 2P", "TE_MATELOK_2P", "1", "24V 电源输入 (>=19.2a)", "XT60 / JST_VH"],
        ["J_MOT", "JST B3P-VH", "JST_B3P-VH", "1", "电机 UVW 输出", "JST_B3P-VH"],
        ["J_CAN", "JST B4P-VH", "JST_B4P-VH", "1", "CAN-FD 总线", "JST_B4P-VH"],
        ["J_ENC", "JST B6P-VH", "JST_B6P-VH", "1", "编码器接口 (电机/腱双端)", "JST_B6P-VH"],
    ]
    csv_path = out_dir / "B1_tendon_drive_bom.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return csv_path


# ============================================================================
# 3. 打样清单 HTML
# ============================================================================

def generate_prototyping_list(out_dir: Path, file_list: list[str]) -> Path:
    """生成打样清单 HTML (使用 string.Template 避免 f-string CSS 冲突)."""
    html_path = out_dir / "B1_prototyping_list.html"

    rows = "".join(
        '<tr><td><code>%s</code></td><td>✅ 已生成</td></tr>' % html.escape(f)
        for f in file_list
    )

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    tpl = string.Template("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>B1 打样清单 — 攀岩腱驱动关节板</title>
<style>
body { font-family: 'Microsoft YaHei','SimHei',sans-serif; max-width:960px; margin:0 auto; padding:20px; color:#222; }
h1 { color:#1a5276; border-bottom:3px solid #1a5276; padding-bottom:10px; }
h2 { color:#2c3e50; margin-top:30px; }
table { border-collapse:collapse; width:100%; margin:10px 0; }
th, td { border:1px solid #ccc; padding:8px 12px; text-align:left; }
th { background:#1a5276; color:white; }
tr:nth-child(even) { background:#f8f9fa; }
.spec-grid { display:grid; grid-template-columns:200px 1fr; gap:6px; margin:10px 0; }
.spec-grid b { background:#eaf2f8; padding:4px 10px; }
.spec-grid span { padding:4px 10px; }
.warning { background:#fdf2e9; border-left:4px solid #e67e22; padding:10px 16px; margin:10px 0; }
.critical { background:#fdedec; border-left:4px solid #e74c3c; padding:10px 16px; margin:10px 0; }
code { background:#f0f0f0; padding:1px 5px; border-radius:3px; font-size:13px; }
ol li, ul li { margin:4px 0; }
</style>
</head>
<body>
<h1>🧗 B1 打样清单 — 攀岩腱驱动关节板</h1>
<p><strong>文档版本</strong>: v1.0 | <strong>生成时间</strong>: $timestamp | <strong>来源</strong>: hw_requirements.yaml climbing.motor_drive_tendon</p>

<div class="critical">
<strong>⚠ 阶段声明</strong>: 本文件包由参数化设计引擎自动生成，用于<strong>仿真验证</strong>。
打样前必须由 Layout 工程师完成真实 PCB Layout (Altium/KiCad)，
<strong>不可直接用自动布局文件投板生产</strong>。
</div>

<h2>一、板规格参数</h2>
<div class="spec-grid">
<b>板名</b><span>B1 — 攀岩腱驱动关节板 (TendonDrive-Climbing)</span>
<b>产品系列</b><span>EmbodiedAI SynapticForge</span>
<b>板种</b><span>tendon_drive_joint_board (区别于整机母线板)</span>
<b>桥臂电压</b><span>24V (48V 母线经通道级 buck)</span>
<b>桥臂峰值电流</b><span>19.2A (= 8ch × 4A × 0.6 同时率)</span>
<b>桥臂峰值功率</b><span>460.8W (24V × 19.2A)</span>
<b>母线峰值</b><span>30A (含裕量)</span>
<b>层数</b><span>4 层 (2oz铜厚)</span>
<b>PCB 尺寸</b><span>~55.4mm × 44.3mm (自动布局估算)</span>
<b>最小线宽/间距</b><span>0.25mm / 0.2mm</span>
<b>爬电距离</b><span>3.0mm (30V+)</span>
<b>基板厚度</b><span>1.6mm</span>
<b>板面积</b><span>≤ 2500 mm²</span>
<b>防护等级</b><span>IP54 (墙面汗/尘环境)</span>
<b>工作温度</b><span>-10 ~ +60°C</span>
<b>最大 PCB 加速度</b><span>20g (dyno 捕获冲击上限)</span>
</div>

<h2>二、关键器件清单</h2>
<table>
<tr><th>位号</th><th>型号</th><th>封装</th><th>功能</th><th>特殊要求</th></tr>
<tr><td>U_MCU</td><td>STM32G474VET6</td><td>LQFP100</td><td>主控 MCU (Cortex-M4 170MHz)</td><td>FPU + Cordic + FMAC</td></tr>
<tr><td>U_GATE</td><td>DRV8301</td><td>QFN48</td><td>栅极驱动 半桥 ×3</td><td>60V / 2.3A 驱动</td></tr>
<tr><td>Q1~Q6</td><td>IRFS4410Z</td><td>TO-220-7P</td><td>N-MOSFET 上下桥臂 ×3相</td><td>100V/80A 降额选用</td></tr>
<tr><td>U_CS1/2</td><td>INA240AIDGSR</td><td>MSOP10</td><td>双向电流检测</td><td>PWM 抑制 80V 共模</td></tr>
<tr><td>U_CAN</td><td>TCAN1043A-Q1</td><td>SOIC14</td><td>CAN-FD 收发器</td><td>车规 Grade 0</td></tr>
<tr><td>U_ENC</td><td>AS5048A</td><td>TSSOP14</td><td>磁性编码器</td><td>14-bit 绝对值</td></tr>
<tr><td>U_LC0~7</td><td>AD8422</td><td>SOIC8</td><td>腱张力仪表放大 ×8</td><td>低噪声 高 CMRR</td></tr>
</table>

<div class="warning">
<strong>⚠ 已知的 Gaps / 打样前必查</strong>:
<ul>
<li><strong>编码器位数不一致</strong>: 契约要求 19-bit, 元件库 AS5048A 仅 14-bit — 需换 TLI5012B</li>
<li><strong>热裕量</strong>: 单桥模型估算 51.1°C, 实际需 SiC 模型 + 双脉冲测试修正</li>
<li><strong>无真实 Layout</strong>: 自动布局器生成, 未做 SI/PI/EMC 仿真</li>
<li><strong>腱张力检测未仿真</strong>: 8ch AD8422 走线未做噪声/屏蔽仿真</li>
</ul>
</div>

<h2>三、控制参数</h2>
<div class="spec-grid">
<b>控制模式</b><span>FOC_SENSORED (磁场定向控制)</span>
<b>PWM 载波频率</b><span>20kHz</span>
<b>电流环频率</b><span>10kHz (100µs 周期)</span>
<b>目标下发频率</b><span>200Hz (5ms 周期)</span>
<b>编码器类型</b><span>absolute_19bit (电机端 + 腱端双编码器)</span>
<b>编码器通道</b><span>16 (8 伺服电机端 + 8 腱端)</span>
<b>总线</b><span>CAN-FD 5Mbps, 8 节点</span>
<b>硬件过流</b><span>< 1μs 独立比较器 (8ch)</span>
<b>过温关断</b><span>105°C</span>
<b>看门狗通信丢失</b><span>50ms</span>
<b>停止类别</b><span>CAT_0_IMMEDIATE (立即断电)</span>
<b>跌落保护</b><span>true (握持失效即断电锁止)</span>
</div>

<h2>四、制造文件清单</h2>
<table>
<tr><th>文件名</th><th>状态</th><th>说明</th></tr>
$rows
</table>

<h2>五、推荐打样流程</h2>
<ol>
<li><strong>Layout 工程化</strong> — Altium Designer / KiCad 真实 PCB Layout (4层/2oz/阻抗控制)</li>
<li><strong>SI/PI/EMC 仿真</strong> — 信号完整性、电源完整性、EMC 仿真</li>
<li><strong>设计评审</strong> — 对照清单检查车规级 CAN 收发器等特殊器件</li>
<li><strong>投板打样</strong> — 嘉立创/华秋 (4层 ~50元/5片, 2oz铜厚)</li>
<li><strong>钢网</strong> — QFN48 (U_GATE) 必须开刻钢网</li>
<li><strong>SMT 贴片</strong> — TO-220-7P (MOSFET) 手工焊, 其他回流焊</li>
<li><strong>FCT/ICT 测试</strong> — 100% ICT 飞针 + FCT 功能测试 (固件自动从 foc-firmware 生成)</li>
<li><strong>实机装配</strong> — 腱伺服电机 ×4 + 拮抗 VSA 电机 ×4 + 8ch 张力传感器</li>
<li><strong>HIL 联调</strong> — 接入 HIL 半实物仿真, 验证 FOC 电流环 / PWM / 编码器反馈</li>
</ol>

<h2>六、认证要求</h2>
<ul>
<li>功能安全: ISO 13482 个人护理机器人, <strong>SIL 2</strong></li>
<li>安全功能: STO (Safe Torque Off) < 1μs 硬件独立比较器</li>
<li>爬电/电气间隙: 3.0mm (30V+)</li>
<li>腱驱动安全: 握持失效 → 断电锁止 (drop_protection = true)</li>
</ul>

</body>
</html>""")

    content = tpl.substitute(timestamp=timestamp, rows=rows)
    html_path.write_text(content, encoding="utf-8")
    return html_path


# ============================================================================
# 4. 主流程
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="B1 腱驱动关节板文件包导出")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help=f"输出目录 (默认: {DEFAULT_OUT})")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("B1 攀岩腱驱动关节板 — 文件包导出")
    print(f"输出目录: {out_dir}")
    print("=" * 60)

    # 1. 电路 JSON
    circuit = build_b1_circuit()
    circuit_path = out_dir / "B1_tendon_drive.circuit.json"
    circuit_path.write_text(json.dumps(circuit, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"✅ 电路 JSON  ({circuit_path.name}): "
          f'{len(circuit["components"])} 元件, {len(circuit["wires"])} 导线')

    # 2. EdaLab CLI 导出
    generated_files = [circuit_path.name]

    # PCB (SVG + Gerber + DXF + PCB JSON)
    print("\n▶ 运行 EdaLab PCB 自动布局...")
    result = run_edalab(["pcb", "-i", str(circuit_path), "-o",
                         str(out_dir / "B1_tendon_drive_pcb"), "--format", "all"])
    if result.returncode == 0:
        for ext in ["svg", "pcb.json", "dxf"]:
            fp = out_dir / f"B1_tendon_drive_pcb.{ext}"
            if fp.exists():
                generated_files.append(fp.name)
        # Gerber 文件
        for gbr in sorted(out_dir.glob("B1_tendon_drive_pcb.*.gbr")):
            generated_files.append(gbr.name)
        print(f"   PCB 输出: {'  '.join(result.stdout.strip().splitlines()[-2:])}")
    else:
        print(f"   ⚠ PCB 失败: {result.stderr[:200]}")

    # Schematic SVG
    print("\n▶ 运行 EdaLab 原理图渲染...")
    result = run_edalab(["render", "-i", str(circuit_path),
                         "-o", str(out_dir / "B1_tendon_drive_schematic.svg")])
    if result.returncode == 0:
        generated_files.append("B1_tendon_drive_schematic.svg")
        print("   ✅ 原理图 SVG 已生成")
    else:
        print(f"   ⚠ 渲染失败: {result.stderr[:200]}")

    # SPICE
    print("\n▶ 运行 EdaLab SPICE 网表导出...")
    result = run_edalab(["spice", "-i", str(circuit_path),
                         "-o", str(out_dir / "B1_tendon_drive")])
    if result.returncode == 0:
        generated_files.append("B1_tendon_drive.cir")
        print("   ✅ SPICE 网表 已生成")
    else:
        print(f"   ⚠ SPICE 失败: {result.stderr[:200]}")

    # C 驱动
    print("\n▶ 运行 EdaLab C 驱动生成...")
    result = run_edalab(["cgen", "-i", str(circuit_path),
                         "-o", str(out_dir / "B1_tendon_drive.c")])
    if result.returncode == 0:
        generated_files.append("B1_tendon_drive.c")
        print("   ✅ C 驱动 已生成")
    else:
        print(f"   ⚠ C 驱动失败: {result.stderr[:200]}")

    # 3. BOM CSV
    print("\n▶ 生成 BOM CSV...")
    bom_path = generate_bom_csv(out_dir)
    generated_files.append(bom_path.name)
    print(f"   {bom_path.name}")

    # 4. 打样清单 HTML
    print("\n▶ 生成打样清单 HTML...")
    list_path = generate_prototyping_list(out_dir, generated_files)
    generated_files.append(list_path.name)
    print(f"   {list_path.name}")

    # 汇总
    print("\n" + "=" * 60)
    print("✅ 全部文件已生成:")
    for f in generated_files:
        fp = out_dir / f
        size = fp.stat().st_size if fp.exists() else 0
        print(f"   ({size:>8} bytes)  {f}")
    print("=" * 60)
    print(f"\n总计 {len(generated_files)} 个文件 → {out_dir}")
    print("⚠  打样前必须由 Layout 工程师完成真实 PCB 设计")


if __name__ == "__main__":
    main()
