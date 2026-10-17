#!/usr/bin/env python3
"""b1_fab_inquiry.py — B1 板级文件包 → 工厂打样询价单 (RFQ) 整理导出。

**用途**
    把 `artifacts/b1_tendon_drive_package/` 的板级文件包 (Gerber/BOM/装配图)
    整理成**可直接发给板厂的打样询价单**: 板规格 + 文件清单 + BOM 摘要 +
    询价问题清单 + 交付要求。工厂按此报价即可横向比价。

**信息来源 (单一真源)**
    · 板规格     → contracts/hw_requirements.yaml (climbing.motor_drive_tendon + mechanical)
    · BOM       → artifacts/b1_tendon_drive_package/B1_tendon_drive_bom.csv
    · 文件清单   → artifacts/b1_tendon_drive_package/ 实际文件扫描 (Gerber 层名自动识别)
    · 工艺要求   → docs/16-manufacturing-guide.md 打样交付清单

**诚实边界 (询价单内保留, 不删)**
    · Gerber 由参数化设计引擎自动生成 —— 打样前必须 Layout 工程师评审
    · encoder_type absolute_19bit 与 BOM AS5048A(14bit) 不一致 (known_gaps)
    · 钢网 Gerber 由板厂自动生成
    provenance: 板规格=契约(hw_requirements); BOM=设计产出; 工艺=文档约定

用法:
  python scripts/b1_fab_inquiry.py                  # 生成询价单到包目录
  python scripts/b1_fab_inquiry.py --out-dir path
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "artifacts" / "b1_tendon_drive_package"
CONTRACT = REPO / "contracts" / "hw_requirements.yaml"

#: Gerber 层文件名 → 中文层名 (工厂沟通口径)
GERBER_LAYERS = {
    "board_outline.gbr": "板框 (Outline)",
    "top_copper.gbr": "顶层铜 (Top Copper)",
    "bottom_copper.gbr": "底层铜 (Bottom Copper)",
    "copper_pour.gbr": "内层铺铜 (Copper Pour)",
    "top_silk.gbr": "顶层丝印 (Top Silkscreen)",
}

#: 询价必问问题清单 (工厂回填)
QUOTE_QUESTIONS = [
    "PCB 光板打样单价 (5 / 10 / 20 片阶梯价)? 交期多少个工作日?",
    "SMT 贴片费用如何计 (按焊点 / 按器件)? 双面贴是否加价?",
    "钢网 (0.12mm 厚, 见工艺要求) 是否随板免费提供? 单独订费用?",
    "4 层 2oz 铜厚 + 板厚 1.6mm 是否标准工艺? 有无加价?",
    "最小起订量 (MOQ) 与工程费 (Gerber 审核/工程确认) 各多少?",
    "能否代采 BOM 器件? 缺料时替代料审批流程? (BOM 已列替代料)",
    "QFN48 (DRV8301) 与 LQFP100 (STM32G474) 的贴装良率保障与 X-Ray 检查是否含?",
    "阻抗控制 / 阻焊颜色 / 表面处理 (建议沉金 ENIG?) 选项与价差?",
    "飞针测试 (E-Test) 是否 100%? 测试报告格式?",
    "打样交付物: 光板+贴片成品 还是 分开报价? 交付周期分别是?",
]


def _load_specs() -> dict:
    """从 hw_requirements.yaml 提取 B1 板规格 (纯文本解析, 不引 yaml 依赖)。"""
    text = CONTRACT.read_text(encoding="utf-8")
    specs: dict = {}

    def _grab(pattern: str, cast=float, default=None):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default

    def _grab_str(pattern: str, default: str) -> str:
        m = re.search(pattern, text)
        return m.group(1) if m else default

    specs["bus_voltage_v"] = _grab(r"bus_voltage_v:\s*([\d.]+)")
    specs["motor_rail_v"] = _grab(r"motor_rail_v:\s*([\d.]+)")
    specs["servo_channels"] = _grab(r"servo_channels:\s*(\d+)", int)
    specs["bridge_peak_current_a"] = _grab(r"bridge_peak_current_a:\s*([\d.]+)")
    specs["bridge_peak_power_w"] = _grab(r"bridge_peak_power_w:\s*([\d.]+)")
    specs["pwm_carrier_hz"] = _grab(r"pwm_carrier_hz:\s*(\d+)", int)
    specs["current_loop_hz"] = _grab(r"current_loop_hz:\s*(\d+)", int)
    specs["tendon_load_cells"] = _grab(r"tendon_load_cells:\s*(\d+)", int)
    specs["control_period_ms"] = _grab(r"control_period_ms:\s*([\d.]+)")
    specs["data_bps"] = _grab(r"data_bps:\s*(\d+)", int)
    specs["overtemp_shutdown_c"] = _grab(r"overtemp_shutdown_c:\s*([\d.]+)")
    specs["hardware_overcurrent_us"] = _grab(r"hardware_overcurrent_us:\s*([\d.]+)", int)
    specs["max_layer_count"] = _grab(r"max_layer_count:\s*(\d+)", int)
    specs["max_thickness_mm"] = _grab(r"max_thickness_mm:\s*([\d.]+)")
    specs["max_pcb_area_mm2"] = _grab(r"max_pcb_area_mm2:\s*(\d+)", int)
    specs["ip_class"] = _grab_str(r"enclosure_class:\s*(\S+)", "ip54")
    specs["encoder_type"] = _grab_str(r"encoder_type:\s*(\S+)", "")
    return specs


def _load_bom() -> list:
    """读取 BOM CSV → 行列表 (utf-8-sig: 剔除 BOM 头, 否则首列键名错位)。"""
    bom_path = PKG / "B1_tendon_drive_bom.csv"
    if not bom_path.exists():
        return []
    with bom_path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _scan_files() -> list:
    """扫描包内实际文件 → [(文件名, 大小KB, 用途)] (排除询价单自身, 防计数漂移)。"""
    rows = []
    if not PKG.exists():
        return rows
    for p in sorted(PKG.iterdir()):
        if not p.is_file() or p.name.startswith("B1_fab_inquiry"):
            continue
        name = p.name
        kb = round(p.stat().st_size / 1024, 1)
        purpose = ""
        for key, label in GERBER_LAYERS.items():   # 层名后缀匹配 (含多段后缀)
            if name.endswith(key):
                purpose = label
                break
        if not purpose:
            if name.endswith(".gbr"):
                purpose = "Gerber 层文件"
            elif name.endswith("_bom.csv"):
                purpose = "BOM 物料清单"
            elif name.endswith("_schematic.svg"):
                purpose = "原理图 (装配参考)"
            elif name.endswith("pcb.svg"):
                purpose = "PCB 装配图"
            elif name.endswith("pcb.dxf"):
                purpose = "板框 DXF (机械接口)"
            elif name.endswith("pcb.pcb.json") or name.endswith(".circuit.json"):
                purpose = "设计数据 JSON (工程备份)"
            elif name.endswith(".cir"):
                purpose = "SPICE 网表 (电气验证)"
            elif name.endswith(".c"):
                purpose = "C 驱动骨架 (固件参考)"
            elif name.endswith(".html"):
                purpose = "打样清单 (工艺要求)"
            else:
                purpose = "设计产物"
        rows.append((name, kb, purpose))
    return rows


def build_inquiry() -> str:
    """生成工厂询价单 Markdown。"""
    specs = _load_specs()
    bom = _load_bom()
    files = _scan_files()
    today = date.today().isoformat()
    n_parts = sum(int(r.get("数量", 0) or 0) for r in bom)

    gerber_files = [f for f in files if f[0].endswith(".gbr")]
    line = []
    a = line.append

    a("# B1 腱驱动关节板 — PCB 打样询价单 (RFQ)")
    a("")
    a(f"> **询价单号**: B1-RFQ-{today.replace('-', '')} | **日期**: {today}")
    a("> **需方**: EmbodiedAI 项目组 | **板名**: B1 攀岩腱驱动关节板 "
      "(TendonDrive-Climbing)")
    a("> **随附文件**: `artifacts/b1_tendon_drive_package/` 全套 "
      f"({len(files)} 件, 见 §3 文件清单)")
    a("")
    a("---")
    a("")

    a("## 一、板规格 (工厂评估工艺能力用)")
    a("")
    a("| 参数 | 值 | 来源 |")
    a("|---|---|---|")
    a(f"| 板层 | {specs['max_layer_count']} 层 | hw_requirements.mechanical |")
    a(f"| 板厚 | ≤ {specs['max_thickness_mm']} mm | 同上 |")
    a(f"| 最大板面积 | {specs['max_pcb_area_mm2']} mm² (约 50×50mm 内) | 同上 |")
    a("| 铜厚 | **2 oz** (外层) | docs/16 工艺要求 |")
    a("| 表面处理 | 沉金 ENIG (建议, 请报价确认) | 询价项 |")
    a(f"| 桥臂电压/电流 | {specs['motor_rail_v']}V / "
      f"{specs['bridge_peak_current_a']}A 峰值 | motor_drive_tendon |")
    a(f"| 桥臂峰值功率 | {specs['bridge_peak_power_w']} W | 同上 |")
    a(f"| PWM 载波 / 电流环 | {specs['pwm_carrier_hz']}Hz / "
      f"{specs['current_loop_hz']}Hz | 同上 |")
    a(f"| 伺服通道 | {specs['servo_channels']} 通道 (4 腱组 + 4 拮抗 VSA) | 同上 |")
    a(f"| 腱张力负载单元 | {specs['tendon_load_cells']} 路 (AD8422 仪表放大) | 同上 |")
    a(f"| 通信 | CAN-FD {specs['data_bps'] / 1e6:.0f}Mbps, 控制周期 "
      f"{specs['control_period_ms']}ms | communication |")
    a(f"| 硬件过流关断 | < {specs['hardware_overcurrent_us']}µs (每通道独立比较器) | safety |")
    a(f"| 过温关断 | {specs['overtemp_shutdown_c']}°C | safety |")
    a(f"| 防护等级 | {str(specs['ip_class']).upper()} (墙面汗/尘) | mechanical |")
    a("")
    a(f"> **功率器件要点**: 三相桥 6× TO-220 MOSFET (IRFS4410Z, 100V/80A) + "
      f"DRV8301 QFN48 栅极驱动; 峰值 {specs['bridge_peak_current_a']}A "
      f"(= 8ch × 4A × 0.6 同时率), 走 2oz 铜厚与宽走线, 请评估温升。")
    a("")

    a("## 二、询价数量与阶梯")
    a("")
    a("| 项 | 数量 | 备注 |")
    a("|---|---|---|")
    a("| PCB 光板 | 5 / 10 / 20 片 (三档分别报价) | 首轮验证 5 片, 请给三档价 |")
    a("| SMT 贴片 | 同上三档 | 含钢网 |")
    a(f"| BOM 器件代采 | 共 {len(bom)} 种料 / {n_parts} 件 | 请分「代采/甲供」报价 |")
    a("| 钢网 (0.12mm) | 1 张 | QFN48 焊膏量敏感 |")
    a("")

    a("## 三、随附文件清单")
    a("")
    a("| 文件 | 大小 (KB) | 用途 |")
    a("|---|---|---|")
    for name, kb, purpose in files:
        a(f"| `{name}` | {kb} | {purpose} |")
    a("")
    a(f"> Gerber 共 **{len(gerber_files)} 层**: " +
      "、".join(purpose for _n, _k, purpose in files
               if purpose in GERBER_LAYERS.values()) +
      "。格式 RS-274X。")
    a("")

    a("## 四、BOM 摘要 (详见随附 CSV)")
    a("")
    a("| 位号 | 型号 | 封装 | 数量 | 描述 |")
    a("|---|---|---|---|---|")
    for r in bom:
        a(f"| {r.get('位号', '')} | {r.get('型号', '')} | {r.get('封装', '')} "
          f"| {r.get('数量', '')} | {r.get('描述', '')} |")
    a("")
    a("> BOM 含替代料列 (第 6 列) —— 缺料时请走替代料审批, 勿擅自替换。")
    a("")

    a("## 五、工艺与质量要求")
    a("")
    a("1. **Gerber 为参数化设计引擎自动产出** —— 投板前我方将提供 "
      "Layout 工程师评审签字版; 请先按本单报价, 审核版 Gerber 与本版层名一致。")
    a("2. 钢网厚度 **0.12mm** (QFN48 焊膏量敏感, 见 B1_prototyping_list.html)。")
    a("3. QFN48 (DRV8301) 与 LQFP100 (STM32G474) 建议 **X-Ray 抽检**。")
    a("4. 飞针测试 (E-Test) 100%, 请随货提供测试报告。")
    a("5. 阻焊颜色: 绿 (默认) / 黑 (如有价差请注明)。")
    a("6. 交付: 光板与贴片成品请**分开报价**并注明各自交期。")
    a("")

    a("## 六、请工厂回填的询价问题")
    a("")
    for i, q in enumerate(QUOTE_QUESTIONS, 1):
        a(f"{i}. {q}")
        a("   - **报价/答复**: ________________")
    a("")

    a("## 七、诚实声明 (随单一并送达, 不删减)")
    a("")
    a("- 本包 Gerber/装配图由参数化设计引擎生成, **打样前必须 Layout 工程师"
      "完成真实 PCB Layout 评审** —— 不可直接用自动布局文件量产。")
    a(f"- 已知不一致: 契约 encoder_type = `{specs['encoder_type']}` "
      "(19bit), 而 BOM 编码器为 AS5048A (14bit) —— 19bit 器件项待补 "
      "(hw_requirements.known_gaps), **本轮按 AS5048A 打样**, 不影响首轮验证。")
    a("- 钢网 Gerber 由板厂按顶铜/焊盘自动生成, 我方不提供单独钢网层文件。")
    a("- 单关节台架首轮仅 1 片即用 (docs/16 Layer 0), 5 片为含备份的最小询价量。")
    a("")
    a("---")
    a("")
    a("*本询价单由 `scripts/b1_fab_inquiry.py` 从板级文件包自动整理 "
      f"({today}); 板规格单一真源 = contracts/hw_requirements.yaml。*")

    return "\n".join(line) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="B1 板级包 → 工厂打样询价单")
    p.add_argument("--out-dir", default=str(PKG), help="输出目录 (默认包目录)")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = build_inquiry()
    out = out_dir / "B1_fab_inquiry_RFQ.md"
    out.write_text(text, encoding="utf-8")

    n_files = len(_scan_files())
    n_bom = len(_load_bom())
    print("B1 打样询价单已生成")
    print(f"  输出: {out}")
    print(f"  板规格: 契约 hw_requirements.yaml (motor_drive_tendon)")
    print(f"  文件清单: {n_files} 件 | BOM: {n_bom} 种料")
    print(f"  询价问题: {len(QUOTE_QUESTIONS)} 项 (待工厂回填)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
