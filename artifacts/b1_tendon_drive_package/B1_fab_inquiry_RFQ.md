# B1 腱驱动关节板 — PCB 打样询价单 (RFQ)

> **询价单号**: B1-RFQ-20261017 | **日期**: 2026-10-17
> **需方**: EmbodiedAI 项目组 | **板名**: B1 攀岩腱驱动关节板 (TendonDrive-Climbing)
> **随附文件**: `artifacts/b1_tendon_drive_package/` 全套 (14 件, 见 §3 文件清单)

---

## 一、板规格 (工厂评估工艺能力用)

| 参数 | 值 | 来源 |
|---|---|---|
| 板层 | 6 层 | hw_requirements.mechanical |
| 板厚 | ≤ 1.6 mm | 同上 |
| 最大板面积 | 2500 mm² (约 50×50mm 内) | 同上 |
| 铜厚 | **2 oz** (外层) | docs/16 工艺要求 |
| 表面处理 | 沉金 ENIG (建议, 请报价确认) | 询价项 |
| 桥臂电压/电流 | 24.0V / 19.2A 峰值 | motor_drive_tendon |
| 桥臂峰值功率 | 460.8 W | 同上 |
| PWM 载波 / 电流环 | 20000Hz / 10000Hz | 同上 |
| 伺服通道 | 8 通道 (4 腱组 + 4 拮抗 VSA) | 同上 |
| 腱张力负载单元 | 8 路 (AD8422 仪表放大) | 同上 |
| 通信 | CAN-FD 5Mbps, 控制周期 5.0ms | communication |
| 硬件过流关断 | < 1µs (每通道独立比较器) | safety |
| 过温关断 | 105.0°C | safety |
| 防护等级 | IP54 (墙面汗/尘) | mechanical |

> **功率器件要点**: 三相桥 6× TO-220 MOSFET (IRFS4410Z, 100V/80A) + DRV8301 QFN48 栅极驱动; 峰值 19.2A (= 8ch × 4A × 0.6 同时率), 走 2oz 铜厚与宽走线, 请评估温升。

## 二、询价数量与阶梯

| 项 | 数量 | 备注 |
|---|---|---|
| PCB 光板 | 5 / 10 / 20 片 (三档分别报价) | 首轮验证 5 片, 请给三档价 |
| SMT 贴片 | 同上三档 | 含钢网 |
| BOM 器件代采 | 共 13 种料 / 26 件 | 请分「代采/甲供」报价 |
| 钢网 (0.12mm) | 1 张 | QFN48 焊膏量敏感 |

## 三、随附文件清单

| 文件 | 大小 (KB) | 用途 |
|---|---|---|
| `B1_prototyping_list.html` | 7.2 | 打样清单 (工艺要求) |
| `B1_tendon_drive.c` | 7.3 | C 驱动骨架 (固件参考) |
| `B1_tendon_drive.cir` | 1.1 | SPICE 网表 (电气验证) |
| `B1_tendon_drive.circuit.json` | 39.0 | 设计数据 JSON (工程备份) |
| `B1_tendon_drive_bom.csv` | 1.0 | BOM 物料清单 |
| `B1_tendon_drive_pcb.board_outline.gbr` | 0.1 | 板框 (Outline) |
| `B1_tendon_drive_pcb.bottom_copper.gbr` | 3.5 | 底层铜 (Bottom Copper) |
| `B1_tendon_drive_pcb.copper_pour.gbr` | 48.6 | 内层铺铜 (Copper Pour) |
| `B1_tendon_drive_pcb.dxf` | 14.1 | 板框 DXF (机械接口) |
| `B1_tendon_drive_pcb.pcb.json` | 6.0 | 设计数据 JSON (工程备份) |
| `B1_tendon_drive_pcb.svg` | 107.7 | PCB 装配图 |
| `B1_tendon_drive_pcb.top_copper.gbr` | 2.0 | 顶层铜 (Top Copper) |
| `B1_tendon_drive_pcb.top_silk.gbr` | 0.5 | 顶层丝印 (Top Silkscreen) |
| `B1_tendon_drive_schematic.svg` | 27.8 | 原理图 (装配参考) |

> Gerber 共 **5 层**: 板框 (Outline)、底层铜 (Bottom Copper)、内层铺铜 (Copper Pour)、顶层铜 (Top Copper)、顶层丝印 (Top Silkscreen)。格式 RS-274X。

## 四、BOM 摘要 (详见随附 CSV)

| 位号 | 型号 | 封装 | 数量 | 描述 |
|---|---|---|---|---|
| U_MCU | STM32G474VET6 | LQFP100 | 1 | 主控 Cortex-M4 170MHz FOC MCU |
| U_GATE | DRV8301 | QFN48 | 1 | 60V 半桥栅极驱动 2.3A |
| Q1~Q6 | IRFS4410Z | TO-220-7P | 6 | 100V/80A N-MOSFET 三相桥臂 |
| U_CS1, U_CS2 | INA240AIDGSR | MSOP10 | 2 | 双向电流检测放大器 |
| U_CAN | TCAN1043A-Q1 | SOIC14 | 1 | CAN-FD 5Mbps 车规收发器 |
| U_ENC | AS5048A | TSSOP14 | 1 | 14-bit 绝对磁编码器 |
| U_LC0~U_LC7 | AD8422 | SOIC8 | 8 | 仪表放大器 (腱应变片放大) |
| CBULK1 | 470µF/35V 电解 | RADIAL_10x16 | 1 | 24V 输入储能 |
| CBULK2 | 100nF/50V MLCC | 0805 | 1 | 24V 高频去耦 |
| J_PWR | MATELOK 2P | TE_MATELOK_2P | 1 | 24V 电源输入 (>=19.2a) |
| J_MOT | JST B3P-VH | JST_B3P-VH | 1 | 电机 UVW 输出 |
| J_CAN | JST B4P-VH | JST_B4P-VH | 1 | CAN-FD 总线 |
| J_ENC | JST B6P-VH | JST_B6P-VH | 1 | 编码器接口 (电机/腱双端) |

> BOM 含替代料列 (第 6 列) —— 缺料时请走替代料审批, 勿擅自替换。

## 五、工艺与质量要求

1. **Gerber 为参数化设计引擎自动产出** —— 投板前我方将提供 Layout 工程师评审签字版; 请先按本单报价, 审核版 Gerber 与本版层名一致。
2. 钢网厚度 **0.12mm** (QFN48 焊膏量敏感, 见 B1_prototyping_list.html)。
3. QFN48 (DRV8301) 与 LQFP100 (STM32G474) 建议 **X-Ray 抽检**。
4. 飞针测试 (E-Test) 100%, 请随货提供测试报告。
5. 阻焊颜色: 绿 (默认) / 黑 (如有价差请注明)。
6. 交付: 光板与贴片成品请**分开报价**并注明各自交期。

## 六、请工厂回填的询价问题

1. PCB 光板打样单价 (5 / 10 / 20 片阶梯价)? 交期多少个工作日?
   - **报价/答复**: ________________
2. SMT 贴片费用如何计 (按焊点 / 按器件)? 双面贴是否加价?
   - **报价/答复**: ________________
3. 钢网 (0.12mm 厚, 见工艺要求) 是否随板免费提供? 单独订费用?
   - **报价/答复**: ________________
4. 4 层 2oz 铜厚 + 板厚 1.6mm 是否标准工艺? 有无加价?
   - **报价/答复**: ________________
5. 最小起订量 (MOQ) 与工程费 (Gerber 审核/工程确认) 各多少?
   - **报价/答复**: ________________
6. 能否代采 BOM 器件? 缺料时替代料审批流程? (BOM 已列替代料)
   - **报价/答复**: ________________
7. QFN48 (DRV8301) 与 LQFP100 (STM32G474) 的贴装良率保障与 X-Ray 检查是否含?
   - **报价/答复**: ________________
8. 阻抗控制 / 阻焊颜色 / 表面处理 (建议沉金 ENIG?) 选项与价差?
   - **报价/答复**: ________________
9. 飞针测试 (E-Test) 是否 100%? 测试报告格式?
   - **报价/答复**: ________________
10. 打样交付物: 光板+贴片成品 还是 分开报价? 交付周期分别是?
   - **报价/答复**: ________________

## 七、诚实声明 (随单一并送达, 不删减)

- 本包 Gerber/装配图由参数化设计引擎生成, **打样前必须 Layout 工程师完成真实 PCB Layout 评审** —— 不可直接用自动布局文件量产。
- 已知不一致: 契约 encoder_type = `absolute_19bit` (19bit), 而 BOM 编码器为 AS5048A (14bit) —— 19bit 器件项待补 (hw_requirements.known_gaps), **本轮按 AS5048A 打样**, 不影响首轮验证。
- 钢网 Gerber 由板厂按顶铜/焊盘自动生成, 我方不提供单独钢网层文件。
- 单关节台架首轮仅 1 片即用 (docs/16 Layer 0), 5 片为含备份的最小询价量。

---

*本询价单由 `scripts/b1_fab_inquiry.py` 从板级文件包自动整理 (2026-10-17); 板规格单一真源 = contracts/hw_requirements.yaml。*
