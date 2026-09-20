# 跨项目资产吸收与融合完善计划 — EmbodiedAI × ECAD-MCAD × SynapticForge × EdaLab

> **版本**: v1.0 | **日期**: 2026-09-20
> **定位**: 盘点 ECAD-MCAD 与 SynapticForge（含其底座的 EdaLab）中 EmbodiedAI 可吸收的资产，结合本项目 G1-G13 缺口，制定具体、详尽、前沿创新的完善计划。
> **方法**: 逐文件实读三个项目的核心文档（非推断），交叉映射后分级给吸收方案。
> **来源实读清单**:
> - `workspace/ECAD-MCAD/`: 最终计划.md、全链条技术缺口补充.md、硬件全生命周期衔接方案.md
> - `workspace/SynapticForge/`: README.md、docs/02-tech-architecture/00-architecture-overview.md、docs/03-roadmap/00-master-roadmap.md、docs/02-tech-architecture/02-edalab-capability-audit-v2.md
> - 本项目: docs/architecture-overview.md、docs/07-gap-analysis/project-gap-comprehensive-report.md、docs/04-hardware/nn-to-hardware-pipeline.md、docs/future-exploration-plan.md

---

## 一、为什么吸收：EmbodiedAI 的结构性不对称

本项目（EmbodiedAI）的资产分布现状（据 architecture-overview.md §6 数学模型完备性矩阵 + 07-gap-analysis）：

| 域 | 深度 | 证据 |
|---|---|---|
| RL/决策数学、浮基/脉冲动力学、竞技攀岩数学 | ✅ 完整且原创 | climbing-dynamics-math.md、climbing_dynamics_core.py |
| 触觉-视觉概率融合 | ✅ 已补（多速率 EKF + 机会约束） | tactile-fusion-conclusion.md |
| 硬件工程（FOC/自动关机） | ⚠️ 工程级流程图，无电磁方程、无仿真验证 | hardware-motor-shutdown.md |
| AI 模型训练/部署 | ⚠️ 架构设计有、训练管线无（G8） | vla-model-design.md |
| Sim-to-Real 闭环 | ❌ 缺（G7，P0 级） | gap-analysis-sim2real.md |
| 安全认证 | ❌ 缺（G10） | — |
| 硬件→AI 的数据反馈通道 | ❌ 完全缺失 | nn-to-hardware-pipeline.md 只写了正向 |

而 SynapticForge README 对三项目分工的判断精辟：**EmbodiedAI 单独 = "写了几十万行论文，电机转不起来"**。本计划要吸收的正是让电机转起来的那一半：可制造的硬件工程链 + 数据回流通道。

---

## 二、可吸收资产盘点（实读核对）

### 2.1 来自 ECAD-MCAD

| # | 资产 | 位置 | 形态 | 对本项目的价值 |
|---|---|---|---|---|
| A1 | **NG1-NG22 深化缺口审计**（22 项技术维度，含公式/标准/参数/工具） | 全链条技术缺口补充.md | 文档 | 一次性补齐硬件工程知识面：热(NG1)/PI(NG2)/SI(NG3)/EMC(NG4)/可靠性(NG5)/BMS(NG6)/FPC(NG7)/RF(NG8)/FOC固件(NG9)/OTA(NG10)/CAN-FD·TSN(NG11)/MLOps(NG12)/标定(NG13)/功能安全(NG14)/追溯(NG15)等 |
| A2 | **P0-P9 全生命周期模型** + 逐阶段接口契约 | 硬件全生命周期衔接方案.md | 文档 | 把本项目"极详细但纸面"的硬件需求规格书变成可执行的工程阶段化路径；其 P0 阶段的 `hw_requirements.yaml` schema 正是本项目第 2/4 章缺的机器可读形式 |
| A3 | **ECAD-MCAD 数据交互枢纽设计**（BoardModel 中间模型、IDF3.0/STEP/DXF/IDX 多格式、双向同步、3D 干涉前置、diff 可追踪） | 最终计划.md §3、技术草案.md | 文档 | G11 灵巧手设计需要"结构-电路-传感一体化迭代"，本枢纽正是机电协同的方法论 |
| A4 | **能力联邦模式**（EdaLab 内核 + ngspice/KiCad/FreeCAD/ROS2 外部工具，统一 Job/Artifact/Callback 语义） | 最终计划.md §4 | 文档 | 与本项目"全栈自研"原则的边界协调：核心自研、重仿真器联邦 |
| A5 | **前瞻技术资料清单**（器件 SPICE 模型、CAN-FD/DBC、IPC 标准、阻抗/叠层、认证体系、URDF/Isaac Sim 接口、供应链数据源） | 最终计划.md §10 | 清单 | 直接作为本项目采购/学习/接口对接的 checklist |
| A6 | **S0-S7 路线图与 DoD 验收标准** | 最终计划.md §6/§8 | 文档 | 验收可量化：关机<1μs、DRC 清零、3D 干涉消除、SN↔固件↔校准可追溯 |

### 2.2 来自 SynapticForge

| # | 资产 | 位置 | 形态 | 对本项目的价值 |
|---|---|---|---|---|
| B1 | **hw_requirements.yaml 数据契约**（strategy_version、sensors.frame_rate 由 RL 需求决定等字段） | 00-architecture-overview.md §3.3 | 文档+schema 草案 | G7 的关键一环：AI 需求机器可读化，是"算法→硬件"的正向桥 |
| B2 | **ForgeCore L4 编排层概念**（AI 约束解析→规格分解→器件选型→设计规则转换→EPS 生成） | 00-architecture-overview.md §2.2 | 文档 | 本项目 nn-to-hardware-pipeline.md 目前只有人读的延迟预算链，ForgeCore 是其自动化版本 |
| B3 | **AI×硬件数据飞轮**（硬件遥测→域随机化→RL 更新；"算法改动→硬件验证 ≤7 天"） | 00-architecture-overview.md §3.2 | 文档 | 补上本项目完全缺失的"硬件→AI"反向通道 |
| B4 | 六层融合架构对 EmbodiedAI 资产的引用索引 | reference/embodiedai-roadmap/INDEX.md | 索引 | 反向核对：外部项目认为本项目哪些资产最有价值（接触力学、多速率 EKF、机会约束、20Hz 分层架构）——这正是对外输出面 |

### 2.3 来自 EdaLab（SynapticForge/ECAD-MCAD 的共同底座，已代码级验证）

| # | 资产 | 位置 | 形态 | 状态 |
|---|---|---|---|---|
| C1 | 16 个具身智能引擎（ForgeCore/MotorDrive/ThermalSolver/PI/SI/EMC/BMS/FOC固件/CommProtocol/功能安全/可靠性/HIL/DigitalTwin/数据飞轮/传感器选型/MCAD协同） | edalabcore (Kotlin) | **真实代码** | 编译通过（仅 StructuralFEMEngine.kt:200 余 1 个 Float/Double 错误） |
| C2 | CLI 15 个子命令 + Open API（9 Stage + 4 Kind + 3 Preset，含 `embodied_ai_full`） | EdaCli.kt / edalab_stages.py | 代码 | 已注册，`embodied-ai-full` 串联完整性待端到端验证 |
| C3 | 3 个 P0 力学引擎（质心/惯性、DH 正逆运动学、结构 FEM） | physics/*.kt | 代码 | 2 个编译通过，1 个差一行修复 |

---

## 三、吸收映射矩阵：资产 → 本项目缺口

| 本项目缺口 | 吸收资产 | 吸收方式 | 优先级 |
|---|---|---|---|
| **G7 Sim-to-Real 闭环** | B3 数据飞轮 + A2 的 P8(HIL)/P9(数据回填) + C1 HILBridge/DigitalTwinEngine/ManufacturingDataFlywheel 真实代码 | 适配吸收：把 future-exploration-plan Phase A 的"真机测试+数据分析"升级为"HIL 采集→DR 参数→RL 更新"的契约化闭环 | 🔴 P0 |
| **G8 VLA 训练管线** | A1 的 NG12（ONNX/量化/TensorRT/逐层精度验证/profiling） | 直接吸收：写入 VLA 训练管线的部署段；GelSight 921KB/帧的 GPU 预处理约束（本架构总览 §7）正好由 TensorRT engine 的延迟预算承接 | 🔴 P0 |
| **G10 安全认证** | A1 的 NG14（ISO13849 PL d、STO、双通道急停、HAZOP）+ NG5（DFMEA、降额、Coffin-Manson 焊点疲劳）+ C1 FunctionalSafetyEngine/ReliabilityEngine | 直接吸收 + 代码对接：安全体系从"硬件自动关机一点"扩展为 design-in 的功能安全框架 | 🟠 P1 |
| **G11 灵巧手** | A3 ECAD-MCAD 枢纽（3D 干涉/IDF/STEP/diff）+ NG7 FPC 刚柔 + C1 MCADCoDesignEngine + C3 结构 FEM | 方法论吸收：灵巧手=机电热耦合最密集部件，用枢纽方法做"结构-传感-驱动"协同迭代 | 🟡 P2 |
| **G9 全身运动控制** | C3 KinematicsDynamicsEngine（DH/Jacobian/工作空间）+ MassPropertiesEngine（质心/惯性/稳定性） | 工具对接：双足 ZMP/LIPM 分析的物理参数（质量分布、惯量）由引擎供给 | 🟡 P2 |
| **G6 柔性体动力学** | （无直接资产；C1 StructuralFEMEngine 仅线弹性小变形） | 不吸收，维持原计划自研（FEM vs 质点弹簧 vs 学习近似三选一） | 原计划 |
| **硬件工程深化**（本架构总览"电机驱动/电磁 ⚠️ 工程级"） | A1 的 NG1-NG4/NG6/NG9 全部公式与标准 + C1 对应四个求解器代码 | 知识吸收 + 仿真验证：给 hardware-motor-shutdown.md 的 <1μs 关断设计补上 ngspice 瞬态仿真验证方法（ECAD-MCAD M1 的做法） | 🟠 P1 |
| **正反向通道** | B1 + B2 + A2 的 P0 契约 | 直接吸收：`hw_requirements.yaml` 确立为算法→硬件的唯一契约格式 | 🔴 P0 |

---

## 四、分级吸收清单

### 4.1 直接吸收（零改造，1 周内可完成）

1. **知识体系并入**：把 NG1-NG22 的技术细节按域拆入本项目文档树：
   - NG9/NG10 → `docs/04-hardware/`（新建 foc-firmware-engineering.md、ota-secureboot.md）
   - NG11 → `docs/05-system-software/`（canfd-tsn-comm-stack.md；注意：CAN-FD 帧格式/ID 分配/DBC 正是 ECAD-MCAD 明确向本项目索要的前置输入，**这是本项目欠外部的一笔债，须最先还**）
   - NG12 → `docs/01-ai-model/`（mlops-quantization-deployment.md）
   - NG13 → `docs/03-sensing-perception/`（sensor-calibration.md：张正友+手眼 Tsai/Park+IMU Allan 方差+力标定——本项目 tactile 文档已有标定矩阵黑盒问题，正好补全）
   - NG14/NG5 → `docs/04-hardware/`（functional-safety-framework.md、reliability-dfmea.md）
   - NG6/NG7 → `docs/04-hardware/`（bms-system-design.md、fpc-rigid-flex.md）
2. **hw_requirements.yaml 定为本项目标准输出物**：每当策略/架构文档产生硬件约束（如"100Hz 触觉采样、6 轴力传感、CAN-FD"），同步产出一份带 `strategy_version` 的 yaml。
3. **交叉引用网**：README 与各章头部补"外部资产引用"（ECAD-MCAD 枢纽、SynapticForge 六层架构、EdaLab 16 引擎），双向锚定。

### 4.2 适配吸收（1-3 个月，需要设计映射）

1. **Sim-to-Real 闭环契约化**（G7）：把 future-exploration-plan Phase A Month 3 的"真机测试+数据分析"重定义为五段契约：
   `RL 回放格式 → HIL 遥测 schema → DR 参数更新文件 → 训练配置 diff → 策略版本号`，全部以文件为接口（与 ECAD-MCAD P8-P9 契约对齐），使闭环不依赖人为传递。
2. **延迟预算自动化核对**：nn-to-hardware-pipeline.md 的人算预算（2+15+1+1+10≈29ms）改为 yaml 声明 → 脚本核对 EdaLab 仿真报告（SI 延迟、热降频）的自动校验。
3. **安全框架落地**：NG14 的 PL d/STO 与本项目的机会约束握力调度（f_n ≥ ‖f_t‖/(μ̂−z·σ_μ)）合并为"软硬两级安全滤波器"设计文档。

### 4.3 新建（创新项，见第六节）

前沿创新方向 5 项，均为三项目单独不具备、融合后才成立的增量。

---

## 五、与现有 future-exploration-plan 的合并关系

不推翻原 Phase A/B/C，而是**给原计划补上"硬件接口半边"**：

| 原计划 | 原内容 | 本计划增量 |
|---|---|---|
| Phase A M1 仿真环境 | Isaac Sim/MuJoCo + URDF + Pinocchio | + URDF 由 ECAD-MCAD 关节铝壳 STEP 派生（A5 §10.7 已列 fixture） |
| Phase A M2 RL 训练 | BC + PPO/SAC + DR | + DR 参数结构化（与制造差异对齐，见创新 #1） |
| Phase A M3 Sim-to-Real | 真机测试 | + 五段契约闭环（4.2.1） |
| Phase B M5-6 VLA 小模型 | 训练+评估 | + NG12 部署管线（量化/逐层精度/<50ms 预算核验） |
| Phase B M4-5 全身控制 | ZMP+LIPM | + MassProperties/KinematicsDynamics 引擎供参 |
| Phase C 安全认证框架 | 待建 | + NG14/NG5 全套标准与 DFMEA 模板，起点提前 |

---

## 六、前沿创新方向（融合后才成立的增量）

### 创新 #1：制造差异驱动的域随机化（Manufacturing-Variance-Informed DR）

**洞察来源**：ECAD-MCAD 的 ManufacturingDataFlywheel（ICT/FCT→DR 闭环）里 "DR" 指设计规则（Design Rule）；本项目 RL 侧的 DR 指域随机化（Domain Randomization）——**两者在 EdaLab 数据飞轮上同名不同义，恰好构成一个未被任何人做过的闭环**：

```
ICT/FCT 实测每块板的器件容差（Rds(on) ±20%、相电感偏差、ADC 增益误差）
    → 结构化为 RL 域随机化参数分布（真实容差替代 uniform 拍的脑袋范围）
    → 策略在"真实硬件总体分布"上训练
    → 真机上滑脱率/轨迹误差下降可归因于硬件差异被建模
    → RL 策略的鲁棒性需求反过来收紧产线测试容差（写回 ICT 规则）
```

**验证方式**：用 climbing_dynamics_core.py 的 Monte Carlo 滑脱率验证框架，对比"uniform DR"与"制造容差 DR"两组策略的滑脱率差。
**学术定位**：Sim-to-Real 文献中把制造容差作为 DR 先验的工作极少，且本项目有独特的机会约束滑脱指标可量化收益。

### 创新 #2：物理信息硬件综合（Physics-Informed Hardware Synthesis）

**洞察来源**：SynapticForge ForgeCore 的翻译方向是"AI 需求→硬件"；本项目 G1-G5 数学资产使其可以反向增强这条链：

```
本项目独有数学资产                     → 硬件设计参数（ForgeCore 输入）
─────────────────────────────────────────────────────
μ 的 RLS 在线估计 + σ_μ 不确定度        → 力传感器量程与精度冗余度分配
多速率 EKF (视觉20Hz/触觉100Hz)        → ADC 采样率 + MIPI 带宽 + 时间同步精度
机会约束置信度 z                        → 安全裕度 → MOSFET 降额比
Hertz 接触刚度 k（GelSight ~60N@5mm）   → 指尖骨架结构 FEM 载荷输入
滑脱前兆（切向柔顺下降）                → 触觉帧率下限 + 1kHz 伺服环增益
```

即：**不仅"AI 说要什么硬件"，而是"接触动力学的可微约束告诉硬件必须留多少裕度"**。这使 ForgeCore 从需求翻译器升级为物理约束编译器。

### 创新 #3：策略-硬件协同版本化（Policy-Hardware Co-Versioning）

**洞察来源**：SynapticForge 的 hw_requirements.yaml 已埋了 `strategy_version` 字段，但没有双向化。定义：

- 每个策略版本绑定一份 `hw_requirements.yaml` + 一个 EdaLab Job ID + 制造批次 SN 范围
- 真机数据回流时按 (policy_ver, hw_rev, batch) 三元组归因性能
- 回答"这次失败率上升是策略回归还是新批次硬件差异"——当前业界靠工程师拍脑袋

**验证方式**：HIL 阶段用两个固件参数版本 × 两批仿真器件容差，验证归因正确性。

### 创新 #4：软硬两级安全滤波器（STO × 机会约束）

**洞察来源**：本项目的机会约束握力调度（数学层）与 ECAD-MCAD NG14 的 STO/硬件关断（电气层）从未被形式化为一个系统：

```
L1 硬件确定层 (<1μs):  比较器 + 锁存 + STO —— 数学不参与，物理底线
L2 反射层 (1kHz):      机会约束滤波 f_n ≥ ‖f_t‖/(μ̂−z·σ_μ) —— 概率安全
L3 策略层 (20Hz):      RL 动作经安全包络投影（CWC 接触 wrench 锥）
```

三层各有数学保证（确定性/概率性/集值安全），失效模式互补。可产出形式化安全论证文档，直接支撑 G10 认证与论文。

### 创新 #5：数字孪生动力学互校（Twin-in-the-Loop Dynamics Vetting）

**洞察来源**：EdaLab DigitalTwinEngine（预测性维护向）与本项目 climbing_dynamics_core.py（Hunt-Crossley+摆荡+dyno）是同一物理对象的两套实现：

- HIL 阶段让两者并行跑同一真机输入，输出残差序列
- 残差 → 在线辨识修正孪生参数（本项目 EKF/RLS 资产直接复用）
- 修正后的孪生 → 加速 RL 训练（比真机快、比盲仿真准）

这是"数字孪生"从监控工具升级为**训练资产**的路径，与 SynapticForge 飞轮的"7 天闭环"目标互锁。

---

## 七、分阶段执行计划

### P0 — 契约与知识并入（第 1-2 周）

| 任务 | 交付 | 验收 |
|---|---|---|
| P0.1 还债：CAN-FD DBC + 帧格式/ID 分配 + MCU 载波频率 + PTP 细节 | `docs/05-system-software/canfd-contract/`（DBC + 文档） | ECAD-MCAD §10.2 必需项逐条覆盖；与 nn-to-hardware-pipeline.md §4 帧结构一致 |
| P0.2 hw_requirements.yaml v1 | `contracts/hw_requirements.yaml`（以 20Hz 家庭服务架构为准） | 覆盖架构总览 §7 的 8 条关键约束；schema 可被 SynapticForge 引用 |
| P0.3 NG 知识拆分并入（4.1.1 的 7 份新文档） | docs 树扩充 | 每份含公式/标准/与本项目资产的接口说明 |
| P0.4 交叉引用网 | README + 各章引用 | 三项目双向可达 |

### P1 — 闭环契约化（Month 1-3，与原 Phase A 并行）

| 任务 | 交付 | 验收 |
|---|---|---|
| P1.1 Sim-to-Real 五段契约 | 回放格式/schema/DR 更新文件模板 | 一条数据可从真机遥测走回训练配置，全程无人工搬运 |
| P1.2 延迟预算自动核对脚本 | `scripts/latency_audit.py` | yaml 预算 vs 仿真报告自动 diff，越界报警 |
| P1.3 催化条件：推动 EdaLab 修复 StructuralFEMEngine.kt:200 + embodied-ai-full 端到端 smoke test | EdaLab 侧 patch + 测试报告 | 三力学引擎可用；15 子引擎串联确认 |

### P2 — 安全与硬件深化（Month 3-6）

| 任务 | 交付 | 验收 |
|---|---|---|
| P2.1 创新 #4 软硬两级安全滤波器文档 + 形式化论证 | `docs/04-hardware/safety-filter-architecture.md` | 三层失效模式互补性论证完整；与 NG14 PL d 对齐 |
| P2.2 <1μs 关断设计的 SPICE 验证方法（吸收 ECAD-MCAD M1 做法） | hardware-motor-shutdown.md 增补章 | ngspice 瞬态仿真方案 + 通过判据 |
| P2.3 创新 #2 物理信息硬件综合首批映射表 | `docs/04-hardware/physics-to-hw-mapping.md` | 上表 5 行映射逐条公式化 |

### P3 — 创新验证（Month 6-18）

| 任务 | 交付 | 验收 |
|---|---|---|
| P3.1 创新 #1 制造容差 DR 实验 | 实验报告 | 滑脱率对比显著；可写论文/专利 |
| P3.2 创新 #3 协同版本化原型 | (policy, hw, batch) 追踪 schema | 归因实验正确率 >90% |
| P3.3 创新 #5 孪生互校 | 残差监控 + 参数修正回路 | 孪生修正后 RL 训练加速可量化 |
| P3.4 G11 灵巧手：吸收 A3 枢纽方法启动设计迭代 | 灵巧手机电协同设计文档 v0 | 3D 干涉/传感-驱动布局/弯曲寿命约束纳入 |

---

## 八、风险与对策

| 风险 | 对策 |
|---|---|
| EdaLab embodied-ai-full 未真正串联 15 引擎（审计 2.4 未验证项） | P1.3 作为前置催化任务，未通过则闭环降级为"逐引擎 CLI 调用"模式 |
| 全栈自研原则 vs 能力联邦 | 边界声明：物理/算法/安全数学自研；重仿真器（ngspice/CFD）与机械 CAD 属"工具执行"，不属"设计知识"，联邦不违背原则 |
| 三项目文档版本漂移（均在快速迭代） | 交叉引用带版本号（如"ECAD-MCAD 最终计划 v1.1"）；P0.4 建立引用清单集中维护 |
| 创新 #1 需要真实 ICT/FCT 数据（当前无产线） | 降级路径：先用器件 datasheet 容差合成"伪制造分布"做方法学验证，真机数据到位后替换 |
| NG 知识并入导致文档膨胀失控 | 每份新文档强制含"与本项目现有资产的接口"一节，无接口的不并入 |

---

## 九、总原则（一句话）

**吸收"让电机转起来"的工程链与数据通道，自研"决定怎么转"的数学与安全**——硬件工程知识、生命周期契约、数据飞轮结构直接吸收；接触动力学、机会约束、EKF、物理化奖励保持本项目原创核心；五个融合创新方向作为长期差异化壁垒。
