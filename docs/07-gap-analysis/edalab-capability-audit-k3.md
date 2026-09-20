# EdaLab 全面能力审计报告 (K3 基准)

> **日期**: 2026-09-20 | **审计范围**: 100 个 Kotlin 引擎 + CLI + API Stage 注册
> **审计方法**: 逐文件读取源码 + 数学模型实质验证 + 输入→处理→输出链路追踪

---

## 一、总体统计

| 指标 | 数值 |
|------|------|
| Kotlin 引擎文件数 | **100** |
| 总代码量 | **1,358.8 KB** |
| 包分布 | 26 个功能包 |
| 11 步主链 CLI 命令数 | **15** (含 embodied-ai-full 组合命令) |
| API Stage 注册数 | **11** 新 Stage |

## 二、四根支柱合规性总评

| 支柱 | 评级 | 差距 |
|------|------|------|
| 1. 硬件设计自动化骨干 | **A-** | 11 步主链完整骨架, MotorDrive↔FOC 未真正闭环 |
| 2. 三域耦合 | **B+** | MultiPhysicsCoupler 真公式, 但与 MCAD 引擎未衔接 |
| 3. 安全横切层 | **C+** | 硬编码 PHF/SFF, 无 STL 内部集成, 无制造阶段约束反馈 |
| 4. 制造飞轮 | **B** | ICT/FCT/遥测全纳入, DR YAML 格式非标, 未对接训练框架 |

---

## 三、分项引擎评级明细

### 支柱 1: 硬件设计自动化

| 引擎 | 大小 | 评级 | 实质 |
|------|------|------|------|
| ForgeCore | 36.6 KB / 862 行 | **A-** | 11 类传感器选型 + AI 选型理由 + 风险评估 + EPS 生成。MotorDrive ↔ FOC 闭环断裂 |
| ThermalSolver | 9.8 KB | **B+** | Foster 指数瞬态 + 4 层热阻链。非多级 Cauer 网络 |
| PowerIntegritySolver | 11.0 KB | **A-** | 100Hz-1000MHz 阻抗曲线 + 谐振检测 + 目标阻抗。去耦电容按频段硬编码 |
| SignalIntegritySolver | 11.3 KB | **A-** | Wheeler 微带线公式 + Γ 振反射 + 临界长度。眼图经验化 |
| EMCAnalyzer | 11.9 KB | **B-** | 辐射估算 + PWM 谐波 + 环路面积。非全波仿真, 单偶极近似 |
| MotorDriveEngine | 13.4 KB | **B** | MOSFET/驱动/采样真选型 (DRV8353/INA240/TCAN1043), 固件仅字符串注释 |
| FOCFirmwareGenerator | 29.3 KB | **A** | 真 C: Clarke/Park + SVPWM + PI 抗饱和 + <1μs BKIN 保护 + CMake 构建 |
| CommProtocolGenerator | 4.9 KB | **B** | CANopen PDO 标准映射。DCF 缺子索引/访问类型/映射结构 |
| QLearning | 9.3 KB | **A (寻路) / C (关节)** | DoubleArray + ε-贪婪 + C 代码生成真。但关节运动是虚假文档声称 |
| HDL (Parser/Simulator) | 59 KB | **A** | Verilog-2001 真解析 + 事件驱动仿真 + 子模块层次展平 |
| ChipReuseEngine | 80 KB | **A** | 参数化设计→多格式输出 (Gerber/DXF/CPL/BOM/SVG/SPICE/JSON/MD) |
| PcbExporter | 31 KB | **A** | Gerber/X-2 + 拾放 + BOM SVG 完整 |
| Layout引擎 (9 files) | 103 KB | **A-** | DRC + LVS + 自动布线 + 密度填充 + OASIS + GDS |

**支柱 1 差距**:
- MotorDrive 组件选型结果→未直接喂给 FOCFirmwareGenerator(断裂链)
- 缺主轴电机/无刷电机 FOC 参数自辨识算法
- 无高频变压器/谐振变换器设计 (GaN/SiC)

### 支柱 2: 三域耦合 / 物理场

| 引擎 | 大小 | 评级 | 实质 |
|------|------|------|------|
| MultiPhysicsCoupler | 51.5 KB / 1052 行 | **B+** | 能量流闭环 (热→电→热), EMI→PWM 降额→约束传播 50 次迭代。与 MCAD 无接口 |
| StructuralFEMEngine | 15.0 KB | **B** | 位移/模态/疲劳/跌落计算公式。名不副实"梁+板解析"非真正 FEM |
| KinematicsDynamicsEngine | 13.9 KB | **B+** | 2R 平面 IK + FK + 雅可比 + 奇异检测。动力学仅含惯性×速度无拉格朗日 |
| ElectromechanicalCouplingEngine | 11.4 KB | **A-** | 共振频率 + 二阶传函 + 带宽计算。未做 dq 轴电流-扭矩纹波闭环 |
| RelayClutchDynamicsEngine | 13.4 KB | **A** | 弹跳恢复系数 + 电弧侵蚀 + 滑摩功 + STO 链全程 <20ms。全链路最扎实 |
| MassPropertiesEngine | 13.5 KB | **B+** | 惯性张量 + 电机匹配 + 模态预估值。仅规则体 |
| MCADCoDesignEngine | 7.6 KB | **C+** | 真 IDF 格式 + 包围盒干涉。不解析 STEP/IGES、无实体布尔运算、无 Gerber |
| SensorSelectionEngine | 4.5 KB | **B** | 多传感器融合决策 + 预算分配。仅评分加权 |
| HILBridge | 3.6 KB | **C** | 仅数据类 + createSession/runTest 占位 |
| DigitalTwinEngine | 5.1 KB | **C** | physical/digital 双向 sync 占位, 真正同步算法未实现 |

**支柱 2 差距**:
- MultiPhysicsCoupler 未输出到 MotorDriveEngine (耦合结果→不能自动调整PCB层数/铜皮重量)
- MCAD 纯 IDF/包围盒, 不能做真正的 3D 实体布尔求交
- HIL Bridge 无真正的 ROS2 节点 + 同步数据协议
- Digital Twin 无数据驱动模型更新

### 支柱 3: 安全横切层

| 引擎 | 大小 | 评级 | 实质 |
|------|------|------|------|
| FunctionalSafetyEngine | 4.8 KB | **C** | 无 PFH/SFF 真计算, 无 SIL 等级达标判定, 无时间预算分解。硬编码常量+字符串占位 |
| ReliabilityEngine | 6.0 KB | **B** | 降额表(类 IPC-9592) + MTBF=1e9/ΣFIT 简化公式。FMEA 仅 3×倍数、Ea 不标出来源 |

**支柱 3 差距** — **最大差距**:
- ⚠️ **PFH 无计算** — SFF 硬编码而非公式 (S+DD)/(S+DD+DU)
- ⚠️ **SIL 判定缺失** — 未对比 actual PFH < target PFH
- ⚠️ **时间预算无分解** — STO 响应时间未拆分为: 传感器延迟→比较器→锁存→驱动 propagation→MOSFET gate discharge
- ⚠️ **与制造阶段无集成** — 安全约束未渗透进 MotorDrive 选型 (如 MOSFET SOA 曲线)
- ⚠️ **与 EmbodiedAI STL 无接口** — MultiPhysicsCoupler 的安全链分析结果未传到 EmbodiedAI 的 STL 监控

### 支柱 4: 制造飞轮

| 引擎 | 大小 | 评级 | 实质 |
|------|------|------|------|
| ManufacturingDataFlywheel | 13.9 KB / 347 行 | **B** | ICT 编码器/CAN-FD 利用率/遥测 → DR 配置。exportDrYaml 输出格式非标 |

**支柱 4 差距**:
- DR YAML 输出缺 RL 框架标准顶层键 (`observations:` / `actions:` / `sim:`)
- 未实现 DR→训练框架对接 (Isaac Gym / RLlib / rsl_rl)
- 未做 DR 训练效果的 A/B 对比评估闭环
- → EmbodiedAI K2 已改造此部分 (edalab.py flywheel_e2e)

---

## 四、关键发现 (5 个假阳性 / 4 个最大差距)

### 假阳性 (文档声称与代码不符)

1. **QLearning 关节运动** — 实际仅 4 邻接迷宫寻路, 连续状态和动力学是伪造声称
2. **StructuralFEM** — 名称"FEM", 实际无网格/无刚度矩阵, 仅为梁+板解析公式
3. **FunctionalSafety "generateSafetyCase"** — 输出为字符串描述, 无计算过程
4. **HIL Bridge "runTest"** — 占位未实现, 无真实 ROS2-HIL 协议
5. **DigitalTwin "predictFaults"** — 未做频域特征 + 判别器

### 4 个最大差距 (P0 修复优先级)

| # | 差距 | 影响 | 修复量级 |
|---|------|------|----------|
| G1 | Functional Safety 无 PFH/SFF 真计算 | 安全泛认证无法通过 | 50 行 Kotlin |
| G2 | MotorDrive↔FOC Firmware 闭环断裂 | 软件生成→手动复制规格参数 | 1 个函数接口 |
| G3 | MultiPhysicsCoupler 未输出到 MotorDrive | 耦合结果→不能自动调整铜厚/走线宽度 | 1 个互操作映射 |
| G4 | Safety→Embedded STL 无接口 | 硬件安全分析→不能被 AI 运行时消费 | 序列化契约 + 1 个跨项目调用 |

---

## 五、审计方法与工具

1. **文件统计**: PowerShell `Get-ChildItem *.kt | Measure Length -Sum` = 1,358.8 KB
2. **代码行数**: `wc -l` 逐文件验证
3. **公式校验**: 关键词搜索 (Foster/Wheeler/Γ/MTBF/PFH/SFF/Clarke/SVPWM/...)
4. **链路追踪**: 在 CLI 子命令 API 上追踪 cmdForgeCore()→forgeCore 具体是否有真正实现
5. **交叉验证**: EmbodiedAI 侧 `adapters/edalab.py` FLOW_CHAIN 与 EdaLab 引擎清单 1:1 映射确认

---

## 六、结论

**EdaLab 在硬件设计自动化层面达到了骨架级可用 (A-), 9 个物理场引擎中 6 个有真正数学模型。**

**但在安全层和制造-训练闭环上存在显著缺口:**
- 安全计算不能支撑 IEC 61508/SIL 认证 (P0 差距)
- 三域耦合与 hardware design 未反馈 (P1)
- HDL (Verilog-2001) + FOC C 代码生成 + ChipReuseEngine 已达生产级质量

**整体: 骨架可用, 不能直接作为产品化硬件自动化 PAAS。** P0 差距 (PFH/SFF + 闭环) 修复后可达产品化门槛。
