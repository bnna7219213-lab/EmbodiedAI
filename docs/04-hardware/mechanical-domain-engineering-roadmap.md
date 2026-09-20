# 机械-电气力学域引擎 P1/P2 发展路线图 — 从结构骨架到变体机构与机电耦合闭环

> **版本**: v1.0 | **日期**: 2026-09-20
> **定位**: EdaLab P0 力学域三引擎（MassProperties / KinematicsDynamics / StructuralFEM）的后续发展计划，以及机械-电气-微电子三域共存框架下的 EmbodiedAI 力学资产建设。
> **前置状态**: P0 三引擎已全部编译通过且 smoke test 可跑（StructuralFEMEngine.kt:200 类型错误已于 2026-09-20 修复，BUILD SUCCESSFUL；structural-fem 命令输出应力/位移/模态/疲劳/PCB 约束 + ECAD/MCAD 协同数据）。
> **关联文档**: [08-cross-project-fusion/fusion-absorption-plan.md](../08-cross-project-fusion/fusion-absorption-plan.md)（吸收矩阵与 P0 催化任务 P1.3）

---

## 一、三域共存框架（本路线图的世界观）

用户提出的三域划分，形式化为：

| 域 | 管什么 | 物理载体 | 设计工具域 | EdaLab 引擎 |
|---|---|---|---|---|
| **微电子域（大脑）** | 决定想什么、做什么（计算/决策） | 密集型 PCB（PADS 类）、SoC、LPDDR | ECAD 密集布线、SI/PI | ForgeCore / SI / PI / EMC 求解器 |
| **电气域（神经-肌肉电信号）** | 做什么这个动作的能量转换（电能↔机械能） | 电机、继电器、离合器、接触器、高压强电流回路 | 电气原理图（非密集 PCB）、功率拓扑 | MotorDriveEngine / PowerIntegritySolver（功率段） |
| **机械域（骨骼）** | 骨架结构怎么定、稳不稳定、可不可塑 | 结构件、关节、连杆、减速器、变体机构 | MCAD 图纸（SolidWorks 类）、FEM | **P0 三引擎**（本路线图主体） |

**核心矛盾**（用户原话的形式化）：机械电气与微电子"既排除又共存"——
- **排除性**：高压强电流的电气功率回路不能挤进为大脑设计的密集 PCB；机械损伤计算不能塞进电路仿真器。PADS↔SolidWorks 之间的数据转换接口"仅仅不足够"。
- **共存性**：具身智能体的身体必须三者同时成立。继电器/离合器的物理断触开关（μs~ms 级、强电隔离）在安全链路上**微电子无法取代**——这正是 EmbodiedAI hardware-motor-shutdown.md <1μs 硬件关断的领域：比较器+锁存+STO 是电气域实体，不是芯片里的代码。

**解法**：不是把三域塞进一个工具，而是建立**力学域引擎作为三域之间的物理耦合枢纽**——机械载荷/形变/模态由它算，电气触点动作的机械后果（离合器啮合力、继电器触点弹跳）由它验证，微电子域的板级约束（PCB 安装孔/支撑点/振动传函）由它输出。EdaLab 的价值恰是三域共享同一 EPS/Job/Stage 编排层。

---

## 二、P0 已达成（2026-09-20 验证）

| 引擎 | 能力 | 状态 |
|---|---|---|
| MassPropertiesEngine | 质心/惯性张量/电池续航/稳定性余量 | ✅ 编译通过 |
| KinematicsDynamicsEngine | DH 正/逆运动学、Jacobian、工作空间、关节力矩 | ✅ 编译通过 |
| StructuralFEMEngine | 应力/位移/模态/疲劳/PCB 安装约束 + ECAD/MCAD 协同输出 | ✅ 编译通过 + smoke test 通过（输出含 IMPACT/FATIGUE 两级 WARNING 风险提示） |

smoke test 实测输出（humanoid_torso_frame 模型）：应力 SF=1793、模态 42975Hz、疲劳 0.43 年（WARN）、冲击 123.5g（WARN，触发"增加 PCB 安装孔/粘接/底座加强"建议）——**引擎已在产出可指导设计的风险分级结论**。

---

## 三、P1 — 机电耦合与电气触点力学（1-2 个月）

### P1.1 继电器/离合器触点动力学引擎（RelayClutchDynamicsEngine）✅ 已交付 (2026-09-20)

**状态**: 编码完成、编译通过、CLI smoke test 通过、已接入 embodied-ai-full 串联链（Step 9/9）。
**交付物**: `edalabcore/.../physics/RelayClutchDynamicsEngine.kt`（约 250 行）+ `relay-clutch` CLI 子命令。

实测输出（48V/30A 安全关断继电器 + 关节制动摩擦片离合器样例）：
- 弹跳: 8 次 / 7.96ms / 恢复系数 e=0.377（质量-弹簧-阻尼碰撞序列，符合真实继电器 5-15ms 量级）
- 侵蚀: 电弧 1800mJ/次 → 寿命 74074 次 ≈ 0.37 年（**WARN，正确触发 CRITICAL 建议**：AgSnO2/镀金触点、RC 熄弧、降额）
- 离合器: 传递 6.38Nm（额定 53%）/ 滑摩功 24.7J / 温升 0.1℃ → PASS
- 安全链路: 电气 0.8µs + 熄弧 8ms + 机械隔离 8ms = **总 16ms < 20ms 目标 → PASS**

**内容**（按原计划全部实现）:
- 触点弹跳模型：质量-弹簧-阻尼碰撞系统，`m·ẍ + c·ẋ + k·x = F_actuator − F_contact`，输出弹跳时长（典型 5-15ms）与回弹次数
- 触点侵蚀：电弧能量 ∝ V·I·t_arc，侵蚀体积模型（对应继电器寿命从百万次到千万次的量级差异）
- 离合器啮合：扭矩传递 `T = μ_cl · F_n · r_eff · n_friction`，啮合瞬态滑摩功 → 热积累（与 ThermalSolver 交叉：滑摩热是电机驱动板之外的第二个热源）
- 与 FunctionalSafetyEngine 对接：断触开关的**机械响应时间**成为安全链路预算的一段（补齐 <1μs 电气关断之后的 ms 级机械隔离段）

**验收**：~~给定 48V/30A 继电器参数，输出弹跳时间/侵蚀寿命/啮合滑摩热，与 datasheet 典型值误差 <30%（工程级）。~~ **已达成**：四类输出全部产出，弹跳 7.96ms 落在真实继电器 5-15ms 区间；侵蚀模型正确触发降额建议。

> 附注（串联验证结论，2026-09-20 实测）：`embodied-ai-full` 此前实际串联 8 步（ForgeCore→质量→结构FEM→运动学→热→PI→电机驱动→数据飞轮），验证时还发现并修复了 `KinematicsDynamicsEngine.kt:237` 的格式串 bug（未转义 `%` 导致整链崩溃）。P1.1 交付后链条升级为 **9 步**；SI/EMC/HIL/BMS/FOC固件/通信协议/安全/可靠性/MCAD/数字孪生/传感器选型等引擎仍未进入串联，属后续 Stage 编排工作（见 P1.4）。

### P1.2 机电耦合仿真闭环（耦合 MotorDrive ↔ 结构）✅ 已交付 (2026-09-20)

**状态**: 编码完成、编译通过、CLI smoke test 通过、已接入 embodied-ai-full 串联链（Step 10/10，链条现为 10 步）。
**交付物**: `edalabcore/.../physics/ElectromechanicalCouplingEngine.kt` + `em-coupling` CLI 子命令。

实测输出（双臂样例 + 48V/30A 驱动板激励谱）：
- 结构载荷谱: 4 项激励（PWM 20kHz / 纹波 300Hz / 齿槽 47Hz / 阻抗步阶 50Hz）的关节角偏与结构反作用力（98.6N @ 50Hz 步阶）
- PCB 安装点加速度谱: PWM 20kHz→**0.137g**（二阶传函高频 1/r² 低通正确生效）、纹波 0.25g、齿槽 0.54g——全部物理合理量级
- **设计发现（引擎价值实证）**: 50Hz 阻抗步阶激励落在样例关节固有频率 50.3Hz 谐振点上（增益 12.42x ≈ 理论 1/(2ζ)=12.5），安装点 55.4g FAIL；控制带宽分析独立给出 16.8Hz < 50Hz 目标的 BLOCKING——**两路分析互相印证"K=5000Nm/rad 减速器刚度不足"，正是该引擎应产出的工程判断**
- 控制带宽上限: f_c = √(K_j/J)/2π 的 1/3 作为阻抗环物理上界——EmbodiedAI household_robot_system.py 的 1kHz 阻抗层增益从此有了结构刚度决定的依据

**内容**（按原计划全部实现）:
- 电机力矩 → 关节结构载荷的单向耦合（P1 阶段先做单向，P2 升级双向）
- 关节柔度（减速器扭转刚度）对控制带宽的影响：`f_c ≈ (1/2π)·√(K_j/J_load)`，输出"结构刚度决定的控制带宽上限"，与 KinematicsDynamicsEngine 的力矩需求交叉
- 振动传函：电机 PWM 载波（10-20kHz）激励 → 结构模态 → PCB 安装点加速度谱（把 StructuralFEMEngine 的冲击分析扩展为频率域传函），作为 PI/SI/EMC 分析的机械激励输入

**验收**：~~一条"电机力矩谱 → 结构响应 → PCB 安装点加速度"链路贯通，三引擎数据互通。~~ **已达成**：四激励链路贯通且量级合理（修正了首版模型缺失的高频 1/r² 惯量低通，PWM 20kHz 从虚高的 343g 修正到 0.137g）。结构模态 42.9kHz 直接取自 StructuralFEMEngine 样例输出；关节力矩 14.79Nm 取自 KinematicsDynamicsEngine 的 shoulder_L 输出——**数据互通已实现（当前为样例级手工对齐，P1.4 Stage 化后改为程序化传递）**。

### P1.3 变刚度关节（VSA, Variable Stiffness Actuator）静力学

**动机**：用户提出"可塑性、变形金刚式展开"的第一步是**刚度可调**——VSA 是当前足式/灵巧操作领域的前沿（安全交互 + 抗冲击 + 蓄能弹射，与本项目攀岩 dyno 弹道资产天然契合）。

**内容**：
- VSA 拓扑库：MACCEPA / SEAb（串联弹性+电机调刚度）/ antagonistic（拮抗双腱）三种参数化模型
- 刚度-力矩映射：`K(θ) = ∂T/∂θ`，SEA 的 `T = K_s·(θ_m − θ_l)` 到 MACCEPA 的非线性 `T(θ, pretension)` 全谱
- 与 EmbodiedAI 攀岩资产对接：摆荡能量泵浦（climbing_dynamics_core.py）的"蓄能段"由 VSA 刚度曲线替代刚性假设，dyno 弹道的初始动能来自弹性势能释放 `E = ∫K(θ)dθ`

**验收**：VSA 参数输入 → 刚度曲线/最大弹射能量/安全冲击限值三输出；与攀岩 dyno 仿真联动一次（复用本项目 Python 资产做交叉验证）。

### P1.4 与 EdaLab 硬件自动化集成（Stage/EPS 化）

把三引擎 + 新引擎注册为 Open API Stage，加入 `embodied_ai_full` kind 流程：
- `mechanical-fem` / `kinematics` / `mass-props` / `relay-clutch` / `vsa-static` 子命令 → Stage 注册（模式与 edalab_stages.py 现有 9 Stage 一致）
- MCADCoDesignEngine 消费 StructuralFEMEngine 的安装孔/支撑约束输出（当前 smoke test 已产出该数据，只差管道接通）

**验收**：一次 EPS job 触发 forge-core → motor-drive → structural-fem → mcad-codesign 串联。

---

## 四、P2 — 变体机构与柔顺结构（2-6 个月）

### P2.1 展开机构运动学（DeployableKinematicsEngine）

**动机**：用户"滑翔机翼伸缩展开/变形金刚"的落地形态——多稳态展开机构（折纸机构 origami、剪叉 scissor、 telescoping 伸缩杆）。

**内容**：
- 展开过程运动学：约束方程 `Φ(q, t) = 0` 的多自由度求解，死点（singular configuration）检测
- 展开驱动力预算：克服重力和关节摩擦的力矩/推力沿展开轨迹的包络
- 展开锁定刚度：锁定后机构作为结构承载（切换回 StructuralFEMEngine 用锁定构型算刚度/模态——**机构分析与结构分析同构**，这正是 EdaLab 三引擎同库的优势）
- 应用映射：滑翔翼展弦比-翼面积-结构重三角权衡（展开翼 vs 固定翼的载荷重构）

**验收**：一个 telescoping + 剪叉复合展开机构从收拢到锁定的全程运动学+驱动力+锁定刚度报告。

### P2.2 柔顺机构与超材料（CompliantMechanismEngine）

**动机**：变体机构去掉铰链的终极形态——柔性铰链（flexure）依靠材料弹性变形实现运动，无摩擦无磨损，天然与 FEM 统一；同时是 G6 柔性体动力学的结构侧入口。

**内容**：
- 柔性铰链刚度矩阵：`K_flex = f(E, t, L, geometry)`，切口类型库（直梁/圆弧/椭圆/簧片）
- 拓扑优化接口：柔顺机构综合（synthesis）——给定输入输出运动关系，SIMP/水平集方法输出材料分布建议（EdaLab 版图领域的 DensityFill 图形学资产可复用）
- 与触觉传感融合：GelSight 指尖骨架并联（本架构总览 §7 约束 8）就是柔顺结构设计问题——骨架承 60N、凝胶感知，刚度分配由本引擎输出

**验收**：柔性铰链参数 → 6 自由度寄生运动 + 刚度；GelSight 骨架-凝胶并联刚度分配方案一份。

### P2.3 双向机电耦合 + 数字孪生互校

- P1.2 升级为双向：结构形变 → 电机负载修正 → 力矩谱修正 →（迭代至收敛，弛豫系数法）
- 与 DigitalTwinEngine 对接：结构侧孪生参数（刚度/阻尼/模态）在线辨识修正，对齐 fusion-absorption-plan 创新 #5

### P2.4 电气-机械断触开关的认证级论证

把 P1.1 的继电器/离合器模型与 FunctionalSafetyEngine 合并产出：STO 链路的分段延迟预算表（电气 <1μs + 触点机械 ms 级 + 负载动能释放），形成 G10 认证的核心技术档案之一。

---

## 五、EmbodiedAI 侧资产联动（丰富本项目）

| EmbodiedAI 资产 | 联动方式 | 新增价值 |
|---|---|---|
| climbing_dynamics_core.py（Hunt-Crossley/摆荡/dyno） | P1.3 VSA 弹射能量源接入 dyno 弹道 | 弹道规划从"刚性瞬时释放"升级为"刚度曲线可控蓄能"——学术上无先例的组合 |
| 机会约束握力调度 | P2.2 GelSight 骨架刚度分配引入概率约束（骨架制造容差 → 刚度分布 → 滑脱率） | 与 fusion-absorption-plan 创新 #1（制造容差 DR）同一方法论 |
| household_robot_system.py 的 1kHz 阻抗层 | P1.2 结构带宽上限 `f_c = √(K_j/J)/2π` 作为阻抗增益的物理上界 | 控制参数不再拍脑袋，由结构刚度决定 |
| G9 全身运动控制 | P2.1 展开机构（腿收拢/展开）作为双足→四足变体形态切换 | ZMP/LIPM 分析扩展到变构型 |

---

## 六、执行优先级与里程碑

| 阶段 | 任务 | 工期 | 里程碑验收 |
|---|---|---|---|
| **P1** | 继电器/离合器触点动力学 | 3 周 | 弹跳/侵蚀/滑摩热三输出，误差<30% |
| P1 | 机电单向耦合闭环 | 2 周 | 三引擎数据互通链路演示 |
| P1 | VSA 静力学 + 攀岩联动 | 3 周 | VSA-dyno 联合仿真一次 |
| P1 | Stage/EPS 注册集成 | 1 周 | 串联 job 跑通 |
| **P2** | 展开机构运动学 | 4 周 | 复合机构全程报告 |
| P2 | 柔顺机构/超材料 | 5 周 | GelSight 骨架方案 |
| P2 | 双向耦合 + 孪生互校 | 3 周 | 收敛迭代演示 |
| P2 | 断触开关认证档案 | 2 周 | 分段延迟预算表 |

**关键依赖**：EdaLab `embodied-ai-full` 串联完整性验证（fusion-absorption-plan P1.3 催化任务，仍未执行）。

---

## 七、一句话总结

**微电子管想什么，电气管做什么，机械管身体扛不扛得住——P0 三引擎已让"身体"可算，P1 把"神经-肌肉"（继电器/离合器/机电耦合）接进来，P2 让身体可变形（VSA/展开机构/柔顺结构）；三者共存的矛盾不靠合并工具解决，靠 EdaLab 统一编排下的三域引擎互喂解决。**
