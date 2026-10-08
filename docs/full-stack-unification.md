# 全栈统一理论 — 控制流 / 物理实体 / 数据流三轴的耦合机制

> **版本**: v3.0 | **日期**: 2026-10-08
> **定位**: 在 `full-stack-layering.md` 的分层骨架之上，给出三轴的**统一数学结构**、**耦合机制**、**可执行的操作规程**
> **前置**: `full-stack-layering.md` (v2.0) 描述"每层装什么"；本文档回答"三根轴为什么要分、如何咬合、怎么用它做工程决策"

---

## 第一部分：为什么必须分三轴 —— 世界观的数学基础

### 1.1 三轴的本体论差异

机器人系统不是"一堆功能堆在一起"，而是**三个不同性质的过程同时发生**：

| 轴 | 本体 | 传递的是什么 | 守恒量 | 方向性 |
|---|---|---|---|---|
| **控制轴 L** | **意图流** | 目标设定（做什么） | 任务完成度 | 自顶向下分解 |
| **物理轴 P** | **能量流** | 力 / 热 / 电 / 物质 | 能量（J） | 双向（驱动+反作用） |
| **数据轴 D** | **信息流** | 观测 / 遥测 / 分布 | 信息熵（bit） | 自底向上抽象 |
| **横切轴 S** | **约束流** | 边界条件（不做什么） | 安全余量 | 横向约束 |

**关键洞察**：三轴的正交性体现在**不可相互还原**：
- 控制指令不能变成扭矩——必须经过 P 层的能量转换
- 扭矩不能变成决策——必须经过 D 层的信息抽象
- 数据不能直接驱动硬件——必须先被 L 层的模型解释
- 任何一层都不能"吞掉"安全边界——S 是独立通道

**推导**：为什么"七层控制栈"无法完整描述机器人？
因为同一个物理实体（比如 P4 驱动板）承载多个控制层（L2 FOC + S 硬件关断），同时产生多种数据流（D0 电流采样 + D1 产线测试）——**一物多轴**。任何单轴分层都会把其他轴的信息折叠丢失。

### 1.2 三轴的时间常数谱

三轴的本质差异还体现在**时间常数跨 8 个数量级**：

```
轴      事件                     时间常数         采样/响应
─────────────────────────────────────────────────────────────
S   硬件比较器锁存              0.8 μs          事件触发
L2  FOC 电流环                  100 μs          10 kHz
L2  PWM 载波                    50 μs           20 kHz
L3  阻抗控制                    1 ms            1 kHz
L1  关节机械响应                10 ms           连续
D0  CAN-FD 帧间隔               100 μs - 1 ms   异步
L4  RL Actor 推理               50 ms           20 Hz
L5  MCTS 接触规划               100 ms          10 Hz
L5  触觉滑脱检测                10 ms           100 Hz
L6  VLA 推理                    15-50 ms        20-60 Hz
L6  任务规划                    1 s             1 Hz
P5  结构件共振响应              20 ms           (模态 50 Hz)
P7  散热时间常数                30-300 s        热惯性
D2  遥测聚合                    1 s - 1 min     批处理
D1  产线 ICT/FCT                分钟 - 小时     离散批次
D3  DR 分布更新                 天 - 周         飞轮周期
D4  策略版本发布                周 - 月         发布周期
P0  MOSFET 老化                 月 - 年         累积
```

**工程含义**：每一层只需要"足够快地响应上层，足够慢地不扰动下层"。当 L4 的 50ms 推理试图直接命令 FOC 100μs 电流环时，中间必须有 L3 的 1kHz 阻抗层做**阻抗匹配**——这是控制理论中的级联控制思想，也是为什么**不能跳层**的原因。

---

## 第二部分：三轴耦合的五种机制

### 机制 1：契约统一（唯一真源）

三轴通过 `contracts/` 目录下的四份 YAML/DBC 文件**同步状态**：

| 契约文件 | 承载的轴间耦合 | 生产方 | 消费方 |
|---------|-------------|-------|-------|
| `hw_requirements.yaml` | L6/L4/L2 需求 → P3/P4/P2 PCB 规格 | EmbodiedAI docs/ | ForgeCore (EdaLab) |
| `safety_budget.yaml` | S 约束 → P4 硬件比较器 / L3 软件阈值 | RelayClutchDynamicsEngine | FunctionalSafetyEngine, household_robot_system.py |
| `dr_params/manufacturing_variance.yaml` | D1 产线容差 → L4 RL 域随机化 | ManufacturingDataFlywheel | BC/PPO/SAC trainer |
| `canfd/joint_control.dbc` | D0 总线格式 ↔ L2 电机命令 | CommProtocolGenerator | 真机 CAN 驱动 |

**关键设计**：契约文件是**三轴之间的唯一握手点**——任何一侧修改接口，都必须更新契约并跑通对侧消费方，否则视为契约违约。这避免了"AI 改了模型但硬件侧不知道"或"硬件换了 MOSFET 但 RL 训练分布没更新"的常见断裂。

### 机制 2：时间常数金字塔 + Nyquist 匹配

上层信号采样率必须满足 Nyquist：**采样频率 ≥ 2 × 被观测过程的带宽**。

```
L4 RL (20 Hz) 要稳定控制 L1 关节 (固有频率 ~50 Hz)？
    → 20 Hz < 2×50 Hz，直接控制会激发共振 → 必须由 L3 (1 kHz) 本地闭环高频，L4 只给 20 Hz 目标

L6 VLA (1 Hz) 要输出动作序列？
    → 无法实时控制，必须输出 action_horizon=10 步，让 L4 逐 50ms 执行

L3 阻抗 (1 kHz) 要读触觉 (100 Hz)？
    → 1 kHz ≥ 2×100 Hz ✓，可以在 L3 内部做滑脱反射，无需上交 L4

L2 FOC (10 kHz) 读电流采样？
    → 采样必须 ≥ 20 kHz，否则 PWM 高频分量混叠
```

这就是为什么 `contracts/hw_requirements.yaml` 的 `control_frequencies_hz` 字段必须和 `base_env.py::EnvConfig.dt` 保持一致——**契约就是时间常数的物理实现**。

### 机制 3：安全预算的分段不可挪用

`contracts/safety_budget.yaml` 定义了三段延迟，每段绑定到一个明确的 P 层实体，**任何软件层不得占用**：

| 段 | 预算 | 绑定实体 | 可否软件实现 |
|---|------|---------|------------|
| `electrical_off` | 0.8 μs | P4 驱动板硬件比较器 + 锁存 | ❌ 不可（核心原则 1） |
| `arc_extinction` | 8 ms | 继电器触点物理熄弧 | ❌ 不可 |
| `mechanical_isolation` | 8 ms | 继电器机械行程 | ❌ 不可 |
| **合计** | **16 ms < 20 ms** | — | — |

任何 L 层（包括 L6 VLA 的"看起来很智能的紧急停车"）**不得**挤占这三段——软件只能触发，不能实现。这是 S 层横切的物理意义：**无论软件栈多智能，物理底线独立存在**。

### 机制 4：制造容差 → RL 训练分布的 DR 注入

这是 D3 → L4 的**闭环创新**，也是本项目区别于"仿真里跑得很好，真机上就废"的核心：

```python
# 产线 ICT 实测某个 MOSFET 的 Rds(on) 批次 σ=15%
# (来自 contracts/dr_params/manufacturing_variance.yaml)

# 传到 BC/PPO/SAC 训练时，域随机化不再"拍脑袋 uniform[0.002, 0.005]Ω"
# 而是按真实分布 normal(μ=0.003, σ=15%) 采样
# → 策略天生对"你手上这批硬件的离散度"鲁棒
```

**链条**：
```
D1 产线 ICT/FCT (真实器件容差)
  ↓ ManufacturingDataFlywheel.kt (Kotlin, EdaLab)
D3 dr_params/manufacturing_variance.yaml (分布参数契约)
  ↓ adapters/edalab.py::parse_stage_output (Python, EmbodiedAI)
L4 RL trainer 的 domain_randomization config
  ↓ 训练得到的 policy
D4 experiments/runs/<version>/ (版本三元组: policy+hw+batch)
  ↓ 部署到真机
D2 遥测 → 回流 D1 (飞轮闭环)
```

### 机制 5：Provenance 四级可信度链

任何数值参数必须标注**来源可信度**，这是"理论文档 vs 实测数据"的诚实边界：

```
Level 0: assumed        — 拍脑袋/文献查表 (e.g. TI UCC21710 datasheet 典型值)
Level 1: modeled        — EdaLab 引擎计算输出 (e.g. StructuralFEM 的应力 SF=1793)
Level 2: modeled_hil    — HIL 台架验证的模型 (e.g. FOC 阶跃响应闭环识别)
Level 3: measured_hil   — 真机直接测量 (e.g. ICT/FCT 产线实测)
```

`safety_budget.yaml::provenance_confidence` 字段就是此机制的实例——当前 v1.1 诚实标注为 `assumed`，`sil2_release_ready: false`。**只有当全部安全链路参数升到 Level 2 或以上，才允许声明 SIL2/PL d 达标**。`safety/formal/budget_closure.py` 按此分级决定是否参与形式化闭环。

---

## 第三部分：三轴交叉点矩阵（36 个耦合格）

把控制层（6）和物理层（9）交叉，得到 54 个格子，再加上数据流方向标注，构成完整系统映射。**每格必须回答三个问题**：谁在这里运行？产生什么数据？受哪些约束？

| | P0 裸片 | P1 子系统板 | P2 执行器 | P3 关节模组 | P4 肢体 | P5 半身 | P6 整机 | P7 外装 | P8 包装 |
|---|---|---|---|---|---|---|---|---|---|
| **L6 VLA** | — | Jetson Orin | — | — | — | — | — | — | — |
| **L5 MCTS** | — | Jetson Orin | — | — | — | — | 攀岩场景 | — | — |
| **L4 RL** | — | Jetson Orin | — | 全身协调 | 双足步态 | 骨盆协调 | 20 Hz 全身 | — | — |
| **L3 阻抗** | MCU 核 | 驱动板 MCU | 1kHz 电流环 | 关节阻抗 | 腕部柔顺 | 躯干稳定 | 1kHz 反射 | — | — |
| **L2 FOC** | SiC FET | DRV835x | 3 相电流 | 编码器闭环 |—|—|—|—|—|
| **L1 物理** | PN 结 | 铜走线 | 磁场-转子 | 谐波减速 | 连杆 | 骨架 | 人形结构 | 皮肤 | 纸箱 |

**横切 S 层投射**（每格都有安全约束，略举三例）：
- L2 × P1：硬件过流 <1μs（在 DRV835x nFAULT 引脚上实现）
- L3 × P3：关节力矩上限（ISO 13482 力阈值）
- L6 × P6：整机行为合法性（ISO 13482 + STL 规约）

**数据流 D 层投射**（每格产生数据，略举三例）：
- L2 × P1 → D0：相电流采样（10 kHz × 3 通道）
- L4 × P5 → D2：骨盆 IMU + 关节编码器聚合遥测
- L6 × P6 → D4：版本三元组（policy_hash + hw_batch + firmware_version）

---

## 第四部分：可执行的操作规程

### 4.1 用三轴定位问题的诊断流程

**示例**：真机上机器人抓杯子时滑脱。

1. **先问哪条轴**：是 L（控制指令错）？P（硬件做不到）？D（感知错了）？S（安全边界触发）？
2. **沿控制轴下探**：
   - L6 VLA 输出的 action 是否合理？ → 检查 action_tokenizer 输出分布
   - L4 RL Actor 的力矩目标是否超过摩擦锥？ → 检查 `chance_constrained_grip.py` 的 μ̂、σ_μ
   - L3 阻抗增益是否过大引起振荡？ → 检查 `household_robot_system.py` 的阻抗参数
3. **沿物理轴排查**：
   - P2 触觉传感器 GelSight 的 Hertz 限值 60N 是否被超过？ → 查 D0 触觉原始数据
   - P4 腕部关节减速器刚度是否偏低（K_j 离散 -20%）？ → 查 `manufacturing_variance.yaml::mechanical.joint_stiffness`
   - P1 驱动板电流环增益是否匹配此批电机？ → 查 D1 ICT 实测 Ld/Lq/Rs
4. **沿数据轴验证**：
   - D0 触觉帧是否 100Hz 稳定？ → 检查 CAN-FD 帧间隔
   - D2 遥测里 μ̂ 是否在线收敛？ → 检查 `multirate_ekf.py` 的 RLS 更新
   - D3 DR 分布是否覆盖本批次容差？ → 对比真机 Ld/Lq 和 yaml 分布的 KL 散度
5. **沿横切轴核对**：
   - S 概率层：f_n 是否满足 ≥ ‖f_t‖/(μ̂ − z·σ_μ)？ → 查 `chance_constrained_grip.py` 日志
   - S 形式化层：STL 规约 `GripStable` 的鲁棒度是否为负？ → 查 `stl_specs.py` 输出

**只有三轴全部核对完，才能确定故障原因并定位到具体修复动作。**

### 4.2 新功能的三轴检查表

添加任何新能力（比如"给灵巧手加第 4 指"）必须回答：

- [ ] **L 轴**：哪个控制层要改？`models/` 里哪些模型需要重训？
- [ ] **P 轴**：哪个物理层要改？PCB/结构/电机/传动是否需要变更？
- [ ] **D 轴**：新增哪些传感数据？`contracts/canfd/` 是否要加 DBC 帧？DR 分布是否要扩展？
- [ ] **S 轴**：新增的安全边界是什么？`safety_budget.yaml` 是否要加段？
- [ ] **时间常数**：新能力的响应时间是否匹配现有 L 层时间金字塔？
- [ ] **Provenance**：所有新参数的 provenance 等级是什么？能升到 Level 2+ 吗？

### 4.3 三轴的 CI 断言（可自动化验证）

本项目已经具备的自动化断言（pytest/verify 双轨）：

```python
# 1. 契约一致性: contracts/ 四件套都能被消费方解析
def test_hw_requirements_parseable(): ...
def test_safety_budget_segments_sum(): ...    # 16ms < 20ms 预算
def test_dr_params_distributions_valid(): ...

# 2. 主链全绿: EdaCli 11 步 + Stage 12 项全部 PASS
def test_edalab_chain_green_11_steps(): ...
def test_stage_registry_12_stages(): ...

# 3. 控制频率匹配: base_env.dt × control_freq 与 L4 20 Hz 一致
def test_env_config_matches_contract(): ...

# 4. 安全预算不可挪用: household_robot_system 中 L3 软件路径 < 20 ms - 16 ms
def test_l3_software_budget_reserve(): ...

# 5. Provenance 分级: sil2_release_ready=true 当且仅当全部段 ≥ modeled_hil
def test_sil2_release_gate(): ...
```

### 4.4 三轴演进的版本号规则

任何一次发布必须同时标注三个轴的版本：
```
policy: v1.2.3     # L6 模型权重 + action_tokenizer 版本
hardware: SF-MDB-0001-rev-C   # P1 驱动板批次 + P4 肢体 rev
dr: ICT-2026-W40   # D3 域随机化分布的数据窗口
```
三者任一变更都构成新"版本三元组"（对应 `experiments/runs/<id>/` 目录），保证真机问题可以回溯到"哪个策略 + 哪批硬件 + 哪次 DR"的精确组合。

---

## 第五部分：当前实现状态对照表

| 轴 | 层 | 已实现 | 部分实现 | 未实现 |
|---|---|-------|---------|-------|
| L | L6 VLA | vla_model/trainer/tokenizer/vision_encoder | 多视角融合（当前 1 视角） | RLHF/预训练 |
| L | L5 MCTS | mcts_integration + climbing_dynamics_core | — | 真实 MCTS 主循环接 L4 |
| L | L4 RL | BC/Diffusion/RLAlgorithms | 全身协调 (whole_body.py) | 双足步态 ZMP 完整闭环 |
| L | L3 阻抗 | household_robot_system.py | 触觉反射（部分） | 1kHz 硬实时 Linux |
| L | L2 FOC | foc_board_align/foc_shutdown (Python 侧) | EdaLab FOCFirmwareGenerator (真 C) | 真机部署 |
| L | L1 物理 | mujoco_env/pybullet_env 骨架 | — | 真实人形本体 |
| P | P0 裸片 | — | — | 全部（外部采购） |
| P | P1 子系统板 | EdaLab ChipReuse/Layout/PCB 导出 | ForgeCore 选型 + 热仿真 | 真实 PCB 打样 |
| P | P2 执行器 | MotorDriveEngine + ElectromechanicalCoupling | FOC 固件生成 | 真实电机联调 |
| P | P3 关节模组 | RelayClutch + MassProps | — | 真实关节台架 |
| P | P4 肢体 | KinematicsDynamics + VsaVariableStiffness | StructuralFEM 应力 | 真实肢体制配 |
| P | P5 半身 | MultiPhysicsCoupler | — | 真实半身装配 |
| P | P6 整机 | 整机样例输入（引擎里 hardcode） | — | 真实整机 |
| P | P7 外装 | — | — | 全部 |
| P | P8 包装 | — | — | 全部 |
| D | D0 原始传感 | canfd/joint_control.dbc | hil/twin_fft.py | 真机数据流 |
| D | D1 产线 | ManufacturingDataFlywheel (ICT/FCT 入口) | — | 真实产线 |
| D | D2 遥测 | adapters/bigdata.py 骨架 | adapters/flywheel_live.py | 真实聚合 |
| D | D3 DR | contracts/dr_params/manufacturing_variance.yaml | ManufacturingDataFlywheel | 真实批次数据 |
| D | D4 部署 | experiments/runs/ 目录规范 | — | 版本三元组自动化 |
| S | 确定性 | foc_shutdown.py (设计) + safety_budget.yaml | RelayClutch 引擎验证 | 真机 STO 实测 |
| S | 概率性 | chance_constrained_grip.py + multirate_ekf.py | — | 真机 μ 估计收敛 |
| S | 形式化 | stl_specs + reachability + iso13482 | budget_closure | 覆盖全栈规约集 |
| S | 物理引擎 | EdaLab 19+ 引擎 (P0/P1 四大类) | P1.4 Stage 化（Python 侧 12 项已注册） | HIL 实测闭环 |

---

## 第六部分：三轴融合后的统一判定准则

一个功能/模块/系统是否"产品级"，必须同时通过三轴 + 横切的**四问**：

1. **L 问**：指令能否从 L6 一路分解到 L2 且每层时间常数符合 Nyquist？
2. **P 问**：每个 L 层的物理载体（PCB/电机/结构）是否已有 EdaLab 引擎仿真背书？
3. **D 问**：运行产生的数据能否沿 D0→D4 闭环回到训练，且 DR 分布覆盖真实容差？
4. **S 问**：所有危险动作是否都在 safety_budget.yaml 的分段延迟内可关断，且 provenance ≥ modeled？

**四问全过 → 产品级；缺任一 → demo 级玩具。**

这就是"控制/物理/数据三轴 + 安全横切"的最终判据，也是本项目的北极星。

---

## 附：三轴与其他框架的对应关系

| 本项目 | ROS 2 对应 | ISO 13482 对应 | Figure AI Helix 对应 |
|-------|-----------|--------------|-------------------|
| L6 VLA | 上位任务规划 | 行为合法性 | S2 System 2 |
| L5 MCTS | 行为树/规划器 | — | S2 中间层 |
| L4 RL | MoveIt/控制器 | 速度/力限制 | S1 全身策略 |
| L3 阻抗 | ros2_control | 力/功率限制 | S1 快速反射 |
| L2 FOC | 硬件驱动 | 急停接口 | 嵌入式 |
| P 轴 | URDF/机械 | 机械安规 | 自研执行器 |
| D 轴 | rosbag/遥测 | 日志合规 | 数据飞轮 |
| S 轴 | safety_node | ISO 13482 全部 | 硬件 STO |

---

**一句话收束**：分层是"装什么"，三轴是"为什么这样分"，耦合机制是"怎么咬合"，操作规程是"怎么用"，统一判据是"什么时候算完"。
