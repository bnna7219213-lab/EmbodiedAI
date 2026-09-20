# MultiPhysicsCoupler — 三域耦合联合仿真引擎

> **代号**: P0-C34 | **日期**: 2026-09-20
> **定位**: 解决"三域共存"矛盾的核心计算引擎

---

## 矛盾本质

三域共存 (微电子-电气-机械) 的接口**不仅是**数据格式转换 (IDF/STEP)，
**更是**物理耦合的联合仿真与约束传播:

```
• 能量流: 电池(化学) → 电气(48V/30A) → 电机(电磁) → 机械(力矩/运动) → 接触(力)
• 信号流: 传感器 → ADC → 芯片计算 → PWM/继电器控制 → 能量流开关
• 安全链: 芯片决策 → 看门狗 → 继电器/离合器物理分断 (确定性层)
```

现有 MCADCoDesignEngine 仅做几何数据交换 (IDF/STEP 文件生成 + 干涉检查)，
缺少**物理场间的双向耦合**。MultiPhysicsCoupler 填补这一空缺。

---

## 架构总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        MultiPhysicsCoupler                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐    │
│  │ ⚡ 电气域     │◄──►│ 🔥 热域       │◄──►│ ⚙️ 机械/电磁域           │    │
│  │ MnaSolver    │    │ ThermalSolver│    │ KinematicsDynamics       │    │
│  │ PI / SI / EMC│    │              │    │ MassProperties           │    │
│  │ MotorDrive   │    │              │    │ StructuralFEM            │    │
│  └──────┬───────┘    └──────┬───────┘    └────────────┬─────────────┘    │
│         │                   │                         │                  │
│         └───────────────────┼─────────────────────────┘                  │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                 约束传播引擎 (Constraint Propagator)               │   │
│  │  温度 → 电阻 → 电流 (正反馈)                                      │   │
│  │  PWM频率 → dV/dt → EMI → 滤波需求 → 母线电容                      │   │
│  │  关节力 → 形变 → 编码器信号偏移                                    │   │
│  │  接触刚度 → 触觉传感器带宽需求                                     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  🔋 能量流耦合    📡 信号流耦合    🛡️ 安全链耦合 (确定性层)        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 三大物理流

### 1. 能量流耦合 (Energy Flow)

**链路**: 电池化学能 → 电气功率 → 电磁转矩 → 机械功 → 接触力

```
电池(48V/5Ah) ─[Rint=0.12Ω]─► 母线 ─[PCB R=5mΩ]─► MOSFET ─[Rds=3mΩ]─►
                                                                         │
                                                                             ▼
                                                    三相逆变器 ─► BLDC电机 ─[Ke=0.08]─►
                                                                                             │
                                                                                                 ▼
                                                    接触力 ◄─ 连杆负载 ◄─ 减速器 ◄─[i=100 η=82%]◄─
```

**关键参数**:
- MOSFET Rds(on) 温度系数: +0.4%/°C
- 绕组电阻温度系数: +0.393%/°C (铜)
- 电池内阻低温效应: <10°C 时内阻非线性急剧上升
- 开关损耗: ∝ f_sw × Vds × I × (tr+tf)

**输入**: `EnergyFlowInput` (电池+电气+电机+机械+接触)
**输出**: `EnergyFlowResult` (功率分解+效率+温度风险警告)

### 2. 信号流耦合 (Signal Flow)

**链路**: 物理量 → 传感器 → 电缆 → ADC → 算法 → PWM → 驱动器

```
物理量(力/位置/电流)
    │
    ▼
传感器 (灵敏度, 本底噪声)
    │
    ▼
电缆 (长度, EMI耦合, 电容)
    │
    ▼
ADC (分辨率, SNR, 采样率)
    │
    ▼
算法 (计算延迟, 控制频率)
    │
    ▼
PWM → 栅极驱动 → MOSFET 开关
```

**关键约束**:
- 信号带宽必须 < ADC Nyquist 频率 (采样率/2)
- PWM 频率会串扰 ADC 采样 (高 dV/dt 耦合)
- 总延迟必须 < 控制周期 (1kHz → <1ms)
- EMI 背景场强 × 电缆长度 决定耦合噪声水平

**输入**: `SignalFlowInput` (传感器+ADC+计算+PWM)
**输出**: `SignalFlowResult` (SNR, 延迟, 带宽, EMI 敏感度)

### 3. 安全链耦合 (Safety Chain) — 确定性层

**链路**: 故障检测 → 独立判断 → 硬件执行 → 物理分断

```
故障源 (过流/过温/通信中断/编码器失效/...)
    │
    ▼
检测 (传感器→比较器 < 0.3μs 硬件独立)
    │
    ▼
逻辑 (独立看门狗 MCU 中断响应)
    │
    ▼
执行 (MOSFET STO < 0.5μs / 继电器 / 动态制动 / 机械抱闸)
    │
    ▼
物理分断 (功率路径被硬切断)
```

**确定性保证**:
- STO 响应 < 1μs (硬件级，不依赖软件)
- 双通道冗余 + 诊断覆盖率 >99%
- 独立供电 (主电源失效仍可执行)
- SIL2/3/4 达标: <10ms 完整停止
- 停止角位移 <30° (人机共存安全)

**输入**: `SafetyChainInput` (安全等级+故障源+执行器+负载)
**输出**: `SafetyChainResult` (停止时间+PFH+SFF+HFT+确定性判断)

---

## 约束传播算法

MultiPhysicsCoupler 的核心算法: 迭代求解三域耦合方程直到收敛。

### 耦合变量表

| 变量 | 源域 | 影响域 | 传播方向 | 公式 |
|------|------|--------|----------|------|
| T_mosfet | 热 | 电气 | 反馈 | Rds(T) = Rds₂₅ × (1 + 0.004×(T-25)) |
| T_winding | 热 | 电气 | 反馈 | Rₚ(T) = R₂₅ × (1 + 0.00393×(T-25)) |
| T_battery | 热 | 电气 | 反馈 | Rint低温非线性 |
| f_pwm | 电气 | EMI/热 | 前向 | EMI ∝ log₁₀(f_sw), P_sw ∝ f_sw |
| J_sensor | 机械 | 信号 | 前向 | 振动 → 传感器噪声↑ |
| F_contact | 机械 | 信号 | 前向 | 接触力频率 → 带宽需求 |

### 迭代过程

```
1. 初始独立域分析 (无耦合)
2. 迭代直到收敛:
   a. 读取各节点温度 → 更新 Rds, R_phase, R_int
   b. 计算 EMI 水平 → 必要时降频 PWM
   c. 检查 EMI 敏感度 → 调整控制参数
   d. 重新计算功率损耗 → 更新温度场
   e. 检测残差 < 阈值 → 收敛
3. 交叉域一致性检查
4. 输出收敛结果 + 残差日志
```

### 收敛判据

- 最大相对残差 < 0.1% (参数变化量 / 参数值)
- 最大迭代次数: 50
- 振荡检测: 连续 4+ 步同参数方向翻转

---

## 交叉域约束库

```kotlin
// 默认注入的 5 条跨域约束
CrossDomainConstraint(
    name = "thermal_electrical_resistance",
    source = THERMAL, target = ELECTRICAL,
    direction = BIDIRECTIONAL,
    severity = CRITICAL,
    description = "温度升高 → 电阻增大 → 损耗增大 → 温度升高 (正反馈热失控)"
)

CrossDomainConstraint(
    name = "emi_pwm_derating",
    source = ELECTRICAL, target = ELECTRICAL,
    direction = FORWARD_FEED,
    severity = WARNING,
    description = "PWM频率↑ → EMI↑ → 需降频 → 热损耗↑ → 温度↑"
)

CrossDomainConstraint(
    name = "signal_delay_safety",
    source = ELECTRICAL, target = MECHANICAL,
    direction = BACKWARD_FEED,
    severity = BLOCKING,
    description = "信号延迟 > 安全检测窗口 → 保护动作滞后 → 不安全"
)

CrossDomainConstraint(
    name = "force_sensor_emc",
    source = MECHANICAL, target = ELECTRICAL,
    severity = INFO,
    description = "接触力信号链 EMI ↑ → 滑脱检测误判"
)
```

---

## 与 EmbodiedAI 7 层架构对齐

| 层 | 物理域 | MultiPhysicsCoupler 覆盖 |
|----|--------|-------------------------|
| L1 物理执行 | 机械+电磁+接触 | ✅ 能量流末端 (电机→减速器→连杆→接触力) |
| L2 电机驱动 | 电气+热+安全 | ✅ 能量流中段 (逆变器+MOSFET+电机) |
| L3 反射伺服 | 信号+控制 | ✅ 信号流 (传感器→算法→PWM) |
| L4 运动控制 | 算法+决策 | ⚠️ 接口 (安全链触发条件) |
| L5 接触序列 | 几何+力 | ⚠️ 输入 (接触刚度→传感器带宽需求) |
| L6-L7 | AI / 交互 | ❌ 不涉及 (上层由 EmbodiedAI 主线承担) |

---

## 代码位置

```
EmbodiedAI/
├── edalabcore/src/main/kotlin/edalab/core/coupling/
│   ├── MultiPhysicsCoupler.kt    ← 核心引擎 (~1050 行, 5 个数据类家族)
│   └── CouplingCliDemo.kt        ← CLI 演示入口
└── docs/04-hardware/
    └── multi-physics-coupling-engine.md  ← 本文档
```

---

## 设计决策

| # | 决策 | 理由 |
|---|------|------|
| 1 | 迭代求解 (非联立) | 各域物理模型复杂度差异巨大，联立不现实 |
| 2 | 耦合参数隔离 (paramState) | 避免修改不可变的 Input 数据类 |
| 3 | 局部类型定义 (Vec3, JointSpec_local) | 自包含性 — 完整集成到 edalabcore 后合并 |
| 4 | 安全链独立确定性判断 | 不参与热迭代 (安全性能降级，不允许收敛摆尾) |
| 5 | 50 次迭代上限 | 典型收敛 5-15 次，防死循环 |

---

## 下一步

- [ ] 集成到 EdaCli 主入口 (cmdCouplingAnalysis)
- [ ] 新增 API Stage: CouplingSimulationStage
- [ ] 合并局部类型到 physics 包 (Vector3f, MaterialType)
- [ ]  ForgeCore 接入 — 将耦合仿真结果反馈给 AI→Hardware 翻译
- [ ] HIL Bridge 集成 — 仿真结果与真实硬件数据对比验证
- [ ] 约束库扩展至 15+ 条 (覆盖更多 P0/P1 场景)

---

> **与 EdaLab 原项目的关系**: MultiPhysicsCoupler 是一个**新层** (耦合层)，
> 而非替代原有的电气/热/机械独立求解器。它建立在现有引擎之上，
> 通过迭代调用各域求解器并注入耦合修正，实现真正的三域联合仿真。
