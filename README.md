# EmbodiedAI Roadmap — 端到端具身智能体研发计划

> **项目代号**: EmbodiedAI
>
> **定位**: 不拼装现有方案，从零构建"AI 神经网络模型 → PCB 硬件 → 电机驱动 → 具身智能体"全链条的端到端研发计划。
>
> **版本**: v0.3 | 更新日期: 2026-09-19

---

## 核心原则

1. **全栈自研** — AI 模型从 tokenizer 到训练管线全部自研；硬件从原理图到 PCB 全部自研
2. **AI × 硬件深度耦合** — 端到端神经网络模型直接映射到硬件执行，不做"仿真-现实鸿沟"
3. **渐进式规模** — 从算力受限嵌入式到高算力边缘计算，分阶段递进
4. **安全优先** — 电机驱动必须有硬件级自动关机保护，不能仅靠软件
5. **决策核心** — 强化学习的数学本质指导具身智能体训练，触觉为最稀缺感知
6. **操作系统驱动** — 嵌入式 Linux on ARM 把 AI 决策变成电机旋转

---

## 文档结构 (v0.3)

### 入门与新文档

| 文档 | 内容 |
|---|---|
| [docs/plan.md](docs/plan.md) | **主计划**: 13 章端到端研发计划 (v0.2) |
| [docs/architecture-overview.md](docs/architecture-overview.md) | ★ **新增** ★ 系统架构总览: 7层架构、感知系统、核心方程、实时性预算 |
| [docs/future-exploration-plan.md](docs/future-exploration-plan.md) | ★ **新增** ★ 下一步未来探索计划: 6-18个月分阶段路线图 |
| [CHANGELOG.md](CHANGELOG.md) | ★ **新增** ★ 版本变更日志与关键决策记录 |

### 01 — AI 模型 (`docs/01-ai-model/`)

| 文档 | 内容 |
|---|---|
| [vla-model-design.md](docs/01-ai-model/vla-model-design.md) | VLA (Vision-Language-Action) 模型架构设计 |
| [reinforcement-learning.md](docs/01-ai-model/reinforcement-learning.md) | 强化学习深度理论: MDP → 贝尔曼 → 策略梯度 → Actor-Critic |
| [algorithm-architecture-decision.md](docs/01-ai-model/algorithm-architecture-decision.md) | 20Hz 家庭服务场景的算法架构决策: 分层混合架构 |
| [multimodal-fusion-jetson.md](docs/01-ai-model/multimodal-fusion-jetson.md) | MultiModalFusionPolicy 在 Jetson Orin NX 上的实时性可行性分析 |

### 02 — 接触动力学 (`docs/02-contact-dynamics/`)

| 文档 | 内容 |
|---|---|
| [contact-dynamics-optimization-1.md](docs/02-contact-dynamics/contact-dynamics-optimization-1.md) | 接触动力学优化: 从简化模型到实用公式 (Hunt-Crossley + SOC 摩擦锥) |
| [contact-dynamics-optimization-2.md](docs/02-contact-dynamics/contact-dynamics-optimization-2.md) | 接触动力学优化: 分层 A*/QP/反馈 + MPC 集成 |
| [climbing-dynamics-math.md](docs/02-contact-dynamics/climbing-dynamics-math.md) | ★ **核心** ★ 攀岩竞技动力学与触觉-视觉融合: 完整数学基础 (浮基→脉冲→竞技) |
| [tactile-fusion-planning.md](docs/02-contact-dynamics/tactile-fusion-planning.md) | 触觉融合与接触规划: MCTS+RL+Diffusion 分层架构与实现 |

### 03 — 感知与传感 (`docs/03-sensing-perception/`)

| 文档 | 内容 |
|---|---|
| [tactile-sensing.md](docs/03-sensing-perception/tactile-sensing.md) | 触觉传感稀缺资源: 传感器技术、信号处理、滑脱检测 |
| [tactile-sim2real.md](docs/03-sensing-perception/tactile-sim2real.md) | 触觉 Sim-to-Real 策略选择: TACTO/Taxim/系统辨识+LoRA |
| [tactile-fusion-conclusion.md](docs/03-sensing-perception/tactile-fusion-conclusion.md) | 触觉-视觉融合如何有效应用于攀岩机器人: 分析结论与落地清单 |

### 04 — 硬件 (`docs/04-hardware/`)

| 文档 | 内容 |
|---|---|
| [hardware-motor-shutdown.md](docs/04-hardware/hardware-motor-shutdown.md) | 自动关机电机驱动的完整硬件设计 (<1μs响应) |
| [embedded-linux-arm.md](docs/04-hardware/embedded-linux-arm.md) | 嵌入式 Linux on ARM 全栈: 汇编→驱动→UBOOT→内核→调试 |
| [nn-to-hardware-pipeline.md](docs/04-hardware/nn-to-hardware-pipeline.md) | 从神经网络到硬件执行的全链路编译映射 |
| [figure-ai-benchmark.md](docs/04-hardware/figure-ai-benchmark.md) | Figure AI 深度对标: 技术路线、差距分析、借鉴点 |

### 05 — 系统软件 (`docs/05-system-software/`)

| 文档 | 内容 |
|---|---|
| [os-action-planning.md](docs/05-system-software/os-action-planning.md) | OS 层动作规划: ROS2、实时调度、任务状态机、可解释性 |
| [gui-interface.md](docs/05-system-software/gui-interface.md) | 上层 GUI 交互界面: Qt/QML、3D 可视化、语音交互 |

### 06 — 数据策略 (`docs/06-data-strategy/`)

| 文档 | 内容 |
|---|---|
| [data-collection-strategy.md](docs/06-data-strategy/data-collection-strategy.md) | 数据采集策略: 攀岩 vs 家庭服务的配比分析 |
| [climbing-to-household-migration.md](docs/06-data-strategy/climbing-to-household-migration.md) | 攀岩接触规划迁移到家庭场景: 力闭合+安全约束 |

### 07 — 差距分析 (`docs/07-gap-analysis/`)

| 文档 | 内容 |
|---|---|
| [README.md](docs/07-gap-analysis/README.md) | 差距分析索引 + 缺失环节速查 |
| [project-gap-comprehensive-report.md](docs/07-gap-analysis/project-gap-comprehensive-report.md) | ★ **综合** ★ 项目全面分析: G1-G5 定位 + 补充方案 |
| [gap-analysis-algorithm-architecture.md](docs/07-gap-analysis/gap-analysis-algorithm-architecture.md) | 算法架构差距分析 |
| [gap-analysis-contact-planning.md](docs/07-gap-analysis/gap-analysis-contact-planning.md) | 接触规划差距分析 |
| [gap-analysis-data-strategy.md](docs/07-gap-analysis/gap-analysis-data-strategy.md) | 数据策略差距分析 |
| [gap-analysis-dynamics-hardware.md](docs/07-gap-analysis/gap-analysis-dynamics-hardware.md) | 刚柔耦合+DIY硬件差距分析 |
| [gap-analysis-realtime-tactile.md](docs/07-gap-analysis/gap-analysis-realtime-tactile.md) | 实时触觉差距分析 |
| [gap-analysis-sim2real.md](docs/07-gap-analysis/gap-analysis-sim2real.md) | Sim-to-Real 差距分析 |

---

## 代码

| 文件 | 内容 |
|---|---|
| [src/core/climbing_dynamics_core.py](src/core/climbing_dynamics_core.py) | ★ 核心 ★ 攀岩竞技动力学纯 numpy 实现 (Hert接触 + 摆荡 + dyno + EKF + 滑脱前兆) |
| [src/core/mcts_integration.py](src/core/mcts_integration.py) | NeRF 增强的 MCTS 攀岩规划 + NeRF 价值网络训练 |
| [src/control/household_robot_system.py](src/control/household_robot_system.py) | 完整家庭服务机器人系统: 4层架构 (MCTS + Diffusion + 阻抗 + FOC) |

---

## 系统架构 (7 层)

```
L7 用户交互层:  自然语言 / GUI / 语音
L6 任务规划层:  VLA / LLM → 子任务序列 (~1Hz)
L5 接触序列层:  MCTS + 学习增强 (2-10Hz)
L4 运动控制层:  RL Actor / Diffusion Policy (20Hz)
L3 反射伺服层:  阻抗控制 + 触觉反射 (1kHz)
L2 电机驱动层:  FOC + SVPWM + 硬件自动关断 (10kHz)
L1 物理执行层:  电机 + 减速器 + 传感器
```

详见 [docs/architecture-overview.md](docs/architecture-overview.md)

---

## 13 章总览 (v0.2, 见 plan.md)

```
第 0 章: 具身智能体技术全景与现状审计
第 1 章: 端到端 AI 模型研发计划（神经网络）
第 2 章: 机器人硬件研发计划（PCB + 电机 + 传感器）
第 3 章: 自动关机电机驱动 — 硬件安全设计专题
第 4 章: 从硬件角度设计具身智能体（PCB 依赖全景）
第 5 章: 神经网络模型到硬件执行的编译映射
第 6 章: Figure AI 对标与中国版 Figure 路径
第 7 章: 分阶段实施路线图与里程碑
第 8 章: 验证评估体系与风险分析
第 9 章: 嵌入式 Linux on ARM 全栈
第10 章: 强化学习 — 决策算法的数学本质
第11 章: 触觉传感 — 具身最稀缺的感知
第12 章: OS 层动作规划与可解释性
第13 章: 上层 GUI 交互界面
```

---

## PCB 依赖全景

| 层级 | PCB 承载 | 关键芯片/模块 | 层数 |
|---|---|---|---|
| L1 电源层 | 电源管理 PCB | PMIC、BMS、电机驱动电源轨 | 4 层 |
| L2 感知层 | 传感器融合 PCB | IMU、摄像头接口、力传感器 ADC | 6 层 |
| L3 决策层 | 主控 PCB | Jetson Orin / RK3588 + LPDDR5 | 8 层 |
| L4 执行层 | 电机驱动 PCB | STM32 + DRV8301 + 自动关机电路 | 4 层 |
| L5 互联层 | 通信背板 | CAN-FD / RS485 / Ethernet | 2 层 |

---

## 技术里程碑

```
Phase 0 (M1-M2)     Phase 1 (M3-M5)       Phase 2 (M6-M9)       Phase 3 (M10-M14)
  ████████          ██████████████████    ████████████████████   ██████████████████████
  AI基+HW+Linux     单臂+RL+触觉+GUI        双臂+全身皮肤            通用人形
  嵌入式Linux启动     行为克隆               Sim-to-Real            终身+迁移
  UBOOT+内核+驱动    BC→PPO→SAC           力控+滑脱恢复           安全认证
  FOC电机转动+自动  决策日志+可解释        语音+远程GUI            RL超越人类
  关机               解释                  多模态VLA               LLM+RL融合
```

---

## 核心公式

```python
# 贝尔曼方程 (长期决策的递归分解)
Q_star(s,a) = E[r + γ · max_a' Q_star(s',a')]

# PPO 裁剪目标 (限制策略变化幅度)
L = min(r·A, clip(r, 1-0.2, 1+0.2)·A)

# Hunt-Crossley 软接触力
f = k · δ^1.5 · (1 + b · δ̇)

# 二阶锥摩擦锥约束
‖f_tangent‖₂ ≤ μ · f_normal

# 滑脱检测不等式 (触觉判断)
slipping if ||F_tangent|| > µ · F_normal

# FOC 三相电流约束 (功率电子)
I_a + I_b + I_c = 0,  I_phase ≤ I_max (自动关机 < 1μs)

# 实时调度约束 (操作系统)
worst_case_latency < 100µs (控制周期 1ms 的 10%)

# 浮基动力学 (攀岩机器人本体)
M(q)ν̇ + Cν + g = Sᵀτ + Σ J^{c_i}ᵀ λ_i

# Hertz 接触刚度
k = (4/3) · E* · √R*

# 机会约束握力调度 (μ 不确定性)
f_n ≥ ‖f_t‖ / (μ̂ − z · σ_μ)
```

---

## 已知关键约束

```
1. 电机自动关机响应 < 1μs (硬件级, 不可软件实现)
2. 内核实时调度 jitter < 100µs (PREEMPT_RT + isolcpus)
3. RL 训练需 1000x 仿真迭代 + 100x 真实微调
4. GelSight 输出 921KB/帧 → 需要 GPU 预处理
5. 触觉传感器成本是视觉 50-500x → 数据稀缺
6. 端到端 VLA 推理延迟必须 < 50ms (20Hz 控制)
7. GelSight 软指尖 Hertz 载荷极限 ~60N@5mm → 需骨架并联方案
8. μ 不确定性: 无裕度点估计滑脱率 50% → 需机会约束
```

---

## 下一步行动

详见 [docs/future-exploration-plan.md](docs/future-exploration-plan.md)

30 天快速启动:
1. 搭建 Isaac Sim / MuJoCo 仿真环境
2. 导入 URDF (机械臂 + 桌面 + 杯子)
3. 实现行为克隆训练管线
4. 第一次 Sim-to-Real 迁移尝试

---

## 项目对齐

| 来源 | 提供什么 |
|---|---|
| LLMInnovation | AI 模型研发路线（Phase 0-6，从 tokenizer 到 100B+ 多模态） |
| EdaLab | 硬件设计能力（原理图 → PCB → 仿真 → 芯片设计） |
| Figure AI | 对标公司（Helix 2.5 端到端模型、Figure 03 人形机器人） |
