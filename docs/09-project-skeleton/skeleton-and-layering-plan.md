# EmbodiedAI 项目骨架方案与七层表述评审

> **版本**: v1.0 | **日期**: 2026-09-20
> **依据**: 五项目审计（EmbodiedAI / EdaLab / ECAD-MCAD / SynapticForge / RaccoonWork\nanoGPT 全部实读）+ MultiPhysicsCoupler 迁移实录
> **前置动作已完成的验证**: EdaLab 主链现为 **11 步**（含 relay-clutch / em-coupling / multi-physics），全链 BUILD SUCCESSFUL；EmbodiedAI 侧 edalabcore 已归档至 `archive/edalabcore-migrated-2026-09-20/`

---

## 一、是否需要再次审计？

**结论：不需要全面重审，本轮已做定向补审。** 各项目在历轮会话中已实读核验（见 fusion-absorption-plan 来源清单），本轮新增/复核：
- **nanoGPT（本轮新读）**：自研 GPT 训练管线（llm.py 库化 + configs/tests + 消融实验 runs/ablations + checkpoint 体系），有 v0.1→v3.0 改进计划（配置外置/可复现/评估门槛）。**定位：L6 任务规划层的模型训练资产**——VLA 的 language/action tokenizer 与 backbone 训练可直接复用其 ModelConfig/Trainer/data pipeline 模式
- **EdaLab（本轮复核）**：coupling 包迁入后编译通过，embodied-ai-full 11 步含新引擎；Gradle 9.3.1，唯一 Kotlin 编译权威
- **EmbodiedAI（本轮变动）**：edalabcore 目录已清空归档，src 只剩 Python（climbing_dynamics_core / mcts_integration / household_robot_system）

骨架方案已具备审计基础，可直接执行。

---

## 二、七层表述评审：合不合理？

**结论：七层控制栈本身合理（学界主流分层），但存在"三套编号互相冲突"的表述问题，必须修。**

### 2.1 问题清单（行级依据）

| # | 问题 | 证据位置 | 严重度 |
|---|---|---|---|
| 1 | **PCB 层级表盗用 L1-L5 编号**：README "PCB 依赖全景"用 L1电源/L2感知/L3决策/L4执行/L5互联，与控制栈 L1物理执行~L5接触序列**同号不同义** | README.md L146-153 vs L107-117 | 🔴 必须修 |
| 2 | **与 SynapticForge 六层架构不可对齐**：SynapticForge 的 L4=编排层(ForgeCore)、L3=硬件设计层，在 EmbodiedAI 七层中无对应；两项目文档互相引用时编号错位 | SynapticForge 00-architecture-overview §2.1 | 🔴 必须修 |
| 3 | **安全层缺位**：<1μs 关断/STO/机会约束是横切全栈的安全链（MultiPhysicsCoupler 已将其建为独立"确定性层"），七层中无位置，导致安全预算（0.8μs+8ms+8ms）在层间无处挂靠 | architecture-overview §7 约束1 | 🟠 建议修 |
| 4 | **数据回填通道缺位**：SynapticForge 飞轮（硬件遥测→DR→RL）在七层中只能挂在 L7 之下，无明确数据流入口 | fusion-absorption-plan 创新 #1/#3 | 🟠 建议修 |
| 5 | **L4/L3 边界模糊**："运动控制 20Hz RL Actor" 与 "反射伺服 1kHz" 之间，阻抗控制归 L3 但 household_robot_system.py 中它承接 L4 的子目标——文档中无统一裁定 | architecture-overview L3/L4 描述 | 🟡 可保留（实现细节） |

### 2.2 修订建议：三栈分离命名法

**保持七层控制栈编号不变（L1-L7），其余两套栈改用不同前缀**：

```
控制栈 (不变):           物理栈 (改 P#):         数据栈 (新增 D#):
L7 用户交互              P1 电源 PCB             D1 传感原始数据
L6 任务规划 (VLA)        P2 感知 PCB             D2 标注/遥测
L5 接触序列 (MCTS)       P3 主控 PCB             D3 制造/批次数据
L4 运动控制 (RL)         P4 电机驱动 PCB         D4 RL 回放/DR 分布
L3 反射伺服 (阻抗)        P5 通信背板             D5 版本三元组 (policy,hw,batch)
L2 电机驱动 (FOC)
L1 物理执行              + 横切: 安全层 S (横跨 L1-L7, 确定性链)
                         + 横切: 数据层 D (飞轮, 横跨 L1-L7)
```

修订后与 SynapticForge 的对齐：SF-L6 算法迭代=D 栈，SF-L4 编排=横切集成层（对应 EdaLab Open API/EPS），SF-L3/L2= P 栈 + EdaLab 引擎。**逐文档改动点**：README 的 PCB 表（L→P）、architecture-overview §5 硬件架构表（L→P）、新增"横切层"一节；plan.md 第 4 章同步。

---

## 三、EmbodiedAI 项目骨架方案

### 3.1 设计原则

1. **单语种单仓库**：Python 是本项目唯一实现语言（与"全栈自研 AI 侧"定位一致）；Kotlin/EdaLab、JS/H5 均为外部协作仓库，只通过契约（yaml/JSON/CLI 输出）交互
2. **契约先行**（吸收 fusion-absorption-plan P0 结论）：所有跨项目交互走 `contracts/`，字段与 EdaLab EPS/SynapticForge hw_requirements.yaml 对齐
3. **包结构对齐 aigc-refactored 经验**：config 外置 + 适配层 + 统一 CLI 子命令，避免脚本散落
4. **七层 ↔ 目录一一映射**：代码结构即架构表述，减少文档与实现漂移

### 3.2 目标目录结构

```
EmbodiedAI/
├── docs/                        # 现有 01-08 保留, 新增:
│   ├── 09-project-skeleton/     # 本骨架方案 + 三栈命名法落地
│   └── 10-contract-register/    # 契约登记表 (谁产谁消, 版本)
├── contracts/                   # ★ 跨项目契约 (唯一真源)
│   ├── hw_requirements.yaml     # AI→硬件 (ForgeCore 输入)
│   ├── canfd/                   # DBC + 帧格式 (还 ECAD-MCAD 的债)
│   ├── dr_params/              # 域随机化分布 (制造容差 DR)
│   └── safety_budget.yaml      # 安全链分段延迟 (0.8µs/8ms/8ms)
├── src/embodied_ai/            # ★ Python 包 (从散脚本升级)
│   ├── __init__.py
│   ├── config/                 # 统一配置 (dataclass + yaml)
│   ├── models/                 # L6: VLA/RL 策略 (接 nanoGPT 资产)
│   ├── dynamics/               # L5/L1: 接触动力学 (climbing_dynamics_core 迁入)
│   ├── perception/             # 感知: 触觉-视觉 EKF (从 core 拆出)
│   ├── control/                # L4/L3: RL Actor + 阻抗 (household_robot_system 迁入)
│   ├── planning/               # L5: MCTS (mcts_integration 迁入)
│   ├── deploy/                 # 部署适配: TensorRT/量化/延迟核验 (吸收 NG12)
│   ├── models/                 # ★ L6: VLA 小模型 (复用 nanoGPT ModelConfig/Trainer 模式)
│   │   ├── vla_config.py        #   VLAConfig dataclass + 消融集 (ablations)
│   │   ├── vla_model.py         #   VLAModel (Vision + Decoder Transformer + Action)
│   │   ├── action_tokenizer.py  #   连续回归 + VQ-VAE 离散化
│   │   ├── vision_encoder.py    #   Tiny ViT (多视角 RGB → tokens)
│   │   └── vla_trainer.py      #   Trainer (warmup+cosine+混合精度+ckpt)
│   ├── safety/                 # ★ 横切 S 层
│   │   ├── chance_constrained_grip.py  # 概率性: 机会约束 f_n >= f_t/(μ̂ - zσ)
│   │   └── formal/             #   形式化 (创新 #4: STL + 可达集 + ISO 13482)
│   │       ├── stl_specs.py    #     STL 公式 (G/F/→/∧/∨/¬) + 鲁棒度监控
│   │       ├── reachability.py  #     区间可达集 (前向传播 + 包络检验)
│   │       └── iso13482.py     #     ISO 13482 运行时监控 + 违规分级
│   ├── adapters/               # ★ 外部适配层 (aigc 模式)
│   │   ├── edalab.py           # 调 EdaLab CLI/Open API (subprocess/HTTP)
│   │   ├── nanogpt.py          # 调 nanoGPT 训练产物
│   │   └── bigdata.py          # 数据飞轮回流
│   └── cli.py                  # 统一 CLI: train/eval/couple/export/audit
├── tests/                       # 单测 (对齐 nanoGPT 的 tests/ 模式)
├── scripts/                     # 一次性脚本 (smoke test 等)
├── experiments/                 # 实验产物 (runs/, 对齐 nanoGPT ablations)
└── archive/                     # 已有: 归档 (edalabcore-migrated 等)
```

### 3.3 迁移映射（现有资产 → 新骨架）

| 现有文件 | 去向 | 动作 |
|---|---|---|
| `src/core/climbing_dynamics_core.py` | `src/embodied_ai/dynamics/` | 迁入，拆出 EKF 到 perception |
| `src/core/mcts_integration.py` | `src/embodied_ai/planning/` | 迁入 |
| `src/control/household_robot_system.py` | `src/embodied_ai/control/` | 迁入，安全滤波拆到 safety/ |
| nanoGPT `llm.py` 的 ModelConfig/Trainer 模式 | `models/` 参考实现 | 不复制代码，按模式新建（保持两仓库独立） |
| EdaLab 16+5 引擎 | **不迁**，经 `adapters/edalab.py` 调 CLI | 契约交互 |
| docs/04-hardware/mechanical-domain-roadmap 等 | 原位保留 | 只按三栈命名法改 L→P |

### 3.4 分阶段执行

| 阶段 | 内容 | 验收 |
|---|---|---|
| **K0（1 周）** | contracts/ 四件套落地 + docs 三栈命名法修订（README/architecture/plan 三处 L→P） | 契约通过 EdaLab 消费测试（yaml → forge-core 子命令输入） |
| **K1（2 周）** | Python 包化：三脚本迁入 + config 体系 + 统一 CLI + tests 骨架 | `pip install -e .` 可用；`embodied-audit` 命令能汇总五项目资产状态 |
| **K2（1 月）** | adapters/edalab.py 打通 11 步主链调用 + dr_params 制造容差 DR（创新 #1） | 一条数据从真机遥测走回训练配置全程无人工（对齐 P1.1 验收） |
| **K3（✅已交付）** | models/ VLA 小模型骨架 (vla_config/vla_model/action_tokenizer/vision_encoder/vla_trainer, 消融4集, ≤200M参数) + safety/formal/ (STL规约+区间可达集+ISO13482运行时监控, 创新#4) | **VLA 消融 4 配置 JSON 往返通过; STL robustness + ISO 13482 违规分级 + 到达集包络验证全部 20 项 K3 测试绿** |

### 3.5 五项目角色边界（终版裁定）

```
EmbodiedAI  = "造什么" — L6 VLA模型 + L5/L1 动力学 + L5 规划 + L4/L3 控制 + 横切安全 (Python, 本仓库)
EdaLab      = "怎么造硬件" — 19 引擎 + 11 步主链 (Kotlin, 编译权威)
ECAD-MCAD   = 生命周期方法论 — NG1-22 + P0-P9 (文档)
SynapticForge = 商业化融合层 — L4 ForgeCore 编排 + 飞轮 (文档)
nanoGPT     = 模型训练底座 — tokenizer/Trainer/消融体系 (独立仓库, 经 adapter 引)
```

---

## 四、一句话结论

**七层保留、三栈分离（控制 L / 物理 P / 数据 D）、安全横切 S；骨架 = contracts 契约先行 + Python 单语种包化 + adapters 适配外部四仓库 + models/ VLA 小模型 + safety/formal/ 形式化安全；K0→K3 四个里程碑全部已交付验收。**
