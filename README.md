# EmbodiedAI Roadmap — 端到端具身智能体研发计划

> **项目代号**: EmbodiedAI
>
> **定位**: 不拼装现有方案，从零构建"AI 神经网络模型 → PCB 硬件 → 电机驱动 → 具身智能体"全链条的端到端研发计划。
>
> **版本**: v1.1 | 更新日期: 2026-10-17
>
> **一键验收**: `scripts\verify.bat` (cmd) / `.\scripts\verify.ps1` (PowerShell) → `RESULT=VERIFY_OK`
> 或 `python -m embodied_ai.cli verify` (**35 项**聚合自检, 退出码即结论; 含 pytest = 36 项 + 632 passed, v1.1 实测)
>
> **v1.1 主题**: 验收**诚实化** — 把"绿色但无效"的判据改成能失败的判据
> (V2A 等价性不再自我认证 / SLAM 补噪声失配负对照 / 分频串扰度量修时间基),
> 把静默失败改成显式报错 (认知中枢三处 / Bus 全零观测), provenance 由数据推导
> 而非由导出器断言, 并清除仓库里的个人机器路径。详见
> [CHANGELOG v1.1](CHANGELOG.md)。
>
> **v1.0 主题**: Sprint M 机器人端补全 — 多传感器分频信道融合 (M1) / EKF-SLAM
> 位姿+时间戳 (M2) / VLA 双路径 V+L 与 V-to-action (M3) / 四要素认知中枢 (M4)。
> 详见 [plan_v5.md](plan_v5.md) 与 [CHANGELOG v1.0](CHANGELOG.md)。
>
> **v0.7 主题**: 针对"全链条完成度 30-35%"的评估结论批量补齐缺口 —
> 柔性体 (G6) / 双足 3D (G9) / 灵巧手 (G11) / 终身学习 (G12) / GUI (L7) /
> 认证工程化 (G10) / 训练管线 (BC+PPO+SAC+Diffusion) / 真机闭环 / FOC 关机链。
> 详见 [CHANGELOG v0.7](CHANGELOG.md#v07-2026-10-06--从图纸库到可运行闭环-缺口批量补齐)。

---

## 核心原则

1. **全栈自研** — AI 模型从 tokenizer 到训练管线全部自研；硬件从原理图到 PCB 全部自研
2. **AI × 硬件深度耦合** — 端到端神经网络模型直接映射到硬件执行，不做"仿真-现实鸿沟"
3. **渐进式规模** — 从算力受限嵌入式到高算力边缘计算，分阶段递进
4. **安全优先** — 电机驱动必须有硬件级自动关机保护，不能仅靠软件
5. **决策核心** — 强化学习的数学本质指导具身智能体训练，触觉为最稀缺感知
6. **操作系统驱动** — 嵌入式 Linux on ARM 把 AI 决策变成电机旋转
7. **契约先行** — 跨项目交互一律经 `contracts/` 机器可读契约，禁止文档口头约定

---

## 硬件端调用：v4「jar 入库 + 消费」是唯一方式

> **定位**：本项目调用硬件端（EdaLab 的 EDA 能力 — 原理图 / PCB / 仿真 / 芯片设计）
> 的途径**只有一条**：通过入库的编译产物 `vendor/edalab/*.jar` 消费。
> Python 侧**不**重实现 EDA 引擎，也**不**直连 EdaLab 源码目录。

| 项 | 内容 |
|---|---|
| 载体 | `vendor/edalab/edalabcore.jar`（101 个 Kotlin 文件编译产物）+ `vendor/edalab/kotlin-stdlib-2.2.21.jar`（Kotlin 运行时，**必需配套** — core jar 不自含运行时） |
| 体积 | 3.54 MB（两件） |
| 运行时依赖 | **仅需 JRE**（无需 JDK / gradle / 外部 EdaLab 目录） |
| 覆盖能力 | **12 个 Stage**：forge-core、mass-props、structural-fem、kinematics、thermal-sim、pi-analysis、motor-drive、data-flywheel、relay-clutch、em-coupling、multi-physics、vsa-static |
| 调用入口 | `embodied-audit couple`（主链）/ `couple --eps-job`（Stage 数据管道串联）/ `EdalabAdapter.run_stage()` |
| 单仓自足实证 | **`embodied-audit selfcheck`**（或 `python scripts/demo_selfcontained.py`）— 5 项断言全绿即证明：载体齐全 / 指纹一致 / 仅需 JRE / 隔离可跑 / 后端可观测 |

**为什么是唯一方式**：实测确认 Python 侧真正缺的 EDA 能力（`netlist` / `schematic` /
`router` / `gerber` / `MNA` / `thermal` / `FEM`）全在 Kotlin 侧，重写 16,780 行能力的
收益/成本比很差；jar 入库以 3.54 MB 换来全部能力且零重写。详见 `plan_L5.md`。

### 仓库边界：只提供 jar 编译产物，**不暴露 EdaLab 源码**

这是本项目的**硬规则**，不是可选约定：

| 项 | 状态 |
|---|---|
| `vendor/edalab/*.jar`（2 件，3.54 MB）+ `PROVENANCE.json` | ✅ **入库** —— 唯一的 EDA 依赖来源 |
| EdaLab 的 Kotlin 源码（`*.kt` / `edalabcore/` / `archive/edalabcore-migrated-*`） | ❌ **一律不入库**（`.gitignore` 全面拦截） |
| 本项目自身的 Python 源码 | ✅ 入库 —— **含双轨模式下自行实现的 G 缺口后端** |

**双轨下的 Python 侧代码可以公开**：G6 / G10 / G7 / G2 这类"jar 冻结改不了"的缺口，
本项目在 Python 侧自行实现（如 `src/embodied_ai/hardware/hil_bridge.py`），
经 `@register_python_backend()` 注册为可路由后端。**这部分属于 EmbodiedAI 的原创实现，
与 EdaLab 源码无关**，因此正常入库。

一句话：**EdaLab 的能力以二进制形态提供（jar），本项目的能力以源码形态提供（Python）。**

> 若你同时持有 EdaLab 源码仓库，请自行从那里获取 `.kt` —— 本仓**有意不镜像**，
> 以保证实现边界清晰、版本可追溯（靠 `PROVENANCE.json` 的 sha256 与构建信息）。

#### 一条命令证明"单仓自足"（30 秒，无需外部环境）

```bash
python -m embodied_ai.cli selfcheck          # 人读: 逐项 ✅/❌ + 失败时的修复指引
python -m embodied_ai.cli selfcheck --json   # 机读: JSON（CI 断言退出码）
```

实测输出（本机）：

```
[✅] 1.载体齐全   — core=True (1908KB)
[✅] 2.指纹一致   — PROVENANCE 校验 9 项，全部一致
[✅] 3.运行时可用  — jar 来源=vendor | java=openjdk 17.0.18
[✅] 4.隔离可跑   — 外部 EdaLab/gradle 屏蔽 → 仍 LIVE_OK 且引擎输出正确(含 16.00ms)
[✅] 5.后端可观测  — status=LIVE_OK, jar_source=vendor
RESULT: SELF_CONTAINED_OK ✅
```

> **第 4 项是决定性证据**：把 `EDALAB_HOME` 与 `GRADLE_USER_HOME` 指向**不存在**的路径，
> 仍能真实执行 Kotlin 引擎 —— 这就是"单仓自足"的定义（不依赖外部 EdaLab 目录 / gradle / JDK）。

#### ⚠️ 别人跑不起来怎么办（排障）

**最可能的原因：机器上没有 JRE（跑 jar 必需）** —— 报错形如
`NoClassDefFoundError: kotlin/KotlinNothingValueException` 或 `java: command not found`。

| 症状 | 原因 | 修复 |
|---|---|---|
| `java: command not found` / `java 未找到` | 没装 JRE | 装 **JRE 17+**（不是 JDK 也行，JRE 足够） |
| `NoClassDefFoundError: kotlin/...` | classpath 缺 `kotlin-stdlib` | 确认 `vendor/edalab/kotlin-stdlib-*.jar` 在（core jar **不自含**运行时，两件必须齐） |
| `jar 不存在: 期望 vendor/edalab/edalabcore.jar` | vendor 产物缺失（被 git 误排除 / clone 不完整） | `.gitignore` 已有 `!vendor/edalab/*.jar` 白名单；重新 `git checkout vendor/` 或从原 EdaLab 重建入库 |
| 指纹校验失败（`2.指纹一致 ❌`） | jar 被替换过 | 有意升级 → 更新 `PROVENANCE.json` 的 sha256/size；非有意 → 检查是否被误改 |
| 想指定 java 路径（多版本共存） | — | 设 `JAVA_HOME`，或设 `EMBODIED_JAVA=<java 绝对路径>`（解析顺序：PATH → JAVA_HOME → EMBODIED_JAVA） |

**Windows 提示**：装完 JRE 请**重开终端**（刷新 PATH），再用 `where java` 确认。

#### 贡献者须知（改代码前必读两条）

1. **新增任何调用方式（CLI 命令 / 能力后端）必须注册**：
   - CLI 命令 → `register_command()`（在 `cli.py` 的 `_register_all()` 里注册）；**禁止**直接 `sub.add_parser`
   - Python 能力后端 → `@register_python_backend(name)`（双轨机制，见 `adapters/edalab.py`）
   - 守护：`tests/test_cli_registry.py` 断言"注册表 == 实际可执行命令"（**无孤儿**）；重复注册**抛错**
   - 自检：`embodied-audit catalog` 能看到 = 注册成功；看不到 = 落在角落了
2. **改动 `vendor/edalab/*.jar` 必须走完整流程**（否则静默漂移）：
   ```bash
   # 1) 在原 EdaLab 改 Kotlin 源码 → 重新构建
   cd <EdaLab> && ./gradlew :edalabcore:jar
   # 2) 覆盖入库产物
   cp edalabcore/build/libs/edalabcore.jar <repo>/vendor/edalab/
   # 3) 更新指纹（sha256/size/日期）—— 不更新则 verify 的 L1 闸门会 FAIL
   #    编辑 vendor/edalab/PROVENANCE.json
   # 4) 跑验收：L2 行为锚定会检查引擎输出是否变化
   embodied-audit verify
   # 5) 若锚定值变了 → 同步 contracts/ 并在 CHANGELOG 记原因
   ```
   > **先例**：B3 就是这么发现契约 `vsa_antagonistic_peak_n` 原值 109.0 落后于引擎实测 166.39 的。

### jar 与源码漂移治理（两层闸门）

jar 是**冻结的编译产物** — 原 EdaLab 源码若演进，jar 不会自动更新 → 行为静默漂移。
**B3 已实证过一次**：契约 `vsa_antagonistic_peak_n` 原值 109.0 落后于引擎实测 166.39
（109 是 V-2 修复前的过期口径）。治理分两层，均有自动化守护：

| 层 | 抓什么 | 怎么抓 | 守护 |
|---|---|---|---|
| **L1 指纹** | jar 被替换 / 重新构建 | `vendor/edalab/PROVENANCE.json` 记录的 sha256/size vs 实际文件比对 | `verify` 项 `vendor_jar_provenance` + `tests/test_vendor_provenance.py`（含 4 项反证） |
| **L2 行为锚定** | 源码演进导致引擎输出变化（jar 该更新了） | 契约锚定值 vs 引擎实测：`vsa_antagonistic_peak_n=166.39`、relay-clutch 安全链 16.00ms、forge-core EPS ID | `tests/test_b3_vsa_antagonistic.py`、`test_b2_stage_pipeline.py` |

**jar 更新流程**（原 EdaLab 演进后必做，亦见 `PROVENANCE.json` 的 `update_procedure`）：

1. `cd EdaLab && ./gradlew :edalabcore:jar`
2. 覆盖 `vendor/edalab/edalabcore.jar`
3. 重算 sha256/size 并更新 `vendor/edalab/PROVENANCE.json`
4. 跑 `python -m embodied_ai.cli verify` —— 若 L2 锚定值断言失败，说明引擎输出已变，
   **必须同步更新 `contracts/hw_requirements.yaml` 的对应锚定值并在 CHANGELOG 记录原因**
   （先例：B3 的 109.0 → 166.39）
5. 更新 CHANGELOG 的 vendor 小节

**边界（诚实声明）**：① 需 JRE（非纯 Python）；② Kotlin 侧缺口 jar 冻结不可修
（G2/G3/G5/G6/G7/G10，见 `plan_L5.md` §三）。

---

## 快速开始

```bash
# 1. 安装 (dev 含 pytest; sim/torch 为可选)
pip install -e ".[dev]"

# 1b. 硬件端 EDA 能力开箱可用 — vendor jar 已入库, 仅需 JRE (无需 JDK/gradle/外部 EdaLab)
#     验证单仓自足 (5 项断言: 载体/指纹/运行时/隔离可跑/后端可观测)
python -m embodied_ai.cli selfcheck
#     再看 EDA 能力真的能跑 (12 Stage 主链)
python -m embodied_ai.cli couple --eps-job   # 应 jar_source=vendor

# 2. 全量验收 (推荐入口)
scripts\verify.bat                  # Windows cmd
python -m embodied_ai.cli verify    # 跨平台, 退出码 0 = 全绿

# 3. 单测
python -m pytest -q

# 4. 常用自检 / 命令发现
python -m embodied_ai.cli catalog    # 列出全部命令 (读全局注册表 — 新命令在此可见)
python -m embodied_ai.cli contracts
python -m embodied_ai.cli safety
python -m embodied_ai.cli sim --dr
python -m embodied_ai.cli biped
```

CI (`.github/workflows/ci.yml`) 在 Python 3.10–3.13 × ubuntu/windows 上执行同一套
`verify --no-pytest` + `pytest` + 全模块导入 + CLI 冒烟, 无需 mujoco/pybullet/torch
即可全绿。

---

## 文档结构 (v0.3)

### 入门与新文档

| 文档 | 内容 |
|---|---|
| [docs/plan.md](docs/plan.md) | **主计划**: 13 章端到端研发计划 (v0.2) |
| [docs/architecture-overview.md](docs/architecture-overview.md) | ★ **新增** ★ 系统架构总览: 7层架构、感知系统、核心方程、实时性预算 |
| [docs/future-exploration-plan.md](docs/future-exploration-plan.md) | ★ **新增** ★ 下一步未来探索计划: 6-18个月分阶段路线图 |
| [docs/09-project-skeleton/skeleton-and-layering-plan.md](docs/09-project-skeleton/skeleton-and-layering-plan.md) | ★ **新增** ★ 项目骨架方案 + 三栈分离命名法 (控制L/物理P/数据D/安全S横切) |
| [CHANGELOG.md](CHANGELOG.md) | ★ **新增** ★ 版本变更日志与关键决策记录 |

### 契约目录 (`contracts/`) — 跨项目交互唯一真源

| 契约 | 内容 | 消费方 |
|---|---|---|
| [hw_requirements.yaml](contracts/hw_requirements.yaml) | AI 策略 → 硬件规格 (三栈约束全集) | SynapticForge ForgeCore / EdaLab |
| [canfd/joint_control.dbc](contracts/canfd/joint_control.dbc) | CAN-FD 帧格式/ID 分配/DBC (9 消息: 6 关节 + 同步 + 遥测 + 安全事件) | ECAD-MCAD (债务已还) / HIL |
| [dr_params/manufacturing_variance.yaml](contracts/dr_params/manufacturing_variance.yaml) | 制造差异驱动的域随机化分布 | RL 训练管线 (创新 #1) |
| [safety_budget.yaml](contracts/safety_budget.yaml) | 安全链分段延迟预算 (0.8µs+8ms+8ms=16ms<20ms) | 横切安全层 S / G10 认证 |

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

> v0.5 目录修正: README 此前指向 `src/core/`、`src/control/` 等**不存在**的路径
> (v0.3 重组后未同步)。下表为当前真实路径。

### 仿真环境 (src/embodied_ai/simulation/) — G7 缺口修复

| 文件 | 内容 |
|---|---|
| [base_env.py](src/embodied_ai/simulation/base_env.py) | SimEnv 抽象基类 (对齐 Gymnasium API, 统一仿真/真机接口) |
| [reference_env.py](src/embodied_ai/simulation/reference_env.py) | ★ v0.5 ★ 纯 numpy 参考环境 (零外部依赖, 默认可用, 支撑 Sim2Real 离线验证) |
| [mujoco_env.py](src/embodied_ai/simulation/mujoco_env.py) | MuJoCo 包装器: 桌面抓取 + 双足 + 简单臂 (MJCF 程序化生成; 需 `pip install mujoco`) |
| [pybullet_env.py](src/embodied_ai/simulation/pybullet_env.py) | PyBullet 包装器 (备选平台, URDF 导入; 需 `pip install pybullet`) |
| [domain_randomization.py](src/embodied_ai/simulation/domain_randomization.py) | 域随机化引擎: 5 类别 (动力学/执行器/传感器/视觉/环境) + 课程学习 + 制造容差注入 |
| [sim_to_real.py](src/embodied_ai/simulation/sim_to_real.py) | Sim-to-Real 迁移管线: 差距量化 + 自动诊断 + 建议 + BC 数据采集 |
| `__init__.py` | ★ v0.5 ★ `make_env()` / `available_backends()` 后端自动选择 (auto→mujoco→pybullet→reference) |

### 运动控制 (src/embodied_ai/control/) — G9 缺口修复

| 文件 | 内容 |
|---|---|
| [biped_locomotion.py](src/embodied_ai/control/biped_locomotion.py) | 双足行走: LIPM + ZMP 周期性解析步态 + 捕获点 + 足端轨迹 (v0.5 修复发散) |
| [household_robot_system.py](src/embodied_ai/control/household_robot_system.py) | 家庭服务机器人 20Hz 控制器: 4 层架构 (MCTS + Diffusion + 阻抗 + FOC) |

### 工程化 (v0.5 新增)

| 文件 | 内容 |
|---|---|
| [console.py](src/embodied_ai/console.py) | 终端安全输出层: UTF-8 强制 + 不可编码字符 ASCII 降级 (根治 GBK 崩溃) |
| [verification.py](src/embodied_ai/verification.py) | 聚合自检 (契约/安全/仿真/迁移/步态/CLI/pytest), 退出码即结论 |
| [adapters/nanogpt.py](src/embodied_ai/adapters/nanogpt.py) | nanoGPT 训练底座适配: 仓库探测 + 超参映射 + 训练计划导出 + BC 数据集校验 |
| [scripts/verify.bat](scripts/verify.bat) / [verify.ps1](scripts/verify.ps1) | 一键验收入口 (编排, 不含断言副本) |
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | CI: Python 3.10–3.13 × ubuntu/windows 矩阵 + 全模块导入 + CLI 冒烟 |

### 目标平台 (src/embodied_ai/platform/) — Linux on ARM 能力 (v0.6 新增)

| 文件 | 内容 |
|---|---|
| [arm_linux.py](src/embodied_ai/platform/arm_linux.py) | ★ v0.6 ★ aarch64 目标探测 (架构/板卡/加速器/RT/isolcpus/CAN/内存/温度) + 就绪评估 + 预算缩放 + systemd 单元渲染 |
| [docs/04-hardware/embedded-linux-arm.md 第 7 部分](docs/04-hardware/embedded-linux-arm.md) | 能力层用法、判定项与契约对应、CI 中的 ARM 覆盖 |

此前"目标板能不能跑 20Hz"全靠人登板 `cat /proc/cpuinfo`；现在
`embodied-audit platform --root <目标板快照>` 可**离线判定**并以退出码给结论。
输入全是文件系统，因此在 x86 Windows 开发机与 CI 上都能完整验证。

### v0.7 新增模块 (缺口补齐)

| 文件 | 内容 |
|---|---|
| [models/nn_core.py](src/embodied_ai/models/nn_core.py) | 纯 numpy 训练内核: MLP + 反向传播 + Adam/SGD + 梯度自检 (**无 torch 也能真训练**) |
| [models/diffusion_policy.py](src/embodied_ai/models/diffusion_policy.py) | DDPM 条件扩散策略 (cosine 调度 + ε 预测 + DDIM 少步采样) |
| [models/bc_trainer.py](src/embodied_ai/models/bc_trainer.py) | 行为克隆: 数据集 + 归一化 + 训练器 + 环境内成功率评估 |
| [models/rl_algorithms.py](src/embodied_ai/models/rl_algorithms.py) | PPO (GAE+clip) 与 SAC (双 Q + 自动温度) |
| [dynamics/soft_body.py](src/embodied_ai/dynamics/soft_body.py) | G6 柔性体: XPBD 布料/毛巾 + 抓取 + 打滑判据 |
| [control/whole_body.py](src/embodied_ai/control/whole_body.py) | G9 3D 全身: 支撑多边形 + 6-DoF 腿部 IK + 接触力分配 |
| [hardware/foc_shutdown.py](src/embodied_ai/hardware/foc_shutdown.py) | FOC + SVPWM + 电流环 + 自动关机链时序 + HIL 用例 |
| [hardware/dexterous_hand.py](src/embodied_ai/hardware/dexterous_hand.py) | G11 灵巧手: 力闭合判据 + 4 候选选型矩阵 |
| [simulation/real_robot_env.py](src/embodied_ai/simulation/real_robot_env.py) | 真机闭环: CAN-FD 帧编解码 + 传输层 + 安全闸门 + 延迟统计 |
| [deploy/jetson_bench.py](src/embodied_ai/deploy/jetson_bench.py) | VLA 推理 roofline 基准 + 实测注入标定 |
| [learning/lifelong.py](src/embodied_ai/learning/lifelong.py) | G12 终身学习: EWC + 回放 + forgetting/BWT |
| [safety/certification.py](src/embodied_ai/safety/certification.py) | G10 认证: HAZOP + SIL/PFH/SFF + 架构约束 + V&V + RTM |
| [gui/](src/embodied_ai/gui/) | L7 运行台: 零依赖 Web 仪表盘 + 7 个 JSON API |

### 核心资产

| 文件 | 内容 |
|---|---|
| [dynamics/climbing_dynamics_core.py](src/embodied_ai/dynamics/climbing_dynamics_core.py) | ★ 核心 ★ 攀岩竞技动力学纯 numpy 实现 (Hertz 接触 + 摆荡 + dyno + EKF + 滑脱前兆) |
| [planning/mcts_integration.py](src/embodied_ai/planning/mcts_integration.py) | NeRF 增强的 MCTS 攀岩规划 + NeRF 价值网络训练 |
| [safety/formal/](src/embodied_ai/safety/formal/) | 形式化安全: STL 规约 + 区间可达集 + ISO 13482 + 16ms 预算闭环 (创新 #4) |

---

## 命令行 (`embodied-audit`)

```bash
pip install -e ".[dev]"          # 安装 (dev 含 pytest)

embodied-audit audit             # 五项目资产状态汇总 (需兄弟仓库存在)
embodied-audit contracts         # contracts/ 四件套校验 (DBC 9 消息 / CRC 覆盖)
embodied-audit safety            # 安全链 16ms 预算核对 + SIL2 PFH
embodied-audit couple [--live]   # EdaLab 11 步主链 (dry 默认 / live 真调 Kotlin)
embodied-audit dr                # 飞轮 DR 制造容差闭环 (ICT/FCT/遥测 → DR YAML)

embodied-audit sim    [--scene point_mass] [--backend auto] [--steps 20] [--dr]
embodied-audit sim2real [--episodes 3] [--steps 300] [--noise 0.02]
embodied-audit biped  [--steps 6] [--step-length 0.15] [--step-time 1.0]
embodied-audit nanogpt [--size vla_tiny] [--out output/nanogpt]

# Linux on ARM 目标能力 (v0.6)
embodied-audit platform [--root <目标板快照>] [--as-target] [--strict]
                        [--json probe.json] [--emit-units ./units]

# v0.7 缺口补齐
embodied-audit soft                        # G6 柔性体 (XPBD)
embodied-audit wholebody [--steps 4]       # G9 双足 3D 全身
embodied-audit train  [--algo bc|ppo|sac|diffusion] [--iters N]
embodied-audit foc                         # FOC + SVPWM + 自动关机链
embodied-audit hand                        # G11 灵巧手选型
embodied-audit real   [--transport mock|socketcan|serial]
embodied-audit bench  [--size vla_tiny] [--platform jetson_orin_nx_16gb]
embodied-audit cert   [--md docs/cert.md]  # G10 认证证据包
embodied-audit lifelong                    # G12 终身学习
embodied-audit gui    [--port 8765] [--once]   # L7 运行台

# v0.8 双轨缺口 (Python 后端 + jar oracle 对照)
embodied-audit twin   [--backend auto|jar|python]
                              # G10 电机模型 / G7 频谱判别 / G2 FOC↔板级对齐
embodied-audit selfcheck [--json]   # 单仓自足实证 (5 项断言)

embodied-audit verify [--no-pytest]   # 一键全量自检, 退出码即结论
```

`platform` 退出码：0 = 无 BLOCKING；1 = 至少一项 BLOCKING（缺 CAN、内存不足、非 aarch64…）。
目标板快照可离线评估，无需登板：

```bash
ssh robot@orin 'tar -C / -cf - proc/cpuinfo proc/cmdline proc/meminfo \
                sys/firmware/devicetree sys/class sys/devices/system/cpu dev etc' \
  | tar -C ./snap -xf -
embodied-audit platform --root ./snap
```

`sim` 在无 mujoco/pybullet 环境自动回退到零依赖 `reference` 后端, 因此仿真链路
在 CI 与本机始终可跑通。

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

> **命名规范（三栈分离命名法，见 [docs/09-project-skeleton/skeleton-and-layering-plan.md](docs/09-project-skeleton/skeleton-and-layering-plan.md)）**：控制栈用 L1-L7（见上方系统架构）；物理栈（PCB/硬件）用 **P1-P5**；数据栈用 **D1-D5**。安全层 S 与数据栈 D 为横切层。编号不再混用。

| 物理栈 | PCB | 关键芯片/模块 | 层数 |
|---|---|---|---|
| P1 电源层 | 电源管理 PCB | PMIC、BMS、电机驱动电源轨 | 4 层 |
| P2 感知层 | 传感器融合 PCB | IMU、摄像头接口、力传感器 ADC | 6 层 |
| P3 决策层 | 主控 PCB | Jetson Orin / RK3588 + LPDDR5 | 8 层 |
| P4 执行层 | 电机驱动 PCB | STM32 + DRV8301 + 自动关机电路 | 4 层 |
| P5 互联层 | 通信背板 | CAN-FD / RS485 / Ethernet | 2 层 |

物理栈与控制栈的对应关系：P3（决策 PCB）承载 L6-L7；P2（感知 PCB）服务 L3-L5 的传感输入；P4（电机驱动 PCB）实现 L2；P1/P5 为 L1 物理执行提供能量与互联。**横切安全层 S**（<1μs 硬件关断、STO、机会约束滤波）横跨 L1-L7 与 P1-P5；**横切数据栈 D**（遥测→DR→RL 飞轮）横跨全栈。

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

# v0.7 新增: 扩散策略去噪 (ε 预测)
x_{t-1} = (x_t − (1−α_t)/√(1−ᾱ_t) · ε̂) / √α_t + σ_t z
L = E‖ε̂(x_t, t, c) − ε‖²

# v0.7 新增: XPBD 约束投影 (柔度 α̃ = compliance/h²)
Δλ = (−C − α̃λ) / (w_i + w_j + α̃),  Δp_i = w_i Δλ ∇C

# v0.7 新增: LIPM 解析推进 (不稳定系统不能用显式欧拉)
ξ = x + v/ω (DCM),  x(t+h) = p + (x−p)cosh(ωh) + (v/ω)sinh(ωh)

# v0.7 新增: EWC 持续学习惩罚
L = L_task + λ · Σ_i F_i (θ_i − θ*_i)²,  F_i = E[(∂L/∂θ_i)²]

# v0.7 新增: SIL 定量 (IEC 61508)
PFH = Σ λ_DU · 1e-9,  λ_DU = λ(1−λ_S%)(1−DC%),  SFF = (Σλ_S+Σλ_DD)/Σλ
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

### 三个实质里程碑的当前状态 (v0.7)

评估给出的优先级排序是 **(1) FOC 关节板转动 + 自动关机 <1µs → (2) Jetson Orin NX
上跑一次 VLA 推理 → (3) 仿真策略部署到真实单关节**。v0.7 把它们各自的
**"到货前必须先钉死"** 部分做完了, 剩下的都是必须有实物才能推进的:

| 里程碑 | 已就绪 (v0.7) | 仍缺 (需实物) |
|---|---|---|
| (1) FOC + 自动关机 | SVPWM 伏秒平衡校验、电流环极点配置、关机链 4 段时序模型、**7 条 HIL 测试用例**、停止角→安全速度包络反解 | PCB 打样、示波器实测、继电器/离合器寿命实测 |
| (2) Jetson VLA 推理 | 5 个 SoC 的 roofline 基准、实测注入标定系数、INT8/FP16 切换、与 50ms 闸门对齐 | 登板实测 TensorRT 引擎 (可用 `bench` 把实测值喂回) |
| (3) Sim-to-Real | `RealRobotEnv` 走同一 `SimEnv` 接口、CAN-FD 按 DBC 编解码、安全闸门 + 延迟统计 | 真关节 + `socketcan`/`serial` 传输层的首次通电 |

也就是说: **板子/电机一到, 上述三条都能在当天跑出第一个实测数**, 且实测值
有明确的落点 (`ShutdownChain.stages[*].delay_s` / `bench_vla(measured_ms=...)`
/ `RealRobotEnv.avg_latency_ms`), 不需要临时设计测量方案。

30 天快速启动 (v0.7 后已部分完成, 标注 ✅):
1. ✅ 仿真环境 (零依赖 reference 后端 + DR + MuJoCo/PyBullet 可选)
2. ⬜ 导入 URDF (机械臂 + 桌面 + 杯子) — 需 MuJoCo/PyBullet 环境
3. ✅ 行为克隆训练管线 (`models/bc_trainer.py`, 含环境内成功率验收)
4. ✅ Sim-to-Real 迁移管线 (`simulation/real_robot_env.py`, mock 通路已跑通)

---

## 项目对齐

| 来源 | 提供什么 |
|---|---|
| LLMInnovation | AI 模型研发路线（Phase 0-6，从 tokenizer 到 100B+ 多模态） |
| EdaLab | 硬件设计能力（原理图 → PCB → 仿真 → 芯片设计）— **经 `vendor/edalab/*.jar` 消费，这是本项目调用硬件端的唯一方式**（仅需 JRE，见上文专章） |
| Figure AI | 对标公司（Helix 2.5 端到端模型、Figure 03 人形机器人） |
