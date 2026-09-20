# 变更日志 (CHANGELOG)

本文档记录 EmbodiedAI Roadmap 项目的版本演进与关键决策。

---

## v0.3 (2026-09-19) — 项目重组与结构化

### 结构重组

- **根目录清理**: 将所有 `# ` 前缀的 AI 生成文档归位到 `docs/` 子目录，移除根目录的文件混乱
- **docs/ 分类重组**: 按照技术领域划分为 7 个子目录，每个子目录对应一个技术方向
  - `01-ai-model/` — AI 模型（RL、VLA、算法决策、实时性分析）
  - `02-contact-dynamics/` — 接触动力学（优化、数学基础、融合规划）
  - `03-sensing-perception/` — 感知与传感（触觉传感、Sim2R Conclusion）
  - `04-hardware/` — 硬件（电机驱动、嵌入式Linux、AI硬件对标）
  - `05-system-software/` — 系统软件（OS动作规划、GUI界面）
  - `06-data-strategy/` — 数据策略（采集策略、攀岩迁移分析）
  - `07-gap-analysis/` — 差距分析（6路Gap分析 + 综合报告）
- **代码归位**: `python完整系统实现` → `src/control/household_robot_system.py`，`python与MCTS集成` → `src/core/mcts_integration.py`
- **Agent 输出整合**: Agent 任务的 4 份核心交付件（3份文档 + 1份代码）分别转入对应目录
- **归档**: 原始 agent session 移至 `archive/agent-task-2026-09-19/`

### 新增文档

- `docs/architecture-overview.md` — 系统架构总览（7层架构、感知系统、核心方程、实时性预算、硬件架构、完备性矩阵）
- `CHANGELOG.md` — 本文件，变更日志
- `docs/future-exploration-plan.md` — 下一步未来探索计划

### 关键决策记录 (ADR)

| ADR | 决策 | 理由 |
|---|---|---|
| ADR-001 | 文档按技术领域分目录而非按章节编号 | 原有 plan.md 的 13 章编号与 docs/ 专题文档不对应，领域分类更利于增量扩展 |
| ADR-002 | plan.md 保留在 docs/ 根目录 | 作为总纲性文档，是项目入口，不需要归入子目录 |
| ADR-003 | 原始 agent 输出完整归档而非删除 | 保留研究过程的可追溯性，尤其是迭代中的推理链 |
| ADR-004 | 代码文件从根目录移入 src/ 子目录 | 区分文档与代码，为后续工程化扩展建立基础 |
| ADR-005 | climbing_dynamics_core.py 进入 src/core/ | 代码是项目核心资产（纯 numpy 实现的可验证数学核心），不只是分析产物 |

---

## v0.2 (2026-09-19) — 技术深度扩展

### 新增内容

- 第9章: 嵌入式 Linux on ARM 全栈
- 第10章: 强化学习深度理论
- 第11章: 触觉传感稀缺资源
- 第12章: OS层动作规划与可解释性
- 第13章: 上层GUI交互界面
- 更新 README 文档结构

---

## v0.1 (2026-09-19) — 初始版本

### 初始内容

- 第0-8章: 端到端具身智能体研发计划
- Figure AI 对标分析
- NN-to-Hardware 编译映射
- 硬件自动关机设计
