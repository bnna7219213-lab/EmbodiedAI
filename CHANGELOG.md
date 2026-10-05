# 变更日志 (CHANGELOG)

本文档记录 EmbodiedAI Roadmap 项目的版本演进与关键决策。

---

## v0.6 (2026-10-06) — Linux on ARM (aarch64) 目标能力层

> 主题: `docs/04-hardware/embedded-linux-arm.md` 写了 ARM 裸机 → UBOOT → 根文件系统
> → 内核调试的全栈知识, 但仓库里**没有任何代码消费它**。本版把"登板人肉
> `cat /proc/cpuinfo`"变成可自动判定、可离线评估、有退出码的能力。

### 新增 `src/embodied_ai/platform/`

| 能力 | 入口 |
|---|---|
| 目标画像探测 | `probe_host(root=...)` — 读 `/proc`+`/sys`+`/dev`+`/etc` |
| 就绪评估 | `evaluate_readiness()` — 对照 `hw_requirements.yaml` 判 BLOCKING/WARNING/INFO/N/A |
| 预算缩放 | `arm_budget_overrides()` — 无 NPU/GPU 时保守放大, 回灌 `deploy.audit_latency` |
| 部署产物 | `render_systemd_units()` — 按探测到的 `isolcpus` 生成 `CPUAffinity` |
| CLI | `embodied-audit platform` — 退出码即结论 (`--root` / `--as-target` / `--json` / `--emit-units`) |

探测覆盖: 架构、板卡(设备树 model → Jetson Orin NX / RK3588 / i.MX8MP)、
厂商、加速器(nvidia-gpu / rockchip-vpu / drm / v4l2)、PREEMPT_RT、cpufreq
governor、`isolcpus`、内存、温度、CAN 接口、CPU features(asimd/neon)。

### 两条刻意的设计约束

1. **不可判定 ≠ 失败**: 读不到 `/proc/cpuinfo` (非 Linux 主机) 时, NEON / RT /
   cpufreq 等项标 **N/A** 而非 FAIL。跨平台探测脚本在 Windows 上全项爆红,
   就是没做这一步。
2. **判定与否由调用方显式选择**: 默认 `platform` 只探测信息; 只有 `--as-target`
   或 `--root` (目标板快照) 才做完整判定。否则 CI runner 会因"没有 CAN / 没有 NPU"
   被判 BLOCKING —— 结论正确但对开发机是纯噪声。显式化后: 开发机不假红,
   目标板不漏判。

### 与既有能力连通 (不是另起一套)

`platform.arm_budget_overrides(target)` → `deploy.audit_latency(budget_overrides=...)`,
复用同一个 50ms 闸门。为此给 `audit_latency` 增加了 `budget_overrides` 参数。
契约 15ms 推理预算的前提是 `preprocessing.gpu_preprocess_required: true`;
纯 CPU 的 ARM 板上不放大预算会把"跑不动"误判成"通过"。缩放系数是**保守占位值**,
模块内明确写着必须用目标板实测替换。

### 文档 / CI / 测试

- `docs/04-hardware/embedded-linux-arm.md` 新增**第 7 部分: 工程化能力层**
  (能力清单 / 用法 / 判定项与契约对应 / 两条设计约束 / CI 覆盖)。
- CI 新增两个 job:
  - `arm64` — QEMU + `run-on-arch-action` 在**真实 aarch64 用户态**跑完整测试
    套件与 `verify --no-pytest`, 探测 JSON 作为 artifact;
  - `aarch64-wheel` — 断言 wheel 为 `py3-none-any` (否则 aarch64 上装不上)。
- 新增 `tests/test_k5_arm_linux.py` (26 项): 用**合成的目标板文件系统快照**验证,
  不需要真板卡。`86 → 111 passed`; `verify` 8 项 → **9 项** (新增 `arm_linux`)。

---

## v0.5 (2026-10-06) — 工程化闭环: 可靠性修复 + 自动化 + 零依赖仿真

> 主题: 让"文档上宣称的能力"变成"任何机器上都能跑通并自动判定"的能力。

### A. 真实缺陷修复 (此前会崩 / 会发散 / 会误判)

| # | 缺陷 | 现象 | 修复 |
|---|---|---|---|
| 1 | **CLI emoji 在 GBK 控制台崩溃** | `audit/safety/couple/dr` 四子命令抛 `UnicodeEncodeError: 'gbk' codec can't encode '\u2705'`, 退出码非 0, 4 个测试红 | 新增 `embodied_ai/console.py` 安全输出层: 先 `reconfigure(utf-8)`, 失败则逐字符降级 (✅→[OK], ❌→[FAIL], ⚠→[WARN], →→->); CLI 全面改用 |
| 2 | **Sim-to-Real 报告含非 ASCII** | `report()` 内 ⚠ / • 在 GBK 下 print 即崩 | 全部改 ASCII 标记 |
| 3 | **EdaLab live 真调输出乱码** | JVM `System.out` 走本地 GBK, Python 固定按 UTF-8 解 → 中文变 U+FFFD, `assertIn("总 16.00ms", stdout)` 恒失败 | `_decode_output()` 编码回退链 (utf-8→gbk→gb18030→replace) + `-Dstdout.encoding=UTF-8`; 风险行匹配改为按 `[CRITICAL]/[WARNING]/[BLOCKING]` 标签 |
| 4 | **双足步态数值发散** | 开环把 ZMP 参考喂给 LIPM 正动力学, 含 e^{+ωt} 不稳定模态 → 6.4s 内 CoM 跑到 −1.17e8 m, 稳定性检查 100% 失稳 | 改为「ZMP 钉在支撑脚中心的 LIPM 解析解 + 周期性边界条件」: `v₀ = ω(L/2)·sinh(ωT)/(cosh(ωT)−1)`; 实测 CoM 前进 0.900 m, 0% 失稳, 裕度 0.08 m |
| 5 | **支撑脚 y 硬编码** | `_foot_trajectories` 把支撑脚 y 写死 ±0.09, 与 `build_walking_plans` 符号相反 → 支撑多边形错位 | 一律回读 plan; 新增 `ZMPStabilityChecker(planar=...)` 与 `other_foot_center` 双支撑包围盒 |
| 6 | **边界帧虚假失稳** | 跨步进边界那一帧 `current_step_index` 已指向下一步, 取到另一只脚 → 1600 帧里 6 帧误报 | 新增 `active_step_index` (本帧实际使用的步序号) |
| 7 | **MuJoCo biped 场景 typo** | `self._mjocf = ...` → `self._mjcf` 未定义 → AttributeError | 改名修正 |
| 8 | **后端可用性误判** | `available_backends()` 只探测"模块能否导入", 而 mujoco/pybullet 的 import 在 `__init__` 里才发生 → 报告可用、构造才炸 | 后端类增 `is_available()` 类方法, 探测在**构造前**做 |
| 9 | **episode 从不记录** | 所有 env 都未调用 `record_step`, `export_episode()` 恒为空 → BC 数据链实际是断的 | `ReferenceEnv.step` 落 `record_step` |
| 10 | **`run_bc_collection` NameError** | `max_steps<=0` 时 `next_result` 未定义 | 初始化为 None 并判空 |
| 11 | **DR 制造加载静默失败** | `load_manufacturing_dr` 三处裸 `return`, 且只替换已存在的参数 → 飞轮产出的新参数永远进不了采样集 | 改返回应用条数 + `last_load_applied` 留痕, 采纳新参数, 支持 normal 分布 |
| 12 | **真机发散误报** | 只按平均奖励符号判"发散", 距离型奖励恒负 → 100% 成功也告警 | 改为 `奖励<0 且 成功率<0.5` |
| 13 | **恢复步硬编码质心高度** | `suggest_recovery_step` 写死 `sqrt(9.81/0.8)`, 与 `LIPMParams.z_c` 脱耦 | `com_height`/`gravity` 可注入 + `omega` 属性 |
| 14 | **`capture_point` 属性恒返回 None** | 死代码, 误导调用方 | 删除, 统一到 `calc_capture_point()` |
| 15 | **`LIPM.__init__` 死矩阵** | `self._A` 计算后从未使用且量纲不一致 | 删除 |

### B. 功能补齐

- **零依赖参考环境** `simulation/reference_env.py` — 二阶点质量 + 可切换"真机退化"通道
  (观测/动作噪声 + 执行器延迟)。此前 MuJoCo/PyBullet 都需重量级外部包, 本机与 CI
  均未安装 → **整个仿真栈在默认环境下完全不可执行**。现在 `reset/step/DR/Sim2Real`
  一条链可以离线跑通并断言。
- **`make_env()` / `available_backends()`** — 后端自动选择 (auto→mujoco→pybullet→reference),
  显式指定缺失后端抛明确 `ImportError` 而非静默降级。
- **nanoGPT 适配器** `adapters/nanogpt.py` — 补齐 k1-audit 遗留项 #2: 仓库探测
  (llm.py/train.py/tokenizer.py 契约面) + 超参映射 + 训练计划 JSON 导出 + BC 数据集校验。
- **CLI 四新子命令** `sim` / `sim2real` / `biped` / `nanogpt` + 聚合 `verify`
  — 此前仿真与运动控制栈**没有任何自动化入口**。
- **`verification.py`** — 8 项聚合自检, 单项异常转 FAIL 不中断整体, 退出码即结论。

### C. 结构 / 工程化

- `src/simulation/` → `src/embodied_ai/simulation/`: 顶层 `simulation` 是与业务无关的
  泛化包名, 安装后会污染 site-packages。
- `tests/conftest.py`: 统一 `sys.path` 与子进程环境 (此前三个测试文件各拼一份)。
- `.gitignore` (新增): `output/`、`__pycache__/`、`.pytest_cache/`、`build/` 等;
  并把已误入库的 `output/dr/dr_config.yaml` 与一个 `.pyc` 移出索引 (文件保留在磁盘)。
- `pyproject.toml`: 版本 0.1.0→0.5.0 (与 README/`__init__` 对齐, 此前三处不一致),
  `requires-python>=3.10`, 可选依赖分组 `sim`/`torch`/`dev`, pytest 配置。
- `scripts/verify.bat` / `scripts/verify.ps1`: 一键验收入口 (ASCII-only — PS5.1 按
  ANSI 读无 BOM 的 .ps1, 中文会解析出错)。
- `.github/workflows/ci.yml`: Python 3.10–3.13 × ubuntu/windows 矩阵。

### D. 测试

`48 → 86 passed` (新增 `tests/test_k4_simulation.py` 41 项 + CLI 子命令覆盖)。
新增覆盖此前零测试的: `simulation/*`、`control/biped_locomotion.py`、
`adapters/nanogpt.py`、`console.py`、`verification.py`, 含 4 组明确的回归用例
(GBK 降级 / 步态不发散 / record_step 非空 / max_steps=0 不崩)。

### 关键决策记录 (ADR)

| ADR | 决策 | 理由 |
|---|---|---|
| ADR-006 | 仿真包移入 `embodied_ai` 命名空间 | 顶层 `simulation` 包名过泛, 易与其他包冲突 |
| ADR-007 | 新增零依赖 reference 后端而非强制安装 mujoco | 让仿真链路在 CI/无 GPU 环境可验证; 精度由 MuJoCoEnv 承担, 参考环境只承诺"管线可验证" |
| ADR-008 | 双足改用 ZMP 钉支撑脚中心 + 周期性解析解 | 开环 ZMP 驱动在数学上必然发散; 新方案数值恒有界且物理可解释 |
| ADR-009 | 平面步态判据 (`planar=True`) | 额状面稳定由落脚点(捕获点)控制而非 ZMP; 按 3D 判据检查会 100% 误报 |
| ADR-010 | 一键验收的断言只写在 `verification.py` | 避免脚本与代码两套断言漂移 |
| ADR-011 | nanoGPT 仓库缺失不作为失败 | 外部资产, CI 机器必然没有; 交付物是训练计划 |

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

## v0.4 (2026-10-06) — 代码骨架补全与 G6/G7/G9 缺口修复

### 新增模块

- **src/simulation/** — 仿真环境核心 (G7 缺口修复)
  - `base_env.py` — SimEnv 抽象基类 (对齐 Gymnasium, 统一仿真/真机接口)
  - `mujoco_env.py` — MuJoCo 包装 (程序化 MJCF 场景:桌面抓取 + 双足 + 简单臂)
  - `pybullet_env.py` — PyBullet 包装 (备选平台, URDF 加载)
  - `domain_randomization.py` — DR 引擎 (11 参数/4 类别, 课程学习调度)
  - `sim_to_real.py` — Sim-to-Real 迁移管线 (自动差距诊断 + 建议生成)

- **src/embodied_ai/control/biped_locomotion.py** — 双足运动控制 (G9 缺口修复)
  - LIPM 线性倒立摆模型 (离散化 + 捕获点计算)
  - GaitGenerator: 梯形 ZMP + 三次样条足端轨迹
  - ZMPStabilityChecker: 实时稳定性检查 + 恢复步生成

### 文档更新

- docs/plan.md: 无变更 (参考)
- src/embodied_ai/control/__init__.py: 导出 biped 模块

---

## v0.1 (2026-09-19) — 初始版本

### 初始内容

- 第0-8章: 端到端具身智能体研发计划
- Figure AI 对标分析
- NN-to-Hardware 编译映射
- 硬件自动关机设计
