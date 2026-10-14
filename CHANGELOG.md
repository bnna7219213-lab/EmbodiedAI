# 变更日志 (CHANGELOG)

本文档记录 EmbodiedAI Roadmap 项目的版本演进与关键决策。

---

## v0.9 (2026-10-08) — Gate 2 收官 + 全栈架构顶层文档体系

### Gate 2 达成 (技术层面)

| 判据 | 处置 |
|------|------|
| verify --no-pytest    | 28/28 PASS → RESULT=VERIFY_OK ✅ |
| pytest                | 441 passed / 0 failed / 58 subtests ✅ |
| C3 主链 11 步 + STAGE_REGISTRY 12 项 双轨   | 维持 FLOW_CHAIN=11 (兼容既有测试)；vsa-static 走超集 |
| C4 DR 参数覆盖 | `_diff_vs_contract` 从 electrical 单节扩展为四节全覆盖 (electrical / mechanical / sensing / environment)；新增 `test_c4_dr_full_param_coverage.py` |
| C6 README / CHANGELOG 口径 | 28 verify + 441 pytest 数字同步；CHANGELOG 补 v0.9 条目 |
| C3/C4 中 vendor 侧改的 | 从 Gate 2 判据移出，改由 P7 (edalabcore 单元测试) 承担 |

### 全栈架构顶层文档 (4 份)

| 文档 | 角色 |
|------|------|
| `docs/full-stack-layering.md` (v2) | 分层骨架 + 三轴正交分解 |
| `docs/full-stack-unification.md` (v3) | 三轴耦合机制 + 时间常数谱 |
| `docs/embodied-silicon-to-cognition.md` | 总设计师视角: 信息域/能量域两套物理系统 + 9 层物理栈 + 跨域三核 |
| `docs/mathematical-foundations.md` | 全栈数学基础: Maxwell/半导体/拉格朗日/LCP/模态/Kalman/MPC/Port-Hamiltonian/信息论/VLA/STL/Sobolev 十层不完备性证明 |

---

## v0.8 (进行中, 2026-10-07) — v4「jar 入库 + 消费」定为硬件端唯一调用方式

### 仓库边界：只提供 jar，**不暴露 EdaLab 源码**

| 项 | 处置 |
|---|---|
| `vendor/edalab/*.jar` + `PROVENANCE.json` | **入库**（唯一 EDA 依赖来源，3.54 MB） |
| EdaLab Kotlin 源码（4 个 `.kt`，此前误提交） | **从索引移除** + `.gitignore` 拦截 `*.kt` / `edalabcore/` / `archive/edalabcore-migrated-*`（磁盘文件保留） |
| 训练检查点（13 个 `.pt`，单个 267 MB，合计 **3.24 GB**） | **从索引移除** + `.gitignore` 拦截 `*.pt` —— 超 GitHub 单文件 100 MB 硬限制，且属可由 `scripts/vla_train_full.py` 再生产的产物；数据集 `*.jsonl` 与配方保留 |

> 双轨模式下本项目自行实现的 Python 后端（G6/G10/G7/G2）**正常入库** ——
> 那属于 EmbodiedAI 原创实现，与 EdaLab 源码无关。

> **决策**：本项目调用硬件端（EdaLab EDA 能力）的途径**唯一** —— 经入库的编译产物
> `vendor/edalab/*.jar` 消费。Python 侧不重实现 EDA 引擎、不直连源码目录。
> 版本号随下次发布同步（当前 `pyproject` 仍为 0.7.0）。

### 依赖入库（单仓自足）

| 文件 | 大小 | 作用 |
|---|---|---|
| `vendor/edalab/edalabcore.jar` | 1,954,103 B | 101 个 Kotlin 文件的编译产物（**不自含 Kotlin 运行时**） |
| `vendor/edalab/kotlin-stdlib-2.2.21.jar` | 1,761,445 B | Kotlin 运行时（必需配套，否则 `NoClassDefFoundError`） |

- `.gitignore` 白名单 `!vendor/edalab/*.jar`（`git add --dry-run` 实证两件会入库）
- 覆盖 **12 个 Stage**，live 实测 12/12 `LIVE_OK`、`jar_source=vendor`
- 隔离实证：`EDALAB_HOME` + `GRADLE_USER_HOME` 均指向不存在路径时仍 `ALL OK`
- 最小 classpath 实证：仅两件 jar 直接调 java → relay-clutch rc=0（安全链 16.00ms PASS）、vsa-static rc=0（`vsa_curve_v1_1`）

### 单仓自足实证 + README 显性化（本轮新增）

**动机**：v4 的"单仓自足"此前只写在 README/demo 之外 —— 新加入者（或三个月后的
自己）无法**一条命令跑出证据**。本轮把"自足"变成可执行、可发现、可排障的资产。

| 交付 | 内容 |
|---|---|
| `src/embodied_ai/selfcheck.py` | 5 项可证伪断言：① 载体齐全 ② **指纹一致**（PROVENANCE sha256/size）③ 仅需 JRE ④ **隔离可跑**（EDALAB_HOME+GRADLE_USER_HOME 指向不存在路径仍 LIVE_OK）⑤ 后端可观测（`jar_source`）。**每个失败项都带修复指引** |
| `scripts/demo_selfcontained.py` | 薄壳（转发包内实现，**不重复逻辑** —— 反两处维护） |
| CLI `selfcheck`（`--json`） | 经 `register_command` 注册进全局注册表，`catalog` 可见 |
| `tests/test_selfcheck.py` | 10 项守护：5 项断言齐备 / 失败项必带 hint / vendor 必在本仓 / **已注册且带示例** / `--json` 合法且 ok / **脚本薄壳不重复实现** |
| README | 新增「一条命令证明单仓自足」+「⚠️ 别人跑不起来怎么办」（JRE 缺失/NoClassDefFoundError/jar 缺失/指纹失败 四类症状 → 修复表）+「贡献者须知」（新增调用方式必须注册；改 jar 必须走指纹更新流程） |

实测：`selfcheck` → **5/5 ✅ SELF_CONTAINED_OK**；`verify` **27/27 PASS**；`pytest` **397 passed**。

### 修复：注册表空表陷阱 + 双重注册（本轮由测试抓到）

| 缺陷 | 现象 | 修法 |
|---|---|---|
| **空表陷阱** | 只 `import cli_registry` 的消费方看到**空的注册表**（注册实际发生在 `cli` 导入时）→ 误判"没有命令"，正是"命令落角落"的另一种形态 | `CLI_REGISTRY` 改为 `_LazyRegistry`（首次访问自动 `ensure_registered()` 补齐） |
| **双重注册** | `python -m embodied_ai.cli` 时 cli 作为 `__main__` 执行，懒注册表又 import `embodied_ai.cli` → 二次执行 → `重复注册 (audit)` ValueError | `ensure_registered()` 检测 `__main__.__file__ == cli.py` 时直接返回 |

两个缺陷均由 `tests/test_selfcheck.py` 抓到（前者由"注册表可发现性"断言，后者由 CLI 子进程断言）。

### 任务 B — 双轨缺口 G10 / G7 / G2（本轮完成，详见后续段落）

### B4 收尾 — 腱/凝胶柔体参数进制造容差 DR（2026-10-08）

### 断点 ③ 修复：最小 MissionManager — OS 决策中枢（2026-10-08）

**缺口**：无调度器/行为树/世界模型；`run()` 是 sleep 占位——审计定级 D。

**交付（`runtime/mission.py`，约 290 行）**：

| 组件 | 内容 |
|---|---|
| `MissionTemplate` | 任务模板 = 技能序列 + 关键词 + 超时 + 完成条件；内置 4 个（pick_and_place / grasp_only / wipe_surface / clean_and_pick），**只用控制器已注册技能**（grasp/place/wipe，测试守护不虚构） |
| `MissionState` | 状态机：PENDING → RUNNING → SUCCESS/FAILED/ABORTED；**非法转移必拒**（终态不可复活） |
| `MissionManager` | `parse("一句话")` → 关键词匹配模板（不可识别必报 `MissionError`）→ `run()` 驱动 20Hz 环（每步：安全检查→观测→技能下发→step→完成判定）→ 终态 |
| Bus 集成 | 状态变更发 `SAFETY_EVENT` topic（mission:started/success/...）——复用断点 1 的 Bus，可观测 |

**端到端实测**：
```
"抓起杯子放到桌上" → pick_and_place (grasp→place)
→ 20Hz 环 2 步 → SUCCESS
状态链: pending→running→success
```

**反证路径全部验证**：超时 → FAILED（"timeout: N steps"）；安全事件 → ABORTED；
不可识别指令 → `MissionError`；重复模板注册 → 拒；终态复活 → 拒。

**守护**：`tests/test_mission_manager.py` **15 项**（解析 5 / 状态机 3 / 端到端 3 /
失败路径 3 / 技能覆盖 1）；verify 新增 `mission_manager`。
全量 **521 passed** + verify **31/31 PASS**。

**8 步竖井覆盖**（审计建议）：①指令→②状态机→③技能调度→④20Hz 执行 ✅（本轮）；
⑤CAN 帧→⑥遥测→⑦DR→⑧回流：⑤⑥ 已有（Bus + DBC），⑦⑧ 已有（bigdata 飞轮）——
**竖井骨架全通，待真机数据激活**。

### 断点 1 修复：最小 Bus 抽象（2026-10-08）

**缺口**：层间 import 硬耦合，contracts/ 无运行时载体——审计定级 D。

**交付（`runtime/bus.py`）**：Topic 枚举（6 主题对齐三栈）+ Message 信封（timestamp/seq/source/schema_version）+ InProcessBus（同步 pub/sub，线程安全，单订阅者异常不阻断）+ schema 校验拒垃圾 + make_bus_backed_controller 一行接入 20Hz 环（原行为零破坏）。CAN-FD 帧是一等消息（复用 DBC 600 帧验证路径）。

守护：test_runtime_bus.py 12 项（含反证）；verify runtime_bus。

### 断点 2 接线：load_ckpt → policy → controller（2026-10-08）

**缺口**：训练产出 .pt 但从未进控制环。

**交付（`control/policy_loader.py`）**：PolicyLoader（.pt→重建模型→strict load_state_dict→infer_action，失败必抛 CheckpointError）→ VLAExecutablePolicy（维度适配+provenance）→ attach_policy_to_controller（无 torch 降级 ReplayPolicy 诚实标注）。

实测：backend=vla_ckpt，控制环非零动作，推理 10.5ms。守护：test_policy_loader.py 8 项；verify policy_loading。




**缺口**：层间协作靠 import 硬耦合；contracts/ 只是静态 YAML/DBC，**没有运行时载体**——
审计报告定级 D（"无 Bus 抽象、无消息 schema"）。

**交付（`runtime/bus.py`，约 270 行，零外部依赖）**：

| 组件 | 内容 |
|---|---|
| `Topic` | 枚举化主题（OBSERVATION / ACTION / CAN_JOINT_CMD / CAN_TELEMETRY / SAFETY_EVENT / DR_UPDATE）——对齐三栈（控制L/物理P/数据D/安全S） |
| `Message` | 信封：timestamp / seq（全局单调）/ source / schema_version——每条消息可追溯 |
| `InProcessBus` | 同步 pub/sub（20Hz 环内零序列化开销）；线程安全；单订阅者异常不阻断总线；stats 可观测（published/delivered/dropped/rejected） |
| `_TOPIC_SCHEMAS` | 每 topic 的 payload schema 校验——垃圾消息在入口被拒 |
| `make_bus_backed_controller()` | 一行接入 20Hz 控制器（step 内发布 OBSERVATION + ACTION；原行为完全保留——既有测试零破坏） |

**CAN-FD 是一等公民**：`encode_joint_cmd`/`decode_telemetry`（R4 已 600 帧 0 失配）的帧对象直接作为 Bus payload 流转，解码往返在 Bus 上验证通过。

**端到端实测**：控制器 step → OBSERVATION(1) + ACTION(1) 上 Bus → CAN_JOINT_CMD(3帧) 发布 → 订阅者收到 → `crc_ok=True`；消息 `seq=2, source=controller_20hz` 可追溯。

**诚实边界**：跨进程/跨机器（真实 ROS2 对接）不在此轮——那是 Phase P 硬件侧；内部纯函数调用保持直调（合理耦合不消息化）。

**守护**：`tests/test_runtime_bus.py` **12 项**（含反证：垃圾 payload 必拒 / 坏 topic 必报 / 无订阅不崩 / 单订阅者异常不阻断）；verify 新增 `runtime_bus`。全量 **506 passed** + verify **30/30 PASS**。

### jar ↔ 源码漂移治理（两层闸门）

风险：jar 是冻结产物，原 EdaLab 源码演进不会自动同步 → 静默漂移。

| 层 | 抓什么 | 守护 |
|---|---|---|
| L1 指纹 | jar 被替换/重新构建（sha256 变） | `verify` 项 `vendor_jar_provenance` + `tests/test_vendor_provenance.py`（9 项，含 4 项反证） |
| L2 行为锚定 | 源码演进导致引擎输出变化（jar 该更新了） | `verify_b3_antagonistic`（锚定值 166.39N）、relay-clutch 安全链 16.00ms、forge-core EPS ID |

元数据载体 `vendor/edalab/PROVENANCE.json`（schema `vendor_jar_provenance_v1`），含
sha256/size/来源/构建命令/覆盖 stage/`update_procedure`。

### C1 (P0) — VLA 视觉截断 58% 修复

原 `block_size=256` 远小于 `min_block_size`(603~613，由 196patch×3view=588 视觉 token
决定)；`vla_model.forward` 的截断是 `ctx[:, -block_size:, :]`（保留**尾部**），而序列
开头是视觉 token → **视觉仅保留 42%，约 58% 被静默丢弃**，loss 曲线却看不出异常。

| # | 修复 | 依据 |
|---|---|---|
| 1 | `block_size` 256 → **640** | 延迟实测（Orin NX INT8）：base 0.90→1.15ms、large 1.77→2.28ms，**远低于** 15ms 模型预算与 50ms 闸门 → **无需**牺牲 num_views/img_size |
| 2 | 截断不再静默 | `vla_model` 加 `_truncation_count`/`_dropped_tokens` + 首次/周期告警 |
| 3 | `vision_encoder.pos_embed` `torch.zeros(...)*0.02` → `torch.randn(...)*0.02` | 原式 **恒等于全零**（0×0.02=0），视觉空间位置信息完全丢失 |
| 4 | `vla_trainer.estimate_loss` 兼容 DataLoader | 原对 DataLoader 判否 → batch=None → 静默返回 **loss=0.0**（"训练完美"是假的）；现支持 `iter()` 协议，取不到 batch 返回 **NaN 并告警** |

守护：`verify` 项 **`vla_block_size`**（同时断言"不截断"**且**"不击穿 15ms"）+ `tests/test_c1_vla_blocksize.py` 8 项（含反证）。

### L3 — MCTS 接真实目标（决策不再空转）

| 空转证据（评估 §3.6） | 修复 |
|---|---|
| `search()` 收 `goal_fn` 但**从不调用**，终局靠 `_default_physics` 硬编码 (`timestep>=8 or stability>0.9`) | `goal_fn` 真实接入 `_is_goal()`，供 `_expand`/`_simulate` 消费；`goal_reached` 仅在目标函数达成时置位（物理终态不算） |
| `stability = min(1, n*0.3+0.1)` **单调不减，无失稳通道** | 引入摩擦锥判据 ‖f_t‖ ≤ μ·f_n（切向比按 grasp_type：power .35 / pinch .70 / hook .50），滑脱 → 稳定性下修 |
| `dr_override()` 设的 `_mu_override`/`_k_override` **无读取点**（飞轮断裂） | `_default_physics` 真实消费 μ（优先级：DR 注入 > 岩点几何 > 默认 0.8） |
| 奖励 = `stability_margin`，而它正是物理产出 → **奖励与转移同源** | 多分量：稳定性 + CWC 摩擦锥余量 − 滑脱惩罚 − 能耗 − 路径代价 |

守护：`tests/test_l3_mcts_goal.py` 9 项（含反证：永假目标不得误报达成、异常目标不崩溃、高 μ 不误滑）。
**测试抓到一处实现 bug**：置位逻辑分散在 `_expand`/`_simulate` 两处，导致物理终态被误记为"达成目标"——已收敛到 `_is_goal` 单点。

### 本轮其它落地

- **R4 CAN-FD ↔ DBC 对齐**：修复 `decode_telemetry` 两处位移错（`actual_current_a` 误
  `>>5` 应无移位、`temp_c` 误 `>>5` 应 `>>3`）；DBC 扩 6 关节遥测（消息数 9→14）；
  新增独立参考编码器对照测试 + `verify` 项 `canfd_db_cross_check`（600 帧 0 失配）
- **B2 P1.4 Stage 化**：`STAGE_REGISTRY` 12 项 + `run_eps_job` 数据管道（`FLOW_CHAIN`
  保持 11 步不动以兼容既有测试）；CLI `couple --eps-job`
- **B3 VSA-ANTAGONISTIC 选型**：★ 修正契约漂移 `vsa_antagonistic_peak_n` **109.0 → 166.39**
  （109 是 V-2 修复前过期口径；选型结论不变：166 < SEA 347 < MACCEPA 525）
- **路径清理**：gradle 缓存与 java 回退的硬编码用户路径改为 `GRADLE_USER_HOME` /
  `JAVA_HOME` / `EMBODIED_JAVA` 跨平台解析

> 自检规模：`verify` **26/26 PASS**，pytest **348 passed**（339 + 9 溯源测试）。

---

## v0.7 (2026-10-06) — 从"图纸库"到"可运行闭环" (缺口批量补齐)

> 触发: 全链条完成度评估给出"总体 30-35%", 结论是
> **"一架极其完备的图纸库 + 半数可运行骨架代码, 但没有一块 PCB、一个电机、
> 一条训练数据、一次真机运行"**。本版针对评估列出的缺口逐条落成代码,
> 并把优先级 (1)(2)(3) 三条里程碑的**可验证部分**先行落地。
>
> 自检规模: `embodied-audit verify` 从 8 项扩到 **20 项**, 全绿; pytest 191 通过。

### 新增模块

| 缺口 | 模块 | 内容 | 验收口径 |
|---|---|---|---|
| 训练管线 30% | `models/nn_core.py` | 纯 numpy MLP + 反向 + Adam/SGD (无 torch 也能真训练) | 有限差分 vs 反向 误差 <1e-9 |
| DiffusionPolicy 桩 | `models/diffusion_policy.py` | 真实 DDPM (cosine 调度 + ε 预测 + DDIM 少步) | 损失降 91%, 采样距离 0.055 vs 随机 0.38 |
| 无 BC | `models/bc_trainer.py` | 数据集 + 归一化 + 训练器 + 环境内评估 | 成功率 1.0 (= 专家), MSE 降 1.7e4 倍 |
| 无 PPO/SAC | `models/rl_algorithms.py` | PPO (GAE+clip) / SAC (双 Q + 自动温度) | PPO 末段回报 −61 > 随机 −73 |
| G6 柔性体 5% | `dynamics/soft_body.py` | XPBD 质点—约束网格 + 抓取 + 滑脱判据 | 拉伸 <0.5%, 悬垂收敛, 滑脱判据非恒真 |
| G9 双足 40% | `control/whole_body.py` | 3D ZMP 支撑多边形 + 6-DoF 腿部 IK + 力分配 | IK 往返 1e-16, 0 失稳, 前进 0.6m |
| 优先级 (1) | `hardware/foc_shutdown.py` | SVPWM + FOC 电流环 + 关机链时序 + HIL 用例 | 伏秒平衡 5e-15, 电关断 0.8µs, 停止角 5.3° |
| G11 灵巧手 0% | `hardware/dexterous_hand.py` | 力闭合判据 + 4 候选选型矩阵 | 推荐 F5-欠驱动 (覆盖 1.0 / 2 执行器) |
| Sim-to-Real 35% | `simulation/real_robot_env.py` | CAN-FD 传输层 (mock/socketcan/serial) + 安全闸门 | mock 跑通闭环, 闸门拦截超限动作 |
| 优先级 (2) | `deploy/jetson_bench.py` | roofline 延迟估算 + 实测注入标定 | tiny@OrinNX 0.36ms; 无 NPU 时超预算 |
| G12 终身学习 0% | `learning/lifelong.py` | EWC + 回放 + forgetting/BWT 度量 | 遗忘 0.492 → 0.005, λ 过大会压死可塑性 |
| G10 认证 10% | `safety/certification.py` | HAZOP + SIL/PFH/SFF + 架构约束 + V&V + RTM | PFH 6.6e-8, SFF 96.9%, RTM 100% |
| L7 GUI 0% | `gui/` | 零依赖 `http.server` + 单页仪表盘 + 7 个 JSON API | 起服务→拉 API→下发 STO→关服务 |

### 修掉的既有缺陷 (不是新功能, 是还债)

1. **`VLAConfig.ablations()` 缺 `vla_small`** — CLI `--size vla_small` 与
   `_SIZE_TO_VLA` 支持该档位, 直接调 `ablations()["vla_small"]` 会 KeyError。
2. **STL 缺 `Until` / `Next` 算子** — 模块 docstring 写了 `ρ(X[a]φ,t)=ρ(φ,t+a)`
   但代码里没有; 补 `STLNext` / `STLUntil` 与 `release_only_after_stop` 规约。
3. **`_last_t` 对 ndarray 信号抛 AttributeError** — `_interp` 承诺支持两种信号
   格式, 但 G/F/U 三个时序算子走 `_last_t` 时必崩, 承诺是假的。
4. **`NanoGPTAdapter` 与测试/契约不一致** (4 项预存在失败): `probe()["total"]`
   应为 3 (把 checkpoints/data 降级为可选资产)、`export_training_plan` 计划缺
   `config` 快照、缺 `to_vla_config()` (未知档位静默回退到 tiny)、
   缺 `validate_bc_dataset()` 静态方法 — 现已补齐并加"未知档位抛 ValueError"。
5. **DBC 位置信号量程不足**: `target_pos_deg` 15-bit × 0.001 ⇒ 量程仅 32.767°,
   连一个 ±90° 的关节都发不出去 (实测所有指令被饱和)。改为 ×0.01
   (量程 327.67°, 配合 180° 零点偏置覆盖 ±147°)。
6. `verification.py` 注解用了未导入的 `typing.Dict`。

### 实现过程中踩到并写进注释的坑

- **手写训练框架**: `zero_grad()` 若重新绑定 `grads` list, 优化器持有的旧引用
  会永远读到零梯度 — 参数一动不动, 而 loss 曲线看起来"在跑"。必须原地清零。
- **DDPM 调度表**索引偏移不会让训练 loss 变难看 (训练/采样共用同一张表),
  只会让采样发散到 |x|~60; 且 T=50 的低维扩散必须给 β 加上限 (β=0.9 时
  `1/√α` 单步放大 3.1 倍)。
- **LIPM 必须用解析解推进**: 显式欧拉的截断误差被 +ω 特征根指数放大, 4 步后
  从"每步 0.15m"退化成"末步倒退, 速度 −1.42m/s"; 且必须按 Kajita 解析周期解
  初始化 (x₀=p−a, v₀=aω·coth(ωT/2)), 否则起步就被前方的支撑脚推回去。
- **SAC 重参数化梯度**: `−½ε²` 项对 mean 无梯度, 只经 tanh 修正项求导; 写错
  会把 log_std 推向 −∞ (log π 涨到 +34)。
- **SVPWM** 用零序注入而非七段式扇区表: 扇区表写错时占空比仍在 [0,1]、波形
  仍像正弦, 只有伏秒平衡校验能发现 (初版重建误差 12.5V)。
- **电流环极点配置**: `kp=L·ωc, ki=Rs·ωc` 且须满足 `ωc·T<2`; 早期 kp=8 让
  快极点落在 3.8e4 rad/s, 离散振荡, 稳态误差 131%。

### 仍然真实存在的缺口 (不掩饰)

- **PCB 实物 / 真实电机 / 真机部署: 依然为零**。v0.7 做的是把"板子到货前必须
  钉死的东西"钉死 (调制器正确性、关机链时序、HIL 用例清单、安全闸门),
  但 `ShutdownChain` 里的数值仍是设计值 + 器件手册量级, **不是实测**。
- `RealRobotEnv` 的 `mock` 传输是"带噪声/延迟/丢帧的一阶关节模型", 不是真机;
  `socketcan` / `serial` 两条通路需要 `python-can` / `pyserial`, 且未在任何
  真实板卡上跑过。
- 训练数据仍为**合成**数据 (专家控制器 rollout / 解析函数拟合), 无真实示教。
- VLA 仍无 torch 环境下的真实训练记录; nanoGPT 仓库不在默认路径。

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
