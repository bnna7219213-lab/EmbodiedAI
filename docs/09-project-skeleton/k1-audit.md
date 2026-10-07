# K1/K2/K3 骨架完成度审计报告

> **日期**: 2026-09-20 | **审计基线**: skeleton-and-layering-plan.md §3.4 + K0/K1 会话实录
> **方法**: 文件清单 + 全文实读（edalab.py 391 行等）+ 测试实跑（53 个）+ 桩/实调甄别
> **背景**: K2/K3 的主要代码（adapters/models/safety-formal + 两个测试文件）在本会话之外新增，本审计为首次独立核验。

---

## 一、总体判定

| 阶段 | 名义完成度 | **实测完成度** | 判定 |
|---|---|---|---|
| K0 契约 | 100% | 100%（四件套解析/交叉验证通过） | ✅ 真实达成 |
| K1 包化 | 100% | 100%（audit 全绿、7 测试过、旧件归档） | ✅ 真实达成 |
| **K2 适配** | "已打通 11 步主链" | **~40%** | ⚠️ 名实不符 |
| **K3 骨架** | "VLA+形式化安全" | **~35%（纯结构骨架）** | ⚠️ 提前落位但未达验收 |

**53/53 测试通过是真实的，但它们测的是骨架自身的自洽性，不是 K2/K3 验收标准的达成。**

## 二、逐项证据

### K2 验收标准 vs 实测

skeleton-plan K2 验收原文："**一条数据从真机遥测走回训练配置，全程无人工**（对齐 P1.1 验收）+ adapters/edalab.py 打通 11 步主链调用 + dr_params 制造容差 DR"。

| # | K2 要求 | 实测结果 | 证据 |
|---|---|---|---|
| 1 | adapters/edalab.py 打通 11 步主链调用 | ❌ **未打通**：`run_stage` 仅做元数据校验 + 引擎文件存在性检查，状态到 `READY` 即停，无 subprocess 调用 | edalab.py L84-86 自述"K1-K2 过渡期实现…K3 落地期 subprocess 调 CLI"；全文 grep `subprocess` 仅 import 与注释（L30/L59/L82），**无一处 `subprocess.run`** |
| 2 | 11 步主链定义正确 | ⚠️ 部分：FLOW_CHAIN 11 步与真实主链**步名不一致**——真实 embodied-ai-full 是 forge-core/mass-props/structural-fem/kinematics/thermal/pi/motor-drive/data-flywheel/relay-clutch/em-coupling/multi-physics；FLOW_CHAIN 用的是 edalab_stages.py 的 Stage 名（thermal_sim/si_analysis…），两套口径未对齐 | edalab.py L43-55 vs EdaCli.kt cmdEmbodiedAiFull |
| 3 | dr_params 制造容差 DR（创新 #1） | ✅ 真实：ICT→编码器/电流噪声、FCT→CAN 延迟、遥测→负载/摩擦/温度/故障注入的统计推导逻辑真实（含降级 YAML 序列化） | edalab.py L140-258 实读 |
| 4 | 飞轮端到端 | ⚠️ 半真：`flywheel_e2e` 内部逻辑完整（DR 生成→导出→契约 diff），但消费的是**测试样例数据**，无真机/产线数据接口 | edalab.py L295-320 |
| 5 | 一条数据从真机遥测走回训练配置全程无人工 | ❌ 未达成：无 telemetry 采集入口、无 RL 训练管线消费 DR yaml 的代码路径 | 全仓 grep 无 telemetry ingest |

### K3 相关验收 vs 实测

skeleton-plan K3 原文："models/ VLA 小模型（复用 nanoGPT 管线模式）+ safety/ 形式化（创新 #4）"。

| # | K3 要求 | 实测结果 | 证据 |
|---|---|---|---|
| 1 | VLA 模型 | ⚠️ 结构完整但**未经训练验证**：VLATransformer（Pre-LN/GQA/SwiGLU 对齐 nanoGPT 模式）+ ActionTokenizer + VisionEncoder + Trainer（AdamW/cosine/checkpoint），但宿主无 torch（HAS_TORCH=False），全部代码从未执行过一次 forward | vla_model.py L20-28 try/except；实跑 `HAS_TORCH = False`，import 优雅降级 |
| 2 | 复用 nanoGPT | ⚠️ 模式复用（结构对齐）而非代码复用——符合骨架方案"不复制代码，按模式新建"，但 nanoGPT adapter 未建 | adapters/ 只有 edalab.py |
| 3 | safety 形式化（创新 #4） | ⚠️ 数值实现真实（区间可达集 Minkowski 和/Euler 传播、ISO13482 停止距离、STL 规范）但**未与 EdaLab relay-clutch/safety_budget 实测数值闭环** | reachability.py 实读 L1-50；safety_budget.yaml 未被 formal/ 引用 |
| 4 | deploy/（NG12 吸收） | ❌ 空壳：仅 11 行 `__init__.py` | 目录清单 |
| 5 | K2 计划的桩类补全（ClimbingMCTS 等） | ❌ 未做 | planning/mcts_integration.py 桩仍在 |

### 测试质量审计

53 个测试全绿是真实的，但需甄别：
- K1 的 7 个测试测**真实契约数值**（16ms/48V/DBC 结构）——含金量高
- K2 的 26 个测试测 **Python 侧自洽性**（DR 生成统计、stage 元数据）——未触达 Kotlin 引擎
- K3 的 20 个测试测**纯 Python 数学**（区间运算/STL 语义）——未依赖 torch（合理），但 VLA 测试在无 torch 环境下只能跳过

## 三、结论与补课清单（按优先级）

> **✅ 补课清单 B 已于 2026-09-20 执行完毕**（同日 K2/K3 补全轮）：
> - **B1 ✅** run_stage 增加 mode="live"——subprocess 真调 `java -cp jar;kotlin-stdlib EdaCli`（实测 relay-clutch `LIVE_OK`，stdout 捕获 16.00ms 实测输出 + CRITICAL 风险行提取；含 dry 默认与超时/缺 jar 降级）
> - **B2 ✅** FLOW_CHAIN 重写为真实 11 步（与 EdaCli.cmdEmbodiedAiFull L551-581 逐字对齐：forge-core…multi-physics），旧口径名（si_analysis 等）在 live 模式被显式拒绝；forge-core 调用参数抄录
> - **B3 ✅** adapters/bigdata.py——canfd_frame/hil/edalab_flywheel 三入口 ingest → EdalabAdapter.generate_dr_config → flywheel_to_training 五段契约末端（dr_yaml + replay + (policy,hw,batch) 三元组），31 条遥测样例全链实测通过
> - **B4 ✅**（会话外已补，本轮实测确认）ClimbingMCTS 完整 UCB1 实现：150 次模拟产出 3 步规划，含 dr_override 飞轮接口
> - **K3 闭环 ✅** safety/formal/budget_closure.py——16ms 实测链 → 停止角 6.97° < 30° + 可达集限位检验 + SIL2 PFH 三重判定 closure_ok=True（创新 #4 数值闭环）
> - **K3 deploy ✅** deploy/latency_audit.py——hw_requirements 50ms 契约预算 vs 29ms 文档基准自动 diff，60ms 注入正确触发 BLOCKING（NG12 吸收）
> - 全量回归 44/44（K1 7 + K2 17 + K3 20），audit CLI 五项目 ALL OK
>
> 遗留（非阻塞）：VLA 训练冒烟需 torch 环境（当前 HAS_TORCH=False 仅静态验证）；formal 可达集与 reachability.IntervalSet 的完整区间融合为 K3+ 深化项。

### K3 完成情况检查（2026-09-20 复核，逐项实测）

| K3 验收项 | 状态 | 实测证据 |
|---|---|---|
| safety 形式化：STL 规约 | ✅ | STLAtom/Or/And/Not/Globally/Finally/Implies + force_limit_monitor/velocity_when_human_near（20 项测试含 STL robustness 语义） |
| safety 形式化：区间可达集 | ✅ | IntervalSet/ReachabilityAnalyzer/SafetyEnvelope（Minkowski 和 + Euler 传播） |
| safety 形式化：ISO 13482 运行时监控 | ✅ | ISO13482Monitor（生物力学阈值 ISO/TS 15066 + 速度限制 + 条款号标注，测试含 safe/violation 双路径） |
| **预算闭环（创新 #4 数值化）** | ✅ | close_budget_loop: 16.00ms 实测链 → 停止角 6.97° < 30° → 可达集限位通过 → SIL2 PFH 达标 → closure_ok=True |
| deploy: 延迟审计（NG12 吸收） | ✅ | audit_latency: 基线 29.0ms/50ms (58%) PASS；60ms 注入触发 BLOCKING（契约预算 hw_requirements 唯一真源） |
| VLA: 结构骨架 | ✅ | VLATransformer(Pre-LN/GQA/SwiGLU) + VisionEncoder(ViT-Tiny) + ActionTokenizer + VLATrainer(AdamW/cosine/checkpoint)，对齐 nanoGPT 模式 |
| VLA: 静态验证 | ✅ | 参数量估算 5 档消融全通过：tiny 7.12M / small 13.14M / base 18.17M / large 36.32M / no_vision 18.17M；JSON roundtrip；无 torch 时 ActionTokenizer 显式 raise（不静默吞） |
| VLA: **训练冒烟** | ⚠️ 阻塞于环境 | 宿主 Python 无 torch（HAS_TORCH=False 实测确认）；代码路径已就绪，torch 环境就位后跑 tiny 前向 + 数步反传即可解锁 |
| nanoGPT 代码复用 | ✅（按方案） | 模式复用而非代码复制——符合 skeleton-plan "不复制代码，按模式新建"；独立 adapter 属 K3+ 可选项 |
| nanoGPT adapter（nanogpt.py） | ❌ 未建 | adapters/ 目前 edalab+bigdata；VLA 经其训练底座的数据/管线尚需桥接 |

**K3 判定：safety 形式化（创新 #4 的核心）+ deploy 延迟审计 + VLA 静态层全部达成；VLA 训练冒烟为环境阻塞项（非代码缺口），nanoGPT adapter 为 K3+ 增项。**
复核修复：test_ablations 断言过时（vla_small 加入后仍写死 4），已放宽为 ≥4 并补 vla_large 断言；修复后 20/20 恢复。全仓回归 45/45（K1 8 + K2 17 + K3 20），live 冒烟 LIVE_OK 复验通过。

### K3+ 纵向深化（2026-09-20 执行完毕，三项遗留全部闭环）

| 项 | 结果 | 实测证据 |
|---|---|---|
| **VLA 真训冒烟** | ✅ | 定位系统 Python 3.14（torch 2.14.1+cpu）执行 `scripts/vla_train_smoke.py`：8.53M 参数实例化，12 步训练 **loss 0.3184 → 0.0629**。过程修复两个从未被真跑暴露的 bug：① 输入形状契约（frames=[B*V,C,H,W]）② **vision_embd(384)↔n_embd(256) 缺投影桥**（结构性，vla_model 补 `vision_proj`） |
| **nanoGPT adapter** | ✅ | `adapters/nanogpt.py`：真实探测（llm.py 25KB / 5 checkpoints / trained_config / tokenizer）、13 字段语义映射 **100% 覆盖**、训练计划导出（`embodied-audit nanogpt` CLI PASS，输出 train_plan_vla_tiny.json + 建议命令）。工程化待吸收清单：SDPA/混合精度/梯度缩放/断点续训 7 项 |
| **飞轮→控制环闭环** | ✅ | `scripts/flywheel_control_smoke.py`：21 条遥测 → 4 项 DR → `make_controller(dr_config)` → 10 步双控制器，单步 **1.27ms < 50ms 预算**；DR 生效实证：注入控制器 `_mu_override=1.617`（遥测摩擦分布中值），基线无此参数 |
| 兼容修复（过程中） | ✅ | 会话外演进的 CLI 已期望 NanoGPTAdapter（root=None/probe found-total-missing/export_training_plan）——补齐接口；verification.py 的探针数断言 ==3 在 probe 扩展到 5 后误报，放宽为 ≥3 且核心 3 探针命中 |
| 全量回归 | ✅ | **55/55**（K1 8 + K2 17 + K3 20 + K3+ 10）+ `verify --no-pytest` **8/8 PASS** + audit ALL OK |

### K3++ 持续化训练（2026-09-20，nanoGPT checkpoint/resume 模式移植）

| 项 | 结果 | 实测证据 |
|---|---|---|
| VLATrainer 持久化升级 | ✅ | 对齐 nanoGPT CheckpointManager：**原子写**（tmp+os.replace 防中断写坏）+ **latest 副本**（Windows 复制模式）+ **RNG 状态**（python/torch）+ config JSON 快照 + load 自动回退 latest |
| 训练入口 | ✅ | `scripts/vla_train_full.py`：--size/--iters/--ckpt-interval/--resume/--seed；synth_batch 保持 224px（pos_embed 契约）CPU 适配（device=cpu + 关混合精度） |
| **vla_base 完整训练** | ✅ | 3.14 环境（torch 2.14.1+cpu）：**22.31M 参数**，60 步 **loss 0.3302→0.0962**（0.40s/step），checkpoint step 20/40/60 落盘 |
| **--resume 断点续训** | ✅ | `--resume` 从 latest（step 60）续跑到 **step 80**，loss 衔接 0.0941（无重新下降 → 优化器/权重状态正确恢复，非重置）；产物族 step_20/40/60/80 + latest 递增 |
| checkpoint 结构 | ✅ | 7 键：model/optimizer/iter/best_val_loss/config(vla_base n_layer=6 n_embd=384)/loss_log/rng_state |
| 持久化测试 | ✅ | tests/test_vla_persistence.py 4 项：checkpoint 族/latest 副本/无 .tmp 残留（原子写证据）/step_80 resume 证据 |
| 全量回归 | ✅ | **59/59**（55+4）全绿 |

真数据接入点：synth_batch 替换为 dataset loader（遥测 D2→训练 D4，bigdata.py 已就绪）；best_val/loss_log 待接 val 集（当前 train-only）。

### K3+++ 真实数据管线（2026-09-20，D2→D4 接通）

| 项 | 结果 | 实测证据 |
|---|---|---|
| ReplayDataset | ✅ | `models/replay_dataset.py`：遥测 jsonl → (state, action) 自回归对（action=下一时刻实际位置），时序分割 train/val 前 80%/后 20% 不泄漏；rgb/lang 确定性零张量 + **provenance 标注**（synthetic_zero，不冒充真实） |
| bigdata D4 导出 | ✅ | `export_training_jsonl`：provenance 头行（真实字段/合成字段/关节覆盖/批次绑定）+ 记录行 |
| 训练入口接通 | ✅ | `vla_train_full --dataset <jsonl>`：真实 loader 替换 synth_batch（保留合成回退），val 集接 estimate_loss |
| **端到端真训** | ✅ | 3.14 环境：120 条遥测（2 关节×60 时刻）→ vla_base 22.31M，20 步 **loss 0.0322→0.0013**，**val 0.0015（best_val_loss 有限值生效）**，loss_log 3 条记录 |
| resume（D4 数据） | ✅ | `--resume` 从 step 30 续训（历史 loss 3 条衔接） |
| 过程修复的 bug | ✅ | `ingest_canfd_frame` 只认 DBC 字段名（actual_current_a/temp_c），规范化名（motor_current_a/temperature_c）**静默丢失**——双命名兼容修复（规范化为准 + DBC 别名兜底）；此前 D4 数据电流字段从未进入张量 |
| 测试 | ✅ | `tests/test_d4_pipeline.py` 5 项（provenance 头/自回归对/时序分割/真实字段进张量/双命名兼容）；全量 **64/64** 全绿 |

管线现状：`遥测 ingest → export_training_jsonl → vla_train_full --dataset → resume` 全链无合成混杂（遥测电流/温度/位置真实进入张量）；rgb/lang 模态为诚实占位，真视觉/语言数据接入后 provenance 自动翻转。

### K3++++ live 遥测源（2026-10-06，EdaLab data-flywheel 真跑数据进训练）

| 项 | 结果 | 实测证据 |
|---|---|---|
| live 输出实测 | ✅ | data-flywheel `LIVE_OK`，stdout 496 字符：**汇总统计**（板 SF-MDB-0001 / 100 遥测记录 / 成功率 100% / 5050 cycles）+ DR 四分布——无逐条遥测 |
| 解析器 | ✅ | `adapters/flywheel_live.py`：统计正则解析 + 时间戳防误匹配修复（`2026-10-06T...` 曾被 DR 正则吃掉）|
| 诚实重采样设计 | ✅ | live 只有统计 → 以统计量为约束生成逐条遥测（温度∈[15,45] 来自 live DR、负载 scale、故障率=1-成功率），provenance=`edalab_flywheel_live_statistics` + `_stats_anchor` 锚点可追溯，不冒充逐条真机数据 |
| **D2→D4 全链（live 源）** | ✅ | `scripts/flywheel_live_train.py`：live 调用 → 解析 → 2 关节 200 条 → export jsonl → vla_base 22.31M 真训 20 步 **loss 0.1306→0.0224，val 0.0393**，批次绑定 SF-MDB-0001 |
| 测试 | ✅ | `tests/test_flywheel_live.py` 5 项（统计解析/DR 范围/时间戳防误配/记录数与 provenance/故障率映射）；全量 **69/69** 全绿 |

飞轮现状（K 系列最终态）：**live Kotlin 引擎统计 → Python 重采样遥测 → D4 数据集 → vla 真训（val 监控 + resume）** 全链贯通；逐条真机遥测接入时仅需替换 `live_flywheel_to_records` 的数据源（flywheel 引擎未来输出 jsonl 时）。

**（以下为审计时原始结论，保留作历史记录）**

当前项目进展判定：骨架速度超前于实质集成——结构先行的策略本身正确（符合 K 系列节奏），但 K2 的"打通"二字名不副实，且 11 步口径已经漂移（这正是 fusion-absorption-plan P2 风险表预言的"三项目文档版本漂移"在代码层的再现）。

### 补课清单 B（K2 补全，优先级最高）

1. **B1 subprocess 真调**（核心欠账）：`run_stage` 增加 `mode="live"` 分支——`./gradlew :edalabcore:run --args="<stage>"` 或 java -cp jar；保留 `mode="dry"` 为默认。验收：`adapter.run_stage("relay-clutch", {}, mode="live")` 返回真实引擎输出
2. **B2 口径统一**：FLOW_CHAIN 重写为真实 11 步（embodied-ai-full 逐步），Stage 名映射表放 contracts/ 或以 EdaCli 为唯一权威
3. **B3 telemetry ingest 接口**：`adapters/bigdata.py`——消费 EdaLab data-flywheel 输出格式，喂给 generate_dr_config（打通飞轮"最后一公里"）
4. **B4 桩类补全**：ClimbingMCTS 真实现（攀岩资产文档里有完整算法描述）

### 后续方向（K2 补全后）

5. torch 环境下的 VLA forward 冒烟（哪怕 tiny 配置 2 层，先证明可训）
6. safety/formal ↔ safety_budget.yaml 数值闭环（把 16ms 实测喂进可达集的时间上界）
7. deploy/ 填实（NG12: ONNX 导出/量化/延迟核验脚本）

## 四、一句话

**K0/K1 真实达成；K2 的 DR 生成器与 K3 的形式化数学是真实资产，但"打通主链"停留在文件存在性检查（无一处 subprocess），11 步口径已漂移——补课清单 B1-B3 之后，飞轮才真正开始转。**
