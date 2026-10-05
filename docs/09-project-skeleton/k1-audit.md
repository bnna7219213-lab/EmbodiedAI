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
