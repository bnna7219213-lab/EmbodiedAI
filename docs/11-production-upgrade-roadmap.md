# 生产级升级改造路线图 (Production Upgrade Roadmap) v1.0

> **日期**: 2026-10-07
> **基础**: v0.7 + Phase A (climb-soft-robot) 全层次审计（含关联项目 EdaLab 跨仓库审计）
> **性质**: 本路线图不是"新增功能清单"，而是把审计发现的全部 P0/P1/P2 缺陷收敛为
> 可执行、可验收、带证据出处的升级改造序列。每条任务都绑定审计证据（文件:行号），
> 每个阶段都有量化完成定义（DoD）。
>
> **当前基线（实测）**:
> - `pytest -q`: **251 passed / 2 failed / 35 subtests**（76s）
> - `python -m embodied_ai.cli verify`: **20/22 PASS, RESULT=VERIFY_FAIL ❌**
> - 生产级 5 判据（CI 可复现 / 安全证据独立 / 总线契约一致 / 端到端闭环 / 数据可回流）**全部未达**

---

## 一、生产级判据（目标定义，不可降级）

| # | 判据 | 量化标准 | 当前状态 |
|---|---|---|---|
| G1 | CI 全绿且可复现 | `verify` 22/22 PASS 且退出码 0；在任意干净 clone（含 CI 容器）上 pytest 全绿 | ❌ FAIL + 本机路径锁死 |
| G2 | 安全证据独立无循环 | 安全链每个 `measured_*` 都有物理测量或明确标 `assumed_*`；SIL2 定量模型内部一致；STL 鲁棒度真实参与安全判定 | ❌ 常量查表 + 循环自证 |
| G3 | 总线契约与硬件一致 | CAN-FD 编解码与 `joint_control.dbc` 逐位一致（含独立对照测试）；6 关节全反馈；sync 帧 CRC 正确 | ❌ 位框错位 + 字段重叠污染 |
| G4 | 端到端闭环真实打通 | 攀岩场景在仿真中可 step（墙面+岩点+接触参数）；触觉通道有数据源；决策层 goal/奖励/DR 真实参与；控制层 20Hz 含握力调度；训练层目标真值真单位 | ❌ 仿真零覆盖 + 触觉无源 + 决策空转 + 训练学零 |
| G5 | 数据可回流可再生产 | DR 飞轮命名统一且契约参数全覆盖；VSA/安全数值有 provenance；双仓库版本控制；跨仓库一致性有自动化测试守护 | ❌ 三套命名 + 永久 missing + EdaLab 无 git |

**升级原则**: G1→G5 是递进依赖关系。G1 不修，后四项的验收全部无效（验收链本身是红的）。
G2 是安全红线，与 G1 同级优先。

---

## 二、阶段总览

```
Phase R 止血修复 (1-2 周)      Phase C 契约对齐 (2-4 周)
┌──────────────────────────┐   ┌──────────────────────────────┐
│ R1 verify 两条红          │   │ C1 VLA 视觉截断修复           │
│ R2 安全链去循环自证        │──▶│ C2 训练目标/单位修复          │
│ R3 攀岩验收路径解耦        │   │ C3 主链 12 步对齐             │
│ R4 CAN-FD vs DBC 对齐     │   │ C4 DR 飞轮命名统一            │
│ R5 柔顺峰值循环论证消除    │   │ C5 EdaLab 消费端修复          │
│ R6 硬编码路径清理          │   │ C6 文档口径诚实化             │
│ R7 机会约束可用化          │   │ C7 仿真环境默认修复           │
│ R8 EdaLab 版本控制         │   │ C8-C10 可复现性收敛           │
└──────────────────────────┘   └──────────────────────────────┘
            │                                │
            ▼                                ▼
Phase L 真实闭环 (1-3 月)      Phase P 生产认证 (3-6 月)
┌──────────────────────────┐   ┌──────────────────────────────┐
│ L1 MJCF 攀岩场景          │   │ P1 HIL 实测注入               │
│ L2 接触动力学修复         │   │ P2 FOC 打样 + 关机链实测      │
│ L3 MCTS 真实化            │   │ P3 安全认证真证据             │
│ L4 触觉→握力调度闭环      │   │ P4 真机通电顺序               │
│ L5 训练/RL/STL 修复       │   │ P5 GUI 安全互锁               │
│ L6 灵巧手攀岩任务集       │   │ P6 双仓库 CI                  │
│ L7 本体/腱模型真实化      │   │ P7 edalabcore 单元测试        │
│ L8 其余清单               │   │ P8 性能验收 (20Hz / <50ms)    │
└──────────────────────────┘   └──────────────────────────────┘
```

依赖关系: Phase R 是后续一切的前提；Phase C 依赖 R；Phase L 依赖 C；
Phase P 依赖硬件到货（外部阻塞，接口已备）。

---

## 三、Phase R — 止血修复（1-2 周）

> **DoD**: `verify` 22/22 PASS 且退出码 0；pytest 在临时目录 clone 中全绿；
> `climb_scene` 不依赖仓库外绝对路径；`safety_budget.yaml` 无 `measured_*` 常量自证。

| ID | 任务 | 审计证据出处 | 验收标准 |
|---|---|---|---|
| R1 | 修复 `verify` 两条红 | `tests/test_k1_skeleton.py::TestK1CLI::test_cli_commands`（`verify --no-pytest` SUBFAILED）、`tests/test_k4_simulation.py::TestK4H_Verification::test_run_all_returns_results`（`check='safety_budget'` SUBFAILED） | 先诊断两条失败根因（预期为 R2/R4 的连带），修复后 `verify` 22/22 PASS；`cli_smoke` 不崩即过项补一条"verify 子命令退出码必须为 0"的硬断言 |
| R2 | 安全链去循环自证 | EdaLab `RelayClutchDynamicsEngine.kt:202-212`（`electricalUs=0.8f`/`arcMs=8f`/`mechIsolationMs=8f` 常量查表，形参 `relay/clutch` 零引用）；`contracts/safety_budget.yaml` `measured_us: 0.8`；`MultiPhysicsCoupler.kt:754-757`（`1e-7f` 注释 0.1 FIT 实为 100 FIT）；`src/embodied_ai/safety/formal/budget_closure.py:40-48` 读 `measured_*` | ① EdaLab 侧改为最小物理模型（继电器吸合/电弧能量/机械行程，哪怕一阶近似）或字段改名 `assumed_*`；② EmbodiedAI `safety_budget.yaml` 同步改 `assumed_*` 并加 `validation_status: assumed_pending_hil`；③ PFH 注释单位修正（0.1 FIT→100 FIT）；④ `budget_closure` 增加"assumed 占比 >50% 则安全结论降级为 provisional"的诚实标志 |
| R3 | 攀岩验收路径解耦 | **已闭环 (2026-10-17 复核)**。原证据"`climb_scene.py:310` 写死 `C:\Users\<用户名>\workspace\EdaLab\experiments_vsa_curve.json`；仓库内 JSON fixture 数=0"两项均已不成立: ① 该处现按 `EDALAB_VSA_JSON` → 仓库 `fixtures/experiments_vsa_curve.json` (已入库) → `EDALAB_ROOT` → 抛指明修复路径的 FileNotFoundError 四级解析; ② fixture 数=1 且随仓库分发。外部仓库默认路径已统一走 `embodied_ai.paths.ext_root()`, 不再含个人机器布局 | 剩余待办: `test_climb_cycle.py` 的"fixture 存在"守卫已具备 (fixture 入库); EdaLab 侧补生成脚本仍属 R8 |
| R4 | CAN-FD 编解码与 DBC 对齐 | `src/embodied_ai/simulation/real_robot_env.py:94-95`（`cur_raw=(d[2:4]>>5)&0x7FF` bits 21-31、`temp_raw=(d[3:5]>>5)&0x1FF` bits 29-37，与 DBC bits 16-26/27-35 不符且 29-31 重叠）；`:362` sync 帧 CRC 恒 0；`:376-379` 仅 joint0 遥测 | ① 编解码严格按 `joint_control.dbc` 逐位实现；② **新增独立对照测试**：用按 DBC 独立实现的"参考编码器"喂入参考帧，断言解码值一致（打破 mock 同错全绿的盲区）；③ sync 帧 `crc8(d[:7])`；④ 6 关节遥测全量（DBC 补 ID 304-320 或遥测循环复用 300+节点偏移） |
| R5 | 柔顺捕获峰值循环论证消除 | `src/embodied_ai/simulation/climb_scene.py:318-331`（二分反解 `v_capture_max` 后 `peak` 在构造上 ≈ `f_limit`）；`tests/test_climb_cycle.py:63` `assertAlmostEqual(r.f_soft_peak_n, 60.0, delta=6.0)` | ① `f_soft_peak_n` 改为报告**非循环量**：`v_capture_max/v_full` 比值 + `f_rigid`（104N，已非循环）；② 测试断言改为 `soft_ratio` 与 `v_capture_max` 在物理合理区间；③ 删除"被二分目标值回代"的呈现 |
| R6 | 硬编码本机路径清理 | `src/embodied_ai/cli.py:41-61`（PROJECTS 五兄弟仓库绝对路径）；`src/embodied_ai/adapters/edalab.py:64-77`（gradle 缓存 SHA 路径 + stdlib 2.2.21/2.2.10/2.1.20 三版本混搭）；`src/embodied_ai/adapters/nanogpt.py:21` | ① 全部改环境变量覆盖 + 相对解析 + 缺失时显式 skip；② Kotlin stdlib 统一为 EdaLab `deploy/artifacts/kotlin-stdlib.jar` 单版本；③ `cli.py PROJECTS` 缺失时不影响 verify（已非阻塞则保留，但文档化） |
| R7 | 机会约束握力可用化 | `src/embodied_ai/safety/chance_constrained_grip.py:1-21`（无 `import numpy`，`grip_demo` 必 NameError）；`:8` `z=√2(1−2δ)` 对 δ≠0.05 全错（δ=0.001 得 1.41 vs 正确 3.09）；`:10` `max(denom,0.05)` 掩盖不可行 | ① 补 `import numpy as np`；② z 改 `statistics.NormalDist().inv_cdf(1−δ)`；③ `denom<=0.05` 抛 `InfeasibleGripError`；④ `grip_demo` 纳入 `verification.py` 检查项；⑤ 与 `safety_budget.yaml:41` 的 `z=2.0` 口径对齐（统一 δ=0.05→1.6449 或 δ=0.025→1.96，两处同步） |
| R8 | EdaLab 版本控制 + provenance | EdaLab 无 `.git`/`.gitignore`；`edalabcore/build/libs/*.jar`、`deploy/artifacts/*.jar`、`edalab_out/`、`.box-agent/`、`2026-09-20-488e4f91/` 混放；`experiments_vsa_curve.json` 为孤儿产物（EdaCli.kt:531-590 只 println） | ① `git init` + `.gitignore`（build/、edalab_out/、output/、*.jar、.box-agent*、会话目录）；② `scripts/gen_vsa_curve.sh`（`vsa-static --json > experiments_vsa_curve.json`）并 README 记录 provenance；③ 已入库的 jar/产物从索引剔除 |

---

## 四、Phase C — 契约对齐（2-4 周）

> **DoD**: 跨层契约（CAN/VLA/DR/主链）两端字段、单位、命名有自动化测试守护；
> 默认配置不再静默残废；README 宣称与实现一致。

| ID | 任务 | 审计证据出处 | 验收标准 |
|---|---|---|---|
| C1 | VLA 视觉截断修复 | `src/embodied_ai/models/vla_config.py:114-125`（`block_size<min_block_size` 只 warn）；`src/embodied_ai/models/vla_model.py:194-196`（头部截断丢 ~58% 视觉 token） | ① 默认 `block_size ≥ 608`（vla_base 口径），或 `__post_init__` raise；② 截断发生时打日志并计数；③ `src/embodied_ai/models/vision_encoder.py:45` `torch.zeros(...)*0.02` 改 `torch.randn(...)*0.02`；④ `vla_trainer.py:99-107` 兼容 DataLoader（`iter(loader)` 协议），消除"传 DataLoader 静默得 0.0 损失" |
| C2 | 训练目标真值化 | `src/embodied_ai/models/replay_dataset.py:150-152`（action 张量仅 `[0,0]` 有真值，119 零元素稀释 MSE）；`src/embodied_ai/models/action_tokenizer.py:49,110` + `replay_dataset.py:84-86`（`joint_limits` ±π 弧度 vs `actual_pos_deg` 度） | ① 有效位 loss mask（只对真值位计 MSE）；② 统一单位（ReplayDataset 侧归一化到弧度，或 joint_limits 按度传入并显式标注）；③ 新增"训练目标非零占比"测试 |
| C3 | 主链 12 步对齐 | `src/embodied_ai/adapters/edalab.py:48-62`（FLOW_CHAIN 11 步）；EdaLab `EdaCli.kt:647-691`（cmdEmbodiedAiFull 实际 12 步，含 `vsa-static`；Kotlin 自身 Step 编号 10/11、11/12、12/12 错乱）；docstring 行号引用漂移（声称 L551-581，实际 L647-691） | ① FLOW_CHAIN 补第 12 步 `{"cli":"vsa-static","engine":"VsaVariableStiffnessEngine.kt"}`；② `tests/test_k2_integration.py:20-40` 从硬编码 11 步改为解析 `EdaCli.kt` 提取真步序列做跨仓库断言（或至少加 vsa-static）；③ EdaLab 侧修正 Kotlin Step 编号；④ edalab.py docstring 行号引用改为符号引用（`cmdEmbodiedAiFull`）防漂移 |
| C4 | DR 飞轮命名统一 | EdaLab `ManufacturingDataFlywheel.kt:150-246`（snake_case）；`contracts/dr_params/manufacturing_variance.yaml`（分节+snake_case，`environment.ambient_temp_c`/`electrical.*`）；`edalab-h5/output/.../manufacturing_data_contract.yaml:102-113`（camelCase `ambientTemp`）；`edalab.py:511-524` `_diff_vs_contract` | ① 以契约 YAML 为唯一真源，Kotlin 飞轮与 h5 导出全部改契约命名（`ambient_temp_c` 等）；② 飞轮补产 `electrical` 四参数（mosfet_rds_on/phase_resistance/battery_internal_r/current_sense_gain）；③ `canfd_latency_ms` 语义修正（延迟≠总线利用率，改从 FCT commTests 的实测延迟字段取）；④ 补"飞轮输出 vs 契约"全参数交集测试，消灭 `missing_from_dr` 永真 |
| C5 | EdaLab 消费端修复 | EdaLab `api/edalab_stages.py:73,94`（main 类 `edalab.core.cli.EdaCliKt` 不存在，真名 `EdaCli`）；`:123-134`（编码回退缺失）；`EdaCli.kt:268-283`（thermal-sim `w/h` 形参死参数）；`edalab-h5/output/.../hw_requirements.yaml`（encoderResolution 14 vs 19 矛盾、`keepoutZones: [[object Object]]`） | ① `EdaCliKt`→`EdaCli`；② 复用 EmbodiedAI `_decode_output` 的 GBK 回退链；③ thermal-sim 用实参或删形参；④ h5 契约修内部矛盾（即使暂无消费方） |
| C6 | 文档口径诚实化 | README "verify **20 项**"（实际 21 检查+pytest 共 22 行输出）；README "EdaLab **11 步**主链"（实际 12）；`docs/10-climb-soft-robot/climb-soft-robot-plan.md:57` "F_soft=60.0N 柔顺降 42%"（循环论证残留，见 R5）；`climbing_dynamics_core.py:18-24` docstring 声称 §4.3-4.6/§6.8 | ① README 数字与 `verification.CHECKS` 自动同步（加一致性测试）；② plan.md 的 A2 结论改写为比值口径；③ `climbing_dynamics_core.py` docstring 删不存在章节或补回实现（见 L2）；④ README 顶部增加"生产级缺口声明"区块，链接本文档 |
| C7 | 仿真环境默认修复 | `src/embodied_ai/simulation/reference_env.py:93-105`（默认 `state_dim=12,action_dim=12` → `ValueError`）；`mujoco_env.py:124`（默认场景 `table_pick` 引用不存在 `joint_0..6`）；`:92` `_qpos_start=nq−action_dim` 负索引；`:322` `site_name2id` 返回 -1 当合法；`base_env.py:46`（EnvConfig 默认场景加载即崩） | ① `__init__` 断言 `state_dim ≥ 2·action_dim`；② MJCF 补真实 7-DoF 臂或 motor 绑定实际 joint 数；③ `_qpos_start` 断言 ≥0 并用 `mj_name2id`；④ `ee_id<0` 显式报错；⑤ `mujoco_env.step` 补 `record_step`（BC 导出在 MuJoCo 侧恢复）；⑥ `export_episode(default=str)` 改 `.tolist()`/`.npy` |
| C8 | 可复现性收敛（一） | `src/embodied_ai/planning/mcts_integration.py:165/187/223/387`（全局 random/np.random）；`reference_env.py:153`；`base_env.py:132/142`；`soft_body.py:93`、`wall_contact.py:318`（rng 建了不用，seed 参数零影响）；`mujoco_env.py:217/224` | ① 所有随机源收敛到实例级 `np.random.default_rng(seed)`/`random.Random(seed)`；② `ClimbingMCTS.__init__` 收 `seed`；③ `soft_body`/`wall_contact` 删除无效 seed 参数或真正使用；④ `reference_env.reset(seed)` 不再污染全局 `np.random.seed` |
| C9 | 可复现性收敛（二）：MCTS 状态契约 | `mcts_integration.py:58-59`（`__hash__` 与 numpy 数组字段的 `__eq__` 不一致，一旦启用 set/dict 即崩）；`:90`（`math.log(parent.visits)` 无零保护）；`:454`（`proprio_encoder=nn.Linear(30,64)` 喂入长 12 数组，shape bug 被死代码掩盖） | ① `@dataclass(eq=False)` 或 hash 基于全字段元组；② `ucb1` 父节点零访问保护；③ `proprio_dim` 与 `joint_positions` 长度构造时断言；④ 删除或修复 `NeRFGuidedMCTS.heuristic_value`（shape bug + `scene_feat=randn` 噪声占位 + `self.nerf` 未用） |
| C10 | 触觉通道数据源 | `base_env.py:64`（观测契约含 `tactile`，无任何环境产出）；`mcts_integration.py:311`（`DiffusionPolicy` 默认 `obs_dim_cfg` 含 `"tactile":(4,64,64)`） | 在 `wall_contact`/`climb_scene` 输出虚拟触觉观测（`f_t/f_n/μ̂/σ_μ` 四维）注入 obs，或从 `obs_dim_cfg` 移除——二选一，不允许"契约有、数据无" |

---

## 五、Phase L — 真实闭环（1-3 月）

> **DoD**: 攀岩任务在仿真中可 step 并有行为断言；触觉→μ 估计→握力调度在 20Hz 控制器中真实接入；
> MCTS 有真实目标函数；灵巧手选型由攀岩任务集驱动；训练管线学真值。

| ID | 任务 | 审计证据出处 | 验收标准 |
|---|---|---|---|
| L1 | MJCF 攀岩场景 | `mujoco_env.py:34-39`（SCENE_PATHS 全 None）；三个场景 MJCF 无 `solref/solimp`；`docs/07-gap-analysis` 仿真缺口 | 新增 `climb_wall` MJCF：墙面 geom + ≥12 岩点 geom（位置/法向/摩擦域随机化）+ 绳/site + `solref/solimp` + 指尖 site 力传感器；`make_env("climb_wall")` 在 MuJoCo 可 step；`reference` 后端同步实现点质量攀岩动力学（与 `climbing_dynamics_core` 公式同源） |
| L2 | 接触动力学修复 | `climbing_dynamics_core.py:179,183`（ADMM ρ=200 缩放项不等价，仅 ρ=1 正确）；`:161` `kappa` 死参数；`:163-197` 无 `converged`/无边界保护；`:24` docstring 声称 §4.3-4.6/§6.8 不存在；`hunt_crossley_force`/`hc_damping_from_restitution`/`project_to_cone_3d`/`swing_pumping_sim`/`cwc_membership` 零调用零测试 | ① ADMM 改 `z − (I+ρAᵀA)⁻¹Aᵀ(Az − ρb + y)`，补 ρ 自适应（primal/dual residual balancing）；② `kappa` 实现为第二阶段目标或删除；③ 返回 `converged`/`residual`，`nc<2`/`max_iter<1` 显式抛错；④ 恢复 `__main__` 与 §4.3-4.6/§6.8 演示；⑤ HC 阻尼接入 `dyno_capture_check`（`hc_damping_from_restitution` 反解 b）或移入 archive 并修文档 |
| L3 | MCTS 真实化 | `mcts_integration.py:131`（`goal_fn` 零引用）；`:237`（稳定性单调不减、无失稳通道）；`:248-253`（奖励与状态转移同源）；`:222`（`contact_normal` 恒 `[0,0,1]`）；`:259-271`（`_mu_override/_k_override` 无读取点）；`:472`（"异步线程"实为纯同步） | ① `_select/_simulate` 接入 `goal_fn` 终局判定；② 状态引入 hold 几何（法向/μ/尺寸）与滑脱通道；③ 奖励改 `CWC membership + 滑脱概率 + 能耗 + 路径长度`；④ DR override 真实接入 `_default_physics`；⑤ async 要么实现（`threading`/`multiprocessing` + 结果回调）要么诚实标注同步 |
| L4 | 触觉→握力调度闭环 | `perception/multirate_ekf.py:44-50`（同一 `z_f` 在两个 H 下各更新一次 → σ_μ 低估）；`:31`（μ 无 `[0,1]` 投影）；`chance_constrained_grip.py`（从未接入控制）；`household_robot_system.py:71-84`（20Hz 循环无 μ/握力） | ① EKF 改联合 2×1 观测更新（H=[[1,0,0],[0,f_n,0]]）；② 每次更新后 `μ̂←clip(μ̂,0,1)`；③ `household_robot_system.step` 接入：EKF `μ̂,σ_μ` → `chance_constrained_grip` → 握力设定值下发；④ `grip_demo` 的 μ 收敛断言（\|μ̂_final−0.75\|<0.05）进测试 |
| L5 | 训练/RL/STL 修复 | `rl_algorithms.py:492-493`（SAC `truncated` 当 done）；`:154-165`（PPO `act()` logp 按未裁剪动作算、存裁剪后动作）；`:56`（`target_kl` 死配置）；`stl_specs.py:175-176`（state 信号忽略时间）；`:238-243`（`force_ramp_monitor` 实为瞬时幅值）；`iso13482.py:106-110,187-213`（速度/ramp 公式从未运行，STL 鲁棒度不影响 safe） | ① SAC ReplayBuffer 增 `truncateds`，Q-target 用 `(1−done_terminal)`；② PPO 统一裁剪前后口径（重要性权重用存储动作对应的 logp）；③ 实现 `target_kl` 早停或删字段；④ STL `_interp` 按 `signal["time"]` 索引，G 空窗语义统一；⑤ `force_ramp_monitor` atom 改真变化率；⑥ `iso13482.evaluate` 启用速度/ramp 公式，STL 鲁棒度 ≤0 生成 violation |
| L6 | 灵巧手攀岩任务集 | `dexterous_hand.py:171-180`（`FAMILY_TASKS` 家庭任务集，无攀岩）；`climb_scene.py:224,246`（`recommend()` 算了不用，硬编码 `_f5_hand()`）；腱耦合 `:159-166`（固定 1:1 比，无摩擦/回差/弹性） | ① 新增 `CLIMBING_TASKS`：crimp 指扣 / sidepull 侧拉 / jug 满握 / pinch 捏点 / dyno-lock 空中锁定 / thumb-lock 拇指锁定，各带指力/接触数/包络约束；② `design_matrix` 并列攀岩+家庭评估，推荐结论供 `climb_scene` 真实消费（F5 硬编码移除，加一致性校验）；③ 腱模型补：pulley 路由非线性耦合比、腱摩擦（库仑+黏滞）、回差、弹性伸长 |
| L7 | 本体/控制修复 | `whole_body.py:107`（`c=-sqrt(...)` 强制脚在髋下方，高岩点解错）；`:67,115`（joint_limit 定义不用）；`:161-175`（`SupportPolygon.contains` margin 两分支语义相反）；`:362-388`（`distribute_forces` 2 未知数 3 方程无残差检查，`mu_foot` 死配置）；`biped_locomotion.py:116-125`（落足公式与 docstring 不符，`step_time` 未用）；`:76-85`（`_Ad/_Bd` 死代码且 `_Bd` 疑似量纲错） | ① IK 支持脚在髋上方（pitch 角分支）；② `leg_ik` 检查 `geom.joint_limit`；③ `contains` margin 语义统一为内缩量；④ `distribute_forces` 加残差断言 + 摩擦锥检查（`mu_foot` 启用）；⑤ 落足公式三处统一（`calc_next_foot_placement`/`suggest_recovery_step`/docstring）；⑥ 删 `_Ad/_Bd` 或让 `step()` 统一离散化口径 |
| L8 | 其余已确认修复项 | `lifelong.py:94-96,148-149`（EWC 每任务覆盖 Fisher，≥3 任务早期不受保护）；`bc_trainer.py:87`（`eval()` 反序列化，改 `ast.literal_eval`）；`gui/server.py:43-50,181-192`（无锁并发 + STO→resume 无互锁）；`foc_shutdown.py:90-92`（死区补偿纯共模无效）、`:234`（`i_peak` √1.5 无依据） | ① EWC 改 online 累积 `F←F+F_new`；② `literal_eval`；③ GUI 加 `threading.Lock` + resume 需确认；④ 死区补偿按相电流符号逐相实现或删除；⑤ `i_peak` 系数修正 |

---

## 六、Phase P — 生产认证（3-6 月，外部采购阻塞）

> **DoD**: 每个安全数值有实测注入点；真机通电顺序有明确顺序与回退；
> 双仓库 CI 绿；性能实测达标。

| ID | 任务 | 说明 | 验收标准 |
|---|---|---|---|
| P1 | HIL 实测注入 | v0.7 已留落点：`ShutdownChain.stages[*].delay_s` / `bench_vla(measured_ms=...)` / `RealRobotEnv.avg_latency_ms` | 示波器实测关机链各段延迟注入 `safety_budget.yaml`（`measured_*` 在此之后才恢复真实语义）；`assumed_*` 占比降到 <50% |
| P2 | FOC 打样 + 关机链实测 | `foc_shutdown.py` 的 7 条 HIL 用例转真台架 | 自动关机 <1µs 示波器实测；电流环带宽实测 vs 800Hz 设计 |
| P3 | 安全认证真证据 | `certification.py:100-102`（PFH 直接求和，未按 HFT=1oo2 建模） | PFH 对 hft≥1 子系统按 1oo2 组合失效率或注明保守上界；RTM 每行链接到实测/HIL 证据 |
| P4 | 真机通电顺序 | Phase C 既有顺序：① 电机单关节台架 → ② 指尖柔性传感器标定（soft_body 实测形变曲线替换模型参数）→ ③ 整机墙面静态悬挂（whole_body 实测）→ ④ 首 dyno（全链实测） | 每步有量化通过线；任一步失败回退到仿真标定 |
| P5 | GUI 安全互锁 | `gui/server.py` STO→resume | resume 需二次确认 token；`DashboardState` 加锁 |
| P6 | 双仓库 CI | EdaLab 无 CI；EmbodiedAI CI 不含 EdaLab 侧 | EdaLab `gradle test`（补 P7 后）入 CI；EmbodiedAI `verify` 增 EdaLab 主链 dry 对齐检查（解析 EdaCli.kt） |
| P7 | edalabcore 单元测试 | `edalabcore/src/test` 不存在，build.gradle test 配置死配置 | VSA 引擎（能量/峰值/刚度边界）、RelayClutch（安全链延迟物理模型）、MultiPhysicsCoupler（PFH 单位）核心路径单测 |
| P8 | 性能验收 | VLA 推理 <50ms（20Hz）；控制环 1kHz jitter <100µs | `bench` 实测注入；`platform` 判定 jetson_orin_nx_16gb |

---

## 七、降级与砍除清单（诚实边界，明确不做）

以下模块审计确认为装饰性/死代码，本路线图**不投入修复**，改为诚实标注或删除：

| 模块 | 现状证据 | 处置 |
|---|---|---|
| NeRF 价值网络 | `mcts_integration.py:454`（shape bug + `scene_feat=randn` 占位 + `self.nerf` 未用） | 删除 `NeRFGuidedMCTS` 或标注"未实现"；MCTS 价值用启发式 |
| VQ 动作离散化 | `action_tokenizer.py:64-133`（无 codebook 训练路径，死代码） | 标注"未实现"；L6 前不启用 |
| 异步 MCTS | `mcts_integration.py:472`（纯同步，无线程） | L3 中二选一；若选同步，文档删"异步"宣称 |
| 触觉渲染仿真（TACTO/Taxim） | `docs/03-sensing-perception` 仅策略分析 | 维持文档级，不进代码路线 |
| 神经材质/BC7 等渲染管线 | 属其他项目（c:\engine）范围 | 不纳入 |

---

## 八、风险与依赖

| 风险 | 影响 | 缓解 |
|---|---|---|
| 硬件到货阻塞 Phase P | P1-P4 无法推进 | 接口已备（落点字段在）；Phase R-C-L 全软件可验收 |
| 跨仓库修改需同步（EdaLab Kotlin + EmbodiedAI Python） | 单侧修改导致契约再漂移 | C3/C4 的"解析 EdaCli.kt 做跨仓库断言"是根治手段；EdaLab git init（R8）是前提 |
| 修复 verify 红可能暴露更深失败 | R1 工作量不可控 | 先诊断再修；若 safety_budget 红根因是 R2 连带，则 R1/R2 合并 |
| 大量硬编码路径清理破坏本机习惯 | 短期不便 | 全部改"环境变量优先、本机路径为 fallback"，不一次性删 |
| Phase L 工作量大（1-3 月） | 与 Phase B/C 人力冲突 | L1-L4 优先（闭环主线），L5-L8 可并行分摊 |

---

## 九、验收总表（Exit Criteria）

| 阶段 | 完成定义（DoD） | 验证方式 |
|---|---|---|
| Phase R | `verify` 22/22 PASS 退出码 0；临时 clone pytest 全绿；`safety_budget.yaml` 无常量 `measured_*`；`climb_scene` 无仓库外绝对路径；CAN 解码与独立参考编码器一致 | `scripts\verify.bat` + 干净环境重跑 + 新增契约对照测试 |
| Phase C | 默认配置不静默残废（VLA/EnvConfig/MuJoCo 场景）；跨层契约两端有自动化测试；README 与实现数字一致；所有随机源实例级可复现 | pytest + 契约对照测试 + README 一致性测试 |
| Phase L | `make_env("climb_wall")` 可 step；20Hz 控制器含握力调度链路；MCTS `goal_fn` 生效且奖励含 CWC+滑脱+能耗；灵巧手推荐由 `CLIMBING_TASKS` 驱动；训练目标非零占比 >95% | 新增 L1-L8 专项测试 + 全量回归 |
| Phase P | 安全数值实测注入占比 >50%；真机四步通电全过；双仓库 CI 绿；VLA <50ms、jitter <100µs 实测 | HIL 台架记录 + CI + bench 实测报告 |

---

## 十、与既有文档的关系

- 本文档**取代** `docs/10-climb-soft-robot/climb-soft-robot-plan.md` 中"已交付 ✅"的乐观口径；
  Phase A 保留为历史记录，其 A2 结论（F_soft=60.0N）由本文档 R5 修正为比值口径。
- 本文档的 Phase R/C 对应 `docs/07-gap-analysis/` 的 G1-G5 缺口的工程化收口；
  Phase L 对应 `docs/02-contact-dynamics/` 的仿真/触觉/规划落地；
  Phase P 对应 `docs/08-cross-project-fusion/` 的硬件自动化闭环。
- 每完成一个任务，在对应 `docs/` 文档的缺口清单中打勾并回链本文档任务 ID。
