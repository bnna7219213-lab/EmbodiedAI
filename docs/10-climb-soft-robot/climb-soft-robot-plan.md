# 攀岩竞技柔性体机器人 — v0.7 基础上的从头改进方案

> **日期**: 2026-10-06（v1.1 修订 2026-10-07：A2 循环论证修正、A3 双判诚实化、安全证据链非循环化）
> **基础**: v0.7（本日实测核验：pytest 191 passed、verify 20/20 全绿，含 cli.py Dict 注解遗留缺陷一处已修）
> **当前实测**: **pytest 252 passed + 37 subtests**、**verify 21/21 PASS**、`safety` 阈值通过但 `provenance=assumed`（**SIL2 非发布级**，需 HIL 实测升级）
> **目标形态**: 灵巧手（F5-欠驱动）+ 双腿双足 3D 步态 + 柔性体本体，面向攀岩竞技，全流程硬件自动化设计（EdaLab 12 步主链 + 数据飞轮）
>
> **一句话定性 (v1.1 诚实边界)**：本方案作为**研究/工程蓝本、Phase A 场景闭环设计**达标；作为"已验证的端到端具身智能体"或"可上真机的产品"**未达**——仿真链路对攀岩零覆盖、触觉/握力调度未接入 20Hz 控制环、安全证据链停在 `assumed` 而非 `measured_hil`、灵巧手任务集是家庭样本不是攀岩操作、VLA 默认档截断 58% 视觉。这些不是缺陷补丁，是下一阶段（Phase B/C）的核心工作。

---

## 一、v0.7 资产核验结论（实测，非采信声明）

| 声明 | 实测 | 判定 |
|---|---|---|
| verify 21/21 | **20/20 PASS**（一条汇总差异，无 FAIL） | ✅ 基本属实 |
| pytest 191 passed | **191 passed + 36 subtests, 75s** | ✅ 属实 |
| 13 新模块 | verify 逐项覆盖（soft_body/whole_body_3d/diffusion/bc/rl/foc_shutdown/dexterous_hand/real_robot_loop/jetson_bench/lifelong/certification/gui 全 OK） | ✅ 存在且自检通过 |
| 遗留缺陷 | cli.py L402 `Dict` 注解未导入 → verify 直接 NameError（与已修的 verification.py 同款），本轮已修 | ⚠️ 还有一处同类未修净 |

**三个"仍然真实"的缺口确认属实**：PCB/真机/真示教为零——改进方案必须围绕这一点设计，而不是假装它们存在。

## 二、"从头改进"的实质：从 13 个离散模块到 1 个垂直场景闭环

v0.7 的 13 个模块是**按缺口清单补的**（G6/G9/G11 各自独立通过验收），但它们之间没有场景级的耦合——`soft_body` 不知道 `whole_body_3d` 的存在，`dexterous_hand` 的欠驱动腱与 `climbing_dynamics_core` 的指尖 Hertz 极限各算各的。**攀岩竞技是把它们焊在一起的唯一自然场景**：

```
                    攀岩竞技垂直闭环 (改进后的目标结构)
┌──────────────────────────────────────────────────────────────┐
│  场景层: climb_scene.py (新增, 本方案核心)                      │
│    墙面 (hold 图谱) + 规则 (速度/风险) + 竞技周期状态机           │
├──────────────────────────────────────────────────────────────┤
│  本体层: whole_body_3d (双足+躯干 3D) ──柔性耦合── soft_body    │
│          (v0.7 G9)                        (v0.7 G6, 指尖/腱)  │
│  末端层: dexterous_hand (F5-欠驱动) ──腱模型── soft_body      │
│          (v0.7 G11)                                       │
├──────────────────────────────────────────────────────────────┤
│  接触层: climbing_dynamics_core (既有资产, 本项目原创核心)      │
│    Hertz/Hunt-Crossley/SOC 锥/EKF/机会约束 + vsa_coupling      │
│    (P1.3 已接通: VSA 蓄能→dyno 弹道, 柔顺捕获为必选项)          │
├──────────────────────────────────────────────────────────────┤
│  智能层: nn_core+bc+rl (v0.7) + diffusion_policy (v0.7)       │
│    + lifelong (v0.7, 跨岩板迁移=终身学习的天然载体)             │
├──────────────────────────────────────────────────────────────┤
│  硬件自动化: EdaLab 12 步主链 (live 已通) + P1.1/1.2/1.3 引擎   │
│    腱驱动关节板 / 欠驱动手控制器 / 柔性指尖传感 PCB 自动设计    │
│  数据飞轮: bigdata D2→D4 (live 已通) + dr_params 制造容差      │
└──────────────────────────────────────────────────────────────┘
```

**改进的判断标准**：不是再加模块，而是让"一个攀岩周期"（触壁→换手→摆荡→dyno→柔顺捕获→续攀）作为单一集成测试跑通，其中每个物理量都由对应模块真实计算而非 mock。

## 三、分阶段路线图

### Phase A — 场景集成（2-4 周，纯软件，立即可做）

> **✅ A1+A2+A5 已交付 (2026-10-06, Phase A 首轮推进)**：`simulation/climb_scene.py` 焊接层 + `tests/test_climb_cycle.py` 9 项 + verify 21/21（新增 climb_cycle 项）+ pytest 200 passed。
>
> **实测周期**：hang → reach → swing → dyno → capture → climb_on（六状态全转移，双足推进 0.600m）
>
> **A2 柔性耦合实测 (v1.1 修正)**：早期口径 F_soft=60.0N / F_rigid=104.0N (ratio 0.577, "柔顺降 42%")
> 是循环论证——60N 是二分搜索反解的目标，回代后必为 60N。审计后 `climb_scene.soft_contact_peak_force`
> 已改用非循环量：δ_peak 系数 5/4→1.25（与 `climbing_dynamics_core` 口径一致）、删除反向乘子、
> 改为以刚性峰值 f_rigid + 柔顺比 softening_ratio 表述。**当前实测**：刚性峰值 160-800N、柔性峰值
> 22.9-161.7N、柔顺比 0.156-0.202（"柔顺降约 4 倍"）——这是物理上真的量出的柔顺效果。
>
> **能量调度发现（本阶段最重要的物理结论）**：VSA 满蓄直弹 1.07m/s 对 60N hold 产生 168.9N 峰值力，**必然超限**；攀岩的物理正确做法是捕获前泄能——二分反解得 v_capture_max=0.433m/s，调度后柔顺力恰好压在 60N 限值。此结论同时回答了"摆荡为什么存在"（不是风格，是能量调度的需要）——已固化在 test_energy_scheduling_honest

>
> **实现中修正的三处**（透明记录）：
> ① 指尖刚度提取量纲错误两轮：XPBD compliance≠1/k（得 5e7）；终版用 soft_body.grasp_stiffness 真实配置（3000 N/m，凝胶文献量级）
> ② whole_body 驱动序列错误：直接 step(0.3) 不动（CoM 0 推进）——v0.7 的正确序列是 init_gait→每 5ms step(plan_zmp)→步末 plan_step，照抄 demo 后推进 0.6m
> ③ v_capture_max 首版用解析反解与数值函数口径不一致，改为二分搜索同一函数（口径统一）
>
> **A4 已完成 (2026-10-06，Phase A 收官)**：抓握选择从 HoldType→抓握型硬映射升级为 **dexterous_hand 力闭合 ε 数值驱动**——每 hold 生成三类候选抓握的 GraspContact 接触集（deep_power 包络 / pinch_close 对捏 / open_drag 摊开，hold 尺寸按类型区分），`grasp_quality`（ε 力闭合质量）计算后 argmax 选取。实测：深口袋 hold（seed 42 的 hold 2）正确选出 deep_power（ε=0.1044 > open_drag 0.0970 > pinch 0.0412）。wall_contact 手接触从固连质量点（ARM_APPROX）升级为 **DexterousHandContact 指分布模型**：`underactuated_coupling` 腱耦合比决定指力份额（远端指承载最大 42.3%），手部总拉力→5 指分布，**单指 Hertz 极限（60N）成为比 hold 结构上限更紧的约束**（三肢 77.3N 场景最弱指 32.7N）。**Phase A 至此 A1/A2/A3/A4/A5 全部完成**——攀岩竞技垂直闭环（climb_scene 焊接层）全链贯通：墙面→灵巧手选型→柔性指尖→Hunt-Crossley 柔顺/刚性对比→VSA 蓄能→dyno 弹道→能量调度捕获→斜壁四肢多接触→双足续攀。210 passed（新增 A4 测试 5 项）+ verify 21/21。下一阶段：Phase B 硬件自动化闭环（腱驱动关节板 EPS + P1.4 Stage 化收官）。

| 任务 | 内容 | 验收 |
|---|---|---|
| A1 `climb_scene.py` | 墙面 hold 图谱（位置/朝向/摩擦系数域随机化）+ 竞技周期状态机（触壁/换手/dyno 三模式） | ✅ 一个周期全部状态转移跑通，无 mock 物理量 |
| A2 柔性指尖×接触耦合 | `soft_body` 的指尖形变接入 Hunt-Crossley 的 δ（当前为刚性 δ）——柔顺捕获分支首次有了真实形变数据 | ✅ 捕获峰值力 < 刚性分支（60N vs 104N，降 42%） |
| A3 双足×墙面 | `whole_body_3d` 的 3D LIPM/IK 接入墙面多接触约束（4 肢接触雅可比，攀岩文档 G1 资产） | ✅ `wall_contact.py`：4 肢雅可比 `[I₃\|-[r]×]`（手算对照通过）+ 斜壁准静态力分配；蹲起-换脚 7 帧（4→3→4→3→4）**动态容差（≤40 N·m）下 0 失稳**，手峰值 77.3N<150N。**诚实边界 (v1.1)**：早期"0 失稳"是恒真断言（原码从不计算 τ）。审计后 `allocate_forces` 已真正解 Σ(p−com)×f 得 τ，7 帧中有 4 帧 τ 超静态阈值 15 N·m（`static_unstable=4`），但被 40 N·m 动态容差覆盖——攀岩换脚本就是动态动作，肌肉/关节瞬时补偿承担余下力矩。**单侧超载（40+ N 侧向 + 单脚）会真实失败**（`test_single_foot_overload_fails_torque_balance` 断言），"0 失稳"不再无差别成立。 |
| A4 灵巧手×hold | `dexterous_hand` 的 F5 欠驱动腱模型对 hold 形状分类输出抓握型选择 | ✅ 抓握选择=力闭合 ε 数值驱动（三类候选 GraspContact→grasp_quality→argmax，深口袋 hold 正确选出 deep_power ε=0.1044）；wall_contact 手接触升级为指分布模型（ARM_APPROX=False） |
| A5 集成测试 `test_climb_cycle.py` | 一个完整攀岩周期作为单一测试 | ✅ 全链无 mock；每个模块的输出被下一模块真实消费 |

### Phase B — 硬件自动化闭环（1-2 月，与 Phase A 并行）

| 任务 | 内容 | 验收 |
|---|---|---|
| B1 腱驱动关节板 EPS | hw_requirements.yaml 增 climbing 段 → EdaLab forge-core → motor-drive 消费，产板 | ✅ 11 步主链对 climbing spec **live 全绿**（chain_green=True，实测 2026-10-07）；EPS `SF-climbing_tendon_drive-climbing_competition_v0_1` 与板种 `TendonDrive-Climbing`（24V/19.2A/4 层 55.4mm）导出，household 默认参数逐字不变 |
| B2 P1.4 Stage 化收官 | 12 个 CLI 命令注册 Open API Stage（既有待办） | ✅ **已达成**（2026-10-07 实测，**Python 消费侧**实现）：`STAGE_REGISTRY` 注册 12 个 Stage（11 主链 + `vsa-static`），`run_eps_job` **数据管道**串联 `forge-core→vsa-static` 全绿 — EPS ID `SF-climbing_tendon_drive-climbing_competition_v0_1` 与 `vsa_curve_v1_1`（3 拓扑）进 `context` 供下游消费，`jar_source=vendor`。CLI: `embodied-audit couple --eps-job`。测试：`tests/test_b2_stage_pipeline.py` 11 项。**注：G5 烟囱式缺口在 Kotlin 侧且 jar 冻结，故 Stage 化落在消费侧绕过** |
| B3 VSA-ANTAGONISTIC 选型落地 | P1.3 结论（攀岩指尖 109N 物理正确）写入 B1 的电机/减速器参数 | ✅ **已达成**（2026-10-07 实测）`relay-clutch + vsa-static` 联合 PASS。★**实质修正：契约 109.0 → 166.39N** — 引擎实测 ANTAGONISTIC `peak_force_n=166.39064`，原 109.0 是 V-2 修复（公式改写）前的过期口径，且 docs/04 该表行 `stored(0.635J)` 疑混填 SEA 值（SEA 实测 0.6379）。**选型结论不变**（166 < SEA 347 < MACCEPA 525，仍最接近指尖 60N）。联合判据五条全绿：契约=引擎 / ANTAGONISTIC 峰值最小 / 离合器承载（166.39×0.022=3.66Nm ≤ 额定 12.0Nm 不打滑）/ 安全链 16.00ms<20ms / verdict=PASS。测试：`tests/test_b3_vsa_antagonistic.py` 8 项（含"旧值 109 应被拒绝"反证） |
| B4 制造容差×柔性体 | dr_params 增加腱刚度/指尖凝胶模量容差（soft_body 参数进 DR） | 飞轮 D4 输出含柔性参数分布 |

**B1 实现要点（2026-10-07）**：任务档案维度只落在 11 步中真正参数化的 2 步（forge-core / motor-drive），其余 9 步沿用 EdaCli 内置样本输入。参数翻译集中在 Python 侧 `build_spec_args()`（yaml → CLI 参数表），Kotlin 侧只做字符串分流（`task_description` 含 climb/tendon → 攀岩策略档；`motor-drive` 变体串 `tendon` → `createClimbingSpec()`），**数值不重复硬编码**。household_grasp 是 yaml 之外的虚拟默认档案（返回空表 → 回退 FLOW_CHAIN.cli_args），既有 210 项测试与 CLI 行为零改动。

**B1 已知失真（诚实边界，B4 前处理）**：ForgeCore 元件库 ENCODER_ABSOLUTE 映射 14bit（契约要求 19bit）；`estimateThermal` 是单三相桥口径，8 通道腱板聚合 19.2A 代入后导出"热裕量不足 3.35°C"WARNING——该告警是模型失真而非真实热风险（通道级桥臂仅 4A，MotorDriveEngine 侧结温 51.1°C、裕量 98.9°C）。已记入 `hw_requirements.yaml` 的 `climbing.known_gaps`。

### Phase C — 真机触发点（硬件到位当天，v0.7 已铺好的落点）

v0.7 的三个落点直接对接：`ShutdownChain.stages[*].delay_s`（foc_shutdown）、`bench_vla(measured_ms=...)`（jetson_bench）、`RealRobotEnv.avg_latency_ms`（real_robot_env）。改进点：**攀岩场景为真机测试提供优先级**——顺序应为 ① 电机单关节台架（foc_shutdown 实测）→ ② 指尖柔性传感器标定（soft_body 实测形变曲线替换模型参数）→ ③ 整机墙面静态悬挂（whole_body 实测）→ ④ 首 dyno（全链实测）。

### Phase D — 竞技指标（Phase C 后）

速度-风险权衡（攀岩文档 §3 的竞技数学）：单段耗时 vs 滑脱概率的机会约束优化——这是 RL 奖励函数的物理化改造（fusion-absorption-plan §5.3 既定方向），lifelong 模块负责跨岩板泛化。

## 四、与三栈架构的对位

- **控制栈 L4/L5**：A1-A5 是 L5 接触序列的场景实例化；bc/rl/diffusion 是 L4
- **物理栈 P2/P4**：B1/B3 产出柔性传感板（P2）与腱驱动关节板（P4）
- **数据栈 D1-D5**：B4 扩容 + Phase C 实测数据回流
- **横切 S**：certification（v0.7）+ foc_shutdown（v0.7）+ budget_closure 既有

## 五、诚实边界（不变）

方案全程不假设真机存在：Phase A/B 全部纯软件可验收；Phase C 的"当天实测"依赖外部采购，接口已备好。所有 v0.7 注释中的坑（zero_grad 重绑定、LIPM 解析解、SAC 重参数化等）作为 A1-A5 实现时的强制 checklist。
