# 架构设计文档 — VLA 双路径 / SSH 双通道 / 链路逻辑 / 标定与验证方案 / 真机 E2E 规划

> **日期**: 2026-10-17
> **范围**: plan_v5 Sprint M（M1-M6）与 plan_v6 Sprint N（N1-N6）交付物的架构设计说明，
> 以及两项**模拟→真实**验证方案（e-nose 标定 / V2L 几何演算验证）与 **C 栈接真机**
> 的端到端闭环测试规划。
> **配套代码**: `models/vla_v2a.py` / `os_channel/ssh_admin.py` / `center/*` /
> `perception/chem_bio.py` + `chem_calibration.py` / `hil/e2e_closed_loop.py`
>
> 文档约定：每节标注对应源码文件与 verify 检查项；涉及数值一律标注 provenance
> （measured / assumed / synthetic / 待领域验证），不宣称未实测的精度。

---

## 1. VLA 双路径架构（vision+language→action 与 vision-only→action）

### 1.1 双路径语义

VLA = Vision-Language-Action。用户口径：**纯视觉也能构成判断依据** —— `v→action`
（V-to-action）与 `v+l→action` 是等价的两条决策路径，语言缺省时不应瘫痪。

| 路径 | 输入 | 调用 | 事件 source 标注 |
|---|---|---|---|
| **vla** | vision + lang_ids + state | `VLAPolicy.infer_vla()` | `…\|vla\|…` |
| **v2a** | vision + state（无语言） | `VLAPolicy.infer_v2a()` ≡ `infer_vla(…, None, …)` | `…\|v2a\|…` |

**等价性契约**（`tests/test_vla_v2a.py` + verify `vla_v2a`）：
- `infer_v2a(v, s) == infer_vla(v, None, s)` 数值恒等（同一条计算路径）；
- `modality="v2a"` 时传入语言**必须报错**（否则"纯视觉"契约名存实亡）；
- 语言通道**生效性**：同权重下有/无语言输出必须不同（防语言被静默忽略）；
- `VLAConfig.modality="v2a"` 时 `min_block_size` 不含语言 token（视觉 token 不被挤掉）。

### 1.2 实现分层

```
models/vla_config.py   modality="vla"/"v2a"  (配置域, min_block_size 分派)
models/vla_v2a.py      VLAPolicy             (numpy 推理封装, 零 torch 依赖)
models/vla_model.py    VLAModel (torch)      (训练侧前向, HAS_TORCH 守卫)
        └─ torch_shape_contract()  两边动作形状契约交叉核验 (10,12)
```

**诚实边界**：`VLAPolicy` head 权重为 seeded synthetic（确定性随机投影），
**不是**训练产物 —— 本层守护的是路径等价性与形状契约，不是策略性能。
真实决策精度由 `vla_trainer.py` 检查点提供（P1 后接实机数据）。

---

## 2. SSH 双通道架构（RUNTIME 协作规划 vs MAINTENANCE 停机维护）

### 2.1 双通道语义（核心区分）

> 用户口径：SSH **不用于遥控**。遥控是实时 runtime 与机器人协作、互相完成
> 下一步规划动作；SSH 是让机器人**暂时停下来**修历史 bug / 新增功能程序。

| 通道 | 语义 | 运动指令 | 典型操作 |
|---|---|---|---|
| **RUNTIME** | 实时协作规划（人机互配完成下一步动作，20Hz 控制环内） | ✅ 允许 | 协作规划、遥测回看 |
| **MAINTENANCE** | SSH 停机维护 | ❌ **禁止** | 拉日志、推程序、出厂烧录 |

### 2.2 模式状态机

```
RUNTIME → PAUSED → MAINTENANCE → PROVISIONED → RUNTIME
              ↘__________↺  (PAUSED 可回 RUNTIME)
```

- `ModeController.ALLOWED_TRANSITIONS` 是唯一真源（`contracts/os_channel.yaml` 同步声明）；
- **铁律**：`can_send_motion()` 仅 RUNTIME 为 True（`tests/test_os_ssh.py` 全转换表钉死）；
- `provision_factory_program()` 烧录"认知本体 + 启动清单"（`/opt/embodied/cognition`），
  烧录后 PROVISIONED → RUNTIME 才恢复运动。

### 2.3 执行器注入

```
SshCommandBuilder   纯函数: ssh / scp / 出厂脚本  (零执行)
CommandExecutor     协议: run(argv) -> ExecResult
  ├─ DryRunExecutor     默认: 只记录命令序列 (测试/CI/预演)
  └─ SubprocessExecutor 真跑系统 ssh/scp (真机; 零 paramiko 依赖)
OsChannelAdmin      工作流: enter_maintenance / pull_debug_logs /
                    push_program / provision_factory_program / return_to_runtime
```

**诚实边界**：host/user 为 assumed 示例值（真机部署待实测）；出厂脚本内容
provenance=待领域验证（systemd 路径为模板）。

---

## 3. 链路逻辑：信号分频 → 触觉融合 → 空间智能决策

### 3.1 端到端数据流（CenterBrain 单步）

```
组合信号 (视频流+微波+…)
  │ C1 信号处理: Channelizer.demux() FFT 频带掩蔽解复用 → 各信道子信号
  │            (串扰 <-20dB, verify sensor_fusion)
  ▼
C2 感知融合
  ├─ 触觉-视觉 MultiRateEKF (触觉 100Hz / 视觉 20Hz, μ 在线估计)
  ├─ SLAM EKFSLAM2D (自身+对象三维坐标, 时间戳单调)
  └─ 化学生化 ChemTactileFusion (e-nose/pH/湿度 × 触觉 → 材质/表面状态)
  │
  │  POSE_UPDATE 发布 ──(B8 断链修复: 唯一发布口)──→ 认知中枢 latest_pose()
  ▼
C3 记忆: EventRecord 四要素 (时间/地点/人物/事情) → EpisodicMemory
  ▼
V2L 空间论证: SpatialReasoner.argue()
  ├─ 辅助线 hypotheses (各视角主方向 + 目标方位)
  ├─ 演算 derivation_steps (逐视角 cos 夹角)
  ├─ 结论 conclusion + 决策向量 decision
  └─ 独立复算验证 GeometryVerifier (docs/19 §7)
  ▼
C4 决策: VLAPolicy 双路径 (有语言 vla / 无语言 v2a)
  ▼
act_fn 动作执行 (回调注入: 仿真 / 真机)
  ▼
事件落库 source="center|<path>|v2l"  →  plan() 重算 → 下一步
```

### 3.2 各环节守护

| 环节 | 判据 | 守护 |
|---|---|---|
| 分频 | 串扰 <-20dB（+不分频负对照 >-20dB，判据有区分力） | `check_sensor_fusion` |
| SLAM | 50 步位姿误差 <0.3m（+3σ 失配负对照）；时间戳单调 | `check_slam` |
| POSE_UPDATE | hub.latest_pose() 收到（B8 断链修复） | `check_center_brain` |
| 化学融合 | dry/wet/corrosive 三场景判定 | `check_chem_bio` |
| V2L 论证 | 输入变则论证变（防恒定假论证） | `check_spatial_reasoning` |
| V2L 复算 | 独立复算 + 真值场景 + 变异负对照 | `check_geometry_verification` |
| VLA 双路径 | 等价性 + 拒语言 + 形状契约 | `check_vla_v2a` |
| 全链 | 5 步阶段推进 + 四要素事件递增 | `check_e2e_closed_loop` |

---

## 4. 化学生化传感：真实 e-nose 标定方案（d1）

> 目标：把 `chem_bio.py` 的 synthetic 响应，升级为**可接真实 MOX 气体阵列**的
> 标定协议。代码 = `perception/chem_calibration.py`，verify = `chem_calibration`。

### 4.1 标定协议（六步）

| 步 | 操作 | 产出 |
|---|---|---|
| 1 | 零气（clean air）基线 60s | 每通道 baseline |
| 2 | 标气多点：目标气体 0/10/50/100 ppm 各 30s + 干扰气体单独注入 | 响应-浓度样本 |
| 3 | 全局线性解 `responses = A·conc + b`（lstsq，**超定**） | 交叉敏感矩阵 A |
| 4 | 每通道主气体标定曲线 slope/intercept/R²/RMSE | `ChannelCurve` |
| 5 | 温湿度补偿 + 零气间歇重测漂移监控 | `DriftMonitor` |
| 6 | 验收门禁：RMSE / R² / 补偿后串扰泄漏 三项全过 | `CalibrationReport` |

### 4.2 关键实现与陷阱（已修）

- **欠定/恰定系统必须拒绝**：点数 ≤ 参数数（气体数+截距）时 lstsq 精确插值，
  残差恒 0、R²=1.0，门禁**恒过 = 自证当验收**。`fit()` 现强制
  `n_points > n_params`（`tests/test_chem_calibration.py::test_underdetermined_rejected`）。
- **交叉补偿**：`compensate() = pinv(A)·(response − b)`，单气体注入后非目标气体
  估计应 →0（测试断言 <15ppm）。
- **漂移门禁**：零气重测归一化偏移 >20% → `needs_recalibration()`（assumed 阈值）。

### 4.3 真实硬件操作清单（P1 后）

1. 气瓶/质量流量控制器按 §4.1 浓度点配气（**待领域验证**：气体种类按场景定）；
2. 采集响应 → 生成 `CalibrationPoint` 列表喂 `fit()`；
3. 门禁报告入 `docs/13-domain-parameters-checklist.md`（provenance 升级为 measured）；
4. 漂移监控并入 D2 遥测（每日零气自检）。

---

## 5. V2L 空间智能：几何演算验证方案（d2）

> 目标：把"辅助线→演算→结论"的**数学正确性**从自证升级为独立验证。
> 代码 = `center/geometry_verification.py`，verify = `geometry_verification`。

### 5.1 验证三层

| 层 | 方法 | 判据 |
|---|---|---|
| **独立复算** | 从 `derivation_steps` 提取报告 cos，用独立公式重算逐项比对 | 误差 ≤ TOL |
| **真值场景** | 解析可解的几何场景（对齐 cos=1 / 正交 cos=0 / 反向 cos=-1） | 结论关键字正确 |
| **变异负对照** | `mutate()` 篡改单步 cos → 验证器必须报 mismatch | 检出率 100% |

### 5.2 关键陷阱（已修）

- **容差必须锚定报告精度**：`derivation_steps` 用 `:.3f` 写出 cos（3 位小数），
  复算容差取半最小数位 **5e-4**。曾取 1e-6 → 一切合法论证"复算失败"。
  变异负对照 delta=0.25 >> TOL，检出不受影响。
- **真值场景分段口径**：`SpatialReasoner` 把 12 维特征均分 3 段（4 元素/段），
  每段取前 3 维为方向 —— 场景数据须 `tile([x,y,z,0], 3)`，否则段内错位场景失真。
- **结论一致性判据**：cos≥阈值 → 结论含"推进/对齐"；否则含"转向"。

---

## 6. C 栈接真机：端到端闭环测试规划（d3）

> 目标：给出"中枢决策层 → 真实机器人硬件"的**可执行测试骨架**。
> 代码 = `hil/e2e_closed_loop.py`，verify = `e2e_closed_loop`，与
> [15-hil-test-protocol.md](15-hil-test-protocol.md) 的 48h HIL 协议对齐。

### 6.1 阶段门（顺序推进，禁跳步）

| 阶段 | 内容 | 判据 |
|---|---|---|
| **S1 通电自检** | 电源/编码器/CAN alive，运动禁用 | 遥测通道 >0，无动作下发 |
| **S2 感知回环** | 遥测→CenterBrain→POSE_UPDATE 抵达 hub | `hub.latest_pose()` 非 None |
| **S3 决策回环** | VLA 双路径决策产出动作（不下发） | 动作形状 (action_horizon, action_dim) |
| **S4 运动闭环** | 单步动作下发 + 时延核对 | worst 时延 ≤ 50ms（契约预算） |
| **S5 任务闭环** | 目标→规划→四要素事件落库 | 事件数 +1 |

- **准入门禁**：`_require_phase()` 拒绝跳步/回退（`test_skip_phase_rejected`）；
- **时延负对照**：80ms 通道必须 FAIL 50ms 预算（`check_e2e_closed_loop` 断言）；
- **ESTOP**：触发后后续阶段全部拒绝，通道拒收动作。

### 6.2 硬件接入路径

```
MockHardwareChannel (synthetic 时延, 当前)
   ↓ 换装
SocketCANTransport (simulation/real_robot_env.py, 已有) 包装为 HardwareChannel
   ↓ 注入
E2EClosedLoopHarness(channel=..., center_brain=CenterBrain(...))
   ↓ 逐阶段
docs/15 的 48h HIL 协议 (P1 硬件到货后)
```

### 6.3 与 OS SSH 通道的配合

真机测试全程遵守 §2 状态机：S1-S5 均在 **RUNTIME** 态；出现 bug 需修时
→ `OsChannelAdmin.enter_maintenance()`（**运动自动禁用**）→ 拉日志/推程序
→ `provision_factory_program()` → `return_to_runtime()` 再续跑阶段门。

### 6.4 里程碑（P1 硬件到货后）

| 里程碑 | 内容 | 准出 |
|---|---|---|
| E2E-M1 | 单关节台架 + S1-S2 | 遥测/POSE_UPDATE 通 |
| E2E-M2 | S3-S4 决策→运动 + 时延 | 50ms 预算实测达标 |
| E2E-M3 | S5 任务闭环 + e-nose 实标定 | 四要素事件 + 标定门禁过 |
| E2E-M4 | 全链回归 + OS 维护流程演练 | `run_all()` 真机全 PASS |

---

## 7. provenance 总表（诚实边界）

| 项 | provenance | 说明 |
|---|---|---|
| e-nose 标定数据产生器 | synthetic | 协议/门禁可执行验证用；真实气体种类/浓度点=待领域验证 |
| 标定门禁阈值 (RMSE/R²/泄漏) | assumed | 待硬件工程师按真实数据调整 |
| 漂移阈值 20% | assumed | 待实测老化曲线 |
| V2L 复算容差 5e-4 | 锚定报告精度 | `:.3f` 报告 → 半最小数位 |
| V2L 真值场景 | 解析真值 | 非学习式，可独立验证 |
| E2E mock 时延 | synthetic | 真机时延待 P1 实测 |
| 50ms 端到端预算 | 契约 | `contracts/hw_requirements.yaml compute` |
| OS host/user/出厂脚本 | assumed / 待领域验证 | 示例值/模板 |
| VLA head 权重 | synthetic | 确定性随机投影，非训练产物 |

---

## 8. 测试与验收索引

| 验证项 | 测试文件 | verify check |
|---|---|---|
| VLA 双路径 | `tests/test_vla_v2a.py` | `vla_v2a` |
| SSH 双通道 | `tests/test_os_ssh.py` | `os_channel` |
| 信号分频 | `tests/test_sensor_fusion.py` | `sensor_fusion` |
| 触觉/化学融合 | `tests/test_chem_bio.py` | `chem_bio` |
| 空间论证 | `tests/test_spatial_reasoning.py` | `spatial_reasoning` |
| 中枢编排/断链 | `tests/test_center_brain.py` | `center_brain` |
| **e-nose 标定** | `tests/test_chem_calibration.py` | `chem_calibration` |
| **V2L 几何验证** | `tests/test_geometry_verification.py` | `geometry_verification` |
| **真机 E2E 支架** | `tests/test_e2e_closed_loop.py` | `e2e_closed_loop` |

一键验收：`python -m embodied_ai.cli verify --no-pytest`（45 项，退出码即结论）。

---

## 9. 触觉传感与化学生化信号仿真实现细节（补充）

> 用户要求：补充触觉传感与化学生化信号仿真的**具体实现细节**。本节逐层展开
> §3.1 链路中 "C2 感知融合" 的两个输入分支，含全部判据参数与 provenance。

### 9.1 触觉传感链路（`perception/tactile_closed_loop.py`）

**数据流**（20Hz 控制环内，dt=0.05s）：

```
tactile_readings (切向力 f_t, N, 100Hz 触觉流)
  → MultiRateEKF.predict()            过程: stick/slip 模型 (f_t, μ, v_t 三态)
  → MultiRateEKF.update_tactile(z_f)  触觉观测更新; 滑锥边界时 μ 也可观
  → SlipDetector.check(f_t_est, μ̂, f_n_cmd)
  → 滑脱? ──是──> chance_constrained_grip(f_t_req, μ̂, σ_μ=0.05, δ)
  │                    (InfeasibleGripError → f_n_max 保底)
  └─否──> 节能基线 f_n_baseline
  → GripForcePID.compute(f_n_target, dt) → f_n_cmd
  → f_n_cmd 回写 ekf.f_n (闭环反馈) → 记录 TactileStep
```

**判据与参数**（assumed，1050-episode 训练可调，见 plan_v5 L4尾）：

| 参数 | 值 | 作用 | provenance |
|---|---|---|---|
| `mu_prior` / `mu_prior_std` | 0.8 / 0.1 | EKF μ 初值 | assumed |
| `f_n_init` | 30 N | 初始握力 | assumed |
| `slip_threshold` | **0.7** | `slip_ratio = f_t_est/(μ̂·f_n_cmd) ≥ 0.7` → incipient slip | assumed |
| `kp` / `ki` | 1.0 / 0.1 | GripForcePI 增益 | assumed（L4尾 训练目标） |
| `f_n_max` | 150 N | 握力上限（GelSight 软指尖 Hertz 极限 ~60N@5mm → 骨架并联） | 文献 |
| `delta` | 0.05 | 机会约束失效概率（`statistics.NormalDist().inv_cdf` 取 z） | 契约 `safety_budget.yaml` |
| **收敛判据** | 末 5 步无 slip **且** `n_incipient > 0` | 曾滑脱→被抑制才算 converged（无滑脱场景不算成功） | 诚实判据 |

**硬件前端**（`adapters/afe_ind.py`，verify `afe_ind_frontend`）：
AFE-IND 触觉模拟前端 — 增益 1046.83×（60.4dB）、带宽 10.6kHz、噪声
3.79e-06 V/√Hz、PCB 65 元件；对应 B1 BOM 的 8 路 AD8422 腱应变片仪表放大
（`U_LC0~U_LC7`）。GelSight 视觉触觉 921KB/帧 → GPU 预处理（architecture-overview §7 约束 4）。

**合成数据源**（provenance=synthetic，真机替换为传感器读数）：
- `synthesize_slip_scenario()` — 滑脱发生→抑制的标准场景（60 步 @20Hz）
- `episode_tactile_stream()` — D4 1050 episodes 回放触觉流

### 9.2 化学生化信号仿真（`perception/chem_bio.py`）

**传感模型**（`ChemBioSensor.read()`，provenance=synthetic）：

| 成分 | 维度/范围 | 合成响应模型 |
|---|---|---|
| `enose` MOX 阵列 | **8 通道** | base=0.2 + 材质偏置 + N(0, 0.05) 噪声 |
| 材质偏置 | — | metal→ch0/ch3 +0.6；plastic→ch1/ch5 +0.5；organic→ch2/ch4/ch6 +0.7 |
| 湿度偏置 | — | wet→ch7 +0.8 |
| `humidity` | [0, 1]（`__post_init__` 强校验） | wet→0.9；dry→0.3+U(0, 0.2) |
| `ph` | 0-14 | 场景给定（corrosive 场景 2.5） |
| `temperature_c` | ℃ | 22±1 |

**融合判定链**（`ChemTactileFusion.fuse()`，每条判定写入 `evidence` 证据链，可解释）：

```
① humidity > WET_THRESHOLD(0.8)        → surface = "wet"
② 否则 ph < CORROSIVE_PH(4.0)          → surface = "corrosive"
③ 否则                                  → surface = "dry"
④ tactile_std < SLIP_TACTILE_STD(0.3)   → surface = "slippery" (覆盖, 低波动=低摩擦)
⑤ enose argmax → material: 0/3=metal, 1/5=plastic, 2/4/6=organic, 其它=unknown
confidence = min(1, 0.5 + 0.1×证据条数)  (规则强度, 非学习概率)
```

阈值三项均 provenance=assumed（待领域验证——由真实 e-nose/湿度/pH 探头数据回归）。

**与真实 e-nose 的衔接**：合成响应 → [`chem_calibration.py`](../src/embodied_ai/perception/chem_calibration.py)
多点标定（§4 协议）把 provenance 升级为 measured：
`responses = A·conc + b` 交叉敏感解耦（`pinv(A)`）、零气漂移监控
（>20% → `needs_recalibration`）、RMSE/R²/串扰泄漏三门禁。

### 9.3 融合进链路与数据回流

| 接口 | 位置 | 说明 |
|---|---|---|
| `CenterBrain.step(..., chem_vec=, tactile_feats=)` | `center/center_brain.py` | `ChemTactileFusion.fuse()` → `state["material"]` 进决策上下文 |
| GUI `/api/tactile` | `gui/server.py` | EKF μ̂ / f_t / slip_risk 实时反馈（20Hz 视觉 + 100Hz 触觉） |
| GUI `/api/chem` | `gui/server.py` | 化学向量 + 材质估计 + provenance 标注 |
| D4 回流 | `models/replay_dataset.py` | `MaterialEstimate` + `TactileStep` 序列 → 训练数据（自我进化） |

### 9.4 参数 provenance 汇总（本节）

| 参数 | 值 | provenance |
|---|---|---|
| EKF 过程/观测噪声（R_tac=0.25, R_vis_mu=0.02, Q） | 见 `multirate_ekf.py` | assumed |
| slip_threshold / kp / ki / f_n_max | 0.7 / 1.0 / 0.1 / 150N | assumed（L4尾 训练） |
| 机会约束 δ / σ_μ | 0.05 / 0.05 | 契约 safety_budget.yaml |
| WET_THRESHOLD / CORROSIVE_PH / SLIP_TACTILE_STD | 0.8 / 4.0 / 0.3 | assumed（待真实探头回归） |
| e-nose 8 通道合成响应 | base 0.2 + 偏置 | synthetic（§4 标定后升 measured） |
| AFE-IND 增益/带宽/噪声 | 60.4dB / 10.6kHz / 3.79e-06 V/√Hz | measured（B1 板级仿真） |
