# 代码-硬件联调指南 (Code-Hardware Integration Guide) — 经验与落地

> **日期**: 2026-10-08
> **定位**: 回答四个工程实际问题: ① 制造工艺与代码平台**会不会出问题** ② 联调**具体步骤** ③ **需不需要经验** ④ **没经验如何渐进积累**直到完成
> **前提**: 已读 [docs/16-manufacturing-guide.md](16-manufacturing-guide.md) (5 层制造堆叠) 和 [docs/15-hil-test-protocol.md](15-hil-test-protocol.md) (48h HIL)

---

## 〇、一句话回答

**会出问题吗? 会** — 但问题种类是**有限且可枚举**的 (本文 §2 列出 7 类典型 gap)，都能通过**参数注入 + 双模仿真/实机切换 + 渐进加载**的系统工程方法把 gap 收敛到可接受范围。

**需要经验吗? 需要** — 但**需要的不是"天赋"，而是"可记录的经验"** (本文 §4 把经验拆成 10 个独立可练的子技能)。没经验者按 §5 的 **90 天渐进路径**走，每周投入 10 小时，**可以在 3 个月内达到"独立完成单关节台架联调"的水平**，6 个月达到"全身协调调试"。

---

## 一、代码平台和硬件到底会不会"对不上"?

### 1.1 七类典型 Gap (每类都有代码侧的"接收器")

| # | Gap 类型 | 典型表现 | 硬件侧根因 | 代码侧接收器 (已有) |
|---|---|---|---|---|
| **G1** | **参数漂移** | 代码 `Rs=50 mΩ`, 实测 65 mΩ | 电机个体差异、温升、制造公差 | `MotorSpec.provenance` + `identify_motor_params` CLI |
| **G2** | **模型不匹配** | 仿真 dyno 成功率 80%, 实测 40% | 接触模型简化 (HC 线性 vs 实际非线性) | `assumed_*` 标签 + `provisional` 降级机制 |
| **G3** | **时序抖动** | 仿真 200 Hz 精确, 实际 ±100 µs | CAN-FD 报文排队、RTOS 调度延迟 | `real_robot_env.avg_latency_ms` 注入点 + `jitter` 统计 |
| **G4** | **传感器误差** | 编码器读数 0.1° 温漂 | 磁编码器温度系数、机械偏心 | `sensor_plausibility` 检查 + `DataQualityChecker` |
| **G5** | **制造缺陷** | PCB 虚焊导致间歇失效 | 焊接工艺、连接器可靠性 | `FctResult` / `IctResult` 数据结构 + 飞轮回流 |
| **G6** | **边界条件** | 电源跌落导致 MCU 复位 | 电池内阻、线缆压降、浪涌 | `fault_events` 字段 + `watchdog` 机制 |
| **G7** | **软件-固件版本漂移** | Python FOC 参数与固件编译时不同 | 固件烧录版本未同步 | `FOCConfig.kp_d/ki_d` 双轨对齐 (`foc_board_align.py`) |

### 1.2 Gap 会不会"叠加放大"?

**会**。这是控制系统的典型问题：小参数误差 × 闭环增益 → 大响应偏差。

**例**: Rs 误差 30% → 电流环 ki 偏差 30% → 相位裕度从 60° 降到 35° → 轻微振荡 → 编码器读数噪声放大 → EKF σ_μ 高估 → 握力过补偿 → 能耗上升 20% → 续航从 15 min 降到 12 min。

**解法**: **每层 Gap 单独收敛，不叠加**。这就是为什么 `docs/16` 强调"绝不跨层"——L0 的电流环参数锁定后，L1 的速度环才能整定。

---

## 二、联调的 5 步循环 (Iterative Convergence Loop)

### 2.1 核心循环: **模型 → 预言 → 实测 → 对比 → 修正**

```
┌────────────────────────────────────────────┐
│                                            │
│  ① 代码平台: 用当前参数"预言"硬件行为     │
│     (e.g. 预测电流环阶跃响应: 上升时间 5ms)│
│         ↓                                  │
│  ② 硬件实测: 真实跑一遍, 采数据            │
│     (e.g. 实测上升时间 8ms)                │
│         ↓                                  │
│  ③ 对比: 算 gap                            │
│     (e.g. gap = 3ms = 60% 误差)            │
│         ↓                                  │
│  ④ 归因: 参数漂移? 模型缺陷? 传感器误差?   │
│     (Rs 假设 50 mΩ vs 实测 65 mΩ)          │
│         ↓                                  │
│  ⑤ 修正: 更新参数到代码 or 修硬件          │
│     (把 measured_Rs=65 注入 MotorSpec)     │
│         ↓                                  │
│  └───── 回到 ①, 直到 gap < 阈值 ──────────┘
```

**收敛阈值** (工业惯例):
- 电流环: gap < 5%
- 速度环: gap < 10%
- 位置/阻抗环: gap < 15%
- 任务级 (dyno 成功率): gap < 20%

### 2.2 双模切换 (Dual-Mode Middleware)

**核心**: **代码不修改**，通过**配置文件**切换仿真/实机。

```python
# src/embodied_ai/simulation/make_env.py 已支持:
env = make_env("climb_wall", backend="mujoco")    # 仿真
env = make_env("climb_wall", backend="reference")  # 纯 Python 快速仿真
env = make_env("climb_wall", backend="real")       # 真机 (Hardware-in-the-Loop)

# 联调时, 代码完全一样, 只是后端不同。
# 这保证了 "仿真调好的参数 → 真机直接用" 的可行性。
```

**建议扩展** (Phase P 前新增):

```python
# src/embodied_ai/hal.py (Hardware Abstraction Layer)
class Hal:
    """硬件抽象层 — 让代码统一视角, 底层切换仿真/实机."""
    def read_encoder(self, joint: str) -> float:
        if self.mode == "sim":
            return self.sim_state.qpos[self.joint_idx[joint]]
        else:
            return self.can_bus.read_pdo(joint, "actual_pos_deg")

    def write_current(self, joint: str, iq_ref: float):
        if self.mode == "sim":
            self.sim_state.ctrl[self.joint_idx[joint]] = iq_ref
        else:
            self.can_bus.send_pdo(joint, "target_iq", iq_ref)
```

### 2.3 渐进加载 (Gradual Loading)

联调时不一步到位，按 **6 档负载** 逐步加载:

| 档 | 负载状态 | 验证内容 | 通过判据 |
|---|---|---|---|
| 0% | 空载 (无腱索) | 电机自由旋转, 电流环稳定 | 阶跃响应符合设计 |
| 10% | 空载 + 腱索预紧 2N | 腱张力零点校准, 编码器读数稳定 | 零点漂移 <0.1°/h |
| 25% | 腱索预紧 + 轻负载 10N | 力矩-电流线性度 | R² > 0.99 |
| 50% | 50% 额定 (15N) | 温升 <10°C (10min), 无振荡 | MOSFET Tj < 80°C |
| 80% | 80% 额定 (24N) | 长时间稳定性 (30min) | 无故障 |
| 100% | 额定 (30N) | 峰值性能 | 达到 datasheet |

**渐进加载的意义**: 每档的 gap 单独归因，避免多变量耦合。

---

## 三、具体联调步骤 (可执行清单)

### 3.1 Step-by-Step: 从开箱到全身联调 (典型 3-4 周)

**Day 1-3: 开箱 + 静态检查**
- [ ] 按 [docs/14 §四](14-hardware-procurement-checklist.md) 到货验收
- [ ] PCB 目检 + 上电电源轨检查 (5V/3.3V/24V)
- [ ] 电机线圈阻值测量 (三相平衡 <5% 差异)
- [ ] 编码器 sin/cos 静态信号 (示波器)

**Day 4-7: 电机参数辨识 (G1 Gap 收敛)**
- [ ] `identify_motor_params --method dc_step` → 测 Rs
- [ ] `identify_motor_params --method lcr` → 测 Ld/Lq (1 kHz LCR 表或软件法)
- [ ] `identify_motor_params --method bemf` → 测 Ke/ψ_m
- [ ] `verify_motor_spec` 三方校验 → 通过
- [ ] `inject_hil_measurements --stage motor_params` → 注入 `contracts/`

**Day 8-12: 电流环整定 (L0)**
- [ ] 按 `docs/16 §2.1` 公式算 kp/ki
- [ ] 注入阶跃, 示波器抓响应
- [ ] 若超调 >5%: 降 kp 10% 重测
- [ ] 若稳态误差 >2%: 升 ki 10% 重测
- [ ] 收敛后: 注入 `FOCConfig` (via `foc_set_gains`)

**Day 13-15: 编码器 + CAN (G3 Gap 收敛)**
- [ ] 极对数识别 (HFI 或旋转法)
- [ ] 编码器零位对齐
- [ ] CAN-FD 双向通信 (1 Mbps + 5 Mbps FD)
- [ ] sync 帧时序抓取 (示波器)
- [ ] `avg_latency_ms` 实测注入

**Day 16-20: 单关节闭环 (L0 放行)**
- [ ] 速度环整定 (阶跃+频域)
- [ ] 位置环整定
- [ ] 24h 连续运行 (L0 放行判据)

**Day 21-30: 4 关节单臂 (L1 放行)**
- [ ] 多节点 CAN 同步
- [ ] DH 参数验证 (正逆往返)
- [ ] 轨迹跟踪误差 <5 mm
- [ ] 腱张力传感器标定

**Day 31-45: 双腿平衡 (L2 放行)**
- [ ] IMU 零偏校准
- [ ] ZMP 轨迹测量
- [ ] 30s 站立测试
- [ ] 20N 抗推测试

**Day 46-60: 全身协调 (L3 放行)**
- [ ] QP 层次化权重整定
- [ ] 8 关节 CAN-FD 同步
- [ ] squat / side-step 动作
- [ ] 自碰撞避免验证

**Day 61-90: 攀岩整机 (L4 放行)**
- [ ] 抓握力调度整定 (`chance_constrained_grip`)
- [ ] 静态悬挂 30s
- [ ] 4→3→4 换脚
- [ ] 单次 dyno
- [ ] 5-hold 小路线完攀

### 3.2 每一步的"代码平台动作"

**联调时, 代码平台不是被动接受数据, 而是主动驱动**:

```bash
# 1. 联调前: 代码平台"预言"
python -m embodied_ai.cli predict_step_response --joint j1 --target 2A
# 输出: 预期上升时间 5.2 ms, 超调 3.1%, 稳态误差 0.8%

# 2. 硬件实测: 示波器/示波记录仪采集
# (人工操作, 或脚本自动触发)

# 3. 数据导入: 实测 CSV
python -m embodied_ai.cli hil_import --csv scope_j1_step.csv --expected 5.2ms

# 4. Gap 分析
python -m embodied_ai.cli hil_gap_analyze --expected 5.2ms --actual 8.1ms
# 输出: gap 56%, 归因: Rs 偏差 (假设 50 vs 实测 68 mΩ), 建议更新

# 5. 参数注入
python scripts/inject_hil_measurements.py --param Rs --value 68e-3 --unit ohm
# 自动更新 contracts/hw_requirements.yaml + provenance
```

---

## 四、经验需要吗? 需要哪 10 项?

**结论**: 需要，但**每一项都是可练的独立技能**，不需要"机械工程硕士"背景。

### 4.1 十项独立子技能

| # | 技能 | 无经验起点 | 合格判据 | 典型练习时长 |
|---|---|---|---|---|
| **S1** | **焊接** | 不会用烙铁 | 0603 贴片连续 100 个无桥接, QFN48 无桥接 | 2-4 周 (每天 1h) |
| **S2** | **万用表** | 认识 V/A/Ω | 能测通断、电阻、电压、电流；能判断虚焊 | 1 周 |
| **S3** | **示波器** | 只见过没用过 | 能设置触发、测量上升时间/频率/相位差、抓异常波形 | 2 周 |
| **S4** | **CAN 分析** | 不知 CAN 为何 | 能用 candump/CANable 发送/接收帧, 解析 DBC | 1 周 |
| **S5** | **机械装配** | 拧过螺丝 | 能按公差装配 (H7/g6), 理解预紧力和防松 | 2 周 |
| **S6** | **Python 脚本** | 能跑 python | 能读懂项目 CLI, 能写简单的数据导入脚本 | 2 周 (如已有) |
| **S7** | **PID 整定** | 听说过 | 能用阶跃响应法独立整定 PI, 理解超调/稳态误差 | 3 周 |
| **S8** | **EKF 基础** | 听说过 Kalman | 能读懂 MultiRateEKF 代码, 理解 Q/R 的含义 | 2 周 |
| **S9** | **故障诊断** | 见到故障会慌 | 按附录 A FAQ 逐步排查, 能定位到元件级 | 4 周 (跟做) |
| **S10** | **系统思维** | 只会局部 | 能从"末端误差"反推"哪层参数出问题" | 持续积累 |

### 4.2 每项技能的"最小练习" (不买机器人也能练)

- **S1 焊接**: 淘宝买 **练习板** (¥30/片, 100 个贴片), 1 周出师
- **S2 万用表**: 万用表 ¥100 + 电池 + 电阻 + LED, 1 周
- **S3 示波器**: 入门示波器 (Rigol DHO802, ¥2,000) + 函数发生器 (¥500), 2 周
- **S4 CAN**: CANable Pro (¥250) + USB-CAN 分析仪 (¥150), 1 周
- **S5 机械**: 乐高 MindStorm / 3D 打印零件 + 螺丝套件, 2 周
- **S6 Python**: 本项目代码仓库, 读 `hardware/` 目录, 2 周
- **S7 PID**: 用 Python 仿真一个弹簧-质量-阻尼系统 (100 行代码), 3 周
- **S8 EKF**: 项目 `perception/multirate_ekf.py`, 逐行读懂, 2 周
- **S9 诊断**: 找一块有问题的开发板 (闲鱼 ¥50), 排查到元件级
- **S10 系统思维**: 跟着本文档联调 1 次即入门

---

## 五、没经验如何有序渐进积累 (90 天路径)

### 5.1 分三阶段: **模仿 → 协作 → 独立**

#### 阶段 1: 模仿期 (Day 1-30) — "打下手 + 做小件"

**目标**: 熟悉工具、能做**非关键装配**、能看懂代码
**投入**: 每天 1-2 小时
**预算**: ~¥500 (工具 + 练习板)

| 周 | 内容 | 产出 | 合格判据 |
|---|---|---|---|
| Week 1 | S1 焊接 + S2 万用表 | 焊完 2 块练习板 | 100 个贴片无桥接 |
| Week 2 | S3 示波器 + S4 CAN | 抓一个方波信号, 发一个 CAN 帧 | 能独立设置触发 |
| Week 3 | S6 Python (项目代码) | 跑通 `verify` + `pytest` + `selfcheck` | 理解 adapters/bigdata.py |
| Week 4 | S5 机械 + S1 复习 | 装配一个小机械件 (3D 打印夹具) | 理解公差与预紧 |

#### 阶段 2: 协作期 (Day 31-60) — "有指导地做关键件"

**目标**: 能做**关键装配**、能做**参数辨识**、能做**初步调试**
**投入**: 每天 2-3 小时
**预算**: ~¥3,000 (入门示波器 + CAN 工具 + 电机套件)

| 周 | 内容 | 产出 | 合格判据 |
|---|---|---|---|
| Week 5-6 | B1 板焊接 (关键) | 1 块 B1 板焊完, 通过 ICT | 100% 元件在位, 无短路 |
| Week 7 | 电机参数辨识 | 实测 Rs/Ld/Lq/ψ + 注入 | `verify_motor_spec` 三方一致 |
| Week 8 | 电流环整定 | 单关节台架 FOC 稳定 | 阶跃响应符合设计 ±5% |
| Week 9 | CAN + 编码器 | CAN-FD 双向通信, 编码器读数稳定 | `avg_latency_ms` < 5 ms |

#### 阶段 3: 独立期 (Day 61-90) — "独立完成联调闭环"

**目标**: 能**独立诊断故障**、能**独立整定 PID**、能**指导他人**
**投入**: 每天 2-3 小时
**预算**: ~¥5,000 (完整单关节套件)

| 周 | 内容 | 产出 | 合格判据 |
|---|---|---|---|
| Week 10-11 | 完整单关节联调 (L0) | 24h 连续运行无故障 | `docs/16 §八` L0 放行 |
| Week 12-13 | 故障注入与诊断 | 人为设置 10 类故障, 全部定位 | 附录 A FAQ 全部实践过 |
| Week | 编写调试文档 | 1 份个人调试日志 | 能教别人做 L0 |

**90 天后应达到**: 独立完成单关节台架从开箱到 24h 连续稳定运行，能诊断 10 类常见故障，能独立整定电流环/速度环/位置环，能读懂本项目全部 `hardware/` 代码。

---

## 六、线下生产与代码平台的**有机结合** (方法论)

### 6.1 **"代码平台"不是被动工具, 是主动"驾驶员"**

传统制造流程: 画图 → 打样 → 装配 → 调试 → 出货。代码平台在"调试"阶段才介入。
**本项目方法论**: 代码平台从 Day 1 就介入，做"主动预言 + 数据汇聚 + 决策支持"。

**具体做法**:

| 线下动作 | 代码平台动作 | 联调产出 |
|---|---|---|
| 开箱验收 | `hardware_inventory.py --check` | `hardware_inventory.csv` |
| 焊接 B1 板 | `verify --b1-pcb-check` (编译规则) | 焊接质量记录 |
| 电机参数辨识 | `identify_motor_params` + `verify_motor_spec` | `MotorSpec.provenance` |
| 电流环调参 | `foc_set_gains` + `hil_gap_analyze` | 注入后 `FOCConfig` |
| CAN 联调 | `can_pdo_map` + `avg_latency_ms` | 通信契约验证报告 |
| 阻抗调参 | `impedance_tune` + `zmp_analyze` | 关节阻抗参数 |
| 抓握力整定 | `calibrate_grip` + `chance_constrained_grip` | EKF σ_μ 收敛验证 |
| dyno 联调 | `climb_test --mode dyno` + `energy_scheduling_honest` | 捕获窗口数据 |

### 6.2 **统一数据流** (Single Source of Truth)

所有实测数据**必须**回流入代码平台，形成闭环:

```
硬件实测 (CSV / JSON)
    ↓
 hil_import 脚本
    ↓
experiments/hil_measurements/YYYYMMDD_*.json  (原始数据, 不可改)
    ↓
inject_hil_measurements.py  (自动更新 contracts/)
    ↓
contracts/hw_requirements.yaml / safety_budget.yaml  (唯一真源)
    ↓
代码平台自动读取 (MotorSpec / FOCConfig / EKF)
    ↓
测试用例自动校验 (verify / pytest)
    ↓
Git commit  (带实测 provenance)
```

**硬规则**: `assumed_*` 参数一旦被实测，**必须**在 24h 内替换为 `measured_*`，且 git commit 信息中带 `[measured]` 标签。不允许长期并存（防止"新旧值混用"）。

### 6.3 **失败也是数据** (Failure as Data)

任何联调失败**必须**记录:

```bash
python -m embodied_ai.cli hil_failure_log \
  --stage "L0-current-loop" \
  --symptom "MOSFET 过热 120°C" \
  --expected "Tj < 80°C" \
  --actual "Tj = 120°C" \
  --root-cause "PWM 死区时间 50ns 不足, 升至 150ns 解决" \
  --evidence "scope_20261008_deadtime.png"
```

**价值**: 这些失败记录**和成功记录同等宝贵**——它们就是"经验"的可编码形态。下次遇到同样问题，不用重新踩坑。

---

## 七、决策树: 现在应该做什么?

```
你现在处于哪个阶段?
│
├── 完全没经验, 没硬件, 只有代码
│   → ① 跑通 `verify` + `pytest` + `selfcheck` (熟悉代码)
│   → ② 按 §5 阶段 1 (模仿期) 开始, 买工具 + 练习板
│   → ③ 90 天内走完阶段 1-2
│
├── 有一些经验 (焊过板子), 但没做过机器人
│   → ① 跳过阶段 1, 直接从 §5 阶段 2 (协作期) 开始
│   → ② 按 [docs/14] 采购最小启动集 (~¥3,000)
│   → ③ 按 §3.1 Day 1-20 做完
│
├── 有完整单关节经验, 想做全身
│   → ① 直接按 §3.1 Day 21-60 推进
│   → ② 参考 [docs/16 §三/§四/§五]
│   → ③ 按 [docs/13] 填领域参数
│
└── 有全身经验, 想做攀岩任务
    → ① 直接按 §3.1 Day 61-90 推进
    → ② 重点 [docs/16 §六] 攀岩整机
    → ③ Phase P 认证流程 [docs/15]
```

---

## 八、常见误区 (Pitfalls)

| 误区 | 实际 | 建议做法 |
|---|---|---|
| "我有 20 年经验, 不用按流程" | 20 年经验是"踩过 500 次坑"的总结, 流程就是把这些坑固化下来 | 按流程走, 但可以在流程里加自己的经验 |
| "仿真调好了, 真机直接上" | gap 有 7 类, 必须按 §2.1 循环收敛 | 用渐进加载, 从 10% 开始 |
| "参数调好了就锁定, 不用再动" | 温度/老化/批次差异会导致漂移 | 每季度复测 Rs/Ld/ψ 一次 |
| "调试就是拧电位器" | 现代机器人调试是**系统辨识 + 参数注入** | 用 `identify_motor_params` 这类工具, 不要拍脑袋 |
| "仿真和实机是两套系统" | 应该是同一套代码 + 不同 backend | 用 HAL + `make_env(backend=...)` 切换 |
| "失败是丢人的事" | 失败是最有价值的数据 | `hil_failure_log` 记录, 回流为训练数据 |
| "我买最好的硬件就不会有问题" | 好硬件 + 错参数 = 坏系统 | 参数整定比硬件选型更重要 |
| "一个人能干完所有事" | 机械/电气/软件/控制 4 个领域 | 至少 2 人组队, 1 人专攻 1-2 个领域 |

---

## 九、结语

**"线下生产与代码平台有机结合"不是口号，是 5 个具体机制**:

1. **双模切换** — `make_env(backend=sim/real)` 让同一套代码跑两种模式
2. **参数注入** — `inject_hil_measurements` 把实测值回灌到契约
3. **闭环数据流** — 硬件实测 → JSON → contracts → 代码 → 测试 → git
4. **失败记录** — `hil_failure_log` 把每次踩坑变成可复用资产
5. **渐进加载** — 6 档负载逐步推进, 每档单独归因

**经验不是天赋，是可编码、可积累、可传承的数据**。本文档 + `docs/13/14/15/16` + 本项目代码仓库本身就是**经验数字化**的实践。

**如果你是零经验**: 按本文档 §5 的 90 天路径走，**每周 10 小时投入，3 个月达到独立联调单关节，6 个月达到全身协调**。这是**普通工程师**可达到的目标，不需要任何特殊背景。

**如果你是有经验的工程师**: 跳过阶段 1，直接从 §3.1 的 Day 1 开始，**联调过程本身就是积累**——把每一个失败记录到 `hil_failure_log`，就是把你自己的经验数字化。
