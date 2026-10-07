# 制造与调试工程指南 (Manufacturing & Debugging Guide)

> **日期**: 2026-10-08
> **用途**: Phase P (硬件到货后) 可落地的工程指南。把"组装一台腱驱动攀岩机器人"拆成 **5 层设备堆叠** (Layer 0-4)，每层都回答三个核心问题: **造什么、参数怎么调、问题怎么解**。
> **配套文档**:
> - [docs/14-hardware-procurement-checklist.md](14-hardware-procurement-checklist.md) — 采购清单
> - [docs/15-hil-test-protocol.md](15-hil-test-protocol.md) — HIL 48 小时流程
> - [docs/13-domain-parameters-checklist.md](13-domain-parameters-checklist.md) — 参数 provenance 表
> - [plan_v4.md](../plan_v4.md) — Gate 3/4 整体路线图

---

## 〇、核心设计哲学 (Equipment Philosophy)

### 0.1 渐进式制造 (Layered Manufacturing)

```
Layer 0: 单关节台架 (1 个关节 + 1 块 B1 板)
    ↓ 验证 FOC + 编码器 + 电流环
Layer 1: 完整单臂 (4 关节 : shoulder×2 + elbow×1 + wrist×1)
    ↓ 验证多关节 CAN-FD 同步 + 运动学
Layer 2: 双腿平衡 (4 关节 : hip×2 + knee×2)
    ↓ 验证 ZMP + 站立平衡
Layer 3: 双臂 + 双腿全身 (8 关节)
    ↓ 验证全身协调 + 欠驱动腱耦合
Layer 4: 攀岩整机 (8 关节 + 抓握 + dyno)
    ↓ 验证墙面接触 + 能量调度 + 真实任务
```

**绝不跨层**: 每一层有明确的"放行判据" (硬件就绪 + 参数锁定 + 测试通过)，任一层不通过绝不推进到下一层。

### 0.2 参数整定顺序 (Parameter Tuning Sequence)

```
机械装配 → 编码器对齐 → 电流环 (FOC) → 速度环 → 位置环 → 阻抗控制 → 平衡控制 → 任务规划
    ↓              ↓              ↓           ↓          ↓          ↓           ↓
 旋变/LCR     极对数/偏移    极点配置     低带宽    K/B 调参    Kp/Kd        MCTS grip
```

---

## 一、Layer 0: 单关节台架 (Single Joint Rig)

> **目标**: 1 台 B1 驱动板 + 1 台电机 + 1 只编码器 + 示波器验证 FOC 完整回路
> **周期**: 4-6 小时 (不含采购)
> **涉及 Phase P**: P1-P2 前置

### 1.0 BOM (单关节台架)

| 组件 | 规格 | 数量 | 来源 |
|---|---|---|---|
| B1 驱动板 | 4 层 PCB, 2 oz, 55×45 mm | 1 | 本地打样 (artifacts/b1_tendon_drive_package/) |
| BLDC 电机 | MN3508 或等效 | 1 | 采购 |
| 编码器 | TLI5012B (19-bit) | 1 | 采购 |
| 减速器 | Planetary 1:20 | 1 | 采购 |
| 隔离电源 | 24V/10A DC | 1 | 采购 |
| 示波器 | 4 通道 ≥100 MHz | 1 | 采购/外借 |
| CAN 接口 | CANable Pro (candleLight) | 1 | 采购 |
| 端子/线材/连接器 | JST VH/Molex | 1 批 | 采购 |
| 光学编码器支架 | 3D 打印 (PETG) | 1 | 自制 |
| 电机支架 | 3D 打印 (PETG/ABS) | 1 | 自制 |
| 丝杠+负载块 | M8 丝杠 (模拟负载) | 1 | 采购 |

### 1.1 机械装配检查清单

```markdown
装配前:
  [ ]  PCB 100% 目检 (铜箔无划痕、焊盘无虚焊、过孔无断裂)
  [ ]  电源对地阻抗 > 10 kΩ (防止短路)
  [ ]  编码器支架与电机轴同轴度 < 0.1 mm
  [ ]  减速器输出轴径向跳动 < 0.05 mm
  [ ]  所有螺丝预涂螺纹胶 (防振松)

装配后:
  [ ]  电机手动旋转无卡滞
  [ ]  编码器读数随轴旋转单调单调递增
  [ ]  编码器零位对齐 (D1.5 已定义 procedure)
  [ ]  腱张力传感器零点校准 (无负载读数=0)
  [ ]  相电流霍尔零点校准 (静态读数=0±0.05A)
```

### 1.2 焊接/拼接/组装 常见制造问题与解法

| 问题 | 根因 | 解法 | 预防措施 |
|---|---|---|---|
| **QFN48 桥接** | 焊膏过多/对位偏移 | 用吸锡带重新开法 + 异丙醇清洗 | 开钢网厚度 0.12 mm (BOM 已规定) |
| **BGA 虚焊** | 回流温度曲线不当 | X-ray 检测 + 风枪局部返修 | 严格按锡膏厂商温度曲线 |
| **连接器插反** | JST VH 方向标记不清 | 首次接电前万用表核验 | 插座胶壳加色标 |
| **PCB内层断裂** | 安装应力过大 | 更换新板 + 增加三角形固定筋 | 板框设安装孔、筋板 |
| **编码器支架松动** | 3D 打印收缩率不一致 | 环氧树脂粘接替代螺丝 | 打印前仿真收缩 + 螺纹嵌件 |
| **腱索跳槽** | 滑轮槽宽度不足 | 更换带挡边滑轮 | 槽宽 = 腱径×1.2 |
| **减速器背隙过大** | 装配公差错配 | Harmonic Drive 替代 | 预加载安装 |
| **示波器地线环路** | 多通道共地 | 单端探头 + 隔离变压器 | 所有测量参考同一系统地 |
| **电机启动抖动** | 初始位置辨识不准 | 高频注入法 (HFI) | 静止时强制对齐 d 轴 |
| **CAN 总线间歇断连** | 终端电阻未接 | 总线两端各接 120 Ω | 任何 CAN 网络都必须 |

---

## 二、Layer 0 → 调试核心：参数整定

### 2.1 电流环 (FOC) 参数整定

**物理模型**: PMSM d-q 轴数学模型

```
V_d = Rs·I_d + Ld·dI_d/dt - ω·Lq·I_q
V_q = Rs·I_q + Lq·dI_q/dt + ω·Ld·I_d + ω·Ke
Te = 1.5·pp·(ψ_m·I_q + (Ld-Lq)·I_d·I_q)
```

**调参策略** (极点配置法):

```python
# src/embodied_ai/hardware/motor_model.py 中的 current_loop_gains()
# 给定:
Rs = 0.050  # 相电阻 (实测, 用 LCR 表或直流压降法)
Ld = 15e-6  # d 轴电感 (实测 LCR 表 1 kHz)
Lq = 28e-6  # q 轴电感 (实测)
pp  = 7     # 极对数
ωc  = 2π·800  # 目标带宽 800 Hz (设计值)

# PI 增益 (零极点对消):
kp_d = Ld · ωc  = 15e-6 · 2π·800  = 0.0754  V/A
ki_d = Rs · ωc  = 0.050 · 2π·800   = 251.3    V/(A·s)
kp_q = Lq · ωc  = 28e-6 · 2π·800  = 0.1407  V/A
ki_q = Rs · ωc  = 同 d 轴        = 251.3    V/(A·s)

# 离散化约束:
f_ctrl = 10000 Hz  →  T_ctrl = 100 μs
ωc·T_ctrl = 2π·800·100e-6 = 0.503  < 2  (满足!)
```

**实物调参流程**:

```bash
# Step 1: 测量电机参数 (Layer 0 台架)
python -m embodied_ai.cli identify_motor_params --method dc_step  # 注入阶跃电压, 拟合 Rs/Ld/Lq
python -m embodied_ai.cli identify_motor_params --method hfi       # 高频注入辨识初始位置
python -m embodied_ai.cli identify_motor_params --method afm      # 反电动势法测 Ke

# Step 2: 自动算出 PI 增益
python -c "
from embodied_ai.hardware.motor_model import MotorSpec
m = MotorSpec(Rs_ohm=0.05, Ld_h=15e-6, Lq_h=28e-6, psi_m_wb=0.0028, pole_pairs=7)
print(m.current_loop_gains(bw_hz=800))
"

# Step 3: 注入到 FOC 固件
python -m embodied_ai.cli foc_set_gains --kp-d 0.0754 --ki-d 251.3 --kp-q 0.1407 --ki-q 251.3

# Step 4: 验证 — 当前环阶跃响应
python -m embodied_ai.cli foc_current_loop --joint j1 --target-iq 2.0 --duration 5
# 判据: 上升时间 < 5 ms, 超调 < 5%, 稳态误差 < 2%
# 示波器核验: 10 ms/div, 无振荡
```

**常见电流环问题诊断**:

| 症状 | 根因 | 解法 |
|---|---|---|
| 大电流振荡 (>5A) | kp 过大 (接近离散稳定边界) | 降带宽到 500 Hz, 重算 |
| 稳态误差 >20% | ki 过小或死区 | 补死区补偿 (逐相实现), 升 ki |
| Id ≠ 0 时有转矩 | 编码器角度偏移 | 手动角度辨识 procedure |
| 高速时电流跟踪丢失 | 反电动势饱和 (Vq_max 到顶) | 弱磁 (Id < 0) 或降速 |
| 启动瞬间过流 | 初始位置未知 | 强制 d 轴对齐 (HFI/I-f 启动) |
| MOSFET 异常发热 | 死区时间不足 | 升到 ≥100 ns, 验示波器死区 |

---

## 三、Layer 1: 完整单臂 (Single Arm, 4 Joints)

### 3.0 BOM (与 Layer 0 增量)

| 组件 | 数量 | 备注 |
|---|---|---|
| B1 驱动板 | 4 | 4 关节 (shoulder pitch/roll + elbow + wrist) |
| 电机 | 4 | 各关节规格可能不同 (shoulder 大力矩) |
| 编码器 | 8 | 双端 (电机端 + 腱输出端) |
| 减速器 | 4 | 不同减速比 |
| 腱索 | 4 根 | Dyneema SK75, 直径 1.0 mm |
| 滑轮 | 12 | 含导向轮 |
| 3D 打印结构 | 1 批 | PETG/碳纤维尼龙 |
| 碳纤维管 | 2 | 上臂/前臂连杆 |
| 轴承 | 8 | 深沟球/角接触 |

### 3.1 拼接/组装关键工艺

**碳纤维管 + 金属端头**:
- 拼接方式: 环氧树脂 + 过盈配合 (H7/g6)
- 胶接表面用 180# 砂纸打毛 → 丙酮清洗 → 施胶 (A/B 胶 1:1)
- 固化 24 h @ 25°C 或 2 h @ 60°C
- 验证: 静载 2× 额定扭矩无脱胶

**滑轮 + 腱索**:
- 滑轮槽 = 腱径 × 1.2 → 跳动量 < 0.05 mm
- 腱索张力 = 额定张力 × 50% (防跳槽)
- 接头用钢丝绳夹或冷压端子 (不是绑扎)
- 拼接处加热缩管防磨损

**B1 板 4 模块 CAN-FD 总线**:
- 物理接线: 手拉手 (daisy chain), 不是星形
- 终端电阻: 两端各 120 Ω (仅 1 个在板内, 1 个加在末端)
- 波特率: 仲裁段 1 Mbps / 数据段 5 Mbps
- 节点 ID: 0x10-0x13 (shoulder pitch/roll + elbow + wrist)
- sync 帧每 5 ms (200 Hz)，各节点在 200 Hz 同步
- 帧格式: CAN-FD 64, DLC=15, BRS=1

### 3.2 运动学参数整定

**DH 参数 (4 关节单臂)**:

```
Joint    θ(deg)   d(m)    a(m)    α(deg)   Type
---------------------------------------------------------
j1(sp)   var      0       0       -90      Revolute j2(sr)        0       0.20            Revolute
j3(el)   var      0       0.25    0        Revolute
j4(wr)   var      0       0.18    0        Revolute
```

**正/逆运动学验证**:
```bash
python -m embodied_ai.cli kin_verify --arm left --n_samples 1000
# 判据: 正逆往返误差 < 0.1 mm, 奇异点附近无跳变
```

**速度/位置环增益** (外环带宽 < 电流环 1/5-1/10):
```
速度环 (PI): Kp_v = 0.5, Ki_v = 5  (100 Hz 带宽 → 20 Hz 实际)
位置环 (P):  Kp_p = 50             (20 Hz 带宽)
```

### 3.3 常见问题

| 问题 | 根因 | 解法 |
|---|---|---|
| 多节点 CAN 同步丢帧 | 终端电阻位置不对/波特率容差 | 先断 2 节点单测, 逐加确认 |
| 关节运动有"阶梯感" | 编码器分辨率不足 | 升 14-bit→19-bit, 加 dithering |
| 减速器回程差 >0.5° | 齿轮侧隙 | 换 Harmonic Drive / 消隙弹簧 |
| 腱索松弛 | 热胀冷缩 | 恒张力弹簧 / 定期复紧 |
| 手臂全展时抖动 | 低频共振 + 结构柔性 | 加加速度前馈 / 结构加筋 |
| 腱索与滑轮磨损 | 包角过大/材质不匹配 | 包角 < 90°, 滑轮阳极化 |

---

## 四、Layer 2: 双腿平衡 (Biped Balancing, 4 Joints)

### 4.0 BOM 增量

| 组件 | 数量 | 备注 |
|---|---|---|
| B1 驱动板 | 4 | hip pitch/roll + knee ×2 |
| 电机 | 4 | 大力矩版 (U8 Lite) |
| 编码器 | 8 | 双端 |
| IMU | 1 | ICM-42688 或 BMI088 (6 轴) |
| 力矩传感器 | 2 | 踝关节 6 轴力/力矩 |
| 脚掌 | 2 | 3D 打印 + 橡胶缓冲 |
| 腿结构件 | 1 批 | 碳纤维管 + 7075 铝接头 |

### 4.1 身体平衡控制参数整定

**ZMP (零力矩点) 稳定性判据**:

```
ZMP_x = (Σ m_i(z_i+g)·x_i − Σ m_i·ẍ_i·z_i) / (Σ m_i(z_i+g))

稳定性: ZMP ∈ 支撑多边形 (脚底接触面凸包) → 稳定
        ZMP ↑ 接近边缘 → 临界
        ZMP ∉ 支撑多边形 → 倾倒
```

**平衡控制回路** (从外到内):

```
任务层 (1 Hz): 目标 → 期望 ZMP 位置 → 修正步态
  ↓
步态层 (20 Hz): ZMP 参考 → 单腿柔顺阻抗 → 关节力矩参考
  ↓
阻抗层 (200 Hz): 关节力矩 → 电流指令 (K_BodyImpedance)
  ↓
电流环 (10 kHz): FOC PWM
```

**调参顺序**:

1. **先锁定电流环** (每关节单独调，Layer 0 已完成)
2. **速度环** (截止频率 50-100 Hz，低于电流环 1/10)
3. **阻抗环**: Kp = 500-2000 N/m, Kd = 50-200 N·s/m
4. **ZMP 补偿**: Kp_zmp = 0.5, Ki_zmp = 0.05 (慢积分)
5. **姿态 (俯仰/滚转)**: 由 IMU 反馈, Kp_att = 50, Kd_att = 5

**IMU 融合参数** (EKF):
```
# 状态: [roll, pitch, ω_x, ω_y, b_gx, b_gy]  (6 维)
# 观测: IMU 加速度计 (低频), 陀螺仪 (高频)
Q = diag([1e-4, 1e-4, 1e-3, 1e-3, 1e-6, 1e-6])  # 过程噪声
R_acc = diag([0.01, 0.01, 0.01])                  # 加速度计噪声
R_gyro = diag([0.001, 0.001, 0.001])              # 陀螺仪噪声
```

### 4.2 平衡站立调试流程

```bash
# Step 1: 传感器校准
python -m embodied_ai.cli imu_calibrate --n_samples 2000  # 静止 30s, 估计陀螺零偏
python -m embodied_ai.cli toe_calibrate --joint leg_left # 脚趾传感器零偏

# Step 2: 单腿力控测试
python -m embodied_ai.cli leg_stiffness_test --joint leg_left_knee --target_force 50
# 判据: 阶跃响应 <100 ms, 超调 <10%

# Step 3: 双腿站立 (受限空间 + 安全绳!)
python -m embodied_ai.cli biped_stand --duration 30 --safety_rope yes
# 判据: ZMP 轨迹在支撑多边形内, 站立 30s 不失稳

# Step 4: 外力扰动
python -m embodied_ai.cli biped_push_recovery --force 20 --direction front
# 判据: ZMP 偏移后 1s 内恢复
```

**常见平衡问题**:

| 症状 | 根因 | 解法 |
|---|---|---|
| 站立时缓慢倾倒 | ZMP 积分饱和 或 姿态角偏置 | 积分 anti-windup + IMU 在线校准 |
| 高频抖动 (>5 Hz) | 阻抗 Kd 过小或传感器噪声 | 低通滤波器 (截止 20 Hz) + 降 Kd |
| 起步"打滑" | 脚趾抓地力不足 | 橡胶脚掌 / 升 μ 系数 |
| 两腿不同步 | 左右关节电流环不一致 | 每通道独立调增益 |
| 受推力后恢复过冲 | Kp_zmp 太大 | 降 Kp + 升 Kd (加阻尼) |
| 髋部下沉 | 膝关节力矩不足 | 升电流限制 + 减减速比 |

### 4.3 制造: 腿结构装配要点

- 碳纤维管与铝接头胶合时, 管内壁涂导电胶 (防静电积累)
- 编码器线缆走管内部 (不可外露, 防刮)
- 腱索走向避开运动干涉区 (3D 仿真预检)
- 脚掌加装橡胶衬垫 (增大 μ, 实测 μ_rubber ≈ 0.8-1.0)

---

## 五、Layer 3: 双臂 + 双腿全身协调 (Whole Body, 8 Joints)

### 5.0 全身协调策略

**任务优先级 (QP 优化, 层次化)**:

```
P0 (硬约束): 关节力矩限制, ZMP 稳定性, 自碰撞避免
P1 (主任务): 末端执行器轨迹跟踪
P2 (次任务): 姿态优化 (避免奇异/最大化灵活度)
P3 (最低): 能耗最小化
```

**调参关键**:
```
# 逆运动学权重 (shoulder > elbow > wrist)
w = [10.0, 5.0, 2.0, 1.0]  ← 近端关节高权重, 运动分配更靠基座

# 逆解算 (Jacobian damped pseudoinverse):
θ_dot = J^T (J J^T + λ² I)^(-1) · v_des
λ = 0.05  (阻尼因子, 避免奇异)
```

### 5.1 CAN-FD 组网 (8 节点)

```
节点 ID    关节          帧 ID (RX)     帧 ID (TX)
-----------------------------------------------------
0x10       shoulder_L     0x100          0x180
0x11       shoulder_R     0x100          0x181
0x12       elbow_L        0x100          0x182
0x13       elbow_R        0x100          0x183
0x14       hip_L          0x110          0x190
0x15       hip_R          0x110          0x191
0x16       knee_L         0x110          0x192
0x17       knee_R         0x110          0x193

sync 帧: 0x080 (每 5 ms)
紧急帧: 0x001 (STO, 最高优先级)
```

**同步策略**:
- 各节点接收 sync 帧 → 在 5 ms 窗口内同时更新电流指令
- PDO mapping: rx_pos/rx_vel/rx_iq / tx_pos/tx_vel/tx_temp
- 帧丢失: 100 ms 未收到 sync → 自动降功率到 50% → 500 ms 仍无 → STO

### 5.2 全身装配要点

- 躯干框架用 20×20 mm 铝型材 (强度/重量比, 无应力集中)
- 双臂挂载时, 肩关节轴线与髋关节轴线平行 (避免行走时手臂晃动耦合)
- 髋关节电机 + 减速器尽量靠近质心 (降惯性)
- 腱索走向: 内部走管, 避免缠绕

### 5.3 调试联机关键 (8 节点联网)

```bash
# Step 1: 全节点心跳
candump can0 | grep "0x7"  # 心跳帧 (所有节点广播)
# 预期: 每 1s 8 帧, 每个节点 1 帧

# Step 2: PDO 映射验证
python -m embodied_ai.cli can_pdo_map --node 0x10  # shoulder_L
# 预期: 3 rx, 3 tx 确认

# Step 3: 200 Hz 同步验证 (示波器)
# CANable 收到 sync 帧 → 从节点同时响应 → 看各节点 PDO 帧是否在 200 μs 内到达

# Step 4: 开环全关节旋转
python -m embodied_ai.cli whole_body_motion --pattern squat --duration 10
# 预期: 8 关节同步, 2 秒下蹲, 2 秒起身, 无交替延迟
```

---

## 六、Layer 4: 攀岩整机 (Climbing Full System, 8 Joints + Grip)

### 6.0 攀岩专用增量设备

| 组件 | 数量 | 备注 |
|---|---|---|
| 灵巧手 | 2 | F5-欠驱动, 5 指, 腱驱动 |
| 触觉传感器 (GelSight Mini) | 10 | 每指尖 + 掌面 |
| 张力传感器 | 8 | 腱张力反馈 |
| 抓握力传感器 | 10 | 指尖法向力 |
| 攀岩墙 | 1 | 木质框架 + 树脂 hold |
| 安全绳 | 1 | 动力绳 + 自锁保护器 |
| dyno 捕捉垫 | 1 | 落地保护 |

### 6.1 腱驱灵巧手装配

**手指关节 (3 关节 × 5 指)**:
```
[电机] → [腱索] → [近端指节] → [腱索] → [中间指节] → [腱索] → [远端指节]
   ↓           ↓          ↓            ↓           ↓          ↓
  滑轮      M2.5 轴    M2.0 轴     M2.5 轴    M1.6 轴    M1.2 轴
```

**装配要点**:
- 腱预张力 = 2 N (空载时) → 消除空程
- 滑轮包角 < 90° → 避免自锁
- 关节限位: 近端 0-90°, 中间 0-120°, 远端 0-60°
- 复位弹簧 (扭簧)辅助伸展 (拮抗 VSA)

**拼接/焊接难点**:
- 手指关节为 M1.2 微型螺丝, 需 #00 十字螺丝刀 + 螺丝胶
- 腱索张紧使用张力计 (精度 ±0.1 N), 不可凭手感
- 触觉传感器 (GelSight) 粘贴于指尖, 需光固化胶 (避免起气泡)

### 6.2 攀岩控制参数整定

**抓握力调度** (chance_constrained_grip):

```python
# src/embodied_ai/safety/chance_constrained_grip.py
# 给定: f_t 切向力需求, μ̂ 估计, σ_μ 不确定度, δ 安全目标

# 攀岩高安全: δ = 0.001  →  z = 3.0902
# f_n = f_t / (μ̂ - 3.0902 * σ_μ)

# 例: f_t = 60 N (体重份额), μ̂ = 0.8, σ_μ = 0.05
# denom = 0.8 - 3.0902*0.05 = 0.6455
# f_n = 60 / 0.6455 = 92.9 N  → 每指尖约 18.6 N (5 指分担)

# 若 σ_μ 升到 0.1:
# denom = 0.8 - 3.0902*0.1 = 0.4910
# f_n = 60 / 0.4910 = 122.2 N  → 显著升握力
# → 力闭环 EKF σ_μ 收敛速度直接决定安全性
```

**能量调度参数** (VSA + dyno):

```
v_capture_max = 0.433 m/s      # A3 实测反解, Hunt-Crossley 捕获上限
E_stored_max  = 0.967 J        # ANTAGONISTIC VSA 实测
m_swing       = 1.2 kg         # 摆动质量
g             = 9.81 m/s²
h_max         = v_capture_max² / (2g) = 0.0096 m = 9.6 mm  # 微荡捕获窗口
```

**攀岩动作分解参数**:

| 动作 | 关节 | 速度 | 加速度 | 腱张力预算 | 时间 |
|---|---|---|---|---|---|
| 伸臂够点 | shoulder | 60°/s | 200°/s² | 40 N | 1.0s |
| 收腿蹬墙 | hip | 45°/s | 150°/s² | 80 N | 1.5s |
| 手转握 | gripper | 90°/s | N/A | 60 N | 0.5s |
| dyno 脱手 | all | max | max | 150 N (peak) | 0.1s |
| 捕获 | gripper | 慢收 | N/A | 40 N | 0.3s |

### 6.3 攀岩联调流程

```bash
# Step 1: 抓握校准 (每指尖)
python -m embodied_ai.cli calibrate_grip --finger index_left --n_samples 500
# 自动切 5 档法向力 (5/10/20/30/40 N), 拟合 f(T tendon displacement)

# Step 2: 墙面静态悬挂
python -m embodied_ai.cli climb_test --mode static_hang --duration 30
# 判据: 4 个 hold, 2 双手 2 脚, ZMP 稳, 30s 不掉

# Step 3: 4→3→4 换脚
python -m embodied_ai.cli climb_test --mode foot_transition
# 判据: 4 支撑 → 3 支撑换脚 → 4 支撑, 无失稳

# Step 4: dyno 单跳
python -m embodied_ai.cli climb_test --mode dyno --wall_distance 1.2
# 判据: dyno 起跳 → 抓点成功, 4 次尝试成功率 >50%

# Step 5: 小岩壁完攀
python -m embodied_ai.cli climb_test --mode full_route --wall 1.5m
# 判据: 从底到顶 (5 hold), 总时间 <60s, 无失误
```

**常见攀岩问题**:

| 症状 | 根因 | 解法 |
|---|---|---|
| 指尖在 hold 上打滑 | 法向力不足/μ 估计误差 | 升 δ 安全裕度; 滑粉增 μ |
| dyno 抓点失败 | v_capture 过大 / hold 预估误差 | 提前泄能; 降目标速度 |
| 悬挂时肩部酸痛(仿真) | 力矩分配不均 | QP 优化加力矩最小化 |
| 踝关节刚度不够 | 脚底接触模型简化 | 加 6 轴力/力矩传感器 |
| 腱索疲劳断裂 | 弯曲半径太小 | 滑轮直径 ≥ 10× 腱径 |
| 机体朝墙面旋转 | 水平力不足 | 踏步增大水平力分量 |
| 能量调度误判 | 捕获窗口 miss | IMU 数据提前 50 ms 预测 |

---

## 七、回环 (Closed-Loop) 整定方法论

### 7.1 三环串级 PID 整定

```
内环 (电流环, 10 kHz): FOC PI, 带宽 800 Hz
  ↓
中环 (速度环, 1 kHz): PI, 带宽 80 Hz
  ↓
外环 (位置/阻抗, 200 Hz): P/PD, 带宽 20 Hz
  ↓
任务环 (ZMP/抓握, 20 Hz): QP/优化
```

**每环独立整定步骤**:
1. 断开外环, 本环闭环
2. 注入阶跃信号
3. 测量上升时间、超调、稳态误差、带宽
4. 调参满足: 超调 <5%, <2 次振荡, 稳态误差 <1%
5. 接回外环, 降外环带宽到本环 1/5-1/10

### 7.2 位总线通信延迟管理

```
全链路延迟实测:
  传感器 → 100 μs (编码器) / 200 μs (IMU) / 50 μs (电流传感器)
  处理   → 50 μs (EKF 更新) / 20 μs (PID 计算)
  通信   → 110 μs (CAN-FD 帧发送) + 传播时间 (10 m 线缆 ≈ 50 μs)
  DAC    → 10 μs (PWM 更新)
  ────────────────────────────────
  总计   → < 500 μs (远低于 1000 μs 控制周期)
```

### 7.3 传感器融合 EKF 调参

**流程噪声 Q** (决定 EKF "信任模型"的程度):
```
Q_tactile = diag([0.05, 0.002, 0.02])
# f_t: 0.05 (N²/step) — 切向力变化较快
# μ:   0.002 — 摩擦变化慢
# v_t: 0.02 — 速度适中
```

**观测噪声 R** (决定 EKF "信任传感器"的程度):
```
R_tac = 0.25    # 触觉力传感器实际噪声方差
R_vis = 0.02    # 视觉 μ 估计较准
```

**调参经验**: 初始时 Q 偏大 (EKF 跟观测), 工作稳定后降 Q (EKF 跟模型)。

---

## 八、验收与放行矩阵

| 层 | 设备就绪 | 参数锁定 | 测试通过 | 标志 |
|---|---|---|---|---|
| L0 | B1 板 + 电机 + 编码器 | 电流环 PI | FOC 24h 无故障 | 进入 L1 |
| L1 | 4 关节 + 4 B1 + CAN | 速度/位置环 | 单臂轨迹跟踪 <5mm 误差 | 进入 L2 |
| L2 | 双腿 + 4 B1 + IMU | 阻抗 + ZMP | 30s 站立, 抗 20N 推 | 进入 L3 |
| L3 | 8 关节全身 + 8 B1 | QP 权重 | squat/side-step | 进入 L4 |
| L4 | 手掌 + 触觉 + 绳索 | grip schedule | 单 dyno + 5-hold 路线 | Gate 4 达标 |

---

## 附录 A: 故障排查快速参考 (FAQ)

| # | 症状 | 优先排查 | 修复 |
|---|---|---|---|
| 1 | 上电无任何反应 | 电源指示灯, B1 板 5V LDO | 换 LDO / 查电源 |
| 2 | 电机不转 | 编码器 sin/cos 信号 | 换编码器线序 |
| 3 | 电机转但无力 | 电流环增益, 相位顺序 | 交换任意两相 |
| 4 | 编码器读数乱跳 | 线缆屏蔽接地 | 重接屏蔽 |
| 5 | CAN 卡死 | 终端电阻, 地线环流 | 单节点隔离测试 |
| 6 | 多节点不同步 | sync 帧时序 | 示波器抓 SYNC 时序 |
| 7 | MOSFET 过热 | PWM 死区, 相电流波形 | 升死区时间 |
| 8 | 腱索跳槽 | 滑轮槽张力/包角 | 重排滑轮布局 |
| 9 | 抓握时手指错位 | 腱张力不均 | 逐一校准张力 |
| 10 | 平衡时 ZMP 过大 | IMU 零偏 | 重新校准零偏 |

## 附录 B: 制造文件交接清单

```
B1 板打样交付:
  [ ] Gerber 5 层文件 (已生成)
  [ ] BOM CSV (13 行关键器件, 已生成)
  [ ] 装配图 (B1_tendon_drive_pcb.svg, 已生成)
  [ ] 钢网 Gerber (由板厂自动生成)
  [ ] 坐标文件 (centroid, 由 Altium/KiCad 生成)

Layout 工程师交付:
  [ ] Altium/KiCad 工程文件 (.PcbDoc/.kicad_pcb)
  [ ] Gerber RS-274X 5 层
  [ ] IPC-D-356A 网表
  [ ] 装配图 (带极性标记)
  [ ] DRC/ERC 报告 (0 ERROR)
  [ ] BOM (含 PN, 厂商, 替代料)
```
