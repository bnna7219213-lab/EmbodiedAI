# 刚柔耦合动力学仿真 + 触觉硬件 DIY + Sim-to-Real 仿真栈

> 本文档是对外呼应用户提问 q5 (a+b+c) 的回应: 
> a) 刚柔耦合动力学 
> b) 触觉硬件 DIY 
> c) Sim-to-Real 仿真栈实操
>
> 关键词: URDF、Pinocchio、Drake、GelSight Mini DIY、PyBullet、TACTO

---

## 1. 刚柔耦合的家庭服务机器人动力学建模

### 1.1 为什么必须做?

```
场景: 机器人抓取毛巾并擦拭桌面
  → 毛巾是柔性体, 形变无法用刚体运动学描述
  → 力传感器反馈的力 = 桌面变形 + 毛巾变形 + 手臂柔性
  → 如果你用纯刚体模型做 RL 训练, 迁移到真实硬件会崩溃

场景: 机器人打开柜门
  → 柜门铰链有弹性, 铰链碰撞脉冲宽 5ms
  → FOC 自动关断可能会误触发 (瞬态电流 >30A)
  → 必须用瞬态电流模型去设计电流环参数

结论: 家庭场景的"柔性"不可忽略, 需要刚柔耦合模型
```

### 1.2 URDF 模型 (统一机器人描述格式)

```xml
<!-- robot_description/urd_v1/EmbodiedAI_v1.urdf -->

<robot name="EmbodiedAI_v1">
  <!-- 躯干基座 -->
  <link name="torso_base">
    <inertial>
      <mass value="12.0"/>   <!-- 含电池 -->
      <origin xyz="0 0 0.15"/>
      <inertia ixx="0.15" ixy="0" ixz="0" iyy="0.12" iyz="0" izz="0.10"/>
    </inertial>
    <visual>
      <geometry><box size="0.3 0.25 0.35"/></geometry>
    </visual>
  </link>
  
  <!-- 左肩关节 (柔性) -->
  <joint name="Lshoulder_pitch" type="revolute">
    <parent link="torso_base"/>
    <child link="Lupper_arm"/>
    <origin xyz="0 0.15 0.25"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.5" upper="2.5" effort="30.0" velocity="10.0"/>
    
    <!-- 柔性关节参数: K=刚度, D=阻尼 -->
    <dynamics damping="0.5" friction="0.1"/>
    <!-- 由关节柔性丝杆引起的形变 -->
    <safety_controller soft_lower="-2.4" soft_upper="2.4" k_position="100"/>
  </joint>
  
  <!-- 谐波减速器柔性模型 -->
  <transmission name="Lshoulder_trans">
    <plugin>transmission_interface/SimpleTransmission</plugin>
    <actuator name="Lshoulder_motor" mechanical_reduction="100"/>
    <joint name="Lshoulder_pitch" hardware_interface="hardware_interface/EffortJointInterface"/>
    <!-- 弹性接头: 标称刚度 5000 N/m -->
  </transmission>
  
  <!-- 力传感器 (ATI Gamma 替代品) -->
  <link name="Lforce_sensor">
    <inertial><mass value="0.1"/></inertial>
    <!-- 6 维力/力矩测量点 -->
  </link>
  
  <!-- 触觉指尖 (GelSight) -->
  <link name="Ltactile_tip">
    <inertial><mass value="0.02"/></inertial>
    <!-- 触觉摄像头内参 (已标定) -->
  </link>
</robot>
```

### 1.3 用 Pinocchio 做运动学/动力学计算

```python
# kinematics.py

import pinocchio as pin
import numpy as np

class RobotKinematics:
    """基于 Pinocchio 的运动学/动力学"""
    
    def __init__(self, urdf_path):
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, package_dirs="/robot_description/"
        )
        self.data = self.model.createData()
        
        # 添加柔性关节: Pinocchio 支持软接触
        self.soft_contact_model = self._build_soft_contact()
    
    def forward_kinematics(self, q):
        """前向运动学: 关节角 → 末端位姿"""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # 躯干坐标系下的各 link 位姿
        ee_id = self.model.getFrameId("Leef_tip")
        ee_pose = self.data.oMf[ee_id]  # SE3 变换矩阵
        
        return ee_pose.translation, ee_pose.rotation
    
    def inverse_kinematics(self, target_pos, target_rot=None, q0=None):
        """逆运动学: 末端位姿 → 关节角 (数值解)"""
        if q0 is None:
            q0 = pin.neutral(self.model)
        
        ee_id = self.model.getFrameId("Leef_tip")
        oMdes = pin.SE3(target_rot, target_pos) if target_rot else pin.SE3(np.eye(3), target_pos)
        
        q = q0.copy()
        for _ in range(1000):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            dM = oMdes.actInv(self.data.oMf[ee_id])
            err = pin.log(dM).vector  # 6 维误差
            err_norm = np.linalg.norm(err)
            
            if err_norm < 1e-4:
                break
            
            J = pin.computeFrameJacobian(self.model, self.data, q, ee_id)
            dq = -np.linalg.pinv(J) @ err
            q = pin.integrate(self.model, q, dq * 0.5)
        
        return q, err_norm < 1e-4
    
    def forward_dynamics(self, q, v, tau):
        """前向动力学: (位置, 速度, 力矩) → 加速度 (ABA 算法)"""
        # ABA = Articulated Body Algorithm, O(n) 复杂度
        a = pin.aba(self.model, self.data, q, v, tau)
        return a
    
    def inverse_dynamics(self, q, v, a):
        """逆动力学: 计算达到目标加速度所需的力矩 (RNEA)"""
        tau = pin.rnea(self.model, self.data, q, v, a)
        return tau
    
    def mass_matrix(self, q):
        """质量矩阵 M(q) [n×n]"""
        M = pin.crba(self.model, self.data, q)
        return M
    
    def _build_soft_contact(self):
        """柔性接触模型 (用刚体近似软接触力)"""
        # 6 维接触力 = [Fx, Fy, Fz, Mx, My, Mz]
        # 接触刚度 K_c 和阻尼 D_c
        K_c = np.diag([1000, 1000, 1000, 10, 10, 10])  # N/m, Nm/rad
        D_c = np.diag([50, 50, 50, 0.5, 0.5, 0.5])
        return {'K': K_c, 'D': D_c}
    
    def compute_contact_force(self, penetration, velocity):
        """柔性接触力 (Penalty-based, 无真实形变)"""
        K = self.soft_contact_model['K']
        D = self.soft_contact_model['D']
        F = K @ penetration - D @ velocity
        return np.maximum(F, 0)  # 只允许压力, 无拉力
```

## 2. GelSight Mini 完整 DIY 手册

### 2.1 BOM 与成本

| 部件 | 型号/规格 | 数量 | 单价 | 总额 |
|---|---|---|---|---|
| 弹性体 | Solaris Silicone XP-565 ( Shore 20A) | 1 块 | $25 | $25 |
| 示点子 | 0.2mm 直径不锈钢 | ~1000 个 | $8 (整包) | $8 |
| 光学玻璃 | 5mm 钢化玻璃 50×70mm | 1 片 | $3 | $3 |
| 摄像头 | OV5640 500万像素 (USB 3.0) | 1 个 | $15 | $15 |
| 漫射 LED | 暖白光 (3000K) 贴片 0603 | 3×3 = 9 个 | $0.3 | $3 |
| LED 驱动 | AL8805 恒流调光 | 1 个 | $1 | $1 |
| 柔性 PCB | 10mm×10mm 6 层刚挠板 | 1 块 | $5 | $5 |
| 3D 打印外壳 | PETG (耐化不熔性) | 1 件 (打印) | ~$2 | $2 |
| M2 不锈钢螺丝 | 长度 5mm | 4 个 | $1 | $1 |
| **总成本** | | | | **~$63** |

### 2.2 弹性体与光学关键参数

```
弹性体 (GelSight 的核心):

材料: Solaris XP-565 (双组分铂催化硅)
  参数:
    -  Shore A 硬度: 20 (接近人类指尖)     ← 关键, 硬度过高无纹理, 过低形变滞后
    -  拉伸强度: 4.5 MPa
    -  断裂伸长率: 450%
    -  透光度: >92% @ 550nm              ← 关键, 透光要高
    -  折射率: 1.41                       ← 与玻璃匹配
    -  催化比例: A:B = 1:1
  
  ⚠ 关键步骤:
    1. 真空脱气 5 分钟 (去除微小气泡, 否则破裂)
    2. 固化温度 25°C 24h (升温加快老化)
    3. 表面喷示点子 (0.2mm 不锈钢球随机分布)
    4. 喷哑光黑漆作为基底层 (吸收内部反射)

光学系统结构:
  ┌─────────────────────────────────────┐
  │  LED 阵列 (3×3) 漫射照明             │ ← 紧贴弹性体表面
  │  ═══════════════════════════════════ │
  │  示点子层 (5μm 不锈钢反射球)          │
  │  ═══════════════════════════════════ │
  │  反射镀层 (铝膜, A/R 镀膜)            │
  │  ═══════════════════════════════════ │
  │  弹性体 (Solaris 硅)                 │
  │  ═══════════════════════════════════ │
  │  5mm 光学玻璃                         │
  │  ═══════════════════════════════════ │
  │  相机 (OV5640, 光圈 f/2.8, 30fps)    │
  └─────────────────────────────────────┘
  
  视场: 25mm × 20mm
  空间分辨率: ~50μm (GelSight 精度极限)
  力分辨率: 0.05N
  帧率: 30Hz (受 USB3.0 限制)
```

### 2.3 触觉信号处理代码

```python
# gelsight_mini_driver.py

class GelSightMiniDriver:
    """
    GelSight Mini v1 完整驱动
    1. 视频采集 (OpenCV + FFMPEG)
    2. 图像处理 (背景减/特征提取)
    3. 力估计 (by标定矩阵)
    4. 滑脱检测 (光流法)
    """
    
    def __init__(self, device_id=0, calib_path="calib_gelsight_v1.pkl"):
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.bg = None                 # 背景图像
        self.calib = self._load_calib(calib_path)
        self.prev_frame = None
        self.flow = None               # 光流缓存
        
    def capture_raw(self):
        """抓取原始帧"""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("GelSight 摄像头读取失败")
        return frame  # (480, 640, 3) uint8
    
    def process_frame(self, frame):
        """
        完整处理管线 (GPU 加速)
        Returns:
            tactile_image: (3, 224, 224) GPU tensor
            force_6dof: (6,) [Fx, Fy, Fz, Mx, My, Mz] N, Nm
            slip_prob: (2,) [p_noslip, p_slip]
        """
        # 1. 裁剪弹性体区域 (椭圆 mask)
        mask = self.ellipse_mask  # 预计算
        cropped = frame[mask]
        
        # 2. 背景减法
        if self.bg is None:
            self.bg = cropped.copy()
        diff = cv2.absdiff(cropped, self.bg)
        
        # 3. 颜色空间转换 (RGB → HSV, 只取强度差异)
        hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]  # V通道
        
        # 4. ROI 裁剪 + resize 到 64×48 (GelSight Mini 原生)
        tactile_img = cv2.resize(value, (64, 48))
        
        # 5. 3 通道伪 RGB (H=V, S=0, V=0 → but 换成三帧不同曝光)
        tactile_3ch = np.stack([tactile_img, 
                                 self.prev_frame if self.prev_frame is not None else tactile_img,
                                 self.prev_prev_frame if self.prev_prev_frame is not None else tactile_img], axis=2)
        
        # 6. 转为 GPU tensor
        tensor = torch.from_numpy(tactile_3ch).permute(2, 0, 1).unsqueeze(0).cuda().float()
        
        # 7. 力估计 (通过标定矩阵)
        feat = self.calib_encoder(tensor)          # GPU 编码
        force = self.calib_matrix @ feat.cpu().numpy().flatten()[:6]
        
        # 8. 滑脱检测 (光流)
        if self.prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, value, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            slip_prob = np.clip(np.mean(flow_mag) / 10.0, 0, 1)
        else:
            slip_prob = 0.0
        
        self.prev_prev_frame = self.prev_frame
        self.prev_frame = value.copy()
        
        return tensor, force, slip_prob
```

## 3. Sim-to-Real 仿真栈实操

### 3.1 仿真框架选择

| 框架 | 实时性 | 触觉支持 | GPU 并行 | 易用性 | 本计划选用 |
|---|---|---|---|---|---|
| PyBullet | 弱 | TACTO (插件) | 否 | ★★★★ | Phase 0-1 |
| MuJoCo | 中等 | 有原生软接触 (但慢) | 否 | ★★★ | Phase 1 |
| Isaac Sim | 强 | 有 (简化物理) | ★★★★ (数千环境并行) | ★★ | Phase 2+ |

### 3.2 PyBullet + TACTO 实操骨架

```python
# sim_env_tacto.py

import pybullet as p
import pybullet_data
import tacto
import cv2

class TactileSimWithTACTO:
    """基于 TACTO 的视触觉仿真环境"""
    
    def __init__(self):
        # 1. 连接 PyBullet (带 GUI 调试, 训练时用 DIRECT)
        self.physics = p.connect(p.DIRECT)  # 训练时; p.GUI for debug
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240)  # 240Hz 物理
        
        # 2. 加载地面 + 物体
        self.plane_id = p.loadURDF("plane.urdf")
        self.obj_id = p.loadURDF("ycb_objects/006_mustard_bottle.urdf",
                                  basePosition=[0.5, 0, 0.5],
                                  globalScaling=1.0)
        
        # 3. 创建触觉传感器 (GelSight)
        self.bg = cv2.imread("conf/bg_digit.jpg")
        self.digit = tacto.Sensor(width=320, height=240, background=self.bg)
        
        # 4. 添加到机器人指尖
        # (实际用法: 先加载 URDF 机器人, 然后 add_camera)
        self.robot_id = p.loadURDF("embodied_v1.urdf", useFixedBase=True)
        # 假设 robot 的 link index 5 是 tactile tip
        self.digit.add_camera(self.robot_id, [5])
        
        # 5. 仿真步长 <= 1ms (与真实 GelSight 30Hz 错开)
        self.sim_substeps = 8  # 每 8 步后抓取触觉
        
    def reset(self):
        """重置环境"""
        p.resetBasePositionAndOrientation(self.obj_id, [0.5, 0, 0.5], [0,0,0,1])
        p.resetJointStates(self.robot_id, self.default_joints)
        for _ in range(10): p.stepSimulation()  # wait 稳定
        return self._get_obs()
    
    def step(self, action):
        """动作 (关节角度) → 物理步进 → 触觉 + 奖励"""
        # 1. 应用关节位置
        for i, target in enumerate(action):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=target, force=30, maxVelocity=5
            )
        
        # 2. 物理步进
        for _ in range(self.sim_substeps):
            p.stepSimulation()
        
        # 3. 渲染触觉 (TACTO 模拟)
        if self.step_count % self.sim_substeps == 0:
            color, depth = self.digit.render()
            self.current_tactile = color
        
        # 4. 获取本体感受
        joint_states = p.getJointStates(self.robot_id, range(7))
        positions = np.array([s[0] for s in joint_states])
        forces = np.array([s[3] for s in joint_states])
        
        # 5. 奖励计算
        reward = self._compute_reward()
        
        self.step_count += 1
        return self._get_obs(), reward, self._is_done(), {}
    
    def _get_obs(self):
        return {
            'tactile': self.current_tactile,             # (240, 320, 3)
            'visual': self._get_rgb_from_robot_cam(),     # (224, 224, 3)
            'proprio': np.concatenate([positions, forces]),
        }
```

## 4. 推荐方案

### 4.1 分阶段计划

| Phase | 硬件/仿真 | 范围 | 工具 |
|---|---|---|---|
| Phase 0 | 刚体 URDF (无柔性) | 仅双臂 | Pinocchio + PyBullet |
| Phase 0+ | 触觉集成 TACTO | 家庭场景 (杯子/毛巾) | PyBullet + TACTO |
| Phase 1 | 柔性关节 URDF | 全身 (含腿) | MuJoCo |
| Phase 2 | GelSight Mini DIY → 真实部署 | 指尖 3 个 | USB3.0 + Jetson |
| Phase 3 | Isaac Sim 迁移 | 全部 | Orin 本地仿真 |

### 4.2 关键里程碑

- M0: URDF 模型无误 + Pinocchio 运动学 IK 收敛 <1ms
- M1: TACTO 仿真输出 ~30Hz 触觉 + 与真实数据 low-level 对齐
- M2: GelSight Mini 标定后, 力估计误差 <5%
- M3: PyBullet 策略迁移到真实机械臂, 抓取成功率 >60%
- M4: 在线 SAC 微调, 真实成功率 >80% (同 reinforcement-learning.md 的 D3 目标)
