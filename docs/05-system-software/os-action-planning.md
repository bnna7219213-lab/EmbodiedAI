# OS 层动作规划与可解释性 — 具身智能体的操作系统

> 本文档回答核心问题: "操作系统如何承载 AI 动作规划? 如何让 AI 的决策过程可解释?"
>
> 关键词: ROS2、实时调度、动作规划、任务引擎、决策日志、可解释性、人机协作

---

## 1. 操作系统层级的职责定义

### 1.1 三层软件架构

```
具身智能体软件不是在"裸 AI 模型"上运行, 而是由操作系统调度分层的:

┌─────────────────────────────────────────────────────────────┐
│ 高层: LLM/VLA 任务理解 (1Hz, 慢)                             │
│   "去厨房拿一杯水" → 子任务链                                │
│   → 子任务链: [导航到厨房 → 定位水杯 → 抓取 → 返回 → 递给]    │
│   → 输出: TaskGraph (有向无环图)                             │
│   运行在: CPU 0-3 (本地核心, 分担 GPU 算力和推理延迟)         │
├─────────────────────────────────────────────────────────────┤
│ 中层: RL/规划控制器 (20Hz, 中速)                              │
│   根据当前子任务路径, 推理 "当前步应该移动到哪"               │
│   → 输入: [视觉/触觉/本体/当前子任务 ID]                     │
│   → 输出: 关节目标角度序列 (20步 @ 100ms)                    │
│   运行在: CPU 4-5 (隔离核心之一, 保证实时)                   │
├─────────────────────────────────────────────────────────────┤
│ 底层: FOC 电机控制 (10kHz, 快)                                │
│   → 输入: 目标角度 (已有 100μs)                              │
│   → 输出: SVPWM 占空比                                       │
│   → 实际执行物理运动                                         │
│   运行在: Cortex-M 核心 (F4/G4/H7 MCU, 裸机)                │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 操作系统在每一层做什么

```
┌────────┬────────────────────────────────────┬────────────────────────┐
│ 层级    │ 操作系统提供的                      │ 本计划实现              │
├────────┼────────────────────────────────────┼────────────────────────┤
│ CPU 调度│ 实时优先级分配                      │ SCHED_FIFO + isolcpus  │
│        │ 任务隔离 (控制跑专用核心)            │ CPU 4-7 隔离           │
├────────┼────────────────────────────────────┼────────────────────────┤
│ 内存    │ 避免 page fault                   │ mlockall + 大页         │
│        │ 预分配缓冲区, 不运行时分配           │ 启动阶段全部预分配      │
├────────┼────────────────────────────────────┼────────────────────────┤
│ 中断    │ 路由控制 (实时核心不处理非实时中断)  │ IRQ affinity           │
│        │ GPIO/SPI/CAN 中断分时               │ threaded irq           │
├────────┼────────────────────────────────────┼────────────────────────┤
│ 通信    │ 进程间通信 (ROS2 topic/service)     │ DDS (CycloneDDS, RMw)  │
│        │ 核间通信 (RPmsg)                   │ OpenAMP                │
├────────┼────────────────────────────────────┼────────────────────────┤
│ 时间    │ 高精度定时 / 计时                   │ hrtimers + TSC         │
│        │ 统一时钟同步                       │ PTP/gPTP               │
├────────┼────────────────────────────────────┼────────────────────────┤
│ 设备    │ 设备文件抽象                       │ CAN class / SPI class  │
│        │ 驱动管理                           │ Device Tree overlays   │
└────────┴────────────────────────────────────┴────────────────────────┘
```

---

## 2. ROS2 与实时调度深度整合

### 2.1 ROS2 节点拓扑

```
具身智能体 ROS2 节点图:

                    ┌─────────────────────────┐
                    │  /task_planner (1Hz)    │
                    │  VLA 模型推理服务端       │
                    │  订阅: 用户语音/文本      │
                    │  发布: /task_graph        │
                    └────────────┬────────────┘
                                 │ TaskGraph
                                 ▼
                    ┌─────────────────────────┐
                    │  /state_machine (10Hz)  │
                    │  任务状态机 (SMACH)       │
                    │  订阅: /task_graph        │
                    │  发布: /current_subtask    │
                    │       /state_machine_state │
                    └────────────┬────────────┘
                                 │ Subtask + State
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
┌──────────────────┐ ┌───────────────────┐ ┌──────────────────────┐
│ /motion_planner  │ │ /safety_monitor   │ │ /led_interface       │
│ (20Hz, RT core)  │ │ (10Hz)            │ │ (1Hz) 状态显示       │
│ 订阅: 子任务      │ │ 订阅: 关节状态    │ │                      │
│ 发布: 目标角度    │ │ 发布: 安全等级    │ └──────────────────────┘
└────────┬─────────┘ └─────────┬─────────┘
         │ Target Joints       │ Safety Level
         ▼                     ▼
┌──────────────────────────────────────────┐
│  /joint_bridge (CAN 总线网关)            │
│  → CAN 帧 → 各关节 MCU (物理执行)        │
│  ← SPI/编码器 ← 实际关节角度反馈          │
└──────────────────────────────────────────┘
```

### 2.2 ROS2 实时配置

```yaml
# /etc/ros2/robot_realtime.yaml

# 实时调度参数 (每个节点可以独立设置)
task_planner:
  cpu_affinity: [0, 1]           # 运行在 CPU 0-1 (非实时核心)
  scheduler: "SCHED_OTHER"       # 普通调度 (不要求实时)
  priority: 0
  
motion_planner:
  cpu_affinity: [4, 5]           # 运行在 CPU 4-5 (隔离核心)
  scheduler: "SCHED_FIFO"        # 先进先出, 除非高优先级抢占
  priority: 80                   # 1-99, 越高越优先
  
safety_monitor:
  cpu_affinity: [6, 7]           # 最高安全核心
  scheduler: "SCHED_FIFO"
  priority: 99                   # 最高优先级 (超过 motion_planner)
  
joint_bridge:
  cpu_affinity: [4, 5]
  scheduler: "SCHED_FIFO"
  priority: 85
```

```bash
# 启动时设置 ROS2 executor 实时性
# /etc/systemd/system/robot-rt.service

[Unit]
Description=EmbodiedAI Robot RT Services
After=network.target

[Service]
Type=simple
User=robot
ExecStartPre=/usr/bin/set_cpu_governor.sh performance
ExecStartPre=/usr/bin/irq_affinity.sh
ExecStart=/opt/ros2/launch/robot.launch.py
LimitRTPRIO=99               # 允许实时优先级
LimitMEMLOCK=infinity        # 允许锁定内存
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=50
MemoryLimit=6G

[Install]
WantedBy=multi-user.target
```

---

## 3. 动作规划引擎 (中层)

### 3.1 动作空间定义与输出

```
机器人全身动作 (Action Space):

动作维度              类型       范围            更新频率
─────────────────────────────────────────────────────────
12 × 关节角度目标       float32   [-π, π] rad     20Hz
12 × 关节速度限制       float32   [0, 20] rad/s   5Hz (可调)
12 × 力矩限制           float32   [0, peak] Nm    5Hz (可调)

2 × 手指开合度           float    [0, 1]          20Hz

移动底盘 (可选):
  3 × [x, y, θ]        float    [0,∞)           20Hz

动作执行:
  - 每 50ms, motion_planner 发布新的 12 维目标角度
  - 关节 MCU 接收 CAN 帧 → 内部插值到 1kHz → FOC 跟踪
```

### 3.2 运动规划算法选择

```
┌─────────────┬──────────────────────┬────────────────────┬────────────┐
│ 算法         │ 适用                  │ 实时性              │ 本计划选用 │
├─────────────┼──────────────────────┼────────────────────┼────────────┤
│ RL 策略      │ 传感器丰富的操作      │ 推理 ~15ms         │ ★ 主要    │
│ (RL policy) │ (抓取/放置/擦拭)      │ GPU 加速           │            │
├─────────────┼──────────────────────┼────────────────────┼────────────┤
│ OMPL        │ 臂运动 (避障)         │ 规划 50ms-5s       │ 备选      │
│             │ (点到点运动)          │ 不保证实时          │ 导航场景    │
├─────────────┼──────────────────────┼────────────────────┼────────────�│
│ Diffusion   │ 复杂序列动作          │ 推理 20-100ms      │ 未来      │
│ Policy       │ (学习人类创造的运动)   │ 采样过程           │ Phase 3+   │
├─────────────┼──────────────────────┼────────────────────┼────────────┤
│ MPC         │ 脚式机器人平衡        │ 求解 1-5ms         │ 未来      │
│ (模型预测)   │                     │ 依赖动力学模型      │ (人形)     │
├─────────────┼──────────────────────┼────────────────────┼────────────┤
│ IK 求解      │ 末端执行器路径        │ 解析解 < 1ms       │ 辅助      │
│             │                     │ 数值解 5ms         │            │
└─────────────┴──────────────────────┴────────────────────┴────────────┘

推荐模式:
  - 日常操作: RL 策略 (端到端, 对传感器噪声鲁棒)
  - 灵巧手指操作: Diffusion Policy (动作自然)
  - 导航+抓取结合: OMPL 粗规划 + RL 精调
```

---

## 4. 任务状态机与动作执行引擎

### 4.1 SMACH 状态机 (ROS2 中的行为树)

```
任务: "去厨房拿一个杯子"

状态机 (SMACH 或 BehaviorTree.CPP):

              ┌──────────────────────┐
              │      INITIALIZE       │
              │  - 自检各传感器       │
              │  - 加载语义地图       │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │     NAVIGATE         │
              │  - 目标: 厨房坐标     │
              │  - 避障路径规划       │
              └──────────┬───────────┘
                         │ 到达
                         ▼
              ┌──────────────────────┐
              │  DETECT_OBJECT       │
              │  - VLM 扫描桌面      │
              │  - 语义搜索 "杯子"    │
              │  - 获取杯子位姿       │
              └──────────┬───────────┘
                         │ 定位成功
                         ▼
              ┌──────────────────────┐
              │  PREGRASP_APPROACH   │
              │  - 预抓取位姿         │
              │  - RL 策略: 靠近     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  GRASP               │
              │  - RL 策略: 抓取      │
              │  - 触觉检测: 稳定?    │
              │  - 力反馈环           │
              └──────────┬───────────┘
                         │ 抓取成功
                         ▼
              ┌──────────────────────┐
              │  POSTGRASP_RETRACT   │
              │  - 后撤到安全位姿     │
              │  - 触觉监控: 滑脱?    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  NAVIGATE_BACK       │
              │  - 导航回用户位置     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  HANDOVER            │
              │  - 力控制递交杯子     │
              │  - 触觉: 接稳了吗?    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │      SUCCEEDED       │
              └──────────────────────┘

失败处理:
  任何状态失败 → RETRY (重试 N 次)
  RETRY 用尽 → RECOVER (回到安全状态)
  RECOVER 失败 → ABORT (报告用户, 需人工干预)

每个状态内部都是 RL/控制的循环:
  GRASP 内部 (200 步循环):
    Repeat until grasp_stable or timeout:
      1. 摄像头+触觉观测 → VLM 编码 → RL 策略 → 目标角度
      2. CAN 发送目标角度 → FOC 执行
      3. 触觉反馈 → 判断 stable? slip? contact?
      4. 调整策略: stable → 增大握力; slip → 快速增大握力; timeout → 重试
```

---

## 5. 可解释性 (Explainable AI for Robot)

### 5.1 机器人为什么需要可解释性

```
黑盒机器人: 
  "机器人伸出手, 抓住了杯子, 递给我"
  → 用户: 它为什么抓了这个杯子而不是那个?
  → 机器人: 无可奉告 (神经网络黑盒)

可解释机器人:
  "机器人伸出手, 因为它检测到玻璃杯在 30cm 右前方, 
   与您'拿水罐'指令匹配, 使用 5N 握力 (防止摔碎玻璃杯),
   触觉检测到稳定握持, 已到达用户面前"
  → 用户: 理解了, 谢谢。
```

### 5.2 可解释性层级

```
Level 1: 决策日志 (基础)
  每个决策自动记录:
  {
    "t": "2026-09-19T14:23:45.123Z",
    "decision": "GRASP",
    "trigger": "杯子位姿稳定 > 1s",
    "reason_weights": {
      "vlm_match_score": 0.92,
      "distance_to_target": 8.2,
      "tactile_feedback": "stable",
      "action_probability": 0.78,
      "grasp_heuristic": "power grasp"
    },
    "submitted_by": "RL_PPO_v3.4",
    "safety_approved": true
  }

Level 2: 注意力可视化 (中级)
  - 显示模型在图像中的 "注视热力图" (Grad-CAM)
  - 显示模型关注了哪些传感器 (注意力权重 bar)

Level 3: 自然语言解释 (高级)
  - LLM 根据决策日志生成自然语言报告
  - "我选择了抓取红色的杯子, 因为用户指着这个杯子, 
     并且我上次抓取它的成功率是 95%"

Level 4: 意图推测 + 修正协作 (最高级)
  - 机器人主动确认高风险决策: "我要进入碰撞风险区, 是否同意?"
  - 用户实时干预: 语音说 "不对, 抓蓝色的" → 状态机切换到蓝色杯子
```

### 5.3 实现: ROS2 决策日志系统

```python
# decision_logger.py — ROS2 节点

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_msgs.msg import String, Float32MultiArray
from embodied_msgs.msg import TactileState, RobotState, DecisionLog
import json
import time
from datetime import datetime

class DecisionLogger(Node):
    def __init__(self):
        super().__init__('decision_logger')
        
        # 订阅所有关键话题
        self.create_subscription(RobotState, '/robot/state', self.state_cb, 10)
        self.create_subscription(Pose, '/perception/target_pose', self.target_cb, 10)
        self.create_subscription(TactileState, '/tactile/processed', self.tactile_cb, 10)
        self.create_subscription(Float32MultiArray, '/rl/action_probs', self.probs_cb, 10)
        self.create_subscription(String, '/vlm/explanation', self.vlm_exp_cb, 10)
        self.create_subscription(String, '/state_machine/state', self.sm_cb, 10)
        
        # 决策日志发布
        self.log_pub = self.create_publisher(DecisionLog, '/decision_log', 10)
        
        # 持久化存储 (SQLite, 板上)
        self.db = sqlite3.connect('/var/log/robot_decisions.db')
        self.init_db()
        
        # 当前决策上下文
        self.current_context = {}
        
    def init_db(self):
        self.db.execute('''CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            decision_type TEXT,
            state_machine_state TEXT,
            target_object TEXT,
            action_details TEXT,
            model_confidence REAL,
            safety_level INTEGER,
            explanation TEXT,
            success INTEGER DEFAULT NULL
        )''')
        
    def state_cb(self, msg: RobotState):
        self.current_context['robot_state'] = {
            'joint_positions': msg.joint_positions,
            'joint_velocities': msg.joint_velocities,
            'eef_pose': msg.eef_pose,
            'mode': msg.operating_mode
        }
    
    def publish_decision(self, decision_type, details, confidence):
        """当有重要决策时调用"""
        log = DecisionLog()
        log.timestamp = datetime.now().isoformat()
        log.decision_type = decision_type
        log.state = self.current_context.get('sm_state', 'UNKNOWN')
        log.target = self.current_context.get('target_object', '')
        log.model_confidence = confidence
        log.safety_level = self.assess_safety(details)
        log.explanation = self.generate_explanation(details)
        
        # 发布到 ROS2
        self.log_pub.publish(log)
        
        # 持久化
        self.db.execute(
            'INSERT INTO decisions VALUES (NULL,?,?,?,?,?,?,?,?)',
            (log.timestamp, log.decision_type, log.state, log.target,
             json.dumps(details), confidence, log.safety_level, log.explanation)
        )
        self.db.commit()
        
        # 日志输出
        self.get_logger().info(f'[DECISION] {decision_type}: {log.explanation}')
    
    def generate_explanation(self, details) -> str:
        """生成人类可读的决策解释"""
        parts = []
        
        if 'vlm_score' in details:
            parts.append(f'目标匹配度 {details["vlm_score"]:.0%}')
        if 'tactile_state' in details:
            parts.append(f'触觉状态: {details["tactile_state"]}')
        if 'distance' in details:
            parts.append(f'目标距离 {details["distance"]:.1f}cm')
        if 'action_probs' in details:
            top_action = max(details['action_probs'], key=details['action_probs'].get)
            parts.append(f'首选动作 "{top_action}" 概率 {details["action_probs"][top_action]:.0%}')
        
        return '; '.join(parts) if parts else 'no explanation'
    
    def query_decision_history(self, last_n=10):
        """查询最近 N 条决策 (调试/分析用)"""
        cur = self.db.execute(
            'SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?', (last_n,)
        )
        return cur.fetchall()
```

### 5.4 注意力可视化 (Grad-CAM for Robotics)

```python
# visualization.py

class RobotAttentionVisualizer:
    """
    生成:
    1. 摄像头图像的注意力热力图 (Grad-CAM)
    2. 传感器柱状图 (哪个传感器影响了决策)
    3. 决策树路径 (状态机轨迹)
    """
    
    def __init__(self, model):
        self.model = model  # VLA model with hook support
        self.attention_weights = []
        
    def register_hooks(self):
        """在模型关键层注册前向钩子"""
        def attention_hook(module, input, output):
            # output shape: [batch, num_heads, seq, seq]
            weights = output[1] if isinstance(output, tuple) else output
            self.attention_weights.append(weights.detach())
        
        # 在最后一层 Transformer 注册
        self.model.fusion_transformer.layers[-1].attention.register_forward_hook(
            attention_hook
        )
    
    def generate_heatmap(self, image, save_path=None):
        """生成 Grad-CAM 热力图叠加到 RGB 图"""
        # 1. 前向传播
        logits = self.model.visual_encoder(image.unsqueeze(0))
        
        # 2. 取目标类别的梯度
        target_idx = logits.argmax()
        logits[0, target_idx].backward()
        
        # 3. 加权特征图
        gradients = self.model.visual_encoder.get_activations_gradient()
        activations = self.model.visual_encoder.get_activations(image.unsqueeze(0))
        
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # 4. 叠加到原图
        result = 0.5 * image + 0.5 * cam
        
        if save_path:
            cv2.imwrite(save_path, result)
        
        return result
    
    def generate_decision_report(self, session_log):
        """生成一份决策分析报告"""
        import pandas as pd
        
        df = pd.DataFrame(session_log)
        
        report = f"""
        === 具身智能体决策分析报告 ({df.iloc[0]['timestamp'][:10]}) ===
        
        总决策数: {len(df)}
        任务成功率: {df['success'].mean():.1%}
        平均决策置信度: {df['model_confidence'].mean():.2f}
        
        状态机统计:
        {df['state_machine_state'].value_counts().to_string()}
        
        触觉状态分布:
        {df['tactile_state'].value_counts().to_string()}
        
        失败案例分析:
        {df[df['success']==0][['timestamp','decision_type','explanation']].to_string()}
        
        高频决策模式 (可能优化):
        """
        return report
```

---

## 6. 完整启动流程 (从通电到所有服务运行)

```
上电 → BootROM → U-Boot SPL → U-Boot → Kernel → systemd → Robot Services

systemd 服务启动顺序 (After= 依赖链):

After=network.target
└─ robot-rt.target (After=robot-rt-pre.target)
   └─ robot-rt-pre.target
      ├─ /etc/init.d/set_cpu_governor.sh     (调频策略)
      ├─ /etc/init.d/irq_affinity.sh        (中断路由)
      ├─ /etc/init.d/pin_ctrl.sh             (引脚复用)
      └─ /etc/init.d/can_init.sh            (CAN 波特率)

After=robot-rt.target
├─ robot-kernel-modules.service (加载驱动)
│   ├─ robot_joint_can.ko      (CAN 电机驱动)
│   ├─ robot_safety_gpio.ko   (安全监控)
│   ├─ robot_imu_i2c.ko       (IMU 驱动)
│   └─ robot_encoder_spi.ko   (编码器 SPI)

After=robot-kernel-modules.service
├─ ros2-daemon.service          (ROS2 DDS 发现服务)
└─ robot-state-publisher.service (机器人 URDF 发布)

After=ros2-daemon.service
├─ task-planner.service         (VLA 推理)
├─ motion-planner.service       (RL 策略 / RT 核心)
├├ safety-monitor.service       (安全监控 / RT 最高优先级)
├├ joint-bridge.service         (CAN 总线网关)
├├ tactile-processor.service    (触觉数据处理)
├├ perception-pipeline.service  (视觉 / 触觉融合)
├├ decision-logger.service      (决策记录)
├├ state-machine.service        (任务状态机)
└├ gui-interface.service        (交互界面)
```

---

## 7. 调试与诊断

```bash
# 1. 查看所有 ROS2 节点和话题
ros2 node list
ros2 topic list
ros2 topic echo /decision_log
ros2 topic hz /tactile/processed        # 检查频率

# 2. 检查实时调度是否生效
chrt -p $(pgrep motion_planner)
# 应显示: scheduling policy: SCHED_FIFO, priority: 80

# 3. 检查 CPU 隔离是否生效
cat /sys/devices/system/cpu/isolated
# 应显示: 4-5,6-7 (隔离的核心)

# 4. 检查中断路由
cat /proc/interrupts | grep -i spi
# 确认 SPI 中断只落在 CPU 4-5

# 5. cyclictest 实时性能测试
cyclictest -m -S -p 90 -i 200 -h 400 -l 100000
# -S: clock_nanosleep + 多线程
# 结果: min=5us, avg=12us, max=45us
# 要求: max < 100us

# 6. 决策日志查询
sqlite3 /var/log/robot_decisions.db "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 20;"
sqlite3 /var/log/robot_decisions.db "SELECT AVG(success), decision_type FROM decisions GROUP BY decision_type;"

# 7. 检查 CAN 总线状态
ip -details -statistics link show can0
# 关注: state, tx_packets, rx_packets, errors
```
