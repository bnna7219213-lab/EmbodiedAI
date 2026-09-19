# 20Hz 家庭服务场景的算法架构决策

> 本文档是对外呼应用户提问 q4 的回应: 在 20Hz 实时控制下, 选择什么算法架构?
>
> 给出分层决策 + 完整的 Python 实现骨架 (与攀岩篇 MCTS + 强化学习篇 Actor-Critic 对齐)。

---

## 1. 架构选择: 为什么是 "三层 + 安全层"

| 候选架构 | 实时性 | 安全性 | 可解释 | 样本效率 | 结论 |
|---|---|---|---|---|---|
| 纯端到端 RL (PPO) | OK (15ms) | 无保障 | 无 | 低 | ❌ 不安全 |
| 纯 MCTS | 太慢 (100ms+) | 有 (可行域搜索) | 树可查 | 仅需动力学 | ❌ 不实时 |
| Diffusion Policy | 推理慢 (50-200ms) | 无保障 | 无 | 中等 | ❌ 不实时 |
| **LLM + MCTS + RL + FOC** | OK (15ms+1ms+... ) | FOC 自动关断 (1μs) | 每层可查 | 中等偏高 | ✓ 选择 |

总架构:

```
┌─────────────────────────────────────────────────────────────┐
│ 任务层 (1Hz): LLM/VLA                                       │
│   "去厨房拿一个杯子"                                         │
│   → 子任务链: [定位杯子 → 导航到桌旁 → 抓取 → 返回 → 递给]    │
│   → 用 LLM 的自然语言推理 (可解释)                           │
│   → 无实时约束, 秒级响应即可                                  │
└────────────────────────┬────────────────────────────────────┘
                         │ TaskGraph (JSON)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 序列层 (2-10Hz): MCTS 在语义世界中搜索                        │
│   "移动到桌旁" → 哪些路径无碰撞?                              │
│   "抓取" → 力闭合验证 + 接触点优化                            │
│   → 用学习到的 Value Network (Actor-Critic 的 Critic 部分)    │
│    替代手动启发, 加速 MCTS                                    │
│   → 输出: 接触序列 + 安全空间边界                              │
└────────────────────────┬────────────────────────────────────┘
                         │ ContactSequence
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 执行层 (20Hz): RL Actor (PPO/SAC)                            │
│   "当前步的关节角度是多少?"                                   │
│   → 输入: [观测 + 接触序列的当前子目标]                        │
│   → 输出: 12 维关节角度目标                                   │
│   → 用 SAC (最大熵, 高探索, 适合家庭新场景)                   │
└────────────────────────┬────────────────────────────────────┘
                         │ TargetJointAngles (CAN-FD)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 反射层 (10kHz): FOC + 自动关断硬件                             │
│   "电机应转多少?"                                            │
│   → PID 位置环 + PID 电流环 + SVPWM                          │
│   → 独立看门狗 MCU (STM32G4)                                  │
│   → 自动关断: 过流 <1μs / 过温 <10ms / 通信中断 <50ms         │
│   → 硬件级, 不依赖任何软件                                    │
└─────────────────────────────────────────────────────────────┘
```

## 2. 关键 Python 实现

```python
# three_layer_controller.py

class TaskLayer:
    """1Hz: 任务理解与分解"""
    def __init__(self, vlm_model_path):
        self.vla = torch.jit.load(vlm_model_path)
    
    def plan(self, instruction: str, scene_description: dict) -> list:
        """
        instruction: "去厨房拿一个杯子"
        scene_description: 当前场景的物体检测 (VLM 生成的 JSON)
        Returns: 子任务列表
        """
        # VLM 推理 (GPU, ~500ms-2s)
        task_graph = self.vla.plan(instruction, scene_description)
        # 返回: [{"action": "navigate", "target": "kitchen_table"}, ...]
        return task_graph

class SequentialLayer:
    """5Hz: MCTS 在接触空间搜索"""
    def __init__(self, value_network, dynamics_model):
        self.critic = value_network   # SAC 的 Critic, 重用
        self.dynamics = dynamics_model  # 从仿真学习
    
    def search(self, subtask: dict, obs: dict, n_iter=200) -> dict:
        """MCTS 搜索最优接触序列"""
        root = ContactState(subtask, obs)
        
        for _ in range(n_iter):
            node = self._select(root)
            child = self._expand(node)
            reward = self._simulate(child, self.dynamics)
            self._backpropagate(child, reward)
        
        best_contact_sequence = self._best_sequence(root)
        return best_contact_sequence

class ExecutionLayer:
    """20Hz: SAC Actor 输出关节角度"""
    def __init__(self, sac_actor_path, fusion_network_path):
        self.actor = torch.jit.load(sac_actor_path)
        self.fusion = MultiModalFusionPolicy()
        self.fusion.load_state_dict(torch.load(fusion_network_path))
        self.fusion.eval()
    
    def act(self, obs: dict, contact_goal: dict) -> np.ndarray:
        """
        输入: obs (视觉/触觉/本体) + contact_goal (来自 MCTS)
        输出: 12 维关节角度, 范围 [-π, π]
        """
        # 1. 编码观测 (GPU)
        tactile_feat = self.fusion.tactile_encoder(obs['tactile_seq'])
        visual_feat = self.fusion.visual_encoder(obs['visual'])
        proprio_feat = self.fusion.proprio_encoder(obs['proprio'])
        
        # 2. 融合
        state_feat = self.fusion.fusion_attn(
            tactile_feat, visual_feat, proprio_feat
        )
        
        # 3. SAC Actor (仅推理)
        with torch.no_grad():
            action_mean, _ = self.actor(state_feat)
        
        return action_main.cpu().numpy()

class SafetyLayer:
    """硬件级 (FOC + 自动关断): 不在这里运行"""
    # 通过 CAN-FD 自动执行, 由 STM32G4 独立 MCU 处理
    pass


class RobotMainController:
    """主控制器, 严格 20Hz 周期"""
    
    def __init__(self):
        self.task_layer = TaskLayer("vla_plan_v2.pt")
        self.seq_layer = SequentialLayer(
            value_network="sac_critic.pt",
            dynamics_model="learned_dynamics.onnx"
        )
        self.exec_layer = ExecutionLayer(
            sac_actor_path="sac_actor_tactile_v2.pt",
            fusion_network_path="tactile_fusion_v2.pt"
        )
        
        # 通信
        self.can = CANInterface('can0', bitrate=1e6)
        self.tactile_reader = TactileReaderI2C(bus=1)
        self.vision_reader = VisionReaderCSI(channels=[0, 1])
        
    def run(self):
        next_time = time.monotonic()
        
        # 初始任务规划
        current_task_graph = None
        current_subtask_idx = 0
        contact_sequence = None
        
        while True:
            t_start = time.monotonic()
            
            # ===== 传感器采集 =====
            obs = {
                'tactile': self.tactile_reader.read_latest(),
                'visual': self.vision_reader.capture(),
                'proprio': self.can.read_joint_states(),
            }
            
            # ===== 1. 任务层重规划 (仅关键帧) =====
            if self.should_replan_task(obs):
                instruction = self.current_instruction  # VLM 上一次生成的
                scene = self.perception.get_scene(obs)  # VLM 实时场景
                current_task_graph = self.task_layer.plan(instruction, scene)
                current_subtask_idx = 0
            
            # ===== 2. 序列层搜索 (仅状态变化时) =====
            current_subtask = current_task_graph[current_subtask_idx]
            if self.should_replan_sequence(obs, current_subtask):
                contact_sequence = self.seq_layer.search(
                    current_subtask, obs, n_iter=500
                )
            
            # ===== 3. 执行层 =====
            contact_goal = contact_sequence.get_current_contact()
            target_joints = self.exec_layer.act(obs, contact_goal)
            
            # ===== 4. 安全限幅 (CPU, <0.5ms) =====
            target_joints = self.apply_safety_limits(target_joints, obs)
            
            # ===== 5. 发送 CAN =====
            self.can.send_targets(target_joints)
            
            # ===== 6. 决策日志 =====
            self.logger.log({
                't': t_start,
                'task': current_subtask_idx,
                'contact': contact_goal,
                'joint_targets': target_joints.copy(),
                'force': obs['proprio']['torques'],
            })
            
            # ===== 7. 子任务推进 =====
            if self.subtask_done(obs, current_subtask):
                current_subtask_idx += 1
            
            # ===== 8. 精确周期控制 =====
            next_time += 0.05  # 20Hz
            sleep_t = next_time - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                self.handle_overrun()
    
    def apply_safety_limits(self, targets, obs):
        """CPU 上的纯 NumPy 安全限幅 (耗时 < 0.5ms)"""
        # 1. 角度限幅
        targets = np.clip(targets, -np.pi, np.pi)
        
        # 2. 速度限幅 (差分)
        delta = targets - obs['proprio']['positions']
        max_delta = self.max_joint_velocity * 0.05  # 50ms 周期内
        delta = np.clip(delta, -max_delta, max_delta)
        
        # 3. 力限幅 (根据力传感器)
        force_ratio = np.max(np.abs(obs['proprio']['torques'])) / self.max_torque
        if force_ratio > 0.8:
            # 力过大, 停止并维持位置
            delta *= 0.5
        
        return obs['proprio']['positions'] + delta
```

## 3. 关键决策说明

```
1. 为什么用 SAC 而不是 PPO?
   - SAC 最大熵 → 家庭场景多新物体, 探索充分
   - SAC 样本效率高 → 真实数据采集贵
   - SAC 连续动作输出 → 家庭不需要离散选择

2. 为什么用 MCTS 而非纯 RL 做序列?
   - MCTS 有显式约束建模 (碰撞/力闭合/可达)
   - MCTS 用学习到的 Value Net, 不再是盲目搜索
   - MCTS 输出可直接验证安全 (通过 kinematic check)

3. 为什么用 TactileTemporalEncoder (GRU) 而非 Transformer?
   - GRU 参数少 → 适合 Orin NX 实时推理
   - 触觉时序短 (8-12 帧) → GRU 比 Transformer 更高效
   - slip detection 是局部特征 → GRU 足够

4. 自动关断为什么不在此代码中?
   - 独立 MCU (STM32G4), 通过 SPI 命令使能/禁用
   - 紧急关断完全由硬件执行
   - Python 代码只负责"常态控制", MCU 负责"异常关断"
```
