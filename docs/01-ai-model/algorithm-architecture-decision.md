## 🤖 Assistant


# 20Hz家庭服务场景的算法架构决策

## 快速建议：**分层混合架构 = MCTS(高层) + RL(中层) + Diffusion(低层)**

核心论点：**单一算法无法同时满足实时性、泛化性、安全性**。20Hz控制频率下，必须分层解耦。

---

## 一、三种算法的本质对比

### 算法特性矩阵

```python
algorithm_comparison = {
    'MCTS': {
        '计算特性': {
            'latency': '100ms - 10s',  # 每次决策需要数千次模拟
            'throughput': '0.1-10 Hz',  # 无法满足20Hz
            'memory': 'O(b^d)',  # b=分支因子, d=深度
        },
        '适用层级': '离散符号决策',
        '示例': '选择"抓杯子" vs "擦桌子"',
        '优势': '零样本泛化、可解释、硬约束保证',
        '劣势': '连续动作空间爆炸、无法学习动力学',
        '最佳场景': '长期规划（>10步）、新任务零样本',
    },
    
    'RL (PPO/SAC)': {
        '计算特性': {
            'latency': '1-10ms',  # 前向推理快
            'throughput': '100-1000 Hz',  # 可以满足20Hz
            'memory': 'O(1)',  # 神经网络固定大小
        },
        '适用层级': '闭环反应式控制',
        '示例': '根据当前触觉调整抓力',
        '优势': '学习复杂动力学、实时反馈、样本复用',
        '劣势': '需要大量交互数据、泛化差、不可解释',
        '最佳场景': '单一技能优化（如稳定抓取）',
    },
    
    'Diffusion Policy': {
        '计算特性': {
            'latency': '50-200ms',  # 50步去噪
            'throughput': '5-20 Hz',  # 勉强满足20Hz
            'memory': 'O(T)',  # T=action_horizon
        },
        '适用层级': '时序轨迹生成',
        '示例': '生成10步的抓取轨迹',
        '优势': '多模态、平滑轨迹、模仿学习高效',
        '劣势': '开环执行、闭环反馈滞后、长期规划弱',
        '最佳场景': '短期运动原语（2-10步）',
    },
}
```

### 20Hz控制的严格约束

```python
real_time_analysis = {
    '控制周期': '1/20 = 50ms',
    
    '时间分配（必须满足）': {
        'perception': '15ms',      # 触觉读取、视觉处理
        'planning': '25ms',        # ⚠️ 核心瓶颈
        'control': '5ms',          # PID/阻抗控制
        'communication': '5ms',    # 发送电机命令
    },
    
    '各算法能否满足25ms规划预算？': {
        'MCTS': '❌ 100ms+ (超时4倍)',
        'RL': '✅ 1-10ms (充裕)',
        'Diffusion': '⚠️ 50-200ms (超时2-8倍)',
    },
}

# 结论：在20Hz闭环控制中，只有RL能直接使用
# 其他算法必须通过架构设计规避实时性问题
```

---

## 二、推荐架构：三层分层控制

### 架构设计哲学

```python
"""
核心思想：将不同时间尺度的决策分离
- 长期规划（秒级）→ MCTS
- 中期协调（0.1秒级）→ RL
- 短期执行（毫秒级）→ Diffusion/PID

类比：人类驾驶汽车
- 导航（去哪）：离线规划，不需要实时
- 变道超车（如何）：反应式决策，实时调整
- 转向刹车（执行）：肌肉记忆，极快响应
"""

class HierarchicalController:
    """
    20Hz家庭服务机器人控制架构
    """
    def __init__(self):
        # Layer 3: 任务规划器 (1-10 Hz)
        self.task_planner = MCTSPlanner(
            update_freq=1.0,  # 每秒重规划
            horizon=10,       # 规划10步
        )
        
        # Layer 2: 技能协调器 (20 Hz) ⚡ 关键层
        self.skill_coordinator = RLCoordinator(
            policy_net=PPO_Policy(),
            update_freq=20.0,
        )
        
        # Layer 1: 运动原语库 (异步生成)
        self.motion_primitives = {
            'grasp': DiffusionPolicy.load('grasp.pth'),
            'reach': DiffusionPolicy.load('reach.pth'),
            'wipe': DiffusionPolicy.load('wipe.pth'),
        }
        
        # Layer 0: 低层控制 (1000 Hz)
        self.low_level = ImpedanceController()
    
    def control_loop(self):
        """
        主控制循环：20Hz
        """
        rate = Rate(20)  # Hz
        
        while not rospy.is_shutdown():
            t_start = time.time()
            
            # === 1. 感知 (15ms预算) ===
            obs = self._get_observation()  # 5ms
            state = self._estimate_state(obs)  # 10ms
            
            # === 2. 规划 (25ms预算) ===
            action = self._hierarchical_planning(state)  # 详见下文
            
            # === 3. 执行 (5ms预算) ===
            self.robot.apply_action(action)
            
            # === 4. 日志与监控 (5ms预算) ===
            self._log_telemetry(state, action)
            
            # 实时性检查
            elapsed = time.time() - t_start
            if elapsed > 0.050:
                rospy.logwarn(f"Control loop overrun: {elapsed*1000:.1f}ms")
            
            rate.sleep()
    
    def _hierarchical_planning(self, state):
        """
        分层规划：25ms内完成
        """
        # === Layer 3: 任务规划（异步，不阻塞） ===
        if self.task_planner.should_replan():
            # 在后台线程运行MCTS（允许100-1000ms）
            self.task_planner.replan_async(state)
        
        # 获取当前高层目标（非阻塞，1ms）
        current_skill = self.task_planner.get_current_skill()
        
        # === Layer 2: RL协调器（实时，10ms） ===
        # 输入：当前状态 + 高层目标
        # 输出：运动原语选择 + 参数调制
        rl_decision = self.skill_coordinator.forward(
            obs=state,
            goal=current_skill,
        )
        # rl_decision = {
        #     'primitive': 'grasp',
        #     'modulation': [0.8, 0.0, 0.2],  # 速度缩放、力度、方向微调
        # }
        
        # === Layer 1: 运动原语执行（缓存，5ms） ===
        # Diffusion Policy在后台异步生成轨迹
        primitive = self.motion_primitives[rl_decision['primitive']]
        
        if not primitive.has_cached_trajectory():
            # 首次调用：触发后台生成（50-200ms）
            # 当前帧返回安全的默认动作
            return self._get_safe_default_action(state)
        
        # 从缓存读取下一步动作（<1ms）
        action = primitive.get_next_action()
        
        # RL调制（微调动作，1ms）
        action = self._modulate_action(action, rl_decision['modulation'])
        
        return action
```

---

## 三、各层详细设计

### Layer 3: MCTS任务规划器（1Hz，异步）

```python
class AsyncMCTSPlanner:
    """
    后台运行MCTS，不阻塞实时控制
    """
    def __init__(self):
        self.mcts = SafeHouseholdMCTS(...)
        self.planner_thread = None
        self.current_plan = deque()  # 当前技能序列
        self.planning = False
    
    def replan_async(self, state):
        """
        在后台线程运行MCTS（允许100-1000ms）
        """
        if self.planning:
            return  # 已经在规划中，跳过
        
        def _planning_worker():
            self.planning = True
            try:
                # 运行MCTS（耗时操作）
                new_plan = self.mcts.search(
                    init_state=state,
                    time_budget=0.5,  # 500ms预算
                )
                
                # 原子更新计划
                with self.lock:
                    self.current_plan = deque(new_plan)
            finally:
                self.planning = False
        
        self.planner_thread = threading.Thread(target=_planning_worker)
        self.planner_thread.start()
    
    def get_current_skill(self):
        """
        获取当前应执行的技能（非阻塞，<1ms）
        """
        with self.lock:
            if len(self.current_plan) == 0:
                return 'idle'
            return self.current_plan[0]
    
    def mark_skill_completed(self):
        """
        技能完成后调用，推进到下一个
        """
        with self.lock:
            if len(self.current_plan) > 0:
                completed = self.current_plan.popleft()
                rospy.loginfo(f"Skill '{completed}' completed")
    
    def should_replan(self):
        """
        何时触发重规划？
        """
        # 条件1：计划快用完了
        if len(self.current_plan) <= 2:
            return True
        
        # 条件2：环境变化（如人类干预）
        if self._detect_environment_change():
            return True
        
        # 条件3：执行失败
        if self._detect_execution_failure():
            return True
        
        return False

# MCTS的状态空间设计（适配家庭场景）
class HouseholdState:
    """
    MCTS搜索的状态节点
    """
    def __init__(self):
        # 离散化的世界状态
        self.robot_pose = RobotPose()
        self.object_states = {
            'cup': ObjectState(position=[0.3, 0.0, 0.8], 
                               grasped=False),
            'table': ObjectState(clean=False),
        }
        self.human_location = HumanPose()
        self.current_skill = None
    
    def get_actions(self):
        """
        返回可选的技能（离散动作空间）
        """
        actions = []
        
        # 基于当前状态生成候选技能
        if not self.object_states['cup'].grasped:
            actions.append('grasp_cup')
        
        if self.object_states['cup'].grasped:
            actions.extend(['place_cup', 'handover_to_human'])
        
        if not self.object_states['table'].clean:
            actions.append('wipe_table')
        
        # 过滤不可行的（安全检查）
        actions = [a for a in actions if self._is_safe(a)]
        
        return actions
    
    def simulate(self, action):
        """
        快速前向模拟（不需要精确物理）
        """
        next_state = self.copy()
        
        # 简化的状态转移
        if action == 'grasp_cup':
            next_state.object_states['cup'].grasped = True
            next_state.robot_pose = self._compute_grasp_pose()
        
        elif action == 'wipe_table':
            next_state.object_states['table'].clean = True
            next_state.robot_pose = self._compute_wipe_pose()
        
        # 估计执行时间（用于奖励折扣）
        next_state.time += self._estimate_duration(action)
        
        return next_state
    
    def reward(self):
        """
        任务完成度奖励
        """
        reward = 0
        
        # 目标1：桌子干净
        if self.object_states['table'].clean:
            reward += 10
        
        # 目标2：杯子递给人类
        if self._cup_handed_over():
            reward += 20
        
        # 惩罚：时间成本
        reward -= 0.1 * self.time
        
        # 惩罚：接近人类（安全）
        if self._too_close_to_human():
            reward -= 50
        
        return reward
```

### Layer 2: RL技能协调器（20Hz，实时核心）

```python
class RLSkillCoordinator:
    """
    实时RL策略：根据当前状态微调运动原语
    
    训练方法：PPO（更稳定）或SAC（探索性更强）
    """
    def __init__(self):
        # 策略网络（小型，推理快）
        self.policy = PPOPolicy(
            obs_dim=128,   # 编码后的观测
            act_dim=16,    # 动作调制参数
        )
        
        # 状态编码器
        self.encoder = MultiModalEncoder(
            visual_encoder=ResNet18(),
            tactile_encoder=TactileNet(),
            proprio_dim=30,
        )
        
        # 推理优化
        self.policy.eval()
        self.policy = torch.jit.script(self.policy)  # TorchScript加速
    
    @torch.no_grad()
    def forward(self, obs, goal):
        """
        前向推理：10ms预算
        """
        # 1. 编码观测（5ms）
        visual_feat = self.encoder.visual(obs['rgb'])
        tactile_feat = self.encoder.tactile(obs['tactile'])
        proprio_feat = obs['proprio']
        goal_feat = self._encode_goal(goal)
        
        state_vec = torch.cat([
            visual_feat, tactile_feat, proprio_feat, goal_feat
        ])
        
        # 2. 策略推理（3ms）
        action_logits = self.policy(state_vec)
        
        # 3. 采样动作（1ms）
        action = self._sample_action(action_logits)
        
        return {
            'primitive_id': action[0],      # 选择哪个运动原语
            'speed_scale': action[1],       # 速度调制 [0.5, 2.0]
            'force_modulation': action[2],  # 力度调节 [0.5, 1.5]
            'direction_bias': action[3:6],  # 方向微调 [-0.1, 0.1]
        }
    
    def _sample_action(self, logits):
        """
        采样策略（训练时加噪声，部署时确定性）
        """
        if self.training:
            # 训练：随机采样（探索）
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
        else:
            # 部署：贪心（稳定）
            return torch.argmax(logits, dim=-1)

# PPO策略网络（轻量级）
class PPOPolicy(nn.Module):
    """
    小型MLP：保证<10ms推理
    """
    def __init__(self, obs_dim=128, act_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, act_dim),
        )
    
    def forward(self, obs):
        return self.net(obs)

# 训练流程
class PPOTrainer:
    """
    在仿真中训练RL协调器
    """
    def train(self):
        env = VectorizedEnv(n_envs=1024)  # 并行环境
        
        for epoch in range(1000):
            # 收集轨迹
            trajectories = []
            for _ in range(2048):  # 每epoch 2M步
                obs = env.reset()
                done = False
                
                while not done:
                    # RL选择运动原语+调制
                    action = self.policy(obs)
                    
                    # 环境执行（调用Diffusion生成的轨迹）
                    next_obs, reward, done, info = env.step(action)
                    
                    trajectories.append((obs, action, reward))
                    obs = next_obs
            
            # PPO更新
            self._ppo_update(trajectories)
            
            # 真机评估
            if epoch % 50 == 0:
                real_success_rate = evaluate_on_real_robot(self.policy)
                print(f"Epoch {epoch}: Real success = {real_success_rate:.1%}")

# RL的观测空间设计
observation_space = {
    # 视觉
    'rgb': (3, 224, 224),           # 场景理解
    'depth': (1, 224, 224),         # 几何信息
    
    # 触觉（关键！）
    'tactile': (4, 64, 64),         # 4个手指的触觉图像
    'contact_force': (4,),          # 法向力
    'slip_velocity': (4, 2),        # 切向滑移
    
    # 本体感觉
    'joint_pos': (12,),
    'joint_vel': (12,),
    'ee_pose': (7,),                # 位置+四元数
    
    # 任务信息
    'goal_embedding': (64,),        # 来自MCTS的高层目标
    'time_remaining': (1,),         # 技能剩余时间
}

# RL的动作空间设计
action_space = {
    # 离散选择（8个运动原语）
    'primitive_id': Discrete(8),
    
    # 连续调制参数
    'speed': Box(0.5, 2.0),         # 加速/减速
    'force': Box(0.5, 1.5),         # 轻抓/重抓
    'direction': Box(-0.1, 0.1, shape=(3,)),  # 轨迹微调
}

# RL的奖励函数设计
def compute_reward(state, action, next_state):
    """
    密集奖励 + 稀疏奖励
    """
    reward = 0
    
    # === 密集奖励（引导学习） ===
    # 1. 任务进度
    progress = next_state.task_progress - state.task_progress
    reward += 10 * progress
    
    # 2. 触觉反馈（防滑）
    if next_state.slip_detected:
        reward -= 1.0  # 惩罚滑移
    
    # 3. 力控制（不要太用力）
    excessive_force = np.sum(np.maximum(0, next_state.contact_force - 10))
    reward -= 0.1 * excessive_force
    
    # 4. 平滑性（避免抖动）
    jerk = np.linalg.norm(action.acceleration)
    reward -= 0.01 * jerk
    
    # === 稀疏奖励（任务完成） ===
    if next_state.skill_completed:
        reward += 50
    
    # === 安全惩罚（硬约束） ===
    if next_state.human_collision:
        reward -= 1000  # 严厉惩罚
    
    return reward
```

### Layer 1: Diffusion运动原语（异步生成）

```python
class AsyncDiffusionPrimitive:
    """
    后台生成轨迹，主线程无阻塞读取
    """
    def __init__(self, model_path):
        self.model = DiffusionPolicy.load(model_path)
        self.model.eval()
        
        # 轨迹缓存（双缓冲）
        self.trajectory_buffer_A = deque()
        self.trajectory_buffer_B = deque()
        self.active_buffer = 'A'
        
        # 生成线程
        self.generator_thread = threading.Thread(
            target=self._generation_loop,
            daemon=True,
        )
        self.generator_thread.start()
    
    def _generation_loop(self):
        """
        后台持续生成轨迹（50-200ms/次）
        """
        while True:
            # 等待触发信号
            self.generation_event.wait()
            
            # 获取当前观测
            obs = self.latest_obs
            
            # Diffusion生成（耗时操作）
            trajectory = self.model.generate(
                obs=obs,
                n_steps=50,       # 去噪步数
                horizon=10,       # 生成10步轨迹
            )
            
            # 写入非活跃缓冲区
            inactive_buffer = 'B' if self.active_buffer == 'A' else 'A'
            buffer = (self.trajectory_buffer_A if inactive_buffer == 'A' 
                     else self.trajectory_buffer_B)
            
            with self.lock:
                buffer.clear()
                buffer.extend(trajectory)
                
                # 切换缓冲区（原子操作）
                self.active_buffer = inactive_buffer
            
            self.generation_event.clear()
    
    def get_next_action(self):
        """
        主线程调用：从缓存读取（<1ms）
        """
        with self.lock:
            buffer = (self.trajectory_buffer_A if self.active_buffer == 'A'
                     else self.trajectory_buffer_B)
            
            if len(buffer) == 0:
                # 缓存用完，返回安全停止动作
                return self._get_safe_stop_action()
            
            return buffer.popleft()
    
    def update_observation(self, obs):
        """
        更新观测，触发新轨迹生成
        """
        self.latest_obs = obs
        
        # 如果缓存快用完（<3步），触发生成
        active_buffer = (self.trajectory_buffer_A if self.active_buffer == 'A'
                        else self.trajectory_buffer_B)
        
        if len(active_buffer) < 3:
            self.generation_event.set()
    
    def has_cached_trajectory(self):
        """
        检
