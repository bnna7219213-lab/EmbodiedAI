# 数据采集策略分析：攀岩 vs 家庭服务

> 本文档是对外呼应用户提问 q5 和 q6 的回应: 攀岩篇的数据采集配比如何适配家庭服务？
>
> 给出自有的数据采集策略 + 质量保障体系 + 失败案例重用机制

---

## 1. 攀岩篇的数据策略回顾

```
攀岩篇提出的数据配比:
  human_demo      20%       人类攀岩录像 + 动作捕捉
  sim_random      40%       仿真随机探索
  sim_planner     30%       MCTS 规划的最优轨迹
  real_robot      10%       真机成功/失败案例
```

```
对家庭服务的启示:
  · 攀岩数据相对标准化 (规则岩壁/标准道路)
  · 家庭场景极度多样化 (不同厨房/不同杯子/不同障碍)
  → 策略需要大幅调整权重, 增加 "任务多样性" 维度
```

## 2. 家庭服务数据采集策略

### 2.1 推荐数据配比 (按项目阶段)

```
数据配比 (权重可根据阶段调整):
                Phase 0      Phase 1      Phase 2      Phase 3
                ────────     ────────     ────────     ────────
human_demo      0%           15%          20%          25%
sim_random      60%          35%          20%          10%
sim_planner     40%          40%          30%          15%
real_robot      0%           10%          30%          45%
online_adapt    0%           0%           0%           5%

Phase 0 (暂无硬件): 100% 仿真, 不采集
  仿真数据量: 1000万条轨迹 (GPU 集群, ~3天)
  配: sim_random 60% + sim_planner 40%
  无 human_demo (和 real_robot)

Phase 1 (单臂原型): 仿真 90% + 真机 10%
  仿真: 100万条 (降低,因训练量足)
  遥操作: 100条轨迹 (~10h VR 操作)
  
Phase 2 (双臂): 仿真 50% + 真机 50%
  真机: 5,000条 (采集速度 20条/h × 250h)
  
Phase 3 (全身人形): 仿真 25% + 真机 75% + 在线适应 5%
  真机: 50,000条+ (终身学习)
```

### 2.2 家庭场景特有数据挑战

| 挑战 | 攀岩 vs 家庭 | 缓解策略 |
|---|---|---|
| 场景多样性 | 固定岩壁 (100 种) | 真实场景 >1M (百万家庭,不同厨房) | 必须做域随机化 + 场景泛化 |
| 任务多样性 | "向上攀爬" 单一任务 | "抓取/放置/擦拭/倒水/开门/按钮" 数百类 | 任务指令泛化 (LLM 生成变式) |
| 力分布集中 | 握力范围 ~5-30N (攀岩) | 0.01N (羽毛) 到 30N (扳手) | 对数采样力域 |
| 失败代价 | 掉落 | 打碎杯/玻璃/灼伤 | 逐渐加压 + 安全关断 |
| 物体不可控 | 固定岩点 | 杯子装水/空杯子/倒下的杯子 | 物体状态空间综合采样 |

### 2.3 采集工具链

```
完整采集工具链:
  Phase 0   : PyBullet + TACTO → 自动采集仿真数据
  Phase 1+  : 遥操作 (VR + 力反馈手套)
  Phase 2+  : 多模态人类示范 (Kinect + OpenPose + Electrim)
  Phase 3   : 自动采集 (机器人自探索 + 在线标注)
```

## 3. 遥操作数据采集系统

### 3.1 硬件方案: 低成本 VR 遥操作

```
┌─────────────────────────────────────────────────┐
│              VR Teleoperation System              │
│                                                   │
│  ┌─────────────┐     WiFi/     ┌──────────────┐  │
│  │ Oculus Quest│     USB3      │ Robot Onboard│  │
│  │ (控制器)     │ ←─────────→ │ Jetson       │  │
│  └──────┬──────┘              └──────────────┘  │
│         │                                          │
│  ┌──────▼──────────────────────────────┐          │
│  │ 1. 手部关节跟踪 (Quest 手部追踪)      │          │
│  │ 2. 位置映射: Pose → EE pose          │          │
│  │ 3. 力反馈: 无力反馈 (Phase 0)        │         │
│  │ 4. 触觉反馈: 振动 (Phase 1+)         │          │
│  │ 5. 录像: 同步记录 4 路相机           │          │
│  │ 6. 触觉记录: GelSight 同步录制       │          │
│  │ 7. 语言解说: 语音转文本              │          │
│  └─────────────────────────────────────┘          │
└─────────────────────────────────────────────────┘

成本: ~$300 (Quest 2) + $50 (WiFi 路由器)
采集速度: 15-20 条轨迹/小时 (熟练操作员)
```

### 3.2 数据采集代码

```python
# data_collection_system.py

class TeleopDataCollector:
    """
    完整的数据采集系统
    来源: 遥操作 VR / 示教拖动 / 程序化生成
    """
    
    def __init__(self, robot, sensors, save_path):
        self.robot = robot
        self.sensors = sensors
        self.save_path = save_path
        
        # 当前episode
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'info': [],
        }
        
        # 元数据
        self.metadata = {
            'start_time': time.time(),
            'operator_id': None,
            'task_instruction': None,
            'robot_model': 'embodied_v1',
            'sensors': ['gelsight_LR', 'gelsight_RL', 'cam_top', 'imu', 
                         'force_LR', 'force_rl', 'Proprioception'],
        }
    
    def start_episode(self, instruction: str, operator_id: str):
        """开始一条新轨迹"""
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'info': [],
        }
        self.metadata['task_instruction'] = instruction
        self.metadata['operator_id'] = operator_id
        self.metadata['start_time'] = time.time()
        
        print(f"开始采集: '{instruction}' by {operator_id}")
    
    def record_step(self, obs, action, reward):
        """记录一步数据"""
        self.current_episode['observations'].append(obs)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        # 信息型 (不参与训练)
        self.current_episode['info'].append(self._collect_info())
    
    def end_episode(self, success: bool, notes: str):
        """结束并保存"""
        self.metadata['success'] = success
        self.metadata['notes'] = notes
        self.metadata['n_steps'] = len(self.current_episode['observations'])
        self.metadata['duration'] = time.time() - self.metadata['start_time']
        
        # 立即保存到磁盘
        ep = self.current_episode
        ep['metadata'] = self.metadata.copy()
        
        filename = f"ep_{self.metadata['start_time']:.0f}_{int(success)}.pt"
        save_path = os.path.join(self.save_path, filename)
        torch.save(ep, save_path)
        
        print(f"保存: {filename} ({ep['metadata']['n_steps']} steps, success={success})")
    
    def _collect_info(self):
        """采集辅助信息"""
        return {
            'collision': self.robot.check_collision(),
            'slip_detected': self.robot.last_slip_state,
            'force': self.robot.last_tactile_force,
            'cpu_usage': psutil.cpu_percent(),
            'gpu_usage': self._gpu_usage(),
        }
```

## 4. 数据质量保证

### 4.1 自动化数据质量检查

```python
# data_quality_check.py

class DataQualityChecker:
    """
    对每条采集数据做质量评分, 决定是否纳入训练集
    """
    
    def check_episode(self, episode: dict) -> dict:
        """
        返回质量报告 dict
        scores: 0-1, >0.7 为合格
        """
        issues = []
        scores = {}
        
        # 1. 完整性检查 (obs 中无 NaN)
        has_nan = any(np.isnan(o).any() for episode['observations'])
        scores['completeness'] = 0.0 if has_nan else 1.0
        if has_nan: issues.append("obs 有 NaN")
        
        # 2. 时间一致性 (数据时间戳单调递增)
        timestamps = [info.get('timestamp') for info in episode['info']]
        is_monotonic = all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1))
        scores['temporal'] = 1.0 if is_monotonic else 0.0
        if not is_monotonic: issues.append("时间戳乱序")
        
        # 3. 动作合理性 (关节角速度 < 阈值)
        max_delta = 0
        for i in range(1, len(episode['actions'])):
            delta = np.max(np.abs(episode['actions'][i] - episode['actions'][i-1]))
            max_delta = max(max_delta, delta)
        scores['action_smoothness'] = 1.0 if max_delta < 0.5 else max(0, 1 - (max_delta - 0.5))
        if scores['action_smoothness'] < 0.5: issues.append("动作剧跳")
        
        # 4. 奖励信号合理性 (如失败应由零奖励 jump)
        rewards = episode['rewards']
        if len(rewards) > 10:
            reward_zero_ratio = sum(1 for r in rewards if abs(r) < 1e-6) / len(rewards)
            scores['reward_distribution'] = 1.0 - abs(reward_zero_ratio - 0.3)  # 期望 ~30% 零奖励
        else:
            scores['reward_distribution'] = 0.5
        
        # 5. 成功/失败标注一致性
        metadata = episode.get('metadata', {})
        if metadata.get('success') == True:
            total_reward = sum(rewards)
            if total_reward < 0:
                scores['success_consistency'] = 0.3
                issues.append("标注成功但总奖励为负")
            else:
                scores['success_consistency'] = 1.0
        
        # 6. 传感器数据物理合理 (惯导数据应在重力范围)
        proprio = episode['info'][0].get('Proprioception', {})
        if proprio:
            acc = proprio.get('imu', {}).get('accel', np.zeros(3))
            if np.linalg.norm(acc) > 15 * 9.81:  # >15g 不合理
                scores['sensor_plausibility'] = 0.0
                issues.append(f"IMU 加速不合理: {acc}")
            else:
                scores['sensor_plausibility'] = 1.0
            
            forces = proprio.get('forces', np.zeros(6))
            if np.linalg.norm(forces[:3]) > 1000:  # 1kN 不合理
                scores['sensor_plausibility'] *= 0.5
                issues.append("力传感器读数过大")
        
        # 总分
        overall = np.mean(list(scores.values()))
        
        return {
            'overall_quality': overall,
            'scores': scores,
            'issues': issues,
            'recommend': overall > 0.7,  # >70 纳入训练集
        }
```

### 4.2 数据标注方案 (Phase 2+)

```
自动标注流水线:
  1. 遥操作 → 录制完整轨迹
  2. 自动回放 → 
     a) 成功/失败 → 基于最终状态判断 (物体是否被拾取?)
     b) 语言描述 → 由 VLM 自动生成: "示教动作: 抓取红色杯子"
     c) 技能嵌入 → 自动聚类: 轨迹加入 "grasp_cup" 技能簇
  
  3. 人工审核 → 
     a) 筛选错误标注 (自动标注有 ~5% 误差)
     b) 处理边界情况 (杯子装水? 空杯子?)
  
  4. 增广 →
     a) 时间平移/缩放 (同一动作不同速度)
     b) 空间翻转 (对称任务左右镜像)
     c) 指令改写 (同义替换: "拿" → "取" → "给我")

标注质量监控:
  - 每周抽检 100 条, 错误率追踪
  - 训练 → 评估 → 效果不好 → 回到标注流程
```

## 5. 失败数据重用

### 5.1 失败案例驱动学习

```python
# failure_replay_buffer.py

class FailureReplayBuffer:
    """
    失败案例回放缓冲
    专为家庭服务机器人设计 (失败代价高, 数据宝贵)
    
    存储: (obs, action, failure_type, failure_reason)
    SAC 训练时优先采样失败案例
    """
    
    def __init__(self, capacity=100000, priority_alpha=0.7):
        self.buffer = []
        self.capacity = capacity
        self.priority_alpha = priority_alpha
        self.failure_idx = set()  # 失败样本索引
        
    def add(self, episode: dict):
        """添加一条轨迹"""
        is_success = episode.get('metadata', {}).get('success', False)
        base_priority = 1.0
        
        # 失败案例优先级 ×2 (学习效果如下降)
        if not is_success:
            failure_type = self._classify_failure(episode)
            episode['metadata']['failure_type'] = failure_type
            base_priority = 2.0
            self.failure_idx.add(len(self.buffer))
        
        self.buffer.append({
            'episode': episode,
            'priority': base_priority,
            'n_sampled': 0,
            'last_loss': 1.0,
        })
        
        # 超出容量则淘汰优先级最低的
        if len(self.buffer) > self.capacity:
            self._evict()
    
    def sample(self, batch_size: int) -> list:
        """优先级采样 (失败 ×2, 最近训练 ×1.5)"""
        priorities = np.array([
            b['priority'] * (1.2 if b['last_loss'] > 0.5 else 1.0)
            for b in self.buffer
        ])
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        return [self.buffer[i]['episode'] for i in indices]
    
    def _classify_failure(self, episode: dict) -> str:
        """
        失败类型自动分类
        - slip: 滑脱 (触觉但握力不足)
        - collision: 碰撞 (路径规划失败)
        - timeout: 超时 (策略探索不充分)
        -不稳: 速度过大失稳
        """
        final_action = episode['actions'][-1]
        final_obs = episode['observations'][-1]
        
        if 'slip_detected' in final_obs and final_obs['slip_detected']:
            return 'slip'
        elif 'collision' in episode['info'][-1] and episode['info'][-1]['collision']:
            return 'collision'
        elif len(episode['observations']) > 500:  # 超过500步
            return 'timeout'
        else:
            return 'unknown'
```

## 6. 推荐方案

### 6.1 总结

| 维度 | 攀岩 | 家庭服务 | 适配方式 |
|---|---|---|---|
| 场景固定性 | 高 (50 条道) | 低 (百万家庭) | 增加场景域随机化 |
| 数据采集难度 | 高 (需真人攀岩) | 中 (遥操作 VR/拖拽) | 遥操作为主, 自探索为辅 |
| 失败价值 | 单一 (尝试失败) | 昂贵 (可能损坏家具) | 失败 ×2 采样 |
| 数据多样性 | 有限 | 极大 | 任务指令自动生成 |
| 标注方式 | 自动判定 (跌落?) | 需 VLM 判定 | VLM + 人工抽检 |

### 6.2 评估指标

```
数据引擎效率指标:
  - 数据采集速率: 目标 50 条/h (熟练用户)
  - 合格率: >75% (通过 quality_checker)
  - 数据多样性: 覆盖 20 种物体 × 5 种任务 × 3 种场景
  - 失败样本占比: 20% (不要太多, 模型会 "害怕")

训练效果指标:
  - BC → RL 过渡时 reward 提升 >50%
  - 迁移到未见物体 >30% (攀岩篇对标 57%)
  - 连续运行 1000 次无安全触发 (自动关机正确工作)
```
