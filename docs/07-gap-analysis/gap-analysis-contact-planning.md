# 攀岩接触规划迁移到家庭场景的技术分析

> 本文档是对外呼应用户提问 q3+q4 的回应:
> - 攀岩 MCTS 接触规划能否用于家庭/服务场景?
> - 20Hz 实时控制的算法决策是什么?
>
> 关键词:力闭合、桌面接触、混合整数规划、MCTS+RL 双层架构

---

## 1. 攀岩接触规划与家庭场景的本质差异

### 1.1 对比表

| 维度 | 攀岩场景 (Hold + 肢体) | 家庭场景 (物体 + 服务) |
|---|---|---|
| **接触对象** | 固定岩点 (刚体, 已知位姿) | 可动物体 (形状各异, 位姿未知) |
| **接触选择** | 离散 (N 个岩点选 4 个) | 连续 (手放在物体的哪个部位?) |
| **稳定性判据** | 质心在支撑面内 + 摩擦锥 | 力闭合 (Force Closure) + 物体动力学 |
| **失败代价** | 跌 → 灾难性 | 滑脱 → 打碎东西 / 掉物体 |
| **任务序列** | 一步步向上攀 | 取/放/开/关/擦/递 |
| **时间范围** | 分钟级攀爬 (100+ 步) | 秒级任务 (5-50 步) |

### 1.2 核心数学差异

```
攀岩的静态平衡条件:
  ⎧ Σ F_i + m·g = 0                    (力平衡)
  ⎨ Σ M_i = 0                          (力矩平衡)
  ⎩ |F_tangent_i| ≤ μ·F_normal_i       (每个接触点的摩擦锥)
  
  求解: 是否可行? (Feasible → True/False)
  → 线性规划, O(n)

家庭的力闭合条件 (更多约束):
  ⎧ Σ F_i + m_object·g = m_object·a    (物体运动方程)
  ⎨ Σ M_i = I·α + ω×(I·ω)             (物体旋转方程)
  ⎩ 力闭合存在: G·w ⊇ Origin           (凸包判据)
  ⎪ 形封闭 (Form Closure): rank(G) = 6  
  ⎩ 抗外扰动: w_i > 0, 至少 4 个接触点

  关键差异: 家庭场景必须考虑:
    1. 物体在抓取时的加速运动 (惯性力)
    2. 外力扰动 (用户递物体/手碰撞/桌面振动)
    3. 柔性物体 (毛巾/塑料袋变形不可控)
```

## 2. 家庭夹取规划的力闭合判据

```python
# force_closure_check.py

class ForceClosureChecker:
    """
    判断给定抓取构型是否力闭合
    
    基于凸包判据: n 个接触点能产生任意方向的力/力矩
    → 各接触力向量构成的凸包包含原点
    """
    
    def __init__(self, mu=0.5, min_contact_points=2):
        self.mu = mu
        self.min_contacts = min_contact_points
    
    def check(self, contact_points, contact_normals, ext_force=None):
        """
        Args:
            contact_points: (N, 3) 接触点位置
            contact_normals: (N, 3) 接触面单位法线
            ext_force: (6,) 已知外力 [Fx,Fy,Fz,Mx,My,Mz]
        Returns:
            (is_force_closure, margin, min_wrench)
        """
        N = contact_points.shape[0]
        if N < self.min_contacts:
            return False, -1.0, float('inf')
        
        # 1. 构建抓取矩阵 G (6 × 3N)
        # G 将接触力 [f_1, f_2, ..., f_N] 映射到物体 wrench
        G = np.zeros((6, 3 * N))
        for i in range(N):
            p = contact_points[i]
            n = contact_normals[i]
            
            # 力分量
            G[0:3, 3*i:3*i+3] = np.eye(3)  # [fx, fy, fz] for contact i
            
            # 力矩分量 r × f
            cross = np.array([[0, -p[2], p[1]],
                              [p[2], 0, -p[0]],
                              [-p[1], p[0], 0]])
            G[3:6, 3*i:3*i+3] = cross
        
        # 2. 摩擦锥线性化为 4 面金字塔
        # (每个接触有 4 个面法线)
        contact_forces_basis = []
        for i in range(N):
            n = contact_normals[i]
            # 在切平面找两个正交方向
            if abs(n[0]) < 0.9:
                t1 = np.cross(n, [1,0,0])
            else:
                t1 = np.cross(n, [0,1,0])
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            
            # 4 个金字塔面方向
            for sign1 in [1, -1]:
                for sign2 in [1, -1]:
                    face_dir = n + self.mu * (sign1 * t1 + sign2 * t2)
                    contact_forces_basis.append(face_dir)
        
        # 3. 计算接触力凸包
        # 将每个接触的基力投影到 wrench 空间
        wrench_vertices = []
        for bf_dir in contact_forces_basis:
            # 对应到 6D wrench (假设单位力)
            f_full = np.zeros(3 * N)
            # 简化: 用单个接触的最大力方向
            # 完整实现: 用线性规划求解凸包
            pass
        
        # 4. 凸包包含原点 → L1-minimization
        # (实际中用 SciPy 的 linprog)
        from scipy.optimize import linprog
        
        c = np.ones(4 * N)  # 最小化 Σ w
        A_ub = -np.eye(4 * N)
        b_ub = np.zeros(4 * N)
        A_eq = G_friction_cone  # 6 × 4N
        b_eq = -np.array(ext_force) if ext_force is not None else np.zeros(6)
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=(0, None), method='highs')
        
        if result.success:
            margin = result.fun  # 最小 L1-norm
            return True, margin, result.fun
        else:
            return False, -1.0, float('inf')
```

## 3. 双层架构: MCTS 接触选择 + RL 执行

```python
# hybrid_planner.py

class HybridContactPlanner:
    """
    双层架构:
    上层 (10Hz): MCTS 在接触/物体模型上粗选动作
    下层 (1000Hz): RL 策略网络在接触点上精炼力位混合指令
    
    双层之间用 Socket/CAN 通信
    """
    
    def __init__(self, rl_policy, mcts_planner):
        self.rl = rl_policy           # 接触精炼策略
        self.mcts = mcts_planner      # 接触选择规划器
        
        # 共享内存: 目标接触点位置
        self.target_contact = SharedMemory("/target_contact", shape=(6,))
    
    def plan_and_execute(self, obs):
        """每 100ms 调用一次"""
        
        # === 上层: MCTS 选择接触 ===
        current_stance = self.estimate_current_stance(obs)
        
        if self.should_replan(obs):  # 只在关键帧重规划
            action_sequence = self.mcts.search(
                root=current_stance,
                max_iter=500,
                reward_fn=self.task_reward
            )
            chosen_contact = action_sequence[0]  # 选第一步
            self.target_contact.write(chosen_contact)
            
            # 更新 RL 条件的 context
            self.rl.set_contact_goal(chosen_contact)
        
        # === 下层: RL 输出关节角度 ===
        rl_action = self.rl.act(obs)  # ~15ms GPU
        
        return rl_action
    
    def should_replan(self, obs):
        """判断是否需要重规划"""
        # 1. 滑脱检测触发重规划
        if obs.get('slip_detected', False):
            return True
        
        # 2. 当前接触不可行 (力闭合失败)
        fc_ok, margin = self.force_closure_check.check(obs['contacts'])
        if not_fc_ok or margin < self.min_margin:
            return True
        
        # 3. 新物体/VLM 新指令
        if obs.get('new_goal', False):
            return True
        
        # 定时重规划 (每 50 步)
        if self.step_counter % 50 == 0:
            return True
        
        return False


# ===== 20Hz 控制主循环 ======
class RobotControlLoop:
    def __init__(self):
        self.mcts_rl = HybridContactPlanner(
            rl_policy=MultiModalFusionPolicy(),  # 已有
            mcts_planner=ClimbingMCTS()  # 复用并扩展
        )
        self.can = CANBus(channel='can0', bitrate=1e6)
        self.tactile = TactileInterface(spi_bus=0, frame_rate=30)
        
        self.step_hz = 20
        self.step_dt = 1.0 / self.step_hz
    
    def run(self):
        """主控制循环, 严格 20Hz 周期"""
        next_time = time.monotonic()
        
        while True:
            # === 1. 传感器采集 (并行) ===
            obs = {
                'tactile': self.tactile.read_latest(),
                'visual': self.vision.capture(),
                'proprio': self.can.read_joints(),
            }
            
            # === 2. 双层决策 ===
            target_joints = self.mcts_rl.plan_and_execute(obs)
            
            # === 3. 安全限幅 ===
            target_joints = self.safety_limiter.limit(target_joints, obs)
            
            # === 4. 发送到关节 MCU ===
            self.can.send_targets(target_joints)
            
            # === 5. 决策日志 ===
            self.logger.log(obs, target_joints)
            
            # === 6. 精确周期控制 ===
            next_time += self.step_dt
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.overrun_count += 1
                if self.overrun_count > 10:
                    raise RuntimeError("控制循环过载!")
```

## 4. 关键扩展: 家庭 vs 攀岩的 MCTS 差异

```python
# ===== MCTS 适配 =====

class HouseholdMCTS(ClimbingMCTS):
    """攀岩 MCTS 的扩展版本, 适配家庭/服务场景"""
    
    def __init__(self, object_db, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_db = object_db  # 物体的 CAD 模型 + 语义
        
    def get_actions(self, state):
        """
        动作空间扩展:
        - 攀岩: 选肢体 + 选岩点 → O(H·L) ~ 几千种
        - 家庭: 选手指 + 抓取物体 → 选抓取点 → 是连续空间
        
        对策:
          1. 候选抓取点通过学习模型 (GQ-CNN/Contact-GNET) 预筛
          2. 每个物体的候选抓取点降采样到 ~50 个
          3. MCTS 的动作选择从离散集合中做
        """
        candidates = self.object_db.get_grasp_candidates(
            state.target_object_id,
            n_candidates=50,
            min_force_closure_margin=0.3
        )
        return candidates
    
    def is_feasible(self, stance, target_obj):
        """可行性检查: 力闭合 + 碰撞检查 + 运动学可达"""
        # 1. 力闭合
        contact_stance = stance.get_contacts(target_obj)
        fc_result = self.fc_checker.check(*contact_stance)
        if not fc_result[0]:
            return False
        
        # 2. 碰撞自由 (用预计算的碰撞球)
        if self.collision_check.is_colliding(stance):
            return False
        
        # 3. 运动学可达 (用学习到的可达性图)
        if not self.reachability_map.is_reachable(target_obj.pose):
            return False
        
        return True
    
    def heuristic_value(self, state):
        """启发函数: 任务进度 + 稳定性 + 能量效率 (区别于攀岩)"""
        progress = 1.0 - (state.task_step / state.total_steps)
        stability = state.force_closure_margin
        energy_cost = state.joint_effort_penalty
        return 0.5 * progress + 0.3 * stability - 0.2 * energy_cost
```

## 5. 推导: 为什么 MCTS+RL 双层是最优架构

| 层级 | 算法 | 频率 | 作用 | 做不好会怎样? |
|---|---|---|---|---|
| 任务规划 | LLM/VLM | 1Hz | 生成子任务链 | 子任务不可行 |
| 接触选择 | MCTS | 2-10Hz | 选抓手部位/导航路径 | 力夹持不当/碰撞 |
| 运动执行 | RL (PPO/SAC) | 20Hz | 关节角度指令 | 抖动/不自然/避障失败 |
| 阻抗控制 | 导纳控制 | 1000Hz | 力位混合 | 碰到硬物打碎杯 |
| 电流控制 | FOC | 10kHz | 力矩快速跟踪 | 自动关机跳闸 |

```
为什么不用"纯端到端" (单一网络输入观察 → 输出 PWM)?
  1. 安全: 自主关断需要 FOC 级 (硬件级), 不能依赖 AI
  2. 可调试: 出错时, 你希望知道"是高层规划错了, 还是 RL 错了, 还是电流环错了?"
     层次化架构天然支持这一点
  3. 混合自主: MCTS 做高层决策(可解释, 可验证), RL 做低层执行(灵活, 适应噪声)
  4. 认证: ISO/CE 认证要求"可证明安全性", 分层架构满足 (每层有验证)

结论: 分层架构是必须的, 不是benefits-vs-costs tradeoff.
```
