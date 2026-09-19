## 🤖 Assistant


# 接触动力学优化：从简化模型到实用公式

## 一、现有公式的核心问题

### 问题诊断

```python
current_formulation_issues = {
    '数学层面': {
        '互补约束非凸': {
            'problem': 'c_t · d_t = 0 是非凸的',
            'consequence': '局部最优、难以求解',
            'solver_failure_rate': '40-60%',
        },
        
        '摩擦锥近似粗糙': {
            'problem': '线性化摩擦锥（金字塔）误差大',
            'consequence': '过度保守或不安全',
            'error': '±15-30%',
        },
        
        '缺少接触模式先验': {
            'problem': '搜索空间包含大量不可行接触序列',
            'consequence': '计算爆炸',
            'example': '5个接触点 → 2^5=32种模式',
        },
    },
    
    '物理层面': {
        '忽略柔顺性': {
            'problem': '刚体假设 vs 真实软接触',
            'consequence': '碰撞冲击、振动',
            'real_world_impact': '抓取失败率+30%',
        },
        
        '静摩擦 vs 动摩擦': {
            'problem': '单一摩擦系数μ',
            'consequence': '无法建模stiction和滑移转换',
            'example': '杯子抓取时的微滑移',
        },
        
        '滚动接触缺失': {
            'problem': '只考虑滑动摩擦',
            'consequence': '无法处理圆柱物体',
            'tasks_affected': ['倒水', '开门', '推车'],
        },
    },
    
    '计算层面': {
        '求解器依赖强': {
            'problem': '需要专业优化器（SNOPT/IPOPT）',
            'consequence': '20Hz实时性难保证',
            'typical_solve_time': '50-500ms',
        },
        
        '不可微': {
            'problem': '互补约束导致梯度不存在',
            'consequence': '无法用端到端学习',
            'workaround': '需要显式接触序列',
        },
    },
}
```

---

## 二、改进方案：分层接触优化框架

### 核心思想：Contact Mode Enumeration + Smooth Relaxation

```python
"""
策略：将离散的接触模式搜索与连续的力优化解耦

Layer 1: 离散搜索（粗粒度）
  → 找到合理的接触模式序列（哪些面接触？）
  
Layer 2: 连续优化（细粒度）
  → 在给定接触模式下优化力和轨迹

优势：
  1. 减少搜索空间（32 → 3-5个候选模式）
  2. 子问题变凸（可用快速QP求解器）
  3. 可并行（不同模式独立优化）
"""
```

### 改进后的数学公式

```python
"""
===============================================
改进公式：混合整数接触优化 (Hybrid Contact Optimization)
===============================================
"""

import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

class ImprovedContactOptimization:
    """
    分层接触优化：解决原公式的三大问题
    """
    
    def __init__(self, T=50, dt=0.05):
        self.T = T           # 时间步数
        self.dt = dt         # 时间步长
        self.n_contacts = 4  # 接触点数量（如手指）
        
    # ============================================
    # Layer 1: 接触模式规划（离散优化）
    # ============================================
    
    def plan_contact_modes(self, x_init, x_goal):
        """
        问题1改进：用图搜索代替盲目枚举
        
        状态空间：contact_mode = [active_0, active_1, ..., active_n]
        active_i ∈ {0, 1} 表示第i个接触点是否激活
        
        启发式：
          1. 几何可行性（碰撞检测）
          2. 力闭合必要条件（重心在支撑多边形内）
          3. 学习的价值函数（RL/MCTS）
        """
        
        # A* 搜索接触模式序列
        contact_sequence = self._astar_search(
            start_mode=self._detect_initial_contacts(x_init),
            goal_mode=self._detect_goal_contacts(x_goal),
            heuristic=self._learned_contact_heuristic,
        )
        
        return contact_sequence
    
    def _astar_search(self, start_mode, goal_mode, heuristic):
        """
        A*搜索接触模式图
        
        节点：接触模式 [0,1,1,0] (哪些手指接触)
        边：接触建立/断开
        代价：力矩成本 + 转换成本
        """
        from heapq import heappush, heappop
        
        open_set = [(0, start_mode)]
        came_from = {}
        g_score = {start_mode: 0}
        
        while open_set:
            _, current = heappop(open_set)
            
            if current == goal_mode:
                return self._reconstruct_path(came_from, current)
            
            # 探索相邻接触模式
            for neighbor in self._get_neighbor_modes(current):
                # 转换代价：建立/断开接触的力矩成本
                transition_cost = self._contact_transition_cost(
                    current, neighbor
                )
                
                tentative_g = g_score[current] + transition_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # f = g + h（实际代价 + 启发式估计）
                    f_score = tentative_g + heuristic(neighbor, goal_mode)
                    heappush(open_set, (f_score, neighbor))
        
        raise Exception("No valid contact sequence found")
    
    def _learned_contact_heuristic(self, mode, goal):
        """
        学习的启发式函数（用RL/MCTS离线训练）
        
        输入：当前接触模式、目标模式
        输出：到达目标的估计代价
        """
        # 可以是：
        # 1. 简单几何距离
        # 2. 神经网络价值函数
        # 3. MCTS rollout估计
        
        # 示例：汉明距离（需要改变多少个接触）
        hamming = np.sum(mode != goal)
        return hamming * 10.0  # 每次接触变化代价10
    
    # ============================================
    # Layer 2: 给定模式下的轨迹优化（连续优化）
    # ============================================
    
    def optimize_trajectory_with_mode(self, contact_mode_sequence):
        """
        问题2改进：在固定接触模式下，问题变为凸优化（QP）
        
        关键：互补约束c·d=0被替换为：
          - 接触模式已知 → 直接固定c或d
          - 不需要求解互补条件
        """
        
        x_traj = []
        u_traj = []
        f_traj = []
        
        for t, mode_t in enumerate(contact_mode_sequence):
            # 在时刻t，接触模式已知
            active_contacts = np.where(mode_t == 1)[0]
            
            # 凸优化子问题（QP）
            x_t, u_t, f_t = self._solve_convex_subproblem(
                t=t,
                active_contacts=active_contacts,
                x_prev=x_traj[-1] if t > 0 else self.x_init,
            )
            
            x_traj.append(x_t)
            u_traj.append(u_t)
            f_traj.append(f_t)
        
        return x_traj, u_traj, f_traj
    
    def _solve_convex_subproblem(self, t, active_contacts, x_prev):
        """
        凸优化子问题：固定接触模式后的QP
        
        ===================================
        改进公式（单步，凸优化）
        ===================================
        
        min   ||u_t||² + λ₁||x_t - x_goal||² + λ₂||f_t||²
        x,u,f
        
        s.t.  x_{t+1} = f(x_t, u_t, f_t)                     [动力学]
              
              对于 i ∈ active_contacts:
                d_i = 0                                       [接触距离]
                ||f_tan,i|| ≤ μ_s(||f_tan,i||) · f_norm,i   [摩擦锥，改进]
                f_norm,i ≥ 0                                  [单向力]
              
              对于 i ∉ active_contacts:
                f_i = 0                                       [无接触力]
                d_i ≥ 0                                       [无穿透]
              
              ∑ f_i + f_gravity = m · a                       [力平衡]
              ∑ (r_i × f_i) + τ_gravity = I · α              [力矩平衡]
              
              ||u_t|| ≤ u_max                                 [控制限制]
              x_t ∈ X_safe                                    [状态约束]
        """
        
        # 使用CVXPY求解（快速且可靠）
        n_state = 12
        n_ctrl = 12
        n_contact = len(active_contacts)
        
        # 变量
        x = cp.Variable(n_state)
        u = cp.Variable(n_ctrl)
        f_normal = cp.Variable(n_contact)
        f_tangent = cp.Variable((n_contact, 2))  # 2D切向力
        
        # 目标函数
        cost = (
            cp.sum_squares(u) +                    # 控制成本
            10 * cp.sum_squares(x - self.x_goal) + # 跟踪成本
            0.01 * cp.sum_squares(f_normal) +      # 力正则化
            0.01 * cp.sum_squares(f_tangent)
        )
        
        # 约束
        constraints = []
        
        # 1. 线性化动力学（用于实时优化）
        A, B, B_f = self._linearize_dynamics(x_prev)
        x_next = A @ x + B @ u + B_f @ self._stack_forces(f_normal, f_tangent)
        constraints.append(x_next == self._forward_dynamics(x_prev, u, ...))
        
        # 2. 改进的摩擦锥约束（后面详述）
        for i, contact_id in enumerate(active_contacts):
            # 非线性摩擦模型
            mu_effective = self._friction_coefficient(
                f_normal[i], 
                contact_id
            )
            
            # 二阶锥约束（精确摩擦锥）
            constraints.append(
                cp.SOC(mu_effective * f_normal[i], f_tangent[i, :])
            )
            
            # 法向力非负
            constraints.append(f_normal[i] >= 0)
        
        # 3. 力平衡（准静态假设）
        total_force = cp.sum([
            self._contact_jacobian(contact_id) @ 
            cp.hstack([f_normal[i], f_tangent[i, :]])
            for i, contact_id in enumerate(active_contacts)
        ])
        constraints.append(total_force == -self.gravity_wrench)
        
        # 4. 控制限制
        constraints.append(cp.abs(u) <= self.u_max)
        
        # 求解QP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True)
        
        if problem.status != cp.OPTIMAL:
            raise Exception(f"QP solver failed: {problem.status}")
        
        return x.value, u.value, f_normal.value, f_tangent.value
    
    # ============================================
    # 改进3：软接触模型（替代刚体假设）
    # ============================================
    
    def _soft_contact_dynamics(self, x, u, f_contact):
        """
        问题3改进：引入柔顺接触模型
        
        刚体模型：
          d = 0  →  f任意（互补约束）
        
        软接触模型（Hunt-Crossley）：
          f_normal = k · d^n + b · ḋ      (n=1.5, 非线性弹簧阻尼)
          
        优势：
          1. 可微（可用梯度优化）
          2. 更真实（建模材料变形）
          3. 无需互补约束
        """
        
        # 接触距离（有符号距离函数）
        d = self._signed_distance(x)
        d_dot = self._distance_velocity(x)
        
        # Hunt-Crossley模型
        f_normal = np.zeros_like(d)
        
        for i in range(len(d)):
            if d[i] < 0:  # 穿透
                penetration = -d[i]
                velocity = -d_dot[i]
                
                # 非线性弹簧 + 速度依赖阻尼
                k = self.contact_stiffness[i]
                n = 1.5  # 经验值（Hertz接触理论）
                b = self.contact_damping[i]
                
                f_normal[i] = (
                    k * penetration**n * (1 + b * velocity)
                )
            else:
                f_normal[i] = 0
        
        # 摩擦力（Stribeck模型，建模stiction）
        f_tangent = self._stribeck_friction(
            f_normal, 
            relative_velocity=self._tangential_velocity(x)
        )
        
        # 动力学更新
        x_next = self._integrate_dynamics(x, u, f_normal, f_tangent)
        
        return x_next
    
    def _stribeck_friction(self, f_normal, v_tangent):
        """
        改进的摩擦模型：Stribeck曲线
        
        原公式：||f_tan|| ≤ μ · f_normal（库仑摩擦，不连续）
        
        改进：
                      ⎧ μ_s · f_n                      if |v| < v_stick
          f_tan = ⎨ [μ_c + (μ_s-μ_c)·e^(-|v|/v_s)] · f_n · sign(v)  if |v| ≥ v_stick
                      ⎩
        
        参数：
          μ_s: 静摩擦系数（0.8）
          μ_c: 动摩擦系数（0.6）
          v_stick: stiction阈值（0.001 m/s）
          v_s: Stribeck速度（0.01 m/s）
        """
        
        v_norm = np.linalg.norm(v_tangent, axis=-1)
        
        # Stribeck函数（平滑插值）
        mu_static = 0.8
        mu_kinetic = 0.6
        v_stribeck = 0.01
        
        mu_effective = mu_kinetic + (mu_static - mu_kinetic) * np.exp(
            -v_norm / v_stribeck
        )
        
        # 摩擦力方向
        direction = np.where(
            v_norm[:, None] > 1e-6,
            -v_tangent / v_norm[:, None],
            0.0
        )
        
        f_friction = (mu_effective * f_normal)[:, None] * direction
        
        return f_friction
    
    # ============================================
    # 改进4：二阶锥摩擦锥（替代线性近似）
    # ============================================
    
    def _friction_cone_constraint_improved(self, f_normal, f_tangent, mu):
        """
        原公式：||f_tan|| ≤ μ · f_normal（精确但非凸）
        
        常用近似：线性化（金字塔）
          |f_tan_x| ≤ μ/√2 · f_normal
          |f_tan_y| ≤ μ/√2 · f_normal
        
        问题：误差大（保守15-30%）
        
        改进：二阶锥约束（SOC，精确且可用内点法）
          ||f_tan|| ≤ μ · f_normal  ⟺  [f_normal; f_tan] ∈ Lorentz Cone
        
        CVXPY支持：cp.SOC(t, x) 表示 ||x||₂ ≤ t
        """
        
        import cvxpy as cp
        
        # 精确摩擦锥（二阶锥）
        constraint = cp.SOC(
            mu * f_normal,      # 标量（锥的高度）
            f_tangent           # 向量（锥的底面半径）
        )
        
        return constraint
    
    # ============================================
    # 改进5：接触雅可比矩阵（力闭合约束）
    # ============================================
    
    def _force_closure_constraint(self, contact_points, contact_normals):
        """
        原公式：隐式力平衡
        
        改进：显式Grasp Matrix检查
        
        定义：Grasp Wrench Space (GWS)
          W = [w₁, w₂, ..., wₙ]
          wᵢ = [fᵢ; rᵢ × fᵢ]  (6维wrench)
        
        力闭合条件：
          1. GWS包含原点（内部）
          2. 等价于：min_{αᵢ≥0} ||∑ αᵢwᵢ|| > 0 对所有方向
        
        快速检查：Ferrari-Canny指标
          Q = min_{||w||=1} max_{αᵢ≥0} min_i (wᵀwᵢ / ||wᵢ||)
          Q > 0 ⟺ 力闭合
        """
        
        n_contacts = len(contact_points)
        
        # 构建Grasp Matrix (6 × 3n)
        G = np.zeros((6, 3 * n_contacts))
        
        for i, (point, normal) in enumerate(zip(contact_points, contact_normals)):
            # 每个接触点贡献3列（1个法向力 + 2个切向力）
            tangent_1, tangent_2 = self._get_tangent_basis(normal)
            
            for j, direction in enumerate([normal, tangent_1, tangent_2]):
                col_idx = 3 * i + j
                
                # 力部分（前3行）
                G[0:3, col_idx] = direction
                
                # 力矩部分（后3行）
                G[3:6, col_idx] = np.cross(point, direction)
        
        # Ferrari-Canny指标
        epsilon = self._compute_ferrari_canny(G)
        
        if epsilon < 0.01:
            raise Exception("Grasp is not force-closed (ε = {:.3f})".format(epsilon))
        
        return G, epsilon
    
    def _compute_ferrari_canny(self, G):
        """
        计算力闭合质量指标（快速近似）
        """
        from scipy.spatial import ConvexHull
        
        # 采样单位wrench空间
        n_samples = 100
        unit_wrenches = self._sample_unit_sphere_6d(n_samples)
        
        min_distance = np.inf
        
        for w in unit_wrenches:
            # 对每个方向，找最近的可达wrench
            # 解LP：min ||Gα - w||  s.t. α ≥ 0
            alpha = cp.Variable(G.shape[1])
            objective = cp.Minimize(cp.norm(G @ alpha - w))
            constraints = [alpha >= 0]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            distance = problem.value
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    # ============================================
    # 改进6：Model Predictive Control集成
    # ============================================
    
    def mpc_with_contact_planning(self, x_current, horizon=20):
        """
        实时闭环MPC：结合接触规划
        
        流程：
          1. 快速重规划接触模式（10ms）
          2. 优化未来horizon步（30ms）
          3. 执行第一步
          4. 重复（滚动优化）
        """
        
        # 1. 接触模式预测（用快速启发式）
        predicted_modes = self._predict_contact_modes_fast(
            x_current, 
            horizon=horizon
        )
        
        # 2. 求解MPC问题
        u_sequence = []
        x_sequence = [x_current]
        
        for t in range(horizon):
            # 当前接触模式
            mode_t = predicted_modes[t]
            
            # 单步优化（warm-start from previous solution）
            x_t, u_t, f_t = self._solve_convex_subproblem(
                t=t,
                active_contacts=np.where(mode_t == 1)[0],
                x_prev=x_sequence[-1],
            )
            
            x_sequence.append(x_t)
            u_sequence.append(u_t)
        
        # 3. 返回第一步控制（MPC receding horizon）
        return u_sequence[0]
```

---

## 三、完整改进公式总结

### 最终公式（分层优化）

```
===============================================
改进的接触规划与控制公式（生产级）
===============================================

Level 1: 接触模式规划（离散搜索，1Hz）
---------------------------------------------

  Find:  M* = {m₀, m₁, ..., m_T}
         mₜ ∈ {0,1}ⁿ (接触模式，n=接触点数)
  
  Via:   A* search with learned heuristic
  
  Cost:  J_mode = ∑ᵗ [τ_transition(mₜ, mₜ₊₁) + h(mₜ, m_goal)]
  
  Constraints:
    - Geometric feasibility: d_i(x, m) ≤ 0 (无碰撞)
    - Force closure: rank
