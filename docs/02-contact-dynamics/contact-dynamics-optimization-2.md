## 🤖 Assistant

```
  Constraints:
    - Geometric feasibility: d_i(x, m) ≤ 0 (无碰撞)
    - Force closure: rank(G(m)) = 6 且 ε_Ferrari(m) > ε_min
    - Transition validity: |mₜ - mₜ₊₁|₁ ≤ 1 (每次只改变一个接触)


Level 2: 轨迹优化（连续优化，给定模式，20Hz）
---------------------------------------------

  min     ∑ᵗ₌₀ᵀ [‖uₜ‖²_R + ‖xₜ - x_goal‖²_Q + ε‖fₜ‖²]
  x,u,f

  s.t.  动力学（软接触模型）:
        
        xₜ₊₁ = f_dynamics(xₜ, uₜ, fₜ)
        
        其中接触力由Hunt-Crossley模型给出:
        
        f_normal,i = {  k_i · δ^1.5 · (1 + b_i · δ̇)   if δ > 0 (穿透)
                     {  0                              otherwise
        
        δ = -d_i(xₜ)  (穿透深度)
        
        摩擦力（Stribeck模型）:
        
        f_tan,i = -[μ_c + (μ_s - μ_c)·exp(-‖vₜₐₙ‖/v_s)] · f_normal,i · v̂ₜₐₙ
        
        其中:
          μ_s = 0.8     (静摩擦系数)
          μ_c = 0.6     (动摩擦系数)
          v_s = 0.01    (Stribeck速度)
        
        对于接触点 i ∈ active(mₜ):
        
        摩擦锥约束（二阶锥，精确）:
          ‖f_tan,i‖₂ ≤ μ(‖vₜₐₙ,ᵢ‖) · f_normal,i
          
          等价于: [f_normal,i; f_tan,i] ∈ 𝕃³ (Lorentz cone)
        
        力/力矩平衡:
          ∑ᵢ fᵢ + f_gravity = m·aₜ
          ∑ᵢ (rᵢ × fᵢ) + τ_gravity = I·αₜ
        
        控制限制:
          ‖uₜ‖∞ ≤ u_max
          ‖u̇ₜ‖∞ ≤ u̇_max  (加速度限制，平滑性)
        
        状态约束:
          xₜ ∈ 𝒳_safe  (工作空间、关节限位)
        
        安全约束:
          d_human(xₜ) ≥ d_min  (人体距离)
          ‖fₜ‖∞ ≤ f_max        (力限制)


Level 3: 实时反馈控制（PID/Impedance，1kHz）
---------------------------------------------

  给定名义轨迹 (x*, u*, f*), 实时补偿:
  
  u_actual = u*ₜ + K_p(x*ₜ - xₜ) + K_d(ẋ*ₜ - ẋₜ) + K_i∫(x*ₜ - xₜ)dt
  
  或阻抗控制:
  
  f_desired = K(x - x*) + B(ẋ - ẋ*) + M(ẍ - ẍ*)

===============================================
```

---

## 四、关键改进点详解

### 改进1：可微软接触模型

```python
class SoftContactModel:
    """
    Hunt-Crossley非线性弹簧阻尼模型
    
    优势：
      1. 可微 → 可用梯度优化、端到端学习
      2. 无互补约束 → 避免非凸优化
      3. 物理准确 → 匹配真实材料
    """
    
    def __init__(self, stiffness=1e5, damping=100, exponent=1.5):
        """
        参数：
          stiffness: 接触刚度 k (N/m^1.5)
            - 软物体: 1e4 (海绵)
            - 中等: 1e5 (橡胶)
            - 硬物体: 1e6 (木头)
          
          damping: 阻尼系数 b (s/m)
            - 典型值: 50-500
          
          exponent: 非线性指数 n
            - Hertz理论: n=1.5 (球-平面接触)
            - 线性: n=1.0
        """
        self.k = stiffness
        self.b = damping
        self.n = exponent
    
    def compute_contact_force(self, penetration, velocity):
        """
        计算接触力（可微）
        
        公式：f = k · δⁿ · (1 + b · δ̇)
        
        输入：
          penetration: δ (m), 正值表示穿透
          velocity: δ̇ (m/s), 正值表示继续穿透
        
        输出：
          f_normal: (N), 法向接触力
        """
        if penetration <= 0:
            return 0.0
        
        # 非线性弹簧力
        f_elastic = self.k * (penetration ** self.n)
        
        # 速度依赖阻尼（Hunt-Crossley关键创新）
        damping_factor = 1.0 + self.b * velocity
        
        # 总接触力
        f_normal = f_elastic * max(0, damping_factor)
        
        return f_normal
    
    def compute_force_gradient(self, penetration, velocity):
        """
        计算雅可比矩阵（用于基于梯度的优化）
        
        ∂f/∂δ = k·n·δ^(n-1)·(1 + b·δ̇)
        ∂f/∂δ̇ = k·b·δⁿ
        """
        if penetration <= 0:
            return np.zeros(2)
        
        df_dpenetration = (
            self.k * self.n * (penetration ** (self.n - 1)) * 
            (1.0 + self.b * velocity)
        )
        
        df_dvelocity = self.k * self.b * (penetration ** self.n)
        
        return np.array([df_dpenetration, df_dvelocity])
    
    def linearize_around_setpoint(self, delta_0, v_0):
        """
        局部线性化（用于MPC）
        
        f ≈ f_0 + ∂f/∂δ·(δ - δ_0) + ∂f/∂δ̇·(δ̇ - δ̇_0)
        """
        f_0 = self.compute_contact_force(delta_0, v_0)
        jacobian = self.compute_force_gradient(delta_0, v_0)
        
        return f_0, jacobian

# 使用示例：与刚体模型对比
def compare_contact_models():
    """
    对比刚体互补约束 vs 软接触模型
    """
    import matplotlib.pyplot as plt
    
    # 场景：手指以0.1 m/s速度接触桌面
    penetrations = np.linspace(0, 0.005, 100)  # 0-5mm
    velocity = 0.1  # m/s
    
    # 模型1：刚体（互补约束）
    rigid_forces = np.where(
        penetrations > 0,
        np.inf,  # 无穷大力（刚体不允许穿透）
        0
    )
    
    # 模型2：软接触（Hunt-Crossley）
    soft_model = SoftContactModel(stiffness=1e5, damping=100)
    soft_forces = [
        soft_model.compute_contact_force(d, velocity) 
        for d in penetrations
    ]
    
    # 绘图
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(penetrations * 1000, soft_forces, label='Soft (Hunt-Crossley)')
    plt.axhline(y=0, color='r', linestyle='--', label='Rigid (Complementarity)')
    plt.xlabel('Penetration (mm)')
    plt.ylabel('Contact Force (N)')
    plt.title('Contact Force vs Penetration')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    velocities = np.linspace(0, 0.5, 100)
    forces_at_1mm = [
        soft_model.compute_contact_force(0.001, v) 
        for v in velocities
    ]
    plt.plot(velocities, forces_at_1mm)
    plt.xlabel('Impact Velocity (m/s)')
    plt.ylabel('Contact Force (N)')
    plt.title('Velocity-Dependent Damping (δ=1mm)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("关键区别：")
    print("1. 刚体模型：无穿透 → 不可微 → 需要互补约束")
    print("2. 软接触：允许小穿透 → 可微 → 标准优化求解")
    print(f"3. 1mm穿透时的力: {soft_model.compute_contact_force(0.001, 0.1):.1f} N")
```

### 改进2：精确摩擦锥（二阶锥约束）

```python
class FrictionConeConstraint:
    """
    精确摩擦锥：二阶锥规划（SOCP）
    """
    
    def __init__(self, mu_static=0.8, mu_kinetic=0.6):
        self.mu_s = mu_static
        self.mu_k = mu_kinetic
    
    def add_to_optimization(self, problem, f_normal, f_tangent, velocity):
        """
        添加精确摩擦锥约束到CVXPY问题
        
        原始（非凸）：‖f_tan‖₂ ≤ μ·f_normal
        
        转换为二阶锥：(f_normal, f_tan) ∈ Lorentz cone
        即：‖f_tan‖₂ ≤ f_normal / μ
        
        CVXPY表示：cp.SOC(t, x) 约束 ‖x‖₂ ≤ t
        """
        import cvxpy as cp
        
        # 速度依赖的摩擦系数（Stribeck）
        v_norm = cp.norm(velocity)
        v_stribeck = 0.01
        
        # 平滑插值（用软化版本，保持凸性）
        # μ(v) ≈ μ_k + (μ_s - μ_k) / (1 + v/v_s)
        # 注意：精确Stribeck不是凸的，这里用凸近似
        mu_effective = self.mu_k + (self.mu_s - self.mu_k) / (1 + v_norm / v_stribeck)
        
        # 二阶锥约束（精确摩擦锥）
        constraint = cp.SOC(
            mu_effective * f_normal,  # 锥的高度
            f_tangent                  # 锥的底面（2D或3D向量）
        )
        
        problem.constraints.append(constraint)
        
        return problem
    
    def compare_approximations(self):
        """
        对比不同摩擦锥近似的误差
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Circle
        
        mu = 0.6
        f_n = 10.0  # 法向力10N
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 方法1：精确摩擦锥（圆）
        theta = np.linspace(0, 2*np.pi, 100)
        r = mu * f_n
        x_exact = r * np.cos(theta)
        y_exact = r * np.sin(theta)
        
        axes[0].fill(x_exact, y_exact, alpha=0.3, color='green', label='Exact')
        axes[0].set_title('Exact Friction Cone')
        axes[0].set_aspect('equal')
        axes[0].grid(True)
        axes[0].legend()
        
        # 方法2：线性近似（正方形）
        mu_linear = mu / np.sqrt(2)
        square = [
            [-mu_linear*f_n, -mu_linear*f_n],
            [mu_linear*f_n, -mu_linear*f_n],
            [mu_linear*f_n, mu_linear*f_n],
            [-mu_linear*f_n, mu_linear*f_n]
        ]
        axes[1].fill(x_exact, y_exact, alpha=0.2, color='green', label='Exact')
        axes[1].add_patch(Polygon(square, alpha=0.3, color='red', label='Linear Approx'))
        axes[1].set_title('Linear Approximation (Conservative)')
        axes[1].set_aspect('equal')
        axes[1].grid(True)
        axes[1].legend()
        
        # 方法3：多面体近似（八边形）
        n_sides = 8
        theta_poly = np.linspace(0, 2*np.pi, n_sides+1)
        r_poly = mu * f_n / np.cos(np.pi / n_sides)  # 外接
        x_poly = r_poly * np.cos(theta_poly)
        y_poly = r_poly * np.sin(theta_poly)
        
        axes[2].fill(x_exact, y_exact, alpha=0.2, color='green', label='Exact')
        axes[2].fill(x_poly, y_poly, alpha=0.3, color='orange', label='8-sided Approx')
        axes[2].set_title('Polyhedral Approximation')
        axes[2].set_aspect('equal')
        axes[2].grid(True)
        axes[2].legend()
        
        plt.tight_layout()
        
        # 计算误差
        area_exact = np.pi * (mu * f_n)**2
        area_linear = (2 * mu_linear * f_n)**2
        area_poly = 0.5 * n_sides * (r_poly**2) * np.sin(2*np.pi / n_sides)
        
        print("面积对比（越接近越好）：")
        print(f"  精确圆形: {area_exact:.2f}")
        print(f"  线性近似: {area_linear:.2f} (误差: {(area_linear/area_exact - 1)*100:+.1f}%)")
        print(f"  八边形: {area_poly:.2f} (误差: {(area_poly/area_exact - 1)*100:+.1f}%)")
        
        return fig
```

### 改进3：力闭合质量度量

```python
class GraspQualityMetrics:
    """
    力闭合与抓取质量评估
    """
    
    def __init__(self):
        pass
    
    def compute_ferrari_canny_metric(self, grasp_matrix):
        """
        Ferrari-Canny L∞ 指标（最广泛使用）
        
        定义：最坏情况下的wrench抵抗能力
        
        Q = min_{‖w‖=1} max_{α≥0, ‖α‖=1} wᵀGα
        
        物理意义：
          - Q > 0: 力闭合（可抵抗任意方向外力）
          - Q越大，抓取越鲁棒
          - 典型值：0.01-0.1（归一化）
        
        输入：
          grasp_matrix: G ∈ ℝ^(6×n), n=接触点数×3
        
        输出：
          epsilon: 力闭合质量指标
        """
        
        # 采样单位wrench球面
        n_samples = 200
        unit_wrenches = self._sample_unit_sphere(dim=6, n=n_samples)
        
        min_distance = np.inf
        
        for w in unit_wrenches:
            # 对每个wrench方向，找最大投影
            # max_{α≥0, ‖α‖=1} wᵀGα
            
            # 转化为LP
            import cvxpy as cp
            alpha = cp.Variable(grasp_matrix.shape[1])
            objective = cp.Maximize(w @ grasp_matrix @ alpha)
            constraints = [
                alpha >= 0,
                cp.norm(alpha) <= 1
            ]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            projection = problem.value
            min_distance = min(min_distance, projection)
        
        return min_distance
    
    def compute_volume_metric(self, grasp_matrix):
        """
        体积指标：Grasp Wrench Space (GWS) 的体积
        
        思想：可达wrench空间越大，抓取越鲁棒
        
        计算：GWS = ConvexHull(所有可能的wrench组合)
        """
        from scipy.spatial import ConvexHull
        
        # 枚举所有可能的wrench（正交基）
        n_contacts = grasp_matrix.shape[1] // 3
        
        # 采样wrench空间
        wrenches = []
        n_samples_per_contact = 10
        
        for i in range(n_contacts):
            # 每个接触点的力（法向+切向）
            for _ in range(n_samples_per_contact):
                alpha = np.random.rand(grasp_matrix.shape[1])
                alpha[3*i:3*(i+1)] /= np.linalg.norm(alpha[3*i:3*(i+1)])  # 归一化
                
                wrench = grasp_matrix @ alpha
                wrenches.append(wrench)
        
        wrenches = np.array(wrenches)
        
        # 计算凸包体积
        try:
            hull = ConvexHull(wrenches)
            volume = hull.volume
        except:
            volume = 0.0  # 退化情况
        
        return volume
    
    def check_force_closure_necessary_conditions(self, contact_points, contact_normals):
        """
        力闭合必要条件（快速预检）
        
        条件1：至少4个接触点（3D）
        条件2：接触法向量张成全空间
        条件3：物体重心在支撑多边形内（2D投影）
        """
        n_contacts = len(contact_points)
        
        # 条件1
        if n_contacts < 4:
            return False, "需要至少4个接触点"
        
        # 条件2：检查法向量的秩
        normal_matrix = np.array(contact_normals).T
        rank = np.linalg.matrix_rank(normal_matrix)
        if rank < 3:
            return False, f"接触法向量退化（秩={rank}）"
        
        # 条件3：重心在支撑多边形内（XY平面投影）
        contact_2d = contact_points[:, :2]  # 只看XY
        center_of_mass = np.array([0, 0])  # 假设物体中心
        
        from scipy.spatial import Delaunay
        hull = Delaunay(contact_2d)
        inside = hull.find_simplex(center_of_mass) >= 0
        
        if not inside:
            return False, "重心不在支撑多边形内"
        
        return True, "通过必要条件检查"
    
    def _sample_unit_sphere(self, dim, n):
        """
        均匀采样单位球面
        """
        points = np.random.randn(n, dim)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return points

# 使用示例
def analyze_grasp_quality():
    """
    分析不同抓取配置的质量
    """
    
    # 场景1：四指抓取（平行对握）
    contact_points_parallel = np.array([
        [0.05, 0, 0],      # 右侧
        [-0.05, 0, 0],     # 左侧
        [0, 0.05, 0],      # 前侧
        [0, -0.05, 0],     # 后侧
    ])
    
    contact_normals_parallel = np.array([
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
    ])
    
    # 场景2：三指抓取（三角形）
    contact_points_triangle = np.array([
        [0.05, 0, 0],
        [-0.025, 0.043, 0],
        [-0.025, -0.043, 0],
    ])
    
    angle = np.array([0, 120, 240]) * np.pi / 180
    contact_normals_triangle = np.column_stack([
        -np.cos(angle),
        -np.sin(angle),
        np.zeros(3)
    ])
    
    # 构建Grasp Matrix
    evaluator = GraspQualityMetrics()
    
    G_parallel = build_grasp_matrix(contact_points_parallel, contact_normals_parallel)
    G_triangle = build_grasp_matrix(contact_points_triangle, contact_normals_triangle)
    
    # 评估
    q_parallel = evaluator.compute_ferrari_canny_metric(G_parallel)
    q_triangle = evaluator.compute_ferrari_canny_metric(G_triangle)
    
    print("抓取质量对比：")
    print(f"  四指平行: Q = {q_parallel:.4f}")
    print(f"  三指三角: Q = {q_triangle:.4f}")
    print(f"  结论: {'四指更鲁棒' if q_parallel > q_triangle else '三角更鲁棒'}")

def build_grasp_matrix(contact_points, contact_normals):
    """
    构建Grasp Matrix: G ∈ ℝ^(6×3n)
    
    每个接触点贡献3列（1法向+2切向）
    每列是6维wrench: [fx, fy, fz, τx, τy, τz]ᵀ
    """
    n = len(contact_points)
    G = np.zeros((6, 3*n))
    
    for i in range(n):
        point = contact_points[i]
        normal = contact_normals[i]
        
        # 构建正交切向量
        tangent1, tangent2 = get_tangent_basis(normal)
        
        for j, direction in enumerate([normal, tangent1, tangent2
