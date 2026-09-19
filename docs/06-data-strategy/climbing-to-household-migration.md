## 🤖 Assistant


# 攀岩接触规划迁移到家庭场景的技术分析

## 快速建议：**选项 2（扩展物理模型）+ 选项 3（混合学习）的混合架构**

核心论点：家庭任务的**多样性远超攀岩**（100+物体类别 vs 单一岩壁），纯几何规划会失败，但完全端到端又缺乏可解释性和样本效率。**分层架构**是最优解。

---

## 一、四个选项的深度对比

### 选项1：直接语义替换（❌ 不可行）

**为什么攀岩的假设在家庭场景失效**

```python
# 攀岩的核心假设
climbing_assumptions = {
    '环境': '刚性岩壁，静态不变',
    '接触': '4-6个离散点接触',
    '目标': '单一目标（到达顶部）',
    '约束': '摩擦锥 + 支撑多边形（凸优化可解）',
    '时间': '准静态（可忽略动力学）',
}

# 家庭场景的现实
household_reality = {
    '环境': '混杂场景：刚性桌面 + 柔性织物 + 流体',
    '接触': '连续面接触（擦拭）+ 包络抓握 + 滚动',
    '目标': '多目标序列（拿杯子→擦桌→放回→开门）',
    '约束': '物体功能约束（杯口朝上）+ 人体安全 + 美学偏好',
    '时间': '动态交互（液体晃动、门反弹）',
}

# 差异导致的技术鸿沟
gaps = {
    '接触表征': '离散岩点 → 连续曲面 (需要SDF/mesh碰撞)',
    '平衡检查': '重力中心 → 物体稳定性 + 形变 (需要FEM)',
    '可达性': '关节极限 → 避障路径 (需要RRT/优化)',
    '成功定义': '到达目标 → 功能性完成（杯子没洒、桌子擦干净）',
}
```

**如果强行复用会发生什么**
```python
# 尝试用攀岩框架抓咖啡杯
stance = Stance(holds=(
    'thumb_on_handle',   # ✅ 可以表征
    'fingers_on_cup_body',  # ✅ 可以
    None, None  # 只用两指
))

# 问题1：杯子不是刚性岩点
equilibrium_check(stance)  # ❌ 忽略了：杯内液体晃动、杯子可能滑动

# 问题2：接触面不是点
friction_cone_check(stance)  # ❌ 手指-杯子是面接触，摩擦锥简化过度

# 问题3：没有功能约束
if plan_success(stance):  # ✅ 平衡稳定
    # 但结果：杯口朝下，咖啡洒了 ❌
```

**结论**：直接复用会导致系统在简单任务上勉强工作（抓刚性物体），但无法处理家庭场景的80%长尾任务。

---

### 选项2：扩展物理模型（✅ 推荐核心）

**需要扩展的关键模块**

#### 2.1 接触力建模扩展

```python
# 原始：单点接触 + 摩擦锥
class PointContactModel:
    def check_feasibility(self, normal_force, tangent_force, mu):
        return np.linalg.norm(tangent_force) <= mu * normal_force

# 扩展1：面接触（擦拭任务必需）
class SurfaceContactModel:
    """
    适用于：手掌压桌面、抹布擦拭、吸盘吸附
    """
    def __init__(self, contact_mesh):
        self.mesh = contact_mesh
        # 压力分布：P(x,y) = 压强场
        self.pressure_field = self._init_pressure_field()
    
    def compute_wrench(self, pressure_distribution):
        """
        从压力分布积分出等效力-力矩
        """
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        
        for triangle in self.mesh.triangles:
            area = triangle.area
            normal = triangle.normal
            center = triangle.centroid
            
            # 法向力
            p = pressure_distribution(center)
            f_local = p * area * normal
            total_force += f_local
            
            # 力矩（相对参考点）
            total_torque += np.cross(center - self.ref_point, f_local)
        
        return total_force, total_torque
    
    def check_detachment(self, pressure_dist, surface_tension):
        """
        检查是否脱离（考虑粘附力）
        """
        min_pressure = np.min(pressure_dist)
        return min_pressure > -surface_tension  # 负压表示拉力

# 扩展2：软接触（抓柔性物体）
class SoftContactModel:
    """
    适用于：抓海绵、衣物、食物
    基于Hertz接触理论
    """
    def __init__(self, E_object, E_gripper, poisson=0.3):
        # 材料参数
        self.E_star = 1.0 / ((1-poisson**2)/E_object + 
                             (1-poisson**2)/E_gripper)
    
    def penetration_depth(self, force, radius):
        """
        接触深度与力的关系（非线性）
        """
        return (force / (self.E_star * radius**0.5)) ** (2/3)
    
    def contact_area(self, force, radius):
        """
        接触面积随力增大而增大
        """
        delta = self.penetration_depth(force, radius)
        return np.pi * (2 * radius * delta - delta**2)

# 扩展3：滚动接触（推物体）
class RollingContactModel:
    """
    适用于：推箱子、滚球
    """
    def rolling_resistance(self, normal_force, radius, mu_roll):
        """
        滚动阻力矩
        """
        return mu_roll * normal_force * radius
    
    def no_slip_condition(self, v_contact, omega, radius):
        """
        纯滚动条件：v = ω × r
        """
        return np.allclose(v_contact, np.cross(omega, radius))
```

#### 2.2 可变形物体建模

```python
class DeformableObject:
    """
    处理毛巾、海绵、食物等
    """
    def __init__(self, rest_mesh, stiffness_matrix):
        self.rest_shape = rest_mesh
        self.K = stiffness_matrix  # 刚度矩阵（FEM）
        self.current_shape = rest_mesh.copy()
    
    def apply_grasp_force(self, contact_points, forces):
        """
        计算抓取导致的形变
        使用简化的线性FEM
        """
        # 组装载荷向量
        F_ext = self._assemble_load_vector(contact_points, forces)
        
        # 求解平衡方程：K u = F
        displacements = np.linalg.solve(self.K, F_ext)
        
        # 更新形状
        self.current_shape = self.rest_shape + displacements
        
        return self.current_shape
    
    def grasp_quality(self, contact_points):
        """
        评估抓取质量：考虑形变后的接触稳定性
        """
        deformed = self.apply_grasp_force(contact_points, forces)
        
        # 检查：是否仍在摩擦锥内？
        contact_normals = self._compute_deformed_normals(deformed)
        
        # 形变过大 → 物体可能滑出
        max_deformation = np.max(np.linalg.norm(displacements, axis=1))
        if max_deformation > 0.1 * self.characteristic_length:
            return 0.0  # 不可行
        
        return self._compute_ferrari_canny_metric(contact_points, 
                                                   contact_normals)

# 真实案例：抓取毛巾
towel = DeformableObject(
    rest_mesh=load_mesh('towel.obj'),
    stiffness_matrix=compute_fem_stiffness(E=1e5, nu=0.3)  # 柔软材料
)

# MCTS扩展：搜索时考虑形变
class DeformableMCTS(ClimbingMCTS):
    def get_actions(self, state):
        # 原攀岩：枚举离散岩点
        # 现在：需要采样连续接触位置（毛巾表面任意点）
        candidate_contacts = sample_surface_points(
            towel.current_shape, 
            n_samples=50
        )
        
        actions = []
        for contacts in candidate_contacts:
            # 模拟抓取后的形变
            deformed = towel.apply_grasp_force(contacts, nominal_force)
            
            # 检查可行性
            if self._check_stable_grasp(deformed):
                actions.append(contacts)
        
        return actions
```

#### 2.3 人体安全约束（关键区别）

```python
class HumanSafetyChecker:
    """
    家庭机器人必须保证人类安全
    攀岩机器人不需要考虑（无人区域）
    """
    def __init__(self):
        # ISO/TS 15066: 人体疼痛阈值
        self.pain_thresholds = {
            'head': 130,      # 单位：N
            'torso': 140,
            'arm': 150,
            'hand': 140,
        }
        
        # 速度限制（功率限速）
        self.max_speed = 1.5  # m/s（接近人时降速）
        
        # 禁区：人体周围的缓冲区
        self.safety_distance = 0.3  # m
    
    def check_trajectory_safety(self, traj, human_pose):
        """
        规划后验证：轨迹是否安全
        """
        for t, state in enumerate(traj):
            robot_pos = state.end_effector_pos
            
            # 1. 距离检查
            min_dist = self._min_distance_to_human(robot_pos, human_pose)
            if min_dist < self.safety_distance:
                return False, f"Too close at t={t}"
            
            # 2. 速度检查（动能限制）
            v = state.velocity
            if np.linalg.norm(v) > self.max_speed:
                return False, f"Overspeed at t={t}"
            
            # 3. 碰撞预测（提前1秒）
            predicted_collision = self._predict_collision(
                robot_traj=traj[t:t+10],
                human_motion_model=self.human_predictor
            )
            if predicted_collision:
                return False, "Collision risk detected"
        
        return True, "Safe"
    
    def compute_safe_workspace(self, human_pose):
        """
        动态计算机器人允许活动的区域
        """
        # 人体周围的禁区（膨胀包围盒）
        forbidden_zone = self._inflate_human_bbox(human_pose, 
                                                   margin=self.safety_distance)
        
        # 可用工作空间 = 全局空间 - 禁区
        safe_zone = self.global_workspace.subtract(forbidden_zone)
        
        return safe_zone

# 集成到MCTS
class SafeHouseholdMCTS(ClimbingMCTS):
    def __init__(self, *args, human_tracker, **kwargs):
        super().__init__(*args, **kwargs)
        self.safety_checker = HumanSafetyChecker()
        self.human_tracker = human_tracker
    
    def simulate(self, state, depth=20):
        # 原攀岩：只检查机器人自身稳定性
        # 现在：每步都检查人体安全
        
        current = state
        for step in range(depth):
            action = self._sample_action(current)
            next_state = self.apply_action(current, action)
            
            # 关键：人体实时位置
            human_pose = self.human_tracker.get_current_pose()
            
            # 安全检查
            safe, msg = self.safety_checker.check_trajectory_safety(
                [next_state], human_pose
            )
            if not safe:
                return -10.0  # 严厉惩罚不安全动作
            
            current = next_state
        
        return self.heuristic_value(current)
```

#### 2.4 功能性约束（家庭场景特有）

```python
class FunctionalConstraints:
    """
    物体必须保持功能性（攀岩没有这个概念）
    """
    
    @staticmethod
    def cup_upright_constraint(grasp_pose):
        """
        杯子必须口朝上（防止洒液体）
        """
        cup_up_vector = grasp_pose.rotation @ np.array([0, 0, 1])
        gravity_vector = np.array([0, 0, -1])
        
        # 允许±30度倾斜
        angle = np.arccos(np.dot(cup_up_vector, -gravity_vector))
        return angle < np.deg2rad(30)
    
    @staticmethod
    def door_handle_orientation(approach_pose):
        """
        门把手必须从正确方向抓取
        """
        # 门把手的旋转轴
        handle_axis = approach_pose.get_rotation_axis()
        
        # 手指必须垂直于旋转轴（才能施加力矩）
        gripper_axis = approach_pose.rotation @ np.array([0, 1, 0])
        
        perpendicularity = np.abs(np.dot(handle_axis, gripper_axis))
        return perpendicularity < 0.2  # 接近垂直
    
    @staticmethod
    def book_spine_protection(grasp_force):
        """
        抓书时避免损坏书脊
        """
        # 书脊区域的压强限制
        MAX_PRESSURE = 5000  # Pa
        contact_area = 0.001  # m²
        
        return grasp_force / contact_area < MAX_PRESSURE

# 扩展MCTS的奖励函数
def household_reward(state, action):
    # 攀岩：只看高度和稳定性
    # climbing_reward = height * 0.7 + stability * 0.3
    
    # 家庭：多目标优化
    reward = {
        'task_progress': 0.4,     # 完成任务进度
        'safety': 0.3,            # 人体安全
        'efficiency': 0.15,       # 能耗/时间
        'functional': 0.1,        # 功能性约束满足
        'aesthetic': 0.05,        # 美学（例如摆放整齐）
    }
    
    total = 0
    for key, weight in reward.items():
        total += weight * compute_metric(state, action, key)
    
    return total
```

---

### 选项3：混合学习式模型（✅ 高级扩展）

**核心思想**：用神经网络学习难以建模的部分，保留几何规划的优势

```python
class HybridContactPlanner:
    """
    分层架构：
    - 高层：符号规划（MCTS/TAMP）
    - 中层：学习的接触模型
    - 低层：运动原语
    """
    def __init__(self):
        # 传统组件
        self.geometric_planner = MCTS(...)
        
        # 学习组件
        self.contact_predictor = LearnedContactModel()
        self.stability_net = StabilityPredictor()
        self.scene_encoder = NeRFSceneEncoder()
    
    def plan(self, scene, task):
        # 1. 场景理解（学习）
        scene_features = self.scene_encoder(scene.rgbd)
        object_affordances = self.affordance_net(scene_features)
        
        # 2. 候选接触点生成（混合）
        geometric_candidates = self._sample_surface_contacts(scene.mesh)
        learned_candidates = self.contact_predictor(
            scene_features, task_embedding
        )
        all_candidates = geometric_candidates + learned_candidates
        
        # 3. 可行性检查（混合）
        for contact in all_candidates:
            # 几何约束（快速剔除）
            if not self._geometric_feasibility(contact):
                continue
            
            # 学习的稳定性（精细判断）
            stability_score = self.stability_net(
                scene_features, contact, object_mesh
            )
            if stability_score > 0.8:
                feasible_contacts.append(contact)
        
        # 4. 序列搜索（传统MCTS）
        sequence = self.geometric_planner.search(
            init_state, 
            feasible_contacts,
            heuristic=self.stability_net  # 用学习的价值函数引导
        )
        
        return sequence
```

**学习的接触模型实现**

```python
class LearnedContactModel(nn.Module):
    """
    从数据学习：给定物体+任务 → 预测最佳接触位置
    替代手工设计的启发式
    """
    def __init__(self):
        # 输入：物体点云 + 任务嵌入
        self.pointnet = PointNetEncoder(out_dim=256)
        self.task_encoder = nn.Embedding(num_tasks=50, embedding_dim=64)
        
        # 输出：接触位置的热图（在物体表面）
        self.decoder = nn.Sequential(
            nn.Linear(256 + 64, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1024),  # N个候选点的得分
        )
    
    def forward(self, object_pcd, task_id):
        """
        object_pcd: (N, 3) 点云
        task_id: int (0=grasp, 1=push, 2=wipe, ...)
        
        返回: (N,) 每个点作为接触点的得分
        """
        obj_feat = self.pointnet(object_pcd)
        task_feat = self.task_encoder(task_id)
        
        combined = torch.cat([obj_feat, task_feat], dim=-1)
        scores = self.decoder(combined)
        
        return torch.softmax(scores, dim=-1)
    
    def sample_contacts(self, object_pcd, task_id, n_samples=10):
        """
        根据学习的分布采样接触点
        """
        scores = self.forward(object_pcd, task_id)
        
        # 按概率采样（探索-利用平衡）
        indices = torch.multinomial(scores, n_samples, replacement=False)
        
        return object_pcd[indices]

# 训练数据来源
training_data = {
    'human_demo': '遥操作演示（标注成功的接触点）',
    'sim_success': '仿真中成功的轨迹',
    'self_supervised': '机器人自主探索（成功→正样本）',
}

# 训练损失
def train_contact_model(model, data):
    for batch in data:
        obj_pcd = batch['object_pointcloud']
        task = batch['task_id']
        success_contacts = batch['human_labeled_contacts']  # ground truth
        
        # 预测
        pred_scores = model(obj_pcd, task)
        
        # 监督信号：成功的接触点应该有高分
        loss = F.cross_entropy(
            pred_scores,
            success_contacts  # one-hot编码
        )
        
        loss.backward()
        optimizer.step()
```

**NeRF场景表征（前沿方向）**

```python
class NeRFSceneEncoder:
    """
    用NeRF/3DGS编码场景 → 提供连续的几何和外观查询
    优势：处理部分观测、透明物体、镜面反射
    """
    def __init__(self):
        self.nerf = InstantNGP(...)  # Nvidia的快速NeRF
        self.fitted = False
    
    def fit(self, rgbd_sequence):
        """
        从多视角RGBD拟合场景
        """
        self.nerf.train(rgbd_sequence, steps=5000)
        self.fitted = True
    
    def query_sdf(self, points):
        """
        查询任意3D点的SDF值（用于碰撞检测）
        """
        density = self.nerf.query_density(points)
        # 密度→SDF的转换
        sdf = -torch.log(density + 1e-6)
        return sdf
    
    def query_normal(self, points):
        """
        查询表面法线（用于接触规划）
        """
        # 对SDF求梯度
        points.requires_grad = True
        sdf = self.query_sdf(points)
        normals = torch.autograd.grad(sdf.sum(), points)[0]
        return F.normalize(normals, dim=-1)
    
    def sample_grasp_poses(self, n_samples=100):
        """
        在场景中采样可行的抓取位姿
        """
        # 在3D空间随机采样点
        candidates = torch.rand(n_samples, 3) * scene_bounds
        
        # 过滤：必须在物体表面附近
        sdf_values = self.query_sdf(candidates)
        on_surface = (sdf_values.abs() < 0.01)
        
        surface_points = candidates[on_surface]
        normals = self.query_normal(surface_points)
        
        # 构造抓取姿态（法线方向为接近向量）
        grasp_poses = []
        for point, normal in zip(surface_points, normals):
            pose = self._construct_grasp_from_normal(point, normal)
            grasp_poses.append(pose)
        
        return grasp_poses

# 与MCTS集成
class NeRFGuidedMCTS(ClimbingMC
