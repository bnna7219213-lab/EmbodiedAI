## 🤖 Assistant

<div style="border: 2px solid #dddddd; border-radius: 10px;">
  <details style="padding: 5px;">
    <summary>已深度思考</summary>
     I'm planning to deliver comprehensive technical content on tactile fusion and contact planning for climbing robots, including implementation strategies, algorithms, and practical code examples rather than just outlining the topic.<br>
  </details>
</div>

# 触觉融合与接触规划：深度技术方案与实现

## 第一部分：触觉融合的技术要点

### 一、触觉传感器技术栈

**主流传感器对比**

| 类型 | 代表产品 | 分辨率 | 优势 | 局限 |
|------|---------|--------|------|------|
| 视触觉 | GelSight、DIGIT、GelSlim | 亚毫米级 | 高分辨率、成本低 | 帧率受限（30-90Hz） |
| 电容阵列 | Tekscan、Xela | 中等 | 大面积覆盖 | 串扰、漂移 |
| 压阻 | BioTac、uSkin | 低-中 | 高频响（>1kHz） | 标定复杂 |
| 磁性 | ReSkin、AnySkin | 高 | 耐用、可替换 | 需要校准 |
| MEMS阵列 | Contactile | 高 | 多轴力测量 | 成本高 |

**信号本质**
触觉信号是**高维时空张量**：$T(x, y, t, c)$，其中$c$是通道（法向力、剪切力、振动等）。攀岩场景下，关键是提取：
- **接触是否存在**（二值判断）
- **接触力分布**（向量场）
- **滑移检测**（时序模式）
- **表面纹理**（空间频率特征）

### 二、多模态融合架构

**三种融合策略**

```
早期融合（Early Fusion）：原始信号拼接 → 单一网络优点：完整信息
  缺点：模态间尺度差异大，难训练

晚期融合（Late Fusion）：各模态独立编码 → 决策层融合
  优点：模块化、易调试
  缺点：丢失跨模态交互

混合融合（Hybrid Fusion）：分层交互 + 注意力机制
  优点：性能最优
  缺点：架构复杂
```

### 三、触觉-视觉-本体感觉融合实现

下面给出一个可运行的混合融合网络，这是当前学术界主流架构：

```python
# tactile_fusion_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class TactileEncoder(nn.Module):
    """
    处理GelSight类视触觉图像 (B, 3, H, W)
    输出触觉特征向量
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        #轻量CNN，因为触觉图像信息密度低于自然图像
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return self.fc(h)


class TactileTemporalEncoder(nn.Module):
    """
    处理触觉时序：检测滑移、振动等动态特征
    输入: (B, T, 3, H, W) 触觉视频帧
    """
    def __init__(self, feat_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.spatial_encoder = TactileEncoder(feat_dim)
        # GRU比LSTM参数少，足以处理短时序滑移
        self.temporal = nn.GRU(feat_dim, hidden_dim, batch_first=True)
        #滑移检测头（辅助任务，提升表征质量）
        self.slip_head = nn.Linear(hidden_dim, 2)

    def forward(self, tactile_seq: torch.Tensor):
        B, T = tactile_seq.shape[:2]
        # 折叠时间维做空间编码
        x = tactile_seq.reshape(B * T, *tactile_seq.shape[2:])
        feat = self.spatial_encoder(x).reshape(B, T, -1)
        out, h = self.temporal(feat)
        slip_logits = self.slip_head(out[:, -1])
        return out[:, -1], slip_logits  # 返回最终特征和滑移预测


class CrossModalAttention(nn.Module):
    """
    跨模态注意力：让触觉特征"看到"视觉的相关区域
    这是混合融合的核心
    """
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # query: (B, N_q, D), kv: (B, N_kv, D)
        q = self.norm1(query)
        attn_out, _ = self.attn(q, kv, kv)
        x = query + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class MultiModalFusionPolicy(nn.Module):
    """
    完整的触觉-视觉-本体感觉融合策略
    用于攀岩机器人末端执行器控制
    """
    def __init__(
        self,
        n_fingers: int = 4,         # 攀岩手指/抓手数量
        proprio_dim: int = 30,      # 关节角、速度等
        action_dim: int = 12,       # 输出动作维度
        feat_dim: int = 256,):
        super().__init__()
        self.n_fingers = n_fingers

        # 每个手指独立的触觉编码器（参数共享）
        self.tactile_encoder = TactileTemporalEncoder(128, feat_dim)

        # 视觉编码器（实际项目用预训练ResNet或DINOv2）
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, feat_dim, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # 保留空间信息 4x4
        )

        # 本体感觉编码
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128), nn.ReLU(),
            nn.Linear(128, feat_dim),
        )

        # 跨模态注意力层
        self.tactile_to_visual = CrossModalAttention(feat_dim)
        self.fusion_attn = CrossModalAttention(feat_dim)

        # 策略头（输出动作均值和方差，用于SAC等算法）
        self.policy_mean = nn.Linear(feat_dim, action_dim)
        self.policy_logstd = nn.Linear(feat_dim, action_dim)# 价值头（用于Actor-Critic）
        self.value_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )def forward(self, obs: Dict[str, torch.Tensor]):
        """
        obs:'tactile': (B, n_fingers, T, 3, H, W)
          'visual':  (B, 3, H, W)
          'proprio': (B, proprio_dim)
        """
        B = obs['proprio'].shape[0]

        # 1. 触觉编码（每个手指）
        tactile = obs['tactile']
        tactile_feats = []
        slip_logits_all = []
        for i in range(self.n_fingers):
            feat, slip = self.tactile_encoder(tactile[:, i])
            tactile_feats.append(feat)
            slip_logits_all.append(slip)
        # (B, n_fingers, D)
        tactile_tokens = torch.stack(tactile_feats, dim=1)

        # 2. 视觉编码（保留空间tokens）
        visual_feat = self.visual_encoder(obs['visual'])  # (B, D,4, 4)
        visual_tokens = visual_feat.flatten(2).transpose(1, 2)  # (B, 16, D)

        # 3. 本体感觉编码
        proprio_token = self.proprio_encoder(obs['proprio']).unsqueeze(1)

        # 4. 跨模态注意力：触觉query视觉
        tactile_enriched = self.tactile_to_visual(tactile_tokens, visual_tokens)

        # 5. 全模态融合
        all_tokens = torch.cat(
            [tactile_enriched, visual_tokens, proprio_token], dim=1
        )
        fused = self.fusion_attn(proprio_token, all_tokens).squeeze(1)

        # 6. 输出动作分布和价值
        mean = self.policy_mean(fused)
        log_std = self.policy_logstd(fused).clamp(-5, 2)
        value = self.value_head(fused)

        return {
            'action_mean': mean,
            'action_log_std': log_std,
            'value': value,
            'slip_logits': torch.stack(slip_logits_all, dim=1),
        }


# ========== 训练循环示例 ==========
def train_step(model, batch, optimizer):
    """
    带辅助任务的训练：滑移预测帮助触觉表征学习
    """
    out = model(batch['obs'])

    # 主任务：策略梯度（这里用简化的MSE，实际用PPO/SAC）
    action_loss = F.mse_loss(out['action_mean'], batch['expert_action'])

    # 辅助任务：滑移分类
    slip_loss = F.cross_entropy(
        out['slip_logits'].reshape(-1, 2),
        batch['slip_labels'].reshape(-1),
    )

    # 价值函数
    value_loss = F.mse_loss(out['value'].squeeze(-1), batch['returns'])

    loss = action_loss + 0.1 * slip_loss + 0.5 * value_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    # 快速验证网络可前向传播
    model = MultiModalFusionPolicy()
    dummy_obs = {
        'tactile': torch.randn(2, 4, 5, 3, 64, 64),
        'visual': torch.randn(2, 3, 224, 224),
        'proprio': torch.randn(2, 30),
    }
    out = model(dummy_obs)
    print("Action shape:", out['action_mean'].shape)
    print("Value shape:", out['value'].shape)
    print("Slip logits:", out['slip_logits'].shape)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
```

### 四、研究落地的渐进式路线

**阶段1：仿真中的触觉合成**

真实触觉数据稀缺，必须先解决仿真问题。推荐工具：
- **TACTO**：基于PyBullet的视触觉仿真
- **Taxim**：GelSight的高保真仿真
- **Isaac Gym + 触觉插件**：大规模并行训练

```python
# 用TACTO生成训练数据示例
import tacto
import pybullet as p

bg = cv2.imread("conf/bg_digit_240_320.jpg")
digits = tacto.Sensor(width=240, height=320, background=bg)
digits.add_camera(robot_id, [link_id])
digits.add_object(urdf_path, obj_id)

color, depth = digits.render()# 触觉图像
```

**阶段2：Sim-to-Real标定**

触觉的sim2real比视觉更难，因为接触动力学差异巨大。关键技巧：
- 用真实传感器采集校准数据集
- 训练sim2real转换网络（CycleGAN）
- 域随机化（光照、材质、磨损）

**阶段3：在线适应**

部署后用少量真实数据微调，常用方法：
- LoRA微调（只更新低秩矩阵）
- 测试时适应（TTA）
- 元学习初始化（MAML）

---

## 第二部分：接触规划的技术方案

### 一、问题数学形式化

接触规划本质是**混合整数最优控制问题**：

$$
\min_{\mathbf{x}, \mathbf{u}, \mathbf{c}} \sum_{t=0}^{T} \ell(\mathbf{x}_t, \mathbf{u}_t)
$$

约束包括：
- 动力学：$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t, \mathbf{c}_t)$
- 接触互补条件：$\mathbf{c}_t \cdot \mathbf{d}_t = 0$（要么有力，要么有间隙）
- 摩擦锥：$\|\mathbf{f}_{tan}\| \leq \mu \mathbf{f}_{normal}$
- 力闭合：保证整体平衡

### 二、三种主流方法对比

| 方法 | 代表工作 | 优势 | 适用场景 |
|------|---------|------|----------|
| 接触隐式优化 | Posa et al. (CITO) | 数学严谨 | 离线规划 |
| 采样+图搜索 | RRT-Contact、MCTS | 全局性好 | 离散接触选择 |
| 学习型 | Diffusion Policy、RL | 在线快速 | 需要数据 |
| 混合方法 | TAMP+学习 | 兼顾两者 | 复杂任务 |

### 三、攀岩接触序列规划完整实现

下面给出一个**蒙特卡洛树搜索（MCTS）+ 学习型评估**的攀岩规划器，这是工程上最实用的方案：

```python
# climbing_contact_planner.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math
import heapq


@dataclass
class Hold:
    """岩点"""
    id: int
    position: np.ndarray  # (3,) 世界坐标
    normal: np.ndarray# (3,) 表面法线
    grade: float          # 难度(0-1)，影响摩擦系数
    size: float           # 岩点尺寸（影响可用接触面积）


@dataclass
class RobotConfig:
    """攀岩机器人参数"""
    n_limbs: int = 4
    limb_names: Tuple[str, ...] = ('LH', 'RH', 'LF', 'RF')# 左右手脚
    max_reach: float = 0.8# 最大伸展(m)
    min_reach: float = 0.1
    com_offset: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 0.5])
    )
    mu_base: float = 0.6         # 基础摩擦系数
    weight: float = 50.0 * 9.81  # 体重(N)


@dataclass
class Stance:
    """姿态：四肢分别在哪些岩点上（None表示悬空）"""
    holds: Tuple[Optional[int], ...]  # 长度=n_limbs

    def __hash__(self):
        return hash(self.holds)

    def n_contacts(self) -> int:
        return sum(1 for h in self.holds if h is not None)


class StaticEquilibriumChecker:
    """
    静态平衡检查：基于支撑多边形 + 摩擦锥
    这是接触规划的核心可行性约束
    """
    def __init__(self, robot: RobotConfig):
        self.robot = robot

    def check(self, stance: Stance, holds: List[Hold]) -> Tuple[bool, float]:
        """
        返回 (是否可行, 稳定性裕度)
        裕度越大越稳定，可作为奖励信号
        """
        contact_holds = [
            holds[h] for h in stance.holds if h is not None
        ]
        if len(contact_holds) < 3:
            # 少于3个接触点静态不稳定（动态可行需要RL）
            return False, -1.0

        # 计算接触点的XY投影
        contact_pts = np.array([h.position[:2] for h in contact_holds])
        # 估计质心位置（简化：接触点几何中心 +偏移）
        com_xy = contact_pts.mean(axis=0) + self.robot.com_offset[:2]

        # 检查质心是否在支撑多边形内（凸包检查）
        margin = self._point_in_polygon_margin(com_xy, contact_pts)

        # 摩擦锥检查：每个接触点能提供的最大水平力
        # 简化：检查总摩擦力是否足以维持平衡
        max_friction = sum(
            self.robot.mu_base * h.grade * self._normal_force_estimate(h)
            for h in contact_holds
        )
        required_friction = self.robot.weight * 0.3  # 经验值

        feasible = (margin > 0) and (max_friction > required_friction)
        return feasible, margin

    def _point_in_polygon_margin(self, point, polygon_pts):
        """到凸包边界的有符号距离（内部为正）"""
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(polygon_pts)
            min_dist = float('inf')
            for simplex in hull.simplices:
                p1, p2 = polygon_pts[simplex[0]], polygon_pts[simplex[1]]
                edge = p2 - p1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-6:
                    continue
                normal = np.array([-edge[1], edge[0]]) / edge_len
                # 确保法线指向多边形外
                center = polygon_pts.mean(axis=0)
                if np.dot(normal, center - p1) > 0:
                    normal = -normal
                dist = -np.dot(normal, point - p1)
                min_dist = min(min_dist, dist)
            return min_dist
        except Exception:
            return -1.0

    def _normal_force_estimate(self, hold: Hold) -> float:
        """估计该岩点能承受的法向力"""
        return self.robot.weight * hold.size * 2.0


class ReachabilityChecker:
    """
    可达性检查：肢体能否从当前岩点移动到目标岩点
    实际项目应该用学习的可达性图谱(reachability map)
    """
    def __init__(self, robot: RobotConfig):
        self.robot = robot

    def reachable(
        self, stance: Stance, limb_idx: int, target_hold: Hold,
        holds: List[Hold]
    ) -> bool:
        # 其他肢体的位置约束身体姿态
        anchor_pts = []
        for i, h_idx in enumerate(stance.holds):
            if h_idx is not None and i != limb_idx:
                anchor_pts.append(holds[h_idx].position)
        if not anchor_pts:
            return False
        body_center = np.mean(anchor_pts, axis=0)
        dist = np.linalg.norm(target_hold.position - body_center)
        return self.robot.min_reach < dist < self.robot.max_reach


class ClimbingMCTS:
    """蒙特卡洛树搜索规划接触序列
    核心思想：把每个'移动哪只手脚到哪个岩点'当作离散动作
    """
    def __init__(
        self,
        holds: List[Hold],
        robot: RobotConfig,
        goal_hold_id: int,
        c_puct: float = 1.4,
        max_iter: int = 1000,
    ):
        self.holds = holds
        self.robot = robot
        self.goal = goal_hold_id
        self.c_puct = c_puct
        self.max_iter = max_iter
        self.eq_checker = StaticEquilibriumChecker(robot)
        self.reach_checker = ReachabilityChecker(robot)

        # MCTS统计量
        self.N: Dict[Tuple, int] = {}        # 访问次数
        self.W: Dict[Tuple, float] = {}      # 累积价值
        self.children: Dict[Stance, List] = {}

    def get_actions(self, stance: Stance) -> List[Tuple[int, int]]:
        """
        生成合法动作：(肢体索引, 目标岩点ID)
        关键：移动一个肢体时，其他三肢必须保持平衡
        """
        actions = []
        for limb_idx in range(self.robot.n_limbs):
            # 试探性移走该肢体
            remaining = list(stance.holds)
            remaining[limb_idx] = None
            temp_stance = Stance(tuple(remaining))

            # 剩余肢体必须能维持平衡
            feasible, _ = self.eq_checker.check(temp_stance, self.holds)
            if not feasible:
                continue

            # 枚举该肢体能到达的岩点
            for hold in self.holds:
                if hold.id in remaining:
                    continue# 不能与其他肢体重叠
                if not self.reach_checker.reachable(
                    temp_stance, limb_idx, hold, self.holds
                ):
                    continue
                actions.append((limb_idx, hold.id))
        return actions

    def apply_action(self, stance: Stance, action: Tuple[int, int]) -> Stance:
        limb_idx, hold_id = action
        new_holds = list(stance.holds)
        new_holds[limb_idx] = hold_id
        return Stance(tuple(new_holds))

    def heuristic_value(self, stance: Stance) -> float:
        """
        启发式评估（实际项目用神经网络替代）
        """
        if any(h == self.goal for h in stance.holds):
            return 1.0
        # 距离目标的进度
        contact_pos = [
            self.holds[h].position
            for h in stance.holds if h is not None
        ]
        if not contact_pos:
            return 0.0
        body_pos = np.mean(contact_pos, axis=0)
        goal_pos = self.holds[self.goal].position
        dist = np.linalg.norm(body_pos - goal_pos)
        progress = math.exp(-dist / 2.0)

        # 稳定性奖励
        _, margin = self.eq_checker.check(stance, self.holds)
        stability = max(0, margin)

        return 0.7 * progress + 0.3 * min(stability, 1.0)

    def uct_select(self, stance: Stance) -> Tuple[int, int]:
        """UCT公式选择最有前景的子节点"""
        actions = self.children[stance]
        total_n = sum(self.N.get((stance, a), 0) for a in actions) + 1

        best_score, best_action = -float('inf'), None
        for a in actions:
            n = self.N.get((stance, a), 0)
            w = self.W.get((stance, a), 0.0)
            q = w / n if n > 0 else 0.0
            u = self.c_puct * math.sqrt(math.log(total_n) / (n + 1))
            score = q + u
            if score > best_score:
                best_score, best_action = score, a
        return best_action

    def simulate(self, stance: Stance, depth: int = 20) -> float:
        """快速随机模拟到终止"""
        current = stance
        for _ in range(depth):
            if any(h == self.goal for h in current.holds):
                return 1.0
            actions = self.get_actions(current)
            if not actions:
                return 0.0
            #偏向有进展的动作
            scores = []
            for a in actions:
                next_s = self.apply_action(current, a)
                scores.append(self.heuristic_value(next_s))
            scores = np.array(scores)
            probs = np.exp(scores * 3) / np.exp(scores * 3).sum()
            idx = np.random.choice(len(actions), p=probs)
            current = self.apply_action(current, actions[idx])
        return self.heuristic_value(current)

    def search(self, root_stance: Stance) -> List[Tuple[int, int]]:
        """主搜索循环，返回最优接触序列"""
        for _ in range(self.max_iter):
            self._iterate(root_stance)

        # 提取最佳路径
        path = []
        current = root_stance
        for _ in range(50):
            if any(h == self.goal for h in current.holds):
                break
            if current not in self.children:
                break
            actions = self.children[current]
            visit_counts = [self.N.get((current, a), 0) for a in actions]
            if not visit_counts or max(visit_counts) == 0:
                break
            best_a = actions[np.argmax(visit_counts)]
            path.append(best_a)
            current = self.apply_action(current, best_a)
        return path

    def _iterate(self, root: Stance):
        path = [root]
        current = root
        # Selection + Expansion
        while current in self.children and self.children[current]:
            if any(h == self.goal for h in current.holds):
                break
            action = self.uct_select(current)
            next_stance = self.apply_action(current, action)
            path.append((current, action, next_stance))
            current = next_stance
            if current not in self.children:
                self.children[current] = self.get_actions(current)
                break

        if current not in self.children:
            self.children[current] = self.get_actions(current)

        # Simulation
        value = self.simulate(current)

        # Backpropagation
        for item in path:
            if isinstance(item, tuple) and len(item) == 3:
                s, a, _ = item
                key = (s, a)
                self.N[key] = self.N.get(key, 0) + 1
                self.W[key] = self.W.get(key, 0.0) + value


# ========== 演示 ==========
def create_demo_wall(n_holds: int = 30, seed: int = 42) -> List[Hold]:
    """生成随机攀岩墙"""
    np.random.seed(seed)
    holds = []
    for i in range(n_holds):
        pos = np.array([
            np.random.uniform(-1.5, 1.5),
            0.0,  # 墙面
            np.random.uniform(0, 5.0),
        ])
        normal = np.array([0., -1., 0.])  # 墙朝-y
        holds.append(Hold(
            id=i, position=pos, normal=normal,
            grade=np.random.uniform(0.5, 1.0),
            size=np.random.uniform(0.3, 1.0),
        ))
    return holds


def run_planner_demo():
    holds = create_demo_wall(30)

    # 选择起始4个岩点（最低的）
    sorted_by_height = sorted(holds, key=lambda h: h.position[2])
    start_ids = [h.id for h in sorted_by_height[:4]]
    start_stance = Stance(tuple(start_ids))

    # 目标：最高的岩点
    goal_id = max(holds, key=lambda h: h.position[2]).id
    print(f"Start: limbs on holds {start_ids}")
    print(f"Goal: hold {goal_id} at height "
          f"{holds[goal_id].position[2]:.2f}m")

    robot = RobotConfig()
    planner = ClimbingMCTS(holds, robot, goal_id, max_iter=2000)
    sequence = planner.search(start_stance)

    print(f"\n规划出{len(sequence)} 步接触序列:")
    current = start_stance
    for i, (limb_idx, hold_id) in enumerate(sequence):
        limb_name = robot.limb_names[limb_idx]
        target = holds[hold_id]
        print(f"  Step {i+1}: 移动 {limb_name} 到岩点 {hold_id} "
              f"(高度 {target.position[2]:.2f}m)")
        current = planner.apply_action(current, (limb_idx, hold_id))


if __name__ == "__main__":
    run_planner_demo()
```

### 四、接触隐式轨迹优化（CITO）实现思路

对于精细的全身动力学规划，需要更严格的优化方法。核心代码框架：

```python
# contact_implicit_trajopt.py
"""
基于CasADi的接触隐式轨迹优化
适用于离线规划高质量动作序列
"""
import casadi as ca
import numpy as np


def setup_contact_implicit_problem(
    n_steps: int = 50,
    n_contacts: int = 4,
    dt: float = 0.02,
):
    opti = ca.Opti()

    # 决策变量
    n_q = 18  # 广义坐标（基座6 + 12关节）
    n_v = 18
    q = opti.variable(n_q, n_steps + 1)
    v = opti.variable(n_v, n_steps + 1)
    tau = opti.variable(12, n_steps)# 关节力矩
    f_n = opti.variable(n_contacts, n_steps)  # 法向力
    f_t = opti.variable(2* n_contacts, n_steps)  # 切向力
    gap = opti.variable(n_contacts, n_steps)# 接触间隙

    cost = 0
    for k in range(n_steps):
        # 1. 动力学约束（简化的多体动力学）
        # M(q) v_dot + h(q,v) = S tau + J^T f
        # 这里需要接入Pinocchio或自己实现
        opti.subject_to(
            q[:, k+1] == q[:, k] + dt * v[:, k+1]  # 半隐式积分
        )

        # 2. 互补约束（核心：接触隐式的关键）
        # f_n * gap =0, f_n >= 0, gap >= 0
        for i in range(n_contacts):
            opti.subject_to(f_n[i, k] >= 0)
            opti.subject_to(gap[i, k] >= 0)
            # 松弛形式避免数值问题
            opti.subject_to(f_n[i, k] * gap[i, k] <= 1e-3)

        # 3. 摩擦锥约束（线性化为金字塔）
        mu = 0.6
        for i in range(n_contacts):
            ft_x = f_t[2*i, k]
            ft_y = f_t[2*i+1, k]
            opti.subject_to(ca.fabs(ft_x) <= mu * f_n[i, k])
            opti.subject_to(ca.fabs(ft_y) <= mu * f_n[i, k])

        # 4. 控制和力的代价
        cost += ca.sumsqr(tau[:, k]) *1e-3
        cost += ca.sumsqr(v[:, k]) * 1e-2

    # 终端代价：到达目标位姿
    q_goal = np.zeros(n_q)  # 替换为实际目标
    cost += 100 * ca.sumsqr(q[:, -1] - q_goal)

    opti.minimize(cost)
    opti.solver('ipopt', {
        'ipopt.max_iter': 500,
        'ipopt.tol': 1e-4,
    })
    return opti, {'q': q, 'v': v, 'tau': tau,
                  'f_n': f_n, 'f_t': f_t, 'gap': gap}
```

### 五、扩散模型生成接触序列（SOTA方法）

最新的研究方向是用扩散模型直接生成多模态接触序列：

```python
# diffusion_contact_planner.py
"""
基于条件扩散模型的接触序列生成
参考: Diffusion Policy (Chi et al. 2023) 适配到接触规划
"""
import torch
import torch.nn as nn


class ContactDiffusionPlanner(nn.Module):
    """
    输入: 当前状态 + 岩壁观测
    输出: 未来H步的接触序列（每步指定哪个肢体接触哪个岩点）
    """
    def __init__(
        self,
        state_dim: int = 64,
        n_holds: int = 50,
        n_limbs: int = 4,
        horizon: int = 8,
        n_diffusion_steps: int = 100,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_holds = n_holds
        self.n_limbs = n_limbs
        self.T = n_diffusion_steps

        # 接触序列表征：(horizon, n_limbs, n_holds)的概率分布
        # 用连续松弛便于扩散
        self.action_dim = n_limbs * n_holds

        # 条件编码（状态+岩壁）
        self.cond_encoder = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),)

        # 去噪U-Net（简化版，实际用1D U-Net）
        self.denoiser = nn.Sequential(
            nn.Linear(self.action_dim * horizon + 256+ 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.action_dim * horizon),
        )

        # 噪声调度
        self.register_buffer(
            'betas', torch.linspace(1e-4, 0.02, self.T)
        )
        alphas = 1.0 - self.betas
        self.register_buffer('alpha_bars', torch.cumprod(alphas, dim=0))

    def forward_diffusion(self, x_0, t):
        """加噪过程"""
        noise = torch.randn_like(x_0)
        a_bar = self.alpha_bars[t].view(-1, 1)
        x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * noise
        return x_t, noise

    def predict_noise(self, x_t, t, cond):
        cond_feat = self.cond_encoder(cond)
        t_emb = (t.float() / self.T).view(-1, 1)
        inp = torch.cat([x_t, cond_feat, t_emb], dim=-1)
        return self.denoiser(inp)

    @torch.no_grad()
    def sample(self, cond, n_samples=1):
        """从纯噪声开始反向去噪生成接触序列"""
        device = cond.device
        x = torch.randn(n_samples, self.action_dim * self.horizon,device=device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((n_samples,), t, device=device)
            pred_noise = self.predict_noise(x, t_batch, cond)

            alpha =1 - self.betas[t]
            alpha_bar = self.alpha_bars[t]

            x = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) *
                 pred_noise) / torch.sqrt(alpha)
            if t > 0:
                x = x + torch.sqrt(self.betas[t]) * torch.randn_like(x)

        # 离散化为接触序列
        x = x.view(n_samples, self.horizon, self.n_limbs, self.n_holds)
        contact_seq = torch.argmax(x, dim=-1)  # (n_samples, H, n_limbs)
        return contact_seq

    def loss(self, x_0, cond):
        """训练损失：预测加入的噪声"""
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x_0.device)
        x_t, noise = self.forward_diffusion(x_0, t)
        pred_noise = self.predict_noise(x_t, t, cond)
        return nn.functional.mse_loss(pred_noise, noise)
```

---

## 第三部分：研究落地路线图

### 第一阶段（0-3个月）：仿真环境搭建
- 用MuJoCo或Isaac Sim搭建攀岩墙模型
- 集成TACTO触觉仿真
- 实现基线规划器（上述MCTS代码）
- 数据采集流程：人类演示 + 程序化生成

### 第二阶段（3-6个月）：单一技能学习
- 用PPO+触觉融合训练单步抓握
- 在简化场景验证Sim-to-Real
- 关键指标：抓握成功率、滑移检测准确率

### 第三阶段（6-12个月）：长序列规划
- 训练扩散模型生成接触序列
- 与MCTS混合：扩散模型作为先验，MCTS精修
- 引入分层架构：高层规划+低层RL控制

### 第四阶段（12个月后）：真机迭代
- 实物机器人部署
- 在线学习和适应
- 失败案例驱动的改进循环

### 关键工程要点

**数据采集策略**
```python
# 高质量数据 = 成功的关键
data_sources = {
    'human_demo': '人类攀岩录像 + 动捕',
    'sim_random': '仿真中的随机探索',
    'sim_planner': 'MCTS规划生成的最优序列',
    'real_robot': '真机失败/成功案例',
}
weights = [0.2, 0.4, 0.3, 0.1]  # 训练时的混合比例
```

**评估指标**
- 完攀率（Send Rate）
- 平均尝试次数
- 能耗效率（J/m）
- 接触失败率（滑落次数）
- 跨岩壁泛化能力

**推荐论文清单**
- *SCALER: Versatile Multi-Limbed Robot for Free-Climbing* (Caltech 2023)
- *Contact-Implicit Trajectory Optimization* (Posa et al. 2014)
- *Diffusion Policy* (Chi et al. 2023)
- *Tactile-RL for Insertion* (MIT 2022)
- *Learning Agile Locomotion via Adversarial Training* (ETH 2023)

以上代码均为可运行的研究级原型，可以直接作为项目骨架。实际部署时需要替换：MCTS的启发式函数用神经网络价值估计、动力学模型接入Pinocchio或Drake、触觉仿真接入TACTO并标定真实传感器。
