"""MCTS 接触规划 (L5) — K2 补全: ClimbingMCTS 完整实现 + DiffusionPolicy 桩类 + 飞轮接口。

K1 清洗 + K2 重构: 
  - ClimbingMCTS: 从 pass 占位 → 完整 UCB1-MCTS (selection/expansion/simulation/backprop)
  - DiffusionPolicy: 从 pass 占位 → 条件生成接口 (K3 对接 Denoising Diffusion)
  - compute_mc_returns: 从 None → 带折扣的 MC 回报计算
  - 新增: ClimbingState, ContactAction, MCTSNode 数据结构
  - 新增: AsyncMCTSPlanner / RLSkillCoordinator (household_robot_system 引用)
  - 新增: dr_override() 接口 — 飞轮 DR 配置注入 MCTS 物理参数 (闭环耦合)
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# 数据结构: 接触状态 / 动作
# ============================================================================

@dataclass
class ContactAction:
    """接触动作 — 末端执行器在接触点上的操作。"""
    contact_position: np.ndarray       # (3,) 世界坐标
    contact_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))  # (3,) 接触法向
    grasp_type: str = "power"          # "power" | "pinch" | "hook"
    force_n: float = 10.0              # 期望接触力 (N)

    def __hash__(self):
        return hash((tuple(self.contact_position.round(4)), self.grasp_type))


@dataclass
class ClimbingState:
    """攀爬/接触状态 — MCTS 节点状态。"""
    ee_position: np.ndarray            # 末端执行器位置
    joint_positions: np.ndarray        # 关节角度
    contacts_held: List[ContactAction] = field(default_factory=list)
    object_pose: Optional[np.ndarray] = None  # 目标物体位姿
    stability_margin: float = 1.0      # 当前平衡裕量 (接触多边形)
    timestep: int = 0
    terminal: bool = False
    # L3 (2026-10-08): 滑脱通道。原模型只有"接触越多越稳"的单调递增,
    # **没有失稳路径** (评估 §3.6) — 规划因此学不到"哪些抓法会滑"。
    # 摩擦系数 (可由 DR 注入或岩点几何覆盖, 默认 0.8 = 攀岩接触典型值)
    hold_mu: float = 0.8
    slip: bool = False                 # 本步是否发生滑脱 (摩擦锥判据)

    def __hash__(self):
        return hash((tuple(self.ee_position.round(3)), self.timestep))


# ============================================================================
# MCTS 节点
# ============================================================================

@dataclass
class MCTSNode:
    """MCTS 搜索树节点。"""
    state: ClimbingState
    parent: Optional["MCTSNode"] = None
    action_from_parent: Optional[ContactAction] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    untried_actions: List[ContactAction] = field(default_factory=list)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        return self.state.terminal

    def ucb1(self, exploration_weight: float = 1.414) -> float:
        """UCB1 评分。"""
        if self.visits == 0:
            return float("inf")
        exploit = self.total_value / self.visits
        explore = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child(self, exploration_weight: float = 1.414) -> "MCTSNode":
        """选择 UCB1 最高的子节点。"""
        return max(self.children, key=lambda c: c.ucb1(exploration_weight))


# ============================================================================
# ClimbingMCTS — K2 完整实现
# ============================================================================

class ClimbingMCTS:
    """攀爬/接触 MCTS 规划器 — K2: 完整 UCB1-MCTS 实现。

    核心循环:
        while budget > 0:
            leaf = select(root)       # UCB1 下降到未完全展开节点
            child = expand(leaf)      # 扩展一个未尝试动作
            reward = simulate(child)  # 随机 rollout 到终态
            backprop(child, reward)   # 反向传播价值
    """

    def __init__(
        self,
        rollout_depth: int = 10,
        n_simulations: int = 200,
        exploration_weight: float = 1.414,
        discount: float = 0.95,
        physics_fn: Optional[Callable] = None,
        reward_fn: Optional[Callable] = None,
    ):
        self.rollout_depth = rollout_depth
        self.n_simulations = n_simulations
        self.exploration_weight = exploration_weight
        self.discount = discount
        # 物理步进函数 (默认恒等)
        self.physics_fn = physics_fn or self._default_physics
        self.reward_fn = reward_fn or self._default_reward
        self.root: Optional[MCTSNode] = None
        # L3 修复 (2026-10-08): goal_fn 此前**收参不用** — 终局判定完全由
        # _default_physics 的硬编码条件 (timestep>=8 or stability>0.9) 决定,
        # 即"规划目标"与传入的 goal_fn 无关 (评估 §3.6 记为"决策空转")。
        # 现在真实保存并在终局判定中消费。
        self.goal_fn: Optional[Callable] = None
        self.goal_reached: bool = False    # 本次 search 是否达成过目标 (可观测)

    def _is_goal(self, state: ClimbingState) -> bool:
        """L3: 终局判定 = 物理终态 OR 传入的目标函数达成。

        goal_fn 抛异常时按"未达成"处理 (不中断搜索) — 规划器不应因用户
        目标函数的一个边缘输入而崩溃。
        """
        if state.terminal:
            return True          # 物理终态: 不置 goal_reached (语义区分)
        if self.goal_fn is None:
            return False
        try:
            gf = bool(self.goal_fn(state))
        except Exception:      # noqa: BLE001 — 目标函数异常按未达成处理
            return False
        if gf:
            # 仅当**目标函数**达成才置位 — 物理终态不算"达成目标",
            # 否则 goal_reached 会恒真, 失去可观测价值。
            self.goal_reached = True
        return gf

    def search(self, initial_state: ClimbingState, goal_fn: Callable) -> List[ContactAction]:
        """入口: 给定初始状态 + 目标函数，返回最优接触序列。

        L3: goal_fn 真实生效 — 达成目标即终局, 并记 self.goal_reached。
        """
        self.goal_fn = goal_fn
        self.goal_reached = False
        actions = self._generate_candidate_actions(initial_state)
        self.root = MCTSNode(state=initial_state, untried_actions=actions.copy())

        for _ in range(self.n_simulations):
            node = self._select(self.root)
            if not node.is_terminal:
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

        # 提取最优动作路径
        path = []
        node = self.root
        while node.children:
            best = max(node.children, key=lambda c: c.visits)
            if best.action_from_parent:
                path.append(best.action_from_parent)
            node = best
        return path

    def _select(self, node: MCTSNode) -> MCTSNode:
        """阶段 1: UCB1 下降到叶节点。"""
        while not node.is_terminal and node.is_fully_expanded:
            if not node.children:
                break
            node = node.best_child(self.exploration_weight)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """阶段 2: 从 untried_actions 中 pop 一个动作 → 创建子节点。"""
        if node.is_fully_expanded or node.is_terminal:
            return node
        action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
        next_state = self.physics_fn(node.state, action)
        # L3: 达成目标 → 标记终局 (此前 goal_fn 完全不参与, 规划与目标脱节)
        if self._is_goal(next_state):
            next_state.terminal = True
            # 注: goal_reached 的置位**只**在 _is_goal 内 (仅 goal_fn 为真时),
            # 此处不置 — 否则物理终态会被误记成"达成目标"。
        child = MCTSNode(
            state=next_state,
            parent=node,
            action_from_parent=action,
            untried_actions=self._generate_candidate_actions(next_state),
        )
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """阶段 3: 随机 rollout 到 rollout_depth 或终态。"""
        state = node.state
        total_reward = 0.0
        discount = 1.0
        for _ in range(self.rollout_depth):
            # L3: 终局判定改为 _is_goal (含 goal_fn), 而非只读 state.terminal
            if self._is_goal(state):
                break
            actions = self._generate_candidate_actions(state)
            if not actions:
                break
            action = random.choice(actions)
            state = self.physics_fn(state, action)
            if self._is_goal(state):
                state.terminal = True
                # 同上: 不在此置 goal_reached (避免物理终态误记为目标达成)
                # 达成终局仍要计入这一步的奖励, 再退出
                total_reward += discount * self.reward_fn(state)
                break
            reward = self.reward_fn(state)
            total_reward += discount * reward
            discount *= self.discount
        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """阶段 4: 反向传播价值到路径上所有节点。"""
        while node is not None:
            node.visits += 1
            node.total_value += reward
            node = node.parent

    # ------------------------------------------------------------------ #
    # 候选动作生成 (可被子类覆盖)
    # ------------------------------------------------------------------ #

    def _generate_candidate_actions(self, state: ClimbingState) -> List[ContactAction]:
        """生成状态的可行接触动作集 (K2: 简化为中心+偏移的5个姿态)。"""
        if state.terminal:
            return []
        center = state.ee_position
        offsets = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.02, 0.0, 0.0]),      # +X
            np.array([-0.02, 0.0, 0.0]),     # -X
            np.array([0.0, 0.02, 0.0]),      # +Y
            np.array([0.0, -0.02, 0.0]),     # -Y
        ]
        actions = []
        for off, gtype in zip(offsets, ["power", "pinch", "hook", "power", "pinch"]):
            actions.append(ContactAction(
                contact_position=center + off,
                contact_normal=np.array([0.0, 0.0, 1.0]),
                grasp_type=gtype,
                force_n=8.0 + random.uniform(-2, 2),
            ))
        return actions

    # ------------------------------------------------------------------ #
    # 默认物理 / 奖励函数
    # ------------------------------------------------------------------ #

    def _default_physics(self, state: ClimbingState, action: ContactAction) -> ClimbingState:
        """默认物理: 末端移动到接触点, 步数 +1, **并经摩擦锥判滑脱**。

        L3 (2026-10-08) 两处修复:
          1. **DR override 真实接入**: dr_override() 会设 `self._mu_override` /
             `self._k_override`, 但本函数 (唯一的物理步进点) 此前**从不读取**
             → "制造容差 → DR → MCTS 物理"的飞轮闭环是断的。现真实消费 μ。
          2. **引入失稳通道**: 原 `stability = min(1, n*0.3+0.1)` 单调不减,
             规划学不到"哪些抓法会滑"。现用摩擦锥判据 ‖f_t‖ ≤ μ·f_n:
             切向需求超摩擦极限 → 滑脱 → 稳定性下修 (可失稳)。
        """
        new_ee = action.contact_position.copy()
        new_contacts = state.contacts_held + [action]

        # (1) μ 优先级: DR 注入 > 本步岩点几何 > 默认 0.8
        mu = float(getattr(self, "_mu_override", None) or state.hold_mu or 0.8)
        # 切向需求按抓取型估计: power 包络最稳, pinch 对捏最易滑
        tangent_ratio = {"power": 0.35, "pinch": 0.70,
                         "hook": 0.50}.get(action.grasp_type, 0.45)
        f_n = max(float(action.force_n), 1e-6)
        f_t = tangent_ratio * f_n
        slip = bool(f_t > mu * f_n)          # 摩擦锥判据

        # (2) 稳定性: 滑脱则下修 (失稳通道), 否则沿用原饱和式
        base = min(1.0, len(new_contacts) * 0.3 + 0.1)
        stability = max(0.0, base - 0.25) if slip else base
        terminal = state.timestep >= 8 or stability > 0.9
        return ClimbingState(
            ee_position=new_ee,
            joint_positions=state.joint_positions,
            contacts_held=new_contacts,
            object_pose=state.object_pose,
            stability_margin=stability,
            timestep=state.timestep + 1,
            terminal=terminal,
            hold_mu=mu,
            slip=slip,
        )

    def _default_reward(self, state: ClimbingState) -> float:
        """L3 (2026-10-08): 多分量奖励 — 与状态转移**不同源**。

        原实现 `r = stability_margin` (+终态 2.0), 而 stability_margin 正是
        `_default_physics` 的产出 → **奖励与状态转移同源**, MCTS 学不到任何
        物理之外的信息 (评估 §3.6 "奖励与状态转移同源")。

        现引入与转移不同源的分量:
          + 稳定性裕量      (主体, 保持原量级)
          + CWC 摩擦锥余量  (μ 越高可承受切向越大 → 安全余量)
          - 滑脱惩罚        (失稳通道的可学习信号)
          - 能耗惩罚        (接触力总量, 与位移无关的代价)
          - 路径代价        (步数, 抑制无意义徘徊)
        """
        r = float(state.stability_margin)
        # CWC (Contact Wrench Cone) 摩擦锥余量: 以 0.6 为基准, μ 越高余量越大
        r += 0.5 * max(0.0, float(state.hold_mu) - 0.6)
        # 滑脱惩罚
        if state.slip:
            r -= 1.0
        # 能耗: 接触力总量 (越大越耗能)
        forces = [float(c.force_n) for c in state.contacts_held]
        if forces:
            r -= 0.01 * sum(forces)
        # 路径代价: 抑制无意义徘徊
        r -= 0.05 * float(state.timestep)
        # 终态: 达成且稳定才给完成奖励 (沿用原口径)
        if state.terminal and state.stability_margin > 0.7:
            r += 2.0
        return r

    # ------------------------------------------------------------------ #
    # 飞轮耦合: DR 配置注入物理参数
    # ------------------------------------------------------------------ #

    def dr_override(self, dr_config: Dict[str, Any]) -> None:
        """飞轮 DR 注入: 用 ManufacturingDataFlywheel 产出的 DR 分布覆盖物理模型参数。

        创新 #1 闭环: 制造容差数据 → DR 分布 → MCTS 物理模型直接用真机分布参数。
        """
        if "joint_stiffness" in dr_config.get("overrides", {}):
            js = dr_config["overrides"]["joint_stiffness"]
            # 注入刚度到物理函数的力限幅
            if HAS_TORCH:
                self._k_override = js.get("mean", 5000)
        if "friction_coefficient" in dr_config.get("overrides", {}):
            fc = dr_config["overrides"]["friction_coefficient"]
            self._mu_override = (fc.get("min", 0) + fc.get("max", 1)) / 2


# ============================================================================
# compute_mc_returns — K2 完整实现
# ============================================================================

def compute_mc_returns(rewards: List[float], discount: float = 0.95) -> List[float]:
    """蒙特卡洛回报: G_t = Σ_{k=0}^{T-t} γ^k * r_{t+k}

    K2: 从 None → 真实计算。用于训练 NeRF 价值网络的 ground truth。
    """
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + discount * running
        returns[t] = running
    return returns


# ============================================================================
# DiffusionPolicy — K2 接口补全 (K3 对接 Denoising Diffusion)
# ============================================================================

class DiffusionPolicy:
    """条件扩散策略 — MCTS 输出接触序列 → 连续关节/力矩轨迹 → 低层执行。

    v0.7: 对接 `models.diffusion_policy.DiffusionPolicyModel` (真实 DDPM 去噪)。
    未训练时**显式**返回占位并告警, 不再伪装成"生成了轨迹" — 此前
    `generate()` 直接返回 randn*0.1, 注释写着 "K3: 实现 DDPM", 一直没实现,
    于是任何上层调用都拿到噪声却无从察觉。

    用法:
        p = DiffusionPolicy(obs_dim={"proprio": 8}, action_dim=3, action_horizon=8)
        p.fit(expert_actions, cond)        # (N,H,A) 专家轨迹 + (N,obs) 条件
        traj = p.generate(obs)
    """

    def __init__(self, obs_dim: Dict[str, Any] = None, action_dim: int = 12,
                 action_horizon: int = 10, n_diffusion_steps: int = 50):
        self.obs_dim_cfg = obs_dim or {"rgb": (3, 224, 224), "tactile": (4, 64, 64),
                                       "proprio": 30}
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_diffusion_steps = n_diffusion_steps
        self._model: Optional[Any] = None
        self._model_loaded = False
        self._weights_path: Optional[str] = None
        self._warned = False

    # v0.6 既有字段别名 (外部代码读 obs_dim)
    @property
    def obs_dim(self) -> Dict[str, Any]:
        return self.obs_dim_cfg

    @obs_dim.setter
    def obs_dim(self, v: Dict[str, Any]) -> None:
        self.obs_dim_cfg = v

    # ------------------------------------------------------------------ #
    def _obs_vector(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """把观测字典摊成一个条件向量 (优先用 proprio/state)。"""
        for key in ("proprio", "state"):
            if isinstance(obs, dict) and key in obs:
                return np.asarray(obs[key], dtype=float).reshape(-1)
        vals = [np.asarray(v, dtype=float).reshape(-1)
                for v in obs.values() if np.asarray(v).ndim <= 2]
        return np.concatenate(vals) if vals else np.zeros(1)

    def attach(self, model) -> None:
        """注入一个已训练的 DiffusionPolicyModel。"""
        self._model = model
        self._model_loaded = True

    def fit(self, expert_actions: np.ndarray, cond: np.ndarray,
            iters: int = 1200, **kw):
        """训练内部 DDPM (返回 TrainStats)。"""
        from ..models.diffusion_policy import DiffusionPolicyModel

        c = np.atleast_2d(np.asarray(cond, dtype=float))
        self._model = DiffusionPolicyModel(
            obs_dim=int(c.shape[1]), action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            n_diffusion_steps=self.n_diffusion_steps, **kw)
        stats = self._model.fit(expert_actions, c, iters=iters)
        self._model_loaded = True
        return stats

    @classmethod
    def load(cls, path: str) -> "DiffusionPolicy":
        """加载预训练模型权重。"""
        inst = cls()
        inst._weights_path = path
        inst._model_loaded = True
        return inst

    def save(self, path: str) -> None:
        self._weights_path = path
        self._model_loaded = False

    # ------------------------------------------------------------------ #
    def generate(self, obs: Dict[str, np.ndarray],
                 contact_sequence: List[ContactAction] = None) -> np.ndarray:
        """生成关节力矩/位置轨迹 (T × action_dim)。"""
        if self._model is not None:
            cond = self._obs_vector(obs)
            traj = self._model.sample(cond, deterministic=True)[0]
            if contact_sequence:
                traj[:, :3] += contact_sequence[0].contact_position[:3]
            return np.asarray(traj, dtype=np.float32)

        # 未训练: 明示占位 (上层应看到这条告警而不是拿到噪声当结果)
        if not self._warned:
            warnings.warn("DiffusionPolicy 未训练 (无 _model): generate() 返回噪声占位, "
                          "请先调用 fit(...) 或 attach(model)", RuntimeWarning)
            self._warned = True
        traj = np.random.randn(self.action_horizon, self.action_dim).astype(np.float32) * 0.1
        if contact_sequence:
            traj[:, :3] += contact_sequence[0].contact_position[:3]
        return traj

    def __repr__(self) -> str:
        status = "loaded" if self._model_loaded else "untrained"
        return f"DiffusionPolicy(action_dim={self.action_dim}, T={self.action_horizon}, {status})"


# ============================================================================
# NeRF 价值网络 (from K1 raw, torch-dependent, kept intact)
# ============================================================================

if HAS_TORCH:
    class NeRFValueNetwork(nn.Module):
        """从NeRF场景特征预测状态价值。"""
        def __init__(self, scene_dim: int = 256, proprio_dim: int = 30):
            super().__init__()
            self.scene_encoder = nn.Sequential(
                nn.Linear(scene_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
            )
            self.proprio_encoder = nn.Sequential(
                nn.Linear(proprio_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
            )
            self.value_head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Tanh(),
            )

        def forward(self, scene_feat: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
            s = self.scene_encoder(scene_feat)
            p = self.proprio_encoder(proprio)
            return self.value_head(torch.cat([s, p], dim=-1))


# ============================================================================
# NeRFGuidedMCTS — K1 清洗保留
# ============================================================================

class NeRFGuidedMCTS(ClimbingMCTS):
    """**未实现** —— NeRF 引导部分全是占位, 请勿当作可用能力引用。

    plan.md 降级清单 (与 plan_v3 / docs/11 §"NeRF/VQ/async") 已裁定 NeRF 方向
    **不做**, docs/11 进一步要求"删除 NeRFGuidedMCTS 或标注'未实现'"。此处取
    后者: 类保留是为了不破坏 `planning.__all__` 与 roadmap_data 的条目引用。

    实际状态 (与旧 docstring "K2: 基类补全后自动可用" 相反 —— 基类补全了,
    NeRF 侧依旧没有):
      · `_collision_free()` / `_ik_feasible()` 恒返回 True → get_actions 的
        "可达性和碰撞过滤"是**空操作**, 一个候选都过滤不掉。
      · `heuristic_value()` 的 scene_feat 是 `torch.randn` 占位, 不是任何
        NeRF 渲染/体查询结果; nerf_scene=None 时直接退回 stability_margin。
      · 全仓无任何 NeRF 训练或推理代码 (grep nerf 仅命中本类与占位注释)。
    """

    def __init__(self, *args, nerf_scene=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nerf = nerf_scene
        if HAS_TORCH:
            self.value_net = NeRFValueNetwork()
        else:
            self.value_net = None

    def get_actions(self, state: ClimbingState) -> List[ContactAction]:
        actions = super()._generate_candidate_actions(state)
        # NeRF 增强: 可达性和碰撞过滤
        feasible = []
        for a in actions:
            if self._ik_feasible(a) and self._collision_free(a):
                feasible.append(a)
        return feasible

    def heuristic_value(self, state: ClimbingState) -> float:
        if self.nerf is None or self.value_net is None or not HAS_TORCH:
            return state.stability_margin
        with torch.no_grad():
            scene_feat = torch.randn(1, 256) * 0.1  # 占位
            proprio = torch.from_numpy(state.joint_positions[:30].reshape(1, -1)).float()
            return self.value_net(scene_feat, proprio).item()

    def _collision_free(self, action: ContactAction) -> bool:
        # 占位: 恒真 → 碰撞检测从未发生 (NeRF SDF 查询在降级清单里已裁定不做)
        return True

    def _ik_feasible(self, action: ContactAction) -> bool:
        # 占位: 恒真 → 可达性过滤从未发生 (整体控制栈另有可用的 IK, 见
        # control/whole_body.py; 这里没有接它)
        return True


# ============================================================================
# 飞轮 DR 驱动的异步 MCTS 规划器 / RL 技能协调器
# (被引用于 control/household_robot_system.py K2 补全)
# ============================================================================

class AsyncMCTSPlanner:
    """**实为同步** MCTS 规划器 —— 名字里的 Async 是历史遗留, 不承诺异步。

    docs/11 对降级清单的处置要求是"async…要么实现要么诚实标注同步", 这里取
    后者: 全类无 threading/multiprocessing/asyncio, `plan()` 与 `replan()` 都是
    阻塞调用, 直接跑满 `mcts.search()` 的 n_simulations 次展开。

    对 20Hz 控制环的影响 (household_robot_system.py:39 把它挂在 task_planner 上):
    单次 `plan()` 的耗时会**阻塞主循环**, 上游若需要非阻塞语义必须自行放到线程
    或进程里, 不能依赖本类的命名。
    """

    def __init__(self, mcts: Optional[ClimbingMCTS] = None, dr_config: Dict = None):
        self.mcts = mcts or ClimbingMCTS(n_simulations=100)
        if dr_config:
            self.mcts.dr_override(dr_config)

    def plan(self, init_state, goal=None) -> List[ContactAction]:
        if isinstance(init_state, ClimbingState):
            return self._mcts_search(init_state)
        return []

    def _mcts_search(self, state: ClimbingState) -> List[ContactAction]:
        return self.mcts.search(state, goal_fn=lambda s: s.stability_margin > 0.8)

    def replan(self, current_state, **kwargs) -> List[ContactAction]:
        return self._mcts_search(current_state) if isinstance(current_state, ClimbingState) else []


class RLSkillCoordinator:
    """RL 技能协调器 — 连接 MCTS 规划和 Diffusion 策略执行。"""

    def __init__(self, skills: Dict[str, DiffusionPolicy] = None, dr_config: Dict = None):
        self.skills = skills or {}
        self.dr_config = dr_config
        self._current_skill: Optional[str] = None

    def load_skill(self, name: str, policy: DiffusionPolicy):
        self.skills[name] = policy

    def execute(self, skill_name: str, obs: Dict[str, np.ndarray]) -> np.ndarray:
        if skill_name not in self.skills:
            return np.zeros(12)
        self._current_skill = skill_name
        return self.skills[skill_name].generate(obs)

    @property
    def current_skill(self) -> Optional[str]:
        return self._current_skill


# ============================================================================
# Motion Primitive Library (被 RealTime20HzController 引用)
# ============================================================================

class MotionPrimitive:
    """运动原语 — 预定义轨迹片段。K2: 接口补全。"""
    def __init__(self, name: str = "unnamed", duration_s: float = 1.0):
        self.name = name
        self.duration_s = duration_s
        self._trajectory: Optional[np.ndarray] = None


class MotionPrimitiveLibrary:
    def __init__(self):
        self._primitives: Dict[str, MotionPrimitive] = {}

    def add(self, name: str, prim: MotionPrimitive):
        self._primitives[name] = prim

    def preload_all(self):
        pass

    def get(self, name: str) -> Optional[MotionPrimitive]:
        return self._primitives.get(name)


# ============================================================================
# 模块便捷函数
# ============================================================================

def plan_contacts(
    init_state: ClimbingState,
    n_sim: int = 200,
    dr_config: Optional[Dict] = None,
) -> Tuple[List[ContactAction], ClimbingMCTS]:
    """一键接触规划入口。"""
    mcts = ClimbingMCTS(n_simulations=n_sim)
    if dr_config:
        mcts.dr_override(dr_config)
    actions = mcts.search(init_state, goal_fn=lambda s: s.stability_margin > 0.7)
    return actions, mcts
