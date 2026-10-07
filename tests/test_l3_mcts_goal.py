"""test_l3_mcts_goal.py — L3: MCTS 接真实目标 (决策不再空转) + 失稳通道。

评估 §3.6 记录的 L3 空转证据:
  - `search()` 收 `goal_fn` 参数但**从不调用** → 终局判定完全由
    `_default_physics` 的硬编码条件 (timestep>=8 or stability>0.9) 决定,
    即"规划目标"与传入的目标函数无关。
  - `stability = min(1, n*0.3+0.1)` **单调不减**, 无失稳通道。
  - `dr_override()` 设的 `_mu_override`/`_k_override` **无读取点** (飞轮断裂)。
  - 奖励 = stability_margin, 而它正是物理的产出 → **奖励与状态转移同源**。

本测试逐条守护修复, 并对关键项做反证 (证明闸门不是摆设)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from embodied_ai.planning.mcts_integration import (  # noqa: E402
    ClimbingMCTS, ClimbingState, ContactAction,
)


def _state(**kw) -> ClimbingState:
    d = dict(ee_position=np.zeros(3), joint_positions=np.zeros(12), timestep=0)
    d.update(kw)
    return ClimbingState(**d)


def _action(grasp="power", force=10.0) -> ContactAction:
    return ContactAction(
        contact_position=np.array([0.0, 0.0, 0.01]),
        contact_normal=np.array([0.0, 0.0, 1.0]),
        grasp_type=grasp,
        force_n=force,
    )


# ============================================================================
# 1. goal_fn 真实消费 (L3 核心)
# ============================================================================

def test_goal_fn_is_actually_consumed():
    """L3 核心: goal_fn 必须真实生效 (此前收参不用 → 决策空转)。"""
    mcts = ClimbingMCTS(n_simulations=30, rollout_depth=3)
    mcts.search(_state(), goal_fn=lambda s: s.timestep >= 1)
    assert mcts.goal_reached, "goal_fn 未生效: goal_reached 应为 True"


def test_unreachable_goal_not_falsely_reported():
    """反证: 永假目标不得误报达成 (物理终态不算'达成目标')。"""
    mcts = ClimbingMCTS(n_simulations=30, rollout_depth=2)
    mcts.search(_state(), goal_fn=lambda s: False)
    assert not mcts.goal_reached, (
        "永假 goal 竟置 goal_reached=True — 物理终态被误当作目标达成")


def test_goal_fn_exception_does_not_crash_search():
    """目标函数抛异常时按'未达成'处理, 搜索不应崩溃。"""
    def bad(s):
        raise RuntimeError("boom")

    mcts = ClimbingMCTS(n_simulations=10, rollout_depth=2)
    path = mcts.search(_state(), goal_fn=bad)
    assert isinstance(path, list)
    assert not mcts.goal_reached


# ============================================================================
# 2. 失稳通道 (滑脱) — 原稳定性单调不减
# ============================================================================

def test_slip_triggered_at_low_mu():
    """低 μ + 高切向抓取型 → 滑脱, 且稳定性被下修 (存在失稳路径)。"""
    mcts = ClimbingMCTS(n_simulations=5, rollout_depth=1)
    s0 = _state(hold_mu=0.1)                       # 极低摩擦
    s1 = mcts._default_physics(s0, _action(grasp="pinch", force=10.0))
    assert s1.slip, "μ=0.1: 切向 0.7*10=7 > 摩擦极限 0.1*10=1 → 应滑脱"
    nominal = min(1.0, len(s1.contacts_held) * 0.3 + 0.1)
    assert s1.stability_margin < nominal, "滑脱应下修稳定性 (否则仍单调递增)"


def test_no_slip_at_high_mu():
    """反证: 判据不是恒真 — 高 μ 下同一动作不应滑脱。"""
    mcts = ClimbingMCTS(n_simulations=5, rollout_depth=1)
    s0 = _state(hold_mu=1.2)
    s1 = mcts._default_physics(s0, _action(grasp="pinch", force=10.0))
    assert not s1.slip, "μ=1.2: 切向 7 < 极限 12 → 不应滑脱"


# ============================================================================
# 3. DR override 真实接入 (飞轮闭环)
# ============================================================================

def test_dr_override_mu_is_consumed_by_physics():
    """DR 注入的 μ 必须被物理消费 (此前 dr_override 设了却无读取点)。"""
    mcts = ClimbingMCTS(n_simulations=5, rollout_depth=1)
    mcts.dr_override({"overrides": {"friction_coefficient": {"min": 0.05, "max": 0.15}}})
    assert abs(mcts._mu_override - 0.1) < 1e-9
    # 注入 μ=0.1 优先于 state.hold_mu=1.5 → 应滑脱
    s1 = mcts._default_physics(_state(hold_mu=1.5), _action(grasp="pinch", force=10.0))
    assert s1.slip, "DR 注入的 μ 未生效 (物理仍在用 state.hold_mu)"


# ============================================================================
# 4. 奖励与状态转移不同源
# ============================================================================

def test_reward_penalises_slip():
    """滑脱应有惩罚 — 奖励不能只是物理产出 stability 的复读。"""
    mcts = ClimbingMCTS()
    base = dict(ee_position=np.zeros(3), joint_positions=np.zeros(12),
                stability_margin=0.5, hold_mu=0.8)
    ok = ClimbingState(**base, slip=False)
    slipped = ClimbingState(**base, slip=True)
    assert mcts._default_reward(slipped) < mcts._default_reward(ok), (
        "滑脱无惩罚 → 奖励仍与状态转移同源")


def test_reward_rewards_friction_margin():
    """CWC 摩擦锥余量应计入奖励 (μ 越高越安全)。"""
    mcts = ClimbingMCTS()
    mk = lambda mu: ClimbingState(ee_position=np.zeros(3), joint_positions=np.zeros(12),
                                  stability_margin=0.5, hold_mu=mu, slip=False)
    assert mcts._default_reward(mk(1.5)) > mcts._default_reward(mk(0.8)), (
        "摩擦余量未计入奖励")


def test_reward_penalises_path_length():
    """路径代价: 步数越多奖励越低 (抑制无意义徘徊)。"""
    mcts = ClimbingMCTS()
    mk = lambda t: ClimbingState(ee_position=np.zeros(3), joint_positions=np.zeros(12),
                                 stability_margin=0.5, hold_mu=0.8, timestep=t)
    assert mcts._default_reward(mk(6)) < mcts._default_reward(mk(1))
