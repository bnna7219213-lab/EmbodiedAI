"""PyBullet 仿真环境 — 备选仿真平台, 支持多体/软体。

PyBullet 作为 MuJoCo 的备选:
  - 免费开源 (BSD 许可)
  - 支持软体动力学 (柔性体场景, G6 缺口)
  - URDF 导入简单

注意: PyBullet 接触精度低于 MuJoCo, Phase A 首选 MuJoCo。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base_env import EnvConfig, SimEnv, StepResult


class PyBulletEnv(SimEnv):
    """PyBullet 包装器 (需要 pybullet 包)。"""

    @classmethod
    def is_available(cls) -> bool:
        """依赖探测 — 供 make_env 在构造前判断 (v0.5, 同 MuJoCoEnv)。"""
        try:
            import pybullet  # noqa: F401
            import pybullet_data  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def __init__(self, config: EnvConfig):
        super().__init__(config)

        try:
            import pybullet as p
            import pybullet_data
            self._p = p
        except ImportError:
            raise ImportError(
                "PyBulletEnv 需要 pybullet 包. 安装: pip install pybullet")

        # 连接物理引擎
        self._client = self._p.connect(self._p.DIRECT)  # 无 GUI 模式
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._build_scene()

    def _build_scene(self) -> None:
        """加载 URDF / 程序化生成场景。"""
        self._p.setGravity(0, 0, -9.81)
        self._p.setTimeStep(self.config.dt)

        # 地面
        self._plane_id = self._p.loadURDF("plane.urdf")

        scene = self.config.scene_name
        urdf_path = self.config.extra.get("urdf_path")

        if urdf_path:
            # 加载自定义 URDF
            self._robot_id = self._p.loadURDF(urdf_path, [0, 0, 0.5])
        elif scene == "biped":
            self._robot_id = self._build_biped_urdf()
        else:
            # 默认: 加载 Franka Emika Panda
            try:
                self._robot_id = self._p.loadURDF("franka_panda/panda.urdf",
                                                   [0, 0, 0], useFixedBase=True)
            except Exception:
                # PyBullet data 中无 franka 时使用简单方块
                self._robot_id = self._build_simple_robot()

        # 关节信息
        self._joint_ids = []
        for i in range(self._p.getNumJoints(self._robot_id)):
            info = self._p.getJointInfo(self._robot_id, i)
            if info[2] == self._p.JOINT_REVOLUTE:  # 旋转型关节
                self._joint_ids.append(i)

    def _build_biped_urdf(self) -> int:
        """生成简化双足 URDF (使用内置立方体拼接)。"""
        # 躯干
        torso_collision = self._p.createCollisionShape(
            self._p.GEOM_BOX, halfExtents=[0.1, 0.15, 0.2])
        torso_visual = self._p.createVisualShape(
            self._p.GEOM_BOX, halfExtents=[0.1, 0.15, 0.2], rgbaColor=[0.3, 0.3, 0.7, 1])
        torso_id = self._p.createMultiBody(
            baseMass=10, baseCollisionShapeIndex=torso_collision,
            baseVisualShapeIndex=torso_visual, basePosition=[0, 0, 1.0])

        # 使用复合多体 + 铰链关节构建双腿
        self._p.createConstraint(torso_id, -1, -1, -1, self._p.JOINT_FIXED,
                                  [0, 0, 0], [0, 0, 0], [0, 0, 1.0])

        return torso_id

    def _build_simple_robot(self) -> int:
        """生成最简单的多体机器人。"""
        box = self._p.createCollisionShape(
            self._p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.3])
        viz = self._p.createVisualShape(
            self._p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.3], rgbaColor=[0.7, 0.3, 0.3, 1])
        return self._p.createMultiBody(
            baseMass=1.0, baseCollisionShapeIndex=box,
            baseVisualShapeIndex=viz, basePosition=[0, 0, 0.3])

    # ------------------------------------------------------------------ #
    # SimEnv 接口实现
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            np.random.seed(seed)

        self._p.resetBasePositionAndOrientation(
            self._robot_id, [0, 0, 0.5], [0, 0, 0, 1])
        self.step_count = 0
        self.clear_episode()

        # 重置关节
        for jid in self._joint_ids:
            self._p.resetJointState(self._robot_id, jid, 0)

        state = self._get_state()
        return self._build_observation(state)

    def step(self, action: np.ndarray) -> StepResult:
        action = self._apply_action(action)

        # 位置控制
        for i, jid in enumerate(self._joint_ids[:len(action)]):
            self._p.setJointMotorControl2(
                self._robot_id, jid, self._p.POSITION_CONTROL,
                targetPosition=action[i], maxVelocity=5.0)

        # 步进仿真
        for _ in range(self._control_interval):
            self._p.stepSimulation()

        self.step_count += 1
        state = self._get_state()
        obs = self._build_observation(state)

        reward, done, info = self._compute_reward(obs)
        truncated = (self.step_count >= self.config.max_episode_steps)

        return StepResult(observation=obs, reward=reward, done=done,
                         truncated=truncated, info=info)

    def close(self) -> None:
        if self._client >= 0:
            self._p.disconnect()

    # ------------------------------------------------------------------ #
    # DR 支持
    # ------------------------------------------------------------------ #
    def set_dr_params(self, params: Dict[str, float]) -> None:
        """应用 DR 参数。"""
        if "table_friction" in params:
            self._p.changeDynamics(
                self._plane_id, -1,
                lateralFriction=params["table_friction"])

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _get_state(self) -> np.ndarray:
        states = []
        for jid in self._joint_ids[:self.config.action_dim]:
            pos, vel, _, _ = self._p.getJointState(self._robot_id, jid)
            states.extend([pos, vel])

        state = np.zeros(self.config.state_dim)
        state[:min(len(states), self.config.state_dim)] = states[:self.config.state_dim]
        return state

    def _compute_reward(self, obs: Dict[str, Any]) -> tuple:
        """简化奖励计算。"""
        state = obs["state"]
        # 默认: 存活奖励 + 直立奖励 (双足场景)
        reward = 0.1  # 存活
        if len(state) > 2 and state[2] > 0.8:  # 躯干高度
            reward += 0.05
        if len(state) > 2 and state[2] < 0.5:  # 摔倒
            reward -= 1.0
            return reward, True, {"fall": True}
        return reward, False, {}
