"""MuJoCo 仿真环境 — 高速接触动力学 + 渲染。

MuJoCo 是 Phase A 推荐的首选仿真平台:
  - 接触模型精确 (Hertz/Hunt-Crossley 内置支持)
  - 仿真速度快 (实时率 >10x 多关节场景)
  - 原生支持 tendon/equality constraint (攀爬绳索场景)

用法:
  config = EnvConfig(scene_name="franka_table", dt=0.001)
  env = MuJoCoEnv(config)
  obs = env.reset()
  result = env.step(np.zeros(7))
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base_env import EnvConfig, SimEnv, StepResult


class MuJoCoEnv(SimEnv):
    """MuJoCo 包装器 (需要 mujoco 包)。

    内部管理:
      - self.model: mujoco.MjModel
      - self.data: mujoco.MjData
      - self.viewer: 渲染器 (可选)
    """

    # 内置场景 → MJCF 文件映射
    SCENE_PATHS = {
        "franka_table": None,  # 使用程序化生成
        "simple_arm": None,
        "biped": None,
        "quadruped": None,
    }

    @classmethod
    def is_available(cls) -> bool:
        """依赖探测 — 供 make_env/available_backends 在**构造前**判断 (v0.5)。

        此前 import mujoco 发生在 __init__ 内, 而 available_backends() 只做模块导入
        探测, 于是会把"模块能导入但依赖缺失"误判为可用, 之后构造才炸。
        """
        try:
            import mujoco  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def __init__(self, config: EnvConfig):
        super().__init__(config)

        try:
            import mujoco
            self._mujoco = mujoco
        except ImportError:
            raise ImportError(
                "MuJoCoEnv 需要 mujoco 包. 安装: pip install mujoco")

        # 构建仿真场景
        self._build_scene()
        self._viewer = None
        self._rgb_buffer: Optional[np.ndarray] = None

    def _build_scene(self) -> None:
        """程序化生成 MuJoCo 场景 (MJCF XML)。"""
        scene = self.config.scene_name

        if scene in ("franka_table", "table_pick"):
            self._mjcf = self._build_table_scene()
        elif scene == "biped":
            self._mjcf = self._build_biped_scene()
        elif scene == "simple_arm":
            self._mjcf = self._build_simple_arm_scene()
        else:
            # 默认桌面场景
            self._mjcf = self._build_table_scene()

        # 编译模型
        self.model = self._mujoco.MjModel.from_xml_string(self._mjcf)
        self.data = self._mujoco.MjData(self.model)

        # 设置仿真步长
        self.model.opt.timestep = self.config.dt

        # 关节索引
        self._joint_ids = list(range(self.config.action_dim))
        self._qpos_start = self.model.nq - self.config.action_dim

    def _build_table_scene(self) -> str:
        """生成桌面抓取场景 MJCF。"""
        return f"""
        <mujoco model="table_pick">
          <option gravity="0 0 -9.81" timestep="{self.config.dt}" iterations="50" ls_iterations="50">
            <flag warmstart="enable"/>
          </option>
          <default>
            <joint armature="0.1" damping="0.1" limited="true"/>
            <geom friction="0.8 0.005 0.0001" density="500"/>
          </default>
          <worldbody>
            <light diffuse="0.8 0.8 0.8" pos="0 0 3" dir="0 0 -1"/>
            <geom name="floor" type="plane" size="2 2 0.01" rgba="0.9 0.9 0.9 1"/>
            <body name="table" pos="0.5 0 0.4">
              <geom name="table_top" type="box" size="0.4 0.3 0.01" rgba="0.6 0.4 0.2 1"/>
              <geom name="table_leg1" type="cylinder" pos="-0.35 -0.25 -0.2" size="0.02 0.2"/>
              <geom name="table_leg2" type="cylinder" pos="0.35 -0.25 -0.2" size="0.02 0.2"/>
              <geom name="table_leg3" type="cylinder" pos="-0.35 0.25 -0.2" size="0.02 0.2"/>
              <geom name="table_leg4" type="cylinder" pos="0.35 0.25 -0.2" size="0.02 0.2"/>
            </body>
            <body name="cup" pos="0.5 0 0.45">
              <joint name="cup_joint" type="free" damping="0.01"/>
              <geom name="cup_geom" type="cylinder" size="0.03 0.05" mass="0.1" rgba="0.2 0.6 0.9 1"/>
            </body>
            <body name="camera" pos="0.5 0 1.2">
              <camera name="top_cam" mode="fixed" fovy="60" euler="0 0 0"/>
            </body>
          </worldbody>
          <actuator>
            {''.join(f'<motor name="j{i}" joint="joint_{i}" gear="1" ctrllimited="true" ctrlrange="-10 10"/>' for i in range(min(self.config.action_dim, 7)))}
          </actuator>
          <equality/>
        </mujoco>
        """

    def _build_biped_scene(self) -> str:
        """生成双足行走场景 MJCF (简化 2D 行走)。"""
        return f"""
        <mujoco model="biped_walk">
          <option gravity="0 0 -9.81" timestep="{self.config.dt}" iterations="100">
            <flag warmstart="enable"/>
          </option>
          <worldbody>
            <light diffuse="0.8 0.8 0.8" pos="0 0 5" dir="0 0 -1"/>
            <geom name="floor" type="plane" size="10 1 0.01" rgba="0.8 0.8 0.8 1"
                  friction="1.0 0.005 0.0001"/>
            <body name="torso" pos="0 0 1.0">
              <joint name="root" type="free"/>
              <geom name="torso_geom" type="capsule" size="0.05 0.2" mass="10" rgba="0.3 0.3 0.7 1"/>
              <body name="thigh_L" pos="-0.05 -0.1 -0.2">
                <joint name="joint_0" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
                <geom name="thigh_L_geom" type="capsule" size="0.04 0.2" mass="3"/>
                <body name="shin_L" pos="0 -0.1 -0.4">
                  <joint name="joint_1" type="hinge" axis="0 1 0" range="-2.5 0" damping="0.3"/>
                  <geom name="shin_L_geom" type="capsule" size="0.03 0.2" mass="2"/>
                  <body name="foot_L" pos="0 -0 -0.4">
                    <joint name="joint_2" type="hinge" axis="0 1 0" range="-0.5 1.0" damping="0.2"/>
                    <geom name="foot_L_geom" type="box" size="0.08 0.03 0.02" mass="0.5" rgba="0.2 0.2 0.2 1"/>
                  </body>
                </body>
              </body>
              <body name="thigh_R" pos="0.05 -0.1 -0.2">
                <joint name="joint_3" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
                <geom name="thigh_R_geom" type="capsule" size="0.04 0.2" mass="3"/>
                <body name="shin_R" pos="0.05 -0.1 -0.4">
                  <joint name="joint_4" type="hinge" axis="0 1 0" range="-2.5 0" damping="0.3"/>
                  <geom name="shin_R_geom" type="capsule" size="0.03 0.2" mass="2"/>
                  <body name="foot_R" pos="0.05 -0 -0.4">
                    <joint name="joint_5" type="hinge" axis="0 1 0" range="-0.5 1.0" damping="0.2"/>
                    <geom name="foot_R_geom" type="box" size="0.08 0.03 0.02" mass="0.5" rgba="0.2 0.2 0.2 1"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            {''.join(f'<motor name="j{i}" joint="joint_{i}" gear="10" ctrllimited="true" ctrlrange="-50 50"/>' for i in range(min(self.config.action_dim, 6)))}
          </actuator>
        </mujoco>
        """

    def _build_simple_arm_scene(self) -> str:
        """生成简单 3-DOF 机械臂场景。"""
        return f"""
        <mujoco model="simple_arm">
          <option gravity="0 0 -9.81" timestep="{self.config.dt}">
            <flag warmstart="enable"/>
          </option>
          <worldbody>
            <light diffuse="0.8 0.8 0.8" pos="0 0 3" dir="0 0 -1"/>
            <geom name="floor" type="plane" size="5 5 0.01"/>
            <body name="base" pos="0 0 0.5">
              <joint name="joint_0" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
              <geom name="link0" type="capsule" size="0.03 0.25" mass="1"/>
              <body name="link1" pos="0 0 0.5">
                <joint name="joint_1" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
                <geom name="link1" type="capsule" size="0.025 0.25" mass="0.8"/>
                <body name="link2" pos="0 0 0.5">
                  <joint name="joint_2" type="hinge" axis="0 1 0" range="-2.5 1.0"/>
                  <geom name="link2" type="capsule" size="0.02 0.2" mass="0.5"/>
                  <body name="end_effector" pos="0 0 0.4">
                    <geom name="ee" type="sphere" size="0.03" mass="0.1" rgba="1 0 0 1"/>
                    <site name="ee_site" pos="0 0 0"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor name="j0" joint="joint_0" gear="1"/>
            <motor name="j1" joint="joint_1" gear="1"/>
            <motor name="j2" joint="joint_2" gear="1"/>
          </actuator>
        </mujoco>
        """

    # ------------------------------------------------------------------ #
    # SimEnv 接口实现
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """重置到初始状态 + 随机初始化。"""
        if seed is not None:
            np.random.seed(seed)

        self._mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.clear_episode()

        # 添加初始位置随机化
        q0 = np.random.uniform(-0.1, 0.1, size=self.config.action_dim)
        self.data.qpos[self._qpos_start: self._qpos_start + self.config.action_dim] = q0

        # 前向运动学
        self._mujoco.mj_forward(self.model, self.data)

        state = self._get_state()
        images = self._render_image() if self.config.num_views > 0 else None
        return self._build_observation(state, images)

    def step(self, action: np.ndarray) -> StepResult:
        """执行动作 (位置控制) + 仿真步进。"""
        action = self._apply_action(action)

        # 设置目标关节位置
        n_act = min(len(action), len(self.data.ctrl))
        self.data.ctrl[:n_act] = action[:n_act]

        # 多步仿真 (dt 步长 × control_interval = 1/control_freq 秒)
        for _ in range(self._control_interval):
            self._mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # 获取新状态
        state = self._get_state()
        images = self._render_image() if self.config.num_views > 0 else None
        obs = self._build_observation(state, images)

        # 计算奖励
        reward, done, info = self._compute_reward(obs)

        truncated = (self.step_count >= self.config.max_episode_steps)

        return StepResult(observation=obs, reward=reward, done=done,
                         truncated=truncated, info=info)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------ #
    # DR 支持
    # ------------------------------------------------------------------ #
    def set_dr_params(self, params: Dict[str, float]) -> None:
        """应用 DR 参数到模型。"""
        # 质量随机化
        if "link_mass" in params:
            scale = params["link_mass"]
            for i in range(self.model.nbody):
                if i > 0:  # 跳过 world
                    self.model.body_mass[i] *= scale

        # 摩擦随机化
        if "joint_friction" in params:
            scale = params["joint_friction"]
            self.model.dof_frictionloss *= scale

        # 阻尼随机化
        if "joint_damping" in params:
            for i in range(self.model.nv):
                self.model.dof_damping[i] *= params["joint_damping"]

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _get_state(self) -> np.ndarray:
        """获取关节状态 (位置 + 速度)。"""
        q = self.data.qpos[self._qpos_start: self._qpos_start + self.config.action_dim]
        qd = self.data.qvel[self._qpos_start: self._qpos_start + self.config.action_dim]

        # pad 到 state_dim
        state_len = min(len(q) + len(qd), self.config.state_dim)
        state = np.zeros(self.config.state_dim)
        state[:len(q)] = q
        state[len(q):state_len] = qd[:state_len - len(q)]
        return state

    def _render_image(self) -> Optional[np.ndarray]:
        """渲染 RGB 图像。"""
        try:
            if self._viewer is None:
                self._viewer = self._mujoco.Renderer(self.model, *self.config.image_size)
            self._viewer.update_scene(self.data)
            return self._viewer.render()
        except Exception:
            return None

    def _compute_reward(self, obs: Dict[str, Any]) -> tuple:
        """计算桌面抓取奖励。"""
        reward = 0.0
        done = False
        info = {}

        if self.config.scene_name in ("franka_table", "table_pick"):
            # 简单距离奖励: 末端到杯子的距离
            try:
                ee_id = self.model.site_name2id("ee_site")
                ee_pos = self.data.site_xpos[ee_id]
                cup_pos = np.array([0.5, 0, 0.45])
                dist = np.linalg.norm(ee_pos - cup_pos)
                reward = -dist * 0.1  # 距离惩罚

                if dist < 0.05:
                    reward += 10.0
                    done = True
                    info["success"] = True
            except Exception:
                info["success"] = False
        elif self.config.scene_name == "biped":
            # 双足行走奖励: 前进 + 存活
            torso_x = self.data.qpos[0]  # 前进方向
            reward = torso_x * 0.1  # 前进奖励
            # 摔倒检测 (躯干高度 < 0.5)
            if self.data.qpos[2] < 0.5:
                reward -= 10.0
                done = True
                info["fall"] = True

        return reward, done, info
