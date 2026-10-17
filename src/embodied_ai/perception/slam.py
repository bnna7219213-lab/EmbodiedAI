"""SLAM — 三维坐标 + 时间戳 (plan_v5 Sprint M: M2)。

**缺口背景（机器人端三大缺口之一）**
    机器人需要三维坐标信息：**自身所处位置** + **对象（要处理的事件或物体）
    所处位置**，以及**事件发生的时间**。此前项目仅有 MultiRateEKF（接触力/μ
    估计），没有任何自身/对象定位（SLAM）能力。

**设计（最小可用，不过度设计）**
    - `EKFSLAM2D`: 平面 EKF-SLAM —— 机器人恒速状态 [x, y, θ, v, ω] +
      路标 [mx, my] 增广；观测 range/bearing；`predict(dt)` /
      `update_landmark(id, range, bearing, t)`。
    - **z 通道恒速直通**：z 位置/速度一阶直通（完整 3D EKF SLAM 留待后续，
      接口已按 3D 返回 `robot_pose` = [x, y, z, θ]）。
    - `PoseStore`: who-pose-where-when 快照（自身/对象位姿 + 时间戳），
      供认知中枢四要素的"时间/地点"使用。
    - 全部位姿带 `timestamp_s`（事件时间要素）。

**不做的（诚实边界）**
    - 不做完整 3D EKF SLAM / 视觉 SLAM / 回环检测（Gate 3 后按需）
    - 不做真实传感器驱动（观测由上游 SLAM 前端提供, provenance=待领域验证）
    - z 通道是恒速直通近似，**不是**真 3D 估计：机器人 z 按常数 vz 积分，
      路标 z 只是末次观测的直通值（不被 range/bearing 约束、不进协方差）。
      接口按 3D 返回是为了上游不必日后改签名，**精度上只有 x/y 是估计量**。

**反证要求（测试守护）**
    带噪仿真位姿误差收敛 (<0.3m)；路标增广正确；时间戳单调；
    坏路标 id / 空观测不崩。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════
# PoseStore — 对象位置 + 时间
# ══════════════════════════════════════════════════════════════════

@dataclass
class PoseSnapshot:
    """一次位姿快照 — who / where(3D) / when(时间戳)。"""
    who: str                      # "robot" 或 对象/路标 id
    position: np.ndarray          # [x, y, z]
    yaw: float                    # 朝向 (rad, 对象可为 0)
    timestamp_s: float

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64).ravel()
        if self.position.size != 3:
            raise ValueError(
                f"PoseSnapshot.position 必须是 3 维 [x, y, z], 得 {self.position.size} 维 "
                "(robot_pose 是 4 维 [x,y,z,θ], 取 [:3] 传入)")


class PoseStore:
    """位姿快照存取 — 认知中枢"地点+时间"要素的数据源。

    给了 `bus` 时, `add()` 同时把快照发布成 Bus `POSE_UPDATE`。此前 plan_v5 M5
    宣称交付了该 topic, 但它在 src 与 tests 里**零发布者、零订阅者** ——
    "SLAM→PoseStore→认知中枢地点要素"这条数据流只存在于 docstring 里。
    """

    def __init__(self, bus=None, source: str = "slam"):
        self._snapshots: List[PoseSnapshot] = []
        self.bus = bus
        self.source = source

    def add(self, snapshot: PoseSnapshot) -> None:
        self._snapshots.append(snapshot)
        if self.bus is not None:
            # topic 用字符串名 — InProcessBus._to_topic 负责解析, 于是感知层
            # 不必 import 运行层 (与 sensor_fusion 的发布方式一致)
            self.bus.publish(
                "pose_update",
                {"pose": np.asarray(snapshot.position, dtype=np.float64),
                 "who": snapshot.who,
                 "yaw": float(snapshot.yaw),
                 "timestamp_s": float(snapshot.timestamp_s)},
                source=self.source,
            )

    def latest(self, who: str) -> Optional[PoseSnapshot]:
        for s in reversed(self._snapshots):
            if s.who == who:
                return s
        return None

    def history(self, who: Optional[str] = None) -> List[PoseSnapshot]:
        return [s for s in self._snapshots if who is None or s.who == who]

    def __len__(self) -> int:
        return len(self._snapshots)


# ══════════════════════════════════════════════════════════════════
# EKFSLAM2D — 平面 EKF-SLAM (z 直通)
# ══════════════════════════════════════════════════════════════════

class EKFSLAM2D:
    """平面 EKF-SLAM: 机器人恒速模型 + 路标增广 + range/bearing 观测。

    状态 x = [x, y, θ, v, ω, m1x, m1y, m2x, m2y, ...]
    过程: 恒速运动模型 (v, ω 控制输入可选)
    观测: range/bearing 到已知路标; 首次观测自动增广路标。
    z 通道: 恒速直通 (dz/dt = vz, vz 常数), 不参与 EKF。
    """

    #: 过程噪声协方差 Q (assumed — 合成验证用, 真实标定属 待领域验证)
    Q_ROBOT = np.diag([0.05, 0.05, 0.02, 0.1, 0.05])

    #: 观测噪声**标准差** (assumed) — 单一真源, 滤波器与仿真生成都从这里取
    SIGMA_RANGE_M = 0.15
    SIGMA_BEARING_RAD = 0.05

    #: 观测噪声协方差 R = diag(σ²)。
    #: 曾经写成 np.diag([0.15, 0.05]) 而注释标 "std" —— S = HPHᵀ + R 需要的是
    #: **方差**, 于是滤波器把 0.15m 当成 0.387m 的 std, 比真实噪声悲观 2.6x /
    #: 4.5x, 收敛判据 "<0.3m" 是在这个量纲错配下达成的。
    R_OBS = np.diag([SIGMA_RANGE_M ** 2, SIGMA_BEARING_RAD ** 2])

    def __init__(self, x0: Optional[np.ndarray] = None,
                 P0: Optional[np.ndarray] = None):
        x0 = np.zeros(5) if x0 is None else np.asarray(x0, dtype=np.float64).ravel()
        if x0.size != 5:
            raise ValueError("x0 必须是 5 维 [x, y, θ, v, ω]")
        self.x = x0.copy()
        self.P = (np.eye(5) * 0.5 if P0 is None
                  else np.asarray(P0, dtype=np.float64))
        self._landmark_index: Dict[str, int] = {}   # id → 状态中 [mx] 的下标
        self._landmark_z: Dict[str, float] = {}     # id → z 直通值 (不进 EKF 状态)
        self.z = 0.0                    # z 通道位置 (恒速直通)
        self.vz = 0.0                   # z 通道速度 (常数)
        self.last_timestamp_s: Optional[float] = None
        self.last_F: Optional[np.ndarray] = None   # 最近一次 predict 的运动雅可比

    # ---------------- 运动模型 ---------------- #

    def predict(self, dt: float, timestamp_s: Optional[float] = None) -> np.ndarray:
        """恒速运动模型时间更新。"""
        if dt <= 0:
            raise ValueError("dt 必须 > 0")
        if timestamp_s is not None:
            if self.last_timestamp_s is not None and timestamp_s < self.last_timestamp_s:
                raise ValueError("时间戳必须单调递增")
        x, y, theta0, v, omega = self.x[:5]
        x += v * np.cos(theta0) * dt
        y += v * np.sin(theta0) * dt
        theta = theta0 + omega * dt
        self.x[:5] = [x, y, theta, v, omega]
        self.z += self.vz * dt          # z 直通

        n = self.x.size
        F = np.eye(n)
        # 雅可比必须在**积分前的 theta0** 上取 (∂f/∂x|x_k), 用积分后的 theta
        # 会让 F 与真实传播不一致, 协方差随转向持续偏置。
        F[0, 2] = -v * np.sin(theta0) * dt
        F[0, 3] = np.cos(theta0) * dt
        F[1, 2] = v * np.cos(theta0) * dt
        F[1, 3] = np.sin(theta0) * dt
        F[2, 4] = dt
        Q = np.zeros((n, n))
        Q[:5, :5] = self.Q_ROBOT * dt
        self.P = F @ self.P @ F.T + Q
        self.last_F = F                 # 暴露给调试/有限差分核验

        if timestamp_s is not None:
            self.last_timestamp_s = timestamp_s
        return self.x.copy()

    # ---------------- 路标增广 ---------------- #

    def _augment_landmark(self, lid: str, mx: float, my: float,
                          mz: float = 0.0) -> int:
        n = self.x.size
        self.x = np.concatenate([self.x, [mx, my]])
        P_new = np.eye(n + 2) * 1.0
        P_new[:n, :n] = self.P
        self.P = P_new
        self._landmark_index[lid] = n
        self._landmark_z[lid] = float(mz)
        return n

    # ---------------- 观测更新 ---------------- #

    def update_landmark(self, lid: str, range_m: float, bearing_rad: float,
                        timestamp_s: Optional[float] = None,
                        mz: float = 0.0) -> np.ndarray:
        """range/bearing 观测更新; 未知路标自动增广。"""
        if range_m < 0:
            raise ValueError("range_m 必须 >= 0")
        if timestamp_s is not None:
            if self.last_timestamp_s is not None and timestamp_s < self.last_timestamp_s:
                raise ValueError("时间戳必须单调递增")
        rx, ry, theta = self.x[0], self.x[1], self.x[2]
        if lid not in self._landmark_index:
            mx = rx + range_m * np.cos(theta + bearing_rad)
            my = ry + range_m * np.sin(theta + bearing_rad)
            self._augment_landmark(lid, mx, my, mz)
            if timestamp_s is not None:
                self.last_timestamp_s = timestamp_s
            return self.x.copy()

        idx = self._landmark_index[lid]
        self._landmark_z[lid] = float(mz)   # z 直通: 末次观测覆盖 (不进 EKF)
        mx, my = self.x[idx], self.x[idx + 1]
        dx, dy = mx - rx, my - ry
        q = dx * dx + dy * dy
        dist = np.sqrt(q)
        # 预测观测 h(x) = [range, bearing]
        bearing_pred = np.arctan2(dy, dx) - theta
        bearing_pred = (bearing_pred + np.pi) % (2 * np.pi) - np.pi
        z_pred = np.array([dist, bearing_pred])
        z_meas = np.array([range_m, (bearing_rad + np.pi) % (2 * np.pi) - np.pi])

        n = self.x.size
        H = np.zeros((2, n))
        H[0, 0] = -dx / dist
        H[0, 1] = -dy / dist
        H[0, idx] = dx / dist
        H[0, idx + 1] = dy / dist
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1.0
        H[1, idx] = -dy / q
        H[1, idx + 1] = dx / q

        innov = z_meas - z_pred
        innov[1] = (innov[1] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R_OBS
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ innov).ravel()
        I_KH = np.eye(n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_OBS @ K.T

        if timestamp_s is not None:
            self.last_timestamp_s = timestamp_s
        return self.x.copy()

    # ---------------- 查询接口 ---------------- #

    @property
    def robot_pose(self) -> np.ndarray:
        """自身位姿 [x, y, z, θ]（z 为恒速直通值）。"""
        return np.array([self.x[0], self.x[1], self.z, self.x[2]])

    @property
    def robot_xy(self) -> np.ndarray:
        return self.x[:2].copy()

    def landmark_position(self, lid: str) -> np.ndarray:
        """对象位置 [x, y, z]。

        x/y 由 EKF 估计; **z 是直通值** (末次 update_landmark 的 mz), 未被
        range/bearing 观测约束, 协方差里也没有它 —— 不要把 z 当估计量用。
        """
        if lid not in self._landmark_index:
            raise KeyError(f"未知路标 '{lid}'")
        i = self._landmark_index[lid]
        return np.array([self.x[i], self.x[i + 1], self._landmark_z[lid]])

    @property
    def landmark_ids(self) -> List[str]:
        return list(self._landmark_index)


# ══════════════════════════════════════════════════════════════════
# demo — 带噪仿真轨迹
# ══════════════════════════════════════════════════════════════════

def slam_demo(obs_noise_scale: float = 1.0, bus=None) -> dict:
    """带噪仿真: 直线运动 + 2 路标 range/bearing 观测 → 位姿误差收敛。

    Args:
        obs_noise_scale: 观测噪声相对滤波器 σ 的倍率。1.0 = **匹配**滤波
            (生成 σ 与滤波器 R 完全相同, 收敛几乎必然); >1.0 = 失配, 用来检验
            "<0.3m" 到底是滤波器能力还是自评巧合。
        bus: 可选 InProcessBus —— 给了就发布 POSE_UPDATE (SLAM→认知中枢的
            地点要素数据流, 端到端可观测)。
    """
    if obs_noise_scale <= 0:
        raise ValueError("obs_noise_scale 必须 > 0")
    rng = np.random.default_rng(7)
    slam = EKFSLAM2D(x0=np.array([0.0, 0.0, 0.0, 1.0, 0.0]))
    store = PoseStore(bus=bus)
    landmarks_true = {"hold_a": np.array([3.0, 2.0, 0.0]),
                      "hold_b": np.array([5.0, -1.0, 0.0])}
    sigma_range = EKFSLAM2D.SIGMA_RANGE_M * obs_noise_scale
    sigma_bearing = EKFSLAM2D.SIGMA_BEARING_RAD * obs_noise_scale
    pose_true = np.array([0.0, 0.0])
    dt = 0.1
    errors = []
    for step in range(50):
        t = step * dt
        pose_true = pose_true + np.array([1.0, 0.0]) * dt    # 真实: 沿 x 匀速
        slam.predict(dt, timestamp_s=t)
        for lid, mp in landmarks_true.items():
            d = mp[:2] - pose_true
            r = float(np.linalg.norm(d)) + rng.normal(0, sigma_range)
            b = float(np.arctan2(d[1], d[0])) + rng.normal(0, sigma_bearing)
            slam.update_landmark(lid, r, b, timestamp_s=t)
        errors.append(float(np.linalg.norm(slam.robot_xy - pose_true)))
        # robot_pose 是 4 维 [x,y,z,θ]; PoseSnapshot.position 只要 3 维位置,
        # 朝向走 yaw 字段 (此前把 4 维整体塞进 position, 静默带着 θ 当坐标流转)
        store.add(PoseSnapshot("robot", slam.robot_pose[:3].copy(),
                               float(slam.x[2]), t))
    final_err = errors[-1]
    result = {
        "final_position_error_m": final_err,
        "error_curve_tail": errors[-5:],
        "obs_noise_scale": obs_noise_scale,
        "landmarks": {lid: slam.landmark_position(lid).tolist()
                      for lid in slam.landmark_ids},
        "snapshots": len(store),
    }
    if obs_noise_scale == 1.0:
        assert final_err < 0.3, f"位姿误差 {final_err:.3f}m 未收敛 (<0.3m)"
    return result


if __name__ == "__main__":
    r = slam_demo()
    print("SLAM demo (provenance=assumed 噪声 / synthetic 轨迹)")
    print(f"50 步后位姿误差: {r['final_position_error_m']:.3f} m (<0.3m ✓)")
    print(f"路标估计: {r['landmarks']}")
    print(f"位姿快照: {r['snapshots']} 条")
