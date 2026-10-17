"""M2 SLAM 三维坐标 + 时间戳 测试 (plan_v5 Sprint M)。

反证: 位姿误差收敛; 路标增广; 时间戳单调; 坏输入不崩; PoseStore 查询。
"""
import numpy as np
import pytest

from embodied_ai.perception.slam import (
    EKFSLAM2D,
    PoseSnapshot,
    PoseStore,
    slam_demo,
)


class TestPoseStore:
    def test_pose_store(self):
        store = PoseStore()
        store.add(PoseSnapshot("robot", np.zeros(3), 0.0, 0.0))
        store.add(PoseSnapshot("robot", np.ones(3), 0.1, 1.0))
        store.add(PoseSnapshot("obj", np.ones(3) * 2, 0.0, 1.0))
        assert len(store) == 3
        assert store.latest("robot").timestamp_s == 1.0
        assert store.latest("obj").timestamp_s == 1.0
        assert store.latest("none") is None
        assert len(store.history("robot")) == 2
        assert len(store.history()) == 3


class TestEKFSLAM2D:
    def test_predict_moves_robot(self):
        slam = EKFSLAM2D(x0=np.array([0.0, 0.0, 0.0, 1.0, 0.0]))
        slam.predict(0.5, timestamp_s=0.5)
        assert slam.robot_xy[0] == pytest.approx(0.5)
        assert slam.last_timestamp_s == 0.5

    def test_predict_rejects_bad_dt(self):
        slam = EKFSLAM2D()
        with pytest.raises(ValueError):
            slam.predict(0.0)

    def test_landmark_augmentation(self):
        slam = EKFSLAM2D()
        assert slam.landmark_ids == []
        slam.update_landmark("hold", 2.0, 0.5, timestamp_s=0.1)
        assert slam.landmark_ids == ["hold"]
        # 增广后位置应在 range/bearing 对应处
        pos = slam.landmark_position("hold")
        expected = np.array([2.0 * np.cos(0.5), 2.0 * np.sin(0.5)])
        assert pos[:2] == pytest.approx(expected, abs=1e-6)

    def test_unknown_landmark_query_raises(self):
        slam = EKFSLAM2D()
        with pytest.raises(KeyError):
            slam.landmark_position("ghost")

    def test_negative_range_raises(self):
        slam = EKFSLAM2D()
        with pytest.raises(ValueError):
            slam.update_landmark("a", -1.0, 0.0)

    def test_timestamp_monotonic(self):
        slam = EKFSLAM2D()
        slam.predict(0.1, timestamp_s=1.0)
        with pytest.raises(ValueError):
            slam.predict(0.1, timestamp_s=0.5)
        with pytest.raises(ValueError):
            slam.update_landmark("a", 1.0, 0.0, timestamp_s=0.5)

    def test_bad_x0_raises(self):
        with pytest.raises(ValueError):
            EKFSLAM2D(x0=np.zeros(3))


class TestPoseStoreBusWiring:
    """回归: plan_v5 M5 宣称交付 POSE_UPDATE topic, 但它在 src 与 tests 里
    零发布者、零订阅者 —— "SLAM→认知中枢地点要素"只存在于 docstring。"""

    def test_pose_store_publishes_pose_update(self):
        from embodied_ai.runtime.bus import Collector, InProcessBus

        bus = InProcessBus("test_slam_bus")
        got = Collector()
        bus.subscribe("pose_update", got)
        store = PoseStore(bus=bus)
        store.add(PoseSnapshot("robot", np.array([1.0, 2.0, 3.0]), 0.5, 1.25))
        assert len(got.messages) == 1
        p = got.last.payload
        assert np.allclose(p["pose"], [1.0, 2.0, 3.0])
        assert p["who"] == "robot" and p["timestamp_s"] == 1.25
        assert got.last.source == "slam"

    def test_pose_store_without_bus_standalone(self):
        """无 bus 时 PoseStore 必须仍能独立工作 (感知层不硬依赖运行层)。"""
        store = PoseStore()
        store.add(PoseSnapshot("robot", np.zeros(3), 0.0, 0.0))
        assert len(store) == 1

    def test_pose_snapshot_rejects_4d_position(self):
        """robot_pose 是 [x,y,z,θ] 4 维 —— 曾整只塞进 position 而不被察觉。"""
        with pytest.raises(ValueError, match="3 维"):
            PoseSnapshot("robot", np.array([1.0, 2.0, 3.0, 0.4]), 0.0, 0.0)

    def test_slam_demo_publishes_one_message_per_snapshot(self):
        from embodied_ai.runtime.bus import Collector, InProcessBus

        bus = InProcessBus("test_slam_demo_bus")
        got = Collector()
        bus.subscribe("pose_update", got)
        r = slam_demo(bus=bus)
        assert r["snapshots"] == 50
        assert len(got.messages) == 50
        assert got.last.payload["who"] == "robot"
        assert len(got.last.payload["pose"]) == 3

    def test_error_converges(self):
        r = slam_demo()
        assert r["final_position_error_m"] < 0.3
        assert r["snapshots"] == 50

    def test_r_obs_is_variance_not_std(self):
        """R_OBS 对角线必须是 σ² —— 曾把 std 直接填进协方差, 量纲错 2.6x/4.5x。"""
        assert EKFSLAM2D.R_OBS[0, 0] == pytest.approx(EKFSLAM2D.SIGMA_RANGE_M ** 2)
        assert EKFSLAM2D.R_OBS[1, 1] == pytest.approx(EKFSLAM2D.SIGMA_BEARING_RAD ** 2)

    def test_noise_mismatch_degrades_gracefully(self):
        """生成噪声 ≠ 滤波器 R 时仍应有界 (匹配滤波自评之外的真实检验)。"""
        assert slam_demo(obs_noise_scale=2.0)["final_position_error_m"] < 0.3

    def test_convergence_criterion_can_actually_fail(self):
        """负对照: 判据必须有区分力。

        3x 失配下误差实测 ~0.39m > 0.3m —— 证明 "<0.3m" 不是恒真门槛。
        (匹配时实测 ~0.013m, 判据有 23x 余量, 所以必须靠失配来证明它有效。)
        """
        assert slam_demo(obs_noise_scale=3.0)["final_position_error_m"] > 0.3

    def test_bad_noise_scale_raises(self):
        with pytest.raises(ValueError):
            slam_demo(obs_noise_scale=0.0)

    def test_landmark_estimate_near_truth(self):
        r = slam_demo()
        # hold_a 真值 (3, 2), 估计应接近 (噪声 std 0.15m 量级)
        est = np.array(r["landmarks"]["hold_a"][:2])
        assert np.linalg.norm(est - np.array([3.0, 2.0])) < 1.0

    def test_z_passthrough(self):
        slam = EKFSLAM2D()
        slam.vz = 0.2
        slam.predict(1.0, timestamp_s=1.0)
        assert slam.robot_pose[2] == pytest.approx(0.2)
        assert slam.robot_pose.shape == (4,)

    def test_landmark_z_not_silently_dropped(self):
        """mz 曾被 _augment_landmark 收下后丢弃, landmark_position 硬编码 z=0.0
        —— 传 mz=1.75 拿回 0.0, "三维坐标"的对象 z 是假的。"""
        slam = EKFSLAM2D()
        slam.update_landmark("shelf", 2.0, 0.3, timestamp_s=0.1, mz=1.75)
        assert slam.landmark_position("shelf")[2] == pytest.approx(1.75)
        # 后续观测末次覆盖 (z 是直通值, 不是估计量)
        slam.update_landmark("shelf", 2.0, 0.3, timestamp_s=0.2, mz=1.20)
        assert slam.landmark_position("shelf")[2] == pytest.approx(1.20)
        # x/y 仍由 EKF 估计, 未被 z 直通污染
        assert slam.landmark_position("shelf")[:2] == pytest.approx(
            np.array([2.0 * np.cos(0.3), 2.0 * np.sin(0.3)]), abs=1e-3)

    def test_predict_jacobian_matches_motion_model(self):
        """F 必须是运动模型在**积分前**状态上的雅可比。

        曾经 theta 先自增再构造 F, 雅可比取在积分后的 theta 上, 与真实传播
        不一致 —— 协方差会随转向持续偏置, 而位姿误差测试看不出来。
        """
        dt = 0.05
        slam = EKFSLAM2D(x0=np.array([0.5, -0.3, 0.7, 1.2, 0.4]))
        x0 = slam.x.copy()
        slam.predict(dt)
        F = slam.last_F[:5, :5]

        def motion(x):
            px, py, th, v, om = x
            return np.array([px + v * np.cos(th) * dt, py + v * np.sin(th) * dt,
                             th + om * dt, v, om])

        eps = 1e-6
        F_num = np.zeros((5, 5))
        for j in range(5):
            xp, xm = x0.copy(), x0.copy()
            xp[j] += eps
            xm[j] -= eps
            F_num[:, j] = (motion(xp) - motion(xm)) / (2 * eps)
        assert F == pytest.approx(F_num, abs=1e-6)
