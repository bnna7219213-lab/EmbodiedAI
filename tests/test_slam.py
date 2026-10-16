"""M2 SLAM 三维坐标 + 时间戳 测试 (plan_v5 Sprint M)。

反证: 位姿误差收敛; 路标增广; 时间戳单调; 坏输入不崩; PoseStore 查询。
"""
import numpy as np
import pytest

from embodied_ai.perception.slam import (
    EKFSLAM2D,
    Landmark,
    PoseSnapshot,
    PoseStore,
    slam_demo,
)


class TestLandmark:
    def test_position_must_be_3d(self):
        lm = Landmark("a", [1.0, 2.0, 0.0])
        assert lm.position.shape == (3,)
        with pytest.raises(ValueError):
            Landmark("b", [1.0, 2.0])

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

    def test_error_converges(self):
        r = slam_demo()
        assert r["final_position_error_m"] < 0.3
        assert r["snapshots"] == 50

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
