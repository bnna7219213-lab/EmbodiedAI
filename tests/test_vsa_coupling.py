"""VSA-dyno 联动测试 — Kotlin JSON 消费 / 蓄能→弹道注入 / 拓扑选择 / 捕获口径。

数据源: EdaLab vsa-static --json 的 live 产物 (experiments_vsa_curve.json)。
"""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.dynamics.vsa_coupling import (VsaTopologyData, capture_feasibility,
                                               inject_into_dyno, load_vsa_json,
                                               release_velocity, select_topology)

JSON_PATH = Path(r"C:\Users\bnna7\workspace\EdaLab\experiments_vsa_curve.json")


def _tops():
    return load_vsa_json(JSON_PATH)


class TestVsaCoupling(unittest.TestCase):

    def test_load_json_schema(self):
        tops = _tops()
        self.assertEqual(len(tops), 3)
        for t in tops:
            self.assertEqual(len(t.curve), 25)
            self.assertGreater(t.stored_energy_j, 0)
            self.assertGreater(t.bandwidth_max_hz, t.bandwidth_min_hz)

    def test_antagonistic_selected_by_default(self):
        """默认选拓扑: 峰值力最接近指尖 Hertz 60N → ANTAGONISTIC (109N)。"""
        best = select_topology(_tops())
        self.assertEqual(best.topology, "ANTAGONISTIC")

    def test_select_by_name(self):
        best = select_topology(_tops(), name="VSA-MACCEPA-ClimbingArm")
        self.assertEqual(best.topology, "MACCEPA")

    def test_release_velocity_from_stored_energy(self):
        """v = √(2E/m) — 从 Kotlin 蓄能计算, 非硬编码。"""
        t = select_topology(_tops(), name="VSA-MACCEPA-ClimbingArm")
        v = release_velocity(t, direction_deg=45)
        expected = math.sqrt(2 * t.stored_energy_j / t.mass_kg)
        self.assertAlmostEqual(np.hypot(*v), expected, places=5)
        # 与 Kotlin 引擎输出一致 (跨语言数值闭环)
        self.assertAlmostEqual(np.hypot(*v), t.release_velocity_mps, places=3)

    def test_stiffness_interpolation(self):
        t = _tops()[0]
        pts = sorted(t.curve, key=lambda p: p.theta_deg)
        mid = 0.5 * (pts[10].theta_deg + pts[11].theta_deg)
        k = t.stiffness_at(mid)
        self.assertTrue(pts[10].k_nm_per_rad - 1 <= k <= pts[11].k_nm_per_rad + 1)

    def test_inject_into_dyno(self):
        t = select_topology(_tops())
        r = inject_into_dyno(t, np.array([0.0, 0.0]), np.array([0.6, 0.8]))
        self.assertIn("dyno", r)
        for key in ("tf", "reach_req", "c_tf", "reachable"):
            self.assertIn(key, r["dyno"])
        self.assertGreater(r["speed_mps"], 0)
        self.assertGreater(r["dyno"]["tf"], 0)

    def test_capture_feasibility_honest(self):
        """捕获口径诚实: MACCEPA 直弹速度超刚性捕获上限 (需柔顺指尖) — 物理正确结论。"""
        t = select_topology(_tops(), name="VSA-MACCEPA-ClimbingArm")
        v_max, v_vsa, feasible = capture_feasibility(t)
        self.assertGreater(v_vsa, 0)
        # 刚性脉冲捕获上限很小 (μ·F·t/m): 0.8*45*0.005/1.2 = 0.15 m/s
        self.assertAlmostEqual(v_max, 0.8 * 45 * 0.005 / 1.2, places=3)
        self.assertFalse(feasible, "1.27 m/s 直弹必然超刚性上限 → 需 Hunt-Crossley 柔顺")


if __name__ == "__main__":
    unittest.main(verbosity=2)
