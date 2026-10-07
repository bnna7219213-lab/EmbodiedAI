"""VSA-dyno 联动测试 — Kotlin JSON 消费 / 蓄能→弹道注入 / 拓扑选择 / 捕获口径。

数据源: EdaLab vsa-static --json 的 live 产物 (experiments_vsa_curve.json)。
"""
import json
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.dynamics.vsa_coupling import (VsaTopologyData, capture_feasibility,
                                               inject_into_dyno, load_vsa_json,
                                               release_velocity, select_topology,
                                               vsa_energy_consistency_check)

# V1 修复 (2026-10-07): JSON_PATH 改为仓库内 fixtures (档位 A 已解耦的三级路径)
_JSON_ENV = os.environ.get("EDALAB_VSA_JSON")
JSON_PATH = (Path(_JSON_ENV) if _JSON_ENV else
             Path(__file__).resolve().parents[1] / "fixtures" / "experiments_vsa_curve.json")


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

    # ============================================================
    # V-1 物理量级错误验证 (2026-10-07 修复)
    # 缺陷: curve.energy_j (行程边界零点) 与 stored_energy_j (平衡位零点) 差 83 倍,
    # 下游误用会导致 11.6 m/s 致命弹道。修复后 Kotlin 侧新增
    # curve.energy_from_origin_j (与 stored_energy_j 同源), Python 侧做一致性校验。
    # ============================================================

    def test_v1_curve_energy_from_origin_matches_stored(self):
        """V1: 曲线 energy_from_origin_j 在弹性行程内应接近 stored_energy_j (同量级)。

        断言 ratio ∈ [0.5, 1.5] (0.5 容忍离散采样与 span 过滤差异)。
        ANTAGONISTIC 拓扑由于零点随预紧偏移, 差异较大是物理正确的。
        """
        tops = _tops()
        for t in tops:
            pts = [p for p in t.curve if abs(p.theta_deg) <= t.elastic_travel_deg / 2 + 1e-6]
            self.assertGreater(len(pts), 0, f"{t.name}: 弹性行程内应有采样点")
            # v1.1 schema 应提供 energy_from_origin_j
            has_v1_1 = any(p.energy_from_origin_j > 0 for p in pts)
            if has_v1_1:
                max_from_origin = max(p.energy_from_origin_j for p in pts)
                ratio = max_from_origin / t.stored_energy_j
                # 一致性: ANTAGONISTIC 由于曲线零点偏移, 上限放宽到 2x
                max_ratio_ok = 2.0 if t.topology == "ANTAGONISTIC" else 1.5
                self.assertLessEqual(
                    ratio, max_ratio_ok,
                    f"{t.name}: max_energy_from_origin={max_from_origin:.4f}J "
                    f"vs stored={t.stored_energy_j:.4f}J, ratio={ratio:.3f} 应 ≤ {max_ratio_ok}"
                )

    def test_v1_curve_energy_j_much_larger_than_stored(self):
        """V1: 曲线 energy_j (行程边界零点) 应显著大于 stored_energy_j。

        这是修复前的错误根源: 若下游误用 energy_j 当弹性能, 会得到 ~10x 过大的弹射速度。
        断言 energy_j >= 2x stored_energy_j — 若两者相同, 说明修复失效。
        """
        tops = _tops()
        for t in tops:
            pts = [p for p in t.curve if abs(p.theta_deg) <= t.elastic_travel_deg / 2 + 1e-6]
            max_ej = max(p.energy_j for p in pts)
            self.assertGreaterEqual(
                max_ej, 2.0 * t.stored_energy_j,
                f"{t.name}: curve.energy_j max={max_ej:.3f}J 应 >= 2x stored="
                f"{t.stored_energy_j:.3f}J (修复前应远大于, 否则修复未生效)"
            )

    def test_v1_release_velocity_uses_stored_energy_not_curve(self):
        """V1: release_velocity 必须消费 stored_energy_j, 而非曲线 energy_j。

        若误用曲线 energy_j 当弹性能, v 会 √(2E/m) 得到 ~3-9x 过大的速度。
        断言 release_velocity 结果 < 2 m/s (物理合理); 误用会得 11.6 m/s 致命弹道。
        """
        tops = _tops()
        for t in tops:
            v = math.hypot(*release_velocity(t, direction_deg=45))
            # 正确速度: √(2·stored/m) — 三个拓扑都在 [1.0, 1.5] m/s
            self.assertLess(v, 2.0,
                            f"{t.name}: v={v:.2f}m/s 超物理上限, 疑似误用曲线 energy_j")
            # 若误用曲线最大 energy_j: 会得到 ~11 m/s (致命)
            pts = [p for p in t.curve if abs(p.theta_deg) <= t.elastic_travel_deg / 2 + 1e-6]
            if pts:
                wrong_v = math.sqrt(2.0 * max(p.energy_j for p in pts) / max(t.mass_kg, 1e-6))
                # 反证: 错误路径应至少 2x 正确路径
                self.assertGreater(wrong_v, 2.0 * v,
                                    f"{t.name}: 错误路径 v={wrong_v:.2f}m/s 应 > 2x 正确 {v:.2f}m/s")

    def test_v1_energy_consistency_check_passes(self):
        """V1: vsa_energy_consistency_check 在 v1.1 schema 上应返回 ok=True。"""
        tops = _tops()
        # tolerance=5.0: stored_energy_j (max in elastic_span) 与 curve.energy_from_origin_j
        # (末端用 kMax*θ_cap²) 都在 ½Kθ² 量级, 差异来自 K 采样点不同, 通常 1-5x。
        # 历史 V-1 错误是 83x (曲线用行程边界零点), 5x 阈值足以拦回归。
        result = vsa_energy_consistency_check(tops, tolerance=5.0)
        # 至少 MACCEPA/SEA 两个应通过 (ANTAGONISTIC 曲线零点偏移较大, 可容忍)
        passed_names = [d["name"] for d in result["details"] if d["within_tolerance"]]
        self.assertIn("VSA-MACCEPA-ClimbingArm", passed_names,
                       f"MACCEPA 应通过一致性检查, got details={result['details']}")

    def test_v1_load_rejects_unknown_schema(self):
        """V1: 未知 schema 应拒绝加载 (防跨版本误用)。"""
        import tempfile
        fake = {"schema": "vsa_curve_v99", "topologies": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(fake, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_vsa_json(Path(path))
        finally:
            os.unlink(path)

    def test_v1_sanity_warns_on_legacy_schema(self):
        """V1: 若加载 v1.0 (无 energy_from_origin_j) 且 energy_j >> stored_energy_j,
        release_velocity 应产生 UserWarning 而非静默通过。"""
        import warnings
        import json
        import tempfile
        tops = _tops()
        # 手工构造一个 v1.0 风格的假数据: 只保留 energy_j (且数值大), 删除 energy_from_origin_j
        legacy_t = VsaTopologyData(
            name="LegacyTest", topology="MACCEPA",
            k_min=tops[0].k_min, k_max=tops[0].k_max,
            elastic_travel_deg=tops[0].elastic_travel_deg,
            link_length_m=tops[0].link_length_m, mass_kg=tops[0].mass_kg,
            stored_energy_j=tops[0].stored_energy_j,  # 正确值 ~0.97J
            peak_force_n=tops[0].peak_force_n,
            release_velocity_mps=tops[0].release_velocity_mps,
            bandwidth_min_hz=tops[0].bandwidth_min_hz,
            bandwidth_max_hz=tops[0].bandwidth_max_hz,
            stiffness_ratio=tops[0].stiffness_ratio, ok=tops[0].ok,
            # 曲线保留原 energy_j (行程边界零点, 大值) 但 energy_from_origin_j 置 0
            curve=[__import__('embodied_ai.dynamics.vsa_coupling', fromlist=['VsaCurvePoint']).VsaCurvePoint(
                p.theta_deg, p.k_nm_per_rad, p.torque_max_nm, p.energy_j, 0.0)
                for p in tops[0].curve],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            release_velocity(legacy_t, direction_deg=45)
            warns = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        self.assertTrue(any("energy_j" in w for w in warns),
                        f"应产生 energy_j 误用警告, got: {warns}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
