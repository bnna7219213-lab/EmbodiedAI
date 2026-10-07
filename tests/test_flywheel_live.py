"""EdaLab data-flywheel live 输出解析测试 — 统计解析/约束重采样/时间戳防误匹配。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.adapters.flywheel_live import (flywheel_stats_to_telemetry,
                                                live_flywheel_to_records,
                                                parse_flywheel_stdout)

# 2026-10-06 实测的 live stdout 样本 (LIVE_OK, data-flywheel)
LIVE_STDOUT = """=== SynapticForge 制造数据回流闭环 ===
生成 DR 配置: Version v2.0
基于 1 块板子
DR 参数数: 4
  canfd_latency_ms: 0.0 - 2.0 (基于 1 块板子的 CAN-FD 总线利用率)
  load_torque_scale: 1.0 - 1.0 (基于 100 条部署遥测)
  friction_coefficient: 0.0 - 0.0 (基于部署数据中的摩擦变化推断)
  ambient_temp: 15.0 - 45.0 (基于部署数据: 最高运行环境温度 45.0°C)

=== Board Performance Comparison Report ===
Generated: 2026-10-06T01:20:58.888020400Z

Board: SF-MDB-0001
  ICT: PASS
  FCT: PASS
  Telemetry records: 100
  Task success rate: 100.0%
  Total cycles: 5050
"""


class TestFlywheelLiveParser(unittest.TestCase):

    def test_parse_statistics(self):
        s = parse_flywheel_stdout(LIVE_STDOUT)
        self.assertEqual(s["board_id"], "SF-MDB-0001")
        self.assertEqual(s["telemetry_records"], 100)
        self.assertEqual(s["success_rate"], 100.0)
        self.assertEqual(s["total_cycles"], 5050)
        self.assertEqual(s["ict"], "PASS")

    def test_parse_dr_ranges(self):
        s = parse_flywheel_stdout(LIVE_STDOUT)
        self.assertEqual(s["dr_ranges"]["canfd_latency_ms"], [0.0, 2.0])
        self.assertEqual(s["dr_ranges"]["ambient_temp"], [15.0, 45.0])

    def test_no_timestamp_false_match(self):
        """时间戳 2026-10-06T... 不得被误认为 DR 参数 (曾触发 bug)。"""
        s = parse_flywheel_stdout(LIVE_STDOUT)
        for k in s["dr_ranges"]:
            self.assertIn(k, ("canfd_latency_ms", "load_torque_scale",
                              "friction_coefficient", "ambient_temp"))

    def test_stats_to_telemetry_count_and_provenance(self):
        recs = live_flywheel_to_records(LIVE_STDOUT, joint="shoulder_L", seed=7)
        self.assertEqual(len(recs), 100)  # 与统计量的 telemetry_records 一致
        for r in recs:
            self.assertEqual(r["source"], "edalab_flywheel_live_statistics")
            self.assertEqual(r["board_serial"], "SF-MDB-0001")
            self.assertIn("_stats_anchor", r)  # 统计锚点可追溯
        # 温度落在 live DR 范围 [15, 45] 内
        temps = [r["temperature_c"] for r in recs]
        self.assertGreaterEqual(min(temps), 15.0)
        self.assertLessEqual(max(temps), 45.0)

    def test_success_rate_maps_to_faults(self):
        stats = parse_flywheel_stdout(LIVE_STDOUT)
        stats["success_rate"] = 90.0  # 注入 10% 故障率
        recs = flywheel_stats_to_telemetry(stats, seed=1)
        n_fault = sum(1 for r in recs if r["fault_events"])
        self.assertGreater(n_fault, 0, "90% 成功率应产生故障注入记录")


if __name__ == "__main__":
    unittest.main(verbosity=2)
