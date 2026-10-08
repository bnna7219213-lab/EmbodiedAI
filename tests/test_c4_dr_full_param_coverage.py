"""Gate 2 C4 收官: DR 参数四节全覆盖测试。

诊断基线 (v0.8):
  - contracts/dr_params/manufacturing_variance.yaml 含 5 节:
    electrical / mechanical / sensing / environment / feedback_rules
  - 旧版 _diff_vs_contract 仅遍历 electrical → missing_from_dr 对机械/传感/环境
    三节永真 (空), 等于无校验。

本测试类直接断言 _diff_vs_contract 的行为:
  (a) 四节 param 全部进入计算 (within_spec 或 missing_from_dr);
  (b) missing_from_dr 不再"永真" — 为空当且仅当 overrides 真包含四节全部 key;
  (c) 跨节重复 param 名按首次声明节归属 (实际 yaml 无重复)。
"""
import sys
import unittest
from pathlib import Path

# 让 tests/ 能 import 顶层包 (兼容直接 pytest 运行与 IDE)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embodied_ai.adapters.edalab import FLOW_CHAIN, EdalabAdapter


# ---- 契约 yaml 中的四节全部顶层 key (手工清单, 与 yaml 同步) ----

_EXPECTED_CONTRACT_KEYS = {
    "electrical": {"mosfet_rds_on"},
    "mechanical": {"joint_stiffness", "link_mass", "link_length_mm", "gear_ratio", "gear_backlash_rad"},
    "sensing": {"force_sensor_calibration", "tactile_sensitivity", "friction_coefficient"},
    "environment": {"ambient_temp_c", "canfd_latency_ms", "load_torque_scale"},
}

_ALL_CONTRACT_KEYS = set().union(*_EXPECTED_CONTRACT_KEYS.values())


class TestC4DiffVsContractBehavior(unittest.TestCase):
    """行为级断言 — _diff_vs_contract 是否真正校验四节。"""

    def _diff(self, overrides: dict) -> dict:
        """用仅含四节的 contract 调用 SUT。"""
        contract = {
            "electrical": {k: {} for k in _EXPECTED_CONTRACT_KEYS["electrical"]},
            "mechanical": {k: {} for k in _EXPECTED_CONTRACT_KEYS["mechanical"]},
            "sensing": {k: {} for k in _EXPECTED_CONTRACT_KEYS["sensing"]},
            "environment": {k: {} for k in _EXPECTED_CONTRACT_KEYS["environment"]},
        }
        return EdalabAdapter._diff_vs_contract(
            EdalabAdapter,  # 绑定类 (self 位置)
            {"overrides": overrides},
            contract,
        )

    def test_empty_overrides_all_missing(self):
        """overrides 为空 → 四节全部 param 进 missing_from_dr (不再永真空列表)。"""
        diff = self._diff({})
        self.assertEqual(diff["within_spec"], [])
        self.assertEqual(diff["extra_in_dr"], [])
        # 关键: missing_from_dr 长度 = 四节全部 param 数量, 不再为空
        self.assertEqual(len(diff["missing_from_dr"]), len(_ALL_CONTRACT_KEYS))
        # 确认四节都有代表
        for section, keys in _EXPECTED_CONTRACT_KEYS.items():
            rep = next((m for m in diff["missing_from_dr"] if m.startswith(f"{section}.")), None)
            self.assertIsNotNone(rep, f"{section} 没有 param 进 missing_from_dr (永真 bug 未修复)")

    def test_partial_overrides_split_correctly(self):
        """overrides 只填 electrical → electrical 在 within_spec, 其余三节全部在 missing。"""
        diff = self._diff({"mosfet_rds_on": 0.003})
        self.assertIn("electrical.mosfet_rds_on", diff["within_spec"])
        # 其余三节 (mechanical/sensing/environment) 全部在 missing_from_dr 中
        missing_set = set(diff["missing_from_dr"])
        for sec in ("mechanical", "sensing", "environment"):
            for k in _EXPECTED_CONTRACT_KEYS[sec]:
                self.assertIn(f"{sec}.{k}", missing_set,
                              f"{sec}.{k} 应在 missing_from_dr 中")
        # electrical 只有一项 → missing 不再包含其他 electrical 项
        for m in diff["missing_from_dr"]:
            self.assertNotEqual(m, "electrical.mosfet_rds_on")

    def test_full_overrides_all_within_spec(self):
        """overrides 包含四节全部 key → missing_from_dr 为空 (全场 within_spec)。"""
        overrides = {k: 0.001 for k in _ALL_CONTRACT_KEYS}  # dummy values, 只看 key
        diff = self._diff(overrides)
        self.assertEqual(len(diff["missing_from_dr"]), 0)
        self.assertEqual(len(diff["within_spec"]), len(_ALL_CONTRACT_KEYS))
        self.assertEqual(diff["extra_in_dr"], [])

    def test_section_prefix_in_outputs(self):
        """输出带 section 前缀 (防跨节重名歧义)。"""
        diff = self._diff({"mosfet_rds_on": 0.005})
        for item in diff["within_spec"]:
            self.assertIn(".", item, "应含 section 前缀, 如 electrical.mosfet_rds_on")
        diff2 = self._diff({})
        for item in diff2["missing_from_dr"]:
            self.assertRegex(item, r"^(electrical|mechanical|sensing|environment)\.",
                              f"{item} 缺少 section 前缀")


class TestC4ContractYamlHasFiveSections(unittest.TestCase):
    """契约文件结构断言 — 5 节完整存在。"""

    YAML_PATH = Path(__file__).resolve().parent.parent / "contracts" / "dr_params" / "manufacturing_variance.yaml"

    def test_five_sections_present(self):
        if not self.YAML_PATH.exists():
            self.skipTest(f"yaml 不存在: {self.YAML_PATH}")
        import yaml
        with open(self.YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for sec in ("electrical", "mechanical", "sensing", "environment", "feedback_rules"):
            self.assertIn(sec, data, f"契约缺失 {sec} 节")

    def test_expected_keys_present(self):
        if not self.YAML_PATH.exists():
            self.skipTest(f"yaml 不存在: {self.YAML_PATH}")
        import yaml
        with open(self.YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for sec, expected_keys in _EXPECTED_CONTRACT_KEYS.items():
            actual = set(data.get(sec, {}).keys())
            missing = expected_keys - actual
            self.assertEqual(missing, set(), f"{sec} 节缺 key: {missing}")


if __name__ == "__main__":
    unittest.main()
