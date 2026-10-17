"""③ 数据扩充守护测试 — ≥1000 episodes 攀岩闭环数据引擎 (schema v2, seed→物理层 DR)。

这个文件的存在意义是**双向**的, 两个方向都要能失败:

  下界 (数据够不够): episode 数 ≥ 1000, 结构合法, 可被 BC 契约与训练管线消费。
  上界 (标注诚不诚实): provenance 必须是 simulated/assumed, **不得**出现
      real/measured —— v1 版把这些字段标成 real_fields, 而同仓硬件实物为零。

  以及本轮新增的关键一条: **seed 必须真的传进物理层**。
      v1 版的失败模式是"1000 条 episode 的 11 个物理字段唯一值数全为 1", 而当时的
      测试只断言 `0 < soft_ratio < 1` 与"取值有限" —— 一个恒真判据, 对退化完全无感,
      测试名 (`test_physics_real_not_random`) 承诺的多样性检查并不存在。现在它真的在测。

  DR 的合法副作用: soft_ratio 可以 >1 (指尖过硬时柔顺分支反而放大峰值力)。
      旧断言把它当错误, 实际是物理结果 → 改为要求有限且为正, 并单独断言跨 episode 变化。
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA = REPO / "artifacts" / "d4_climb" / "climb_episodes_full.jsonl"

# 必须随 seed 变化的物理量 (DR 的实际落点)
DRIVEN_FIELDS = ("f_soft_peak_n", "f_rigid_peak_n", "soft_ratio",
                 "com_advance_m", "peak_finger_n", "v_capture_max_mps")
# 结构上恒定的字段 —— 记下原因, 防止被当成"物理结论"误用
CONSTANT_FIELDS_REASON = {
    "capture_feasible": "捕获用二分把冲击速度解到 f_soft == hold.max_load_n, "
                        "可行性由**构造**保证, 不是物理结论",
    "wall_frames": "序列帧数由脚本编排决定, 非物理量",
}


def _read():
    with open(DATA, encoding="utf-8") as f:
        header = json.loads(f.readline())
        eps = [json.loads(line) for line in f]
    return header, eps


class TestClimbDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.header, cls.eps = _read()

    # ---------- 下界: 量与结构 ----------

    def test_dataset_exists_and_meets_threshold(self):
        self.assertTrue(DATA.exists(),
                        f"数据文件缺失: {DATA} (运行 scripts/gen_climb_rollouts.py)")
        self.assertGreaterEqual(len(self.eps), 1000,
                                f"episodes 不足: {len(self.eps)} < 1000")

    def test_episode_structure_valid(self):
        """obs/actions 等长、动作纯数值、有 instruction、坐标是**真数组**。"""
        for i, ep in enumerate(self.eps[:20]):
            self.assertEqual(len(ep["observations"]), len(ep["actions"]),
                             f"ep[{i}] obs/actions 不等长")
            self.assertTrue(ep["instruction"], f"ep[{i}] 缺 instruction")
            for a in ep["actions"][:3]:
                self.assertTrue(all(isinstance(x, (int, float)) for x in a),
                                f"ep[{i}] action 非数值")
            # v1 曾把坐标经 default=str 存成 numpy repr 字符串 `'[0.  0.  0.1 ...]'`
            o = ep["observations"][0]
            self.assertIsInstance(o["joint_positions"], list,
                                  "joint_positions 必须是 JSON 数组, 不是 numpy repr 字符串")
            self.assertEqual(len(o["joint_positions"]), 12)
            self.assertIsInstance(o["ee_position"], list)
            self.assertEqual(len(o["ee_position"]), 3)

    def test_validate_bc_dataset_full(self):
        from embodied_ai.adapters.nanogpt import NanoGPTAdapter
        result = NanoGPTAdapter.validate_bc_dataset(self.eps)
        self.assertTrue(result["ok"], f"validate 失败: {result['issues'][:3]}")
        self.assertGreaterEqual(result["n_episodes"], 1000)

    def test_loadable_by_replay_dataset(self):
        ep = self.eps[0]
        self.assertIn("episode_meta", ep)
        for key in ("f_soft_peak_n", "com_advance_m", "v_release"):
            self.assertIn(key, ep["episode_meta"]["physics"])

    # ---------- 上界: provenance 诚实 ----------

    def test_schema_is_v2(self):
        self.assertEqual(self.header["_schema"], "d4_climb_v2")

    def test_provenance_declares_simulated_not_real(self):
        """物理量是**仿真计算**产出, 不是实测; DR 输入参数是 assumed。"""
        prov = self.header["_provenance"]
        self.assertIn("f_soft_peak_n", prov["simulated_fields"])
        self.assertEqual(prov["simulated_provenance"], "simulated (物理引擎计算, 非实测)")
        self.assertEqual(prov["dr_parameter_provenance"],
                         "assumed (DR 区间为本仓设定, 非实测散布)")

    def test_no_field_claimed_real_or_measured(self):
        """反证: provenance 头里出现 real/measured 的过度宣称就必须失败。

        v1 版正是靠 `real_fields` 这个键名把仿真量说成实测。
        """
        blob = json.dumps(self.header, ensure_ascii=False).lower()
        for bad in ('"real_fields"', '"measured_fields"', "实测标定"):
            self.assertNotIn(bad, blob, f"provenance 头出现过度宣称: {bad}")
        self.assertIn("不冒充真实示教", self.header["_provenance"]["note"])

    def test_dr_params_recorded_per_episode(self):
        """每条 episode 必须自带生成它的 DR 参数 —— 否则数据不可审计。"""
        missing = [i for i, ep in enumerate(self.eps[:50])
                   if not ep["episode_meta"].get("dr_params")]
        self.assertEqual(missing, [], f"前 50 条中有 {len(missing)} 条缺 dr_params")
        d = self.eps[0]["episode_meta"]["dr_params"]
        for key in ("compliance_scale", "damping", "k_hc_scale", "f_preload_n",
                    "m_eff_kg", "vsa_energy_scale", "load_limit_sigma_n"):
            self.assertIn(key, d, f"dr_params 缺 {key}")
        self.assertEqual(d["provenance"], "assumed")

    # ---------- 关键新增: seed 是否真的进了物理层 ----------

    def test_physics_varies_with_seed(self):
        """域随机化必须传导到物理量 —— 这才是测试名字该有的含义。

        v1 版此处只有 `0 < soft_ratio < 1` + 有限性, 对"1000 条全是同一个值"完全
        无感。现按字段统计唯一值数与相对极差, 两者都要过。
        """
        vals = {k: [] for k in DRIVEN_FIELDS}
        for ep in self.eps:
            ph = ep["episode_meta"]["physics"]
            for k in DRIVEN_FIELDS:
                vals[k].append(float(ph[k]))

        n = len(self.eps)
        for k, v in vals.items():
            uniq = len(set(round(x, 6) for x in v))
            self.assertGreater(uniq, max(20, 0.5 * n),
                               f"{k} 唯一值数 {uniq}/{n} —— seed 未传导到物理层 "
                               "(v1 的失败模式: 全部相同)")
            lo, hi = min(v), max(v)
            rel_spread = (hi - lo) / max(abs(lo), 1e-9)
            self.assertGreater(rel_spread, 0.15,
                               f"{k} 相对极差 {rel_spread:.3f} < 0.15, DR 形同未生效 "
                               f"[{lo:.4f}, {hi:.4f}]")

    def test_physics_values_finite_and_positive(self):
        """力/位移必须有限且为正 —— 不再用 `soft_ratio < 1` 把合法物理结果当错误
        (指尖过硬时柔顺分支确实会放大峰值力)。"""
        for ep in self.eps[:200]:
            ph = ep["episode_meta"]["physics"]
            for k in DRIVEN_FIELDS:
                v = float(ph[k])
                self.assertTrue(v == v, f"{k} 是 NaN")
                self.assertNotEqual(abs(v), float("inf"), f"{k} 发散")
                self.assertGreater(v, 0.0, f"{k} 应为正, 得 {v}")

    def test_constant_fields_are_documented_as_such(self):
        """确实恒定的字段必须在 provenance 头里写明原因, 不能被当物理结论。"""
        known = self.header["_provenance"].get("known_constant_fields", {})
        for k, reason in CONSTANT_FIELDS_REASON.items():
            self.assertIn(k, known,
                          f"{k} 恒定但未在 known_constant_fields 说明: {reason}")
        self.assertIn("构造", known["capture_feasible"])

    def test_nominal_control_is_still_the_pre_dr_behaviour(self):
        """spread=0 对照的可复现性: ClimbDR() 名义值必须仍等于引入 DR 前的硬编码值。

        若有人改了名义默认值, "DR 是唯一差异来源"这条对照就悄悄失效了。
        """
        from embodied_ai.simulation.climb_scene import ClimbDR
        nominal = ClimbDR()
        self.assertEqual(nominal.compliance_scale, 1.0)
        self.assertEqual(nominal.k_hc_scale, 1.0)
        self.assertEqual(nominal.vsa_energy_scale, 1.0)
        self.assertEqual(nominal.f_preload_n, 8.0)
        self.assertEqual(nominal.m_eff_kg, 1.2)
        self.assertEqual(nominal.load_limit_mean_n, 60.0)
        self.assertEqual(nominal.load_limit_sigma_n, 0.0)
        self.assertEqual(nominal.wall_total_mass_kg, 20.0)
        self.assertEqual(nominal.gait_step_length_m, 0.15)
        self.assertIsNone(nominal.p_star)          # None → (0.4, 0.9) 原值


if __name__ == "__main__":
    unittest.main(verbosity=2)
