"""断点 2 接线守护测试 — load_ckpt → policy → controller。

分层验证:
  A. 受管环境 (无 torch): PolicyLoader 报错路径 + ReplayPolicy 降级 + 接线路由
  B. 有 torch 环境 (3.14): 真装载 + 真推理 + 控制环执行 (端到端)
反证: 坏 ckpt 必须抛 CheckpointError, 不静默零向量。
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from embodied_ai.control.policy_loader import (CheckpointError, PolicyLoader,
                                                ReplayPolicy,
                                                find_latest_checkpoint)

PY314 = r"C:\Users\bnna7\AppData\Local\Programs\Python\Python314\python.exe"
HAS_TORCH_PY = Path(PY314).exists()


def _resolve_test_ckpt() -> Path | None:
    """解析测试用 checkpoint 路径 (CI 无 ckpt 时返回 None)。"""
    ck = find_latest_checkpoint()
    if ck is None or not ck.exists():
        return None
    return ck


class TestPolicyLoaderErrors(unittest.TestCase):
    """A. 加载错误路径 (受管环境可跑 — 不需要 torch)。"""

    def test_missing_ckpt_raises(self):
        """反证: 不存在的路径必须抛 CheckpointError, 不静默。"""
        with self.assertRaises(CheckpointError):
            PolicyLoader(REPO / "nonexistent.pt").load()

    def test_load_without_load_call_raises(self):
        """反证: 未 load() 就 infer 必须报错 (防静默零)。

        CI 无 checkpoint 时用临时路径构造 loader (验证未 load 即 infer 的报错)。
        """
        ck = _resolve_test_ckpt()
        if ck is None:
            ck = Path(tempfile.mktemp(suffix=".pt"))
        loader = PolicyLoader(ck)
        with self.assertRaises(CheckpointError):
            loader.infer_action(None, None, np.zeros(30, dtype=np.float32))

    def test_find_latest_checkpoint(self):
        """有 checkpoint 时验证可发现；无时 skip (CI 环境无训练产出物)。"""
        ck = _resolve_test_ckpt()
        if ck is None:
            self.skipTest("无 checkpoint 文件 — 跳过可发现性检查")
        self.assertTrue(ck.exists())


class TestReplayPolicy(unittest.TestCase):
    """A. 降级路径 (受管环境)。"""

    def test_replay_is_deterministic_and_honest(self):
        """无 ckpt 时走 None 路径 — 验证降级仍保持确定性与诚实标注。"""
        ck = _resolve_test_ckpt()
        p = ReplayPolicy(ck)  # ck 可能为 None
        obs = {"joint_positions": np.ones(12, dtype=np.float32)}
        a1, a2 = p.generate(obs), p.generate(obs)
        np.testing.assert_array_equal(a1, a2, "回放策略必须确定性")
        self.assertEqual(p.provenance["backend"], "replay")
        self.assertIn("不是模型推理", p.provenance["note"])

    def test_replay_produces_nonzero(self):
        p = ReplayPolicy(_resolve_test_ckpt())
        a = p.generate({"joint_positions": np.ones(12, dtype=np.float32)})
        self.assertTrue(np.any(np.abs(a) > 1e-6), "回放不应输出全零")

    def test_replay_handles_none_checkpoint(self):
        """反证: ckpt=None 时 ReplayPolicy 不崩溃, 如实标注 ckpt_exists=False。"""
        p = ReplayPolicy(None)
        self.assertFalse(p.exists, "ckpt=None 时应标 exists=False")
        self.assertEqual(p.provenance["backend"], "replay")
        a = p.generate({"joint_positions": np.ones(12, dtype=np.float32)})
        self.assertTrue(np.any(a != 0), "即使 ckpt=None 也应产出确定性非零输出")


class TestAttachRouting(unittest.TestCase):
    """A. 接线路由 (受管环境 — 走 Replay 降级)。"""

    def test_attach_falls_back_to_replay_without_torch(self):
        try:
            import torch  # noqa: F401
            self.skipTest("本环境有 torch — 路由会走真推理, 由 B 组测试覆盖")
        except ImportError:
            pass
        from embodied_ai.control.household_robot_system import make_controller
        from embodied_ai.control.policy_loader import attach_policy_to_controller

        ctrl = make_controller()
        r = attach_policy_to_controller(ctrl, _resolve_test_ckpt())
        self.assertEqual(r["backend"], "replay")
        obs = {"ee_position": np.zeros(3), "joint_positions": np.ones(12)}
        a = ctrl.step(obs)
        self.assertTrue(np.any(np.abs(a) > 1e-6), "接线后控制环应产出非零动作")


@unittest.skipUnless(HAS_TORCH_PY, "需要 3.14 torch 环境")
class TestEndToEndWithTorch(unittest.TestCase):
    """B. 端到端 (3.14 子进程): 真装载→真推理→控制环。"""

    def _run(self, code: str):
        env = {**os.environ, "PYTHONPATH": str(SRC)}
        return subprocess.run([PY314, "-c", code], capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              env=env, cwd=str(REPO), timeout=300)

    def test_e2e_ckpt_to_controller(self):
        r = self._run(
            "from embodied_ai.control.policy_loader import attach_policy_to_controller, find_latest_checkpoint;"
            "from embodied_ai.control.household_robot_system import make_controller;"
            "import numpy as np;"
            "ctrl = make_controller();"
            "rep = attach_policy_to_controller(ctrl, find_latest_checkpoint());"
            "assert rep['backend'] == 'vla_ckpt', rep['backend'];"
            "a = ctrl.step({'ee_position': np.zeros(3), 'joint_positions': np.zeros(12)});"
            "assert np.any(np.abs(a) > 1e-6), '动作全零 — 权重未真正进入控制环';"
            "p = ctrl.rl_coordinator.skills['grasp'];"
            "assert p.inference_count >= 1;"
            "print('__E2E__OK', p.last_inference_ms)"
        )
        self.assertIn("__E2E__OK", r.stdout,
                      f"端到端失败:\n{r.stdout[-400:]}\n{r.stderr[-400:]}")

    def test_bad_state_dict_raises(self):
        """反证: config 与权重不匹配必须报错 (torch 环境才能触发 state_dict 校验)。"""
        ck_path = str(find_latest_checkpoint())
        code = (
            "import torch, tempfile\n"
            "from embodied_ai.control.policy_loader import PolicyLoader, CheckpointError\n"
            f"ck = torch.load({ck_path!r}, map_location='cpu', weights_only=False)\n"
            "ck['model'] = {}\n"
            "f = tempfile.mktemp(suffix='.pt')\n"
            "torch.save(ck, f)\n"
            "try:\n"
            "    PolicyLoader(f).load()\n"
            "    raise SystemExit('SHOULD_HAVE_RAISED')\n"
            "except CheckpointError:\n"
            "    print('__BADCK__OK')\n"
        )
        r = self._run(code)
        self.assertIn("__BADCK__OK", r.stdout,
                      f"坏 checkpoint 未报错:\n{r.stdout[-300:]}\n{r.stderr[-300:]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
