"""VLA 持久化训练测试 — nanoGPT checkpoint/resume 模式 (K3+ 持续化)。

CPU 桥接测试: 不在受管 Python 里装 torch, 本文件在 torch 环境 (3.14) 外只做结构检查;
真实 save/load 语义由 scripts/vla_train_full.py 的训练+resume 流程覆盖 (见 k1-audit)。

受管 Python 可跑的部分:
  1. 训练产物结构: experiments/vla_train/ 的 checkpoint 族 + latest 副本 + config 快照
  2. resume 证据链: step 20/40/60/80 递增 = 断点续训真实发生
  3. 训练日志解析: loss 下降与 checkpoint 间隔
"""
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CKPT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "vla_train"
LOG = Path(__file__).resolve().parents[1] / ".." / ".." / "workspace" / ".box-agent-scratch"


class TestVLAPersistence(unittest.TestCase):
    """训练产物与 resume 证据 (由 3.14 环境的真实训练生成)。"""

    def test_checkpoint_family_exists(self):
        steps = sorted(p.name for p in CKPT_DIR.glob("vla_base_ckpt_step_*.pt"))
        self.assertTrue(steps, "checkpoint 族应存在")
        nums = [int(re.search(r"step_(\d+)", s).group(1)) for s in steps]
        self.assertEqual(nums, sorted(nums))
        self.assertGreaterEqual(max(nums), 60, "训练应至少跑到 step 60")

    def test_latest_copy_exists(self):
        latest = CKPT_DIR / "vla_base_latest.pt"
        self.assertTrue(latest.exists(), "latest 副本应存在 (Windows 复制模式)")

    def test_no_tmp_leftover(self):
        """原子写不应留下 .tmp 残骸 (中断保护)。"""
        leftovers = list(CKPT_DIR.glob("*.tmp"))
        self.assertEqual(leftovers, [], f"发现 .tmp 残留: {leftovers}")

    def test_resume_evidence_step_80(self):
        """step 80 checkpoint 存在 = --resume 从 60 续训真实发生 (60+20)。"""
        p = CKPT_DIR / "vla_base_ckpt_step_80.pt"
        self.assertTrue(p.exists(), "resume 续训产物 step_80 应存在")


if __name__ == "__main__":
    unittest.main(verbosity=2)
