"""降级清单执行状态守护 — "明确不做"的东西不得在源码里假装能用。

背景: plan.md 与 plan_v3 的降级清单 (NeRF / VQ / async MCTS) 已裁定**不做**,
docs/11 的处置要求是"删除 NeRFGuidedMCTS 或标注'未实现'"、"async…要么实现要么
诚实标注同步"。两处长期违反: NeRFGuidedMCTS 的 docstring 写"基类补全后自动
可用", 而它的碰撞/可达过滤是恒真占位; AsyncMCTSPlanner 的 docstring 写"在独立
线程/进程中跑 MCTS 不阻塞 20Hz 主循环", 而全类无 threading, 且真挂在
household_robot_system 的 task_planner 上。

本文件不测"功能对不对", 测的是**宣称与实现是否一致** —— 一旦占位被实现掉、或
docstring 被改回虚假宣称, 这里会失败并逼作者同步更新口径。
"""
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.planning import AsyncMCTSPlanner, NeRFGuidedMCTS  # noqa: E402
from embodied_ai.planning.mcts_integration import ClimbingState, ContactAction  # noqa: E402


def _action(x: float) -> ContactAction:
    return ContactAction(contact_position=np.array([x, 0.0, 0.0]))


class TestNeRFIsHonestPlaceholder(unittest.TestCase):
    def test_docstring_summary_declares_not_implemented(self):
        """docstring 首行必须写明未实现。

        (只查首行: 正文会引用旧的错误宣称 "基类补全后自动可用" 来说明改了什么,
         那属于有意保留的历史交代, 不该被当成虚假宣称。)
        """
        summary = inspect.getdoc(NeRFGuidedMCTS).splitlines()[0]
        self.assertIn("未实现", summary,
                      f"NeRFGuidedMCTS 首行必须标注未实现, 得: {summary!r}")
        self.assertNotIn("自动可用", summary,
                         "首行已退回虚假宣称: 'K2: 基类补全后自动可用'")

    def test_filters_are_no_ops(self):
        """占位过滤器恒真: 候选一个都过滤不掉。

        若将来接上真实 IK / SDF 查询, 本用例会失败 —— 那时应同步删掉 docstring
        里的"未实现"标注, 而不是悄悄改行为。
        """
        m = NeRFGuidedMCTS()
        for x in (0.0, 99.0, -3.5):
            a = _action(x)
            self.assertTrue(m._collision_free(a), "_collision_free 应为恒真占位")
            self.assertTrue(m._ik_feasible(a), "_ik_feasible 应为恒真占位")

    def test_get_actions_filters_nothing(self):
        """过滤是空操作 → 候选集与基类生成集大小一致。"""
        m = NeRFGuidedMCTS()
        st = ClimbingState(ee_position=np.zeros(3), joint_positions=np.zeros(12))
        candidates = m._generate_candidate_actions(st)
        self.assertEqual(len(m.get_actions(st)), len(candidates),
                         "NeRF 过滤竟然起作用了 —— 请同步更新'未实现'口径")

    def test_no_nerf_backend_exists(self):
        """src 内不得存在 NeRF **后端** (文档/roadmap 里的提及不算)。

        判据是"有没有可执行的 NeRF 实现或依赖", 而不是"有没有出现过 nerf 这个
        词" —— roadmap_data.py 与 planning/__init__.py 只是把它列为不做/未实现。
        """
        root = Path(__file__).resolve().parents[1] / "src"
        self.assertEqual(list(root.rglob("nerf*.py")), [],
                         "出现 NeRF 模块文件, 降级清单口径需重议")
        signals = []
        for p in root.rglob("*.py"):
            if p.name == "mcts_integration.py":   # 占位实现所在文件 (已标注未实现)
                continue
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if s.startswith(("import ", "from ")) and "nerf" in s.lower():
                    signals.append(f"{p.name}: {s}")
        self.assertEqual(signals, [],
                         f"出现 NeRF 后端依赖, 降级清单口径需重议: {signals}")


class TestAsyncPlannerIsHonestSync(unittest.TestCase):
    def test_docstring_summary_declares_sync(self):
        summary = inspect.getdoc(AsyncMCTSPlanner).splitlines()[0]
        self.assertIn("同步", summary,
                      f"AsyncMCTSPlanner 首行必须诚实标注实为同步, 得: {summary!r}")
        self.assertNotIn("独立线程/进程", summary,
                         "首行已退回虚假宣称: '在独立线程/进程中跑 MCTS'")

    def test_no_threading_primitives_in_class(self):
        """守护"实为同步"这一标注: 可执行代码里不得出现线程/进程/异步原语。

        用 AST 剥掉所有 docstring 再扫描 —— 类 docstring 本身要解释"没有这些
        原语", 直接对源码字符串扫描会自我误报。
        """
        import ast

        tree = ast.parse(inspect.getsource(AsyncMCTSPlanner))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body = node.body[1:]
        cls = tree.body[0]
        body = ast.unparse(ast.Module(body=cls.body, type_ignores=[]))
        for kw in ("threading", "multiprocessing", "asyncio", "concurrent.futures",
                   "Thread(", "Process("):
            self.assertNotIn(kw, body,
                             f"AsyncMCTSPlanner 代码出现 {kw!r} —— 若真实现了异步, "
                             "请同步删除 docstring 里的'实为同步'标注并更新本用例")

    def test_plan_blocks_the_caller(self):
        """plan() 阻塞调用方 —— 上游 20Hz 环若假设它立即返回就会踩坑。"""
        import time

        class _SlowMCTS:
            def search(self, state, goal_fn=None):
                time.sleep(0.05)
                return []

        p = AsyncMCTSPlanner()
        p.mcts = _SlowMCTS()
        st = ClimbingState(ee_position=np.zeros(3), joint_positions=np.zeros(12))
        t0 = time.perf_counter()
        p.plan(st)
        self.assertGreater(time.perf_counter() - t0, 0.04,
                           "plan() 不再阻塞调用方 —— 异步可能已被实现, 请更新口径")

    def test_control_stack_still_uses_it(self):
        """该同步规划器确实挂在生产 20Hz 控制栈上 (命名误导有实际影响面)。"""
        from embodied_ai.control.household_robot_system import RealTime20HzController

        src = inspect.getsource(RealTime20HzController.__init__)
        self.assertIn("AsyncMCTSPlanner", src,
                      "若已从控制栈摘除, '命名误导'的风险已消失, 可精简本守护")


class TestDeadCodeStaysDead(unittest.TestCase):
    """slam.py 的 Landmark dataclass 曾全仓零实例化、只被自己的测试用 ——
    "为测试而存在的类"是死代码, 已删除。本用例防止它被重新引入。"""

    def test_landmark_class_removed(self):
        from embodied_ai.perception import slam

        self.assertFalse(hasattr(slam, "Landmark"),
                         "slam.Landmark 是死代码 (src 内零引用), 不应回归")


if __name__ == "__main__":
    unittest.main(verbosity=2)
