"""K3+ 集成冒烟: 飞轮 DR → 20Hz 控制环 (闭环演示)。

链路 (skeleton-plan K2/K3+ 收尾):
  合成遥测 → BigdataAdapter.flywheel_to_training → dr_overrides
    → make_controller(dr_config=...) → RealTime20HzController.step()
  断言: ① 闭环不抛异常 ② DR 注入改变控制器物理参数 (对比无 DR 基线) ③ 20Hz 时序预算

运行: PYTHONPATH=src python scripts/flywheel_control_smoke.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.adapters.bigdata import BigdataAdapter
from embodied_ai.control.household_robot_system import make_controller


def main() -> int:
    # 1. 飞轮: 合成遥测 → DR
    bd = BigdataAdapter(output_dir=Path("output/dr_ctrl_smoke"))
    for i in range(20):
        bd.ingest_canfd_frame({
            "actual_current_a": 8.0 + (i % 6) * 0.9,
            "temp_c": 38.0 + (i % 7),
        }, joint="shoulder_L", board_serial="SN-CTRL-001")
    bd.ingest_canfd_frame({"actual_current_a": 12.0, "temp_c": 55.0,
                           "fault_events": [{"fault_type": "OVERCURRENT"}]},
                          joint="shoulder_L", board_serial="SN-CTRL-001")
    train_cfg = bd.flywheel_to_training(strategy_v="ctrl_v1")
    dr = train_cfg["dr_overrides"]
    print(f"[1] 飞轮: {train_cfg['replay_records']} 条遥测 → {len(dr)} 项 DR")
    for k in sorted(dr):
        print(f"    {k}: {dr[k].get('min', dr[k].get('mean', '?'))} ~ {dr[k].get('max', '?')}")

    # 2. make_controller + DR 注入
    ctrl_dr = make_controller(dr_config={"overrides": dr})
    ctrl_base = make_controller()  # 无 DR 基线

    # 3. 同一观测跑双控制器 (确定性对比)
    obs = {
        "ee_position": np.array([0.3, 0.0, 1.0]),
        "joint_positions": np.zeros(12),
        "rgb": np.zeros((3, 3, 64, 64)),
        "lang": "抓取杯子",
    }
    t0 = time.perf_counter()
    n_steps = 10
    a_dr = None
    for _ in range(n_steps):
        a_dr = ctrl_dr.step(obs)
        ctrl_base.step(obs)
    loop_ms = (time.perf_counter() - t0) / n_steps * 1000

    print(f"[2] 控制环: {n_steps} 步双控制器跑通, 动作输出 shape={getattr(a_dr, 'shape', type(a_dr).__name__)}")
    print(f"[3] 平均单步 {loop_ms:.2f}ms (20Hz 预算 50ms)")

    # 4. DR 生效验证: MCTS 物理参数被真机分布改写 (dr_override → _mu_override)
    mu_dr = getattr(ctrl_dr.task_planner.mcts, "_mu_override", None)
    mu_base = getattr(ctrl_base.task_planner.mcts, "_mu_override", None)
    dr_effective = mu_dr is not None and mu_base is None
    print(f"[4] DR 注入链: _mu_override DR={mu_dr} / 基线={mu_base} → {'生效 ✅' if dr_effective else '未生效 ❌'}")

    ok = loop_ms < 50.0 and dr_effective
    print("FLYWHEEL→CONTROL SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
