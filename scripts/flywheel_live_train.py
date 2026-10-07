"""EdaLab live 遥测源 → D2→D4 → vla 训练 端到端 (遥测源切换演示)。

链路:
  [1] live 调 EdaLab data-flywheel (Kotlin 引擎真跑)
  [2] stdout → 解析统计 → 统计约束遥测 (provenance=edalab_flywheel_live_statistics)
  [3] BigdataAdapter.ingest (D2) → export_training_jsonl (D4)
  [4] vla_train_full --dataset 真实 loader 训练

运行 (3.14 环境):
    PYTHONPATH=src python3.14 scripts/flywheel_live_train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.adapters.bigdata import BigdataAdapter
from embodied_ai.adapters.edalab import EdalabAdapter
from embodied_ai.adapters.flywheel_live import (live_flywheel_to_records,
                                                parse_flywheel_stdout)


def main() -> int:
    # [1] live 调 data-flywheel
    adapter = EdalabAdapter()
    r = adapter.run_stage("data_flywheel", {}, mode="live")
    assert r["status"] == "LIVE_OK", f"live 调用失败: {r.get('error', r.get('stderr', ''))[:200]}"
    print(f"[1] data-flywheel live: LIVE_OK, stdout {len(r['stdout'])} 字符")

    # [2] 解析统计
    stats = parse_flywheel_stdout(r["stdout"])
    print(f"[2] 解析: 板={stats.get('board_id')} 记录数={stats.get('telemetry_records')} "
          f"成功率={stats.get('success_rate')}% cycles={stats.get('total_cycles')}")
    print(f"    DR 分布: {stats.get('dr_ranges')}")

    # [3] ingest → export
    bd = BigdataAdapter(output_dir=Path("experiments/d4_live"))
    for joint in ("shoulder_L", "elbow_L"):
        recs = live_flywheel_to_records(r["stdout"], joint=joint, seed=hash(joint) % 10000)
        n = bd.ingest_hil_telemetry(recs)
        assert n == len(recs)
    out = bd.export_training_jsonl()
    print(f"[3] D4 导出: {out['path']} ({out['n_records']} 条, 2 关节)")

    # [4] 训练 (子进程调 vla_train_full, 同一 python 解释器)
    import subprocess
    py = sys.executable
    cmd = [py, str(Path(__file__).parent / "vla_train_full.py"),
           "--size", "vla_base", "--iters", "20", "--ckpt-interval", "20",
           "--eval-interval", "10", "--dataset", out["path"],
           "--ckpt-dir", "experiments/vla_train_live"]
    print(f"[4] 训练: {' '.join(cmd[1:6])} ...")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env={**__import__("os").environ,
                                                 "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
    tail = "\n".join(proc.stdout.strip().splitlines()[-6:])
    print(tail)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
