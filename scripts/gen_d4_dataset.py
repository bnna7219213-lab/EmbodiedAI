"""生成 D4 训练数据集 — 从合成遥测源 (演示端到端; 真机遥测替换 ingest 即同管线)。

数据扩充 (任务 ③): 120 → >=1000 episodes。
  每条 episode 是一个关节的完整运动轨迹 (20-60 步), 含 episode_id 与 scenario 标签。
  全部 episodes: 1050 条 (150/scenario x 7 类), 总记录数 ~30k+。

  场景覆盖 climbing / household 两大领域:
    climb_grip        静态握持     (高电流维持 · 位置稳定)
    climb_reach       伸臂够点     (位置斜坡 · 中等电流)
    climb_dyno        动态跳脱     (阶跃冲击 · 大电流尖峰)
    climb_cross       交叉横移     (正弦轨迹 · 电流跟随)
    tendon_pump       VSA 蓄能     (电流振荡 · 温升累积)
    household_grasp   轻抓取       (缓斜坡 · 低电流)
    slip_recovery     滑脱恢复     (位置漂移 + 电流尖峰 · 真实失败模式)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_ai.adapters.bigdata import BigdataAdapter


# 场景定义: (名称, 适用关节列表, 该场景下 episode 数, 力/位/温特征)
SCENARIOS = [
    dict(name="climb_grip",     joints=["shoulder_L", "shoulder_R", "elbow_L", "elbow_R"],   n=150,
         current_range=(15.0, 25.0), pos_jitter=2.0,  temp_rise=0.05, label="hold"),
    dict(name="climb_reach",    joints=["shoulder_L", "shoulder_R", "elbow_L", "elbow_R"],   n=150,
         current_range=(8.0, 18.0),  pos_jitter=5.0,  temp_rise=0.08, label="reach"),
    dict(name="climb_dyno",     joints=["shoulder_L", "shoulder_R"],                          n=150,
         current_range=(20.0, 40.0),pos_jitter=15.0, temp_rise=0.20, label="dyno"),
    dict(name="climb_cross",    joints=["elbow_L", "elbow_R", "shoulder_L", "shoulder_R"],   n=150,
         current_range=(6.0, 14.0),  pos_jitter=8.0,  temp_rise=0.06, label="cross"),
    dict(name="tendon_pump",    joints=["shoulder_L", "shoulder_R"],                          n=150,
         current_range=(10.0, 30.0),pos_jitter=4.0,  temp_rise=0.15, label="vsa_pump"),
    dict(name="household_grasp",joints=["shoulder_L", "elbow_L", "wrist_L"],                  n=150,
         current_range=(2.0, 6.0),   pos_jitter=3.0,  temp_rise=0.03, label="grasp"),
    dict(name="slip_recovery",  joints=["shoulder_L", "shoulder_R", "elbow_L", "elbow_R"],   n=150,
         current_range=(12.0, 35.0),pos_jitter=12.0, temp_rise=0.10, label="slip"),
]


def generate_episode(bd: BigdataAdapter, scenario: dict, ep_idx: int, rng: np.random.Generator) -> int:
    """为单个 episode 生成遥测记录。返回生成的记录条数。"""
    joint = scenario["joints"][ep_idx % len(scenario["joints"])]
    ep_id = f"ep_{scenario['label']}_{ep_idx:04d}"

    n_steps = int(rng.integers(20, 61))           # 20-60 步 / episode
    dt = float(rng.uniform(0.02, 0.08))            # 20-80ms 控制周期

    c_lo, c_hi = scenario["current_range"]
    pos = float(rng.uniform(-80, 80))
    temp = float(rng.uniform(35.0, 42.0))
    base_current = float(rng.uniform(c_lo, c_hi))

    records = 0
    slip_event = (scenario["name"] == "slip_recovery")
    slip_step = int(rng.integers(int(n_steps * 0.3), int(n_steps * 0.7))) if slip_event else -1
    dyno_step = int(rng.integers(3, 8)) if scenario["name"] == "climb_dyno" else -1
    freq = float(rng.uniform(0.5, 2.0))

    for t in range(n_steps):
        ts = float(t) * dt

        # 位置: 场景模式 (斜坡/正弦/抖动/冲击)
        if scenario["name"] == "climb_reach":
            pos += 2.0 * dt + rng.normal(0, scenario["pos_jitter"] * 0.1)
        elif scenario["name"] == "climb_cross":
            pos = 60.0 * np.sin(2 * np.pi * freq * ts) + rng.normal(0, 2.0)
        elif scenario["name"] == "climb_grip":
            pos += rng.normal(0, scenario["pos_jitter"] * 0.05)
        else:
            pos += rng.normal(0, scenario["pos_jitter"] * 0.2)

        # 电流: 跟随运动 + 温升 + 特殊事件
        current = base_current + rng.normal(0, (c_hi - c_lo) * 0.15)
        if t == dyno_step:
            current += rng.uniform(15.0, 30.0)        # dyno 电流尖峰
        if t == slip_step:
            current += rng.uniform(10.0, 20.0)        # 滑脱冲击
        if scenario["name"] == "tendon_pump":
            current += 8.0 * np.sin(2 * np.pi * 1.5 * ts)  # VSA 蓄能振荡

        # 温度: 累积上升 + 噪声
        temp += scenario["temp_rise"] * dt * (1.0 + current / 30.0) + rng.normal(0, 0.05)

        # 故障事件 (仅 slip 场景 + 10% 其他)
        faults = []
        if t == slip_step:
            faults.append({"type": "slip_detected", "severity": "medium", "ts": ts})
        elif rng.random() < 0.02:
            faults.append({"type": "soft_limit", "joint": joint, "ts": ts})

        bd.ingest_canfd_frame({
            "timestamp_s": ts,
            "actual_current_a": float(np.clip(current, 0.0, 60.0)),
            "temp_c": float(temp),
            "actual_pos_deg": float(np.clip(pos, -90, 90)),
            "fault_events": faults,
            "episode_id": ep_id,
            "scenario": scenario["name"],
            "n_steps": n_steps,
        }, joint=joint, board_serial="SN-D4-001", source="replay_gen")
        records += 1
    return records


def main() -> int:
    bd = BigdataAdapter(output_dir=Path("experiments/d4_dataset"))
    rng = np.random.default_rng(42)

    total_episodes = sum(s["n"] for s in SCENARIOS)
    total_records = 0
    for scenario in SCENARIOS:
        for i in range(scenario["n"]):
            total_records += generate_episode(bd, scenario, i, rng)

    out = bd.export_training_jsonl()
    print(f"D4 数据集 (扩展版): {out['path']}")
    print(f"Episode 数: {total_episodes} (目标 >=1000)")
    print(f"记录数: {total_records} (总遥测帧)")
    print(f"场景: {', '.join(s['name'] for s in SCENARIOS[:4])} / ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
