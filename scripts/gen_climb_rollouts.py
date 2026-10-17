"""③ 数据扩充: 用攀岩闭环当数据引擎 — 批量生成 D4 训练 episodes。

**定位**（评分 50 分域的提分点: 数据仅 120 条半合成 → ≥1000 episodes）

数据引擎 = `climb_scene.run_climb_cycle`（六状态全链闭环, Phase A 收官产物）。
每 episode 的物理量都是**计算链产出**（MCTS 规划 / 柔顺接触 / VSA 蓄能 / 斜壁多接触
力分配 / 解析 LIPM 步态）——不是随机数剧本, 也不是文献值的拷贝。

**provenance = simulated, 不是 measured**
    这些物理量由物理引擎算出, 但全仓硬件实物为零 → 没有任何一条来自实测; 且其输入
    参数 (ClimbDR 的 DR 区间) 是 **assumed**。v1 版把这批字段标成 `real_fields`,
    与 D4 那次同类错误, 已更正为 `simulated_fields` + `dr_parameter_provenance`。

**seed 必须同时驱动两处**（v2 的核心修复）
    v1 版 seed 只流进 `generate_wall` 的场景采样, 而指尖柔度/阻尼、Hertz 耦合刚度、
    预压、等效质量、VSA 蓄能、弹道几何、体重/重心、步态步长全是硬编码常量 ——
    于是"1000 episodes 的 11 个物理字段唯一值数全为 1", 所谓 seed 驱动的域随机化
    根本没进物理层。现在 seed 同时喂 `ClimbDR.from_seed(seed)`。
    （`--spread 0` 可退回名义配置, 用于对照: 那时物理量应当重新变成恒定。）

**失稳配置显式跳过并计数**: 全维联合放开时约 11% 的 seed 会把斜壁多接触推出可行域
（体重×手摩擦×侧向载荷×重心高的联合交互, 任一单维放开都是 0 失败）。跳过的比例
被打印出来 —— 产量与覆盖率不能假装总是 1000。

**D4 JSONL 格式**（对齐 validate_bc_dataset 的 BC 契约）:
  首行 provenance: {"_schema": "d4_climb_v2", "_provenance": {...}}
  每行一个 episode: {"observations": [...], "actions": [...],
                     "instruction": str, "episode_meta": {..., "dr_params": {...}}}
  observations 的坐标是**真 JSON 数组**（v1 用 default=str 把 numpy 数组存成了
  repr 字符串 `'[0.  0.  0.1 ...]'`, 下游必须写宽容解析器才不炸）。

用法:
  python scripts/gen_climb_rollouts.py                 # 默认 1000 episodes
  python scripts/gen_climb_rollouts.py --n 50 --quick   # 快速验证
  python scripts/gen_climb_rollouts.py --spread 0        # 名义配置对照组
  python scripts/gen_climb_rollouts.py --out custom.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from embodied_ai.simulation.climb_scene import ClimbDR, run_climb_cycle  # noqa: E402

# 相位 → 动作模板（攀岩闭环的物理语义: 每个相位有特征性的关节动作方向）
_PHASE_ACTIONS = {
    "hang": np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0,   # 悬挂: 轻微腕部维持
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "reach": np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0,  # 伸展: 肩部前伸
                       0.0, 0.0, 0.1, 0.0, 0.0, 0.0], dtype=np.float32),
    "swing": np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.0,  # 摆荡: 躯干摆动
                       0.0, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "dyno": np.array([0.8, 0.3, 0.5, 0.0, 0.0, 0.0,   # 弹射: 全力伸展
                      0.0, 0.0, 0.4, 0.0, 0.0, 0.0], dtype=np.float32),
    "capture": np.array([-0.4, -0.1, -0.2, 0.0, 0.0, 0.0,  # 捕获: 减速回收
                         0.0, 0.0, -0.1, 0.0, 0.0, 0.0], dtype=np.float32),
    "climb_on": np.array([0.5, 0.2, 0.3, 0.0, 0.0, 0.0,  # 续攀: 稳定上升
                          0.0, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=np.float32),
}


def _phase_noise(seed: int, phase: str, step: int, n_dims: int = 12) -> np.ndarray:
    """逐维、随 seed 变化的微扰。

    原实现是 `0.05 * np.sin(step*0.5 + hash(phase)%10)` —— 一个**标量**广播到 12 维。
    模板里为 0 的那 7 个维因此拿到完全相同的值 (实测 dim 3/4/5/6/9/10/11 的 std
    全等于 0.03278), 12 维动作空间实际只有 6 种不同取值, 前 4 维占 95.6% 方差。
    现在每个维独立采样 (确定于 seed/phase/step, 可复现)。
    """
    rng = np.random.default_rng(
        (seed * 1_000_003 + _phase_key(phase) * 31 + step) % (2 ** 32))
    return rng.normal(0.0, 0.05, size=n_dims)


def _phase_key(phase: str) -> int:
    """相位的**稳定**散列。

    绝不能用内置 hash(): 字符串 hash 受 PYTHONHASHSEED 随机化, 同一 seed 在不同
    进程会生成不同数据集 —— 那会让"数据集可复现"这句话变成假的 (且重跑覆盖时
    静默改变全部样本)。
    """
    return int(hashlib.sha256(phase.encode("utf-8")).hexdigest()[:8], 16) % 9973


def _obs_from_cycle(r, phase_idx: int, step: int, seed: int) -> Dict[str, Any]:
    """从 climb_cycle 结果构造第 phase_idx 相位、第 step 步的观测。

    joint_positions 现在是**真数组** (list), 不再是 numpy repr 字符串 ——
    原实现经 `json.dumps(..., default=str)` 落盘成 `'[0.  0.  0.1 ...]'`,
    下游必须写宽容解析器才不炸, 而 validate_bc_dataset 照样判 PASS。
    观测的 12 维取自本步动作模板 + 逐维微扰 + 相位特征幅值 (与动作解耦:
    状态是"当前构型", 动作是"下一步指令", 二者不该逐维相等)。
    """
    phase = r.phases_visited[phase_idx % len(r.phases_visited)]
    base = _PHASE_ACTIONS[phase]
    amp = 1.0 + 0.3 * float(np.sin(step * 0.3 + phase_idx))
    joint = np.clip(base * amp + _phase_noise(seed, phase, step) * 0.4, -1.0, 1.0)
    return {
        "joint_positions": joint.astype(np.float32).tolist(),
        "ee_position": [float(0.3), 0.0, float(1.0 + 0.1 * phase_idx)],
        "phase": phase,
        "step": step,
    }


def _action_from_phase(phase: str, step: int, seed: int) -> np.ndarray:
    """动作 = 相位模板 + 逐维微扰（攀岩语义的动作监督信号）。"""
    base = _PHASE_ACTIONS[phase]
    return np.clip(base + _phase_noise(seed, phase, step), -1, 1).astype(np.float32)


def generate_one_episode(seed: int, n_steps_per_phase: int = 3,
                         spread: float = 1.0) -> Optional[Dict[str, Any]]:
    """跑一次攀岩闭环 → 一个完整 episode（observations/actions 等长）。

    seed 现在**同时**驱动两处:
      1. `generate_wall` 的场景采样 (μ / 坐标 / hold 载荷极限) —— 原先只有这个;
      2. `ClimbDR.from_seed(seed)` 的物理参数组 (指尖柔度与阻尼 / Hertz 耦合刚度 /
         预压 / 等效质量 / VSA 蓄能 / 弹道几何 / 体重与重心 / 步态步长)。
    缺了 2 时, 11 个物理字段在 1000 条 episode 里全是同一个值 —— 所谓"seed 驱动的
    域随机化"根本没进物理层 (见 climb_scene.ClimbDR 的 docstring)。

    返回 None 表示该 seed 的 DR 抽样落在失稳配置 (斜壁多接触不可行), 由调用方
    **显式计数**后跳过 —— 不静默丢样本, 否则"1000 episodes"的产量与覆盖率会失真。
    """
    dr = ClimbDR.from_seed(seed, spread=spread)
    try:
        r = run_climb_cycle(seed=seed, dr=dr)
    except AssertionError:
        return None

    observations: List[Dict] = []
    actions: List[np.ndarray] = []
    global_step = 0
    for pi, phase in enumerate(r.phases_visited):
        for s in range(n_steps_per_phase):
            observations.append(_obs_from_cycle(r, pi, s, seed))
            actions.append(_action_from_phase(phase, global_step, seed))
            global_step += 1

    return {
        "observations": observations,
        "actions": [a.tolist() for a in actions],   # 纯 list — validate 契约要标量序列
        "instruction": f"攀岩竞技: 从 seed {seed} 的墙面开始, "
                       f"完成 {r.phases_visited[0]} 到 {r.phases_visited[-1]} 的攀爬周期",
        "episode_meta": {
            "seed": seed,
            "phases": r.phases_visited,
            "n_steps": len(actions),
            # 本 episode 实际使用的物理参数 (可审计"这条数据由哪组 DR 参数生成")
            "dr_params": r.dr_params,
            # 攀岩闭环算出的物理量。**provenance=simulated**: 由 climb_scene 的
            # 真实计算链 (MCTS/柔顺接触/VSA/斜壁力分配/LIPM 步态) 产出, 但**不是
            # 实测** —— 输入参数是 assumed 的 DR 区间。此前标 real_fields 与本仓
            # "硬件实物为零"的事实冲突, 已更正。
            "physics": {
                "f_soft_peak_n": r.f_soft_peak_n,
                "f_rigid_peak_n": r.f_rigid_peak_n,
                "soft_ratio": r.soft_ratio,
                "v_release": list(r.v_release),
                "dyno_tf_s": r.dyno_tf_s,
                "capture_feasible": r.capture_feasible,
                "com_advance_m": r.com_advance_m,
                "wall_frames": r.wall_frames,
                "wall_unstable": r.wall_unstable,
                "peak_finger_n": r.peak_finger_n,
                "v_capture_max_mps": r.v_capture_max_mps,
                "full_speed_infeasible": r.full_speed_infeasible,
            },
        },
    }


def generate_dataset(n_episodes: int = 1000, seed_start: int = 0,
                     n_steps_per_phase: int = 3,
                     progress_every: int = 100,
                     spread: float = 1.0) -> List[Dict]:
    """批量生成 episodes, 直到攒够 n_episodes 或 seed 预算耗尽。

    产出率与跳过数**如实打印** —— 失稳配置被跳过的比例是数据集覆盖率的一部分,
    不能当成"总是 1000 条"。
    """
    out: List[Dict] = []
    t0 = time.time()
    skipped = tried = 0
    while len(out) < n_episodes and tried < n_episodes * 4:
        seed = seed_start + tried
        tried += 1
        ep = generate_one_episode(seed=seed, n_steps_per_phase=n_steps_per_phase,
                                  spread=spread)
        if ep is None:
            skipped += 1
            continue
        out.append(ep)
        if progress_every and len(out) % progress_every == 0:
            dt = time.time() - t0
            eta = dt / len(out) * (n_episodes - len(out))
            print(f"  [{len(out)}/{n_episodes}] {dt:.0f}s elapsed, ETA {eta:.0f}s "
                  f"(跳过失稳 {skipped})")
    print(f"  产量: {len(out)}/{n_episodes} episodes, 尝试 {tried} 个 seed, "
          f"跳过失稳配置 {skipped} ({100.0 * skipped / max(tried, 1):.1f}%)")
    return out


def export_d4_jsonl(dataset: List[Dict], out_path: Path,
                    spread: float = 1.0) -> Dict[str, Any]:
    """导出 D4 JSONL（首行 provenance + 每 episode 一行）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    physics_keys = ["f_soft_peak_n", "f_rigid_peak_n", "soft_ratio", "v_release",
                    "dyno_tf_s", "com_advance_m", "peak_finger_n",
                    "v_capture_max_mps", "wall_max_hand_n"]
    provenance = {
        "_schema": "d4_climb_v2",
        "_provenance": {
            "source": "climb_scene.run_climb_cycle (六状态攀岩闭环) + ClimbDR.from_seed",
            "n_episodes": len(dataset),
            "n_steps_total": sum(len(e["actions"]) for e in dataset),
            # ⚠ 用 simulated 而不是 real: 这些量由 climb_scene 的**计算链**产出
            #   (MCTS / Hunt-Crossley 柔顺接触 / VSA 蓄能 / 斜壁多接触力分配 /
            #    解析 LIPM 步态), 但**没有任何一条来自实测** —— 全仓硬件实物为零。
            #   v1 数据集把这批字段标成 real_fields, 与 D4 那次同类错误。
            "simulated_fields": physics_keys,
            "simulated_provenance": "simulated (物理引擎计算, 非实测)",
            "dr_parameter_provenance": "assumed (DR 区间为本仓设定, 非实测散布)",
            "dr_spread": spread,
            "synthetic_fields": ["rgb (absent — 数据集无图像)",
                                 "lang (instruction 模板文本, 非语言模型)"],
            "engine": "Phase A 攀岩闭环 (climb_scene + wall_contact + dexterous_hand + VSA)",
            "dr_dimensions": [
                "hold μ / hold 载荷极限", "指尖柔度(结构/剪切/弯曲) / 阻尼",
                "Hertz 耦合刚度 k_hc", "指尖预压 f_preload / 末端等效质量 m_eff",
                "VSA 蓄能量缩放", "弹道 p_star / r_max / margin",
                "target_hold 索引", "斜壁体重 / 重心高 / 手摩擦 / 侧向载荷",
                "步态步长 / 步态重心高"],
            "known_constant_fields": {
                "capture_feasible": "恒 True —— 捕获用二分把冲击速度解到 "
                                    "f_soft == hold.max_load_n, 故可行性由构造保证, "
                                    "**不是**物理结论, 不可当验收信号使用",
                "full_speed_infeasible": "恒 True —— 名义 DR 区间内满速直弹始终超限",
                "wall_frames": "恒 7 —— 序列帧数由脚本编排决定, 非物理量",
                "dyno_tf_s": "仅约 9 个不同取值 —— 弹道积分按 dt 量化",
            },
            "note": "物理量由攀岩闭环计算链产出 (simulated), 输入参数为 assumed DR 区间; "
                    "rgb 缺失、lang 为模板文本 —— 不冒充真实示教, 更不冒充实测。",
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(provenance, ensure_ascii=False) + "\n")
        for ep in dataset:
            # 不再用 default=str: 观测坐标已是真数组, 若还有 numpy 残留应当**报错**
            # 而不是被 str() 悄悄转成字符串 (v1 就是这么把坐标存成 numpy repr 的)。
            f.write(json.dumps(ep, ensure_ascii=False, allow_nan=False) + "\n")
    return {"path": str(out_path), "n_episodes": len(dataset),
            "n_steps": provenance["_provenance"]["n_steps_total"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="攀岩闭环批量数据生成 (③ 数据扩充)")
    ap.add_argument("--n", type=int, default=1000, help="episode 数")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--steps-per-phase", type=int, default=3)
    ap.add_argument("--out", default="artifacts/d4_climb/climb_episodes.jsonl")
    ap.add_argument("--spread", type=float, default=1.0,
                    help="DR 幅度: 0.0=名义配置(物理量恒定, 用于对照), 1.0=假定区间")
    ap.add_argument("--quick", action="store_true", help="快速模式 (n=50, 少步数)")
    args = ap.parse_args()

    if args.quick:
        args.n = min(args.n, 50)
        args.steps_per_phase = 2

    print(f"=== 攀岩闭环数据引擎: {args.n} episodes (DR spread={args.spread}) ===")
    data = generate_dataset(args.n, seed_start=args.seed_start,
                            n_steps_per_phase=args.steps_per_phase,
                            spread=args.spread)
    out = export_d4_jsonl(data, Path(args.out), spread=args.spread)
    print(f"产出: {out['path']}")
    print(f"  episodes: {out['n_episodes']}")
    print(f"  总步数: {out['n_steps']}")

    # 多样性自证: DR 生效的话, 关键物理量的唯一值数应接近 episode 数。
    # 这一条是本轮修复的**验收判据** —— v1 数据集它全部等于 1, 说明 seed 没进物理层。
    for key in ("f_soft_peak_n", "soft_ratio", "com_advance_m", "peak_finger_n"):
        vals = {round(e["episode_meta"]["physics"][key], 5) for e in data}
        print(f"  {key:16s} 唯一值数 = {len(vals):5d} / {len(data)}")

    # validate 校验
    from embodied_ai.adapters.nanogpt import NanoGPTAdapter
    # validate_bc_dataset 是静态方法 — 直接用类调用
    result = NanoGPTAdapter.validate_bc_dataset(data)
    print(f"  validate_bc_dataset: {'PASS' if result['ok'] else 'FAIL: ' + str(result['issues'][:3])}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
