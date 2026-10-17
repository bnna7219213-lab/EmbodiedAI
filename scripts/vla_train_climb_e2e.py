"""1000 episodes 攀岩闭环 × 触觉闭环 → VLA 真实训练 + 端到端数据质量核验。

为什么要单独写这个脚本 (而不是复用 vla_train_full.py --dataset):

    现有 `--dataset` 路径消费的是 **扁平遥测 jsonl** (build_pairs: 每条记录
    motor_current_a/temperature_c/actual_pos_deg → state[3], 下一位置 → action[1]),
    而 d4_climb 是 **episode 结构** (observations/actions/instruction/episode_meta)。
    两者 schema 不兼容 —— 直接把 climb 数据集喂给旧路径会在 load_telemetry_jsonl
    里读出 0 条可用记录。本脚本走 episode 侧接入, 并把"数据质量"当作**被测对象**
    而不是前提假设。

与旧路径的三处实质差别 (都是旧路径的数据质量问题, 见 REPORT 段输出):

1. **视觉不再用零张量冒充输入**。ReplayDataset 每步喂 (3,3,224,224) 全零 frames,
   VLAModel 仍走 vision_encoder → 训练在"有视觉输入"的名义下实际拿不到任何视觉信息。
   这里改为 `frames=None` (VLAModel 契约本身允许), 名义与实参一致。
2. **语言通道用真实 instruction 派生 token**, 不是 8 个 0。旧路径 lang 恒零,
   语言 embedding 只收到同一个 id → 语言分支等于一个常数偏置, 却在 README 里算
   "VLA 双路径已打通"。
3. **切分按 episode 而非全局 80/20**。build_pairs 的全局时序切分会把同一 episode 的
   相邻帧分置 train/val 两侧; 监督信号是"下一帧位置", 于是 val 几乎是 train 的
   邻域 → 泄漏, val loss 失去意义。

provenance (逐通道, 输出里也打印):
    state 关节/末端  = real_computed  (climb_scene 闭环算出, 非 rng)
    state 触觉 μ̂/σ/fn = synthetic      (synth_tactile_obs 确定性 stick-slip + EKF +
                                   机会约束握力; μ_true 由 episode 物理量按**未标定**
                                   映射导出 → assumed)
    language         = real_text      (instruction 模板文本, 非语言模型生成)
    action           = real_computed  (相位模板 + 微扰)
    vision           = absent         (数据集中不存在, 显式 None 而非零张量)

用法:
    python scripts/vla_train_climb_e2e.py --iters 300
    python scripts/vla_train_climb_e2e.py --size vla_tiny --iters 50 --report out.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA = REPO / "artifacts" / "d4_climb" / "climb_episodes_full.jsonl"


# ------------------------------------------------------------------ #
# 1. 载入 + 数据质量核验 (训练前, 作为被测对象)
# ------------------------------------------------------------------ #

def _vec(s) -> np.ndarray:
    """observations 里的坐标存成了 numpy repr **字符串** → 解析成 float 向量。"""
    if isinstance(s, (list, tuple)):
        return np.asarray(s, dtype=np.float64)
    txt = str(s).strip().strip("[]")
    try:
        v = ast.literal_eval(txt if "," in txt else txt.replace("\n", " ").split())
        return np.asarray(v, dtype=np.float64).ravel()
    except Exception:  # noqa: BLE001 — 解析失败必须显式可见, 不静默补零
        return np.fromstring(txt, sep=" ")


def load_episodes(path: Path = DATA) -> List[dict]:
    eps = []
    with open(path, encoding="utf-8") as f:
        f.readline()                     # provenance header
        for line in f:
            line = line.strip()
            if line:
                eps.append(json.loads(line))
    return eps


def audit(episodes: List[dict]) -> dict:
    """把数据集当作被测对象: 可解析性 / 维度一致 / 退化维度 / 多样性 / 可用样本数。"""
    n_bad_vec, n_bad_len = 0, 0
    step_counts, instrs, act_stack = [], [], []
    dim_std_acc, dim_mean_acc = [], []
    phase_hist: Dict[str, int] = {}
    phys_unique: Dict[str, int] = {}

    for ep in episodes:
        obs, acts = ep["observations"], ep["actions"]
        step_counts.append(len(obs))
        if len(obs) != len(acts):
            n_bad_len += 1
        instrs.append(str(ep.get("instruction", "")))
        for o, a in zip(obs, acts):
            j = _vec(o.get("joint_positions"))
            e = _vec(o.get("ee_position"))
            if j.size != 12 or not np.all(np.isfinite(j)) or e.size != 3:
                n_bad_vec += 1
            phase_hist[str(o.get("phase", "?"))] = phase_hist.get(str(o.get("phase", "?")), 0) + 1
            act_stack.append(np.asarray(a, dtype=np.float64))

        ph = ep.get("episode_meta", {}).get("physics", {})
        for k, v in ph.items():
            if isinstance(v, (int, float)):
                phys_unique.setdefault(k, set()).add(round(float(v), 6))

    A = np.stack(act_stack) if act_stack else np.zeros((1, 1))
    per_dim_std = A.std(axis=0)
    per_dim_mean = A.mean(axis=0)

    var_by_dim = A.var(axis=0)
    order = np.argsort(var_by_dim)[::-1]
    top4_share = float(var_by_dim[order[:4]].sum() / max(var_by_dim.sum(), 1e-12))
    # 防"探针把自己做成常量": 语言 token 必须真的随 instruction 变化
    lang_rows = {tuple(lang_tokens(s).tolist()) for s in instrs}

    phys_summary = {k: len(v) for k, v in phys_unique.items()}
    return {
        "n_episodes": len(episodes),
        "n_steps_total": int(sum(step_counts)),
        "steps_min": int(min(step_counts)), "steps_max": int(max(step_counts)),
        "episodes_with_obs_act_mismatch": n_bad_len,
        "bad_vectors": n_bad_vec,
        "action_dim": int(A.shape[1]),
        "action_constant_dims": [int(i) for i in np.where(per_dim_std < 1e-9)[0]],
        "action_per_dim_std": [round(float(x), 5) for x in per_dim_std],
        "action_per_dim_mean": [round(float(x), 5) for x in per_dim_mean],
        "action_total_variance": float(A.var(axis=0).sum()),
        "action_variance_share_top4_dims": top4_share,
        "action_distinct_std_values": int(len(np.unique(np.round(per_dim_std, 6)))),
        "unique_instructions": len(set(instrs)),
        "unique_lang_token_rows": len(lang_rows),
        "lang_channel_degenerate": len(lang_rows) <= 1,
        "phases": phase_hist,
        "physics_field_unique_value_counts": phys_summary,
        "physics_degenerate_fields": sorted(k for k, n in phys_summary.items() if n <= 1),
    }


# ------------------------------------------------------------------ #
# 2. 触觉闭环: 每个 episode 跑一遍 EKF + 机会约束握力, 取每步调度量
# ------------------------------------------------------------------ #

class _PassthroughController:
    """占位控制器 — 我们要的是触觉环产出的 μ̂/σ_μ/f_n, 不是它的动作。"""
    def step(self, obs):
        return np.zeros(12)


def mu_from_physics(ep: dict) -> float:
    """由 episode 物理量导出 μ_true。

    ⚠ 这是一条 **assumed 映射, 无任何标定依据**: 峰值指触力 / 柔顺峰值力 的比值
    与摩擦系数并非因果关系。这里用它只为让触觉输入随 episode 变化、从而能检验
    "上游信号变化是否传导到训练" —— 不可用于任何物理结论。
    """
    ph = ep.get("episode_meta", {}).get("physics", {})
    fs = float(ph.get("f_soft_peak_n", 60.0)) or 60.0
    fp = float(ph.get("peak_finger_n", 30.0))
    return float(np.clip(fp / fs, 0.30, 0.95))


def tactile_features(ep: dict, n_steps: int) -> np.ndarray:
    """跑 TactileGripLoop, 返回 [n_steps, 3] = (μ̂, σ_μ, f_n_setpoint/40)。

    f_t 序列来自 synth_tactile_obs (确定性 stick→slip 物理模型, provenance=synthetic),
    EKF 与机会约束握力是**生产实现** (multirate_ekf / chance_constrained_grip)。
    """
    from embodied_ai.control.tactile_grip_loop import TactileGripLoop, synth_tactile_obs

    mu_true = mu_from_physics(ep)
    obs_stream = synth_tactile_obs(mu_true=mu_true, f_n=40.0,
                                   n_steps=n_steps, seed=int(ep["episode_meta"]["seed"]))
    ctrl = _PassthroughController()
    loop = TactileGripLoop(ctrl, bus=None, f_n_nominal=40.0)
    feats = np.zeros((n_steps, 3), dtype=np.float64)
    for i, o in enumerate(obs_stream):
        ctrl.step(o)
        feats[i] = (loop.last_mu_hat, loop.last_sigma_mu,
                    loop.last_f_n_setpoint / 40.0)
    return feats


# ------------------------------------------------------------------ #
# 3. 语言通道: instruction → 确定性 token (真实文本派生, 非零)
# ------------------------------------------------------------------ #

def lang_tokens(text: str, n: int = 8, vocab: int = 256) -> np.ndarray:
    """instruction → 确定性 token 序列 (占位 tokenizer, 非语言模型)。

    ⚠ 两个必须知道的前提:
    1. 这里对**全文**取 sha256 摘要再展开成 n 个 id, 刻意丢弃词序与语义 —— 它只
       检验"语言通道能否在管线里真的可变并传导到训练", 不检验任何语言理解。
    2. 由于 1000 条 instruction 之间的唯一差异就是 seed 数字, 这个摘要**等价于
       seed 的指纹**。因此若观测到"语言通道有用", 含义是模型在背诵 seed, 不是
       在理解指令 —— 报告里必须这样限定, 不能读成 VLA 的语言能力得到验证。

    首版按"前 n 个字符各取一个 id"实现, 而 instruction 的前 8 个字符在 1000 条里
    完全相同 (`'攀岩竞技: 从 s'`), 差异出现在第 17 位 → lang 恒为同一向量, 于是
    "打乱语言后 val 完全不变" (Δ=0.0) 是**探针自身失效**的假象, 不是数据结论。
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    ids = [digest[i] % vocab for i in range(n)]
    return np.asarray(ids, dtype=np.int64)


# ------------------------------------------------------------------ #
# 4. 样本构造: state(30) ← 关节12 + 末端3 + 相位one-hot + 触觉3
#              action ← 未来 horizon 步的 12 维动作
# ------------------------------------------------------------------ #

PHASES = ("hang", "reach", "swing", "dyno", "capture", "climb_on")


def build_samples(episodes: List[dict], horizon: int, tactile_every: int = 3) -> dict:
    """按 episode 构造样本; 触觉量按 tactile_every 步重跑一次 (EKF 收敛是慢过程,
    逐步重跑代价高且短 episode 内 μ̂ 变化有限)。"""
    X, Y, L, EPID = [], [], [], []
    skipped_short = 0
    for ei, ep in enumerate(episodes):
        acts = [np.asarray(a, dtype=np.float64) for a in ep["actions"]]
        obs = ep["observations"]
        T = len(obs)
        if T <= horizon:
            skipped_short += 1
            continue
        joints = np.stack([_vec(o["joint_positions"]) for o in obs])      # [T,12]
        ee = np.stack([_vec(o["ee_position"]) for o in obs])               # [T,3]
        ph = np.stack([_phase_onehot(o.get("phase", "")) for o in obs])    # [T,6]
        tac = np.tile(_episode_tac(ep, T, tactile_every), (1, 1))           # [T,3]

        lang = lang_tokens(str(ep.get("instruction", "")))
        for t in range(T - horizon):
            state = np.concatenate([joints[t], ee[t], ph[t], tac[t]])       # 24 维
            target = np.stack(acts[t + 1: t + 1 + horizon])                 # [H,12]
            X.append(state); Y.append(target); L.append(lang); EPID.append(ei)

    return {"X": np.asarray(X, dtype=np.float32),
            "Y": np.asarray(Y, dtype=np.float32),
            "L": np.asarray(L, dtype=np.int64),
            "epid": np.asarray(EPID, dtype=np.int64),
            "skipped_short_episodes": skipped_short}


def _phase_onehot(p) -> np.ndarray:
    v = np.zeros(len(PHASES), dtype=np.float32)
    if p in PHASES:
        v[PHASES.index(p)] = 1.0
    else:
        v[:] = 0.5                                   # 未知相位 → 中性, 不假装是某个相位
    return v


_TAC_CACHE: Dict[int, np.ndarray] = {}


def _episode_tac(ep: dict, T: int, every: int) -> np.ndarray:
    key = int(ep["episode_meta"]["seed"]) * 1000 + T
    if key not in _TAC_CACHE:
        _TAC_CACHE[key] = tactile_features(ep, max(T, 8))
    full = _TAC_CACHE[key]
    return full[:T] if full.shape[0] >= T else np.pad(full, ((0, T - full.shape[0]), (0, 0)))


# ------------------------------------------------------------------ #
# 5. 切分 (按 episode, 防邻帧泄漏) + 基线
# ------------------------------------------------------------------ #

def _arrays(S: dict) -> dict:
    return {k: v for k, v in S.items() if isinstance(v, np.ndarray)}


def episode_split(S: dict, val_ratio: float = 0.2, seed: int = 0):
    ep_ids = np.unique(S["epid"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ep_ids)
    n_val = max(1, int(len(perm) * val_ratio))
    val_eps = set(perm[:n_val].tolist())
    m_val = np.isin(S["epid"], list(val_eps))
    return (
        {k: v[~m_val] for k, v in _arrays(S).items()},
        {k: v[m_val] for k, v in _arrays(S).items()},
        {"n_train_episodes": int(len(ep_ids) - len(val_eps)),
         "n_val_episodes": int(len(val_eps)),
         "val_episode_overlap_with_train": 0},   # 按 episode 切 → 结构性为 0
    )


def leakage_probe(S: dict) -> dict:
    """量化"若按全局时序 80/20 切"会泄漏多少: val 首样本在 train 中的最近邻距离。"""
    X, epid = S["X"], S["epid"]
    cut = int(len(X) * 0.8)
    tr, va = X[:cut], X[cut:]
    if len(tr) == 0 or len(va) == 0:
        return {}
    d = np.linalg.norm(va[:200][:, None, :] - tr[None, :, :], axis=2).min(axis=1)
    same_ep = int(np.sum(epid[cut:cut + 200] == epid[:cut][-1]))
    return {"naive_global_split_median_nn_distance": float(np.median(d)),
            "naive_split_val_samples_with_epid_equal_last_train": same_ep}


def constant_baseline_mse(Ytr: np.ndarray, Yva: np.ndarray) -> float:
    """逐维训练集均值预测 —— 任何模型至少要打过它, 否则模型没学到东西。"""
    mu = Ytr.mean(axis=0)
    return float(((Yva - mu) ** 2).mean())


# ------------------------------------------------------------------ #
# 6. 训练 + 消融
# ------------------------------------------------------------------ #

def make_batch(S: dict, idx: np.ndarray, cfg, device, shuffle_lang=False,
               zero_tactile=False, lang_seed=0):
    import torch
    X = S["X"][idx].copy()
    if zero_tactile:
        X[:, -3:] = 0.0
    L = S["L"][idx].copy()
    if shuffle_lang:
        L = L[np.random.default_rng(lang_seed).permutation(len(L))]
    B, H = len(idx), cfg.action_horizon
    state = np.zeros((B, cfg.state_dim), dtype=np.float32)
    n = min(X.shape[1], cfg.state_dim)
    state[:, :n] = X[:, :n]
    action = np.zeros((B, H, cfg.action_dim), dtype=np.float32)
    Y = S["Y"][idx]
    action[:, :min(H, Y.shape[1]), :] = Y[:, :H, :]
    lang = torch.from_numpy(L.astype(np.int64))
    return (None, lang, torch.from_numpy(state), torch.from_numpy(action))


def predict(trainer, S: dict, idx: np.ndarray, cfg, **kw) -> np.ndarray:
    import torch
    frames, lang, state, action = make_batch(S, idx, cfg, cfg.device, **kw)
    trainer.model.eval()
    with torch.no_grad():
        pred, _ = trainer.model(None, lang, state, None)
    return pred.cpu().numpy()


def main() -> int:
    import torch
    # Windows GBK 控制台: 输出含中文/希腊字母, 强制 UTF-8 否则 UnicodeEncodeError
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description="1000 eps × 触觉闭环 → VLA 训练 + 数据质量核验")
    ap.add_argument("--size", default="vla_tiny", choices=list(
        __import__("embodied_ai.models.vla_config", fromlist=["VLAConfig"]).VLAConfig.ablations().keys()))
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-after", type=int, default=60, help="跑这些步后再评估")
    ap.add_argument("--max-episodes", type=int, default=0, help="0=全部 1000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    from embodied_ai.models.vla_config import VLAConfig
    from embodied_ai.models.vla_model import VLAModel
    from embodied_ai.models.vla_trainer import VLATrainer

    t0 = time.time()
    print("=== 1) 载入 1000 episodes + 数据质量核验 ===")
    episodes = load_episodes()
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    rep = {"provenance": {
        "state_joint_ee_phase": "real_computed (climb_scene 闭环)",
        "state_tactile": "synthetic (synth_tactile_obs) + assumed μ 映射 (未标定)",
        "language": "real_text (instruction 模板)",
        "action": "real_computed (相位模板+微扰)",
        "vision": "absent (frames=None; 数据集中不存在图像)",
    }}
    rep["audit"] = audit(episodes)
    a = rep["audit"]
    print(f"  episodes={a['n_episodes']}  steps={a['n_steps_total']} "
          f"(min/max={a['steps_min']}/{a['steps_max']})")
    print(f"  obs/act 长度不一致 episode: {a['episodes_with_obs_act_mismatch']}")
    print(f"  解析异常坐标: {a['bad_vectors']}")
    print(f"  action 恒为常数的维度: {a['action_constant_dims'] or '无'}")
    print(f"  action 逐维 std: {a['action_per_dim_std']}")
    print(f"  instruction 唯一数: {a['unique_instructions']} / {a['n_episodes']}")
    print(f"  lang token 行唯一数: {a['unique_lang_token_rows']} "
          f"(退化={a['lang_channel_degenerate']})")
    print(f"  action 方差前 4 维占比: {100*a['action_variance_share_top4_dims']:.1f}%"
          f"  | 不同 std 取值数: {a['action_distinct_std_values']} / 12")
    print(f"  物理字段唯一值数: {a['physics_field_unique_value_counts']}")
    print(f"  **退化为单一值的物理字段**: {a['physics_degenerate_fields']}")
    if a["lang_channel_degenerate"]:
        raise AssertionError(
            "语言通道退化为常量 —— 消融探针会失去意义 (val 对 lang 的 Δ 恒为 0), "
            "先修 tokenizer 再谈语言通道结论")

    print("\n=== 2) 触觉闭环特征 (EKF + 机会约束握力) ===")
    S = build_samples(episodes, horizon=VLAConfig.ablations()[args.size].action_horizon)
    print(f"  样本数={len(S['X'])}  state 维={S['X'].shape[1]}  "
          f"target={S['Y'].shape[1:]}  跳过过短 episode={S['skipped_short_episodes']}")
    tac = S["X"][:, -3:]
    print(f"  触觉 mu_hat 范围 [{tac[:,0].min():.3f}, {tac[:,0].max():.3f}] "
          f"sigma_mu 范围 [{tac[:,1].min():.4f}, {tac[:,1].max():.4f}]  "
          f"f_n 归一范围 [{tac[:,2].min():.3f}, {tac[:,2].max():.3f}]")
    rep["samples"] = {"n": int(len(S["X"])), "state_dim": int(S["X"].shape[1]),
                      "skipped_short": S["skipped_short_episodes"],
                      "tactile_mu_range": [float(tac[:, 0].min()), float(tac[:, 0].max())],
                      "tactile_sigma_range": [float(tac[:, 1].min()), float(tac[:, 1].max())]}

    print("\n=== 3) 切分 (按 episode) + 泄漏探针 ===")
    Str, Sva, split_info = episode_split(S)
    rep["split"] = split_info
    rep["leakage_probe"] = leakage_probe(S)
    print(f"  train_eps={split_info['n_train_episodes']} val_eps={split_info['n_val_episodes']}")
    print(f"  全局时序切分的 val→train 最近邻中位距离: "
          f"{rep['leakage_probe'].get('naive_global_split_median_nn_distance', float('nan')):.5f}"
          f"  (≈0 ⇒ 邻帧泄漏, val loss 无意义)")

    cfg = VLAConfig.ablations()[args.size]
    cfg.device = "cpu"; cfg.use_mixed_precision = False; cfg.seed = args.seed
    torch.manual_seed(args.seed)
    model = VLAModel(cfg)
    trainer = VLATrainer(cfg, model)
    n_params = sum(p.numel() for p in model.parameters())

    base_mse = constant_baseline_mse(Str["Y"], Sva["Y"])
    var_y = float(Sva["Y"].var(axis=0).sum())
    print(f"\n  常数均值基线 val MSE = {base_mse:.6f}   (val 目标逐维方差和 = {var_y:.6f})")

    print("\n=== 4) 训练 ===")
    rng = np.random.default_rng(args.seed)
    losses = []
    for it in range(args.iters):
        idx = rng.integers(0, len(Str["X"]), size=args.batch_size)
        batch = make_batch(Str, idx, cfg, cfg.device)
        losses.append(trainer.train_step(batch))
        if (it + 1) % 50 == 0 or it == 0:
            print(f"  iter {it+1:4d}  loss={losses[-1]:.6f}  "
                  f"(前50均值 {np.mean(losses[-50:]):.6f})  {time.time()-t0:.0f}s")

    print("\n=== 5) 评估: 训练后 val + 通道消融 ===")
    vidx = np.arange(len(Sva["X"]))
    n_eval = min(160, len(vidx))
    eidx = vidx[:n_eval]

    def mse_of(**kw):
        pred = predict(trainer, Sva, eidx, cfg, **kw)
        tgt = Sva["Y"][:n_eval, :cfg.action_horizon, :]
        return float(((pred - tgt) ** 2).mean())

    m_full = mse_of()
    m_lang_shuf = mse_of(shuffle_lang=True, lang_seed=7)
    m_tac_zero = mse_of(zero_tactile=True)
    r2 = 1.0 - m_full / base_mse

    # 逐维: 常数基线只吃"该维方差", 所以按维算解释率才知道模型到底在预测什么。
    # 数据侧若存在"多个维是同一模板常数", 总体解释率会被少数几个维主导。
    pred = predict(trainer, Sva, eidx, cfg)
    tgt = Sva["Y"][:n_eval, :cfg.action_horizon, :]
    tr_mu_dim = Str["Y"].mean(axis=0)          # [H, 12] — 逐 (horizon, dim) 的常数基线
    per_dim = []
    for d in range(cfg.action_dim):
        pv = float(((tgt[:, :, d] - tr_mu_dim[:, d]) ** 2).mean())   # 常数基线该维 MSE
        mv = float(((tgt[:, :, d] - pred[:, :, d]) ** 2).mean())     # 模型该维 MSE
        var_d = float(tgt[:, :, d].var())
        per_dim.append({"dim": d, "variance": round(var_d, 6),
                        "const_mse": round(pv, 6), "model_mse": round(mv, 6),
                        "explained": round(1 - mv / pv, 4) if pv > 1e-12 else None})
    informative = [p for p in per_dim if p["variance"] > 1e-4]
    print("  逐维解释率 (var>1e-4 视为有信息):")
    for p in per_dim:
        tag = "" if p["variance"] > 1e-4 else "   ← 该维几乎是常数模板"
        print(f"    dim{p['dim']:2d} var={p['variance']:.6f} "
              f"const={p['const_mse']:.6f} model={p['model_mse']:.6f} "
              f"explained={p['explained']}{tag}")
    print(f"  val MSE (全通道)        = {m_full:.8f}")
    print(f"  val MSE (语言随机打乱)  = {m_lang_shuf:.8f}   Δ={m_lang_shuf-m_full:+.3e}")
    print(f"  val MSE (触觉置零)      = {m_tac_zero:.8f}   Δ={m_tac_zero-m_full:+.3e}")
    print(f"  常数基线                = {base_mse:.8f}")
    print(f"  相对常数基线的方差解释率 = {100*r2:.1f}%")
    # persistence 对照: 把当前关节角直接抄成未来 H 步动作。若它优于常数基线,
    # 说明模型分数主要来自"动作≈状态"的自相关而非学习。
    H = cfg.action_horizon
    pers = Sva["X"][:n_eval, :cfg.action_dim][:, None, :].repeat(H, axis=1)
    mse_pers = float(((tgt - pers) ** 2).mean())
    rep["eval"] = {"iters": args.iters, "params": n_params,
                   "final50_loss": float(np.mean(losses[-50:])),
                   "val_mse_full": m_full, "val_mse_lang_shuffled": m_lang_shuf,
                   "val_mse_tactile_zeroed": m_tac_zero,
                   "val_mse_constant_baseline": base_mse,
                   "val_mse_persistence_baseline": mse_pers,
                   "variance_explained_vs_constant": r2,
                   "variance_explained_by_persistence": 1 - mse_pers / base_mse,
                   "per_dim": per_dim,
                   "n_informative_action_dims": len(informative),
                   "n_val_evaluated": n_eval}
    print(f"  persistence 基线 (抄当前关节角) val MSE = {mse_pers:.8f} "
          f"(解释率 {100*(1-mse_pers/base_mse):.1f}%)"
          f"  ⇒ {'模型学到的是状态→动作映射, 非自相关' if mse_pers > m_full else '模型主要在抄状态'}")

    print("\n=== 6) 判定 ===")
    verdict = []
    verdict.append(("管道可跑通 (1000 eps → VLA 训练步)", True))
    ok_beats = m_full < base_mse
    verdict.append((f"模型优于常数基线 ({'是' if ok_beats else '否'})", ok_beats))
    verdict.append(("模型优于 persistence 基线 (不只是抄当前关节角)", m_full < mse_pers))
    verdict.append(("语言通道携带可用信息 (打乱后 val 变差)", m_lang_shuf > m_full))
    verdict.append(("触觉通道携带可用信息 (置零后 val 变差)", m_tac_zero > m_full))
    # 注意: 不能用"var > 1e-4 的维数"来判退化 —— 实测 7 个维的 var=0.001048 只是
    # 勉强越过阈值, 它们其实是**同一个模板常数加了同一份噪声** (std 完全相等),
    # 那条判据会给出假绿灯。改用"不同 std 取值数"直接量结构冗余。
    distinct_std = a["action_distinct_std_values"]
    verdict.append((f"动作空间结构非冗余 (不同 std 取值数 ≥ 8/12, 实测 {distinct_std}; "
                    f"前 4 维占方差 {100*a['action_variance_share_top4_dims']:.1f}%)",
                    distinct_std >= 8))
    for label, ok in verdict:
        print(f"  [{'✅' if ok else '❌'}] {label}")
    rep["verdict"] = [{"check": l, "ok": bool(o)} for l, o in verdict]
    rep["runtime_s"] = round(time.time() - t0, 1)

    print(f"\n耗时 {rep['runtime_s']}s | 参数 {n_params:,} | "
          f"final50 loss {rep['eval']['final50_loss']:.6f}")
    if args.report:
        Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"报告已写出: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
