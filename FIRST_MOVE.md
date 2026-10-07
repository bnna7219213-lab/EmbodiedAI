# FIRST_MOVE — 今天动手做的事

> **日期**: 2026-10-08
> **定位**: 读完 docs/13-17 等一系列规划文档后，**最想知道的问题是: 我现在能做什么?** 本文档给你 5 件事，每件事 15-30 分钟，都能在今天做完。
> **前提**: 已 clone 到此仓库（已满足），有 Python 3.10+（已装 3.12）。

---

## 第 1 步: 跑通验收体系 (15 min)

**目的**: 确认你本地的代码平台是健康的。

```bash
cd C:\Users\bnna7\workspace\git\EmbodiedAI

# 1. 跑 verify (28 项全绿意味着代码平台正常)
python -m embodied_ai.cli verify --no-pytest
# 期望输出: 汇总: 28/28 PASS  / RESULT=VERIFY_OK

# 2. 跑 selfcheck (5 项全绿意味着"单仓自足"成立)
python -m embodied_ai.cli selfcheck
# 期望输出: [✅] 1.载体齐全  [✅] 2.指纹一致  [✅] 3.运行时可用  [✅] 4.隔离可跑  [✅] 5.后端可观测
#           RESULT: SELF_CONTAINED_OK

# 3. 跑 pytest (441 项全绿)
python -m pytest tests/ -q
# 期望输出: 441 passed in 90s

# 如果某步失败, 记下失败项的名称，这是你的第一个 "待办"
```

**这一步的价值**: 即使你什么硬件都没有，代码平台本身就有 441 个可运行的测试。这些测试就是你的"编程作业"。

---

## 第 2 步: 读懂你代码仓库的一个模块 (20 min)

**目的**: 从"文档读者"变成"代码读者"。

挑下面一个文件，**从头到尾读一遍**, 不跳过任何注释:

```bash
# 推荐顺序 (由浅入深):
# (a) 最安全, 纯 Python, 无硬件依赖:
src/embodied_ai/safety/chance_constrained_grip.py

# (b) 中等, 有 numpy, 纯算法:
src/embodied_ai/perception/multirate_ekf.py

# (c) 本项目最新原创:
src/embodied_ai/perception/tactile_closed_loop.py
```

**读的时候做这件事**: 找到下面这个函数

```python
# 在 chance_constrained_grip.py 里:
def chance_constrained_grip(f_t_req, mu_hat, sigma_mu, delta=0.05, min_margin=0.05):
```

**脑子算一遍**: 如果 `f_t_req=60, mu_hat=0.8, sigma_mu=0.05, delta=0.001` (攀岩高安全), `f_n` 应该是多少?

```
z = norm.ppf(1 - 0.001) = 3.0902
denom = 0.8 - 3.0902 * 0.05 = 0.8 - 0.1545 = 0.6455
f_n = 60 / 0.6455 = 92.90 N
```

**立刻验证**: 打开 Python 交互模式。

```bash
python -c "
from embodied_ai.safety.chance_constrained_grip import chance_constrained_grip
result = chance_constrained_grip(60.0, mu_hat=0.8, sigma_mu=0.05, delta=0.001)
print(f'f_n = {result:.2f} N')
print(f'和手算 92.90 N 一致? {\"YES\" if abs(result - 92.90) < 0.1 else \"NO — 检查\"}')
"

# 期望输出: f_n = 92.90 N / 和手算一致? YES
```

**这一步的价值**: 把你从"纸面理解"变成"手脑并用"。这种快速验证的习惯是工程师的核心素养。

---

## 第 3 步: 做一个小改动 (10 min)

**目的**: 完成你的第一次"代码提交前体验"。

```bash
# 1. 创建一个新分支 (命名规范: feature/xxx 或 fix/xxx)
git checkout -b feature/first-move-trial

# 2. 打开 tactile_closed_loop.py, 微调 SlipDetector 的 threshold

# 文件位置: src/embodied_ai/perception/tactile_closed_loop.py
# 找到 SlipDetector.__init__ 的 threshold=0.7
# 把它改成 threshold=0.6 (更敏感)

# 3. 跑测试确认没有回归
python -m pytest tests/test_tactile_closed_loop.py -q
# 期望: 17 passed (所有测试应该仍然通过，因为测试设了余量)

# 4. 如果测试通过，撤销改动 (这是 trial 分支):
git checkout -- src/embodied_ai/perception/tactile_closed_loop.py
git checkout main  # 回到主分支
git branch -D feature/first-move-trial

# 这一步不需要真正提交。目的只是让你体验 "改代码 → 跑测试" 的完整流程。
```

---

## 第 4 步: 画一个 B1 板框的 SVG (30 min, 无需 EDA 软件)

**目的**: 让你对"你要造的板子"有**物理尺寸直觉**。

```bash
# 用本项目工具:
python -c "
from scripts.export_b1_package import build_b1_circuit
from pathlib import Path
import json

circuit = build_b1_circuit()
print(f'元件数: {len(circuit[\"components\"])}')
print(f'导线数: {len(circuit[\"wires\"])}')

# 看前 3 个元件:
for c in circuit['components'][:3]:
    print(f'  {c[\"refDes\"]:8s} x={c[\"x\"]:3.0f} y={c[\"y\"]:3.0f} {c[\"spec\"][\"name\"]:40s} {c[\"spec\"][\"footprint\"]}')

# PCB 实际尺寸 (在 MotorDriveEngine.kt 注释中注释):
# 55.4mm × 44.3mm
# B1 板上有 27 个元件, 46 条导线
# 对照物: 一张名片是 90×54 mm, 所以 B1 板比名片的一半还小
print()
print('B1 板实际大小: 55.4 mm × 44.3 mm ≈ 2.2 inch × 1.75 inch')
print('对照: iPhone 主板 ≈ 90×55 mm, 一块乐高积木 32×32 mm')
print('可以放一个名片在桌上, 想象一半大小, 那就是你要造的板子')
"
```

**画图强化想象**: 拿一张名片放桌上。B1 板大概是名片一半那么大 (2.2×1.75 英寸)。上面要放 27 个元件和走 46 条线。想象一下---

```bash
# 如果你想看 PCB 布局:
# artifacts/b1_tendon_drive_package/B1_tendon_drive_pcb.svg
# 用浏览器打开这个 SVG 就能看到自动布局效果
start artifacts/b1_tendon_drive_package/B1_tendon_drive_pcb.svg
```

---

## 第 5 步: 做你的第一个 "Lab 练习" (20 min)

**目的**: 实际编程，不是纸面阅读。

**Lab 目标**: 对 D4 数据集做分析，找出哪个 scenario 的平均力最大。

```python
# 新建一个临时脚本:
# C:\Users\bnna7\workspace\git\EmbodiedAI\scripts\my_first_lab.py

import json
from collections import defaultdict
from pathlib import Path

D4_FILE = Path("experiments/d4_dataset/training_replay_v1.jsonl")

# 读数据
records = []
with open(D4_FILE) as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        if i == 0: continue  # 跳过 header
        records.append(obj)

# 按 scenario 分组, 统计平均 motor_current_a
scenario_currents = defaultdict(list)
for r in records:
    sc = r.get("scenario", "unknown")
    # 把电流当成切向力近似: 1A ≈ 0.5N
    scenario_currents[sc].append(r.get("motor_current_a", 0.0) * 0.5)

# 打印结果:
print("D4 数据集 scenario 平均切向力 (N):")
for sc, forces in sorted(scenario_currents.items(), key=lambda x: -sum(x[1])/len(x[1])):
    avg_f = sum(forces) / len(forces)
    print(f"  {sc:20s} 平均 {avg_f:6.1f} N  (基于 {len(forces):5d} 帧)")
```

```bash
# 运行:
python scripts/my_first_lab.py

# 期望输出 (示例):
# D4 数据集 scenario 平均切向力 (N):
#   climb_dyno            平均  124.8 N  (基于  5720 帧)
#   climb_grip            平均 443.5 N  (基于  5810 帧)
#   climb_reach           平均  268.3 N  (基于  5705 帧)
#   ...
```

**动脑**: dyno 场景的平均力最大 (124 N) — 这和物理直觉一致吗? 攀岩 dyno 时切向力最高, 因为要对抗重力向上跳。对, 合理。

**动脑**: 如果 `slip_detection_threshold=0.7`, 在 dyno 场景里哪些帧会触发滑脱? 用这个判断写第二行查询:

```python
# 接上, 计算 slip ratio (假设 μ̂=0.8, f_n_cmd=100 N)
MU_HAT = 0.8
F_N_CMD = 100.0
THRESHOLD = 0.7

slip_frames = []
for r in records:
    f_t = r.get("motor_current_a", 0.0) * 0.5
    ratio = f_t / (MU_HAT * F_N_CMD)
    if ratio >= THRESHOLD:
        slip_frames.append((r.get("scenario"), f_t, ratio))

print(f"\n滑脱阈值 {THRESHOLD} 下: 共 {len(slip_frames)} 帧触发 incipient slip (占总记录 {len(slip_frames)/len(records)*100:.1f}%)")
```

---

## 今天的收获

| 完成的事 | 你获得了什么 |
|---|---|
| verify 全绿 | 你能确认代码平台健康 |
| 读懂一个模块 | 从文档进入到代码 |
| 改一个参数 + 跑测试 | "改代码" 的完整流程直觉 |
| 看 B1 PCB SVG | 对你将要造的东西有物理尺寸直觉 |
| Lab 动手查 D4 数据集 | 从"看数据" → "分析数据" |

**下一步**: 读 docs/18-practical-programming-labs.md 做正式 Lab 练习。
