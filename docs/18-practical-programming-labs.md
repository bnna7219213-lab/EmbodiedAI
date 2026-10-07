# 实用编程实验 (Practical Programming Labs)

> **日期**: 2026-10-08
> **定位**: 10 个编程动手实验, 每个 15-30 分钟, 覆盖从"读代码"到"写新模块"的能力递进。
> **前提**: 已 clone 仓库, Python 3.10+, 已完成 [FIRST_MOVE.md](../FIRST_MOVE.md) 第 1 步。
> **规则**: 
>   - 每道 Lab 都有 "代码模板" + "期望输出" + "思考问题"
>   - 改写代码后**必须**跑 `pytest` 确认没回归
>   - 失败的 Lab 记录到 `experiments/lab_notebook.md`

---

## Lab 1: 读懂电流环 PI 公式 (读代码 + 验证)

**目标**: 读懂 `src/embodied_ai/hardware/motor_model.py` 的 `current_loop_gains()`, 手算验证。

**步骤**:

```bash
# 1. 打开文件, 找到 current_loop_gains 方法
cat src/embodied_ai/hardware/motor_model.py | grep -n "current_loop_gains" 

# 2. 新建脚本:
cat > scripts/lab_01_pi_gains.py << 'PYEOF'
import numpy as np
from src.embodied_ai.hardware.motor_model import MotorSpec

# B1 腱驱动电机参数 (来自 contracts/hw_requirements.yaml climbing.motor_drive_tendon)
# 假设值 (Phase P1 measured 之前):
m = MotorSpec(
    Rs_ohm=0.050,   # 50 mΩ 相电阻
    Ld_h=15e-6,     # 15 µH d 轴电感
    Lq_h=28e-6,     # 28 µH q 轴电感
    psi_m_wb=0.0028, # 2.8 mWb 永磁磁链
    pole_pairs=7,    # 7 对极
    provenance={"source": "contracts/hw_requirements.yaml climbing.motor_drive_tendon (assumed)"}
)

# 不同带宽下的 PI 增益:
print("电流环 PI 增益 (极点配置 kp=L·ωc, ki=Rs·ωc):")
print(f"{'带宽(Hz)':>10s} {'kp_d':>10s} {'ki_d':>10s} {'kp_q':>10s} {'ki_q':>10s} {'ωc·T_ctrl':>10s} {'稳定?':>8s}")
for bw in [200, 400, 800, 1000, 1200]:
    g = m.current_loop_gains(bw_hz=bw)
    wc = 2 * np.pi * bw
    T_ctrl = 1.0 / m.f_ctrl_hz  # 100 µs
    stability = "YES" if wc * T_ctrl < 2.0 else "UNSTABLE!"
    print(f"{bw:>10d} {g['kp_d']:>10.4f} {g['ki_d']:>10.1f} {g['kp_q']:>10.4f} {g['ki_q']:>10.1f} {wc*T_ctrl:>10.3f} {stability:>8s}")

print(f"\n推荐带宽: 800 Hz (kp_d={m.current_loop_gains(800)['kp_d']:.4f}, ki_d={m.current_loop_gains(800)['ki_d']:.1f})")
PYEOF

python scripts/lab_01_pi_gains.py
# (注: sys.path 需要从 repo root 跑)
```

**期望输出示例**:
```
电流环 PI 增益 (极点配置 kp=L·ωc, ki=Rs·ωc):
     带宽(Hz)       kp_d       ki_d       kp_q       ki_q    ωc·T_ctrl     稳定?
       200     0.0188       62.8     0.0352       62.8      0.126      YES
       400     0.0377      125.7     0.0703      125.7      0.251      YES
       800     0.0754      251.3     0.1407      251.3      0.503      YES
      1000     0.0942      314.2     0.1759      314.2      0.628      YES
      1200     0.1131      377.0     0.2111      377.0      0.754      YES

推荐带宽: 800 Hz (kp_d=0.0754, ki_d=251.3)
```

**思考**:
- 带宽从 200 → 1200 Hz, kp 线性增长 6 倍。**实际应该选哪个?** 带宽越高响应越快, 但越容易受离散化影响。工程通常选 f_ctrl/10 = 1000 Hz 以下。
- `ωc·T_ctrl < 2` 是离散稳定条件。**哪一档会让电路振荡?**

---

## Lab 2: 写第一个测试 (TDD 入门)

**目标**: 测试驱动开发 — 先写测试, 确认它 FAIL, 然后让代码 PASS。

**场景**: `chance_constrained_grip` 的 `InfeasibleGripError` 应该对不可行输入抛出。

```bash
# 1. 新建测试文件:
cat > tests/test_lab_02_grip.py << 'PYEOF'
"""Lab 2: InfeasibleGripError 的正确性"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embodied_ai.safety.chance_constrained_grip import (
    chance_constrained_grip, InfeasibleGripError, _z_1_minus_delta
)
import pytest

class TestChanceConstrainedGripLab:
    """Lab 2 的练习题 — 填写 [???] """

    def test_z_value_delta_005(self):
        """δ=0.05 时 z 值应该是 1.6449"""
        z = _z_1_minus_delta(0.05)
        assert abs(z - 1.6449) < 0.001, f"z={z}, 期望 1.6449"

    def test_z_value_delta_001(self):
        """δ=0.001 时 z 值应该是 3.0902 (攀岩高安全)"""
        z = _z_1_minus_delta(0.001)
        assert abs(z - 3.0902) < 0.001

    def test_infeasible_raises(self):
        """σ_μ 过大 (摩擦不确定度过高) → 机会约束无解"""
        with pytest.raises(InfeasibleGripError):
            chance_constrained_grip(10.0, mu_hat=0.3, sigma_mu=0.5, delta=0.001)

    def test_known_value_correct(self):
        """已知输入 → 已知输出"""
        # f_t=10N, μ̂=0.8, σ_μ=0.05, δ=0.05
        # z=1.6449, denom=0.8-1.6449*0.05=0.7178, f_n=13.93
        result = chance_constrained_grip(10.0, 0.8, 0.05, delta=0.05)
        assert abs(result - 13.93) < 0.1, f"f_n={result}, 期望 13.93"

    def test_safety_increases_with_stricter_delta(self):
        """更严格的 δ → 更大握力 (安全裕度更高)"""
        f_n_05 = chance_constrained_grip(10.0, 0.8, 0.05, delta=0.05)
        f_n_001 = chance_constrained_grip(10.0, 0.8, 0.05, delta=0.001)
        assert f_n_001 > f_n_05, f"δ=0.001 应该更大, 但 {f_n_001} < {f_n_05}"

    def test_exercise_for_you(self):
        """🎯 你的练习题: 攀岩 dyno 场景 f_t=120N, μ̂=0.85, σ_μ=0.08, δ=0.001
        手算 f_n, 然后写断言验证。答案填在下面:
        z = ???
        denom = ???
        f_n = ?? ASSERT = ???
        """
        # TODO: 取消注释并完成
        # result = chance_constrained_grip(120.0, 0.85, 0.08, delta=0.001)
        # assert abs(result - ???) < 0.5
        pass  # 先让它 SKIP, 你完成上面的计算后改 pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
PYEOF

# 2. 看测试失败:
python -m pytest tests/test_lab_02_grip.py -v 2>&1 | tail -15
# 期望: test_exercise_for_you FAIL — 其他 5 个 PASS

# 3. 完成练习题, 重新跑, 全部 PASSED
```

**期望最终输出**: 6 passed。

---

## Lab 3: 用 EKF 估计摩擦系数 (算法实战)

**目标**: 用 `MultiRateEKF` 对一个**已知 μ=0.75** 的合成触觉序列做估计, 验证 EKF 收敛。

```bash
cat > scripts/lab_03_ekf_estimate.py << 'PYEOF'
"""Lab 3: 用 EKF 估计 μ"""
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from embodied_ai.perception.multirate_ekf import MultiRateEKF
from pathlib import Path

# 合成一段"已知真实 μ=0.75"的触觉数据
# 物理: f_t 从 0 线性上升到 μ*f_n = 0.75*30N = 22.5N, 保持 2s
dt = 0.01  # 100 Hz
T = 2.0    # 2s
f_n_true = 30.0
mu_true = 0.75

steps = int(T / dt)
true_f_t = np.linspace(0, mu_true * f_n_true, steps)
# 加测量噪声
np.random.seed(42)
measurements = true_f_t + np.random.normal(0, 0.3, steps)

# EKF 估计
ekf = MultiRateEKF(mu_prior_mean=0.8, mu_prior_std=0.1, f_n=f_n_true)
mu_trace = []
f_t_trace = []
for z in measurements:
    ekf.predict()
    ekf.update_tactile(z)
    f_t_trace.append(ekf.x[0])
    mu_trace.append(ekf.x[1])

# 分析:
final_mu_err = abs(mu_trace[-1] - mu_true)
print(f"真实 μ = {mu_true}")
print(f"最终估计 μ̂ = {mu_trace[-1]:.4f}")
print(f"误差: {final_mu_err:.4f} (目标 <0.05)")
print(f"{'✅ EKF收敛' if final_mu_err < 0.05 else '⚠️ EKF误差偏大, 检查R/Q'}")

# 打印每 0.5s 的估计过程:
print(f"\n{'t(s)':>6s} {'f_t_est':>8s} {'μ̂':>8s} {'f_t_real':>8s} {'z_noisy':>8s}")
for i in range(0, steps, 50):
    t = i * dt
    print(f"{t:6.2f} {f_t_trace[i]:8.3f} {mu_trace[i]:8.4f} {true_f_t[i]:8.3f} {measurements[i]:8.3f}")
PYEOF

python scripts/lab_03_ekf_estimate.py
```

**期望结果**: μ 最终估计值应该收敛到 ~0.75±0.05。如果不收敛, 调整 `R_tac` 或 `mu_prior_std`。

---

## Lab 4: 闭环触觉→握力 (端到端 L4 mini)

**目标**: 用 Lab 3 的 EKF 估计 + `chance_constrained_grip`, 感受完整触觉→握力闭环。

```bash
cat > scripts/lab_04_tactile_loop.py << 'PYEOF'
"""Lab 4: 触觉→EKF→握力调度→PI 完整 mini 闭环"""
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from collections import namedtuple
from embodied_ai.perception.multirate_ekf import MultiRateEKF
from embodied_ai.safety.chance_constrained_grip import chance_constrained_grip, InfeasibleGripError
from pathlib import Path

# 合成扰动序列: 静态握持 → 扰动上升 → 恢复
def make_disturbance(steps=200, dt=0.05):
    """Step 0-99: 静态握持 f_t≈2N; Step 100-150: 扰动升到 20N; Step 150-199: 回到 5N"""
    readings = np.zeros(steps)
    readings[:100] = np.random.normal(2.0, 0.3, 100)
    readings[100:150] = np.linspace(2.0, 20.0, 50) + np.random.normal(0, 0.5, 50)
    readings[150:] = np.random.normal(5.0, 0.3, 50)
    return readings

# 灵敏度为 0.6 的 SlipDetector:
class SimpleSlipDetector:
    def __init__(self, threshold=0.6): self.threshold = threshold
    def check(self, f_t_est, mu_est, f_n_cmd):
        denom = mu_est * f_n_cmd
        if denom < 0.1: return False, 0.0
        ratio = f_t_est / denom
        return ratio >= self.threshold, ratio

# PI:
class SimplePI:
    def __init__(self, kp=1.0, ki=0.1, f_n_max=150.0):
        self.kp, self.ki, self.f_n_max = kp, ki, f_n_max
        self.integral = 0.0
    def compute(self, target, dt):
        self.integral += target * dt
        return float(np.clip(self.kp*target + self.ki*self.integral, 0.0, self.f_n_max))

# 闭环:
readings = make_disturbance()
ekf = MultiRateEKF(mu_prior_mean=0.8, mu_prior_std=0.1, f_n=30.0)
slip = SimpleSlipDetector(threshold=0.6)
pi = SimplePI(kp=1.0, ki=0.1, f_n_max=150.0)

f_n_cmd = 30.0
Step = namedtuple('Step', ['t','f_t_m','f_t_e','mu','ratio','slip','f_n_cmd','target'])
log = []

for i, z in enumerate(readings):
    ekf.predict(); ekf.update_tactile(z)
    f_t_e, mu_e, _ = ekf.x
    slip_det, ratio = slip.check(f_t_e, mu_e, f_n_cmd)

    if slip_det:
        try:
            target = chance_constrained_grip(f_t_e, mu_e, sigma_mu=0.05, delta=0.05)
        except InfeasibleGripError:
            target = 150.0
    else:
        target = 20.0  # 节能基线

    f_n_cmd = pi.compute(target, 0.05)
    log.append(Step(i*0.05, z, f_t_e, mu_e, ratio, slip_det, f_n_cmd, target))

# 分析:
final_10 = log[-10:]
slip_steps = [i for i, s in enumerate(log) if s.slip]
max_force = max(s.f_n_cmd for s in log)
converged = not any(s.slip for s in final_10) and len(slip_steps) > 0

print(f"滑脱触发: {len(slip_steps)} 帧 (step {slip_steps[0] if slip_steps else 'N/A'} - {slip_steps[-1] if slip_steps else 'N/A'})")
print(f"最大握力: {max_force:.1f} N")
print(f"最终握力 (恢复): {np.mean([s.f_n_cmd for s in final_10]):.1f} N")
print(f"闭环收敛: {'✅' if converged else '⚠️'}")

# 打印关键帧:
print(f"\n{'t(s)':>6s} {'f_t_m':>7s} {'f_t_e':>7s} {'μ̂':>7s} {'ratio':>6s} {'slip':>5s} {'f_n':>7s}")
for i in range(0, len(log), 25):
    s = log[i]
    print(f"{s.t:6.2f} {s.f_t_m:7.1f} {s.f_t_e:7.1f} {s.mu:7.3f} {s.ratio:6.3f} {'Y' if s.slip else '.':>5s} {s.f_n_cmd:7.1f}")
PYEOF

python scripts/lab_04_tactile_loop.py
```

**期望结果**: 扰动出现 → slip 触发 → f_n_cmd 上升 → 扰动消失 → f_n_cmd 下降回到节能基线。

---

## Lab 5: 写一个新模块 (为新需求写代码)

**目标**: 添加一个 `SlipDetector` 的**变体** (速率限制版)，然后集成进闭环。

```bash
cat > scripts/lab_05_new_module.py << 'PYEOF'
"""Lab 5: 增量开发 — 添加 RateLimitedSlipDetector, 避免瞬间扰动误触发"""
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# 继承你已有的 SlipDetector, 添加"必须在 incipient 区间连续 N 帧才确认"的逻辑

class RateLimitedSlipDetector:
    """在 SlipDetector 基础上增加: slip_ratio 持续 >= threshold 连续 N 帧 → 确认滑脱"""
    def __init__(self, threshold=0.6, confirm_frames=3):
        self.threshold = threshold
        self.confirm_frames = confirm_frames
        self._buffer = []

    def check(self, f_t_est, mu_est, f_n_cmd) -> tuple[bool, float]:
        denom = mu_est * f_n_cmd
        if denom < 0.1:
            ratio = 0.0
        else:
            ratio = f_t_est / denom
        # 缓冲:
        self._buffer.append(ratio >= self.threshold)
        if len(self._buffer) > self.confirm_frames:
            self._buffer.pop(0)
        # 必须连续确认:
        confirmed = len(self._buffer) == self.confirm_frames and all(self._buffer)
        return confirmed, float(np.clip(ratio, 0.0, 5.0))

# 测试它:
det = RateLimitedSlipDetector(threshold=0.6, confirm_frames=3)

# 序列: slip_ratio 连续 0.5 → 0.7 → 0.8 → 0.9 → 0.4
test_inputs = [(15.0, 0.8, 30.0),  # ratio = 0.625
               (18.0, 0.8, 30.0),  # ratio = 0.75
               (22.0, 0.8, 30.0),  # ratio = 0.917
               (25.0, 0.8, 30.0),  # ratio = 1.04
               (10.0, 0.8, 30.0)]  # ratio = 0.417

print("RateLimitedSlipDetector 测试 (confirm_frames=3):")
for i, (f_t, mu, f_n) in enumerate(test_inputs):
    confirmed, ratio = det.check(f_t, mu, f_n)
    status = "✅ 确认" if confirmed else "·   累积"
    print(f"  frame {i}: ratio={ratio:.3f} {status}")

# 期望: frame 0-1 不确认（只有 1-2 帧进入阈值）, frame 3 确认（连续 3 帧 ≥0.6）, frame 4 不确认
PYEOF

python scripts/lab_05_new_module.py
```

**思考**: `confirm_frames=3` 在 200 Hz 控制速率下 = **15 ms 确认延迟**。这对攀岩 dyno 够吗? 通常 dyno 起跳到抓点只要 100-300 ms, 15 ms 延迟是可以接受的。

---

## Lab 6-10 (选做 / 进阶)

| Lab | 名称 | 难度 | 涉及代码 |
|---|---|---|---|
| **6** | **PID 整定仿真** — 用 Python 仿真一个二阶系统 (弹簧-质量-阻尼), 手动调 PID, 观察超调 | ⭐⭐ | 新建脚本即可 |
| **7** | **CAN 帧解码** — 写一个独立参考编码器 (不 import 被测代码), 验证 DBC 解码 | ⭐⭐⭐ | `tests/test_lab_07_reference_encoder.py` + `contracts/canfd/joint_control.dbc` |
| **8** | **RK4 轨迹积分** — 给 `climbing_dynamics_core.py` 的 dyno 轨迹做一次 4 阶 Runge-Kutta 积分, 看数值稳定性 | ⭐⭐⭐ | `src/embodied_ai/dynamics/climbing_dynamics_core.py` |
| **9** | **QP 优化入门** — 用 `scipy.optimize.minimize` 实现 ZMP 平衡 (2 腿站立) | ⭐⭐⭐⭐ | 新建脚本 |
| **10** | **端到端攀爬仿真** — 把 L1-L6 集成进 `make_env("climb_wall")`, 用 reference 后端跑一个 5-step 攀爬任务 | ⭐⭐⭐⭐⭐ | 需要 L1 完整 |

每道 Lab 的完成判据都是: **有一个可运行的脚本 + 至少 1 个测试断言 PASSED**。

---

## Lab 使用规范

1. **每个 Lab 用自己的脚本文件** (`scripts/lab_XX_xxx.py`), 不修改项目本体代码
2. **Lab 失败时**: 在 `experiments/lab_notebook.md` 记录: 做了什么 / 怎么失败 / 学到什么
3. **Lab 成功时**: 如果有价值, 转化为正式测试或模块 (提 PR)
4. **Lab 与正式项目的边界**: Lab 脚本可以随时删, 但如果你发现 Lab 代码有复用价值, 按 `pyproject.toml` 的测试/代码规范整理进 `src/` 或 `tests/`

---

## 完成 Lab 后的下一步

| 完成 Lab 数 | 你应该能做 |
|---|---|
| Lab 1-2 | 读懂 PID 公式, 写 TDD 测试 |
| Lab 3-4 | 理解 EKF 和触觉闭环的完整数据流 |
| Lab 5 | 能独立写一个功能完整的 Python 新模块 |
| Lab 6-7 | 能做"参考实现" (独立于被测代码做 cross-check) |
| Lab 8-10 | 能独立设计一个完整的从感知→决策→控制链路 |

**准备好了?** 做 Lab 10, 然后是 [FIRST_MOVE.md 第 5 步的 Lab 练习](../FIRST_MOVE.md)。
