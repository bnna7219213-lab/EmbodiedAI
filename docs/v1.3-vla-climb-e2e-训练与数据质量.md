# VLA × 1000 episodes × 触觉闭环 — 端到端训练与数据质量核验

> 日期: 2026-10-17 · 脚本: `scripts/vla_train_climb_e2e.py`
> 运行: `python scripts/vla_train_climb_e2e.py --iters 400 --batch-size 16 --report output/vla_climb_e2e.json`
> （`output/` 已在 `.gitignore` 内，JSON 是本地产物、按上式可再生产；本文件抄录全部关键数值）
> 规模: 1000 episodes / 18000 步 → 13000 训练样本 · `vla_tiny` 8,534,588 参数 · CPU 21.7s

## 结论一句话

**管道成立，模型学到的是真的状态→动作映射；但这批数据无法验证 VLA 的语言与触觉能力——
根因是 seed 驱动的域随机化没有传到物理层，导致 11 个物理字段在 1000 条 episode 里全是同一个值。**

## 判定表

| 检查 | 结果 | 实测值 |
|---|---|---|
| 1000 eps → VLA 训练步跑通 | ✅ | loss 0.0414 → 0.0142 (final50 均值) |
| 优于常数均值基线 | ✅ | val MSE 0.014005 vs 0.024330 → **解释率 42.4%** |
| 优于 persistence 基线（不是只抄当前关节角） | ✅ | 抄关节角 val MSE 0.046050，**解释率 −89.3%**（比常数还差）→ 42.4% 不是自相关白捡的 |
| 语言通道携带可用信息 | ❌ | 打乱 lang 后 val MSE 变化 **Δ = −2.88e−06**（不升反降）→ 模型对语言完全不变 |
| 触觉通道携带可用信息 | ⚠ 名义 ✅ 实际可忽略 | 置零后 Δ = +7.6e−07（相对 5e−5）。`mu_hat` 全域只跨 [0.552, 0.572]，**区间宽 0.02** |
| 动作空间结构非冗余 | ❌ | 12 维中只有 **6 种不同 std 取值**；前 4 维占 **95.6%** 方差；dim 3/4/5/6/9/10/11 的 var 全等于 0.001048（同一模板常数 + 同一份噪声） |

## 为什么"触觉/语言通道"测不出东西

三者是**同一个 bug 的三个症状**：

1. `gen_climb_rollouts.py` 的 seed 只驱动攀岩闭环的**部分**分支，VSA/耦合参数是常量
   → `f_soft_peak_n`≡59.99999、`f_rigid_peak_n`≡115.4978、`soft_ratio`≡0.51949、
   `peak_finger_n`≡32.6986、`com_advance_m`≡0.6000、`wall_frames`≡7 …
   **11 个数值物理字段的唯一值数全部为 1**。
   （`f_soft_peak_n`=60.0 恰是本仓反复引用的指尖 Hertz 极限，怀疑是在该处被 clip 饱和。）
2. 我的触觉特征 μ_true 由 `peak_finger_n / f_soft_peak_n` 导出（**assumed 映射，无标定**）。
   分子分母都是常量 ⇒ 1000 条 episode 的 μ_true 相同 ⇒ EKF+机会约束握力的输出
   几乎恒定，模型学不到"触觉变化 → 握力调整"。
3. 1000 条 instruction 只在第 17 位的 seed 数字上不同，而 seed 并不改变动作 ⇒
   语言 token 即使被正确编码也是**与目标无关的噪声**。模型正确地学会了忽略它。

所以 **不能**把这次运行读成"VLA 双路径 / 触觉闭环已验证"。它验证的是：
数据装载、维度契约、触觉环与握力调度能作为状态特征进入训练、episode 级切分可用、
以及模型能真学到关节构型→未来动作的可预测结构。

## 顺带测出的既有管线问题（本次未改，属后续）

| 发现 | 证据 | 影响 |
|---|---|---|
| **旧 `--dataset` 路径与本数据集 schema 不兼容** | `build_pairs` 需要扁平遥测（`motor_current_a/temperature_c/actual_pos_deg`），climb 数据集是 episode 结构（`observations/actions/instruction/episode_meta`） | 直接把 d4_climb 喂给 `vla_train_full.py --dataset` 会读出 0 条可用记录 |
| **旧路径 val 集形同虚设** | 全局时序 80/20 切分下，val 首样本到 train 最近邻的中位距离 = **5e−5** | 监督信号是"下一帧"，邻帧跨集 ⇒ 泄漏，val loss 无意义。本次改为按 episode 切分（结构性重叠=0） |
| **旧路径"视觉输入"是零张量** | `ReplayDataset.__getitem__` 造 `(3,3,224,224)` 全零 frames | 名义 VLA、实际无视觉信息，且白烧 22M 参数的视觉编码器算力。本次 `frames=None`（模型契约允许） |
| **旧路径语言是 8 个 0** | 同上，`lang = np.zeros(8)` | 语言 embedding 只收到同一 id ⇒ 语言分支等于常数偏置 |
| **坐标被存成 numpy repr 字符串** | `observations[i]["joint_positions"] == '[0.  0.  0.1 0.  ...]'` | JSON 里放数组才对；下游必须写宽容解析器才不炸 |
| **`validate_bc_dataset` 契约过松** | `adapters/nanogpt.py:223+` 只查 actions 长度与有限性，**不查 observations 的数值形状** | 上述字符串编码、维度缺失都能过校验；"validate PASS"不代表可用 |
| **`episode_tactile_stream()` 对本数据集是死函数** | `perception/tactile_closed_loop.py:254` 读 `episode["__records"]`（climb episode 无此键）→ 返回空数组；且内部用 `1A ≈ 0.5N` 这一**无标定换算**（与我从 `FIRST_MOVE.md` 删掉的是同一个） | 谁按名字用它取触觉流，会静默拿到空序列 |

## 两个我自己踩的坑（已修，且已加防回归）

1. **探针把自己做成常量**：`lang_tokens` 首版取"前 8 个字符各一 id"，而 1000 条 instruction
   的前 8 个字符完全相同（`'攀岩竞技: 从 s'`），差异在第 17 位 → lang 恒为同一向量，
   于是"打乱语言 Δ=0.000e+00"是**探针失效**而非数据结论。已改为对全文取摘要，并在 audit
   里加 `unique_lang_token_rows` + **退化为常量直接中止**。
2. **假绿灯判据**：原先用"var > 1e−4 的维数"判动作空间退化，实测 12/12 通过——但那 7 个维
   var=0.001048 只是勉强越过阈值，实为同一常数模板。已改用"不同 std 取值数 ≥ 8"，
   现如实报 ❌（实测 6）。

> 教训与 v1.1 同源：**判据必须能失败**。这两条若没被基线/persistence 对照压出来，
> 就会变成两条绿色的空判据。

## 要让这批数据真正能验 VLA，下一步的顺序

1. **先修 seed → 物理层的传导**（`gen_climb_rollouts.py`：把 seed 注入 VSA 刚度/阻尼、
   hold μ、耦合参数，而不是只注入微扰），并检查 `f_soft_peak_n` 是否被 clip 在 60 N。
2. 修好后重跑本脚本，判据自然翻绿：`physics_degenerate_fields` 应为空、
   `mu_hat` 区间应明显宽于 0.02、语言/触觉消融的 Δ 应可测。
3. 再给 `validate_bc_dataset` 补 observations 的数值形状契约（拒绝字符串坐标、
   校验维度=12/3），把"校验通过"变成有意义的信号。
4. 最后才谈真机：`e2e_real_robot_test.py` 的 5 阶段门已就位，但当前数据连仿真侧的
   触觉/语言变量性都没建立。
