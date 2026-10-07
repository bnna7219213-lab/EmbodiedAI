# EmbodiedAI × EdaLab — L5 落地计划（v4：jar 入库 + 消费模式，**已执行**）

**版本**: plan_L5.md v4 (2026-10-07) — 用户决策 `jar_in_repo` 后的最终方案
**决策依据**: 实测发现 Python 侧缺 MNA/PCB/Gerber/FEM/热仿真（全在 Kotlin 里），而 G1/G9/G11 的 Python 实现反强于 Kotlin 查表版 → **不重写引擎，把编译产物入库消费**

---

## 零、三次方案演进（结论：v4 最省且满足 L5）

| 版本 | 思路 | 问题 |
|---|---|---|
| v1 | 嵌 16 引擎 | 丢掉 15K 行已交付能力 |
| v2 | 全量嵌 101 Kotlin 文件 | 把形式当内容（搬代码≠拿能力），双语言维护 |
| v3 | 保留能力、用 Python 重实现（17d） | 未论证"为何不直接用 jar"；重写 16,780 行能力收益/成本差 |
| **v4 ✅** | **jar 入库 + 消费** | **已实测落地**（见 §一） |

**关键事实（实测）**：
- Python 侧**真正缺**的 EDA 能力：`netlist`(0 文件) / `schematic`(0) / `router`(0) / `gerber`(0 真实现) / `mna`(0，2 处命中是 "Gymnasium" 误匹配) / `thermal`(0 真实现) / `fem`(0) —— **全在 Kotlin**
- Python 侧**已有且更强**：G1 PFH/SFF（`certification.py` 真公式 vs Kotlin `calculateSff` 硬编码查表）、G11 DR YAML（`bigdata.py`）、G9 STL 消费（`safety/formal/`）
- jar 体积小（core 1.95MB/681 类 + stdlib 1.76MB/974 类 = **3.6MB**）

---

## 一、v4 已执行的改造（本轮实测证据）

### 1.1 jar 入库

```
vendor/edalab/
├── edalabcore.jar           1,954,103 B  (681 类, 含全部 101 Kotlin 文件编译产物)
└── kotlin-stdlib-2.2.21.jar 1,761,445 B  (974 类, Kotlin 运行时, jar 不自含)
```
- 实测：**仅这两件**即可运行（`vendor_only.bat` 验证 → relay-clutch 输出完整 16.00ms 安全链）
- `.gitignore` 加 `!vendor/edalab/*.jar` 白名单（防误忽略）

### 1.2 路径解析改造（`adapters/edalab.py`）

```python
VENDOR_DIR = <repo>/vendor/edalab        # ① 入库 jar (单仓自足)
EDALAB_ROOT = C:\...\EdaLab              # ② 外部回退 (开发机)
gradle cache                             # ③ stdlib 兜底

resolve_jar_paths() → {"core", "stdlib", "source"}   # source: vendor|external|missing
build_classpath()   → "vendor_core;vendor_stdlib;[jdk8];[jdk7]"  (路径统一正斜杠)
```
- `EDALAB_HOME` 环境变量可覆盖外部回退目录（供 CI 显式验证"无外部依赖"）
- `_run_live()` 改用 `build_classpath()`，结果记 `jar_source` 字段（可观测）

### 1.3 audit 命令改造（`cli.py`）

探针分两类：
- `own_probes`：本仓探针（`vendor/edalab/*.jar`，相对仓库根）
- `probes`：外部探针（相对项目 root）

实测输出：
```
[OK     ] EdaLab        怎么造硬件 — 36 引擎 + 12 步主链 (Kotlin; vendor jar 已入库)
           root=C:\Users\bnna7\workspace\EdaLab
           本仓探针 2/2 (vendor 入库)
           Kotlin 引擎文件 (...): 37
           注: v0.8: 能力经 vendor/edalab jar 消费 (单仓自足, 仅需 JRE)
```

### 1.4 验证证据（全部实测）

| 验证 | 结果 |
|---|---|
| 仅 vendor 两件跑引擎 | ✅ `vendor_only.bat` → relay-clutch 完整输出（16.00ms PASS） |
| adapter live 调用 | ✅ `LIVE_OK`, `jar_source: vendor`, 含 `16.00ms` |
| **隔离验证**（EDALAB_HOME→不存在路径） | ✅ 仍 `LIVE_OK`, `jar_source: vendor` —— **单仓自足的决定性证据** |
| audit | ✅ 五项目 ALL OK（vendor 2/2） |
| verify | ✅ **25/25 PASS** |
| pytest 全量 | ✅ **316 passed** + 55 subtests |
| 新增回归测试 | ✅ `tests/test_vendored_jars.py` 4/4（含隔离验证固化为 CI 守护） |

---

## 二、L5 判据（v4，与实际能力对齐）

| Gate | 判据 | 验收命令 | 状态 |
|---|---|---|---|
| **G1 单仓可跑** | clone 后仅需 JRE，无需 gradle/JDK/外部 EdaLab | `EDALAB_HOME=/nonexistent python -m embodied_ai.cli audit` | ✅ 已达成 |
| **G2 EDA 能力可用** | 12 步主链经 vendor jar 真实执行 | `python -m embodied_ai.cli couple --live` | ✅ 已达成 |
| **G3 安全闭环** | PFH/SFF + STL 消费 + 预算闭环 | `verify` 的 safety/stl/budget checks | ✅ 已达成 |
| **G4 飞轮闭合** | ICT/FCT → DR → RL 训练 | `python scripts/flywheel_live_train.py` | ✅ 已达成 |
| **G5 硬件产出** | 输出原理图+PCB+BOM 文件包（工厂可打样，**不含实物到货**） | `python -m embodied_ai.cli couple --live`（motor-drive 步骤） | ✅ 能力就绪 |
| **G6 缺口透明** | 未修的 Kotlin 缺口（G2/G3/G5/G6/G7/G10）如实记录，不假装修复 | 本文档 §三 | ✅ 已记录 |

**关于"硬件"（v4 修正 v2 的错误框架）**：平台定位是**产出制造文件包**（原理图/PCB/BOM → 工厂打样），**"板卡到货"不是 L5 前置条件**。L5 止于 G5。

---

## 三、已知缺口（诚实记录，未修）

以下 Kotlin 侧缺口在 v4 中**保持原状**（因 jar 入库=冻结编译产物，不改源码）：

| 缺口 | 内容 | v4 处置 | 后续可选 |
|---|---|---|---|
| G2 | MotorDrive ↔ FOC 参数未对齐 | 保持 | 需改 Kotlin 源码（超出 jar 入库范围） |
| G3 | MultiPhysics → MotorDrive 字段缺失 | 保持 | 同上 |
| G5 | CLI 主链为"烟囱式"（各步用样本输入，非数据管道） | 保持 | 同上 |
| G6 | HILBridge 用随机数（未接 MNA + CAN-FD） | 保持 | 同上 |
| G7 | DigitalTwin 阈值告警（无 FFT/判别器） | 保持 | 同上 |
| G10 | MotorDriveSpec 缺 Ke/flux/Ld/Lq/Rs | 保持 | 同上 |

**若需修这些缺口**：两条路 ① 在**新仓内**重建 Kotlin 源码并编译（不碰原 EdaLab），走 v2/v3 路径；② 在 Python 侧实现对应能力（如 G1 已在 Python 侧更强）。

---

## 四、目录结构（v4 实际）

```
EmbodiedAI/
├── vendor/edalab/                  ← ★ 新增：入库 jar（单仓自足的载体，3.6MB）
│   ├── edalabcore.jar              (681 类 / 101 Kotlin 文件的编译产物)
│   └── kotlin-stdlib-2.2.21.jar    (974 类 / Kotlin 运行时)
├── src/embodied_ai/
│   ├── adapters/edalab.py          ← 改：vendor 优先解析 + EDALAB_HOME + jar_source
│   ├── cli.py                      ← 改：audit 支持本仓/外部双类探针
│   └── ...（其余不动）
├── tests/test_vendored_jars.py     ← 新增：4 项守护（含隔离验证）
├── contracts/ · artifacts/ · docs/
└── （无 edalabcore/ 源码目录 —— v4 不迁源码）
```

---

## 五、遗留与边界

**已解决**：单仓自足（仅需 JRE）、全部 EDA 能力可用（MNA/PCB/Gerber/FEM/热/HDL/版图等均在 jar 内）、零重写成本

**边界（诚实声明）**：
1. **需 JRE**（不是纯 Python 运行）——如需完全无 JVM，走 v3 的 Python 重实现路径
2. **Kotlin 缺口不可修**（jar 冻结）——6 项 G# 缺口如实保留
3. **jar 与源码漂移 — ✅ 已治理** (2026-10-07)：新增 `vendor/edalab/PROVENANCE.json`
   记录两件 jar 的 sha256 / size / 来源 / 构建命令 / 覆盖 stage。治理分两层：
   - **L1 指纹**：jar 被替换或重新构建 → sha256 变化 → 立刻暴露。
     守护 = `verify` 项 `vendor_jar_provenance` + `tests/test_vendor_provenance.py`
     （9 项，含 4 项反证：篡改 hash / size 不符 / 声明缺失 jar / 元数据缺失，均须 FAIL）。
   - **L2 行为锚定**：源码演进导致引擎输出变化（jar 该更新了，L1 抓不到这种情形）。
     契约锚定值 vs 引擎实测 —— `vsa_antagonistic_peak_n=166.39`、relay-clutch 安全链
     16.00ms、forge-core EPS ID。守护 = `test_b3_vsa_antagonistic.py`
     / `test_b2_stage_pipeline.py`。
   - 更新流程见 `PROVENANCE.json` 的 `update_procedure`，README 有专章。
   - **先例**：B3 发现契约 109.0 落后于引擎实测 166.39（V-2 修复前口径）—— 正是
     L2 层要抓的漂移类型。
4. **vendor 二进制入库**（3.6MB）——git 仓库体积增加，但换来 clone 即用

**下一步可选**：
- A. 若需要修 G2/G3/G5/G6/G7/G10 → 在新仓重建 Kotlin 源码（v2/v3 路径）
- B. 若需要无 JVM → 按实际痛点逐块 Python 重实现（MNA 优先，用 Kotlin 作 oracle 对照）
- C. 维持 v4 → 继续用 vendor 消费模式推进 Phase B（腱驱动关节板 EPS、Stage 化）— **✅ 已执行（2026-10-07）**

### C 已执行结果：B2 P1.4 Stage 化收官（消费侧）

| 项 | 内容 |
|---|---|
| B1 腱驱动关节板 EPS | （v4 前已 ✅ 完成）EPS `SF-climbing_tendon_drive-climbing_competition_v0_1` |
| **B2 Stage 化（本轮）** | `STAGE_REGISTRY` 12 个 Stage（11 主链 + `vsa-static`）+ `run_eps_job` 数据管道 |
| 关键设计 | `FLOW_CHAIN` **保持 11 步不动**（既有 331 项测试与 `chain_green(expect_steps=11)` 依赖它）；编排基于**超集** `STAGE_REGISTRY`，零破坏 |
| 为什么在消费侧 | G5「主链烟囱式」缺口在 Kotlin 侧，vendor jar 冻结改不了源码 → Python 侧把 stage 串成管道，上游产物进 `context` |
| 实测验收 | `couple --eps-job` → `steps=2/2 ok=YES`，EPS ID + `vsa_curve_v1_1`(3 拓扑)，`jar_source=vendor` |
| 回归 | pytest **331 passed**（+11 B2 测试）、verify **26/26 PASS** |

**B3 已一并完成（同轮）**：VSA-ANTAGONISTIC 选型落地，`run_eps_job(["relay_clutch","vsa_static"])` 联合 PASS。

| 项 | 内容 |
|---|---|
| ★ 实质修正 | 契约 `vsa_antagonistic_peak_n` **109.0 → 166.39**（引擎实测 166.39064；109 是 V-2 修复前过期口径，docs/04 该表行 stored 疑混填 SEA 值） |
| 选型结论 | **不变** —— ANTAGONISTIC 仍是三拓扑峰值最小者（166 < SEA 347 < MACCEPA 525），最接近指尖 Hertz 极限 60N |
| 联合五判据 | 契约=引擎 / 峰值最小 / 离合器承载 3.66Nm ≤ 12.0Nm / 安全链 16.00ms<20ms / verdict=PASS |
| 显式化 | 力臂 `fingertip_lever_arm_m: 0.022`（原散见 `servo_peak_torque_nm` 注释） |
| 回归 | pytest **339 passed**（+8 B3）、verify **26/26 PASS** |

**Phase B 现状**：B1 ✅ / B2 ✅ / B3 ✅ 全部完成。

---

**文档位置**: `plan_L5.md` v4
**前置**: `plan_v3.md`(Phase A)、K1-K3 交付、`docs/10-climb-soft-robot/`
**决策记录**: 用户选择 `jar_in_repo`（超时默认项）
**本轮已验证**: vendor 入库 ✅ / 隔离验证 ✅ / verify 25/25 ✅ / pytest 316 ✅
