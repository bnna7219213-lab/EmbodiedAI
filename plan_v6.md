# EmbodiedAI 计划 v6 (Plan v6) — Sprint N：中枢决策层(C1-C5) / OS SSH 通道 / GUI×触觉×化学生化 / V2L 空间智能

> **日期**: 2026-10-17
> **前生**: [plan.md](plan.md) v1 → [plan_v2.md](plan_v2.md) → [plan_v3.md](plan_v3.md) → [plan_v4.md](plan_v4.md) → [plan_v5.md](plan_v5.md)
> **本版定位**: v5 (Sprint M) 补齐了机器人端四大缺口（多传感器信道/VLA 双路径/SLAM/认知中枢），
> 但用户指出"**还有某些部分不全，整条链路串连有缺点**"。本版围绕五点补齐：
>
> 1. **架构补充**：7 层控制栈之上增加**中枢决策层**（nanoGPT / VLA / 记忆 / 感知 / 信号处理）→ 新增 **C 栈 C1-C5**（L1-L7 编号不动，docs/09 硬约束）
> 2. **机器人 Linux OS SSH**：Linux on ARM（Jetson/WSL）作 OS，SSH 远程 debug / 编写出厂初始程序。**双通道语义**：RUNTIME=实时协作规划（非遥控），MAINTENANCE=停机维护（修 bug/装程序）
> 3. **GUI × 触觉反馈**：触觉占比大，自我进化需触觉 + **化学生化信号仿真传感融合**
> 4. **V2L 空间智能**：动作规划与可解释性由 Video-to-Language 空间智能决定（想象空间思考: 辅助线→演算→结论）
> 5. **链路串联修复**：含 CHANGELOG B8 实锤的 `POSE_UPDATE` 零发布/零订阅断链

---

## 一、实测基线锚点 (2026-10-17, Sprint N 完成后)

```
verify --no-pytest:   42/42 PASS, RESULT=VERIFY_OK   (v5 为 37 项, 本轮 +5: N1-N5)
pytest:               (见 README 实测行)             (新增约 75 用例)
新增源码:             src/embodied_ai/center/layer_registry.py   (N1 层注册表单一真源)
                      src/embodied_ai/center/center_brain.py     (N2 中枢编排 facade)
                      src/embodied_ai/center/spatial_reasoning.py (N5 V2L 空间论证)
                      src/embodied_ai/os_channel/ssh_admin.py    (N3 OS SSH 双通道)
                      src/embodied_ai/perception/chem_bio.py     (N4 化学生化融合)
                      contracts/os_channel.yaml                  (N3 通道契约)
新增测试:             tests/test_layer_registry.py (11) / test_center_brain.py (11)
                      tests/test_os_ssh.py (17) / test_chem_bio.py (18)
                      tests/test_spatial_reasoning.py (9)
新 CLI (组 OS/中枢):  os-status / os-provision / center-demo / v2l-demo / chem-demo
新 GUI routes:        /api/layer_stack /api/tactile /api/chem
```

### 基线对照 (v5→v6)

| 版本 | verify (no-pytest) | 关键交付 | 链路完整性 |
|---|---|---|---|
| v5 plan_v5.md | 37 项 PASS | Sprint M: 信道/V2A/SLAM/认知中枢 | 模块齐, **串联有断点** (B8) |
| **v6 plan_v6.md** | **42 项 PASS** | **Sprint N: C 栈 + OS SSH + GUI×触觉×化学生化 + V2L** | **全链闭合** (POSE_UPDATE 修复) |

---

## 二、Sprint N 任务表

| ID | 任务 | 交付 | 验收 DoD | 量级 | 阻塞 |
|---|---|---|---|---|---|
| **N1** | 层注册表（单一真源） | `center/layer_registry.py`：`LayerSpec` + `LAYER_REGISTRY`（L1-L7 照旧 + P1-P5 + S + D1-D5 + **C1-C5**）+ `layers_by_stack()/as_dict()/validate_registry()`；`gui/server.py:_layer_stack()` 改读注册表 | C1-C5 齐全（nanoGPT/VLA/记忆/感知/分频覆盖）；L1-L7 编号无增删；module_refs 全可 import；GUI 与注册表同源；`test_layer_registry.py` 11 项 | 1d | 无 |
| **N2** | 中枢编排 CenterBrain | `center/center_brain.py`：`CenterBrain.step()` 固定链路（分频→触觉+EKF+化学生化→**POSE_UPDATE 发布（修复 B8）**→记忆→V2L 论证→VLA 双路径决策→act_fn→四要素事件 source="center"）；`run_demo()` | POSE_UPDATE 真发布且 `hub.latest_pose()` 收到；双路径 v2a/vla 正确；四要素齐全；`test_center_brain.py` 11 项 | 1d | N1 |
| **N3** | OS SSH 双通道 | `os_channel/ssh_admin.py`：`ChannelKind`(RUNTIME 协作规划 / MAINTENANCE 停机维护，**非遥控**) + `ModeController` 状态机（RUNTIME→PAUSED→MAINTENANCE→PROVISIONED→RUNTIME，仅 RUNTIME 允许运动）+ `SshCommandBuilder`(ssh/scp/出厂脚本) + `DryRunExecutor`/`SubprocessExecutor` + `OsChannelAdmin`(enter_maintenance/pull_debug_logs/push_program/provision_factory_program/return_to_runtime)；`contracts/os_channel.yaml` | 状态机全转换表（合法+非法边）；MAINTENANCE 拒收运动指令；dry-run 命令含 ssh/scp token；provisioning 序列完整；未进维护态 push/pull 报错；`test_os_ssh.py` 17 项 | 2d | 无 |
| **N4** | GUI×触觉×化学生化 | `perception/chem_bio.py`：`ChemBioSensor`(e-nose 8 通道/pH/湿度/温度, synthetic) + `ChemTactileFusion.fuse()`(材质/表面状态估计+证据链)；GUI routes `/api/layer_stack` `/api/tactile` `/api/chem` | 三场景 dry/wet/corrosive 判定正确；融合可复现；输入变则估计变；GUI routes 键齐全；`test_chem_bio.py` 18 项 | 1d | 无 |
| **N5** | V2L 空间智能 | `center/spatial_reasoning.py`：`SpatialReasoner.argue()`→`SpatialArgument`(辅助线 hypotheses / 演算 derivation_steps / 结论 / 语言摘要 / 决策向量)；`to_event(source="v2l")` 喂认知中枢 | 辅助线≥3、推演≥3；结论与决策一致；输入变则论证变（防恒定假论证）；`test_spatial_reasoning.py` 9 项 | 1d | 无 |
| **N6** | 串联 + CLI/verify/文档 | CLI 组 `OS/中枢`（os-status/os-provision/center-demo/v2l-demo/chem-demo）；verify +5 项（layer_registry/center_brain/os_channel/chem_bio/spatial_reasoning），CHECKS 37→**42**；`plan_v6.md`；README/CHANGELOG 口径同步 | verify **42/42**；pytest 全绿；5 demo 可运行；README 项数==len(CHECKS)（test_version_consistency 守护） | 1d | N1-N5 |

**Sprint N 准出声明**: "架构层注册表收口（L+C 栈单一真源）；中枢编排端到端闭环且 POSE_UPDATE 断链修复；OS SSH 双通道（维护态禁运动）；触觉+化学生化融合入 GUI；V2L 空间论证可解释；verify 42/42 + pytest 全绿 + 5 demo 可运行。"

---

## 三、C 栈中枢决策层（C1-C5）

> 用户口径：中枢决策层含 **nanoGPT、VLA、记忆、感知、信号处理**。保持 L1-L7 编号不变，新增 C 栈。

| 层 | 名称 | 构成 | 模块 |
|---|---|---|---|
| **C1** | 信号处理 | 多传感器分频信道（视频信号流 + 微波等组合信号解复用） | `perception.sensor_fusion` |
| **C2** | 感知 | 触觉-视觉多速率 EKF + SLAM 三维坐标/时间 + 触觉闭环 | `perception.multirate_ekf` / `slam` / `tactile_closed_loop` |
| **C3** | 记忆 | 四要素情景记忆（时间/地点/人物/事情）+ 记忆召回 | `cognition.cognitive_hub` |
| **C4** | 决策 | nanoGPT + VLA 双路径（v+l→action / v→action） | `adapters.nanogpt` / `models.vla_v2a` / `vla_model` |
| **C5** | 中枢编排 | 端到端链路编排（信号→感知→记忆→论证→决策→动作→事件） | `center.center_brain` / `cognition.closed_loop` |

---

## 四、断链修复清单（对应"整条链路串连有缺点"）

| # | 断链 | 修复 | 守护 |
|---|---|---|---|
| 1 | C1-C5 与 L1-L7 映射只在文档、GUI 硬编码 | N1 层注册表收口，GUI 改读注册表 | `test_layer_registry.py` + `check_layer_registry` |
| 2 | **POSE_UPDATE 零发布/零订阅**（CHANGELOG B8 实锤）——"SLAM→认知中枢地点要素"只存在于 docstring | N2 `CenterBrain.publish_pose()` 唯一发布口，hub 订阅 `latest_pose()` | `test_center_brain.py` + `check_center_brain` |
| 3 | 触觉/化学生化 → 决策输入无通道 | N4 `ChemTactileFusion` → `CenterBrain.step` | `test_chem_bio.py` + `check_chem_bio` |
| 4 | V2L 论证与 VLA 决策各自为战 | N5 `SpatialReasoner.to_event(source="v2l")` 落库 + N2 编排串入决策前 | `test_spatial_reasoning.py` + `check_spatial_reasoning` |
| 5 | 全链无 smoke | N2 `run_demo()` + `check_center_brain` | verify 项 |

---

## 五、OS SSH 双通道语义（核心区分）

| 通道 | 语义 | 运动指令 | 用途 |
|---|---|---|---|
| **RUNTIME** | 实时 runtime 协作规划（人机互相配合完成下一步动作） | ✅ 允许 | 遥控协作（非单向遥控命令） |
| **MAINTENANCE** | SSH 停机维护 | ❌ **禁止** | 修历史 bug / 新增功能程序 / 拉日志 / 出厂烧录 |

模式状态机：`RUNTIME → PAUSED → MAINTENANCE → PROVISIONED → RUNTIME`
**铁律**：仅 RUNTIME 允许运动指令（`ModeController.can_send_motion()`，测试钉死）。

---

## 六、provenance 标注（诚实边界）

| 参数/实现 | provenance | 说明 |
|---|---|---|
| 化学生化传感响应（e-nose/pH/湿度） | **synthetic** | 合成仿真；真实化学传感标定=待领域验证 |
| 融合阈值（WET_THRESHOLD/CORROSIVE_PH/SLIP_TACTILE_STD） | **assumed** | 规则参数，待领域验证 |
| OS 契约 host/board 名 | **assumed** | 示例值，真机部署待实测 |
| 出厂脚本内容 | **待领域验证** | systemd/cognition 路径为模板 |
| V2L 论证 caption | **模板化** | 规则式几何推理，非语言模型输出 |
| CenterBrain | 纯编排 facade | 不重复实现既有模块（注册表 module_refs 为证） |

---

## 七、风险与应对

| 风险 | 应对 |
|---|---|
| verify/README 数字口径漂移（历史最高频事故） | N6 DoD 强制同步；test_version_consistency 自动守护 README 项数==len(CHECKS) |
| 合成化学生化/传感被误读为硬件标定 | docstring + plan_v6 provenance 表强制标注 |
| 状态机转换表遗漏边角 | 全转换表测试（每条合法边 + 每条非法边） |
| CenterBrain 变成重复实现 | 约束为纯编排 facade，能力一律引用既有模块 |
| SubprocessExecutor 真跑 ssh 无真机 | 默认 DryRunExecutor（测试/CI）；真机注入 SubprocessExecutor |

---

## 八、本计划的"不做的事"

- 不改 L1-L7 控制栈编号语义（docs/09 硬约束）
- 不做 paramiko 真实 SSH 连接管理（系统 ssh/scp 足够，零第三方依赖）
- 不做遥控遥操作（RUNTIME 通道是协作规划，不是遥控命令下发）
- 不做真实化学传感器驱动/标定（Phase P 硬件到货后）
- 不做语言模型 caption 生成（V2L caption 模板化，大模型属后续）
- 不做动态插件层发现（层集合是架构约定，非运行时扩展点）
