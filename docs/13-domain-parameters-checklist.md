# 领域参数清单 (Domain Parameters Checklist) — Gate 3 准入

> **日期**: 2026-10-08
> **用途**: Gate 3 的核心准入条件是"**领域参数已由专家填入验证**"（见 [docs/12-capability-fit-review.md §五](12-capability-fit-review.md)）。
> 本文档列出全部需要专家确认/填入的参数，按数据来源分三类：
>
> - **[文献]** 可由 AI 从公开文献/手册自动填入，附引用即可（自动验证）
> - **[专家]** 必须由领域专家（机械/控制/摩擦学/机器人）人工确认，AI 只能给出候选值
> - **[实测]** 必须硬件到货后从物理实测取，目前只能占位 `assumed_*`
>
> **验收标准（Gate 3）**: 每条参数都有 provenance 字段（来源/负责人/日期）；`assumed_*` 占比 <50%；专家确认签字（或声明"基于文献，等待实测"）。

---

## 一、接触动力学参数 (L1/L2)

| # | 参数 | 用途 | 当前值 | 类型 | 来源/建议 | Provenance |
|---|---|---|---|---|---|---|
| D1.1 | 攀岩墙摩擦系数 μ_wall | 攀岩场景接触 | 0.8 (假设) | **[专家]** | 岩壁类型（树脂/聚碳酸酯/天然岩）差异 0.5-1.2，需按实际 hold 材质确认 | `wall_material_mu` |
| D1.2 | 指尖胶皮摩擦系数 μ_fingertip | 指尖接触 | 0.9 (假设) | **[专家]** | 软胶 (Shore A 20-40) vs 硬胶 (Shore A 60) 差 2x，需按设计选型 | `fingertip_rubber_mu` |
| D1.3 | Hunt-Crossley 刚度 K_hc | 接触冲击模型 | 50000 N/m (文献) | [文献] | Hertz 接触文献值，对 20 kg 机器人合理量级 | `hc_stiffness_n_per_m` |
| D1.4 | Hunt-Crossley 阻尼 B_hc | 接触冲击模型 | 反解自 e=0.5 | [文献] | 反解公式 `hc_damping_from_restitution` | `hc_damping_n_s_per_m` |
| D1.5 | 指尖软胶厚度 | 柔性体 XPBD | 8 mm | **[专家]** | 设计选型 (GelSight 类 3-10 mm) | `fingertip_pad_thickness_mm` |
| D1.6 | 指尖软胶杨氏模量 | 柔性体 XPBD | 0.5 MPa (文献) | [文献] | 软胶 Shore A 30 约 0.5-1 MPa | `fingertip_youngs_modulus_mpa` |
| D1.7 | ADMM 收敛容差 | 接触优化 | 1e-6 | [文献] | 数值优化标准 | `admm_tolerance` |
| D1.8 | ADMM 最大迭代 | 接触优化 | 200 | [文献] | 数值优化标准 | `admm_max_iter` |

## 二、机器人本体参数 (L7)

| # | 参数 | 用途 | 当前值 | 类型 | 来源/建议 | Provenance |
|---|---|---|---|---|---|---|
| D2.1 | 整机质量 M | 攀岩动力学 | 20 kg | **[专家]** | 设计选型 (攀爬机器人通常 10-30 kg) | `robot_mass_kg` |
| D2.2 | 质心高度 h_com | 稳定性 | 0.6 m | **[专家]** | 结构设计决定 | `com_height_m` |
| D2.3 | 肩/肘连杆长度 | IK | 0.35 / 0.30 m | **[专家]** | 机械图纸 | `upper_arm_length_m`, `forearm_length_m` |
| D2.4 | 大腿/小腿长度 | 双足 LIPM | 0.40 / 0.40 m | **[专家]** | 机械图纸 | `thigh_length_m`, `shin_length_m` |
| D2.5 | 髋间距 | 双足 | 0.18 m | **[专家]** | 机械图纸 | `hip_width_m` |
| D2.6 | 脚掌尺寸 | 双足 | 0.20 × 0.10 m | **[专家]** | 机械图纸 | `foot_length_m`, `foot_width_m` |
| D2.7 | 关节力矩上限 τ_max | 控制 | 30 Nm (肩) / 20 Nm (肘/膝) | **[专家]** | 电机+减速器选型 | `joint_torque_limits_nm` |
| D2.8 | 关节速度上限 ω_max | 控制 | 6 rad/s | **[专家]** | 电机选型 | `joint_speed_limits_rad_s` |
| D2.9 | 关节限位 | IK | ±120° / ±150° / ±90° | **[专家]** | 机械结构决定 | `joint_limits_deg` |

## 三、电机与驱动参数 (L4/B1)

| # | 参数 | 用途 | 当前值 | 类型 | 来源/建议 | Provenance |
|---|---|---|---|---|---|---|
| D3.1 | 腱伺服电机额定力矩 | B1 腱驱动 | 1.4 Nm (从 Hertz 60N 反解) | **[专家]** | 选型 (BLDC 24V / PMSM) | `tendon_motor_rated_torque_nm` |
| D3.2 | 腱伺服电机峰值电流 | B1 | 4 A (设计) | **[专家]** | 电机 datasheet | `tendon_motor_peak_current_a` |
| D3.3 | 腱伺服电机 KV | B1 | 200 KV (假设) | **[专家]** | 电机 datasheet | `tendon_motor_kv_rpm_v` |
| D3.4 | 极对数 pp | FOC | 7 (假设) | **[专家]** | 电机 datasheet | `motor_pole_pairs` |
| D3.5 | 相电感 Ld/Lq | FOC | 15/28 µH (假设) | **[专家]** | 电机 datasheet | `motor_ld_uh`, `motor_lq_uh` |
| D3.6 | 磁链 ψ | FOC | 2.8 mWb (假设) | **[专家]** | 电机 datasheet | `motor_flux_linkage_mwb` |
| D3.7 | 相电阻 Rs | FOC | 50 mΩ (假设) | **[专家]** | 电机 datasheet | `motor_resistance_mohm` |
| D3.8 | 减速器刚度 K_j | VSA | 5000 Nm/rad (文献) | **[专家]** | Harmonic Drive / Planetary 选型 | `reducer_stiffness_nm_per_rad` |
| D3.9 | 腱模量 E_tendon | 腱弹性 | 150 GPa (文献) | [文献] | Dyneema/Steel | `tendon_modulus_gpa` |
| D3.10 | 腱摩擦系数 | 腱模型 (L6) | 0.15 (文献) | [文献] | UHMWPE vs 铝 pulley | `tendon_friction_coefficient` |
| D3.11 | Pulley 路由半径 | 腱模型 (L6) | 10 mm | **[专家]** | 机械图纸 | `pulley_radius_mm` |

## 四、触觉传感器参数 (L4)

| # | 参数 | 用途 | 当前值 | 类型 | 来源/建议 | Provenance |
|---|---|---|---|---|---|---|
| D4.1 | 触觉帧率 | 控制 | 100 Hz (契约) | [专家] | GelSight 类 30-90 Hz，电容阵列可达 1 kHz | `tactile_frame_rate_hz` |
| D4.2 | 触觉阵列分辨率 | 感知 | 8×8 (假设) | **[专家]** | 传感器选型 | `tactile_resolution` |
| D4.3 | 触觉测量噪声 σ | EKF R 矩阵 | 0.5 N (假设) | **[实测]** | 传感器 datasheet + 真机标定 | `tactile_noise_std_n` |
| D4.4 | 滑脱检测阈值 | SlipDetector | 0.7 | **[专家]** | 经验值 (文献 0.6-0.8) | `slip_detection_threshold` |
| D4.5 | EKF R_tac | 触觉观测噪声 | 0.25 | [文献] | 初始值，需真机标定 | `ekf_r_tac` |
| D4.6 | EKF R_vis_mu | 视觉 μ 观测噪声 | 0.02 | [文献] | 初始值，需真机标定 | `ekf_r_vis_mu` |
| D4.7 | PI 增益 kp / ki | GripForcePID | 1.0 / 0.1 (默认) | **[实测]** | 1050 eps 训练后替换 (L4尾) | `grip_pid_kp`, `grip_pid_ki` |
| D4.8 | 滑脱概率目标 δ | 机会约束 | 0.05 (家用) / 0.001 (攀岩) | **[专家]** | 安全等级决定 | `grip_slip_probability_target` |

## 五、攀岩场景参数 (L1)

| # | 参数 | 用途 | 当前值 | 类型 | 来源/建议 | Provenance |
|---|---|---|---|---|---|---|
| D5.1 | 墙面倾角 | 攀岩 | -10° (A3) | **[专家]** | 场景定义 (仰角/垂直/屋檐) | `wall_overhang_deg` |
| D5.2 | 岩点间距分布 | MJCF 场景 | 均匀 0.3-0.6 m (假设) | **[专家]** | 竞赛赛道规则 / 自定义 | `hold_spacing_range_m` |
| D5.3 | 岩点尺寸分布 | MJCF 场景 | 30-100 mm | **[专家]** | 竞赛赛道规则 | `hold_size_range_mm` |
| D5.4 | 岩点法向分布 | MJCF 场景 | 半球均匀 (假设) | **[专家]** | 场景定义 | `hold_normal_distribution` |
| D5.5 | dyno 距离 | 场景 | 0.6-1.2 m (A2 实测) | **[专家]** | 场景定义 | `dyno_distance_m` |
| D5.6 | 4→3→4 换脚序列时间 | 场景 | A3 实测 77.3N 峰值 | **[专家]** | 场景定义 | `foot_transition_time_s` |
| D5.7 | 绳索张力 (安全绳) | 接触 | 50 N (假设) | **[专家]** | 场景定义 (是否带保护) | `safety_rope_tension_n` |

## 六、VSA 参数 (L2/L6)

| # | 参数 | 用途 | 当前值 | 类型 | 来源/建议 | Provenance |
|---|---|---|---|---|---|---|
| D6.1 | ANTAGONISTIC kMin/kMax | VSA 静力学 | 60-520 Nm/rad (B3 实测) | **[专家]** | 选型 | `vsa_k_min_nm_per_rad`, `vsa_k_max_nm_per_rad` |
| D6.2 | 弹性行程 | VSA | 12° | **[专家]** | 选型 | `vsa_elastic_travel_deg` |
| D6.3 | 腱连杆长度 | VSA | 0.15 m | **[专家]** | 机械图纸 | `vsa_link_length_m` |
| D6.4 | 腱质量 | VSA | 0.2 kg | **[专家]** | 机械图纸 | `vsa_mass_kg` |
| D6.5 | 指尖 Hertz 极限 | 安全 | 60 N (A4 实测) | **[专家]** | 指尖材料 + 几何 | `fingertip_hertz_limit_n` |

## 七、安全与认证参数 (L5)

| # | 参数 | 用途 | 当前值 | 类型 | 来源/建议 | Provenance |
|---|---|---|---|---|---|---|
| D7.1 | 硬件关断 <1µs | STO | assumed 0.8 µs | **[实测]** | Phase P1 示波器 | `hardware_shutdown_us` |
| D7.2 | 电弧时间 | STO | assumed 8 ms | **[实测]** | Phase P1 示波器 | `arc_quench_ms` |
| D7.3 | 机械隔离时间 | STO | assumed 8 ms | **[实测]** | Phase P1 示波器 | `mech_isolation_ms` |
| D7.4 | PFH | 安全 | assumed 6.6e-8 | **[实测]** | Phase P3 认证 | `pfh_per_hour` |
| D7.5 | SFF | 安全 | 96.9% (文献) | [文献] | 元件 MTBF 手册 | `safe_failure_fraction` |
| D7.6 | 过温关断温度 | 安全 | 105°C | **[专家]** | MOSFET Tj max 降额 | `overtemp_shutdown_c` |
| D7.7 | 过流关断阈值 | 安全 | 40 A (B1) | **[专家]** | MOSFET 降额 + 熔断 | `overcurrent_threshold_a` |
| D7.8 | 通信丢失看门狗 | 安全 | 50 ms | **[专家]** | CAN-FD 通信周期决定 | `comm_loss_watchdog_ms` |

## 八、AI/训练参数 (L5)

| # | 参数 | 用途 | 当前值 | 类型 | 来源/建议 | Provenance |
|---|---|---|---|---|---|---|
| D8.1 | D4 episode 长度 | 训练 | 20-60 步 (1050 eps) | [文献] | 已固化 | `d4_episode_length_steps` |
| D8.2 | RL 折扣 γ | PPO/SAC | 0.99 | [文献] | 标准 | `rl_gamma` |
| D8.3 | PPO clip ε | PPO | 0.2 | [文献] | 标准 | `ppo_clip_epsilon` |
| D8.4 | PPO GAE λ | PPO | 0.95 | [文献] | 标准 | `ppo_gae_lambda` |
| D8.5 | SAC α 初始 | SAC | auto | [文献] | 自动温度 | `sac_alpha_init` |
| D8.6 | Replay 容量 | SAC | 100k | [文献] | 标准 | `sac_replay_capacity` |
| D8.7 | target_kl | PPO | 0.02 (死配置) | [文献] | 标准 | `ppo_target_kl` |
| D8.8 | STL time window | STL | 0.5 s | [文献] | 标准 | `stl_time_window_s` |

## 九、硬件平台参数 (Phase P)

| # | 参数 | 用途 | 当前值 | 类型 | 来源/建议 | Provenance |
|---|---|---|---|---|---|---|
| D9.1 | Jetson Orin NX TOPS | 算力 | 100 (INT8) | [文献] | NVIDIA 手册 | `jetson_orin_nx_tops` |
| D9.2 | Jetson Orin NX TDP | 功耗 | 25 W | [文献] | NVIDIA 手册 | `jetson_orin_nx_tdp_w` |
| D9.3 | CAN-FD 数据段速率 | 通信 | 5 Mbps | [专家] | 总线设计 | `canfd_data_bps` |
| D9.4 | CAN-FD 仲裁段速率 | 通信 | 1 Mbps | [专家] | 总线设计 | `canfd_arb_bps` |
| D9.5 | 电池容量 | 续航 | 48 V / 320 Wh | **[专家]** | 电池选型 (21700 × N) | `battery_capacity_wh` |
| D9.6 | 电池内阻 | BMS | 30 mΩ (假设) | **[实测]** | 电池 datasheet + 实测 | `battery_internal_r_mohm` |
| D9.7 | VLA 推理延迟 | 性能 | assumed 20 ms | **[实测]** | Phase P8 实测 | `vla_inference_ms` |
| D9.8 | 控制环 jitter | 性能 | assumed <100 µs | **[实测]** | Phase P8 实测 | `control_loop_jitter_us` |

---

## 十、Provenance 字段 schema (建议)

```yaml
parameter_id: D1.1          # 本清单 ID
symbol: wall_material_mu    # 代码/契约中的字段名
value: 0.8
unit: dimensionless
source: literature          # literature | expert | measured | assumed
source_ref: "ISO ..."       # 文献引用 / 专家姓名 / 测量设备+编号
confidence: high            # high | medium | low
validation_status: validated | provisional
expert_sign_off: ""         # 专家签字 (必填, Gate 3 准入)
date: 2026-10-08
notes: ""
```

---

## 十一、填入流程

1. **AI 预填**: 按本清单扫描代码/契约/文献，自动填 `[文献]` 类参数 + 附引用；给 `[专家]` 类参数填候选值 + 说明
2. **专家审阅**: 逐条审阅 `[专家]` 类参数，签字确认或修改
3. **落盘**: 填入 `contracts/hw_requirements.yaml` 或 `contracts/safety_budget.yaml`（按语义归档）；本清单记录 provenance
4. **验证**: `verify` 新增 `domain_params_provenance` 项（每条 `[专家]` 参数必须填 `expert_sign_off` 字段才算通过）
5. **Gate 3 达成**: 所有 `[专家]` 参数有 `expert_sign_off`，所有 `[实测]` 参数改为 `assumed_*` 等待 Phase P

---

## 十二、总结

**参数总数**: 61 条 (D1.1 - D9.8)

| 类型 | 数量 | 状态 | 说明 |
|---|---|---|---|
| [文献] | 22 | AI 可自动填入 | 只需引用 |
| [专家] | 27 | **必须专家参与** | Gate 3 主要外部阻塞 |
| [实测] | 12 | 必须硬件到货 | Phase P 前全部 `assumed_*` 占位 |

**关键结论**: Gate 3 的**软件侧工作 AI 可独立完成**（Sprint L 里 L1-L8），但 **L-Domain 领域参数专家验证必须由领域专家参与**——本清单即是为这个外部阻塞做好**全部准备**：把"专家填入"变成"读这份清单 + 签字"的最小动作。

**一旦专家可用**: 按第 11 节流程操作，预计 1-2 天内完成 61 条参数的填入与验证。
