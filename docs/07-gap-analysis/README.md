# 差距分析文档索引

本目录包含从不同维度对 EmbodiedAI 项目进行的差距分析，定位当前技术水平与目标之间的距离。

## 文档清单

| 文档 | 分析维度 | 核心结论 |
|---|---|---|
| [gap-analysis-algorithm-architecture.md](gap-analysis-algorithm-architecture.md) | 算法架构 | 单一算法无法满足20Hz实时+泛化+安全，需分层混合架构 |
| [gap-analysis-contact-planning.md](gap-analysis-contact-planning.md) | 接触规划 | 攀岩MCTS迁移到家庭场景需要连续化扩展 |
| [gap-analysis-data-strategy.md](gap-analysis-data-strategy.md) | 数据策略 | 需在仿真与真机数据之间动态配比 |
| [gap-analysis-dynamics-hardware.md](gap-analysis-dynamics-hardware.md) | 刚柔耦合+DIY硬件 | 柔性体建模和触觉DIY是家庭场景核心瓶颈 |
| [gap-analysis-realtime-tactile.md](gap-analysis-realtime-tactile.md) | 实时触觉 | Jetson Orin NX 上可行的融合网络延迟分析 |
| [gap-analysis-sim2real.md](gap-analysis-sim2real.md) | Sim-to-Real | 触觉域差距比视觉大一个数量级 |
| [project-gap-comprehensive-report.md](project-gap-comprehensive-report.md) | **综合报告** | **全项目 5 大缺失环节 + G1-G5 定位 + 补充方案** |

## 综合报告阅读顺序

建议按以下顺序阅读：

1. 先读 [project-gap-comprehensive-report.md](project-gap-comprehensive-report.md) 了解全局
2. 根据感兴趣的维度读具体文档，例如关心触觉 → gap-analysis-realtime-tactile.md
3. 实现层面问题参考 gap-analysis-dynamics-hardware.md 的 URDF/TACTO 实操

## 缺失环节速查

```
G1  浮基/质心动力学        → 已补齐 → docs/02-contact-dynamics/climbing-dynamics-math.md §1
G2  冲击/脉冲动力学        → 已补齐 → docs/02-contact-dynamics/climbing-dynamics-math.md §2
G3  竞技攀岩运动数学       → 已补齐 → docs/02-contact-dynamics/climbing-dynamics-math.md §3
G4  触觉-视觉融合概率基础   → 已补齐 → docs/02-contact-dynamics/climbing-dynamics-math.md §4
G5  滑脱检测动力学升级     → 已补齐 → docs/02-contact-dynamics/climbing-dynamics-math.md §4
G6  柔性体动力学          → 未补齐 → 未来探索计划 P0-A
```
