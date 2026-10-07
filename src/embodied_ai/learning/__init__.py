"""终身学习 / 持续学习 (G12) — v0.7 把"纯文档级提及"变成可度量的方法。

缺口背景:
  评估报告: 「终身学习 (G12) 完成度 0% — 纯文档级提及」。家庭场景的任务是
  开放的 (今天学会用这个杯子, 明天换一个碗), 顺序微调必然带来**灾难性遗忘**;
  没有度量就没有改进, 所以这里先把度量和实现都落地。

方法:
  - EWC (Elastic Weight Consolidation, Kirkpatrick et al. 2017): 用 Fisher 信息
    对角近似衡量"哪些参数对旧任务重要", 二次惩罚把它们锚住;
  - 经验回放 (replay): 保存旧任务的少量样本, 与新任务混合训练;
  - 组合: EWC + replay 是持续学习最实用的一对基线。

度量 (持续学习的标准指标):
  - forgetting:   学完后续任务后, 旧任务误差的**上升量**
  - backward transfer (BWT): 后续任务对旧任务的平均影响 (负 = 遗忘)
  - plasticity:   新任务能达到的精度 (防止"锚太死导致学不会")
"""
from .lifelong import (EWCRegularizer, LifelongConfig, LifelongLearner,
                       ReplayMemory, compute_fisher, lifelong_demo,
                       make_synthetic_tasks)

__all__ = ["EWCRegularizer", "LifelongConfig", "LifelongLearner", "ReplayMemory",
           "compute_fisher", "lifelong_demo", "make_synthetic_tasks"]
