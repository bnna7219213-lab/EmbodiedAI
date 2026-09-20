# embodied_ai 包

三栈分离架构 (控制L / 物理P / 数据D / 安全S横切) 的 Python 实现。

## 使用

```bash
# 环境有 setuptools 时: pip install -e .  → 命令行 embodied-audit 可用
# 无 setuptools 环境的等价方式:
PYTHONPATH=src python -m embodied_ai.cli audit      # 五项目资产状态汇总
PYTHONPATH=src python -m embodied_ai.cli contracts  # 契约四件套校验
PYTHONPATH=src python -m embodied_ai.cli safety     # 安全链预算核对
```

## 模块映射

| 模块 | 控制栈 | 来源 |
|---|---|---|
| dynamics/ | L5/L1 数学 | climbing_dynamics_core.py 原样迁移 |
| perception/ | 传感 | MultiRateEKF 拆分自 climbing_dynamics_core |
| planning/ | L5 | mcts_integration.py (K1 清洗为合法 Python) |
| control/ | L4/L3 | household_robot_system.py (K1 清洗) |
| safety/ | 横切 S | chance_constrained_grip 拆分自 climbing_dynamics_core |
| config/ | — | contracts/ 四件套加载校验 |
| adapters/ | K2 | edalab/nanogpt/bigdata 适配器 |

## K1 清洗说明

mcts_integration.py 与 household_robot_system.py 的原始文件为对话式 markdown
（含 `## 🤖 Assistant` 头与 code fence），不是合法 Python——K1 已清洗为可解析
模块，逻辑未改；引用但未定义的基类（ClimbingMCTS 等）以桩替代，K2 重构补全。
原始文件归档于 `archive/src-pre-k1-2026-09-20/`。
