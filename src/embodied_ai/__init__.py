"""EmbodiedAI 包 — 三栈分离架构 (控制L / 物理P / 数据D / 安全S横切)。

子模块按控制栈 L 层映射:
- models:     L6 VLA 小模型 (Vision-Language-Action, 复用 nanoGPT 管线模式)
- dynamics:   L5/L1 接触动力学 (climbing_dynamics_core 迁入)
- perception: 触觉-视觉感知 (多速率 EKF)
- planning:   L5 MCTS 接触规划
- control:    L4/L3 RL Actor / Diffusion / 20Hz 控制 + 双足步态
- simulation: L1/L3 仿真闭环 (参考环境 / MuJoCo / PyBullet + DR + Sim2Real)
- safety:     横切安全层 S (形式化 STL/ISO 13482 + 概率性机会约束)
- deploy:     部署 (量化/TensorRT/延迟核验)
- platform:   Linux on ARM 目标能力 (探测 / 就绪评估 / 预算缩放 / systemd 单元)
- adapters:   外部仓库适配 (EdaLab / nanoGPT / bigdata)
- config:     契约加载 (contracts/ 四件套)
- console:    终端安全输出 (Windows GBK 编码降级)
- verification: 聚合自检 (CLI verify / CI / scripts 共用)

K3 新增: models/ (VLA) + safety/formal/ (形式化安全 — 创新 #4)
v0.5 新增: simulation/ (零依赖参考环境 + 后端自动选择) + console/ + verification/
v0.6 新增: platform/ (Linux on ARM 目标能力层)
"""

from . import console  # noqa: F401 — 导入即提供安全输出能力

__version__ = "0.6.0"
