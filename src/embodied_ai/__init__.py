"""EmbodiedAI 包 — 三栈分离架构 (控制L / 物理P / 数据D / 安全S横切)。

子模块按控制栈 L 层映射:
- models:     L6 VLA 小模型 (Vision-Language-Action, 复用 nanoGPT 管线模式)
              v0.7 + nn_core (零依赖训练内核) / diffusion_policy / bc_trainer /
              rl_algorithms (PPO+SAC)
- dynamics:   L5/L1 接触动力学 (climbing_dynamics_core) + 柔性体 (soft_body)
- perception: 触觉-视觉感知 (多速率 EKF) + 多传感器分频融合 + EKF-SLAM
- planning:   L5 MCTS 接触规划
- control:    L4/L3 RL Actor / Diffusion / 20Hz 控制 + 双足步态 (2D + 3D 全身)
- simulation: L1/L3 仿真闭环 (参考环境 / MuJoCo / PyBullet + DR + Sim2Real)
              v0.7 + real_robot_env (真机 CAN-FD 闭环)
- hardware:   v0.7 新增 — FOC/SVPWM/自动关机链 + 灵巧手设计决策
- learning:   v0.7 新增 — 终身学习 (EWC + 回放 + 遗忘度量)
- safety:     横切安全层 S (形式化 STL/ISO 13482 + 机会约束 + 认证工程化)
- deploy:     部署 (延迟核验 + VLA 推理 roofline 基准)
- gui:        v0.7 新增 — 零依赖 Web 仪表盘 (L7)
- runtime:    进程内消息总线 (Topic/Message/pub-sub) + MissionManager 决策中枢
- cognition:  v1.0 新增 — 四要素认知中枢 (时间/地点/人物/事情 的记录-分析-
              计算-规划-记忆)
- platform:   Linux on ARM 目标能力 (探测 / 就绪评估 / 预算缩放 / systemd 单元)
- adapters:   外部仓库适配 (EdaLab / nanoGPT / bigdata)
- config:     契约加载 (contracts/ 四件套)
- console:    终端安全输出 (Windows GBK 编码降级)
- verification: 聚合自检 (CLI verify / CI / scripts 共用)

K3 新增: models/ (VLA) + safety/formal/ (形式化安全 — 创新 #4)
v0.5 新增: simulation/ (零依赖参考环境 + 后端自动选择) + console/ + verification/
v0.6 新增: platform/ (Linux on ARM 目标能力层)
v0.7 新增: 见 CHANGELOG v0.7 — 缺口补齐 (G6/G9/G10/G11/G12/L7/训练管线/真机闭环)
v1.0 新增: 见 CHANGELOG v1.0 — Sprint M 感知/认知栈 (分频融合 / SLAM / V2A /
           认知中枢) + runtime 消息总线与 MissionManager
v1.1 新增: 见 CHANGELOG v1.1 — 验收诚实化 (判据去自我认证 / 静默失败改显式
           报错 / provenance 整改 / 单仓自足去个人路径)
v1.4 新增: 见 CHANGELOG v1.4 — seed→物理层域随机化 (ClimbDR) + 攀岩数据集 v2
           重生 + VLA×触觉端到端实跑与消融效应量门槛
"""

from . import console  # noqa: F401 — 导入即提供安全输出能力

#: 版本号**单一真源** — pyproject.toml 经 [tool.setuptools.dynamic] 静态提取
#: 此值, verification.VERSION 也从这里取。不要再在别处写死版本字面量
#: (tests/test_version_consistency.py 守护四方一致)。
__version__ = "1.4.0"
