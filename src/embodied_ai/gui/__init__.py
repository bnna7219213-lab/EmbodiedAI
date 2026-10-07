"""顶层 GUI 交互界面 (L7) — v0.7 把"仅有文档方案、0% 代码"落地。

缺口背景:
  评估报告: 「GUI 交互界面 (L7) 完成度 0% — docs/05-system-software/gui-interface.md
  仅有文档方案, 无任何 Qt/QML 或 Web 前端代码」。

技术选型 (写在这里以免再被问一次):
  **不引入 Qt/PySide、不引入 React/Vite**, 用 Python 标准库 `http.server` +
  单文件原生 JS。理由:
    1. 项目运行时依赖只有 numpy/pyyaml — 引入 GUI 框架会把安装体积抬一个数量级,
       并让 CI (无显示/无 node) 无法验证;
    2. 具身机器人的操作台往往是**另一台机器**上的浏览器, Web 天然是 C/S 架构;
    3. 全部可离线自检: 起服务 → HTTP 拉 JSON → 断言结构 → 关服务, 无需截图。

后续若确实要做本地高刷 3D 视口 (点云/网格), 再引 Qt6 并**保留本 HTTP 层**做
远程遥测 — 两者不冲突。
"""
from .server import DashboardState, gui_demo, make_server, poll_once, routes

__all__ = ["DashboardState", "make_server", "routes", "poll_once", "gui_demo"]
