"""零依赖 Web 仪表盘服务端 (L7)。

路由 (全部返回 JSON, 除 `/` 返回静态页面):
  GET  /                → 单页仪表盘
  GET  /api/status      → 总览: 版本 / 层栈 / 契约 / 模块可达性
  GET  /api/safety      → 安全链预算闭环 + 认证包结论 + 最近事件
  GET  /api/sim?steps=N → 跑一段参考环境仿真, 返回回报曲线与延迟
  GET  /api/biped       → 3D 步态: 前进距离 / 稳定性 / 关节角
  GET  /api/hand        → 灵巧手选型矩阵 + 推荐
  GET  /api/platform    → Linux on ARM 就绪评估 (当前主机)
  POST /api/command     → {"cmd": "sto"|"resume"|"hold"} 下发运行期指令
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

__all__ = ["DashboardState", "routes", "make_server", "poll_once", "gui_demo"]

_STATIC = Path(__file__).parent / "static" / "index.html"


# --------------------------------------------------------------------------- #
class DashboardState:
    """仪表盘状态容器: 缓存重活结果, 记录运行期指令。"""

    def __init__(self):
        self.mode = "IDLE"            # IDLE | RUNNING | HOLD | STO
        self.events: list = []
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self.ttl_s = 5.0

    def log(self, level: str, msg: str) -> None:
        self.events.append({"t": round(time.time(), 3), "level": level, "msg": msg})
        self.events = self.events[-50:]

    def cached(self, key: str, fn: Callable[[], Any]) -> Any:
        now = time.time()
        hit = self._cache.get(key)
        if hit and now - hit[0] < self.ttl_s:
            return hit[1]
        val = fn()
        self._cache[key] = (now, val)
        return val


# --------------------------------------------------------------------------- #
def _layer_stack() -> Dict[str, Any]:
    return {
        "L7": "用户交互 (本 Web 仪表盘 / 自然语言)",
        "L6": "任务规划 (VLA + LLM, ~1Hz)",
        "L5": "接触序列 (MCTS + DR, 2-10Hz)",
        "L4": "运动控制 (RL Actor / PPO / SAC / Diffusion, 20Hz)",
        "L3": "反射伺服 (阻抗 + 触觉反射, 1kHz)",
        "L2": "电机驱动 (FOC + SVPWM + 硬件关断, 10kHz)",
        "L1": "物理执行 (电机 + 减速器 + 传感器)",
        "P1-P5": "物理栈: 电源 / 感知 / 决策 / 执行 / 互联 PCB",
        "S": "横切安全层 (<1µs 硬件关断 + STL + ISO13482)",
        "D": "横切数据栈 (遥测 → DR → RL 飞轮)",
    }


def _module_status() -> Dict[str, str]:
    """各模块可达性 (导入即视为可用)。"""
    out = {}
    checks = {
        "contact_dynamics": "embodied_ai.dynamics.climbing_dynamics_core",
        "multirate_ekf": "embodied_ai.perception.multirate_ekf",
        "mcts_planning": "embodied_ai.planning.mcts_integration",
        "biped_locomotion": "embodied_ai.control.biped_locomotion",
        "whole_body_3d": "embodied_ai.control.whole_body",
        "soft_body": "embodied_ai.dynamics.soft_body",
        "simulation": "embodied_ai.simulation",
        "real_robot": "embodied_ai.simulation.real_robot_env",
        "foc_shutdown": "embodied_ai.hardware.foc_shutdown",
        "dexterous_hand": "embodied_ai.hardware.dexterous_hand",
        "lifelong": "embodied_ai.learning.lifelong",
        "certification": "embodied_ai.safety.certification",
        "jetson_bench": "embodied_ai.deploy.jetson_bench",
        "arm_linux": "embodied_ai.platform.arm_linux",
    }
    import importlib
    for k, mod in checks.items():
        try:
            importlib.import_module(mod)
            out[k] = "ok"
        except Exception as e:  # noqa: BLE001
            out[k] = f"error: {type(e).__name__}"
    return out


# --------------------------------------------------------------------------- #
def routes(state: DashboardState) -> Dict[str, Callable[[Dict[str, list]], Any]]:
    """路由表: path → handler(query)。"""
    def api_status(_q):
        from .. import __version__
        from ..config import load_contract
        mods = state.cached("mods", _module_status)
        contracts = {}
        for name in ("hw_requirements", "safety_budget",
                     "dr_params/manufacturing_variance"):
            try:
                load_contract(name)
                contracts[name] = "ok"
            except Exception as e:  # noqa: BLE001
                contracts[name] = f"error: {type(e).__name__}"
        return {"version": __version__, "mode": state.mode,
                "layers": _layer_stack(), "modules": mods,
                "contracts": contracts,
                "events": state.events[-10:]}

    def api_safety(_q):
        from ..config import load_contract
        from ..safety.certification import build_default_package
        from ..safety.formal import close_budget_loop
        sb = load_contract("safety_budget")
        # v1.1 字段: measured_* → assumed_* (向后兼容)
        def _g(s, *keys):
            for k in keys:
                if k in s:
                    return s[k]
            return 0.0
        total_ms = sum((_g(s, "assumed_us", "measured_us") / 1000 if "budget_us" in s
                       else _g(s, "assumed_ms", "measured_ms"))
                      for s in sb["segments"])
        # v1.1 修复: joint_speed_rad_s=8.0 会停在 inf, 让仪表盘显示不合理。
        # 用 0.5 rad/s (在预算内可停) 作为默认视图值, 同时暴露 8 rad/s 的诚实失败。
        closure = close_budget_loop(joint_speed_rad_s=0.5)
        closure_fast = close_budget_loop(joint_speed_rad_s=8.0)
        pkg = state.cached("cert", lambda: build_default_package().evaluate())
        return {"chain_total_ms": round(total_ms, 4),
                "budget_max_ms": sb["acceptance"]["total_break_max_ms"],
                "stop_angle_deg": round(closure.stop_angle_deg, 2),
                "closure_ok": bool(closure.closure_ok),
                "closure_ok_at_8rad_s": bool(closure_fast.closure_ok),
                "stop_angle_deg_at_8rad_s": closure_fast.stop_angle_deg,
                "provenance": sb.get("metadata", {}).get("provenance_confidence", "assumed"),
                "sil2_release_ready": sb.get("metadata", {}).get("sil2_release_ready", False),
                "sil2_pfh_target": sb["acceptance"]["sil2_pfh_target"],
                "certification": pkg,
                "mode": state.mode}

    def api_sim(q):
        steps = int(parse_qs_value(q, "steps", "40"))
        from ..simulation import DRConfig, DomainRandomizer, EnvConfig, make_env
        env = make_env(EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3,
                                 dt=0.01, max_episode_steps=steps), backend="reference")
        dr = DomainRandomizer(DRConfig(seed=0, enable_visual=False,
                                       enable_environment=False))
        env.set_dr_params(dr.sample_params())
        obs = env.reset(seed=0)
        rets = []
        for _ in range(steps):
            r = env.step(-0.5 * obs["state"][:3] - 0.1 * obs["state"][3:6])
            rets.append(round(float(r.reward), 4))
            obs = r.observation
            if r.done or r.truncated:
                break
        env.close()
        return {"steps": len(rets), "returns": rets,
                "final_return": float(np_mean(rets))}

    def api_biped(_q):
        from ..control.whole_body import WholeBody3D, WholeBodyConfig, LegGeometry
        wb = state.cached("biped", lambda: _run_biped())
        return wb

    def api_hand(_q):
        from ..hardware.dexterous_hand import design_matrix, recommend
        rows = state.cached("hand", design_matrix)
        return {"matrix": rows, "recommend": recommend(rows)}

    def api_platform(_q):
        from ..platform import (ArmRequirements, evaluate_readiness, probe_host,
                                summarize, target_to_dict)
        t = probe_host()
        checks = evaluate_readiness(t, ArmRequirements.from_contract(),
                                    as_target=False)
        return {"target": target_to_dict(t), "summary": summarize(t, checks)}

    def api_command(q):
        body = q.get("_body", [b"{}"])[0]
        try:
            payload = json.loads(body.decode("utf-8") if isinstance(body, bytes)
                                 else str(body))
        except Exception:  # noqa: BLE001
            payload = {}
        cmd = str(payload.get("cmd", "")).lower()
        if cmd == "sto":
            state.mode = "STO"
            state.log("critical", "STO 下发: 请求安全链关断")
        elif cmd == "hold":
            state.mode = "HOLD"
            state.log("warn", "暂停运动")
        elif cmd == "resume":
            state.mode = "RUNNING"
            state.log("info", "恢复运行")
        else:
            return {"ok": False, "error": f"未知指令: {cmd}"}
        return {"ok": True, "mode": state.mode}

    return {
        "/api/status": api_status,
        "/api/safety": api_safety,
        "/api/sim": api_sim,
        "/api/biped": api_biped,
        "/api/hand": api_hand,
        "/api/platform": api_platform,
        "/api/command": api_command,
    }


def parse_qs_value(q: Dict[str, list], key: str, default: str = "") -> str:
    v = q.get(key)
    return default if not v else str(v[0])


def np_mean(rets):
    return sum(rets) / len(rets) if rets else 0.0


def _run_biped() -> Dict[str, Any]:
    import numpy as np
    from ..control.whole_body import LegGeometry, WholeBody3D, WholeBodyConfig
    wb = WholeBody3D(WholeBodyConfig(), LegGeometry())
    wb.init_gait()
    dt = 0.005
    unstable = 0
    for _ in range(4):
        for _ in range(int(wb.cfg.step_time / dt)):
            st = wb.step(dt, zmp_target=wb.plan_zmp(False))
            unstable += 0 if st.stable else 1
        wb.plan_step()
    return {"advance_m": round(float(wb.com[0]), 4),
            "unstable_frames": unstable,
            "lateral_m": round(float(wb.com[1]), 4),
            "joint_limit": wb.geom.joint_limit,
            "q_left": [round(float(x), 4) for x in wb.state.q_left],
            "q_right": [round(float(x), 4) for x in wb.state.q_right]}


# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    state: DashboardState = None      # 由 make_server 注入
    table: Dict[str, Callable] = {}

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):  # pragma: no cover
            pass

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:          # noqa: N802 — http.server 约定
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            if _STATIC.exists():
                self._send(200, _STATIC.read_bytes(), "text/html; charset=utf-8")
            else:  # pragma: no cover — 静态页随包分发
                self._send(404, b"static/index.html missing", "text/plain")
            return
        h = self.table.get(u.path)
        if h is None:
            self._json({"error": f"no route: {u.path}"}, 404)
            return
        try:
            self._json(h(parse_qs(u.query)))
        except Exception as e:  # noqa: BLE001 — 仪表盘不应因单点异常掉线
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:         # noqa: N802
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b"{}"
        h = self.table.get(u.path)
        if h is None:
            self._json({"error": f"no route: {u.path}"}, 404)
            return
        try:
            self._json(h({"_body": [body], **parse_qs(u.query)}))
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args) -> None:   # 静音访问日志
        pass


def make_server(state: Optional[DashboardState] = None,
                host: str = "127.0.0.1", port: int = 0):
    """创建 (不启动) 一个仪表盘 HTTP 服务。port=0 → 由内核分配。"""
    st = state or DashboardState()
    table = routes(st)
    handler = type("DashboardHandler", (_Handler,), {"state": st, "table": table})
    srv = ThreadingHTTPServer((host, port), handler)
    return srv, st


def poll_once(port: int, host: str = "127.0.0.1") -> Dict[str, Any]:
    """拉一次全部只读 API (自检/冒烟用)。"""
    import urllib.request
    out = {}
    for path in ("/api/status", "/api/safety", "/api/sim?steps=20", "/api/biped",
                 "/api/hand", "/api/platform"):
        with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=30) as r:
            out[path] = json.loads(r.read().decode("utf-8"))
    return out


def gui_demo(port: int = 0) -> Dict[str, Any]:
    """起服务 → 拉 API → 断言 → 关服务。全程无外部依赖、无浏览器。"""
    srv, st = make_server(port=port)
    host, real_port = srv.server_address[0], srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        data = poll_once(real_port, host)
        status = data["/api/status"]
        safety = data["/api/safety"]
        biped = data["/api/biped"]
        hand = data["/api/hand"]

        # 指令通道: STO 必须改变 mode 并留下事件
        import urllib.request
        req = urllib.request.Request(
            f"http://{host}:{real_port}/api/command",
            data=json.dumps({"cmd": "sto"}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            cmd_res = json.loads(r.read().decode("utf-8"))
        with urllib.request.urlopen(f"http://{host}:{real_port}/api/status",
                                    timeout=10) as r:
            after = json.loads(r.read().decode("utf-8"))

        return {
            "port": int(real_port),
            "static_html_bytes": int(_STATIC.stat().st_size) if _STATIC.exists() else 0,
            "modules_ok": sum(1 for v in status["modules"].values() if v == "ok"),
            "modules_total": len(status["modules"]),
            "chain_total_ms": safety["chain_total_ms"],
            "cert_verdict": safety.get("certification", {}).get("verdict", "ERROR"),
            "biped_advance_m": biped["advance_m"],
            "biped_unstable": biped["unstable_frames"],
            "hand_recommended": hand["recommend"]["recommended"],
            "sto_accepted": bool(cmd_res.get("ok")),
            "mode_after_sto": after["mode"],
            "events_logged": len(after["events"]),
            "ok": bool(_STATIC.exists()
                       and all(v == "ok" for v in status["modules"].values())
                       and len(status["contracts"]) == 3
                       and safety["chain_total_ms"] < safety["budget_max_ms"]
                       and safety.get("certification", {}).get("verdict") == "PASS"
                       and biped["advance_m"] > 0.1
                       and biped["unstable_frames"] == 0
                       and hand["recommend"]["recommended"]
                       and cmd_res.get("ok") and after["mode"] == "STO"),
        }
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)
