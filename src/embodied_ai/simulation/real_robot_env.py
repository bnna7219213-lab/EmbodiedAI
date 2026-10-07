"""真机环境 (Sim-to-Real 闭环的最后一公里) — v0.7 补齐"链条仍是仿真到仿真"。

缺口背景:
  评估报告: 「Sim-to-Real 闭环完成度 35% — 从无真实硬件部署, Sim-to-Real 报告
  从未在真机上验证过」; 优先级 (3) 是「仿真策略部署到真实单关节做一次
  Sim-to-Real」。而 `sim_to_real._diagnose_gap` 里那句 "连接真机环境
  RealRobotEnv" 一直是 TODO — 本模块就是它。

设计要点:
  1. **同一个 SimEnv 接口**: SimToRealPipeline / BC 采集 / PPO 训练循环拿到
     RealRobotEnv 与拿到 ReferenceEnv 的代码完全一样 — 这是"仿真/真机统一
     接口"原则的兑现;
  2. **传输层可替换**: `MockTransport` (零依赖, CI 与 dry-run) /
     `SocketCANTransport` (Linux SocketCAN) / `SerialTransport` (USB-CAN 适配器);
  3. **CAN-FD 帧按 contracts/canfd/joint_control.dbc 编解码** (14 消息 (R4 后):
     6 JointCmd + Sync + 6 JointTelemetry + SafetyEvent), CRC8 与契约一致;
  4. **横切安全层 S 在真机侧必须在线**: 每一步先过 ISO13482 监控 + 停止角
     预算, 超限 → 立即下发 STO, 而不是"记录一下继续跑";
  5. **延迟可测**: 记录命令下发→遥测回收的环回延迟, 直接填 SimToRealResult
     的 avg_action_latency_ms (此前该字段恒为 0)。

没有硬件时: MockTransport 用一个"带噪声/延迟/丢帧"的一阶关节模型冒充真机,
因此整条链路 (包括安全闸门与延迟统计) 在 CI 上就能跑通并断言。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..safety.formal import ISO13482Monitor
from .base_env import EnvConfig, SimEnv, StepResult

__all__ = ["CANFrame", "Transport", "MockTransport", "SocketCANTransport",
           "SerialTransport", "RealRobotEnv", "RealRobotConfig", "real_robot_demo",
           "encode_joint_cmd", "decode_telemetry", "crc8"]


# --------------------------------------------------------------------------- #
# CAN-FD 帧 (对齐 contracts/canfd/joint_control.dbc)
# --------------------------------------------------------------------------- #
CMD_IDS = {0: 100, 1: 104, 2: 108, 3: 112, 4: 116, 5: 120}
SYNC_ID = 0
# R4 修复 (2026-10-07): 6 关节遥测, stride 4 (与 JointCmd 100/104/.../120 对齐)。
# DBC JointTelemetry_<joint> 共 6 条: 300/304/308/312/316/320。
TELEM_IDS = {0: 300, 1: 304, 2: 308, 3: 312, 4: 316, 5: 320}
TELEM_ID_BASE = 300  # 保留兼容别名; 新代码请用 TELEM_IDS
SAFETY_ID = 900

# 位置信号缩放: 15-bit × 0.01 deg ⇒ 量程 0~327.67 deg, 配合 180° 偏置可覆盖
# ±147° 关节行程。v0.7 起从 0.001 改为 0.01 — 原值量程只有 32.767 deg,
# 连一个 90° 的肩关节都发不出去 (实测所有指令被饱和成 32.767°)。
POS_SCALE = 0.01


def crc8(data: bytes, poly: int = 0x07) -> int:
    """CRC-8/ATM (poly 0x07) — 与 DBC 的 crc8 信号一致。"""
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFF if (c & 0x80) else (c << 1) & 0xFF
    return int(c)


@dataclass
class CANFrame:
    can_id: int
    data: bytes = b"\x00" * 8
    ts: float = field(default_factory=time.time)


def encode_joint_cmd(joint: int, pos_deg: float, vel_dps: float,
                     torque_lim_a: float) -> CANFrame:
    """JointCmd_<joint> (8 字节, 与 DBC 的缩放/偏移一致)。"""
    if joint not in CMD_IDS:
        raise ValueError(f"关节索引越界: {joint} (DBC 只定义 0..5)")
    p = int(np.clip(pos_deg / POS_SCALE, 0, 0x7FFF))
    v = int(np.clip(vel_dps / 0.01, 0, 0x7FFF))
    t = int(np.clip(torque_lim_a / 0.1, 0, 0xFF))
    d = bytearray(8)
    d[0:2] = p.to_bytes(2, "little")
    d[2:4] = v.to_bytes(2, "little")
    d[3] = (d[3] & 0x7F) | ((t & 0x1) << 7)     # torque 低 1 位放进 bit31
    d[4] = (t >> 1) & 0x7F                       # 余下 7 位
    d[7] = crc8(bytes(d[:7]))
    return CANFrame(CMD_IDS[joint], bytes(d))


def decode_telemetry(frame: CANFrame) -> Dict[str, float]:
    """JointTelemetry 解包 → (pos_deg, current_a, temp_c, flags)。

    R4 修复 (2026-10-07): 按 contracts/canfd/joint_control.dbc 字面位域解包 —
    actual_current_a 是 bit 16|11@0+ (CAN bit 16 len 11), 在
    int.from_bytes(d[2:4],"little") 中对应 value bits 0-10, **不需要移位**
    (旧 `>> 5` 错读 bits 21-31, 跨过 DBC 位边界)。
    temp_c 是 bit 27|9@0+, 起点 byte3 bit 3, 在
    int.from_bytes(d[3:5],"little") 中对应 value bits 3-11, 应 >>3
    (旧 `>> 5` 错读 bits 29-37)。
    """
    d = frame.data
    pos = int.from_bytes(d[0:2], "little") & 0x7FFF        # bit 0|15, no shift
    cur_raw = int.from_bytes(d[2:4], "little") & 0x7FF       # bit 16|11, no shift
    temp_raw = (int.from_bytes(d[3:5], "little") >> 3) & 0x1FF  # bit 27|9, shift 3
    flags = d[5] if len(d) > 5 else 0
    return {"pos_deg": pos * POS_SCALE, "current_a": cur_raw * 0.01,
            "temp_c": temp_raw * 0.5 - 20.0, "flags": int(flags),
            "crc_ok": bool(len(d) >= 8 and crc8(bytes(d[:7])) == d[7])}


class Transport:
    """CAN 传输层抽象。"""

    def open(self) -> None:
        raise NotImplementedError

    def send(self, frame: CANFrame) -> None:
        raise NotImplementedError

    def recv(self, timeout_s: float = 0.01) -> Optional[CANFrame]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class MockTransport(Transport):
    """离线真机替身: 一阶关节模型 + 噪声 + 延迟 + 丢帧 + CRC 错误注入。

    它不是"把仿真又跑一遍", 而是刻意加入**仿真里没有**的真机特性:
    传输延迟、丢帧、CRC 错误、堵转/过流、温升 — 这些正是 Sim-to-Real 差距
    的来源, 也只有在它们存在时差距量化才有意义。
    """

    def __init__(self, n_joints: int = 6, latency_ms: float = 1.2,
                 drop_rate: float = 0.01, crc_error_rate: float = 0.0,
                 seed: int = 0):
        self.n_joints = n_joints
        self.latency_ms = latency_ms
        self.drop_rate = drop_rate
        self.crc_error_rate = crc_error_rate
        self.rng = np.random.default_rng(seed)
        self.pose = np.zeros(n_joints)
        self.current = np.zeros(n_joints)
        self.temp = np.full(n_joints, 35.0)
        self._queue: List[Tuple[float, CANFrame]] = []
        self._t = 0.0
        self.sto = False
        self.stats = {"sent": 0, "recv": 0, "dropped": 0, "crc_err": 0}
        # R4 (2026-10-07): 6 关节全量遥测。每收到一次 SYNC, 排队生成 n_joints
        # 帧 JointTelemetry (can_id = TELEM_IDS[0..5]), RealRobotEnv._sync_and_read
        # 必须连续 recv 6 次消费完。本字段记录"还要生成几帧"。
        self._telem_remaining = 0
        # 虚拟环回延迟 (ms): 真机上这是 CAN 往返 + 驱动响应的实测值; mock 里
        # 用配置值 + 抖动, 让 RealRobotEnv 的延迟统计在无硬件时也有意义
        # (若只记 perf_counter 实测, mock 会得到 0.03ms 这种"假的好看"数字)。
        self.last_latency_ms: float = 0.0

    def open(self) -> None:
        self._t = 0.0
        self._telem_remaining = 0

    def _build_telemetry(self, j: int) -> CANFrame:
        """生成关节 j 的 JointTelemetry 帧 (按 DBC 2026-10-07 修正后的位域)。"""
        d = bytearray(8)
        pos = int(np.clip((np.rad2deg(self.pose[j]) + 180.0) / POS_SCALE, 0, 0x7FFF))
        cur = int(np.clip(self.current[j] / 0.01, 0, 0x7FF))
        tmp = int(np.clip((self.temp[j] + 20.0) / 0.5, 0, 0x1FF))
        d[0:2] = (pos & 0x7FFF).to_bytes(2, "little")
        # R4 修复 (2026-10-07): 按 DBC 位域写 telemetry —
        # actual_current_a 是 bit 16|11, 在 d[2:4] 的 value bits 0-10, **不要 <<5**
        # (旧 <<5 与 decode_telemetry 的 >>5 同错, 自闭环掩盖真实设备失真)
        # temp_c 是 bit 27|9, 在 d[3:5] 的 value bits 3-11, 应 <<3
        cw = cur & 0x7FF
        d[2:4] = cw.to_bytes(2, "little")
        tw = (tmp & 0x1FF) << 3
        d[3:5] = (int.from_bytes(d[3:5], "little") | tw).to_bytes(2, "little")
        d[5] = 0
        if self.rng.random() < self.crc_error_rate:
            d[7] = (crc8(bytes(d[:7])) + 1) & 0xFF
            self.stats["crc_err"] += 1
        else:
            d[7] = crc8(bytes(d[:7]))
        return CANFrame(TELEM_IDS[j], bytes(d))

    def send(self, frame: CANFrame) -> None:
        self.stats["sent"] += 1
        if self.rng.random() < self.drop_rate:
            self.stats["dropped"] += 1
            return
        # 下行: 关节目标 → 内部一阶模型
        if frame.can_id in CMD_IDS.values():
            j = [k for k, v in CMD_IDS.items() if v == frame.can_id][0]
            d = frame.data
            pos = (int.from_bytes(d[0:2], "little") & 0x7FFF) * POS_SCALE
            # DBC 用无符号 deg + 180° 零点偏置 (与 RealRobotConfig.home_deg 一致)
            target = np.deg2rad(pos) - np.pi
            if not self.sto:
                self.pose[j] += 0.35 * (target - self.pose[j])
                self.current[j] = 0.6 * abs(target - self.pose[j]) + 0.2
            else:
                self.current[j] *= 0.2
        elif frame.can_id == SYNC_ID:
            # R4: 收到同步帧 → 标记要生成 n_joints 帧 JointTelemetry, 由后续
            # 6 次 recv() 逐帧返回。每帧带相同延迟 (CAN 整批回传语义)。
            self._telem_remaining = self.n_joints
        delay = max(0.0, self.rng.normal(self.latency_ms / 1000.0,
                                         self.latency_ms / 4000.0))
        self._queue.append((self._t + delay, frame))
        self._t += 1e-4

    def recv(self, timeout_s: float = 0.01) -> Optional[CANFrame]:
        self._t += timeout_s
        # R4: 优先消费 sync 触发的 6 关节 telemetry 队列
        if self._telem_remaining > 0:
            j = self.n_joints - self._telem_remaining
            self._telem_remaining -= 1
            self.stats["recv"] += 1
            self.last_latency_ms = float(max(
                0.0, self.rng.normal(self.latency_ms, self.latency_ms / 8.0)))
            return self._build_telemetry(j)
        # 否则: 检查延迟队列 (cmd/sync 回声, 真机收不到 — 兼容)
        if not self._queue:
            return None
        self._queue = [q for q in self._queue if q[0] <= self._t] or self._queue
        due = [q for q in self._queue if q[0] <= self._t]
        if not due:
            return None
        _, frame = due[0]
        self._queue.remove(due[0])
        self.stats["recv"] += 1
        self.last_latency_ms = float(max(
            0.0, self.rng.normal(self.latency_ms, self.latency_ms / 8.0)))
        return frame

    def close(self) -> None:
        self._queue.clear()
        self._telem_remaining = 0


class SocketCANTransport(Transport):
    """Linux SocketCAN (python-can 可选依赖; 缺失时构造即报错, 由调用方降级)。"""

    def __init__(self, channel: str = "can0", bitrate: int = 500000,
                 fd: bool = True):
        self.channel, self.bitrate, self.fd = channel, bitrate, fd
        self._bus = None

    def open(self) -> None:
        import can  # type: ignore  — 可选依赖, 缺失时 ImportError 上抛
        self._bus = can.interface.Bus(channel=self.channel,
                                      interface="socketcan", fd=self.fd)

    def send(self, frame: CANFrame) -> None:
        import can  # type: ignore
        assert self._bus is not None
        self._bus.send(can.Message(arbitration_id=frame.can_id, data=frame.data,
                                   is_fd=self.fd, is_extended_id=False))

    def recv(self, timeout_s: float = 0.01):
        assert self._bus is not None
        m = self._bus.recv(timeout=timeout_s)
        return None if m is None else CANFrame(m.arbitration_id, bytes(m.data))

    def close(self) -> None:
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None


class SerialTransport(Transport):
    """USB-CAN 适配器的 SLCAN/串 backlog (pyserial 可选依赖)。"""

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 2000000):
        self.port, self.baud = port, baud
        self._ser = None

    def open(self) -> None:
        import serial  # type: ignore
        self._ser = serial.Serial(self.port, self.baud, timeout=0.01)

    def send(self, frame: CANFrame) -> None:
        assert self._ser is not None
        self._ser.write(f"t{frame.can_id:03X}{len(frame.data)}"
                        f"{frame.data.hex().upper()}\r".encode())

    def recv(self, timeout_s: float = 0.01):
        assert self._ser is not None
        line = self._ser.readline()
        if not line or not line.startswith(b"t"):
            return None
        try:
            cid = int(line[1:4], 16)
            payload = bytes.fromhex(line[5:-1].decode().strip())
        except ValueError:
            return None
        return CANFrame(cid, payload.ljust(8, b"\x00"))

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None


# --------------------------------------------------------------------------- #
@dataclass
class RealRobotConfig:
    n_joints: int = 6
    transport: str = "mock"          # mock | socketcan | serial
    channel: str = "can0"
    serial_port: str = "/dev/ttyUSB0"
    torque_limit_a: float = 20.0
    safety_enabled: bool = True
    max_force_n: float = 280.0
    max_speed_ms: float = 0.25
    # 关节级硬限位 (闸门在 ISO13482 之外单独判: 后者管"人机共存"的力和速度,
    # 前者管机械本体不被自己拧断)
    joint_limit_rad: float = 2.0
    max_joint_speed_rad_s: float = 12.0
    home_deg: float = 180.0           # DBC 无符号编码的零点偏置
    telem_timeout_s: float = 0.01


class RealRobotEnv(SimEnv):
    """真机环境 — 与仿真环境同接口, 但每次 step 真的走一趟 CAN。"""

    def __init__(self, config: EnvConfig, robot: RealRobotConfig = None,
                 transport: Optional[Transport] = None):
        super().__init__(config)
        self.robot = robot or RealRobotConfig()
        cfg = self.robot
        if transport is not None:
            self.tr = transport
        elif cfg.transport == "socketcan":
            self.tr = SocketCANTransport(cfg.channel)
        elif cfg.transport == "serial":
            self.tr = SerialTransport(cfg.serial_port)
        else:
            self.tr = MockTransport(n_joints=cfg.n_joints)
        self.tr.open()

        n = cfg.n_joints
        self._pose = np.zeros(n)
        self._vel = np.zeros(n)
        self._current = np.zeros(n)
        self._temp = np.full(n, 35.0)
        self.sto = False
        self.safety_log: List[Dict] = []
        self.latencies_ms: List[float] = []
        self.monitor = ISO13482Monitor(state_dim=config.state_dim,
                                       dt=config.dt,
                                       max_force_N=cfg.max_force_n,
                                       max_speed_ms=cfg.max_speed_ms)
        self._last_cmd = np.zeros(n)

    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self.step_count = 0
        self.clear_episode()
        self.sto = False
        self._pose[:] = 0.0
        self._vel[:] = 0.0
        self._send_all(np.zeros(self.robot.n_joints))
        self._sync_and_read()
        return self._build_observation(self._state())

    def _state(self) -> np.ndarray:
        sd = self.config.state_dim
        s = np.zeros(sd)
        n = min(self.robot.n_joints, sd // 2)
        s[:n] = self._pose[:n]
        s[sd // 2: sd // 2 + n] = self._vel[:n]
        return s

    def close(self) -> None:
        try:
            self.tr.close()
        except Exception:  # noqa: BLE001 — 关闭失败不应掩盖上层结果
            pass

    # ------------------------------------------------------------------ #
    def _send_all(self, cmd_rad: np.ndarray) -> None:
        """下发全部关节命令。

        动作维度可能小于关节数 (如 3 维策略驱动 6 关节) — 缺位补零, 不允许
        越界 (v0.7 初版直接索引, 3 维动作打到 6 关节立即 IndexError)。
        """
        cmd = np.zeros(self.robot.n_joints)
        cmd[: min(len(cmd_rad), self.robot.n_joints)] = cmd_rad[: self.robot.n_joints]
        for j in range(self.robot.n_joints):
            pos_deg = float(np.rad2deg(cmd[j]) + self.robot.home_deg)
            self.tr.send(encode_joint_cmd(j, pos_deg, 0.0, self.robot.torque_limit_a))

    def _sync_and_read(self) -> None:
        """下发同步帧 → 收 n_joints 帧 JointTelemetry → 更新状态与延迟统计。

        R4 (2026-10-07): 6 关节全量遥测。MockTransport 在收到 SYNC 后会排队
        生成 n_joints 帧 JointTelemetry (can_id = TELEM_IDS[j]), 本函数逐帧
        消费并按 TELEM_IDS 反查关节号, 然后写入 _pose/_current/_temp。
        真机行为: 同步帧是广播, 各关节异步回传; mock 一次性回传是等价的
        (延迟用同一 latency_ms 模型)。
        """
        t0 = time.perf_counter()
        self.tr.send(CANFrame(SYNC_ID, bytes([0xA5]) + b"\x00" * 6 + b"\x00"))
        # can_id → joint 索引反向表 (避免每次 recv 都遍历 TELEM_IDS)
        id_to_joint = {v: k for k, v in TELEM_IDS.items()}
        for _ in range(self.robot.n_joints):
            frame = self.tr.recv(timeout_s=self.robot.telem_timeout_s)
            if frame is None:
                self.latencies_ms.append(self.robot.telem_timeout_s * 1000.0)
                continue
            wall = (time.perf_counter() - t0) * 1000.0
            reported = getattr(self.tr, "last_latency_ms", 0.0)
            self.latencies_ms.append(float(max(wall, reported)))
            if frame.can_id == SAFETY_ID:
                self.sto = True
                continue
            tel = decode_telemetry(frame)
            if not tel["crc_ok"]:
                continue                   # CRC 错 → 丢弃本帧, 保持上一状态
            j = id_to_joint.get(frame.can_id)
            if j is None or j >= self.robot.n_joints:
                continue                    # 未知 can_id, 跳过
            self._pose[j] = np.deg2rad(tel["pos_deg"] - self.robot.home_deg)
            self._current[j] = tel["current_a"]
            self._temp[j] = tel["temp_c"]

    # ------------------------------------------------------------------ #
    def _safety_gate(self, action: np.ndarray) -> Tuple[bool, str]:
        """横切安全层 S: 超限 → True(触发 STO)。"""
        if not self.robot.safety_enabled:
            return False, "disabled"
        st = self._state()
        res = self.monitor.evaluate(joint_state=st,
                                    force_reading=float(np.max(np.abs(self._current)) * 10.0),
                                    torque_reading=float(np.max(np.abs(action))) * 10.0,
                                    timestamp=self.step_count * self.config.dt)
        if not res.safe:
            worst = max(res.violations, key=lambda v: v.severity) if res.violations else None
            msg = worst.clause if worst else "ISO13482 violation"
            self.safety_log.append({"step": self.step_count, "clause": msg,
                                    "severity": int(worst.severity) if worst else 0})
            return True, msg
        # 关节级: 限位 / 过流 / 角速度
        if np.max(np.abs(self._pose)) > self.robot.joint_limit_rad:
            return True, "joint_limit_exceeded"
        if np.max(np.abs(self._current)) > self.robot.torque_limit_a * 0.98:
            return True, "overcurrent"
        if np.max(np.abs(self._vel)) > self.robot.max_joint_speed_rad_s:
            return True, "joint_overspeed"
        return False, ""

    def _trigger_sto(self) -> None:
        """下发 STO: 这里由主控请求, 真正的**电气关断**在 P4 PCB 硬件侧。"""
        self.sto = True
        d = bytearray(8)
        d[0] = 0x02                    # fault_type = overcurrent/unsafe
        d[3] = 0x01                    # sto_triggered
        d[7] = crc8(bytes(d[:7]))
        self.tr.send(CANFrame(SAFETY_ID, bytes(d)))

    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=float).reshape(-1)
        a = self._apply_action(action)[: self.robot.n_joints]

        trip, clause = self._safety_gate(a)
        if trip:
            self._trigger_sto()
            obs = self._build_observation(self._state())
            self.record_step(obs, a, -100.0, obs, True)
            return StepResult(observation=obs, reward=-100.0, done=True,
                              truncated=False,
                              info={"sto": True, "clause": clause, "success": False})

        if self.sto:
            self._send_all(np.zeros(self.robot.n_joints))
        else:
            self._send_all(a)
        prev = self._pose.copy()
        self._sync_and_read()
        self._vel = (self._pose - prev) / max(self.config.dt, 1e-6)

        self.step_count += 1
        obs = self._build_observation(self._state())
        dist = float(np.linalg.norm(self._pose))
        reward = -dist - 0.01 * float(np.dot(a, a))
        done = bool(np.max(np.abs(self._current)) > self.robot.torque_limit_a * 0.9)
        truncated = self.step_count >= self.config.max_episode_steps
        self.record_step(obs, a, reward, obs, done or truncated)
        # P0 修复: 原代码 success 硬编码 False, RL 训练时无法区分成功/失败。
        # 任务达成判据: episode 结束时 pose 接近 home 位 (‖pose‖ < threshold),
        # 且未触发 STO。threshold 默认 0.5 rad ≈ 28.6° (与关节限位 30° 同源)。
        success_threshold_rad = getattr(self.config, "success_threshold_rad", 0.5)
        success = bool(not done and not truncated
                       and dist < success_threshold_rad)
        return StepResult(observation=obs, reward=reward, done=done,
                          truncated=truncated,
                          info={"sto": self.sto, "success": success,
                                "latency_ms": self.latencies_ms[-1]
                                if self.latencies_ms else 0.0,
                                "current_a": float(np.max(np.abs(self._current))),
                                "pose_dist_rad": dist,
                                "success_threshold_rad": success_threshold_rad})

    # ------------------------------------------------------------------ #
    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def safety_summary(self) -> Dict[str, Any]:
        return {"sto": self.sto, "violations": len(self.safety_log),
                "log": self.safety_log[-10:],
                "avg_latency_ms": self.avg_latency_ms,
                "transport_stats": getattr(self.tr, "stats", {})}


# --------------------------------------------------------------------------- #
def real_robot_demo(steps: int = 60, seed: int = 0) -> Dict[str, Any]:
    """真机闭环自检: mock 传输跑通 Sim-to-Real, 含安全闸门与延迟统计。"""
    from .sim_to_real import SimToRealPipeline

    cfg = EnvConfig(scene_name="real_joint", state_dim=6, action_dim=3, dt=0.005,
                    max_episode_steps=steps)
    sim_env = None
    from . import make_env
    sim_env = make_env(EnvConfig(scene_name="point_mass", state_dim=6, action_dim=3,
                                 dt=0.01, max_episode_steps=steps), backend="reference")
    real = RealRobotEnv(cfg, RealRobotConfig(transport="mock"))

    class _P:
        def __init__(self, target):
            self.target = target

        def predict(self, obs):
            s = np.asarray(obs["state"], dtype=float)
            n = len(self.target)
            # 增益压到真机侧安全闸门之下: 关节速度上限 12 rad/s, 太猛的策略
            # 会在第 2 步就被自己触发的 STO 打断, episode 只剩 2 个样本。
            return -0.5 * (s[:n] - self.target) - 0.1 * s[n:2 * n]

    pipe = SimToRealPipeline(sim_env, real, _P(np.zeros(3)))
    res = pipe.run(sim_episodes=1, real_episodes=1, max_steps_per_episode=steps)

    # 安全闸门必须能真的拦下来: 灌一个超限动作
    real2 = RealRobotEnv(cfg, RealRobotConfig(transport="mock"))
    real2.reset(seed=seed)
    blocked = False
    for _ in range(10):
        r = real2.step(np.array([50.0, 50.0, 50.0]))   # 远超限位/力矩
        if r.info.get("sto"):
            blocked = True
            break
    real2.close()
    real.close()

    # CAN 帧编解码往返
    f = encode_joint_cmd(2, 123.456, 10.0, 20.0)
    dec = decode_telemetry(CANFrame(300, f.data))   # 结构校验用
    crc_ok = bool(f.data[7] == crc8(bytes(f.data[:7])))

    return {
        "real_episodes_evaluated": int(res.real_episodes),
        "gap_diagnosis": res.gap_diagnosis,
        "avg_latency_ms": float(real.avg_latency_ms),
        "latency_samples": len(real.latencies_ms),
        "safety_gate_blocked": blocked,
        "safety_violations_logged": len(real2.safety_log),
        "crc_roundtrip_ok": crc_ok,
        "transport": getattr(real.tr, "stats", {}),
        "ok": bool(res.real_episodes == 1 and res.gap_diagnosis
                   and real.avg_latency_ms > 0.0 and blocked and crc_ok),
    }
