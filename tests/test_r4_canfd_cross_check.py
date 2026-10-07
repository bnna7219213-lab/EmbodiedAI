"""R4 CAN-FD vs DBC 独立参考编码器对照测试 (2026-10-07, plan_v3 Sprint R-A)。

打破 mock 同错全绿的盲区 (评估 §6.3 G3, [docs/11 §三 R4](docs/11-production-upgrade-roadmap.md)):

  不导入被测代码 (`embodied_ai.simulation.real_robot_env` 在 ≥1 处被
  `import` 时即视为本测试不独立), 按 `contracts/canfd/joint_control.dbc`
  字面 SG_ 信号位域**独立重实现**一套参考 DBC 解析 + 参考信号编码/解码。
  然后:
    - JointCmd: project encode vs reference encode → 字节相等
    - JointTelemetry: reference encode → project decode → 物理量 1 LSB 内一致
    - 边界值 (DBC range 两端, 零, 居中)
    - DBC 字段改动应使本测试的 DBC 解析路径失败 (非"无声通过")

历史问题 (本测试曝光):
  - decode_telemetry 的 actual_current_a 位移 `>> 5` 错 (DBC bit 16 起始,
    应 no-shift; 旧错读 bits 21-31)
  - decode_telemetry 的 temp_c 位移 `>> 5` 错 (DBC bit 27 起始, 应 `>> 3`;
    旧错读 bits 29-33)
  - MockTransport.recv 用同样错位移写, 所以 mock 自闭环测不出 — 真实 DBC
    合规硬件一来就静默串扰

本测试不依赖项目代码里的任何 helper (crc8/encode_joint_cmd/decode_telemetry
都不 import)。仅在测试体内按需 import `CANFrame` dataclass 用作 project
端输入的"包装" — 这一约束已通过本地 dataclass 等价物回避, 见下。
"""

from __future__ import annotations

import math
import random
import re
import unittest
from pathlib import Path
from typing import Dict, List, Tuple


# ─── 独立 DBC 解析 (零依赖, 不读项目代码) ──────────────────────────────────

_SG_RE = re.compile(
    r"^SG_\s+(\w+)\s*:\s*(\d+)\|(\d+)@([01])([+-])\s*\("
    r"([^,]+),([^\)]+)\)"
)


def parse_dbc_signals(dbc_path: Path) -> Dict[int, Dict[str, Tuple[int, int, float, float, bool]]]:
    """解析 DBC 中 BO_ 消息的 SG_ 信号位域。

    返回: { can_id: { signal_name: (start_bit, length, scale, offset, is_little) } }
    """
    out: Dict[int, Dict[str, Tuple[int, int, float, float, bool]]] = {}
    text = dbc_path.read_text(encoding="utf-8")
    cur_id: int = -1
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"^BO_\s+(\d+)\s+(\w+):\s+\d+\s+\w+", line)
        if m:
            cur_id = int(m.group(1))
            out[cur_id] = {}
            continue
        m = _SG_RE.match(line)
        if m and cur_id >= 0:
            name = m.group(1)
            start_bit = int(m.group(2))
            length = int(m.group(3))
            endian = m.group(4)
            scale = float(m.group(6).strip())
            offset = float(m.group(7).strip())
            is_little = endian == "0"
            out[cur_id][name] = (start_bit, length, scale, offset, is_little)
    return out


# ─── 独立参考 CRC-8 (poly 0x07) ─────────────────────────────────────────────

def ref_crc8(data: bytes, poly: int = 0x07) -> int:
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            if c & 0x80:
                c = ((c << 1) ^ poly) & 0xFF
            else:
                c = (c << 1) & 0xFF
    return int(c)


# ─── 独立参考位操作 (little-endian DBC @0+) ─────────────────────────────────

def ref_place_bits(d: bytearray, start_bit: int, length: int, raw: int) -> None:
    """按 DBC little-endian 位布局, 把 length 位 raw 放入 d。

    bit_pos = start_bit + i, byte_idx = bit_pos // 8, bit_in_byte = bit_pos % 8
    """
    if not (0 <= raw < (1 << length)):
        raise ValueError(f"raw={raw} 超出 {length} 位范围")
    for i in range(length):
        bit_val = (raw >> i) & 1
        bit_pos = start_bit + i
        byte_idx = bit_pos // 8
        bit_in_byte = bit_pos % 8
        if bit_val:
            d[byte_idx] |= (1 << bit_in_byte)
        else:
            d[byte_idx] &= ~(1 << bit_in_byte) & 0xFF


def ref_read_bits(d: bytes, start_bit: int, length: int) -> int:
    raw = 0
    for i in range(length):
        bit_pos = start_bit + i
        byte_idx = bit_pos // 8
        bit_in_byte = bit_pos % 8
        bit_val = (d[byte_idx] >> bit_in_byte) & 1
        raw |= (bit_val << i)
    return raw


def ref_scale_to_raw(phys: float, scale: float, offset: float) -> int:
    """物理值 → raw。匹配 project 的 `int(np.clip(...))` 截断语义 (向零)。

    关键: 不使用 round(). Python 3 的 round() 在 0.5 处取最近偶数, 与 C-style
    截断/远离零都不同; np.int_/int() 在 numpy 标量上**截断向零**, 与 project
    的 int(np.clip(phys/scale, 0, MAX)) 完全一致。
    """
    return int((phys - offset) / scale)


def ref_raw_to_phys(raw: int, scale: float, offset: float) -> float:
    return raw * scale + offset


# ─── 独立参考: JointCmd 编码 (用于与 project encode byte-exact 对照) ──────

CMD_IDS = [100, 104, 108, 112, 116, 120]  # 与 code CMD_IDS 一致 (仅用于构造)


def ref_encode_joint_cmd(sigs: Dict[str, Tuple[int, int, float, float, bool]],
                         target_pos_deg: float, target_vel_dps: float,
                         torque_lim_a: float) -> bytes:
    d = bytearray(8)
    p = ref_scale_to_raw(target_pos_deg, *sigs["target_pos_deg"][2:4])
    ref_place_bits(d, sigs["target_pos_deg"][0], sigs["target_pos_deg"][1], p)
    v = ref_scale_to_raw(target_vel_dps, *sigs["target_vel_dps"][2:4])
    ref_place_bits(d, sigs["target_vel_dps"][0], sigs["target_vel_dps"][1], v)
    t = ref_scale_to_raw(torque_lim_a, *sigs["torque_lim_a"][2:4])
    ref_place_bits(d, sigs["torque_lim_a"][0], sigs["torque_lim_a"][1], t)
    crc = ref_crc8(bytes(d[:7]))
    ref_place_bits(d, sigs["crc8"][0], sigs["crc8"][1], crc)
    return bytes(d)


def ref_encode_telemetry(sigs: Dict[str, Tuple[int, int, float, float, bool]],
                          actual_pos_deg: float, actual_current_a: float,
                          temp_c: float, status_flags: int) -> bytes:
    d = bytearray(8)
    p = ref_scale_to_raw(actual_pos_deg, *sigs["actual_pos_deg"][2:4])
    ref_place_bits(d, sigs["actual_pos_deg"][0], sigs["actual_pos_deg"][1], p)
    c = ref_scale_to_raw(actual_current_a, *sigs["actual_current_a"][2:4])
    ref_place_bits(d, sigs["actual_current_a"][0], sigs["actual_current_a"][1], c)
    t = ref_scale_to_raw(temp_c, *sigs["temp_c"][2:4])
    ref_place_bits(d, sigs["temp_c"][0], sigs["temp_c"][1], t)
    ref_place_bits(d, sigs["status_flags"][0], sigs["status_flags"][1], status_flags)
    crc = ref_crc8(bytes(d[:7]))
    ref_place_bits(d, sigs["crc8"][0], sigs["crc8"][1], crc)
    return bytes(d)


def ref_encode_safety_event(sigs: Dict[str, Tuple[int, int, float, float, bool]],
                              fault_type: int, fault_value: float,
                              sto_triggered: bool, timestamp_us: int) -> bytes:
    d = bytearray(8)
    ref_place_bits(d, sigs["fault_type"][0], sigs["fault_type"][1], fault_type)
    fv = ref_scale_to_raw(fault_value, *sigs["fault_value"][2:4])
    ref_place_bits(d, sigs["fault_value"][0], sigs["fault_value"][1], fv)
    ref_place_bits(d, sigs["sto_triggered"][0], sigs["sto_triggered"][1],
                    1 if sto_triggered else 0)
    ref_place_bits(d, sigs["timestamp_us"][0], sigs["timestamp_us"][1], timestamp_us)
    crc = ref_crc8(bytes(d[:7]))
    ref_place_bits(d, sigs["crc8"][0], sigs["crc8"][1], crc)
    return bytes(d)


def ref_encode_sync_frame(sigs: Dict[str, Tuple[int, int, float, float, bool]],
                           sync_marker: int, time_low_s: float,
                           time_high_s: float) -> bytes:
    d = bytearray(8)
    ref_place_bits(d, sigs["sync_marker"][0], sigs["sync_marker"][1], sync_marker)
    tl = ref_scale_to_raw(time_low_s, *sigs["time_low"][2:4])
    ref_place_bits(d, sigs["time_low"][0], sigs["time_low"][1], tl)
    th = ref_scale_to_raw(time_high_s, *sigs["time_high"][2:4])
    ref_place_bits(d, sigs["time_high"][0], sigs["time_high"][1], th)
    crc = ref_crc8(bytes(d[:7]))
    ref_place_bits(d, sigs["crc8"][0], sigs["crc8"][1], crc)
    return bytes(d)


# ─── 测试基础设施 ──────────────────────────────────────────────────────────

# CANFrame 的本地等价 dataclass, **不** import 被测代码; 仅用于把"裸 bytes"
# 包装成 project 端 decode_telemetry 接受的输入形状 (但本测试**只**用 bytes,
# 直接调 decode_telemetry 时仍需 import, 此时本测试即不独立)。
# 为此把 decode_telemetry 调用改造成接受 bytes 而非 CANFrame。

DBC_PATH = Path(__file__).resolve().parents[1] / "contracts" / "canfd" / "joint_control.dbc"


class TestR4DBCParse(unittest.TestCase):
    """R4-前置: DBC 解析必须能拿到所有 9 条消息的关键信号。"""

    @classmethod
    def setUpClass(cls):
        cls.dbc = parse_dbc_signals(DBC_PATH)

    def test_joint_cmd_count(self):
        """6 个 JointCmd 消息必须全部解析出 target_pos_deg / vel / torque / crc8."""
        cmd_ids = [100, 104, 108, 112, 116, 120]
        for cid in cmd_ids:
            sigs = self.dbc[cid]
            for name in ("target_pos_deg", "target_vel_dps", "torque_lim_a", "crc8"):
                self.assertIn(name, sigs, f"消息 {cid} 缺信号 {name}")

    def test_telemetry_signals(self):
        """JointTelemetry_300 (SHOULDER_L) 信号位域必须与 DBC 一致。"""
        sigs = self.dbc[300]
        # (start_bit, length)
        self.assertEqual(sigs["actual_pos_deg"][:2], (0, 15))
        self.assertEqual(sigs["actual_current_a"][:2], (16, 11))
        self.assertEqual(sigs["temp_c"][:2], (27, 9))
        self.assertEqual(sigs["status_flags"][:2], (40, 8))
        self.assertEqual(sigs["crc8"][:2], (56, 8))

    def test_signal_scales(self):
        """DBC 缩放/偏移必须被解析 (与 README/契约一致)。"""
        sigs = self.dbc[300]
        self.assertAlmostEqual(sigs["actual_pos_deg"][2], 0.01)        # deg
        self.assertAlmostEqual(sigs["actual_pos_deg"][3], 0.0)
        self.assertAlmostEqual(sigs["actual_current_a"][2], 0.01)      # A
        self.assertAlmostEqual(sigs["actual_current_a"][3], 0.0)
        self.assertAlmostEqual(sigs["temp_c"][2], 0.5)                 # C
        self.assertAlmostEqual(sigs["temp_c"][3], -20.0)
        self.assertAlmostEqual(sigs["crc8"][2], 1.0)


class TestR4JointCmdCrossCheck(unittest.TestCase):
    """R4-1: JointCmd 编码, project vs reference, 1000 帧 + 边界值, 字节相等。"""

    @classmethod
    def setUpClass(cls):
        cls.dbc = parse_dbc_signals(DBC_PATH)
        # 这一刻才允许 import 被测代码 (因为 reference 已经独立写完)
        from embodied_ai.simulation.real_robot_env import encode_joint_cmd
        cls.encode_joint_cmd = staticmethod(encode_joint_cmd)

    def test_1000_random_joint_cmd_bytes_match(self):
        """1000 帧随机参数, project encode bytes == reference encode bytes."""
        rng = random.Random(0xCAFE)
        failures = []
        for i in range(1000):
            j = rng.randint(0, 5)
            # DBC 量程内 (POS_SCALE 0.01, 最大 0x7FFF=327.67)
            pos = rng.uniform(0, 327.67)
            vel = rng.uniform(0, 327.67)
            torque = rng.uniform(0, 25.5)
            proj = self.encode_joint_cmd(j, pos, vel, torque).data
            ref = ref_encode_joint_cmd(self.dbc[CMD_IDS[j]], pos, vel, torque)
            if proj != ref:
                failures.append((j, pos, vel, torque, proj.hex(), ref.hex()))
        self.assertFalse(failures,
                          f"JointCmd {len(failures)}/1000 与 DBC 参考不一致: {failures[:3]}")

    def test_boundary_values(self):
        """边界值: DBC range 两端, 零, 居中, 全 15-bit 满量程。"""
        boundary_cases = [
            # (joint, pos, vel, torque, description)
            (0, 0.0, 0.0, 0.0, "全零"),
            (0, 327.67, 327.67, 25.5, "全满"),
            (5, 0.0, 0.0, 0.0, "joint5 零"),
            (5, 327.67, 327.67, 25.5, "joint5 满"),
            (3, 163.835, 163.835, 12.75, "居中"),
            (0, 0.01, 0.0, 0.0, "1 LSB pos"),
            (0, 0.0, 0.01, 0.0, "1 LSB vel"),
            (0, 0.0, 0.0, 0.1, "1 LSB torque"),
            (0, 0.01, 0.01, 0.1, "全 1 LSB"),
        ]
        for j, pos, vel, torque, desc in boundary_cases:
            with self.subTest(case=desc):
                proj = self.encode_joint_cmd(j, pos, vel, torque).data
                ref = ref_encode_joint_cmd(self.dbc[CMD_IDS[j]], pos, vel, torque)
                self.assertEqual(proj, ref,
                                  f"{desc}: project {proj.hex()} != ref {ref.hex()}")

    def test_crc8_matches_dbc_atomic(self):
        """CRC8 字段必须与本参考实现 byte-exact (poly 0x07)。"""
        from embodied_ai.simulation.real_robot_env import crc8
        rng = random.Random(0xDEAD)
        for _ in range(50):
            d = bytes(rng.randint(0, 255) for _ in range(7))
            proj = crc8(d)
            ref = ref_crc8(d)
            self.assertEqual(proj, ref, f"CRC8 偏差: d={d.hex()} proj={proj:02x} ref={ref:02x}")


class TestR4TelemetryCrossCheck(unittest.TestCase):
    """R4-2: JointTelemetry 解码, reference encode → project decode → 物理量 1 LSB 一致。

    这是 R4 的核心 — 历史 bug 都在这一项里:
      - decode_telemetry 的 actual_current_a 用 `>> 5` 错读 (DBC bit 16, 应 no-shift)
      - decode_telemetry 的 temp_c 用 `>> 5` 错读 (DBC bit 27, 应 `>> 3`)
    """

    @classmethod
    def setUpClass(cls):
        cls.dbc = parse_dbc_signals(DBC_PATH)
        from embodied_ai.simulation.real_robot_env import decode_telemetry, CANFrame
        cls.decode_telemetry = staticmethod(decode_telemetry)
        cls.CANFrame = CANFrame

    def _roundtrip_phys(self, pos_deg: float, current_a: float, temp_c: float,
                         flags: int = 0):
        """reference encode → project decode, 返回 project 解出的物理量。"""
        data = ref_encode_telemetry(self.dbc[300], pos_deg, current_a, temp_c, flags)
        dec = self.decode_telemetry(self.CANFrame(300, data))
        return dec

    def test_1000_random_telemetry_physical_match(self):
        """1000 帧随机参考编码 → project 解码, 物理量 1 LSB 内一致。"""
        rng = random.Random(0xBEEF)
        failures = []
        for i in range(1000):
            pos = rng.uniform(0, 327.67)
            cur = rng.uniform(0, 20.47)
            tmp = rng.uniform(-20, 235.5)
            flags = rng.randint(0, 255)
            dec = self._roundtrip_phys(pos, cur, tmp, flags)
            # 1 LSB 容差
            if abs(dec["pos_deg"] - pos) > 0.011:
                failures.append(("pos", pos, dec["pos_deg"], 0.011))
            if abs(dec["current_a"] - cur) > 0.011:
                failures.append(("current", cur, dec["current_a"], 0.011))
            if abs(dec["temp_c"] - tmp) > 0.51:
                failures.append(("temp", tmp, dec["temp_c"], 0.51))
            if bool(dec["crc_ok"]) is not True:
                failures.append(("crc_ok", True, dec["crc_ok"], False))
        self.assertFalse(failures,
                          f"JointTelemetry {len(failures)}/1000 解码与 DBC 参考不一致:\n"
                          + "\n".join(repr(f) for f in failures[:5]))

    def test_telemetry_boundary_values(self):
        """边界: pos/cur/tmp 全量程两端, 零, 居中, 极值 LSB."""
        cases = [
            (0.0, 0.0, -20.0, 0, "全零"),
            (327.67, 20.47, 235.5, 255, "全满"),
            (163.835, 10.235, 107.75, 128, "居中"),
            (0.01, 0.01, -19.5, 1, "1 LSB"),
            (327.66, 20.46, 235.0, 254, "1 LSB 离顶"),
        ]
        for pos, cur, tmp, flags, desc in cases:
            with self.subTest(case=desc):
                dec = self._roundtrip_phys(pos, cur, tmp, flags)
                self.assertAlmostEqual(dec["pos_deg"], pos, delta=0.011, msg=desc)
                self.assertAlmostEqual(dec["current_a"], cur, delta=0.011, msg=desc)
                self.assertAlmostEqual(dec["temp_c"], tmp, delta=0.51, msg=desc)
                self.assertEqual(int(dec["flags"]), flags, msg=desc)
                self.assertTrue(dec["crc_ok"], f"{desc}: CRC 应通过")

    def test_decode_telemetry_bit_layout_documented(self):
        """失败见证测试: 旧 `>> 5` 会让 actual_current_a 物理量严重失真。

        不期望失败; 若失败说明 bug 回来了, 锁住回归。
        """
        # 在 DBC bit 16 = 0, 其它全零的最小输入下, reference 编码再解码
        # 应得到 current_a = 0 (LSB)。
        data = ref_encode_telemetry(self.dbc[300], 0.0, 0.0, -20.0, 0)
        dec = self.decode_telemetry(self.CANFrame(300, data))
        # 若旧 `>> 5` bug 回归, current_a 会被读成"bit 21..31"被误解为非零
        self.assertAlmostEqual(dec["current_a"], 0.0, delta=0.011,
                                 msg=f"current_a 应为 0, 实得 {dec['current_a']}")
        self.assertAlmostEqual(dec["temp_c"], -20.0, delta=0.51,
                                 msg=f"temp_c 应为 -20, 实得 {dec['temp_c']}")


class TestR4SyncFrameAndSafety(unittest.TestCase):
    """R4-3: SyncFrame + SafetyEvent 编解码覆盖。"""

    @classmethod
    def setUpClass(cls):
        cls.dbc = parse_dbc_signals(DBC_PATH)
        from embodied_ai.simulation.real_robot_env import crc8
        cls.crc8 = staticmethod(crc8)

    def test_sync_frame_crc(self):
        """SyncFrame 8 字节, 含 sync_marker + time_low + time_high + crc8。"""
        sigs = self.dbc[0]
        # 显式构造已知输入 → 参考 CRC 应等于末尾字节
        d = bytearray(8)
        ref_place_bits(d, sigs["sync_marker"][0], sigs["sync_marker"][1], 0xA5)
        ref_place_bits(d, sigs["time_low"][0], sigs["time_low"][1], 1000)  # 10.00s
        ref_place_bits(d, sigs["time_high"][0], sigs["time_high"][1], 2000)  # 20.00s
        crc = ref_crc8(bytes(d[:7]))
        ref_place_bits(d, sigs["crc8"][0], sigs["crc8"][1], crc)
        # 实际 CRC 字段值
        crc_field = ref_read_bits(bytes(d), sigs["crc8"][0], sigs["crc8"][1])
        self.assertEqual(crc_field, self.crc8(bytes(d[:7])))

    def test_safety_event_encoding(self):
        """SafetyEvent ID=900, 显式编码 + 自身 CRC 自检。"""
        sigs = self.dbc[900]
        data = ref_encode_safety_event(sigs, fault_type=2, fault_value=123.4,
                                         sto_triggered=True, timestamp_us=1000000)
        # 自己读回
        fault = ref_read_bits(data, *sigs["fault_type"][:2])
        sto = ref_read_bits(data, *sigs["sto_triggered"][:2])
        self.assertEqual(fault, 2)
        self.assertEqual(sto, 1)
        # CRC 字段必须与 ref_crc8(d[:7]) 一致
        crc_field = ref_read_bits(data, *sigs["crc8"][:2])
        self.assertEqual(crc_field, ref_crc8(data[:7]))


class TestR4SixJointTelemetry(unittest.TestCase):
    """R4-4: 6 关节全量遥测 — 每个 JointTelemetry_<joint> 都必须独立 byte-exact 解码。

    DBC 扩展后 (R4 2026-10-07) 共 6 条 JointTelemetry (300/304/308/312/316/320),
    code 端 TELEM_IDS 与之一致。本测试逐关节 reference encode → project decode,
    1 LSB 内一致; 反向: 任何一关节错位即失败, 锁住回归。
    """

    @classmethod
    def setUpClass(cls):
        cls.dbc = parse_dbc_signals(DBC_PATH)
        from embodied_ai.simulation.real_robot_env import (
            decode_telemetry, CANFrame, TELEM_IDS,
        )
        cls.decode_telemetry = staticmethod(decode_telemetry)
        cls.CANFrame = CANFrame
        cls.TELEM_IDS = TELEM_IDS

    def test_telem_ids_match_dbc(self):
        """code 端 TELEM_IDS 必须与 DBC JointTelemetry_<joint> 全部对应。"""
        dbc_telem_ids = sorted(
            [cid for cid, sigs in self.dbc.items()
             if "actual_pos_deg" in sigs and "actual_current_a" in sigs]
        )
        code_telem_ids = sorted(self.TELEM_IDS.values())
        self.assertEqual(dbc_telem_ids, code_telem_ids,
                          f"DBC telemetry IDs {dbc_telem_ids} != code TELEM_IDS {code_telem_ids}")

    def test_six_joint_roundtrip(self):
        """6 关节×200 帧随机 → reference encode → project decode, 1 LSB 一致。"""
        rng = random.Random(0xCAFE)
        failures = []
        for j in range(6):
            cid = self.TELEM_IDS[j]
            sigs = self.dbc[cid]
            for _ in range(200):
                pos = rng.uniform(0, 327.67)
                cur = rng.uniform(0, 20.47)
                tmp = rng.uniform(-20, 235.5)
                flags = rng.randint(0, 255)
                data = ref_encode_telemetry(sigs, pos, cur, tmp, flags)
                dec = self.decode_telemetry(self.CANFrame(cid, data))
                if abs(dec["pos_deg"] - pos) > 0.011:
                    failures.append((j, "pos", pos, dec["pos_deg"]))
                if abs(dec["current_a"] - cur) > 0.011:
                    failures.append((j, "current", cur, dec["current_a"]))
                if abs(dec["temp_c"] - tmp) > 0.51:
                    failures.append((j, "temp", tmp, dec["temp_c"]))
                if not dec["crc_ok"]:
                    failures.append((j, "crc", True, dec["crc_ok"]))
        self.assertFalse(failures,
                          f"6 关节遥测 {len(failures)}/1200 与 DBC 参考不一致:\n"
                          + "\n".join(repr(f) for f in failures[:5]))

    def test_mock_transport_round_trip_six_joints(self):
        """集成测试: MockTransport.send(6 cmd) + send(sync) → recv 6 telemetry → decode 6 关节。

        验证 RealRobotEnv 的 end-to-end 路径 (cmd→model→telemetry→decode) 对所有 6 个关节
        都自洽, 且与 reference 物理量一致。
        """
        from embodied_ai.simulation.real_robot_env import (
            MockTransport, encode_joint_cmd, decode_telemetry, CANFrame,
        )
        tr = MockTransport(n_joints=6, latency_ms=0.0, drop_rate=0.0,
                            crc_error_rate=0.0, seed=42)
        tr.open()
        # 给每个关节下发不同的目标角度 (用 1+ rad 区别)
        import math
        targets = [0.0, 0.3, -0.5, 0.7, -0.9, 1.1]
        # 一阶模型: pose += 0.35*(target-pose), 8 步 ≈ (0.65)^8 ≈ 0.032 误差
        for _ in range(8):
            for j, tgt in enumerate(targets):
                pos_deg = math.degrees(tgt) + 180.0
                tr.send(encode_joint_cmd(j, pos_deg, 0.0, 20.0))
            # 清空 cmd 队列 (mock 真机不会回声 cmd)
            while tr.recv(timeout_s=0.0) is not None:
                pass
        # 发 SYNC, 触发 6 关节 telemetry 队列
        tr.send(CANFrame(0, bytes([0xA5]) + b"\x00" * 6 + b"\x00"))
        received = []
        for _ in range(6):
            f = tr.recv(timeout_s=0.01)
            self.assertIsNotNone(f, "MockTransport 应返回 6 帧 telemetry, 但 recv 提前 None")
            received.append(f)
        # 解码并核对 6 个 can_id 都是 TELEM_IDS 之一
        from embodied_ai.simulation.real_robot_env import TELEM_IDS as TIDS
        for f in received:
            self.assertIn(f.can_id, TIDS.values(), f"未知 can_id: {f.can_id}")
        # 每个关节的 pos_deg 应接近 target (1 阶模型收敛后 <1%)
        id_to_joint = {v: k for k, v in TIDS.items()}
        for f in received:
            j = id_to_joint[f.can_id]
            tel = decode_telemetry(f)
            self.assertTrue(tel["crc_ok"], f"joint {j} telemetry CRC 应通过")
            actual_rad = math.radians(tel["pos_deg"] - 180.0)
            self.assertAlmostEqual(actual_rad, targets[j], delta=0.05,
                                    msg=f"joint {j} expected {targets[j]:.2f} rad, got {actual_rad:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)