"""bigdata 适配器 — 飞轮最后一公里: 遥测采集 → DR → 训练配置 (B3)。

五段契约末端 (fusion-absorption-plan 4.2.1):
  真机/HIL 遥测 (D2) → ingest_telemetry 规范化
    → EdalabAdapter.generate_dr_config (D3→D4: ICT/FCT/遥测 → DR 分布)
    → replay_to_training_config (D4→训练: DR yaml + 回放清单 + 策略版本)
    → RL 训练管线直接消费, 全程无人工

数据来源格式:
  - EdaLab ManufacturingDataFlywheel 输出 (data-flywheel 引擎 stdout/JSON 遥测记录)
  - HIL 遥测 schema (d2_hil_telemetry, contracts/hw_requirements.data_flywheel)
  - 真机 CAN-FD JointTelemetry 帧 (contracts/canfd/joint_control.dbc BO_300)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class TelemetryRecord:
    """规范化遥测记录 (D2 schema)。"""
    timestamp_s: float
    joint_name: str
    motor_current_a: Optional[float] = None
    temperature_c: Optional[float] = None
    actual_pos_deg: Optional[float] = None
    fault_events: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "hil"  # hil | fleet | canfd_frame
    board_serial: str = "unknown"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s, "joint_name": self.joint_name,
            "motor_current_a": self.motor_current_a, "temperature_c": self.temperature_c,
            "actual_pos_deg": self.actual_pos_deg, "fault_events": self.fault_events,
            "source": self.source, "board_serial": self.board_serial,
        }


class BigdataAdapter:
    """遥测 → DR → 训练配置 飞轮末段适配器。

    用法:
        bd = BigdataAdapter()
        bd.ingest_canfd_frame({"actual_current_a": 12.3, "temp_c": 45.2, "status_flags": 0},
                               joint="shoulder_L")
        bd.ingest_hil_csv(Path("output/hil/telemetry.csv"))     # 或批量
        training_cfg = bd.flywheel_to_training(strategy_v="v2.0")
        # training_cfg["dr_yaml_path"] 可直接喂 RL 训练管线
    """

    def __init__(self, output_dir: Path = Path("output/dr")):
        self.output_dir = Path(output_dir)
        self.records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 采集入口 (D2 → 规范化)
    # ------------------------------------------------------------------ #
    def ingest_canfd_frame(self, frame: Dict[str, Any], joint: str = "unknown",
                           board_serial: str = "unknown", source: str = "canfd_frame") -> TelemetryRecord:
        """消费 DBC BO_300 JointTelemetry 帧字段 (contracts/canfd)。"""
        rec = TelemetryRecord(
            timestamp_s=float(frame.get("timestamp_s", time.time())),
            joint_name=joint,
            motor_current_a=frame.get("actual_current_a"),
            temperature_c=frame.get("temp_c"),
            actual_pos_deg=frame.get("actual_pos_deg"),
            fault_events=frame.get("fault_events", []),
            source=source, board_serial=board_serial,
        )
        self.records.append(rec.as_dict())
        return rec

    def ingest_hil_telemetry(self, rows: List[Dict[str, Any]]) -> int:
        """批量消费 HIL 遥测行 (d2_hil_telemetry schema: motor_current_a/temperature_c/fault_events)。"""
        n = 0
        for r in rows:
            self.ingest_canfd_frame(r, joint=r.get("joint_name", "unknown"),
                                    board_serial=r.get("board_serial", "unknown"), source="hil")
            n += 1
        return n

    def ingest_edalab_flywheel_output(self, flywheel_result: Dict[str, Any]) -> int:
        """消费 EdaLab ManufacturingDataFlywheel 的输出结构 (Board Performance Report / telemetry records)。"""
        n = 0
        for brd in flywheel_result.get("boards", []):
            for tel in brd.get("telemetry_records", []):
                self.ingest_canfd_frame(tel, board_serial=brd.get("board_id", "unknown"),
                                        source="edalab_flywheel")
                n += 1
        # 兼容直接挂顶层的 records
        for tel in flywheel_result.get("telemetry_records", []):
            self.ingest_canfd_frame(tel, source="edalab_flywheel")
            n += 1
        return n

    # ------------------------------------------------------------------ #
    # 飞轮末端 (D4 → 训练配置)
    # ------------------------------------------------------------------ #
    def flywheel_to_training(self, strategy_v: str = "v2.0",
                             ict_data: Optional[List[Dict]] = None,
                             fct_data: Optional[List[Dict]] = None,
                             dr_params_path: Optional[Path] = None) -> Dict[str, Any]:
        """遥测 → EdalabAdapter.generate_dr_config → 训练配置包 (五段契约末端)。

        返回可直接被 RL 管线消费的配置:
          dr_yaml_path:   DR 分布 yaml (制造容差 + 遥测推断)
          replay_records: 归一化回放清单 (策略版本绑定)
          policy_binding: (policy_ver, hw_rev, batch) 三元组 (创新 #3)
        """
        from .edalab import EdalabAdapter
        adapter = EdalabAdapter()

        dr_config = adapter.generate_dr_config(
            strategy_v=strategy_v, ict_data=ict_data, fct_data=fct_data,
            telemetry_data=self.records if self.records else None,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dr_yaml_path = adapter.export_dr_yaml(dr_config, self.output_dir / f"dr_{strategy_v}.yaml")

        serials = sorted({r["board_serial"] for r in self.records if r.get("board_serial") != "unknown"})
        training_cfg = {
            "dr_yaml_path": str(dr_yaml_path),
            "dr_overrides": dr_config["overrides"],
            "recommended_training_epochs": dr_config["recommended_training_epochs"],
            "recommended_lr_multiplier": dr_config["recommended_lr_multiplier"],
            "replay_records": len(self.records),
            "replay_schema": "d4_replay_v1",
            "policy_binding": {
                "policy_ver": strategy_v,
                "hw_rev": "SF-MDB-0001",
                "batch_sn": serials[0] if len(serials) == 1 else f"{len(serials)}_boards",
            },
            "flywheel_generated_at": time.time(),
        }
        return training_cfg

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def dump_records(self, path: Optional[Path] = None) -> Path:
        """遥测记录落盘 (D2 存档, JSON lines)。"""
        path = Path(path) if path else self.output_dir / "telemetry_records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path
