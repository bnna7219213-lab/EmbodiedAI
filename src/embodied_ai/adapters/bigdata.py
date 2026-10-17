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


#: 只有这些 source 才算**实测**遥测注入 (provenance = measured)。
#: 其余 (如 gen_d4_dataset.py 的 "replay_gen"、data-flywheel 引擎的
#: "edalab_flywheel"、ingest 默认值 "canfd_frame"/"unknown") 一律归 synthetic。
#: 单一真源: `export_training_jsonl()` 据此推导 provenance, 不再由导出器断言
#: 字段"是真实的" —— 此前 real_fields 标签与自家 provenance 四选一铁律冲突。
#: models/replay_dataset.py 复用本常量, 两处 vocabulary 不得各写一套
#: (tests/test_d4_pipeline.py 有交叉一致性守护)。
MEASURED_SOURCES = ("hil", "canfd_live", "bench_inject",
                    "real", "telemetry", "measured")

#: 遥测物理字段 (参与 provenance 归类; 存在 ≠ 实测)
TELEMETRY_FIELDS = ("motor_current_a", "temperature_c", "actual_pos_deg",
                    "actual_current_a", "temp_c")


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
        """消费遥测帧 — 同时接受 DBC BO_300 字段名与规范化字段名 (双命名兼容)。

        DBC 别名: actual_current_a→motor_current_a, temp_c→temperature_c
        (修复: 此前只认 DBC 名, 规范化名静默丢失 — gen_d4_dataset 触发)

        frame 中超出 TelemetryRecord schema 的字段 (如 episode_id, scenario,
        n_steps 等 D4 扩展元数据) 自动透传, 进入 records 的 dict 表示。
        """
        motor = frame.get("motor_current_a", frame.get("actual_current_a"))
        temp = frame.get("temperature_c", frame.get("temp_c"))
        rec = TelemetryRecord(
            timestamp_s=float(frame.get("timestamp_s", time.time())),
            joint_name=frame.get("joint_name", joint),
            motor_current_a=motor,
            temperature_c=temp,
            actual_pos_deg=frame.get("actual_pos_deg"),
            fault_events=frame.get("fault_events", []),
            source=frame.get("source", source),
            board_serial=frame.get("board_serial", board_serial),
        )
        row = rec.as_dict()
        # 透传 frame 中的扩展元数据字段 (D4 episode_id / scenario 等)
        schema_keys = set(row.keys())
        for k, v in frame.items():
            if k not in schema_keys:
                row[k] = v
        self.records.append(row)
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

    # ------------------------------------------------------------------ #
    # D4 导出: 遥测 → 训练数据集文件 (replay_dataset 消费)
    # ------------------------------------------------------------------ #
    def export_training_jsonl(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """导出 D2 遥测为 D4 训练用 jsonl (含 provenance 头行)。

        格式: 第一行 {"_schema": "d4_replay_v1", "_provenance": {...}}; 后续为记录行。
        消费方: embodied_ai.models.replay_dataset.load_telemetry_jsonl。
        """
        path = Path(path) if path else self.output_dir / "training_replay_v1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        sources = sorted({str(r.get("source", "")) for r in self.records})
        # 字段"存在"不等于字段"实测"。此前这里无条件把 motor_current_a /
        # temperature_c / actual_pos_deg 标成 real_fields, 而 scripts/
        # gen_d4_dataset.py 产出的那三个字段全是 rng 剧本 (source="replay_gen"),
        # 且全仓硬件实物为零 —— 电机电流/温度不可能有实测值。该标签直接违反项目
        # 铁律"所有新物理参数标注 provenance: measured/assumed/synthetic/待领域
        # 验证 四选一"。现在 provenance 由记录的 source 推导, 不再由导出器断言。
        measured = bool(sources) and all(s in MEASURED_SOURCES for s in sources)
        present = [f for f in TELEMETRY_FIELDS
                   if any(r.get(f) is not None for r in self.records)]
        header = {
            "_schema": "d4_replay_v1",
            "_provenance": {
                "source_records": len(self.records),
                "record_sources": sources,
                "telemetry_fields": present,
                "telemetry_provenance": "measured" if measured else "synthetic",
                "synthetic_fields": (["rgb", "lang"] if measured
                                     else sorted(set(present) | {"rgb", "lang"})),
                "generation": ("hardware/board ingest" if measured else
                               "rng-script (scripts/gen_d4_dataset.py) — 无物理引擎、"
                               "无硬件在环; 场景形状由手写规则 + 均匀/高斯扰动构成"),
                "joints": sorted({r.get("joint_name", "unknown") for r in self.records}),
                "policy_binding": {
                    "hw_rev": "SF-MDB-0001",
                    "batch_sn": sorted({r.get("board_serial", "unknown") for r in self.records}),
                    # 板卡序列号是生成器写死的占位符, 不代表真实批次
                    "binding_provenance": "declared" if measured else "synthetic",
                },
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(_json.dumps(header, ensure_ascii=False) + "\n")
            for r in self.records:
                f.write(_json.dumps(r, ensure_ascii=False) + "\n")
        return {"path": str(path), "n_records": len(self.records), "schema": "d4_replay_v1"}
