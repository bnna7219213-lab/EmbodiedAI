"""EdaLab 适配器 — 打通 11 步主链调用 + dr_params 制造容差 DR 落地 (K2 创新 #1)。

桥接 EmbodiedAI Python 层 (RL 训练/策略层) ↔ EdaLab Kotlin 硬件设计引擎 (11 步主链 + ManufacturingDataFlywheel)。

11 步主链 (对应 edalab_stages.py KIND_FLOWS + ForgeCore 新增):
  1. forge_core     — AI策略→硬件需求翻译
  2. thermal_sim    — 热仿真收敛
  3. pi_analysis    — PDN 电源完整性
  4. si_analysis    — 信号完整性
  5. emc_precomp    — EMC 预合规
  6. motor_drive    — FOC 电机驱动 PCB 自动设计
  7. mcad_co_design — 机电协同布局
  8. foc_firmware   — FOC 固件生成 (C)
  9. canfd_protocol — CAN-FD PDO 映射 + DCF
  10. coupling_analysis — 三域联合仿真 (MultiPhysicsCoupler)
  11. data_flywheel — 制造数据回流 → DR 配置生成

飞轮闭环真机数据流:
  产线 ICT/FCT → ManufacturingDataFlywheel.ingest → generateDrConfig → exportDrYaml
  → RL domain_randomization → 新策略 → 部署 → 产线 (闭环)

依赖:
  - edalab CLI 已 build 且 PATH 中可调用
  - contracts/dr_params/manufacturing_variance.yaml 已就位
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # 降级处理


# 11 步主链的阶段注册顺序 (与 edalab_stages.py KIND_FLOWS + ForgeCore 引擎对齐)
FLOW_CHAIN: List[Dict[str, Any]] = [
    {"name": "forge_core",      "category": "ai_input",    "est_sec": 5,  "engine": "ForgeCore.kt"},
    {"name": "thermal_sim",     "category": "simulation",  "est_sec": 8,  "engine": "ThermalSolver.kt"},
    {"name": "pi_analysis",     "category": "simulation",  "est_sec": 10, "engine": "PowerIntegritySolver.kt"},
    {"name": "si_analysis",     "category": "simulation",  "est_sec": 10, "engine": "SignalIntegritySolver.kt"},
    {"name": "emc_precomp",     "category": "simulation",  "est_sec": 8,  "engine": "EMCAnalyzer.kt"},
    {"name": "motor_drive",     "category": "design",      "est_sec": 15, "engine": "MotorDriveEngine.kt"},
    {"name": "mcad_co_design",  "category": "co-design",   "est_sec": 12, "engine": "MCADCoDesignEngine.kt"},
    {"name": "foc_firmware",    "category": "firmware",    "est_sec": 6,  "engine": "FOCFirmwareGenerator.kt"},
    {"name": "canfd_protocol",  "category": "protocol",    "est_sec": 5,  "engine": "CommProtocolGenerator.kt"},
    {"name": "coupling_analysis", "category": "coupling", "est_sec": 12, "engine": "MultiPhysicsCoupler.kt"},
    {"name": "data_flywheel",   "category": "mfg_telemetry","est_sec": 20, "engine": "ManufacturingDataFlywheel.kt"},
]

# 默认项目根
EDALAB_ROOT = Path(r"C:\Users\bnna7\workspace\EdaLab")
EDALAB_CLI = EDALAB_ROOT / "edalabcore" / "edalab"  # CLI 入口 (Kotlin jar 或 gradlew)


class EdalabAdapter:
    """EdaLab Python ↔ Kotlin 桥接适配器。

    用法:
        adapter = EdalabAdapter()
        result = adapter.run_stage("forge_core", {"control_freq_hz": 20, "compute_tops": 30})
        dr = adapter.generate_dr_config(strategy_v="v2.0")
        adapter.export_dr_yaml(dr, Path("output/dr_config.yaml"))
    """

    def __init__(self, edalab_root: Path = EDALAB_ROOT, verbose: bool = False):
        self.root = edalab_root
        self.verbose = verbose
        self._stage_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 11 步主链 — 单阶段触发
    # ------------------------------------------------------------------ #

    def run_stage(self, stage_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """触发单阶段 (通过 edalab CLI subprocess 或 direct function call)。

        K1-K2 过渡期实现: 先验证 11 步元数据完整性 + 参数契约匹配；
        K3 落地期: subprocess 调 CLI jar 或 gRPC/HTTP 桥。
        """
        stage = next((s for s in FLOW_CHAIN if s["name"] == stage_name), None)
        if stage is None:
            raise ValueError(f"未知阶段: {stage_name}. 合法: {[s['name'] for s in FLOW_CHAIN]}")

        rec = {
            "stage": stage_name,
            "engine_file": stage["engine"],
            "params": params,
            "status": "QUEUED",
            "started_at": time.time(),
        }

        # 参数契约检查
        contract_ok, contract_msg = self._check_param_contract(stage_name, params)
        if not contract_ok:
            rec["status"] = "CONTRACT_FAIL"
            rec["error"] = contract_msg
            self._stage_log.append(rec)
            return rec

        # K1-K2: 验证引擎文件存在 + 参数契约匹配 → READY
        engine_path = self._engine_path(stage["engine"])
        if engine_path and engine_path.exists():
            rec["status"] = "READY"
            rec["engine_resolved"] = str(engine_path)
        else:
            rec["status"] = "ENGINE_MISSING"
            rec["engine_resolved"] = None
            if self.verbose:
                print(f"  ⚠ 引擎文件未找到: {stage['engine']} (K3 编译后生效)")

        rec["finished_at"] = time.time()
        self._stage_log.append(rec)
        return rec

    def run_full_chain(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """顺序执行完整 11 步主链，返回每阶段结果。"""
        results = []
        cumulative = dict(params)  # 前序阶段输出传递到后续
        for stage in FLOW_CHAIN:
            merged = {**cumulative, **params.get(stage["name"], {})}
            result = self.run_stage(stage["name"], merged)
            results.append(result)
            if result["status"] in ("CONTRACT_FAIL",):
                break  # 失败止链
            # 模拟前序输出传递
            cumulative[f"{stage['name']}_ok"] = True
        return results

    # ------------------------------------------------------------------ #
    # DR 配置生成 — 制造容差驱动的域随机化 (创新 #1)
    # ------------------------------------------------------------------ #

    def generate_dr_config(self, strategy_v: str = "v1.0",
                           ict_data: Optional[List[Dict]] = None,
                           fct_data: Optional[List[Dict]] = None,
                           telemetry_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """基于 ICT/FCT/遥测数据生成 RL DR 配置。

        实际生产路径: 调用 ManufacturingDataFlywheel.generateDrConfig + exportDrYaml。
        K2 实现: Python 侧独立生成逻辑 (与 Kotlin 引擎算法对齐，供 RL 训练管线直接消费)。
        """
        overrides: Dict[str, Dict[str, Any]] = {}
        based_serials: List[str] = []

        # 辅助变量预定义 (避免条件分支未赋值导致的 NameError)
        flat: List[Dict] = []
        faults: List[List[Dict]] = []
        by_type: Dict[str, int] = {}

        # 1. ICT → 编码器/电流传感器噪声
        if ict_data:
            based_serials.extend(d.get("board_serial", "unknown") for d in ict_data)
            encoder_vals = []
            current_vals = []
            for d in ict_data:
                for k, v in d.get("component_values", {}).items():
                    if k.startswith("ENC"):
                        encoder_vals.append(v)
                    elif "CURR" in k or k.startswith("CS_"):
                        current_vals.append(v)
            if encoder_vals:
                mean = sum(encoder_vals) / len(encoder_vals)
                std = self._std(encoder_vals)
                overrides["encoder_noise_std"] = {
                    "distribution": "normal",
                    "min": max(0, mean - 3 * std),
                    "max": mean + 3 * std,
                    "mean": mean,
                    "std": std,
                    "rationale": f"基于 {len(encoder_vals)} 个编码器 ICT 实测偏差",
                }
            if current_vals:
                mean_curr = sum(current_vals) / len(current_vals)
                overrides["current_sensor_noise_std"] = {
                    "distribution": "uniform",
                    "min": 0,
                    "max": mean_curr * 0.05,
                    "rationale": f"基于 {len(current_vals)} 个电流传感器 ICT 实测偏差",
                }

        # 2. FCT → 通信延迟
        if fct_data:
            based_serials.extend(d.get("board_serial", "unknown") for d in fct_data)
            can_utils = [d["comm_tests"]["can_bus_utilization"]
                         for d in fct_data if "comm_tests" in d and "can_bus_utilization" in d["comm_tests"]]
            if can_utils:
                avg = sum(can_utils) / len(can_utils)
                overrides["canfd_latency_ms"] = {
                    "distribution": "uniform",
                    "min": 0,
                    "max": min(avg, 2.0),
                    "rationale": f"基于 {len(can_utils)} 块板子 FCT CAN-FD 利用率",
                }

        # 3. 遥测 → 负载/摩擦/温度/故障注入
        if telemetry_data:
            torques = [d["motor_current_a"] for d in telemetry_data if "motor_current_a" in d]
            temps = [d["temperature_c"] for d in telemetry_data if "temperature_c" in d]
            faults = [d.get("fault_events", []) for d in telemetry_data]

            if torques:
                t_std = self._std(torques)
                overrides["load_torque_scale"] = {
                    "distribution": "loguniform",
                    "min": max(0.5, 1.0 - t_std),
                    "max": 1.0 + t_std,
                    "rationale": f"基于 {len(torques)} 条遥测电流",
                }
                overrides["friction_coefficient"] = {
                    "distribution": "uniform",
                    "min": 0,
                    "max": t_std * 2,
                    "rationale": "摩擦 (从部署电流变化推断)",
                }

            if temps:
                overrides["ambient_temp"] = {
                    "distribution": "uniform",
                    "min": 15,
                    "max": min(max(temps), 45),
                    "rationale": f"部署最高环境温度: {max(temps):.1f}°C",
                }

            # 故障率 → 注入
            if faults:
                flat = [f for sub in faults for f in sub]
                total = max(len(telemetry_data), 1)
                by_type: Dict[str, int] = {}
                for f in flat:
                    by_type[f.get("fault_type", "UNKNOWN")] = by_type.get(f.get("fault_type", "UNKNOWN"), 0) + 1
                for ftype, count in by_type.items():
                    rate = count / total
                    if rate > 0.01:
                        overrides[f"fault_inject_{ftype.lower()}"] = {
                            "distribution": "uniform",
                            "min": rate * 0.5,
                            "max": rate * 2,
                            "rationale": f"真机故障率: {rate * 100:.1f}%",
                        }

        return {
            "config_version": strategy_v,
            "based_on_serials": based_serials,
            "overrides": overrides,
            "recommended_training_epochs": 200 + min(len(flat) if faults else 0, 1000),
            "recommended_lr_multiplier": 2.0 if ( faults and any(
                (count / max(len(telemetry_data), 1)) > 0.1
                for count in by_type.values()
            )) else 1.0 if faults else 1.0,
            "generated_at": time.time(),
        }

    def export_dr_yaml(self, config: Dict[str, Any], out_path: Path) -> Path:
        """导出 DR 配置为 YAML (与 Kotlin exportDrYaml 格式对齐)。"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if yaml is not None:
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        else:
            # 无 PyYAML 降级: 手写简单 YAML 输出
            lines = self._to_yaml(config, indent=0)
            out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def load_contract_dr_params(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """加载 contracts/dr_params/manufacturing_variance.yaml 作为 DR 基准。

        优先复用 config.load_contract (已内建 mini-YAML 解析器适配无 PyYAML 环境)；
        仅在 path 显式指定时走独立文件读取。
        """
        if path is None:
            # 复用 config 加载器 (已支持 PyYAML 缺失时的 mini-parser)
            from ..config import load_contract
            return load_contract("dr_params/manufacturing_variance")
        if not path.exists():
            raise FileNotFoundError(f"DR params 契约缺失: {path}")
        if yaml is not None:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            from ..config.loader import _yaml_load
            return _yaml_load(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ #
    # 端到端飞轮验证
    # ------------------------------------------------------------------ #

    def flywheel_e2e(self, ict_data=None, fct_data=None, telemetry_data=None,
                     output_dir: Path = Path("output/dr")) -> Dict[str, Any]:
        """端到端飞轮: 制造数据 → DR → YAML 输出。"""
        # Step 1: 生成 DR 配置
        config = self.generate_dr_config(
            strategy_v="v2.0_e2e",
            ict_data=ict_data,
            fct_data=fct_data,
            telemetry_data=telemetry_data,
        )

        # Step 2: 导出 YAML
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = self.export_dr_yaml(config, output_dir / "dr_config.yaml")

        # Step 3: 加载制造容差契约做差异比对
        contract = self.load_contract_dr_params()
        diff = self._diff_vs_contract(config, contract)

        return {
            "config": config,
            "yaml_path": str(yaml_path),
            "contract_name": "manufacturing_variance",
            "variance_diff": diff,
            "flywheel_ok": True,
        }

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #

    def _engine_path(self, engine_file: str) -> Optional[Path]:
        base = self.root / "edalabcore" / "src" / "main" / "kotlin" / "edalab" / "core"
        for subdir in ["design", "physics", "simulation", "safety", "bms", "hil", "ai", "mfg"]:
            candidate = base / subdir / engine_file
            if candidate.exists():
                return candidate
        # coupling 包
        candidate = base / "coupling" / engine_file
        if candidate.exists():
            return candidate
        return None

    def _check_param_contract(self, stage: str, params: Dict) -> tuple:
        """参数契约校验 (K2 基础版, K3 对接 JSON Schema)。"""
        # 11 步共享核心参数
        shared = {"strategy_version", "control_freq_hz", "compute_tops"}
        if "control_freq_hz" in params and params["control_freq_hz"] > 10000:
            return False, "control_freq_hz 超出合理范围 (最大 10kHz)"
        return True, "OK"

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    def _to_yaml(self, obj: Any, indent: int) -> List[str]:
        """极简 YAML 序列化 (无 PyYAML 时降级)。"""
        lines = []
        prefix = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{k}:")
                    lines.extend(self._to_yaml(v, indent + 1))
                else:
                    lines.append(f"{prefix}{k}: {v}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.extend(self._to_yaml(item, indent + 1))
                else:
                    lines.append(f"{prefix}- {item}")
        return lines

    def _diff_vs_contract(self, config: Dict, contract: Dict) -> Dict[str, Any]:
        """DR 配置 vs 制造容差契约的差异报告。"""
        diff: Dict[str, Any] = {"extra_in_dr": [], "missing_from_dr": [], "within_spec": []}
        contract_electrical = contract.get("electrical", {})
        config_overrides = config.get("overrides", {})
        for param_name in contract_electrical:
            yaml_key = param_name
            if yaml_key in config_overrides:
                diff["within_spec"].append(param_name)
            else:
                diff["missing_from_dr"].append(param_name)
        for param_name in config_overrides:
            if param_name not in contract_electrical and param_name not in ("canfd_latency_ms", "load_torque_scale", "friction_coefficient", "ambient_temp"):
                diff["extra_in_dr"].append(param_name)
        return diff

    @property
    def stage_log(self) -> List[Dict[str, Any]]:
        return list(self._stage_log)
