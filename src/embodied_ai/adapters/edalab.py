"""EdaLab 适配器 — 11 步主链真调 + dr_params 制造容差 DR 落地 (K2 创新 #1)。

桥接 EmbodiedAI Python 层 (RL 训练/策略层) ↔ EdaLab Kotlin 硬件设计引擎。

真实 11 步主链 (与 EdaCli.cmdEmbodiedAiFull 逐字对齐, B2 口径统一 2026-09-20):
  1.  forge-core      — AI策略→硬件需求翻译   (EdaCli L551)
  2.  mass-props      — 质量/惯性/重心/续航    (L554)
  3.  structural-fem  — 结构FEM/模态/疲劳      (L557)
  4.  kinematics      — 运动学/动力学/工作空间 (L560)
  5.  thermal-sim     — 热仿真                 (L563)
  6.  pi-analysis     — 电源完整性             (L566)
  7.  motor-drive     — FOC 电机驱动 PCB       (L569)
  8.  data-flywheel   — 制造数据闭环           (L572)
  9.  relay-clutch    — 继电器/离合器安全断触链 (L575)
  10. em-coupling     — 机电耦合→PCB加速度谱   (L578)
  11. multi-physics   — 三域耦合联合仿真       (L581)

live 模式: java -cp <jar>;<kotlin-stdlib...> edalab.core.cli.EdaCli <cli-name>
  (2026-09-20 实测: relay-clutch PASS, UTF-8 正常)
dry 模式 (默认): 元数据校验 + 引擎文件存在性检查 (K2 早期行为, 保留为 CI 快速路径)
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
    yaml = None


# ------------------------------------------------------------------ #
# B2: 真实 11 步主链 (以 EdaCli.kt cmdEmbodiedAiFull 为唯一权威)
# ------------------------------------------------------------------ #
# cli: EdaCli 子命令名 (live 调用名) — 与 EdaCli.kt dispatch 逐字一致
# engine: Kotlin 引擎源文件 (dry 模式存在性探测)
# cli_args: 主链调用时 EdaCli 实际传入的参数 (L551-581 抄录)
FLOW_CHAIN: List[Dict[str, Any]] = [
    {"step": 1,  "name": "forge_core",     "cli": "forge-core",     "engine": "ForgeCore.kt",
     "cli_args": ["v1.0", "household_grasp", "20", "30", "25", "canfd", "5"]},
    {"step": 2,  "name": "mass_props",     "cli": "mass-props",     "engine": "MassPropertiesEngine.kt", "cli_args": []},
    {"step": 3,  "name": "structural_fem", "cli": "structural-fem", "engine": "StructuralFEMEngine.kt", "cli_args": []},
    {"step": 4,  "name": "kinematics",     "cli": "kinematics",     "engine": "KinematicsDynamicsEngine.kt", "cli_args": []},
    {"step": 5,  "name": "thermal_sim",    "cli": "thermal-sim",    "engine": "ThermalSolver.kt",
     "cli_args": ["40", "40", "2", "40"]},
    {"step": 6,  "name": "pi_analysis",    "cli": "pi-analysis",    "engine": "PowerIntegritySolver.kt", "cli_args": []},
    {"step": 7,  "name": "motor_drive",    "cli": "motor-drive",    "engine": "MotorDriveEngine.kt", "cli_args": []},
    {"step": 8,  "name": "data_flywheel",  "cli": "data-flywheel",  "engine": "ManufacturingDataFlywheel.kt", "cli_args": []},
    {"step": 9,  "name": "relay_clutch",   "cli": "relay-clutch",   "engine": "RelayClutchDynamicsEngine.kt", "cli_args": []},
    {"step": 10, "name": "em_coupling",    "cli": "em-coupling",    "engine": "ElectromechanicalCouplingEngine.kt", "cli_args": []},
    {"step": 11, "name": "multi_physics",  "cli": "multi-physics",  "engine": "MultiPhysicsCoupler.kt", "cli_args": []},
]

EDALAB_ROOT = Path(r"C:\Users\bnna7\workspace\EdaLab")
EDALAB_JAR = EDALAB_ROOT / "edalabcore" / "build" / "libs" / "edalabcore.jar"
EDALAB_MAIN_CLASS = "edalab.core.cli.EdaCli"

# Kotlin 运行时 classpath (2026-09-20 实测可跑组合; gradle cache 位置)
_KOTLIN_STDLIB = ("C:/Users/bnna7/.gradle/caches/modules-2/files-2.1/org.jetbrains.kotlin"
                  "/kotlin-stdlib/2.2.21/fa374a986e128314c3db00a20aae55f72a258511/kotlin-stdlib-2.2.21.jar")
_KOTLIN_JDK8 = ("C:/Users/bnna7/.gradle/caches/modules-2/files-2.1/org.jetbrains.kotlin"
                "/kotlin-stdlib-jdk8/2.2.10/80cc9a776a058eeda7053731350bc41af88a8a5a/kotlin-stdlib-jdk8-2.2.10.jar")
_KOTLIN_JDK7 = ("C:/Users/bnna7/.gradle/caches/modules-2/files-2.1/org.jetbrains.kotlin"
                "/kotlin-stdlib-jdk7/2.1.20/72672df5e83231744c00550955b10acb71ed49d6/kotlin-stdlib-jdk7-2.1.20.jar")

# 可选 java 路径 (PATH 无 java 时回退到 Gradle JDK)
_JAVA_FALLBACK = Path(r"C:\Users\bnna7\.jdks\ms-17.0.18\bin\java.exe")


def _decode_output(data: bytes) -> str:
    """把子进程字节流解成文本 — 带编码回退链 (v0.5 修复)。

    问题: Windows 中文环境下 JVM 的 System.out 用本地编码 (GBK) 打印, 而此前 Python
    侧固定 `encoding="utf-8", errors="replace"`, 中文被静默替换成 U+FFFD。结果
    `assertIn("总 16.00ms", stdout)` 恒失败, 且人工阅读日志全是乱码。

    策略: 先按 UTF-8 严格解码; 若失败或含替换字符, 依次回退 GBK/GB18030; 最后
    latin-1 兜底 (永不抛异常)。
    """
    if not data:
        return ""
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            text = data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        if "\ufffd" not in text:
            return text
    return data.decode("utf-8", errors="replace")


def _resolve_java() -> str:
    import shutil
    which = shutil.which("java")
    if which:
        return which
    if _JAVA_FALLBACK.exists():
        return str(_JAVA_FALLBACK)
    raise FileNotFoundError("java 不可用: PATH 无 java 且回退路径不存在")


class EdalabAdapter:
    """EdaLab Python ↔ Kotlin 桥接适配器。

    用法:
        adapter = EdalabAdapter()
        # live 真调 (subprocess → Kotlin 引擎)
        r = adapter.run_stage("relay_clutch", {}, mode="live")
        assert r["status"] == "LIVE_OK" and "16.00ms" in r["stdout"]
        # dry 元数据校验 (CI 快速路径)
        r = adapter.run_stage("relay_clutch", {}, mode="dry")
        # 全链
        results = adapter.run_full_chain({}, mode="live")
    """

    def __init__(self, edalab_root: Path = EDALAB_ROOT, verbose: bool = False,
                 java_path: Optional[str] = None, timeout_sec: int = 120):
        self.root = Path(edalab_root)
        self.verbose = verbose
        self.timeout = timeout_sec
        self._java = java_path  # 惰性解析
        self._stage_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 单阶段触发 (B1: dry / live 双模式)
    # ------------------------------------------------------------------ #
    def run_stage(self, stage_name: str, params: Dict[str, Any],
                  mode: str = "dry") -> Dict[str, Any]:
        """触发单阶段。

        mode="dry":  元数据校验 + 引擎文件存在性 (秒回, CI 用)
        mode="live": subprocess 真调 EdaCli → Kotlin 引擎执行, 捕获 stdout
        """
        stage = next((s for s in FLOW_CHAIN if s["name"] == stage_name), None)
        if stage is None:
            raise ValueError(f"未知阶段: {stage_name}. 合法: {[s['name'] for s in FLOW_CHAIN]}")

        rec: Dict[str, Any] = {
            "step": stage["step"], "stage": stage_name, "mode": mode,
            "cli": stage["cli"], "engine_file": stage["engine"], "params": params,
            "started_at": time.time(),
        }

        if mode == "dry":
            ok, msg = self._check_param_contract(stage_name, params)
            if not ok:
                rec.update(status="CONTRACT_FAIL", error=msg)
            else:
                engine_path = self._engine_path(stage["engine"])
                if engine_path:
                    rec.update(status="READY", engine_resolved=str(engine_path))
                else:
                    rec.update(status="ENGINE_MISSING", engine_resolved=None)
            rec["finished_at"] = time.time()
            self._stage_log.append(rec)
            return rec

        if mode == "live":
            return self._run_live(rec, stage)

        raise ValueError(f"mode 必须为 dry|live, 收到: {mode}")

    def _run_live(self, rec: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, Any]:
        """B1: subprocess 真调。"""
        jar = EDALAB_JAR if self.root == EDALAB_ROOT else self.root / "edalabcore" / "build" / "libs" / "edalabcore.jar"
        if not jar.exists():
            rec.update(status="LIVE_FAIL", error=f"jar 不存在 (先 ./gradlew :edalabcore:jar): {jar}",
                       finished_at=time.time())
            self._stage_log.append(rec)
            return rec
        cp = f"{jar};{_KOTLIN_STDLIB};{_KOTLIN_JDK8};{_KOTLIN_JDK7}"
        try:
            java = self._java or _resolve_java()
        except FileNotFoundError as e:
            rec.update(status="LIVE_FAIL", error=str(e), finished_at=time.time())
            self._stage_log.append(rec)
            return rec

        # -Dstdout.encoding/-Dstderr.encoding: JDK 17+ 会让 System.out 走 UTF-8,
        # 省掉一次编码回退; 老 JDK 忽略这两个属性, 由 _decode_output 兜底。
        cmd = [java, "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8",
               "-Dstderr.encoding=UTF-8", "-cp", cp, EDALAB_MAIN_CLASS,
               stage["cli"], *stage["cli_args"]]
        rec["cmd"] = cmd
        try:
            proc = subprocess.run(cmd, capture_output=True,
                                  cwd=str(self.root), timeout=self.timeout)
            stdout = _decode_output(proc.stdout)
            stderr = _decode_output(proc.stderr)
            rec.update(returncode=proc.returncode,
                       stdout=stdout, stderr=stderr[-2000:])
            rec["status"] = "LIVE_OK" if proc.returncode == 0 else "LIVE_NONZERO"
            # 提取风险行 ([CRITICAL]/[WARNING]/[BLOCKING] 或 -> [xxx] / 符号前缀)
            rec["risk_lines"] = [
                l.strip() for l in stdout.splitlines()
                if any(tag in l for tag in ("[CRITICAL]", "[WARNING]", "[BLOCKING]", "[INFO]"))
            ]
        except subprocess.TimeoutExpired:
            rec.update(status="LIVE_TIMEOUT", error=f"超过 {self.timeout}s")
        except Exception as e:  # noqa: BLE001
            rec.update(status="LIVE_ERROR", error=repr(e))
        rec["finished_at"] = time.time()
        self._stage_log.append(rec)
        return rec

    def run_full_chain(self, params: Dict[str, Any], mode: str = "dry") -> List[Dict[str, Any]]:
        """顺序执行完整 11 步主链。live 模式下逐引擎真调。"""
        results = []
        for stage in FLOW_CHAIN:
            r = self.run_stage(stage["name"], params, mode=mode)
            results.append(r)
            if r["status"] in ("CONTRACT_FAIL", "LIVE_FAIL", "LIVE_ERROR", "LIVE_TIMEOUT"):
                break
        return results

    # ------------------------------------------------------------------ #
    # DR 配置生成 — 制造容差驱动的域随机化 (创新 #1, K2 真实资产, 保留)
    # ------------------------------------------------------------------ #
    def generate_dr_config(self, strategy_v: str = "v1.0",
                           ict_data: Optional[List[Dict]] = None,
                           fct_data: Optional[List[Dict]] = None,
                           telemetry_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """基于 ICT/FCT/遥测数据生成 RL DR 配置。"""
        overrides: Dict[str, Dict[str, Any]] = {}
        based_serials: List[str] = []

        flat: List[Dict] = []
        faults: List[List[Dict]] = []
        by_type: Dict[str, int] = {}

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
                    "min": max(0, mean - 3 * std), "max": mean + 3 * std,
                    "mean": mean, "std": std,
                    "rationale": f"基于 {len(encoder_vals)} 个编码器 ICT 实测偏差",
                }
            if current_vals:
                mean_curr = sum(current_vals) / len(current_vals)
                overrides["current_sensor_noise_std"] = {
                    "distribution": "uniform", "min": 0, "max": mean_curr * 0.05,
                    "rationale": f"基于 {len(current_vals)} 个电流传感器 ICT 实测偏差",
                }

        if fct_data:
            based_serials.extend(d.get("board_serial", "unknown") for d in fct_data)
            can_utils = [d["comm_tests"]["can_bus_utilization"]
                         for d in fct_data if "comm_tests" in d and "can_bus_utilization" in d["comm_tests"]]
            if can_utils:
                avg = sum(can_utils) / len(can_utils)
                overrides["canfd_latency_ms"] = {
                    "distribution": "uniform", "min": 0, "max": min(avg, 2.0),
                    "rationale": f"基于 {len(can_utils)} 块板子 FCT CAN-FD 利用率",
                }

        if telemetry_data:
            torques = [d["motor_current_a"] for d in telemetry_data if "motor_current_a" in d]
            temps = [d["temperature_c"] for d in telemetry_data if "temperature_c" in d]
            faults = [d.get("fault_events", []) for d in telemetry_data]

            if torques:
                t_std = self._std(torques)
                overrides["load_torque_scale"] = {
                    "distribution": "loguniform",
                    "min": max(0.5, 1.0 - t_std), "max": 1.0 + t_std,
                    "rationale": f"基于 {len(torques)} 条遥测电流",
                }
                overrides["friction_coefficient"] = {
                    "distribution": "uniform", "min": 0, "max": t_std * 2,
                    "rationale": "摩擦 (从部署电流变化推断)",
                }
            if temps:
                overrides["ambient_temp"] = {
                    "distribution": "uniform", "min": 15, "max": min(max(temps), 45),
                    "rationale": f"部署最高环境温度: {max(temps):.1f}°C",
                }
            if faults:
                flat = [f for sub in faults for f in sub]
                total = max(len(telemetry_data), 1)
                for f in flat:
                    by_type[f.get("fault_type", "UNKNOWN")] = by_type.get(f.get("fault_type", "UNKNOWN"), 0) + 1
                for ftype, count in by_type.items():
                    rate = count / total
                    if rate > 0.01:
                        overrides[f"fault_inject_{ftype.lower()}"] = {
                            "distribution": "uniform", "min": rate * 0.5, "max": rate * 2,
                            "rationale": f"真机故障率: {rate * 100:.1f}%",
                        }

        return {
            "config_version": strategy_v,
            "based_on_serials": based_serials,
            "overrides": overrides,
            "recommended_training_epochs": 200 + min(len(flat) if faults else 0, 1000),
            "recommended_lr_multiplier": 2.0 if (faults and any(
                (count / max(len(telemetry_data), 1)) > 0.1
                for count in by_type.values()
            )) else 1.0,
            "generated_at": time.time(),
        }

    def export_dr_yaml(self, config: Dict[str, Any], out_path: Path) -> Path:
        """导出 DR 配置为 YAML (与 Kotlin exportDrYaml 格式对齐)。"""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if yaml is not None:
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        else:
            lines = self._to_yaml(config, indent=0)
            out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def load_contract_dr_params(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """加载 contracts/dr_params/manufacturing_variance.yaml 作为 DR 基准。"""
        if path is None:
            from ..config import load_contract
            return load_contract("dr_params/manufacturing_variance")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DR params 契约缺失: {path}")
        if yaml is not None:
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        from ..config.loader import _yaml_load
        return _yaml_load(path.read_text(encoding="utf-8"))

    def flywheel_e2e(self, ict_data=None, fct_data=None, telemetry_data=None,
                     output_dir: Path = Path("output/dr")) -> Dict[str, Any]:
        """端到端飞轮: 制造数据 → DR → YAML 输出 → 契约 diff。"""
        config = self.generate_dr_config(strategy_v="v2.0_e2e", ict_data=ict_data,
                                          fct_data=fct_data, telemetry_data=telemetry_data)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = self.export_dr_yaml(config, output_dir / "dr_config.yaml")
        contract = self.load_contract_dr_params()
        diff = self._diff_vs_contract(config, contract)
        return {"config": config, "yaml_path": str(yaml_path),
                "contract_name": "manufacturing_variance",
                "variance_diff": diff, "flywheel_ok": True}

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #
    def _engine_path(self, engine_file: str) -> Optional[Path]:
        base = self.root / "edalabcore" / "src" / "main" / "kotlin" / "edalab" / "core"
        for subdir in ["design", "physics", "simulation", "safety", "bms", "hil", "ai", "mfg", "coupling"]:
            candidate = base / subdir / engine_file
            if candidate.exists():
                return candidate
        return None

    def _check_param_contract(self, stage: str, params: Dict) -> tuple:
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
        lines: List[str] = []
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
        diff: Dict[str, Any] = {"extra_in_dr": [], "missing_from_dr": [], "within_spec": []}
        contract_electrical = contract.get("electrical", {})
        config_overrides = config.get("overrides", {})
        for param_name in contract_electrical:
            if param_name in config_overrides:
                diff["within_spec"].append(param_name)
            else:
                diff["missing_from_dr"].append(param_name)
        for param_name in config_overrides:
            if param_name not in contract_electrical and param_name not in (
                    "canfd_latency_ms", "load_torque_scale", "friction_coefficient", "ambient_temp"):
                diff["extra_in_dr"].append(param_name)
        return diff

    @property
    def stage_log(self) -> List[Dict[str, Any]]:
        return list(self._stage_log)
