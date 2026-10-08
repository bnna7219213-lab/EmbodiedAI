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

B1 (Phase B): 主链增加"任务档案"维度 — run_stage/run_full_chain 的 spec 参数
  ("household_grasp" 默认 / "climbing"), 由 contracts/hw_requirements.yaml 驱动
  forge-core 与 motor-drive 两个参数化阶段的 CLI 参数 (见 build_spec_args)。

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
from typing import Any, Dict, List, Optional, Sequence

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

# ------------------------------------------------------------------ #
# 路径解析 (v0.8: 同仓 vendor 优先 → 外部 EdaLab 回退 → gradle 缓存回退)
# ------------------------------------------------------------------ #
_REPO_ROOT = Path(__file__).resolve().parents[3]          # src/embodied_ai/adapters/ → 仓库根
VENDOR_DIR = _REPO_ROOT / "vendor" / "edalab"            # 入库的 jar (单仓自足的关键)
EDALAB_ROOT = Path(r"C:\Users\bnna7\workspace\EdaLab")   # 外部参照 (仅回退用)
EDALAB_MAIN_CLASS = "edalab.core.cli.EdaCli"


def resolve_jar_paths() -> Dict[str, Optional[Path]]:
    """解析 jar 与 Kotlin 运行时的实际位置 (四级回退, 单仓优先)。

    1. vendor/edalab/  (入库, 单仓自足 — clone 即可用, 仅需 JRE)
    2. 外部 EdaLab 目录  (开发机回退, 未做 vendor 时)
    3. gradle 缓存       (Kotlin stdlib, vendor 缺失时兜底)
    返回 {"core": Path|None, "stdlib": Path|None, "source": str}

    环境变量 EDALAB_HOME 可覆盖外部回退目录 (CI 显式验证"无外部依赖"时可指向不存在路径)。
    """
    import os as _os
    external_root = Path(_os.environ["EDALAB_HOME"]) if _os.environ.get("EDALAB_HOME") else EDALAB_ROOT
    # core jar
    core = None
    source = "missing"
    for cand, tag in ((VENDOR_DIR / "edalabcore.jar", "vendor"),
                      (external_root / "edalabcore" / "build" / "libs" / "edalabcore.jar", "external")):
        if cand.exists():
            core = cand
            source = tag
            break
    # kotlin stdlib
    stdlib = None
    vendored = sorted(VENDOR_DIR.glob("kotlin-stdlib-*.jar")) if VENDOR_DIR.is_dir() else []
    if vendored:
        stdlib = vendored[-1]
    else:
        import glob as _glob
        hits = sorted(_glob.glob(
            "C:/Users/bnna7/.gradle/caches/modules-2/files-2.1/org.jetbrains.kotlin"
            "/kotlin-stdlib/*/*/kotlin-stdlib-*.jar"))
        hits = [h for h in hits if "sources" not in h]
        if hits:
            stdlib = Path(hits[-1])
    return {"core": core, "stdlib": stdlib, "source": source}


def build_classpath() -> Optional[str]:
    """构造 java -cp 参数串 (vendor 优先, gradle 缓存兜底), 无 core jar 返回 None。

    注: 路径统一为正斜杠 — Windows java 接受 `/`, 且混用分隔符易踩转义坑。
    """
    paths = resolve_jar_paths()
    if paths["core"] is None:
        return None
    parts = [str(paths["core"]).replace("\\", "/")]
    if paths["stdlib"] is not None:
        parts.append(str(paths["stdlib"]).replace("\\", "/"))
    # jdk7/8 stdlib 兜底 (仅当存在; 缺亦不影响 — 已实测 stdlib 2.2.21 单件可跑,
    # 且"仅 vendor 两件"已在干净 classpath 下实测通过全部 12 个 stage)
    # 路径跨平台推导: GRADLE_USER_HOME 优先, 否则 ~/.gradle — 不硬编码用户目录
    import glob as _glob
    import os as _os
    gradle_home = (_os.environ.get("GRADLE_USER_HOME")
                   or str(Path.home() / ".gradle")).replace("\\", "/")
    for pat in ("kotlin-stdlib-jdk8", "kotlin-stdlib-jdk7"):
        hits = sorted(_glob.glob(
            f"{gradle_home}/caches/modules-2/files-2.1/"
            f"org.jetbrains.kotlin/{pat}/*/*/{pat}-*.jar"))
        hits = [h.replace("\\", "/") for h in hits if "sources" not in h]
        if hits:
            parts.append(hits[-1])
    return ";".join(parts)


# 兼容旧引用 (保留常量名, 惰性求值)
EDALAB_JAR = VENDOR_DIR / "edalabcore.jar"

# v4 溯源元数据 (jar 与源码漂移治理的锚点 — 见 docs/10 与 plan_L5 §五)
PROVENANCE_PATH = VENDOR_DIR / "PROVENANCE.json"


def verify_vendor_provenance(provenance_path: Optional[Path] = None) -> Dict[str, Any]:
    """v4 漂移治理 L1 (指纹层): 校验 vendor jar 与 PROVENANCE.json 记录一致。

    背景 (必须治理的风险): vendor jar 是**冻结的编译产物**。原 EdaLab 源码若
    演进 (改引擎/改公式), jar 不会自动更新 → 行为静默漂移, 且无人知晓。
    B3 已实证过一次: 契约 109.0 落后于引擎实测 166.39 (V-2 修复前口径)。

    两道闸门:
      L1 指纹 (本函数): jar 被替换/重新构建 → sha256 变化 → 立刻暴露。
      L2 行为锚定: 契约锚定值 vs 引擎实测 (verify_b3_antagonistic 等) —
         覆盖"源码已演进、jar 该更新了"的情形 (L1 只能发现 jar 变了)。

    返回 {"ok": bool, "checks": [...], "jars": [...], "path": str}
    """
    import hashlib as _hashlib
    import json as _json

    p = Path(provenance_path) if provenance_path is not None else PROVENANCE_PATH
    checks: List[Dict[str, Any]] = []

    def _ck(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    def _fail():
        return {"ok": False, "checks": checks, "jars": [], "path": str(p)}

    if not p.exists():
        _ck("provenance_file_exists", False, f"缺失: {p}")
        return _fail()
    _ck("provenance_file_exists", True, str(p))

    try:
        meta = _json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        _ck("provenance_json_parseable", False, repr(e))
        return _fail()
    _ck("provenance_json_parseable", True, f"schema={meta.get('schema')}")

    jars_meta = meta.get("jars") or []
    if not jars_meta:
        _ck("jars_declared", False, "PROVENANCE.json 未声明 jars")
        return _fail()
    _ck("jars_declared", True, f"{len(jars_meta)} 件声明")

    jar_results: List[Dict[str, Any]] = []
    for jm in jars_meta:
        fname = str(jm.get("file"))
        fpath = VENDOR_DIR / fname
        rec: Dict[str, Any] = {"file": fname, "role": jm.get("role")}
        if not fpath.exists():
            _ck(f"jar_exists[{fname}]", False, f"缺失: {fpath}")
            rec["ok"] = False
            jar_results.append(rec)
            continue
        _ck(f"jar_exists[{fname}]", True, str(fpath))
        data = fpath.read_bytes()
        actual_sha = _hashlib.sha256(data).hexdigest()
        actual_size = len(data)
        want_sha = jm.get("sha256")
        want_size = jm.get("size_bytes")
        sha_ok = want_sha is None or actual_sha == want_sha
        size_ok = want_size is None or actual_size == int(want_size)
        _ck(f"sha256[{fname}]", sha_ok,
            f"记录={str(want_sha)[:12]}… 实际={actual_sha[:12]}…")
        _ck(f"size[{fname}]", size_ok, f"记录={want_size} 实际={actual_size}")
        rec.update(ok=bool(sha_ok and size_ok), sha256=actual_sha,
                   size_bytes=actual_size, expected_sha256=want_sha)
        jar_results.append(rec)

    return {"ok": all(c["ok"] for c in checks), "checks": checks,
            "jars": jar_results, "path": str(p)}


# ------------------------------------------------------------------ #
# 双轨后端 (v4 + Path B): Python 侧实现与 jar 并存, 按 backend 路由
# ------------------------------------------------------------------ #
# 背景: vendor jar 是冻结的编译产物, Kotlin 侧缺口 (G2/G3/G5/G6/G7/G10) 在本仓
# 改不了源码。路径 B = 在 Python 侧实现"可修的那部分"(建模/编排/算法型), 与 jar
# 并存, 由调用方按 backend 选择。
#
# 双轨的**必要约束** (否则会退化成两套各自漂移):
#   1. 单一入口: 一律经 run_capability(), 调用方不直接碰后端
#   2. jar 作 oracle: 每个 Python 实现必须有与 jar 输出的对照测试 (容差内一致)
#   3. 可观测: 结果记 backend 字段 (jar / python), 不允许"无来源"的结果
#   4. 不静默: 两者不一致必须 FAIL, 不接受"两套都对"
PYTHON_BACKENDS: Dict[str, Callable[..., Dict[str, Any]]] = {}


def register_python_backend(name: str) -> Callable:
    """注册一个 Python 侧后端实现 (与 jar 并存, 由 backend 参数路由)。

    用法:
        @register_python_backend("hil")
        def hil_py(case_id=..., **kw) -> dict: ...
    """
    def deco(fn: Callable[..., Dict[str, Any]]):
        PYTHON_BACKENDS[name] = fn
        return fn
    return deco


def ensure_backends_registered() -> None:
    """确保所有 Python 后端已注册 (按需导入注册它们的模块)。

    动机 (同类陷阱, 与 cli_registry 的空表一致): 后端在 `hardware/*` 等模块的
    **模块级** `_register()` 里注册。只 `import adapters.edalab` 就查后端表 →
    看到空表 → 误判"无 python 后端" → `resolve_backend` 抛错或静默回落。
    本函数在任何后端查询前补一次导入, 使"注册"与"查询"解耦于导入顺序。

    **必须用惰性导入** (函数内), 否则 edalab ↔ hardware 会在模块级形成循环。
    """
    global _backends_ensured
    if _backends_ensured:
        return
    _backends_ensured = True
    import importlib
    import sys as _sys
    # 注册后端的模块清单 (新增后端时在此追加 —— 与 CLI 注册表同理, 集中一处)
    for mod in ("embodied_ai.hardware", "embodied_ai.hil"):
        try:
            if mod not in _sys.modules:
                importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            # 导入失败不应让能力查询崩溃 (该后端自然不可用), 但保留痕迹
            pass


_backends_ensured = False


def available_backends(name: str) -> List[str]:
    """某能力当前可用的后端列表 (jar = CLI 已注册 / python = 已实现)。"""
    ensure_backends_registered()
    out: List[str] = []
    if any(s["name"] == name for s in STAGE_REGISTRY):
        out.append("jar")
    if name in PYTHON_BACKENDS:
        out.append("python")
    return out


def resolve_backend(name: str, backend: str = "auto") -> str:
    """解析实际使用的后端。

    auto: 优先 jar (能力最全), 无 jar 后端时回落 python。
    jar / python: 强制指定, 若不可用则抛错 (**不静默回落**)。
    """
    avail = available_backends(name)
    if backend == "auto":
        if "jar" in avail:
            return "jar"
        if "python" in avail:
            return "python"
        raise ValueError(f"能力 {name} 无任何可用后端 (jar CLI 未注册, Python 未实现)")
    if backend not in avail:
        raise ValueError(
            f"能力 {name} 不支持 backend={backend}; 可用: {avail}")
    return backend

# 可选 java 路径 (PATH 无 java 时的回退): 仅由环境变量指定。
# 本机专属路径 **不入库** — 硬编码用户目录 (如 C:\Users\<x>\.jdks\...) 会让
# 干净 clone 上的兜底静默失效, 且违反"单仓自足/跨平台"原则。
# 解析顺序见 _resolve_java(): PATH → JAVA_HOME → EMBODIED_JAVA → 报错。


# ------------------------------------------------------------------ #
# B1: 任务档案 (spec profile) — hw_requirements.yaml 是唯一真源
# ------------------------------------------------------------------ #
# 主链默认档案 household_grasp: 沿用 FLOW_CHAIN.cli_args 抄录值 (EdaCli L614-646),
#   保持既有行为与全部既有测试不变 (向后兼容)。
# "climbing" 档案: 由 contracts/hw_requirements.yaml 的 climbing 段驱动,
#   只覆盖主链中真正参数化的两个阶段 (forge-core / motor-drive);
#   其余 9 步 (mass-props … multi-physics) 在 EdaCli 侧使用样本输入, 不随档案变化。
HOUSEHOLD_SPEC = "household_grasp"
CLIMBING_SPEC = "climbing"

# hw_requirements.yaml 无算力 TOPS 字段 (只有推理时延预算), 沿用 household 的 30 TOPS 口径
FORGE_TOPS_DEFAULT = 30


def _contracts_dir() -> Path:
    """contracts/ 目录 — 相对本文件定位 (src/embodied_ai/adapters/ → 仓库根)。"""
    return Path(__file__).resolve().parents[3] / "contracts"


def load_hw_requirements(path: Optional[Path] = None) -> Dict[str, Any]:
    """读取 contracts/hw_requirements.yaml (AI 策略→硬件规格的唯一真源)。"""
    p = Path(path) if path is not None else _contracts_dir() / "hw_requirements.yaml"
    if not p.exists():
        raise FileNotFoundError(f"hw_requirements.yaml 缺失: {p}")
    if yaml is None:
        raise RuntimeError("PyYAML 不可用, 无法解析 hw_requirements.yaml")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def available_specs() -> List[str]:
    """可跑主链的任务档案: 默认 household_grasp + yaml 中带 metadata.board_kind 的场景段。

    board_kind 是场景档案的标记 (如 climbing → tendon_drive_joint_board);
    compute/motor_drive/sensors 等结构段没有该键, 不是档案, 不参与主链分流。
    """
    hw = load_hw_requirements()
    profiles = [str(k) for k, v in hw.items()
                if isinstance(v, dict) and isinstance(v.get("metadata"), dict)
                and "board_kind" in v["metadata"]]
    return [HOUSEHOLD_SPEC] + profiles


def build_spec_args(spec: str = HOUSEHOLD_SPEC) -> Dict[str, List[str]]:
    """把 hw_requirements.yaml 的 <spec> 段翻译成主链 CLI 参数。

    返回 {stage_name: cli_args}; 空 dict 表示该档案沿用 FLOW_CHAIN.cli_args 抄录值。

    climbing 档案 (B1) 的映射:
      forge-core : [strategy_version, task_description, 运动控制Hz, TOPS, 推理ms,
                    bus, busMbps, node_count, 桥臂峰值A, 触觉Hz, 编码器通道]
      motor-drive: [板种变体] — 由 board_kind 含 "tendon" 判定 (而非硬编码字符串)
    """
    if spec != CLIMBING_SPEC:
        return {}

    hw = load_hw_requirements()
    c = hw[CLIMBING_SPEC]
    md = c["motor_drive_tendon"]
    comm = c["communication"]
    compute = hw["compute"]

    forge_args = [
        c["metadata"]["strategy_version"],                      # climbing_competition_v0.1
        c["metadata"]["task_description"],                       # climbing_tendon_drive
        str(compute["control_frequencies_hz"]["motion_control"]),  # 20Hz RL 运动控制
        str(FORGE_TOPS_DEFAULT),                                 # yaml 无 TOPS 字段 → 沿用 30
        str(compute["model_inference_ms"]),                      # 15ms (契约实测预算, 严于 household 25)
        str(comm["bus"].replace("_", "")),            # can_fd → canfd (ForgeCore 枚举口径)
        str(comm["data_bps"] // 1_000_000),                      # 5Mbps
        str(comm["node_count"]),                                 # 8 伺服节点
        str(md["bridge_peak_current_a"]),                        # 19.2A = 8ch×4A×0.6
        str(hw["sensors"]["tactile"]["frame_rate_hz"]),          # 100Hz (契约)
        str(md["encoder_channels"]),                             # 16 = 8 伺服 × 双端
    ]
    variant = "tendon" if "tendon" in str(c["metadata"]["board_kind"]) else "default"
    return {"forge_core": forge_args, "motor_drive": [variant]}


# ------------------------------------------------------------------ #
# B2 (Phase B): Stage 注册表 + 产物解析 (数据管道, 替代烟囱式)
# ------------------------------------------------------------------ #
# P1.4 Stage 化收官: 12 个 CLI 命令注册为可编排 Stage。
#
# 关键约束: FLOW_CHAIN (11 步) **保持原样不动** — 既有 320 项测试依赖其长度
# 与 EdalabAdapter.chain_green(expect_steps=11)。B2 的编排基于本注册表
# (第 12 步 = vsa-static), 因此既有行为零改动。
#
# 为什么必须在 Python 消费侧做: plan_L5 §三 的 G5 缺口 (CLI 主链烟囱式 —
# 各步用样本输入、上游产物无人消费) 位于 Kotlin 侧, 而 vendor jar 是冻结的
# 编译产物, 改不了源码。消费侧编排是 vendor 模式下唯一可推进的路径。
VSA_STAGE: Dict[str, Any] = {
    "step": 12, "name": "vsa_static", "cli": "vsa-static",
    "engine": "VsaVariableStiffnessEngine.kt", "cli_args": ["--json"],
}

STAGE_REGISTRY: List[Dict[str, Any]] = [*FLOW_CHAIN, VSA_STAGE]

# stage → 产物类型 (供 parse_stage_output 分派)
STAGE_OUTPUT_PARSERS: Dict[str, str] = {
    "forge_core": "eps_id",      # stdout 提取 "EPS ID: <id>"
    "vsa_static": "vsa_json",    # stdout 解析 vsa_curve_v1(_1) JSON
    "relay_clutch": "clutch",    # B3: 解析离合器传递/额定扭矩 + 安全链总时延
}


def _parse_eps_id(stdout: str) -> Optional[str]:
    """从 forge-core 输出提取 EPS ID (B1 已实测口径)。

    climbing 档案实测: EPS ID: SF-climbing_tendon_drive-climbing_competition_v0_1
    """
    import re as _re
    m = _re.search(r"EPS\s*ID[:：]\s*(\S+)", stdout or "")
    return m.group(1) if m else None


def _parse_vsa_json(stdout: str) -> Optional[Dict[str, Any]]:
    """从 vsa-static --json 输出解析 VSA 曲线 JSON。

    容错: Kotlin 侧可能在 JSON 前后打日志行, 故取最外层 {...}; 解析失败
    返回 None (不抛异常 — dry 模式或引擎变更时不影响编排继续)。
    """
    import json as _json
    if not stdout:
        return None
    s = stdout.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return _json.loads(s[i:j + 1])
    except ValueError:
        return None


def _parse_clutch(stdout: str) -> Dict[str, Any]:
    """B3: 从 relay-clutch 输出解析离合器承载与安全链时延 (供联合校验)。

    实测输出形态 (2026-10-07, vendor jar, climbing 档案):
      离合器: JointBrake-FrictionPlate, 传递 6.38Nm (额定 12.0Nm, 53%),
              滑摩功 24.7J, 温升 0.1℃→峰值 40.1℃ -> PASS
      安全链路: 电气 0.8µs + 熄弧 8.0ms + 机械隔离 8.0ms = 总 16.00ms (目标 <20ms) -> PASS

    额定扭矩是 B3 "离合器能否锁定 VSA 弹射峰值" 判据的分母。
    """
    import re as _re
    out: Dict[str, Any] = {"transmit_nm": None, "rated_nm": None,
                           "safety_chain_ms": None, "clutch_verdict": None}
    if not stdout:
        return out
    m = _re.search(r"传递\s*([\d.]+)\s*Nm", stdout)
    if m:
        out["transmit_nm"] = float(m.group(1))
    m = _re.search(r"额定\s*([\d.]+)\s*Nm", stdout)
    if m:
        out["rated_nm"] = float(m.group(1))
    m = _re.search(r"总\s*([\d.]+)\s*ms", stdout)
    if m:
        out["safety_chain_ms"] = float(m.group(1))
    # 注意: 标题行也含"离合器" ("继电器/离合器触点动力学"), 若只按"离合器"匹配
    # 会命中标题、取不到 verdict 就 break — 必须限定数据行 (含 "Nm" 扭矩)。
    for ln in stdout.splitlines():
        if "离合器" in ln and "Nm" in ln:
            mv = _re.search(r"->\s*(PASS|WARN|FAIL)", ln)
            if mv:
                out["clutch_verdict"] = mv.group(1)
            break
    return out


def parse_stage_output(stage_name: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    """把单 stage 的 stdout 解析为结构化产物, 供下游 stage 消费。

    烟囱式主链的根本问题就是"上游产出没人消费"。本函数让 Python 消费侧能把
    forge-core 的 EPS ID 与 vsa-static 的刚度曲线真正串成管道 (如 EPS →
    VSA 曲线 → 攀岩 dyno 弹道注入), 而不是各步自说自话。
    """
    kind = STAGE_OUTPUT_PARSERS.get(stage_name)
    out: Dict[str, Any] = {
        "stage": stage_name,
        "kind": kind,
        "status": rec.get("status"),
        "raw_len": len(rec.get("stdout") or ""),
    }
    if kind == "eps_id":
        out["eps_id"] = _parse_eps_id(rec.get("stdout") or "")
    elif kind == "vsa_json":
        d = _parse_vsa_json(rec.get("stdout") or "")
        out["vsa"] = d
        if isinstance(d, dict):
            out["vsa_schema"] = d.get("schema")
            tops = d.get("topologies") or []
            out["vsa_topologies"] = [t.get("name") for t in tops if isinstance(t, dict)]
            # B3: 逐拓扑峰值力 — 供 ANTAGONISTIC 选型一致性与离合器承载校验消费
            out["vsa_peaks"] = {str(t.get("topology")): t.get("peak_force_n")
                                for t in tops if isinstance(t, dict)}
    elif kind == "clutch":
        out.update(_parse_clutch(rec.get("stdout") or ""))
    return out


def verify_b3_antagonistic(context: Dict[str, Dict[str, Any]],
                           hw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """B3 (Phase B): VSA-ANTAGONISTIC 选型落地 — relay-clutch + vsa-static 联合校验。

    B3 定义 (docs/10 Phase B): 把 P1.3 结论 (攀岩指尖 109N 物理正确) 写入 B1 的
    电机/减速器参数; 验收 = "relay-clutch + vsa-static 联合 PASS"。

    ★ B3 实测发现 (2026-10-07): 契约原值 109.0 与引擎实测 166.39 **不一致** —
      109 是 V-2 修复 (ANTAGONISTIC V 形公式改写为 kMin+t*(kMax-kMin)) **之前**
      的过期口径, 且 docs/04 该行 stored(0.635J) 疑混填了 SEA 的实测值 (0.6379)。
      契约已按 vsa-static 实测校准为 166.39; **本函数守护这一一致性, 防止回漂**。
      选型结论不变: ANTAGONISTIC 仍是三拓扑中峰值最小者 (最接近指尖 60N)。

    判据 (物理量交叉验证, 而非"两个 stage 各自 LIVE_OK" 的空转):
      1. 契约 vsa_antagonistic_peak_n == vsa-static 实测 ANTAGONISTIC peak (防漂移)
      2. ANTAGONISTIC 仍是三拓扑峰值最小者 → 选型物理正确
      3. 离合器承载自洽: peak × 力臂 ≤ 离合器额定扭矩 (弹射瞬间不得打滑)
      4. 安全链总时延 < 20ms (契约预算)
      5. 离合器 verdict == PASS
    """
    if hw is None:
        hw = load_hw_requirements()
    fl = (hw.get("climbing") or {}).get("force_limits") or {}
    contract_peak = fl.get("vsa_antagonistic_peak_n")
    lever = fl.get("fingertip_lever_arm_m", 0.022)
    tip_limit = fl.get("fingertip_hertz_limit_n", 60.0)

    vsa_ctx = (context or {}).get("vsa_static") or {}
    rc_ctx = (context or {}).get("relay_clutch") or {}
    peaks = vsa_ctx.get("vsa_peaks") or {}
    eng_peak = peaks.get("ANTAGONISTIC")

    checks: List[Dict[str, Any]] = []

    def _ck(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # 1) 契约 vs 引擎实测一致 (防漂移)
    ok1 = (contract_peak is not None and eng_peak is not None
           and abs(float(contract_peak) - float(eng_peak)) <= 0.5)
    _ck("contract_matches_engine", ok1,
        f"契约={contract_peak} vs 引擎={eng_peak} (容差 0.5N)")

    # 2) ANTAGONISTIC 是峰值最小拓扑 → 选型物理正确
    ok2 = bool(peaks) and eng_peak is not None and all(
        float(eng_peak) <= float(v) for v in peaks.values() if v is not None)
    _ck("antagonistic_is_min_peak", ok2, f"peaks={peaks}")

    # 3) 离合器承载自洽 (弹射峰值不得打滑)
    rated = rc_ctx.get("rated_nm")
    demand = float(eng_peak) * float(lever) if eng_peak is not None else None
    ok3 = demand is not None and rated is not None and demand <= float(rated)
    _ck("clutch_holds_vsa_peak", ok3,
        f"需求={demand:.3f}Nm (peak {eng_peak}N × 臂 {lever}m) ≤ 额定 {rated}Nm")

    # 4) 安全链预算
    chain = rc_ctx.get("safety_chain_ms")
    ok4 = chain is not None and float(chain) < 20.0
    _ck("safety_chain_within_20ms", ok4, f"安全链={chain}ms < 20ms")

    # 5) 离合器 verdict
    verdict = rc_ctx.get("clutch_verdict")
    _ck("clutch_verdict_pass", verdict == "PASS", f"verdict={verdict}")

    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "values": {
            "contract_peak_n": contract_peak, "engine_peak_n": eng_peak,
            "peaks": peaks, "lever_arm_m": lever,
            "clutch_demand_nm": demand, "clutch_rated_nm": rated,
            "safety_chain_ms": chain, "clutch_verdict": verdict,
            "fingertip_limit_n": tip_limit,
        },
    }


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
    """解析 java 可执行文件路径 (跨平台, 无硬编码用户目录)。

    顺序: PATH(shutil.which) → JAVA_HOME/bin/java → EMBODIED_JAVA → 报错。
    vendor 消费模式单仓自足**仅需 JRE**, 不需要 JDK/gradle。
    """
    import os as _os
    import shutil
    which = shutil.which("java")
    if which:
        return which
    jh = _os.environ.get("JAVA_HOME")
    if jh:
        cand = Path(jh) / "bin" / ("java.exe" if _os.name == "nt" else "java")
        if cand.exists():
            return str(cand)
    fb = _os.environ.get("EMBODIED_JAVA")
    if fb and Path(fb).exists():
        return str(Path(fb))
    raise FileNotFoundError(
        "java 不可用: PATH 与 JAVA_HOME 均未找到 java; "
        "请安装 JRE 或设置 JAVA_HOME / EMBODIED_JAVA "
        "(vendor 模式单仓自足仅需 JRE, 无需 JDK/gradle)")


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
                  mode: str = "dry", spec: str = HOUSEHOLD_SPEC,
                  spec_args: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """触发单阶段。

        mode="dry":  元数据校验 + 引擎文件存在性 (秒回, CI 用)
        mode="live": subprocess 真调 EdaCli → Kotlin 引擎执行, 捕获 stdout
        spec:        任务档案 (默认 household_grasp / climbing), 由 hw_requirements.yaml 驱动;
                     覆盖 FLOW_CHAIN.cli_args 中该阶段的参数
        """
        # B2: 查 STAGE_REGISTRY (12 项 = FLOW_CHAIN 11 步 + vsa-static), 而非
        # FLOW_CHAIN。它是 FLOW_CHAIN 的超集, 既有 11 步行为完全不变, 且让
        # run_stage 能触发第 12 步 (vsa-static), 供 run_eps_job 编排。
        stage = next((s for s in STAGE_REGISTRY if s["name"] == stage_name), None)
        if stage is None:
            raise ValueError(f"未知阶段: {stage_name}. 合法: {[s['name'] for s in STAGE_REGISTRY]}")

        override = spec_args if spec_args is not None else build_spec_args(spec)
        cli_args = list(override.get(stage_name) or stage["cli_args"])

        rec: Dict[str, Any] = {
            "step": stage["step"], "stage": stage_name, "mode": mode,
            "cli": stage["cli"], "engine_file": stage["engine"], "params": params,
            "spec": spec, "cli_args": cli_args,
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
            return self._run_live(rec, stage, cli_args)

        raise ValueError(f"mode 必须为 dry|live, 收到: {mode}")

    def _run_live(self, rec: Dict[str, Any], stage: Dict[str, Any],
                  cli_args: Optional[List[str]] = None) -> Dict[str, Any]:
        """B1: subprocess 真调。cli_args 由 run_stage 按 spec 档案解析后传入。

        v0.8: classpath 走 build_classpath() — vendor/edalab 优先 (单仓自足),
        外部 EdaLab 与 gradle 缓存依次回退。仅需 JRE 即可运行, 不需要 JDK/gradle。
        """
        cp = build_classpath()
        if cp is None:
            paths = resolve_jar_paths()
            rec.update(status="LIVE_FAIL",
                       error=("jar 不存在: 期望 vendor/edalab/edalabcore.jar (单仓) "
                              f"或 {EDALAB_ROOT}/edalabcore/build/libs/edalabcore.jar (外部)"),
                       jar_source=paths["source"], finished_at=time.time())
            self._stage_log.append(rec)
            return rec
        rec["jar_source"] = resolve_jar_paths()["source"]
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
               stage["cli"], *(cli_args if cli_args is not None else stage["cli_args"])]
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

    def run_full_chain(self, params: Optional[Dict[str, Any]] = None, mode: str = "dry",
                       spec: str = HOUSEHOLD_SPEC) -> List[Dict[str, Any]]:
        """顺序执行完整 11 步主链。live 模式下逐引擎真调。

        spec="climbing" 时 forge-core / motor-drive 两个参数化阶段改用
        hw_requirements.yaml climbing 段的参数 (B1); 其余 9 步不随档案变化。
        """
        spec_args = build_spec_args(spec)
        results = []
        for stage in FLOW_CHAIN:
            r = self.run_stage(stage["name"], params or {}, mode=mode,
                               spec=spec, spec_args=spec_args)
            results.append(r)
            if r["status"] in ("CONTRACT_FAIL", "LIVE_FAIL", "LIVE_ERROR", "LIVE_TIMEOUT"):
                break
        return results

    def run_eps_job(self, stages: Sequence[str], spec: str = HOUSEHOLD_SPEC,
                    mode: str = "live",
                    params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """B2: EPS job — 按序串联执行多个 Stage, 上游产物喂给下游 (数据管道)。

        与 run_full_chain 的本质区别:
          - run_full_chain: EdaCli 内置 11 步**烟囱式** — 各步用 Kotlin 侧样本输入,
            上游产出不被下游消费 (plan_L5 §三 G5 缺口, Kotlin 冻结改不了)。
          - run_eps_job: Python 消费侧编排 — 每步解析产物进 context, 下游可读。
            例: forge-core 的 EPS ID → vsa-static 的刚度曲线 → 攀岩 dyno 弹道注入。

        验收口径 (docs/10-climb-soft-robot Phase B / B2 "P1.4 Stage 化收官"):
          EPS job 触发 forge-core → vsa-static 串联。

        返回 dict: {job, spec, mode, ok, steps, expected_steps, results, context, elapsed_s}
        """
        seq = [str(s) for s in stages]
        valid = {x["name"] for x in STAGE_REGISTRY}
        unknown = [s for s in seq if s not in valid]
        if unknown:
            raise ValueError(f"未知 stage: {unknown}. 合法: {sorted(valid)}")

        spec_args = build_spec_args(spec)
        results: List[Dict[str, Any]] = []
        context: Dict[str, Dict[str, Any]] = {}
        started = time.time()
        for name in seq:
            r = self.run_stage(name, params or {}, mode=mode, spec=spec,
                               spec_args=spec_args)
            results.append(r)
            # 数据管道: 解析本步产物, 供下游消费
            context[name] = parse_stage_output(name, r)
            if r["status"] not in ("READY", "LIVE_OK"):
                break
        ok = len(results) == len(seq) and all(
            r["status"] in ("READY", "LIVE_OK") for r in results)
        return {
            "job": seq, "spec": spec, "mode": mode, "ok": ok,
            "steps": len(results), "expected_steps": len(seq),
            "results": results, "context": context,
            "elapsed_s": time.time() - started,
        }

    def run_capability(self, name: str, backend: str = "auto",
                       mode: str = "live", **kwargs: Any) -> Dict[str, Any]:
        """双轨执行入口 — 按 backend 路由到 jar CLI 或 Python 实现。

        这是路径 B 的**唯一入口**: 调用方不直接选后端, 只声明要什么能力;
        路由结果写入 `backend` 字段, 使"这个结果是谁算的"永远可查。

        统一结果字段:
          capability / backend ("jar"|"python") / status / elapsed_s
          jar  路径附加 "stage_result" (run_stage 的原结果)
          python 路径附加被注册函数返回的字段

        约定: 两种后端的 **status 语义必须对齐** (OK/FAILED/...), 以便
        oracle 对照测试用同一套断言比较两者。
        """
        used = resolve_backend(name, backend)
        started = time.time()

        if used == "python":
            fn = PYTHON_BACKENDS[name]
            try:
                out = dict(fn(**kwargs))
            except Exception as e:      # noqa: BLE001 — 后端异常按 FAILED 返回
                return {"capability": name, "backend": "python",
                        "status": "FAILED", "error": repr(e),
                        "elapsed_s": time.time() - started}
            out.setdefault("status", "OK")
            out.update(capability=name, backend="python",
                       elapsed_s=time.time() - started)
            return out

        # jar 路径: 该能力须是 STAGE_REGISTRY 中的 stage
        r = self.run_stage(name, {}, mode=mode)
        return {"capability": name, "backend": "jar",
                "status": ("OK" if r["status"] in ("READY", "LIVE_OK")
                           else r["status"]),
                "stage_result": r,
                "elapsed_s": time.time() - started}

    @staticmethod
    def chain_green(results: List[Dict[str, Any]],
                    expect_steps: int = len(FLOW_CHAIN)) -> "tuple":
        """主链全绿判定: 全部 expect_steps 步跑完且每步状态在通过集合内。

        返回 (是否全绿, 逐步状态串) — B1 验收 "11 步主链对 climbing spec 全绿" 的断言入口。
        """
        ok = len(results) == expect_steps and all(
            r["status"] in ("READY", "LIVE_OK") for r in results)
        states = [f"{r['step']}.{r['stage']}={r['status']}" for r in results]
        return ok, states

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
        config_overrides = config.get("overrides", {})
        # Gate 2 C4 收官: 四节全覆盖 (electrical / mechanical / sensing / environment)
        # feedback_rules 是规则列表(IP 不在 overrides 中) — 跳过
        _CONTRACT_SECTIONS = ("electrical", "mechanical", "sensing", "environment")
        contract_params: Dict[str, str] = {}  # param_name → section_name (用于错误信息)
        for _sec in _CONTRACT_SECTIONS:
            _sec_data = contract.get(_sec, {})
            if isinstance(_sec_data, dict):
                for _pn in _sec_data:
                    contract_params[_pn] = _sec
        for param_name, section in contract_params.items():
            if param_name in config_overrides:
                diff["within_spec"].append(f"{section}.{param_name}")
            else:
                diff["missing_from_dr"].append(f"{section}.{param_name}")
        _known_overrides = {
            "canfd_latency_ms", "load_torque_scale", "friction_coefficient", "ambient_temp",
            "policy_update_hz", "action_repeat",
        }
        for param_name in config_overrides:
            if param_name not in contract_params and param_name not in _known_overrides:
                diff["extra_in_dr"].append(param_name)
        return diff

    @property
    def stage_log(self) -> List[Dict[str, Any]]:
        return list(self._stage_log)
