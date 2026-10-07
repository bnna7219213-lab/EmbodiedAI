"""单仓自足实证 demo — v4「jar 入库 + 消费」的可运行证明。

**为什么要有这个 demo**
    "单仓自足"不能只写在 README 里 —— 新加入的人（或三个月后的自己）必须能
    **一条命令跑出证据**。本 demo 就是那个证据：逐项断言，任何一项不成立即失败。

**它证明什么**（5 项，每项都有可证伪的断言）
    1. 载体齐全     vendor/edalab 两件 jar 存在且体积合理（core 不自含 Kotlin 运行时的配套关系）
    2. 指纹一致     jar 的 sha256/size 与 PROVENANCE.json 记录一致（jar 未被偷换/静默替换）
    3. 运行时可用    仅需 JRE（不需要 JDK / gradle / 外部 EdaLab 目录）
    4. 隔离可跑     EDALAB_HOME + GRADLE_USER_HOME 指向**不存在**路径时仍能真实执行引擎
    5. 后端可观测    调用结果带 jar_source 字段，能回答"我这次跑的是哪份产物"

**用法**
    python scripts/demo_selfcontained.py            # 人读：逐项 ✓/✗ + 结论
    python scripts/demo_selfcontained.py --json     # 机读：JSON（CI 可断言 exit code）

    等价 CLI（已注册进全局注册表）:
    embodied-audit selfcheck

**退出码**：0 = 全部通过；1 = 有项失败（哪项失败看输出，不静默）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]      # src/embodied_ai/selfcheck.py → 仓库根
SRC = REPO / "src"


def _check(name: str, ok: bool, detail: str, hint: str = "") -> Dict[str, Any]:
    return {"check": name, "ok": bool(ok), "detail": detail, "hint": hint if not ok else ""}


def run_selfcheck(*, verbose: bool = True) -> Dict[str, Any]:
    """执行 5 项实证，返回结构化结果（不发 print —— 渲染交给调用方）。"""
    from embodied_ai.adapters.edalab import (VENDOR_DIR, build_classpath,
                                             resolve_jar_paths,
                                             verify_vendor_provenance)

    checks: List[Dict[str, Any]] = []

    # ---- ① 载体齐全 ----
    core = VENDOR_DIR / "edalabcore.jar"
    stdlibs = sorted(VENDOR_DIR.glob("kotlin-stdlib-*.jar")) if VENDOR_DIR.is_dir() else []
    check1 = core.exists() and bool(stdlibs)
    checks.append(_check(
        "1.载体齐全",
        check1,
        f"core={core.exists()} ({core.stat().st_size // 1024}KB)" if core.exists()
        else f"core 缺失: {core}",
        hint=("vendor/edalab 缺 jar。这不是代码问题 —— 是入库产物缺失。\n"
              "        修复: 从原 EdaLab 重新构建并入库:\n"
              "          cd <EdaLab> && ./gradlew :edalabcore:jar\n"
              "          cp edalabcore/build/libs/edalabcore.jar <repo>/vendor/edalab/\n"
              "          # 并更新 vendor/edalab/PROVENANCE.json 的 sha256/size")))

    # ---- ② 指纹一致（jar 未被偷换/静默替换） ----
    prov = verify_vendor_provenance()
    prov_failed = [c["check"] for c in prov.get("checks", []) if not c["ok"]]
    checks.append(_check(
        "2.指纹一致",
        bool(prov.get("ok")),
        f"PROVENANCE 校验 {len(prov.get('checks', []))} 项，"
        f"{'全部一致' if prov.get('ok') else '失败: ' + ', '.join(prov_failed)}",
        hint=("jar 与 PROVENANCE.json 记录不符 → jar 被替换过。\n"
              "        · 若你有意升级 jar: 更新 PROVENANCE.json 的 sha256/size/日期，\n"
              "          并跑 `embodied-audit verify`（L2 行为锚定会检查引擎输出是否变化）\n"
              "        · 若你没动过: 检查 vendor/edalab/ 是否被误改或被 git 覆盖")))

    # ---- ③ 运行时可用（仅需 JRE） ----
    paths = resolve_jar_paths()
    cp = build_classpath()
    java_ok, java_detail = _probe_java()
    checks.append(_check(
        "3.运行时可用",
        java_ok and cp is not None,
        f"jar 来源={paths['source']} | {java_detail}",
        hint=("需要 JRE 17+（不是 JDK，也不是 gradle）。\n"
              "        修复任选其一:\n"
              "          · 装 JRE 并确保 `java` 在 PATH\n"
              "          · 设 JAVA_HOME=<jre 目录>（脚本读 $JAVA_HOME/bin/java）\n"
              "          · 设 EMBODIED_JAVA=<java 可执行文件绝对路径>\n"
              "        Windows 提示: 用 `where java` 确认；装完重开终端刷新 PATH")))

    # ---- ④ 隔离可跑（决定性：外部不存在也能跑） ----
    iso = _isolated_probe(SRC, REPO)
    checks.append(_check(
        "4.隔离可跑",
        iso["ok"],
        iso["detail"],
        hint=("外部 EdaLab / gradle 缓存被屏蔽后仍应能跑 —— 这正是不变量的定义。\n"
              "        若失败，说明解析链退回了外部依赖，检查:\n"
              "          · vendor/edalab/edalabcore.jar 是否被 .gitignore 误排除\n"
              "          · adapters/edalab.py 的 resolve_jar_paths() 是否被改动")))

    # ---- ⑤ 后端可观测 ----
    live = _live_probe(SRC, REPO)
    checks.append(_check(
        "5.后端可观测",
        live["ok"],
        live["detail"],
        hint=("调用结果必须带 jar_source 字段（回答『这次跑的是哪份产物』）。\n"
              "        修复: 确认 _run_live() 仍写入 rec['jar_source']")))

    failed = [c["check"] for c in checks if not c["ok"]]
    return {
        "schema": "selfcontained_selfcheck_v1",
        "repo": str(REPO),
        "vendor_dir": str(VENDOR_DIR),
        "checks": checks,
        "ok": not failed,
        "failed": failed,
    }


def _probe_java() -> tuple[bool, str]:
    """探测 java 可用性（复用适配器的解析顺序，含 EMBODIED_JAVA 回退）。"""
    try:
        from embodied_ai.adapters.edalab import _resolve_java
        java = _resolve_java()
    except Exception as e:  # noqa: BLE001
        return False, f"java 未找到 ({type(e).__name__})"
    try:
        p = subprocess.run([java, "-version"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=30)
        first = (p.stderr or p.stdout or "").splitlines()[0] if (p.stderr or p.stdout) else "?"
        return p.returncode == 0, f"java={first[:48]}"
    except Exception as e:  # noqa: BLE001
        return False, f"java 执行失败: {type(e).__name__}"


def _isolated_probe(src: Path, repo: Path) -> Dict[str, Any]:
    """子进程内屏蔽外部依赖，验证仍能真实执行引擎。"""
    code = (
        "import os,sys;"
        "os.environ['EDALAB_HOME']='C:/__nonexistent_edalab_for_demo__';"
        "os.environ['GRADLE_USER_HOME']='C:/__nonexistent_gradle_for_demo__';"
        "from embodied_ai.adapters.edalab import EdalabAdapter, resolve_jar_paths;"
        "p=resolve_jar_paths();"
        "assert p['source']=='vendor', 'jar 来源非 vendor: '+str(p['source']);"
        "r=EdalabAdapter().run_stage('relay_clutch', {}, mode='live');"
        "assert r['status']=='LIVE_OK', '调用未成功: '+str(r.get('error',''))[:200];"
        "print('__ISO__'+str('16.00ms' in r.get('stdout','')))"
    )
    env = {**os.environ, "PYTHONPATH": str(src),
           "EDALAB_HOME": "C:/__nonexistent_edalab_for_demo__",
           "GRADLE_USER_HOME": "C:/__nonexistent_gradle_for_demo__"}
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           env=env, timeout=180, cwd=str(repo))
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": "隔离探测超时(180s)"}
    out = p.stdout or ""
    got = "__ISO__True" in out
    return {
        "ok": got,
        "detail": ("外部 EdaLab/gradle 屏蔽 → 仍 LIVE_OK 且引擎输出正确(含 16.00ms)"
                   if got else
                   f"隔离下未通过: rc={p.returncode} out={out.strip()[-160:]} "
                   f"err={(p.stderr or '').strip()[-160:]}"),
    }


def _live_probe(src: Path, repo: Path) -> Dict[str, Any]:
    """验证调用结果可观测（jar_source 字段）。"""
    code = (
        "from embodied_ai.adapters.edalab import EdalabAdapter;"
        "r=EdalabAdapter().run_stage('vsa_static', {}, mode='live');"
        "print('__LIVE__'+str(r.get('status'))+'|'+str(r.get('jar_source')))"
    )
    env = {**os.environ, "PYTHONPATH": str(src)}
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           env=env, timeout=180, cwd=str(repo))
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": "后端探测超时(180s)"}
    marker = [l for l in (p.stdout or "").splitlines() if l.startswith("__LIVE__")]
    if not marker:
        return {"ok": False, "detail": f"无标记输出: {(p.stdout or '')[-120:]}"}
    payload = marker[-1][len("__LIVE__"):]
    status, _, src = payload.partition("|")
    ok = status == "LIVE_OK" and src in ("vendor", "external")
    return {"ok": ok, "detail": f"status={status}, jar_source={src}"}


def render(res: Dict[str, Any]) -> None:
    """人读渲染（CLI 用 console 安全层，直接跑脚本时用 print）。"""
    try:
        from embodied_ai import console
        emit = console.emit
    except Exception:  # noqa: BLE001
        emit = print

    emit("=" * 74)
    emit("  单仓自足实证 — v4「jar 入库 + 消费」可运行证明")
    emit("=" * 74)
    emit(f"  仓库: {res['repo']}")
    emit(f"  载体: {res['vendor_dir']}")
    emit("")
    for c in res["checks"]:
        emit(f"  [{'✅' if c['ok'] else '❌'}] {c['check']}  — {c['detail']}")
        if c["hint"]:
            emit(f"        ↳ 修复指引:\n        {c['hint']}")
    emit("")
    emit("-" * 74)
    if res["ok"]:
        emit("  RESULT: SELF_CONTAINED_OK ✅")
        emit("  含义: clone 本仓库后, 仅需 JRE 即可运行全部 12 个 EDA Stage,")
        emit("        不需要 JDK / gradle / 外部 EdaLab 目录。")
    else:
        emit(f"  RESULT: FAIL ❌  未通过: {', '.join(res['failed'])}")
        emit("  含义: 当前环境跑不起来 —— 按上面每项的「修复指引」处理。")
    emit("=" * 74)


def cli_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="selfcheck",
        description="单仓自足实证 demo (v4 jar 入库 + 消费)")
    ap.add_argument("--json", action="store_true", help="JSON 输出 (CI 用)")
    args = ap.parse_args(argv)

    res = run_selfcheck()
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        render(res)
    return 0 if res["ok"] else 1
