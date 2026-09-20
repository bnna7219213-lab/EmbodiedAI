"""契约加载器 — contracts/ 四件套加载校验。

兼容: 无 PyYAML 时使用内置 mini-YAML 解析器 (覆盖 contracts 格式:
block-style 嵌套 dict、list、行内注释、科学记数法、quoted string)。
"""
from pathlib import Path

try:
    import yaml as _yaml
    def _yaml_load(text: str):
        return _yaml.safe_load(text)
except ImportError:
    def _yaml_load(text: str):
        return _mini_yaml_parse(text)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_DIR = REPO_ROOT / "contracts"

_REQUIRED_KEYS = {
    "hw_requirements.yaml": ["metadata", "compute", "motor_drive", "sensors", "power", "communication", "safety"],
    "safety_budget.yaml": ["metadata", "segments", "software_filters", "acceptance"],
    "dr_params/manufacturing_variance.yaml": ["metadata", "electrical", "mechanical", "sensing"],
}

_DBC_REQUIRED_MESSAGES = [0, 100, 104, 108, 112, 116, 120, 300, 900]


def load_contract(name: str) -> dict:
    rel = f"{name}.yaml"
    path = CONTRACTS_DIR / rel
    if not path.exists():
        raise FileNotFoundError(f"契约不存在: {path}")
    data = _yaml_load(path.read_text(encoding="utf-8"))
    required = _REQUIRED_KEYS.get(rel, [])
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"契约 {rel} 缺失顶层键: {missing}")
    return data


def validate_dbc() -> dict:
    path = CONTRACTS_DIR / "canfd" / "joint_control.dbc"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    ids, crc_ok, current_has_crc = [], True, False
    for l in lines:
        if l.startswith("BO_ "):
            if ids and not current_has_crc:
                crc_ok = False
            ids.append(int(l.split()[1]))
            current_has_crc = False
        elif "crc8" in l:
            current_has_crc = True
    if ids and not current_has_crc:
        crc_ok = False
    return {
        "path": str(path),
        "message_count": len(ids),
        "required_ok": all(m in ids for m in _DBC_REQUIRED_MESSAGES),
        "unique_ids": len(ids) == len(set(ids)),
        "crc_coverage": crc_ok,
    }


# ---- 迷你 YAML 解析器 (PyYAML 不可用时降级) ----

def _strip_comment(s: str) -> str:
    """剥离行内注释 (# 不在引号内)。"""
    in_quote: str | None = None
    for i, c in enumerate(s):
        if c in "\"'":
            if in_quote is None:
                in_quote = c
            elif in_quote == c:
                in_quote = None
        elif c == "#" and in_quote is None:
            return s[:i].rstrip()
    return s


def _parse_scalar(s: str):
    s = _strip_comment(s).strip()
    if not s:
        return None
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _get_indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _mini_yaml_parse(text: str):
    """递归下降 mini-YAML 解析器。

    支持: block-style dict/list、行内 # 注释、引号字符串、int/float/bool。
    """
    lines = text.splitlines()

    def parse_block(start: int, base_indent: int):
        """解析从 start 开始、base_indent 为最小缩进的块。

        返回 (parsed_value, next_line_index)。
        parsed_value 可能是 dict 或 list。
        """
        result: dict = {}
        items_list: list = None  # None=未确定类型, []=list
        i = start

        while i < len(lines):
            raw = lines[i]
            stripped = raw.strip()
            if stripped == "" or stripped.startswith("#"):
                i += 1
                continue

            ci = _get_indent(raw)
            if ci < base_indent:
                break  # 缩进减小, 跳出当前块

            # 当前 key/entry 的缩进 (应 == base_indent)
            line = _strip_comment(stripped)

            if line.startswith("- "):
                # === List 条目 ===
                if items_list is None:
                    items_list = []
                # 收集完整 item block (第 0 行 + 所有缩进行)
                item_start = i
                item_lines_raw = []
                while i < len(lines):
                    r = lines[i]
                    s = r.strip()
                    if s == "" or s.startswith("#"):
                        item_lines_raw.append(r)
                        i += 1
                        continue
                    ci2 = _get_indent(r)
                    if item_lines_raw and ci2 < (base_indent + 1):
                        break
                    if not item_lines_raw and ci2 < base_indent:
                        break
                    if not item_lines_raw and ci2 > base_indent:
                        break  # 意外缩进, 视为上级
                    item_lines_raw.append(r)
                    i += 1

                first = item_lines_raw[0].strip()[2:]  # 去掉 "- "
                rest = item_lines_raw[1:]

                if ":" in first and not first.startswith('"') and not first.startswith("'"):
                    k, v = first.split(":" , 1)
                    item_dict = {k.strip(): _parse_scalar(v)}
                    # 解析缩进后续行
                    for sl in rest:
                        sl_s = sl.strip()
                        if sl_s == "" or sl_s.startswith("#"):
                            continue
                        sl_line = _strip_comment(sl_s)
                        if ":" in sl_line:
                            sk, sv = sl_line.split(":", 1)
                            item_dict[sk.strip()] = _parse_scalar(sv)
                    items_list.append(item_dict)
                else:
                    items_list.append(_parse_scalar(first))
            elif ":" in line:
                # === Dict 条目 ===
                items_list = None  # 明确是 dict
                key, val = line.split(":", 1)
                key = key.strip()
                val_str = val.strip()
                i += 1
                if val_str:
                    result[key] = _parse_scalar(val_str)
                else:
                    # 嵌套块: 收集缩进更大的行
                    sub_start = i
                    sub_lines = []
                    while i < len(lines):
                        r = lines[i]
                        if r.strip() == "" or r.strip().startswith("#"):
                            sub_lines.append(r)
                            i += 1
                            continue
                        if _get_indent(r) <= base_indent:
                            break
                        sub_lines.append(r)
                        i += 1
                    if sub_lines:
                        sub_parsed, _ = parse_block(sub_start, base_indent + 2)
                        result[key] = sub_parsed
                    else:
                        result[key] = None
            else:
                i += 1  # 跳过无法解析的行

        if items_list is not None:
            return items_list, i
        return result, i

    parsed, _ = parse_block(0, 0)
    return parsed if parsed else {}
