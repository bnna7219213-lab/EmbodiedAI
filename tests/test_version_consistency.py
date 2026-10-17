"""版本口径一致性守护。

背景: CHANGELOG 已宣称 v1.0 (2026-10-17), 而 pyproject.toml /
`embodied_ai.__version__` / `verification.VERSION` / README 头部四处仍写 0.7.0
—— `cli verify` 的 banner 与 `gui/server.py` 的 `/api/status` 对外都报 0.7.0。
项目已有"verify/pytest 数字四方一致"的守护先例 (plan_v4 C6), 版本号纳入同一
机制: 单一真源 + 本测试钉死。
"""
import re
from pathlib import Path

import pytest

import embodied_ai
from embodied_ai import verification

REPO_ROOT = Path(__file__).resolve().parents[1]


def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


class TestVersionSingleSource:
    def test_verification_version_is_not_a_literal(self):
        """verification.VERSION 必须取自包, 不得再写死字面量。"""
        assert verification.VERSION == embodied_ai.__version__

    def test_pyproject_uses_dynamic_version(self):
        """pyproject 必须经 [tool.setuptools.dynamic] 静态提取 __version__。

        若改回 version = "x.y.z" 字面量, 这条会失败 —— 那正是四处漂移的成因。
        """
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'dynamic = ["version"]' in text, \
            "pyproject 应声明 dynamic version, 而非写死字面量"
        assert 'attr = "embodied_ai.__version__"' in text
        assert not re.search(r'^version\s*=\s*"\d', text, re.MULTILINE), \
            "pyproject 不应再有静态 version 字面量"

    def test_version_is_semver(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", embodied_ai.__version__), \
            f"__version__ 应为 x.y.z, 得 {embodied_ai.__version__!r}"


class TestDocsAgreeWithCode:
    def test_readme_header_version(self):
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r"\*\*版本\*\*:\s*v(\d+\.\d+)", text)
        assert m, "README 头部找不到 **版本**: vX.Y"
        assert m.group(1) == _major_minor(embodied_ai.__version__), (
            f"README 头部 v{m.group(1)} 与代码 {embodied_ai.__version__} 不一致")

    def test_changelog_latest_entry_matches(self):
        text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        m = re.search(r"^##\s+v(\d+\.\d+)", text, re.MULTILINE)
        assert m, "CHANGELOG 找不到 ## vX.Y 条目"
        assert m.group(1) == _major_minor(embodied_ai.__version__), (
            f"CHANGELOG 最新条目 v{m.group(1)} 与代码 "
            f"{embodied_ai.__version__} 不一致")

    def test_no_stale_in_progress_label_above_latest(self):
        """"(进行中)" 标签不得出现在已发布版本之后 (v0.8 曾遗留)。"""
        text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        entries = re.findall(r"^##\s+v(\d+\.\d+)(.*)$", text, re.MULTILINE)
        assert entries, "CHANGELOG 无版本条目"
        for ver, rest in entries[1:]:
            assert "进行中" not in rest, \
                f"v{ver} 仍标注'进行中', 但其后已有更新版本发布"


class TestVerifyCountConsistency:
    """README 宣称的 verify 项数必须等于 verification.CHECKS 实际长度。"""

    def test_readme_verify_count(self):
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r"\*\*(\d+)\s*项\*\*聚合自检", text)
        assert m, "README 找不到 '**N 项**聚合自检'"
        # CHECKS 不含 check_pytest (run_all 按 skip_pytest 另行追加),
        # 所以 len(CHECKS) 正是 README 的 --no-pytest 口径。
        assert int(m.group(1)) == len(verification.CHECKS), (
            f"README 写 {m.group(1)} 项, verification.CHECKS 实际 "
            f"{len(verification.CHECKS)} 项")

    def test_check_names_unique(self):
        names = [n for n, _ in verification.CHECKS]
        assert len(names) == len(set(names)), "CHECKS 存在重名项"


class TestNoPersonalPathsInRepo:
    """仓库内不得出现写死的个人机器路径 (含用户名)。

    此前 adapters/cli/FIRST_MOVE 多处写死 `C:\\Users\\<用户名>\\...`, 直接违反
    README "clone 后仅需 JRE" 的单仓自足承诺, 且把用户名发布进了公开仓库。
    外部仓库默认路径一律走 embodied_ai.paths.ext_root()。
    """

    _SCAN_ROOTS = ("src", "scripts", "tests", "docs")

    @pytest.mark.parametrize("rel", _SCAN_ROOTS)
    def test_no_hardcoded_user_dir(self, rel):
        offenders = []
        for p in (REPO_ROOT / rel).rglob("*"):
            if not p.is_file() or p.suffix not in (".py", ".md", ".toml", ".yaml", ".yml"):
                continue
            if "__pycache__" in p.parts:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                # 用户名位必须是个真名字: [A-Za-z0-9._-]+ 匹配不到 `<用户名>` 这类
                # 占位写法, 所以本文件里解释历史问题的注释不会自我误报。
                if re.search(r"C:[\\/]+Users[\\/]+[A-Za-z0-9._-]+[\\/]", line):
                    offenders.append(f"{p.relative_to(REPO_ROOT)}: {line.strip()[:100]}")
        assert offenders == [], "发现写死的个人主目录:\n" + "\n".join(offenders)


class TestNoStaleVerifyCounts:
    """操作性文档不得残留旧的 verify/pytest 口径 (曾长期停在 28/28 + 441)。

    这两份是**新人上手 / HIL 操作手册**, 照做的人会以为跑出 35/35 是异常。
    """

    _OPERATIONAL_DOCS = ("FIRST_MOVE.md", "docs/15-hil-test-protocol.md")
    _STALE = ("28/28", "441 passed", "441 项", "441+")

    @pytest.mark.parametrize("rel", _OPERATIONAL_DOCS)
    def test_no_stale_counts(self, rel):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        hits = [s for s in self._STALE if s in text]
        assert hits == [], f"{rel} 残留旧验收口径 {hits} — 请改为'以实时输出为准'的写法"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
