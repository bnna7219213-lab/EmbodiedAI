"""e3 B1 打样询价单生成 测试 (docs/19 §8 附)。

反证: 询价单含板规格/BOM 位号/Gerber 层名/询价问题/诚实声明;
文件计数自洽 (不含询价单自身); BOM 首列键名不因 BOM 头错位。
"""
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import b1_fab_inquiry as rfq  # noqa: E402


class TestInquiryContent:
    def test_builds_text(self):
        text = rfq.build_inquiry()
        assert "打样询价单" in text
        assert "B1 腱驱动关节板" in text

    def test_specs_from_contract(self):
        text = rfq.build_inquiry()
        assert "19.2A" in text          # 桥臂峰值电流 (契约值)
        assert "2 oz" in text           # 铜厚 (工艺要求)
        assert "CAN-FD 5Mbps" in text

    def test_bom_designators_present(self):
        """BOM 位号必须解析出来 (utf-8-sig 修复守护)。"""
        text = rfq.build_inquiry()
        assert "| U_MCU |" in text, "BOM 位号列为空 — CSV BOM 头错位"
        assert "STM32G474VET6" in text

    def test_gerber_layers_named(self):
        text = rfq.build_inquiry()
        m = re.search(r"Gerber 共 \*\*(\d+) 层\*\*", text)
        assert m and int(m.group(1)) == 5
        assert "板框 (Outline)" in text
        assert "顶层铜 (Top Copper)" in text

    def test_quote_questions_present(self):
        text = rfq.build_inquiry()
        for kw in ("单价", "交期", "钢网", "MOQ", "X-Ray", "E-Test"):
            assert kw in text, f"询价问题缺『{kw}』"

    def test_honest_disclaimer_present(self):
        text = rfq.build_inquiry()
        assert "Layout 工程师" in text
        assert "AS5048A" in text and "19bit" in text   # known_gaps 明示


class TestFileScan:
    def test_excludes_self(self):
        """文件计数不得包含询价单自身 (防计数漂移)。"""
        names = [n for n, _k, _p in rfq._scan_files()]
        assert all(not n.startswith("B1_fab_inquiry") for n in names)

    def test_gerber_purposes_labeled(self):
        rows = rfq._scan_files()
        gerbers = [r for r in rows if r[0].endswith(".gbr")]
        assert len(gerbers) == 5
        assert all(r[2] != "设计产物" for r in gerbers), "Gerber 层名未识别"


class TestMain:
    def test_main_writes_file(self, tmp_path):
        out_dir = tmp_path
        rc = rfq.main(["--out-dir", str(out_dir)])
        assert rc == 0
        out = out_dir / "B1_fab_inquiry_RFQ.md"
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "询价单号" in text
