"""HTML parser splits 10-Ks into Item 1A / 7 / 7A and writes parsed JSON."""

from datetime import date
from pathlib import Path

import pytest
from filing_rag.ingest.catalog import FilingRef
from filing_rag.ingest.parse import ParseError, parse_html, write_parsed
from filing_rag.settings import PROJECT_ROOT

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_10k.html"
MSFT_HTML = PROJECT_ROOT / "data/raw/000095017024087843.html"

FILING = FilingRef(
    ticker="MSFT",
    cik="0000789019",
    company_name="MICROSOFT CORP",
    accession="0000950170-24-087843",
    form="10-K",
    filing_date=date(2024, 7, 30),
    period_of_report=date(2024, 6, 30),
    fiscal_year=2024,
    primary_doc="msft-20240630.htm",
    edgar_url="https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
)


def _parse_fixture() -> object:
    return parse_html(FIXTURE.read_text(), FILING)


def test_synthetic_extracts_requested_items_not_item_8() -> None:
    parsed = _parse_fixture()
    assert [section.item_code for section in parsed.sections] == ["1A", "7", "7A"]
    joined = "\n".join(section.text for section in parsed.sections)
    assert "Financial Statements" not in joined
    assert "scoped out of the corpus" not in joined


def test_toc_is_not_used_for_item_1a() -> None:
    risk = _parse_fixture().sections[0]
    assert "Visible risk factor body" in risk.text
    assert "TABLE OF CONTENTS" not in risk.text


def test_xbrl_hidden_stripped_visible_kept() -> None:
    risk = _parse_fixture().sections[0]
    assert "XBRL visible text stays" in risk.text
    assert "secret taxonomy junk" not in risk.text


def test_tables_flattened_to_pipes() -> None:
    risk = _parse_fixture().sections[0]
    assert "Cyber | High" in risk.text


def test_exhibits_after_anchor_are_dropped() -> None:
    parsed = _parse_fixture()
    joined = "\n".join(section.text for section in parsed.sections)
    assert "subsidiaries must not appear" not in joined


def test_char_offsets_point_into_section_text() -> None:
    parsed = _parse_fixture()
    risk = parsed.sections[0]
    assert risk.char_end > risk.char_start
    assert risk.text.startswith("Item 1A") or "Risk Factors" in risk.text


def test_write_parsed_upserts_by_accession(tmp_path: Path) -> None:
    parsed = _parse_fixture()
    path = write_parsed(parsed, tmp_path)
    assert path.name == "000095017024087843.json"
    again = write_parsed(parsed, tmp_path)
    assert again == path
    assert '"item_code": "1A"' in path.read_text()


def test_missing_item_raises() -> None:
    html = (
        "<html><body><p>Item 1A. Risk Factors</p>"
        "<p>Only risks here, no MD&amp;A.</p></body></html>"
    )
    with pytest.raises(ParseError, match="7"):
        parse_html(html, FILING)


@pytest.mark.skipif(not MSFT_HTML.exists(), reason="no live MSFT 10-K cache")
def test_live_msft_10k_sections() -> None:
    parsed = parse_html(MSFT_HTML.read_bytes(), FILING)
    assert [section.item_code for section in parsed.sections] == ["1A", "7", "7A"]
    risk, mda, market = parsed.sections
    assert "RISK FACTORS" in risk.text.upper()
    assert "DISCUSSION AND ANALYSIS" in mda.text.upper()
    assert "MARKET RISK" in market.text.upper()
    assert not any(section.item_code == "8" for section in parsed.sections)
    assert "ITEM 8." not in mda.text.upper().split("ITEM 7A")[0]
