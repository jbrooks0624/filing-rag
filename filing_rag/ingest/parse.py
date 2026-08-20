"""Parse a 10-K primary document into Item 1A / 7 / 7A sections."""

from __future__ import annotations

import re
from collections.abc import Sequence
from html import escape
from pathlib import Path

from pydantic import BaseModel
from selectolax.parser import HTMLParser

from filing_rag.ingest.cache import accession_key
from filing_rag.ingest.catalog import FilingRef
from filing_rag.settings import get_settings

DEFAULT_ITEMS = ("1A", "7", "7A")
ITEM_TITLES = {
    "1A": "Risk Factors",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
}

HIDDEN_BLOCK = re.compile(r"(?is)<ix:hidden\b[^>]*>.*?</ix:hidden>")
IX_TAG = re.compile(r"(?i)</?ix:[a-z0-9_.-]+\b[^>]*>")
# Require a period after the item code so running headers ("Item 1A") and
# narrative ("Item 1A of this Form 10-K") do not look like section starts.
ITEM_HEADER = re.compile(r"(?im)^\s*item\s+(\d{1,2}[A-Z]?)\s*\.\s*(.*)$")
EXHIBIT_ID = re.compile(r"(?i)^EX-?\d")


class ParseError(ValueError):
    """The filing HTML could not be split into the requested items."""


class Section(BaseModel):
    item_code: str
    item_title: str
    text: str
    char_start: int
    char_end: int


class ParsedFiling(BaseModel):
    ticker: str
    cik: str
    company_name: str = ""
    accession: str
    form: str
    filing_date: str
    period_of_report: str
    fiscal_year: int
    primary_doc: str
    edgar_url: str
    sections: list[Section]


def parse_html(
    html: str | bytes,
    filing: FilingRef,
    items: Sequence[str] = DEFAULT_ITEMS,
) -> ParsedFiling:
    """Clean HTML and emit the requested items, skipping Item 8 and exhibits."""
    text = _clean_text(html)
    wanted = [_normalize_item(code) for code in items]
    spans = _best_item_spans(text)
    sections: list[Section] = []
    missing: list[str] = []
    for code in wanted:
        span = spans.get(code)
        if span is None:
            missing.append(code)
            continue
        start, end = span
        sections.append(
            Section(
                item_code=code,
                item_title=ITEM_TITLES.get(code, f"Item {code}"),
                text=text[start:end].strip(),
                char_start=start,
                char_end=end,
            )
        )
    if missing:
        raise ParseError(f"{filing.ticker} {filing.accession} missing Item {', '.join(missing)}")
    return ParsedFiling(
        ticker=filing.ticker,
        cik=filing.cik,
        company_name=filing.company_name,
        accession=filing.accession,
        form=filing.form,
        filing_date=filing.filing_date.isoformat(),
        period_of_report=filing.period_of_report.isoformat(),
        fiscal_year=filing.fiscal_year,
        primary_doc=filing.primary_doc,
        edgar_url=filing.edgar_url,
        sections=sections,
    )


def write_parsed(parsed: ParsedFiling, dest_dir: Path | None = None) -> Path:
    """Upsert `data/parsed/{accession_nodash}.json`."""
    directory = dest_dir if dest_dir is not None else get_settings().parsed_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{accession_key(parsed.accession)}.json"
    path.write_text(parsed.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _normalize_item(code: str) -> str:
    return code.strip().upper()


def _as_str(html: str | bytes) -> str:
    if isinstance(html, bytes):
        return html.decode("utf-8", errors="replace")
    return html


def _clean_text(html: str | bytes) -> str:
    raw = HIDDEN_BLOCK.sub("", _as_str(html))
    raw = IX_TAG.sub("", raw)
    tree = HTMLParser(raw)
    tree.strip_tags(["script", "style"], recursive=True)
    _drop_after_exhibit_anchor(tree)
    _flatten_tables(tree)
    root = tree.body or tree.root
    text = root.text(separator="\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _drop_after_exhibit_anchor(tree: HTMLParser) -> None:
    """Remove the first EX-* anchor and every node after it in document order."""
    anchor = None
    for node in tree.css("[id], [name]"):
        ident = (node.attributes or {}).get("id") or (node.attributes or {}).get("name") or ""
        if EXHIBIT_ID.match(ident):
            anchor = node
            break
    if anchor is None:
        return
    current = anchor
    while current is not None:
        sibling = current.next
        while sibling is not None:
            following = sibling.next
            sibling.decompose()
            sibling = following
        current = current.parent
    anchor.decompose()


def _flatten_tables(tree: HTMLParser) -> None:
    for table in list(tree.css("table")):
        rows: list[str] = []
        for row in table.css("tr"):
            cells = [cell.text(separator=" ", strip=True) for cell in row.css("th, td")]
            cells = [re.sub(r"\s+", " ", cell) for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        replacement = "\n".join(rows)
        if replacement:
            fragment = HTMLParser(f"<div>{escape(replacement)}</div>")
            node = fragment.body.child if fragment.body is not None else None
            if node is not None:
                table.replace_with(node)
                continue
        table.decompose()


def _best_item_spans(text: str) -> dict[str, tuple[int, int]]:
    """For each item code, keep the occurrence with the largest following span.

    The table of contents matches first and is short; the body match is long.
    """
    matches = list(ITEM_HEADER.finditer(text))
    best: dict[str, tuple[int, int, int]] = {}
    for index, match in enumerate(matches):
        code = _normalize_item(match.group(1))
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        span = end - start
        previous = best.get(code)
        if previous is None or span > previous[0]:
            best[code] = (span, start, end)
    return {code: (start, end) for code, (_, start, end) in best.items()}
