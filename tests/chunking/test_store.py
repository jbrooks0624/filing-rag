"""Parsed JSON in, chunk JSON out under data/chunks/{strategy}/."""

from pathlib import Path

import pytest
from filing_rag.chunking.store import (
    chunked_exists,
    chunked_path,
    iter_parsed,
    load_chunked,
    load_parsed,
    write_chunked,
)
from filing_rag.chunking.types import Chunk, ChunkedFiling
from filing_rag.ingest.parse import ParsedFiling, Section, write_parsed

ACCESSION = "0000950170-24-087843"


def _parsed() -> ParsedFiling:
    return ParsedFiling(
        ticker="MSFT",
        cik="0000789019",
        company_name="MICROSOFT CORP",
        accession=ACCESSION,
        form="10-K",
        filing_date="2024-07-30",
        period_of_report="2024-06-30",
        fiscal_year=2024,
        primary_doc="msft-20240630.htm",
        edgar_url="https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
        sections=[
            Section(
                item_code="1A",
                item_title="Risk Factors",
                text="Cybersecurity could harm results.",
                char_start=0,
                char_end=32,
            )
        ],
    )


def _chunk(parsed: ParsedFiling) -> Chunk:
    section = parsed.sections[0]
    return Chunk(
        ticker=parsed.ticker,
        cik=parsed.cik,
        accession=parsed.accession,
        fiscal_year=parsed.fiscal_year,
        edgar_url=parsed.edgar_url,
        item_code=section.item_code,
        item_title=section.item_title,
        strategy="fixed",
        chunk_index=0,
        char_start=0,
        char_end=len(section.text),
        token_count=5,
        text=section.text,
    )


def test_load_parsed_missing_raises(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="parsed filing not found"):
        load_parsed(parsed_dir, ACCESSION)


def test_load_parsed_round_trip(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed"
    write_parsed(_parsed(), parsed_dir)
    loaded = load_parsed(parsed_dir, ACCESSION)
    assert loaded.ticker == "MSFT"
    assert loaded.accession == ACCESSION
    assert [section.item_code for section in loaded.sections] == ["1A"]


def test_iter_parsed_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="parsed directory not found"):
        list(iter_parsed(tmp_path / "parsed"))


def test_iter_parsed_yields_sorted(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed"
    first = _parsed()
    second = first.model_copy(update={"accession": "0001018724-24-000001", "ticker": "AMZN"})
    write_parsed(second, parsed_dir)
    write_parsed(first, parsed_dir)
    tickers = [filing.ticker for filing in iter_parsed(parsed_dir)]
    assert tickers == ["MSFT", "AMZN"]


def test_write_chunked_uses_strategy_dir(tmp_path: Path) -> None:
    parsed = _parsed()
    chunked = ChunkedFiling.from_parsed(parsed, "fixed", [_chunk(parsed)])
    path = write_chunked(chunked, tmp_path)
    assert path == tmp_path / "fixed" / "000095017024087843.json"
    assert path.exists()
    loaded = load_chunked(tmp_path, "fixed", ACCESSION)
    assert loaded.strategy == "fixed"
    assert loaded.chunks[0].text == "Cybersecurity could harm results."
    assert loaded.chunks[0].item_code == "1A"
    assert chunked_exists(tmp_path, "fixed", ACCESSION)
    assert not chunked_exists(tmp_path, "structural", ACCESSION)


def test_write_chunked_upserts(tmp_path: Path) -> None:
    parsed = _parsed()
    chunked = ChunkedFiling.from_parsed(parsed, "fixed", [_chunk(parsed)])
    first = write_chunked(chunked, tmp_path)
    again = write_chunked(chunked, tmp_path)
    assert again == first


def test_load_chunked_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="chunked filing not found"):
        load_chunked(tmp_path, "fixed", ACCESSION)


def test_chunked_path_strips_dashes(tmp_path: Path) -> None:
    path = chunked_path(tmp_path, "semantic", ACCESSION)
    assert path == tmp_path / "semantic" / "000095017024087843.json"
