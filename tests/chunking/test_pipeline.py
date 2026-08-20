"""Chunker.run is idempotent, isolates errors, and never hits EDGAR."""

from pathlib import Path

import pytest
from filing_rag.chunking.pipeline import Chunker
from filing_rag.chunking.store import chunked_exists, load_chunked
from filing_rag.chunking.tokenize import WhitespaceTokenCounter
from filing_rag.ingest.parse import ParsedFiling, Section, write_parsed
from filing_rag.settings import PROJECT_ROOT, Settings

MSFT_ACCESSION = "0000950170-24-087843"


class UnitEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _settings(tmp_path: Path) -> Settings:
    corpus = tmp_path / "corpus.yaml"
    corpus.write_text(
        """
form_types: ["10-K"]
fiscal_years: [2024]
items: ["1A", "7", "7A"]
companies:
  - ticker: MSFT
    cik: "0000789019"
    sector: tech
  - ticker: JPM
    cik: "0000019617"
    sector: banks
""".strip()
        + "\n"
    )
    return Settings(
        data_dir=tmp_path / "data",
        corpus_path=corpus,
        chunking_path=PROJECT_ROOT / "config" / "chunking.yaml",
    )


def _parsed(*, ticker: str, accession: str, cik: str) -> ParsedFiling:
    text = (
        "Cybersecurity Risk\n"
        "We face cybersecurity threats that could disrupt operations.\n"
        "Competition\n"
        "The markets in which we compete change rapidly. Alpha stays. Zeta switches."
    )
    return ParsedFiling(
        ticker=ticker,
        cik=cik,
        accession=accession,
        form="10-K",
        filing_date="2024-07-30",
        period_of_report="2024-06-30",
        fiscal_year=2024,
        primary_doc=f"{ticker.lower()}-20240630.htm",
        edgar_url="https://example.com",
        sections=[
            Section(
                item_code="1A",
                item_title="Risk Factors",
                text=text,
                char_start=0,
                char_end=len(text),
            ),
            Section(
                item_code="7",
                item_title="MD&A",
                text="Revenue grew year over year.",
                char_start=0,
                char_end=28,
            ),
            Section(
                item_code="7A",
                item_title="Market Risk",
                text="Interest rates may fluctuate.",
                char_start=0,
                char_end=29,
            ),
        ],
    )


def _write_msft(settings: Settings) -> None:
    write_parsed(
        _parsed(ticker="MSFT", accession=MSFT_ACCESSION, cik="0000789019"),
        settings.parsed_dir,
    )


def _chunker(tmp_path: Path, settings: Settings | None = None) -> Chunker:
    resolved = settings or _settings(tmp_path)
    return Chunker.from_config(
        resolved.corpus_path,
        resolved.chunking_path,
        settings=resolved,
        counter=WhitespaceTokenCounter(),
        embedder=UnitEmbedder(),
    )


def test_first_run_writes_all_strategies(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_msft(settings)
    result = _chunker(tmp_path, settings).run(tickers=["MSFT"], fiscal_years=[2024])
    assert result.ok
    assert result.chunked == 3
    assert result.skipped == 0
    assert result.chunks_written > 0
    for strategy in ("fixed", "structural", "semantic"):
        assert chunked_exists(settings.chunks_dir, strategy, MSFT_ACCESSION)
        loaded = load_chunked(settings.chunks_dir, strategy, MSFT_ACCESSION)
        assert loaded.strategy == strategy
        assert loaded.chunks
        assert all(chunk.token_count <= 512 for chunk in loaded.chunks)


def test_second_run_skips(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_msft(settings)
    chunker = _chunker(tmp_path, settings)
    first = chunker.run(tickers=["MSFT"], strategies=["fixed"])
    second = chunker.run(tickers=["MSFT"], strategies=["fixed"])
    assert first.chunked == 1
    assert second.skipped == 1
    assert second.chunked == 0
    assert second.chunks_written == 0


def test_force_rewrites(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_msft(settings)
    chunker = _chunker(tmp_path, settings)
    chunker.run(tickers=["MSFT"], strategies=["fixed"])
    forced = chunker.run(tickers=["MSFT"], strategies=["fixed"], force=True)
    assert forced.chunked == 1
    assert forced.skipped == 0


def test_missing_parsed_is_an_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.parsed_dir.mkdir(parents=True)
    result = _chunker(tmp_path, settings).run(tickers=["MSFT"], strategies=["fixed"])
    assert not result.ok
    assert result.chunked == 0
    assert result.errors[0].ticker == "MSFT"
    assert "parsed filing not found" in result.errors[0].message


def test_missing_parsed_dir_is_an_error(tmp_path: Path) -> None:
    result = _chunker(tmp_path).run(tickers=["MSFT"], strategies=["fixed"])
    assert not result.ok
    assert "parsed directory not found" in result.errors[0].message


def test_one_missing_filing_does_not_stop_the_other(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_msft(settings)
    result = _chunker(tmp_path, settings).run(tickers=["MSFT", "JPM"], strategies=["fixed"])
    assert result.chunked == 1
    assert len(result.errors) == 1
    assert result.errors[0].ticker == "JPM"


def test_unknown_year_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2020"):
        _chunker(tmp_path).run(tickers=["MSFT"], fiscal_years=[2020], strategies=["fixed"])


def test_unknown_strategy_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="recursive"):
        _chunker(tmp_path).run(tickers=["MSFT"], strategies=["recursive"])


def test_unknown_ticker_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="NVDA"):
        _chunker(tmp_path).run(tickers=["NVDA"], strategies=["fixed"])
