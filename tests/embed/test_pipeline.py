"""Indexer.run is idempotent, isolates errors, and never opens Postgres."""

from pathlib import Path

import pytest
from filing_rag.chunking.config import STRATEGIES
from filing_rag.chunking.store import write_chunked
from filing_rag.chunking.types import Chunk, ChunkedFiling
from filing_rag.embed.pipeline import Indexer
from filing_rag.embed.store import EMBEDDING_DIM, MemoryChunkStore
from filing_rag.ingest.parse import ParsedFiling, Section, write_parsed
from filing_rag.settings import PROJECT_ROOT, Settings

MSFT_ACCESSION = "0000950170-24-087843"


class UnitEmbedder:
    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts.extend(texts)
        vectors: list[list[float]] = []
        for index, _ in enumerate(texts):
            vector = [0.0] * EMBEDDING_DIM
            vector[index % EMBEDDING_DIM] = 1.0
            vectors.append(vector)
        return vectors


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
    text = "Cybersecurity risk could disrupt operations."
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
            )
        ],
    )


def _chunked(parsed: ParsedFiling, strategy: str, text: str = "alpha risk") -> ChunkedFiling:
    chunk = Chunk(
        ticker=parsed.ticker,
        cik=parsed.cik,
        accession=parsed.accession,
        fiscal_year=parsed.fiscal_year,
        edgar_url=parsed.edgar_url,
        item_code="1A",
        item_title="Risk Factors",
        strategy=strategy,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
        text=text,
    )
    return ChunkedFiling.from_parsed(parsed, strategy, [chunk])


def _write_msft(settings: Settings, strategies: tuple[str, ...] = ("fixed",)) -> None:
    parsed = _parsed(ticker="MSFT", accession=MSFT_ACCESSION, cik="0000789019")
    write_parsed(parsed, settings.parsed_dir)
    for strategy in strategies:
        write_chunked(_chunked(parsed, strategy), settings.chunks_dir)


def _indexer(
    tmp_path: Path,
    settings: Settings | None = None,
    *,
    store: MemoryChunkStore | None = None,
    embedder: UnitEmbedder | None = None,
) -> tuple[Indexer, MemoryChunkStore, UnitEmbedder]:
    resolved = settings or _settings(tmp_path)
    memory = store if store is not None else MemoryChunkStore()
    encoder = embedder if embedder is not None else UnitEmbedder()
    indexer = Indexer.from_config(
        resolved.corpus_path,
        resolved.chunking_path,
        settings=resolved,
        store=memory,
        embedder=encoder,
    )
    return indexer, memory, encoder


def test_first_run_embeds_and_builds_hnsw(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_msft(settings, strategies=STRATEGIES)
    indexer, store, embedder = _indexer(tmp_path, settings)
    result = indexer.run(tickers=["MSFT"], fiscal_years=[2024])
    assert result.ok
    assert result.indexed == 3
    assert result.skipped == 0
    assert result.embedded == 3
    assert embedder.calls == 3
    assert store.hnsw_built
    assert [item.strategy for item in result.index_stats] == list(STRATEGIES)
    for strategy in STRATEGIES:
        assert store.count(strategy, MSFT_ACCESSION) == 1
        assert store.unembedded(strategy, MSFT_ACCESSION) == []


def test_second_run_skips_without_embedding(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_msft(settings)
    indexer, _, embedder = _indexer(tmp_path, settings)
    first = indexer.run(tickers=["MSFT"], strategies=["fixed"])
    second = indexer.run(tickers=["MSFT"], strategies=["fixed"])
    assert first.indexed == 1
    assert first.embedded == 1
    assert second.skipped == 1
    assert second.indexed == 0
    assert second.embedded == 0
    assert embedder.calls == 1


def test_force_reembeds(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_msft(settings)
    indexer, _, embedder = _indexer(tmp_path, settings)
    indexer.run(tickers=["MSFT"], strategies=["fixed"])
    forced = indexer.run(tickers=["MSFT"], strategies=["fixed"], force=True)
    assert forced.indexed == 1
    assert forced.skipped == 0
    assert forced.embedded == 1
    assert embedder.calls == 2


def test_missing_chunk_json_is_an_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    write_parsed(
        _parsed(ticker="MSFT", accession=MSFT_ACCESSION, cik="0000789019"),
        settings.parsed_dir,
    )
    result = _indexer(tmp_path, settings)[0].run(tickers=["MSFT"], strategies=["fixed"])
    assert not result.ok
    assert result.indexed == 0
    assert result.errors[0].ticker == "MSFT"
    assert result.errors[0].strategy == "fixed"
    assert "chunked filing not found" in result.errors[0].message


def test_missing_parsed_is_an_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.parsed_dir.mkdir(parents=True)
    result = _indexer(tmp_path, settings)[0].run(tickers=["MSFT"], strategies=["fixed"])
    assert not result.ok
    assert result.errors[0].ticker == "MSFT"
    assert "parsed filing not found" in result.errors[0].message


def test_missing_parsed_dir_is_an_error(tmp_path: Path) -> None:
    result = _indexer(tmp_path)[0].run(tickers=["MSFT"], strategies=["fixed"])
    assert not result.ok
    assert "parsed directory not found" in result.errors[0].message


def test_one_missing_filing_does_not_stop_the_other(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_msft(settings)
    result = _indexer(tmp_path, settings)[0].run(
        tickers=["MSFT", "JPM"], strategies=["fixed"]
    )
    assert result.indexed == 1
    assert len(result.errors) == 1
    assert result.errors[0].ticker == "JPM"


def test_unknown_year_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2020"):
        _indexer(tmp_path)[0].run(
            tickers=["MSFT"], fiscal_years=[2020], strategies=["fixed"]
        )


def test_unknown_strategy_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="recursive"):
        _indexer(tmp_path)[0].run(tickers=["MSFT"], strategies=["recursive"])


def test_unknown_ticker_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="NVDA"):
        _indexer(tmp_path)[0].run(tickers=["NVDA"], strategies=["fixed"])
