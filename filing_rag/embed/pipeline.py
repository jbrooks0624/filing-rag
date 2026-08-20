"""Orchestrate parsed + chunk JSON → Postgres embeddings + HNSW."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from filing_rag.chunking.config import STRATEGIES, ChunkingConfig, load_chunking
from filing_rag.chunking.store import iter_parsed, load_chunked
from filing_rag.corpus import CorpusConfig, load_corpus
from filing_rag.embed.encoder import BgeEmbedder, Embedder
from filing_rag.embed.store import (
    ChunkStore,
    IndexStat,
    PostgresChunkStore,
)
from filing_rag.ingest.parse import ParsedFiling
from filing_rag.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingError:
    message: str
    ticker: str | None = None
    fiscal_year: int | None = None
    accession: str | None = None
    strategy: str | None = None


@dataclass
class IndexResult:
    indexed: int = 0
    skipped: int = 0
    embedded: int = 0
    errors: list[IndexingError] = field(default_factory=list)
    index_stats: list[IndexStat] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class Indexer:
    """Callable index pipeline. Re-runs skip rows that already have embeddings unless forced."""

    def __init__(
        self,
        corpus: CorpusConfig,
        chunking: ChunkingConfig,
        settings: Settings,
        store: ChunkStore,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.corpus = corpus
        self.chunking = chunking
        self.settings = settings
        self.store = store
        self._embedder = embedder

    @classmethod
    def from_config(
        cls,
        corpus_path: str | Path | None = None,
        chunking_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        store: ChunkStore | None = None,
        embedder: Embedder | None = None,
    ) -> Indexer:
        resolved = settings or get_settings()
        corpus = Path(corpus_path) if corpus_path is not None else resolved.corpus_path
        chunking = Path(chunking_path) if chunking_path is not None else resolved.chunking_path
        return cls(
            load_corpus(corpus),
            load_chunking(chunking),
            resolved,
            store if store is not None else PostgresChunkStore.from_settings(resolved),
            embedder=embedder,
        )

    def run(
        self,
        tickers: Sequence[str] | None = None,
        fiscal_years: Sequence[int] | None = None,
        strategies: Sequence[str] | None = None,
        *,
        force: bool = False,
    ) -> IndexResult:
        companies = self.corpus.select_companies(list(tickers) if tickers else None)
        years = self._resolve_years(fiscal_years)
        names = self._resolve_strategies(strategies)
        self._ensure_encoder_available()
        result = IndexResult()
        try:
            self.store.ensure_schema()
        except Exception as exc:  # noqa: BLE001 — schema is a global failure
            result.errors.append(IndexingError(message=str(exc)))
            return result
        expected = {(company.ticker, year) for company in companies for year in years}
        try:
            parsed_filings = [
                filing
                for filing in iter_parsed(self.settings.parsed_dir)
                if (filing.ticker, filing.fiscal_year) in expected
            ]
        except FileNotFoundError as exc:
            result.errors.append(IndexingError(message=str(exc)))
            return result
        found = {(filing.ticker, filing.fiscal_year) for filing in parsed_filings}
        for ticker, year in sorted(expected - found):
            result.errors.append(
                IndexingError(
                    message="parsed filing not found",
                    ticker=ticker,
                    fiscal_year=year,
                )
            )
        for filing in parsed_filings:
            for strategy in names:
                self._index_filing(filing, strategy, result, force=force)
        try:
            result.index_stats = list(self.store.ensure_hnsw())
        except Exception as exc:  # noqa: BLE001 — isolate index build
            result.errors.append(IndexingError(message=str(exc)))
        return result

    def _resolve_years(self, fiscal_years: Sequence[int] | None) -> list[int]:
        if fiscal_years is None:
            return list(self.corpus.fiscal_years)
        years = list(fiscal_years)
        unknown = sorted(set(years) - set(self.corpus.fiscal_years))
        if unknown:
            raise ValueError(f"fiscal years not in corpus: {unknown}")
        return years

    def _resolve_strategies(self, strategies: Sequence[str] | None) -> list[str]:
        known = list(STRATEGIES)
        if strategies is None:
            return known
        names = [name.strip().lower() for name in strategies]
        unknown = sorted({name for name in names if name not in known})
        if unknown:
            raise ValueError(f"unknown strategies: {unknown}. Known: {', '.join(known)}")
        ordered: list[str] = []
        for name in names:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _ensure_encoder_available(self) -> None:
        if self._embedder is not None:
            return
        probe = BgeEmbedder.from_config(self.chunking)
        probe.ensure_available()
        self._embedder = probe

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embedder = self._embedder
        if embedder is None:
            embedder = BgeEmbedder.from_config(self.chunking)
            self._embedder = embedder
        if isinstance(embedder, BgeEmbedder):
            embedder.warm()
        return embedder.embed(texts)

    def _index_filing(
        self,
        filing: ParsedFiling,
        strategy: str,
        result: IndexResult,
        *,
        force: bool,
    ) -> None:
        try:
            chunked = load_chunked(self.settings.chunks_dir, strategy, filing.accession)
        except FileNotFoundError:
            result.errors.append(
                IndexingError(
                    message="chunked filing not found",
                    ticker=filing.ticker,
                    fiscal_year=filing.fiscal_year,
                    accession=filing.accession,
                    strategy=strategy,
                )
            )
            return
        if (
            not force
            and self.store.count(strategy, filing.accession) > 0
            and not self.store.unembedded(strategy, filing.accession)
        ):
            logger.info("skip %s FY%s %s", filing.ticker, filing.fiscal_year, strategy)
            result.skipped += 1
            return
        try:
            self.store.upsert_filing(filing)
            self.store.upsert_chunks(chunked)
            if force:
                self.store.clear_embeddings(strategy, filing.accession)
            pending = self.store.unembedded(strategy, filing.accession)
            if pending:
                vectors = self._embed_texts([row.text for row in pending])
                self.store.write_embeddings([row.id for row in pending], vectors)
            result.indexed += 1
            result.embedded += len(pending)
            logger.info(
                "indexed %s FY%s %s (%d chunks)",
                filing.ticker,
                filing.fiscal_year,
                strategy,
                len(pending),
            )
        except Exception as exc:  # noqa: BLE001 — isolate one filing × strategy
            logger.warning(
                "error %s FY%s %s: %s",
                filing.ticker,
                filing.fiscal_year,
                strategy,
                exc,
            )
            result.errors.append(
                IndexingError(
                    message=str(exc),
                    ticker=filing.ticker,
                    fiscal_year=filing.fiscal_year,
                    accession=filing.accession,
                    strategy=strategy,
                )
            )
