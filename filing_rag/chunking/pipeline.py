"""Orchestrate parsed filings → strategy chunkers → data/chunks/{strategy}/."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from filing_rag.chunking.config import STRATEGIES, ChunkingConfig, load_chunking
from filing_rag.chunking.fixed import chunk_filing as chunk_fixed
from filing_rag.chunking.semantic import BgeSentenceEmbedder, Embedder
from filing_rag.chunking.semantic import chunk_filing as chunk_semantic
from filing_rag.chunking.store import chunked_exists, iter_parsed, write_chunked
from filing_rag.chunking.structural import chunk_filing as chunk_structural
from filing_rag.chunking.tokenize import HuggingFaceTokenCounter, TokenCounter
from filing_rag.chunking.types import ChunkedFiling
from filing_rag.corpus import CorpusConfig, load_corpus
from filing_rag.ingest.parse import ParsedFiling
from filing_rag.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkError:
    message: str
    ticker: str | None = None
    fiscal_year: int | None = None
    accession: str | None = None
    strategy: str | None = None


@dataclass
class ChunkResult:
    chunked: int = 0
    skipped: int = 0
    chunks_written: int = 0
    errors: list[ChunkError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class Chunker:
    """Callable chunking pipeline. Re-runs skip existing strategy files unless forced."""

    def __init__(
        self,
        corpus: CorpusConfig,
        chunking: ChunkingConfig,
        settings: Settings,
        *,
        counter: TokenCounter | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.corpus = corpus
        self.chunking = chunking
        self.settings = settings
        self._counter = counter
        self._embedder = embedder

    @classmethod
    def from_config(
        cls,
        corpus_path: str | Path | None = None,
        chunking_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        counter: TokenCounter | None = None,
        embedder: Embedder | None = None,
    ) -> Chunker:
        resolved = settings or get_settings()
        corpus = Path(corpus_path) if corpus_path is not None else resolved.corpus_path
        chunking = Path(chunking_path) if chunking_path is not None else resolved.chunking_path
        return cls(
            load_corpus(corpus),
            load_chunking(chunking),
            resolved,
            counter=counter,
            embedder=embedder,
        )

    def run(
        self,
        tickers: Sequence[str] | None = None,
        fiscal_years: Sequence[int] | None = None,
        strategies: Sequence[str] | None = None,
        *,
        force: bool = False,
    ) -> ChunkResult:
        companies = self.corpus.select_companies(list(tickers) if tickers else None)
        years = self._resolve_years(fiscal_years)
        names = self._resolve_strategies(strategies)
        if "semantic" in names:
            self._require_embedder()
        result = ChunkResult()
        expected = {(company.ticker, year) for company in companies for year in years}
        try:
            parsed_filings = [
                filing
                for filing in iter_parsed(self.settings.parsed_dir)
                if (filing.ticker, filing.fiscal_year) in expected
            ]
        except FileNotFoundError as exc:
            result.errors.append(ChunkError(message=str(exc)))
            return result
        found = {(filing.ticker, filing.fiscal_year) for filing in parsed_filings}
        for ticker, year in sorted(expected - found):
            result.errors.append(
                ChunkError(
                    message="parsed filing not found",
                    ticker=ticker,
                    fiscal_year=year,
                )
            )
        for filing in parsed_filings:
            for strategy in names:
                self._chunk_filing(filing, strategy, result, force=force)
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

    def _counter_or_default(self) -> TokenCounter:
        if self._counter is None:
            self._counter = HuggingFaceTokenCounter.from_config(self.chunking)
        return self._counter

    def _require_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = BgeSentenceEmbedder.from_config(self.chunking)
        if isinstance(self._embedder, BgeSentenceEmbedder):
            self._embedder.warm()
        return self._embedder

    def _apply(self, filing: ParsedFiling, strategy: str) -> ChunkedFiling:
        counter = self._counter_or_default()
        if strategy == "fixed":
            return chunk_fixed(filing, config=self.chunking, counter=counter)
        if strategy == "structural":
            return chunk_structural(filing, config=self.chunking, counter=counter)
        if strategy == "semantic":
            return chunk_semantic(
                filing,
                config=self.chunking,
                counter=counter,
                embedder=self._require_embedder(),
            )
        raise ValueError(f"unknown strategy {strategy!r}")

    def _chunk_filing(
        self,
        filing: ParsedFiling,
        strategy: str,
        result: ChunkResult,
        *,
        force: bool,
    ) -> None:
        if not force and chunked_exists(self.settings.chunks_dir, strategy, filing.accession):
            logger.info("skip %s FY%s %s", filing.ticker, filing.fiscal_year, strategy)
            result.skipped += 1
            return
        try:
            chunked = self._apply(filing, strategy)
            write_chunked(chunked, self.settings.chunks_dir)
            result.chunked += 1
            result.chunks_written += len(chunked.chunks)
            logger.info(
                "chunked %s FY%s %s (%d chunks)",
                filing.ticker,
                filing.fiscal_year,
                strategy,
                len(chunked.chunks),
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
                ChunkError(
                    message=str(exc),
                    ticker=filing.ticker,
                    fiscal_year=filing.fiscal_year,
                    accession=filing.accession,
                    strategy=strategy,
                )
            )
