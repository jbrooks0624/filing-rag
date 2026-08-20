"""Orchestrate catalog → fetch → parse → write for the configured corpus."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from filing_rag.corpus import CorpusConfig, load_corpus
from filing_rag.ingest.cache import DiskCache, accession_key
from filing_rag.ingest.catalog import FilingRef, select_company_filings
from filing_rag.ingest.client import EdgarClient
from filing_rag.ingest.parse import parse_html, write_parsed
from filing_rag.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestError:
    message: str
    ticker: str | None = None
    fiscal_year: int | None = None
    accession: str | None = None


@dataclass
class IngestResult:
    fetched: int = 0
    skipped: int = 0
    parsed: int = 0
    errors: list[IngestError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class Ingestor:
    """Callable ingest pipeline. Re-runs are cache-first and idempotent."""

    def __init__(
        self,
        corpus: CorpusConfig,
        settings: Settings,
        client: EdgarClient | None = None,
    ) -> None:
        self.corpus = corpus
        self.settings = settings
        self._client = client
        self._cache = DiskCache(settings.raw_dir)

    @classmethod
    def from_config(
        cls,
        corpus_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        client: EdgarClient | None = None,
    ) -> Ingestor:
        resolved = settings or get_settings()
        path = Path(corpus_path) if corpus_path is not None else resolved.corpus_path
        return cls(load_corpus(path), resolved, client=client)

    def run(
        self,
        tickers: Sequence[str] | None = None,
        fiscal_years: Sequence[int] | None = None,
        *,
        force: bool = False,
        parse_only: bool = False,
    ) -> IngestResult:
        companies = self.corpus.select_companies(list(tickers) if tickers else None)
        years = self._resolve_years(fiscal_years)
        result = IngestResult()
        with self._open_client() as client:
            for company in companies:
                try:
                    submissions = client.submissions(
                        company.cik,
                        force=force and not parse_only,
                        since=f"{min(years)}-01-01",
                        until=f"{max(years) + 1}-12-31",
                    )
                    refs = select_company_filings(
                        submissions,
                        company,
                        fiscal_years=years,
                        form_types=self.corpus.form_types,
                    )
                except Exception as exc:  # noqa: BLE001 — isolate one company, keep going
                    result.errors.append(
                        IngestError(message=str(exc), ticker=company.ticker)
                    )
                    continue
                for ref in refs:
                    self._ingest_filing(
                        client,
                        ref,
                        result,
                        force=force,
                        parse_only=parse_only,
                    )
        return result

    def _resolve_years(self, fiscal_years: Sequence[int] | None) -> list[int]:
        if fiscal_years is None:
            return list(self.corpus.fiscal_years)
        years = list(fiscal_years)
        unknown = sorted(set(years) - set(self.corpus.fiscal_years))
        if unknown:
            raise ValueError(f"fiscal years not in corpus: {unknown}")
        return years

    @contextmanager
    def _open_client(self) -> Iterator[EdgarClient]:
        if self._client is not None:
            yield self._client
            return
        with EdgarClient.from_settings(self.settings) as client:
            yield client

    def _ingest_filing(
        self,
        client: EdgarClient,
        ref: FilingRef,
        result: IngestResult,
        *,
        force: bool,
        parse_only: bool,
    ) -> None:
        parsed_path = self.settings.parsed_dir / f"{accession_key(ref.accession)}.json"
        if not force and not parse_only and parsed_path.exists():
            logger.info("skip %s FY%s", ref.ticker, ref.fiscal_year)
            result.skipped += 1
            return
        try:
            html = self._load_html(client, ref, result, force=force, parse_only=parse_only)
            parsed = parse_html(html, ref, items=self.corpus.items)
            write_parsed(parsed, self.settings.parsed_dir)
            result.parsed += 1
            logger.info(
                "parsed %s FY%s (%d sections)",
                ref.ticker,
                ref.fiscal_year,
                len(parsed.sections),
            )
        except Exception as exc:  # noqa: BLE001 — isolate one filing, keep going
            logger.warning("error %s FY%s: %s", ref.ticker, ref.fiscal_year, exc)
            result.errors.append(
                IngestError(
                    message=str(exc),
                    ticker=ref.ticker,
                    fiscal_year=ref.fiscal_year,
                    accession=ref.accession,
                )
            )

    def _load_html(
        self,
        client: EdgarClient,
        ref: FilingRef,
        result: IngestResult,
        *,
        force: bool,
        parse_only: bool,
    ) -> bytes:
        cached = self._cache.get_html(ref.accession)
        if parse_only:
            if cached is None:
                raise FileNotFoundError(
                    f"parse-only: no cached HTML for {ref.ticker} {ref.accession}"
                )
            return cached
        if force or cached is None:
            body = client.primary_doc(
                ref.cik,
                ref.accession,
                ref.primary_doc,
                force=force,
            )
            result.fetched += 1
            return body
        return cached
