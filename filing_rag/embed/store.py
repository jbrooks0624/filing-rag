"""Postgres upserts for filings, sections, and chunk embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Protocol

import psycopg
from pgvector.psycopg import register_vector

from filing_rag.chunking.config import STRATEGIES
from filing_rag.chunking.types import ChunkedFiling
from filing_rag.ingest.parse import ParsedFiling
from filing_rag.settings import Settings, get_settings

EMBEDDING_DIM = 768
HNSW_INDEXES = {
    "fixed": "chunks_hnsw_fixed",
    "structural": "chunks_hnsw_structural",
    "semantic": "chunks_hnsw_semantic",
}


class StoreError(RuntimeError):
    """Schema, upsert, or connection failed."""


@dataclass(frozen=True)
class ChunkRow:
    id: int
    text: str
    strategy: str
    accession: str
    chunk_index: int


@dataclass(frozen=True)
class IndexStat:
    strategy: str
    name: str
    bytes: int
    build_ms: float | None = None


@dataclass(frozen=True)
class SearchRow:
    """Citation payload for dense search and sparse index builds."""

    id: int
    text: str
    ticker: str
    fiscal_year: int
    item_code: str
    accession: str
    char_start: int
    char_end: int
    edgar_url: str
    strategy: str
    chunk_index: int
    score: float | None = None


class ChunkStore(Protocol):
    def ensure_schema(self) -> None: ...

    def upsert_filing(self, filing: ParsedFiling) -> int: ...

    def upsert_chunks(self, chunked: ChunkedFiling) -> list[int]: ...

    def unembedded(self, strategy: str, accession: str) -> list[ChunkRow]: ...

    def count(self, strategy: str, accession: str) -> int: ...

    def clear_embeddings(self, strategy: str, accession: str) -> int: ...

    def write_embeddings(
        self, chunk_ids: Sequence[int], vectors: Sequence[Sequence[float]]
    ) -> None: ...

    def search_dense(
        self,
        vector: Sequence[float],
        strategy: str,
        k: int,
        *,
        tickers: Sequence[str] | None = None,
        fiscal_years: Sequence[int] | None = None,
        item_codes: Sequence[str] | None = None,
    ) -> list[SearchRow]: ...

    def iter_chunks(self, strategy: str) -> Iterable[SearchRow]: ...

    def ensure_hnsw(self) -> list[IndexStat]: ...

    def index_stats(self) -> list[IndexStat]: ...


def hnsw_statement(strategy: str) -> str:
    if strategy not in HNSW_INDEXES:
        raise ValueError(f"unknown strategy {strategy!r}")
    name = HNSW_INDEXES[strategy]
    return (
        f"CREATE INDEX IF NOT EXISTS {name} ON chunks USING hnsw "
        f"(embedding vector_cosine_ops) WHERE strategy = '{strategy}'"
    )


DENSE_SEARCH_SQL = """
SELECT
    c.id,
    1 - (c.embedding <=> %s::vector) AS score,
    c.text,
    c.ticker,
    c.fiscal_year,
    c.item_code,
    c.accession,
    c.char_start,
    c.char_end,
    f.edgar_url,
    c.strategy,
    c.chunk_index
FROM chunks c
JOIN filings f ON f.id = c.filing_id
WHERE c.strategy = %s
  AND c.embedding IS NOT NULL
  AND (%s::text[] IS NULL OR c.ticker = ANY(%s))
  AND (%s::int[] IS NULL OR c.fiscal_year = ANY(%s))
  AND (%s::text[] IS NULL OR c.item_code = ANY(%s))
ORDER BY c.embedding <=> %s::vector
LIMIT %s
"""


def dense_search_sql() -> str:
    return DENSE_SEARCH_SQL


def validate_embeddings(
    chunk_ids: Sequence[int], vectors: Sequence[Sequence[float]]
) -> None:
    if len(chunk_ids) != len(vectors):
        raise ValueError(
            f"chunk_ids ({len(chunk_ids)}) and vectors ({len(vectors)}) length mismatch"
        )
    for vector in vectors:
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(f"expected {EMBEDDING_DIM}-d embeddings, got {len(vector)}")


def require_strategy(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    return strategy


def require_k(k: int) -> int:
    if k < 1:
        raise ValueError("k must be >= 1")
    return k


def query_vector(vector: Sequence[float]) -> list[float]:
    if len(vector) != EMBEDDING_DIM:
        raise ValueError(f"expected {EMBEDDING_DIM}-d embeddings, got {len(vector)}")
    return list(vector)


def metadata_matches(
    *,
    ticker: str,
    fiscal_year: int,
    item_code: str,
    tickers: Sequence[str] | None = None,
    fiscal_years: Sequence[int] | None = None,
    item_codes: Sequence[str] | None = None,
) -> bool:
    if tickers is not None and ticker not in tickers:
        return False
    if fiscal_years is not None and fiscal_year not in fiscal_years:
        return False
    if item_codes is not None and item_code not in item_codes:
        return False
    return True


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


class PostgresChunkStore:
    """psycopg store. Connects lazily so constructing the object does not open a socket."""

    def __init__(self, database_url: str, migrations_dir: Path) -> None:
        self.database_url = database_url
        self.migrations_dir = migrations_dir
        self._conn: psycopg.Connection | None = None
        self._hnsw_build_ms: dict[str, float] = {}
        self._vector_registered = False

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> PostgresChunkStore:
        resolved = settings or get_settings()
        return cls(resolved.require_database_url(), resolved.migrations_dir)

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def ensure_schema(self) -> None:
        path = self.migrations_dir / "001_init.sql"
        sql = path.read_text(encoding="utf-8")
        conn = self._connect()
        conn.execute(sql, prepare=False)
        conn.commit()
        self._register_vector(conn)

    def upsert_filing(self, filing: ParsedFiling) -> int:
        conn = self._connection()
        row = conn.execute(
            """
            INSERT INTO filings (
                accession, ticker, cik, company_name, form, filing_date,
                period_of_report, fiscal_year, primary_doc, edgar_url
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (accession) DO UPDATE SET
                ticker = EXCLUDED.ticker,
                cik = EXCLUDED.cik,
                company_name = EXCLUDED.company_name,
                form = EXCLUDED.form,
                filing_date = EXCLUDED.filing_date,
                period_of_report = EXCLUDED.period_of_report,
                fiscal_year = EXCLUDED.fiscal_year,
                primary_doc = EXCLUDED.primary_doc,
                edgar_url = EXCLUDED.edgar_url
            RETURNING id
            """,
            (
                filing.accession,
                filing.ticker,
                filing.cik,
                filing.company_name,
                filing.form,
                filing.filing_date,
                filing.period_of_report,
                filing.fiscal_year,
                filing.primary_doc,
                filing.edgar_url,
            ),
        ).fetchone()
        assert row is not None
        filing_id = int(row[0])
        for section in filing.sections:
            conn.execute(
                """
                INSERT INTO sections (
                    filing_id, item_code, item_title, text, char_start, char_end
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (filing_id, item_code) DO UPDATE SET
                    item_title = EXCLUDED.item_title,
                    text = EXCLUDED.text,
                    char_start = EXCLUDED.char_start,
                    char_end = EXCLUDED.char_end
                """,
                (
                    filing_id,
                    section.item_code,
                    section.item_title,
                    section.text,
                    section.char_start,
                    section.char_end,
                ),
            )
        conn.commit()
        return filing_id

    def upsert_chunks(self, chunked: ChunkedFiling) -> list[int]:
        conn = self._connection()
        filing_row = conn.execute(
            "SELECT id FROM filings WHERE accession = %s",
            (chunked.accession,),
        ).fetchone()
        if filing_row is None:
            raise StoreError(f"filing not upserted: {chunked.accession}")
        filing_id = int(filing_row[0])
        section_ids = {
            str(item_code): int(section_id)
            for item_code, section_id in conn.execute(
                "SELECT item_code, id FROM sections WHERE filing_id = %s",
                (filing_id,),
            )
        }
        ids: list[int] = []
        for chunk in chunked.chunks:
            section_id = section_ids.get(chunk.item_code)
            if section_id is None:
                raise StoreError(
                    f"section {chunk.item_code} missing for {chunked.accession}"
                )
            row = conn.execute(
                """
                INSERT INTO chunks (
                    filing_id, section_id, strategy, chunk_index, char_start, char_end,
                    token_count, text, ticker, fiscal_year, item_code, accession
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (section_id, strategy, chunk_index) DO UPDATE SET
                    filing_id = EXCLUDED.filing_id,
                    char_start = EXCLUDED.char_start,
                    char_end = EXCLUDED.char_end,
                    token_count = EXCLUDED.token_count,
                    ticker = EXCLUDED.ticker,
                    fiscal_year = EXCLUDED.fiscal_year,
                    item_code = EXCLUDED.item_code,
                    accession = EXCLUDED.accession,
                    embedding = CASE
                        WHEN chunks.text IS NOT DISTINCT FROM EXCLUDED.text
                        THEN chunks.embedding
                        ELSE NULL
                    END,
                    text = EXCLUDED.text
                RETURNING id
                """,
                (
                    filing_id,
                    section_id,
                    chunk.strategy,
                    chunk.chunk_index,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.token_count,
                    chunk.text,
                    chunk.ticker,
                    chunk.fiscal_year,
                    chunk.item_code,
                    chunk.accession,
                ),
            ).fetchone()
            assert row is not None
            ids.append(int(row[0]))
        conn.commit()
        return ids

    def unembedded(self, strategy: str, accession: str) -> list[ChunkRow]:
        conn = self._connection()
        rows = conn.execute(
            """
            SELECT id, text, strategy, accession, chunk_index
            FROM chunks
            WHERE strategy = %s AND accession = %s AND embedding IS NULL
            ORDER BY chunk_index
            """,
            (strategy, accession),
        ).fetchall()
        return [
            ChunkRow(
                id=int(row[0]),
                text=str(row[1]),
                strategy=str(row[2]),
                accession=str(row[3]),
                chunk_index=int(row[4]),
            )
            for row in rows
        ]

    def count(self, strategy: str, accession: str) -> int:
        conn = self._connection()
        row = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE strategy = %s AND accession = %s",
            (strategy, accession),
        ).fetchone()
        assert row is not None
        return int(row[0])

    def clear_embeddings(self, strategy: str, accession: str) -> int:
        conn = self._connection()
        row = conn.execute(
            """
            UPDATE chunks SET embedding = NULL
            WHERE strategy = %s AND accession = %s
            """,
            (strategy, accession),
        )
        conn.commit()
        return row.rowcount

    def write_embeddings(
        self, chunk_ids: Sequence[int], vectors: Sequence[Sequence[float]]
    ) -> None:
        validate_embeddings(chunk_ids, vectors)
        conn = self._connection()
        for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
            result = conn.execute(
                "UPDATE chunks SET embedding = %s WHERE id = %s",
                (list(vector), chunk_id),
            )
            if result.rowcount != 1:
                raise StoreError(f"chunk id {chunk_id} not found")
        conn.commit()

    def search_dense(
        self,
        vector: Sequence[float],
        strategy: str,
        k: int,
        *,
        tickers: Sequence[str] | None = None,
        fiscal_years: Sequence[int] | None = None,
        item_codes: Sequence[str] | None = None,
    ) -> list[SearchRow]:
        query = query_vector(vector)
        strategy = require_strategy(strategy)
        k = require_k(k)
        ticker_filter = list(tickers) if tickers is not None else None
        year_filter = list(fiscal_years) if fiscal_years is not None else None
        item_filter = list(item_codes) if item_codes is not None else None
        conn = self._connection()
        try:
            rows = conn.execute(
                DENSE_SEARCH_SQL,
                (
                    query,
                    strategy,
                    ticker_filter,
                    ticker_filter,
                    year_filter,
                    year_filter,
                    item_filter,
                    item_filter,
                    query,
                    k,
                ),
            ).fetchall()
        except psycopg.Error as exc:
            raise StoreError(f"dense search failed: {exc}") from exc
        return [
            SearchRow(
                id=int(row[0]),
                score=float(row[1]),
                text=str(row[2]),
                ticker=str(row[3]),
                fiscal_year=int(row[4]),
                item_code=str(row[5]),
                accession=str(row[6]),
                char_start=int(row[7]),
                char_end=int(row[8]),
                edgar_url=str(row[9]),
                strategy=str(row[10]),
                chunk_index=int(row[11]),
            )
            for row in rows
        ]

    def iter_chunks(self, strategy: str) -> Iterable[SearchRow]:
        strategy = require_strategy(strategy)
        conn = self._connection()
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.text,
                c.ticker,
                c.fiscal_year,
                c.item_code,
                c.accession,
                c.char_start,
                c.char_end,
                f.edgar_url,
                c.strategy,
                c.chunk_index
            FROM chunks c
            JOIN filings f ON f.id = c.filing_id
            WHERE c.strategy = %s
            ORDER BY c.accession, c.chunk_index
            """,
            (strategy,),
        )
        for row in rows:
            yield SearchRow(
                id=int(row[0]),
                text=str(row[1]),
                ticker=str(row[2]),
                fiscal_year=int(row[3]),
                item_code=str(row[4]),
                accession=str(row[5]),
                char_start=int(row[6]),
                char_end=int(row[7]),
                edgar_url=str(row[8]),
                strategy=str(row[9]),
                chunk_index=int(row[10]),
            )

    def ensure_hnsw(self) -> list[IndexStat]:
        conn = self._connection()
        for strategy in STRATEGIES:
            name = HNSW_INDEXES[strategy]
            existed = conn.execute(
                "SELECT 1 FROM pg_class WHERE relname = %s",
                (name,),
            ).fetchone() is not None
            started = perf_counter()
            conn.execute(hnsw_statement(strategy), prepare=False)
            elapsed_ms = (perf_counter() - started) * 1000
            self._hnsw_build_ms[strategy] = 0.0 if existed else elapsed_ms
        conn.commit()
        return self.index_stats()

    def index_stats(self) -> list[IndexStat]:
        conn = self._connection()
        stats: list[IndexStat] = []
        for strategy in STRATEGIES:
            name = HNSW_INDEXES[strategy]
            row = conn.execute(
                "SELECT pg_relation_size(%s::regclass)",
                (name,),
            ).fetchone()
            stats.append(
                IndexStat(
                    strategy=strategy,
                    name=name,
                    bytes=int(row[0]) if row is not None else 0,
                    build_ms=self._hnsw_build_ms.get(strategy),
                )
            )
        return stats

    def _connection(self) -> psycopg.Connection:
        conn = self._connect()
        if not self._vector_registered:
            self._register_vector(conn)
        return conn

    def _connect(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg.connect(self.database_url)
            except psycopg.OperationalError as exc:
                raise StoreError(
                    "could not connect to Postgres. Start it with: docker compose up -d"
                ) from exc
            self._vector_registered = False
        return self._conn

    def _register_vector(self, conn: psycopg.Connection) -> None:
        try:
            register_vector(conn)
        except psycopg.ProgrammingError as exc:
            if "vector type not found" in str(exc).lower():
                raise StoreError(
                    "pgvector extension is missing. Use docker compose up -d "
                    "(image pgvector/pgvector:pg17), not a stock Postgres."
                ) from exc
            raise
        self._vector_registered = True


@dataclass
class _MemChunk:
    id: int
    filing_id: int
    section_id: int
    strategy: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    text: str
    ticker: str
    fiscal_year: int
    item_code: str
    accession: str
    embedding: list[float] | None = None


@dataclass
class MemoryChunkStore:
    """In-memory ChunkStore. Tests inject this; it never opens a socket."""

    filings: dict[str, int] = field(default_factory=dict)
    edgar_urls: dict[str, str] = field(default_factory=dict)
    sections: dict[tuple[int, str], int] = field(default_factory=dict)
    chunks: dict[int, _MemChunk] = field(default_factory=dict)
    hnsw_built: bool = False
    _next_filing_id: int = 1
    _next_section_id: int = 1
    _next_chunk_id: int = 1

    def ensure_schema(self) -> None:
        return None

    def upsert_filing(self, filing: ParsedFiling) -> int:
        filing_id = self.filings.get(filing.accession)
        if filing_id is None:
            filing_id = self._next_filing_id
            self._next_filing_id += 1
            self.filings[filing.accession] = filing_id
        self.edgar_urls[filing.accession] = filing.edgar_url
        for section in filing.sections:
            key = (filing_id, section.item_code)
            if key not in self.sections:
                self.sections[key] = self._next_section_id
                self._next_section_id += 1
        return filing_id

    def upsert_chunks(self, chunked: ChunkedFiling) -> list[int]:
        filing_id = self.filings.get(chunked.accession)
        if filing_id is None:
            raise StoreError(f"filing not upserted: {chunked.accession}")
        ids: list[int] = []
        by_key = {
            (row.section_id, row.strategy, row.chunk_index): row
            for row in self.chunks.values()
        }
        for chunk in chunked.chunks:
            section_id = self.sections.get((filing_id, chunk.item_code))
            if section_id is None:
                raise StoreError(
                    f"section {chunk.item_code} missing for {chunked.accession}"
                )
            existing = by_key.get((section_id, chunk.strategy, chunk.chunk_index))
            if existing is None:
                record = _MemChunk(
                    id=self._next_chunk_id,
                    filing_id=filing_id,
                    section_id=section_id,
                    strategy=chunk.strategy,
                    chunk_index=chunk.chunk_index,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    token_count=chunk.token_count,
                    text=chunk.text,
                    ticker=chunk.ticker,
                    fiscal_year=chunk.fiscal_year,
                    item_code=chunk.item_code,
                    accession=chunk.accession,
                )
                self._next_chunk_id += 1
                self.chunks[record.id] = record
                ids.append(record.id)
                continue
            if existing.text != chunk.text:
                existing.embedding = None
            existing.char_start = chunk.char_start
            existing.char_end = chunk.char_end
            existing.token_count = chunk.token_count
            existing.text = chunk.text
            existing.ticker = chunk.ticker
            existing.fiscal_year = chunk.fiscal_year
            existing.item_code = chunk.item_code
            existing.accession = chunk.accession
            ids.append(existing.id)
        return ids

    def unembedded(self, strategy: str, accession: str) -> list[ChunkRow]:
        rows = [
            row
            for row in self.chunks.values()
            if row.strategy == strategy and row.accession == accession and row.embedding is None
        ]
        rows.sort(key=lambda row: row.chunk_index)
        return [
            ChunkRow(
                id=row.id,
                text=row.text,
                strategy=row.strategy,
                accession=row.accession,
                chunk_index=row.chunk_index,
            )
            for row in rows
        ]

    def count(self, strategy: str, accession: str) -> int:
        return sum(
            1
            for row in self.chunks.values()
            if row.strategy == strategy and row.accession == accession
        )

    def clear_embeddings(self, strategy: str, accession: str) -> int:
        cleared = 0
        for row in self.chunks.values():
            if row.strategy == strategy and row.accession == accession:
                row.embedding = None
                cleared += 1
        return cleared

    def write_embeddings(
        self, chunk_ids: Sequence[int], vectors: Sequence[Sequence[float]]
    ) -> None:
        validate_embeddings(chunk_ids, vectors)
        for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
            row = self.chunks.get(chunk_id)
            if row is None:
                raise StoreError(f"chunk id {chunk_id} not found")
            row.embedding = list(vector)

    def search_dense(
        self,
        vector: Sequence[float],
        strategy: str,
        k: int,
        *,
        tickers: Sequence[str] | None = None,
        fiscal_years: Sequence[int] | None = None,
        item_codes: Sequence[str] | None = None,
    ) -> list[SearchRow]:
        query = query_vector(vector)
        strategy = require_strategy(strategy)
        k = require_k(k)
        scored: list[tuple[float, _MemChunk]] = []
        for row in self.chunks.values():
            if row.strategy != strategy or row.embedding is None:
                continue
            if not metadata_matches(
                ticker=row.ticker,
                fiscal_year=row.fiscal_year,
                item_code=row.item_code,
                tickers=tickers,
                fiscal_years=fiscal_years,
                item_codes=item_codes,
            ):
                continue
            scored.append((cosine_similarity(query, row.embedding), row))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [self._search_row(row, score=score) for score, row in scored[:k]]

    def iter_chunks(self, strategy: str) -> Iterable[SearchRow]:
        strategy = require_strategy(strategy)
        rows = [row for row in self.chunks.values() if row.strategy == strategy]
        rows.sort(key=lambda row: (row.accession, row.chunk_index))
        for row in rows:
            yield self._search_row(row)

    def _search_row(self, row: _MemChunk, score: float | None = None) -> SearchRow:
        return SearchRow(
            id=row.id,
            text=row.text,
            ticker=row.ticker,
            fiscal_year=row.fiscal_year,
            item_code=row.item_code,
            accession=row.accession,
            char_start=row.char_start,
            char_end=row.char_end,
            edgar_url=self.edgar_urls.get(row.accession, ""),
            strategy=row.strategy,
            chunk_index=row.chunk_index,
            score=score,
        )

    def ensure_hnsw(self) -> list[IndexStat]:
        self.hnsw_built = True
        return self.index_stats()

    def index_stats(self) -> list[IndexStat]:
        return [
            IndexStat(
                strategy=strategy,
                name=HNSW_INDEXES[strategy],
                bytes=0,
                build_ms=0.0 if self.hnsw_built else None,
            )
            for strategy in STRATEGIES
        ]
