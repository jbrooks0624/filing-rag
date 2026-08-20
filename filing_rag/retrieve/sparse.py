"""Sparse BM25 via bm25s. Not Postgres full-text search."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import bm25s

from filing_rag.embed.store import SearchRow, require_k
from filing_rag.retrieve.dense import hit_from_row
from filing_rag.retrieve.types import Filters, Hit

META_NAME = "meta.json"
STOPWORDS = "english"


class SparseError(RuntimeError):
    """BM25 index build, load, or search failed."""


def _tokenize_corpus(texts: Sequence[str]) -> Any:
    return bm25s.tokenize(list(texts), stopwords=STOPWORDS, show_progress=False)


def _tokenize_query(query: str) -> list[str]:
    tokenized = bm25s.tokenize(
        query, stopwords=STOPWORDS, show_progress=False, return_ids=False
    )
    if not tokenized:
        return []
    return list(tokenized[0])


def _row_payload(row: SearchRow) -> dict[str, object]:
    return {
        "id": row.id,
        "text": row.text,
        "ticker": row.ticker,
        "fiscal_year": row.fiscal_year,
        "item_code": row.item_code,
        "accession": row.accession,
        "char_start": row.char_start,
        "char_end": row.char_end,
        "edgar_url": row.edgar_url,
        "strategy": row.strategy,
        "chunk_index": row.chunk_index,
    }


def _row_from_payload(payload: dict[str, object]) -> SearchRow:
    return SearchRow(
        id=int(payload["id"]),
        text=str(payload["text"]),
        ticker=str(payload["ticker"]),
        fiscal_year=int(payload["fiscal_year"]),
        item_code=str(payload["item_code"]),
        accession=str(payload["accession"]),
        char_start=int(payload["char_start"]),
        char_end=int(payload["char_end"]),
        edgar_url=str(payload["edgar_url"]),
        strategy=str(payload["strategy"]),
        chunk_index=int(payload["chunk_index"]),
    )


class SparseIndex:
    """One BM25 index. Rows stay aligned with corpus positions."""

    def __init__(
        self,
        rows: tuple[SearchRow, ...],
        *,
        k1: float,
        b: float,
        retriever: Any | None = None,
    ) -> None:
        self.rows = rows
        self.k1 = k1
        self.b = b
        self._retriever = retriever

    @classmethod
    def build(cls, rows: Sequence[SearchRow], *, k1: float, b: float) -> SparseIndex:
        records = tuple(rows)
        if not records:
            return cls((), k1=k1, b=b)
        retriever = bm25s.BM25(k1=k1, b=b)
        retriever.index(
            _tokenize_corpus([row.text for row in records]),
            show_progress=False,
        )
        return cls(records, k1=k1, b=b, retriever=retriever)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / META_NAME).write_text(
            json.dumps([_row_payload(row) for row in self.rows]),
            encoding="utf-8",
        )
        if self._retriever is not None:
            self._retriever.save(str(path), show_progress=False)

    @classmethod
    def load(cls, path: Path) -> SparseIndex:
        meta_path = path / META_NAME
        if not meta_path.exists():
            raise SparseError(f"BM25 metadata missing: {meta_path}")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = tuple(_row_from_payload(item) for item in payload)
        if not rows:
            return cls((), k1=1.5, b=0.75)
        retriever = bm25s.BM25.load(str(path), load_corpus=False, show_progress=False)
        return cls(rows, k1=float(retriever.k1), b=float(retriever.b), retriever=retriever)

    @classmethod
    def load_or_build(
        cls,
        path: Path,
        rows: Sequence[SearchRow],
        *,
        k1: float,
        b: float,
        force: bool = False,
    ) -> SparseIndex:
        if not force and (path / META_NAME).exists():
            return cls.load(path)
        index = cls.build(rows, k1=k1, b=b)
        index.save(path)
        return index

    def search(
        self,
        query: str,
        *,
        k: int,
        filters: Filters | None = None,
    ) -> list[Hit]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        k = require_k(k)
        resolved = filters or Filters()
        allowed = [
            index
            for index, row in enumerate(self.rows)
            if resolved.matches(
                ticker=row.ticker,
                fiscal_year=row.fiscal_year,
                item_code=row.item_code,
            )
        ]
        if not allowed:
            return []
        scores = self._scores(query)
        ordered = sorted(allowed, key=lambda index: (-scores[index], index))[:k]
        return [
            hit_from_row(replace(self.rows[index], score=scores[index]), rank)
            for rank, index in enumerate(ordered, start=1)
        ]

    def _scores(self, query: str) -> list[float]:
        if self._retriever is None or not self.rows:
            return [0.0] * len(self.rows)
        tokens = _tokenize_query(query)
        if not tokens:
            return [0.0] * len(self.rows)
        return [float(score) for score in self._retriever.get_scores(tokens)]


def search_sparse(
    index: SparseIndex,
    query: str,
    *,
    k: int,
    filters: Filters | None = None,
) -> list[Hit]:
    return index.search(query, k=k, filters=filters)
