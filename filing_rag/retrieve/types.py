"""Citation-bearing retrieval hits. No page numbers — EDGAR HTML has none."""

from __future__ import annotations

from dataclasses import dataclass, field

MODES = ("dense", "sparse", "hybrid")
SNIPPET_LIMIT = 96


def snippet(text: str, limit: int = SNIPPET_LIMIT) -> str:
    """Collapse whitespace and cap length. CLI search and SSE citations share this."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


@dataclass(frozen=True)
class Filters:
    """Metadata scope. ``None`` means unrestricted; an empty tuple matches nothing."""

    tickers: tuple[str, ...] | None = None
    fiscal_years: tuple[int, ...] | None = None
    item_codes: tuple[str, ...] | None = None

    def matches(self, *, ticker: str, fiscal_year: int, item_code: str) -> bool:
        if self.tickers is not None and ticker not in self.tickers:
            return False
        if self.fiscal_years is not None and fiscal_year not in self.fiscal_years:
            return False
        if self.item_codes is not None and item_code not in self.item_codes:
            return False
        return True


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    score: float
    rank: int
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


@dataclass(frozen=True)
class RetrieveTimings:
    encode_ms: float = 0.0
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    fuse_ms: float = 0.0
    rerank_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.encode_ms + self.dense_ms + self.sparse_ms + self.fuse_ms + self.rerank_ms
        )


@dataclass(frozen=True)
class RetrieveResult:
    hits: tuple[Hit, ...] = ()
    mode: str = "hybrid"
    strategy: str = "fixed"
    reranked: bool = False
    timings: RetrieveTimings = field(default_factory=RetrieveTimings)
