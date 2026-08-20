"""Generation usage, timings, and numbered citation blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from filing_rag.retrieve.types import Hit, RetrieveResult, snippet


@dataclass(frozen=True)
class Usage:
    """Token counts from the OpenAI-compatible usage object."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class GenerateTimings:
    generate_ms: float = 0.0


@dataclass(frozen=True)
class CitationBlock:
    """One retrieved chunk as numbered context. Built from Hit, never from gold quotes."""

    index: int
    ticker: str
    fiscal_year: int
    item_code: str
    accession: str
    edgar_url: str
    text: str

    @property
    def citation_tag(self) -> str:
        return f"[{self.ticker} FY{self.fiscal_year} Item {self.item_code}]"

    @property
    def header(self) -> str:
        return (
            f"[{self.index}] {self.ticker} FY{self.fiscal_year} Item {self.item_code} "
            f"accession={self.accession} url={self.edgar_url}"
        )

    @classmethod
    def from_hit(cls, hit: Hit, index: int) -> CitationBlock:
        return cls(
            index=index,
            ticker=hit.ticker,
            fiscal_year=hit.fiscal_year,
            item_code=hit.item_code,
            accession=hit.accession,
            edgar_url=hit.edgar_url,
            text=hit.text,
        )


def blocks_from_hits(hits: Sequence[Hit]) -> tuple[CitationBlock, ...]:
    """Number hits in list order (1-based), not by ``Hit.rank``."""
    return tuple(CitationBlock.from_hit(hit, index) for index, hit in enumerate(hits, start=1))


@dataclass(frozen=True)
class GenerateResult:
    """One completion: text, token usage, API dollars, and wall time."""

    text: str = ""
    usage: Usage = field(default_factory=Usage)
    usd: float = 0.0
    timings: GenerateTimings = field(default_factory=GenerateTimings)
    model: str = ""


@dataclass(frozen=True)
class AskResult:
    """Retrieve then generate. CLI and eval both use this."""

    retrieve: RetrieveResult = field(default_factory=RetrieveResult)
    generate: GenerateResult = field(default_factory=GenerateResult)


@dataclass(frozen=True)
class StreamCitation:
    """One retrieved hit in a ``CitationEvent``. Snippet only — never ``hit.text``."""

    rank: int
    score: float
    ticker: str
    fiscal_year: int
    item_code: str
    accession: str
    edgar_url: str
    snippet: str

    @classmethod
    def from_hit(cls, hit: Hit) -> StreamCitation:
        return cls(
            rank=hit.rank,
            score=hit.score,
            ticker=hit.ticker,
            fiscal_year=hit.fiscal_year,
            item_code=hit.item_code,
            accession=hit.accession,
            edgar_url=hit.edgar_url,
            snippet=snippet(hit.text),
        )


@dataclass(frozen=True)
class CitationEvent:
    """One stream event: all retrieved hits, in rank order."""

    citations: tuple[StreamCitation, ...] = ()

    @classmethod
    def from_hits(cls, hits: Sequence[Hit]) -> CitationEvent:
        return cls(tuple(StreamCitation.from_hit(hit) for hit in hits))


@dataclass(frozen=True)
class TokenEvent:
    delta: str


@dataclass(frozen=True)
class DoneEvent:
    """Final stream event: full answer, usage, retrieve timings, and serving latency."""

    text: str = ""
    usage: Usage = field(default_factory=Usage)
    usd: float = 0.0
    encode_ms: float = 0.0
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    fuse_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    serving_ms: float = 0.0
    model: str = ""
    strategy: str = ""
    mode: str = ""
    rerank: bool = False
    k: int = 0


StreamEvent = CitationEvent | TokenEvent | DoneEvent
