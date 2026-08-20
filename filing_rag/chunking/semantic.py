"""Semantic chunking: split at sentence-similarity drops, then cap oversized spans."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

from filing_rag.chunking.cap import TokenSpan, cap_spans
from filing_rag.chunking.config import ChunkingConfig
from filing_rag.chunking.tokenize import TokenCounter
from filing_rag.chunking.types import Chunk, ChunkedFiling
from filing_rag.ingest.parse import ParsedFiling, Section

STRATEGY = "semantic"
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SemanticChunkError(RuntimeError):
    """Semantic chunking could not run (missing extra, embed failure, …)."""


class BgeSentenceEmbedder:
    """Lazy sentence-transformers wrapper. Import is deferred so tests stay torch-free."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    @classmethod
    def from_config(cls, config: ChunkingConfig) -> BgeSentenceEmbedder:
        return cls(config.semantic.encoder)

    def ensure_available(self) -> None:
        """Raise if sentence-transformers is not installed, without loading weights."""
        try:
            import sentence_transformers
        except ImportError as exc:
            raise SemanticChunkError(
                "Semantic chunking requires sentence-transformers. "
                "Install with: uv sync --dev"
            ) from exc
        version = getattr(sentence_transformers, "__version__", "0")
        if version.split(".", 1)[0] in {"0", "1", "2"}:
            raise SemanticChunkError(
                f"sentence-transformers {version} cannot load bge models. "
                "Reinstall with: uv sync --dev"
            )

    def warm(self) -> None:
        """Download and load the encoder once."""
        self.ensure_available()
        self._load()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise SemanticChunkError(
                    "Semantic chunking requires sentence-transformers. "
                    "Install with: uv sync --dev"
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise SemanticChunkError(
                    f"failed to load encoder {self.model_name}: {exc}"
                ) from exc
        return self._model


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences. Offsets are into `text`."""
    if not text.strip():
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_BOUNDARY.finditer(text):
        end = match.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def split_on_similarity(
    text: str,
    *,
    embedder: Embedder,
    breakpoint_percentile: int,
) -> list[tuple[int, int]]:
    """Group sentences; break where cosine *distance* is at/above the percentile.

    `breakpoint_percentile=95` keeps the largest ~5% of adjacent drops (LangChain default).
    """
    sentences = sentence_spans(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [(sentences[0][0], sentences[-1][1])]
    vectors = embedder.embed([text[start:end].strip() for start, end in sentences])
    if len(vectors) != len(sentences):
        raise SemanticChunkError(
            f"embedder returned {len(vectors)} vectors for {len(sentences)} sentences"
        )
    distances = [
        1.0 - _cosine(vectors[index], vectors[index + 1]) for index in range(len(vectors) - 1)
    ]
    threshold = _percentile(distances, breakpoint_percentile)
    groups: list[tuple[int, int]] = []
    group_start = 0
    for index, distance in enumerate(distances):
        if distance > threshold:
            groups.append((sentences[group_start][0], sentences[index][1]))
            group_start = index + 1
    groups.append((sentences[group_start][0], sentences[-1][1]))
    return groups


def chunk_section(
    section: Section,
    filing: ParsedFiling,
    *,
    config: ChunkingConfig,
    counter: TokenCounter,
    embedder: Embedder,
    start_index: int = 0,
) -> list[Chunk]:
    bounds = split_on_similarity(
        section.text,
        embedder=embedder,
        breakpoint_percentile=config.semantic.breakpoint_percentile,
    )
    candidates = [
        TokenSpan(
            text=section.text[start:end],
            char_start=start,
            char_end=end,
            token_count=counter.count(section.text[start:end]),
        )
        for start, end in bounds
        if section.text[start:end].strip()
    ]
    spans = cap_spans(
        candidates,
        max_tokens=config.max_tokens,
        size=config.fixed.size,
        overlap=config.fixed.overlap,
        counter=counter,
    )
    return [
        Chunk.from_span(
            span,
            filing,
            section,
            strategy=STRATEGY,
            chunk_index=start_index + index,
        )
        for index, span in enumerate(spans)
    ]


def chunk_filing(
    parsed: ParsedFiling,
    *,
    config: ChunkingConfig,
    counter: TokenCounter,
    embedder: Embedder,
) -> ChunkedFiling:
    chunks: list[Chunk] = []
    for section in parsed.sections:
        chunks.extend(
            chunk_section(
                section,
                parsed,
                config=config,
                counter=counter,
                embedder=embedder,
                start_index=len(chunks),
            )
        )
    return ChunkedFiling.from_parsed(parsed, STRATEGY, chunks)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def _percentile(values: Sequence[float], percent: int) -> float:
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    if percent <= 0:
        return ordered[0]
    if percent >= 100:
        return ordered[-1]
    rank = (len(ordered) - 1) * (percent / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight
