"""Semantic strategy: sentence embeddings, percentile breakpoints, then cap."""

from collections.abc import Sequence

import pytest
from filing_rag.chunking.config import (
    ChunkingConfig,
    FixedConfig,
    SemanticConfig,
    StructuralConfig,
)
from filing_rag.chunking.semantic import (
    STRATEGY,
    BgeSentenceEmbedder,
    SemanticChunkError,
    chunk_filing,
    chunk_section,
    sentence_spans,
    split_on_similarity,
)
from filing_rag.chunking.tokenize import WhitespaceTokenCounter
from filing_rag.ingest.parse import ParsedFiling, Section

TEXT = (
    "Alpha topic stays here. Alpha keeps going. Alpha still the same. "
    "Zeta switches topics. Zeta remains aside."
)


class ScriptedEmbedder:
    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self.vectors = [list(vector) for vector in vectors]

    def embed(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == len(self.vectors), (texts, len(self.vectors))
        return [list(vector) for vector in self.vectors]


def _config(*, max_tokens: int = 512, size: int = 400, percentile: int = 95) -> ChunkingConfig:
    overlap = 80 if size > 80 else max(0, size - 1)
    return ChunkingConfig(
        tokenizer="x",
        max_tokens=max_tokens,
        fixed=FixedConfig(size=size, overlap=overlap),
        structural=StructuralConfig(max_header_chars=120),
        semantic=SemanticConfig(breakpoint_percentile=percentile, encoder="x"),
    )


def _filing(text: str) -> ParsedFiling:
    return ParsedFiling(
        ticker="MSFT",
        cik="0000789019",
        accession="0000950170-24-087843",
        form="10-K",
        filing_date="2024-07-30",
        period_of_report="2024-06-30",
        fiscal_year=2024,
        primary_doc="msft-20240630.htm",
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


def test_sentence_spans_split_on_period() -> None:
    spans = sentence_spans(TEXT)
    sentences = [TEXT[start:end].strip() for start, end in spans]
    assert sentences == [
        "Alpha topic stays here.",
        "Alpha keeps going.",
        "Alpha still the same.",
        "Zeta switches topics.",
        "Zeta remains aside.",
    ]


def test_similarity_break_at_known_sentence() -> None:
    embedder = ScriptedEmbedder(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.01, 0.99],
        ]
    )
    bounds = split_on_similarity(TEXT, embedder=embedder, breakpoint_percentile=95)
    groups = [TEXT[start:end].strip() for start, end in bounds]
    assert len(groups) == 2
    assert groups[0].startswith("Alpha topic stays here.")
    assert "Zeta switches topics." in groups[1]
    assert "Alpha still the same." in groups[0]
    assert "Alpha still the same." not in groups[1]


def test_chunk_section_uses_breakpoint_and_offsets() -> None:
    filing = _filing(TEXT)
    embedder = ScriptedEmbedder(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.01, 0.99],
        ]
    )
    chunks = chunk_section(
        filing.sections[0],
        filing,
        config=_config(),
        counter=WhitespaceTokenCounter(),
        embedder=embedder,
    )
    assert [chunk.strategy for chunk in chunks] == [STRATEGY, STRATEGY]
    assert chunks[0].chunk_index == 0
    for chunk in chunks:
        assert TEXT[chunk.char_start : chunk.char_end] == chunk.text


def test_oversize_semantic_group_is_capped() -> None:
    words = " ".join(f"Word{i:03d}." for i in range(12))
    filing = _filing(words)
    count = len(sentence_spans(words))
    embedder = ScriptedEmbedder([[1.0, 0.0]] * count)
    chunks = chunk_section(
        filing.sections[0],
        filing,
        config=_config(max_tokens=4, size=4, percentile=95),
        counter=WhitespaceTokenCounter(),
        embedder=embedder,
    )
    assert chunks
    assert all(chunk.token_count <= 4 for chunk in chunks)
    assert all(chunk.strategy == "semantic" for chunk in chunks)


def test_chunk_filing_passes_embedder() -> None:
    filing = _filing("Only one sentence remains.")
    result = chunk_filing(
        filing,
        config=_config(),
        counter=WhitespaceTokenCounter(),
        embedder=ScriptedEmbedder([[1.0]]),
    )
    assert result.strategy == "semantic"
    assert len(result.chunks) == 1
    assert result.chunks[0].text.strip() == "Only one sentence remains."


def test_bge_embedder_requires_extra() -> None:
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is not None:
        pytest.skip("sentence-transformers is installed")
    embedder = BgeSentenceEmbedder("BAAI/bge-base-en-v1.5")
    with pytest.raises(SemanticChunkError, match="uv sync --dev"):
        embedder.ensure_available()
