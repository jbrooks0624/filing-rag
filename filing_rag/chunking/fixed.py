"""Fixed-token chunking: 400-token windows with 80-token overlap."""

from __future__ import annotations

from filing_rag.chunking.cap import ensure_max, window_split
from filing_rag.chunking.config import ChunkingConfig
from filing_rag.chunking.tokenize import TokenCounter
from filing_rag.chunking.types import Chunk, ChunkedFiling
from filing_rag.ingest.parse import ParsedFiling, Section

STRATEGY = "fixed"


def chunk_section(
    section: Section,
    filing: ParsedFiling,
    *,
    config: ChunkingConfig,
    counter: TokenCounter,
    start_index: int = 0,
) -> list[Chunk]:
    """Split one section with the configured token window. Empty text yields no chunks."""
    spans = window_split(
        section.text,
        size=config.fixed.size,
        overlap=config.fixed.overlap,
        counter=counter,
        char_offset=0,
    )
    return [
        Chunk.from_span(
            ensure_max(span, config.max_tokens, counter),
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
) -> ChunkedFiling:
    """Chunk every section. `chunk_index` is sequential across the filing."""
    chunks: list[Chunk] = []
    for section in parsed.sections:
        chunks.extend(
            chunk_section(
                section,
                parsed,
                config=config,
                counter=counter,
                start_index=len(chunks),
            )
        )
    return ChunkedFiling.from_parsed(parsed, STRATEGY, chunks)
