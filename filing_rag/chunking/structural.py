"""Structural chunking: split on subsection headers, then cap oversized spans."""

from __future__ import annotations

from collections.abc import Sequence

from filing_rag.chunking.cap import TokenSpan, cap_spans
from filing_rag.chunking.config import ChunkingConfig
from filing_rag.chunking.tokenize import TokenCounter
from filing_rag.chunking.types import Chunk, ChunkedFiling
from filing_rag.ingest.parse import ParsedFiling, Section

STRATEGY = "structural"

SMALL_WORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "with"}
)


def looks_like_header(line: str, *, max_header_chars: int) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > max_header_chars:
        return False
    if _is_table_row(stripped):
        return False
    return _is_all_caps(stripped) or _is_title_case(stripped)


def split_on_headers(text: str, *, max_header_chars: int) -> list[tuple[int, int]]:
    """Return (start, end) spans. No headers → the whole section. Empty text → []."""
    if not text.strip():
        return []
    header_starts = _header_starts(text, max_header_chars=max_header_chars)
    if not header_starts:
        return [(0, len(text))]
    bounds: list[tuple[int, int]] = []
    first = header_starts[0]
    if first > 0 and text[:first].strip():
        bounds.append((0, first))
    for index, start in enumerate(header_starts):
        end = header_starts[index + 1] if index + 1 < len(header_starts) else len(text)
        bounds.append((start, end))
    return bounds


def chunk_section(
    section: Section,
    filing: ParsedFiling,
    *,
    config: ChunkingConfig,
    counter: TokenCounter,
    start_index: int = 0,
) -> list[Chunk]:
    bounds = split_on_headers(section.text, max_header_chars=config.structural.max_header_chars)
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
) -> ChunkedFiling:
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


def _is_table_row(line: str) -> bool:
    return " | " in line or line.count("|") >= 2


def _is_all_caps(line: str) -> bool:
    letters = [char for char in line if char.isalpha()]
    return len(letters) >= 2 and all(char.isupper() for char in letters)


def _is_title_case(line: str) -> bool:
    words = line.split()
    if not words:
        return False
    return all(_word_is_titled(word) for word in words) and any(
        any(char.isupper() for char in word) for word in words
    )


def _word_is_titled(word: str) -> bool:
    cleaned = word.strip(".,:;()[]\"'")
    if not cleaned:
        return True
    if cleaned.lower() in SMALL_WORDS:
        return True
    if cleaned[:1].isdigit():
        return True
    return cleaned[:1].isupper()


def _iter_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    start = 0
    for line in text.splitlines(keepends=True):
        lines.append((start, line))
        start += len(line)
    return lines


def _header_starts(text: str, *, max_header_chars: int) -> list[int]:
    lines = _iter_lines(text)
    starts: list[int] = []
    for index, (offset, line) in enumerate(lines):
        stripped = line.strip()
        if not looks_like_header(stripped, max_header_chars=max_header_chars):
            continue
        prose = _following_prose(lines, index + 1, max_header_chars=max_header_chars)
        if prose is None or len(prose) <= len(stripped):
            continue
        starts.append(offset)
    return starts


def _following_prose(
    lines: Sequence[tuple[int, str]],
    start: int,
    *,
    max_header_chars: int,
) -> str | None:
    for _, line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if looks_like_header(stripped, max_header_chars=max_header_chars):
            continue
        return stripped
    return None
