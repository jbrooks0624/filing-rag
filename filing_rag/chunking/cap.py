"""Hard-cap chunks at max_tokens using a sliding token window."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from filing_rag.chunking.config import ChunkingConfig
from filing_rag.chunking.tokenize import TokenCounter


@dataclass(frozen=True)
class TokenSpan:
    text: str
    char_start: int
    char_end: int
    token_count: int


def window_split(
    text: str,
    *,
    size: int,
    overlap: int,
    counter: TokenCounter,
    char_offset: int = 0,
) -> list[TokenSpan]:
    """Sliding token windows. Short text is a single span covering the whole string."""
    if size <= 0:
        raise ValueError("window size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")
    token_spans = list(counter.offsets(text))
    if not text or not token_spans:
        return []
    n = len(token_spans)
    if n <= size:
        return [
            _clip(
                text,
                local_start=0,
                local_end=len(text),
                char_offset=char_offset,
                max_tokens=size,
                counter=counter,
            )
        ]
    step = size - overlap
    pieces: list[TokenSpan] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        local_start = 0 if start == 0 else token_spans[start][0]
        local_end = len(text) if end == n else token_spans[end - 1][1]
        pieces.append(
            _clip(
                text,
                local_start=local_start,
                local_end=local_end,
                char_offset=char_offset,
                max_tokens=size,
                counter=counter,
            )
        )
        if end == n:
            break
        start += step
    return pieces


def enforce_cap(
    text: str,
    *,
    max_tokens: int,
    size: int,
    overlap: int,
    counter: TokenCounter,
    char_offset: int = 0,
) -> list[TokenSpan]:
    """Keep text as one span if it fits; otherwise window-split and clip to max_tokens."""
    if not text.strip():
        return []
    count = counter.count(text)
    if count <= max_tokens:
        return [
            TokenSpan(
                text=text,
                char_start=char_offset,
                char_end=char_offset + len(text),
                token_count=count,
            )
        ]
    pieces = window_split(
        text,
        size=size,
        overlap=overlap,
        counter=counter,
        char_offset=char_offset,
    )
    return [ensure_max(piece, max_tokens, counter) for piece in pieces]


def cap_spans(
    spans: Sequence[TokenSpan],
    *,
    max_tokens: int,
    size: int,
    overlap: int,
    counter: TokenCounter,
) -> list[TokenSpan]:
    """Apply `enforce_cap` to each candidate, preserving section-relative offsets."""
    capped: list[TokenSpan] = []
    for span in spans:
        capped.extend(
            enforce_cap(
                span.text,
                max_tokens=max_tokens,
                size=size,
                overlap=overlap,
                counter=counter,
                char_offset=span.char_start,
            )
        )
    return capped


def cap_from_config(
    text: str,
    config: ChunkingConfig,
    counter: TokenCounter,
    *,
    char_offset: int = 0,
) -> list[TokenSpan]:
    return enforce_cap(
        text,
        max_tokens=config.max_tokens,
        size=config.fixed.size,
        overlap=config.fixed.overlap,
        counter=counter,
        char_offset=char_offset,
    )


def _clip(
    text: str,
    *,
    local_start: int,
    local_end: int,
    char_offset: int,
    max_tokens: int,
    counter: TokenCounter,
) -> TokenSpan:
    piece = text[local_start:local_end]
    count = counter.count(piece)
    if count > max_tokens:
        piece = counter.truncate(piece, max_tokens)
        count = counter.count(piece)
    return TokenSpan(
        text=piece,
        char_start=char_offset + local_start,
        char_end=char_offset + local_start + len(piece),
        token_count=count,
    )


def ensure_max(span: TokenSpan, max_tokens: int, counter: TokenCounter) -> TokenSpan:
    if span.token_count <= max_tokens:
        return span
    truncated = counter.truncate(span.text, max_tokens)
    return TokenSpan(
        text=truncated,
        char_start=span.char_start,
        char_end=span.char_start + len(truncated),
        token_count=counter.count(truncated),
    )
