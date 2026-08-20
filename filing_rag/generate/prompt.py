"""Citation-forcing prompt. Contexts come from Hits, never gold quotes."""

from __future__ import annotations

from collections.abc import Sequence

from filing_rag.generate.config import GenerationConfig
from filing_rag.generate.types import CitationBlock, blocks_from_hits
from filing_rag.retrieve.types import Hit

CITATION_FORMAT = "[TICKER FYyear Item CODE]"
CITATION_EXAMPLE = "[MSFT FY2024 Item 1A]"


def system_prompt(refusal_phrase: str) -> str:
    phrase = refusal_phrase.strip()
    return (
        "You answer questions about SEC 10-K filings using only the numbered contexts.\n"
        "\n"
        "Rules:\n"
        "- Use only information in the contexts. Do not use outside knowledge.\n"
        f"- Cite every claim as {CITATION_FORMAT}, for example {CITATION_EXAMPLE}.\n"
        "- If the contexts do not contain enough information to answer, "
        f"reply with exactly: {phrase}"
    )


def format_contexts(blocks: Sequence[CitationBlock]) -> str:
    if not blocks:
        return "(none)"
    parts: list[str] = []
    for block in blocks:
        parts.append(f"{block.header}\n{block.text}")
    return "\n\n".join(parts)


def user_prompt(query: str, blocks: Sequence[CitationBlock]) -> str:
    return f"Question:\n{query}\n\nContexts:\n{format_contexts(blocks)}"


def build_messages(
    query: str,
    hits: Sequence[Hit],
    config: GenerationConfig,
) -> tuple[dict[str, str], ...]:
    """System + user chat messages. Empty query raises."""
    stripped = query.strip()
    if not stripped:
        raise ValueError("query must be non-empty")
    blocks = blocks_from_hits(hits)
    return (
        {"role": "system", "content": system_prompt(config.refusal_phrase)},
        {"role": "user", "content": user_prompt(stripped, blocks)},
    )
