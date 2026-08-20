"""Citation-forcing prompt. Contexts come from hits, never gold quotes."""

import pytest
from filing_rag.generate.config import GenerationConfig
from filing_rag.generate.prompt import (
    CITATION_EXAMPLE,
    CITATION_FORMAT,
    build_messages,
    system_prompt,
)
from filing_rag.generate.types import CitationBlock, blocks_from_hits
from filing_rag.retrieve.types import Hit

ACCESSION = "0000950170-24-087843"
GOLD_QUOTE = (
    "we have experienced cybersecurity incidents in which such actors have gained "
    "unauthorized access"
)


def _hit(
    *,
    chunk_id: int = 1,
    rank: int = 3,
    text: str = "Azure revenue grew 30 percent.",
    ticker: str = "MSFT",
    fiscal_year: int = 2024,
    item_code: str = "1A",
    accession: str = ACCESSION,
    edgar_url: str = "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft.htm",
) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        score=1.0,
        rank=rank,
        text=text,
        ticker=ticker,
        fiscal_year=fiscal_year,
        item_code=item_code,
        accession=accession,
        char_start=0,
        char_end=10,
        edgar_url=edgar_url,
        strategy="fixed",
        chunk_index=0,
    )


def test_system_prompt_includes_refusal_and_citation_format() -> None:
    config = GenerationConfig()
    text = system_prompt(config.refusal_phrase)
    assert "only the numbered contexts" in text
    assert "Do not use outside knowledge" in text
    assert CITATION_FORMAT in text
    assert CITATION_EXAMPLE in text
    assert config.refusal_phrase in text
    assert GOLD_QUOTE not in text


def test_build_messages_numbers_contexts_in_list_order() -> None:
    config = GenerationConfig()
    first = _hit(chunk_id=10, rank=9, text="First chunk.")
    second = _hit(
        chunk_id=11,
        rank=2,
        ticker="JPM",
        fiscal_year=2023,
        item_code="7",
        accession="0000019617-24-000001",
        edgar_url="https://example.com/jpm",
        text="Second chunk.",
    )
    system, user = build_messages("What did they disclose?", [first, second], config)
    assert system["role"] == "system"
    assert user["role"] == "user"
    content = user["content"]
    assert content.startswith("Question:\nWhat did they disclose?\n\nContexts:\n")
    assert "[1] MSFT FY2024 Item 1A accession=" in content
    assert ACCESSION in content
    assert first.edgar_url in content
    assert "First chunk." in content
    assert (
        "[2] JPM FY2023 Item 7 accession=0000019617-24-000001 url=https://example.com/jpm"
        in content
    )
    assert "Second chunk." in content
    assert GOLD_QUOTE not in content
    assert GOLD_QUOTE not in system["content"]


def test_build_messages_empty_hits_uses_none_contexts() -> None:
    config = GenerationConfig()
    system, user = build_messages("What did NVDA disclose?", (), config)
    assert "Contexts:\n(none)" in user["content"]
    assert config.refusal_phrase in system["content"]


def test_build_messages_rejects_empty_query() -> None:
    config = GenerationConfig()
    with pytest.raises(ValueError, match="query must be non-empty"):
        build_messages("  ", [_hit()], config)


def test_blocks_from_hits_are_1_based_and_ignore_rank() -> None:
    hits = [_hit(rank=7, text="a"), _hit(rank=1, text="b")]
    blocks = blocks_from_hits(hits)
    assert [block.index for block in blocks] == [1, 2]
    assert blocks[0].citation_tag == "[MSFT FY2024 Item 1A]"
    assert blocks[0].text == "a"
    assert GOLD_QUOTE not in blocks[0].text


def test_citation_block_header() -> None:
    block = CitationBlock.from_hit(_hit(), index=1)
    assert block.header.startswith("[1] MSFT FY2024 Item 1A accession=")
    assert ACCESSION in block.header
    assert "url=https://www.sec.gov/" in block.header
