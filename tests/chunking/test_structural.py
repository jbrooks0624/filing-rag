"""Structural strategy: subsection headers, then the 512-token cap."""

from filing_rag.chunking.config import (
    ChunkingConfig,
    FixedConfig,
    SemanticConfig,
    StructuralConfig,
)
from filing_rag.chunking.structural import (
    STRATEGY,
    chunk_filing,
    chunk_section,
    looks_like_header,
    split_on_headers,
)
from filing_rag.chunking.tokenize import WhitespaceTokenCounter
from filing_rag.ingest.parse import ParsedFiling, Section

ACCESSION = "0000950170-24-087843"

HEADERED = """\
Item 1A. Risk Factors
Investors should consider the following risks carefully before investing.

Cybersecurity Risk
We face cybersecurity threats that could disrupt operations and harm results of operations.

Revenue | 2024 | 2023
Cyber | High | Medium

COMPETITION
The markets in which we compete are highly competitive and change rapidly.
"""


def _config(*, size: int = 400, overlap: int = 80, max_tokens: int = 512) -> ChunkingConfig:
    return ChunkingConfig(
        tokenizer="x",
        max_tokens=max_tokens,
        fixed=FixedConfig(size=size, overlap=overlap),
        structural=StructuralConfig(max_header_chars=120),
        semantic=SemanticConfig(breakpoint_percentile=95, encoder="x"),
    )


def _filing(text: str) -> ParsedFiling:
    return ParsedFiling(
        ticker="MSFT",
        cik="0000789019",
        accession=ACCESSION,
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


def test_table_row_is_not_a_header() -> None:
    assert not looks_like_header("Cyber | High | Medium", max_header_chars=120)
    assert looks_like_header("Cybersecurity Risk", max_header_chars=120)
    assert looks_like_header("COMPETITION", max_header_chars=120)


def test_headers_become_boundaries() -> None:
    spans = split_on_headers(HEADERED, max_header_chars=120)
    texts = [HEADERED[start:end] for start, end in spans]
    joined = " ".join(texts)
    assert "Cybersecurity Risk" in joined
    assert "COMPETITION" in joined
    assert any(text.lstrip().startswith("Cybersecurity Risk") for text in texts)
    assert any(text.lstrip().startswith("COMPETITION") for text in texts)
    assert not any("Cyber | High" in text and "COMPETITION" in text for text in texts)


def test_headerless_section_is_one_chunk_then_capped() -> None:
    text = " ".join(f"w{i:03d}" for i in range(20))
    filing = _filing(text)
    chunks = chunk_section(
        filing.sections[0], filing, config=_config(), counter=WhitespaceTokenCounter()
    )
    assert len(chunks) == 1
    assert chunks[0].strategy == STRATEGY
    assert chunks[0].text == text
    assert text[chunks[0].char_start : chunks[0].char_end] == chunks[0].text


def test_headerless_oversize_section_is_window_split() -> None:
    text = " ".join(f"w{i:03d}" for i in range(20))
    filing = _filing(text)
    chunks = chunk_section(
        filing.sections[0],
        filing,
        config=_config(size=8, overlap=2, max_tokens=8),
        counter=WhitespaceTokenCounter(),
    )
    assert len(chunks) > 1
    assert all(chunk.token_count <= 8 for chunk in chunks)
    assert all(chunk.strategy == "structural" for chunk in chunks)


def test_chunk_filing_labels_structural() -> None:
    filing = _filing(HEADERED)
    result = chunk_filing(filing, config=_config(), counter=WhitespaceTokenCounter())
    assert result.strategy == "structural"
    assert result.chunks
    assert all(chunk.item_code == "1A" for chunk in result.chunks)
    assert any("Cybersecurity Risk" in chunk.text for chunk in result.chunks)
    assert any("COMPETITION" in chunk.text for chunk in result.chunks)
