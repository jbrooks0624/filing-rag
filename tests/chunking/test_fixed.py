"""Fixed-token strategy: 400/80 window, section-relative offsets."""

from filing_rag.chunking.config import (
    ChunkingConfig,
    FixedConfig,
    SemanticConfig,
    StructuralConfig,
    load_chunking,
)
from filing_rag.chunking.fixed import STRATEGY, chunk_filing, chunk_section
from filing_rag.chunking.tokenize import WhitespaceTokenCounter
from filing_rag.ingest.parse import ParsedFiling, Section

ACCESSION = "0000950170-24-087843"


def _config(*, size: int = 400, overlap: int = 80, max_tokens: int = 512) -> ChunkingConfig:
    return ChunkingConfig(
        tokenizer="x",
        max_tokens=max_tokens,
        fixed=FixedConfig(size=size, overlap=overlap),
        structural=StructuralConfig(max_header_chars=120),
        semantic=SemanticConfig(breakpoint_percentile=95, encoder="x"),
    )


def _section(text: str, item_code: str = "1A", item_title: str = "Risk Factors") -> Section:
    return Section(
        item_code=item_code,
        item_title=item_title,
        text=text,
        char_start=0,
        char_end=len(text),
    )


def _filing(*texts: str) -> ParsedFiling:
    items = [("1A", "Risk Factors"), ("7", "MD&A"), ("7A", "Market Risk")]
    sections = [
        _section(text, item_code=code, item_title=title)
        for (code, title), text in zip(items, texts, strict=False)
    ]
    return ParsedFiling(
        ticker="MSFT",
        cik="0000789019",
        company_name="MICROSOFT CORP",
        accession=ACCESSION,
        form="10-K",
        filing_date="2024-07-30",
        period_of_report="2024-06-30",
        fiscal_year=2024,
        primary_doc="msft-20240630.htm",
        edgar_url="https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
        sections=sections,
    )


def _words(n: int) -> str:
    return " ".join(f"w{i:03d}" for i in range(n))


def test_empty_section_yields_no_chunks() -> None:
    filing = _filing("   ")
    chunks = chunk_section(
        filing.sections[0], filing, config=_config(), counter=WhitespaceTokenCounter()
    )
    assert chunks == []


def test_short_section_is_one_chunk_with_offsets() -> None:
    text = "Cybersecurity could harm results."
    filing = _filing(text)
    chunks = chunk_section(
        filing.sections[0], filing, config=_config(), counter=WhitespaceTokenCounter()
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.strategy == STRATEGY
    assert chunk.chunk_index == 0
    assert chunk.ticker == "MSFT"
    assert chunk.fiscal_year == 2024
    assert chunk.item_code == "1A"
    assert chunk.char_start == 0
    assert chunk.char_end == len(text)
    assert chunk.text == text
    assert text[chunk.char_start : chunk.char_end] == chunk.text
    assert chunk.token_count == 4


def test_overlap_windows_and_reconstructed_offsets() -> None:
    text = _words(5)
    filing = _filing(text)
    chunks = chunk_section(
        filing.sections[0],
        filing,
        config=_config(size=3, overlap=1, max_tokens=8),
        counter=WhitespaceTokenCounter(),
    )
    assert [chunk.token_count for chunk in chunks] == [3, 3]
    assert chunks[0].text.split() == ["w000", "w001", "w002"]
    assert chunks[1].text.split() == ["w002", "w003", "w004"]
    assert chunks[1].chunk_index == 1
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.token_count <= 8


def test_chunk_index_is_sequential_across_sections() -> None:
    filing = _filing(_words(5), _words(5), "short")
    result = chunk_filing(
        filing,
        config=_config(size=3, overlap=1, max_tokens=8),
        counter=WhitespaceTokenCounter(),
    )
    assert result.strategy == "fixed"
    assert [chunk.item_code for chunk in result.chunks] == ["1A", "1A", "7", "7", "7A"]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1, 2, 3, 4]


def test_long_section_never_exceeds_max_tokens() -> None:
    filing = _filing(_words(900))
    result = chunk_filing(filing, config=load_chunking(), counter=WhitespaceTokenCounter())
    assert result.chunks
    assert all(chunk.token_count <= 512 for chunk in result.chunks)
    assert all(chunk.token_count <= 400 for chunk in result.chunks)
    assert result.chunks[0].token_count == 400
