"""On-disk chunk records. Offsets are into the parent section's text."""

from pydantic import BaseModel, Field

from filing_rag.chunking.cap import TokenSpan
from filing_rag.ingest.parse import ParsedFiling, Section


class Chunk(BaseModel):
    ticker: str
    cik: str
    accession: str
    fiscal_year: int
    edgar_url: str
    item_code: str
    item_title: str
    strategy: str
    chunk_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    token_count: int = Field(ge=0)
    text: str

    @classmethod
    def from_span(
        cls,
        span: TokenSpan,
        filing: ParsedFiling,
        section: Section,
        *,
        strategy: str,
        chunk_index: int,
    ) -> "Chunk":
        return cls(
            ticker=filing.ticker,
            cik=filing.cik,
            accession=filing.accession,
            fiscal_year=filing.fiscal_year,
            edgar_url=filing.edgar_url,
            item_code=section.item_code,
            item_title=section.item_title,
            strategy=strategy,
            chunk_index=chunk_index,
            char_start=span.char_start,
            char_end=span.char_end,
            token_count=span.token_count,
            text=span.text,
        )


class ChunkedFiling(BaseModel):
    ticker: str
    cik: str
    accession: str
    fiscal_year: int
    edgar_url: str
    strategy: str
    chunks: list[Chunk]

    @classmethod
    def from_parsed(
        cls,
        parsed: ParsedFiling,
        strategy: str,
        chunks: list[Chunk],
    ) -> "ChunkedFiling":
        return cls(
            ticker=parsed.ticker,
            cik=parsed.cik,
            accession=parsed.accession,
            fiscal_year=parsed.fiscal_year,
            edgar_url=parsed.edgar_url,
            strategy=strategy,
            chunks=chunks,
        )
