"""Dense cosine search via ChunkStore."""

from collections.abc import Sequence

from filing_rag.embed.store import ChunkStore, SearchRow
from filing_rag.retrieve.types import Filters, Hit


def search_dense(
    store: ChunkStore,
    vector: Sequence[float],
    *,
    strategy: str,
    k: int,
    filters: Filters | None = None,
) -> list[Hit]:
    resolved = filters or Filters()
    rows = store.search_dense(
        vector,
        strategy,
        k,
        tickers=resolved.tickers,
        fiscal_years=resolved.fiscal_years,
        item_codes=resolved.item_codes,
    )
    return [hit_from_row(row, rank) for rank, row in enumerate(rows, start=1)]


def hit_from_row(row: SearchRow, rank: int) -> Hit:
    return Hit(
        chunk_id=row.id,
        score=0.0 if row.score is None else row.score,
        rank=rank,
        text=row.text,
        ticker=row.ticker,
        fiscal_year=row.fiscal_year,
        item_code=row.item_code,
        accession=row.accession,
        char_start=row.char_start,
        char_end=row.char_end,
        edgar_url=row.edgar_url,
        strategy=row.strategy,
        chunk_index=row.chunk_index,
    )
