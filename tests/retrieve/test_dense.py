"""Dense search maps store rows to ranked Hits. No Postgres, no torch."""

from filing_rag.chunking.types import Chunk, ChunkedFiling
from filing_rag.embed.store import EMBEDDING_DIM, MemoryChunkStore
from filing_rag.ingest.parse import ParsedFiling, Section
from filing_rag.retrieve.dense import search_dense
from filing_rag.retrieve.types import Filters

ACCESSION = "0000950170-24-087843"


def _unit(seed: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[seed % EMBEDDING_DIM] = 1.0
    return vector


def _filing(*, ticker: str = "MSFT", accession: str = ACCESSION) -> ParsedFiling:
    return ParsedFiling(
        ticker=ticker,
        cik="0000789019",
        accession=accession,
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
                text="Cybersecurity risk.",
                char_start=0,
                char_end=19,
            )
        ],
    )


def _index(store: MemoryChunkStore, filing: ParsedFiling, text: str, vector: list[float]) -> None:
    store.upsert_filing(filing)
    chunked = ChunkedFiling.from_parsed(
        filing,
        "fixed",
        [
            Chunk(
                ticker=filing.ticker,
                cik=filing.cik,
                accession=filing.accession,
                fiscal_year=filing.fiscal_year,
                edgar_url=filing.edgar_url,
                item_code="1A",
                item_title="Risk Factors",
                strategy="fixed",
                chunk_index=0,
                char_start=0,
                char_end=len(text),
                token_count=len(text.split()),
                text=text,
            )
        ],
    )
    ids = store.upsert_chunks(chunked)
    store.write_embeddings(ids, [vector])


def test_search_dense_returns_ranked_hits_with_citations() -> None:
    store = MemoryChunkStore()
    _index(store, _filing(), "cyber risk", _unit(0))
    hits = search_dense(store, _unit(0), strategy="fixed", k=5)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.rank == 1
    assert hit.score == 1.0
    assert hit.ticker == "MSFT"
    assert hit.fiscal_year == 2024
    assert hit.item_code == "1A"
    assert hit.accession == ACCESSION
    assert hit.edgar_url == "https://example.com"
    assert hit.strategy == "fixed"
    assert hit.text == "cyber risk"


def test_search_dense_passes_filters() -> None:
    store = MemoryChunkStore()
    _index(store, _filing(), "msft", _unit(0))
    _index(
        store,
        _filing(ticker="JPM", accession="0000019617-24-000001"),
        "jpm",
        _unit(0),
    )
    hits = search_dense(
        store,
        _unit(0),
        strategy="fixed",
        k=5,
        filters=Filters(tickers=("MSFT",)),
    )
    assert [hit.ticker for hit in hits] == ["MSFT"]
    assert hits[0].rank == 1
