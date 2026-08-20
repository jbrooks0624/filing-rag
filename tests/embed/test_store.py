"""ChunkStore contract via MemoryChunkStore. No Postgres, no torch."""

import pytest
from filing_rag.chunking.config import STRATEGIES
from filing_rag.chunking.types import Chunk, ChunkedFiling
from filing_rag.embed.store import (
    EMBEDDING_DIM,
    HNSW_INDEXES,
    MemoryChunkStore,
    PostgresChunkStore,
    StoreError,
    dense_search_sql,
    hnsw_statement,
    validate_embeddings,
)
from filing_rag.ingest.parse import ParsedFiling, Section
from filing_rag.settings import Settings

ACCESSION = "0000950170-24-087843"


def _unit(seed: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[seed % EMBEDDING_DIM] = 1.0
    return vector


def _filing(
    *,
    ticker: str = "MSFT",
    cik: str = "0000789019",
    accession: str = ACCESSION,
    fiscal_year: int = 2024,
    edgar_url: str = "https://example.com",
    item_code: str = "1A",
) -> ParsedFiling:
    return ParsedFiling(
        ticker=ticker,
        cik=cik,
        accession=accession,
        form="10-K",
        filing_date="2024-07-30",
        period_of_report="2024-06-30",
        fiscal_year=fiscal_year,
        primary_doc="msft-20240630.htm",
        edgar_url=edgar_url,
        sections=[
            Section(
                item_code=item_code,
                item_title="Risk Factors",
                text="Cybersecurity risk.",
                char_start=0,
                char_end=19,
            )
        ],
    )


def _chunked(
    *texts: str,
    filing: ParsedFiling | None = None,
    strategy: str = "fixed",
    item_code: str = "1A",
) -> ChunkedFiling:
    parsed = filing or _filing()
    chunks = [
        Chunk(
            ticker=parsed.ticker,
            cik=parsed.cik,
            accession=parsed.accession,
            fiscal_year=parsed.fiscal_year,
            edgar_url=parsed.edgar_url,
            item_code=item_code,
            item_title="Risk Factors",
            strategy=strategy,
            chunk_index=index,
            char_start=0,
            char_end=len(text),
            token_count=len(text.split()),
            text=text,
        )
        for index, text in enumerate(texts)
    ]
    return ChunkedFiling.from_parsed(parsed, strategy, chunks)


def test_hnsw_statements_are_partial_per_strategy() -> None:
    assert tuple(HNSW_INDEXES) == STRATEGIES
    for strategy in STRATEGIES:
        sql = hnsw_statement(strategy)
        assert "USING hnsw" in sql
        assert "vector_cosine_ops" in sql
        assert f"WHERE strategy = '{strategy}'" in sql
        assert "IF NOT EXISTS" in sql
        assert HNSW_INDEXES[strategy] in sql


def test_dense_search_sql_casts_query_to_vector() -> None:
    sql = dense_search_sql()
    assert sql.count("%s::vector") == 2
    assert "vector_cosine_ops" not in sql
    assert "ORDER BY c.embedding <=> %s::vector" in sql


def test_postgres_store_from_settings_does_not_connect() -> None:
    settings = Settings(database_url="postgresql://filing:filing@localhost:5432/filing_rag")
    store = PostgresChunkStore.from_settings(settings)
    assert store._conn is None
    assert store._vector_registered is False
    assert store.database_url == settings.database_url


def test_upsert_then_unembedded_then_write() -> None:
    store = MemoryChunkStore()
    store.upsert_filing(_filing())
    ids = store.upsert_chunks(_chunked("alpha", "beta"))
    assert store.count("fixed", ACCESSION) == 2
    pending = store.unembedded("fixed", ACCESSION)
    assert [row.chunk_index for row in pending] == [0, 1]
    store.write_embeddings(ids, [_unit(0), _unit(1)])
    assert store.unembedded("fixed", ACCESSION) == []
    assert store.chunks[ids[0]].embedding == _unit(0)


def test_same_text_upsert_keeps_embedding() -> None:
    store = MemoryChunkStore()
    store.upsert_filing(_filing())
    ids = store.upsert_chunks(_chunked("alpha"))
    store.write_embeddings(ids, [_unit(0)])
    store.upsert_chunks(_chunked("alpha"))
    assert store.unembedded("fixed", ACCESSION) == []
    assert store.chunks[ids[0]].embedding == _unit(0)


def test_changed_text_upsert_clears_embedding() -> None:
    store = MemoryChunkStore()
    store.upsert_filing(_filing())
    ids = store.upsert_chunks(_chunked("alpha"))
    store.write_embeddings(ids, [_unit(0)])
    store.upsert_chunks(_chunked("omega"))
    pending = store.unembedded("fixed", ACCESSION)
    assert len(pending) == 1
    assert pending[0].text == "omega"
    assert store.chunks[ids[0]].embedding is None


def test_clear_embeddings_for_force() -> None:
    store = MemoryChunkStore()
    store.upsert_filing(_filing())
    ids = store.upsert_chunks(_chunked("alpha"))
    store.write_embeddings(ids, [_unit(0)])
    assert store.clear_embeddings("fixed", ACCESSION) == 1
    assert len(store.unembedded("fixed", ACCESSION)) == 1


def test_upsert_chunks_requires_filing() -> None:
    store = MemoryChunkStore()
    with pytest.raises(StoreError, match="filing not upserted"):
        store.upsert_chunks(_chunked("alpha"))


def test_write_embeddings_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError, match="768-d"):
        validate_embeddings([1], [[0.0, 1.0]])


def test_ensure_hnsw_records_stats() -> None:
    store = MemoryChunkStore()
    stats = store.ensure_hnsw()
    assert store.hnsw_built
    assert [item.strategy for item in stats] == list(STRATEGIES)
    assert all(item.build_ms == 0.0 for item in stats)


def test_search_dense_ranks_by_cosine_and_skips_nulls() -> None:
    store = MemoryChunkStore()
    store.upsert_filing(_filing())
    ids = store.upsert_chunks(_chunked("match", "other", "pending"))
    store.write_embeddings(ids[:2], [_unit(0), _unit(1)])
    hits = store.search_dense(_unit(0), "fixed", 5)
    assert [row.text for row in hits] == ["match", "other"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.0)
    assert hits[0].edgar_url == "https://example.com"
    assert all(row.score is not None for row in hits)


def test_search_dense_filters_before_k() -> None:
    store = MemoryChunkStore()
    msft = _filing()
    jpm = _filing(
        ticker="JPM",
        cik="0000019617",
        accession="0000019617-24-000001",
        edgar_url="https://example.com/jpm",
    )
    store.upsert_filing(msft)
    store.upsert_filing(jpm)
    msft_ids = store.upsert_chunks(_chunked("msft", filing=msft))
    jpm_ids = store.upsert_chunks(_chunked("jpm", filing=jpm))
    store.write_embeddings(msft_ids, [_unit(1)])
    store.write_embeddings(jpm_ids, [_unit(0)])
    unrestricted = store.search_dense(_unit(0), "fixed", 1)
    assert [row.ticker for row in unrestricted] == ["JPM"]
    filtered = store.search_dense(_unit(0), "fixed", 1, tickers=("MSFT",))
    assert [row.ticker for row in filtered] == ["MSFT"]
    assert filtered[0].edgar_url == "https://example.com"
    assert store.search_dense(_unit(0), "fixed", 5, tickers=()) == []


def test_iter_chunks_includes_unembedded() -> None:
    store = MemoryChunkStore()
    store.upsert_filing(_filing())
    ids = store.upsert_chunks(_chunked("alpha", "beta"))
    store.write_embeddings(ids[:1], [_unit(0)])
    rows = list(store.iter_chunks("fixed"))
    assert [row.text for row in rows] == ["alpha", "beta"]
    assert rows[0].score is None
    assert rows[0].edgar_url == "https://example.com"
    assert list(store.iter_chunks("structural")) == []


def test_search_dense_rejects_bad_k_and_dim() -> None:
    store = MemoryChunkStore()
    with pytest.raises(ValueError, match="k must be"):
        store.search_dense(_unit(0), "fixed", 0)
    with pytest.raises(ValueError, match="768-d"):
        store.search_dense([0.0, 1.0], "fixed", 1)
    with pytest.raises(ValueError, match="unknown strategy"):
        store.search_dense(_unit(0), "mystery", 1)
