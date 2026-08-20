"""Retriever.search composes dense / sparse / hybrid. No Postgres, no torch."""

from pathlib import Path

import pytest
from filing_rag.chunking.types import Chunk, ChunkedFiling
from filing_rag.embed.store import EMBEDDING_DIM, MemoryChunkStore
from filing_rag.ingest.parse import ParsedFiling, Section
from filing_rag.retrieve.pipeline import Retriever
from filing_rag.retrieve.rerank import IdentityReranker
from filing_rag.retrieve.sparse import SparseIndex
from filing_rag.retrieve.types import MODES, Filters, Hit
from filing_rag.settings import PROJECT_ROOT, Settings

ACCESSION = "0000950170-24-087843"


def _unit(seed: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[seed % EMBEDDING_DIM] = 1.0
    return vector


class FakeEncoder:
    def encode(self, query: str) -> list[float]:
        del query
        return _unit(0)


class RecordingReranker:
    def __init__(self) -> None:
        self.called = False
        self.queries: list[str] = []

    def rerank(self, query: str, hits: list[Hit], *, k: int) -> list[Hit]:
        self.called = True
        self.queries.append(query)
        return IdentityReranker().rerank(query, hits, k=k)


def _filing() -> ParsedFiling:
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
                text="Cybersecurity risk.",
                char_start=0,
                char_end=19,
            )
        ],
    )


def _chunk(filing: ParsedFiling, text: str, index: int) -> Chunk:
    return Chunk(
        ticker=filing.ticker,
        cik=filing.cik,
        accession=filing.accession,
        fiscal_year=filing.fiscal_year,
        edgar_url=filing.edgar_url,
        item_code="1A",
        item_title="Risk Factors",
        strategy="fixed",
        chunk_index=index,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
        text=text,
    )


def _populated_store() -> MemoryChunkStore:
    filing = _filing()
    store = MemoryChunkStore()
    store.upsert_filing(filing)
    chunked = ChunkedFiling.from_parsed(
        filing,
        "fixed",
        [
            _chunk(filing, "Cybersecurity risk could disrupt operations.", 0),
            _chunk(filing, "Liquidity and capital resources remained stable.", 1),
        ],
    )
    ids = store.upsert_chunks(chunked)
    store.write_embeddings(ids, [_unit(0), _unit(1)])
    return store


def _retriever(tmp_path: Path) -> tuple[Retriever, RecordingReranker]:
    store = _populated_store()
    sparse = SparseIndex.build(
        list(store.iter_chunks("fixed")),
        k1=1.5,
        b=0.75,
    )
    recorder = RecordingReranker()
    settings = Settings(
        data_dir=tmp_path / "data",
        corpus_path=PROJECT_ROOT / "config" / "corpus.yaml",
        chunking_path=PROJECT_ROOT / "config" / "chunking.yaml",
        retrieval_path=PROJECT_ROOT / "config" / "retrieval.yaml",
    )
    retriever = Retriever.from_config(
        settings=settings,
        store=store,
        encoder=FakeEncoder(),
        reranker=recorder,
        sparse=sparse,
    )
    return retriever, recorder


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rerank", [False, True])
def test_six_configs_return_ranked_hits(tmp_path: Path, mode: str, rerank: bool) -> None:
    retriever, recorder = _retriever(tmp_path)
    result = retriever.search(
        "cybersecurity",
        strategy="fixed",
        mode=mode,
        k=2,
        rerank=rerank,
    )
    assert result.mode == mode
    assert result.strategy == "fixed"
    assert result.reranked is rerank
    assert recorder.called is rerank
    assert 1 <= len(result.hits) <= 2
    assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))
    assert result.hits[0].ticker == "MSFT"
    if mode == "dense":
        assert result.timings.sparse_ms == 0.0
        assert result.timings.fuse_ms == 0.0
    if mode == "sparse":
        assert result.timings.encode_ms == 0.0
        assert result.timings.dense_ms == 0.0
        assert result.timings.fuse_ms == 0.0
    if not rerank:
        assert result.timings.rerank_ms == 0.0


def test_empty_query_and_unknown_mode(tmp_path: Path) -> None:
    retriever, _ = _retriever(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        retriever.search("  ", strategy="fixed")
    with pytest.raises(ValueError, match="unknown mode"):
        retriever.search("cyber", strategy="fixed", mode="keyword")
    with pytest.raises(ValueError, match="unknown strategy"):
        retriever.search("cyber", strategy="recursive")


def test_filters_scope_dense_search(tmp_path: Path) -> None:
    retriever, _ = _retriever(tmp_path)
    none = retriever.search(
        "cybersecurity",
        strategy="fixed",
        mode="dense",
        k=5,
        filters=Filters(tickers=()),
    )
    assert none.hits == ()
    hits = retriever.search(
        "cybersecurity",
        strategy="fixed",
        mode="dense",
        k=5,
        filters=Filters(tickers=("MSFT",)),
    )
    assert hits.hits
    assert {hit.ticker for hit in hits.hits} == {"MSFT"}
    assert hits.hits[0].edgar_url == "https://example.com"


def test_rerank_receives_raw_query(tmp_path: Path) -> None:
    retriever, recorder = _retriever(tmp_path)
    retriever.search("cyber risk", strategy="fixed", mode="dense", k=1, rerank=True)
    assert recorder.queries == ["cyber risk"]
