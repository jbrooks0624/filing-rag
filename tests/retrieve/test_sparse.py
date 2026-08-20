"""Real BM25 via bm25s. Tests write only under tmp_path."""

from pathlib import Path

import pytest
from filing_rag.embed.store import SearchRow
from filing_rag.retrieve.sparse import SparseError, SparseIndex, search_sparse
from filing_rag.retrieve.types import Filters


def _row(
    chunk_id: int,
    text: str,
    *,
    ticker: str = "MSFT",
    accession: str = "0000950170-24-087843",
    item_code: str = "1A",
    edgar_url: str = "https://example.com",
) -> SearchRow:
    return SearchRow(
        id=chunk_id,
        text=text,
        ticker=ticker,
        fiscal_year=2024,
        item_code=item_code,
        accession=accession,
        char_start=0,
        char_end=len(text),
        edgar_url=edgar_url,
        strategy="fixed",
        chunk_index=chunk_id - 1,
    )


def _index() -> SparseIndex:
    return SparseIndex.build(
        [
            _row(1, "The company sells software licenses and cloud services."),
            _row(
                2,
                "Cybersecurity cybersecurity incidents at the bank were material.",
                ticker="JPM",
                accession="0000019617-24-000001",
                edgar_url="https://example.com/jpm",
            ),
            _row(3, "Liquidity and capital resources remained stable."),
        ],
        k1=1.5,
        b=0.75,
    )


def test_search_ranks_lexical_match() -> None:
    hits = search_sparse(_index(), "cybersecurity", k=3)
    assert hits[0].ticker == "JPM"
    assert hits[0].rank == 1
    assert hits[0].score > hits[1].score
    assert hits[0].edgar_url == "https://example.com/jpm"
    assert hits[0].item_code == "1A"


def test_filters_apply_before_k() -> None:
    index = _index()
    unrestricted = index.search("cybersecurity", k=1)
    assert [hit.ticker for hit in unrestricted] == ["JPM"]
    filtered = index.search("cybersecurity", k=1, filters=Filters(tickers=("MSFT",)))
    assert [hit.ticker for hit in filtered] == ["MSFT"]
    assert filtered[0].text.startswith("The company sells")
    assert index.search("cybersecurity", k=5, filters=Filters(tickers=())) == []


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "indexes" / "fixed"
    original = _index()
    original.save(path)
    loaded = SparseIndex.load(path)
    assert loaded.k1 == 1.5
    assert loaded.b == 0.75
    assert [row.id for row in loaded.rows] == [row.id for row in original.rows]
    first = original.search("cybersecurity", k=1)[0]
    second = loaded.search("cybersecurity", k=1)[0]
    assert first.chunk_id == second.chunk_id
    assert first.score == pytest.approx(second.score)


def test_load_or_build_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "indexes" / "fixed"
    rows = [_row(1, "Cybersecurity risk.")]
    first = SparseIndex.load_or_build(path, rows, k1=1.5, b=0.75)
    changed = [_row(1, "CHANGED interest rates.")]
    second = SparseIndex.load_or_build(path, changed, k1=1.5, b=0.75)
    assert second.rows[0].text == first.rows[0].text
    forced = SparseIndex.load_or_build(path, changed, k1=1.5, b=0.75, force=True)
    assert forced.rows[0].text == "CHANGED interest rates."


def test_empty_index_and_missing_meta(tmp_path: Path) -> None:
    empty = SparseIndex.build([], k1=1.5, b=0.75)
    assert empty.search("cybersecurity", k=1) == []
    path = tmp_path / "indexes" / "fixed"
    empty.save(path)
    loaded = SparseIndex.load(path)
    assert loaded.rows == ()
    with pytest.raises(SparseError, match="metadata missing"):
        SparseIndex.load(tmp_path / "missing")


def test_search_rejects_empty_query_and_k() -> None:
    index = _index()
    with pytest.raises(ValueError, match="non-empty"):
        index.search("  ", k=1)
    with pytest.raises(ValueError, match="k must be"):
        index.search("cybersecurity", k=0)
