"""RRF fuses ranked Hit lists. No Postgres, no torch, no bm25s."""

import pytest
from filing_rag.retrieve.config import load_retrieval
from filing_rag.retrieve.hybrid import DEFAULT_RRF_K, rrf
from filing_rag.retrieve.types import Hit


def _hit(chunk_id: int, rank: int, *, text: str = "", score: float = 0.0) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        text=text or f"chunk-{chunk_id}",
        ticker="MSFT",
        fiscal_year=2024,
        item_code="1A",
        accession="0000950170-24-087843",
        char_start=0,
        char_end=1,
        edgar_url="https://example.com",
        strategy="fixed",
        chunk_index=chunk_id,
    )


def test_rrf_k_matches_config() -> None:
    assert DEFAULT_RRF_K == 60
    assert load_retrieval().rrf_k == DEFAULT_RRF_K


def test_rrf_adds_reciprocal_ranks_and_unions_ids() -> None:
    dense = [_hit(1, 1, text="dense-1"), _hit(2, 2, text="dense-2")]
    sparse = [_hit(2, 1, text="sparse-2"), _hit(3, 2, text="sparse-3")]
    fused = rrf([dense, sparse], rrf_k=60)
    assert [hit.chunk_id for hit in fused] == [2, 1, 3]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
    assert [hit.rank for hit in fused] == [1, 2, 3]


def test_rrf_keeps_payload_from_first_ranking() -> None:
    dense = [_hit(2, 2, text="from-dense")]
    sparse = [_hit(2, 1, text="from-sparse")]
    fused = rrf([dense, sparse])
    assert fused[0].text == "from-dense"


def test_rrf_counts_duplicate_in_one_ranking_once() -> None:
    dense = [_hit(1, 1), _hit(1, 2)]
    fused = rrf([dense], rrf_k=60)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_cuts_to_k() -> None:
    dense = [_hit(1, 1), _hit(2, 2), _hit(3, 3)]
    fused = rrf([dense, []], k=2)
    assert [hit.chunk_id for hit in fused] == [1, 2]


def test_rrf_empty_rankings() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_rrf_rejects_bad_constants() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        rrf([[_hit(1, 1)]], rrf_k=0)
    with pytest.raises(ValueError, match="k must be"):
        rrf([[_hit(1, 1)]], k=0)
