"""Rerank reorders Hits. Tests never load torch."""

import importlib.util

import pytest
from filing_rag.retrieve.config import load_retrieval
from filing_rag.retrieve.rerank import BgeReranker, IdentityReranker, RerankError
from filing_rag.retrieve.types import Hit


def _hit(chunk_id: int, rank: int, *, text: str, score: float = 0.0) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        text=text,
        ticker="MSFT",
        fiscal_year=2024,
        item_code="1A",
        accession="0000950170-24-087843",
        char_start=0,
        char_end=len(text),
        edgar_url="https://example.com",
        strategy="fixed",
        chunk_index=chunk_id,
    )


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = list(pairs)
        return [2.0 if "cyber" in passage else 0.1 for _, passage in pairs]


def test_identity_keeps_order_and_rewrites_ranks() -> None:
    hits = [
        _hit(3, 3, text="third", score=0.2),
        _hit(1, 1, text="first", score=0.9),
        _hit(2, 2, text="second", score=0.5),
    ]
    out = IdentityReranker().rerank("ignored", hits, k=2)
    assert [hit.chunk_id for hit in out] == [3, 1]
    assert [hit.rank for hit in out] == [1, 2]
    assert [hit.score for hit in out] == [0.2, 0.9]


def test_from_config_uses_yaml_model() -> None:
    reranker = BgeReranker.from_config()
    assert reranker.model_name == load_retrieval().rerank.model
    assert reranker.model_name == "BAAI/bge-reranker-base"
    assert reranker._model is None


def test_bge_rerank_scores_pairs_and_reorders() -> None:
    fake = FakeCrossEncoder()
    reranker = BgeReranker("BAAI/bge-reranker-base")
    reranker._model = fake
    hits = [
        _hit(1, 1, text="liquidity resources"),
        _hit(2, 2, text="cybersecurity risk"),
    ]
    out = reranker.rerank("cyber risk", hits, k=2)
    assert fake.pairs == [
        ("cyber risk", "liquidity resources"),
        ("cyber risk", "cybersecurity risk"),
    ]
    assert [hit.chunk_id for hit in out] == [2, 1]
    assert [hit.rank for hit in out] == [1, 2]
    assert out[0].score == 2.0
    assert out[1].score == 0.1
    assert reranker._model is fake


def test_bge_rerank_cuts_to_k() -> None:
    fake = FakeCrossEncoder()
    reranker = BgeReranker("unused")
    reranker._model = fake
    hits = [
        _hit(1, 1, text="other"),
        _hit(2, 2, text="cyber"),
        _hit(3, 3, text="also other"),
    ]
    out = reranker.rerank("q", hits, k=1)
    assert [hit.chunk_id for hit in out] == [2]
    assert out[0].rank == 1


def test_bge_empty_hits_do_not_load_model() -> None:
    reranker = BgeReranker("BAAI/bge-reranker-base")
    assert reranker.rerank("cyber risk", [], k=5) == []
    assert reranker._model is None


def test_bge_rejects_empty_query() -> None:
    reranker = BgeReranker("BAAI/bge-reranker-base")
    with pytest.raises(ValueError, match="non-empty"):
        reranker.rerank("  ", [_hit(1, 1, text="x")], k=1)
    assert reranker._model is None


def test_bge_reranker_requires_extra() -> None:
    if importlib.util.find_spec("sentence_transformers") is not None:
        pytest.skip("sentence-transformers is installed")
    reranker = BgeReranker("BAAI/bge-reranker-base")
    with pytest.raises(RerankError, match="uv sync --dev"):
        reranker.ensure_available()
