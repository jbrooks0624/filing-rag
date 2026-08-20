"""Retrieval YAML and settings paths."""

import pytest
from filing_rag.retrieve.config import RetrievalConfig, load_retrieval
from pydantic import ValidationError


def test_retrieval_defaults() -> None:
    config = load_retrieval()
    assert config.k == 5
    assert config.candidate_k == 50
    assert config.rrf_k == 60
    assert config.query_prefix == "Represent this sentence for searching relevant passages: "
    assert config.bm25.k1 == 1.5
    assert config.bm25.b == 0.75
    assert config.rerank.model == "BAAI/bge-reranker-base"


def test_candidate_k_must_cover_k() -> None:
    with pytest.raises(ValidationError, match="candidate_k"):
        RetrievalConfig.model_validate(
            {
                "k": 10,
                "candidate_k": 5,
                "rrf_k": 60,
                "query_prefix": "prefix: ",
                "bm25": {"k1": 1.5, "b": 0.75},
                "rerank": {"model": "x"},
            }
        )


def test_bm25_b_must_be_unit_interval() -> None:
    with pytest.raises(ValidationError, match="b"):
        RetrievalConfig.model_validate(
            {
                "k": 5,
                "candidate_k": 50,
                "rrf_k": 60,
                "query_prefix": "prefix: ",
                "bm25": {"k1": 1.5, "b": 1.5},
                "rerank": {"model": "x"},
            }
        )


def test_query_prefix_must_be_nonempty() -> None:
    with pytest.raises(ValidationError, match="query_prefix"):
        RetrievalConfig.model_validate(
            {
                "k": 5,
                "candidate_k": 50,
                "rrf_k": 60,
                "query_prefix": "",
                "bm25": {"k1": 1.5, "b": 0.75},
                "rerank": {"model": "x"},
            }
        )