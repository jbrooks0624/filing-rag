"""Hit / Filters / RetrieveResult are frozen citation payloads."""

from dataclasses import FrozenInstanceError, replace

import pytest
from filing_rag.retrieve import (
    MODES,
    SNIPPET_LIMIT,
    Filters,
    Hit,
    RetrieveResult,
    RetrieveTimings,
    snippet,
)


def _hit() -> Hit:
    return Hit(
        chunk_id=1,
        score=0.9,
        rank=1,
        text="Cybersecurity risk.",
        ticker="MSFT",
        fiscal_year=2024,
        item_code="1A",
        accession="0000950170-24-087843",
        char_start=0,
        char_end=19,
        edgar_url="https://example.com",
        strategy="fixed",
        chunk_index=0,
    )


def test_modes_are_dense_sparse_hybrid() -> None:
    assert MODES == ("dense", "sparse", "hybrid")


def test_unrestricted_filters_match_everything() -> None:
    filters = Filters()
    assert filters.matches(ticker="MSFT", fiscal_year=2024, item_code="1A")
    assert filters.matches(ticker="JPM", fiscal_year=2022, item_code="7")


def test_filters_apply_before_k_semantics() -> None:
    filters = Filters(tickers=("MSFT",), fiscal_years=(2024,), item_codes=("1A", "7"))
    assert filters.matches(ticker="MSFT", fiscal_year=2024, item_code="1A")
    assert filters.matches(ticker="MSFT", fiscal_year=2024, item_code="7")
    assert not filters.matches(ticker="JPM", fiscal_year=2024, item_code="1A")
    assert not filters.matches(ticker="MSFT", fiscal_year=2023, item_code="1A")
    assert not filters.matches(ticker="MSFT", fiscal_year=2024, item_code="7A")


def test_empty_filter_tuple_matches_nothing() -> None:
    filters = Filters(tickers=())
    assert not filters.matches(ticker="MSFT", fiscal_year=2024, item_code="1A")


def test_hit_is_frozen() -> None:
    hit = _hit()
    with pytest.raises(FrozenInstanceError):
        hit.score = 0.1  # type: ignore[misc]


def test_hit_replace_updates_score_and_rank() -> None:
    hit = replace(_hit(), score=0.2, rank=3)
    assert hit.score == 0.2
    assert hit.rank == 3
    assert hit.ticker == "MSFT"
    assert hit.edgar_url == "https://example.com"


def test_snippet_collapses_whitespace_and_caps_length() -> None:
    assert snippet("  hello   world  ") == "hello world"
    assert snippet("a" * SNIPPET_LIMIT) == "a" * SNIPPET_LIMIT
    trimmed = snippet("a" * (SNIPPET_LIMIT + 1))
    assert len(trimmed) == SNIPPET_LIMIT
    assert trimmed.endswith("...")
    assert trimmed[: SNIPPET_LIMIT - 3] == "a" * (SNIPPET_LIMIT - 3)


def test_retrieve_result_defaults() -> None:
    result = RetrieveResult()
    assert result.hits == ()
    assert result.mode == "hybrid"
    assert result.strategy == "fixed"
    assert result.reranked is False
    assert result.timings.total_ms == 0.0


def test_retrieve_result_carries_hits_and_timings() -> None:
    hit = _hit()
    timings = RetrieveTimings(encode_ms=12.0, dense_ms=4.0, sparse_ms=3.0, fuse_ms=1.0)
    result = RetrieveResult(
        hits=(hit,),
        mode="hybrid",
        strategy="structural",
        reranked=True,
        timings=timings,
    )
    assert result.hits[0].item_code == "1A"
    assert result.reranked is True
    assert result.timings.total_ms == 20.0
    assert result.timings.rerank_ms == 0.0
