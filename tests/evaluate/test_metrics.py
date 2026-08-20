"""Section collapse, recall@k, MRR, nDCG. No Postgres, no torch."""

import math

import pytest
from filing_rag.evaluate.metrics import (
    collapse_sections,
    mean_scores,
    mrr,
    ndcg_at,
    recall_at,
    score_question,
)
from filing_rag.evaluate.types import Question
from filing_rag.retrieve.types import Hit

ACCESSION = "0000950170-24-087843"
JPM_ACCESSION = "0000019617-24-000001"
MSFT_1A = (ACCESSION, "1A")
JPM_1A = (JPM_ACCESSION, "1A")
MSFT_7 = (ACCESSION, "7")


def _hit(
    accession: str,
    item_code: str,
    rank: int,
    *,
    chunk_id: int | None = None,
) -> Hit:
    return Hit(
        chunk_id=chunk_id if chunk_id is not None else rank,
        score=1.0 / rank,
        rank=rank,
        text=f"{accession} {item_code}",
        ticker="MSFT" if accession == ACCESSION else "JPM",
        fiscal_year=2024,
        item_code=item_code,
        accession=accession,
        char_start=0,
        char_end=1,
        edgar_url="https://example.com",
        strategy="fixed",
        chunk_index=rank,
    )


def _question(*, answerable: bool = True, citations: list[dict] | None = None) -> Question:
    if not answerable:
        return Question.model_validate(
            {
                "id": "ua-001",
                "type": "unanswerable",
                "query": "What did NVDA disclose about GPU supply?",
                "answerable": False,
                "answer": "",
                "citations": [],
            }
        )
    if citations is None:
        citations = [{"accession": ACCESSION, "item_code": "1A", "quote": "Cyber."}]
    return Question.model_validate(
        {
            "id": "sh-001",
            "type": "single_hop",
            "query": "What cybersecurity risks does Microsoft disclose?",
            "answerable": True,
            "answer": "Cybersecurity risk.",
            "citations": citations,
        }
    )


def test_collapse_keeps_first_section_and_drops_later_chunks() -> None:
    hits = [
        _hit(ACCESSION, "1A", 1, chunk_id=1),
        _hit(ACCESSION, "1A", 2, chunk_id=2),
        _hit(JPM_ACCESSION, "1A", 3, chunk_id=3),
        _hit(ACCESSION, "1A", 4, chunk_id=4),
    ]
    assert collapse_sections(hits) == (MSFT_1A, JPM_1A)


def test_collapse_empty() -> None:
    assert collapse_sections([]) == ()


def test_recall_at_one_vs_five() -> None:
    gold = {MSFT_1A, JPM_1A}
    sections = (MSFT_7, MSFT_1A, JPM_1A)
    assert recall_at(gold, sections, 1) == 0.0
    assert recall_at(gold, sections, 5) == 1.0
    assert recall_at(gold, sections, 10) == 1.0


def test_recall_two_gold_sections() -> None:
    gold = {MSFT_1A, JPM_1A}
    sections = (MSFT_1A, MSFT_7)
    assert recall_at(gold, sections, 5) == 0.5


def test_recall_undefined_for_empty_gold() -> None:
    with pytest.raises(ValueError, match="empty gold"):
        recall_at([], (MSFT_1A,), 5)


def test_mrr_is_reciprocal_of_first_gold() -> None:
    gold = {JPM_1A}
    assert mrr(gold, (MSFT_1A, JPM_1A)) == 0.5
    assert mrr(gold, (JPM_1A, MSFT_1A)) == 1.0
    assert mrr(gold, (MSFT_1A, MSFT_7)) == 0.0


def test_ndcg_perfect_is_one() -> None:
    gold = {MSFT_1A}
    assert ndcg_at(gold, (MSFT_1A,), 10) == pytest.approx(1.0)


def test_ndcg_relevant_at_rank_two() -> None:
    gold = {JPM_1A}
    # DCG = 1 / log2(3), IDCG = 1 / log2(2) = 1
    expected = 1.0 / math.log2(3)
    assert ndcg_at(gold, (MSFT_1A, JPM_1A), 10) == pytest.approx(expected)


def test_duplicate_chunks_do_not_inflate_scores() -> None:
    gold = {MSFT_1A}
    one = [_hit(ACCESSION, "1A", 1)]
    many = [_hit(ACCESSION, "1A", rank, chunk_id=rank) for rank in range(1, 6)]
    collapsed_one = collapse_sections(one)
    collapsed_many = collapse_sections(many)
    assert collapsed_many == (MSFT_1A,)
    assert recall_at(gold, collapsed_many, 5) == recall_at(gold, collapsed_one, 5) == 1.0
    assert mrr(gold, collapsed_many) == mrr(gold, collapsed_one) == 1.0
    assert ndcg_at(gold, collapsed_many, 10) == pytest.approx(
        ndcg_at(gold, collapsed_one, 10)
    )


def test_duplicate_gold_chunks_do_not_count_as_two_sections() -> None:
    gold = {MSFT_1A, JPM_1A}
    hits = [
        _hit(ACCESSION, "1A", 1, chunk_id=1),
        _hit(ACCESSION, "1A", 2, chunk_id=2),
        _hit(JPM_ACCESSION, "1A", 3, chunk_id=3),
    ]
    sections = collapse_sections(hits)
    assert recall_at(gold, sections, 1) == 0.5
    assert recall_at(gold, sections, 5) == 1.0
    assert mrr(gold, sections) == 1.0


def test_score_question_fills_recall_ks() -> None:
    question = _question(
        citations=[
            {"accession": ACCESSION, "item_code": "1A", "quote": "Cyber."},
            {"accession": JPM_ACCESSION, "item_code": "1A", "quote": "Rates."},
        ]
    )
    hits = [
        _hit(ACCESSION, "7", 1, chunk_id=1),
        _hit(ACCESSION, "1A", 2, chunk_id=2),
        _hit(JPM_ACCESSION, "1A", 3, chunk_id=3),
    ]
    score = score_question(question, hits)
    assert score.answerable is True
    assert score.recall_at_1 == 0.0
    assert score.recall_at_5 == 1.0
    assert score.recall_at_10 == 1.0
    assert score.mrr == 0.5
    assert score.retrieved_sections == (MSFT_7, MSFT_1A, JPM_1A)


def test_unanswerable_metrics_are_none() -> None:
    score = score_question(_question(answerable=False), [_hit(ACCESSION, "1A", 1)])
    assert score.answerable is False
    assert score.recall_at_1 is None
    assert score.recall_at_5 is None
    assert score.recall_at_10 is None
    assert score.mrr is None
    assert score.ndcg_at_10 is None
    assert score.retrieved_sections == (MSFT_1A,)


def test_mean_excludes_unanswerable() -> None:
    hits = [_hit(ACCESSION, "1A", 1)]
    rows = [
        score_question(_question(), hits),
        score_question(_question(answerable=False), hits),
    ]
    mean = mean_scores(rows)
    assert mean is not None
    assert mean.n == 1
    assert mean.recall_at_5 == 1.0
    assert mean.mrr == 1.0
    assert mean.ndcg_at_10 == pytest.approx(1.0)


def test_mean_none_when_only_unanswerable() -> None:
    assert mean_scores([score_question(_question(answerable=False), [])]) is None


def test_rejects_nonpositive_k() -> None:
    gold = {MSFT_1A}
    with pytest.raises(ValueError, match="k must be"):
        recall_at(gold, (MSFT_1A,), 0)
    with pytest.raises(ValueError, match="k must be"):
        ndcg_at(gold, (MSFT_1A,), 0)
