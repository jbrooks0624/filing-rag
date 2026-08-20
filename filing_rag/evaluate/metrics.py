"""Section-level IR metrics. Hits collapse to unique (accession, item_code)."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

from filing_rag.embed.store import require_k
from filing_rag.evaluate.types import (
    NDCG_K,
    MeanScore,
    QueryScore,
    Question,
    SectionKey,
)
from filing_rag.retrieve.types import Hit


def collapse_sections(hits: Sequence[Hit]) -> tuple[SectionKey, ...]:
    """Unique sections in first-seen order. Later chunks from the same section drop."""
    seen: set[SectionKey] = set()
    ordered: list[SectionKey] = []
    for hit in hits:
        key = (hit.accession, hit.item_code)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return tuple(ordered)


def recall_at(
    gold: Collection[SectionKey],
    sections: Sequence[SectionKey],
    k: int,
) -> float:
    """``|gold ∩ sections[:k]| / |gold|``. Undefined when gold is empty."""
    unique = set(gold)
    if not unique:
        raise ValueError("recall is undefined for empty gold")
    retrieved = set(sections[: require_k(k)])
    return len(retrieved & unique) / len(unique)


def mrr(gold: Collection[SectionKey], sections: Sequence[SectionKey]) -> float:
    """Reciprocal rank of the first gold section, else 0. Undefined when gold is empty."""
    unique = set(gold)
    if not unique:
        raise ValueError("mrr is undefined for empty gold")
    for rank, section in enumerate(sections, start=1):
        if section in unique:
            return 1.0 / rank
    return 0.0


def ndcg_at(
    gold: Collection[SectionKey],
    sections: Sequence[SectionKey],
    k: int,
) -> float:
    """Binary nDCG@k on the collapsed section ranking. Undefined when gold is empty."""
    unique = set(gold)
    if not unique:
        raise ValueError("ndcg is undefined for empty gold")
    cutoff = require_k(k)
    gains = [1.0 if section in unique else 0.0 for section in sections[:cutoff]]
    if len(gains) < cutoff:
        gains.extend([0.0] * (cutoff - len(gains)))
    dcg = _dcg(gains)
    ideal = _dcg([1.0] * min(len(unique), cutoff))
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def score_question(question: Question, hits: Sequence[Hit]) -> QueryScore:
    """Score one question. Unanswerable rows keep retrieved sections and null metrics."""
    sections = collapse_sections(hits)
    gold = question.gold_sections
    if not question.answerable:
        return QueryScore(
            question_id=question.id,
            answerable=False,
            gold_sections=gold,
            retrieved_sections=sections,
            recall_at_1=None,
            recall_at_5=None,
            recall_at_10=None,
            mrr=None,
            ndcg_at_10=None,
        )
    return QueryScore(
        question_id=question.id,
        answerable=True,
        gold_sections=gold,
        retrieved_sections=sections,
        recall_at_1=recall_at(gold, sections, 1),
        recall_at_5=recall_at(gold, sections, 5),
        recall_at_10=recall_at(gold, sections, 10),
        mrr=mrr(gold, sections),
        ndcg_at_10=ndcg_at(gold, sections, NDCG_K),
    )


def mean_scores(rows: Sequence[QueryScore]) -> MeanScore | None:
    """Macro-average quality metrics over answerable rows. None if there are none."""
    scored = [row for row in rows if row.answerable]
    if not scored:
        return None
    n = len(scored)
    return MeanScore(
        n=n,
        recall_at_1=_mean([row.recall_at_1 for row in scored]),
        recall_at_5=_mean([row.recall_at_5 for row in scored]),
        recall_at_10=_mean([row.recall_at_10 for row in scored]),
        mrr=_mean([row.mrr for row in scored]),
        ndcg_at_10=_mean([row.ndcg_at_10 for row in scored]),
    )


def _dcg(gains: Sequence[float]) -> float:
    return sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
        if gain
    )


def _mean(values: Sequence[float | None]) -> float:
    present = [value for value in values if value is not None]
    if not present:
        raise ValueError("mean is undefined for empty values")
    return sum(present) / len(present)
