"""Reciprocal rank fusion. No backends — fuse ranked Hit lists by chunk_id."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from filing_rag.embed.store import require_k
from filing_rag.retrieve.types import Hit

DEFAULT_RRF_K = 60


def rrf(
    rankings: Sequence[Sequence[Hit]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    k: int | None = None,
) -> list[Hit]:
    """Fuse rankings with ``score = Σ 1 / (rrf_k + rank)``.

    A chunk that appears twice in one ranking is counted once, using its first
    rank. Citation payload comes from the first ranking that mentioned it.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    scores: dict[int, float] = {}
    payloads: dict[int, Hit] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for hit in ranking:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + hit.rank)
            payloads.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    fused = [
        replace(payloads[chunk_id], score=scores[chunk_id], rank=rank)
        for rank, chunk_id in enumerate(ordered, start=1)
    ]
    if k is None:
        return fused
    return fused[: require_k(k)]
