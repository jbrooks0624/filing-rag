"""Persist eval rows as JSONL and format the ablation table."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

from filing_rag.evaluate.types import (
    ConfigReport,
    EvalResult,
    QueryRow,
    RagConfigReport,
    RagEvalResult,
    RagQueryRow,
    SectionKey,
)

JSONL_NAME = "eval-retrieval.jsonl"
RAG_JSONL_NAME = "eval-rag.jsonl"


def write_jsonl(rows: Sequence[QueryRow], path: Path) -> Path:
    """Overwrite ``path`` with one JSON object per row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row_payload(row)) + "\n")
    return path


def load_jsonl(path: Path) -> tuple[QueryRow, ...]:
    """Read Stage 1 JSONL written by ``write_jsonl``."""
    if not path.is_file():
        raise ValueError(f"Stage 1 JSONL not found: {path}")
    rows: list[QueryRow] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rows.append(_query_row(json.loads(line)))
    return tuple(rows)


def row_payload(row: QueryRow) -> dict[str, object]:
    return {
        "strategy": row.strategy,
        "mode": row.mode,
        "rerank": row.rerank,
        "question_id": row.question_id,
        "type": row.question_type,
        "query": row.query,
        "answerable": row.answerable,
        "gold_sections": _sections(row.gold_sections),
        "retrieved_sections": _sections(row.retrieved_sections),
        "recall_at_1": row.recall_at_1,
        "recall_at_5": row.recall_at_5,
        "recall_at_10": row.recall_at_10,
        "mrr": row.mrr,
        "ndcg_at_10": row.ndcg_at_10,
        "encode_ms": row.encode_ms,
        "dense_ms": row.dense_ms,
        "sparse_ms": row.sparse_ms,
        "fuse_ms": row.fuse_ms,
        "rerank_ms": row.rerank_ms,
        "total_ms": row.total_ms,
    }


def percentile(values: Sequence[float], p: float) -> float:
    """Linear interpolation over the sorted sample. ``p`` is in ``[0, 1]``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if not values:
        raise ValueError("percentile is undefined for empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * p
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return float(ordered[lo])
    weight = index - lo
    return float(ordered[lo]) * (1.0 - weight) + float(ordered[hi]) * weight


def format_summary(result: EvalResult) -> str:
    n_answerable = result.n_questions - result.n_unanswerable
    return (
        f"configs={result.n_configs} questions={n_answerable} "
        f"skipped_unanswerable={result.n_unanswerable}"
    )


def format_table(reports: Sequence[ConfigReport]) -> str:
    lines: list[str] = []
    for report in reports:
        rerank = "on" if report.rerank else "off"
        if report.scores is None:
            quality = "recall@5=n/a  mrr=n/a  ndcg@10=n/a"
        else:
            quality = (
                f"recall@5={report.scores.recall_at_5:.3f}  "
                f"mrr={report.scores.mrr:.3f}  "
                f"ndcg@10={report.scores.ndcg_at_10:.3f}"
            )
        lines.append(
            f"{report.strategy:11} {report.mode:7} {rerank:3}  {quality}  "
            f"p50={report.p50_ms:.0f}ms p95={report.p95_ms:.0f}ms"
        )
    return "\n".join(lines)


def write_rag_jsonl(rows: Sequence[RagQueryRow], path: Path) -> Path:
    """Overwrite ``path`` with one JSON object per Stage 2 row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(rag_row_payload(row)) + "\n")
    return path


def load_rag_jsonl(path: Path) -> tuple[RagQueryRow, ...]:
    """Read Stage 2 JSONL written by ``write_rag_jsonl``."""
    if not path.is_file():
        raise ValueError(f"Stage 2 JSONL not found: {path}")
    rows: list[RagQueryRow] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rows.append(_rag_query_row(json.loads(line)))
    return tuple(rows)


def rag_row_payload(row: RagQueryRow) -> dict[str, object]:
    return {
        "strategy": row.strategy,
        "mode": row.mode,
        "rerank": row.rerank,
        "question_id": row.question_id,
        "type": row.question_type,
        "query": row.query,
        "answerable": row.answerable,
        "response": row.response,
        "refused": row.refused,
        "retrieved_sections": _sections(row.retrieved_sections),
        "faithfulness": row.faithfulness,
        "context_precision": row.context_precision,
        "context_recall": row.context_recall,
        "relevancy": row.relevancy,
        "judge_ms": row.judge_ms,
        "judge_usd": row.judge_usd,
        "encode_ms": row.encode_ms,
        "dense_ms": row.dense_ms,
        "sparse_ms": row.sparse_ms,
        "fuse_ms": row.fuse_ms,
        "rerank_ms": row.rerank_ms,
        "total_ms": row.total_ms,
        "generate_ms": row.generate_ms,
        "serving_ms": row.serving_ms,
        "prompt_tokens": row.prompt_tokens,
        "completion_tokens": row.completion_tokens,
        "usd": row.usd,
    }


def format_rag_summary(result: RagEvalResult) -> str:
    n_answerable = result.n_questions - result.n_unanswerable
    refusal = f"{result.refusal_rate:.3f}" if result.refusal_rate is not None else "n/a"
    return (
        f"strategy={result.strategy} configs={result.n_configs} "
        f"questions={n_answerable} skipped_unanswerable={result.n_unanswerable} "
        f"refusal={refusal}"
    )


def format_rag_table(reports: Sequence[RagConfigReport]) -> str:
    lines: list[str] = []
    for report in reports:
        rerank = "on" if report.rerank else "off"
        if report.ragas is None:
            quality = "faith=n/a  ctx_p=n/a  ctx_r=n/a  rel=n/a"
        else:
            quality = (
                f"faith={report.ragas.faithfulness:.3f}  "
                f"ctx_p={report.ragas.context_precision:.3f}  "
                f"ctx_r={report.ragas.context_recall:.3f}  "
                f"rel={report.ragas.relevancy:.3f}"
            )
        lines.append(
            f"{report.mode:7} {rerank:3}  {quality}  "
            f"p50={report.p50_ms:.0f}ms p95={report.p95_ms:.0f}ms "
            f"usd={report.usd:.6f}"
        )
    return "\n".join(lines)


def _sections(keys: Sequence[SectionKey]) -> list[dict[str, str]]:
    return [{"accession": accession, "item_code": item} for accession, item in keys]


def _parse_sections(raw: object) -> tuple[SectionKey, ...]:
    if not isinstance(raw, list):
        return ()
    keys: list[SectionKey] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        keys.append((str(item.get("accession", "")), str(item.get("item_code", ""))))
    return tuple(keys)


def _query_row(payload: dict[str, object]) -> QueryRow:
    return QueryRow(
        strategy=str(payload.get("strategy", "")),
        mode=str(payload.get("mode", "")),
        rerank=bool(payload.get("rerank", False)),
        question_id=str(payload.get("question_id", "")),
        question_type=str(payload.get("type", "")),
        query=str(payload.get("query", "")),
        answerable=bool(payload.get("answerable", False)),
        gold_sections=_parse_sections(payload.get("gold_sections")),
        retrieved_sections=_parse_sections(payload.get("retrieved_sections")),
        recall_at_1=_optional_float(payload.get("recall_at_1")),
        recall_at_5=_optional_float(payload.get("recall_at_5")),
        recall_at_10=_optional_float(payload.get("recall_at_10")),
        mrr=_optional_float(payload.get("mrr")),
        ndcg_at_10=_optional_float(payload.get("ndcg_at_10")),
        encode_ms=_float(payload.get("encode_ms")),
        dense_ms=_float(payload.get("dense_ms")),
        sparse_ms=_float(payload.get("sparse_ms")),
        fuse_ms=_float(payload.get("fuse_ms")),
        rerank_ms=_float(payload.get("rerank_ms")),
        total_ms=_float(payload.get("total_ms")),
    )


def _rag_query_row(payload: dict[str, object]) -> RagQueryRow:
    return RagQueryRow(
        strategy=str(payload.get("strategy", "")),
        mode=str(payload.get("mode", "")),
        rerank=bool(payload.get("rerank", False)),
        question_id=str(payload.get("question_id", "")),
        question_type=str(payload.get("type", "")),
        query=str(payload.get("query", "")),
        answerable=bool(payload.get("answerable", False)),
        response=str(payload.get("response", "")),
        refused=bool(payload.get("refused", False)),
        retrieved_sections=_parse_sections(payload.get("retrieved_sections")),
        faithfulness=_optional_float(payload.get("faithfulness")),
        context_precision=_optional_float(payload.get("context_precision")),
        context_recall=_optional_float(payload.get("context_recall")),
        relevancy=_optional_float(payload.get("relevancy")),
        judge_ms=_optional_float(payload.get("judge_ms")),
        judge_usd=_optional_float(payload.get("judge_usd")),
        encode_ms=_float(payload.get("encode_ms")),
        dense_ms=_float(payload.get("dense_ms")),
        sparse_ms=_float(payload.get("sparse_ms")),
        fuse_ms=_float(payload.get("fuse_ms")),
        rerank_ms=_float(payload.get("rerank_ms")),
        total_ms=_float(payload.get("total_ms")),
        generate_ms=_float(payload.get("generate_ms")),
        serving_ms=_float(payload.get("serving_ms")),
        prompt_tokens=_int(payload.get("prompt_tokens")),
        completion_tokens=_int(payload.get("completion_tokens")),
        usd=_float(payload.get("usd")),
    )


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _float(value: object) -> float:
    parsed = _optional_float(value)
    return 0.0 if parsed is None else parsed


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value
