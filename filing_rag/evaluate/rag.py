"""Stage 2 RAG eval. Tests inject Searcher, Answerer, and RagasScorer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from filing_rag.embed.store import require_k
from filing_rag.evaluate.golden import load_golden
from filing_rag.evaluate.metrics import collapse_sections
from filing_rag.evaluate.pipeline import Searcher, eval_grid
from filing_rag.evaluate.ragas import CollectionsRagasScorer, RagasScorer, RagasScores, mean_ragas
from filing_rag.evaluate.refusal import is_refusal, refusal_rate
from filing_rag.evaluate.store import JSONL_NAME, RAG_JSONL_NAME, percentile, write_rag_jsonl
from filing_rag.evaluate.types import (
    GoldenSet,
    GridConfig,
    RagConfigReport,
    RagEvalResult,
    RagQueryRow,
)
from filing_rag.generate.config import GenerationConfig
from filing_rag.generate.pipeline import Generator
from filing_rag.generate.types import GenerateResult
from filing_rag.retrieve.pipeline import Retriever
from filing_rag.retrieve.types import Hit, RetrieveTimings
from filing_rag.settings import Settings, get_settings

MISSING_STAGE1 = (
    "Stage 1 JSONL not found: {path}. Pass --strategy or run eval-retrieval first."
)
EMPTY_STAGE1 = (
    "Stage 1 JSONL has no answerable rows: {path}. Pass --strategy or run eval-retrieval first."
)


class Answerer(Protocol):
    def generate(self, query: str, hits: Sequence[Hit]) -> GenerateResult: ...


def winning_strategy(path: Path) -> str:
    """Chunker with the highest mean recall@5. Tie-break nDCG@10, then MRR."""
    if not path.is_file():
        raise ValueError(MISSING_STAGE1.format(path=path))
    by_strategy: dict[str, list[tuple[float, float, float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("answerable"):
                continue
            recall = row.get("recall_at_5")
            ndcg = row.get("ndcg_at_10")
            mrr = row.get("mrr")
            strategy = row.get("strategy")
            if recall is None or ndcg is None or mrr is None or not strategy:
                continue
            by_strategy.setdefault(str(strategy), []).append(
                (float(recall), float(ndcg), float(mrr))
            )
    if not by_strategy:
        raise ValueError(EMPTY_STAGE1.format(path=path))

    def score(strategy: str) -> tuple[float, float, float]:
        rows = by_strategy[strategy]
        n = len(rows)
        return (
            sum(item[0] for item in rows) / n,
            sum(item[1] for item in rows) / n,
            sum(item[2] for item in rows) / n,
        )

    return max(by_strategy, key=score)


class RagEvaluator:
    """Callable Stage 2 harness. Tests inject golden set, retriever, generator, scorer."""

    def __init__(
        self,
        golden: GoldenSet,
        settings: Settings,
        retriever: Searcher,
        generator: Answerer,
        scorer: RagasScorer,
        *,
        refusal_phrase: str | None = None,
    ) -> None:
        self.golden = golden
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.scorer = scorer
        self.refusal_phrase = (
            refusal_phrase if refusal_phrase is not None else GenerationConfig().refusal_phrase
        )

    @classmethod
    def from_config(
        cls,
        golden_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        retriever: Searcher | None = None,
        generator: Answerer | None = None,
        scorer: RagasScorer | None = None,
        parsed_dir: Path | None = None,
    ) -> RagEvaluator:
        resolved = settings or get_settings()
        path = Path(golden_path) if golden_path is not None else resolved.golden_path
        quotes_dir = parsed_dir if parsed_dir is not None else resolved.parsed_dir
        searcher = retriever if retriever is not None else Retriever.from_config(settings=resolved)
        answerer = (
            generator
            if generator is not None
            else Generator.from_config(settings=resolved, retriever=searcher)
        )
        judge = scorer
        if judge is None:
            live = CollectionsRagasScorer.from_config(settings=resolved)
            live.ensure_available()
            judge = live
        return cls(
            load_golden(path, parsed_dir=quotes_dir),
            resolved,
            searcher,
            answerer,
            judge,
        )

    def run(
        self,
        *,
        strategy: str | None = None,
        modes: Sequence[str] | None = None,
        rerank: bool | None = None,
        k: int | None = None,
        force: bool = False,
        output: Path | None = None,
        retrieval_jsonl: Path | None = None,
    ) -> RagEvalResult:
        if not self.golden.questions:
            raise ValueError("golden set is empty")
        top_k = require_k(k) if k is not None else None
        resolved = strategy
        if resolved is None:
            resolved = winning_strategy(
                retrieval_jsonl
                if retrieval_jsonl is not None
                else self.settings.results_dir / JSONL_NAME
            )
        configs = eval_grid(strategies=(resolved,), modes=modes, rerank=rerank)
        winner = configs[0].strategy
        rebuilt: set[str] = set()
        rows: list[RagQueryRow] = []
        for config in configs:
            for question in self.golden.questions:
                rebuild = force and config.strategy not in rebuilt
                if rebuild:
                    rebuilt.add(config.strategy)
                retrieved = self.retriever.search(
                    question.query,
                    strategy=config.strategy,
                    mode=config.mode,
                    k=top_k,
                    rerank=config.rerank,
                    force=rebuild,
                )
                generated = self.generator.generate(question.query, retrieved.hits)
                refused = is_refusal(generated.text, self.refusal_phrase)
                ragas = None
                if question.answerable:
                    ragas = self.scorer.score(
                        user_input=question.query,
                        response=generated.text,
                        retrieved_contexts=[hit.text for hit in retrieved.hits],
                        reference=question.answer,
                    )
                rows.append(
                    _row(
                        config,
                        question.id,
                        question.type,
                        question.query,
                        question.answerable,
                        retrieved.hits,
                        retrieved.timings,
                        generated,
                        refused,
                        ragas,
                    )
                )
        reports = tuple(_report(config, rows) for config in configs)
        dest = output if output is not None else self.settings.results_dir / RAG_JSONL_NAME
        write_rag_jsonl(rows, dest)
        return RagEvalResult(
            strategy=winner,
            reports=reports,
            rows=tuple(rows),
            output=dest,
        )


def _row(
    config: GridConfig,
    question_id: str,
    question_type: str,
    query: str,
    answerable: bool,
    hits: Sequence[Hit],
    timings: RetrieveTimings,
    generated: GenerateResult,
    refused: bool,
    ragas: RagasScores | None,
) -> RagQueryRow:
    total_ms = timings.total_ms
    generate_ms = generated.timings.generate_ms
    return RagQueryRow(
        strategy=config.strategy,
        mode=config.mode,
        rerank=config.rerank,
        question_id=question_id,
        question_type=question_type,
        query=query,
        answerable=answerable,
        response=generated.text,
        refused=refused,
        retrieved_sections=collapse_sections(hits),
        faithfulness=None if ragas is None else ragas.faithfulness,
        context_precision=None if ragas is None else ragas.context_precision,
        context_recall=None if ragas is None else ragas.context_recall,
        relevancy=None if ragas is None else ragas.relevancy,
        judge_ms=None if ragas is None else ragas.judge_ms,
        judge_usd=None,
        encode_ms=timings.encode_ms,
        dense_ms=timings.dense_ms,
        sparse_ms=timings.sparse_ms,
        fuse_ms=timings.fuse_ms,
        rerank_ms=timings.rerank_ms,
        total_ms=total_ms,
        generate_ms=generate_ms,
        serving_ms=total_ms + generate_ms,
        prompt_tokens=generated.usage.prompt_tokens,
        completion_tokens=generated.usage.completion_tokens,
        usd=generated.usd,
    )


def _report(config: GridConfig, rows: Sequence[RagQueryRow]) -> RagConfigReport:
    scoped = [
        row
        for row in rows
        if row.strategy == config.strategy
        and row.mode == config.mode
        and row.rerank == config.rerank
    ]
    ragas_rows: list[RagasScores | None] = []
    for row in scoped:
        if not row.answerable:
            ragas_rows.append(None)
            continue
        ragas_rows.append(
            RagasScores(
                faithfulness=row.faithfulness or 0.0,
                context_precision=row.context_precision or 0.0,
                context_recall=row.context_recall or 0.0,
                relevancy=row.relevancy or 0.0,
                judge_ms=row.judge_ms or 0.0,
            )
        )
    refused = [row.refused for row in scoped if not row.answerable]
    latencies = [row.serving_ms for row in scoped]
    dollars = [row.usd for row in scoped]
    return RagConfigReport(
        strategy=config.strategy,
        mode=config.mode,
        rerank=config.rerank,
        ragas=mean_ragas(ragas_rows),
        refusal_rate=refusal_rate(refused) if refused else None,
        p50_ms=percentile(latencies, 0.50),
        p95_ms=percentile(latencies, 0.95),
        usd=sum(dollars) / len(dollars),
        n_questions=len(scoped),
        n_unanswerable=sum(1 for row in scoped if not row.answerable),
    )
