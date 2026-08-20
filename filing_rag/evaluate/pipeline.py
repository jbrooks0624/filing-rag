"""Run the retrieval eval grid. Tests inject a Searcher; live path uses Retriever."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from filing_rag.chunking.config import STRATEGIES
from filing_rag.embed.store import require_k, require_strategy
from filing_rag.evaluate.golden import load_golden
from filing_rag.evaluate.metrics import mean_scores, score_question
from filing_rag.evaluate.store import JSONL_NAME, percentile, write_jsonl
from filing_rag.evaluate.types import (
    EVAL_K,
    ConfigReport,
    EvalResult,
    GoldenSet,
    GridConfig,
    QueryRow,
    QueryScore,
)
from filing_rag.retrieve.pipeline import Retriever
from filing_rag.retrieve.types import MODES, Filters, RetrieveResult
from filing_rag.settings import Settings, get_settings


class Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        strategy: str,
        mode: str = "hybrid",
        k: int | None = None,
        rerank: bool = False,
        filters: Filters | None = None,
        force: bool = False,
    ) -> RetrieveResult: ...


def eval_grid(
    *,
    strategies: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
    rerank: bool | None = None,
) -> tuple[GridConfig, ...]:
    """Cartesian product of chunkers × modes × rerank. ``None`` means the full axis."""
    resolved_strategies = _resolve_names(strategies, STRATEGIES, "strategy")
    resolved_modes = _resolve_names(modes, MODES, "mode")
    if rerank is None:
        flags: tuple[bool, ...] = (False, True)
    else:
        flags = (rerank,)
    grid = tuple(
        GridConfig(strategy=strategy, mode=mode, rerank=flag)
        for strategy in resolved_strategies
        for mode in resolved_modes
        for flag in flags
    )
    if not grid:
        raise ValueError("eval grid is empty")
    return grid


class Evaluator:
    """Callable Stage 1 harness. Tests inject golden set, retriever, and output path."""

    def __init__(
        self,
        golden: GoldenSet,
        settings: Settings,
        retriever: Searcher,
    ) -> None:
        self.golden = golden
        self.settings = settings
        self.retriever = retriever

    @classmethod
    def from_config(
        cls,
        golden_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        retriever: Searcher | None = None,
        parsed_dir: Path | None = None,
    ) -> Evaluator:
        resolved = settings or get_settings()
        path = Path(golden_path) if golden_path is not None else resolved.golden_path
        quotes_dir = parsed_dir if parsed_dir is not None else resolved.parsed_dir
        return cls(
            load_golden(path, parsed_dir=quotes_dir),
            resolved,
            retriever if retriever is not None else Retriever.from_config(settings=resolved),
        )

    def run(
        self,
        *,
        strategies: Sequence[str] | None = None,
        modes: Sequence[str] | None = None,
        rerank: bool | None = None,
        k: int | None = None,
        force: bool = False,
        output: Path | None = None,
    ) -> EvalResult:
        if not self.golden.questions:
            raise ValueError("golden set is empty")
        top_k = require_k(k if k is not None else EVAL_K)
        configs = eval_grid(strategies=strategies, modes=modes, rerank=rerank)
        rebuilt: set[str] = set()
        rows: list[QueryRow] = []
        for config in configs:
            for question in self.golden.questions:
                rebuild = force and config.strategy not in rebuilt
                if rebuild:
                    rebuilt.add(config.strategy)
                result = self.retriever.search(
                    question.query,
                    strategy=config.strategy,
                    mode=config.mode,
                    k=top_k,
                    rerank=config.rerank,
                    force=rebuild,
                )
                score = score_question(question, result.hits)
                timings = result.timings
                rows.append(
                    QueryRow(
                        strategy=config.strategy,
                        mode=config.mode,
                        rerank=config.rerank,
                        question_id=question.id,
                        question_type=question.type,
                        query=question.query,
                        answerable=question.answerable,
                        gold_sections=score.gold_sections,
                        retrieved_sections=score.retrieved_sections,
                        recall_at_1=score.recall_at_1,
                        recall_at_5=score.recall_at_5,
                        recall_at_10=score.recall_at_10,
                        mrr=score.mrr,
                        ndcg_at_10=score.ndcg_at_10,
                        encode_ms=timings.encode_ms,
                        dense_ms=timings.dense_ms,
                        sparse_ms=timings.sparse_ms,
                        fuse_ms=timings.fuse_ms,
                        rerank_ms=timings.rerank_ms,
                        total_ms=timings.total_ms,
                    )
                )
        reports = tuple(_report(config, rows) for config in configs)
        dest = output if output is not None else self.settings.results_dir / JSONL_NAME
        write_jsonl(rows, dest)
        return EvalResult(reports=reports, rows=tuple(rows), output=dest)


def _report(config: GridConfig, rows: Sequence[QueryRow]) -> ConfigReport:
    scoped = [
        row
        for row in rows
        if row.strategy == config.strategy
        and row.mode == config.mode
        and row.rerank == config.rerank
    ]
    scores = mean_scores(
        [
            QueryScore(
                question_id=row.question_id,
                answerable=row.answerable,
                gold_sections=row.gold_sections,
                retrieved_sections=row.retrieved_sections,
                recall_at_1=row.recall_at_1,
                recall_at_5=row.recall_at_5,
                recall_at_10=row.recall_at_10,
                mrr=row.mrr,
                ndcg_at_10=row.ndcg_at_10,
            )
            for row in scoped
        ]
    )
    latencies = [row.total_ms for row in scoped]
    return ConfigReport(
        strategy=config.strategy,
        mode=config.mode,
        rerank=config.rerank,
        scores=scores,
        p50_ms=percentile(latencies, 0.50),
        p95_ms=percentile(latencies, 0.95),
        n_questions=len(scoped),
        n_unanswerable=sum(1 for row in scoped if not row.answerable),
    )


def _resolve_names(
    names: Sequence[str] | None,
    known: Sequence[str],
    kind: str,
) -> tuple[str, ...]:
    allowed = list(known)
    if names is None:
        return tuple(allowed)
    resolved: list[str] = []
    unknown: list[str] = []
    for raw in names:
        name = raw.strip().lower()
        if kind == "strategy":
            try:
                name = require_strategy(name)
            except ValueError:
                unknown.append(name)
                continue
        elif name not in allowed:
            unknown.append(name)
            continue
        if name not in resolved:
            resolved.append(name)
    if unknown:
        label = "strategies" if kind == "strategy" else "modes"
        raise ValueError(f"unknown {label}: {unknown}. Known: {', '.join(allowed)}")
    return tuple(resolved)
