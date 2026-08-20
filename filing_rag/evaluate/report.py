"""Ablation markdown from Stage 1/2 JSONL. No live retrieval or LLM calls."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from filing_rag.chunking.config import STRATEGIES
from filing_rag.evaluate.metrics import mean_scores
from filing_rag.evaluate.pipeline import _report as retrieval_report
from filing_rag.evaluate.pipeline import eval_grid
from filing_rag.evaluate.rag import _report as rag_report
from filing_rag.evaluate.rag import winning_strategy
from filing_rag.evaluate.store import (
    JSONL_NAME,
    RAG_JSONL_NAME,
    format_rag_table,
    format_table,
    load_jsonl,
    load_rag_jsonl,
)
from filing_rag.evaluate.types import (
    EVAL_K,
    ConfigReport,
    QueryRow,
    QueryScore,
    RagConfigReport,
    RagQueryRow,
)
from filing_rag.settings import PROJECT_ROOT, Settings, get_settings

ABLATION_START = "<!-- ablation:start -->"
ABLATION_END = "<!-- ablation:end -->"
ABLATION_NAME = "ablation.md"
STAGE2_K = 5
MISSING_MARKERS = (
    f"README is missing {ABLATION_START} / {ABLATION_END} markers. "
    "Add an Ablation section before running filing-rag report."
)


def run_report(
    *,
    retrieval_jsonl: Path | None = None,
    rag_jsonl: Path | None = None,
    output: Path | None = None,
    readme: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Write ``results/ablation.md`` and inject headlines into the README markers."""
    resolved = settings or get_settings()
    stage1_path = (
        retrieval_jsonl if retrieval_jsonl is not None else resolved.results_dir / JSONL_NAME
    )
    stage2_path = rag_jsonl if rag_jsonl is not None else resolved.results_dir / RAG_JSONL_NAME
    dest = output if output is not None else resolved.results_dir / ABLATION_NAME
    readme_path = readme if readme is not None else PROJECT_ROOT / "README.md"

    stage1_rows = load_jsonl(stage1_path)
    winner = winning_strategy(stage1_path)
    stage2_rows: tuple[RagQueryRow, ...] = ()
    if stage2_path.is_file():
        stage2_rows = load_rag_jsonl(stage2_path)
    elif rag_jsonl is not None:
        raise ValueError(f"Stage 2 JSONL not found: {stage2_path}")

    markdown, headlines = render_ablation(stage1_rows, stage2_rows, winner=winner)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown, encoding="utf-8")
    inject_readme(readme_path, headlines)
    return dest


def render_ablation(
    stage1_rows: Sequence[QueryRow],
    stage2_rows: Sequence[RagQueryRow] = (),
    *,
    winner: str | None = None,
) -> tuple[str, str]:
    """Return ``(ablation.md body, README headlines)``."""
    if winner is None:
        raise ValueError("winner is required")
    stage1_reports = _reports_for(stage1_rows, retrieval_report)
    winner_rows = [row for row in stage1_rows if row.strategy == winner]
    winner_reports = _reports_for(winner_rows, retrieval_report, strategies=(winner,))
    stage2_winner = [row for row in stage2_rows if row.strategy == winner]
    stage2_reports = (
        _reports_for(stage2_winner, rag_report, strategies=(winner,)) if stage2_winner else ()
    )
    headlines = _headlines(stage1_rows, winner_reports, stage2_reports, winner=winner)
    parts = [
        f"### Stage 1 retrieval (k={EVAL_K})",
        "",
        "```",
        format_table(stage1_reports),
        "```",
        "",
    ]
    if stage2_reports:
        parts.extend(
            [
                f"### Stage 2 generation (k={STAGE2_K})",
                "",
                "```",
                format_rag_table(stage2_reports),
                "```",
                "",
            ]
        )
    else:
        parts.extend(["_Stage 2 JSONL not found; generation table omitted._", ""])
    parts.extend(["### Definition of done", "", headlines])
    return "\n".join(parts).rstrip() + "\n", headlines


def inject_readme(path: Path, headlines: str) -> None:
    text = path.read_text(encoding="utf-8")
    if ABLATION_START not in text or ABLATION_END not in text:
        raise ValueError(MISSING_MARKERS)
    before, rest = text.split(ABLATION_START, 1)
    _, after = rest.split(ABLATION_END, 1)
    path.write_text(
        f"{before}{ABLATION_START}\n{headlines.rstrip()}\n{ABLATION_END}{after}",
        encoding="utf-8",
    )


def _reports_for(
    rows: Sequence[QueryRow] | Sequence[RagQueryRow],
    reporter,
    *,
    strategies: Sequence[str] | None = None,
) -> tuple:
    present = {(row.strategy, row.mode, row.rerank) for row in rows}
    grid = eval_grid(strategies=strategies)
    return tuple(
        reporter(config, rows)
        for config in grid
        if (config.strategy, config.mode, config.rerank) in present
    )


def _headlines(
    stage1_rows: Sequence[QueryRow],
    winner_reports: Sequence[ConfigReport],
    stage2_reports: Sequence[RagConfigReport],
    *,
    winner: str,
) -> str:
    return "\n".join(
        [
            f"1. {_headline_winner(stage1_rows, winner)}",
            f"2. {_headline_lift(winner_reports, winner)}",
            f"3. {_headline_cost(winner_reports, stage2_reports)}",
            f"4. {_headline_faith(stage2_reports)}",
        ]
    )


def _headline_winner(rows: Sequence[QueryRow], winner: str) -> str:
    means: list[tuple[str, float]] = []
    for strategy in STRATEGIES:
        scores = mean_scores([_as_score(row) for row in rows if row.strategy == strategy])
        if scores is None:
            continue
        means.append((strategy, scores.recall_at_5))
    by_name = dict(means)
    win_r5 = by_name[winner]
    others = [(name, value) for name, value in means if name != winner]
    if not others:
        return (
            f"{winner} is the winning chunker at k={EVAL_K} "
            f"(mean recall@5={win_r5:.3f})."
        )
    gaps = " and ".join(
        f"{win_r5 - value:+.3f} vs {name}" for name, value in others
    )
    return (
        f"{winner} is the winning chunker at k={EVAL_K} "
        f"(mean recall@5={win_r5:.3f}), {gaps}."
    )


def _headline_lift(reports: Sequence[ConfigReport], winner: str) -> str:
    hybrid = _find_config(reports, mode="hybrid", rerank=True)
    dense = _find_config(reports, mode="dense", rerank=False)
    if (
        hybrid is None
        or dense is None
        or hybrid.scores is None
        or dense.scores is None
    ):
        return (
            f"On {winner} at k={EVAL_K}, hybrid+rerank recall@5 lift vs dense "
            "with rerank off is n/a (config pair missing)."
        )
    lift = hybrid.scores.recall_at_5 - dense.scores.recall_at_5
    return (
        f"On {winner} at k={EVAL_K}, hybrid+rerank recall@5={hybrid.scores.recall_at_5:.3f}, "
        f"a {lift:+.3f} lift vs dense with rerank off "
        f"({dense.scores.recall_at_5:.3f})."
    )


def _headline_cost(
    stage1: Sequence[ConfigReport],
    stage2: Sequence[RagConfigReport],
) -> str:
    hybrid2 = _find_config(stage2, mode="hybrid", rerank=True)
    dense2 = _find_config(stage2, mode="dense", rerank=False)
    if hybrid2 is not None and dense2 is not None:
        return (
            f"At k={STAGE2_K}, hybrid+rerank p95={hybrid2.p95_ms:.0f}ms and "
            f"${hybrid2.usd:.6f}/query vs dense with rerank off "
            f"p95={dense2.p95_ms:.0f}ms and ${dense2.usd:.6f}/query."
        )
    hybrid1 = _find_config(stage1, mode="hybrid", rerank=True)
    dense1 = _find_config(stage1, mode="dense", rerank=False)
    if hybrid1 is None or dense1 is None:
        return (
            f"At k={EVAL_K}, hybrid+rerank vs dense with rerank off p95 n/a; "
            "dollars/query n/a."
        )
    return (
        f"At k={EVAL_K}, hybrid+rerank p95={hybrid1.p95_ms:.0f}ms vs dense "
        f"with rerank off p95={dense1.p95_ms:.0f}ms; dollars/query n/a."
    )


def _headline_faith(reports: Sequence[RagConfigReport]) -> str:
    hybrid = _find_config(reports, mode="hybrid", rerank=True)
    if hybrid is None or hybrid.ragas is None or hybrid.refusal_rate is None:
        return (
            f"Winner hybrid+rerank faithfulness at k={STAGE2_K} is n/a; "
            "refusal rate n/a (no Stage 2 rows for that config)."
        )
    n_unans = hybrid.n_unanswerable
    return (
        f"Winner hybrid+rerank faithfulness is {hybrid.ragas.faithfulness:.3f} "
        f"at k={STAGE2_K}; refusal rate on that config is {hybrid.refusal_rate:.3f} "
        f"({n_unans} unanswerable)."
    )


def _find_config(reports: Sequence, *, mode: str, rerank: bool):
    for report in reports:
        if report.mode == mode and report.rerank is rerank:
            return report
    return None


def _as_score(row: QueryRow) -> QueryScore:
    return QueryScore(
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
