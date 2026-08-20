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
from filing_rag.retrieve.types import MODES
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
    """Write ``results/ablation.md`` and inject the full block into the README markers."""
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

    markdown = render_ablation(stage1_rows, stage2_rows, winner=winner)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown, encoding="utf-8")
    inject_readme(readme_path, markdown)
    return dest


def render_ablation(
    stage1_rows: Sequence[QueryRow],
    stage2_rows: Sequence[RagQueryRow] = (),
    *,
    winner: str | None = None,
) -> str:
    """Return the ablation.md body (tables, headlines, and reads)."""
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
    reads = _reads(winner_reports, stage2_reports, winner=winner)
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
    parts.extend(["### Definition of done", "", headlines, ""])
    if reads:
        parts.extend(["### Reads", "", reads, ""])
    return "\n".join(parts).rstrip() + "\n"


def inject_readme(path: Path, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    if ABLATION_START not in text or ABLATION_END not in text:
        raise ValueError(MISSING_MARKERS)
    before, rest = text.split(ABLATION_START, 1)
    _, after = rest.split(ABLATION_END, 1)
    path.write_text(
        f"{before}{ABLATION_START}\n{body.rstrip()}\n{ABLATION_END}{after}",
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
            f"2. {_headline_best(winner_reports, winner)}",
            f"3. {_headline_stage2(stage2_reports)}",
            f"4. {_headline_refusal(stage2_reports)}",
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


def _headline_best(reports: Sequence[ConfigReport], winner: str) -> str:
    best = _best_retrieval(reports)
    if best is None or best.scores is None:
        return (
            f"On {winner} at k={EVAL_K}, best retrieval config is n/a "
            "(no scored configs)."
        )
    r5 = _r5(best)
    bits = [
        f"On {winner} at k={EVAL_K}, {_label(best)} is the best retrieval config "
        f"(recall@5={r5:.3f})"
    ]
    hybrid_on = _find_config(reports, mode="hybrid", rerank=True)
    if (
        hybrid_on is not None
        and hybrid_on.scores is not None
        and (best.mode, best.rerank) != ("hybrid", True)
    ):
        hybrid_r5 = _r5(hybrid_on)
        gap = r5 - hybrid_r5
        bits.append(f"{gap:+.3f} vs hybrid+rerank ({hybrid_r5:.3f})")
    lead = ", ".join(bits) + "."
    decomp = _decomposition_sentences(reports)
    if not decomp:
        return lead
    return lead + " " + " ".join(decomp)


def _headline_stage2(reports: Sequence[RagConfigReport]) -> str:
    if not reports:
        return (
            f"At k={STAGE2_K}, Stage 2 faithfulness is n/a "
            "(no Stage 2 rows)."
        )
    best = _best_faith(reports)
    if best is None or best.ragas is None:
        return f"At k={STAGE2_K}, Stage 2 faithfulness is n/a (no scored configs)."
    lead = (
        f"At k={STAGE2_K}, {_label(best)} has the best faithfulness "
        f"({best.ragas.faithfulness:.3f})"
    )
    if not best.rerank:
        on = _find_config(reports, mode=best.mode, rerank=True)
        if on is not None and on.ragas is not None:
            lead += f"; turning rerank on drops it to {on.ragas.faithfulness:.3f}"
            lead += " " + _p50_clause(best.p50_ms, on.p50_ms)
    lead += "."
    if _all_rerank_hurt_ctx_p(reports):
        lead += " Context precision falls for every rerank-on config."
    return lead


def _headline_refusal(reports: Sequence[RagConfigReport]) -> str:
    if not reports:
        return (
            "Refusal rate n/a; context precision n/a; dollars/query n/a "
            "(no Stage 2 rows)."
        )
    parts: list[str] = []
    rates = [report.refusal_rate for report in reports if report.refusal_rate is not None]
    n_unans = max((report.n_unanswerable for report in reports), default=0)
    if rates:
        noun = "question" if n_unans == 1 else "questions"
        if all(abs(rate - rates[0]) < 1e-9 for rate in rates):
            parts.append(
                f"Refusal rate is {rates[0]:.3f} on all {n_unans} unanswerable {noun}"
            )
        else:
            parts.append(
                f"Refusal rate ranges {min(rates):.3f}–{max(rates):.3f} "
                f"on {n_unans} unanswerable {noun}"
            )
    precs = [
        report.ragas.context_precision for report in reports if report.ragas is not None
    ]
    if precs:
        parts.append(
            f"context precision is the remaining lever "
            f"({min(precs):.2f}–{max(precs):.2f})"
        )
    usds = [report.usd for report in reports]
    if usds:
        lo, hi = min(usds), max(usds)
        if lo > 0 and (hi - lo) / lo <= 0.10:
            spread = (hi - lo) / lo * 100
            parts.append(
                f"generation cost is flat (${lo:.6f}–${hi:.6f}, ~{spread:.0f}% spread)"
            )
        else:
            parts.append(f"generation cost is ${lo:.6f}–${hi:.6f}/query")
    if not parts:
        return (
            "Refusal rate n/a; context precision n/a; dollars/query n/a "
            "(no Stage 2 quality rows)."
        )
    return "; ".join(parts) + "."


def _reads(
    winner_reports: Sequence[ConfigReport],
    stage2_reports: Sequence[RagConfigReport],
    *,
    winner: str,
) -> str:
    paragraphs: list[str] = []
    stage1 = _reads_stage1(winner_reports, winner=winner)
    if stage1:
        paragraphs.append(stage1)
    stage2 = _reads_stage2(stage2_reports)
    if stage2:
        paragraphs.append(stage2)
    leftover = _reads_leftover(stage2_reports)
    if leftover:
        paragraphs.append(leftover)
    return "\n\n".join(paragraphs)


def _reads_stage1(reports: Sequence[ConfigReport], *, winner: str) -> str:
    best = _best_retrieval(reports)
    if best is None or best.scores is None:
        return ""
    hybrid_on = _find_config(reports, mode="hybrid", rerank=True)
    lines: list[str] = []
    if (
        hybrid_on is not None
        and hybrid_on.scores is not None
        and (best.mode, best.rerank) != ("hybrid", True)
    ):
        lines.append(
            f"Stage 1 (k={EVAL_K}). {_cap(_label(best))} is the best retrieval config "
            f"(recall@5={_r5(best):.3f}), not hybrid+rerank "
            f"({_r5(hybrid_on):.3f}). Decomposition on {winner}:"
        )
    else:
        lines.append(
            f"Stage 1 (k={EVAL_K}). {_cap(_label(best))} is the best retrieval config "
            f"(recall@5={_r5(best):.3f})."
        )
    bullets = _decomposition_bullets(reports)
    if bullets:
        lines.append("")
        lines.extend(bullets)
        lines.append("")
    redundant = _fusion_redundant(reports)
    if redundant:
        lines.append(redundant)
    return "\n".join(lines).rstrip()


def _reads_stage2(reports: Sequence[RagConfigReport]) -> str:
    if not reports:
        return ""
    best = _best_faith(reports)
    if best is None or best.ragas is None:
        return ""
    prefix = (
        f"Stage 2 (k={STAGE2_K}) disagrees. "
        if not best.rerank
        else f"Stage 2 (k={STAGE2_K}). "
    )
    parts = [
        f"{prefix}Best faithfulness is {_label(best)} "
        f"({best.ragas.faithfulness:.3f})."
    ]
    if not best.rerank:
        on = _find_config(reports, mode=best.mode, rerank=True)
        if on is not None and on.ragas is not None:
            parts.append(
                f"Turning rerank on drops it to {on.ragas.faithfulness:.3f} "
                f"{_p50_clause(best.p50_ms, on.p50_ms)}."
            )
    drops = _ctx_p_drops(reports)
    if _all_rerank_hurt_ctx_p(reports) and drops:
        pairs = ", ".join(
            f"{off:.3f}→{on:.3f}" for _mode, off, on in drops
        )
        parts.append(
            f"Context precision falls on every rerank-on config ({pairs})."
        )
        parts.append(
            "A cross-encoder optimizes topical query-chunk relevance, not answer "
            "supportiveness; at serving k it swaps in on-topic chunks that do not "
            "carry the evidence."
        )
    if not best.rerank:
        parts.append(
            "Reranking improves retrieval depth at k="
            f"{EVAL_K} and degrades answer faithfulness at serving k={STAGE2_K}."
        )
    return " ".join(parts)


def _reads_leftover(reports: Sequence[RagConfigReport]) -> str:
    if not reports:
        return ""
    bits: list[str] = []
    rates = [report.refusal_rate for report in reports if report.refusal_rate is not None]
    n_unans = max((report.n_unanswerable for report in reports), default=0)
    if rates and all(abs(rate - rates[0]) < 1e-9 for rate in rates):
        noun = "question" if n_unans == 1 else "questions"
        bits.append(
            f"Refusal is {rates[0]:.3f} on all {n_unans} unanswerable {noun}."
        )
    precs = [
        report.ragas.context_precision for report in reports if report.ragas is not None
    ]
    if precs:
        lo, hi = min(precs), max(precs)
        if hi <= 0.40:
            bits.append(
                f"Context precision is the weak spot ({lo:.2f}–{hi:.2f}): "
                "roughly two-thirds of retrieved context is not supporting the answer."
            )
        else:
            bits.append(f"Context precision ranges {lo:.2f}–{hi:.2f}.")
    usds = [report.usd for report in reports]
    if usds:
        lo, hi = min(usds), max(usds)
        if lo > 0 and (hi - lo) / lo <= 0.10:
            spread = (hi - lo) / lo * 100
            bits.append(
                f"Generation cost is effectively flat (${lo:.6f}–${hi:.6f}, "
                f"~{spread:.0f}% spread); retrieval config does not move spend — "
                "k and chunk size do."
            )
    return " ".join(bits)


def _decomposition_sentences(reports: Sequence[ConfigReport]) -> list[str]:
    cells = _fusion_cells(reports)
    if cells is None:
        return []
    dense_off, dense_on, hybrid_off, hybrid_on = cells
    return [
        (
            f"Reranking alone, on dense: {dense_off:.3f} → {dense_on:.3f} "
            f"({dense_on - dense_off:+.3f})."
        ),
        (
            f"Fusion alone, no rerank: {dense_off:.3f} → {hybrid_off:.3f} "
            f"({hybrid_off - dense_off:+.3f})."
        ),
        (
            f"Reranking on top of fusion: {hybrid_off:.3f} → {hybrid_on:.3f} "
            f"({hybrid_on - hybrid_off:+.3f})."
        ),
    ]


def _decomposition_bullets(reports: Sequence[ConfigReport]) -> list[str]:
    return [f"- {sentence.rstrip('.')}" for sentence in _decomposition_sentences(reports)]


def _fusion_redundant(reports: Sequence[ConfigReport]) -> str:
    cells = _fusion_cells(reports)
    if cells is None:
        return ""
    dense_off, dense_on, hybrid_off, hybrid_on = cells
    if dense_on <= hybrid_on:
        return ""
    text = (
        "Fusion and reranking recover largely the same chunks. Once you rerank, "
        "fusion adds almost nothing."
    )
    rerank_lift = dense_on - dense_off
    fusion_lift = hybrid_off - dense_off
    if fusion_lift > 0 and rerank_lift >= 2 * fusion_lift:
        text += (
            " Reranking alone beats fusion alone by ~2x. "
            "Hybrid fusion did not pay for itself."
        )
    else:
        text += " Hybrid fusion did not pay for itself."
    return text


def _fusion_cells(
    reports: Sequence[ConfigReport],
) -> tuple[float, float, float, float] | None:
    dense_off = _find_config(reports, mode="dense", rerank=False)
    dense_on = _find_config(reports, mode="dense", rerank=True)
    hybrid_off = _find_config(reports, mode="hybrid", rerank=False)
    hybrid_on = _find_config(reports, mode="hybrid", rerank=True)
    if not all(
        report is not None and report.scores is not None
        for report in (dense_off, dense_on, hybrid_off, hybrid_on)
    ):
        return None
    assert dense_off is not None and dense_off.scores is not None
    assert dense_on is not None and dense_on.scores is not None
    assert hybrid_off is not None and hybrid_off.scores is not None
    assert hybrid_on is not None and hybrid_on.scores is not None
    return (
        _r5(dense_off),
        _r5(dense_on),
        _r5(hybrid_off),
        _r5(hybrid_on),
    )


def _best_retrieval(reports: Sequence[ConfigReport]) -> ConfigReport | None:
    scored = [report for report in reports if report.scores is not None]
    if not scored:
        return None

    def key(report: ConfigReport) -> tuple[float, float, float]:
        scores = report.scores
        assert scores is not None
        return (scores.recall_at_5, scores.ndcg_at_10, scores.mrr)

    return max(scored, key=key)


def _best_faith(reports: Sequence[RagConfigReport]) -> RagConfigReport | None:
    scored = [report for report in reports if report.ragas is not None]
    if not scored:
        return None

    def key(report: RagConfigReport) -> float:
        ragas = report.ragas
        assert ragas is not None
        return ragas.faithfulness

    return max(scored, key=key)


def _ctx_p_drops(
    reports: Sequence[RagConfigReport],
) -> list[tuple[str, float, float]]:
    by_mode: dict[str, dict[bool, RagConfigReport]] = {}
    for report in reports:
        if report.ragas is None:
            continue
        by_mode.setdefault(report.mode, {})[report.rerank] = report
    drops: list[tuple[str, float, float]] = []
    for mode in MODES:
        pair = by_mode.get(mode, {})
        if True not in pair or False not in pair:
            continue
        off = pair[False].ragas.context_precision
        on = pair[True].ragas.context_precision
        drops.append((mode, off, on))
    return drops


def _all_rerank_hurt_ctx_p(reports: Sequence[RagConfigReport]) -> bool:
    drops = _ctx_p_drops(reports)
    if not drops:
        return False
    return all(on < off for _mode, off, on in drops)


def _p50_clause(off_ms: float, on_ms: float) -> str:
    if off_ms > 0 and on_ms / off_ms >= 2.5:
        return f"and nearly triples p50 ({off_ms:.0f}ms → {on_ms:.0f}ms)"
    return f"and raises p50 from {off_ms:.0f}ms to {on_ms:.0f}ms"


def _label(report: ConfigReport | RagConfigReport) -> str:
    if report.rerank:
        return f"{report.mode}+rerank"
    return f"{report.mode} with rerank off"


def _cap(label: str) -> str:
    return label[:1].upper() + label[1:]


def _r5(report: ConfigReport) -> float:
    scores = report.scores
    assert scores is not None
    return round(scores.recall_at_5, 3)


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
