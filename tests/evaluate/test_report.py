"""Ablation report from fixture JSONL. No Postgres, no LLM, no results/."""

from pathlib import Path

import pytest
from filing_rag.evaluate.report import (
    ABLATION_END,
    ABLATION_START,
    run_report,
)
from filing_rag.evaluate.store import load_jsonl, write_jsonl, write_rag_jsonl
from filing_rag.evaluate.types import QueryRow, RagQueryRow

_FIXED_STAGE1 = (
    ("dense", False, 0.656, 25.0),
    ("dense", True, 0.789, 955.0),
    ("sparse", False, 0.656, 1.0),
    ("sparse", True, 0.667, 1149.0),
    ("hybrid", False, 0.722, 26.0),
    ("hybrid", True, 0.733, 1189.0),
)
_FIXED_STAGE2 = (
    ("dense", False, 0.732, 0.350, 1427.0, 0.000577),
    ("dense", True, 0.743, 0.271, 4190.0, 0.000584),
    ("sparse", False, 0.719, 0.341, 1491.0, 0.000548),
    ("sparse", True, 0.665, 0.272, 4085.0, 0.000560),
    ("hybrid", False, 0.819, 0.340, 1713.0, 0.000582),
    ("hybrid", True, 0.722, 0.280, 4463.0, 0.000578),
)


def _stage1(
    *,
    strategy: str,
    mode: str,
    rerank: bool,
    recall: float | None,
    answerable: bool = True,
    total_ms: float = 10.0,
    question_id: str = "q1",
) -> QueryRow:
    gold = (("acc", "1A"),) if answerable else ()
    return QueryRow(
        strategy=strategy,
        mode=mode,
        rerank=rerank,
        question_id=question_id,
        question_type="single_hop" if answerable else "unanswerable",
        query="q",
        answerable=answerable,
        gold_sections=gold,
        retrieved_sections=gold,
        recall_at_1=recall,
        recall_at_5=recall,
        recall_at_10=recall,
        mrr=recall,
        ndcg_at_10=recall,
        encode_ms=0.0,
        dense_ms=0.0,
        sparse_ms=0.0,
        fuse_ms=0.0,
        rerank_ms=0.0,
        total_ms=total_ms,
    )


def _stage2(
    *,
    mode: str,
    rerank: bool,
    answerable: bool,
    refused: bool = False,
    faithfulness: float | None = None,
    context_precision: float | None = None,
    serving_ms: float = 80.0,
    usd: float = 0.0001,
    question_id: str = "q1",
    strategy: str = "fixed",
) -> RagQueryRow:
    precision = context_precision if context_precision is not None else faithfulness
    return RagQueryRow(
        strategy=strategy,
        mode=mode,
        rerank=rerank,
        question_id=question_id,
        question_type="single_hop" if answerable else "unanswerable",
        query="q",
        answerable=answerable,
        response="Not in the corpus." if refused else "ok",
        refused=refused,
        retrieved_sections=(("acc", "1A"),),
        faithfulness=faithfulness,
        context_precision=precision,
        context_recall=faithfulness,
        relevancy=faithfulness,
        judge_ms=1.0 if answerable else None,
        judge_usd=None,
        encode_ms=0.0,
        dense_ms=0.0,
        sparse_ms=0.0,
        fuse_ms=0.0,
        rerank_ms=0.0,
        total_ms=10.0,
        generate_ms=serving_ms - 10.0,
        serving_ms=serving_ms,
        prompt_tokens=1,
        completion_tokens=1,
        usd=usd,
    )


def _grid_stage1() -> tuple[QueryRow, ...]:
    rows = [
        _stage1(
            strategy="fixed",
            mode=mode,
            rerank=rerank,
            recall=recall,
            total_ms=total_ms,
        )
        for mode, rerank, recall, total_ms in _FIXED_STAGE1
    ]
    rows.append(
        _stage1(
            strategy="fixed",
            mode="hybrid",
            rerank=True,
            recall=None,
            answerable=False,
            total_ms=1189.0,
            question_id="ua-1",
        )
    )
    rows.append(
        _stage1(strategy="structural", mode="dense", rerank=False, recall=0.400, total_ms=40.0)
    )
    rows.append(
        _stage1(strategy="semantic", mode="dense", rerank=False, recall=0.200, total_ms=30.0)
    )
    return tuple(rows)


def _grid_stage2() -> tuple[RagQueryRow, ...]:
    rows: list[RagQueryRow] = []
    for mode, rerank, faith, ctx_p, serving_ms, usd in _FIXED_STAGE2:
        rows.append(
            _stage2(
                mode=mode,
                rerank=rerank,
                answerable=True,
                faithfulness=faith,
                context_precision=ctx_p,
                serving_ms=serving_ms,
                usd=usd,
            )
        )
        rows.append(
            _stage2(
                mode=mode,
                rerank=rerank,
                answerable=False,
                refused=True,
                serving_ms=serving_ms,
                usd=usd,
                question_id="ua-1",
            )
        )
    return tuple(rows)


def _readme(path: Path, *, markers: bool = True) -> Path:
    if markers:
        path.write_text(
            f"before\n{ABLATION_START}\nplaceholder\n{ABLATION_END}\nafter\n",
            encoding="utf-8",
        )
    else:
        path.write_text("no markers here\n", encoding="utf-8")
    return path


def test_load_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = _grid_stage1()
    path = tmp_path / "eval-retrieval.jsonl"
    write_jsonl(rows, path)
    loaded = load_jsonl(path)
    assert loaded[0].strategy == "fixed"
    assert loaded[0].recall_at_5 == pytest.approx(0.656)
    assert loaded[6].answerable is False
    assert loaded[6].recall_at_5 is None


def test_report_winner_gap_and_lift(tmp_path: Path) -> None:
    retrieval = tmp_path / "eval-retrieval.jsonl"
    rag = tmp_path / "eval-rag.jsonl"
    write_jsonl(_grid_stage1(), retrieval)
    write_rag_jsonl(_grid_stage2(), rag)
    readme = _readme(tmp_path / "README.md")
    output = tmp_path / "ablation.md"
    dest = run_report(
        retrieval_jsonl=retrieval,
        rag_jsonl=rag,
        output=output,
        readme=readme,
    )
    assert dest == output
    body = output.read_text(encoding="utf-8")
    injected = readme.read_text(encoding="utf-8")
    assert injected.startswith("before\n")
    assert injected.endswith("after\n")
    assert body.rstrip() in injected
    assert "Stage 1 retrieval (k=10)" in injected
    assert "Stage 2 generation (k=5)" in injected
    assert "fixed is the winning chunker at k=10" in injected
    assert "mean recall@5=0.704" in injected
    assert "dense+rerank is the best retrieval config (recall@5=0.789)" in injected
    assert "+0.056 vs hybrid+rerank (0.733)" in injected
    assert "Reranking alone, on dense: 0.656 → 0.789 (+0.133)" in injected
    assert "Fusion alone, no rerank: 0.656 → 0.722 (+0.066)" in injected
    assert "Reranking on top of fusion: 0.722 → 0.733 (+0.011)" in injected
    assert "hybrid with rerank off has the best faithfulness (0.819)" in injected
    assert "turning rerank on drops it to 0.722" in injected
    assert "nearly triples p50 (1713ms → 4463ms)" in injected
    assert "Context precision falls for every rerank-on config" in injected
    assert "Refusal rate is 1.000 on all 1 unanswerable question" in injected
    assert "generation cost is flat" in injected
    assert "fusion adds almost nothing" in injected
    assert "Stage 2 (k=5) disagrees" in injected
    assert "degrades answer faithfulness at serving k=5" in injected
    assert "not hybrid+rerank" in injected


def test_report_missing_stage2_uses_stage1_p95(tmp_path: Path) -> None:
    retrieval = tmp_path / "eval-retrieval.jsonl"
    write_jsonl(_grid_stage1(), retrieval)
    readme = _readme(tmp_path / "README.md")
    output = tmp_path / "ablation.md"
    run_report(
        retrieval_jsonl=retrieval,
        output=output,
        readme=readme,
        rag_jsonl=None,
        settings=_settings(tmp_path),
    )
    body = output.read_text(encoding="utf-8")
    assert "Stage 2 JSONL not found" in body
    injected = readme.read_text(encoding="utf-8")
    assert "Stage 1 retrieval (k=10)" in injected
    assert "dense+rerank is the best retrieval config (recall@5=0.789)" in injected
    assert "Stage 2 faithfulness is n/a" in injected
    assert "Refusal rate n/a" in injected
    assert "dollars/query n/a" in injected
    assert "Stage 2 generation" not in injected
    assert "disagrees" not in injected


def test_report_missing_markers_fails(tmp_path: Path) -> None:
    retrieval = tmp_path / "eval-retrieval.jsonl"
    write_jsonl(_grid_stage1(), retrieval)
    readme = _readme(tmp_path / "README.md", markers=False)
    with pytest.raises(ValueError, match="missing"):
        run_report(
            retrieval_jsonl=retrieval,
            output=tmp_path / "ablation.md",
            readme=readme,
            settings=_settings(tmp_path),
        )


def _settings(tmp_path: Path):
    from filing_rag.settings import Settings

    return Settings(results_dir=tmp_path)
