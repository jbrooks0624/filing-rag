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
    serving_ms: float = 80.0,
    usd: float = 0.0001,
    question_id: str = "q1",
) -> RagQueryRow:
    return RagQueryRow(
        strategy="structural",
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
        context_precision=faithfulness,
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
    return (
        _stage1(strategy="fixed", mode="dense", rerank=False, recall=0.400, total_ms=20.0),
        _stage1(strategy="semantic", mode="dense", rerank=False, recall=0.200, total_ms=30.0),
        _stage1(
            strategy="structural",
            mode="dense",
            rerank=False,
            recall=0.800,
            total_ms=40.0,
        ),
        _stage1(
            strategy="structural",
            mode="hybrid",
            rerank=True,
            recall=1.000,
            total_ms=80.0,
        ),
        _stage1(
            strategy="structural",
            mode="hybrid",
            rerank=True,
            recall=None,
            answerable=False,
            total_ms=80.0,
            question_id="ua-1",
        ),
    )


def _grid_stage2() -> tuple[RagQueryRow, ...]:
    return (
        _stage2(
            mode="dense",
            rerank=False,
            answerable=True,
            faithfulness=0.700,
            serving_ms=80.0,
            usd=0.000080,
        ),
        _stage2(
            mode="dense",
            rerank=False,
            answerable=False,
            refused=True,
            serving_ms=80.0,
            usd=0.000080,
            question_id="ua-1",
        ),
        _stage2(
            mode="hybrid",
            rerank=True,
            answerable=True,
            faithfulness=0.900,
            serving_ms=200.0,
            usd=0.000100,
        ),
        _stage2(
            mode="hybrid",
            rerank=True,
            answerable=False,
            refused=True,
            serving_ms=200.0,
            usd=0.000100,
            question_id="ua-1",
        ),
    )


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
    assert loaded[0].recall_at_5 == pytest.approx(0.400)
    assert loaded[-1].answerable is False
    assert loaded[-1].recall_at_5 is None


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
    assert "Stage 1 retrieval (k=10)" in body
    assert "Stage 2 generation (k=5)" in body
    injected = readme.read_text(encoding="utf-8")
    assert injected.startswith("before\n")
    assert injected.endswith("after\n")
    assert "structural is the winning chunker at k=10" in injected
    assert "mean recall@5=0.900" in injected
    assert "+0.500 vs fixed" in injected
    assert "+0.700 vs semantic" in injected
    assert "hybrid+rerank recall@5=1.000" in injected
    assert "+0.200 lift vs dense with rerank off (0.800)" in injected
    assert "At k=5, hybrid+rerank p95=" in injected
    assert "$0.000100/query" in injected
    assert "faithfulness is 0.900 at k=5" in injected
    assert "refusal rate on that config is 1.000 (1 unanswerable)" in injected


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
    assert "At k=10, hybrid+rerank p95=80ms vs dense with rerank off p95=40ms" in injected
    assert "dollars/query n/a" in injected
    assert "faithfulness at k=5 is n/a" in injected


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
