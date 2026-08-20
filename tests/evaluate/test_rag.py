"""RagEvaluator grid + JSONL. Injected retriever/generator/scorer; no API, no ragas."""

import json
from pathlib import Path

import pytest
import yaml
from filing_rag.evaluate.rag import RagEvaluator, winning_strategy
from filing_rag.evaluate.ragas import RagasScores
from filing_rag.evaluate.store import format_rag_summary, format_rag_table
from filing_rag.evaluate.types import GoldenSet
from filing_rag.generate.types import GenerateResult, GenerateTimings, Usage
from filing_rag.retrieve.types import Hit, RetrieveResult, RetrieveTimings
from filing_rag.settings import PROJECT_ROOT, Settings

ACCESSION = "0000950170-24-087843"
JPM_ACCESSION = "0000019617-24-000001"
REFUSAL = "Not in the corpus."


def _hit(accession: str, item_code: str, rank: int) -> Hit:
    return Hit(
        chunk_id=rank,
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


def _questions() -> list[dict]:
    return [
        {
            "id": "sh-001",
            "type": "single_hop",
            "query": "What cybersecurity risks does Microsoft disclose?",
            "answerable": True,
            "answer": "Cybersecurity risk.",
            "citations": [
                {"accession": ACCESSION, "item_code": "1A", "quote": "Cyber."},
            ],
        },
        {
            "id": "ws-001",
            "type": "within_sector",
            "query": "How do Microsoft and JPMorgan describe interest-rate risk?",
            "answerable": True,
            "answer": "Both disclose rate risk.",
            "citations": [
                {"accession": ACCESSION, "item_code": "7A", "quote": "Rates."},
                {"accession": JPM_ACCESSION, "item_code": "7A", "quote": "Market."},
            ],
        },
        {
            "id": "ua-001",
            "type": "unanswerable",
            "query": "What did NVDA disclose about GPU supply?",
            "answerable": False,
            "answer": "",
            "citations": [],
        },
    ]


def _golden() -> GoldenSet:
    return GoldenSet.model_validate({"questions": _questions()})


def _write_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(yaml.safe_dump({"questions": _questions()}), encoding="utf-8")
    return path


def _stage1_row(
    strategy: str,
    *,
    answerable: bool = True,
    recall_at_5: float | None = 1.0,
    ndcg_at_10: float | None = 0.8,
    mrr: float | None = 0.7,
    question_id: str = "sh-001",
) -> dict:
    return {
        "strategy": strategy,
        "mode": "dense",
        "rerank": False,
        "question_id": question_id,
        "answerable": answerable,
        "recall_at_5": recall_at_5,
        "ndcg_at_10": ndcg_at_10,
        "mrr": mrr,
    }


def _write_stage1(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.hits = {
            "What cybersecurity risks does Microsoft disclose?": (
                _hit(ACCESSION, "1A", 1),
            ),
            "How do Microsoft and JPMorgan describe interest-rate risk?": (
                _hit(ACCESSION, "7A", 1),
                _hit(JPM_ACCESSION, "7A", 2),
            ),
            "What did NVDA disclose about GPU supply?": (_hit(ACCESSION, "1A", 1),),
        }

    def search(
        self,
        query: str,
        *,
        strategy: str,
        mode: str = "hybrid",
        k: int | None = None,
        rerank: bool = False,
        filters=None,
        force: bool = False,
    ) -> RetrieveResult:
        self.calls.append(
            {
                "query": query,
                "strategy": strategy,
                "mode": mode,
                "k": k,
                "rerank": rerank,
                "filters": filters,
                "force": force,
            }
        )
        return RetrieveResult(
            hits=self.hits[query],
            mode=mode,
            strategy=strategy,
            reranked=rerank,
            timings=RetrieveTimings(
                encode_ms=10.0,
                dense_ms=4.0,
                rerank_ms=80.0 if rerank else 0.0,
            ),
        )


class FakeGenerator:
    def __init__(self, *, refuse_answerable: bool = False) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.refuse_answerable = refuse_answerable

    def generate(self, query: str, hits) -> GenerateResult:
        self.calls.append((query, tuple(hits)))
        if "NVDA" in query or self.refuse_answerable:
            text = REFUSAL
        else:
            text = f"Answer for {query}"
        return GenerateResult(
            text=text,
            usage=Usage(prompt_tokens=100, completion_tokens=20),
            usd=0.0001,
            timings=GenerateTimings(generate_ms=50.0),
            model="gpt-5.6-luna",
        )


class FakeRagas:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def score(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str,
    ) -> RagasScores:
        self.calls.append(
            {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": list(retrieved_contexts),
                "reference": reference,
            }
        )
        return RagasScores(
            faithfulness=0.9,
            context_precision=0.8,
            context_recall=0.7,
            relevancy=0.6,
            judge_ms=12.0,
        )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        golden_path=tmp_path / "golden.yaml",
        corpus_path=PROJECT_ROOT / "config" / "corpus.yaml",
        chunking_path=PROJECT_ROOT / "config" / "chunking.yaml",
        retrieval_path=PROJECT_ROOT / "config" / "retrieval.yaml",
    )


def _evaluator(
    tmp_path: Path,
    *,
    refuse_answerable: bool = False,
) -> tuple[RagEvaluator, FakeRetriever, FakeGenerator, FakeRagas]:
    retriever = FakeRetriever()
    generator = FakeGenerator(refuse_answerable=refuse_answerable)
    scorer = FakeRagas()
    evaluator = RagEvaluator(
        _golden(),
        _settings(tmp_path),
        retriever,
        generator,
        scorer,
    )
    return evaluator, retriever, generator, scorer


def test_winning_strategy_picks_highest_recall_at_5(tmp_path: Path) -> None:
    path = _write_stage1(
        tmp_path / "eval.jsonl",
        [
            _stage1_row("fixed", recall_at_5=0.5, ndcg_at_10=0.9, mrr=0.9),
            _stage1_row("structural", recall_at_5=0.8, ndcg_at_10=0.1, mrr=0.1),
            _stage1_row("fixed", answerable=False, recall_at_5=None),
        ],
    )
    assert winning_strategy(path) == "structural"


def test_winning_strategy_tie_breaks_ndcg_then_mrr(tmp_path: Path) -> None:
    path = _write_stage1(
        tmp_path / "eval.jsonl",
        [
            _stage1_row("fixed", recall_at_5=1.0, ndcg_at_10=0.5, mrr=0.9),
            _stage1_row("structural", recall_at_5=1.0, ndcg_at_10=0.8, mrr=0.1),
        ],
    )
    assert winning_strategy(path) == "structural"
    path = _write_stage1(
        tmp_path / "eval.jsonl",
        [
            _stage1_row("fixed", recall_at_5=1.0, ndcg_at_10=0.8, mrr=0.2),
            _stage1_row("semantic", recall_at_5=1.0, ndcg_at_10=0.8, mrr=0.9),
        ],
    )
    assert winning_strategy(path) == "semantic"


def test_winning_strategy_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="Pass --strategy or run eval-retrieval"):
        winning_strategy(missing)


def test_winning_strategy_only_unanswerable_raises(tmp_path: Path) -> None:
    path = _write_stage1(
        tmp_path / "eval.jsonl",
        [_stage1_row("fixed", answerable=False, recall_at_5=None)],
    )
    with pytest.raises(ValueError, match="no answerable rows"):
        winning_strategy(path)


def test_empty_golden_set_raises(tmp_path: Path) -> None:
    evaluator = RagEvaluator(
        GoldenSet(),
        _settings(tmp_path),
        FakeRetriever(),
        FakeGenerator(),
        FakeRagas(),
    )
    with pytest.raises(ValueError, match="golden set is empty"):
        evaluator.run(strategy="fixed", output=tmp_path / "out.jsonl")


def test_run_writes_jsonl_skips_ragas_on_unanswerable(tmp_path: Path) -> None:
    evaluator, retriever, generator, scorer = _evaluator(tmp_path)
    dest = tmp_path / "eval-rag.jsonl"
    result = evaluator.run(
        strategy="fixed",
        modes=["dense"],
        rerank=None,
        output=dest,
    )
    assert result.strategy == "fixed"
    assert result.n_configs == 2
    assert result.n_questions == 3
    assert result.n_unanswerable == 1
    assert result.refusal_rate == pytest.approx(1.0)
    assert len(result.rows) == 6
    assert dest.is_file()
    parsed = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]
    assert len(parsed) == 6
    answerable = [row for row in parsed if row["answerable"]]
    unanswerable = [row for row in parsed if not row["answerable"]]
    assert all(row["faithfulness"] == 0.9 for row in answerable)
    assert all(row["faithfulness"] is None for row in unanswerable)
    assert all(row["refused"] is True for row in unanswerable)
    assert all(row["refused"] is False for row in answerable)
    assert all(row["usd"] == 0.0001 for row in parsed)
    assert all(row["serving_ms"] == row["total_ms"] + row["generate_ms"] for row in parsed)
    assert parsed[0]["generate_ms"] == 50.0
    assert parsed[0]["prompt_tokens"] == 100
    assert parsed[0]["judge_usd"] is None
    assert retriever.calls[0]["k"] is None
    assert retriever.calls[0]["filters"] is None
    assert len(scorer.calls) == 4
    assert all("NVDA" not in call["user_input"] for call in scorer.calls)
    assert len(generator.calls) == 6
    report = result.reports[0]
    assert report.ragas is not None
    assert report.ragas.faithfulness == pytest.approx(0.9)
    assert report.usd == pytest.approx(0.0001)
    assert report.n_unanswerable == 1
    assert report.refusal_rate == pytest.approx(1.0)
    assert report.p50_ms == pytest.approx(64.0)
    assert "results" not in str(dest)


def test_answerable_refusal_still_runs_ragas(tmp_path: Path) -> None:
    evaluator, _retriever, _generator, scorer = _evaluator(
        tmp_path, refuse_answerable=True
    )
    result = evaluator.run(
        strategy="structural",
        modes=["hybrid"],
        rerank=False,
        output=tmp_path / "out.jsonl",
    )
    answerable = [row for row in result.rows if row.answerable]
    assert all(row.refused for row in result.rows)
    assert all(row.faithfulness == 0.9 for row in answerable)
    assert len(scorer.calls) == 2
    assert all(call["response"] == REFUSAL for call in scorer.calls)


def test_run_picks_winner_from_stage1_jsonl(tmp_path: Path) -> None:
    stage1 = _write_stage1(
        tmp_path / "eval-retrieval.jsonl",
        [
            _stage1_row("fixed", recall_at_5=0.4),
            _stage1_row("semantic", recall_at_5=0.9),
        ],
    )
    evaluator, retriever, _generator, _scorer = _evaluator(tmp_path)
    result = evaluator.run(
        modes=["sparse"],
        rerank=False,
        retrieval_jsonl=stage1,
        output=tmp_path / "out.jsonl",
    )
    assert result.strategy == "semantic"
    assert {call["strategy"] for call in retriever.calls} == {"semantic"}


def test_force_rebuilds_sparse_once_per_strategy(tmp_path: Path) -> None:
    evaluator, retriever, _generator, _scorer = _evaluator(tmp_path)
    evaluator.run(
        strategy="fixed",
        modes=["dense", "hybrid"],
        rerank=False,
        force=True,
        output=tmp_path / "out.jsonl",
    )
    forced = [call for call in retriever.calls if call["force"]]
    assert [call["strategy"] for call in forced] == ["fixed"]
    assert len(retriever.calls) == 6


def test_k_is_passed_through(tmp_path: Path) -> None:
    evaluator, retriever, _generator, _scorer = _evaluator(tmp_path)
    evaluator.run(
        strategy="fixed",
        modes=["dense"],
        rerank=False,
        k=5,
        output=tmp_path / "out.jsonl",
    )
    assert all(call["k"] == 5 for call in retriever.calls)


def test_from_config_uses_injected_backends(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path)
    retriever = FakeRetriever()
    generator = FakeGenerator()
    scorer = FakeRagas()
    settings = Settings(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        golden_path=yaml_path,
        corpus_path=PROJECT_ROOT / "config" / "corpus.yaml",
        chunking_path=PROJECT_ROOT / "config" / "chunking.yaml",
        retrieval_path=PROJECT_ROOT / "config" / "retrieval.yaml",
    )
    evaluator = RagEvaluator.from_config(
        settings=settings,
        retriever=retriever,
        generator=generator,
        scorer=scorer,
        parsed_dir=tmp_path / "parsed",
    )
    result = evaluator.run(
        strategy="fixed",
        modes=["sparse"],
        rerank=False,
        output=tmp_path / "out.jsonl",
    )
    assert result.output == tmp_path / "out.jsonl"
    assert len(retriever.calls) == 3
    summary = format_rag_summary(result)
    assert "strategy=fixed configs=1 questions=2 skipped_unanswerable=1" in summary
    assert "refusal=1.000" in summary
    table = format_rag_table(result.reports)
    assert "sparse" in table
    assert "faith=0.900" in table
    assert "usd=0.000100" in table


def test_missing_strategy_without_stage1_raises(tmp_path: Path) -> None:
    evaluator, _retriever, _generator, _scorer = _evaluator(tmp_path)
    with pytest.raises(ValueError, match="Pass --strategy or run eval-retrieval"):
        evaluator.run(output=tmp_path / "out.jsonl")
