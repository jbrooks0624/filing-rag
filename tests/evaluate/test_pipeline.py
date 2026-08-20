"""Evaluator grid + JSONL. Injected retriever; no Postgres, no torch, no results/."""

import json
from pathlib import Path

import pytest
import yaml
from filing_rag.evaluate.pipeline import Evaluator, eval_grid
from filing_rag.evaluate.store import format_summary, format_table, percentile
from filing_rag.evaluate.types import GoldenSet
from filing_rag.retrieve.types import Hit, RetrieveResult, RetrieveTimings
from filing_rag.settings import PROJECT_ROOT, Settings

ACCESSION = "0000950170-24-087843"
JPM_ACCESSION = "0000019617-24-000001"


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


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._n = 0
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
        self._n += 1
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
                encode_ms=float(self._n * 10),
                dense_ms=1.0,
                rerank_ms=80.0 if rerank else 0.0,
            ),
        )


def _evaluator(
    tmp_path: Path, retriever: FakeRetriever | None = None
) -> tuple[Evaluator, FakeRetriever]:
    fake = retriever or FakeRetriever()
    settings = Settings(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        golden_path=tmp_path / "golden.yaml",
        corpus_path=PROJECT_ROOT / "config" / "corpus.yaml",
        chunking_path=PROJECT_ROOT / "config" / "chunking.yaml",
        retrieval_path=PROJECT_ROOT / "config" / "retrieval.yaml",
    )
    return Evaluator(_golden(), settings, fake), fake


def test_eval_grid_default_is_eighteen() -> None:
    grid = eval_grid()
    assert len(grid) == 18
    assert grid[0].label == "fixed/dense/off"
    assert grid[-1].label == "semantic/hybrid/rerank"
    assert sum(1 for cell in grid if cell.rerank) == 9


def test_eval_grid_narrows_axes() -> None:
    grid = eval_grid(strategies=["fixed"], modes=["dense", "hybrid"], rerank=False)
    assert [(c.strategy, c.mode, c.rerank) for c in grid] == [
        ("fixed", "dense", False),
        ("fixed", "hybrid", False),
    ]


def test_eval_grid_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown strategies"):
        eval_grid(strategies=["recursive"])
    with pytest.raises(ValueError, match="unknown modes"):
        eval_grid(modes=["keyword"])
    with pytest.raises(ValueError, match="empty"):
        eval_grid(strategies=[], modes=["dense"])


def test_empty_golden_set_raises(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", results_dir=tmp_path / "results")
    evaluator = Evaluator(GoldenSet(), settings, FakeRetriever())
    with pytest.raises(ValueError, match="golden set is empty"):
        evaluator.run(output=tmp_path / "out.jsonl")


def test_run_writes_jsonl_and_copies_timings(tmp_path: Path) -> None:
    evaluator, fake = _evaluator(tmp_path)
    dest = tmp_path / "eval.jsonl"
    result = evaluator.run(
        strategies=["fixed", "structural"],
        modes=["dense"],
        rerank=None,
        output=dest,
    )
    assert result.n_configs == 4
    assert result.n_questions == 3
    assert result.n_unanswerable == 1
    assert len(result.rows) == 12
    assert dest.is_file()
    parsed = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]
    assert len(parsed) == 12
    answerable = [row for row in parsed if row["answerable"]]
    unanswerable = [row for row in parsed if not row["answerable"]]
    assert all(row["recall_at_5"] == 1.0 for row in answerable)
    assert all(row["mrr"] is None for row in unanswerable)
    assert all("encode_ms" in row and "total_ms" in row for row in parsed)
    first = parsed[0]
    assert first["gold_sections"] == [{"accession": ACCESSION, "item_code": "1A"}]
    assert first["encode_ms"] == 10.0
    assert first["total_ms"] == 11.0
    assert fake.calls[0]["k"] == 10
    assert fake.calls[0]["filters"] is None
    assert "results" not in str(dest)
    report = result.reports[0]
    assert report.scores is not None
    assert report.scores.n == 2
    assert report.n_unanswerable == 1
    assert report.scores.recall_at_5 == 1.0


def test_force_rebuilds_sparse_once_per_strategy(tmp_path: Path) -> None:
    evaluator, fake = _evaluator(tmp_path)
    evaluator.run(
        strategies=["fixed", "structural"],
        modes=["dense"],
        rerank=False,
        force=True,
        output=tmp_path / "out.jsonl",
    )
    forced = [call for call in fake.calls if call["force"]]
    assert [call["strategy"] for call in forced] == ["fixed", "structural"]
    assert len(fake.calls) == 6


def test_from_config_uses_injected_retriever(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path)
    fake = FakeRetriever()
    settings = Settings(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        golden_path=yaml_path,
        corpus_path=PROJECT_ROOT / "config" / "corpus.yaml",
        chunking_path=PROJECT_ROOT / "config" / "chunking.yaml",
        retrieval_path=PROJECT_ROOT / "config" / "retrieval.yaml",
    )
    evaluator = Evaluator.from_config(
        settings=settings,
        retriever=fake,
        parsed_dir=tmp_path / "parsed",
    )
    result = evaluator.run(
        strategies=["fixed"],
        modes=["sparse"],
        rerank=False,
        output=tmp_path / "out.jsonl",
    )
    assert result.output == tmp_path / "out.jsonl"
    assert len(fake.calls) == 3
    assert "configs=1 questions=2 skipped_unanswerable=1" in format_summary(result)
    table = format_table(result.reports)
    assert "fixed" in table
    assert "sparse" in table
    assert "recall@5=1.000" in table


def test_percentile_linear_interpolation() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 0.50) == 30.0
    assert percentile(values, 0.95) == pytest.approx(48.0)
    assert percentile([7.0], 0.95) == 7.0
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.5)
