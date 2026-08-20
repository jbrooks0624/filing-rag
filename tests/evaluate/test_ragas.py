"""RAGAS adapter with injected metrics. Never imports ragas or torch."""

from types import SimpleNamespace

import pytest
from filing_rag.evaluate.ragas import (
    CollectionsRagasScorer,
    RagasError,
    RagasScores,
    _bge_embeddings,
    mean_ragas,
)
from filing_rag.generate.config import GenerationConfig
from filing_rag.settings import Settings
from ragas.embeddings.base import BaseRagasEmbedding


class FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value
        self.kwargs: dict | None = None

    async def ascore(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(value=self.value)


def _scorer() -> tuple[CollectionsRagasScorer, dict[str, FakeMetric]]:
    metrics = {
        "faithfulness": FakeMetric(0.9),
        "context_precision": FakeMetric(0.8),
        "context_recall": FakeMetric(0.7),
        "relevancy": FakeMetric(0.6),
    }
    scorer = CollectionsRagasScorer(
        GenerationConfig(),
        Settings(openai_api_key="sk-test"),
        metrics=metrics,
    )
    return scorer, metrics


def test_score_runs_injected_metrics() -> None:
    scorer, metrics = _scorer()
    scores = scorer.score(
        user_input="How fast did Azure grow?",
        response="Azure grew 30%. [MSFT FY2024 Item 7]",
        retrieved_contexts=["Azure and other cloud services growth of 30%."],
        reference="Azure grew 30%.",
    )
    assert scores == RagasScores(
        faithfulness=0.9,
        context_precision=0.8,
        context_recall=0.7,
        relevancy=0.6,
        judge_ms=scores.judge_ms,
    )
    assert scores.judge_ms >= 0
    assert metrics["faithfulness"].kwargs == {
        "user_input": "How fast did Azure grow?",
        "response": "Azure grew 30%. [MSFT FY2024 Item 7]",
        "retrieved_contexts": ["Azure and other cloud services growth of 30%."],
    }
    assert metrics["context_precision"].kwargs is not None
    assert metrics["context_precision"].kwargs["reference"] == "Azure grew 30%."
    assert metrics["relevancy"].kwargs == {
        "user_input": "How fast did Azure grow?",
        "response": "Azure grew 30%. [MSFT FY2024 Item 7]",
    }


def test_empty_user_input_raises() -> None:
    scorer, _metrics = _scorer()
    with pytest.raises(ValueError, match="user_input"):
        scorer.score(
            user_input="  ",
            response="x",
            retrieved_contexts=[],
            reference="y",
        )


def test_mean_ragas_skips_unanswerable_nones() -> None:
    answerable = RagasScores(
        faithfulness=1.0,
        context_precision=0.5,
        context_recall=0.25,
        relevancy=0.0,
        judge_ms=10.0,
    )
    mean = mean_ragas([answerable, None, answerable])
    assert mean is not None
    assert mean.faithfulness == pytest.approx(1.0)
    assert mean.context_precision == pytest.approx(0.5)
    assert mean.context_recall == pytest.approx(0.25)
    assert mean.relevancy == pytest.approx(0.0)
    assert mean.judge_ms == pytest.approx(10.0)


def test_mean_ragas_all_unanswerable_is_none() -> None:
    assert mean_ragas([None, None]) is None


def test_bge_embeddings_is_modern_base() -> None:
    class FakeEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text)), 0.0] for text in texts]

    embeddings = _bge_embeddings(FakeEmbedder())
    assert isinstance(embeddings, BaseRagasEmbedding)
    assert embeddings.embed_text("azure") == [5.0, 0.0]
    assert embeddings.embed_texts(["a", "bb"]) == [[1.0, 0.0], [2.0, 0.0]]


def test_ensure_available_without_ragas() -> None:
    try:
        import ragas
    except ImportError:
        scorer = CollectionsRagasScorer(GenerationConfig(), Settings(openai_api_key="sk-test"))
        with pytest.raises(RagasError, match="uv sync --dev"):
            scorer.ensure_available()
        return
    del ragas
    pytest.skip("ragas is installed")
