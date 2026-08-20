"""RAGAS collections adapter. Lazy-imports ragas so unit tests stay extra-free."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from filing_rag.generate.config import GenerationConfig
from filing_rag.settings import Settings, get_settings

INSTALL_HINT = "Stage 2 eval requires ragas. Install with: uv sync --dev"


class RagasError(RuntimeError):
    """RAGAS scoring could not run (missing extra, metric failure, …)."""


@dataclass(frozen=True)
class RagasScores:
    """Four collections metrics. ``relevancy`` is v0.4 AnswerRelevancy."""

    faithfulness: float
    context_precision: float
    context_recall: float
    relevancy: float
    judge_ms: float = 0.0


class RagasScorer(Protocol):
    def score(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: Sequence[str],
        reference: str,
    ) -> RagasScores: ...


class Metric(Protocol):
    def ascore(self, **kwargs: object) -> object: ...


class CollectionsRagasScorer:
    """Live path builds Faithfulness / ContextPrecision / ContextRecall / AnswerRelevancy.

    Tests inject ``metrics`` so this module never imports ragas under pytest.
    """

    def __init__(
        self,
        config: GenerationConfig,
        settings: Settings,
        *,
        metrics: Mapping[str, Metric] | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self._metrics = dict(metrics) if metrics is not None else None

    @classmethod
    def from_config(
        cls,
        *,
        settings: Settings | None = None,
        config: GenerationConfig | None = None,
        metrics: Mapping[str, Metric] | None = None,
    ) -> CollectionsRagasScorer:
        return cls(
            config or GenerationConfig(),
            settings or get_settings(),
            metrics=metrics,
        )

    def ensure_available(self) -> None:
        """Raise if ragas is not installed, without constructing metrics."""
        try:
            import ragas
        except ModuleNotFoundError as exc:
            if exc.name == "ragas":
                raise RagasError(INSTALL_HINT) from exc
            raise RagasError(
                f"ragas is installed but failed to import ({exc}). "
                "Reinstall with: uv sync --dev"
            ) from exc
        except ImportError as exc:
            raise RagasError(INSTALL_HINT) from exc
        del ragas

    def score(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: Sequence[str],
        reference: str,
    ) -> RagasScores:
        if not user_input.strip():
            raise ValueError("user_input must be non-empty")
        metrics = self._require_metrics()
        started = perf_counter()
        try:
            values = asyncio.run(
                _score_async(
                    metrics,
                    user_input=user_input,
                    response=response,
                    retrieved_contexts=list(retrieved_contexts),
                    reference=reference,
                )
            )
        except RagasError:
            raise
        except Exception as exc:
            raise RagasError(f"ragas scoring failed: {exc}") from exc
        return RagasScores(
            faithfulness=values["faithfulness"],
            context_precision=values["context_precision"],
            context_recall=values["context_recall"],
            relevancy=values["relevancy"],
            judge_ms=(perf_counter() - started) * 1000,
        )

    def _require_metrics(self) -> dict[str, Metric]:
        if self._metrics is not None:
            return self._metrics
        self.ensure_available()
        self._metrics = _build_metrics(self.config, self.settings)
        return self._metrics


def mean_ragas(rows: Sequence[RagasScores | None]) -> RagasScores | None:
    """Macro-average over scored rows. ``None`` entries (unanswerable) are skipped."""
    scored = [row for row in rows if row is not None]
    if not scored:
        return None
    n = len(scored)
    return RagasScores(
        faithfulness=sum(row.faithfulness for row in scored) / n,
        context_precision=sum(row.context_precision for row in scored) / n,
        context_recall=sum(row.context_recall for row in scored) / n,
        relevancy=sum(row.relevancy for row in scored) / n,
        judge_ms=sum(row.judge_ms for row in scored) / n,
    )


async def _score_async(
    metrics: Mapping[str, Metric],
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    reference: str,
) -> dict[str, float]:
    faithfulness = await _ascore(
        metrics["faithfulness"],
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
    )
    context_precision = await _ascore(
        metrics["context_precision"],
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    context_recall = await _ascore(
        metrics["context_recall"],
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    relevancy = await _ascore(
        metrics["relevancy"],
        user_input=user_input,
        response=response,
    )
    return {
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "relevancy": relevancy,
    }


async def _ascore(metric: Metric, **kwargs: object) -> float:
    result = metric.ascore(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return _metric_value(result)


def _metric_value(result: object) -> float:
    value = getattr(result, "value", result)
    if value is None:
        raise RagasError("ragas metric returned no value")
    return float(value)


def _build_metrics(config: GenerationConfig, settings: Settings) -> dict[str, Metric]:
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    from filing_rag.chunking.config import load_chunking
    from filing_rag.embed.encoder import BgeEmbedder

    client = AsyncOpenAI(
        api_key=settings.require_api_key(),
        base_url=settings.require_base_url(),
    )
    llm = llm_factory(config.judge_model, client=client)
    args = getattr(llm, "model_args", None)
    if not isinstance(args, dict):
        raise RagasError("ragas llm has no model_args to sanitize for Luna")
    config.apply_chat_kwargs(args)
    embedder = BgeEmbedder.from_config(load_chunking(settings.chunking_path))
    embedder.ensure_available()
    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "relevancy": AnswerRelevancy(llm=llm, embeddings=_bge_embeddings(embedder)),
    }


def _bge_embeddings(embedder: object) -> object:
    """Local bge as ragas 0.4 ``BaseRagasEmbedding`` (collections reject duck types)."""
    from ragas.embeddings.base import BaseRagasEmbedding

    class BgeEmbeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: object) -> list[float]:
            del kwargs
            return _embed(embedder, [text])[0]

        async def aembed_text(self, text: str, **kwargs: object) -> list[float]:
            return self.embed_text(text, **kwargs)

        def embed_texts(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            del kwargs
            return _embed(embedder, texts)

        async def aembed_texts(
            self, texts: list[str], **kwargs: object
        ) -> list[list[float]]:
            return self.embed_texts(texts, **kwargs)

    return BgeEmbeddings()


def _embed(embedder: object, texts: list[str]) -> list[list[float]]:
    embed = getattr(embedder, "embed")
    return embed(list(texts))
