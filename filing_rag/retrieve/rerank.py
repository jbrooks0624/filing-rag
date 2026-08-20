"""Cross-encoder rerank. Lazy-imports sentence-transformers so tests stay torch-free."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol

from filing_rag.embed.store import require_k
from filing_rag.retrieve.config import RetrievalConfig, load_retrieval
from filing_rag.retrieve.types import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit], *, k: int) -> list[Hit]: ...


class RerankError(RuntimeError):
    """Reranking could not run (missing extra, load failure, …)."""


class IdentityReranker:
    """Rerank-off. Keeps incoming order, cuts to k, rewrites ranks."""

    def rerank(self, query: str, hits: Sequence[Hit], *, k: int) -> list[Hit]:
        del query
        k = require_k(k)
        return [replace(hit, rank=rank) for rank, hit in enumerate(hits[:k], start=1)]


class BgeReranker:
    """bge-reranker-base. Scores (query, hit.text); never adds the dense query prefix."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    @classmethod
    def from_config(cls, retrieval: RetrievalConfig | None = None) -> BgeReranker:
        config = retrieval if retrieval is not None else load_retrieval()
        return cls(config.rerank.model)

    def ensure_available(self) -> None:
        try:
            import sentence_transformers
        except ImportError as exc:
            raise RerankError(
                "Reranking requires sentence-transformers. "
                "Install with: uv sync --dev"
            ) from exc
        version = getattr(sentence_transformers, "__version__", "0")
        if version.split(".", 1)[0] in {"0", "1", "2"}:
            raise RerankError(
                f"sentence-transformers {version} cannot load bge models. "
                "Reinstall with: uv sync --dev"
            )

    def warm(self) -> None:
        self.ensure_available()
        self._load()

    def rerank(self, query: str, hits: Sequence[Hit], *, k: int) -> list[Hit]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        k = require_k(k)
        if not hits:
            return []
        pairs = [(query, hit.text) for hit in hits]
        raw = self._load().predict(pairs)
        scored = [
            replace(hit, score=float(score))
            for hit, score in zip(hits, raw, strict=True)
        ]
        scored.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return [replace(hit, rank=rank) for rank, hit in enumerate(scored[:k], start=1)]

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankError(
                    "Reranking requires sentence-transformers. "
                    "Install with: uv sync --dev"
                ) from exc
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                raise RerankError(
                    f"failed to load reranker {self.model_name}: {exc}"
                ) from exc
        return self._model
