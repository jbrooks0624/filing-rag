"""Passage encoder. Lazy-imports sentence-transformers so tests stay torch-free."""

from __future__ import annotations

from typing import Any, Protocol

from filing_rag.chunking.config import ChunkingConfig

DEFAULT_BATCH_SIZE = 32


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbedError(RuntimeError):
    """Passage encoding could not run (missing extra, load failure, …)."""


class BgeEmbedder:
    """bge-base-en-v1.5 passage encoder. Never adds the query instruction prefix."""

    def __init__(self, model_name: str, *, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = None

    @classmethod
    def from_config(cls, config: ChunkingConfig) -> BgeEmbedder:
        return cls(config.semantic.encoder)

    def ensure_available(self) -> None:
        """Raise if sentence-transformers is not installed, without loading weights."""
        try:
            import sentence_transformers
        except ImportError as exc:
            raise EmbedError(
                "Indexing requires sentence-transformers. Install with: uv sync --dev"
            ) from exc
        version = getattr(sentence_transformers, "__version__", "0")
        if version.split(".", 1)[0] in {"0", "1", "2"}:
            raise EmbedError(
                f"sentence-transformers {version} cannot load bge models. "
                "Reinstall with: uv sync --dev"
            )

    def warm(self) -> None:
        """Download and load the encoder once."""
        self.ensure_available()
        self._load()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            # Passages only — the bge query prefix is added at retrieval time.
            vectors = model.encode(batch, normalize_embeddings=True)
            out.extend(list(map(float, vector)) for vector in vectors)
        return out

    def embed_query(self, text: str, *, prefix: str) -> list[float]:
        """Encode one query. Caller supplies the bge instruction prefix."""
        return self.embed([prefix + text])[0]

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbedError(
                    "Indexing requires sentence-transformers. "
                    "Install with: uv sync --dev"
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbedError(f"failed to load encoder {self.model_name}: {exc}") from exc
        return self._model
