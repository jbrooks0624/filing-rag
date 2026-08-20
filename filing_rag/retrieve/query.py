"""Query encoder. Adds the bge instruction prefix; passages stay unprefixed."""

from __future__ import annotations

from typing import Protocol

from filing_rag.chunking.config import load_chunking
from filing_rag.embed.encoder import BgeEmbedder
from filing_rag.retrieve.config import RetrievalConfig, load_retrieval


class QueryEncoder(Protocol):
    def encode(self, query: str) -> list[float]: ...


class QueryPassageEncoder(Protocol):
    def embed_query(self, text: str, *, prefix: str) -> list[float]: ...


class BgeQueryEncoder:
    """Wraps a passage embedder so retrieval never encodes the raw query."""

    def __init__(self, embedder: QueryPassageEncoder, prefix: str) -> None:
        self.embedder = embedder
        self.prefix = prefix

    @classmethod
    def from_config(
        cls,
        retrieval: RetrievalConfig | None = None,
        *,
        embedder: QueryPassageEncoder | None = None,
    ) -> BgeQueryEncoder:
        config = retrieval if retrieval is not None else load_retrieval()
        resolved = embedder if embedder is not None else BgeEmbedder.from_config(load_chunking())
        return cls(resolved, config.query_prefix)

    def encode(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        return self.embedder.embed_query(query, prefix=self.prefix)
