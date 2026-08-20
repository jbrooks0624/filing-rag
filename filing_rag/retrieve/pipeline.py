"""Orchestrate dense / sparse / hybrid retrieval with optional rerank."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Literal

from filing_rag.embed.encoder import BgeEmbedder
from filing_rag.embed.store import ChunkStore, PostgresChunkStore, require_k, require_strategy
from filing_rag.retrieve.config import RetrievalConfig, load_retrieval
from filing_rag.retrieve.dense import search_dense
from filing_rag.retrieve.hybrid import rrf
from filing_rag.retrieve.query import BgeQueryEncoder, QueryEncoder
from filing_rag.retrieve.rerank import BgeReranker, Reranker
from filing_rag.retrieve.sparse import SparseIndex
from filing_rag.retrieve.types import (
    MODES,
    Filters,
    Hit,
    RetrieveResult,
    RetrieveTimings,
)
from filing_rag.settings import Settings, get_settings

Mode = Literal["dense", "sparse", "hybrid"]


class Retriever:
    """Callable retrieval pipeline. Tests inject store, encoder, sparse index, reranker."""

    def __init__(
        self,
        retrieval: RetrievalConfig,
        settings: Settings,
        store: ChunkStore,
        *,
        encoder: QueryEncoder | None = None,
        reranker: Reranker | None = None,
        sparse: SparseIndex | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.settings = settings
        self.store = store
        self._encoder = encoder
        self._reranker = reranker
        self._sparse = sparse
        self._sparse_cache: dict[str, SparseIndex] = {}

    @classmethod
    def from_config(
        cls,
        retrieval_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        store: ChunkStore | None = None,
        encoder: QueryEncoder | None = None,
        reranker: Reranker | None = None,
        sparse: SparseIndex | None = None,
    ) -> Retriever:
        resolved = settings or get_settings()
        path = Path(retrieval_path) if retrieval_path is not None else resolved.retrieval_path
        return cls(
            load_retrieval(path),
            resolved,
            store if store is not None else PostgresChunkStore.from_settings(resolved),
            encoder=encoder,
            reranker=reranker,
            sparse=sparse,
        )

    def search(
        self,
        query: str,
        *,
        strategy: str,
        mode: str = "hybrid",
        k: int | None = None,
        rerank: bool = False,
        filters: Filters | None = None,
        force: bool = False,
    ) -> RetrieveResult:
        if not query.strip():
            raise ValueError("query must be non-empty")
        strategy = require_strategy(strategy.strip().lower())
        resolved_mode = self._resolve_mode(mode)
        top_k = require_k(k if k is not None else self.retrieval.k)
        pool = self.retrieval.candidate_k if rerank or resolved_mode == "hybrid" else top_k
        scope = filters or Filters()
        encode_ms = dense_ms = sparse_ms = fuse_ms = rerank_ms = 0.0
        dense_hits: list[Hit] = []
        sparse_hits: list[Hit] = []

        if resolved_mode in {"dense", "hybrid"}:
            started = perf_counter()
            vector = self._encode(query)
            encode_ms = (perf_counter() - started) * 1000
            started = perf_counter()
            dense_hits = search_dense(
                self.store,
                vector,
                strategy=strategy,
                k=pool,
                filters=scope,
            )
            dense_ms = (perf_counter() - started) * 1000

        if resolved_mode in {"sparse", "hybrid"}:
            started = perf_counter()
            index = self._sparse_index(strategy, force=force)
            sparse_hits = index.search(query, k=pool, filters=scope)
            sparse_ms = (perf_counter() - started) * 1000

        if resolved_mode == "dense":
            hits = dense_hits
        elif resolved_mode == "sparse":
            hits = sparse_hits
        else:
            started = perf_counter()
            fuse_k = pool if rerank else top_k
            hits = rrf(
                [dense_hits, sparse_hits],
                rrf_k=self.retrieval.rrf_k,
                k=fuse_k,
            )
            fuse_ms = (perf_counter() - started) * 1000

        if rerank:
            started = perf_counter()
            hits = self._rerank(query, hits, k=top_k)
            rerank_ms = (perf_counter() - started) * 1000
        else:
            hits = hits[:top_k]

        return RetrieveResult(
            hits=tuple(hits),
            mode=resolved_mode,
            strategy=strategy,
            reranked=rerank,
            timings=RetrieveTimings(
                encode_ms=encode_ms,
                dense_ms=dense_ms,
                sparse_ms=sparse_ms,
                fuse_ms=fuse_ms,
                rerank_ms=rerank_ms,
            ),
        )

    def _resolve_mode(self, mode: str) -> Mode:
        name = mode.strip().lower()
        known: dict[str, Mode] = {item: item for item in MODES}
        resolved = known.get(name)
        if resolved is None:
            raise ValueError(f"unknown mode {mode!r}. Known: {', '.join(MODES)}")
        return resolved

    def _encode(self, query: str) -> list[float]:
        encoder = self._encoder
        if encoder is None:
            encoder = BgeQueryEncoder.from_config(self.retrieval)
            embedder = encoder.embedder
            if isinstance(embedder, BgeEmbedder):
                embedder.ensure_available()
                embedder.warm()
            self._encoder = encoder
        return encoder.encode(query)

    def _rerank(self, query: str, hits: Sequence[Hit], *, k: int) -> list[Hit]:
        reranker = self._reranker
        if reranker is None:
            live = BgeReranker.from_config(self.retrieval)
            live.ensure_available()
            live.warm()
            reranker = live
            self._reranker = reranker
        return reranker.rerank(query, hits, k=k)

    def _sparse_index(self, strategy: str, *, force: bool) -> SparseIndex:
        if self._sparse is not None:
            return self._sparse
        if not force and strategy in self._sparse_cache:
            return self._sparse_cache[strategy]
        index = SparseIndex.load_or_build(
            self.settings.indexes_dir / strategy,
            list(self.store.iter_chunks(strategy)),
            k1=self.retrieval.bm25.k1,
            b=self.retrieval.bm25.b,
            force=force,
        )
        self._sparse_cache[strategy] = index
        return index
