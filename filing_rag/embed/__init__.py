"""Callable embedding / vector-store package."""

from filing_rag.embed.encoder import BgeEmbedder, Embedder, EmbedError
from filing_rag.embed.pipeline import Indexer, IndexingError, IndexResult
from filing_rag.embed.store import (
    EMBEDDING_DIM,
    HNSW_INDEXES,
    ChunkRow,
    ChunkStore,
    IndexStat,
    MemoryChunkStore,
    PostgresChunkStore,
    SearchRow,
    StoreError,
    dense_search_sql,
    hnsw_statement,
)

__all__ = [
    "EMBEDDING_DIM",
    "HNSW_INDEXES",
    "BgeEmbedder",
    "ChunkRow",
    "ChunkStore",
    "EmbedError",
    "Embedder",
    "IndexResult",
    "IndexStat",
    "Indexer",
    "IndexingError",
    "MemoryChunkStore",
    "PostgresChunkStore",
    "SearchRow",
    "StoreError",
    "dense_search_sql",
    "hnsw_statement",
]
