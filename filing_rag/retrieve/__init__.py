"""Callable retrieval package."""

from filing_rag.retrieve.config import (
    Bm25Config,
    RerankConfig,
    RetrievalConfig,
    load_retrieval,
)
from filing_rag.retrieve.dense import search_dense
from filing_rag.retrieve.hybrid import rrf
from filing_rag.retrieve.pipeline import Retriever
from filing_rag.retrieve.query import BgeQueryEncoder, QueryEncoder
from filing_rag.retrieve.rerank import BgeReranker, IdentityReranker, Reranker, RerankError
from filing_rag.retrieve.sparse import SparseError, SparseIndex, search_sparse
from filing_rag.retrieve.types import (
    MODES,
    SNIPPET_LIMIT,
    Filters,
    Hit,
    RetrieveResult,
    RetrieveTimings,
    snippet,
)

__all__ = [
    "MODES",
    "SNIPPET_LIMIT",
    "BgeQueryEncoder",
    "BgeReranker",
    "Bm25Config",
    "Filters",
    "Hit",
    "IdentityReranker",
    "QueryEncoder",
    "RerankConfig",
    "RerankError",
    "Reranker",
    "RetrievalConfig",
    "RetrieveResult",
    "RetrieveTimings",
    "Retriever",
    "SparseError",
    "SparseIndex",
    "load_retrieval",
    "rrf",
    "search_dense",
    "search_sparse",
    "snippet",
]
