"""Callable chunking package."""

from filing_rag.chunking.cap import (
    TokenSpan,
    cap_from_config,
    cap_spans,
    enforce_cap,
    window_split,
)
from filing_rag.chunking.config import (
    STRATEGIES,
    ChunkingConfig,
    FixedConfig,
    SemanticConfig,
    StructuralConfig,
    load_chunking,
)
from filing_rag.chunking.fixed import chunk_filing, chunk_section
from filing_rag.chunking.pipeline import Chunker, ChunkError, ChunkResult
from filing_rag.chunking.semantic import (
    BgeSentenceEmbedder,
    Embedder,
    SemanticChunkError,
)
from filing_rag.chunking.store import (
    chunked_exists,
    chunked_path,
    iter_parsed,
    load_chunked,
    load_parsed,
    parsed_path,
    write_chunked,
)
from filing_rag.chunking.tokenize import (
    HuggingFaceTokenCounter,
    TokenCounter,
    WhitespaceTokenCounter,
)
from filing_rag.chunking.types import Chunk, ChunkedFiling

__all__ = [
    "STRATEGIES",
    "BgeSentenceEmbedder",
    "Chunk",
    "ChunkedFiling",
    "Chunker",
    "ChunkError",
    "ChunkResult",
    "ChunkingConfig",
    "Embedder",
    "FixedConfig",
    "HuggingFaceTokenCounter",
    "SemanticChunkError",
    "SemanticConfig",
    "StructuralConfig",
    "TokenCounter",
    "TokenSpan",
    "WhitespaceTokenCounter",
    "cap_from_config",
    "cap_spans",
    "chunk_filing",
    "chunk_section",
    "chunked_exists",
    "chunked_path",
    "enforce_cap",
    "iter_parsed",
    "load_chunked",
    "load_chunking",
    "load_parsed",
    "parsed_path",
    "window_split",
    "write_chunked",
]
