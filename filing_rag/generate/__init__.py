"""Citation-forcing generation over retrieved hits."""

from filing_rag.generate.client import (
    ChatClient,
    GenerateError,
    OpenAIResponsesClient,
    StreamClient,
)
from filing_rag.generate.config import (
    LUNA_INPUT_PER_MILLION,
    LUNA_OUTPUT_PER_MILLION,
    GenerationConfig,
)
from filing_rag.generate.pipeline import Generator, Searcher
from filing_rag.generate.prompt import (
    CITATION_EXAMPLE,
    CITATION_FORMAT,
    build_messages,
    format_contexts,
    system_prompt,
    user_prompt,
)
from filing_rag.generate.types import (
    AskResult,
    CitationBlock,
    CitationEvent,
    DoneEvent,
    GenerateResult,
    GenerateTimings,
    StreamCitation,
    StreamEvent,
    TokenEvent,
    Usage,
    blocks_from_hits,
)

__all__ = [
    "CITATION_EXAMPLE",
    "CITATION_FORMAT",
    "LUNA_INPUT_PER_MILLION",
    "LUNA_OUTPUT_PER_MILLION",
    "AskResult",
    "ChatClient",
    "CitationBlock",
    "CitationEvent",
    "DoneEvent",
    "GenerateError",
    "GenerateResult",
    "GenerateTimings",
    "GenerationConfig",
    "Generator",
    "OpenAIResponsesClient",
    "Searcher",
    "StreamCitation",
    "StreamClient",
    "StreamEvent",
    "TokenEvent",
    "Usage",
    "blocks_from_hits",
    "build_messages",
    "format_contexts",
    "system_prompt",
    "user_prompt",
]
