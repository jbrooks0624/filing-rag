"""Token counting for chunk windows. Production uses the bge tokenizer; tests inject a stand-in."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from filing_rag.chunking.config import ChunkingConfig

WORD = re.compile(r"\S+")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def offsets(self, text: str) -> Sequence[tuple[int, int]]: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


def truncate_to_tokens(text: str, max_tokens: int, offsets: Sequence[tuple[int, int]]) -> str:
    """Keep the original prefix that covers at most `max_tokens` tokens."""
    if max_tokens <= 0 or not offsets:
        return ""
    if len(offsets) <= max_tokens:
        return text
    return text[: offsets[max_tokens - 1][1]]


class WhitespaceTokenCounter:
    """One token per whitespace-separated word. Used in tests; not the bge tokenizer."""

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in WORD.finditer(text)]

    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        return truncate_to_tokens(text, max_tokens, self.offsets(text))


class HuggingFaceTokenCounter:
    """Wraps a HuggingFace `tokenizers.Tokenizer` (no torch)."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        tokenizer.no_truncation()
        tokenizer.no_padding()
        tokenizer.post_processor = None
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        filename: str = "tokenizer.json",
    ) -> HuggingFaceTokenCounter:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        return cls(Tokenizer.from_file(path))

    @classmethod
    def from_config(cls, config: ChunkingConfig) -> HuggingFaceTokenCounter:
        return cls.from_pretrained(config.tokenizer)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoding = self._tokenizer.encode(text)
        return [(start, end) for start, end in encoding.offsets if end > start]

    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        return truncate_to_tokens(text, max_tokens, self.offsets(text))
