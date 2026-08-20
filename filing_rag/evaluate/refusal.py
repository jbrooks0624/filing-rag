"""Locked-phrase refusal. No LLM judge."""

from __future__ import annotations

import string
from collections.abc import Sequence

from filing_rag.generate.config import GenerationConfig

_PUNCT = str.maketrans("", "", string.punctuation)


def is_refusal(text: str, phrase: str | None = None) -> bool:
    """True iff the locked phrase appears, case-insensitive, ignoring punctuation."""
    needle = _normalize(phrase if phrase is not None else GenerationConfig().refusal_phrase)
    if not needle:
        return False
    return needle in _normalize(text)


def refusal_rate(refused: Sequence[bool]) -> float:
    """Fraction of True values. Undefined when ``refused`` is empty."""
    if not refused:
        raise ValueError("refusal rate is undefined for empty values")
    return sum(1 for item in refused if item) / len(refused)


def _normalize(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCT).split())
