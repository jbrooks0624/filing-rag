"""Locked-phrase refusal. No LLM, no ragas."""

import pytest
from filing_rag.evaluate.refusal import is_refusal, refusal_rate
from filing_rag.generate.config import GenerationConfig


def test_default_phrase_matches_with_punctuation_and_case() -> None:
    phrase = GenerationConfig().refusal_phrase
    assert phrase == "Not in the corpus."
    assert is_refusal("Not in the corpus.")
    assert is_refusal("not in the corpus")
    assert is_refusal("NOT IN THE CORPUS!")
    assert is_refusal("I cannot answer. Not in the corpus.")


def test_refusal_is_false_without_the_phrase() -> None:
    assert not is_refusal("")
    assert not is_refusal("Microsoft discloses cybersecurity incidents.")
    assert not is_refusal("The corpus includes nine filers.")


def test_custom_phrase() -> None:
    assert is_refusal("I refuse.", phrase="I refuse.")
    assert not is_refusal("Not in the corpus.", phrase="I refuse.")


def test_refusal_rate() -> None:
    assert refusal_rate([True, True, False, True, True]) == pytest.approx(0.8)
    assert refusal_rate([False, False]) == 0.0
    assert refusal_rate([True]) == 1.0


def test_refusal_rate_empty_raises() -> None:
    with pytest.raises(ValueError, match="undefined"):
        refusal_rate(())
