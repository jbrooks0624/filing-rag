"""GenerationConfig defaults and Luna dollar arithmetic."""

import pytest
from filing_rag.generate.config import (
    LUNA_INPUT_PER_MILLION,
    LUNA_OUTPUT_PER_MILLION,
    GenerationConfig,
)
from filing_rag.generate.types import Usage


def test_generation_defaults() -> None:
    config = GenerationConfig()
    assert config.model == "gpt-5.6-luna"
    assert config.reasoning_effort == "none"
    assert config.max_output_tokens == 800
    assert config.judge_model == "gpt-5.6-luna"
    assert not hasattr(config, "temperature")
    assert not hasattr(config, "max_tokens")
    assert config.refusal_phrase == "Not in the corpus."
    assert config.input_per_million == LUNA_INPUT_PER_MILLION
    assert config.output_per_million == LUNA_OUTPUT_PER_MILLION


def test_cost_usd_from_token_counts() -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
    assert GenerationConfig().cost_usd(usage) == pytest.approx(0.80)


def test_cost_usd_zero_tokens() -> None:
    assert GenerationConfig().cost_usd(Usage()) == 0.0


def test_responses_kwargs_are_luna_safe() -> None:
    kwargs = GenerationConfig().responses_kwargs()
    assert kwargs == {
        "model": "gpt-5.6-luna",
        "reasoning": {"effort": "none"},
        "max_output_tokens": 800,
        "store": False,
    }
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs
    assert "top_p" not in kwargs


def test_apply_chat_kwargs_strips_instructor_defaults() -> None:
    args = {
        "temperature": 0.01,
        "top_p": 0.1,
        "max_tokens": 1024,
        "system_prompt": None,
    }
    out = GenerationConfig().apply_chat_kwargs(args)
    assert out is args
    assert "temperature" not in out
    assert "top_p" not in out
    assert "max_tokens" not in out
    assert out["max_completion_tokens"] == 4096
    assert out["reasoning_effort"] == "none"
