"""Fixed generation settings. Change this class, not a YAML file."""

from __future__ import annotations

from dataclasses import dataclass

from filing_rag.generate.types import Usage

# GPT-5.6 Luna list prices (short context), dollars per million tokens.
# https://developers.openai.com/api/docs/pricing
LUNA_INPUT_PER_MILLION = 0.20
LUNA_OUTPUT_PER_MILLION = 1.20

# Instructor/RAGAS structured output needs headroom; 800 is only the serving cap.
JUDGE_MAX_COMPLETION_TOKENS = 4096

# InstructorLLM defaults that Luna Chat Completions reject.
_LEGACY_CHAT_KEYS = ("max_tokens", "temperature", "top_p")


@dataclass(frozen=True)
class GenerationConfig:
    """GPT-5.6 Luna. Responses for serving; Chat Completions for the RAGAS judge.

    Responses (``generate``): ``max_output_tokens``, ``reasoning.effort``.
    Do not send ``temperature`` or ``max_tokens``.

    Chat Completions (RAGAS/instructor): ``max_completion_tokens``,
    ``reasoning_effort``. Do not send ``temperature``, ``top_p``, or ``max_tokens``.
    ragas 0.4.3 treats ``gpt-5.6-*`` as a legacy model (``int('5.6')`` fails),
    so we strip Instructor defaults ourselves.
    """

    model: str = "gpt-5.6-luna"
    reasoning_effort: str = "none"
    max_output_tokens: int = 800
    judge_model: str = "gpt-5.6-luna"
    refusal_phrase: str = "Not in the corpus."
    input_per_million: float = LUNA_INPUT_PER_MILLION
    output_per_million: float = LUNA_OUTPUT_PER_MILLION

    def cost_usd(self, usage: Usage) -> float:
        """``prompt * input / 1e6 + completion * output / 1e6``."""
        return (
            usage.prompt_tokens * self.input_per_million
            + usage.completion_tokens * self.output_per_million
        ) / 1_000_000

    def responses_kwargs(self) -> dict[str, object]:
        """Keyword args for ``client.responses.create`` besides input/stream payload."""
        return {
            "model": self.model,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }

    def apply_chat_kwargs(self, model_args: dict[str, object]) -> dict[str, object]:
        """Rewrite Instructor/RAGAS Chat Completions kwargs for Luna."""
        for key in _LEGACY_CHAT_KEYS:
            model_args.pop(key, None)
        model_args["max_completion_tokens"] = max(
            self.max_output_tokens, JUDGE_MAX_COMPLETION_TOKENS
        )
        model_args["reasoning_effort"] = self.reasoning_effort
        return model_args
