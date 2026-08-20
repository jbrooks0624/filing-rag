"""OpenAI Responses API client. Tests inject a fake SDK; no live calls."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from time import perf_counter
from typing import Protocol, Self, runtime_checkable

from openai import APIStatusError, OpenAI, OpenAIError

from filing_rag.generate.config import GenerationConfig
from filing_rag.generate.types import GenerateResult, GenerateTimings, Usage
from filing_rag.settings import Settings, get_settings

DEFAULT_TIMEOUT = 120.0


class ChatClient(Protocol):
    def complete(self, messages: Sequence[dict[str, str]]) -> GenerateResult: ...


@runtime_checkable
class StreamClient(Protocol):
    """Narrow streaming surface for ``ask_stream``. Not part of ``ChatClient``."""

    def stream(self, messages: Sequence[dict[str, str]]) -> Iterator[str | Usage]: ...


class GenerateError(RuntimeError):
    """The Responses API returned an error or an unreadable stream."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class OpenAIResponsesClient:
    """``client.responses.create`` with streaming text deltas and usage."""

    def __init__(
        self,
        config: GenerationConfig,
        api_key: str,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        sdk: object | None = None,
    ) -> None:
        key = api_key.strip()
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
                "an API key for the OpenAI Responses endpoint."
            )
        self.config = config
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self._owns_sdk = sdk is None
        self._sdk = sdk or OpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=timeout,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        config: GenerationConfig | None = None,
        sdk: object | None = None,
    ) -> OpenAIResponsesClient:
        resolved = settings or get_settings()
        return cls(
            config or GenerationConfig(),
            resolved.require_api_key(),
            resolved.require_base_url(),
            sdk=sdk,
        )

    def close(self) -> None:
        if self._owns_sdk:
            closer = getattr(self._sdk, "close", None)
            if closer is not None:
                closer()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def stream(self, messages: Sequence[dict[str, str]]) -> Iterator[str | Usage]:
        """Yield non-empty text deltas, then a final ``Usage``."""
        if not messages:
            raise ValueError("messages must be non-empty")
        instructions, input_items = _split_instructions(messages)
        try:
            create = self._sdk.responses.create
            extra: dict[str, object] = dict(self.config.responses_kwargs())
            if instructions:
                extra["instructions"] = instructions
            events = create(
                input=input_items,
                stream=True,
                **extra,
            )
            yield from _iter_stream(events)
        except GenerateError:
            raise
        except APIStatusError as exc:
            raise GenerateError(
                _status_message(exc),
                status_code=exc.status_code,
            ) from exc
        except OpenAIError as exc:
            raise GenerateError(f"generation request failed: {exc}") from exc

    def complete(self, messages: Sequence[dict[str, str]]) -> GenerateResult:
        started = perf_counter()
        text, usage = _join_stream(self.stream(messages))
        elapsed_ms = (perf_counter() - started) * 1000
        return GenerateResult(
            text=text,
            usage=usage,
            usd=self.config.cost_usd(usage),
            timings=GenerateTimings(generate_ms=elapsed_ms),
            model=self.config.model,
        )


def _split_instructions(
    messages: Sequence[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    """First system message becomes ``instructions``; the rest is ``input``."""
    instructions: str | None = None
    items: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system" and instructions is None:
            instructions = content
            continue
        items.append({"role": role, "content": content})
    if not items:
        raise ValueError("messages must include a non-system turn")
    return instructions, items


def _iter_stream(events: Iterable[object]) -> Iterator[str | Usage]:
    usage = Usage()
    for event in events:
        etype = getattr(event, "type", None)
        if etype == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            if isinstance(delta, str) and delta:
                yield delta
            continue
        if etype == "response.completed":
            parsed = _usage_from_completed(event)
            if parsed is not None:
                usage = parsed
            continue
        if etype in {"error", "response.failed"}:
            raise GenerateError(_event_error(event))
    yield usage


def _join_stream(items: Iterable[str | Usage]) -> tuple[str, Usage]:
    pieces: list[str] = []
    usage = Usage()
    for item in items:
        if isinstance(item, Usage):
            usage = item
        else:
            pieces.append(item)
    return "".join(pieces), usage


def _usage_from_completed(event: object) -> Usage | None:
    response = getattr(event, "response", None)
    raw = getattr(response, "usage", None) if response is not None else None
    if raw is None:
        return None
    prompt = getattr(raw, "input_tokens", None)
    if prompt is None:
        prompt = getattr(raw, "prompt_tokens", 0)
    completion = getattr(raw, "output_tokens", None)
    if completion is None:
        completion = getattr(raw, "completion_tokens", 0)
    return Usage(
        prompt_tokens=_token_count(prompt),
        completion_tokens=_token_count(completion),
    )


def _token_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _event_error(event: object) -> str:
    for attr in ("message", "error"):
        value = getattr(event, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, str):
            nested = getattr(value, "message", None)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return "generation request failed"


def _status_message(exc: APIStatusError) -> str:
    detail = str(exc).strip()
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                detail = message.strip()
        elif isinstance(error, str) and error.strip():
            detail = error.strip()
    prefix = f"generation request failed ({exc.status_code})"
    if detail:
        return f"{prefix}: {detail}"
    return prefix
