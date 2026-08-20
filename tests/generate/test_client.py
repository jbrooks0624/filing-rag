"""OpenAI Responses client. Fake SDK; no live API."""

from dataclasses import dataclass

import httpx
import pytest
from filing_rag.generate.client import GenerateError, OpenAIResponsesClient
from filing_rag.generate.config import GenerationConfig
from filing_rag.generate.types import Usage
from filing_rag.settings import Settings
from openai import APIStatusError


@dataclass
class FakeDelta:
    type: str = "response.output_text.delta"
    delta: str = ""


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class FakeResponse:
    usage: FakeUsage | None = None


@dataclass
class FakeCompleted:
    type: str = "response.completed"
    response: FakeResponse | None = None


@dataclass
class FakeFailed:
    type: str = "response.failed"
    message: str = "boom"


class FakeResponses:
    def __init__(self, events: list[object] | None = None) -> None:
        self.kwargs: dict | None = None
        self.events = events or [
            FakeDelta(delta="Azure "),
            FakeDelta(delta="grew 30%."),
            FakeCompleted(response=FakeResponse(FakeUsage(1_000_000, 500_000))),
        ]
        self.error: Exception | None = None

    def create(self, **kwargs: object) -> list[object]:
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return list(self.events)


class FakeSDK:
    def __init__(self, responses: FakeResponses | None = None) -> None:
        self.responses = responses or FakeResponses()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _client(sdk: FakeSDK | None = None) -> OpenAIResponsesClient:
    fake = sdk or FakeSDK()
    return OpenAIResponsesClient(
        GenerationConfig(),
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        sdk=fake,
    )


def test_complete_streams_text_and_usage() -> None:
    sdk = FakeSDK()
    with _client(sdk) as client:
        result = client.complete(
            [
                {"role": "system", "content": "Answer from context."},
                {"role": "user", "content": "How fast did Azure grow?"},
            ]
        )
    assert result.text == "Azure grew 30%."
    assert result.usage.prompt_tokens == 1_000_000
    assert result.usage.completion_tokens == 500_000
    assert result.usd == pytest.approx(0.80)
    assert result.model == "gpt-5.6-luna"
    assert result.timings.generate_ms >= 0

    kwargs = sdk.responses.kwargs
    assert kwargs is not None
    assert kwargs["model"] == "gpt-5.6-luna"
    assert kwargs["reasoning"] == {"effort": "none"}
    assert kwargs["max_output_tokens"] == 800
    assert kwargs["stream"] is True
    assert kwargs["store"] is False
    assert kwargs["instructions"] == "Answer from context."
    assert kwargs["input"] == [
        {"role": "user", "content": "How fast did Azure grow?"},
    ]
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs
    assert "top_p" not in kwargs


def test_stream_yields_deltas_then_usage() -> None:
    sdk = FakeSDK()
    messages = [
        {"role": "system", "content": "Answer from context."},
        {"role": "user", "content": "How fast did Azure grow?"},
    ]
    with _client(sdk) as client:
        items = list(client.stream(messages))
        result = client.complete(messages)
    assert items[:-1] == ["Azure ", "grew 30%."]
    usage = items[-1]
    assert isinstance(usage, Usage)
    assert usage.prompt_tokens == 1_000_000
    assert usage.completion_tokens == 500_000
    assert result.text == "".join(items[:-1])
    assert result.usage == usage
    kwargs = sdk.responses.kwargs
    assert kwargs is not None
    assert kwargs["stream"] is True
    assert kwargs["max_output_tokens"] == 800
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs
    assert "top_p" not in kwargs


def test_complete_skips_empty_deltas() -> None:
    sdk = FakeSDK(
        FakeResponses(
            [
                FakeDelta(delta=""),
                FakeDelta(delta="ok"),
                FakeCompleted(response=FakeResponse(FakeUsage(2, 1))),
            ]
        )
    )
    with _client(sdk) as client:
        result = client.complete([{"role": "user", "content": "hi"}])
    assert result.text == "ok"
    assert result.usage.prompt_tokens == 2


def test_failed_event_raises_generate_error() -> None:
    sdk = FakeSDK(FakeResponses([FakeFailed(message="model overloaded")]))
    with _client(sdk) as client, pytest.raises(GenerateError, match="model overloaded"):
        client.complete([{"role": "user", "content": "hi"}])


def test_401_raises_generate_error() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(401, request=request, json={"error": {"message": "invalid api key"}})
    sdk = FakeSDK()
    sdk.responses.error = APIStatusError(
        "invalid api key",
        response=response,
        body={"error": {"message": "invalid api key"}},
    )
    with _client(sdk) as client, pytest.raises(GenerateError, match="invalid api key") as caught:
        client.complete([{"role": "user", "content": "hi"}])
    assert caught.value.status_code == 401


def test_empty_messages_raise() -> None:
    with _client() as client, pytest.raises(ValueError, match="messages must be non-empty"):
        client.complete([])


def test_system_only_messages_raise() -> None:
    with _client() as client, pytest.raises(ValueError, match="non-system"):
        client.complete([{"role": "system", "content": "Be brief."}])


def test_from_settings_requires_api_key() -> None:
    settings = Settings(openai_api_key="  ")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIResponsesClient.from_settings(settings)


def test_constructor_rejects_blank_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIResponsesClient(
            GenerationConfig(),
            api_key="  ",
            base_url="https://api.openai.com/v1",
        )
