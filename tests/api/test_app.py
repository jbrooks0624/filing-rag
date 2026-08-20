"""FastAPI app. Skips until fastapi is installed (post-sync). No Postgres, no OpenAI."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field

import pytest

if importlib.util.find_spec("fastapi") is None:
    pytest.skip("fastapi not installed", allow_module_level=True)

from fastapi.testclient import TestClient
from filing_rag.api.app import create_app
from filing_rag.chunking.config import STRATEGIES
from filing_rag.embed.store import StoreError
from filing_rag.generate.client import GenerateError
from filing_rag.generate.config import GenerationConfig
from filing_rag.generate.types import (
    CitationEvent,
    DoneEvent,
    StreamCitation,
    TokenEvent,
    Usage,
)
from filing_rag.retrieve.config import load_retrieval
from filing_rag.retrieve.types import MODES, Filters
from filing_rag.settings import Settings

ACCESSION = "0000950170-24-087843"


@dataclass
class FakeStreamer:
    captured: dict = field(default_factory=dict)
    error: Exception | None = None
    events: tuple = ()

    def __post_init__(self) -> None:
        if self.events:
            return
        self.events = (
            CitationEvent(
                citations=(
                    StreamCitation(
                        rank=1,
                        score=0.91,
                        ticker="MSFT",
                        fiscal_year=2024,
                        item_code="1A",
                        accession=ACCESSION,
                        edgar_url="https://example.com",
                        snippet="Azure revenue grew 30 percent.",
                    ),
                )
            ),
            TokenEvent(delta="Azure "),
            TokenEvent(delta="grew 30%."),
            DoneEvent(
                text="Azure grew 30%.",
                usage=Usage(prompt_tokens=10, completion_tokens=4),
                usd=0.0000068,
                encode_ms=5.0,
                dense_ms=2.0,
                generate_ms=12.0,
                serving_ms=19.0,
                model="gpt-5.6-luna",
                strategy="structural",
                mode="hybrid",
                rerank=True,
                k=5,
            ),
        )

    def ask_stream(self, query, **kwargs):
        self.captured = {"query": query, **kwargs}
        if self.error is not None:
            raise self.error
        yield from self.events


def _client(
    streamer: FakeStreamer | None = None,
    *,
    ping=lambda: None,
    settings: Settings | None = None,
) -> tuple[TestClient, FakeStreamer]:
    fake = streamer or FakeStreamer()
    app = create_app(fake, settings=settings, ping=ping)
    return TestClient(app), fake


def _sse_events(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    name = "message"
    data: list[str] = []
    for line in body.splitlines():
        if line.startswith("event:"):
            name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].lstrip())
        elif line == "":
            if data:
                events.append((name, json.loads("\n".join(data))))
            name = "message"
            data = []
    if data:
        events.append((name, json.loads("\n".join(data))))
    return events


def test_healthz_ok() -> None:
    client, _ = _client()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_503_on_db_failure() -> None:
    def boom() -> None:
        raise StoreError("connection refused")

    client, _ = _client(ping=boom)
    response = client.get("/healthz")
    assert response.status_code == 503
    assert "connection refused" in response.json()["detail"]


def test_healthz_does_not_init_generator(monkeypatch) -> None:
    def boom(**kwargs):
        raise AssertionError("Generator.from_config must not run on /healthz")

    monkeypatch.setattr("filing_rag.api.app.Generator.from_config", boom)
    client = TestClient(create_app(ping=lambda: None))
    assert client.get("/healthz").status_code == 200
    assert client.get("/configs").status_code == 200


def test_configs_from_yaml_and_generation() -> None:
    client, _ = _client()
    response = client.get("/configs")
    assert response.status_code == 200
    payload = response.json()
    retrieval = load_retrieval()
    generation = GenerationConfig()
    assert payload["strategies"] == list(STRATEGIES)
    assert payload["modes"] == list(MODES)
    assert payload["rerank"] == [False, True]
    assert payload["k"] == retrieval.k
    assert payload["candidate_k"] == retrieval.candidate_k
    assert payload["model"] == generation.model
    assert payload["refusal_phrase"] == generation.refusal_phrase


def test_query_sse_citations_tokens_done() -> None:
    client, fake = _client()
    response = client.post(
        "/query",
        json={
            "query": "How fast did Azure grow?",
            "strategy": "structural",
            "mode": "dense",
            "rerank": True,
            "k": 5,
            "ticker": ["MSFT"],
            "year": [2024],
            "item": ["1A"],
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(response.text)
    assert [name for name, _ in events] == ["citations", "token", "token", "done"]
    citations = events[0][1]["citations"]
    assert citations[0]["ticker"] == "MSFT"
    assert citations[0]["snippet"] == "Azure revenue grew 30 percent."
    assert "text" not in citations[0]
    assert events[1][1] == {"delta": "Azure "}
    assert events[2][1] == {"delta": "grew 30%."}
    done = events[3][1]
    assert done["text"] == "Azure grew 30%."
    assert done["usage"] == {"prompt_tokens": 10, "completion_tokens": 4}
    assert done["serving_ms"] == 19.0
    assert done["k"] == 5
    assert fake.captured["query"] == "How fast did Azure grow?"
    assert fake.captured["strategy"] == "structural"
    assert fake.captured["mode"] == "dense"
    assert fake.captured["k"] == 5
    assert fake.captured["rerank"] is True
    assert fake.captured["filters"] == Filters(
        tickers=("MSFT",),
        fiscal_years=(2024,),
        item_codes=("1A",),
    )


def test_query_defaults() -> None:
    client, fake = _client()
    response = client.post("/query", json={"query": "rates", "strategy": "fixed"})
    assert response.status_code == 200
    assert fake.captured["mode"] == "hybrid"
    assert fake.captured["rerank"] is False
    assert fake.captured["k"] is None
    assert fake.captured["filters"] is None


def test_query_400_empty_query() -> None:
    client, fake = _client()
    response = client.post("/query", json={"query": "   ", "strategy": "fixed"})
    assert response.status_code == 400
    assert "non-empty" in response.json()["detail"]
    assert fake.captured == {}


def test_query_400_unknown_strategy() -> None:
    client, _ = _client()
    response = client.post("/query", json={"query": "rates", "strategy": "recursive"})
    assert response.status_code == 400
    assert "recursive" in response.json()["detail"]


def test_query_400_unknown_mode() -> None:
    client, _ = _client()
    response = client.post(
        "/query",
        json={"query": "rates", "strategy": "fixed", "mode": "lexical"},
    )
    assert response.status_code == 400
    assert "lexical" in response.json()["detail"]


def test_query_400_invalid_k() -> None:
    client, _ = _client()
    response = client.post("/query", json={"query": "rates", "strategy": "fixed", "k": 0})
    assert response.status_code == 400
    assert "k must be >= 1" in response.json()["detail"]


def test_query_503_store_error() -> None:
    fake = FakeStreamer(error=StoreError("index missing"))
    client, _ = _client(fake)
    response = client.post("/query", json={"query": "rates", "strategy": "fixed"})
    assert response.status_code == 503
    assert "index missing" in response.json()["detail"]


def test_query_503_generate_error() -> None:
    fake = FakeStreamer(error=GenerateError("model overloaded"))
    client, _ = _client(fake)
    response = client.post("/query", json={"query": "rates", "strategy": "fixed"})
    assert response.status_code == 503
    assert "model overloaded" in response.json()["detail"]


def test_query_503_missing_api_key() -> None:
    client = TestClient(
        create_app(
            settings=Settings(openai_api_key="  "),
            ping=lambda: None,
        )
    )
    response = client.post("/query", json={"query": "rates", "strategy": "fixed"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_query_inits_generator_once(monkeypatch) -> None:
    fake = FakeStreamer()
    calls = {"n": 0}

    def from_config(**kwargs):
        calls["n"] += 1
        return fake

    monkeypatch.setattr("filing_rag.api.app.Generator.from_config", from_config)
    client = TestClient(create_app(ping=lambda: None))
    assert client.get("/healthz").status_code == 200
    assert client.get("/configs").status_code == 200
    assert calls["n"] == 0
    assert client.post("/query", json={"query": "rates", "strategy": "fixed"}).status_code == 200
    assert client.post("/query", json={"query": "rates", "strategy": "fixed"}).status_code == 200
    assert calls["n"] == 1
    assert fake.captured["query"] == "rates"
