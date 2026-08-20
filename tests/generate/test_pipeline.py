"""Generator.generate / ask with injected client and retriever. No API, no Postgres."""

from dataclasses import replace

import pytest
from filing_rag.generate.config import GenerationConfig
from filing_rag.generate.pipeline import Generator
from filing_rag.generate.prompt import build_messages
from filing_rag.generate.types import (
    CitationEvent,
    DoneEvent,
    GenerateResult,
    GenerateTimings,
    TokenEvent,
    Usage,
)
from filing_rag.retrieve.types import Filters, Hit, RetrieveResult, RetrieveTimings, snippet
from filing_rag.settings import Settings

ACCESSION = "0000950170-24-087843"


def _hit() -> Hit:
    return Hit(
        chunk_id=1,
        score=0.91,
        rank=1,
        text="Azure revenue grew 30 percent.",
        ticker="MSFT",
        fiscal_year=2024,
        item_code="1A",
        accession=ACCESSION,
        char_start=0,
        char_end=30,
        edgar_url="https://example.com",
        strategy="fixed",
        chunk_index=0,
    )


class FakeClient:
    def __init__(self, result: GenerateResult | None = None) -> None:
        self.messages: list[dict[str, str]] | None = None
        self.result = result or GenerateResult(
            text="Azure grew 30%. [MSFT FY2024 Item 7]",
            usage=Usage(prompt_tokens=10, completion_tokens=4),
            usd=0.01,
            timings=GenerateTimings(generate_ms=12.0),
            model="gpt-5.6-luna",
        )

    def complete(self, messages):
        self.messages = list(messages)
        return self.result


class FakeRetriever:
    def __init__(self, result: RetrieveResult | None = None) -> None:
        self.captured: dict = {}
        self.result = result or RetrieveResult(
            hits=(_hit(),),
            mode="hybrid",
            strategy="fixed",
            reranked=False,
            timings=RetrieveTimings(encode_ms=5.0, dense_ms=2.0),
        )

    def search(
        self,
        query,
        *,
        strategy,
        mode="hybrid",
        k=None,
        rerank=False,
        filters=None,
        force=False,
    ):
        self.captured = {
            "query": query,
            "strategy": strategy,
            "mode": mode,
            "k": k,
            "rerank": rerank,
            "filters": filters,
            "force": force,
        }
        return self.result


class FakeStreamClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None
        self.deltas = ("Azure ", "grew 30%.")
        self.usage = Usage(prompt_tokens=10, completion_tokens=4)

    def complete(self, messages):
        raise AssertionError("ask_stream must not call complete()")

    def stream(self, messages):
        self.messages = list(messages)
        yield from self.deltas
        yield self.usage


def test_generate_sends_citation_prompt_to_client() -> None:
    client = FakeClient()
    generator = Generator(GenerationConfig(), client)
    hit = _hit()
    result = generator.generate("How fast did Azure grow?", [hit])
    assert result.text.startswith("Azure grew 30%")
    expected = build_messages("How fast did Azure grow?", [hit], GenerationConfig())
    assert client.messages == list(expected)


def test_ask_retrieves_then_generates() -> None:
    client = FakeClient()
    retriever = FakeRetriever()
    generator = Generator.from_config(
        settings=Settings(openai_api_key="sk-test"),
        client=client,
        retriever=retriever,
    )
    filters = Filters(tickers=("MSFT",), fiscal_years=(2024,), item_codes=("1A",))
    result = generator.ask(
        "How fast did Azure grow?",
        strategy="structural",
        mode="dense",
        k=5,
        rerank=True,
        filters=filters,
        force=True,
    )
    assert retriever.captured == {
        "query": "How fast did Azure grow?",
        "strategy": "structural",
        "mode": "dense",
        "k": 5,
        "rerank": True,
        "filters": filters,
        "force": True,
    }
    assert result.retrieve.hits[0].ticker == "MSFT"
    assert result.retrieve.timings.encode_ms == 5.0
    assert result.generate.text.startswith("Azure grew 30%")
    assert result.generate.timings.generate_ms == 12.0
    assert client.messages is not None
    assert "Azure revenue grew 30 percent." in client.messages[1]["content"]


def test_ask_stream_citations_tokens_done() -> None:
    client = FakeStreamClient()
    retriever = FakeRetriever()
    generator = Generator.from_config(
        settings=Settings(openai_api_key="sk-test"),
        client=client,
        retriever=retriever,
    )
    filters = Filters(tickers=("MSFT",), fiscal_years=(2024,), item_codes=("1A",))
    events = list(
        generator.ask_stream(
            "How fast did Azure grow?",
            strategy="structural",
            mode="dense",
            k=5,
            rerank=True,
            filters=filters,
            force=True,
        )
    )
    assert retriever.captured == {
        "query": "How fast did Azure grow?",
        "strategy": "structural",
        "mode": "dense",
        "k": 5,
        "rerank": True,
        "filters": filters,
        "force": True,
    }
    citations, *tokens, done = events
    assert isinstance(citations, CitationEvent)
    assert len(citations.citations) == 1
    cite = citations.citations[0]
    hit = _hit()
    assert cite.rank == hit.rank
    assert cite.score == hit.score
    assert cite.ticker == hit.ticker
    assert cite.fiscal_year == hit.fiscal_year
    assert cite.item_code == hit.item_code
    assert cite.accession == hit.accession
    assert cite.edgar_url == hit.edgar_url
    assert cite.snippet == snippet(hit.text)
    assert not hasattr(cite, "text")
    assert [event.delta for event in tokens] == ["Azure ", "grew 30%."]
    assert all(isinstance(event, TokenEvent) for event in tokens)
    assert isinstance(done, DoneEvent)
    assert done.text == "Azure grew 30%."
    assert done.usage == client.usage
    assert done.usd == pytest.approx(GenerationConfig().cost_usd(client.usage))
    assert done.encode_ms == 5.0
    assert done.dense_ms == 2.0
    assert done.serving_ms == pytest.approx(7.0 + done.generate_ms)
    assert done.model == "gpt-5.6-luna"
    assert done.strategy == "fixed"
    assert done.mode == "hybrid"
    assert done.rerank is False
    assert done.k == 5
    expected = build_messages("How fast did Azure grow?", [hit], GenerationConfig())
    assert client.messages == list(expected)


def test_citation_event_uses_snippet_not_full_text() -> None:
    hit = replace(_hit(), text="word " * 80)
    event = CitationEvent.from_hits([hit])
    cite = event.citations[0]
    assert cite.snippet == snippet(hit.text)
    assert len(cite.snippet) == 96
    assert cite.snippet != hit.text
    assert not hasattr(cite, "text")


def test_ask_stream_requires_stream_client() -> None:
    generator = Generator.from_config(
        settings=Settings(openai_api_key="sk-test"),
        client=FakeClient(),
        retriever=FakeRetriever(),
    )
    with pytest.raises(TypeError, match="does not support streaming"):
        list(generator.ask_stream("How fast did Azure grow?", strategy="fixed"))
