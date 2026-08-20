"""Retrieve then generate. Tests inject ChatClient and Searcher."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from time import perf_counter
from typing import Protocol

from filing_rag.generate.client import ChatClient, OpenAIResponsesClient, StreamClient
from filing_rag.generate.config import GenerationConfig
from filing_rag.generate.prompt import build_messages
from filing_rag.generate.types import (
    AskResult,
    CitationEvent,
    DoneEvent,
    GenerateResult,
    StreamEvent,
    TokenEvent,
    Usage,
)
from filing_rag.retrieve.pipeline import Retriever
from filing_rag.retrieve.types import Filters, Hit, RetrieveResult
from filing_rag.settings import Settings, get_settings


class Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        strategy: str,
        mode: str = "hybrid",
        k: int | None = None,
        rerank: bool = False,
        filters: Filters | None = None,
        force: bool = False,
    ) -> RetrieveResult: ...


class Generator:
    """Callable generation pipeline. ``generate`` is retrieve-agnostic; ``ask`` searches first."""

    def __init__(
        self,
        config: GenerationConfig,
        client: ChatClient,
        *,
        retriever: Searcher | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self._retriever = retriever
        self.settings = settings or get_settings()

    @classmethod
    def from_config(
        cls,
        *,
        settings: Settings | None = None,
        config: GenerationConfig | None = None,
        client: ChatClient | None = None,
        retriever: Searcher | None = None,
    ) -> Generator:
        resolved = settings or get_settings()
        generation = config or GenerationConfig()
        chat = (
            client
            if client is not None
            else OpenAIResponsesClient.from_settings(resolved, config=generation)
        )
        return cls(generation, chat, retriever=retriever, settings=resolved)

    def generate(self, query: str, hits: Sequence[Hit]) -> GenerateResult:
        messages = build_messages(query, hits, self.config)
        return self.client.complete(messages)

    def ask(
        self,
        query: str,
        *,
        strategy: str,
        mode: str = "hybrid",
        k: int | None = None,
        rerank: bool = False,
        filters: Filters | None = None,
        force: bool = False,
    ) -> AskResult:
        retrieved = self._searcher().search(
            query,
            strategy=strategy,
            mode=mode,
            k=k,
            rerank=rerank,
            filters=filters,
            force=force,
        )
        generated = self.generate(query, retrieved.hits)
        return AskResult(retrieve=retrieved, generate=generated)

    def ask_stream(
        self,
        query: str,
        *,
        strategy: str,
        mode: str = "hybrid",
        k: int | None = None,
        rerank: bool = False,
        filters: Filters | None = None,
        force: bool = False,
    ) -> Iterator[StreamEvent]:
        client = self._stream_client()
        retrieved = self._searcher().search(
            query,
            strategy=strategy,
            mode=mode,
            k=k,
            rerank=rerank,
            filters=filters,
            force=force,
        )
        yield CitationEvent.from_hits(retrieved.hits)
        messages = build_messages(query, retrieved.hits, self.config)
        started = perf_counter()
        pieces: list[str] = []
        usage = Usage()
        for item in client.stream(messages):
            if isinstance(item, Usage):
                usage = item
                continue
            pieces.append(item)
            yield TokenEvent(delta=item)
        generate_ms = (perf_counter() - started) * 1000
        timings = retrieved.timings
        yield DoneEvent(
            text="".join(pieces),
            usage=usage,
            usd=self.config.cost_usd(usage),
            encode_ms=timings.encode_ms,
            dense_ms=timings.dense_ms,
            sparse_ms=timings.sparse_ms,
            fuse_ms=timings.fuse_ms,
            rerank_ms=timings.rerank_ms,
            generate_ms=generate_ms,
            serving_ms=timings.total_ms + generate_ms,
            model=self.config.model,
            strategy=retrieved.strategy,
            mode=retrieved.mode,
            rerank=retrieved.reranked,
            k=k if k is not None else len(retrieved.hits),
        )

    def _searcher(self) -> Searcher:
        if self._retriever is None:
            self._retriever = Retriever.from_config(settings=self.settings)
        return self._retriever

    def _stream_client(self) -> StreamClient:
        client = self.client
        if not isinstance(client, StreamClient):
            raise TypeError(f"{type(client).__name__} does not support streaming")
        return client
