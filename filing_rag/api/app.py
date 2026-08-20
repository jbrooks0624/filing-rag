"""FastAPI factory. Tests inject a generator and a DB ping; live path lazy-inits."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict
from typing import Protocol

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from filing_rag.api.schemas import ConfigsResponse, HealthResponse, QueryRequest
from filing_rag.chunking.config import STRATEGIES
from filing_rag.embed.encoder import EmbedError
from filing_rag.embed.store import StoreError, require_k, require_strategy
from filing_rag.generate.client import GenerateError
from filing_rag.generate.config import GenerationConfig
from filing_rag.generate.pipeline import Generator
from filing_rag.generate.types import CitationEvent, DoneEvent, StreamEvent, TokenEvent
from filing_rag.retrieve.config import load_retrieval
from filing_rag.retrieve.rerank import RerankError
from filing_rag.retrieve.sparse import SparseError
from filing_rag.retrieve.types import MODES, Filters
from filing_rag.settings import Settings, get_settings

_SERVING_ERRORS = (
    EmbedError,
    GenerateError,
    RerankError,
    SparseError,
    StoreError,
)


class AskStreamer(Protocol):
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
    ) -> Iterator[StreamEvent]: ...


def create_app(
    generator: AskStreamer | None = None,
    *,
    settings: Settings | None = None,
    ping: Callable[[], None] | None = None,
) -> FastAPI:
    """Build the API. ``/healthz`` never constructs a retriever or loads torch."""
    resolved = settings or get_settings()
    held = generator
    app = FastAPI(title="filing-rag", docs_url=None, redoc_url=None)

    def current_generator() -> AskStreamer:
        nonlocal held
        if held is None:
            try:
                held = Generator.from_config(settings=resolved)
            except ValueError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return held

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        try:
            if ping is not None:
                ping()
            else:
                _ping_postgres(resolved)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return HealthResponse(status="ok")

    @app.get("/configs", response_model=ConfigsResponse)
    def configs() -> ConfigsResponse:
        retrieval = load_retrieval(resolved.retrieval_path)
        generation = GenerationConfig()
        return ConfigsResponse(
            strategies=list(STRATEGIES),
            modes=list(MODES),
            rerank=[False, True],
            k=retrieval.k,
            candidate_k=retrieval.candidate_k,
            model=generation.model,
            refusal_phrase=generation.refusal_phrase,
        )

    @app.post("/query")
    def query(body: QueryRequest) -> StreamingResponse:
        strategy, mode, k = _validate_query(body)
        streamer = current_generator()
        events = streamer.ask_stream(
            body.query,
            strategy=strategy,
            mode=mode,
            k=k,
            rerank=body.rerank,
            filters=_filters(body),
        )
        try:
            first = next(events)
        except StopIteration:
            first = None
        except _SERVING_ERRORS as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise _value_error(exc) from exc

        def publish() -> Iterator[str]:
            if first is not None:
                yield _sse(first)
            try:
                for event in events:
                    yield _sse(event)
            except _SERVING_ERRORS as exc:
                yield _sse_error(str(exc))

        return StreamingResponse(publish(), media_type="text/event-stream")

    return app


def _ping_postgres(settings: Settings) -> None:
    with psycopg.connect(settings.require_database_url()) as conn:
        conn.execute("SELECT 1")


def _validate_query(body: QueryRequest) -> tuple[str, str, int | None]:
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query must be non-empty")
    try:
        strategy = require_strategy(body.strategy.strip().lower())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    mode = body.mode.strip().lower()
    if mode not in MODES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown mode {body.mode!r}. Known: {', '.join(MODES)}",
        )
    k = body.k
    if k is not None:
        try:
            k = require_k(k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return strategy, mode, k


def _filters(body: QueryRequest) -> Filters | None:
    if not (body.ticker or body.year or body.item):
        return None
    return Filters(
        tickers=tuple(body.ticker) if body.ticker else None,
        fiscal_years=tuple(body.year) if body.year else None,
        item_codes=tuple(body.item) if body.item else None,
    )


def _value_error(exc: ValueError) -> HTTPException:
    detail = str(exc)
    if "OPENAI_API_KEY" in detail:
        return HTTPException(status_code=503, detail=detail)
    return HTTPException(status_code=400, detail=detail)


def _sse(event: StreamEvent) -> str:
    if isinstance(event, CitationEvent):
        name = "citations"
    elif isinstance(event, TokenEvent):
        name = "token"
    elif isinstance(event, DoneEvent):
        name = "done"
    else:
        raise TypeError(f"unsupported stream event: {type(event)!r}")
    return f"event: {name}\ndata: {json.dumps(asdict(event))}\n\n"


def _sse_error(message: str) -> str:
    return f"event: error\ndata: {json.dumps({'message': message})}\n\n"
