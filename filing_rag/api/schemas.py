"""Request and JSON response models. SSE payloads are dataclasses, not these."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ConfigsResponse(BaseModel):
    strategies: list[str]
    modes: list[str]
    rerank: list[bool]
    k: int
    candidate_k: int
    model: str
    refusal_phrase: str


class QueryRequest(BaseModel):
    """Same flags as CLI generate. ``force`` is not exposed."""

    query: str
    strategy: str
    mode: str = "hybrid"
    rerank: bool = False
    k: int | None = None
    ticker: list[str] | None = None
    year: list[int] | None = None
    item: list[str] | None = None
