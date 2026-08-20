"""Retrieval definition loaded from config/retrieval.yaml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from filing_rag.settings import get_settings


class Bm25Config(BaseModel):
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)


class RerankConfig(BaseModel):
    model: str = Field(min_length=1)


class RetrievalConfig(BaseModel):
    k: int = Field(gt=0)
    candidate_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    query_prefix: str = Field(min_length=1)
    bm25: Bm25Config
    rerank: RerankConfig

    @model_validator(mode="after")
    def pool_covers_k(self) -> "RetrievalConfig":
        if self.candidate_k < self.k:
            raise ValueError("candidate_k must be >= k")
        return self


def load_retrieval(path: Path | None = None) -> RetrievalConfig:
    retrieval_path = path if path is not None else get_settings().retrieval_path
    with retrieval_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return RetrievalConfig.model_validate(payload)
