"""Chunking definition loaded from config/chunking.yaml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from filing_rag.settings import get_settings

STRATEGIES = ("fixed", "structural", "semantic")


class FixedConfig(BaseModel):
    size: int = Field(gt=0)
    overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def overlap_less_than_size(self) -> "FixedConfig":
        if self.overlap >= self.size:
            raise ValueError("fixed.overlap must be less than fixed.size")
        return self


class StructuralConfig(BaseModel):
    max_header_chars: int = Field(gt=0)


class SemanticConfig(BaseModel):
    breakpoint_percentile: int = Field(ge=0, le=100)
    encoder: str


class ChunkingConfig(BaseModel):
    tokenizer: str
    max_tokens: int = Field(gt=0)
    fixed: FixedConfig
    structural: StructuralConfig
    semantic: SemanticConfig

    @model_validator(mode="after")
    def window_fits_cap(self) -> "ChunkingConfig":
        if self.fixed.size > self.max_tokens:
            raise ValueError("fixed.size must be <= max_tokens")
        return self

    @property
    def strategies(self) -> tuple[str, ...]:
        return STRATEGIES


def load_chunking(path: Path | None = None) -> ChunkingConfig:
    chunking_path = path if path is not None else get_settings().chunking_path
    with chunking_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ChunkingConfig.model_validate(payload)
