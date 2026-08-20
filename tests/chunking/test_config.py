"""Chunking YAML and settings paths."""

import pytest
from filing_rag.chunking.config import ChunkingConfig, load_chunking
from pydantic import ValidationError


def test_chunking_defaults() -> None:
    config = load_chunking()
    assert config.tokenizer == "BAAI/bge-base-en-v1.5"
    assert config.max_tokens == 512
    assert config.fixed.size == 400
    assert config.fixed.overlap == 80
    assert config.structural.max_header_chars == 120
    assert config.semantic.breakpoint_percentile == 95
    assert config.semantic.encoder == "BAAI/bge-base-en-v1.5"
    assert config.strategies == ("fixed", "structural", "semantic")


def test_overlap_must_be_less_than_size() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        ChunkingConfig.model_validate(
            {
                "tokenizer": "x",
                "max_tokens": 512,
                "fixed": {"size": 80, "overlap": 80},
                "structural": {"max_header_chars": 120},
                "semantic": {"breakpoint_percentile": 95, "encoder": "x"},
            }
        )


def test_fixed_size_must_fit_cap() -> None:
    with pytest.raises(ValidationError, match="max_tokens"):
        ChunkingConfig.model_validate(
            {
                "tokenizer": "x",
                "max_tokens": 256,
                "fixed": {"size": 400, "overlap": 80},
                "structural": {"max_header_chars": 120},
                "semantic": {"breakpoint_percentile": 95, "encoder": "x"},
            }
        )
