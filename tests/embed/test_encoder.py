"""Passage encoder: lazy import, no query prefix, no torch at import time."""

import importlib.util

import pytest
from filing_rag.chunking.config import load_chunking
from filing_rag.embed.encoder import BgeEmbedder, EmbedError


def test_from_config_uses_bge_encoder_name() -> None:
    embedder = BgeEmbedder.from_config(load_chunking())
    assert embedder.model_name == "BAAI/bge-base-en-v1.5"
    assert embedder.batch_size == 32


def test_embed_empty_does_not_load_model() -> None:
    embedder = BgeEmbedder("BAAI/bge-base-en-v1.5")
    assert embedder.embed([]) == []
    assert embedder._model is None


def test_embed_query_prepends_prefix_without_loading(monkeypatch) -> None:
    embedder = BgeEmbedder("BAAI/bge-base-en-v1.5")
    seen: list[str] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        seen.extend(texts)
        return [[1.0, 0.0]]

    monkeypatch.setattr(embedder, "embed", fake_embed)
    vector = embedder.embed_query(
        "cyber risk",
        prefix="Represent this sentence for searching relevant passages: ",
    )
    assert seen == ["Represent this sentence for searching relevant passages: cyber risk"]
    assert vector == [1.0, 0.0]
    assert embedder._model is None


def test_bge_embedder_requires_extra() -> None:
    if importlib.util.find_spec("sentence_transformers") is not None:
        pytest.skip("sentence-transformers is installed")
    embedder = BgeEmbedder("BAAI/bge-base-en-v1.5")
    with pytest.raises(EmbedError, match="uv sync --dev"):
        embedder.ensure_available()
