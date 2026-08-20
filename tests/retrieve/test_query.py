"""Query encoder prepends the bge prefix; tests never load torch."""

import pytest
from filing_rag.retrieve.config import load_retrieval
from filing_rag.retrieve.query import BgeQueryEncoder


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def embed_query(self, text: str, *, prefix: str) -> list[float]:
        self.calls.append((text, prefix))
        return [0.5, 0.5]


def test_from_config_uses_yaml_prefix() -> None:
    fake = FakeEmbedder()
    encoder = BgeQueryEncoder.from_config(embedder=fake)
    assert encoder.prefix == load_retrieval().query_prefix
    assert encoder.prefix.startswith("Represent this sentence")
    assert encoder.embedder is fake


def test_encode_never_sends_raw_query() -> None:
    fake = FakeEmbedder()
    prefix = "Represent this sentence for searching relevant passages: "
    encoder = BgeQueryEncoder(fake, prefix=prefix)
    vector = encoder.encode("cybersecurity risk")
    assert vector == [0.5, 0.5]
    assert fake.calls == [("cybersecurity risk", prefix)]


def test_encode_rejects_empty_query() -> None:
    fake = FakeEmbedder()
    encoder = BgeQueryEncoder(fake, prefix="prefix: ")
    with pytest.raises(ValueError, match="non-empty"):
        encoder.encode("")
    with pytest.raises(ValueError, match="non-empty"):
        encoder.encode("   ")
    assert fake.calls == []
