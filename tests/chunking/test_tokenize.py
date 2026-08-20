"""TokenCounter and 512-token cap. Tests never download HuggingFace weights."""

from pathlib import Path

import pytest
from filing_rag.chunking.cap import TokenSpan, cap_from_config, cap_spans, enforce_cap, window_split
from filing_rag.chunking.config import load_chunking
from filing_rag.chunking.tokenize import HuggingFaceTokenCounter, WhitespaceTokenCounter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def _words(n: int) -> str:
    return " ".join(f"w{i:03d}" for i in range(n))


def test_whitespace_count_and_truncate() -> None:
    counter = WhitespaceTokenCounter()
    text = "alpha beta gamma delta"
    assert counter.count(text) == 4
    assert counter.offsets(text) == [(0, 5), (6, 10), (11, 16), (17, 22)]
    assert counter.truncate(text, 2) == "alpha beta"
    assert counter.truncate(text, 4) == text
    assert counter.truncate(text, 0) == ""


def test_window_split_short_text_is_one_span() -> None:
    text = _words(10)
    spans = window_split(text, size=400, overlap=80, counter=WhitespaceTokenCounter())
    assert len(spans) == 1
    assert spans[0].text == text
    assert spans[0].char_start == 0
    assert spans[0].char_end == len(text)
    assert spans[0].token_count == 10


def test_window_split_overlap() -> None:
    text = _words(5)
    spans = window_split(text, size=3, overlap=1, counter=WhitespaceTokenCounter())
    assert [span.token_count for span in spans] == [3, 3]
    assert spans[0].text.split() == ["w000", "w001", "w002"]
    assert spans[1].text.split() == ["w002", "w003", "w004"]
    assert text[spans[0].char_start : spans[0].char_end] == spans[0].text
    assert text[spans[1].char_start : spans[1].char_end] == spans[1].text


def test_window_split_respects_char_offset() -> None:
    text = _words(3)
    spans = window_split(
        text, size=400, overlap=80, counter=WhitespaceTokenCounter(), char_offset=10
    )
    assert spans[0].char_start == 10
    assert spans[0].char_end == 10 + len(text)


def test_enforce_cap_leaves_short_text() -> None:
    text = _words(20)
    spans = enforce_cap(
        text, max_tokens=512, size=400, overlap=80, counter=WhitespaceTokenCounter()
    )
    assert len(spans) == 1
    assert spans[0].token_count == 20
    assert spans[0].text == text


def test_enforce_cap_splits_long_text_under_max() -> None:
    text = _words(900)
    spans = enforce_cap(
        text, max_tokens=512, size=400, overlap=80, counter=WhitespaceTokenCounter()
    )
    assert len(spans) > 1
    assert all(span.token_count <= 512 for span in spans)
    assert all(span.token_count <= 400 for span in spans)
    assert spans[0].token_count == 400
    assert text[spans[0].char_start : spans[0].char_end] == spans[0].text


def test_enforce_cap_empty_and_whitespace() -> None:
    counter = WhitespaceTokenCounter()
    assert enforce_cap("", max_tokens=512, size=400, overlap=80, counter=counter) == []
    assert enforce_cap("   \n", max_tokens=512, size=400, overlap=80, counter=counter) == []


def test_cap_spans_splits_only_oversize_candidates() -> None:
    counter = WhitespaceTokenCounter()
    short = TokenSpan(text=_words(3), char_start=0, char_end=len(_words(3)), token_count=3)
    long_text = _words(10)
    long = TokenSpan(text=long_text, char_start=100, char_end=100 + len(long_text), token_count=10)
    capped = cap_spans([short, long], max_tokens=6, size=4, overlap=1, counter=counter)
    assert capped[0].text == short.text
    assert capped[0].char_start == 0
    assert all(span.token_count <= 6 for span in capped)
    assert all(span.char_start >= 100 for span in capped[1:])


def test_window_split_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        window_split("a b c", size=2, overlap=2, counter=WhitespaceTokenCounter())


def _word_tokenizer() -> Tokenizer:
    vocab = {"[UNK]": 0, **{f"w{i:03d}": i + 1 for i in range(20)}}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def test_hf_counter_wraps_in_memory_tokenizer() -> None:
    counter = HuggingFaceTokenCounter(_word_tokenizer())
    text = "w000 w001 w002"
    assert counter.count(text) == 3
    assert counter.truncate(text, 2).split() == ["w000", "w001"]


def test_hf_from_pretrained_uses_downloaded_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = tmp_path / "tokenizer.json"
    _word_tokenizer().save(str(saved))

    def fake_download(*args: object, **kwargs: object) -> str:
        assert kwargs.get("repo_id") == "local/test" or (args and args[0] == "local/test")
        return str(saved)

    monkeypatch.setattr("filing_rag.chunking.tokenize.hf_hub_download", fake_download)
    counter = HuggingFaceTokenCounter.from_pretrained("local/test")
    assert counter.count("w000 w001") == 2


def test_cap_from_config_uses_yaml_defaults() -> None:
    config = load_chunking()
    text = _words(20)
    spans = cap_from_config(text, config, WhitespaceTokenCounter())
    assert len(spans) == 1
    assert spans[0].token_count == 20
