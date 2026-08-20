"""Corpus config and settings load as expected."""

import pytest
from filing_rag.corpus import load_corpus
from filing_rag.settings import Settings, get_settings


def test_corpus_has_nine_companies_and_three_years() -> None:
    corpus = load_corpus()
    assert [company.ticker for company in corpus.companies] == [
        "MSFT",
        "GOOGL",
        "AMZN",
        "JPM",
        "BAC",
        "GS",
        "PFE",
        "MRK",
        "LLY",
    ]
    assert corpus.form_types == ["10-K"]
    assert corpus.fiscal_years == [2022, 2023, 2024]
    assert corpus.items == ["1A", "7", "7A"]
    assert corpus.company("msft").cik == "0000789019"
    assert len(corpus.companies) * len(corpus.fiscal_years) == 27


def test_unknown_ticker_raises() -> None:
    with pytest.raises(KeyError, match="NVDA"):
        load_corpus().company("NVDA")


def test_settings_data_paths_live_under_data() -> None:
    settings = get_settings()
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.parsed_dir == settings.data_dir / "parsed"
    assert settings.chunks_dir == settings.data_dir / "chunks"
    assert settings.indexes_dir == settings.data_dir / "indexes"
    assert settings.corpus_path.name == "corpus.yaml"
    assert settings.chunking_path.name == "chunking.yaml"
    assert settings.retrieval_path.name == "retrieval.yaml"
    assert settings.golden_path.name == "golden_set.yaml"
    assert settings.golden_path.parent.name == "eval"
    assert settings.results_dir.name == "results"
    assert settings.migrations_dir.name == "migrations"
    assert settings.openai_base_url == "https://api.openai.com/v1"


def test_require_user_agent_fails_when_missing() -> None:
    settings = Settings(edgar_user_agent="  ")
    with pytest.raises(ValueError, match="EDGAR_USER_AGENT"):
        settings.require_user_agent()


def test_require_database_url_fails_when_missing() -> None:
    settings = Settings(database_url="  ")
    with pytest.raises(ValueError, match="DATABASE_URL"):
        settings.require_database_url()


def test_database_url_defaults_to_compose() -> None:
    settings = Settings()
    assert settings.require_database_url() == "postgresql://filing:filing@localhost:5432/filing_rag"


def test_require_api_key_fails_when_missing() -> None:
    settings = Settings(openai_api_key="  ")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.require_api_key()


def test_require_base_url_strips_trailing_slash() -> None:
    settings = Settings(openai_base_url="https://api.openai.com/v1/")
    assert settings.require_base_url() == "https://api.openai.com/v1"


def test_require_base_url_fails_when_missing() -> None:
    settings = Settings(openai_base_url="  ")
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        settings.require_base_url()
