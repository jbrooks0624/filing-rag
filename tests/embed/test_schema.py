"""Schema SQL is the contract; tests read the file, they do not open Postgres."""

from filing_rag.settings import PROJECT_ROOT, Settings

SCHEMA = (PROJECT_ROOT / "migrations" / "001_init.sql").read_text(encoding="utf-8")


def test_settings_point_at_init_migration() -> None:
    settings = Settings()
    path = settings.migrations_dir / "001_init.sql"
    assert path.is_file()
    assert path.name == "001_init.sql"


def test_schema_creates_filings_sections_chunks() -> None:
    assert "CREATE EXTENSION IF NOT EXISTS vector" in SCHEMA
    assert "CREATE TABLE IF NOT EXISTS filings" in SCHEMA
    assert "CREATE TABLE IF NOT EXISTS sections" in SCHEMA
    assert "CREATE TABLE IF NOT EXISTS chunks" in SCHEMA
    assert "embedding vector(768)" in SCHEMA
    assert "UNIQUE (section_id, strategy, chunk_index)" in SCHEMA


def test_chunks_denormalize_filter_columns() -> None:
    assert "CREATE INDEX IF NOT EXISTS chunks_ticker_idx ON chunks (ticker)" in SCHEMA
    assert "CREATE INDEX IF NOT EXISTS chunks_fiscal_year_idx ON chunks (fiscal_year)" in SCHEMA
    assert "CREATE INDEX IF NOT EXISTS chunks_item_code_idx ON chunks (item_code)" in SCHEMA
    assert "CREATE INDEX IF NOT EXISTS chunks_strategy_accession_idx" in SCHEMA


def test_schema_does_not_prebuild_hnsw_or_fts() -> None:
    lowered = SCHEMA.lower()
    assert "using hnsw" not in lowered
    assert "tsvector" not in lowered
    assert "to_tsvector" not in lowered
