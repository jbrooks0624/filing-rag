-- Applied by Indexer on first run. Idempotent (IF NOT EXISTS).
-- HNSW is created after embeddings exist; do not add it here.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS filings (
    id BIGSERIAL PRIMARY KEY,
    accession TEXT NOT NULL UNIQUE,
    ticker TEXT NOT NULL,
    cik TEXT NOT NULL,
    company_name TEXT NOT NULL DEFAULT '',
    form TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    period_of_report TEXT NOT NULL,
    fiscal_year INT NOT NULL,
    primary_doc TEXT NOT NULL,
    edgar_url TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sections (
    id BIGSERIAL PRIMARY KEY,
    filing_id BIGINT NOT NULL REFERENCES filings (id) ON DELETE CASCADE,
    item_code TEXT NOT NULL,
    item_title TEXT NOT NULL,
    text TEXT NOT NULL,
    char_start INT NOT NULL,
    char_end INT NOT NULL,
    UNIQUE (filing_id, item_code)
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    filing_id BIGINT NOT NULL REFERENCES filings (id) ON DELETE CASCADE,
    section_id BIGINT NOT NULL REFERENCES sections (id) ON DELETE CASCADE,
    strategy TEXT NOT NULL CHECK (strategy IN ('fixed', 'structural', 'semantic')),
    chunk_index INT NOT NULL,
    char_start INT NOT NULL,
    char_end INT NOT NULL,
    token_count INT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(768),
    ticker TEXT NOT NULL,
    fiscal_year INT NOT NULL,
    item_code TEXT NOT NULL,
    accession TEXT NOT NULL,
    UNIQUE (section_id, strategy, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_ticker_idx ON chunks (ticker);
CREATE INDEX IF NOT EXISTS chunks_fiscal_year_idx ON chunks (fiscal_year);
CREATE INDEX IF NOT EXISTS chunks_item_code_idx ON chunks (item_code);
CREATE INDEX IF NOT EXISTS chunks_strategy_accession_idx ON chunks (strategy, accession);
