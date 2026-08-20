"""Corpus definition loaded from config/corpus.yaml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from filing_rag.settings import get_settings


class Company(BaseModel):
    ticker: str
    cik: str
    sector: str

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> str:
        return str(value).strip().upper()

    @field_validator("cik", mode="before")
    @classmethod
    def pad_cik(cls, value: object) -> str:
        return str(value).strip().zfill(10)


class CorpusConfig(BaseModel):
    form_types: list[str]
    fiscal_years: list[int]
    items: list[str]
    companies: list[Company] = Field(min_length=1)

    def company(self, ticker: str) -> Company:
        key = ticker.strip().upper()
        for company in self.companies:
            if company.ticker == key:
                return company
        known = ", ".join(c.ticker for c in self.companies)
        raise KeyError(f"Unknown ticker {ticker!r}. Known: {known}")

    def select_companies(self, tickers: list[str] | None = None) -> list[Company]:
        if not tickers:
            return list(self.companies)
        return [self.company(ticker) for ticker in tickers]


def load_corpus(path: Path | None = None) -> CorpusConfig:
    corpus_path = path if path is not None else get_settings().corpus_path
    with corpus_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return CorpusConfig.model_validate(payload)
