"""Pick the corpus 10-Ks from an EDGAR submissions JSON payload."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from pydantic import BaseModel, field_validator

from filing_rag.corpus import Company
from filing_rag.ingest.client import pad_cik, primary_doc_url


class CatalogError(ValueError):
    """A requested filing could not be resolved from submissions history."""


class FilingRef(BaseModel):
    """One primary document selected for ingest."""

    ticker: str
    cik: str
    company_name: str = ""
    accession: str
    form: str
    filing_date: date
    period_of_report: date
    fiscal_year: int
    primary_doc: str
    edgar_url: str

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> str:
        return str(value).strip().upper()

    @field_validator("cik", mode="before")
    @classmethod
    def pad_cik_value(cls, value: object) -> str:
        return pad_cik(str(value))


def fiscal_year_from_report_date(report_date: date) -> int:
    """Fiscal year is the calendar year of period_of_report, not the filing date.

    MSFT FY2024 ends 2024-06-30; a calendar-year filer ending 2024-12-31 and
    filing in February 2025 is still FY2024.
    """
    return report_date.year


def select_filings(
    submissions: dict[str, Any],
    *,
    ticker: str,
    fiscal_years: Sequence[int],
    form_types: Sequence[str] = ("10-K",),
    cik: str | None = None,
) -> list[FilingRef]:
    """Return one filing per requested fiscal year.

    Only exact form matches are kept (``10-K/A`` is not ``10-K``). If more than
    one match shares a year, the latest ``filingDate`` wins. Missing years raise
    ``CatalogError``.
    """
    wanted_years = list(fiscal_years)
    wanted_forms = set(form_types)
    resolved_cik = pad_cik(cik if cik is not None else str(submissions.get("cik", "")))
    company_name = str(submissions.get("name", ""))
    chosen: dict[int, FilingRef] = {}

    for row in _recent_rows(submissions):
        if row["form"] not in wanted_forms:
            continue
        if not row["report_date"]:
            continue
        period = date.fromisoformat(row["report_date"])
        year = fiscal_year_from_report_date(period)
        if year not in wanted_years:
            continue
        filing_date = date.fromisoformat(row["filing_date"])
        ref = FilingRef(
            ticker=ticker,
            cik=resolved_cik,
            company_name=company_name,
            accession=row["accession"],
            form=row["form"],
            filing_date=filing_date,
            period_of_report=period,
            fiscal_year=year,
            primary_doc=row["primary_doc"],
            edgar_url=primary_doc_url(resolved_cik, row["accession"], row["primary_doc"]),
        )
        existing = chosen.get(year)
        if existing is None or ref.filing_date > existing.filing_date:
            chosen[year] = ref

    missing = [year for year in wanted_years if year not in chosen]
    if missing:
        raise CatalogError(
            f"No {sorted(wanted_forms)} filing for {ticker} fiscal year(s) {missing}"
        )
    return [chosen[year] for year in wanted_years]


def select_company_filings(
    submissions: dict[str, Any],
    company: Company,
    *,
    fiscal_years: Sequence[int],
    form_types: Sequence[str] = ("10-K",),
) -> list[FilingRef]:
    return select_filings(
        submissions,
        ticker=company.ticker,
        cik=company.cik,
        fiscal_years=fiscal_years,
        form_types=form_types,
    )


def _recent_rows(submissions: dict[str, Any]) -> list[dict[str, str]]:
    try:
        recent = submissions["filings"]["recent"]
    except (KeyError, TypeError) as exc:
        raise CatalogError("submissions JSON is missing filings.recent") from exc
    accessions = recent.get("accessionNumber") or []
    rows: list[dict[str, str]] = []
    for index, accession in enumerate(accessions):
        rows.append(
            {
                "accession": str(accession),
                "filing_date": str(recent["filingDate"][index]),
                "report_date": str(recent["reportDate"][index]).strip(),
                "form": str(recent["form"][index]),
                "primary_doc": str(recent["primaryDocument"][index]),
            }
        )
    return rows
