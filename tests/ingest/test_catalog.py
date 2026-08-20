"""Catalog selects one 10-K per fiscal year from columnar submissions JSON."""

import json

import pytest
from filing_rag.corpus import Company
from filing_rag.ingest.catalog import (
    CatalogError,
    select_company_filings,
    select_filings,
)
from filing_rag.settings import PROJECT_ROOT

MSFT_CACHE = PROJECT_ROOT / "data/raw/submissions/CIK0000789019.json"


def _submissions(
    *,
    forms: list[str],
    report_dates: list[str],
    filing_dates: list[str],
    accessions: list[str] | None = None,
    docs: list[str] | None = None,
) -> dict:
    n = len(forms)
    return {
        "cik": "0000789019",
        "name": "MICROSOFT CORP",
        "filings": {
            "recent": {
                "accessionNumber": accessions or [f"0000789019-24-{i:06d}" for i in range(n)],
                "filingDate": filing_dates,
                "reportDate": report_dates,
                "form": forms,
                "primaryDocument": docs or [f"doc-{i}.htm" for i in range(n)],
            }
        },
    }


def test_msft_june_report_date_is_that_fiscal_year() -> None:
    payload = _submissions(
        forms=["10-K", "10-Q", "8-K"],
        report_dates=["2024-06-30", "2024-03-31", ""],
        filing_dates=["2024-07-30", "2024-04-24", "2024-08-01"],
        accessions=["0000950170-24-087843", "q", "8k"],
        docs=["msft-20240630.htm", "q.htm", "8k.htm"],
    )
    refs = select_filings(payload, ticker="MSFT", fiscal_years=[2024])
    assert len(refs) == 1
    assert refs[0].fiscal_year == 2024
    assert refs[0].period_of_report.isoformat() == "2024-06-30"
    assert refs[0].primary_doc == "msft-20240630.htm"
    assert refs[0].edgar_url.endswith("/789019/000095017024087843/msft-20240630.htm")


def test_calendar_year_filer_uses_report_date_not_filing_date() -> None:
    payload = _submissions(
        forms=["10-K"],
        report_dates=["2024-12-31"],
        filing_dates=["2025-02-04"],
        accessions=["0001018724-25-000001"],
        docs=["amzn-20241231.htm"],
    )
    refs = select_filings(
        payload,
        ticker="AMZN",
        cik="0001018724",
        fiscal_years=[2024],
    )
    assert refs[0].fiscal_year == 2024
    assert refs[0].filing_date.isoformat() == "2025-02-04"


def test_amendments_are_ignored() -> None:
    payload = _submissions(
        forms=["10-K/A", "10-K"],
        report_dates=["2024-06-30", "2024-06-30"],
        filing_dates=["2024-08-15", "2024-07-30"],
        accessions=["amend", "original"],
        docs=["a.htm", "k.htm"],
    )
    refs = select_filings(payload, ticker="MSFT", fiscal_years=[2024])
    assert refs[0].accession == "original"
    assert refs[0].form == "10-K"


def test_latest_filing_date_wins_when_two_10ks_share_a_year() -> None:
    payload = _submissions(
        forms=["10-K", "10-K"],
        report_dates=["2023-06-30", "2023-06-30"],
        filing_dates=["2023-07-27", "2023-08-01"],
        accessions=["first", "second"],
        docs=["a.htm", "b.htm"],
    )
    refs = select_filings(payload, ticker="MSFT", fiscal_years=[2023])
    assert refs[0].accession == "second"


def test_selects_three_years_in_requested_order() -> None:
    payload = _submissions(
        forms=["10-K", "10-K", "10-K"],
        report_dates=["2022-06-30", "2024-06-30", "2023-06-30"],
        filing_dates=["2022-07-28", "2024-07-30", "2023-07-27"],
        accessions=["fy22", "fy24", "fy23"],
        docs=["22.htm", "24.htm", "23.htm"],
    )
    refs = select_filings(payload, ticker="msft", fiscal_years=[2022, 2023, 2024])
    assert [ref.fiscal_year for ref in refs] == [2022, 2023, 2024]
    assert [ref.accession for ref in refs] == ["fy22", "fy23", "fy24"]


def test_missing_year_raises() -> None:
    payload = _submissions(
        forms=["10-K"],
        report_dates=["2024-06-30"],
        filing_dates=["2024-07-30"],
    )
    with pytest.raises(CatalogError, match="2022"):
        select_filings(payload, ticker="MSFT", fiscal_years=[2022, 2024])


def test_select_company_filings_uses_corpus_company() -> None:
    payload = _submissions(
        forms=["10-K"],
        report_dates=["2024-06-30"],
        filing_dates=["2024-07-30"],
        accessions=["acc"],
        docs=["msft.htm"],
    )
    company = Company(ticker="MSFT", cik="0000789019", sector="tech")
    refs = select_company_filings(payload, company, fiscal_years=[2024])
    assert refs[0].ticker == "MSFT"
    assert refs[0].cik == "0000789019"


@pytest.mark.skipif(not MSFT_CACHE.exists(), reason="no live MSFT submissions cache")
def test_live_msft_cache_resolves_fy2022_2024() -> None:
    payload = json.loads(MSFT_CACHE.read_text())
    refs = select_filings(payload, ticker="MSFT", fiscal_years=[2022, 2023, 2024])
    assert [ref.fiscal_year for ref in refs] == [2022, 2023, 2024]
    assert all(ref.form == "10-K" for ref in refs)
    assert all(ref.period_of_report.month == 6 for ref in refs)
    assert refs[-1].primary_doc == "msft-20240630.htm"
