"""Ingestor.run is idempotent, isolates per-filing errors, and honors parse_only."""

from pathlib import Path

import pytest
from filing_rag.ingest.cache import DiskCache, accession_key
from filing_rag.ingest.client import EdgarClient, primary_doc_url, submissions_url
from filing_rag.ingest.pipeline import Ingestor
from filing_rag.settings import Settings
from pytest_httpx import HTTPXMock

USER_AGENT = "filing-rag test@example.com"
MSFT_CIK = "0000789019"
JPM_CIK = "0000019617"
ACCESSION = "0000950170-24-087843"
DOC = "msft-20240630.htm"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_10k.html"


def _settings(tmp_path: Path) -> Settings:
    corpus = tmp_path / "corpus.yaml"
    corpus.write_text(
        """
form_types: ["10-K"]
fiscal_years: [2024]
items: ["1A", "7", "7A"]
companies:
  - ticker: MSFT
    cik: "0000789019"
    sector: tech
  - ticker: JPM
    cik: "0000019617"
    sector: banks
""".strip()
        + "\n"
    )
    return Settings(edgar_user_agent=USER_AGENT, data_dir=tmp_path / "data", corpus_path=corpus)


def _submissions(cik: str, *, accession: str = ACCESSION, doc: str = DOC) -> dict:
    return {
        "cik": cik,
        "name": "TEST CORP",
        "filings": {
            "recent": {
                "accessionNumber": [accession],
                "filingDate": ["2024-07-30"],
                "reportDate": ["2024-06-30"],
                "form": ["10-K"],
                "primaryDocument": [doc],
            }
        },
    }


def _ingestor(tmp_path: Path) -> Ingestor:
    settings = _settings(tmp_path)
    client = EdgarClient(USER_AGENT, DiskCache(settings.raw_dir))
    return Ingestor.from_config(settings.corpus_path, settings=settings, client=client)


def test_first_run_fetches_and_parses(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    html = FIXTURE.read_bytes()
    httpx_mock.add_response(url=submissions_url(MSFT_CIK), json=_submissions(MSFT_CIK))
    httpx_mock.add_response(url=primary_doc_url(MSFT_CIK, ACCESSION, DOC), content=html)
    result = _ingestor(tmp_path).run(tickers=["MSFT"])
    assert result.fetched == 1
    assert result.parsed == 1
    assert result.skipped == 0
    assert result.ok
    parsed = tmp_path / "data/parsed" / f"{accession_key(ACCESSION)}.json"
    assert parsed.exists()
    assert '"item_code": "1A"' in parsed.read_text()


def test_second_run_skips_without_http(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    html = FIXTURE.read_bytes()
    httpx_mock.add_response(url=submissions_url(MSFT_CIK), json=_submissions(MSFT_CIK))
    httpx_mock.add_response(url=primary_doc_url(MSFT_CIK, ACCESSION, DOC), content=html)
    ingestor = _ingestor(tmp_path)
    first = ingestor.run(tickers=["MSFT"])
    second = ingestor.run(tickers=["MSFT"])
    assert first.fetched == 1
    assert second.skipped == 1
    assert second.fetched == 0
    assert second.parsed == 0
    assert len(httpx_mock.get_requests()) == 2


def test_parse_only_does_not_hit_http(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    cache = DiskCache(settings.raw_dir)
    cache.put_json(cache.submissions_path(MSFT_CIK), _submissions(MSFT_CIK))
    cache.put_html(ACCESSION, FIXTURE.read_bytes())
    client = EdgarClient(USER_AGENT, cache)
    result = Ingestor.from_config(
        settings.corpus_path, settings=settings, client=client
    ).run(tickers=["MSFT"], parse_only=True)
    assert result.parsed == 1
    assert result.fetched == 0
    assert httpx_mock.get_requests() == []


def test_parse_only_errors_when_html_missing(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    cache = DiskCache(settings.raw_dir)
    cache.put_json(cache.submissions_path(MSFT_CIK), _submissions(MSFT_CIK))
    result = Ingestor.from_config(
        settings.corpus_path,
        settings=settings,
        client=EdgarClient(USER_AGENT, cache),
    ).run(tickers=["MSFT"], parse_only=True)
    assert not result.ok
    assert result.parsed == 0
    assert "parse-only" in result.errors[0].message
    assert httpx_mock.get_requests() == []


def test_force_refetches_primary_doc(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    html = FIXTURE.read_bytes()
    httpx_mock.add_response(url=submissions_url(MSFT_CIK), json=_submissions(MSFT_CIK))
    httpx_mock.add_response(url=primary_doc_url(MSFT_CIK, ACCESSION, DOC), content=html)
    httpx_mock.add_response(url=submissions_url(MSFT_CIK), json=_submissions(MSFT_CIK))
    httpx_mock.add_response(url=primary_doc_url(MSFT_CIK, ACCESSION, DOC), content=html)
    ingestor = _ingestor(tmp_path)
    ingestor.run(tickers=["MSFT"])
    forced = ingestor.run(tickers=["MSFT"], force=True)
    assert forced.fetched == 1
    assert forced.parsed == 1
    assert forced.skipped == 0
    assert len(httpx_mock.get_requests()) == 4


def test_one_company_failure_does_not_stop_the_other(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    html = FIXTURE.read_bytes()
    httpx_mock.add_response(url=submissions_url(MSFT_CIK), json=_submissions(MSFT_CIK))
    httpx_mock.add_response(url=primary_doc_url(MSFT_CIK, ACCESSION, DOC), content=html)
    httpx_mock.add_response(url=submissions_url(JPM_CIK), status_code=404, text="missing")
    result = _ingestor(tmp_path).run(tickers=["MSFT", "JPM"])
    assert result.parsed == 1
    assert result.fetched == 1
    assert len(result.errors) == 1
    assert result.errors[0].ticker == "JPM"


def test_unknown_year_raises_before_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2020"):
        _ingestor(tmp_path).run(tickers=["MSFT"], fiscal_years=[2020])
