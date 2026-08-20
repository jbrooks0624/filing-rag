"""EdgarClient sends a User-Agent, caches hits, and backs off on 403/429."""

from pathlib import Path

import pytest
from filing_rag.ingest.cache import DiskCache
from filing_rag.ingest.client import (
    EdgarClient,
    EdgarHttpError,
    TokenBucket,
    extra_submissions_url,
    primary_doc_url,
    submissions_url,
)
from filing_rag.settings import Settings
from pytest_httpx import HTTPXMock

USER_AGENT = "filing-rag test@example.com"
CIK = "0000789019"
ACCESSION = "0000789019-24-000000"
PRIMARY_DOC = "msft-10k.htm"
SUBMISSIONS = submissions_url(CIK)
PRIMARY = primary_doc_url(CIK, ACCESSION, PRIMARY_DOC)


def _client(tmp_path: Path, **kwargs: object) -> EdgarClient:
    return EdgarClient(USER_AGENT, DiskCache(tmp_path), **kwargs)


def test_submissions_url_pads_cik() -> None:
    assert SUBMISSIONS == "https://data.sec.gov/submissions/CIK0000789019.json"


def test_primary_doc_url_uses_unpadded_cik() -> None:
    assert PRIMARY == (
        "https://www.sec.gov/Archives/edgar/data/789019/000078901924000000/msft-10k.htm"
    )


def test_submissions_sends_user_agent(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    httpx_mock.add_response(url=SUBMISSIONS, json={"cik": CIK})
    with _client(tmp_path) as client:
        payload = client.submissions(CIK)
    assert payload == {"cik": CIK}
    request = httpx_mock.get_request()
    assert request.headers["User-Agent"] == USER_AGENT


def test_submissions_skips_http_on_cache_hit(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put_json(cache.submissions_path(CIK), {"cik": CIK, "cached": True})
    with EdgarClient(USER_AGENT, cache) as client:
        payload = client.submissions(CIK)
    assert payload == {"cik": CIK, "cached": True}
    assert httpx_mock.get_requests() == []


def test_primary_doc_skips_http_on_cache_hit(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put_html(ACCESSION, b"<html>cached</html>")
    with EdgarClient(USER_AGENT, cache) as client:
        body = client.primary_doc(CIK, ACCESSION, PRIMARY_DOC)
    assert body == b"<html>cached</html>"
    assert httpx_mock.get_requests() == []


def test_primary_doc_writes_cache_on_fetch(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    httpx_mock.add_response(url=PRIMARY, content=b"<html>live</html>")
    cache = DiskCache(tmp_path)
    with EdgarClient(USER_AGENT, cache) as client:
        body = client.primary_doc(CIK, ACCESSION, PRIMARY_DOC)
    assert body == b"<html>live</html>"
    assert cache.get_html(ACCESSION) == b"<html>live</html>"
    assert cache.meta_path(ACCESSION).exists()


def test_force_refetch_ignores_cache(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put_html(ACCESSION, b"<html>stale</html>")
    httpx_mock.add_response(url=PRIMARY, content=b"<html>fresh</html>")
    with EdgarClient(USER_AGENT, cache) as client:
        body = client.primary_doc(CIK, ACCESSION, PRIMARY_DOC, force=True)
    assert body == b"<html>fresh</html>"
    assert cache.get_html(ACCESSION) == b"<html>fresh</html>"


def test_retries_on_403_then_succeeds(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    sleeps: list[float] = []
    httpx_mock.add_response(url=SUBMISSIONS, status_code=403)
    httpx_mock.add_response(url=SUBMISSIONS, json={"cik": CIK})
    with _client(tmp_path, sleep=sleeps.append) as client:
        payload = client.submissions(CIK)
    assert payload == {"cik": CIK}
    assert sleeps == [10.0]


def test_retries_exhausted_raise(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    sleeps: list[float] = []
    httpx_mock.add_response(url=SUBMISSIONS, status_code=429)
    httpx_mock.add_response(url=SUBMISSIONS, status_code=429)
    httpx_mock.add_response(url=SUBMISSIONS, status_code=429)
    with _client(tmp_path, sleep=sleeps.append, max_attempts=3) as client:
        with pytest.raises(EdgarHttpError) as exc_info:
            client.submissions(CIK)
    assert exc_info.value.status_code == 429
    assert sleeps == [10.0, 20.0]


def test_non_retryable_error_raises_immediately(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    httpx_mock.add_response(url=SUBMISSIONS, status_code=404, text="missing")
    with _client(tmp_path) as client:
        with pytest.raises(EdgarHttpError) as exc_info:
            client.submissions(CIK)
    assert exc_info.value.status_code == 404


def test_empty_user_agent_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="User-Agent"):
        EdgarClient("  ", DiskCache(tmp_path))


def test_token_bucket_sleeps_when_empty() -> None:
    sleeps: list[float] = []
    clock = {"now": 0.0}

    def monotonic() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    bucket = TokenBucket(rate=8.0, capacity=1.0, sleep=sleep, monotonic=monotonic)
    bucket.acquire()
    bucket.acquire()
    assert sleeps == [pytest.approx(0.125)]


def test_from_settings_requires_user_agent(tmp_path: Path) -> None:
    settings = Settings(edgar_user_agent="", data_dir=tmp_path)
    with pytest.raises(ValueError, match="EDGAR_USER_AGENT"):
        EdgarClient.from_settings(settings)


def test_submissions_merges_overlapping_extra_pages(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    extra_name = "CIK0000789019-submissions-001.json"
    httpx_mock.add_response(
        url=SUBMISSIONS,
        json={
            "cik": CIK,
            "filings": {
                "recent": {
                    "accessionNumber": ["new"],
                    "filingDate": ["2024-07-30"],
                    "reportDate": ["2024-06-30"],
                    "form": ["10-K"],
                    "primaryDocument": ["new.htm"],
                },
                "files": [
                    {
                        "name": extra_name,
                        "filingFrom": "2022-01-01",
                        "filingTo": "2023-06-01",
                        "filingCount": 1,
                    },
                    {
                        "name": "too-old.json",
                        "filingFrom": "2010-01-01",
                        "filingTo": "2015-01-01",
                        "filingCount": 1,
                    },
                ],
            },
        },
    )
    httpx_mock.add_response(
        url=extra_submissions_url(extra_name),
        json={
            "accessionNumber": ["old"],
            "filingDate": ["2023-07-28"],
            "reportDate": ["2022-06-30"],
            "form": ["10-K"],
            "primaryDocument": ["old.htm"],
        },
    )
    with _client(tmp_path) as client:
        payload = client.submissions(CIK, since="2022-01-01", until="2024-12-31")
    recent = payload["filings"]["recent"]
    assert recent["accessionNumber"] == ["new", "old"]
    assert recent["reportDate"] == ["2024-06-30", "2022-06-30"]
    requested = [str(request.url) for request in httpx_mock.get_requests()]
    assert extra_submissions_url(extra_name) in requested
    assert extra_submissions_url("too-old.json") not in requested
