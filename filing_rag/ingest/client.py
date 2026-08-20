"""Rate-limited EDGAR HTTP client.

Fair-access rules: https://www.sec.gov/os/accessing-edgar-data
APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Self

import httpx

from filing_rag.ingest.cache import DiskCache, accession_key
from filing_rag.settings import Settings, get_settings

DATA_SEC_BASE = "https://data.sec.gov"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_RATE = 8.0
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_INITIAL_BACKOFF = 10.0
RETRYABLE_STATUS = {403, 429}


class EdgarHttpError(RuntimeError):
    def __init__(self, url: str, status_code: int, detail: str = "") -> None:
        self.url = url
        self.status_code = status_code
        message = f"EDGAR request failed ({status_code}) for {url}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class TokenBucket:
    """Pace requests at `rate` tokens per second, with an optional burst capacity."""

    def __init__(
        self,
        rate: float = DEFAULT_RATE,
        capacity: float | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = rate if capacity is None else capacity
        self._tokens = self.capacity
        self._updated_at = monotonic()
        self._sleep = sleep
        self._monotonic = monotonic

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            now = self._monotonic()
            elapsed = max(0.0, now - self._updated_at)
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._updated_at = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            wait = (tokens - self._tokens) / self.rate
            self._sleep(wait)


def pad_cik(cik: str | int) -> str:
    return str(cik).strip().zfill(10)


def unpad_cik(cik: str | int) -> str:
    return str(int(str(cik).strip()))


def submissions_url(cik: str | int) -> str:
    return f"{DATA_SEC_BASE}/submissions/CIK{pad_cik(cik)}.json"


def primary_doc_url(cik: str | int, accession: str, filename: str) -> str:
    return f"{ARCHIVES_BASE}/{unpad_cik(cik)}/{accession_key(accession)}/{filename}"


def extra_submissions_url(name: str) -> str:
    filename = name.rsplit("/", 1)[-1]
    return f"{DATA_SEC_BASE}/submissions/{filename}"


class EdgarClient:
    """The only module that talks to EDGAR. Everything else reads the disk cache."""

    def __init__(
        self,
        user_agent: str,
        cache: DiskCache,
        *,
        rate: float = DEFAULT_RATE,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        timeout: float = 30.0,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        agent = user_agent.strip()
        if not agent:
            raise ValueError(
                "EDGAR User-Agent is empty. Set EDGAR_USER_AGENT to a name and contact email."
            )
        self.user_agent = agent
        self.cache = cache
        self.max_attempts = max_attempts
        self.initial_backoff = initial_backoff
        self._sleep = sleep
        self._limiter = TokenBucket(rate, sleep=sleep)
        self._owns_http = http is None
        self._http = http or httpx.Client(
            headers={
                "User-Agent": agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=timeout,
            follow_redirects=True,
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None, **kwargs: Any) -> EdgarClient:
        resolved = settings or get_settings()
        return cls(
            user_agent=resolved.require_user_agent(),
            cache=DiskCache(resolved.raw_dir),
            **kwargs,
        )

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def submissions(
        self,
        cik: str | int,
        *,
        force: bool = False,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        path = self.cache.submissions_path(pad_cik(cik))
        if force or (cached := self.cache.get_json(path)) is None:
            payload = self._get_json(submissions_url(cik))
            self.cache.put_json(path, payload)
        else:
            payload = cached
        return self._with_older_filings(payload, force=force, since=since, until=until)

    def _with_older_filings(
        self,
        submissions: dict[str, Any],
        *,
        force: bool,
        since: str | None,
        until: str | None,
    ) -> dict[str, Any]:
        """Merge paginated `filings.files` pages into `filings.recent`.

        High-volume filers (banks) push 10-Ks out of the most-recent window.
        """
        files = submissions.get("filings", {}).get("files") or []
        if not files or (since is None and until is None):
            return submissions
        since = since or "0000-01-01"
        until = until or "9999-12-31"
        recent = {
            key: list(values) if isinstance(values, list) else values
            for key, values in submissions["filings"]["recent"].items()
        }
        for spec in files:
            if spec.get("filingTo", "0000-01-01") < since:
                continue
            if spec.get("filingFrom", "9999-12-31") > until:
                continue
            page = self._extra_submissions(spec["name"], force=force)
            for key, values in page.items():
                if not isinstance(values, list):
                    continue
                if key in recent and isinstance(recent[key], list):
                    recent[key].extend(values)
                else:
                    recent[key] = list(values)
        merged = dict(submissions)
        merged["filings"] = {**submissions["filings"], "recent": recent}
        return merged

    def _extra_submissions(self, name: str, *, force: bool) -> dict[str, Any]:
        path = self.cache.extra_submissions_path(name)
        if not force:
            cached = self.cache.get_json(path)
            if cached is not None:
                return cached
        payload = self._get_json(extra_submissions_url(name))
        self.cache.put_json(path, payload)
        return payload

    def primary_doc(
        self,
        cik: str | int,
        accession: str,
        filename: str,
        *,
        force: bool = False,
    ) -> bytes:
        if not force:
            cached = self.cache.get_html(accession)
            if cached is not None:
                return cached
        url = primary_doc_url(cik, accession, filename)
        body = self._get_bytes(url)
        self.cache.put_html(
            accession,
            body,
            meta={
                "url": url,
                "cik": pad_cik(cik),
                "accession": accession,
                "primary_doc": filename,
                "fetched_at": datetime.now(UTC).isoformat(),
            },
        )
        return body

    def _get_json(self, url: str) -> dict[str, Any]:
        payload = self._request(url).json()
        if not isinstance(payload, dict):
            raise EdgarHttpError(url, 200, "expected a JSON object")
        return payload

    def _get_bytes(self, url: str) -> bytes:
        return self._request(url).content

    def _request(self, url: str) -> httpx.Response:
        backoff = self.initial_backoff
        last_error: EdgarHttpError | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._limiter.acquire()
            response = self._http.get(url)
            if response.status_code in RETRYABLE_STATUS:
                last_error = EdgarHttpError(
                    url,
                    response.status_code,
                    "rate limit or missing User-Agent; backing off",
                )
                if attempt == self.max_attempts:
                    break
                self._sleep(backoff)
                backoff *= 2
                continue
            if response.is_error:
                raise EdgarHttpError(url, response.status_code, response.text[:300])
            return response
        assert last_error is not None
        raise last_error
