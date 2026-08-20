"""Disk cache round-trip for primary documents and submissions JSON."""

import json
from pathlib import Path

from filing_rag.ingest.cache import DiskCache, accession_key


def test_accession_key_strips_dashes() -> None:
    assert accession_key("0000789019-24-000000") == "000078901924000000"
    assert accession_key("000078901924000000") == "000078901924000000"


def test_html_round_trip_writes_sidecar_meta(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put_html(
        "0000789019-24-000000",
        b"<html>msft</html>",
        meta={"url": "https://www.sec.gov/example"},
    )
    assert cache.get_html("000078901924000000") == b"<html>msft</html>"
    meta = json.loads(cache.meta_path("0000789019-24-000000").read_text())
    assert meta["bytes"] == 17
    assert meta["url"] == "https://www.sec.gov/example"


def test_submissions_json_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    path = cache.submissions_path("789019")
    cache.put_json(path, {"cik": "0000789019"})
    assert path == tmp_path / "submissions" / "CIK0000789019.json"
    assert cache.get_json(path) == {"cik": "0000789019"}
