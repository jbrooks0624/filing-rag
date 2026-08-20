"""Disk cache for raw EDGAR responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def accession_key(accession: str) -> str:
    """Strip dashes from an accession number for filesystem and archive URLs."""
    return accession.replace("-", "").strip()


class DiskCache:
    """Store primary documents as `{accession_nodash}.html` plus a sidecar meta file."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def html_path(self, accession: str) -> Path:
        return self.root / f"{accession_key(accession)}.html"

    def meta_path(self, accession: str) -> Path:
        return self.root / f"{accession_key(accession)}.meta.json"

    def submissions_path(self, cik: str) -> Path:
        padded = str(cik).strip().zfill(10)
        directory = self.root / "submissions"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"CIK{padded}.json"

    def extra_submissions_path(self, name: str) -> Path:
        directory = self.root / "submissions"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / Path(name).name

    def get_html(self, accession: str) -> bytes | None:
        path = self.html_path(accession)
        if not path.exists():
            return None
        return path.read_bytes()

    def put_html(self, accession: str, body: bytes, meta: dict[str, Any] | None = None) -> Path:
        path = self.html_path(accession)
        path.write_bytes(body)
        payload = {"bytes": len(body), **(meta or {})}
        self.meta_path(accession).write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def get_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"cached JSON at {path} is not an object")
        return payload

    def put_json(self, path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path
