"""Read parsed filings and upsert chunk JSON under data/chunks/{strategy}/."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from filing_rag.chunking.types import ChunkedFiling
from filing_rag.ingest.cache import accession_key
from filing_rag.ingest.parse import ParsedFiling
from filing_rag.settings import get_settings


def parsed_path(parsed_dir: Path, accession: str) -> Path:
    return parsed_dir / f"{accession_key(accession)}.json"


def chunked_path(chunks_dir: Path, strategy: str, accession: str) -> Path:
    return chunks_dir / strategy / f"{accession_key(accession)}.json"


def load_parsed(parsed_dir: Path, accession: str) -> ParsedFiling:
    """Load one parsed filing. Missing files raise, they are not skipped."""
    path = parsed_path(parsed_dir, accession)
    if not path.exists():
        raise FileNotFoundError(f"parsed filing not found: {path}")
    return ParsedFiling.model_validate_json(path.read_text(encoding="utf-8"))


def iter_parsed(parsed_dir: Path) -> Iterator[ParsedFiling]:
    """Yield every parsed filing in `parsed_dir`, sorted by filename."""
    if not parsed_dir.exists():
        raise FileNotFoundError(f"parsed directory not found: {parsed_dir}")
    for path in sorted(parsed_dir.glob("*.json")):
        yield ParsedFiling.model_validate_json(path.read_text(encoding="utf-8"))


def chunked_exists(chunks_dir: Path, strategy: str, accession: str) -> bool:
    return chunked_path(chunks_dir, strategy, accession).exists()


def load_chunked(chunks_dir: Path, strategy: str, accession: str) -> ChunkedFiling:
    path = chunked_path(chunks_dir, strategy, accession)
    if not path.exists():
        raise FileNotFoundError(f"chunked filing not found: {path}")
    return ChunkedFiling.model_validate_json(path.read_text(encoding="utf-8"))


def write_chunked(chunked: ChunkedFiling, dest_dir: Path | None = None) -> Path:
    """Upsert `data/chunks/{strategy}/{accession_nodash}.json`."""
    directory = dest_dir if dest_dir is not None else get_settings().chunks_dir
    path = chunked_path(directory, chunked.strategy, chunked.accession)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chunked.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
