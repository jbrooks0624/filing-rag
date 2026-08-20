"""Callable EDGAR ingest package."""

from filing_rag.ingest.cache import DiskCache, accession_key
from filing_rag.ingest.catalog import CatalogError, FilingRef, select_filings
from filing_rag.ingest.client import EdgarClient, EdgarHttpError
from filing_rag.ingest.parse import ParsedFiling, ParseError, parse_html, write_parsed
from filing_rag.ingest.pipeline import Ingestor, IngestResult

__all__ = [
    "CatalogError",
    "DiskCache",
    "EdgarClient",
    "EdgarHttpError",
    "FilingRef",
    "Ingestor",
    "IngestResult",
    "ParsedFiling",
    "ParseError",
    "accession_key",
    "parse_html",
    "select_filings",
    "write_parsed",
]
