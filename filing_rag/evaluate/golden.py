"""Load and validate eval/golden_set.yaml."""

from pathlib import Path

import yaml

from filing_rag.chunking.store import load_parsed, parsed_path
from filing_rag.evaluate.types import GoldenSet
from filing_rag.settings import get_settings


class QuoteCheckError(ValueError):
    """A gold quote is missing from the cited parsed section."""


def load_golden(
    path: Path | None = None,
    *,
    parsed_dir: Path | None = None,
) -> GoldenSet:
    golden_path = path if path is not None else get_settings().golden_path
    with golden_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        payload = {}
    golden = GoldenSet.model_validate(payload)
    if parsed_dir is not None:
        check_quotes(golden, parsed_dir)
    return golden


def check_quotes(golden: GoldenSet, parsed_dir: Path) -> None:
    """Require each citation quote to appear in the cited section text.

    No-op when ``parsed_dir`` does not exist yet (ingest has not been run).
    """
    if not parsed_dir.exists():
        return
    errors: list[str] = []
    for question in golden.questions:
        for citation in question.citations:
            path = parsed_path(parsed_dir, citation.accession)
            if not path.exists():
                errors.append(
                    f"{question.id}: parsed filing not found for {citation.accession}"
                )
                continue
            filing = load_parsed(parsed_dir, citation.accession)
            section = next(
                (item for item in filing.sections if item.item_code == citation.item_code),
                None,
            )
            if section is None:
                errors.append(
                    f"{question.id}: Item {citation.item_code} missing in {citation.accession}"
                )
                continue
            if citation.quote not in section.text:
                errors.append(
                    f"{question.id}: quote not in {citation.accession} Item {citation.item_code}"
                )
    if errors:
        raise QuoteCheckError("\n".join(errors))
