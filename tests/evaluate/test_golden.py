"""Golden-set YAML schema and quote checks."""

from pathlib import Path

import pytest
import yaml
from filing_rag.evaluate.golden import QuoteCheckError, load_golden
from filing_rag.evaluate.types import (
    EVAL_K,
    ITEM_CODES,
    NDCG_K,
    QUESTION_TYPES,
    RECALL_KS,
    GoldenSet,
    Question,
)
from filing_rag.ingest.parse import ParsedFiling, Section, write_parsed
from pydantic import ValidationError

ACCESSION = "0000950170-24-087843"
JPM_ACCESSION = "0000019617-24-000001"


def _question(
    *,
    id: str = "sh-001",
    type: str = "single_hop",
    query: str = "What cybersecurity risks does Microsoft disclose?",
    answerable: bool = True,
    answer: str = "Cybersecurity risk.",
    citations: list[dict] | None = None,
) -> dict:
    if citations is None and answerable:
        citations = [
            {
                "accession": ACCESSION,
                "item_code": "1A",
                "quote": "Cybersecurity risk could disrupt operations.",
            }
        ]
    return {
        "id": id,
        "type": type,
        "query": query,
        "answerable": answerable,
        "answer": answer,
        "citations": citations or [],
    }


def _golden(*questions: dict) -> GoldenSet:
    return GoldenSet.model_validate({"questions": list(questions)})


def _write_yaml(tmp_path: Path, *questions: dict) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(yaml.safe_dump({"questions": list(questions)}), encoding="utf-8")
    return path


def test_committed_golden_set_counts() -> None:
    golden = load_golden()
    assert len(golden.questions) == 50
    assert [q.type for q in golden.questions].count("single_hop") == 20
    assert [q.type for q in golden.questions].count("temporal") == 10
    assert [q.type for q in golden.questions].count("within_sector") == 10
    assert [q.type for q in golden.questions].count("cross_sector") == 5
    assert [q.type for q in golden.questions].count("unanswerable") == 5
    assert len(golden.answerable) == 45
    assert len(golden.unanswerable) == 5
    assert all(q.citations for q in golden.answerable)
    assert all(not q.citations for q in golden.unanswerable)


def test_eval_constants() -> None:
    assert QUESTION_TYPES == (
        "single_hop",
        "temporal",
        "within_sector",
        "cross_sector",
        "unanswerable",
    )
    assert ITEM_CODES == ("1A", "7", "7A")
    assert EVAL_K == 10
    assert RECALL_KS == (1, 5, 10)
    assert NDCG_K == 10


def test_load_golden_from_path(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        _question(),
        _question(
            id="ws-001",
            type="within_sector",
            query="How do Microsoft and JPMorgan describe interest-rate risk?",
            citations=[
                {
                    "accession": ACCESSION,
                    "item_code": "7A",
                    "quote": "Interest rate risk.",
                },
                {
                    "accession": JPM_ACCESSION,
                    "item_code": "7A",
                    "quote": "Market risk from rates.",
                },
            ],
        ),
        _question(
            id="ua-001",
            type="unanswerable",
            query="What did NVDA disclose about GPU supply in its FY2024 10-K?",
            answerable=False,
            answer="",
            citations=[],
        ),
    )
    golden = load_golden(path)
    assert [q.id for q in golden.questions] == ["sh-001", "ws-001", "ua-001"]
    assert len(golden.answerable) == 2
    assert len(golden.unanswerable) == 1
    assert golden.questions[0].citations[0].section_key == (ACCESSION, "1A")
    assert golden.questions[1].gold_sections == (
        (ACCESSION, "7A"),
        (JPM_ACCESSION, "7A"),
    )


def test_item_code_normalizes_and_rejects_unknown() -> None:
    question = Question.model_validate(_question(citations=[
        {"accession": ACCESSION, "item_code": "1a", "quote": "Cyber."}
    ]))
    assert question.citations[0].item_code == "1A"
    with pytest.raises(ValidationError, match="item_code"):
        Question.model_validate(
            _question(citations=[{"accession": ACCESSION, "item_code": "8", "quote": "x"}])
        )


def test_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(_question(type="multi_hop"))


def test_empty_query_rejected() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(_question(query="   "))


def test_answerable_requires_citations() -> None:
    with pytest.raises(ValidationError, match="at least one citation"):
        Question.model_validate(_question(citations=[]))


def test_unanswerable_rejects_citations() -> None:
    with pytest.raises(ValidationError, match="empty citations"):
        Question.model_validate(
            _question(
                id="ua-001",
                type="unanswerable",
                answerable=False,
                citations=[
                    {"accession": ACCESSION, "item_code": "1A", "quote": "Cyber."}
                ],
            )
        )


def test_unanswerable_must_not_be_answerable() -> None:
    with pytest.raises(ValidationError, match="answerable: false"):
        Question.model_validate(
            _question(id="ua-001", type="unanswerable", answerable=True, citations=[])
        )


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate question ids"):
        _golden(_question(), _question(id="sh-001", query="Another query."))


def test_gold_sections_dedupes_same_citation() -> None:
    question = Question.model_validate(
        _question(
            citations=[
                {"accession": ACCESSION, "item_code": "1A", "quote": "First span."},
                {"accession": ACCESSION, "item_code": "1A", "quote": "Second span."},
            ]
        )
    )
    assert question.gold_sections == ((ACCESSION, "1A"),)


def test_check_quotes_noop_when_parsed_dir_missing(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, _question())
    golden = load_golden(path, parsed_dir=tmp_path / "parsed")
    assert len(golden.questions) == 1


def test_check_quotes_accepts_substring(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed"
    write_parsed(
        ParsedFiling(
            ticker="MSFT",
            cik="0000789019",
            accession=ACCESSION,
            form="10-K",
            filing_date="2024-07-30",
            period_of_report="2024-06-30",
            fiscal_year=2024,
            primary_doc="msft-20240630.htm",
            edgar_url="https://example.com",
            sections=[
                Section(
                    item_code="1A",
                    item_title="Risk Factors",
                    text="Cybersecurity risk could disrupt operations. More prose.",
                    char_start=0,
                    char_end=60,
                )
            ],
        ),
        parsed_dir,
    )
    path = _write_yaml(tmp_path, _question())
    golden = load_golden(path, parsed_dir=parsed_dir)
    assert golden.questions[0].id == "sh-001"


def test_check_quotes_rejects_missing_span(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed"
    write_parsed(
        ParsedFiling(
            ticker="MSFT",
            cik="0000789019",
            accession=ACCESSION,
            form="10-K",
            filing_date="2024-07-30",
            period_of_report="2024-06-30",
            fiscal_year=2024,
            primary_doc="msft-20240630.htm",
            edgar_url="https://example.com",
            sections=[
                Section(
                    item_code="1A",
                    item_title="Risk Factors",
                    text="Unrelated risk factor body.",
                    char_start=0,
                    char_end=27,
                )
            ],
        ),
        parsed_dir,
    )
    path = _write_yaml(tmp_path, _question())
    with pytest.raises(QuoteCheckError, match="quote not in"):
        load_golden(path, parsed_dir=parsed_dir)


def test_check_quotes_rejects_missing_parsed_filing(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    path = _write_yaml(tmp_path, _question())
    with pytest.raises(QuoteCheckError, match="parsed filing not found"):
        load_golden(path, parsed_dir=parsed_dir)
