"""Golden-set questions. Citations are section-level, not chunk ids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from filing_rag.evaluate.ragas import RagasScores

QUESTION_TYPES = (
    "single_hop",
    "temporal",
    "within_sector",
    "cross_sector",
    "unanswerable",
)
ITEM_CODES = ("1A", "7", "7A")
EVAL_K = 10
RECALL_KS = (1, 5, 10)
NDCG_K = 10

QuestionType = Literal[
    "single_hop",
    "temporal",
    "within_sector",
    "cross_sector",
    "unanswerable",
]


class Citation(BaseModel):
    """Ground-truth section. Matching uses accession + item_code, not the quote."""

    model_config = ConfigDict(frozen=True)

    accession: str = Field(min_length=1)
    item_code: str
    quote: str = Field(min_length=1)

    @field_validator("accession", "quote", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("item_code", mode="before")
    @classmethod
    def normalize_item(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("item_code")
    @classmethod
    def known_item(cls, value: str) -> str:
        if value not in ITEM_CODES:
            known = ", ".join(ITEM_CODES)
            raise ValueError(f"unknown item_code {value!r}. Known: {known}")
        return value

    @property
    def section_key(self) -> tuple[str, str]:
        return (self.accession, self.item_code)


class Question(BaseModel):
    """One eval query. Unanswerable questions have no citations."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    type: QuestionType
    query: str = Field(min_length=1)
    answerable: bool
    answer: str = ""
    citations: tuple[Citation, ...] = ()

    @field_validator("id", "query", "answer", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def citations_match_answerable(self) -> Question:
        if self.type == "unanswerable":
            if self.answerable:
                raise ValueError(f"{self.id}: unanswerable questions must set answerable: false")
            if self.citations:
                raise ValueError(f"{self.id}: unanswerable questions must have empty citations")
            return self
        if not self.answerable:
            raise ValueError(f"{self.id}: {self.type} questions must set answerable: true")
        if not self.citations:
            raise ValueError(f"{self.id}: answerable questions need at least one citation")
        return self

    @property
    def gold_sections(self) -> tuple[tuple[str, str], ...]:
        seen: list[tuple[str, str]] = []
        for citation in self.citations:
            key = citation.section_key
            if key not in seen:
                seen.append(key)
        return tuple(seen)


class GoldenSet(BaseModel):
    """Root of eval/golden_set.yaml."""

    model_config = ConfigDict(frozen=True)

    questions: tuple[Question, ...] = ()

    @model_validator(mode="after")
    def unique_ids(self) -> GoldenSet:
        seen: set[str] = set()
        dupes: list[str] = []
        for question in self.questions:
            if question.id in seen:
                dupes.append(question.id)
            seen.add(question.id)
        if dupes:
            raise ValueError(f"duplicate question ids: {', '.join(dupes)}")
        return self

    @property
    def answerable(self) -> tuple[Question, ...]:
        return tuple(q for q in self.questions if q.answerable)

    @property
    def unanswerable(self) -> tuple[Question, ...]:
        return tuple(q for q in self.questions if not q.answerable)


SectionKey = tuple[str, str]


@dataclass(frozen=True)
class QueryScore:
    """Per-query metrics. Quality fields are None for unanswerable questions."""

    question_id: str
    answerable: bool
    gold_sections: tuple[SectionKey, ...]
    retrieved_sections: tuple[SectionKey, ...]
    recall_at_1: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    mrr: float | None
    ndcg_at_10: float | None


@dataclass(frozen=True)
class MeanScore:
    """Macro-average over answerable QueryScores."""

    n: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


@dataclass(frozen=True)
class GridConfig:
    """One cell of the 18-config ablation grid."""

    strategy: str
    mode: str
    rerank: bool

    @property
    def label(self) -> str:
        flag = "rerank" if self.rerank else "off"
        return f"{self.strategy}/{self.mode}/{flag}"


@dataclass(frozen=True)
class QueryRow:
    """One JSONL record: one question scored under one grid config."""

    strategy: str
    mode: str
    rerank: bool
    question_id: str
    question_type: str
    query: str
    answerable: bool
    gold_sections: tuple[SectionKey, ...]
    retrieved_sections: tuple[SectionKey, ...]
    recall_at_1: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    mrr: float | None
    ndcg_at_10: float | None
    encode_ms: float
    dense_ms: float
    sparse_ms: float
    fuse_ms: float
    rerank_ms: float
    total_ms: float


@dataclass(frozen=True)
class ConfigReport:
    """Macro quality plus latency for one grid config."""

    strategy: str
    mode: str
    rerank: bool
    scores: MeanScore | None
    p50_ms: float
    p95_ms: float
    n_questions: int
    n_unanswerable: int


@dataclass(frozen=True)
class EvalResult:
    """Full Stage 1 run: per-config reports, raw rows, and the JSONL path."""

    reports: tuple[ConfigReport, ...] = ()
    rows: tuple[QueryRow, ...] = ()
    output: Path | None = None

    @property
    def n_configs(self) -> int:
        return len(self.reports)

    @property
    def n_questions(self) -> int:
        return len({row.question_id for row in self.rows})

    @property
    def n_unanswerable(self) -> int:
        return len({row.question_id for row in self.rows if not row.answerable})


@dataclass(frozen=True)
class RagQueryRow:
    """One JSONL record: generate + RAGAS under one retrieval config."""

    strategy: str
    mode: str
    rerank: bool
    question_id: str
    question_type: str
    query: str
    answerable: bool
    response: str
    refused: bool
    retrieved_sections: tuple[SectionKey, ...]
    faithfulness: float | None
    context_precision: float | None
    context_recall: float | None
    relevancy: float | None
    judge_ms: float | None
    judge_usd: float | None
    encode_ms: float
    dense_ms: float
    sparse_ms: float
    fuse_ms: float
    rerank_ms: float
    total_ms: float
    generate_ms: float
    serving_ms: float
    prompt_tokens: int
    completion_tokens: int
    usd: float


@dataclass(frozen=True)
class RagConfigReport:
    """Macro RAGAS, refusal, serving latency, and generation dollars for one config."""

    strategy: str
    mode: str
    rerank: bool
    ragas: RagasScores | None
    refusal_rate: float | None
    p50_ms: float
    p95_ms: float
    usd: float
    n_questions: int
    n_unanswerable: int


@dataclass(frozen=True)
class RagEvalResult:
    """Full Stage 2 run: per-config reports, raw rows, and the JSONL path."""

    strategy: str = ""
    reports: tuple[RagConfigReport, ...] = ()
    rows: tuple[RagQueryRow, ...] = ()
    output: Path | None = None

    @property
    def n_configs(self) -> int:
        return len(self.reports)

    @property
    def n_questions(self) -> int:
        return len({row.question_id for row in self.rows})

    @property
    def n_unanswerable(self) -> int:
        return len({row.question_id for row in self.rows if not row.answerable})

    @property
    def refusal_rate(self) -> float | None:
        refused = [row.refused for row in self.rows if not row.answerable]
        if not refused:
            return None
        return sum(1 for item in refused if item) / len(refused)
