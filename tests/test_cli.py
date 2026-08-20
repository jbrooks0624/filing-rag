"""CLI maps flags onto ingest / chunk / index / search / generate / eval-retrieval / eval-rag."""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from filing_rag.chunking.pipeline import ChunkError, ChunkResult
from filing_rag.cli import app
from filing_rag.embed.pipeline import IndexingError, IndexResult
from filing_rag.embed.store import IndexStat
from filing_rag.evaluate.ragas import RagasScores
from filing_rag.evaluate.types import (
    ConfigReport,
    EvalResult,
    MeanScore,
    QueryRow,
    RagConfigReport,
    RagEvalResult,
    RagQueryRow,
)
from filing_rag.generate.types import AskResult, GenerateResult, GenerateTimings, Usage
from filing_rag.ingest.pipeline import IngestError, IngestResult
from filing_rag.retrieve.types import Filters, Hit, RetrieveResult, RetrieveTimings
from typer.testing import CliRunner

runner = CliRunner()


@dataclass
class FakeIngestor:
    captured: dict = field(default_factory=dict)
    result: IngestResult = field(default_factory=IngestResult)

    @classmethod
    def from_config(cls) -> FakeIngestor:
        return instances[0]

    def run(
        self,
        tickers=None,
        fiscal_years=None,
        *,
        force=False,
        parse_only=False,
    ) -> IngestResult:
        self.captured.update(
            tickers=tickers,
            fiscal_years=fiscal_years,
            force=force,
            parse_only=parse_only,
        )
        return self.result


instances: list[FakeIngestor] = []


def _install(monkeypatch, result: IngestResult | None = None) -> FakeIngestor:
    fake = FakeIngestor(result=result or IngestResult(fetched=1, parsed=1))
    instances.clear()
    instances.append(fake)
    monkeypatch.setattr("filing_rag.cli.Ingestor", FakeIngestor)
    return fake


def test_ingest_help() -> None:
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "--ticker" in result.stdout
    assert "--year" in result.stdout
    assert "--force" in result.stdout
    assert "--parse-only" in result.stdout


def test_ingest_passes_flags(monkeypatch) -> None:
    fake = _install(monkeypatch)
    result = runner.invoke(
        app,
        ["ingest", "--ticker", "MSFT", "--ticker", "GOOGL", "--year", "2024", "--force"],
    )
    assert result.exit_code == 0
    assert fake.captured == {
        "tickers": ["MSFT", "GOOGL"],
        "fiscal_years": [2024],
        "force": True,
        "parse_only": False,
    }
    assert "fetched=1 skipped=0 parsed=1 errors=0" in result.stdout


def test_ingest_parse_only(monkeypatch) -> None:
    fake = _install(monkeypatch, IngestResult(parsed=1))
    result = runner.invoke(app, ["ingest", "--parse-only"])
    assert result.exit_code == 0
    assert fake.captured["parse_only"] is True
    assert fake.captured["tickers"] is None


def test_ingest_exits_one_on_errors(monkeypatch) -> None:
    _install(
        monkeypatch,
        IngestResult(errors=[IngestError(message="boom", ticker="JPM", fiscal_year=2024)]),
    )
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "errors=1" in result.stdout
    assert "JPM" in result.output
    assert "boom" in result.output


def test_ingest_exits_one_on_unknown_ticker(monkeypatch) -> None:
    class Boom(FakeIngestor):
        def run(self, *args, **kwargs) -> IngestResult:
            raise KeyError("Unknown ticker 'NVDA'")

    instances.clear()
    instances.append(Boom())
    monkeypatch.setattr("filing_rag.cli.Ingestor", Boom)
    result = runner.invoke(app, ["ingest", "--ticker", "NVDA"])
    assert result.exit_code == 1
    assert "NVDA" in result.output


@dataclass
class FakeChunker:
    captured: dict = field(default_factory=dict)
    result: ChunkResult = field(default_factory=ChunkResult)

    @classmethod
    def from_config(cls) -> FakeChunker:
        return chunkers[0]

    def run(
        self,
        tickers=None,
        fiscal_years=None,
        strategies=None,
        *,
        force=False,
    ) -> ChunkResult:
        self.captured.update(
            tickers=tickers,
            fiscal_years=fiscal_years,
            strategies=strategies,
            force=force,
        )
        return self.result


chunkers: list[FakeChunker] = []


def _install_chunker(monkeypatch, result: ChunkResult | None = None) -> FakeChunker:
    fake = FakeChunker(result=result or ChunkResult(chunked=1, chunks_written=4))
    chunkers.clear()
    chunkers.append(fake)
    monkeypatch.setattr("filing_rag.cli.Chunker", FakeChunker)
    return fake


def test_chunk_help() -> None:
    result = runner.invoke(app, ["chunk", "--help"])
    assert result.exit_code == 0
    assert "--ticker" in result.stdout
    assert "--year" in result.stdout
    assert "--strategy" in result.stdout
    assert "--force" in result.stdout


def test_chunk_passes_flags(monkeypatch) -> None:
    fake = _install_chunker(monkeypatch)
    result = runner.invoke(
        app,
        [
            "chunk",
            "--ticker",
            "MSFT",
            "--year",
            "2024",
            "--strategy",
            "fixed",
            "--strategy",
            "structural",
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert fake.captured == {
        "tickers": ["MSFT"],
        "fiscal_years": [2024],
        "strategies": ["fixed", "structural"],
        "force": True,
    }
    assert "chunked=1 skipped=0 chunks=4 errors=0" in result.stdout


def test_chunk_defaults_pass_none(monkeypatch) -> None:
    fake = _install_chunker(monkeypatch)
    result = runner.invoke(app, ["chunk"])
    assert result.exit_code == 0
    assert fake.captured == {
        "tickers": None,
        "fiscal_years": None,
        "strategies": None,
        "force": False,
    }


def test_chunk_exits_one_on_errors(monkeypatch) -> None:
    _install_chunker(
        monkeypatch,
        ChunkResult(
            errors=[
                ChunkError(
                    message="parsed filing not found",
                    ticker="JPM",
                    fiscal_year=2024,
                    strategy="fixed",
                )
            ]
        ),
    )
    result = runner.invoke(app, ["chunk"])
    assert result.exit_code == 1
    assert "errors=1" in result.stdout
    assert "JPM" in result.output
    assert "parsed filing not found" in result.output


def test_chunk_exits_one_on_unknown_strategy(monkeypatch) -> None:
    class Boom(FakeChunker):
        def run(self, *args, **kwargs) -> ChunkResult:
            raise ValueError(
                "unknown strategies: ['recursive']. Known: fixed, structural, semantic"
            )

    chunkers.clear()
    chunkers.append(Boom())
    monkeypatch.setattr("filing_rag.cli.Chunker", Boom)
    result = runner.invoke(app, ["chunk", "--strategy", "recursive"])
    assert result.exit_code == 1
    assert "recursive" in result.output


@dataclass
class FakeIndexer:
    captured: dict = field(default_factory=dict)
    result: IndexResult = field(default_factory=IndexResult)

    @classmethod
    def from_config(cls) -> FakeIndexer:
        return indexers[0]

    def run(
        self,
        tickers=None,
        fiscal_years=None,
        strategies=None,
        *,
        force=False,
    ) -> IndexResult:
        self.captured.update(
            tickers=tickers,
            fiscal_years=fiscal_years,
            strategies=strategies,
            force=force,
        )
        return self.result


indexers: list[FakeIndexer] = []


def _install_indexer(monkeypatch, result: IndexResult | None = None) -> FakeIndexer:
    fake = FakeIndexer(
        result=result
        or IndexResult(
            indexed=1,
            embedded=12,
            index_stats=[
                IndexStat(
                    strategy="fixed",
                    name="chunks_hnsw_fixed",
                    bytes=2048,
                    build_ms=12.4,
                )
            ],
        )
    )
    indexers.clear()
    indexers.append(fake)
    monkeypatch.setattr("filing_rag.cli.Indexer", FakeIndexer)
    return fake


def test_index_help() -> None:
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "--ticker" in result.stdout
    assert "--year" in result.stdout
    assert "--strategy" in result.stdout
    assert "--force" in result.stdout


def test_index_passes_flags(monkeypatch) -> None:
    fake = _install_indexer(monkeypatch)
    result = runner.invoke(
        app,
        [
            "index",
            "--ticker",
            "MSFT",
            "--year",
            "2024",
            "--strategy",
            "fixed",
            "--strategy",
            "structural",
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert fake.captured == {
        "tickers": ["MSFT"],
        "fiscal_years": [2024],
        "strategies": ["fixed", "structural"],
        "force": True,
    }
    assert "indexed=1 skipped=0 embedded=12 errors=0" in result.stdout
    assert "hnsw fixed=2.0KiB 12ms" in result.stdout


def test_index_defaults_pass_none(monkeypatch) -> None:
    fake = _install_indexer(monkeypatch)
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 0
    assert fake.captured == {
        "tickers": None,
        "fiscal_years": None,
        "strategies": None,
        "force": False,
    }


def test_index_exits_one_on_errors(monkeypatch) -> None:
    _install_indexer(
        monkeypatch,
        IndexResult(
            errors=[
                IndexingError(
                    message="chunked filing not found",
                    ticker="JPM",
                    fiscal_year=2024,
                    strategy="fixed",
                )
            ]
        ),
    )
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 1
    assert "errors=1" in result.stdout
    assert "JPM" in result.output
    assert "chunked filing not found" in result.output


def test_index_exits_one_on_unknown_strategy(monkeypatch) -> None:
    class Boom(FakeIndexer):
        def run(self, *args, **kwargs) -> IndexResult:
            raise ValueError(
                "unknown strategies: ['recursive']. Known: fixed, structural, semantic"
            )

    indexers.clear()
    indexers.append(Boom())
    monkeypatch.setattr("filing_rag.cli.Indexer", Boom)
    result = runner.invoke(app, ["index", "--strategy", "recursive"])
    assert result.exit_code == 1
    assert "recursive" in result.output


@dataclass
class FakeRetriever:
    captured: dict = field(default_factory=dict)
    result: RetrieveResult = field(default_factory=RetrieveResult)

    @classmethod
    def from_config(cls) -> FakeRetriever:
        return retrievers[0]

    def search(
        self,
        query,
        *,
        strategy,
        mode="hybrid",
        k=None,
        rerank=False,
        filters=None,
        force=False,
    ) -> RetrieveResult:
        self.captured.update(
            query=query,
            strategy=strategy,
            mode=mode,
            k=k,
            rerank=rerank,
            filters=filters,
            force=force,
        )
        return self.result


retrievers: list[FakeRetriever] = []


def _install_retriever(monkeypatch, result: RetrieveResult | None = None) -> FakeRetriever:
    fake = FakeRetriever(
        result=result
        or RetrieveResult(
            hits=(
                Hit(
                    chunk_id=1,
                    score=0.91,
                    rank=1,
                    text="Cybersecurity risk could disrupt operations.",
                    ticker="MSFT",
                    fiscal_year=2024,
                    item_code="1A",
                    accession="0000950170-24-087843",
                    char_start=0,
                    char_end=44,
                    edgar_url="https://example.com",
                    strategy="fixed",
                    chunk_index=0,
                ),
            ),
            mode="hybrid",
            strategy="fixed",
            reranked=True,
            timings=RetrieveTimings(
                encode_ms=12.0,
                dense_ms=4.0,
                sparse_ms=3.0,
                fuse_ms=1.0,
                rerank_ms=80.0,
            ),
        )
    )
    retrievers.clear()
    retrievers.append(fake)
    monkeypatch.setattr("filing_rag.cli.Retriever", FakeRetriever)
    return fake


def test_search_help() -> None:
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "--strategy" in result.stdout
    assert "--mode" in result.stdout
    assert "--rerank" in result.stdout
    assert "--k" in result.stdout
    assert "--ticker" in result.stdout
    assert "--year" in result.stdout
    assert "--item" in result.stdout
    assert "--force" in result.stdout


def test_search_requires_strategy() -> None:
    result = runner.invoke(app, ["search", "cybersecurity risk"])
    assert result.exit_code != 0


def test_search_passes_flags(monkeypatch) -> None:
    fake = _install_retriever(monkeypatch)
    result = runner.invoke(
        app,
        [
            "search",
            "cybersecurity risk",
            "--strategy",
            "fixed",
            "--mode",
            "hybrid",
            "--rerank",
            "--k",
            "5",
            "--ticker",
            "MSFT",
            "--year",
            "2024",
            "--item",
            "1A",
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert fake.captured["query"] == "cybersecurity risk"
    assert fake.captured["strategy"] == "fixed"
    assert fake.captured["mode"] == "hybrid"
    assert fake.captured["k"] == 5
    assert fake.captured["rerank"] is True
    assert fake.captured["force"] is True
    assert fake.captured["filters"] == Filters(
        tickers=("MSFT",),
        fiscal_years=(2024,),
        item_codes=("1A",),
    )
    assert "hits=1 mode=hybrid strategy=fixed rerank=on" in result.stdout
    assert "encode=12ms dense=4ms sparse=3ms fuse=1ms rerank=80ms" in result.stdout
    assert "1 0.9100 MSFT FY2024 1A 0000950170-24-087843" in result.stdout
    assert "Cybersecurity risk" in result.stdout


def test_search_defaults(monkeypatch) -> None:
    fake = _install_retriever(monkeypatch)
    result = runner.invoke(
        app,
        ["search", "interest rates", "--strategy", "structural"],
    )
    assert result.exit_code == 0
    assert fake.captured == {
        "query": "interest rates",
        "strategy": "structural",
        "mode": "hybrid",
        "k": None,
        "rerank": False,
        "filters": None,
        "force": False,
    }


def test_search_exits_one_on_unknown_mode(monkeypatch) -> None:
    class Boom(FakeRetriever):
        def search(self, *args, **kwargs) -> RetrieveResult:
            raise ValueError("unknown mode 'keyword'. Known: dense, sparse, hybrid")

    retrievers.clear()
    retrievers.append(Boom())
    monkeypatch.setattr("filing_rag.cli.Retriever", Boom)
    result = runner.invoke(
        app,
        ["search", "cyber", "--strategy", "fixed", "--mode", "keyword"],
    )
    assert result.exit_code == 1
    assert "keyword" in result.output


@dataclass
class FakeEvaluator:
    captured: dict = field(default_factory=dict)
    result: EvalResult = field(default_factory=EvalResult)

    @classmethod
    def from_config(cls) -> FakeEvaluator:
        return evaluators[0]

    def run(
        self,
        *,
        strategies=None,
        modes=None,
        rerank=None,
        k=None,
        force=False,
        output=None,
    ) -> EvalResult:
        self.captured.update(
            strategies=strategies,
            modes=modes,
            rerank=rerank,
            k=k,
            force=force,
            output=output,
        )
        return self.result


evaluators: list[FakeEvaluator] = []


def _eval_result() -> EvalResult:
    rows = (
        QueryRow(
            strategy="fixed",
            mode="dense",
            rerank=False,
            question_id="sh-001",
            question_type="single_hop",
            query="What cybersecurity risks does Microsoft disclose?",
            answerable=True,
            gold_sections=(("0000950170-24-087843", "1A"),),
            retrieved_sections=(("0000950170-24-087843", "1A"),),
            recall_at_1=1.0,
            recall_at_5=1.0,
            recall_at_10=1.0,
            mrr=1.0,
            ndcg_at_10=1.0,
            encode_ms=10.0,
            dense_ms=4.0,
            sparse_ms=0.0,
            fuse_ms=0.0,
            rerank_ms=0.0,
            total_ms=14.0,
        ),
        QueryRow(
            strategy="fixed",
            mode="dense",
            rerank=False,
            question_id="ua-001",
            question_type="unanswerable",
            query="What did NVDA disclose about GPU supply?",
            answerable=False,
            gold_sections=(),
            retrieved_sections=(("0000950170-24-087843", "1A"),),
            recall_at_1=None,
            recall_at_5=None,
            recall_at_10=None,
            mrr=None,
            ndcg_at_10=None,
            encode_ms=10.0,
            dense_ms=4.0,
            sparse_ms=0.0,
            fuse_ms=0.0,
            rerank_ms=0.0,
            total_ms=14.0,
        ),
    )
    return EvalResult(
        reports=(
            ConfigReport(
                strategy="fixed",
                mode="dense",
                rerank=False,
                scores=MeanScore(
                    n=1,
                    recall_at_1=1.0,
                    recall_at_5=0.8,
                    recall_at_10=1.0,
                    mrr=0.75,
                    ndcg_at_10=0.82,
                ),
                p50_ms=20.0,
                p95_ms=40.0,
                n_questions=2,
                n_unanswerable=1,
            ),
        ),
        rows=rows,
        output=Path("results/eval-retrieval.jsonl"),
    )


def _install_evaluator(monkeypatch, result: EvalResult | None = None) -> FakeEvaluator:
    fake = FakeEvaluator(result=result or _eval_result())
    evaluators.clear()
    evaluators.append(fake)
    monkeypatch.setattr("filing_rag.cli.Evaluator", FakeEvaluator)
    return fake


def test_eval_retrieval_help() -> None:
    result = runner.invoke(app, ["eval-retrieval", "--help"])
    assert result.exit_code == 0
    assert "--strategy" in result.stdout
    assert "--mode" in result.stdout
    assert "--rerank-only" in result.stdout
    assert "--no-rerank" in result.stdout
    assert "--k" in result.stdout
    assert "--force" in result.stdout
    assert "--output" in result.stdout


def test_eval_retrieval_passes_flags(monkeypatch) -> None:
    fake = _install_evaluator(monkeypatch)
    result = runner.invoke(
        app,
        [
            "eval-retrieval",
            "--strategy",
            "fixed",
            "--strategy",
            "structural",
            "--mode",
            "dense",
            "--mode",
            "hybrid",
            "--rerank-only",
            "--k",
            "10",
            "--force",
            "--output",
            "tmp/eval.jsonl",
        ],
    )
    assert result.exit_code == 0
    assert fake.captured == {
        "strategies": ["fixed", "structural"],
        "modes": ["dense", "hybrid"],
        "rerank": True,
        "k": 10,
        "force": True,
        "output": Path("tmp/eval.jsonl"),
    }
    assert "configs=1 questions=1 skipped_unanswerable=1" in result.stdout
    assert "recall@5=0.800" in result.stdout
    assert "p50=20ms p95=40ms" in result.stdout


def test_eval_retrieval_defaults(monkeypatch) -> None:
    fake = _install_evaluator(monkeypatch)
    result = runner.invoke(app, ["eval-retrieval"])
    assert result.exit_code == 0
    assert fake.captured == {
        "strategies": None,
        "modes": None,
        "rerank": None,
        "k": None,
        "force": False,
        "output": None,
    }


def test_eval_retrieval_no_rerank(monkeypatch) -> None:
    fake = _install_evaluator(monkeypatch)
    result = runner.invoke(app, ["eval-retrieval", "--no-rerank"])
    assert result.exit_code == 0
    assert fake.captured["rerank"] is False


def test_eval_retrieval_rejects_both_rerank_flags(monkeypatch) -> None:
    fake = _install_evaluator(monkeypatch)
    result = runner.invoke(app, ["eval-retrieval", "--rerank-only", "--no-rerank"])
    assert result.exit_code == 1
    assert "not both" in result.output
    assert fake.captured == {}


def test_eval_retrieval_exits_one_on_unknown_strategy(monkeypatch) -> None:
    class Boom(FakeEvaluator):
        def run(self, *args, **kwargs) -> EvalResult:
            raise ValueError(
                "unknown strategies: ['recursive']. Known: fixed, structural, semantic"
            )

    evaluators.clear()
    evaluators.append(Boom())
    monkeypatch.setattr("filing_rag.cli.Evaluator", Boom)
    result = runner.invoke(app, ["eval-retrieval", "--strategy", "recursive"])
    assert result.exit_code == 1
    assert "recursive" in result.output


@dataclass
class FakeGenerator:
    captured: dict = field(default_factory=dict)
    result: AskResult = field(default_factory=AskResult)

    @classmethod
    def from_config(cls) -> FakeGenerator:
        return generators[0]

    def ask(
        self,
        query,
        *,
        strategy,
        mode="hybrid",
        k=None,
        rerank=False,
        filters=None,
        force=False,
    ) -> AskResult:
        self.captured.update(
            query=query,
            strategy=strategy,
            mode=mode,
            k=k,
            rerank=rerank,
            filters=filters,
            force=force,
        )
        return self.result


generators: list[FakeGenerator] = []


def _ask_result() -> AskResult:
    return AskResult(
        retrieve=RetrieveResult(
            hits=(
                Hit(
                    chunk_id=1,
                    score=0.91,
                    rank=1,
                    text="Cybersecurity risk could disrupt operations.",
                    ticker="MSFT",
                    fiscal_year=2024,
                    item_code="1A",
                    accession="0000950170-24-087843",
                    char_start=0,
                    char_end=44,
                    edgar_url="https://example.com",
                    strategy="fixed",
                    chunk_index=0,
                ),
            ),
            mode="hybrid",
            strategy="fixed",
            reranked=True,
        ),
        generate=GenerateResult(
            text="Microsoft discloses cybersecurity incidents. [MSFT FY2024 Item 1A]",
            usage=Usage(prompt_tokens=400, completion_tokens=80),
            usd=0.000176,
            timings=GenerateTimings(generate_ms=120.0),
            model="gpt-5.6-luna",
        ),
    )


def _install_generator(monkeypatch, result: AskResult | None = None) -> FakeGenerator:
    fake = FakeGenerator(result=result or _ask_result())
    generators.clear()
    generators.append(fake)
    monkeypatch.setattr("filing_rag.cli.Generator", FakeGenerator)
    return fake


def test_generate_help() -> None:
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--strategy" in result.stdout
    assert "--mode" in result.stdout
    assert "--rerank" in result.stdout
    assert "--k" in result.stdout
    assert "--ticker" in result.stdout
    assert "--year" in result.stdout
    assert "--item" in result.stdout
    assert "--force" in result.stdout


def test_generate_requires_strategy() -> None:
    result = runner.invoke(app, ["generate", "cybersecurity risk"])
    assert result.exit_code != 0


def test_generate_passes_flags(monkeypatch) -> None:
    fake = _install_generator(monkeypatch)
    result = runner.invoke(
        app,
        [
            "generate",
            "cybersecurity risk",
            "--strategy",
            "fixed",
            "--mode",
            "hybrid",
            "--rerank",
            "--k",
            "5",
            "--ticker",
            "MSFT",
            "--year",
            "2024",
            "--item",
            "1A",
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert fake.captured["query"] == "cybersecurity risk"
    assert fake.captured["strategy"] == "fixed"
    assert fake.captured["mode"] == "hybrid"
    assert fake.captured["k"] == 5
    assert fake.captured["rerank"] is True
    assert fake.captured["force"] is True
    assert fake.captured["filters"] == Filters(
        tickers=("MSFT",),
        fiscal_years=(2024,),
        item_codes=("1A",),
    )
    assert "1 0.9100 MSFT FY2024 1A 0000950170-24-087843" in result.stdout
    assert "Microsoft discloses cybersecurity incidents" in result.stdout
    assert "generate=120ms prompt=400 completion=80 usd=0.000176" in result.stdout


def test_generate_defaults(monkeypatch) -> None:
    fake = _install_generator(monkeypatch)
    result = runner.invoke(
        app,
        ["generate", "interest rates", "--strategy", "structural"],
    )
    assert result.exit_code == 0
    assert fake.captured == {
        "query": "interest rates",
        "strategy": "structural",
        "mode": "hybrid",
        "k": None,
        "rerank": False,
        "filters": None,
        "force": False,
    }


def test_generate_exits_one_on_unknown_mode(monkeypatch) -> None:
    class Boom(FakeGenerator):
        def ask(self, *args, **kwargs) -> AskResult:
            raise ValueError("unknown mode 'keyword'. Known: dense, sparse, hybrid")

    generators.clear()
    generators.append(Boom())
    monkeypatch.setattr("filing_rag.cli.Generator", Boom)
    result = runner.invoke(
        app,
        ["generate", "cyber", "--strategy", "fixed", "--mode", "keyword"],
    )
    assert result.exit_code == 1
    assert "keyword" in result.output


@dataclass
class FakeRagEvaluator:
    captured: dict = field(default_factory=dict)
    result: RagEvalResult = field(default_factory=RagEvalResult)

    @classmethod
    def from_config(cls) -> FakeRagEvaluator:
        return rag_evaluators[0]

    def run(
        self,
        *,
        strategy=None,
        modes=None,
        rerank=None,
        k=None,
        force=False,
        output=None,
        retrieval_jsonl=None,
    ) -> RagEvalResult:
        self.captured.update(
            strategy=strategy,
            modes=modes,
            rerank=rerank,
            k=k,
            force=force,
            output=output,
            retrieval_jsonl=retrieval_jsonl,
        )
        return self.result


rag_evaluators: list[FakeRagEvaluator] = []


def _rag_eval_result() -> RagEvalResult:
    rows = (
        RagQueryRow(
            strategy="structural",
            mode="dense",
            rerank=False,
            question_id="sh-001",
            question_type="single_hop",
            query="What cybersecurity risks does Microsoft disclose?",
            answerable=True,
            response="Microsoft discloses cybersecurity incidents.",
            refused=False,
            retrieved_sections=(("0000950170-24-087843", "1A"),),
            faithfulness=0.9,
            context_precision=0.8,
            context_recall=0.7,
            relevancy=0.6,
            judge_ms=12.0,
            judge_usd=None,
            encode_ms=10.0,
            dense_ms=4.0,
            sparse_ms=0.0,
            fuse_ms=0.0,
            rerank_ms=0.0,
            total_ms=14.0,
            generate_ms=50.0,
            serving_ms=64.0,
            prompt_tokens=100,
            completion_tokens=20,
            usd=0.0001,
        ),
        RagQueryRow(
            strategy="structural",
            mode="dense",
            rerank=False,
            question_id="ua-001",
            question_type="unanswerable",
            query="What did NVDA disclose about GPU supply?",
            answerable=False,
            response="Not in the corpus.",
            refused=True,
            retrieved_sections=(("0000950170-24-087843", "1A"),),
            faithfulness=None,
            context_precision=None,
            context_recall=None,
            relevancy=None,
            judge_ms=None,
            judge_usd=None,
            encode_ms=10.0,
            dense_ms=4.0,
            sparse_ms=0.0,
            fuse_ms=0.0,
            rerank_ms=0.0,
            total_ms=14.0,
            generate_ms=50.0,
            serving_ms=64.0,
            prompt_tokens=100,
            completion_tokens=20,
            usd=0.0001,
        ),
    )
    return RagEvalResult(
        strategy="structural",
        reports=(
            RagConfigReport(
                strategy="structural",
                mode="dense",
                rerank=False,
                ragas=RagasScores(
                    faithfulness=0.9,
                    context_precision=0.8,
                    context_recall=0.7,
                    relevancy=0.6,
                    judge_ms=12.0,
                ),
                refusal_rate=1.0,
                p50_ms=64.0,
                p95_ms=80.0,
                usd=0.0001,
                n_questions=2,
                n_unanswerable=1,
            ),
        ),
        rows=rows,
        output=Path("results/eval-rag.jsonl"),
    )


def _install_rag_evaluator(
    monkeypatch, result: RagEvalResult | None = None
) -> FakeRagEvaluator:
    fake = FakeRagEvaluator(result=result or _rag_eval_result())
    rag_evaluators.clear()
    rag_evaluators.append(fake)
    monkeypatch.setattr("filing_rag.cli.RagEvaluator", FakeRagEvaluator)
    return fake


def test_eval_rag_help() -> None:
    result = runner.invoke(app, ["eval-rag", "--help"])
    assert result.exit_code == 0
    assert "--strategy" in result.stdout
    assert "--mode" in result.stdout
    assert "--rerank-only" in result.stdout
    assert "--no-rerank" in result.stdout
    assert "--k" in result.stdout
    assert "--force" in result.stdout
    assert "--output" in result.stdout
    assert "--retrieval-jsonl" in result.stdout


def test_eval_rag_passes_flags(monkeypatch) -> None:
    fake = _install_rag_evaluator(monkeypatch)
    result = runner.invoke(
        app,
        [
            "eval-rag",
            "--strategy",
            "structural",
            "--mode",
            "dense",
            "--mode",
            "hybrid",
            "--rerank-only",
            "--k",
            "5",
            "--force",
            "--output",
            "tmp/eval-rag.jsonl",
            "--retrieval-jsonl",
            "tmp/eval-retrieval.jsonl",
        ],
    )
    assert result.exit_code == 0
    assert fake.captured == {
        "strategy": "structural",
        "modes": ["dense", "hybrid"],
        "rerank": True,
        "k": 5,
        "force": True,
        "output": Path("tmp/eval-rag.jsonl"),
        "retrieval_jsonl": Path("tmp/eval-retrieval.jsonl"),
    }
    assert "strategy=structural configs=1 questions=1 skipped_unanswerable=1" in result.stdout
    assert "refusal=1.000" in result.stdout
    assert "faith=0.900" in result.stdout
    assert "p50=64ms p95=80ms" in result.stdout
    assert "usd=0.000100" in result.stdout


def test_eval_rag_defaults(monkeypatch) -> None:
    fake = _install_rag_evaluator(monkeypatch)
    result = runner.invoke(app, ["eval-rag"])
    assert result.exit_code == 0
    assert fake.captured == {
        "strategy": None,
        "modes": None,
        "rerank": None,
        "k": None,
        "force": False,
        "output": None,
        "retrieval_jsonl": None,
    }


def test_eval_rag_no_rerank(monkeypatch) -> None:
    fake = _install_rag_evaluator(monkeypatch)
    result = runner.invoke(app, ["eval-rag", "--no-rerank"])
    assert result.exit_code == 0
    assert fake.captured["rerank"] is False


def test_eval_rag_rejects_both_rerank_flags(monkeypatch) -> None:
    fake = _install_rag_evaluator(monkeypatch)
    result = runner.invoke(app, ["eval-rag", "--rerank-only", "--no-rerank"])
    assert result.exit_code == 1
    assert "not both" in result.output
    assert fake.captured == {}


def test_eval_rag_exits_one_on_unknown_strategy(monkeypatch) -> None:
    class Boom(FakeRagEvaluator):
        def run(self, *args, **kwargs) -> RagEvalResult:
            raise ValueError(
                "unknown strategies: ['recursive']. Known: fixed, structural, semantic"
            )

    rag_evaluators.clear()
    rag_evaluators.append(Boom())
    monkeypatch.setattr("filing_rag.cli.RagEvaluator", Boom)
    result = runner.invoke(app, ["eval-rag", "--strategy", "recursive"])
    assert result.exit_code == 1
    assert "recursive" in result.output


def test_serve_help() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "127.0.0.1" in result.stdout
    assert "8000" in result.stdout


def test_help_lists_serve() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.stdout


@pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is not None,
    reason="fastapi already installed",
)
def test_serve_exits_when_fastapi_missing() -> None:
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1
    assert "fastapi" in result.output.lower()


def test_report_help() -> None:
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    assert "--retrieval-jsonl" in result.stdout
    assert "--rag-jsonl" in result.stdout
    assert "--output" in result.stdout
    assert "--readme" in result.stdout


def test_help_lists_report() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "report" in result.stdout


def test_report_writes_ablation(monkeypatch, tmp_path: Path) -> None:
    from filing_rag.evaluate.report import ABLATION_END, ABLATION_START
    from filing_rag.evaluate.store import write_jsonl
    from filing_rag.evaluate.types import QueryRow
    from filing_rag.settings import Settings

    monkeypatch.setattr(
        "filing_rag.evaluate.report.get_settings",
        lambda: Settings(results_dir=tmp_path),
    )

    row = QueryRow(
        strategy="fixed",
        mode="dense",
        rerank=False,
        question_id="q1",
        question_type="single_hop",
        query="q",
        answerable=True,
        gold_sections=(("acc", "1A"),),
        retrieved_sections=(("acc", "1A"),),
        recall_at_1=1.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr=1.0,
        ndcg_at_10=1.0,
        encode_ms=0.0,
        dense_ms=0.0,
        sparse_ms=0.0,
        fuse_ms=0.0,
        rerank_ms=0.0,
        total_ms=10.0,
    )
    retrieval = tmp_path / "eval-retrieval.jsonl"
    write_jsonl((row,), retrieval)
    readme = tmp_path / "README.md"
    readme.write_text(f"{ABLATION_START}\n{ABLATION_END}\n", encoding="utf-8")
    output = tmp_path / "ablation.md"
    result = runner.invoke(
        app,
        [
            "report",
            "--retrieval-jsonl",
            str(retrieval),
            "--output",
            str(output),
            "--readme",
            str(readme),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert "wrote" in result.stdout
    assert "fixed is the winning chunker at k=10" in readme.read_text(encoding="utf-8")

