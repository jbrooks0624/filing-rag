"""Command-line entry point for filing-rag."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from filing_rag.chunking import Chunker, SemanticChunkError
from filing_rag.embed import EmbedError, Indexer, IndexStat, StoreError
from filing_rag.evaluate import (
    Evaluator,
    RagasError,
    RagEvaluator,
    format_rag_summary,
    format_rag_table,
    format_summary,
    format_table,
    run_report,
)
from filing_rag.generate import AskResult, GenerateError, Generator
from filing_rag.ingest import Ingestor
from filing_rag.retrieve import (
    Filters,
    RerankError,
    Retriever,
    RetrieveResult,
    SparseError,
    snippet,
)

app = typer.Typer(
    no_args_is_help=True,
    help="SEC filing RAG toolkit.",
)


@app.callback()
def main() -> None:
    """SEC filing RAG toolkit."""


@app.command()
def ingest(
    ticker: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these tickers. Repeatable."),
    ] = None,
    year: Annotated[
        list[int] | None,
        typer.Option(help="Limit to these fiscal years. Repeatable."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Refetch from EDGAR even if cache exists."),
    ] = False,
    parse_only: Annotated[
        bool,
        typer.Option("--parse-only", help="Re-parse cached HTML; never hit the network."),
    ] = False,
) -> None:
    """Pull and parse SEC filings from EDGAR."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = Ingestor.from_config().run(
            tickers=ticker or None,
            fiscal_years=year or None,
            force=force,
            parse_only=parse_only,
        )
    except (KeyError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"fetched={result.fetched} skipped={result.skipped} "
        f"parsed={result.parsed} errors={len(result.errors)}"
    )
    for error in result.errors:
        where = " ".join(
            part
            for part in (error.ticker, str(error.fiscal_year) if error.fiscal_year else None)
            if part
        )
        prefix = f"{where}: " if where else ""
        typer.echo(f"  {prefix}{error.message}", err=True)
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def chunk(
    ticker: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these tickers. Repeatable."),
    ] = None,
    year: Annotated[
        list[int] | None,
        typer.Option(help="Limit to these fiscal years. Repeatable."),
    ] = None,
    strategy: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these strategies (fixed, structural, semantic). Repeatable."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-chunk even if output JSON already exists."),
    ] = False,
) -> None:
    """Chunk parsed filings. Never calls EDGAR."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = Chunker.from_config().run(
            tickers=ticker or None,
            fiscal_years=year or None,
            strategies=strategy or None,
            force=force,
        )
    except (KeyError, ValueError, SemanticChunkError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"chunked={result.chunked} skipped={result.skipped} "
        f"chunks={result.chunks_written} errors={len(result.errors)}"
    )
    for error in result.errors:
        where = " ".join(
            part
            for part in (
                error.ticker,
                str(error.fiscal_year) if error.fiscal_year else None,
                error.strategy,
            )
            if part
        )
        prefix = f"{where}: " if where else ""
        typer.echo(f"  {prefix}{error.message}", err=True)
    if not result.ok:
        raise typer.Exit(code=1)


def _format_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f}MiB"
    if n >= 1024:
        return f"{n / 1024:.1f}KiB"
    return f"{n}B"


def _hnsw_line(stats: list[IndexStat]) -> str:
    parts: list[str] = []
    for stat in stats:
        piece = f"{stat.strategy}={_format_bytes(stat.bytes)}"
        if stat.build_ms is not None:
            piece += f" {stat.build_ms:.0f}ms"
        parts.append(piece)
    return "hnsw " + " ".join(parts)


@app.command("index")
def index_cmd(
    ticker: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these tickers. Repeatable."),
    ] = None,
    year: Annotated[
        list[int] | None,
        typer.Option(help="Limit to these fiscal years. Repeatable."),
    ] = None,
    strategy: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these strategies (fixed, structural, semantic). Repeatable."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-embed even if vectors already exist."),
    ] = False,
) -> None:
    """Load chunk JSON into pgvector. Never calls EDGAR."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = Indexer.from_config().run(
            tickers=ticker or None,
            fiscal_years=year or None,
            strategies=strategy or None,
            force=force,
        )
    except (KeyError, ValueError, EmbedError, StoreError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"indexed={result.indexed} skipped={result.skipped} "
        f"embedded={result.embedded} errors={len(result.errors)}"
    )
    if result.index_stats:
        typer.echo(_hnsw_line(result.index_stats))
    for error in result.errors:
        where = " ".join(
            part
            for part in (
                error.ticker,
                str(error.fiscal_year) if error.fiscal_year else None,
                error.strategy,
            )
            if part
        )
        prefix = f"{where}: " if where else ""
        typer.echo(f"  {prefix}{error.message}", err=True)
    if not result.ok:
        raise typer.Exit(code=1)


def _search_summary(result: RetrieveResult) -> str:
    rerank = "on" if result.reranked else "off"
    timing = result.timings
    return (
        f"hits={len(result.hits)} mode={result.mode} strategy={result.strategy} "
        f"rerank={rerank} encode={timing.encode_ms:.0f}ms dense={timing.dense_ms:.0f}ms "
        f"sparse={timing.sparse_ms:.0f}ms fuse={timing.fuse_ms:.0f}ms "
        f"rerank={timing.rerank_ms:.0f}ms"
    )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Natural-language query.")],
    strategy: Annotated[
        str,
        typer.Option(help="Chunking strategy: fixed, structural, or semantic."),
    ],
    mode: Annotated[
        str,
        typer.Option(help="Retrieval mode: dense, sparse, or hybrid."),
    ] = "hybrid",
    rerank: Annotated[
        bool,
        typer.Option("--rerank", help="Cross-encoder rerank over candidate_k."),
    ] = False,
    k: Annotated[
        int | None,
        typer.Option(help="Top-k hits. Defaults to config/retrieval.yaml."),
    ] = None,
    ticker: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these tickers. Repeatable."),
    ] = None,
    year: Annotated[
        list[int] | None,
        typer.Option(help="Limit to these fiscal years. Repeatable."),
    ] = None,
    item: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these item codes (1A, 7, 7A). Repeatable."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild the bm25s index for this strategy."),
    ] = False,
) -> None:
    """Search indexed chunks. Never calls EDGAR."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    filters = None
    if ticker or year or item:
        filters = Filters(
            tickers=tuple(ticker) if ticker else None,
            fiscal_years=tuple(year) if year else None,
            item_codes=tuple(item) if item else None,
        )
    try:
        result = Retriever.from_config().search(
            query,
            strategy=strategy,
            mode=mode,
            k=k,
            rerank=rerank,
            filters=filters,
            force=force,
        )
    except (KeyError, ValueError, EmbedError, StoreError, RerankError, SparseError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(_search_summary(result))
    for hit in result.hits:
        typer.echo(
            f"{hit.rank} {hit.score:.4f} {hit.ticker} FY{hit.fiscal_year} "
            f"{hit.item_code} {hit.accession}"
        )
        typer.echo(f"  {snippet(hit.text)}")


@app.command("generate")
def generate_cmd(
    query: Annotated[str, typer.Argument(help="Natural-language query.")],
    strategy: Annotated[
        str,
        typer.Option(help="Chunking strategy: fixed, structural, or semantic."),
    ],
    mode: Annotated[
        str,
        typer.Option(help="Retrieval mode: dense, sparse, or hybrid."),
    ] = "hybrid",
    rerank: Annotated[
        bool,
        typer.Option("--rerank", help="Cross-encoder rerank over candidate_k."),
    ] = False,
    k: Annotated[
        int | None,
        typer.Option(help="Top-k hits. Defaults to config/retrieval.yaml."),
    ] = None,
    ticker: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these tickers. Repeatable."),
    ] = None,
    year: Annotated[
        list[int] | None,
        typer.Option(help="Limit to these fiscal years. Repeatable."),
    ] = None,
    item: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these item codes (1A, 7, 7A). Repeatable."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild the bm25s index for this strategy."),
    ] = False,
) -> None:
    """Retrieve then generate an answer with citations. Never calls EDGAR."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    filters = None
    if ticker or year or item:
        filters = Filters(
            tickers=tuple(ticker) if ticker else None,
            fiscal_years=tuple(year) if year else None,
            item_codes=tuple(item) if item else None,
        )
    try:
        result = Generator.from_config().ask(
            query,
            strategy=strategy,
            mode=mode,
            k=k,
            rerank=rerank,
            filters=filters,
            force=force,
        )
    except (
        KeyError,
        ValueError,
        EmbedError,
        StoreError,
        RerankError,
        SparseError,
        GenerateError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    _print_ask(result)


def _print_ask(result: AskResult) -> None:
    for hit in result.retrieve.hits:
        typer.echo(
            f"{hit.rank} {hit.score:.4f} {hit.ticker} FY{hit.fiscal_year} "
            f"{hit.item_code} {hit.accession}"
        )
        typer.echo(f"  {snippet(hit.text)}")
    generated = result.generate
    if generated.text:
        typer.echo(generated.text)
    typer.echo(
        f"generate={generated.timings.generate_ms:.0f}ms "
        f"prompt={generated.usage.prompt_tokens} "
        f"completion={generated.usage.completion_tokens} "
        f"usd={generated.usd:.6f}"
    )


def _rerank_axis(rerank_only: bool, no_rerank: bool) -> bool | None:
    if rerank_only and no_rerank:
        raise ValueError("use --rerank-only or --no-rerank, not both")
    if rerank_only:
        return True
    if no_rerank:
        return False
    return None


@app.command("eval-retrieval")
def eval_retrieval(
    strategy: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these strategies (fixed, structural, semantic). Repeatable."),
    ] = None,
    mode: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these modes (dense, sparse, hybrid). Repeatable."),
    ] = None,
    rerank_only: Annotated[
        bool,
        typer.Option("--rerank-only", help="Only run rerank-on configs."),
    ] = False,
    no_rerank: Annotated[
        bool,
        typer.Option("--no-rerank", help="Only run rerank-off configs."),
    ] = False,
    k: Annotated[
        int | None,
        typer.Option(help="Top-k hits scored. Defaults to 10."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild bm25s indexes before scoring."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(help="JSONL path. Defaults to results/eval-retrieval.jsonl."),
    ] = None,
) -> None:
    """Score the golden set across retrieval configs. Never calls EDGAR."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        rerank = _rerank_axis(rerank_only, no_rerank)
        result = Evaluator.from_config().run(
            strategies=strategy or None,
            modes=mode or None,
            rerank=rerank,
            k=k,
            force=force,
            output=output,
        )
    except (KeyError, ValueError, EmbedError, StoreError, RerankError, SparseError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_summary(result))
    table = format_table(result.reports)
    if table:
        typer.echo(table)


@app.command("eval-rag")
def eval_rag(
    strategy: Annotated[
        str | None,
        typer.Option(help="Chunking strategy. Omit to pick the Stage 1 winner."),
    ] = None,
    mode: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these modes (dense, sparse, hybrid). Repeatable."),
    ] = None,
    rerank_only: Annotated[
        bool,
        typer.Option("--rerank-only", help="Only run rerank-on configs."),
    ] = False,
    no_rerank: Annotated[
        bool,
        typer.Option("--no-rerank", help="Only run rerank-off configs."),
    ] = False,
    k: Annotated[
        int | None,
        typer.Option(help="Top-k hits. Defaults to config/retrieval.yaml."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild bm25s indexes before scoring."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(help="JSONL path. Defaults to results/eval-rag.jsonl."),
    ] = None,
    retrieval_jsonl: Annotated[
        Path | None,
        typer.Option(help="Stage 1 JSONL used to pick a winner."),
    ] = None,
) -> None:
    """Score the golden set with generation and RAGAS. Calls an LLM."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        rerank = _rerank_axis(rerank_only, no_rerank)
        result = RagEvaluator.from_config().run(
            strategy=strategy,
            modes=mode or None,
            rerank=rerank,
            k=k,
            force=force,
            output=output,
            retrieval_jsonl=retrieval_jsonl,
        )
    except (
        KeyError,
        ValueError,
        EmbedError,
        StoreError,
        RerankError,
        SparseError,
        GenerateError,
        RagasError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_rag_summary(result))
    table = format_rag_table(result.reports)
    if table:
        typer.echo(table)


@app.command()
def report(
    retrieval_jsonl: Annotated[
        Path | None,
        typer.Option(help="Stage 1 JSONL. Defaults to results/eval-retrieval.jsonl."),
    ] = None,
    rag_jsonl: Annotated[
        Path | None,
        typer.Option(help="Stage 2 JSONL. Defaults to results/eval-rag.jsonl."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Markdown path. Defaults to results/ablation.md."),
    ] = None,
    readme: Annotated[
        Path | None,
        typer.Option(help="README to inject headlines into. Defaults to README.md."),
    ] = None,
) -> None:
    """Write ablation tables and inject definition-of-done headlines. No LLM calls."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        dest = run_report(
            retrieval_jsonl=retrieval_jsonl,
            rag_jsonl=rag_jsonl,
            output=output,
            readme=readme,
        )
    except (KeyError, ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {dest}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Serve the RAG HTTP API. Does not call EDGAR. Missing API key is not a start failure."""
    try:
        import uvicorn

        from filing_rag.api.app import create_app
    except ImportError as exc:
        typer.echo(
            "serving requires fastapi and uvicorn. Install with: uv sync",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    uvicorn.run(create_app(), host=host, port=port)
