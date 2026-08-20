# FILING-RAG

SEC filing RAG with a measured retrieval evaluation harness.

## Use case

Ask a question against Items 1A, 7, and 7A of 27 10-Ks (MSFT / GOOGL / AMZN, JPM / BAC / GS, PFE / MRK / LLY × FY2022–FY2024). Retrieval returns cited chunks; generation must answer with `[TICKER FYyear Item CODE]` citations or reply exactly `Not in the corpus.` when the filings do not support it. The golden set uses NVDA, WFC, and ABBV as those out-of-corpus probes.

That covers the question types in [`eval/golden_set.yaml`](eval/golden_set.yaml): a fact in one filing, a year-over-year change at one company, a comparison inside tech / banks / pharma, or a comparison across sectors. `--ticker` / `--year` / `--item` (and the same fields on `POST /query`) scope retrieval when the slice is already known.

The other use is the harness itself. Stage 1 scores all 18 retrieval configs with no LLM calls; Stage 2 generates and runs RAGAS on the winning chunker’s 6 configs; `filing-rag report` writes those numbers into this README. `filing-rag serve` and Compose `app` expose the same retrieve-then-generate path over HTTP. There is no product UI.

## Setup

Python 3.14 (pinned). Copy `.env.example` to `.env` and set a descriptive EDGAR User-Agent (name + contact email).

```bash
cp .env.example .env
uv sync --dev
docker compose up -d db
uv run filing-rag --help
```

Postgres is `pgvector/pgvector:pg17` on port 5432 (`filing` / `filing` / `filing_rag`). Override with `DATABASE_URL` in `.env`.

## Ingest

```bash
uv run filing-rag ingest
uv run filing-rag ingest --ticker MSFT --year 2024
uv run filing-rag ingest --parse-only
uv run filing-rag ingest --force
# equivalent: uv run python -m filing_rag ingest
```

`--ticker` and `--year` can be repeated. Re-runs skip filings already in `data/parsed/`. `--parse-only` re-parses cached HTML and does not call EDGAR.

## Chunk

```bash
uv run filing-rag chunk --strategy fixed --strategy structural
uv run filing-rag chunk --strategy semantic
```

Writes `data/chunks/{fixed,structural,semantic}/`. Every chunk is ≤ 512 tokens. A second run skips existing files.

## Index

Requires Compose Postgres (same bge encoder as chunking).

```bash
docker compose up -d db
uv run filing-rag index
```

Loads parsed + chunk JSON into pgvector, embeds with `bge-base-en-v1.5` (passages only, no query prefix), then builds one partial HNSW index per strategy. Expect `indexed=81 skipped=0 embedded=12157` on the 27-filing corpus. A second run should report `skipped=81`.

## Search

Requires Compose Postgres and an indexed corpus.

```bash
uv run filing-rag search "cybersecurity risk" --strategy fixed
uv run filing-rag search "cybersecurity risk" --strategy structural --mode dense --ticker MSFT --year 2024 --item 1A
uv run filing-rag search "interest rates" --strategy semantic --mode hybrid --rerank --k 5
```

`--strategy` is required (`fixed`, `structural`, or `semantic`). `--mode` is `dense`, `sparse`, or `hybrid` (default hybrid). Six configs: those three modes × rerank on/off. Queries get the bge instruction prefix; passage embeddings stay unprefixed. `--force` rebuilds the bm25s index for that strategy.

## Generate

Requires Compose Postgres, an indexed corpus, and `OPENAI_API_KEY` in `.env`.

```bash
uv run filing-rag generate "cybersecurity risk" --strategy structural
uv run filing-rag generate "interest rates" --strategy fixed --mode hybrid --rerank --k 5
```

`--strategy` is required. Other flags match search. Prints citation lines, the answer, then `generate=…ms prompt=… completion=… usd=…`. Answers are forced to cite `[TICKER FYyear Item CODE]`. If the retrieved chunks do not support an answer, the model is instructed to reply with exactly `Not in the corpus.`

Generator is `gpt-5.6-luna` with `reasoning_effort=none`. USD is Luna list price on prompt + completion tokens. Local embed/rerank stay milliseconds.

## Eval

Requires Compose Postgres and an indexed corpus. This stage makes **no LLM calls**.

```bash
uv run filing-rag eval-retrieval
uv run filing-rag eval-retrieval --strategy fixed --mode hybrid --rerank-only --k 10
uv run filing-rag eval-retrieval --strategy semantic --mode sparse --no-rerank
```

Omit flags to score all **18 configs**: `{fixed, structural, semantic}` × `{dense, sparse, hybrid}` × `{rerank off, on}`. Repeatable `--strategy` / `--mode` narrow the grid. `--rerank-only` and `--no-rerank` keep one side of the rerank axis (do not combine them). `--force` rebuilds bm25s once per strategy. `--output` defaults to `results/eval-retrieval.jsonl` (gitignored).

Queries come from [`eval/golden_set.yaml`](eval/golden_set.yaml): 50 questions (20 single-hop, 10 same-company temporal, 10 within-sector, 5 cross-sector, 5 unanswerable targeting NVDA / WFC / ABBV). Ground truth is **section-level** `(accession, item_code)`, not chunk ids, so recall is comparable across chunkers. A retrieved chunk is a hit when that pair matches a gold citation. Hits collapse to unique sections before scoring so overlapping fixed windows do not inflate nDCG. Gold `quote` fields are human evidence; they are not used for matching.

Metrics on the collapsed ranking, `k=10`: recall@{1,5,10}, MRR, nDCG@10. Macro-average is over the **45 answerable** questions; the 5 unanswerable rows are still retrieved and written to JSONL (latency) but excluded from quality means. Per-config p50 / p95 come from `RetrieveTimings.total_ms`. No dollar column — there is no API spend in Stage 1.

Stdout is a summary line plus one row per config, for example:

```
configs=18 questions=45 skipped_unanswerable=5
fixed       dense   off  recall@5=0.656  mrr=0.543  ndcg@10=0.559  p50=25ms p95=84ms
```

## Eval RAG

Requires Compose Postgres, an indexed corpus, Stage 1 JSONL (or `--strategy`), and `OPENAI_API_KEY`. This stage **does** call an LLM: once to generate, then RAGAS judges the 45 answerable answers.

```bash
uv run filing-rag eval-rag
uv run filing-rag eval-rag --strategy structural --mode hybrid --rerank-only --k 5
uv run filing-rag eval-rag --strategy fixed --no-rerank
```

Omit `--strategy` to pick the Stage 1 winner: the chunker with the highest mean recall@5 in `results/eval-retrieval.jsonl` (tie-break nDCG@10, then MRR). Pass `--retrieval-jsonl` to point at a different Stage 1 file. Missing JSONL without `--strategy` fails and tells you to run `eval-retrieval` first.

Grid is **6 configs** on that one chunker: `{dense, sparse, hybrid}` × `{rerank off, on}`. Repeatable `--mode` plus `--rerank-only` / `--no-rerank` narrow it. `k` defaults to **5** (`config/retrieval.yaml`), the serving default, not Stage 1's eval `k=10`. `--output` defaults to `results/eval-rag.jsonl` (gitignored).

Per question: retrieve, generate, then a locked-phrase refusal check (`Not in the corpus.`, case-insensitive). The 5 unanswerable questions skip RAGAS (faithfulness is undefined when the right answer is a refusal) but still generate and count toward `usd` and `refusal`. Answerable refusals still go through RAGAS so a false refusal looks bad. RAGAS collections metrics: Faithfulness, ContextPrecision, ContextRecall, AnswerRelevancy (reported as `relevancy`). The judge model is the same Luna generator — that eval bias is intentional and called out.

Macro-average is over the **45 answerable** questions. `usd` is mean **generation** dollars over all **50** questions (judge tokens are `judge_ms` / `judge_usd` in JSONL and are not added to `usd`). p50 / p95 are serving latency: `retrieve.total_ms + generate_ms`. `refusal` is the fraction of unanswerable rows that contained the locked phrase.

Stdout:

```
strategy=fixed configs=6 questions=45 skipped_unanswerable=5 refusal=1.000
dense   off  faith=0.732  ctx_p=0.350  ctx_r=0.441  rel=0.599  p50=1427ms p95=4357ms usd=0.000577
```

## API

Same retrieve-then-generate path as `filing-rag generate`, over HTTP. No UI. `GET /healthz` does not load torch and does not require `OPENAI_API_KEY`; `POST /query` returns 503 if the key is missing.

Local (MPS on Apple Silicon; this is the fast path):

```bash
uv run filing-rag serve
uv run filing-rag serve --host 127.0.0.1 --port 8000
```

Compose app (CPU torch in the image). Image size **9.21GB** — linux/arm64 torch pulls NVIDIA CUDA wheels even though serving is CPU. An empty `HF_HOME` volume (`models`) downloads `bge-base-en-v1.5` and `bge-reranker-base` on first `/query` — expect minutes.

```bash
docker compose up -d db
docker compose up -d app
```

```bash
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/configs
curl -N -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"cybersecurity risk","strategy":"structural","mode":"hybrid","rerank":false}'
```

`POST /query` requires `query` and `strategy`. Defaults match CLI generate: `mode=hybrid`, `rerank=false`, `k` from `config/retrieval.yaml`. Optional `ticker` / `year` / `item` lists become metadata filters. SSE events: one `citations` (snippets, not full chunk text), then `token` deltas, then `done` (answer, usage, retrieve timings, `serving_ms`).

## Ablation

Run `uv run filing-rag report` after `eval-rag` to fill this block from Stage 1/2 JSONL. Writes `results/ablation.md` and injects the tables and findings here.

<!-- ablation:start -->
### Stage 1 retrieval (k=10)

```
fixed       dense   off  recall@5=0.656  mrr=0.543  ndcg@10=0.559  p50=25ms p95=84ms
fixed       dense   on   recall@5=0.789  mrr=0.621  ndcg@10=0.633  p50=955ms p95=1016ms
fixed       sparse  off  recall@5=0.656  mrr=0.482  ndcg@10=0.495  p50=1ms p95=2ms
fixed       sparse  on   recall@5=0.667  mrr=0.586  ndcg@10=0.580  p50=1149ms p95=1173ms
fixed       hybrid  off  recall@5=0.722  mrr=0.519  ndcg@10=0.556  p50=26ms p95=53ms
fixed       hybrid  on   recall@5=0.733  mrr=0.641  ndcg@10=0.630  p50=1189ms p95=1398ms
structural  dense   off  recall@5=0.622  mrr=0.450  ndcg@10=0.491  p50=27ms p95=66ms
structural  dense   on   recall@5=0.756  mrr=0.612  ndcg@10=0.637  p50=925ms p95=997ms
structural  sparse  off  recall@5=0.544  mrr=0.423  ndcg@10=0.448  p50=2ms p95=2ms
structural  sparse  on   recall@5=0.689  mrr=0.554  ndcg@10=0.570  p50=978ms p95=1117ms
structural  hybrid  off  recall@5=0.656  mrr=0.429  ndcg@10=0.487  p50=29ms p95=50ms
structural  hybrid  on   recall@5=0.722  mrr=0.560  ndcg@10=0.599  p50=1108ms p95=1297ms
semantic    dense   off  recall@5=0.589  mrr=0.456  ndcg@10=0.484  p50=27ms p95=123ms
semantic    dense   on   recall@5=0.722  mrr=0.576  ndcg@10=0.583  p50=990ms p95=1380ms
semantic    sparse  off  recall@5=0.533  mrr=0.435  ndcg@10=0.434  p50=1ms p95=2ms
semantic    sparse  on   recall@5=0.678  mrr=0.518  ndcg@10=0.552  p50=1182ms p95=1603ms
semantic    hybrid  off  recall@5=0.622  mrr=0.455  ndcg@10=0.484  p50=27ms p95=46ms
semantic    hybrid  on   recall@5=0.733  mrr=0.607  ndcg@10=0.609  p50=1191ms p95=1588ms
```

### Stage 2 generation (k=5)

```
dense   off  faith=0.732  ctx_p=0.350  ctx_r=0.441  rel=0.599  p50=1427ms p95=4357ms usd=0.000577
dense   on   faith=0.743  ctx_p=0.271  ctx_r=0.526  rel=0.641  p50=4190ms p95=6579ms usd=0.000584
sparse  off  faith=0.719  ctx_p=0.341  ctx_r=0.444  rel=0.502  p50=1491ms p95=3890ms usd=0.000548
sparse  on   faith=0.665  ctx_p=0.272  ctx_r=0.444  rel=0.615  p50=4085ms p95=5820ms usd=0.000560
hybrid  off  faith=0.819  ctx_p=0.340  ctx_r=0.522  rel=0.624  p50=1713ms p95=4582ms usd=0.000582
hybrid  on   faith=0.722  ctx_p=0.280  ctx_r=0.500  rel=0.617  p50=4463ms p95=6469ms usd=0.000578
```

### Definition of done

1. fixed is the winning chunker at k=10 (mean recall@5=0.704), +0.039 vs structural and +0.057 vs semantic.
2. On fixed at k=10, dense+rerank is the best retrieval config (recall@5=0.789), +0.056 vs hybrid+rerank (0.733). Reranking alone, on dense: 0.656 → 0.789 (+0.133). Fusion alone, no rerank: 0.656 → 0.722 (+0.066). Reranking on top of fusion: 0.722 → 0.733 (+0.011).
3. At k=5, hybrid with rerank off has the best faithfulness (0.819); turning rerank on drops it to 0.722 and nearly triples p50 (1713ms → 4463ms). Context precision falls for every rerank-on config.
4. Refusal rate is 1.000 on all 5 unanswerable questions; context precision is the remaining lever (0.27–0.35); generation cost is flat ($0.000548–$0.000584, ~6% spread).

### Reads

Stage 1 (k=10). Dense+rerank is the best retrieval config (recall@5=0.789), not hybrid+rerank (0.733). Decomposition on fixed:

- Reranking alone, on dense: 0.656 → 0.789 (+0.133)
- Fusion alone, no rerank: 0.656 → 0.722 (+0.066)
- Reranking on top of fusion: 0.722 → 0.733 (+0.011)

Fusion and reranking recover largely the same chunks. Once you rerank, fusion adds almost nothing. Reranking alone beats fusion alone by ~2x. Hybrid fusion did not pay for itself.

Stage 2 (k=5) disagrees. Best faithfulness is hybrid with rerank off (0.819). Turning rerank on drops it to 0.722 and nearly triples p50 (1713ms → 4463ms). Context precision falls on every rerank-on config (0.350→0.271, 0.341→0.272, 0.340→0.280). A cross-encoder optimizes topical query-chunk relevance, not answer supportiveness; at serving k it swaps in on-topic chunks that do not carry the evidence. Reranking improves retrieval depth at k=10 and degrades answer faithfulness at serving k=5.

Refusal is 1.000 on all 5 unanswerable questions. Context precision is the weak spot (0.27–0.35): roughly two-thirds of retrieved context is not supporting the answer. Generation cost is effectively flat ($0.000548–$0.000584, ~6% spread); retrieval config does not move spend — k and chunk size do.
<!-- ablation:end -->

## What I'd do differently

Item 8 table-aware retrieval was scoped out. The corpus is pipe-flattened 1A / 7 / 7A, which is a compromise: numbers survive as text, but row/column structure does not.

bm25s is in-process and memory-bound. Each strategy's sparse index lives on disk under `data/indexes/` and is loaded into the serving process; it does not use Postgres full-text search.

HNSW was left at pgvector defaults. No `m` / `ef_construction` sweep; the partial indexes are one per chunking strategy and otherwise stock.

A single Luna model (`gpt-5.6-luna`) is both the generator and the RAGAS judge. That eval bias is intentional and called out in Stage 2, but a held-out judge would be the fairer number.

ragas 0.4 still talks Chat Completions. Serving uses the Responses API. `gpt-5.6-luna` also breaks ragas's GPT-5 version remap (`int('5.6')`), so Chat Completions kwargs are rewritten in `GenerationConfig.apply_chat_kwargs` rather than trusted from Instructor.

The Docker app is CPU torch. Local `uv run` on Apple Silicon uses MPS. Those two latency profiles are not interchangeable; p95 in the ablation is the local serving path.

Stage 2 is sequential and writes JSONL only at the end. There is no incremental dump, so a crash loses the run.

## Future additions

Context precision is the remaining quality-and-cost lever: most retrieved chunks are not supporting the answer. Rank or filter for answer supportiveness, not just topical query-chunk relevance. Retrieval mode does not move generation spend; k and chunk size do.

Sparse/bm25s is optional. Once you rerank, hybrid fusion added almost nothing, so in-process bm25s is not load-bearing. If sparse stays, move it into Postgres (BM25 or `ts_rank` labeled honestly) so serving is not memory-bound. Sweep HNSW `m` / `ef_construction`. Install CPU torch wheels in Docker so the app image is not 9.21GB of unused CUDA.

Item 8 with table-aware retrieval, plus 10-Qs / 8-Ks and a larger issuer set, so answers can use financial statements and more than annual risk/MD&A text.

A held-out judge model, ragas on the Responses API, incremental Stage 2 JSONL, and a Postgres `eval_results` table so a crash does not lose a run and the ablation is regenerable from the database.

A small UI over `POST /query` SSE (citations, tokens, `done`). Keep API dollars and local embed/rerank milliseconds as separate columns.

## Techniques

Chunking, embedding, and retrieval are configured in `config/chunking.yaml` and `config/retrieval.yaml`. Token counts always use the `BAAI/bge-base-en-v1.5` HuggingFace tokenizer (no PyTorch). Every chunk is hard-capped at 512 tokens.

### Chunking

Each strategy runs independently on each parsed 10-K section (items 1A, 7, 7A). Chunks keep character offsets into the parent section and write to `data/chunks/{strategy}/`. A second run skips existing files unless `--force`.

#### Fixed

Sliding token windows: **400 tokens, 80-token overlap** (step 320). A section shorter than 400 tokens is one chunk. Windows that still exceed 512 tokens are truncated. Ignores document structure; neighboring chunks share 80 tokens so a boundary does not drop a sentence.

#### Structural

Split on in-section subsection headers, then cap anything still over 512 tokens with the same 400/80 windows as fixed.

A line is treated as a header when it is ≤ 120 characters, is not a table row (`|` separators), and is either ALL CAPS (at least two letters) or Title Case (small words like `of`/`the` allowed). The next non-header prose line must also be longer than the header, so short titles without body text are skipped. If a section has no headers, the whole section is one candidate. Leading prose before the first header is kept as its own span.

#### Semantic

Split at sentence-similarity drops, then cap oversized groups with the same 400/80 windows.

1. Split the section into sentences on `.!?` followed by whitespace and a capital letter, quote, or parenthesis.
2. Embed each sentence with `BAAI/bge-base-en-v1.5` (L2-normalized).
3. Compute cosine **distance** (`1 − cosine`) between adjacent sentences.
4. Break where distance is above the **95th percentile** of those distances (LangChain default: keep the largest ~5% of topic drops).

A one-sentence section stays one chunk.

### Embedding

Indexing loads parsed + chunk JSON into Postgres (`pgvector/pgvector:pg17`) and encodes **passage text only**.

| Detail | Value |
| --- | --- |
| Model | `BAAI/bge-base-en-v1.5` via sentence-transformers |
| Dimension | 768 |
| Normalization | L2 (`normalize_embeddings=True`) |
| Batch size | 32 |
| Query prefix on passages | none (BGE is asymmetric; the instruction prefix is added only at search time) |
| Storage | `chunks.embedding vector(768)` |
| ANN index | one partial HNSW per strategy: `USING hnsw (embedding vector_cosine_ops) WHERE strategy = '…'` |

Dense score at search time is cosine similarity: `1 − (embedding <=> query)`. Re-runs skip filings that already have embeddings unless `--force`. Expect `indexed=81 skipped=0 embedded=12157` on the 27-filing × 3-strategy corpus.

### Retrieval

Search is always scoped to one chunking strategy. Optional filters: `--ticker`, `--year`, `--item`. Defaults live in `config/retrieval.yaml`.

| Setting | Default | Role |
| --- | --- | --- |
| `k` | 5 | Hits returned |
| `candidate_k` | 50 | Pool size for hybrid fusion and/or rerank |
| `rrf_k` | 60 | RRF smoothing constant |
| `query_prefix` | `Represent this sentence for searching relevant passages: ` | Prepended to the query before dense encoding |
| BM25 `k1` / `b` | 1.5 / 0.75 | Term-frequency saturation and length normalization |
| Reranker | `BAAI/bge-reranker-base` | Cross-encoder over `(query, chunk)` pairs |

Dense and `--rerank` use the local bge models from `uv sync --dev`. Sparse-only can run once `data/indexes/{strategy}/` exists.

#### Dense

Encode the query **with** the BGE instruction prefix (passages stay unprefixed). Approximate nearest-neighbor cosine search over the strategy’s HNSW index. Pool size is `k` unless hybrid or rerank is on, in which case it is `candidate_k`.

#### Sparse

Okapi BM25 via **bm25s** (not Postgres full-text search). English stopwords. One on-disk index per strategy at `data/indexes/{strategy}/`. `--force` rebuilds it from the current Postgres chunks.

#### Hybrid (default)

Run dense and sparse in parallel over the same `candidate_k` pool, then fuse with reciprocal rank fusion:

```
score(chunk) = Σ 1 / (rrf_k + rank)
```

A chunk that appears twice in one ranking is counted once, using its first rank. Citation metadata comes from the first ranking that mentioned it. Without rerank, fusion cuts to `k`; with rerank, fusion keeps `candidate_k` for the cross-encoder.

#### Rerank

Optional second stage. `bge-reranker-base` scores raw `(query, hit.text)` pairs — it does **not** use the dense query prefix — then sorts by that score and cuts to `k`. When rerank is off, the first-stage ranking is truncated to `k`.

The six retrieval configs are the product of `{dense, sparse, hybrid} × {rerank off, rerank on}`.

## Tests

Unit tests mock EDGAR with `pytest-httpx`. They never hit the network and never write under `data/`.

```bash
uv run pytest
```

## Live corpus (27 filings)

Requires a real `EDGAR_USER_AGENT` in `.env` (name + contact email). This is the only step that talks to the SEC. Cached responses live in `data/raw/` (gitignored); parsed sections in `data/parsed/` (also gitignored).

```bash
uv run python -m filing_rag ingest
```

Expect 27 HTML files in `data/raw/` and 27 JSON files in `data/parsed/`, each with items 1A, 7, and 7A. A second run should report `skipped=27`. High-volume filers (JPM, BAC, GS) paginate filing history; ingest follows those extra EDGAR submission files automatically.
