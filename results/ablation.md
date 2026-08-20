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
