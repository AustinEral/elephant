# Elephant Benchmark Protocol

Single source of truth for all benchmark methodology. All other benchmark docs reference this file.

## Dataset

| Field | Value |
|---|---|
| Dataset | [LoCoMo](https://arxiv.org/abs/2402.17753) (ACL 2024) |
| Total conversations | 10 |
| Total questions (Cat.1-4) | 1,540 |
| Total questions (incl Cat.5) | 1,986 |
| Categories evaluated | **1-4 only** (single-hop, temporal, multi-hop, open-domain) |
| Category 5 (adversarial) | **Excluded** — consistent with Mnemis, Backboard, and other recent systems |

### Image handling

Per the [LoCoMo evaluation protocol](https://arxiv.org/abs/2402.17753), images are replaced with their BLIP-2 captions inline in the conversation text. This is the default behavior. See `--raw-json` flag for alternative.

## Answering Stack

| Component | Model | Notes |
|---|---|---|
| Extraction (retain) | Configurable via `LLM_MODEL` | Structured fact extraction from conversations |
| Entity resolution | Same as extraction model | LLM verification of embedding-matched entities |
| Consolidation | Same as extraction model | Topic-scoped observation synthesis |
| Reflection (reflect) | Same as extraction model | Agentic retrieval + synthesis loop |
| Embeddings | bge-small-en-v1.5 | Local ONNX, 384 dimensions |
| Reranker | ms-marco-MiniLM-L-6-v2 | Local ONNX cross-encoder |

## Retrieval Config

| Parameter | Value | Env var |
|---|---|---|
| Retrieval channels | 4 (semantic, keyword, graph, temporal) | — |
| Fusion | Reciprocal Rank Fusion (RRF) | — |
| Retriever limit | 20 per channel | `RETRIEVER_LIMIT` |
| Rerank top-N | 50 | `RERANK_TOP_N` |
| Reranker max seq length | 512 tokens | `RERANKER_MAX_SEQ_LEN` |

## Memory Config

| Parameter | Value | Env var |
|---|---|---|
| Consolidation batch size | 8 | `CONSOLIDATION_BATCH_SIZE` |
| Consolidation recall budget | 512 tokens | `CONSOLIDATION_RECALL_BUDGET` |
| Consolidation timing | After all sessions ingested | `--consolidate-per-session` to change |
| Temporal annotations | Yes — facts include `| occurred: date` suffix | — |

## Reflect Agent Config

| Parameter | Value |
|---|---|
| Max iterations | 8 |
| Tools | search_observations, recall, expand, done |
| Forced sequence | iter 0→search_observations, 1→recall, 2..N-1→auto, last→done-only |

## Evaluation Protocol

| Field | Value |
|---|---|
| Primary metric | LLM-as-judge binary accuracy (CORRECT/WRONG) |
| Secondary metric | Token F1 (reference only, not used for ranking) |
| Judge model | Configurable via `JUDGE_MODEL` (defaults to `LLM_MODEL`) |
| Judge temperature | 0 |
| Judge prompt | [`bench/locomo/judge_answer.txt`](/bench/locomo/judge_answer.txt) — binary correctness with tolerance for format differences |

## Scoring

- **Accuracy** = correct / total per category and overall
- **Per-conversation breakdown** reported when multiple conversations run
- Questions with no response (timeout/error) count as incorrect

## Comparison Policy

- We only compare Elephant directly to runs that disclose category scope, judge model, and question count
- Single-conversation results are stress tests, not leaderboard claims
- Full-benchmark results (all 10 conversations, Cat.1-4) are separated from per-conversation slices
- Competitor numbers are tagged by evidence type (self-reported, reproduced, comparison-table)

## Reproduction

```bash
# Start Elephant
docker compose up -d

# Full benchmark (all 10 conversations)
cargo run --release --bin locomo-bench -- --tag full --question-concurrency 5

# Single conversation gate
cargo run --release --bin locomo-bench -- --tag conv26 --max-conversations 1 --question-concurrency 5

# View results
cargo run --release --bin view -- bench/locomo/results/full.json
```

All results saved as JSON in `bench/locomo/results/`. Schema defined in [results-format.md](results-format.md).
