# LoCoMo Benchmark

Evaluates long-term conversational memory using the [LoCoMo dataset](https://arxiv.org/abs/2402.17753) (ACL 2024).

**Categories**: single-hop, multi-hop, temporal, open-domain, unanswerable

**Pipeline**: Ingest sessions → consolidate → ask questions via reflect → score with LLM judge

## Setup

```bash
# Download dataset
mkdir -p data
curl -o data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

# Build
cargo build --release --bin locomo-bench --bin view
```

The judge LLM can be configured independently via `JUDGE_PROVIDER`, `JUDGE_API_KEY`, `JUDGE_MODEL` env vars (falls back to `LLM_*` equivalents).

## Quick start

```bash
# Start Elephant
docker compose up -d

# Quick smoke test
cargo run --release --bin locomo-bench -- --tag quick --max-conversations 1 --max-sessions 1 --max-questions 5

# Full benchmark (all 10 conversations, ~2000 questions)
cargo run --release --bin locomo-bench -- --tag baseline

# Compare two runs
cargo run --release --bin view -- bench/locomo/results/quick.json bench/locomo/results/baseline.json
```

## Arguments

| Flag | Description | Default |
|---|---|---|
| `--tag <name>` | Save results to `results/<name>.json` | `locomo` |
| `--output <path>` | Override output path (ignores tag) | |
| `--max-conversations <N>` | Limit number of conversations | all |
| `--max-sessions <N>` | Limit ingested sessions per conversation | all |
| `--max-questions <N>` | Limit questions per conversation | all |
| `--concurrency <N>` | Parallel conversations | 1 |
| `--question-concurrency <N>` | Parallel questions per conversation | 1 |
| `--no-consolidate` | Skip consolidation after ingestion | consolidate on |
| `--consolidate-per-session` | Consolidate after each session (incremental) | off |
| `--resume <path>` | Reuse bank IDs from previous results (skip ingestion) | |
| `--bank-id <id>` | Reuse a specific bank (skip ingestion) | |
| `--ingest-only` | Ingest and consolidate only, skip questions | off |
| `--judge-model <name>` | Override judge model | from env |
| `--api-url <url>` | Elephant server URL | `http://localhost:3001` |
| `--data <path>` | Dataset path | `data/locomo10.json` |

### View tool

```bash
# View a single result
cargo run --release --bin view -- bench/locomo/results/baseline.json

# Compare two runs
cargo run --release --bin view -- bench/locomo/results/reranker.json bench/locomo/results/baseline.json

# Filter to a single conversation
cargo run --release --bin view -- --conv conv-26 bench/locomo/results/reranker.json

# Compare a single conversation across runs
cargo run --release --bin view -- --conv conv-26 bench/locomo/results/reranker.json bench/locomo/results/baseline.json
```

| Flag | Description |
|---|---|
| `--conv <id>` | Filter results to a single conversation |

## Scoring

- **LLM-as-judge** (primary): Binary CORRECT/WRONG via the configured LLM.
- **Token F1** (reference): Token-level overlap between prediction and gold answer. Included for diagnostics only.

## Baseline results

Single conversation (conv-26), 154 questions, with consolidation:

| Category | Accuracy | n |
|---|---|---|
| multi-hop | 91.9% | 37 |
| open-domain | 77.1% | 70 |
| single-hop | 78.1% | 32 |
| temporal | 76.9% | 13 |
| unanswerable | 100.0% | 2 |
| **TOTAL** | **81.2%** | **154** |

Model: Sonnet 4.6, Judge: Sonnet 4.6, Embeddings: bge-small-en-v1.5 (local)
