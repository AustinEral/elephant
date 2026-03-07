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
| `--raw-json` | Send raw dataset JSON per session (see below) | off |
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

## Image handling

The LoCoMo dataset includes images shared during conversations. Per the [paper's evaluation protocol](https://arxiv.org/abs/2402.17753), images are replaced with their BLIP-2 captions inline in the conversation text. This is the default behavior.

### `--raw-json` mode

Some competing implementations send the raw dataset JSON per session, which includes extra metadata fields (`query`, `img_url`, `dia_id`) beyond what the LoCoMo paper specifies as fair input. In particular, the `query` field contains the image search term used during dataset construction, which often directly answers benchmark questions. The `--raw-json` flag reproduces this behavior for comparison purposes.

## Scoring

- **LLM-as-judge** (primary): Binary CORRECT/WRONG via the configured LLM.
- **Token F1** (reference): Token-level overlap between prediction and gold answer. Included for diagnostics only.

## Results

Single conversation (conv-26), 154 questions, with consolidation:

| Category | Accuracy | n |
|---|---|---|
| temporal | 100.0% | 13 |
| multi-hop | 91.9% | 37 |
| open-domain | 91.4% | 70 |
| single-hop | 81.2% | 32 |
| unanswerable | 100.0% | 2 |
| **TOTAL** | **90.3%** | **154** |

Model: Sonnet 4.6, Judge: Sonnet 4.6, Embeddings: bge-small-en-v1.5 (local), Reranker: ms-marco-MiniLM-L-6-v2 (local)
