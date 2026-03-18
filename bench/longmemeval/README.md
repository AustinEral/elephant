# LongMemEval Benchmark

Evaluates long-term memory abilities using the [LongMemEval dataset](https://arxiv.org/abs/2410.10813) (Wu et al., 2024).

500 manually curated questions testing five core abilities: information extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention. Each question has its own conversation history — 500 independent banks per full run.

Two dataset sizes:
- **S** (~50 sessions per instance, ~115k tokens)
- **M** (~500 sessions per instance, ~1.5M tokens)

## Setup

```bash
# Download datasets from HuggingFace
mkdir -p data
# https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
# Place longmemeval_s_cleaned.json and/or longmemeval_m_cleaned.json in data/

# Build
cargo build --release --bin longmemeval-bench --bin longmemeval-view
```

Same environment variables as Elephant itself (`DATABASE_URL`, `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, embedding/reranker vars). Optional prompt-cache envs are also supported: `LLM_PROMPT_CACHE_ENABLED`, `OPENAI_PROMPT_CACHE_KEY`, `OPENAI_PROMPT_CACHE_RETENTION`, and `ANTHROPIC_PROMPT_CACHE_TTL`. Judge defaults to GPT-4o, overridable with `--judge-model` or `JUDGE_MODEL` env var.

Use an isolated Postgres instance for benchmark runs (see [LoCoMo README](../locomo/README.md#setup) for Docker instructions).

## Quick start

```bash
# Smoke test (1 instance)
cargo run --release --bin longmemeval-bench -- run --profile smoke --tag quick

# Full S benchmark (500 questions)
cargo run --release --bin longmemeval-bench -- run --profile full-s --tag baseline

# Full M benchmark (500 questions, ~500 sessions each)
cargo run --release --bin longmemeval-bench -- run --profile full-m --tag baseline-m

# Ingest only (create banks, skip QA)
cargo run --release --bin longmemeval-bench -- ingest --profile full-s --tag ingest

# QA only (reuse banks from ingest artifact)
cargo run --release --bin longmemeval-bench -- \
  qa bench/longmemeval/results/local/ingest.json \
  --tag qa-run

# Run specific instances
cargo run --release --bin longmemeval-bench -- \
  run --profile full-s \
  --instance q_123 --instance q_456 \
  --tag subset

# Parallel instances (4 at a time)
cargo run --release --bin longmemeval-bench -- \
  run --profile full-s \
  --instance-jobs 4 \
  --tag parallel

# Inspect results
cargo run --release --bin longmemeval-view -- bench/longmemeval/results/local/baseline.json

# Compare two runs
cargo run --release --bin longmemeval-view -- \
  bench/longmemeval/results/local/baseline.json \
  bench/longmemeval/results/local/ablation.json
```

## Flags

| Flag | Description |
|---|---|
| `run` | Ingest + consolidate + QA (full pipeline) |
| `ingest` | Ingest + consolidate only, no QA |
| `qa <artifact>` | Score against existing banks from ingest artifact |
| `--profile <name>` | `smoke` (1 instance), `full-s` (S dataset), `full-m` (M dataset) |
| `--config <path>` | JSON overlay on top of profile |
| `--tag <name>` | Output stem in `results/local/` |
| `--out <path>` | Explicit output path |
| `--instance <id>` | Run specific question instance (repeatable) |
| `--instance-limit <n>` | Limit number of instances |
| `--instance-jobs <n>` | Parallel instance processing (default: 1) |
| `--session-limit <n>` | Limit sessions ingested per instance (debug) |
| `--ingest-format <mode>` | `text` (default) or `json` |
| `--consolidation <mode>` | `end` (default), `per-session`, or `off` |
| `--judge-model <model>` | Override judge model (default: gpt-4o) |
| `--dataset <path>` | Override dataset file path |
| `--force` | Allow overwriting existing output |

## Output artifacts

Three files per run (same pattern as LoCoMo):

- **Summary JSON** (`<tag>.json`) — config, manifest, per-category accuracy, stage metrics, total stage usage
- **Questions JSONL** (`<tag>.questions.jsonl`) — per-question judge outcome
- **Debug JSONL** (`<tag>.debug.jsonl`) — retrieved context, reflect traces

Results default to `bench/longmemeval/results/local/`.

## Question categories

| Category | Count | Description |
|---|--:|---|
| single-session-user | ~150 | Facts from user messages in one session |
| single-session-assistant | ~80 | Facts from assistant messages in one session |
| single-session-preference | ~30 | User preferences expressed in one session |
| multi-session | ~120 | Information spanning multiple sessions |
| knowledge-update | ~50 | Changed user information over time |
| temporal-reasoning | ~40 | Time-based questions |
| abstention | ~30 | False-premise questions (should refuse) |

Abstention questions are identified by `_abs` suffix on `question_id` and routed to a dedicated judge prompt.

## View tool

```bash
# Single result
cargo run --release --bin longmemeval-view -- results/local/baseline.json

# With per-question table
cargo run --release --bin longmemeval-view -- results/local/baseline.json --verbose

# Compare two runs (shows delta per category)
cargo run --release --bin longmemeval-view -- results/local/a.json results/local/b.json
```
