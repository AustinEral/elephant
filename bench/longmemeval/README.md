# LongMemEval Benchmark

Evaluates long-term memory abilities using the [LongMemEval dataset](https://arxiv.org/abs/2410.10813) (Wu et al., 2024).

500 manually curated questions testing five core abilities: information extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention. Each question has its own conversation history — 500 independent banks per full run.

Two dataset sizes:
- **S** (~50 sessions per instance, ~115k tokens)
- **M** (~500 sessions per instance, ~1.5M tokens)

Current checked-in profiles use:
- round ingest
- `retain_chunk_max_tokens = 2048`
- `retain_chunk_overlap_tokens = 64`

## Setup

```bash
# Download datasets from HuggingFace
mkdir -p data
# https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
# Place longmemeval_s_cleaned.json and/or longmemeval_m_cleaned.json in data/

# Build
cargo build -p elephant-bench --release --bin longmemeval-bench --bin longmemeval-view
```

LongMemEval does not read the root runtime `.env`.

The benchmark input surface is:

- checked-in contract profile TOML in `bench/longmemeval/profiles/`
- optional execution overlay TOML passed via `--config`
- benchmark secrets from `bench/secrets.example.env` or process env

Set up benchmark secrets separately:

```bash
cp bench/secrets.example.env bench/secrets.env
```

Fill in the keys you need:

- `ELEPHANT_BENCH_RUNTIME_API_KEY`
- `ELEPHANT_BENCH_JUDGE_API_KEY`
- `ELEPHANT_BENCH_EMBEDDING_API_KEY` only for API-backed embeddings
- `ELEPHANT_BENCH_RERANKER_API_KEY` only for API-backed rerankers

The standard local execution defaults are:

- benchmark Postgres: `postgres://postgres:postgres@localhost:5433/elephant_bench`
- repo-local ONNX Runtime under `lib/onnxruntime-*/lib` is auto-detected for local embeddings/reranking
- local embeddings: `models/bge-small-en-v1.5`
- local reranker: `models/ms-marco-MiniLM-L-6-v2`

Use an isolated Postgres instance for benchmark runs (see [LoCoMo README](../locomo/README.md#setup) for Docker instructions).

## Quick start

```bash
# Smoke test (1 instance, first session only)
cargo run --release --bin longmemeval-bench -- \
  run --profile smoke --secrets-env-file bench/secrets.env --tag quick

# Fast probe (1 session only; choose an instance with --instance)
cargo run --release --bin longmemeval-bench -- \
  run --profile probe --instance 8aef76bc --secrets-env-file bench/secrets.env --tag probe

# Full S benchmark (500 questions)
cargo run --release --bin longmemeval-bench -- \
  run --profile full-s --secrets-env-file bench/secrets.env --tag baseline

# Full M benchmark (500 questions, ~500 sessions each)
cargo run --release --bin longmemeval-bench -- \
  run --profile full-m --secrets-env-file bench/secrets.env --tag baseline-m

# Ingest only (create banks, skip QA)
cargo run --release --bin longmemeval-bench -- \
  ingest --profile full-s --secrets-env-file bench/secrets.env --tag ingest

# QA only (reuse banks from ingest artifact)
cargo run --release --bin longmemeval-bench -- \
  qa bench/longmemeval/results/local/ingest.json \
  --secrets-env-file bench/secrets.env \
  --tag qa-run

# Execution-only overlay example
cargo run --release --bin longmemeval-bench -- \
  run --profile full-s \
  --config bench/longmemeval/configs/instance-jobs-4.toml \
  --secrets-env-file bench/secrets.env \
  --tag local-run

# Inspect results
cargo run -p elephant-bench --release --bin longmemeval-view -- bench/longmemeval/results/local/baseline.json

# Verify a shard set before merge/publication
cargo run --release --bin longmemeval-bench -- \
  verify \
  bench/longmemeval/results/local/shard-a.json \
  bench/longmemeval/results/local/shard-b.json

# Doctor a shard set before publishing or promoting it
cargo run --release --bin longmemeval-bench -- \
  doctor \
  bench/longmemeval/results/local/shard-a.json \
  bench/longmemeval/results/local/shard-b.json

# Compare two runs
cargo run -p elephant-bench --release --bin longmemeval-view -- \
  bench/longmemeval/results/local/baseline.json \
  bench/longmemeval/results/local/ablation.json
```

## Flags

| Flag | Description |
|---|---|
| `run` | Ingest + consolidate + QA (full pipeline) |
| `ingest` | Ingest + consolidate only, no QA |
| `qa <artifact>` | Score against existing banks from ingest artifact |
| `verify <artifact...>` | Validate artifact structure and shard compatibility without running the benchmark |
| `doctor <artifact...>` | Check publication readiness and canonical-slice coverage from artifact provenance |
| `--profile <name>` | `smoke` (1 fixed instance, first session only), `probe` (first session only for a chosen instance), `kimi-smoke` (Kimi K2.6 via OpenRouter), `full-s` (S dataset), `full-m` (M dataset) |
| `--config <path>` | TOML execution overlay on top of profile |
| `--instance <id>` | Execution-only shard selector within the profile slice |
| `--instance-limit <n>` | Execution-only shard window within the profile slice |
| `--instance-offset <n>` | Execution-only shard offset within the profile slice |
| `--secrets-env-file <path>` | Benchmark secrets env file |
| `--tag <name>` | Output stem in `results/local/` |
| `--out <path>` | Explicit output path |
| `--instance-jobs <n>` | Parallel instance processing from execution overlay / CLI override |
| `--judge-model <model>` | `qa` only; changes the resolved benchmark contract for that QA pass |
| `--dataset <path>` | Override dataset file path |
| `--force` | Allow overwriting existing output |

Contract-affecting slice, ingest, consolidation, and judge defaults live in the checked-in profile. Execution-time shard controls stay outside the contract hash.

`verify` is the first publication-hygiene command. It checks artifact structure, per-artifact consistency, and multi-artifact shard compatibility for LongMemEval result artifacts.

`doctor` builds on `verify`. When the artifact contract explicitly lists the canonical instance slice, `doctor` also checks that the artifact set fully covers that slice before promotion or publication.

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
cargo run -p elephant-bench --release --bin longmemeval-view -- results/local/baseline.json

# With per-question table
cargo run -p elephant-bench --release --bin longmemeval-view -- results/local/baseline.json --verbose

# Compare two runs (shows delta per category)
cargo run -p elephant-bench --release --bin longmemeval-view -- results/local/a.json results/local/b.json
```
