# LoCoMo Benchmark

Evaluates long-term conversational memory using the [LoCoMo dataset](https://arxiv.org/abs/2402.17753) (ACL 2024).

## What changed

The benchmark now runs Elephant **in process** instead of talking to a running server over HTTP.

That change was deliberate:

- stage-token accounting is now precise
- retain/reflect/consolidate are benchmarked through the real runtime wiring
- turn provenance is still available when `--ingest turn` is used

## Scope

- Categories scored: **1-4 only**
- Category 5: excluded in code
- Default ingestion granularity: **session-level**
- Image handling: BLIP captions inline by default

## Setup

```bash
# Download dataset
mkdir -p data
curl -o data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

# Build
cargo build --release --bin locomo-bench --bin view
```

The runner uses the same environment variables as Elephant itself:

- `DATABASE_URL`
- `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`
- `RETAIN_LLM_MODEL`, `REFLECT_LLM_MODEL` if split models are desired
- embedding and reranker env vars

Judge env vars remain independently overridable through `JUDGE_*`.

For benchmark hygiene, use an isolated Postgres instance instead of your normal development database. Docker is the easiest option:

```bash
docker run -d \
  --name elephant-bench-pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=elephant_bench \
  -p 5433:5432 \
  -v elephant-bench-pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:pg16

export DATABASE_URL=postgres://postgres:postgres@localhost:5433/elephant_bench
```

Using a named volume makes the benchmark database persistent:

- container crash or restart: data stays
- `docker stop` / `docker start`: data stays
- delete and recreate the container: data stays, as long as the volume is kept
- delete the volume: data is gone

To fully reset the benchmark database:

```bash
docker rm -f elephant-bench-pg
docker volume rm elephant-bench-pgdata
```

Recommended workflow:

- keep one dedicated benchmark Postgres container
- reset the database between benchmark series
- do not reuse your normal local dev database for benchmark claims

Results layout is split by purpose:

- transient local outputs stay in `bench/locomo/results/local/`
- final merged artifacts should be promoted intentionally to `bench/locomo/results/canonical/`
- deprecated legacy artifacts live in `bench/locomo/results/archive/legacy-v0/`

The serious runner surface is now profile-driven. Versioned profile files live in `bench/locomo/profiles/`, and `--config <path>` can layer additional JSON overrides on top.

Reusable local configs can live alongside the benchmark in `bench/locomo/configs/`.

Profiles and configs are intentionally different:

- profiles are canonical benchmark shapes, like `full`, `smoke`, and `legacy-raw`
- configs are small local overlays that change only a few knobs on top of a profile

In practice, use `--profile full` as the base benchmark contract and use `--config` only for operator convenience, like changing question parallelism.

The CLI is intentionally strict. Old flag aliases were removed so benchmark commands stay unambiguous.

## Quick start

```bash
# Quick smoke test
cargo run --release --bin locomo-bench -- run --profile smoke --tag quick

# Full benchmark
cargo run --release --bin locomo-bench -- run --profile full --tag baseline

# Single conversation gate
cargo run --release --bin locomo-bench -- \
  run \
  --profile full \
  --conversation conv-26 \
  --tag conv26

# Ingest only
cargo run --release --bin locomo-bench -- ingest --profile full --tag ingest

# QA only from existing banks
cargo run --release --bin locomo-bench -- \
  qa \
  bench/locomo/results/local/ingest.json \
  --out bench/locomo/results/local/ingest-qa.json

# Single conversation with 5 question workers
cargo run --release --bin locomo-bench -- \
  run \
  --profile full \
  --config bench/locomo/configs/question-jobs-5.json \
  --conversation conv-26 \
  --tag series1-conv-26

# Merge disjoint subset artifacts into one canonical result
cargo run --release --bin locomo-bench -- \
  merge \
  bench/locomo/results/local/batch-a.json \
  bench/locomo/results/local/batch-b.json \
  --out bench/locomo/results/canonical/full.json

# Inspect a canonical run
cargo run --release --bin view -- bench/locomo/results/canonical/full.json

# Compare two runs
cargo run --release --bin view -- \
  bench/locomo/results/canonical/baseline.json \
  bench/locomo/results/canonical/ablation.json
```

By default:

- `run` and `ingest` write to `bench/locomo/results/local/`
- `merge` also writes to `bench/locomo/results/local/`
- `qa <artifact>` writes back to the source artifact unless `--out` is set

Use `--out bench/locomo/results/canonical/<name>.json` when you intentionally want to promote a merged artifact into the canonical record.

## Important flags

| Flag | Description |
|---|---|
| `run` | Fresh ingest, consolidate, then score QA |
| `ingest` | Ingest and consolidate only; do not run QA |
| `qa <artifact>` | Score QA against bank ids from an existing artifact; skips ingest and consolidation |
| `merge <artifact>...` | Combine compatible subset artifacts into one canonical summary + sidecars |
| `--profile <name>` | Load a versioned benchmark profile (`full`, `smoke`, `legacy-raw`) |
| `--config <path>` | Apply JSON overrides on top of the selected profile |
| `--tag <name>` | Name the output stem in `results/local/` by default |
| `--conversation <id>` | Run a specific conversation; repeat to run an explicit set |
| `--ingest <turn|session|raw-json>` | Choose explicit turn ingest, the default session ingest, or unfair raw-json reproduction |
| `--consolidation <end|per-session|off>` | Control when consolidation runs |
| `--conversation-jobs <n>` | Parallel conversations |
| `--question-jobs <n>` | Parallel questions per conversation |
| `--session-limit <n>` | Debug-only session slice |
| `--question-limit <n>` | Debug-only question slice |
| `--force` | Allow overwriting an existing output path and sidecars |

Fresh `run`, `ingest`, and `merge` outputs now refuse to overwrite existing summary/sidecar files unless you pass `--force`. `qa` still allows in-place updates when it writes back to its source artifact by default.

## Merge constraints

`merge` is strict. It is meant to assemble one canonical artifact from disjoint subset runs of the same benchmark contract, not to average or reconcile different protocols.

The input artifacts must match on:

- dataset fingerprint
- protocol version
- mode
- ingest mode and consolidation mode
- category filter and slice limits
- judge model and Elephant runtime stack
- prompt hashes and runtime tuning knobs

They must also have:

- disjoint conversation scope
- no duplicate `question_id` values
- new-style sidecars present: `*.questions.jsonl` and `*.debug.jsonl`

These fields are treated as provenance notes, not blockers:

- profile label
- commit
- dirty-tree state
- question and conversation concurrency

If a full benchmark is assembled from batches, the merged artifact is the one to treat as canonical.

## Artifact quality

A serious run now records:

- manifest and run provenance
- prompt hashes and runtime tuning knobs
- run-level and per-conversation timing
- per-stage token/call/latency metrics
- per-conversation bank construction stats
- per-question judge outcome in `*.questions.jsonl`
- retrieved facts and reflect traces in `*.debug.jsonl`
- retrieved turn refs when turn provenance is preserved
- evidence hit / evidence recall
- merge provenance when a full benchmark is assembled from disjoint subset runs

Schema: [results-format.md](/docs/results-format.md)

## Publication status

The checked-in historical artifacts are still legacy outputs from the old harness. Use the new runner for any benchmark claim or comparison work.
