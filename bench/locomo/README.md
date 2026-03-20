# LoCoMo Benchmark

Evaluates long-term conversational memory using the [LoCoMo dataset](https://arxiv.org/abs/2402.17753) (ACL 2024).

The checked-in canonical result is a full 10-conversation run (series1, 1,540 questions, Cat.1–4). Per-conversation artifacts live in `results/canonical/series1/`.

Related docs:

- [protocol.md](protocol.md) — benchmark methodology and publication standard
- [publish.md](publish.md) — public bundle format and Pages/Releases workflow
- [result-card.md](result-card.md) — current checked-in reference run
- [competitors.md](competitors.md) — competitor methodology notes
- [results-format.md](results-format.md) — artifact schema
- [results/README.md](results/README.md) — results layout
- [internal/README.md](internal/README.md) — benchmark planning and internal comparison notes

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
- optional reasoning-effort envs:
  - `RETAIN_EXTRACT_REASONING_EFFORT`
  - `RETAIN_RESOLVE_REASONING_EFFORT`
  - `RETAIN_GRAPH_REASONING_EFFORT`
  - `REFLECT_REASONING_EFFORT`
  - `CONSOLIDATE_REASONING_EFFORT`
  - `OPINION_MERGE_REASONING_EFFORT`
- optional prompt-cache envs:
  - `LLM_PROMPT_CACHE_ENABLED`
  - `OPENAI_PROMPT_CACHE_KEY`
  - `OPENAI_PROMPT_CACHE_RETENTION`
  - `ANTHROPIC_PROMPT_CACHE_TTL`
- embedding and reranker env vars

Judge env vars remain independently overridable through `JUDGE_*`.

For OpenAI `gpt-5.4-mini`, `REFLECT_REASONING_EFFORT=high` is a useful non-default benchmark tuning when you want stronger reflect tool use, at the cost of higher latency and token usage.

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
- historical artifacts live in `bench/locomo/results/archive/legacy-v0/`

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
  --tag conv-26-q5

# Merge disjoint subset artifacts into one canonical result
cargo run --release --bin locomo-bench -- \
  merge \
  bench/locomo/results/local/batch-a.json \
  bench/locomo/results/local/batch-b.json \
  --out bench/locomo/results/canonical/full.json

# Export a Pages-friendly publish bundle from one canonical artifact
cargo run --release --bin locomo-bench -- \
  publish \
  bench/locomo/results/canonical/full.json \
  --out bench/locomo/published \
  --run-id 2026-03-10-series1

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
- `qa <artifact>` uses `--tag` for output path, or writes back to the source artifact if neither `--tag` nor `--out` is set
- `publish <artifact>` writes a Pages-friendly bundle to `bench/locomo/published/`

Use `--out bench/locomo/results/canonical/<name>.json` when you intentionally want to promote a merged artifact into the canonical record.
Use `publish` after that to stage a public bundle with `summary.json` plus gzipped sidecars.

## Important flags

| Flag | Description |
|---|---|
| `run` | Fresh ingest, consolidate, then score QA |
| `ingest` | Ingest and consolidate only; do not run QA |
| `qa <artifact>` | Score QA against bank ids from an existing artifact; skips ingest and consolidation |
| `merge <artifact>...` | Combine compatible subset artifacts into one canonical summary + sidecars |
| `publish <artifact>` | Export a publishable bundle with `index.json`, `summary.json`, and `questions.jsonl.gz` |
| `--profile <name>` | Load a versioned benchmark profile (`full`, `smoke`, `legacy-raw`) |
| `--config <path>` | Apply JSON overrides on top of the selected profile |
| `--tag <name>` | Name the output stem in `results/local/` by default |
| `--conversation <id>` | Run a specific conversation; repeat to run an explicit set |
| `--ingest <mode>` | Choose ingest mode: `turn`, `session` (default), or `raw-json` |
| `--consolidation <mode>` | Control consolidation timing: `end`, `per-session`, or `off` |
| `--conversation-jobs <n>` | Parallel conversations |
| `--question-jobs <n>` | Parallel questions per conversation |
| `--run-id <name>` | Set the published run id for `publish` |
| `--include-debug` | Also export `debug.jsonl.gz` for `publish` |
| `--session-limit <n>` | Debug-only session slice |
| `--question-limit <n>` | Debug-only question slice |
| `--force` | Allow overwriting an existing output path and sidecars |

Fresh `run`, `ingest`, and `merge` outputs now refuse to overwrite existing summary/sidecar files unless you pass `--force`. `qa` also refuses to overwrite when `--tag` resolves to an existing file; pass `--force` to override.

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
- per-stage token/call/cache/latency metrics
- per-conversation bank construction stats
- per-question judge outcome in `*.questions.jsonl`
- retrieved facts and reflect traces in `*.debug.jsonl`
- retrieved turn refs when turn provenance is preserved
- evidence hit / evidence recall
- merge provenance when a full benchmark is assembled from disjoint subset runs

Schema: [results-format.md](results-format.md)

## Publishing

The benchmark-native artifact contract under `bench/locomo/results/` is not the same as the public publication bundle.

Use this flow:

1. Run or merge a canonical artifact under `bench/locomo/results/canonical/`
2. Export a public bundle with `locomo-bench publish`
3. Copy that bundle into a dedicated public benchmarks repo
4. Keep optional `debug.jsonl.gz` files in GitHub Releases rather than the default Pages payload

Public bundle details: [publish.md](publish.md)

## Publication status

Use the current runner and current artifact schema for any benchmark claim or comparison work.
