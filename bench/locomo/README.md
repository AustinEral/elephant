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

LoCoMo does not read the root runtime `.env`.

The benchmark input surface is:

- checked-in contract profile TOML in `bench/locomo/profiles/`
- optional execution overlay TOML passed via `--config`
- benchmark secrets from `bench/secrets.example.env` or process env

Benchmark secrets are namespaced and separate from the runtime server:

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
- local embeddings: `models/bge-small-en-v1.5`
- local reranker: `models/ms-marco-MiniLM-L-6-v2`

Use `--config` only when you need to change execution/provenance settings such as dataset path, output path, concurrency, or local model/database locations.

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
```

That matches the default benchmark execution database URL, so no extra runtime env is needed if you use the standard local benchmark setup.

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

The serious runner surface is profile-driven. Versioned profile files live in `bench/locomo/profiles/`, and `--config <path>` layers an execution-only TOML overlay on top.

Reusable local configs can live alongside the benchmark in `bench/locomo/configs/`.

Profiles and configs are intentionally different:

- profiles are canonical benchmark contracts, like `full`, `smoke`, and `legacy-raw`
- configs are small execution overlays that change only machine-local knobs on top of a profile

In practice, use `--profile full` as the base benchmark contract and use `--config` only for operator convenience, like changing question parallelism.

The CLI is intentionally strict. Old flag aliases were removed so benchmark commands stay unambiguous.

## Quick start

```bash
# Quick smoke test
cargo run --release --bin locomo-bench -- \
  run --profile smoke --secrets-env-file bench/secrets.env --tag quick

# Full benchmark
cargo run --release --bin locomo-bench -- \
  run --profile full --secrets-env-file bench/secrets.env --tag baseline

# Ingest only
cargo run --release --bin locomo-bench -- \
  ingest --profile full --secrets-env-file bench/secrets.env --tag ingest

# QA only from existing banks
cargo run --release --bin locomo-bench -- \
  qa \
  bench/locomo/results/local/ingest.json \
  --secrets-env-file bench/secrets.env \
  --out bench/locomo/results/local/ingest-qa.json

# Execution-only overlay with 5 question workers
cargo run --release --bin locomo-bench -- \
  run \
  --profile full \
  --config bench/locomo/configs/question-jobs-5.toml \
  --secrets-env-file bench/secrets.env \
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
| `--config <path>` | Apply a TOML execution overlay on top of the selected profile |
| `--secrets-env-file <path>` | Load benchmark API keys from a separate benchmark secrets env file |
| `--tag <name>` | Name the output stem in `results/local/` by default |
| `--run-id <name>` | Set the published run id for `publish` |
| `--include-debug` | Also export `debug.jsonl.gz` for `publish` |
| `--force` | Allow overwriting an existing output path and sidecars |

Fresh `run`, `ingest`, and `merge` outputs now refuse to overwrite existing summary/sidecar files unless you pass `--force`. `qa` also refuses to overwrite when `--tag` resolves to an existing file; pass `--force` to override.

Contract-affecting slice/ingest/consolidation settings live in the checked-in profile, not in ad hoc CLI flags.

## Merge constraints

`merge` is strict. It is meant to assemble one canonical artifact from disjoint subset runs of the same benchmark contract, not to average or reconcile different protocols.

The input artifacts must match on:

- dataset fingerprint
- contract hash
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
