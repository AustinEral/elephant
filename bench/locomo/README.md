# LoCoMo Benchmark

Evaluates long-term conversational memory using the [LoCoMo dataset](https://arxiv.org/abs/2402.17753) (ACL 2024).

## What changed

The benchmark now runs Elephant **in process** instead of talking to a running server over HTTP.

That change was deliberate:

- stage-token accounting is now precise
- retain/reflect/consolidate are benchmarked through the real runtime wiring
- turn provenance can be compared to LoCoMo `evidence`

## Scope

- Categories scored: **1-4 only**
- Category 5: excluded in code
- Default ingestion granularity: **turn-level**
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

The serious runner surface is now profile-driven. Versioned profile files live in `bench/locomo/profiles/`, and `--config <path>` can layer additional JSON overrides on top.

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
  bench/locomo/results/ingest.json \
  --out bench/locomo/results/ingest-qa.json

# Inspect a run
cargo run --release --bin view -- bench/locomo/results/baseline.json

# Compare two runs
cargo run --release --bin view -- \
  bench/locomo/results/baseline.json \
  bench/locomo/results/ablation.json
```

## Important flags

| Flag | Description |
|---|---|
| `run` | Fresh ingest, consolidate, then score QA |
| `ingest` | Ingest and consolidate only; do not run QA |
| `qa <artifact>` | Score QA against bank ids from an existing artifact; skips ingest and consolidation |
| `--profile <name>` | Load a versioned benchmark profile (`full`, `smoke`, `legacy-raw`) |
| `--config <path>` | Apply JSON overrides on top of the selected profile |
| `--tag <name>` | Save results to `bench/locomo/results/<name>.json` |
| `--conversation <id>` | Run a specific conversation; repeat to run an explicit set |
| `--ingest <turn|session|raw-json>` | Choose turn-level ingest, legacy session ingest, or unfair raw-json reproduction |
| `--consolidation <end|per-session|off>` | Control when consolidation runs |
| `--conversation-jobs <n>` | Parallel conversations |
| `--question-jobs <n>` | Parallel questions per conversation |
| `--session-limit <n>` | Debug-only session slice |
| `--question-limit <n>` | Debug-only question slice |

## Artifact quality

A serious run now records:

- manifest and run provenance
- per-stage token/call/latency metrics
- per-question judge outcome
- retrieved facts
- retrieved turn refs
- evidence hit / evidence recall

Schema: [results-format.md](/docs/results-format.md)

## Publication status

The checked-in historical artifacts are still legacy outputs from the old harness. Use the new runner for any benchmark claim or comparison work.
