# Elephant Benchmark Result Card

## Run: series1-conv-26 (2026-03-10)

### Status

| Field | Value |
|---|---|
| Publication status | **Clean single-conversation reference run** |
| Why not a headline claim | `conv-26` only, no rerun/variance note, not full LoCoMo |
| Required action | Run the full Cat.1-4 benchmark on frozen config and add a rerun or judge-variance note |

### Scope

| Field | Value |
|---|---|
| Dataset | LoCoMo (ACL 2024) |
| Conversations | `conv-26` only (1 of 10) |
| Sessions ingested | 19 |
| Questions scored | 152 |
| Categories | Cat.1-4 only |
| Ingest mode | Session-level |
| Consolidation | End-of-run |
| Question concurrency | 5 |

Full protocol: [benchmark-protocol.md](benchmark-protocol.md)

### Stack

| Component | Model |
|---|---|
| Extraction (retain) | Sonnet 4.6 (`claude-sonnet-4-6`) |
| Reflection (reflect) | Sonnet 4.6 (`claude-sonnet-4-6`) |
| Consolidation | Sonnet 4.6 (`claude-sonnet-4-6`) |
| Judge | Sonnet 4.6 (`claude-sonnet-4-6`) |
| Embeddings | bge-small-en-v1.5 (local ONNX) |
| Reranker | ms-marco-MiniLM-L-6-v2 (local ONNX) |

### Results

| Slice | Accuracy | n |
|---|---|---|
| Clean Cat.1-4 reference slice | **94.7%** | **152** |

Category breakdown:

| Category | Accuracy | n |
|---|---|---|
| Single-hop | 93.8% | 32 |
| Multi-hop | 94.6% | 37 |
| Temporal | 84.6% | 13 |
| Open-domain | **97.1%** | 70 |

### Efficiency

| Metric | Value |
|---|---|
| Total runtime | 135m 15s |
| Ingest | 81m 38s |
| Consolidation | 32m 03s |
| QA | 21m 34s |
| Avg QA time | 8.5s/question |
| Token usage (total) | 4,998,181 |
| Token usage (excluding judge) | 4,899,479 |

### Variance

Not yet measured. Elephant still needs either a clean rerun or a judge-only variance pass before this should be cited beyond internal reference use.

### Reproduction

```bash
cargo run --release --bin locomo-bench -- \
  run \
  --profile full \
  --config bench/locomo/configs/question-jobs-5.json \
  --conversation conv-26 \
  --tag series1-conv-26
```

Artifact: `bench/locomo/results/local/series1-conv-26.json`
Schema: [results-format.md](results-format.md)

### Caveats

- **Single conversation only** — useful reference slice, not a full benchmark claim
- **No variance data yet** — still single-run only
- **Session ingest** — appropriate for current Elephant benchmarking, but turn-level evidence-trace metrics are weaker in this mode
- **Same model for all stages** — extraction, reflection, consolidation, and judging all use Sonnet 4.6
- **BLIP-2 image captions** — images replaced with captions per LoCoMo evaluation protocol
