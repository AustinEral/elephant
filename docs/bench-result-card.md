# Elephant Benchmark Result Card

## Run: temporal-consolidation (2026-03-08)

### Scope

| Field | Value |
|---|---|
| Dataset | LoCoMo (ACL 2024) |
| Conversations | conv-26 only (1 of 10) |
| Sessions ingested | 26 |
| Questions | 154 |
| LoCoMo categories evaluated | **Category 1-4 only** (single-hop, temporal, multi-hop, open-domain) |
| Category 5 (adversarial) | None present in this slice |
| Local answer subtypes | unanswerable = 2 |

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

| Category | Accuracy | n |
|---|---|---|
| Temporal | **100.0%** | 13 |
| Multi-hop | 94.6% | 37 |
| Open-domain | 94.3% | 70 |
| Single-hop | 90.6% | 32 |
| **Overall** | **94.2%** | **154** |

Local subtypes: unanswerable 100.0% (2/2)

### Efficiency

| Metric | Value |
|---|---|
| Total runtime | 103m 10s |
| Avg reflect time | 38.6s/question |
| Token usage | Not yet instrumented |

### Variance

Not yet measured. A judge-only rerun and a full rerun are planned to establish variance bounds. Backboard's benchmark notes 2-3% variance across runs with GPT-4.1 as judge; similar variance expected here.

### Reproduction

```bash
# Commit: f58610f
# Requires: Elephant server running on localhost:3001

# Full run (fresh bank, ingestion + consolidation + questions)
cargo run --release --bin locomo-bench -- \
  --tag temporal-consolidation \
  --max-conversations 1 \
  --question-concurrency 5

# Questions only (reuse existing bank)
cargo run --release --bin locomo-bench -- \
  --tag temporal-consolidation-rerun \
  --max-conversations 1 \
  --question-concurrency 5 \
  --resume bench/locomo/results/temporal-consolidation.json
```

Results JSON: `bench/locomo/results/temporal-consolidation.json`
Schema: [results-schema.md](results-schema.md)

### Caveats

- **Single conversation only** — this is a stress test, not a full leaderboard claim
- **No token/cost data yet** — instrumentation required before full run
- **No variance data yet** — single run, no repeat measurements
- **Same model for all stages** — extraction, reflection, consolidation, and judging all use Sonnet 4.6
- **BLIP-2 image captions** — images replaced with captions per LoCoMo evaluation protocol
