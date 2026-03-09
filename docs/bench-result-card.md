# Elephant Benchmark Result Card

## Run: temporal-consolidation (2026-03-08)

### Status

| Field | Value |
|---|---|
| Publication status | **Not leaderboard-valid yet** |
| Why | This artifact predates the 2026-03-09 harness fix that hard-excludes LoCoMo Category 5 |
| Required action | Rerun with the current harness before using externally |

### Scope

| Field | Value |
|---|---|
| Dataset | LoCoMo (ACL 2024) |
| Conversations | conv-26 only (1 of 10) |
| Sessions ingested | 26 |
| Questions scored in legacy JSON | 154 |
| Protocol-correct Cat.1-4 slice | 152 |
| Cat.5 leakage in legacy JSON | 2 (`unanswerable`) |

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
| Legacy artifact (includes leaked Cat.5) | 94.2% | 154 |
| Protocol-correct Cat.1-4 slice | **94.1%** | **152** |

Cat.1-4 breakdown from the protocol-correct slice:

| Category | Accuracy | n |
|---|---|---|
| Temporal | **100.0%** | 13 |
| Multi-hop | 94.6% | 37 |
| Open-domain | 94.3% | 70 |
| Single-hop | 90.6% | 32 |

### Efficiency

| Metric | Value |
|---|---|
| Total runtime | 103m 10s |
| Avg reflect time | 38.6s/question |
| Token usage | Not yet instrumented |

### Variance

Not yet measured. A judge-only rerun and a fresh end-to-end rerun are still required to establish variance bounds. Backboard reports roughly 2-3% judge variance with GPT-4.1; Elephant has no measured variance yet.

### Reproduction

```bash
# Legacy reproduction command (kept for traceability)
cargo run --release --bin locomo-bench -- \
  run \
  --profile full \
  --conversation conv-26 \
  --tag temporal-consolidation-legacy-shape

# Required next step: rerun with the current harness to get a clean Cat.1-4 artifact
cargo run --release --bin locomo-bench -- \
  run \
  --profile full \
  --conversation conv-26 \
  --tag temporal-consolidation-rerun
```

Legacy results JSON: `bench/locomo/results/temporal-consolidation.json`
Schema: [results-format.md](results-format.md)

### Caveats

- **Single conversation only** — still a stress test, not a full leaderboard claim
- **Legacy artifact** — includes 2 Category 5 rows and should not be used for public comparison
- **Fresh rerun still needed** — the checked-in JSON is legacy, even though the current harness now records token/cost data
- **No variance data yet** — still single-run only
- **Same model for all stages** — extraction, reflection, consolidation, and judging all use Sonnet 4.6
- **BLIP-2 image captions** — images replaced with captions per LoCoMo evaluation protocol
