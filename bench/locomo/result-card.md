# Elephant Benchmark Result Card

## Run: series1 (2026-03-10)

### Status

| Field | Value |
|---|---|
| Publication status | **Full 10-conversation benchmark** |
| Series | series1 |
| Artifacts | 10 per-conversation artifacts in `results/canonical/series1/` |

### Scope

| Field | Value |
|---|---|
| Dataset | LoCoMo (ACL 2024) |
| Conversations | All 10 |
| Sessions ingested | 272 |
| Turns ingested | 5,882 |
| Questions scored | 1,540 |
| Categories | Cat.1-4 only |
| Ingest mode | Session-level |
| Consolidation | End-of-run |
| Question concurrency | 5 |

Full protocol: [protocol.md](protocol.md)

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
| Full Cat.1-4 | **91.2%** | **1,540** |

Category breakdown:

| Category | Accuracy | n |
|---|---|---|
| Open-domain | 93.8% | 841 |
| Multi-hop | 92.5% | 321 |
| Single-hop | 90.4% | 282 |
| Temporal | 66.7% | 96 |

Per conversation:

| Conversation | Accuracy | n |
|---|---|---|
| conv-41 | 96.1% | 152 |
| conv-26 | 94.7% | 152 |
| conv-47 | 92.7% | 150 |
| conv-44 | 91.9% | 123 |
| conv-50 | 91.8% | 158 |
| conv-49 | 91.0% | 156 |
| conv-48 | 90.6% | 191 |
| conv-43 | 89.3% | 178 |
| conv-30 | 88.9% | 81 |
| conv-42 | 86.4% | 199 |

### Efficiency

| Metric | Value |
|---|---|
| Total runtime | 1,590m |
| Ingest | 973m |
| Consolidation | 387m |
| QA | 230m |
| Avg QA time | 9.0s/question |
| Total tokens | 52,531,350 |

### Variance

Not yet measured. A judge-only variance pass or clean rerun would strengthen the result.

### Reproduction

```bash
# Run individual conversations
cargo run --release --bin locomo-bench -- \
  run \
  --profile full \
  --conversation conv-26 \
  --question-jobs 5 \
  --tag series1-conv-26

# Repeat for each conversation, then merge
cargo run --release --bin locomo-bench -- \
  merge \
  bench/locomo/results/canonical/series1/series1-conv-*.json \
  --out bench/locomo/results/canonical/series1.json
```

Artifacts: `bench/locomo/results/canonical/series1/`
Schema: [results-format.md](results-format.md)

### Caveats

- **No variance data yet** — single-run only
- **Session ingest** — turn-level evidence-trace metrics are weaker in this mode
- **Same model for all stages** — extraction, reflection, consolidation, and judging all use Sonnet 4.6
- **BLIP-2 image captions** — images replaced with captions per LoCoMo evaluation protocol
- **Temporal category** — 66.7% is the weakest area, driven by date resolution gaps
