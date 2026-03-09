# Benchmark Results Format

JSON format for benchmark output files in `bench/locomo/results/`.

## Top-level schema

```json
{
  "benchmark": "locomo",
  "timestamp": "2026-03-09T21:15:02Z",
  "commit": "abc123def456",
  "tag": "baseline",

  "judge_model": "anthropic/claude-sonnet-4-6",
  "retain_model": "anthropic/claude-sonnet-4-6",
  "reflect_model": "anthropic/claude-sonnet-4-6",
  "embedding_model": "local/bge-small-en-v1.5",
  "reranker_model": "local/ms-marco-MiniLM-L-6-v2",
  "consolidation_strategy": "end",

  "total_questions": 1540,
  "accuracy": 0.9312,
  "mean_f1": 0.4123,
  "mean_evidence_recall": 0.6641,

  "manifest": {
    "protocol_version": "2026-03-10-config-v1",
    "profile": "full",
    "mode": "qa",
    "dataset_path": "data/locomo10.json",
    "dataset_fingerprint": "9f7f4c0a5fbb2df2",
    "selected_conversations": [],
    "session_limit": null,
    "question_limit": null,
    "ingestion_granularity": "turn"
  },

  "stage_metrics": {
    "retain_extract": {
      "prompt_tokens": 123,
      "completion_tokens": 45,
      "calls": 10,
      "errors": 0,
      "latency_ms": 2110
    }
  },

  "total_stage_usage": {
    "prompt_tokens": 12345,
    "completion_tokens": 678,
    "calls": 300,
    "errors": 1,
    "latency_ms": 543210
  },

  "bank_ids": {
    "conv-26": "01KK623GTJJB2WW3RKHSDSCDT6"
  },

  "turn_refs": {
    "01KKA...": "D1:3"
  },

  "per_category": {},
  "per_conversation": {},
  "results": [ ... ],
  "total_time_s": 1234.5
}
```

## Per-question record

```json
{
  "question_id": "f8cc48",
  "sample_id": "conv-26",
  "question": "When did Melanie paint a sunrise?",
  "ground_truth": "2022",
  "hypothesis": "Melanie painted a lake sunrise in 2022.",
  "category_name": "multi-hop",
  "f1": 1.0,
  "judge_correct": true,
  "judge_reasoning": "The generated answer correctly identifies 2022.",
  "confidence": 0.85,
  "elapsed_s": 25.4,
  "status": "ok",
  "evidence_refs": ["D1:3"],
  "retrieved_turn_refs": ["D1:3"],
  "evidence_hit": true,
  "evidence_recall": 1.0,
  "retrieved_context": [
    {
      "id": "01KK54YT6PV3KG9GRYNV65TV0K",
      "content": "Melanie painted a lake sunrise in 2022...",
      "score": 0.9998,
      "network": "observation",
      "source_turn_id": "01KKA...",
      "source_turn_ref": "D1:3"
    }
  ]
}
```

## What matters most

### Manifest

This is the reproducibility contract. It records:

- profile / config selection
- subcommand mode (`run`, `ingest`, or `qa`)
- dataset fingerprint
- command line
- category filter
- explicit conversation selection
- ingestion granularity
- image handling mode
- concurrency
- consolidation mode
- dirty-worktree status

For `qa`, the manifest still records the original ingest and consolidation settings from the source artifact because those settings define the banks being evaluated.

### Stage metrics

These are summed across the run and keyed by benchmark stage:

- `retain_extract`
- `retain_resolve`
- `retain_graph`
- `retain_opinion`
- `reflect`
- `consolidate`
- `opinion_merge`
- `judge`

### Evidence fields

LoCoMo includes supporting `evidence` refs for nearly every question. Elephant now records:

- `evidence_refs`: dataset gold evidence
- `retrieved_turn_refs`: evidence refs recovered by retrieval provenance
- `evidence_hit`: whether any gold evidence was retrieved
- `evidence_recall`: fraction of gold evidence refs retrieved

## Compatibility notes

- New runs serialize banks as `bank_ids`. Older Elephant results used `banks`; the harness reads both.
- `turn_refs` only exists for runs that preserve turn provenance, which is the new default turn-level ingestion path.
- Older result files may include `category_name: "unanswerable"` from the legacy Cat.5 leakage bug. Those are legacy artifacts.
- The `view` tool now surfaces evidence recall and stage metrics for new-style result files while remaining backward-compatible with older artifacts.
