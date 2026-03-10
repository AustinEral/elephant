# Benchmark Results Format

Benchmark runs now write three artifacts under `bench/locomo/results/`:

- summary JSON: `<tag>.json`
- question records: `<tag>.questions.jsonl`
- debug records: `<tag>.debug.jsonl`

Recommended layout:

- `bench/locomo/results/local/` for local `run` and `ingest` outputs
- `bench/locomo/results/local/` for default `merge` outputs too
- `bench/locomo/results/canonical/` for promoted merged artifacts
- `bench/locomo/results/archive/legacy-v0/` for pre-refactor historical JSONs

Historical pre-refactor JSONs are archived under `bench/locomo/results/archive/legacy-v0/`. They are not part of the current artifact contract.

The summary file is the canonical run manifest and aggregate metric record. Per-question payloads live in sidecars so publication-grade runs do not balloon into one giant nested JSON document.

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
    "ingestion_granularity": "session",
    "prompt_hashes": {
      "judge": "60c6fd2d7e6f8f4b",
      "retain_extract": "d3f7f98d1dfd1dc4",
      "reflect_agent": "6f7e9859dfde0c5a"
    },
    "runtime_config": {
      "chunk_max_tokens": 512,
      "chunk_overlap_tokens": 64,
      "retriever_limit": 40,
      "recall_rrf_k": 60.0,
      "rerank_top_n": 50,
      "reflect_max_iterations": 8,
      "reflect_max_tokens": null,
      "reflect_budget_tokens": 4096,
      "judge_temperature": 0.0,
      "judge_max_tokens": 200,
      "judge_max_attempts": 3,
      "qa_updates_memory": false
    },
    "source_artifact": {
      "path": "bench/locomo/results/local/ingest.json",
      "fingerprint": "c31f96709a4cf9de",
      "mode": "ingest",
      "tag": "ingest",
      "commit": "abc123def456"
    },
    "source_artifacts": []
  },

  "artifacts": {
    "questions_path": "baseline.questions.jsonl",
    "debug_path": "baseline.debug.jsonl"
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
  "per_conversation": {
    "conv-26": {
      "bank_id": "01KK623GTJJB2WW3RKHSDSCDT6",
      "accuracy": 0.92,
      "mean_f1": 0.41,
      "mean_evidence_recall": 0.68,
      "count": 154,
      "ingest_time_s": 188.2,
      "consolidation_time_s": 7.3,
      "qa_time_s": 612.0,
      "total_time_s": 807.5,
      "bank_stats": {
        "sessions_ingested": 35,
        "turns_ingested": 620,
        "facts_stored": 910,
        "entities_resolved": 460,
        "links_created": 2300,
        "opinions_reinforced": 12,
        "opinions_weakened": 3,
        "observations_created": 201,
        "observations_updated": 44,
        "final_fact_count": 1111,
        "final_observation_count": 201,
        "final_opinion_count": 78,
        "final_entity_count": 154
      },
      "stage_metrics": {
        "retain_extract": {
          "prompt_tokens": 123,
          "completion_tokens": 45,
          "calls": 10,
          "errors": 0,
          "latency_ms": 2110
        }
      }
    }
  },
  "total_time_s": 1234.5
}
```

The summary file may omit `results` entirely or serialize it as an empty array. The `view` tool loads question sidecars automatically when `artifacts.questions_path` is present.

`manifest.source_artifact` is used for `qa` runs that point at one existing artifact. `manifest.source_artifacts` is used for `merge` runs that combine multiple compatible subset artifacts into one canonical result.

Compatibility here means the merged artifacts shared the same benchmark contract: dataset fingerprint, protocol/config knobs, prompt hashes, runtime config, and disjoint conversation/question scope.

If merged source runs differed only in provenance-style fields, the merged manifest may normalize those to mixed values instead of pretending there was one canonical source value. Today that means:

- `profile: "mixed"` when source profile labels differed
- `question_concurrency: 0` or `conversation_concurrency: 0` when source runs used different concurrency

## Question sidecar record

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
  "elapsed_s": 25.4,
  "status": "ok",
  "final_source_ids": ["01KK54YT6PV3KG9GRYNV65TV0K"],
  "evidence_refs": ["D1:3"],
  "retrieved_turn_refs": ["D1:3"],
  "evidence_hit": true,
  "evidence_recall": 1.0,
  "qa_stage_metrics": {
    "reflect": {
      "prompt_tokens": 321,
      "completion_tokens": 44,
      "calls": 2,
      "errors": 0,
      "latency_ms": 1840
    },
    "judge": {
      "prompt_tokens": 88,
      "completion_tokens": 12,
      "calls": 1,
      "errors": 0,
      "latency_ms": 420
    }
  }
}
```

## Debug sidecar record

```json
{
  "question_id": "f8cc48",
  "sample_id": "conv-26",
  "question": "When did Melanie paint a sunrise?",
  "final_done": {
    "iteration": 2,
    "assistant_content": "",
    "raw_arguments": {
      "response": "Melanie painted a lake sunrise in 2022 [01KK54YT6PV3KG9GRYNV65TV0K].",
      "source_ids": ["01KK54YT6PV3KG9GRYNV65TV0K"]
    },
    "used_fallback": false,
    "stop_reason": "tool_use",
    "response": "Melanie painted a lake sunrise in 2022 [01KK54YT6PV3KG9GRYNV65TV0K].",
    "source_ids": ["01KK54YT6PV3KG9GRYNV65TV0K"]
  },
  "reflect_trace": [
    {
      "iteration": 1,
      "tool_name": "recall",
      "query": "sunrise",
      "returned_fact_ids": ["01KK54YT6PV3KG9GRYNV65TV0K"],
      "new_fact_ids": ["01KK54YT6PV3KG9GRYNV65TV0K"],
      "facts_returned": 1,
      "total_tokens": 34,
      "latency_ms": 55
    }
  ],
  "retrieved_context": [
    {
      "id": "01KK54YT6PV3KG9GRYNV65TV0K",
      "content": "Melanie painted a lake sunrise in 2022...",
      "score": 0.9998,
      "network": "observation",
      "source_turn_id": "01KKA...",
      "source_turn_ref": "D1:3",
      "evidence_ids": ["01KK4..."],
      "retrieval_sources": ["semantic", "keyword"],
      "support_turn_ids": ["01KKA..."],
      "support_turn_refs": ["D1:3"]
    }
  ]
}
```

`final_done` is the raw final `done()` tool payload captured from the reflect loop. It exists to explain blank or malformed answers: you can see the original tool arguments, whether fallback parsing was used, the provider `stop_reason`, and the normalized response/source ids that actually fed the benchmark result.

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
- prompt hashes for every LLM prompt template that can affect benchmark behavior
- runtime tuning knobs that affect retrieval, reflection, and consolidation
- source-artifact provenance for `qa`

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

Question records also carry `qa_stage_metrics`, which are the question-scoped `reflect` and `judge` costs only.

Per-conversation summaries now carry their own `stage_metrics`, so QA-only runs can be merged on top of an earlier ingest artifact without losing the ingest/consolidation cost picture.

Per-conversation summaries also carry:

- `ingest_time_s`
- `consolidation_time_s`
- `qa_time_s`
- `total_time_s`

These are the right default timing granularity. We do not persist per-turn or per-session timing by default in the publication artifact. `total_time_s` at the top level is wall-clock run time; summed per-conversation times can be larger when conversations run concurrently.

### Bank stats

Per-conversation summaries also carry `bank_stats`, which are the publication-facing bank construction counters:

- `sessions_ingested`
- `turns_ingested`
- `facts_stored`
- `entities_resolved`
- `links_created`
- `opinions_reinforced`
- `opinions_weakened`
- `observations_created`
- `observations_updated`
- `final_fact_count`
- `final_observation_count`
- `final_opinion_count`
- `final_entity_count`

These let a benchmark reader distinguish "answer quality changed" from "the bank was materially different."

### Evidence fields

LoCoMo includes supporting `evidence` refs for nearly every question. Elephant now records:

- `evidence_refs`: dataset gold evidence
- `retrieved_turn_refs`: evidence refs recovered by retrieval provenance
- `evidence_hit`: whether any gold evidence was retrieved
- `evidence_recall`: fraction of gold evidence refs retrieved
- `final_source_ids`: the fact ids cited by the final answer
- `support_turn_refs`: transitive support refs recovered through fact `evidence_ids`, not just direct `source_turn_id`

The debug sidecar carries the heavy provenance payload:

- `reflect_trace`: tool/query history for the reflect loop
- `retrieved_context`: retrieved facts with direct and transitive turn provenance

## Compatibility notes

- New runs serialize banks as `bank_ids`. Older Elephant results used `banks`; the harness reads both.
- `turn_refs` only exists for runs that preserve turn provenance, typically `--ingest turn`.
- Older result files may include `category_name: "unanswerable"` from the legacy Cat.5 leakage bug. Those are legacy artifacts.
- The `view` tool now reads question sidecars automatically for new-style result files while remaining backward-compatible with older artifacts.
