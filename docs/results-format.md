# Benchmark Results Schema

JSON schema for benchmark output files in `bench/locomo/results/`.

## Top-level structure

```json
{
  "benchmark": "locomo",
  "timestamp": "2026-03-08T02:37:45Z",
  "tag": "temporal-consolidation",
  "commit": "f58610f",

  "judge_model": "anthropic/claude-sonnet-4-6",
  "server_info": {
    "llm_model": "anthropic/claude-sonnet-4-6",
    "embedding_model": "local/bge-small-en-v1.5",
    "reranker_model": "local/ms-marco-MiniLM-L-6-v2"
  },

  "bank_ids": {
    "conv-26": "01KK623GTJJB2WW3RKHSDSCDT6"
  },

  "token_usage": {
    "retain": { "prompt_tokens": 0, "completion_tokens": 0, "calls": 0 },
    "consolidation": { "prompt_tokens": 0, "completion_tokens": 0, "calls": 0 },
    "reflect": { "prompt_tokens": 0, "completion_tokens": 0, "calls": 0 },
    "judge": { "prompt_tokens": 0, "completion_tokens": 0, "calls": 0 },
    "total": { "prompt_tokens": 0, "completion_tokens": 0, "calls": 0 }
  },

  "results": [ ... ]
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
  "retrieved_context": [
    {
      "id": "01KK54YT6PV3KG9GRYNV65TV0K",
      "content": "Melanie painted a lake sunrise in 2022...",
      "score": 0.9998,
      "network": "observation"
    }
  ]
}
```

## Field definitions

### Top-level

| Field | Type | Description |
|---|---|---|
| `benchmark` | string | Always `"locomo"` |
| `timestamp` | ISO 8601 | When the run started |
| `tag` | string | Human-readable run identifier |
| `commit` | string | Git commit SHA (planned, not yet implemented) |
| `judge_model` | string | Model used for binary correctness judging |
| `server_info` | object | Models used by the Elephant server |
| `bank_ids` | object | Map of conversation ID → bank ULID |
| `token_usage` | object | Per-stage token counts (planned, not yet implemented) |
| `results` | array | Per-question results |

### Per-question

| Field | Type | Description |
|---|---|---|
| `question_id` | string | Truncated hash of the question |
| `sample_id` | string | Conversation identifier (e.g., `conv-26`) |
| `question` | string | The question text |
| `ground_truth` | string | Gold answer from LoCoMo dataset |
| `hypothesis` | string | Elephant's generated answer |
| `category_name` | string | LoCoMo category: `single-hop`, `multi-hop`, `temporal`, `open-domain` |
| `f1` | float | Token-level F1 score (0.0 - 1.0) |
| `judge_correct` | bool | Whether the LLM judge scored this correct |
| `judge_reasoning` | string | Judge's explanation |
| `confidence` | float | Judge confidence score |
| `elapsed_s` | float | Reflect wall-clock time in seconds |
| `retrieved_context` | array | Facts retrieved during reflect, ordered by score |

### Retrieved context entry

| Field | Type | Description |
|---|---|---|
| `id` | string | Fact ULID |
| `content` | string | Fact text |
| `score` | float | Reranker score |
| `network` | string | Memory network: `observation`, `world`, `experience`, `opinion` |

### Token usage (per stage)

| Field | Type | Description |
|---|---|---|
| `prompt_tokens` | int | Total prompt/input tokens |
| `completion_tokens` | int | Total completion/output tokens |
| `calls` | int | Number of LLM API calls |

## Notes

- `token_usage` and `commit` fields are planned but not yet implemented in the bench harness
- `retrieved_context` was added in the `reflect-synthesis` run; older result files may not have it
- The view tool uses `#[serde(default)]` to gracefully read results with missing fields
- `category_name` values map to LoCoMo Categories 1-4; Category 5 (adversarial) is not used
- Local answer subtypes (e.g., `unanswerable`) are tracked in `category_name` but are not LoCoMo categories
