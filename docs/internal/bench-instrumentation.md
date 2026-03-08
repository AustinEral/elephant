# Benchmark Instrumentation Plan

Token and cost tracking for credible benchmark publication.

**Priority: REQUIRED before full 10-conversation run.** Mnemis and EverMemOS both publish stage-by-stage token tables. Without cost data, our results are incomplete and we risk having to rerun the full benchmark.

Protocol reference: [benchmark-protocol.md](benchmark-protocol.md)

## What to Track

### Per-stage token counts

| Stage | Prompt Tokens | Completion Tokens | LLM Calls | Notes |
|---|---|---|---|---|
| **Retain** (extraction) | per session | per session | 1 per session | Fact extraction from conversation text |
| **Entity resolution** | per session | per session | 0-N per session | LLM verification calls (may be 0 if embedding-only) |
| **Consolidation** | per batch | per batch | 1 per batch | Topic-scoped observation synthesis |
| **Reflect** (per question) | per iteration | per iteration | 1-8 per question | Agentic loop: search + recall + synthesis |
| **Judge** (per question) | per question | per question | 1 per question | Binary correct/wrong scoring |

### Aggregates to publish

- Total tokens (prompt + completion) per stage
- Total tokens across all stages
- Tokens per question (total / question count)
- Average response time per question (reflect only)
- Total wall-clock runtime
- LLM call count per stage

## Implementation

### 1. LLM client token tracking

The LLM client already returns token usage in `CompletionResponse`. We need to aggregate it.

```rust
struct TokenUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

struct StageMetrics {
    stage: String,          // "retain", "consolidation", "reflect", "judge"
    total_prompt: u64,
    total_completion: u64,
    call_count: u64,
}
```

Options:
- **Thread-local accumulator**: Each pipeline stage writes to a thread-local counter, bench harness reads at stage boundaries
- **Return from pipeline**: Each pipeline operation returns its token usage alongside its result
- **Metrics collector**: Shared `Arc<Metrics>` passed through the pipeline

The cleanest approach is returning token usage from each pipeline call, since the bench harness already controls the call sequence.

### 2. Retain token tracking

`RetainResult` should include token usage from extraction + entity resolution.

### 3. Consolidation token tracking

`ConsolidationReport` should include token usage from all LLM calls during consolidation.

### 4. Reflect token tracking

`ReflectResult` already has `elapsed_s`. Add token usage from all agentic loop iterations.

### 5. Judge token tracking

The bench harness makes judge calls directly. Track tokens there.

### 6. Bench output format

Add to the bench results JSON:

```json
{
  "token_usage": {
    "retain": { "prompt": 123456, "completion": 7890, "calls": 26 },
    "consolidation": { "prompt": 45678, "completion": 2345, "calls": 67 },
    "reflect": { "prompt": 234567, "completion": 12345, "calls": 462 },
    "judge": { "prompt": 56789, "completion": 1234, "calls": 154 },
    "total": { "prompt": 460490, "completion": 23814, "calls": 709 }
  },
  "tokens_per_question": 3145,
  "total_runtime_s": 1234.5
}
```

## Competitor Reference Points

| System | Tokens/Question | Total Tokens | Runtime |
|---|---|---|---|
| EverMemOS (GPT-4.1-mini) | ~14.3k (incl judge) | ~22M | not published |
| Mnemis | ~36.4k | ~56M | 8,622s (~5.6s/q) |
| Elephant | unknown | unknown | ~38s/q reflect only |

Our reflect time (~38s/q) is high but includes full agentic loop. Token count is the more meaningful comparison.

## Priority

**Phase 1 — required before the full 10-conversation run.** If we do the full run without token accounting, we will likely have to rerun it. EverMemOS publishes a stage-by-stage LoCoMo token table for 1,540 questions (9.42M add, 10.27M search+answer, 2.38M evaluate for GPT-4.1-mini). Mnemis publishes 56M total tokens and 8,622s runtime. These disclosures are now part of what makes a result credible.

Results schema: [results-format.md](results-format.md)
