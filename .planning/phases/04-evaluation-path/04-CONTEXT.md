# Phase 4: Evaluation Path - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Score all 500 LongMemEval questions including 30 false-premise abstention questions. Reflect agent answers each question, LLM judge scores correctness, results aggregated into per-category accuracy across 7 question types. Configurable judge model (GPT-4o default). No reflect agent modifications, no concurrency, no resume — just the QA evaluation path wired into the existing pipeline stub.

</domain>

<decisions>
## Implementation Decisions

### Evaluation philosophy
- Baseline-first: no changes to reflect agent behavior. Pass raw question, see how Elephant performs out of the box
- If temporal reasoning or abstention scores are low, that's data for future tuning — not a harness problem to solve now
- Match LongMemEval paper's evaluation protocol exactly — don't carry over LoCoMo assumptions

### Judge prompts
- Write LongMemEval-specific judge prompts, NOT reuse LoCoMo's judge_answer.txt
- Follow whatever criteria the LongMemEval paper defines for correctness
- Two prompts: factual (standard questions) and abstention (false-premise questions identified by `_abs` suffix)
- Abstention scoring follows LongMemEval's own definition of correct refusal
- Prompts live in `bench/longmemeval/` (benchmark-specific, not shared)

### Judge infrastructure
- Extract `llm_judge()`, `build_judge_client()`, and judge response parsing to `bench/common/`
- Both LoCoMo and LongMemEval share the machinery (client setup, retry logic, JSON extraction, metering)
- Each benchmark provides its own prompt template — common infra is prompt-agnostic
- Judge model defaults to GPT-4o, overridable via `--judge-model` (EVAL-01)

### Temporal context
- No modifications to ReflectQuery or the reflect agent for baseline
- Pass the raw question string to reflect as-is
- question_date is available per instance if needed later — temporal context injection is a tuning knob, not a baseline requirement

### Scoring
- Overall accuracy: correct / 500 (all questions in denominator, no exclusions) (EVAL-05)
- Per-category breakdown across 7 types with question counts (EVAL-04)
- Categories: single-session-user, single-session-assistant, single-session-preference, multi-session, knowledge-update, temporal-reasoning, abstention
- `reporting_category()` and `compute_per_category()` already implemented

### Claude's Discretion
- Exact judge prompt wording (guided by LongMemEval paper's evaluation criteria)
- How to structure the extracted common judge module
- Error handling for judge failures (retry count, fallback behavior)
- Debug JSONL content: reflect trace, retrieved context, tool call history

</decisions>

<specifics>
## Specific Ideas

- "Stay cautious. Don't change behavior until we see regression. We need baseline before we can move."
- "Target what LongMemEval wants. Don't get stuck in our ways or preferred evaluations. Do what the bench asks."
- LoCoMo iteration history proves the pattern: baseline first (81.2%), then targeted prompt tuning drove accuracy from 81% to 94%

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bench/locomo/locomo.rs`: `llm_judge()` (lines 664-711) — extract to common, make prompt-agnostic
- `bench/locomo/locomo.rs`: `build_judge_client()` (lines 713-744) — extract to common
- `bench/locomo/locomo.rs`: Judge retry logic, JSON extraction, metering — extract to common
- `bench/longmemeval/longmemeval.rs`: `QuestionResult`, `QuestionDebugRecord`, `ReflectTraceEntry`, `RetrievedFactEntry` — already defined
- `bench/longmemeval/longmemeval.rs`: `compute_per_category()` (lines 936-960) — already implemented
- `bench/longmemeval/dataset.rs`: `is_abstention()`, `reporting_category()` — already implemented

### Established Patterns
- LoCoMo QA flow: reflect call -> hypothesis -> llm_judge -> QuestionResult -> append JSONL
- Judge env vars: JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_API_KEY (falls back to LLM_* vars)
- JSONL incremental flush: write question/debug records as each completes
- `with_scoped_collector()` for per-instance stage metrics

### Integration Points
- QA stub at `bench/longmemeval/longmemeval.rs` lines 1169-1197 — replace with real evaluation
- `runtime.reflect` — existing reflect agent, called with `ReflectQuery { bank_id, question, budget_tokens }`
- `bench/common/` — target for extracted judge infrastructure
- `bench/locomo/locomo.rs` — source for judge extraction, leave delegating wrappers per Phase 1 convention

</code_context>

<deferred>
## Deferred Ideas

- Temporal context injection into reflect (prepend question_date to question or extend ReflectQuery) — tune after baseline
- Reflect agent modifications for abstention handling — engine change, not harness
- Token F1 secondary metric (EVAL-07, v2)
- Evidence session tracking against answer_session_ids (EVAL-08, v2)
- Stage-level cost metrics (EVAL-09, v2)

</deferred>

---

*Phase: 04-evaluation-path*
*Context gathered: 2026-03-15*
