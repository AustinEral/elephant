# Phase 4: Evaluation Path - Research

**Researched:** 2026-03-15
**Domain:** LLM-as-judge evaluation, LongMemEval scoring protocol, common judge extraction
**Confidence:** HIGH

## Summary

This phase replaces the QA stub in `bench/longmemeval/longmemeval.rs` (lines 1169-1197) with a real evaluation path: call reflect, call an LLM judge with task-specific prompts, compute per-category accuracy. The upstream LongMemEval Python evaluation script (`evaluate_qa.py`) defines exact judge prompts per question type, which we must faithfully replicate. The LoCoMo harness already has a working `llm_judge()` + `build_judge_client()` + JSON parsing pipeline that establishes the exact pattern.

The user has locked a baseline-first philosophy: no reflect agent modifications, no temporal context injection, no special abstention handling. The raw question goes to reflect as-is. Separate judge prompts are used only on the judging side: factual prompt for standard questions, abstention prompt for `_abs`-suffixed questions.

**Primary recommendation:** Extract `build_judge_client()`, `JudgeResponse`, JSON response parsing, and retry logic to `bench/common/judge.rs`. Write 5 LongMemEval judge prompt files (one per question type + abstention) in `bench/longmemeval/`. Replace the QA stub with: reflect call -> select prompt by question type/abstention -> call common judge -> fill QuestionResult/QuestionDebugRecord -> flush JSONL. Accuracy = correct / total_questions.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Baseline-first: no changes to reflect agent behavior. Pass raw question, see how Elephant performs out of the box
- If temporal reasoning or abstention scores are low, that's data for future tuning -- not a harness problem to solve now
- Match LongMemEval paper's evaluation protocol exactly -- don't carry over LoCoMo assumptions
- Write LongMemEval-specific judge prompts, NOT reuse LoCoMo's judge_answer.txt
- Follow whatever criteria the LongMemEval paper defines for correctness
- Two prompts: factual (standard questions) and abstention (false-premise questions identified by _abs suffix)
- Abstention scoring follows LongMemEval's own definition of correct refusal
- Prompts live in bench/longmemeval/ (benchmark-specific, not shared)
- Extract llm_judge(), build_judge_client(), and judge response parsing to bench/common/
- Both LoCoMo and LongMemEval share the machinery (client setup, retry logic, JSON extraction, metering)
- Each benchmark provides its own prompt template -- common infra is prompt-agnostic
- Judge model defaults to GPT-4o, overridable via --judge-model (EVAL-01)
- No modifications to ReflectQuery or the reflect agent for baseline
- Pass the raw question string to reflect as-is
- question_date is available per instance if needed later -- temporal context injection is a tuning knob, not a baseline requirement
- Overall accuracy: correct / 500 (all questions in denominator, no exclusions) (EVAL-05)
- Per-category breakdown across 7 types with question counts (EVAL-04)
- Categories: single-session-user, single-session-assistant, single-session-preference, multi-session, knowledge-update, temporal-reasoning, abstention
- reporting_category() and compute_per_category() already implemented

### Claude's Discretion
- Exact judge prompt wording (guided by LongMemEval paper's evaluation criteria)
- How to structure the extracted common judge module
- Error handling for judge failures (retry count, fallback behavior)
- Debug JSONL content: reflect trace, retrieved context, tool call history

### Deferred Ideas (OUT OF SCOPE)
- Temporal context injection into reflect (prepend question_date to question or extend ReflectQuery) -- tune after baseline
- Reflect agent modifications for abstention handling -- engine change, not harness
- Token F1 secondary metric (EVAL-07, v2)
- Evidence session tracking against answer_session_ids (EVAL-08, v2)
- Stage-level cost metrics (EVAL-09, v2)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | Configurable LLM judge -- GPT-4o default, overridable via `--judge-model` | `build_judge_client()` from LoCoMo (lines 713-744) extracted to `bench/common/judge.rs`. `--judge-model` already parsed into `config.judge_model`. Default "gpt-4o" when JUDGE_MODEL env var absent. |
| EVAL-02 | Two judge prompt variants: factual (standard) and abstention (false-premise) | Upstream Python `get_anscheck_prompt()` defines 5 factual variants + 1 abstention prompt. Factual variants differ by question type. Abstention prompt checks "correctly identifies the question as unanswerable". |
| EVAL-03 | All 500 questions scored including 30 false-premise abstention questions | QA stub loop already iterates all instances. Replace stub body with reflect + judge calls. No filtering or skipping. |
| EVAL-04 | Per-category accuracy breakdown across 7 question types | `compute_per_category()` already implemented (lines 936-956). QuestionResult.category populated by `reporting_category()`. Works once judge_correct is properly set. |
| EVAL-05 | Overall accuracy metric (correct / total across all 500 questions) | Sum judge_correct / total_questions in summary output. No exclusions, no category weighting. |
| EVAL-06 | Abstention questions identified by `_abs` suffix and scored with abstention judge | `is_abstention()` already implemented in dataset.rs (line 48). Route to abstention prompt when true, factual prompt otherwise. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| elephant::llm | internal | LlmClient trait, build_client(), extract_json() | Already used by LoCoMo judge |
| elephant::llm::retry | internal | RetryingLlmClient, RetryPolicy | Already wraps LoCoMo judge client |
| elephant::metrics | internal | MeteredLlmClient, MetricsCollector, LlmStage::Judge | Already meters LoCoMo judge calls |
| elephant::types | internal | ReflectQuery, ReflectResult, CompletionRequest, Message | Already used for reflect calls |
| bench/common/ | internal | New judge.rs module for shared judge infra | Extraction target per user decision |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| serde / serde_json | 1.x | JudgeResponse parsing, JSONL serialization | Already in use everywhere |
| std::time::Instant | stdlib | Per-question elapsed timing | QA timing per instance |
| include_str!() | stdlib | Load judge prompt text files at compile time | LoCoMo pattern for prompts |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| include_str! prompts | Runtime file loading | include_str! matches LoCoMo pattern, simpler deployment |
| 5 separate factual prompt files | Single template with conditionals | Separate files match upstream's per-type prompt design, easier to tune individually |
| Single "factual" prompt file | Per-type prompts | User says "two prompts" but upstream has per-type variants; recommend per-type for fidelity |

## Architecture Patterns

### Recommended Project Structure
```
bench/
  common/
    mod.rs          # existing: pub mod fingerprint, io; ADD: pub mod judge
    fingerprint.rs  # existing
    io.rs           # existing
    judge.rs        # NEW: build_judge_client(), llm_judge(), JudgeResponse
  longmemeval/
    longmemeval.rs  # MODIFY: replace QA stub with real evaluation
    dataset.rs      # existing (unchanged)
    ingest.rs       # existing (unchanged)
    prompts/        # NEW directory
      judge_factual.txt                 # standard questions (ssu, ssa, multi)
      judge_temporal.txt                # temporal-reasoning variant
      judge_knowledge_update.txt        # knowledge-update variant
      judge_preference.txt              # single-session-preference variant
      judge_abstention.txt              # false-premise questions
  locomo/
    locomo.rs       # MODIFY: replace inline judge with delegating wrappers to common
    judge_answer.txt # existing (unchanged, LoCoMo-specific)
```

### Pattern 1: Common Judge Module
**What:** Extract prompt-agnostic judge infrastructure to `bench/common/judge.rs`
**When to use:** Both LoCoMo and LongMemEval judge calls

The common module provides:
- `JudgeResponse` struct (reasoning + label fields)
- `build_judge_client(metrics, override_model)` -> Arc<dyn LlmClient>
- `judge_label(override_model)` -> String (provider/model label for manifest)
- `llm_judge(judge, prompt)` -> Result<(bool, String)> -- takes a fully-rendered prompt string, handles retry + JSON parsing

Key design: the common `llm_judge()` receives a **pre-rendered prompt string** (not question/answer/response separately). Each benchmark is responsible for rendering its own prompt template. This keeps the common module truly prompt-agnostic.

```rust
// bench/common/judge.rs

use std::env;
use std::sync::Arc;
use serde::Deserialize;
use elephant::llm::{self, LlmClient, Provider, ProviderConfig};
use elephant::llm::retry::{RetryPolicy, RetryingLlmClient};
use elephant::metrics::{LlmStage, MeteredLlmClient, MetricsCollector};
use elephant::types::{CompletionRequest, Message};

const JUDGE_TEMPERATURE: f32 = 0.0;
const JUDGE_MAX_TOKENS: usize = 200;
const JUDGE_MAX_ATTEMPTS: usize = 3;

#[derive(Debug, Deserialize)]
pub struct JudgeResponse {
    pub reasoning: String,
    pub label: String,
}

/// Call the LLM judge with a fully-rendered prompt string.
/// Returns (is_correct, reasoning).
pub async fn llm_judge(
    judge: &dyn LlmClient,
    rendered_prompt: &str,
) -> Result<(bool, String), String> {
    // Same retry + JSON extraction logic as LoCoMo
}

/// Build a metered, retrying judge client from env vars.
pub fn build_judge_client(
    metrics: Arc<MetricsCollector>,
    override_model: Option<String>,
) -> Arc<dyn LlmClient> {
    // Same as LoCoMo build_judge_client
}

/// Return "provider/model" label for manifest.
pub fn judge_label(override_model: &Option<String>) -> String {
    // Same as LoCoMo judge_label
}
```

### Pattern 2: Per-Type Prompt Selection
**What:** Select judge prompt based on question type and abstention status
**When to use:** Rendering the prompt before calling common judge

The upstream evaluation script has 5 distinct factual prompts (by question type) plus 1 abstention prompt. We match this exactly:

```rust
fn select_judge_prompt(instance: &LongMemEvalInstance) -> &'static str {
    if instance.is_abstention() {
        return JUDGE_ABSTENTION;
    }
    match instance.question_type {
        QuestionType::SingleSessionUser
        | QuestionType::SingleSessionAssistant
        | QuestionType::MultiSession => JUDGE_FACTUAL,
        QuestionType::TemporalReasoning => JUDGE_TEMPORAL,
        QuestionType::KnowledgeUpdate => JUDGE_KNOWLEDGE_UPDATE,
        QuestionType::SingleSessionPreference => JUDGE_PREFERENCE,
    }
}
```

### Pattern 3: QA Evaluation Loop (replacing stub)
**What:** The core evaluation flow for each instance
**When to use:** Lines 1169-1197 of longmemeval.rs

```rust
// Per-instance QA evaluation (replacing the stub)
let qa_start = Instant::now();

// 1. Call reflect
let reflect_result = with_scoped_collector(
    metrics.clone(),
    runtime.reflect.reflect(&ReflectQuery {
        bank_id: bank_id.parse().unwrap(),
        question: instance.question.clone(),
        budget_tokens: REFLECT_BUDGET_TOKENS,
    }),
).await;

// 2. Process reflect result
let (hypothesis, retrieved_context, reflect_trace, final_done, status, error) =
    match reflect_result { ... };

// 3. Judge
let (judge_correct, judge_reasoning, status, error) = if hypothesis.is_empty() {
    (false, error_msg, status, error)
} else {
    let prompt_template = select_judge_prompt(instance);
    let rendered = render_judge_prompt(prompt_template, &instance.question,
                                        &instance.answer_string(), &hypothesis);
    match llm_judge(judge.as_ref(), &rendered).await {
        Ok((correct, reasoning)) => (correct, reasoning, status, error),
        Err(e) => (false, e.clone(), "judge_error".into(), Some(e)),
    }
};

// 4. Build QuestionResult + QuestionDebugRecord, flush JSONL
```

### Pattern 4: LoCoMo Delegating Wrappers
**What:** After extraction, LoCoMo calls common functions through thin wrappers
**When to use:** Minimize diff in locomo.rs per Phase 1 convention

```rust
// bench/locomo/locomo.rs -- delegating wrappers
fn build_judge_client(
    metrics: Arc<MetricsCollector>,
    override_model: Option<String>,
) -> Arc<dyn LlmClient> {
    common::judge::build_judge_client(metrics, override_model)
}

async fn llm_judge(
    judge: &dyn LlmClient,
    question: &str,
    gold_answer: &str,
    generated_answer: &str,
) -> Result<(bool, String), String> {
    let rendered = JUDGE_PROMPT
        .replace("{question}", question)
        .replace("{gold_answer}", gold_answer)
        .replace("{generated_answer}", generated_answer);
    common::judge::llm_judge(judge, &rendered).await
}
```

### Anti-Patterns to Avoid
- **Modifying ReflectQuery for temporal context:** Locked decision -- baseline first, no changes to reflect agent
- **Reusing LoCoMo's judge_answer.txt:** Locked decision -- LongMemEval has its own evaluation criteria
- **Single "factual" prompt for all types:** Upstream uses per-type prompts with meaningful differences (temporal off-by-one tolerance, preference rubric evaluation, knowledge-update prior-info tolerance)
- **Excluding abstention from denominator:** EVAL-05 says correct / 500, all questions in denominator
- **JSON mode for judge calls:** Upstream uses `max_tokens: 10` and checks for "yes" in response. Our pattern uses structured JSON response with reasoning field for richer debugging data. This is a deliberate divergence from upstream that LoCoMo already established.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LLM client construction | Manual HTTP/auth/retry | `build_judge_client()` from common | Env var fallback chain, metering, retry all handled |
| JSON extraction from LLM | Custom regex parsing | `elephant::llm::extract_json()` | Handles markdown fences, surrounding prose, nested objects |
| JSONL flushing | Manual file I/O | `common::io::append_jsonl()` | Already handles create+append+serialize |
| Judge response parsing | Custom string matching | `JudgeResponse` serde + extract_json fallback | LoCoMo pattern, handles both clean JSON and wrapped JSON |

## Common Pitfalls

### Pitfall 1: Prompt Placeholder Mismatch
**What goes wrong:** Judge prompt uses `{question}` placeholder but code passes `.replace("{q}", ...)`
**Why it happens:** Copy-paste from LoCoMo prompt which uses different placeholders
**How to avoid:** Each prompt file should use consistent placeholders: `{question}`, `{answer}`, `{response}`. Verify with a test that all placeholders get replaced.
**Warning signs:** Judge always returns incorrect (sees literal `{question}` in prompt)

### Pitfall 2: Abstention Prompt Has Different Semantics
**What goes wrong:** Using factual prompt for abstention questions. Factual prompt checks "does response contain correct answer" but abstention gold answer is an explanation of WHY it's unanswerable, not a reference answer.
**Why it happens:** Treating all questions identically
**How to avoid:** `is_abstention()` gates prompt selection. Abstention prompt asks "does model correctly identify question as unanswerable" not "does response match gold answer."
**Warning signs:** Abstention accuracy is 0% or near 0%

### Pitfall 3: Judge Model Default
**What goes wrong:** Judge client construction fails because no JUDGE_MODEL env var is set and default isn't applied
**Why it happens:** LoCoMo requires JUDGE_MODEL env var. LongMemEval needs GPT-4o default per EVAL-01.
**How to avoid:** In `build_judge_client`, when `override_model` is None and JUDGE_MODEL env var is unset, default to "gpt-4o". This is LongMemEval-specific behavior, not common module behavior. Apply the default in the LongMemEval harness before passing to common.
**Warning signs:** Panic on startup with "JUDGE_MODEL must be set"

### Pitfall 4: Accuracy Computation Denominator
**What goes wrong:** Accuracy excludes errored questions or skipped instances
**Why it happens:** Natural inclination to only count successfully-judged questions
**How to avoid:** `correct / total_questions` where total_questions = all_question_results.len(). Errored questions count as incorrect (judge_correct = false).
**Warning signs:** Accuracy denominator doesn't match 500 for full runs

### Pitfall 5: Missing Imports After Extraction
**What goes wrong:** After extracting judge functions to common, LoCoMo or LongMemEval fail to compile
**Why it happens:** Imports not updated, or `use` paths break
**How to avoid:** Run `cargo build --bin longmemeval-bench --bin locomo-bench` after extraction to verify both binaries compile.
**Warning signs:** Compilation errors in CI

## Code Examples

### Upstream Judge Prompts (from LongMemEval evaluate_qa.py)

**Factual (single-session-user, single-session-assistant, multi-session):**
```
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
```

**Temporal-reasoning (adds off-by-one tolerance):**
```
[Same as factual] In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}
[...]
```

**Knowledge-update (tolerates prior + updated answer):**
```
[Same as factual] If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}
[...]
```

**Single-session-preference (rubric-based):**
```
I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {question}

Rubric: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
```

**Abstention (false-premise, `_abs` suffix):**
```
I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {answer}

Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only.
```

### Adaptation Decision: JSON vs Yes/No Response Format

The upstream evaluation uses `max_tokens: 10` and checks `'yes' in response.lower()` -- a minimal binary signal with no reasoning captured.

Our LoCoMo judge pattern uses structured JSON response `{"reasoning": "...", "label": "CORRECT"}` with `max_tokens: 200`. This provides richer debugging data at the cost of slightly more tokens.

**Recommendation:** Keep the LoCoMo JSON pattern for our judge prompts. Append the JSON instruction to each upstream prompt template:

```
[upstream prompt body]

First, provide a short (one sentence) explanation of your reasoning, then finish with your verdict.

Return response in JSON format with keys: "reasoning" and "label" (CORRECT or WRONG).
```

This replaces the upstream's "Answer yes or no only" suffix. The `JudgeResponse.label` is then checked via `eq_ignore_ascii_case("CORRECT")` (existing LoCoMo pattern).

### Reflect Call Pattern (from LoCoMo)

```rust
const REFLECT_BUDGET_TOKENS: usize = 4096;

let reflect_result = runtime
    .reflect
    .reflect(&ReflectQuery {
        bank_id,
        question: instance.question.clone(),
        budget_tokens: REFLECT_BUDGET_TOKENS,
    })
    .await;
```

ReflectResult fields needed for debug record:
- `response` -> hypothesis
- `retrieved_context: Vec<RetrievedFact>` -> mapped to RetrievedFactEntry
- `trace: Vec<ReflectTraceStep>` -> mapped to ReflectTraceEntry
- `final_done: Option<ReflectDoneTrace>` -> stored directly

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Binary yes/no judge | JSON {reasoning, label} judge | LoCoMo implementation | Richer failure analysis data |
| Single factual prompt | Per-type prompts (5 variants) | LongMemEval paper (Oct 2024) | Temporal off-by-one tolerance, preference rubric, knowledge-update tolerance |
| Inline judge in each benchmark | Shared common module | This phase | Consistent judge behavior, DRY |

## Open Questions

1. **Judge model default application point**
   - What we know: EVAL-01 says "GPT-4o default." LoCoMo requires env var. Common module should stay generic.
   - What's unclear: Should the default be in common or in LongMemEval harness?
   - Recommendation: Apply default in LongMemEval harness before calling common. Pass `Some("gpt-4o".into())` when both `config.judge_model` and JUDGE_MODEL env var are absent. Common module stays strict (requires model from somewhere).

2. **Prompt hash for manifest**
   - What we know: `BenchmarkPromptHashes.judge` is currently empty string. Need to hash the selected judge prompt.
   - What's unclear: Hash all 5+1 prompt files, or just the one used? Multiple prompts are used per run.
   - Recommendation: Hash all 6 prompt files concatenated (sorted by filename) into a single judge prompt hash. This uniquely identifies the judge prompt set.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in #[test] + cargo test |
| Config file | None -- Cargo auto-discovers |
| Quick run command | `cargo test --bin longmemeval-bench` |
| Full suite command | `cargo test --bin longmemeval-bench --bin locomo-bench` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | Judge model defaults to GPT-4o, overridable | unit | `cargo test --bin longmemeval-bench judge_model_default` | Wave 0 |
| EVAL-02 | Correct prompt selected by type/abstention | unit | `cargo test --bin longmemeval-bench select_judge_prompt` | Wave 0 |
| EVAL-03 | All questions scored (no skipping) | integration | Manual: run smoke profile, verify 0 questions with status "qa-not-implemented" | manual-only -- requires live LLM |
| EVAL-04 | Per-category accuracy breakdown | unit | `cargo test --bin longmemeval-bench compute_per_category` | Exists (from Phase 3) |
| EVAL-05 | Overall accuracy = correct/total | unit | `cargo test --bin longmemeval-bench accuracy_computation` | Wave 0 |
| EVAL-06 | Abstention prompt for _abs questions | unit | `cargo test --bin longmemeval-bench abstention_prompt_selection` | Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --bin longmemeval-bench --bin locomo-bench`
- **Per wave merge:** `cargo test` (full workspace)
- **Phase gate:** Full suite green before verify

### Wave 0 Gaps
- [ ] `bench/common/judge.rs` -- needs #[cfg(test)] module with: JudgeResponse parsing, extract_json fallback, build_judge_client label generation
- [ ] `bench/longmemeval/longmemeval.rs` -- needs tests for: select_judge_prompt routing, prompt placeholder rendering, accuracy computation with mixed results
- [ ] LoCoMo compilation check -- `cargo test --bin locomo-bench` passes after extraction

## Sources

### Primary (HIGH confidence)
- LongMemEval upstream `evaluate_qa.py` -- exact judge prompts per question type, binary scoring logic, GPT-4o as default judge model. Retrieved from GitHub repo `xiaowu0162/LongMemEval`
- LoCoMo `llm_judge()` implementation -- `bench/locomo/locomo.rs` lines 664-711. JSON response format, retry logic, extract_json fallback
- LoCoMo `build_judge_client()` -- `bench/locomo/locomo.rs` lines 713-744. Env var fallback chain, metering, retry wrapping
- LongMemEval harness types -- `bench/longmemeval/longmemeval.rs` lines 735-781. QuestionResult, QuestionDebugRecord, ReflectTraceEntry, RetrievedFactEntry already defined
- LongMemEval `compute_per_category()` -- `bench/longmemeval/longmemeval.rs` lines 936-956. Already implemented and correct
- LongMemEval `is_abstention()` + `reporting_category()` -- `bench/longmemeval/dataset.rs` lines 48-67. Already implemented

### Secondary (MEDIUM confidence)
- [LongMemEval paper](https://arxiv.org/abs/2410.10813) -- evaluation protocol, GPT-4o judge with 97% human agreement
- LongMemEval `print_qa_metrics.py` -- confirms per-type accuracy aggregation pattern, task-averaged vs overall accuracy distinction

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, just reorganizing
- Architecture: HIGH -- direct adaptation of LoCoMo pattern + upstream Python prompts
- Pitfalls: HIGH -- identified from actual LoCoMo code and upstream evaluation script differences

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (stable domain, evaluation protocol unlikely to change)
