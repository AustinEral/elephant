# Technology Stack

**Project:** LongMemEval Benchmark Integration
**Researched:** 2026-03-15

## Executive Summary

The LongMemEval harness needs **zero new crate dependencies**. The existing Elephant dependency set (`serde`, `serde_json`, `chrono`, `tokio`, `tabled`) already covers everything required for dataset parsing, async execution, judge evaluation, and artifact output. The main "stack" work is designing the right serde types for LongMemEval's JSON schema and adapting the judge prompt for LongMemEval's task-specific evaluation protocol.

## Recommended Stack

### Core Framework (already in Cargo.toml)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `serde` + `serde_json` | 1.0.228 / 1.0.149 | Dataset deserialization, artifact serialization | Already used by LoCoMo harness; LongMemEval's JSON format is straightforward |
| `chrono` | 0.4.44 | Timestamp parsing for `haystack_dates` and `question_date` | Already used for LoCoMo session dates; same patterns apply |
| `tokio` | 1.49.0 | Async runtime, concurrency (Semaphore, mpsc) | Same concurrency patterns as LoCoMo (question-jobs, instance-jobs) |
| `tabled` | 0.20.0 | CLI output formatting for view tool | Already used by LoCoMo view tool |
| `ulid` | 1.2.1 | Bank ID generation (one per question instance) | Already used for LoCoMo bank IDs |

**Confidence: HIGH** -- these are already compiled and proven in the LoCoMo harness.

### Dataset Parsing (no new deps)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `serde_json::Value` | (bundled) | Handle mixed-type `answer` field | LongMemEval's `answer` field is sometimes a string, sometimes an integer. Use `serde_json::Value` and coerce to string at parse time, same pattern as LoCoMo's `answer: Option<serde_json::Value>` |

**Confidence: HIGH** -- LoCoMo already uses `serde_json::Value` for the answer field for exactly this reason.

### Infrastructure (already in Cargo.toml)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `elephant` (crate) | 0.1.0 | In-process runtime (retain, consolidate, reflect) | Same integration pattern as LoCoMo: `build_runtime_from_env`, direct API calls |
| `sqlx` + `pgvector` | 0.8.6 / 0.4.1 | Per-question bank storage | Each of 500 questions gets its own bank; same DB infra as LoCoMo |
| `reqwest` | 0.13.2 | LLM API calls (judge, retain, reflect) | Already used by LLM providers |

**Confidence: HIGH** -- identical infrastructure to LoCoMo.

### Supporting Libraries (already in Cargo.toml)

| Library | Version | Purpose | When Used |
|---------|---------|---------|-----------|
| `tracing` | 0.1 | Structured logging during ingestion/qa | All stages |
| `futures` | 0.3 | `futures::stream::buffer_unordered` for parallel question processing | QA phase with concurrency control |
| `regex` | 1.12.3 | Potential date normalization, answer extraction | Judge response parsing fallback |
| `uuid` | 1.21.0 | Bank ID conversion (ULID to UUID) | Bank creation |

**Confidence: HIGH** -- all already in the dependency tree.

## What NOT to Add

| Crate | Why Not |
|-------|---------|
| `hf-hub` / HuggingFace SDK | PROJECT.md explicitly scopes this out: "manual download keeps things simple and avoids HF SDK dependency." Use `wget`/`curl` to download `longmemeval_s_cleaned.json` and `longmemeval_m_cleaned.json` to `data/` |
| `csv` | LongMemEval is pure JSON. No CSV anywhere in the pipeline |
| `polars` / `arrow` | Overkill for 500-item JSON arrays. `serde_json` handles it trivially |
| `indicatif` (progress bars) | LoCoMo doesn't use progress bars; maintain consistency. Use `tracing` for progress |
| `clap` | LoCoMo parses CLI args manually with `std::env::args()`. Adding clap for one harness creates inconsistency. Follow the same manual parsing pattern |
| Any Python interop (`pyo3`) | Don't call LongMemEval's Python evaluation scripts. Reimplement the judge logic in Rust to match the LoCoMo pattern |

**Confidence: HIGH** -- these are clear anti-patterns given the existing codebase.

## LongMemEval Dataset Schema (Serde Types)

Based on the official repository (`xiaowu0162/LongMemEval`) and the cleaned dataset (`xiaowu0162/longmemeval-cleaned`), the dataset format is:

```rust
/// Top-level: the file is a JSON array of LongMemEvalInstance.
/// File: longmemeval_s_cleaned.json or longmemeval_m_cleaned.json
type Dataset = Vec<LongMemEvalInstance>;

#[derive(Debug, Deserialize)]
struct LongMemEvalInstance {
    question_id: String,
    /// One of: "single-session-user", "single-session-assistant",
    /// "single-session-preference", "multi-session",
    /// "knowledge-update", "temporal-reasoning"
    /// Abstention variants have "_abs" suffix on question_id, not a separate type.
    question_type: String,
    question: String,
    /// The date/time of the question (after all sessions).
    question_date: String,
    /// Ground truth. Usually a string, but sometimes an integer in the
    /// raw dataset. Use serde_json::Value and coerce to string.
    answer: serde_json::Value,
    /// Parallel arrays: haystack_sessions[i] occurred at haystack_dates[i].
    /// Each session is a list of turns.
    haystack_sessions: Vec<Vec<Turn>>,
    /// ISO-ish date strings, one per session.
    haystack_dates: Vec<String>,
    /// Which session indices contain the evidence for this question.
    answer_session_ids: Vec<usize>,
    /// Optional: session IDs with has_answer markers on specific turns.
    #[serde(default)]
    haystack_session_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Turn {
    role: String,        // "user" or "assistant"
    content: String,     // turn text
    #[serde(default)]
    has_answer: bool,    // true if this turn contains evidence
}
```

**Confidence: MEDIUM** -- Field names confirmed from the LongMemEval retrieval code (`run_retrieval.py`) which accesses `question_id`, `question_type`, `question`, `answer`, `question_date`, `haystack_sessions`, `haystack_dates`, `answer_session_ids`, `haystack_session_ids`. Turn structure (`role`, `content`, `has_answer`) confirmed from the same source. The `answer` mixed-type issue is confirmed by the HuggingFace dataset viewer error. Exact nesting should be validated against the actual downloaded JSON file before implementation.

## Key Architectural Differences from LoCoMo

These are not stack differences but shape the serde types and harness structure:

| Aspect | LoCoMo | LongMemEval |
|--------|--------|-------------|
| **Instances** | 10 conversations, ~150 questions each | 500 questions, each with its own conversation |
| **Bank model** | 1 bank per conversation (shared across questions) | 1 bank per question (500 banks for a full run) |
| **Session format** | `session_N_dialogue` keys in a flat HashMap | `haystack_sessions` as a Vec of Vec of turns |
| **Date format** | `"3:00 PM on 5 January, 2023"` | ISO-ish date strings in `haystack_dates` array |
| **Categories** | 5 numeric (1-5, only 1-4 scored) | 7 string-valued types, all scored including abstention |
| **Judge** | Single prompt, CORRECT/WRONG JSON response | Task-specific prompts, yes/no binary response |
| **Answer type** | `Option<serde_json::Value>` (cat-5 may be null) | `serde_json::Value` (always present, mixed types) |
| **Scale** | ~1,540 questions total, 10 conversations | 500 questions, each with 50-500 sessions |

## Judge Evaluation Approach

LongMemEval's official evaluation uses **task-specific judge prompts** (unlike LoCoMo's single universal prompt). The six prompt variants are:

1. **Standard** (single-session-user, single-session-assistant, multi-session): "Does the response contain the correct answer? Yes/No." Accepts equivalent formulations, rejects partial info.
2. **Temporal-reasoning**: Same as standard but explicitly allows off-by-one errors for day/week/month calculations.
3. **Knowledge-update**: Accepts responses containing the updated answer even if they also mention previous info.
4. **Single-session-preference**: Evaluates against a rubric; accepts responses correctly utilizing personal information without requiring all rubric points.
5. **Abstention** (`_abs` suffix on question_id): Checks if model correctly identifies the question as unanswerable.

**Implementation recommendation**: Create 5 judge prompt templates (one per evaluation variant above) stored as `.txt` files in `bench/longmemeval/`. The harness selects the template based on `question_type` and whether `question_id` ends with `_abs`. This differs from LoCoMo which uses a single `judge_answer.txt`.

For comparability with published baselines (GPT-4o: 60.6%), the judge should output yes/no (matching LongMemEval's protocol) rather than CORRECT/WRONG (LoCoMo's protocol). The harness should convert to binary accuracy internally.

**Confidence: HIGH** -- Judge prompt structure confirmed from `evaluate_qa.py` source code in the LongMemEval repository.

## Scoring Metrics

LongMemEval uses simpler metrics than LoCoMo:

| Metric | LongMemEval | LoCoMo |
|--------|-------------|--------|
| Primary | Binary accuracy (LLM judge yes/no) | Binary accuracy + token F1 |
| Grouping | Per question-type + overall + task-averaged | Per category + per conversation |
| Abstention | Scored (30 false-premise questions) | Excluded (category 5 not scored) |
| Evidence recall | Not in official protocol | Evidence hit + evidence recall |
| Retrieval metrics | Optional Recall@k, NDCG@k (separate evaluation) | Not tracked |

**Implementation recommendation**: Compute binary accuracy per question-type and overall. Also compute token F1 for internal diagnostics (reuse LoCoMo's `token_f1` function) even though the official protocol does not require it. Track a "task-averaged" accuracy (mean of per-type accuracies) as this is the published comparison metric.

**Confidence: HIGH** -- Confirmed from `print_qa_metrics.py` source.

## Installation

No new crate installations needed. The Cargo.toml binary target is the only addition:

```toml
[[bin]]
name = "longmemeval-bench"
path = "bench/longmemeval/longmemeval.rs"

[[bin]]
name = "longmemeval-view"
path = "bench/longmemeval/view.rs"
```

Dataset download (manual, per PROJECT.md):

```bash
mkdir -p data
# From HuggingFace xiaowu0162/longmemeval-cleaned
wget -O data/longmemeval_s_cleaned.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
wget -O data/longmemeval_m_cleaned.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json"
```

## Code Sharing with LoCoMo

Candidate functions to share (extract to `bench/common/` or a shared module):

| Function | Current Location | Shareable? |
|----------|-----------------|------------|
| `build_judge_client` | `locomo.rs` | Yes -- identical LLM client setup |
| `token_f1` / `normalize_answer` | `locomo.rs` | Yes -- same F1 computation |
| Artifact writing (JSONL sidecars) | `locomo.rs` | Yes -- same three-artifact pattern |
| `MetricsCollector` integration | `locomo.rs` | Yes -- same stage-level metrics |
| Profile/config loading | `locomo.rs` | Partially -- different profile shapes |
| Judge prompt / parsing | `locomo.rs` | No -- LongMemEval needs task-specific prompts |
| Dataset types | `locomo.rs` | No -- completely different schemas |
| Session parsing | `locomo.rs` | No -- different formats |

**Recommendation**: Extract `build_judge_client`, `token_f1`, `normalize_answer`, and artifact-writing helpers into a shared `bench/common.rs` module. Keep dataset parsing, judge prompts, and session handling separate per benchmark.

**Confidence: MEDIUM** -- The sharing boundaries are clear from reading the code, but the exact refactoring depends on how tightly coupled the artifact-writing code is to LoCoMo-specific types. This should be validated during implementation.

## Sources

- [LongMemEval GitHub Repository](https://github.com/xiaowu0162/LongMemEval) -- ICLR 2025, dataset structure, evaluation code
- [LongMemEval Paper (arXiv:2410.10813)](https://arxiv.org/abs/2410.10813) -- Dataset format, evaluation protocol, baseline results
- [longmemeval-cleaned Dataset (HuggingFace)](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) -- Cleaned dataset files, mixed-type answer field issue
- [LongMemEval evaluate_qa.py](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py) -- Judge prompt templates, scoring logic
- [LongMemEval print_qa_metrics.py](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/print_qa_metrics.py) -- Metric aggregation, category groupings
- [LongMemEval run_retrieval.py](https://github.com/xiaowu0162/LongMemEval/blob/main/src/retrieval/run_retrieval.py) -- Dataset field access patterns, haystack_sessions structure
- [Backboard LongMemEval Results](https://github.com/Backboard-io/Backboard-longmemEval-results) -- 93.4% accuracy, per-question bank isolation approach
- [EverMemOS Evaluation Framework](https://github.com/EverMind-AI/EverMemOS/tree/main/evaluation) -- Cross-benchmark evaluation, format conversion patterns
- [serde_json crate](https://crates.io/crates/serde_json) -- Current version 1.0.149, already in Cargo.toml

