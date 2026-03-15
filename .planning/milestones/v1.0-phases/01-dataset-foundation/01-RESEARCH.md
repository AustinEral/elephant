# Phase 1: Dataset Foundation - Research

**Researched:** 2026-03-15
**Domain:** LongMemEval dataset parsing, Rust serde deserialization, FNV hashing, bench infrastructure
**Confidence:** HIGH

## Summary

Phase 1 involves parsing the LongMemEval dataset files (`longmemeval_s_cleaned.json` at 277MB, `longmemeval_m_cleaned.json` at 2.74GB) into correctly typed Rust structs, validating semantic constraints, computing deterministic fingerprints, and extracting shared bench infrastructure into `bench/common/`. The dataset contains 500 question instances each with conversation histories (sessions), timestamps, and mixed-type answer fields.

The core technical challenge is straightforward serde deserialization with a few wrinkles: the `answer` field contains mixed types (strings and integers), `question_type` uses 6 string variants (with abstention identified by `_abs` suffix on `question_id`, not a separate type), and `haystack_sessions`/`haystack_dates` must be zip-validated. The existing LoCoMo harness provides proven patterns for FNV fingerprinting, JSONL helpers, and sidecar paths that should be extracted to `bench/common/`.

**Primary recommendation:** Use `serde_json::Value` for the raw `answer` field with an `answer_to_string()` coercion function (matching LoCoMo's pattern), a 6-variant `QuestionType` enum with `#[serde(rename_all = "kebab-case")]`, and two-layer validation (serde parse + semantic pass).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Extract shared bench infrastructure into `bench/common/` module -- both harnesses import from it
- Share where it's natural: FNV fingerprinting, artifact I/O (JSONL helpers, sidecar paths), judge evaluation, stage metrics
- Refactor LoCoMo when straightforward and aligned with multi-bench future; defer painful refactors as tracked tech debt
- Two-layer validation: (1) serde JSON parse (fail-on-first), (2) semantic validation pass collecting all errors before reporting
- All validation runs upfront before any pipeline work begins
- Semantic validation checks: question_type membership, session/date length match (DATA-07), answer coercion success, required fields present
- Dataset files live in `data/` (gitignored), helpful error with download instructions pointing to `xiaowu0162/longmemeval-cleaned` on HuggingFace
- Manual download only -- no HF SDK dependency

### Claude's Discretion
- Serde struct design for LongMemEval instances (field types, enum variants)
- Question type enum variant naming (mapped from the 7 LongMemEval categories)
- Answer coercion implementation (`serde_json::Value` to string)
- FNV fingerprinting extraction approach from LoCoMo
- Internal module organization within `bench/longmemeval/`

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Harness loads `longmemeval_s_cleaned.json` (~50 sessions/instance, ~115k tokens) | Serde deserialization of top-level JSON array; `fs::read()` + `serde_json::from_slice()` pattern from LoCoMo; 277MB file |
| DATA-02 | Harness loads `longmemeval_m_cleaned.json` (~500 sessions/instance, ~1.5M tokens) | Same pattern but 2.74GB file; `from_slice` over `from_str` avoids double-allocation |
| DATA-03 | All 500 questions parsed with correct question_type categorization (7 types) | 6 `question_type` enum variants + abstention detected via `_abs` suffix on `question_id`; see QuestionType enum design |
| DATA-04 | Mixed-type `answer` field handled via `serde_json::Value` coercion to string | `answer_to_string()` pattern already proven in LoCoMo; handles String, Number, fallback |
| DATA-05 | Dataset fingerprinting (FNV1a-64 hash) stored in manifest for reproducibility | `fnv1a64()` and `fnv1a64_hex()` functions exist in LoCoMo; extract to `bench/common/` |
| DATA-06 | `question_date` parsed and passed to reflect agent as temporal context | Date format is `YYYY/MM/DD (Day) HH:MM` (e.g., `2023/05/25 (Thu) 14:30`); parse as string, consumed in Phase 4 |
| DATA-07 | `haystack_sessions` and `haystack_dates` zip-validated (equal length assertion) | Semantic validation pass checks `haystack_sessions.len() == haystack_dates.len()` per instance |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| serde | 1.0.228 | Struct serialization/deserialization | Already in Cargo.toml, standard for Rust JSON |
| serde_json | 1.0.149 | JSON parsing, `Value` type for mixed-type fields | Already in Cargo.toml |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| thiserror | 2.0.18 | Error types for validation errors | Already in Cargo.toml; use for `DatasetError` enum |
| anyhow | 1.0.102 | Error handling in binary/test code | Already in dev-dependencies |

### No New Dependencies Needed
All required functionality is covered by existing crate dependencies. FNV hashing is hand-rolled (6 lines, already in LoCoMo). No new crates needed for this phase.

## Architecture Patterns

### Recommended Project Structure
```
bench/
├── common/
│   ├── mod.rs           # Re-exports
│   ├── fingerprint.rs   # fnv1a64(), fnv1a64_hex()
│   └── io.rs            # append_jsonl(), sidecar_path(), write_jsonl_records()
├── longmemeval/
│   ├── mod.rs           # Re-exports
│   └── dataset.rs       # LongMemEval types, loading, validation
├── locomo/
│   └── locomo.rs        # Existing harness (imports from bench::common)
└── view.rs              # Existing view tool
```

### Pattern 1: LongMemEval Dataset Types

**What:** Serde structs matching the upstream JSON schema exactly.

**Schema (from upstream `sample_haystack_and_timestamp.py`):**
```json
{
  "question_id": "single_hop_1",
  "question_type": "single-session-user",
  "question": "What is the name of...",
  "answer": "John" | 42,
  "question_date": "2023/05/25 (Thu) 14:30",
  "haystack_dates": ["2023/05/20 (Mon) 10:15", ...],
  "haystack_session_ids": ["session_123", ...],
  "haystack_sessions": [
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    ...
  ],
  "answer_session_ids": ["answer_session_456", ...]
}
```

**Recommended Rust types:**
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuestionType {
    SingleSessionUser,
    SingleSessionAssistant,
    SingleSessionPreference,
    MultiSession,
    TemporalReasoning,
    KnowledgeUpdate,
}

#[derive(Debug, Deserialize)]
pub struct LongMemEvalInstance {
    pub question_id: String,
    pub question_type: QuestionType,
    pub question: String,
    pub answer: serde_json::Value,  // Mixed string/int
    pub question_date: String,
    pub haystack_dates: Vec<String>,
    pub haystack_session_ids: Vec<String>,
    pub haystack_sessions: Vec<Vec<Turn>>,
    pub answer_session_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Turn {
    pub role: String,
    pub content: String,
}
```

**Why 6 variants, not 7:** The upstream `evaluate_qa.py` and `print_qa_metrics.py` both use exactly 6 `question_type` values. Abstention is NOT a separate `question_type` -- it is identified by the `_abs` suffix on `question_id`. The "7 categories" in the paper/requirements counts abstention as a reporting category, but in the data schema it is orthogonal to `question_type`. The type system should reflect the data reality (6 enum variants), with a helper method `is_abstention(&self) -> bool` on the instance checking for the `_abs` suffix.

### Pattern 2: Two-Layer Validation

**What:** First layer is serde's structural parse (fail-fast on malformed JSON). Second layer is a semantic validation pass that collects ALL errors before reporting.

**When to use:** After successful serde parse, before any pipeline work.

```rust
pub struct ValidationError {
    pub instance_index: usize,
    pub question_id: String,
    pub errors: Vec<String>,
}

pub fn validate_dataset(instances: &[LongMemEvalInstance]) -> Result<(), Vec<ValidationError>> {
    let mut all_errors = Vec::new();
    for (i, inst) in instances.iter().enumerate() {
        let mut errors = Vec::new();

        // DATA-07: zip validation
        if inst.haystack_sessions.len() != inst.haystack_dates.len() {
            errors.push(format!(
                "haystack_sessions ({}) and haystack_dates ({}) length mismatch",
                inst.haystack_sessions.len(), inst.haystack_dates.len()
            ));
        }

        // Session IDs length match
        if inst.haystack_sessions.len() != inst.haystack_session_ids.len() {
            errors.push(format!(
                "haystack_sessions ({}) and haystack_session_ids ({}) length mismatch",
                inst.haystack_sessions.len(), inst.haystack_session_ids.len()
            ));
        }

        // Answer coercion check
        if inst.answer.is_null() || inst.answer.is_object() || inst.answer.is_array() {
            errors.push(format!("answer is not a string or number: {:?}", inst.answer));
        }

        // question_date non-empty
        if inst.question_date.is_empty() {
            errors.push("question_date is empty".into());
        }

        if !errors.is_empty() {
            all_errors.push(ValidationError {
                instance_index: i,
                question_id: inst.question_id.clone(),
                errors,
            });
        }
    }

    if all_errors.is_empty() { Ok(()) } else { Err(all_errors) }
}
```

### Pattern 3: Answer Coercion

**What:** Convert `serde_json::Value` to `String`, handling mixed types.

**Source:** Direct port from LoCoMo's `answer_to_string()` in `bench/locomo/locomo.rs:558-563`.

```rust
pub fn answer_to_string(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}
```

### Pattern 4: Dataset Loading with Missing-File Help

**What:** Load dataset file, provide helpful error if missing.

```rust
pub fn load_dataset(path: &Path) -> Result<Vec<LongMemEvalInstance>, String> {
    if !path.exists() {
        return Err(format!(
            "Dataset file not found: {}\n\n\
             Download from HuggingFace:\n\
             \n\
             # Install git-lfs first, then:\n\
             git clone https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned\n\
             cp longmemeval-cleaned/longmemeval_s_cleaned.json data/\n\
             cp longmemeval-cleaned/longmemeval_m_cleaned.json data/",
            path.display()
        ));
    }

    let raw_bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    let instances: Vec<LongMemEvalInstance> = serde_json::from_slice(&raw_bytes)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;

    Ok(instances)
}
```

### Pattern 5: Shared FNV Fingerprinting (bench/common/)

**What:** Extract `fnv1a64()` and `fnv1a64_hex()` from LoCoMo into `bench/common/fingerprint.rs`.

**Source:** `bench/locomo/locomo.rs:595-606` -- these are standalone pure functions with no dependencies.

```rust
// bench/common/fingerprint.rs
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn fnv1a64_hex(data: &str) -> String {
    format!("{:016x}", fnv1a64(data.as_bytes()))
}
```

**Fingerprint computation:** Hash the raw dataset bytes (same as LoCoMo: `format!("{:016x}", fnv1a64(&raw_bytes))`).

### Anti-Patterns to Avoid
- **Parsing `question_date` into chrono types now:** The format is `YYYY/MM/DD (Day) HH:MM` which is non-standard. Store as `String` in Phase 1; Phase 4 decides how to pass it to reflect. Premature parsing adds complexity for no value.
- **Creating an `Abstention` variant in `QuestionType`:** The data doesn't have `"abstention"` as a `question_type` value. Abstention is a cross-cutting concern detected by `_abs` suffix. Adding a fake enum variant would cause serde deserialization to fail.
- **Streaming JSON parse:** Both files are arrays. The S file is 277MB, the M file is 2.74GB. `serde_json::from_slice()` on the raw bytes works (LoCoMo uses this pattern for its 2.8MB dataset). The M file at 2.74GB will need significant RAM but is still feasible with `from_slice` on a development machine with 16GB+ RAM. If memory becomes an issue, `serde_json::StreamDeserializer` can be used later, but don't optimize prematurely.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FNV1a-64 hashing | New hash function | Extract existing `fnv1a64()` from LoCoMo | Proven, deterministic, 6 lines |
| JSONL append | New file I/O helpers | Extract existing `append_jsonl()` from LoCoMo | Handles create + append correctly |
| Sidecar paths | New path generation | Extract existing `sidecar_path()` from LoCoMo | Convention-preserving |
| JSON enum deserialization | Custom deserializer | `#[serde(rename_all = "kebab-case")]` | Serde handles `"single-session-user"` <-> `SingleSessionUser` automatically |
| Mixed-type answer handling | Custom deserializer | `serde_json::Value` + `answer_to_string()` | Proven pattern from LoCoMo |

## Common Pitfalls

### Pitfall 1: Assuming 7 question_type enum variants
**What goes wrong:** Creating an `Abstention` variant causes serde to fail because the data never contains `"abstention"` as a `question_type` value.
**Why it happens:** The paper and requirements refer to "7 categories" but abstention is identified by `_abs` suffix on `question_id`, not a separate `question_type`.
**How to avoid:** Use 6 `QuestionType` variants. Detect abstention with `question_id.ends_with("_abs")`. Report abstention as a 7th category in evaluation output (Phase 4).
**Warning signs:** Serde parse failure on the first abstention question.

### Pitfall 2: Using `from_str` instead of `from_slice` for large files
**What goes wrong:** `fs::read_to_string()` + `serde_json::from_str()` forces UTF-8 validation AND then serde parses, doubling memory for the string allocation.
**Why it happens:** `from_str` is more commonly seen in examples.
**How to avoid:** Use `fs::read()` + `serde_json::from_slice()`. This validates UTF-8 during parse, using less peak memory.
**Warning signs:** OOM on the 2.74GB M dataset.

### Pitfall 3: Not validating haystack_session_ids length
**What goes wrong:** `haystack_sessions`, `haystack_dates`, and `haystack_session_ids` must all have the same length. Forgetting `haystack_session_ids` in the zip-validation.
**Why it happens:** DATA-07 mentions only sessions and dates, but session_ids is a third parallel array.
**How to avoid:** Validate all three arrays have equal length in the semantic pass.
**Warning signs:** Index-out-of-bounds during ingestion (Phase 2).

### Pitfall 4: LoCoMo refactor breaking existing tests
**What goes wrong:** Extracting functions from `locomo.rs` to `bench/common/` breaks the LoCoMo binary or its inline tests.
**Why it happens:** Functions were `fn` (private) in locomo.rs; moving them requires `pub` and adjusting imports.
**How to avoid:** After extraction, run `cargo test --bin locomo-bench` and `cargo build --bin locomo-bench` to verify LoCoMo still works.
**Warning signs:** Compilation errors in locomo.rs after refactor.

### Pitfall 5: Module path resolution for bench binaries
**What goes wrong:** `bench/common/mod.rs` isn't automatically visible to `bench/locomo/locomo.rs` or `bench/longmemeval/` binaries because Rust binary targets don't share a module tree.
**Why it happens:** Each `[[bin]]` target is an independent crate root. They can't use `mod common;` to import sibling files.
**How to avoid:** Use `#[path = "../common/mod.rs"] mod common;` in each binary, or restructure as `bench/common.rs` with `#[path]` attributes. Alternatively, put shared code in `src/bench/` as part of the library crate and import via `elephant::bench::common`.
**Warning signs:** "file not found for module `common`" compiler error.

## Code Examples

### Loading and Validating a Dataset (Complete Flow)
```rust
// Source: Synthesized from LoCoMo patterns + LongMemEval schema

use std::path::Path;

fn load_and_validate(path: &Path) -> Result<(Vec<LongMemEvalInstance>, String), String> {
    // Layer 1: Read + parse (fail-fast on malformed JSON)
    let raw_bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    let fingerprint = format!("{:016x}", common::fnv1a64(&raw_bytes));

    let instances: Vec<LongMemEvalInstance> = serde_json::from_slice(&raw_bytes)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;

    // Layer 2: Semantic validation (collect all errors)
    validate_dataset(&instances)?;

    Ok((instances, fingerprint))
}
```

### Abstention Detection
```rust
// Source: Upstream evaluate_qa.py uses `'_abs' in entry['question_id']`

impl LongMemEvalInstance {
    pub fn is_abstention(&self) -> bool {
        self.question_id.contains("_abs")
    }

    /// Returns the reporting category (7 categories: 6 question types + abstention)
    pub fn reporting_category(&self) -> &str {
        if self.is_abstention() {
            "abstention"
        } else {
            match self.question_type {
                QuestionType::SingleSessionUser => "single-session-user",
                QuestionType::SingleSessionAssistant => "single-session-assistant",
                QuestionType::SingleSessionPreference => "single-session-preference",
                QuestionType::MultiSession => "multi-session",
                QuestionType::TemporalReasoning => "temporal-reasoning",
                QuestionType::KnowledgeUpdate => "knowledge-update",
            }
        }
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `longmemeval` (original HF dataset) | `longmemeval-cleaned` (cleaned dataset) | ~2024 | Cleaned version removes noisy sessions that interfere with answer correctness |
| Single evaluation prompt | Per-type judge prompts | LongMemEval paper | Different prompt templates for different question types + abstention |
| `question_type` includes abstention | `_abs` suffix on `question_id` | Dataset design | Abstention is orthogonal to question type |

## Open Questions

1. **Exact question_id patterns for abstention**
   - What we know: `_abs` suffix on `question_id` identifies abstention (from `evaluate_qa.py`: `'_abs' in entry['question_id']`)
   - What's unclear: Whether `_abs` always appears at the very end, or could be followed by a number (e.g., `single_hop_1_abs` vs `single_hop_1_abs_2`)
   - Recommendation: Use `.contains("_abs")` matching upstream Python code, not `.ends_with("_abs")`

2. **M dataset memory requirements**
   - What we know: `longmemeval_m_cleaned.json` is 2.74GB. `serde_json::from_slice()` will parse it but peak memory will be raw bytes + deserialized structs.
   - What's unclear: Exact peak memory usage -- could be 6-8GB depending on serde's intermediate allocations.
   - Recommendation: Use `from_slice` (not `from_str`). If OOM occurs on CI or small machines, add a streaming fallback in a future phase. This is a data-loading concern, not a Phase 1 blocker.

3. **`has_answer` field in turns**
   - What we know: Some turns in the upstream generation code include a `has_answer: true` field. The cleaned dataset generation explicitly strips this: `[{'role': y['role'], 'content': y['content']} for y in x['session']]`.
   - What's unclear: Whether any traces of `has_answer` remain in the cleaned files.
   - Recommendation: Define `Turn` with only `role` and `content`. If `has_answer` appears, serde will ignore unknown fields by default (no `#[serde(deny_unknown_fields)]`).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[cfg(test)]` + `#[test]` |
| Config file | None (Cargo's built-in test runner) |
| Quick run command | `cargo test --lib` |
| Full suite command | `cargo test` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Load S dataset (500 instances) | integration | `cargo test --test longmemeval_dataset -- load_s` | No -- Wave 0 |
| DATA-02 | Load M dataset (500 instances) | integration | `cargo test --test longmemeval_dataset -- load_m` | No -- Wave 0 |
| DATA-03 | QuestionType enum deserialization (6 variants) | unit | `cargo test --bin longmemeval-bench -- question_type` | No -- Wave 0 |
| DATA-04 | Answer coercion (string, int, fallback) | unit | `cargo test --bin longmemeval-bench -- answer_to_string` | No -- Wave 0 |
| DATA-05 | FNV fingerprint determinism | unit | `cargo test --bin longmemeval-bench -- fingerprint` | No -- Wave 0 |
| DATA-06 | question_date parsed as string | unit | `cargo test --bin longmemeval-bench -- question_date` | No -- Wave 0 |
| DATA-07 | Haystack sessions/dates length validation | unit | `cargo test --bin longmemeval-bench -- zip_validate` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test` (unit tests, no external deps needed for Phase 1)
- **Per wave merge:** `cargo test` + `cargo build --bin longmemeval-bench` + `cargo build --bin locomo-bench`
- **Phase gate:** All tests green + successful parse of both S and M datasets (manual, requires data files)

### Wave 0 Gaps
- [ ] `bench/common/` module -- shared FNV, JSONL, sidecar helpers
- [ ] `bench/longmemeval/dataset.rs` -- types, loading, validation
- [ ] Unit tests for QuestionType deserialization, answer coercion, zip-validation
- [ ] Integration tests for dataset loading (marked `#[ignore]` since they need data files in `data/`)
- [ ] Cargo.toml `[[bin]]` target for `longmemeval-bench`

## Sources

### Primary (HIGH confidence)
- [xiaowu0162/LongMemEval GitHub repo](https://github.com/xiaowu0162/LongMemEval) -- dataset structure, question types, evaluation code
- `src/evaluation/evaluate_qa.py` (raw GitHub fetch) -- definitive judge prompts, abstention detection via `_abs` in `question_id`, 6 question type values
- `src/evaluation/print_qa_metrics.py` (raw GitHub fetch) -- confirms exactly 6 `question_type` values with abstention tracked separately
- `data/custom_history/sample_haystack_and_timestamp.py` (raw GitHub fetch) -- definitive output JSON schema with all field names and types
- [xiaowu0162/longmemeval-cleaned HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) -- confirms mixed-type `answer` field (ArrowTypeError on int/string mix), file sizes (277MB S, 2.74GB M)
- `bench/locomo/locomo.rs` (local codebase) -- FNV fingerprinting, answer_to_string, JSONL helpers, sidecar paths

### Secondary (MEDIUM confidence)
- [LongMemEval paper (arXiv:2410.10813)](https://arxiv.org/html/2410.10813v1) -- 7 question types (6 + abstention), dataset composition, evaluation methodology

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all crates already in Cargo.toml, no new dependencies
- Architecture: HIGH -- JSON schema verified from upstream source code, patterns from existing LoCoMo harness
- Pitfalls: HIGH -- module path issue verified by Rust language rules, mixed-type answer confirmed by HuggingFace error and upstream code

**Research date:** 2026-03-15
**Valid until:** Stable dataset format, valid for 90+ days unless upstream pushes new data version
