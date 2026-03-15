# Phase 6: Wire Temporal Context & Session Limit - Research

**Researched:** 2026-03-15
**Domain:** LongMemEval benchmark harness gap closure (Rust)
**Confidence:** HIGH

## Summary

This phase closes two audit gaps identified in the v1.0 milestone audit. Both are straightforward wiring tasks -- the infrastructure exists but the connections are missing.

**Gap 1 (DATA-06):** `question_date` is parsed and stored as a `String` field on `LongMemEvalInstance` (Phase 1), but `ReflectQuery` only has three fields (`bank_id`, `question`, `budget_tokens`). The QA loop at line 1444 of `longmemeval.rs` constructs a `ReflectQuery` without temporal context. The reflect agent's system prompt never receives temporal grounding. Fix: add an `Option<String>` temporal context field to `ReflectQuery`, populate it from `instance.question_date` in the QA loop, and inject it into the user message or system prompt in `reflect_inner()`.

**Gap 2 (session_limit):** `--session-limit N` is parsed into `RunConfig.session_limit`, serialized into the manifest, printed in config summary, and restored during QA resume -- but it is never forwarded to `IngestConfig` or applied in `ingest_instance()`. LoCoMo handles this identically via `options.max_sessions` which slices the session iterator (line 2442 of `locomo.rs`). Fix: add `session_limit: Option<usize>` to `IngestConfig`, pass it from `RunConfig`, and apply `.take(n)` on the session iterator in `ingest_instance()`.

**Primary recommendation:** Implement both fixes in a single plan. Both are wiring-only changes with no new dependencies. Follow the LoCoMo reference patterns exactly.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-06 | `question_date` parsed and passed to reflect agent as temporal context | ReflectQuery needs temporal_context field; reflect_inner() must inject into prompt; QA loop must populate from instance.question_date |
</phase_requirements>

## Standard Stack

### Core

No new dependencies needed. All changes are in existing files:

| File | Purpose | Change Type |
|------|---------|-------------|
| `src/types/pipeline.rs` | `ReflectQuery` struct | Add `temporal_context: Option<String>` field |
| `src/reflect/mod.rs` | `reflect_inner()` | Inject temporal context into user message |
| `bench/longmemeval/ingest.rs` | `IngestConfig`, `ingest_instance()` | Add `session_limit` field, apply `.take()` |
| `bench/longmemeval/longmemeval.rs` | QA loop, `IngestConfig` construction | Wire `question_date` to `ReflectQuery`, `session_limit` to `IngestConfig` |
| `src/server/handlers.rs` | HTTP reflect handler | No change needed (already takes `ReflectQuery` from JSON body) |
| `src/mcp/mod.rs` | MCP reflect tool | No change needed (constructs `ReflectQuery` without temporal context, which is fine for non-bench usage) |

### Supporting

| Library | Purpose |
|---------|---------|
| `chrono` | Already in scope for date parsing (`parse_haystack_date`) |
| `serde` | Already derived on all affected structs |

## Architecture Patterns

### Pattern 1: Optional Field Addition to ReflectQuery

**What:** Add `temporal_context: Option<String>` to `ReflectQuery` with `#[serde(default)]` for backward compatibility.

**Why Option<String> not DateTime:** The `question_date` in LongMemEval is a formatted string like `"2023/05/25 (Thu) 14:30"`. The reflect agent needs human-readable temporal context in its prompt, not a machine timestamp. Passing the string avoids unnecessary parsing then re-formatting. The MCP and HTTP API callers don't use this field, so `Option` with `#[serde(default)]` keeps backward compat.

**Where it goes in the prompt:** Inject as a prefix to the user message, not the system prompt. The system prompt is bank-profile-specific (disposition, directives, mission). Temporal context is query-specific and belongs with the question. Pattern:

```rust
// In reflect_inner(), replace:
let mut messages: Vec<Message> = vec![Message::text("user", query.question.clone())];

// With:
let user_content = if let Some(ref tc) = query.temporal_context {
    format!("[Current date: {tc}]\n\n{}", query.question)
} else {
    query.question.clone()
};
let mut messages: Vec<Message> = vec![Message::text("user", user_content)];
```

This is the same pattern as the `[Date: YYYY-MM-DD]` prefix used for session ingestion (INGEST-02). The reflect agent will see temporal grounding naturally as part of the question context.

### Pattern 2: Session Limit via Iterator .take()

**What:** Slice the session iterator to limit ingestion count.

**Reference pattern:** LoCoMo `locomo.rs` line 2442-2445:
```rust
let ingest_sessions = options
    .max_sessions
    .map(|m| m.min(total_sessions))
    .unwrap_or(total_sessions);
// then: for idx in 1..=ingest_sessions { ... }
```

LongMemEval equivalent in `ingest_instance()` line 261-266:
```rust
// Current:
for (idx, (session, date_str)) in instance
    .haystack_sessions
    .iter()
    .zip(instance.haystack_dates.iter())
    .enumerate()
{ ... }

// Fixed:
let session_count = config.session_limit
    .map(|n| n.min(instance.haystack_sessions.len()))
    .unwrap_or(instance.haystack_sessions.len());

for (idx, (session, date_str)) in instance
    .haystack_sessions
    .iter()
    .zip(instance.haystack_dates.iter())
    .take(session_count)
    .enumerate()
{ ... }
```

Also update `total_sessions` reporting to distinguish available vs selected.

### Pattern 3: IngestConfig Wiring

**What:** Forward `RunConfig.session_limit` to `IngestConfig.session_limit` at the construction site in the spawn closure.

**Current construction site** (longmemeval.rs line 1347-1350):
```rust
let ingest_config = IngestConfig {
    format: ingest_format,
    consolidation,
};
```

**Fixed:**
```rust
let ingest_config = IngestConfig {
    format: ingest_format,
    consolidation,
    session_limit,
};
```

The `session_limit` value needs to be captured in the spawn closure alongside `ingest_format` and `consolidation` (both already captured as copies at line 1315-1316).

### Anti-Patterns to Avoid

- **Adding temporal_context to system prompt:** System prompt is for bank personality and agent instructions. Temporal context is query-specific, belongs in user message.
- **Parsing question_date to DateTime then back to string:** Unnecessary round-trip. Pass the string directly. The bench harness already has `parse_haystack_date()` but it is for timestamps used in `RetainInput`, not for display.
- **Making temporal_context a required field:** Would break all existing callers (MCP, HTTP API, LoCoMo bench). Must be `Option<String>` with `#[serde(default)]`.
- **Applying session_limit after enumeration:** Must `.take()` before `.enumerate()` so indices stay correct and `total_sessions` reporting is accurate.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Date formatting | Custom date parser for `question_date` | Pass raw string directly | `question_date` is already human-readable ("2023/05/25 (Thu) 14:30"), no parsing needed |
| Iterator slicing | Manual index bounds checking | `.take(n)` on iterator chain | Rust standard library; cleaner, no off-by-one risk |

## Common Pitfalls

### Pitfall 1: Breaking Serde Backward Compatibility on ReflectQuery

**What goes wrong:** Adding a required field to `ReflectQuery` breaks deserialization of existing JSON that doesn't include the new field.
**Why it happens:** `ReflectQuery` is `#[derive(Serialize, Deserialize)]` and used in HTTP API, MCP, and benchmarks.
**How to avoid:** Use `#[serde(default)]` on the new field. The existing roundtrip test `reflect_query_roundtrip()` at pipeline.rs line 421 must still pass.
**Warning signs:** Test `reflect_query_roundtrip` fails.

### Pitfall 2: session_limit Captured as Reference Not Copy

**What goes wrong:** The spawn closure needs `session_limit: Option<usize>` as a value, not a reference. `usize` is `Copy` and `Option<usize>` is `Copy`, so this should work naturally, but if captured from a reference to `config.session_limit` the borrow checker will complain.
**How to avoid:** Capture `let session_limit = config.session_limit;` outside the spawn, same as `ingest_format` and `consolidation` are captured at lines 1315-1316.

### Pitfall 3: IngestStats.sessions_ingested Count Mismatch

**What goes wrong:** After applying session_limit, the `sessions_ingested` stat should reflect the actual number ingested (capped), not the total available.
**How to avoid:** `sessions_ingested` is already incremented per-iteration in the loop, so limiting the iterator naturally limits the count. Just verify `total_sessions` in the log message shows both available and selected when a limit is active.

### Pitfall 4: Temporal Context Format for LLM Consumption

**What goes wrong:** Raw `question_date` format `"2023/05/25 (Thu) 14:30"` might confuse the LLM or be inconsistent with the `[Date: YYYY-MM-DD]` prefix format used in ingested sessions.
**How to avoid:** Use `parse_date_prefix()` from `ingest.rs` to normalize the date format. This produces `[Date: 2023-05-25]` which matches the format already in the bank's memories. Alternatively, pass the full datetime string for more precision. Either approach works -- the key is consistency with what the memory bank contains.
**Recommendation:** Use `parse_date_prefix(&instance.question_date)` for consistency, then format as `[Current date: 2023-05-25]` in the user message.

## Code Examples

### Adding temporal_context to ReflectQuery

```rust
// src/types/pipeline.rs - ReflectQuery
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReflectQuery {
    pub bank_id: BankId,
    pub question: String,
    pub budget_tokens: usize,
    /// Optional temporal context for grounding time-sensitive questions.
    #[serde(default)]
    pub temporal_context: Option<String>,
}
```

### Injecting temporal context in reflect_inner()

```rust
// src/reflect/mod.rs - in reflect_inner(), around line 222
let user_content = if let Some(ref tc) = query.temporal_context {
    format!("{tc}\n\n{}", query.question)
} else {
    query.question.clone()
};
let mut messages: Vec<Message> = vec![Message::text("user", user_content)];
```

### Populating temporal context in QA loop

```rust
// bench/longmemeval/longmemeval.rs - around line 1444
runtime.reflect.reflect(&ReflectQuery {
    bank_id: bank_id_str.parse().unwrap(),
    question: instance.question.clone(),
    budget_tokens: REFLECT_BUDGET_TOKENS,
    temporal_context: Some(
        ingest::parse_date_prefix(&instance.question_date)
    ),
})
```

### Adding session_limit to IngestConfig

```rust
// bench/longmemeval/ingest.rs - IngestConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestConfig {
    pub format: IngestFormat,
    pub consolidation: ConsolidationMode,
    /// Maximum number of sessions to ingest. None = all sessions.
    #[serde(default)]
    pub session_limit: Option<usize>,
}
```

### Applying session_limit in ingest_instance()

```rust
// bench/longmemeval/ingest.rs - in ingest_instance(), around line 261
let ingest_count = config.session_limit
    .map(|n| n.min(total_sessions))
    .unwrap_or(total_sessions);

info!(
    question_id = %instance.question_id,
    sessions_available = total_sessions,
    sessions_selected = ingest_count,
    format = ?config.format,
    consolidation = ?config.consolidation,
    "starting ingestion"
);

for (idx, (session, date_str)) in instance
    .haystack_sessions
    .iter()
    .zip(instance.haystack_dates.iter())
    .take(ingest_count)
    .enumerate()
{ ... }
```

### Wiring session_limit in the spawn closure

```rust
// bench/longmemeval/longmemeval.rs - around line 1315
let ingest_format = config.ingest_format;
let consolidation = config.consolidation;
let session_limit = config.session_limit;  // ADD THIS

// ... in spawn closure, around line 1347:
let ingest_config = IngestConfig {
    format: ingest_format,
    consolidation,
    session_limit,  // ADD THIS
};
```

## State of the Art

No new libraries, patterns, or breaking changes relevant. All changes are internal wiring within an existing codebase.

| Concern | Status |
|---------|--------|
| `ReflectQuery` temporal support | Missing -- this phase adds it |
| `session_limit` wiring | Parsed but dead code -- this phase activates it |
| LoCoMo reference implementation | Already working with `max_sessions` -- follow that pattern |

## Open Questions

1. **Temporal context format in user message**
   - What we know: LongMemEval `question_date` is `"2023/05/25 (Thu) 14:30"`. Sessions in the bank use `[Date: YYYY-MM-DD]` prefix format.
   - What's unclear: Whether to pass raw datetime string or normalize to ISO date only.
   - Recommendation: Normalize via `parse_date_prefix()` for consistency with bank content. This drops the time component but matches the format the LLM already sees in memories.

2. **Whether MCP/API callers should get temporal_context**
   - What we know: The field is `Option<String>` so callers can optionally provide it. Current MCP `ReflectParams` doesn't expose it.
   - What's unclear: Whether to expose it in MCP now.
   - Recommendation: Don't change MCP/API now. `Option<String>` with `#[serde(default)]` means they can adopt it later without a breaking change. This phase is bench-focused.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (Rust built-in) |
| Config file | Cargo.toml |
| Quick run command | `cargo test --bin longmemeval-bench` |
| Full suite command | `cargo test --all` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-06-a | ReflectQuery has temporal_context field | unit | `cargo test -p elephant reflect_query_roundtrip -- --exact` | Yes (existing test, needs update) |
| DATA-06-b | Temporal context injected into user message | unit | `cargo test -p elephant temporal_context_in_user_message -- --exact` | No -- Wave 0 |
| DATA-06-c | QA loop passes question_date to ReflectQuery | unit | `cargo test --bin longmemeval-bench reflect_query_includes_temporal -- --exact` | No -- Wave 0 |
| GAP-SL-a | IngestConfig has session_limit field | unit | `cargo test --bin longmemeval-bench ingest_config_default -- --exact` | Yes (existing test, needs update) |
| GAP-SL-b | ingest_instance respects session_limit | unit | `cargo test --bin longmemeval-bench -- session_limit` | No -- Wave 0 (needs mock runtime) |
| GAP-SL-c | session_limit wired from RunConfig | unit | `cargo test --bin longmemeval-bench -- session_limit` | No (covered by GAP-SL-b) |

### Sampling Rate
- **Per task commit:** `cargo test --bin longmemeval-bench && cargo test -p elephant -- reflect`
- **Per wave merge:** `cargo test --all`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] Update `reflect_query_roundtrip` test in `src/types/pipeline.rs` -- add `temporal_context: Some("...")` to roundtrip
- [ ] New test `temporal_context_in_user_message` in `src/reflect/mod.rs` -- verify user message prefix
- [ ] Update `ingest_config_default` test in `bench/longmemeval/ingest.rs` -- verify `session_limit: None` default
- [ ] New test for session_limit slicing behavior (mock or logic test, not integration)

## Sources

### Primary (HIGH confidence)
- `src/types/pipeline.rs` -- `ReflectQuery` struct definition (3 fields, no temporal)
- `src/reflect/mod.rs` -- `reflect_inner()` system prompt and message construction
- `bench/longmemeval/longmemeval.rs` -- QA loop ReflectQuery construction (line 1444), RunConfig (line 214), IngestConfig construction (line 1347)
- `bench/longmemeval/ingest.rs` -- `IngestConfig` struct (2 fields), `ingest_instance()` session loop (line 261)
- `bench/locomo/locomo.rs` -- Reference pattern for `max_sessions` slicing (line 2442-2445)
- `.planning/v1.0-MILESTONE-AUDIT.md` -- Gap descriptions and evidence

### Secondary (MEDIUM confidence)
- `bench/longmemeval/dataset.rs` -- `LongMemEvalInstance.question_date` field type (String, line 37)
- `src/server/handlers.rs` -- HTTP reflect handler (takes `Json<ReflectQuery>`, line 134)
- `src/mcp/mod.rs` -- MCP reflect tool (constructs ReflectQuery, line 187)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all changes are in files I read directly
- Architecture: HIGH -- both patterns have reference implementations in the codebase (LoCoMo session_limit, session date prefix)
- Pitfalls: HIGH -- serde compat and iterator behavior are well-understood

**Research date:** 2026-03-15
**Valid until:** indefinite (internal codebase, no external dependencies)
