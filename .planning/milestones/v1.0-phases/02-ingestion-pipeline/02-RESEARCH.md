# Phase 2: Ingestion Pipeline - Research

**Researched:** 2026-03-15
**Domain:** LongMemEval benchmark ingestion -- per-instance bank creation, session formatting, and consolidation
**Confidence:** HIGH

## Summary

Phase 2 builds the ingestion pipeline for LongMemEval benchmark instances. Each of the 500 questions gets its own isolated memory bank. Sessions (50 per instance in S dataset, 500 in M) are ingested sequentially with date prefixes parsed from `haystack_dates`. Two formatting modes are required: plain text (our approach) and raw JSON (for competitive comparison with Hindsight). Consolidation runs in one of three modes: end-of-ingestion, per-session, or off.

The existing LoCoMo bench harness (`bench/locomo/locomo.rs`) provides a thorough reference implementation covering bank creation, session-level ingestion with `RetainInput`, consolidation modes, metrics collection, and progress logging. The LongMemEval implementation is structurally simpler because the dataset format is cleaner (explicit arrays instead of dynamic session keys), but requires LongMemEval-specific date parsing and turn formatting.

**Primary recommendation:** Follow the LoCoMo ingestion pattern closely. The function `ingest_instance()` in `bench/longmemeval/ingest.rs` should create a bank, iterate sessions sequentially calling `runtime.retain.retain()`, then optionally consolidate. Return an `IngestResult` struct with bank_id, metrics, and timing data for Phase 3's artifact system.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Support two ingest formats via configuration: plain text (default) and raw JSON
- Plain text: `[Date: YYYY-MM-DD]\n\nuser: content\nassistant: content`
- Raw JSON: `json.dumps(cleaned_turns)` matching Hindsight/Vectorize approach
- Configurable via `IngestFormat` enum (text/json), passed in `IngestConfig`
- Date prefix always `[Date: YYYY-MM-DD]` (strip day-of-week and time from LongMemEval's full format)
- `context` field on `RetainInput`: `None` for session-level ingestion
- Three consolidation modes: `end` (default), `per-session`, `off` via `ConsolidationMode` enum
- Per-bank isolation: consolidation only operates within a single instance's bank
- Pool sizing is NOT a Phase 2 concern -- sequential ingestion uses the existing pool. Phase 5 addresses INGEST-05
- `ingest_instance()` returns `IngestResult` with `question_id -> bank_id` mapping for resume support
- Track per-instance stage metrics matching LoCoMo's `StageUsage`
- Ingestion logic in `bench/longmemeval/ingest.rs` -- separate from binary entry point
- LongMemEval-specific session formatters (not shared with LoCoMo)
- Runtime passed in as parameter, not built internally
- Function signature: `async fn ingest_instance(instance: &LongMemEvalInstance, runtime: &Runtime, config: &IngestConfig) -> Result<IngestResult>`

### Claude's Discretion
- `IngestConfig` and `IngestResult` struct design details
- How to strip `has_answer` field from turns before raw JSON formatting
- Error handling strategy for individual session failures within an instance
- Whether to log per-session progress or only per-instance

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INGEST-01 | Per-instance bank isolation -- one bank per question (500 banks for full run) | Bank creation via `store.create_bank()` with `MemoryBank` struct. LoCoMo pattern at line 2509-2522. Each instance gets a unique `BankId::new()`. |
| INGEST-02 | Session-level ingestion with date prefix (`[Date: 2023-01-15] content`) | Plain text format function producing `[Date: YYYY-MM-DD]\n\nrole: content` per session. Date parsed from `haystack_dates` entries like `"2023/05/20 (Sat) 02:21"`. |
| INGEST-03 | Consolidation modes: end (default), per-session, off | `ConsolidationMode` enum with `enabled()` and `per_session()` helpers. LoCoMo pattern at lines 1091-1136. Consolidator called via `runtime.consolidator.consolidate_with_progress()`. |
| INGEST-04 | Sessions ingested sequentially with timestamps from `haystack_dates` | Sequential loop over `instance.haystack_sessions.iter().zip(instance.haystack_dates.iter())`. Timestamp parsed to `DateTime<Utc>` for `RetainInput.timestamp`. |
| INGEST-05 | Postgres connection pool explicitly sized for concurrent bank operations | DEFERRED to Phase 5 per user decision. Phase 2 uses sequential ingestion with default pool. Document as not addressed in this phase. |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `chrono` | (already in deps) | Date parsing from `haystack_dates` format | Already used throughout codebase for `DateTime<Utc>` |
| `serde` / `serde_json` | (already in deps) | Raw JSON formatting, `IngestConfig` serialization | Already used throughout |
| `tokio` | (already in deps) | Async runtime for retain/consolidate calls | Already the project async runtime |
| `anyhow` | (already in deps) | Error handling in bench code | Project convention for non-library code |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `tracing` | (already in deps) | Structured logging during ingestion | Per-session progress logging |

### Alternatives Considered
None -- all dependencies are already in the project. No new crates needed.

**Installation:**
No new dependencies required.

## Architecture Patterns

### Recommended Project Structure
```
bench/
  longmemeval/
    mod.rs           # pub mod dataset; pub mod ingest;
    dataset.rs       # Existing: LongMemEvalInstance, Turn, load_dataset
    ingest.rs        # NEW: ingest_instance(), IngestConfig, IngestResult, formatters
    longmemeval.rs   # Binary entry point (Phase 3 expands this)
```

### Pattern 1: Session Formatting (Plain Text)

**What:** Convert a session's turns into the date-prefixed plain text format for retain.
**When to use:** Default `IngestFormat::Text` mode.
**Example:**
```rust
// LongMemEval haystack_dates format: "2023/05/20 (Sat) 02:21"
// Output format: "[Date: 2023-05-20]\n\nuser: content\nassistant: content"

fn format_session_text(turns: &[Turn], date_str: &str) -> String {
    let date_prefix = parse_date_prefix(date_str); // "[Date: 2023-05-20]"
    let dialogue = turns
        .iter()
        .map(|t| format!("{}: {}", t.role, t.content))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{date_prefix}\n\n{dialogue}")
}

/// Parse "2023/05/20 (Sat) 02:21" -> "[Date: 2023-05-20]"
fn parse_date_prefix(date_str: &str) -> String {
    // Take everything before the first space-paren: "2023/05/20"
    // Replace slashes with dashes: "2023-05-20"
    let date_part = date_str.split(" (").next().unwrap_or(date_str);
    let iso_date = date_part.replace('/', "-");
    format!("[Date: {iso_date}]")
}
```

### Pattern 2: Session Formatting (Raw JSON)

**What:** Serialize turns as JSON, stripping `has_answer` field, matching Hindsight's approach.
**When to use:** `IngestFormat::Json` mode for competitive comparison.
**Example:**
```rust
fn format_session_json(turns: &[Turn], session_id: &str, date_str: &str) -> String {
    // Build cleaned turn list (role + content only, no has_answer)
    let cleaned: Vec<serde_json::Value> = turns
        .iter()
        .map(|t| serde_json::json!({"role": t.role, "content": t.content}))
        .collect();
    serde_json::to_string(&cleaned).unwrap_or_default()
}
```
Note: Since `Turn` struct only has `role` and `content` fields (serde ignores unknown fields during deserialization), the Turn struct itself is already clean. For raw JSON mode, we just serialize the Turn array directly. The `has_answer` field is never deserialized into our struct.

### Pattern 3: Date Parsing for Timestamp

**What:** Parse LongMemEval date strings into `DateTime<Utc>` for `RetainInput.timestamp`.
**When to use:** Every session ingestion.
**Example:**
```rust
// LongMemEval format: "2023/05/20 (Sat) 02:21"
fn parse_haystack_date(date_str: &str) -> DateTime<Utc> {
    let trimmed = date_str.trim();
    // Try: "2023/05/20 (Sat) 02:21" -> strip day-of-week
    // Parse: "2023/05/20 02:21"
    let cleaned = trimmed
        .split('(')
        .next()
        .unwrap_or(trimmed)
        .trim()
        .to_string()
        + " "
        + trimmed.split(')').nth(1).unwrap_or("00:00").trim();

    if let Ok(ndt) = NaiveDateTime::parse_from_str(&cleaned, "%Y/%m/%d %H:%M") {
        return ndt.and_utc();
    }
    // Fallback: try just the date portion
    if let Ok(nd) = NaiveDate::parse_from_str(
        trimmed.split(" (").next().unwrap_or(trimmed), "%Y/%m/%d"
    ) {
        return nd.and_hms_opt(0, 0, 0).unwrap().and_utc();
    }
    Utc::now() // last resort fallback, same as LoCoMo
}
```

### Pattern 4: Bank Creation (from LoCoMo)

**What:** Create an isolated bank per instance.
**When to use:** Start of each `ingest_instance()` call.
**Example:**
```rust
// Source: bench/locomo/locomo.rs lines 2509-2522
let bank = MemoryBank {
    id: BankId::new(),
    name: format!("longmemeval-{}", instance.question_id),
    mission: "Long-term conversational memory benchmark".into(),
    directives: vec![],
    disposition: Disposition::default(),
    embedding_model: runtime.embeddings.model_name().to_string(),
    embedding_dimensions: runtime.embeddings.dimensions() as u16,
};
runtime.store.create_bank(&bank).await?;
```

### Pattern 5: Sequential Session Ingestion (from LoCoMo)

**What:** Iterate sessions, call `retain()` for each, accumulate stats.
**When to use:** Core ingestion loop.
**Example:**
```rust
// Source: bench/locomo/locomo.rs lines 2541-2587
for (session, date_str) in instance.haystack_sessions.iter()
    .zip(instance.haystack_dates.iter())
{
    let content = match config.format {
        IngestFormat::Text => format_session_text(session, date_str),
        IngestFormat::Json => format_session_json(session, session_id, date_str),
    };
    let timestamp = parse_haystack_date(date_str);

    match runtime.retain.retain(&RetainInput {
        bank_id: bank.id,
        content,
        timestamp,
        turn_id: None,
        context: None,
        custom_instructions: None,
        speaker: None,
    }).await {
        Ok(resp) => { /* accumulate stats */ }
        Err(e) => { /* log and continue */ }
    }
}
```

### Pattern 6: Consolidation After Ingestion (from LoCoMo)

**What:** Run consolidation in the configured mode after sessions are ingested.
**When to use:** After ingestion loop completes, or per-session.
**Example:**
```rust
// Source: bench/locomo/locomo.rs lines 2706-2734
match config.consolidation {
    ConsolidationMode::End => {
        let report = runtime.consolidator
            .consolidate_with_progress(bank_id, Some(progress_tx))
            .await?;
    }
    ConsolidationMode::PerSession => {
        // Called inside the session loop after each session
    }
    ConsolidationMode::Off => { /* no-op */ }
}
```

### Anti-Patterns to Avoid
- **Building runtime inside ingest_instance():** Runtime construction is expensive (pool, models, etc). Pass it in. The LoCoMo harness builds it once and shares via `Arc<ElephantRuntime>`.
- **Parsing dates with regex:** Use `chrono::NaiveDateTime::parse_from_str` with explicit format strings. The LoCoMo bench has a `parse_session_date()` function as reference.
- **Sharing banks across instances:** Each question MUST have its own bank. Cross-contamination violates the benchmark protocol (INGEST-01).
- **Using `turn_id` for session-level ingestion:** Session-level retain calls set `turn_id: None`. Only turn-level ingestion (not used in LongMemEval Phase 2) sets turn IDs.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bank creation | Custom DB inserts | `store.create_bank(&MemoryBank{..})` | Handles ID generation, embedding model metadata |
| Retain pipeline | Custom fact extraction | `runtime.retain.retain(&RetainInput{..})` | Full pipeline: chunk -> extract -> resolve -> graph -> opinion |
| Consolidation | Custom observation synthesis | `runtime.consolidator.consolidate_with_progress(..)` | Handles batching, progress, topic-scoped synthesis |
| Metrics collection | Manual token counting | `MetricsCollector` + `with_scoped_collector` | Thread-safe, per-stage tracking |
| Date prefix parsing | Manual string splitting | `chrono::NaiveDate::parse_from_str` | Handles edge cases, validated parsing |

**Key insight:** The entire retain and consolidation pipeline already exists. Phase 2 is a thin orchestration layer that creates banks, formats sessions, and calls existing pipelines.

## Common Pitfalls

### Pitfall 1: Date Format Mismatch
**What goes wrong:** LongMemEval dates are `"2023/05/20 (Sat) 02:21"` -- different from LoCoMo's `"3:00 PM on 5 March, 2023"`. Using LoCoMo's parser will fail silently and fall back to `Utc::now()`.
**Why it happens:** Copy-paste from LoCoMo without adapting the date format.
**How to avoid:** Write a dedicated `parse_haystack_date()` for LongMemEval's format. Test it against actual dataset values.
**Warning signs:** All timestamps are the same (current time) in the database.

### Pitfall 2: has_answer Field Leakage in Raw JSON Mode
**What goes wrong:** The `has_answer` field from turns leaks into the raw JSON content sent to retain, giving the system unfair hints about which turns contain answer information.
**Why it happens:** Serializing raw dataset JSON without filtering fields.
**How to avoid:** Since our `Turn` struct only deserializes `role` and `content`, serialize the struct (not raw JSON). The `has_answer` field is dropped during deserialization and never reaches the formatter.
**Warning signs:** Benchmark results suspiciously better than expected; JSON output contains `has_answer`.

### Pitfall 3: Consolidation Running on Wrong Bank
**What goes wrong:** Consolidation operates on facts from a previous instance's bank.
**Why it happens:** Bug in bank_id passing or forgetting to scope consolidation to current bank.
**How to avoid:** `consolidate()` takes `bank_id` parameter and is inherently scoped. Just pass the correct bank_id.
**Warning signs:** Observation counts don't match expected fact counts.

### Pitfall 4: Date Prefix Format Inconsistency
**What goes wrong:** Date prefix in content doesn't match what the extraction/consolidation prompts expect.
**Why it happens:** Using different date formats in different places.
**How to avoid:** The established format is `[Date: YYYY-MM-DD]` (matching the existing bench harness pattern). Use this consistently. The LoCoMo harness uses `Date: {date_str}` (no brackets) but the context decision specifies brackets.
**Warning signs:** Temporal annotations not appearing in consolidated observations.

### Pitfall 5: Forgetting to Return Bank ID for Resume
**What goes wrong:** Phase 3's resume support can't find bank IDs for completed instances.
**Why it happens:** `IngestResult` doesn't include the bank_id mapping, or it's lost.
**How to avoid:** `IngestResult` must contain `question_id` and `bank_id` as primary fields. Phase 3 will persist this to the ingest artifact.
**Warning signs:** `qa` subcommand can't find banks.

## Code Examples

### IngestConfig Struct Design
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestConfig {
    /// Session formatting: text (default) or json
    pub format: IngestFormat,
    /// Consolidation strategy: end (default), per-session, or off
    pub consolidation: ConsolidationMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IngestFormat {
    Text,
    Json,
}

impl Default for IngestFormat {
    fn default() -> Self { Self::Text }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConsolidationMode {
    End,
    PerSession,
    Off,
}

impl Default for ConsolidationMode {
    fn default() -> Self { Self::End }
}

impl ConsolidationMode {
    pub fn enabled(self) -> bool { !matches!(self, Self::Off) }
    pub fn per_session(self) -> bool { matches!(self, Self::PerSession) }
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            format: IngestFormat::default(),
            consolidation: ConsolidationMode::default(),
        }
    }
}
```

### IngestResult Struct Design
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResult {
    /// Which question this result is for.
    pub question_id: String,
    /// The bank created for this instance.
    pub bank_id: BankId,
    /// Per-stage LLM usage metrics.
    pub stage_metrics: BTreeMap<LlmStage, StageUsage>,
    /// Ingestion statistics.
    pub stats: IngestStats,
    /// Wall-clock timing.
    pub timing: IngestTiming,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestStats {
    pub sessions_ingested: usize,
    pub facts_stored: usize,
    pub entities_resolved: usize,
    pub links_created: usize,
    pub opinions_reinforced: usize,
    pub opinions_weakened: usize,
    pub session_failures: usize,
    /// Consolidation stats (if consolidation was run)
    pub observations_created: usize,
    pub observations_updated: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestTiming {
    pub ingest_time_s: f64,
    pub consolidation_time_s: f64,
    pub total_time_s: f64,
}
```

### Full ingest_instance Skeleton
```rust
pub async fn ingest_instance(
    instance: &LongMemEvalInstance,
    runtime: &ElephantRuntime,
    config: &IngestConfig,
) -> Result<IngestResult> {
    let metrics = Arc::new(MetricsCollector::new());
    let mut stats = IngestStats::default();

    // 1. Create bank
    let bank = MemoryBank {
        id: BankId::new(),
        name: format!("longmemeval-{}", instance.question_id),
        mission: "Long-term conversational memory benchmark".into(),
        directives: vec![],
        disposition: Disposition::default(),
        embedding_model: runtime.embeddings.model_name().to_string(),
        embedding_dimensions: runtime.embeddings.dimensions() as u16,
    };
    runtime.store.create_bank(&bank).await?;

    // 2. Ingest sessions sequentially
    let ingest_start = Instant::now();
    let total_sessions = instance.haystack_sessions.len();

    for (idx, (session, date_str)) in instance.haystack_sessions.iter()
        .zip(instance.haystack_dates.iter())
        .enumerate()
    {
        let content = match config.format {
            IngestFormat::Text => format_session_text(session, date_str),
            IngestFormat::Json => format_session_json(session),
        };
        let timestamp = parse_haystack_date(date_str);

        match runtime.retain.retain(&RetainInput {
            bank_id: bank.id,
            content,
            timestamp,
            turn_id: None,
            context: None,
            custom_instructions: None,
            speaker: None,
        }).await {
            Ok(resp) => {
                stats.sessions_ingested += 1;
                stats.facts_stored += resp.facts_stored;
                stats.entities_resolved += resp.entities_resolved;
                stats.links_created += resp.links_created;
                stats.opinions_reinforced += resp.opinions_reinforced;
                stats.opinions_weakened += resp.opinions_weakened;
            }
            Err(e) => {
                eprintln!("[{}] session {}/{} failed: {e}",
                    instance.question_id, idx + 1, total_sessions);
                stats.session_failures += 1;
            }
        }

        // Per-session consolidation
        if config.consolidation.per_session() {
            // consolidate after each session
            match runtime.consolidator.consolidate(bank.id).await {
                Ok(cr) => {
                    stats.observations_created += cr.observations_created;
                    stats.observations_updated += cr.observations_updated;
                }
                Err(e) => eprintln!("[{}] consolidation failed: {e}", instance.question_id),
            }
        }
    }
    let ingest_time_s = ingest_start.elapsed().as_secs_f64();

    // 3. End-of-ingestion consolidation
    let mut consolidation_time_s = 0.0;
    if config.consolidation.enabled() && !config.consolidation.per_session() {
        let t0 = Instant::now();
        match runtime.consolidator.consolidate(bank.id).await {
            Ok(cr) => {
                stats.observations_created += cr.observations_created;
                stats.observations_updated += cr.observations_updated;
            }
            Err(e) => eprintln!("[{}] consolidation failed: {e}", instance.question_id),
        }
        consolidation_time_s = t0.elapsed().as_secs_f64();
    }

    Ok(IngestResult {
        question_id: instance.question_id.clone(),
        bank_id: bank.id,
        stage_metrics: metrics.snapshot(),
        stats,
        timing: IngestTiming {
            ingest_time_s,
            consolidation_time_s,
            total_time_s: ingest_time_s + consolidation_time_s,
        },
    })
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| LoCoMo dynamic session keys (`session_1`, `session_1_dialogue`) | LongMemEval explicit arrays (`haystack_sessions`, `haystack_dates`) | Dataset design difference | Much simpler iteration -- just zip arrays |
| LoCoMo `"3:00 PM on 5 March, 2023"` date format | LongMemEval `"2023/05/20 (Sat) 02:21"` format | Dataset design difference | Need new parser, but format is more consistent |
| Turn struct with `speaker`, `text`, `blip_caption`, `dia_id` | Turn struct with `role`, `content` (plus `has_answer` to strip) | Dataset design difference | Simpler turns, but need to handle `has_answer` for raw JSON |

**Key difference from LoCoMo:** LongMemEval ingestion is simpler because the data is cleanly structured (arrays instead of dynamic JSON keys). No `get_session_turns()` / `session_count()` key-hunting needed.

## Open Questions

1. **Metrics scoping with `with_scoped_collector`**
   - What we know: LoCoMo uses `with_scoped_collector(conversation_metrics.clone(), ...)` to route LLM metrics to the right collector
   - What's unclear: Whether to use this same pattern or just pass `MetricsCollector` through `BuildRuntimeOptions`
   - Recommendation: Use `with_scoped_collector` for consistency with LoCoMo. It's already proven.

2. **Raw JSON context string**
   - What we know: Hindsight passes `context="Session {id} - you are the assistant in this conversation - happened on {date}"` alongside the JSON content
   - What's unclear: Whether to use `RetainInput.context` for this or embed it in the content
   - Recommendation: For Phase 2, keep `context: None` as decided. The date prefix in the content serves the temporal anchoring purpose. Hindsight's context string could be explored in a future ablation.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `#[tokio::test]` |
| Config file | Cargo.toml (existing) |
| Quick run command | `cargo test --lib -p elephant -- ingest` |
| Full suite command | `cargo test` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INGEST-01 | Bank created per instance | unit | `cargo test --test longmemeval_ingest -- bank_creation` | No -- Wave 0 |
| INGEST-02 | Session formatted with date prefix | unit | `cargo test -p elephant bench::longmemeval::ingest -- format_session` | No -- Wave 0 |
| INGEST-03 | Consolidation modes all work | unit | `cargo test -p elephant bench::longmemeval::ingest -- consolidation_mode` | No -- Wave 0 |
| INGEST-04 | Sessions ingested sequentially with timestamps | unit | `cargo test -p elephant bench::longmemeval::ingest -- sequential` | No -- Wave 0 |
| INGEST-05 | Pool sizing for concurrent ops | N/A | Deferred to Phase 5 | N/A |

### Sampling Rate
- **Per task commit:** `cargo test --lib -- longmemeval`
- **Per wave merge:** `cargo test`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `bench/longmemeval/ingest.rs` -- unit tests for format_session_text, format_session_json, parse_haystack_date, parse_date_prefix
- [ ] Tests for IngestConfig/IngestResult serialization roundtrip
- [ ] Tests for ConsolidationMode helper methods (enabled, per_session)

## Sources

### Primary (HIGH confidence)
- `bench/locomo/locomo.rs` -- LoCoMo ingestion pattern (bank creation, session loop, consolidation, metrics)
- `bench/longmemeval/dataset.rs` -- LongMemEvalInstance, Turn struct definitions
- `src/types/pipeline.rs` -- RetainInput, RetainOutput contracts
- `src/runtime.rs` -- ElephantRuntime struct, build_runtime_from_env
- `src/consolidation/observation.rs` -- Consolidator trait, consolidate_with_progress
- `src/types/bank.rs` -- MemoryBank struct, Disposition
- `src/storage/mod.rs` -- MemoryStore trait, create_bank
- `data/longmemeval_s_cleaned.json` -- actual dataset format verification

### Secondary (MEDIUM confidence)
- LongMemEval date format verified against actual dataset files (5 instances checked)
- `has_answer` field presence verified: 10,960 of 246,750 turns have it, but our Turn struct already drops it during deserialization

### Tertiary (LOW confidence)
- Hindsight's raw JSON approach with context string -- referenced from CONTEXT.md discussion notes, not directly verified against Hindsight source

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project, no new deps
- Architecture: HIGH -- directly follows proven LoCoMo pattern with simpler data format
- Pitfalls: HIGH -- verified against actual dataset, identified concrete format differences
- Date parsing: HIGH -- verified actual `haystack_dates` format against 5 dataset instances

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (stable -- internal project patterns, no external dependency changes)
