# Phase 5: Concurrency, Resume, and View Tool - Research

**Researched:** 2026-03-15
**Domain:** Tokio concurrency, Postgres pool sizing, CLI view tooling
**Confidence:** HIGH

## Summary

Phase 5 adds three capabilities to the LongMemEval bench harness: (1) parallel instance processing via `--instance-jobs N` with a tokio semaphore, (2) resume support where `qa` reuses bank_ids from an ingest artifact, and (3) a standalone `longmemeval-view` binary for inspecting results. All three are well-understood patterns with direct precedent in the existing LoCoMo harness code.

The concurrency model follows LoCoMo's exact `Semaphore + tokio::spawn + Arc<Mutex<SharedState>>` pattern (locomo.rs:3424-3471). The pool sizing change is a single-line swap from `PgPool::connect()` to `PgPoolOptions::new().max_connections(N).connect()` in runtime.rs:257. The view tool is a new `[[bin]]` target that reads the existing LongMemEval artifact JSON format. Comparison mode (pulled forward from VIEW-04) adds delta columns per category.

**Primary recommendation:** Follow LoCoMo's SharedResults pattern exactly -- extract the sequential for-loop body into an async function, wrap shared state in Arc<Mutex<>>, spawn tasks gated by semaphore, collect handles.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- `--instance-jobs N` controls parallel instance processing via tokio semaphore (same pattern as LoCoMo's conversation_jobs)
- Default: 1 (sequential, opt into parallelism explicitly)
- Postgres pool auto-scales: `max_connections = instance_jobs * 3`, capped at a reasonable max (~50)
- JSONL sidecars written incrementally as each instance completes (mutex on file), preserving partial results on crash
- Progress output: simple atomic counters `[42/500] q_123 ✓ ingest 12.3s qa 2.1s` -- lines interleave but each is self-contained
- Resume is ingestion-level only -- `qa` reuses banks but always re-runs reflect + judge on all questions
- No question-level resume (no partial-state merge complexity)
- No `--resume` flag on `run` -- resume is strictly the ingest->qa two-step workflow
- Two-step: `longmemeval-bench ingest` -> `longmemeval-bench qa <artifact.json>`
- Minimal default output: config summary + per-category accuracy table (7 rows + overall) + stage metrics + total time
- Per-question table only with `--verbose` (loads from JSONL sidecar, not summary JSON)
- Stage metrics (token counts/costs per stage) shown in default view, not gated behind --verbose
- Basic two-file comparison supported in Phase 5 (pull forward from VIEW-04)
- Delta column per category when two artifacts provided: `longmemeval-view a.json b.json`
- Single-file mode for solo inspection

### Claude's Discretion
- Exact pool scaling formula and cap value
- Semaphore + tokio::spawn implementation details
- Mutex vs channel pattern for incremental JSONL writes
- View tool table formatting and column widths
- How to handle comparison when category counts differ between runs

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLI-05 | `--instance-jobs N` concurrency control for parallel instance processing | Semaphore pattern from LoCoMo (locomo.rs:3424-3471), SharedResults pattern for incremental writes, pool sizing via PgPoolOptions |
| CLI-10 | Resume via `qa` subcommand reusing bank_ids from ingest artifact | Already implemented: qa subcommand loads artifact, validates bank_ids (longmemeval.rs:1043-1118). Just needs to skip re-ingestion correctly under concurrency |
| INGEST-05 | Postgres connection pool explicitly sized for concurrent bank operations | PgPoolOptions::new().max_connections(N) replacing PgPool::connect() in runtime.rs:257 |
| VIEW-01 | Separate `longmemeval-view` binary (independent of LoCoMo view) | New `[[bin]]` target in Cargo.toml, reads LongMemEval BenchmarkOutput JSON format |
| VIEW-02 | Per-category accuracy display with question counts | 7 category rows + overall from per_category HashMap in BenchmarkOutput |
| VIEW-03 | Single-artifact view mode with config, summary, and question tables | Follows LoCoMo view.rs pattern: SingleConfigRow, SingleSummaryRow, StageRow types with tabled |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| tokio | 1.49.0 | Semaphore, Mutex, spawn, async runtime | Already in Cargo.toml with `full` features |
| sqlx | 0.8.6 | PgPoolOptions for pool sizing | Already in Cargo.toml |
| tabled | 0.20.0 | Table formatting for view tool | Already in Cargo.toml, used by LoCoMo view |
| serde/serde_json | 1.0.x | Artifact deserialization for view tool | Already in Cargo.toml |

### Supporting
No new dependencies needed. All required functionality exists in the current dependency tree.

## Architecture Patterns

### Pattern 1: Semaphore-Gated Concurrency (from LoCoMo)

**What:** Spawn all tasks immediately, gate execution with semaphore permits.
**When to use:** Parallel instance processing with bounded concurrency.
**Source:** `bench/locomo/locomo.rs:3424-3471`

```rust
let semaphore = Arc::new(Semaphore::new(config.instance_jobs));
let shared = Arc::new(Mutex::new(SharedState { /* ... */ }));
let mut handles = Vec::new();

for (idx, instance) in instances.into_iter().enumerate() {
    let sem = semaphore.clone();
    let runtime = runtime.clone();
    let shared = shared.clone();
    // move all needed data into the task

    handles.push(tokio::spawn(async move {
        let _permit = sem.acquire().await
            .map_err(|e| format!("semaphore closed: {e}"))?;
        process_instance(/* ... */).await
    }));
}

for handle in handles {
    match handle.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => eprintln!("instance failed: {e}"),
        Err(e) => eprintln!("instance task panicked: {e}"),
    }
}
```

**Key implementation detail:** The LoCoMo pattern does NOT abort on individual task failures -- it logs errors and continues. This preserves partial results. The LongMemEval harness currently calls `std::process::exit(1)` on ingest errors (line 1238). Under concurrency, this needs to become a per-instance error that gets recorded, not a process abort.

### Pattern 2: SharedResults with Mutex + Incremental Flush (from LoCoMo)

**What:** Arc<Mutex<SharedState>> holds accumulated results, flushes summary JSON after each instance completes.
**When to use:** Crash-resilient incremental output during concurrent execution.
**Source:** `bench/locomo/locomo.rs:2196-2293`

```rust
struct SharedState {
    results: Vec<QuestionResult>,
    banks: HashMap<String, String>,
    output_path: PathBuf,
    questions_path: PathBuf,
    debug_path: PathBuf,
    // ... other accumulated state
}

impl SharedState {
    fn push_and_flush(&mut self, result: QuestionResult, debug: QuestionDebugRecord) {
        append_jsonl(&self.questions_path, &result);
        append_jsonl(&self.debug_path, &debug);
        self.results.push(result);
        self.flush_summary();
    }

    fn flush_summary(&self) {
        // Rewrite full summary JSON with current accumulated state
    }
}
```

**Why Mutex over channels:** The flush pattern needs to read ALL accumulated results to rewrite the summary JSON. A channel-based approach would need a dedicated writer task that maintains state -- more complex with no benefit since the critical section (append + flush) is fast I/O, not compute-bound.

### Pattern 3: Pool Sizing for Concurrency

**What:** Replace `PgPool::connect()` with `PgPoolOptions::new().max_connections(N).connect()`.
**Where:** `src/runtime.rs:257`
**Source:** sqlx documentation

```rust
// Before
let pool = sqlx::PgPool::connect(&database_url).await?;

// After
use sqlx::postgres::PgPoolOptions;
let pool = PgPoolOptions::new()
    .max_connections(max_connections)
    .connect(&database_url)
    .await?;
```

**Recommended formula:** `min(instance_jobs * 3, 50)` where the multiplier accounts for concurrent retain + consolidate + reflect operations per instance. Cap at 50 to stay within Postgres default `max_connections = 100` (leaving room for other clients). The `BuildRuntimeOptions` struct needs a new `max_pool_connections: Option<u32>` field.

### Pattern 4: View Tool Binary Structure

**What:** Standalone binary reading LongMemEval BenchmarkOutput JSON.
**Where:** `bench/longmemeval/view.rs` as new file, `[[bin]]` in Cargo.toml.
**Source:** LoCoMo view.rs pattern

The view tool needs its OWN deserialization types with `#[serde(default)]` on all fields (reader resilience), NOT the writer types from longmemeval.rs. This matches the LoCoMo convention where view.rs has separate struct definitions from locomo.rs.

Key difference from LoCoMo view: LongMemEval has no `per_conversation` data -- it has `per_category` as a HashMap<String, CategoryResult> directly in the summary JSON. Questions come from the JSONL sidecar.

### Recommended Project Structure
```
bench/
  longmemeval/
    longmemeval.rs      # (existing) harness binary -- add concurrency
    view.rs             # (NEW) longmemeval-view binary
    dataset.rs          # (existing) unchanged
    ingest.rs           # (existing) unchanged
  common/
    io.rs               # (existing) append_jsonl, sidecar_path -- already shared
src/
  runtime.rs            # (modify) PgPoolOptions with configurable max_connections
Cargo.toml              # (modify) add [[bin]] for longmemeval-view
```

### Anti-Patterns to Avoid
- **process::exit() under concurrency:** The current harness calls `std::process::exit(1)` on ingest failure (line 1238). Under tokio::spawn, this kills the entire process including all other tasks. Convert to per-instance error handling that records the error and continues.
- **Shared mutable state without Mutex:** `append_jsonl` does `OpenOptions::append()` which is safe for file-level atomicity on individual lines, but the summary JSON rewrite needs Mutex protection to avoid interleaved writes.
- **Unbounded pool connections:** Don't set pool to `instance_jobs * N` without a cap. Postgres has limits and each connection consumes memory.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bounded concurrency | Custom task queue or thread pool | `tokio::sync::Semaphore` | Proven pattern, already used in LoCoMo |
| Thread-safe shared state | Raw Atomic operations for complex state | `Arc<Mutex<SharedState>>` | Semaphore already serializes heavy work; Mutex critical section is just I/O flush |
| Table formatting | println!-based manual alignment | `tabled` crate with `Tabled` derive | Already used by LoCoMo view, handles alignment, styling, ANSI |
| JSONL file writes | Custom file locking | `append_jsonl()` from `bench/common/io.rs` | Already shared between harnesses |
| Delta formatting | Manual ANSI color logic | `fmt_pct_delta()`, `fmt_cost_delta_u64()` from view.rs | Copy the helper functions from LoCoMo view |

## Common Pitfalls

### Pitfall 1: process::exit kills concurrent tasks
**What goes wrong:** `std::process::exit(1)` on line 1238 of longmemeval.rs will terminate the entire process when running under concurrency, losing partial results from other concurrent instances.
**Why it happens:** The sequential loop was designed with fail-fast semantics. Under concurrency, fail-fast means killing N-1 other in-flight tasks.
**How to avoid:** Replace `process::exit(1)` in the per-instance worker with `Err(...)` return. The spawned task logs the error and records a failed result. The main loop continues.
**Warning signs:** Any `process::exit()` or `panic!()` call inside a `tokio::spawn` closure.

### Pitfall 2: Sidecar truncation races
**What goes wrong:** The current code truncates sidecar files at startup (longmemeval.rs:1169-1170). Under concurrency this is fine -- it happens before any tasks spawn. But if someone tries to add "resume within a run," re-truncation would lose data.
**Why it happens:** Truncation is a one-time setup step, but its placement matters.
**How to avoid:** Keep truncation in the main function before spawning any tasks, exactly as LoCoMo does.
**Warning signs:** Truncation calls inside per-instance worker functions.

### Pitfall 3: MetricsCollector is global, not per-instance
**What goes wrong:** `with_scoped_collector` uses thread-local state. Under concurrency, if two instances run on the same tokio worker thread, their metrics could bleed together.
**Why it happens:** The scoped collector pattern was designed for sequential execution.
**How to avoid:** Check how LoCoMo handles this -- it uses `with_scoped_collector` inside `tokio::spawn` closures and it works because the scope is per-future, not per-thread. The `metrics.clone()` pattern clones the Arc, not the collector. Each `with_scoped_collector` call creates its own scope. This should work correctly under concurrency.
**Warning signs:** Stage metrics totals not adding up when running with instance_jobs > 1.

### Pitfall 4: Summary JSON race on flush
**What goes wrong:** Without a Mutex, two concurrent tasks could both call `flush_summary()` simultaneously, leading to a corrupted or partial summary JSON file.
**Why it happens:** `fs::write()` is not atomic -- it truncates then writes. Two concurrent writes can interleave.
**How to avoid:** All mutations to shared state AND the summary flush must happen inside the same Mutex lock. LoCoMo's `SharedResults.push_and_flush()` method does this correctly because it's called via `shared.lock().await.push_and_flush(...)`.
**Warning signs:** Summary JSON with missing fields or truncated content.

### Pitfall 5: Move semantics with tokio::spawn
**What goes wrong:** `tokio::spawn` requires `'static` futures. References to loop variables don't work -- everything must be moved or cloned.
**Why it happens:** Rust's borrow checker enforces that spawned tasks outlive the loop iteration.
**How to avoid:** Clone Arc-wrapped values, move owned data. The LoCoMo pattern shows exactly what needs cloning: `sem.clone()`, `runtime.clone()`, `shared.clone()`. Instance data should be moved by value (owned, not borrowed).
**Warning signs:** Compiler errors about lifetimes and `'static` bounds.

### Pitfall 6: View tool reading wrong field names
**What goes wrong:** LongMemEval artifact format differs from LoCoMo. Fields like `per_category` (LongMemEval) vs `per_conversation` (LoCoMo), `question_id` as the instance key vs `sample_id`.
**Why it happens:** Copy-paste from LoCoMo view without adapting to LongMemEval schema.
**How to avoid:** Define view tool types from the LongMemEval BenchmarkOutput schema, not from LoCoMo. Use `#[serde(default)]` on all fields for forward/backward compatibility.
**Warning signs:** Empty tables or missing data in view output.

## Code Examples

### Converting Sequential Loop to Concurrent (LongMemEval-specific)

The current sequential loop at longmemeval.rs:1219-1381 needs restructuring:

```rust
// 1. Build shared state (before spawning)
let shared = Arc::new(Mutex::new(SharedState {
    results: Vec::new(),
    banks: existing_banks,
    output_path: output_path.clone(),
    questions_path: questions_path.clone(),
    debug_path: debug_path.clone(),
    // ... carry forward all state needed for summary flush
}));

// 2. Spawn instances with semaphore
let semaphore = Arc::new(Semaphore::new(config.instance_jobs));
let mut handles = Vec::new();
let total_instances = instances.len();

for instance in instances {
    let sem = semaphore.clone();
    let runtime = runtime.clone();
    let judge = judge.clone();
    let metrics = metrics.clone();
    let shared = shared.clone();
    let command = command;

    handles.push(tokio::spawn(async move {
        let _permit = sem.acquire().await.map_err(|e| format!("{e}"))?;
        // ... process instance (ingest + qa)
        // ... push result to shared.lock().await
        Ok::<(), String>(())
    }));
}

// 3. Await all handles
for handle in handles {
    match handle.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => eprintln!("instance failed: {e}"),
        Err(e) => eprintln!("task panicked: {e}"),
    }
}
```

### Pool Sizing in BuildRuntimeOptions

```rust
// runtime.rs
pub struct BuildRuntimeOptions {
    pub metrics: Option<Arc<MetricsCollector>>,
    pub max_pool_connections: Option<u32>,
}

// In build_runtime_from_env:
let max_conns = options.max_pool_connections.unwrap_or(10);
let pool = PgPoolOptions::new()
    .max_connections(max_conns)
    .connect(&database_url)
    .await?;
```

### LongMemEval View Tool - Config Table

```rust
// bench/longmemeval/view.rs
use tabled::{Table, Tabled};
use tabled::settings::style::Style;
use tabled::settings::{Alignment, Modify};
use tabled::settings::object::Columns;

#[derive(Tabled)]
struct ConfigRow {
    #[tabled(rename = "config")]
    key: String,
    value: String,
}

#[derive(Tabled)]
struct CategoryRow {
    category: String,
    #[tabled(rename = "acc")]
    accuracy: String,
    #[tabled(rename = "n")]
    count: usize,
}
```

### LongMemEval View Tool - Comparison Mode

```rust
#[derive(Tabled)]
struct ComparisonCategoryRow {
    category: String,
    #[tabled(rename = "A")]
    acc_a: String,
    #[tabled(rename = "B")]
    acc_b: String,
    #[tabled(rename = "delta")]
    delta: String,
    #[tabled(rename = "n(A)")]
    n_a: usize,
    #[tabled(rename = "n(B)")]
    n_b: usize,
}
```

When category counts differ between runs (e.g., partial run A vs full run B), show both n values. Delta is computed from accuracies, which are independent of count.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Sequential per-instance loop | Semaphore-bounded concurrency | This phase | Full 500-instance runs go from ~N*T to ~N*T/jobs wallclock |
| PgPool::connect() (default ~10 conns) | PgPoolOptions with explicit sizing | This phase | Prevents pool exhaustion under concurrency |
| No view tool for LongMemEval | Standalone longmemeval-view binary | This phase | Can inspect results without re-running bench |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `#[tokio::test]` |
| Config file | None (Cargo convention) |
| Quick run command | `cargo test --bin longmemeval-bench -- --nocapture` |
| Full suite command | `cargo test` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLI-05 | `--instance-jobs` CLI parsing | unit | `cargo test --bin longmemeval-bench parse_instance_jobs -- --nocapture` | Partially (parse tests exist at line 1486+) |
| CLI-10 | QA mode reuses bank_ids from artifact | unit | `cargo test --bin longmemeval-bench qa_ -- --nocapture` | Partially (QA parse tests exist) |
| INGEST-05 | Pool auto-scales with instance_jobs | unit | `cargo test --lib -- runtime --nocapture` | Wave 0 |
| VIEW-01 | longmemeval-view binary compiles and runs | smoke | `cargo build --bin longmemeval-view` | Wave 0 |
| VIEW-02 | Per-category accuracy display | unit | `cargo test --bin longmemeval-view -- --nocapture` | Wave 0 |
| VIEW-03 | Single-artifact view mode | unit | `cargo test --bin longmemeval-view -- --nocapture` | Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --bin longmemeval-bench --bin longmemeval-view -- --nocapture`
- **Per wave merge:** `cargo test`
- **Phase gate:** Full suite green before verify

### Wave 0 Gaps
- [ ] `bench/longmemeval/view.rs` -- new binary source file
- [ ] Tests for pool sizing formula in runtime or longmemeval harness
- [ ] Tests for view tool output (can test data formatting functions in isolation)

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `bench/locomo/locomo.rs:3424-3471` (semaphore pattern)
- Direct code inspection: `bench/locomo/locomo.rs:2196-2293` (SharedResults pattern)
- Direct code inspection: `bench/view.rs` (LoCoMo view tool -- 1755 lines of reference)
- Direct code inspection: `bench/longmemeval/longmemeval.rs` (current sequential harness -- 2712 lines)
- Direct code inspection: `src/runtime.rs:257` (pool creation)
- Direct code inspection: `bench/common/io.rs` (shared JSONL utilities)

### Secondary (MEDIUM confidence)
- sqlx PgPoolOptions API: `max_connections()` method (from training data, matches codebase pitfalls doc at `.planning/research/PITFALLS.md:58`)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, no new dependencies
- Architecture: HIGH -- direct LoCoMo precedent for all three features, ~2000 lines of reference code
- Pitfalls: HIGH -- identified from concrete code inspection of both harnesses
- View tool: HIGH -- LoCoMo view.rs provides complete template (1755 lines)

**Research date:** 2026-03-15
**Valid until:** indefinite (internal codebase patterns, no external API dependencies)
