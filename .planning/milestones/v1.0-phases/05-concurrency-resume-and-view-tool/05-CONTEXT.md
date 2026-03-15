# Phase 5: Concurrency, Resume, and View Tool - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Parallel instance execution via `--instance-jobs N`, resume support through ingest→qa two-step workflow, and a standalone `longmemeval-view` binary for inspecting results. Requirements: CLI-05, CLI-10, INGEST-05, VIEW-01, VIEW-02, VIEW-03.

</domain>

<decisions>
## Implementation Decisions

### Concurrency model
- `--instance-jobs N` controls parallel instance processing via tokio semaphore (same pattern as LoCoMo's conversation_jobs)
- Default: 1 (sequential, opt into parallelism explicitly)
- Postgres pool auto-scales: `max_connections = instance_jobs * 3`, capped at a reasonable max (~50)
- JSONL sidecars written incrementally as each instance completes (mutex on file), preserving partial results on crash
- Progress output: simple atomic counters `[42/500] q_123 ✓ ingest 12.3s qa 2.1s` — lines interleave but each is self-contained

### Resume behavior
- Resume is ingestion-level only — `qa` reuses banks but always re-runs reflect + judge on all questions
- No question-level resume (no partial-state merge complexity)
- No `--resume` flag on `run` — resume is strictly the ingest→qa two-step workflow
- Two-step: `longmemeval-bench ingest` → `longmemeval-bench qa <artifact.json>`

### View tool — default display
- Minimal default output: config summary + per-category accuracy table (7 rows + overall) + stage metrics + total time
- Per-question table only with `--verbose` (loads from JSONL sidecar, not summary JSON)
- Stage metrics (token counts/costs per stage) shown in default view, not gated behind --verbose

### View tool — comparison
- Basic two-file comparison supported in Phase 5 (pull forward from VIEW-04)
- Delta column per category when two artifacts provided: `longmemeval-view a.json b.json`
- Single-file mode for solo inspection

### Claude's Discretion
- Exact pool scaling formula and cap value
- Semaphore + tokio::spawn implementation details
- Mutex vs channel pattern for incremental JSONL writes
- View tool table formatting and column widths
- How to handle comparison when category counts differ between runs

</decisions>

<specifics>
## Specific Ideas

- LoCoMo's Semaphore pattern at `bench/locomo/locomo.rs:3424-3471` is the direct reference for instance-level parallelism
- Progress counter format: `[N/total] question_id ✓/✗ ingest Xs qa Ys`
- View tool preview mockup confirmed by user — config block + accuracy table + stage metrics + time

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bench/locomo/locomo.rs:3424-3471`: Semaphore + tokio::spawn concurrency pattern for conversations — adapt for instances
- `bench/view.rs`: LoCoMo view tool with comparison mode, table formatting, ConfigRow/SummaryRow/MetricsRow types — reference for longmemeval-view
- `bench/common/io.rs`: `append_jsonl()`, `sidecar_path()` — already shared
- `bench/longmemeval/longmemeval.rs:1043-1118`: Existing artifact loading and bank_id validation for qa mode

### Established Patterns
- Pool: `sqlx::PgPool::connect()` with default settings (currently ~5 connections) — needs explicit `PgPoolOptions::max_connections()`
- LoCoMo view: single-file and comparison modes with delta coloring, per-conversation and per-question tables
- Incremental JSONL flush pattern already used in both harnesses

### Integration Points
- `bench/longmemeval/longmemeval.rs`: Main loop (lines 1212+) needs parallelization with semaphore
- `src/runtime.rs:257`: Pool creation needs `PgPoolOptions` with configurable max_connections
- `Cargo.toml`: New `[[bin]]` target for `longmemeval-view`
- `bench/longmemeval/longmemeval.rs`: `instance_jobs` field already in config, currently unused

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-concurrency-resume-and-view-tool*
*Context gathered: 2026-03-15*
