---
phase: 05-concurrency-resume-and-view-tool
plan: 02
subsystem: benchmarking
tags: [longmemeval, view-tool, cli, tabled, serde]

requires:
  - phase: 03-cli-artifact-infrastructure
    provides: common bench IO (sidecar_path), tabled rendering patterns
  - phase: 04-evaluation-path
    provides: LongMemEval BenchmarkOutput schema, QuestionResult JSONL sidecar
provides:
  - longmemeval-view binary for inspecting benchmark results
  - single-file and two-file comparison display modes
  - verbose per-question table from JSONL sidecar
affects: []

tech-stack:
  added: []
  patterns:
    - "View-local deserialization types with #[serde(default)] on ALL fields for reader resilience"
    - "BTreeMap<String, ViewStageUsage> for stage_metrics (string keys, not enum -- forward/backward compat)"

key-files:
  created:
    - bench/longmemeval/view.rs
  modified:
    - Cargo.toml

key-decisions:
  - "Used String keys for stage_metrics instead of LlmStage enum (avoids elephant::metrics dependency, handles forward compat)"
  - "ViewPromptHashes and ViewRuntimeConfig use #[serde(flatten)] with BTreeMap for unknown fields"

patterns-established:
  - "LongMemEval view tool mirrors LoCoMo view.rs patterns: #[path] common module, tabled tables, ANSI-colored deltas"

requirements-completed: [VIEW-01, VIEW-02, VIEW-03]

duration: 3min
completed: 2026-03-15
---

# Phase 5 Plan 2: LongMemEval View Tool Summary

**Standalone longmemeval-view binary with single-file config/accuracy/stage display, two-file comparison with delta columns, and verbose per-question table from JSONL sidecar**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-15T22:41:49Z
- **Completed:** 2026-03-15T22:44:43Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- longmemeval-view binary compiles and runs as standalone tool
- Single-file mode: config summary table, per-category accuracy table (sorted alphabetically + overall row), stage metrics table, total time
- Comparison mode: side-by-side config, per-category accuracy with colored delta column, stage metrics with token delta
- Verbose mode: per-question table from JSONL sidecar via sidecar_path; comparison mode shows A/B correct marks
- 23 unit tests covering arg parsing, formatting, delta computation, serialization resilience

## Task Commits

Each task was committed atomically:

1. **Task 1: Create longmemeval-view binary with single-file and comparison modes** - `4edf94f` (feat)

**Plan metadata:** (pending)

## Files Created/Modified
- `bench/longmemeval/view.rs` - Standalone longmemeval-view binary (1053 lines)
- `Cargo.toml` - Added [[bin]] target for longmemeval-view

## Decisions Made
- Used String keys for stage_metrics BTreeMap instead of LlmStage enum -- avoids depending on elephant::metrics in the view binary and handles forward/backward compatibility when new stages are added
- ViewPromptHashes and ViewRuntimeConfig use `#[serde(flatten)]` with BTreeMap to absorb unknown fields gracefully
- Followed LoCoMo view.rs patterns: `#[path = "../common/mod.rs"]` module inclusion, tabled crate rendering, ANSI color codes for positive/negative deltas

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ambiguous float type in test**
- **Found during:** Task 1 (unit tests)
- **Issue:** `(0.90 - 0.85).abs()` is ambiguous numeric type in Rust
- **Fix:** Added explicit `f64` type annotation
- **Files modified:** bench/longmemeval/view.rs
- **Verification:** cargo test passes
- **Committed in:** 4edf94f (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Trivial type annotation fix. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- View tool ready for use with any LongMemEval benchmark output JSON
- Comparison mode enables rapid A/B analysis of benchmark runs

## Self-Check: PASSED

- FOUND: bench/longmemeval/view.rs
- FOUND: longmemeval-view in Cargo.toml
- FOUND: commit 4edf94f
- PASS: 1053 lines (>= 200 min)

---
*Phase: 05-concurrency-resume-and-view-tool*
*Completed: 2026-03-15*
