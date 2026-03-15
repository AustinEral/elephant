---
phase: 05-concurrency-resume-and-view-tool
plan: 01
subsystem: bench
tags: [tokio, semaphore, concurrency, postgres-pool, longmemeval]

# Dependency graph
requires:
  - phase: 04-evaluation-path
    provides: sequential per-instance loop in longmemeval.rs
provides:
  - Semaphore-gated concurrent instance processing via --instance-jobs N
  - Configurable Postgres pool sizing via BuildRuntimeOptions.max_pool_connections
  - SharedState incremental flush pattern for crash-resilient output
  - Per-instance error isolation (no process::exit in worker)
affects: [05-02, bench-runs]

# Tech tracking
tech-stack:
  added: [tokio::sync::Semaphore, tokio::sync::Mutex, PgPoolOptions]
  patterns: [SharedState incremental flush, semaphore-gated spawn loop, AtomicUsize progress counter]

key-files:
  created: []
  modified:
    - src/runtime.rs
    - bench/longmemeval/longmemeval.rs
    - bench/longmemeval/dataset.rs
    - bench/locomo/locomo.rs

key-decisions:
  - "Added Clone derive to LongMemEvalInstance for tokio::spawn ownership"
  - "Pool sizing formula: min(instance_jobs * 3, 50) matching plan spec"
  - "Judge client already Arc<dyn LlmClient> from build_judge_client -- no conversion needed"

patterns-established:
  - "SharedState pattern: Mutex-protected struct with push_and_flush/record_bank/flush methods for incremental crash-resilient output"
  - "Semaphore-gated spawn: clone Arcs, move owned data into spawned task, acquire permit inside"

requirements-completed: [CLI-05, CLI-10, INGEST-05]

# Metrics
duration: 4min
completed: 2026-03-15
---

# Phase 5 Plan 1: Concurrent Instance Processing Summary

**Semaphore-gated parallel instance processing with configurable Postgres pool sizing and incremental crash-resilient output**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-15T22:42:19Z
- **Completed:** 2026-03-15T22:46:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- `--instance-jobs N` controls concurrency via tokio::Semaphore (default 1 = sequential)
- Postgres pool sized to min(instance_jobs * 3, 50) via PgPoolOptions
- Individual instance failures logged but don't kill other concurrent instances
- JSONL sidecars and summary JSON written incrementally after each instance completes via SharedState
- QA mode reuses bank_ids from artifact under concurrency
- Progress output: `[N/total] question_id ok/err ingest Xs qa Ys`

## Task Commits

Each task was committed atomically:

1. **Task 1: Add max_pool_connections and SharedState infrastructure** - `c0ac02f` (feat)
2. **Task 2: Semaphore-gated concurrent instance execution** - `a04e657` (feat)

## Files Created/Modified
- `src/runtime.rs` - Added max_pool_connections to BuildRuntimeOptions, PgPoolOptions for configurable pool sizing
- `bench/longmemeval/longmemeval.rs` - SharedState struct with incremental flush, semaphore-gated concurrent loop replacing sequential iteration
- `bench/longmemeval/dataset.rs` - Added Clone derive to LongMemEvalInstance for spawned task ownership
- `bench/locomo/locomo.rs` - Updated BuildRuntimeOptions struct literal with new field

## Decisions Made
- Added Clone to LongMemEvalInstance rather than restructuring to avoid clones -- instances are small relative to session data, and this is the simplest approach for moving into spawned tasks
- Judge client (build_judge_client) already returns Arc<dyn LlmClient>, so no Box-to-Arc conversion was needed
- Pool default is 10 when max_pool_connections is None (server/other callers), scaled by instance_jobs only for bench

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added Clone derive to LongMemEvalInstance**
- **Found during:** Task 1
- **Issue:** LongMemEvalInstance lacked Clone, needed for moving into tokio::spawn
- **Fix:** Added `Clone` to derive macro on the struct
- **Files modified:** bench/longmemeval/dataset.rs
- **Verification:** cargo build passes, all 139 tests pass
- **Committed in:** c0ac02f (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Trivial derive addition required for correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Concurrency infrastructure complete, ready for full 500-instance benchmark runs
- Plan 05-02 can build on this for resume/view tool features

---
*Phase: 05-concurrency-resume-and-view-tool*
*Completed: 2026-03-15*
