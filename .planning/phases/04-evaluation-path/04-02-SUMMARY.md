---
phase: 04-evaluation-path
plan: 02
subsystem: benchmarking
tags: [longmemeval, reflect, judge, accuracy, evaluation]

requires:
  - phase: 04-evaluation-path/01
    provides: judge infrastructure, prompt templates, select/render/hash helpers
provides:
  - Real QA evaluation path (reflect -> judge -> score) in longmemeval-bench
  - compute_accuracy helper function
  - Accuracy and judge_model populated in summary artifact
affects: [05-execution-harness]

tech-stack:
  added: []
  patterns: [scoped-metrics-for-reflect, ingest-only-continue-pattern]

key-files:
  created: []
  modified: [bench/longmemeval/longmemeval.rs]

key-decisions:
  - "Factored accuracy into compute_accuracy() helper for testability"
  - "Judge client built lazily (only for Run/Qa, not Ingest)"
  - "Reflect errors count as wrong in denominator (no exclusions per EVAL-05)"

patterns-established:
  - "reflect -> judge -> score pipeline pattern for bench evaluation"

requirements-completed: [EVAL-03, EVAL-04, EVAL-05]

duration: 3min
completed: 2026-03-15
---

# Phase 4 Plan 2: QA Evaluation Path Summary

**Real reflect -> judge -> score pipeline replacing QA stub, with compute_accuracy helper and 5 accuracy unit tests**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-15T13:54:49Z
- **Completed:** 2026-03-15T13:58:11Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Replaced QA stub with full reflect -> judge -> score pipeline in per-instance loop
- Ingest-only mode preserved with `continue` early-exit pattern
- Judge client built conditionally (only for Run/Qa commands)
- Accuracy computed as correct/total with no exclusions (EVAL-05)
- Summary artifact now contains real judge_model, accuracy, and judge prompt hash
- Removed "NOTE: QA scoring not yet implemented" message, replaced with accuracy printout
- Added compute_accuracy() helper with 5 unit tests (all correct, mixed, all wrong, errors counted wrong, empty)

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire QA evaluation into per-instance loop** - `f8d9a62` (feat)

## Files Created/Modified
- `bench/longmemeval/longmemeval.rs` - Added QA evaluation pipeline (reflect -> judge -> score), compute_accuracy helper, 5 accuracy tests

## Decisions Made
- Factored accuracy computation into `compute_accuracy()` helper for direct unit testing
- Judge client built only when needed (Run/Qa commands), skipped for Ingest
- Reflect errors produce empty hypothesis, which skips judge call and counts as wrong (no exclusions from denominator)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- LongMemEval bench now produces real benchmark scores via `longmemeval-bench run`
- Ready for Phase 5 (execution harness) to orchestrate full benchmark runs

---
*Phase: 04-evaluation-path*
*Completed: 2026-03-15*
