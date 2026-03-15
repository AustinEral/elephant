---
phase: 01-dataset-foundation
plan: 01
subsystem: testing
tags: [serde, fnv, benchmark, longmemeval, jsonl]

# Dependency graph
requires: []
provides:
  - "bench/common/ shared infrastructure (FNV fingerprinting, JSONL I/O, sidecar paths)"
  - "LongMemEval dataset types (QuestionType, Turn, LongMemEvalInstance)"
  - "Dataset loading with validation and fingerprinting"
  - "longmemeval-bench binary target"
affects: [01-dataset-foundation, 02-ingestion-pipeline, 03-harness-scaffold, 04-evaluation-judge]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "#[path] mod import for sharing code between binary targets"
    - "Two-layer validation: serde parse then semantic collect-all pass"
    - "answer_to_string coercion for mixed-type JSON fields"
    - "FNV1a-64 fingerprinting for dataset reproducibility"

key-files:
  created:
    - bench/common/mod.rs
    - bench/common/fingerprint.rs
    - bench/common/io.rs
    - bench/longmemeval/mod.rs
    - bench/longmemeval/dataset.rs
    - bench/longmemeval/longmemeval.rs
  modified:
    - bench/locomo/locomo.rs
    - Cargo.toml

key-decisions:
  - "Delegating wrapper functions in locomo.rs instead of direct common:: calls to minimize diff"
  - "6 QuestionType variants with is_abstention() helper matching upstream _abs suffix detection"

patterns-established:
  - "#[path = '../common/mod.rs'] mod common for sharing code between bench binaries"
  - "Semantic validation collects all errors before reporting (not fail-fast)"

requirements-completed: [DATA-03, DATA-04, DATA-05, DATA-06, DATA-07]

# Metrics
duration: 3min
completed: 2026-03-15
---

# Phase 1 Plan 1: Dataset Foundation Summary

**Shared bench/common/ infrastructure (FNV, JSONL I/O) extracted from LoCoMo, LongMemEval dataset types with 6-variant QuestionType enum, two-layer validation, and longmemeval-bench binary target**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-15T10:38:29Z
- **Completed:** 2026-03-15T10:42:14Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Extracted fnv1a64, fnv1a64_hex, sidecar_path, append_jsonl to bench/common/ with full test coverage
- LoCoMo bench compiles and all 29 tests pass after extraction (zero breakage)
- LongMemEval dataset types: QuestionType (6 variants), Turn, LongMemEvalInstance with mixed-type answer coercion
- Semantic validation catches session/date/id length mismatches and invalid answer types, collecting all errors
- load_dataset provides helpful HuggingFace download instructions when file is missing
- 56 total tests passing (29 locomo-bench + 27 longmemeval-bench)

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract shared bench infrastructure to bench/common/** - `ba37559` (feat)
2. **Task 2: Create LongMemEval dataset types, loading, validation, and binary target** - `30d3bd6` (feat)

## Files Created/Modified
- `bench/common/mod.rs` - Re-exports for shared bench utilities
- `bench/common/fingerprint.rs` - FNV1a-64 hashing with 5 unit tests
- `bench/common/io.rs` - JSONL append and sidecar path helpers with 3 unit tests
- `bench/longmemeval/mod.rs` - Module re-export
- `bench/longmemeval/dataset.rs` - LongMemEval types, loading, validation with 19 unit tests
- `bench/longmemeval/longmemeval.rs` - Binary entry point (stub for Phase 3)
- `bench/locomo/locomo.rs` - Updated to import from bench/common/ via #[path]
- `Cargo.toml` - Added longmemeval-bench binary target

## Decisions Made
- Used delegating wrapper functions in locomo.rs (e.g., `fn fnv1a64(data: &[u8]) -> u64 { common::fnv1a64(data) }`) instead of replacing all call sites with `common::` prefix. This minimizes the diff and avoids touching every call site in a 3700-line file.
- Kept answer_to_string in both locomo.rs and dataset.rs as separate copies per plan instruction (LoCoMo's version is LoCoMo-specific until Phase 4 judge work unifies them).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- bench/common/ is ready for any future bench binary to import via `#[path]`
- LongMemEval dataset types are ready for ingestion pipeline (Phase 2)
- longmemeval-bench binary compiles and is ready for harness implementation (Phase 3)
- Dataset files must be downloaded from HuggingFace before integration testing

## Self-Check: PASSED

All 6 created files verified on disk. Both task commits (ba37559, 30d3bd6) verified in git log.

---
*Phase: 01-dataset-foundation*
*Completed: 2026-03-15*
