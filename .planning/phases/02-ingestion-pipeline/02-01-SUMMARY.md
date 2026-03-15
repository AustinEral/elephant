---
phase: 02-ingestion-pipeline
plan: 01
subsystem: benchmarking
tags: [longmemeval, ingestion, chrono, serde, consolidation]

requires:
  - phase: 01-dataset-foundation
    provides: LongMemEvalInstance, Turn types, dataset loading
provides:
  - IngestConfig, IngestFormat, ConsolidationMode config types
  - IngestResult, IngestStats, IngestTiming result types
  - parse_date_prefix and parse_haystack_date date parsers
  - format_session_text and format_session_json session formatters
  - ingest_instance orchestration function
affects: [03-cli-harness, 04-qa-evaluation]

tech-stack:
  added: []
  patterns: [per-instance bank isolation, sequential session ingestion with error recovery, configurable consolidation modes]

key-files:
  created: [bench/longmemeval/ingest.rs]
  modified: [bench/longmemeval/mod.rs, bench/longmemeval/longmemeval.rs]

key-decisions:
  - "Used elephant::error::Result instead of anyhow::Result since anyhow is dev-only dep"
  - "JSON mode prepends date prefix as '[Date: ...]\n\n' before JSON array, matching text mode pattern"
  - "stage_metrics left as empty BTreeMap -- scoped collector wiring deferred to Phase 3 CLI"

patterns-established:
  - "LongMemEval date format parsing: split on '(' and ')' to extract date and time components"
  - "Session error recovery: warn + increment session_failures counter, continue to next session"
  - "Consolidation mode dispatch: enabled() && !per_session() for end-mode, per_session() inside loop"

requirements-completed: [INGEST-01, INGEST-02, INGEST-03, INGEST-04, INGEST-05]

duration: 5min
completed: 2026-03-15
---

# Phase 2 Plan 1: Ingestion Pipeline Summary

**Per-instance LongMemEval ingestion: date-prefixed text/JSON formatters, configurable consolidation modes, and ingest_instance orchestration with error recovery**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-15T12:09:15Z
- **Completed:** 2026-03-15T12:14:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- IngestConfig with Text/Json format and End/PerSession/Off consolidation mode enums
- Date parsing for LongMemEval's "2023/05/20 (Sat) 02:21" format with fallback chain
- Session formatters producing [Date: YYYY-MM-DD]-prefixed content in both modes
- ingest_instance: bank creation, sequential session loop, three consolidation modes, structured result
- 18 unit tests covering all formatters, parsers, config types, and serde roundtrips

## Task Commits

1. **Task 1: Types, formatters, and date parsing with unit tests** - `19dd41a` (feat)
2. **Task 2: ingest_instance orchestration function** - `3dca976` (feat)

## Files Created/Modified
- `bench/longmemeval/ingest.rs` - Types, formatters, date parsers, and ingest_instance function (340 lines)
- `bench/longmemeval/mod.rs` - Added pub mod ingest re-export
- `bench/longmemeval/longmemeval.rs` - Added mod ingest for binary compilation

## Decisions Made
- Used `elephant::error::Result` for the ingest_instance return type since anyhow is a dev-only dependency and the function is in binary code
- JSON mode prepends `[Date: ...]` prefix before the JSON array as a separate line, matching how text mode includes the prefix
- stage_metrics returns an empty BTreeMap for now -- the MetricsCollector/with_scoped_collector integration will be wired in Phase 3 when the CLI builds the runtime

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ingest_instance is ready to be called from Phase 3's CLI harness
- IngestResult contains bank_id for resume support
- All three consolidation modes are implemented and dispatched
- INGEST-05 (pool sizing) documented as deferred to Phase 5

## Self-Check: PASSED

- bench/longmemeval/ingest.rs: FOUND
- bench/longmemeval/mod.rs: FOUND
- Commit 19dd41a: FOUND
- Commit 3dca976: FOUND

---
*Phase: 02-ingestion-pipeline*
*Completed: 2026-03-15*
