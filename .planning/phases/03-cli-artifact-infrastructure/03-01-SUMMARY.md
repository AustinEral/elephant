---
phase: 03-cli-artifact-infrastructure
plan: 01
subsystem: bench
tags: [cli, longmemeval, config, profiles]

requires:
  - phase: 02-ingestion-pipeline
    provides: IngestFormat, ConsolidationMode enums and ingest_instance function
provides:
  - BenchCommand enum (Run, Ingest, Qa)
  - RunProfile enum (Smoke, FullS, FullM) with config_path()
  - RunConfig, FileRunConfig, CliOverrides, ParsedCli structs
  - parse_args_from with full flag parsing
  - resolve_fresh_config (3-layer: profile -> config JSON -> CLI)
  - resolve_qa_config with validate_qa_overrides
  - default_output_path resolution
  - Three profile JSON files (smoke, full-s, full-m)
affects: [03-02, pipeline-orchestration, qa-evaluation]

tech-stack:
  added: []
  patterns: [manual-arg-parsing, three-layer-config-resolution, profile-json-system]

key-files:
  created:
    - bench/longmemeval/profiles/smoke.json
    - bench/longmemeval/profiles/full-s.json
    - bench/longmemeval/profiles/full-m.json
    - bench/longmemeval/results/local/.gitkeep
  modified:
    - bench/longmemeval/longmemeval.rs
    - bench/longmemeval/ingest.rs

key-decisions:
  - "Added FromStr and as_str to IngestFormat and ConsolidationMode in ingest.rs for CLI parsing"
  - "QA resolve_qa_config starts with default RunConfig (artifact loading deferred to pipeline wiring)"
  - "instance_limit field added to FileRunConfig and RunConfig matching LoCoMo question_limit pattern"

patterns-established:
  - "Manual arg parsing with parse_cli_overrides -> parse_args_from -> resolve_*_config chain"
  - "Profile JSON in bench/longmemeval/profiles/*.json loaded by RunProfile.config_path()"
  - "Three-layer config: profile defaults -> --config JSON overlay -> CLI flag overrides"

requirements-completed: [CLI-01, CLI-02, CLI-03, CLI-04, CLI-06, CLI-07, CLI-11]

duration: 3min
completed: 2026-03-15
---

# Phase 3 Plan 1: CLI and Config Foundation Summary

**Manual CLI parser with 3 subcommands, 3-layer config resolution, profile JSON system, and 62 unit tests for the longmemeval-bench binary**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-15T13:03:12Z
- **Completed:** 2026-03-15T13:06:38Z
- **Tasks:** 1
- **Files modified:** 6

## Accomplishments
- Full CLI type system: BenchCommand, RunProfile, RunConfig, FileRunConfig, CliOverrides, ParsedCli
- parse_args_from handles all subcommands (run/ingest/qa) and 14 flags with error handling
- Three-layer config resolution: profile JSON -> config overlay -> CLI flags
- QA override validation rejects 7 ingestion-related flags, allows 6 qa-only flags
- Three profile JSON files with correct defaults (smoke: instance_limit=1, full-s: S dataset, full-m: M dataset)
- 62 new unit tests covering parsing, validation, config resolution, output paths, and profile loading

## Task Commits

1. **Task 1: CLI types, parsing, config resolution, and profile system** - `1932953` (feat)

## Files Created/Modified
- `bench/longmemeval/longmemeval.rs` - CLI types, parsing, config resolution, main stub, 62 unit tests
- `bench/longmemeval/ingest.rs` - Added FromStr and as_str impls for IngestFormat and ConsolidationMode
- `bench/longmemeval/profiles/smoke.json` - Smoke profile (instance_limit=1, S dataset)
- `bench/longmemeval/profiles/full-s.json` - Full-S profile (S dataset, no instance limit)
- `bench/longmemeval/profiles/full-m.json` - Full-M profile (M dataset)
- `bench/longmemeval/results/local/.gitkeep` - Results directory placeholder

## Decisions Made
- Added FromStr and as_str directly to IngestFormat and ConsolidationMode in ingest.rs rather than parsing manually in the CLI -- cleaner and consistent with LoCoMo patterns
- resolve_qa_config uses RunConfig::default() as base since BenchmarkOutput type doesn't exist yet (Plan 02 will add artifact loading)
- main() prints config summary then "pipeline not yet wired (Plan 02)" -- valid parse-and-print run, no exit(1)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added FromStr/as_str to ingest.rs enums**
- **Found during:** Task 1
- **Issue:** Plan noted IngestFormat and ConsolidationMode lack FromStr/as_str needed for CLI parsing
- **Fix:** Added FromStr and as_str impls to both enums in ingest.rs
- **Files modified:** bench/longmemeval/ingest.rs
- **Verification:** Unit tests for FromStr and as_str pass
- **Committed in:** 1932953

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Plan explicitly called out this addition as needed. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CLI foundation complete, ready for Plan 02 pipeline orchestration
- All types and config resolution in place for wiring ingest/consolidate/qa pipeline
- Profile system tested and working

---
*Phase: 03-cli-artifact-infrastructure*
*Completed: 2026-03-15*
