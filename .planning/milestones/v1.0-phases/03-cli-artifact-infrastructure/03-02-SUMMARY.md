---
phase: 03-cli-artifact-infrastructure
plan: 02
subsystem: bench
tags: [longmemeval, artifacts, pipeline, orchestration, manifest]

requires:
  - phase: 03-cli-artifact-infrastructure
    plan: 01
    provides: BenchCommand, RunProfile, RunConfig, parse_args, resolve_fresh_config, resolve_qa_config
  - phase: 02-ingestion-pipeline
    provides: ingest_instance, IngestConfig, IngestResult
provides:
  - BenchmarkOutput, BenchmarkManifest, CategoryResult, QuestionResult, QuestionDebugRecord artifact types
  - ensure_output_paths_are_safe output safety check
  - git_commit_sha, git_dirty_worktree helpers
  - load_benchmark_output, run_config_from_artifact for QA mode
  - benchmark_prompt_hashes, benchmark_runtime_config for manifest population
  - compute_per_category aggregation
  - Full main() pipeline orchestration (parse -> config -> dataset -> runtime -> ingest loop -> artifacts)
affects: [04-qa-evaluation, 05-concurrency-view]

tech-stack:
  added: [chrono, tokio-main]
  patterns: [three-artifact-output, incremental-jsonl-flush, manifest-reproducibility, scoped-metrics-collector]

key-files:
  created:
    - bench/longmemeval/results/local/.gitkeep
  modified:
    - bench/longmemeval/longmemeval.rs

key-decisions:
  - "resolve_qa_config now loads artifact via load_benchmark_output + run_config_from_artifact instead of starting with defaults"
  - "QA stub writes status=qa-not-implemented for run, status=ingest-only for ingest"
  - "with_scoped_collector wraps ingest_instance for per-instance stage metrics"
  - "Manifest protocol_version: 2026-03-15-longmemeval-v1"
  - "git_dirty_worktree filters bench/longmemeval/results/ paths from dirty check"

patterns-established:
  - "Three-artifact output: summary JSON + questions JSONL + debug JSONL, sidecars truncated at start"
  - "BenchmarkManifest with full reproducibility contract (dataset fingerprint, prompt hashes, runtime config, git state)"
  - "Per-instance loop: ingest -> QA stub -> append JSONL -> accumulate results"

requirements-completed: [CLI-01, CLI-02, CLI-03, CLI-08, CLI-09]

duration: 4min
completed: 2026-03-15
---

# Phase 3 Plan 2: Artifact Types and Pipeline Orchestration Summary

**Artifact type system with serde roundtrip tests, output safety checks, git helpers, full main() pipeline orchestration producing three-artifact output (summary JSON + question JSONL + debug JSONL) with reproducibility manifest**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-15T13:10:58Z
- **Completed:** 2026-03-15T13:15:16Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- 10 artifact types (BenchmarkOutput, BenchmarkManifest, CategoryResult, QuestionResult, QuestionDebugRecord, ReflectTraceEntry, RetrievedFactEntry, BenchmarkArtifacts, SourceArtifact, BenchmarkPromptHashes/RuntimeConfig) with full serde support
- Output safety: ensure_output_paths_are_safe blocks overwrites without --force, allows with --force, prevents qa from overwriting source artifact
- Full main() pipeline: parse_args -> safety check -> load dataset -> filter instances -> build runtime -> per-instance ingest loop -> write three artifacts
- resolve_qa_config upgraded to load artifact and extract config (was placeholder from Plan 01)
- with_scoped_collector wiring for per-instance stage metrics (resolves Phase 2 deferred item)
- 123 total tests passing (62 from Plan 01 + 61 new including 15 artifact serde roundtrips, 4 output safety, 2 git helpers, load/extract tests, compute_per_category)

## Task Commits

1. **Task 1: Artifact types and output safety** - `30a1e21` (feat)
2. **Task 2: Pipeline orchestration and main()** - `20d57fe` (feat)

## Files Created/Modified
- `bench/longmemeval/longmemeval.rs` - Artifact types, output safety, git helpers, full main() pipeline orchestration, 61 new unit tests

## Decisions Made
- resolve_qa_config now properly loads the artifact file and extracts config from its manifest, replacing the placeholder from Plan 01
- QA stub writes "qa-not-implemented" status for run command and "ingest-only" for ingest command, producing valid artifacts for Phase 4
- with_scoped_collector wraps each ingest_instance call so MetricsCollector captures stage-level usage per instance
- git_dirty_worktree filters out bench/longmemeval/results/ paths from the dirty check (matching LoCoMo pattern for bench/locomo/results/)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed parse_qa_subcommand_with_path test after resolve_qa_config change**
- **Found during:** Task 1
- **Issue:** Existing test used a non-existent "path.json" artifact path, which now fails because resolve_qa_config loads the artifact
- **Fix:** Updated test to create a temp file with valid BenchmarkOutput JSON
- **Files modified:** bench/longmemeval/longmemeval.rs
- **Verification:** Test passes
- **Committed in:** 30a1e21

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Necessary fix for test that assumed the old placeholder resolve_qa_config. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Pipeline skeleton complete: run/ingest produce valid artifacts with bank mappings
- QA scoring stubbed with clear status markers, ready for Phase 4 judge implementation
- Manifest includes full reproducibility contract for publication
- Stage metrics wired via scoped collector

---
*Phase: 03-cli-artifact-infrastructure*
*Completed: 2026-03-15*
