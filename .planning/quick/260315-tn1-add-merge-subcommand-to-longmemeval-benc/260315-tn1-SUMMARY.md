---
phase: quick
plan: 01
subsystem: bench/longmemeval
tags: [merge, cli, benchmark-tooling]
key-files:
  modified:
    - bench/longmemeval/longmemeval.rs
decisions: []
metrics:
  duration: 6 min
  completed: "2026-03-16T04:32:50Z"
  tasks_completed: 2
  tasks_total: 2
  tests_added: 11
  tests_total: 152
---

# Quick Task 260315-tn1: Add merge subcommand to longmemeval-bench Summary

Merge subcommand for longmemeval-bench, following the locomo-bench merge pattern exactly. Enables combining subset benchmark artifacts from parallel runs into a single canonical result file.

## Task Results

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | Add merge subcommand infrastructure and merge logic | a80cb12 | Done |
| 2 | Add merge unit tests | 6ef2a30 | Done |

## What Was Built

**Task 1 -- Merge infrastructure (521 lines added):**
- `BenchCommand::Merge` variant and `as_str()` mapping
- `source_artifacts: Vec<SourceArtifact>` field on `BenchmarkManifest`
- `merge_artifacts: Vec<PathBuf>` on `ParsedCli` and `BenchInvocation`
- CLI parsing: `merge` subcommand with positional artifact paths, min-2 validation
- `validate_merge_overrides` -- rejects all flags except --out/--tag/--force
- `resolve_merge_config` -- minimal config for merge (only output/tag/force)
- `default_output_path` Merge arm: `results/local/{tag_or_merged}.json`
- `ensure_output_paths_are_safe` updated with merge-input overwrite protection
- `print_help` updated with merge usage line
- Merge helpers: `read_jsonl_records`, `write_jsonl_records`, `artifact_relative_path`, `LoadedArtifactBundle`, `load_artifact_bundle`, `merge_source_artifact`, `ensure_merge_compatible`, `merge_profile_value`, `merge_concurrency_value`, `warn_if_mixed`
- `merge_artifacts` main function: loads bundles, validates compatibility (models, prompt hashes, runtime config), detects overlapping question_ids and duplicate banks, merges results/debug/banks/stage_metrics, sorts by question_id, writes sidecars and output JSON with recomputed accuracy and source_artifacts provenance
- Early-return merge dispatch in `main` (no runtime needed)

**Task 2 -- Tests (303 lines, 11 new tests):**
- CLI parsing: `parse_merge_subcommand`, `parse_merge_with_tag`, `parse_merge_with_out`, `parse_merge_requires_two_inputs`, `parse_merge_rejects_profile`, `parse_merge_rejects_instance_jobs`
- Output paths: `default_output_merge_with_tag`, `default_output_merge_no_tag`
- Merge logic: `merge_combines_disjoint_subset_artifacts` (end-to-end with fixture artifacts), `merge_rejects_overlapping_questions`, `merge_output_must_differ_from_inputs_without_force`
- Updated `bench_command_as_str` to include Merge

## Deviations from Plan

None -- plan executed exactly as written.

## Verification

- All 152 longmemeval-bench tests pass (141 existing + 11 new)
- Clean build (no errors)
- Help text shows merge subcommand

## Self-Check: PASSED
