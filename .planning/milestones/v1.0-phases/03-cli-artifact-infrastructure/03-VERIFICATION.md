---
phase: 03-cli-artifact-infrastructure
verified: 2026-03-15T14:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 3: CLI and Artifact Infrastructure Verification Report

**Phase Goal:** User can run the harness via CLI with subcommands, profiles, and get reproducible three-artifact output
**Verified:** 2026-03-15
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `longmemeval-bench run` executes ingest+consolidate pipeline and writes results | VERIFIED | `main()` routes `BenchCommand::Run` through ingest loop + writes summary JSON + truncates and flushes JSONL sidecars; QA stub with `status="qa-not-implemented"` and printed notice are explicit Phase 4 placeholders per plan |
| 2 | `longmemeval-bench ingest` creates banks and writes ingest artifact without running QA | VERIFIED | `status="ingest-only"` for ingest command; `ingest_instance` called in per-instance loop; QA path skipped by command match |
| 3 | `longmemeval-bench qa` loads existing artifact, validates bank_ids, and stubs QA scoring | VERIFIED | `resolve_qa_config` loads artifact via `load_benchmark_output` + `run_config_from_artifact`; bank_id validation loop in `main()` at line 1039; `qa` reuses existing `banks` map from artifact |
| 4 | Profile selection controls dataset, instance subset, and consolidation mode | VERIFIED | `resolve_fresh_config` applies three-layer resolution: profile JSON → `--config` overlay → CLI flags; `smoke.json` (instance_limit=1, S dataset), `full-s.json` (S dataset), `full-m.json` (M dataset) all verified |
| 5 | Output artifacts land in `bench/longmemeval/results/local/` by default | VERIFIED | `default_output_path` returns `bench/longmemeval/results/local/{stem}.json`; directory placeholder `results/local/.gitkeep` exists; JSONL sidecars via `sidecar_path()` follow same prefix |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `bench/longmemeval/longmemeval.rs` | CLI types, parsing, config resolution, artifact types, main() | VERIFIED | 2329 lines; `async fn main`, `parse_args_from`, `resolve_fresh_config`, `resolve_qa_config`, `BenchmarkOutput`, `BenchmarkManifest`, all required types present |
| `bench/longmemeval/profiles/smoke.json` | Smoke profile: instance_limit=1, S dataset | VERIFIED | Contains `instance_limit:1`, `dataset:"data/longmemeval_s_cleaned.json"`, `consolidation:"end"` |
| `bench/longmemeval/profiles/full-s.json` | Full-S profile: S dataset, no instance limit | VERIFIED | Contains `dataset:"data/longmemeval_s_cleaned.json"`, no instance_limit |
| `bench/longmemeval/profiles/full-m.json` | Full-M profile: M dataset | VERIFIED | Contains `dataset:"data/longmemeval_m_cleaned.json"` |
| `bench/longmemeval/results/local/.gitkeep` | Default output directory placeholder | VERIFIED | File exists (0 bytes) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `parse_args_from` | `resolve_fresh_config` / `resolve_qa_config` | `ParsedCli.overrides` feeds into config resolution | WIRED | Lines 429-437: command match routes to `resolve_fresh_config(overrides)` or `resolve_qa_config(artifact_path, overrides)` |
| `RunProfile.config_path()` | `bench/longmemeval/profiles/*.json` | `load_json_config` reads profile JSON | WIRED | Line 476: `load_json_config(&profile.config_path())?.apply(&mut config)` |
| `main()` | `ingest::ingest_instance()` | per-instance loop calling ingest_instance | WIRED | Lines 1152-1163: `with_scoped_collector(metrics.clone(), ingest::ingest_instance(instance, &runtime, &ingest_config))` |
| `main()` | `common::io::append_jsonl()` | incremental JSONL flush after each instance | WIRED | Lines 1187, 1197: `append_jsonl(&questions_path, &qr)` and `append_jsonl(&debug_path, &dr)` |
| `BenchmarkOutput` | `BenchmarkManifest` | manifest field populated with full reproducibility contract | WIRED | Lines 1111-1133: `BenchmarkManifest` constructed with protocol_version, dataset_fingerprint, prompt_hashes, runtime_config, git state, command string |
| `main()` | `ensure_output_paths_are_safe()` | safety check before any writes | WIRED | Lines 971-975: called immediately after `default_output_path`, before dataset load or sidecar truncation |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CLI-01 | 03-01, 03-02 | `longmemeval-bench` binary with `run` subcommand | SATISFIED | Binary registered in Cargo.toml (line 50); `BenchCommand::Run` routes through full pipeline in `main()` |
| CLI-02 | 03-01, 03-02 | `ingest` subcommand (ingest + consolidate only, no QA) | SATISFIED | `BenchCommand::Ingest` in parse + main loop writes `status="ingest-only"` QuestionResult |
| CLI-03 | 03-01, 03-02 | `qa` subcommand (score against existing banks from ingest artifact) | SATISFIED | `resolve_qa_config` loads artifact; main validates bank_ids; QA scoring stubbed for Phase 4 |
| CLI-04 | 03-01 | Profile system with `smoke`, `full-s`, `full-m` | SATISFIED | `RunProfile` enum with `config_path()` + three JSON files verified |
| CLI-06 | 03-01 | `--config` JSON overlay on top of profile | SATISFIED | `resolve_fresh_config` at line 478: loads `--config` JSON and calls `.apply()` on top of profile defaults |
| CLI-07 | 03-01 | `--instance` flag (repeatable) | SATISFIED | `parse_cli_overrides` accumulates `overrides.instances.push(...)` for each `--instance`; test `parse_multiple_instances` passes |
| CLI-08 | 03-01, 03-02 | Three-artifact output: summary JSON, question JSONL, debug JSONL | SATISFIED | `sidecar_path` for questions + debug; JSONL truncated at start; `append_jsonl` called per instance; summary written at end |
| CLI-09 | 03-01, 03-02 | Manifest with full reproducibility contract | SATISFIED | `BenchmarkManifest` includes dataset_fingerprint, prompt_hashes, runtime_config, git commit/dirty, full CLI invocation string, protocol_version |
| CLI-11 | 03-01 | Results default to `bench/longmemeval/results/local/` | SATISFIED | `default_output_path` constructs `bench/longmemeval/results/local/{stem}.json`; test `default_output_run_no_tag` passes |

All 9 phase requirements accounted for. No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `bench/longmemeval/longmemeval.rs` | 1238 | `NOTE: QA scoring not yet implemented (Phase 4)` | Info | Intentional per plan — QA is Phase 4 scope. Not a blocker. |

No blocker anti-patterns. The QA scoring stub is by design (Phase 4 prerequisite).

### Human Verification Required

#### 1. End-to-end run with Postgres and LLM

**Test:** `cargo run --bin longmemeval-bench -- run --profile smoke --tag verify-test` against a running local stack (Postgres + LLM configured in `.env`)
**Expected:** Creates a bank, ingests 1 instance, writes `bench/longmemeval/results/local/verify-test.json` with non-empty `banks` map; companion `verify-test.questions.jsonl` and `verify-test.debug.jsonl` each contain 1 line. Manifest has non-empty `dataset_fingerprint` and valid `commit` hash.
**Why human:** Requires Postgres connection pool + working LLM credentials; cannot be verified statically.

#### 2. Ingest artifact usable by qa subcommand

**Test:** Run `ingest --profile smoke --tag ingest-test`, then run `qa bench/longmemeval/results/local/ingest-test.json --tag qa-test`
**Expected:** QA run loads artifact, skips ingest, processes same instance with `status="qa-not-implemented"`. Output path is `qa-test.json`, not overwriting the ingest artifact.
**Why human:** Requires live Postgres with persisted bank from step 1.

### Gaps Summary

No gaps. All five observable truths verified, all nine requirements satisfied, all key links wired, 123 unit tests pass, build clean (4 warnings, no errors).

---

_Verified: 2026-03-15_
_Verifier: Claude (gsd-verifier)_
