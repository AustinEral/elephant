---
phase: 05-concurrency-resume-and-view-tool
verified: 2026-03-15T23:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 5: Concurrency, Resume, and View Tool Verification Report

**Phase Goal:** User can run instances in parallel, resume interrupted runs, and inspect results with a standalone view tool
**Verified:** 2026-03-15T23:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

Plan 05-01 truths (requirements CLI-05, CLI-10, INGEST-05):

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `--instance-jobs N` processes N instances concurrently | VERIFIED | `Semaphore::new(config.instance_jobs)` at longmemeval.rs:1313, semaphore-gated `tokio::spawn` loop at lines 1318-1560 |
| 2 | Default instance_jobs=1 preserves sequential behavior | VERIFIED | `instance_jobs: 1` default at longmemeval.rs:243; semaphore with size 1 = sequential |
| 3 | Postgres pool auto-scales to min(instance_jobs * 3, 50) | VERIFIED | longmemeval.rs:1193 `Some(std::cmp::min(config.instance_jobs as u32 * 3, 50))`, consumed by runtime.rs:259-263 via PgPoolOptions |
| 4 | Individual instance failures are logged but do not kill other concurrent instances | VERIFIED | Ingest errors produce a failed `QuestionResult` with `status: "ingest_error"`, call `push_and_flush`, then `return Err(err_msg)`. Join loop at lines 1561+ logs via `eprintln!` without aborting |
| 5 | QA mode skips ingestion and reuses bank_ids from artifact under concurrency | VERIFIED | Spawned task checks `BenchCommand::Qa` and reads `shared.lock().await.banks.get(&instance.question_id)` at longmemeval.rs:1336-1345; `existing_banks` loaded at startup and placed into SharedState |
| 6 | JSONL sidecars and summary JSON are written incrementally as each instance completes | VERIFIED | `SharedState::push_and_flush` appends to JSONL sidecars via `append_jsonl` and rewrites summary JSON via `flush()` after every instance |
| 7 | Progress output shows [N/total] question_id with per-instance timing | VERIFIED | Three progress `eprintln!` sites at lines 1393-1398, 1430-1435, 1542-1547: `"[{done}/{total_instances}] {} ok/err ingest {:.1}s qa {:.1}s"` |

Plan 05-02 truths (requirements VIEW-01, VIEW-02, VIEW-03):

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 8 | longmemeval-view binary compiles and runs as a standalone tool | VERIFIED | `cargo build --bin longmemeval-view` finishes successfully; `[[bin]]` target at Cargo.toml:54; 1053-line file at bench/longmemeval/view.rs |
| 9 | Default output shows config summary + per-category accuracy table + stage metrics + total time | VERIFIED | `view_single` at view.rs:502 renders: config table (build_config_rows), category table (build_category_rows), stage metrics table (build_stage_rows), then `"Total time: ..."` |
| 10 | Per-question table appears only with --verbose flag, loaded from JSONL sidecar | VERIFIED | `if verbose { let questions = load_questions(Path::new(path)); ... }` at view.rs:540; `load_questions` calls `sidecar_path(path, "questions")` then parses JSONL |
| 11 | Two-file comparison shows delta column per category | VERIFIED | `view_compare` at view.rs:577 builds `CompareCategoryRow` with `delta: fmt_pct_delta(ca.accuracy, cb.accuracy)` |
| 12 | Stage metrics (token counts per stage) shown in default view | VERIFIED | `build_stage_rows` at view.rs:482 iterates `stage_metrics` BTreeMap, emits `StageRow { stage, requests, total_tok }`; rendered in `view_single` unconditionally |

**Score:** 9/9 must-have groups verified (12 individual truths all pass)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `bench/longmemeval/longmemeval.rs` | Semaphore-gated concurrent instance loop with SharedState | VERIFIED | 2869 lines; `Semaphore::new` at line 1313; `SharedState` struct at line 1028 with `push_and_flush`/`record_bank`/`flush` methods |
| `src/runtime.rs` | Configurable pool sizing via BuildRuntimeOptions | VERIFIED | 490 lines; `pub max_pool_connections: Option<u32>` at line 154; `PgPoolOptions::new().max_connections(max_conns)` at lines 260-262 |
| `bench/longmemeval/view.rs` | Standalone longmemeval-view binary (min 200 lines) | VERIFIED | 1053 lines; full single/comparison/verbose display modes |
| `Cargo.toml` | `[[bin]]` target for longmemeval-view | VERIFIED | `name = "longmemeval-view"` confirmed at Cargo.toml:54 |

### Key Link Verification

Plan 05-01 key links:

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| bench/longmemeval/longmemeval.rs | src/runtime.rs | BuildRuntimeOptions.max_pool_connections | WIRED | longmemeval.rs:1191-1194 passes `max_pool_connections: Some(std::cmp::min(config.instance_jobs as u32 * 3, 50))` to `build_runtime_from_env` which consumes it at runtime.rs:259 |
| bench/longmemeval/longmemeval.rs | bench/common/io.rs | append_jsonl inside Mutex-protected SharedState | WIRED | `SharedState::push_and_flush` calls `append_jsonl(&self.questions_path, &result)` and `append_jsonl(&self.debug_path, &debug)` at lines 1051-1052 |

Plan 05-02 key links:

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| bench/longmemeval/view.rs | bench/longmemeval/results/ | reads BenchmarkOutput JSON artifact via serde_json::from_str | WIRED | `serde_json::from_str(&raw)` at view.rs:386 within `load_file` function |
| bench/longmemeval/view.rs | bench/common/io.rs | sidecar_path for verbose question table | WIRED | `use common::io::sidecar_path` at view.rs:22; called in `load_questions` at view.rs:393 |

### Requirements Coverage

All six Phase 5 requirement IDs explicitly declared in plan frontmatter:

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CLI-05 | 05-01-PLAN.md | `--instance-jobs N` concurrency control | SATISFIED | `Semaphore::new(config.instance_jobs)` gates all spawned tasks; parse at longmemeval.rs:448 |
| CLI-10 | 05-01-PLAN.md | Resume via `qa` subcommand reusing bank_ids | SATISFIED | QA path reads `shared.lock().await.banks.get(question_id)` from pre-loaded existing_banks |
| INGEST-05 | 05-01-PLAN.md | Postgres pool explicitly sized for concurrent bank operations | SATISFIED | `PgPoolOptions::new().max_connections(min(instance_jobs*3, 50))` in runtime.rs |
| VIEW-01 | 05-02-PLAN.md | Separate `longmemeval-view` binary | SATISFIED | Standalone binary at bench/longmemeval/view.rs with its own `fn main()` |
| VIEW-02 | 05-02-PLAN.md | Per-category accuracy display with question counts | SATISFIED | `build_category_rows` emits `CategoryRow { category, acc, n }` sorted alphabetically with overall row last |
| VIEW-03 | 05-02-PLAN.md | Single-artifact view mode with config, summary, and question tables | SATISFIED | `view_single` renders config table, category table, stage metrics, total time; verbose adds per-question table |

No orphaned requirements: REQUIREMENTS.md traceability table maps CLI-05, CLI-10, INGEST-05, VIEW-01, VIEW-02, VIEW-03 all to Phase 5 (lines 126-135). All six are covered by plans 05-01 and 05-02.

Note: INGEST-05 appears in both Phase 2 and Phase 5 roadmap requirement lists. Phase 2 plan did not claim it. Phase 5 plan 05-01 claims and implements it. REQUIREMENTS.md traceability maps it exclusively to Phase 5. No conflict.

### Anti-Patterns Found

Scanned all four files modified in this phase:

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| bench/longmemeval/longmemeval.rs | `std::process::exit` present (8 occurrences) | Info | All in startup/validation paths (CLI arg errors, missing dataset, QA validation). None inside per-instance worker. Acceptable use — not the stub pattern the plan required removing. |
| bench/longmemeval/view.rs | 19 compiler warnings (unused items) | Info | Dead code warnings, no behavioral impact. Binary compiles and all 23 tests pass. |

No TODO/FIXME/placeholder comments found in phase-modified files. No empty implementations or stub returns in worker code.

### Human Verification Required

The following aspects cannot be verified programmatically:

1. **Actual parallel speedup**
   - Test: Run `longmemeval-bench run --instance-jobs 4` against a real dataset
   - Expected: Wall-clock time roughly 4x faster than sequential; 4 instances active simultaneously
   - Why human: Cannot simulate concurrent Postgres + LLM calls in static analysis

2. **Crash resilience / incremental output**
   - Test: Kill a running bench mid-flight; inspect the output JSON
   - Expected: Completed instances are preserved in the artifact; no data loss for finished work
   - Why human: Requires a live run with intentional interruption

3. **Terminal output readability under concurrency**
   - Test: Run with `--instance-jobs 4`; observe progress lines in terminal
   - Expected: `[N/total] question_id ok/err ingest Xs qa Ys` lines are distinct and readable despite interleaving
   - Why human: Interleaved eprintln output can only be assessed during a real run

4. **longmemeval-view rendering**
   - Test: Run `longmemeval-view bench/longmemeval/results/local/<some-result>.json`
   - Expected: Config table, category table with 7 categories + overall, stage metrics, total time all display correctly formatted
   - Why human: Visual table rendering requires actual output inspection

### Gaps Summary

No gaps found. All automated checks passed.

Both binaries compile (`cargo build --bin longmemeval-bench`, `cargo build --bin longmemeval-view`). All 139 longmemeval-bench tests pass, all 23 longmemeval-view tests pass. All four phase artifacts exist with substantive implementations, all key links are wired, all six requirement IDs are satisfied.

---

_Verified: 2026-03-15T23:00:00Z_
_Verifier: Claude (gsd-verifier)_
