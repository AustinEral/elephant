---
phase: 5
slug: concurrency-resume-and-view-tool
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `#[tokio::test]` |
| **Config file** | None (Cargo convention) |
| **Quick run command** | `cargo test --bin longmemeval-bench --bin longmemeval-view -- --nocapture` |
| **Full suite command** | `cargo test` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --bin longmemeval-bench --bin longmemeval-view -- --nocapture`
- **After every plan wave:** Run `cargo test`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | CLI-05 | unit | `cargo test --bin longmemeval-bench parse_instance_jobs` | Partially | ⬜ pending |
| 05-01-02 | 01 | 1 | INGEST-05 | unit | `cargo test --lib -- runtime` | ❌ W0 | ⬜ pending |
| 05-01-03 | 01 | 1 | CLI-05 | unit | `cargo test --bin longmemeval-bench concurren` | ❌ W0 | ⬜ pending |
| 05-02-01 | 02 | 1 | CLI-10 | unit | `cargo test --bin longmemeval-bench qa_` | Partially | ⬜ pending |
| 05-03-01 | 03 | 2 | VIEW-01 | smoke | `cargo build --bin longmemeval-view` | ❌ W0 | ⬜ pending |
| 05-03-02 | 03 | 2 | VIEW-02 | unit | `cargo test --bin longmemeval-view category` | ❌ W0 | ⬜ pending |
| 05-03-03 | 03 | 2 | VIEW-03 | unit | `cargo test --bin longmemeval-view single` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `bench/longmemeval/view.rs` — new binary source file stub
- [ ] Tests for pool sizing formula in runtime or longmemeval harness
- [ ] Tests for view tool output (data formatting functions in isolation)
- [ ] Tests for concurrency behavior (semaphore gating, shared state)

*Existing infrastructure partially covers CLI-05 parse tests and CLI-10 QA parse tests.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Concurrent instances produce correct interleaved progress output | CLI-05 | Requires multi-instance run with real data | Run `longmemeval-bench run --instance-jobs 3` on test dataset, verify `[N/M]` lines appear for all instances |
| Summary JSON is valid after interrupted concurrent run | CLI-05 | Requires forced interruption | Start concurrent run, Ctrl-C mid-execution, verify summary JSON is parseable |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
