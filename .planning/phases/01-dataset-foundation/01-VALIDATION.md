---
phase: 1
slug: dataset-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[cfg(test)]` + `#[test]` |
| **Config file** | None (Cargo's built-in test runner) |
| **Quick run command** | `cargo test --bin longmemeval-bench` |
| **Full suite command** | `cargo test` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --bin longmemeval-bench`
- **After every plan wave:** Run `cargo test` + `cargo build --bin longmemeval-bench` + `cargo build --bin locomo-bench`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | DATA-05 | unit | `cargo test --bin longmemeval-bench -- fingerprint` | No -- W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | DATA-03 | unit | `cargo test --bin longmemeval-bench -- question_type` | No -- W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | DATA-04 | unit | `cargo test --bin longmemeval-bench -- answer_to_string` | No -- W0 | ⬜ pending |
| 1-01-04 | 01 | 1 | DATA-06 | unit | `cargo test --bin longmemeval-bench -- question_date` | No -- W0 | ⬜ pending |
| 1-01-05 | 01 | 1 | DATA-07 | unit | `cargo test --bin longmemeval-bench -- zip_validate` | No -- W0 | ⬜ pending |
| 1-02-01 | 02 | 1 | DATA-01 | integration | `cargo test --test longmemeval_dataset -- load_s` | No -- W0 | ⬜ pending |
| 1-02-02 | 02 | 1 | DATA-02 | integration | `cargo test --test longmemeval_dataset -- load_m` | No -- W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `bench/common/mod.rs` — shared FNV fingerprinting, JSONL helpers, sidecar path utilities
- [ ] `bench/longmemeval/longmemeval.rs` — binary target with dataset types, loading, validation
- [ ] Unit tests for QuestionType deserialization, answer coercion, zip-validation, fingerprinting
- [ ] Integration test file `tests/longmemeval_dataset.rs` — dataset loading tests (marked `#[ignore]`, need data files)
- [ ] `Cargo.toml` `[[bin]]` target for `longmemeval-bench`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Load S dataset (500 instances) | DATA-01 | Requires 277MB data file download | Download `longmemeval_s_cleaned.json` to `data/`, run `cargo test --test longmemeval_dataset -- load_s` |
| Load M dataset (all instances) | DATA-02 | Requires 2.74GB data file download | Download `longmemeval_m_cleaned.json` to `data/`, run `cargo test --test longmemeval_dataset -- load_m` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
