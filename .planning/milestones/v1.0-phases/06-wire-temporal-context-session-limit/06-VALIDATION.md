---
phase: 6
slug: wire-temporal-context-session-limit
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) |
| **Config file** | Cargo.toml |
| **Quick run command** | `cargo test --bin longmemeval-bench && cargo test -p elephant -- reflect` |
| **Full suite command** | `cargo test --all` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --bin longmemeval-bench && cargo test -p elephant -- reflect`
- **After every plan wave:** Run `cargo test --all`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | DATA-06-a | unit | `cargo test -p elephant reflect_query_roundtrip -- --exact` | ✅ (update) | ⬜ pending |
| 06-01-02 | 01 | 1 | DATA-06-b | unit | `cargo test -p elephant temporal_context_in_user_message -- --exact` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | DATA-06-c | unit | `cargo test --bin longmemeval-bench reflect_query_includes_temporal -- --exact` | ❌ W0 | ⬜ pending |
| 06-02-01 | 02 | 1 | GAP-SL-a | unit | `cargo test --bin longmemeval-bench ingest_config_default -- --exact` | ✅ (update) | ⬜ pending |
| 06-02-02 | 02 | 1 | GAP-SL-b | unit | `cargo test --bin longmemeval-bench -- session_limit` | ❌ W0 | ⬜ pending |
| 06-02-03 | 02 | 1 | GAP-SL-c | unit | `cargo test --bin longmemeval-bench -- session_limit` | ❌ (covered by SL-b) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Update `reflect_query_roundtrip` test in `src/types/pipeline.rs` — add `temporal_context: Some("...")` to roundtrip
- [ ] New test `temporal_context_in_user_message` in `src/reflect/mod.rs` — verify user message prefix
- [ ] Update `ingest_config_default` test in `bench/longmemeval/ingest.rs` — verify `session_limit: None` default
- [ ] New test for session_limit slicing behavior (mock or logic test, not integration)

*Existing infrastructure covers framework install — cargo test already configured.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
