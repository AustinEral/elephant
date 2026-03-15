---
phase: 2
slug: ingestion-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `#[tokio::test]` |
| **Config file** | Cargo.toml (existing) |
| **Quick run command** | `cargo test --lib -p elephant -- ingest` |
| **Full suite command** | `cargo test` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --lib -- longmemeval`
- **After every plan wave:** Run `cargo test`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | INGEST-01 | unit | `cargo test --test longmemeval_ingest -- bank_creation` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | INGEST-02 | unit | `cargo test -p elephant -- format_session` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | INGEST-03 | unit | `cargo test -p elephant -- consolidation_mode` | ❌ W0 | ⬜ pending |
| 02-01-04 | 01 | 1 | INGEST-04 | unit | `cargo test -p elephant -- sequential` | ❌ W0 | ⬜ pending |
| 02-01-05 | 01 | 1 | INGEST-05 | N/A | Deferred to Phase 5 | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `bench/longmemeval/ingest.rs` — unit tests for format_session_text, format_session_json, parse_haystack_date, parse_date_prefix
- [ ] Tests for IngestConfig/IngestResult serialization roundtrip
- [ ] Tests for ConsolidationMode helper methods (enabled, per_session)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Pool sizing handles concurrent ops | INGEST-05 | Deferred to Phase 5 | N/A |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
