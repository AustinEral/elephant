---
phase: 03
slug: cli-artifact-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (built-in) |
| **Config file** | Cargo.toml [[bin]] section |
| **Quick run command** | `cargo test --bin longmemeval-bench` |
| **Full suite command** | `cargo test --bin longmemeval-bench && cargo test --lib` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --bin longmemeval-bench`
- **After every plan wave:** Run `cargo test --bin longmemeval-bench && cargo test --lib`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | CLI-01 | unit | `cargo test --bin longmemeval-bench -- run_subcommand` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | CLI-02 | unit | `cargo test --bin longmemeval-bench -- ingest_subcommand` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 1 | CLI-03 | unit | `cargo test --bin longmemeval-bench -- qa_subcommand` | ❌ W0 | ⬜ pending |
| 03-01-04 | 01 | 1 | CLI-04 | unit | `cargo test --bin longmemeval-bench -- profile_loads` | ❌ W0 | ⬜ pending |
| 03-01-05 | 01 | 1 | CLI-06 | unit | `cargo test --bin longmemeval-bench -- config_overlay` | ❌ W0 | ⬜ pending |
| 03-01-06 | 01 | 1 | CLI-07 | unit | `cargo test --bin longmemeval-bench -- instance_flag` | ❌ W0 | ⬜ pending |
| 03-01-07 | 01 | 1 | CLI-08 | unit | `cargo test --bin longmemeval-bench -- sidecar_paths` | ✅ partial | ⬜ pending |
| 03-01-08 | 01 | 1 | CLI-09 | unit | `cargo test --bin longmemeval-bench -- manifest` | ❌ W0 | ⬜ pending |
| 03-01-09 | 01 | 1 | CLI-11 | unit | `cargo test --bin longmemeval-bench -- default_output` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] CLI parsing tests for all subcommands (parse_args_from unit tests)
- [ ] Profile loading tests (smoke, full-s, full-m defaults)
- [ ] Config resolution tests (layered apply)
- [ ] Default output path tests
- [ ] QA override validation tests
- [ ] Profile JSON files: `bench/longmemeval/profiles/smoke.json`, `full-s.json`, `full-m.json`
- [ ] Results directory: `bench/longmemeval/results/local/.gitkeep`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `run` executes full pipeline end-to-end | CLI-01 | Requires Postgres + LLM | Run `cargo run --bin longmemeval-bench -- run --profile smoke --tag test` against local stack |
| `ingest` writes bank mappings artifact | CLI-02 | Requires Postgres | Run `cargo run --bin longmemeval-bench -- ingest --profile smoke --tag test` and verify output JSON |
| `qa` reads existing artifact | CLI-03 | Requires existing banks | Run `qa` against ingest artifact from above |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
