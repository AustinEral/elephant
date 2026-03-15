---
phase: 4
slug: evaluation-path
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in #[test] + cargo test |
| **Config file** | None — Cargo auto-discovers |
| **Quick run command** | `cargo test --bin longmemeval-bench` |
| **Full suite command** | `cargo test --bin longmemeval-bench --bin locomo-bench` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --bin longmemeval-bench --bin locomo-bench`
- **After every plan wave:** Run `cargo test` (full workspace)
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 1 | EVAL-01 | unit | `cargo test --bin longmemeval-bench judge_model_default` | Wave 0 | ⬜ pending |
| TBD | 01 | 1 | EVAL-02 | unit | `cargo test --bin longmemeval-bench select_judge_prompt` | Wave 0 | ⬜ pending |
| TBD | 01 | 1 | EVAL-06 | unit | `cargo test --bin longmemeval-bench abstention_prompt_selection` | Wave 0 | ⬜ pending |
| TBD | 02 | 2 | EVAL-03 | integration | Manual: run smoke profile, verify 0 "qa-not-implemented" | manual-only | ⬜ pending |
| TBD | 02 | 2 | EVAL-04 | unit | `cargo test --bin longmemeval-bench compute_per_category` | ✅ exists | ⬜ pending |
| TBD | 02 | 2 | EVAL-05 | unit | `cargo test --bin longmemeval-bench accuracy_computation` | Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `bench/common/judge.rs` — #[cfg(test)] module with: JudgeResponse parsing, extract_json fallback, build_judge_client label generation
- [ ] `bench/longmemeval/longmemeval.rs` — tests for: select_judge_prompt routing, prompt placeholder rendering, accuracy computation with mixed results
- [ ] LoCoMo compilation check — `cargo test --bin locomo-bench` passes after judge extraction

*Existing infrastructure partially covers: compute_per_category test exists from Phase 3*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| All 500 questions scored | EVAL-03 | Requires live LLM for reflect + judge | Run smoke profile, verify all questions have status != "qa-not-implemented" |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
