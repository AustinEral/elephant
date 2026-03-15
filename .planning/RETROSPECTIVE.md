# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — LongMemEval Benchmark Integration

**Shipped:** 2026-03-15
**Phases:** 6 | **Plans:** 10 | **Execution time:** 0.60 hours

### What Was Built
- Full LongMemEval benchmark harness (`longmemeval-bench`) with run/ingest/qa subcommands
- Dataset parsing for 500 questions across 7 categories with mixed-type answer coercion
- Per-instance bank ingestion pipeline with configurable consolidation (end/per-session/off)
- 5 judge prompt variants (factual, temporal, knowledge-update, preference, abstention)
- Semaphore-gated concurrency with crash-resilient incremental output
- Standalone view tool (`longmemeval-view`) for result inspection
- Shared bench infrastructure (bench/common/) extracted from LoCoMo

### What Worked
- Extracting shared infrastructure (judge, fingerprinting, JSONL I/O) early in Phase 1 paid off — reused cleanly across all later phases
- Phase-by-phase layering (dataset → ingest → CLI → eval → concurrency) meant each phase had a solid foundation
- Profile system (smoke/full-s/full-m) enabled fast iteration during development
- Three-artifact output pattern from LoCoMo transferred cleanly to LongMemEval
- Delegating wrapper pattern minimized LoCoMo diff when extracting common modules

### What Was Inefficient
- DATA-06 (temporal context wiring) was assigned to Phase 1 but needed Phase 4 infrastructure — required a gap closure phase (Phase 6)
- session_limit flag was parsed in CLI but not wired through to ingest until Phase 6 — silently did nothing for phases 3-5
- Phase verifications caught these gaps but not until the milestone audit; earlier cross-phase integration checks could catch wiring gaps sooner
- INGEST-05 (pool sizing) was claimed by Phase 2 SUMMARY but actually implemented in Phase 5 — documentation tracking mismatch

### Patterns Established
- `#[path = "../common/mod.rs"] mod common;` pattern for sharing code between binary targets
- Two-layer validation: serde parse then semantic collect-all pass
- SharedState with incremental flush for crash-resilient concurrent output
- View tool uses `#[serde(default)]` on ALL fields for forward/backward compatibility
- `[Current date: ...]` prefix for temporal context, distinct from ingested `[Date: ...]` prefixes

### Key Lessons
1. Requirements that span multiple phases (like DATA-06) should be explicitly tracked as "partial" until the final wire-up phase completes — don't mark complete at the data layer
2. CLI flags should be wired end-to-end when introduced, not left as no-ops — silent no-op flags are worse than missing flags
3. Per-instance bank architecture (500 banks vs LoCoMo's 10) drives different engineering choices around pool sizing and concurrency
4. 5 judge prompt variants (vs LoCoMo's 1) significantly improved accuracy for temporal and knowledge-update questions

### Cost Observations
- Model mix: quality profile (opus for orchestration, sonnet for subagents)
- 10 plans executed in ~36 minutes total (3.6 min average)
- Notable: Phase 6 (gap closure) added 5 min for what should have been caught earlier — invest in cross-phase wiring checks

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Execution Time | Phases | Key Change |
|-----------|---------------|--------|------------|
| v1.0 | 0.60 hours | 6 | Initial milestone — established bench harness patterns |

### Cumulative Quality

| Milestone | Tests | Requirements | Coverage |
|-----------|-------|-------------|----------|
| v1.0 | 162 | 32/32 | All satisfied, 8 tech debt items |

### Top Lessons (Verified Across Milestones)

1. Wire CLI flags end-to-end when introduced — silent no-ops accumulate as gap closure work
2. Cross-phase requirements need explicit tracking at each phase boundary
