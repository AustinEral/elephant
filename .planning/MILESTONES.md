# Milestones

## v1.0 LongMemEval Benchmark Integration (Shipped: 2026-03-15)

**Phases completed:** 6 phases, 10 plans
**Lines of code:** 5,131 Rust (bench modules)
**Timeline:** 18 days (2026-02-25 → 2026-03-15)
**Git range:** ba37559..2dd4d0c (15 feat commits)

**Key accomplishments:**
- Shared bench infrastructure (FNV fingerprinting, JSONL I/O) extracted from LoCoMo
- LongMemEval dataset parsing with 7 question types and mixed-type answer coercion
- Per-instance bank ingestion with configurable consolidation modes (end, per-session, off)
- Full CLI with run/ingest/qa subcommands, profile system, and three-artifact output with manifest
- Dual judge evaluation with 5 prompt variants (factual, temporal, knowledge-update, preference, abstention) and per-category accuracy
- Semaphore-gated concurrency, resume support, standalone view tool, and temporal context wiring

**Tech debt accepted:**
- 8 items across 4 phases (compiler warnings, unused fields, empty stage_metrics) — see milestone audit

---

