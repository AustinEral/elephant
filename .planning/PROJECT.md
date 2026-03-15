# LongMemEval Benchmark Integration

## What This Is

A LongMemEval benchmark harness for the Elephant memory engine — a complete evaluation pipeline that loads 500 curated questions from LongMemEval (arXiv:2410.10813), ingests per-instance conversation histories into isolated banks, runs the reflect agent with temporal context, scores with a configurable LLM judge (5 prompt variants), and outputs publication-quality three-artifact results with full reproducibility manifest.

## Core Value

Produce publication-quality benchmark results that back Elephant's claims as a serious competitor to other agentic memory systems, with the same rigor and artifact hygiene established by the LoCoMo harness.

## Requirements

### Validated

- ✓ LoCoMo benchmark harness with run/ingest/qa/merge subcommands — existing
- ✓ Profile-driven configuration (full, smoke, legacy-raw) — existing
- ✓ Three-artifact output (summary JSON, question JSONL, debug JSONL) — existing
- ✓ Manifest-based reproducibility (prompt hashes, runtime config, dataset fingerprint) — existing
- ✓ Stage-level metrics instrumentation — existing
- ✓ Configurable LLM judge with binary accuracy scoring — existing
- ✓ View/comparison tool for LoCoMo artifacts — existing
- ✓ Evidence-aware evaluation (retrieval vs synthesis gap analysis) — existing
- ✓ LongMemEval harness binary (`longmemeval-bench`) with run/ingest/qa subcommands — v1.0
- ✓ Support both LongMemEvalS (~50 sessions) and LongMemEvalM (~500 sessions) — v1.0
- ✓ Dataset loading with validation and FNV1a-64 fingerprinting — v1.0
- ✓ All 500 questions scored including 30 false-premise (abstention) questions — v1.0
- ✓ Seven question categories with per-category accuracy breakdown — v1.0
- ✓ Per-question bank ingestion (isolated bank per question) — v1.0
- ✓ Consolidation modes (end, per-session, off) — v1.0
- ✓ Profile system (smoke, full-s, full-m) — v1.0
- ✓ Configurable judge (GPT-4o default, overridable via --judge-model) — v1.0
- ✓ Three-artifact output with manifest reproducibility contract — v1.0
- ✓ Per-category accuracy breakdown across all 7 question types — v1.0
- ✓ Separate view tool (`longmemeval-view`) — v1.0
- ✓ Concurrency controls (--instance-jobs N) with Postgres pool auto-sizing — v1.0
- ✓ Resume support via qa subcommand (reuse banks from ingest artifact) — v1.0
- ✓ Temporal context wiring (question_date forwarded to reflect agent) — v1.0
- ✓ Session limit support (--session-limit N) — v1.0

### Active

(No active requirements — next milestone TBD)

### Out of Scope

- Auto-downloading dataset from HuggingFace — manual download keeps things simple
- Extending the existing LoCoMo view tool — separate viewer keeps concerns decoupled
- Custom history compilation — we use fixed S and M datasets
- MemoryChat optimizations — we benchmark Elephant's pipeline, not theirs
- Retrieval-only metrics (Recall@k, NDCG@k) — focus on end-to-end QA accuracy
- Modifying reflect agent for abstention — if it can't handle false-premise naturally, that's a separate follow-up
- Token F1 secondary metric (deferred to v2)
- Evidence session tracking against answer_session_ids (deferred to v2)
- Stage-level cost metrics per stage (deferred to v2)
- Merge subcommand for combining batch artifacts (deferred to v2)
- Comparison mode in view tool (deferred to v2)
- Turn-level ingestion mode (deferred to v2)

## Context

**Current state:** v1.0 shipped. 5,131 LOC Rust across bench modules. Two binaries: `longmemeval-bench` (2,869 lines) and `longmemeval-view` (1,053 lines). Shared infrastructure in `bench/common/` (fingerprinting, JSONL I/O, judge module). 162 tests passing.

**LongMemEval** (Wu et al., 2024) evaluates five core long-term memory abilities: information extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention. 500 manually curated questions with scalable conversation histories. Two fixed settings: S (~50 sessions) and M (~500 sessions).

**Publication goal**: Results must meet the same standard as LoCoMo series1 — full artifact set, reproducible manifest, category breakdown, stage metrics, cost disclosure, git provenance. Enables credible comparison against published baselines (GPT-4o: 60.6%, commercial chat assistants: 32-58%).

## Constraints

- **Tech stack**: Rust, same crate as Elephant — binary targets in Cargo.toml
- **Dataset**: Manual download to `data/` directory
- **Code sharing**: Common infrastructure in bench/common/, independent harness files
- **Artifact format**: Compatible with LoCoMo three-artifact pattern with LongMemEval-specific fields

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Mirror LoCoMo subcommands (run/ingest/qa) | Consistency across benchmarks, proven workflow | ✓ Good |
| Include all 500 questions including abstention | See how reflect handles false premises naturally | ✓ Good |
| Configurable judge (GPT-4o default) | Matches LoCoMo pattern; paper comparability when needed | ✓ Good |
| Separate view tool | Keeps concerns decoupled, independent maintenance | ✓ Good |
| Manual dataset download | Avoids HF SDK dependency | ✓ Good |
| Both S and M settings | S for iteration, M for stress testing / publication | ✓ Good |
| Share infra pragmatically | Common judge/IO/fingerprint, independent harness | ✓ Good |
| Delegating wrappers for extracted common modules | Minimizes diff in locomo.rs during extraction | ✓ Good |
| 6 QuestionType variants + is_abstention() helper | Maps cleanly to upstream _abs suffix convention | ✓ Good |
| elephant::error::Result for bench code | anyhow is dev-only dependency | ✓ Good |
| Pool sizing: min(instance_jobs * 3, 50) | Prevents pool exhaustion under concurrency | ✓ Good |
| Raw question_date as temporal_context | Preserves time-of-day precision from dataset | ✓ Good |
| [Current date: ...] prefix in reflect | Distinct from ingested [Date: ...] prefixes | ✓ Good |
| View tool uses String keys + #[serde(default)] everywhere | Forward-compatible, reads any version of artifacts | ✓ Good |

---
*Last updated: 2026-03-15 after v1.0 milestone*
