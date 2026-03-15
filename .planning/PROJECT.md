# LongMemEval Benchmark Integration

## What This Is

A LongMemEval benchmark harness for the Elephant memory engine, enabling rigorous evaluation against the LongMemEval benchmark (arXiv:2410.10813) — 500 curated questions testing five core long-term memory abilities across multi-session conversation histories. This gives Elephant publishable, verifiable results comparable to other agentic memory systems (Mnemis, EverMemOS, Backboard, Hindsight).

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

### Active

- [ ] LongMemEval harness binary (`longmemeval-bench`) with run/ingest/qa/merge subcommands
- [ ] Support both LongMemEvalS (~50 sessions, ~115k tokens) and LongMemEvalM (~500 sessions, ~1.5M tokens)
- [ ] Dataset loading from manually downloaded HuggingFace files (`xiaowu0162/longmemeval-cleaned`)
- [ ] All 500 questions scored, including 30 false-premise (abstention) questions
- [ ] Seven question categories: single-session-user, single-session-assistant, single-session-preference, multi-session, knowledge-update, temporal-reasoning, false-premise
- [ ] Per-question bank ingestion (each question has its own conversation history)
- [ ] Session-level and turn-level ingestion modes (matching LoCoMo flexibility)
- [ ] Consolidation modes (end, per-session, off)
- [ ] Profile system (full-s, full-m, smoke) for reproducible benchmark shapes
- [ ] Configurable judge (default Sonnet, overridable to GPT-4o for paper comparability)
- [ ] Three-artifact output matching LoCoMo contract (summary JSON, question JSONL, debug JSONL)
- [ ] Manifest with full reproducibility contract (dataset fingerprint, prompt hashes, runtime config, git commit)
- [ ] Stage-level metrics (retain, consolidate, reflect, judge)
- [ ] Per-category accuracy breakdown across all 7 question types
- [ ] Merge subcommand for assembling full runs from batch slices
- [ ] Separate view tool for LongMemEval artifacts (`longmemeval-view` binary)
- [ ] Concurrency controls (question-jobs, instance-jobs) for parallel execution
- [ ] Resume support via qa subcommand (reuse banks from ingest artifact)

### Out of Scope

- Auto-downloading dataset from HuggingFace — manual download keeps things simple and avoids HF SDK dependency
- Extending the existing LoCoMo view tool — separate viewer to keep concerns decoupled
- Custom history compilation (LongMemEval's algorithm for generating variable-length histories) — we use the fixed S and M datasets
- Implementing LongMemEval's proposed MemoryChat optimizations (session decomposition, fact-augmented key expansion, time-aware query expansion) — we benchmark Elephant's existing pipeline, not theirs
- Retrieval-only metrics (Recall@k, NDCG@k) — focus on end-to-end QA accuracy like LoCoMo
- Modifying Elephant's reflect agent for abstention — if it can't handle false-premise questions naturally, that's a separate follow-up

## Context

**LongMemEval** (Wu et al., 2024) evaluates five core long-term memory abilities:
1. **Information Extraction** — recall specific details from conversation history
2. **Multi-Session Reasoning** — synthesize information across multiple sessions
3. **Knowledge Updates** — recognize changes in user information over time
4. **Temporal Reasoning** — reason about timestamps and time references
5. **Abstention** — refuse to answer when information wasn't mentioned

The benchmark uses 500 manually curated questions embedded in scalable conversation histories. Each question has its own conversation history (unlike LoCoMo where questions share 10 conversations). Two fixed settings: S (~50 sessions) and M (~500 sessions).

**Existing infrastructure**: The LoCoMo harness (`bench/locomo/locomo.rs`, 148KB) establishes patterns for CLI parsing, profile-driven config, artifact output, metrics instrumentation, judge evaluation, and merge validation. The LongMemEval harness should follow these patterns where sensible without coupling the two implementations.

**Dataset**: `longmemeval_s_cleaned.json` and `longmemeval_m_cleaned.json` from HuggingFace (`xiaowu0162/longmemeval-cleaned`). Each instance has: `question_id`, `question_type`, `question`, `question_date`, `answer`, `haystack_sessions` (array of turns with `role`, `content`, `has_answer`), `haystack_dates`, `answer_session_ids`.

**Publication goal**: Results must meet the same standard as LoCoMo series1 — full artifact set, reproducible manifest, category breakdown, stage metrics, cost disclosure, git provenance. This enables credible comparison against published LongMemEval baselines (GPT-4o: 60.6%, commercial chat assistants: 32-58%).

## Constraints

- **Tech stack**: Rust, same crate as Elephant — new binary target in Cargo.toml
- **Dataset**: Manual download to `data/` directory (no HF SDK dependency)
- **Code sharing**: Share common infrastructure where it makes good sense (judge, metrics, artifact writing), but don't couple the LoCoMo and LongMemEval harnesses
- **Artifact format**: Compatible with LoCoMo's three-artifact pattern but with LongMemEval-specific fields (question_type categories, per-instance bank tracking)
- **Ingestion model**: Each of the 500 questions requires its own bank (own conversation history), unlike LoCoMo's 10 shared conversations — this is the main architectural difference

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Mirror LoCoMo subcommands (run/ingest/qa/merge) | Consistency across benchmarks, proven workflow | — Pending |
| Include all 500 questions including abstention | See how reflect handles false premises naturally; fix later if needed | — Pending |
| Configurable judge (not GPT-4o only) | Matches LoCoMo pattern; can use GPT-4o for paper comparability when needed | — Pending |
| Separate view tool (not extending LoCoMo viewer) | Keeps concerns decoupled, each benchmark maintained independently | — Pending |
| Manual dataset download | Avoids HF SDK dependency, simple `wget` instructions | — Pending |
| Both S and M settings | S for iteration speed, M for stress testing / full publication claims | — Pending |
| Share infra pragmatically | Common judge/metrics/artifact code where sensible, independent harness files | — Pending |

---
*Last updated: 2026-03-15 after initialization*
