# Project Research Summary

**Project:** LongMemEval Benchmark Integration
**Domain:** Benchmark harness for Rust memory engine (LongMemEval ICLR 2025)
**Researched:** 2026-03-15
**Confidence:** HIGH

## Executive Summary

The LongMemEval harness is a well-scoped engineering task, not a research problem. The benchmark protocol is fully specified in the paper and source repository, the existing Elephant/LoCoMo codebase provides ~80% of the required infrastructure, and the stack requires zero new dependencies. The core engineering challenge is scaling: LongMemEval demands 500 independent banks (one per question) vs LoCoMo's 10 shared banks. Every design decision flows from this structural difference. The recommended approach is a clean `run_instance()` implementation — not an adaptation of LoCoMo's `run_conversation()` — with a single-level semaphore-gated concurrency model and incremental artifact flushing to survive multi-hour runs.

The competitive landscape sets a clear target: 85%+ to be credible, 90%+ to be noteworthy. Hindsight (same architecture family) already hits 89-91.4%, and Elephant's LoCoMo result of 94.2% suggests the engine is capable. The main risks are measurement validity (judge model, dataset version, abstention scoring) rather than algorithmic capability. Getting the harness infrastructure right — category labels, judge prompts, timestamp propagation, and resume durability — is the path to a defensible published result.

Two design decisions must be locked before the first scored run and cannot be changed without invalidating results: (1) use `longmemeval_s_cleaned.json`, not the original dataset, to match post-2025 published baselines; (2) use GPT-4o as the default judge to match the paper's canonical evaluation protocol. Everything else can be tuned. The dataset schema is confirmed from official source code, the evaluation protocol is documented with >97% human agreement, and the architectural patterns are directly derived from the existing codebase.

## Key Findings

### Recommended Stack

No new Rust crate dependencies are needed. The existing `serde`/`serde_json`, `tokio`, `chrono`, `tabled`, `ulid`, `sqlx`, `reqwest`, and `futures` already cover every requirement. The `answer` field requires `serde_json::Value` (same workaround as LoCoMo's optional answer field) due to mixed string/integer types in the dataset. Two new binary targets are added to `Cargo.toml`: `longmemeval-bench` and `longmemeval-view`.

**Core technologies:**
- `serde_json::Value` for answer field: handles mixed-type answers — same pattern LoCoMo already uses
- `tokio::sync::Semaphore`: instance-level concurrency control — single axis, simpler than LoCoMo's two-axis model
- `sqlx::PgPoolOptions::new().max_connections(20)`: explicit pool sizing — default pool of 10 is insufficient for concurrent bank operations across 500 instances
- `chrono`: date parsing for `haystack_dates` parallel array — requires zip-with-length-assertion against `haystack_sessions`
- `ElephantRuntime` (existing): retain, consolidate, reflect — imported directly, same as LoCoMo

**What NOT to add:** HuggingFace SDK (manual download), `clap` (LoCoMo uses manual arg parsing), any Python interop (reimplement judge in Rust).

### Expected Features

**Must have (table stakes):**
- All 500 questions scored — partial runs are not credible; denominator must always be 500 (not 470 after excluding abstention)
- Per-category accuracy breakdown — 7 types: single-session-user, single-session-assistant, single-session-preference, multi-session, knowledge-update, temporal-reasoning, abstention
- GPT-4o judge — the paper's canonical judge; using Sonnet requires explicit disclosure and calibration
- Per-instance bank isolation — one bank per question, no sharing; this is non-negotiable for protocol compliance
- LongMemEval-S support — the standard evaluation setting; every published result uses S
- `question_date` passthrough to reflect — temporal reasoning questions require the "current date" context
- run/ingest/qa subcommands — iteration speed is critical; rebuilding 500 banks is too expensive to do repeatedly
- Full manifest/provenance and three-artifact output — reproducibility contract, direct carry-over from LoCoMo
- Abstention scoring — 30 false-premise questions must be scored with a separate judge prompt

**Should have (competitive):**
- Smoke profile for fast iteration — essential for development velocity with 500-question scale
- Stage-level cost metrics — low-cost competitive advantage, already built for LoCoMo
- Retrieval provenance in debug sidecar — enables post-hoc evidence recall analysis without rerunning
- merge subcommand — batch execution across multiple runs, needed for reliability on long runs

**Defer (v2+):**
- LongMemEval-M support — ~750M tokens total, extremely expensive; no competitor has published M results
- Evidence session tracking / Recall@k metrics — data is in the sidecar; compute after first baseline exists
- Multi-judge variance analysis — run GPT-4o first, then explore alternatives
- Cross-benchmark comparison view — defer until both LoCoMo and LongMemEval have canonical results

### Architecture Approach

The harness reuses ElephantRuntime directly (same as LoCoMo) but replaces the two-level conversation×question loop with a single-level instance loop. Each "instance" is a question with its own bank: create bank → ingest sessions sequentially → consolidate → reflect with question_date → judge → record. Concurrency is controlled by a single semaphore (`--instance-jobs`). Artifact output follows the LoCoMo three-artifact pattern (summary JSON, question JSONL, debug JSONL) but uses `per_instance` instead of `per_conversation` schema. The LoCoMo harness code is not reused — patterns are copied, not abstracted, to keep the schemas independent.

**Major components:**
1. Dataset Loader — parse `Vec<LongMemEvalInstance>` from JSON, zip `haystack_sessions` with `haystack_dates` with length assertion
2. Instance Runner (`run_instance()`) — full lifecycle: bank create → session ingest → consolidate → reflect → judge
3. Shared Results (`Arc<Mutex<SharedResults>>`) — incremental artifact flushing keyed by `question_id`, atomic writes
4. Judge — two prompts: factual judge (matches answer) and abstention judge (detects refusal); GPT-4o canonical
5. CLI + Profile System — run/ingest/qa/merge subcommands; smoke/full-s/full-m profiles
6. View Tool (`longmemeval-view`) — separate binary, independent of LoCoMo view tool

### Critical Pitfalls

1. **Category-label mapping bug** — use `question_type` strings directly as category keys; never create a numeric enum mapping. This exact bug (swapped single-hop/multi-hop) already happened in the Hindsight LoCoMo comparison. With 7 categories instead of 5, the surface area is larger.

2. **Abstention scoring omitted or wrong** — implement two judge prompts (factual and abstention); always count all 500 questions in the denominator. LoCoMo's pattern of excluding category 5 must NOT be carried forward — LongMemEval scores all 500.

3. **Timestamps not propagated** — zip `haystack_dates` with `haystack_sessions` with a length assertion; pass `question_date` to reflect as "current date" context. Temporal reasoning accuracy goes near zero if timestamps are wrong, wasting Elephant's biggest accuracy gain (temporal annotations in consolidation).

4. **Wrong dataset version** — hardcode `_cleaned` filename suffix; record dataset fingerprint in manifest; fail loudly if filename doesn't match. The original dataset was deprecated in September 2025.

5. **500 banks exhaust Postgres connections** — configure pool explicitly to `max_connections(20+)`; implement `--instance-jobs` concurrency limit; write bank mapping incrementally. Default pool of 10 connections is insufficient for concurrent bank operations.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Dataset Foundation
**Rationale:** Everything depends on correct dataset parsing. Failures here invalidate all downstream work and are cheap to catch early with unit tests against the real dataset file.
**Delivers:** `LongMemEvalInstance` struct with serde deserialization, session/turn types, question type parsing, dataset loading and validation
**Addresses:** table stakes for all 500 questions, per-category breakdown
**Avoids:** Category-label mapping bug (Pitfall 1), wrong dataset version (Pitfall 5), timestamp propagation bug (Pitfall 4), `has_answer` misinterpretation (Pitfall 11)
**Research flag:** None — schema is confirmed from official source code with HIGH confidence

### Phase 2: Ingestion Infrastructure
**Rationale:** Ingestion is the novel code; QA reuses existing reflect/judge patterns. Ingestion must work correctly before investing in QA tooling. Infrastructure decisions made here (pool size, concurrency model, bank naming) affect every subsequent phase.
**Delivers:** `run_instance()` ingest path, bank creation, session formatting with date prefix, per-instance bank stats, Postgres pool configuration
**Addresses:** per-instance bank isolation, `question_date` passthrough setup, `Date:` prefix for temporal annotations
**Avoids:** 500 banks exhaust Postgres (Pitfall 3), wall-clock explosion (Pitfall 7), `has_answer` filter during ingestion (Pitfall 11)
**Research flag:** None — directly mirrors LoCoMo patterns with well-understood modifications

### Phase 3: CLI, Profiles, and Artifact Infrastructure
**Rationale:** Output infrastructure must exist before QA can record results. Building this phase before QA avoids the trap of writing QA code without a place to store results.
**Delivers:** run/ingest/qa/merge subcommands, smoke/full-s/full-m profiles, summary JSON + question JSONL + debug JSONL output, SharedResults with incremental flushing, manifest with dataset fingerprint
**Addresses:** run/ingest/qa subcommands (table stakes), full manifest/provenance (table stakes), three-artifact output (table stakes)
**Avoids:** Resume/merge complexity (Pitfall 8), consolidation strategy inconsistency (Pitfall 9)
**Research flag:** None — established patterns from LoCoMo with schema changes only

### Phase 4: QA Path (reflect + judge)
**Rationale:** Extends the ingest-only instance runner with the full lifecycle. Judge implementation is critical to measurement validity and must be done correctly before the first scored run.
**Delivers:** reflect integration with question_date context, two judge prompts (factual + abstention), per-category accuracy breakdown, overall accuracy metric, token F1 scoring
**Addresses:** all 500 questions scored, GPT-4o judge, per-category accuracy, abstention scoring, question_date passthrough
**Avoids:** Abstention scoring wrong (Pitfall 2), judge model mismatch (Pitfall 6), category totals wrong (Pitfall 1)
**Research flag:** None for factual judge (port from LoCoMo). The abstention judge prompt should be ported directly from `evaluate_qa.py` rather than invented — this is known, well-documented behavior.

### Phase 5: Concurrency, Resume, and Polish
**Rationale:** Get the serial path completely correct first, then add concurrency. Debugging race conditions while core logic is still being developed multiplies complexity.
**Delivers:** `--instance-jobs N` parallelism with semaphore, resume from ingest artifact (skip already-completed instances), merge subcommand with question-level deduplication, progress reporting
**Addresses:** merge subcommand (differentiator), smoke profile (differentiator)
**Avoids:** Wall-clock explosion (Pitfall 7), resume losing ingestion work (Pitfall 8), merge with overlapping questions (Pitfall 14)
**Research flag:** None — standard concurrency patterns from LoCoMo, well-understood

### Phase 6: View Tool and First Full Run
**Rationale:** The view tool is needed to interpret results from the first full S run. The full run validates that all phases work end-to-end at scale and produces the first publishable number.
**Delivers:** `longmemeval-view` binary, per-category display with judge model flagging, first full LongMemEval-S scored run
**Addresses:** stage-level cost metrics (differentiator), cross-benchmark comparison (deferred to v2)
**Avoids:** Judge model mismatch going unnoticed in published results (Pitfall 6)
**Research flag:** None — view tool is standard display code; first full run may surface performance issues to address

### Phase Ordering Rationale

- Dataset parsing first because every subsequent component depends on correctly-typed data; this is the only phase with zero Elephant infrastructure dependencies and is fully testable in isolation
- Ingestion before QA because ingestion is the novel code; QA reuses established reflect/judge patterns
- CLI/artifacts before QA because you need somewhere to store results before writing QA code
- Concurrency last because serial correctness must be established before introducing concurrency; race conditions in still-evolving code are very expensive to debug
- View tool and full run last because they validate the complete pipeline; the view tool is only useful once results exist

### Research Flags

Phases with standard patterns (skip research-phase for all phases):
- **Phase 1:** LongMemEval schema is confirmed from official source code (`run_retrieval.py`) — HIGH confidence
- **Phase 2:** Session formatting and bank lifecycle mirror LoCoMo exactly — first-party source
- **Phase 3:** Three-artifact pattern, manifest, SharedResults all directly ported from LoCoMo — first-party source
- **Phase 4:** Factual judge ported from LoCoMo; abstention judge ported from `evaluate_qa.py` — both well-documented
- **Phase 5:** Semaphore concurrency model established in LoCoMo — first-party source
- **Phase 6:** View tool is display code; no complex logic

The one area requiring careful validation before Phase 1 closes: turn structure field names (`role`, `content`, `has_answer`) confirmed from source code inspection but should be cross-checked against the actual downloaded JSON file before implementing serde types. This is a MEDIUM confidence item that becomes HIGH on first successful load.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All dependencies already in Cargo.toml and proven in LoCoMo; dataset schema confirmed from official source code |
| Features | HIGH | Evaluation protocol documented in paper with >97% human agreement; competitive landscape well-measured |
| Architecture | HIGH | Directly derived from existing LoCoMo harness (4340 lines, first-party source); LongMemEval structural differences are clear and well-understood |
| Pitfalls | HIGH | Most pitfalls are from first-party experience (category-label bug already happened; LoCoMo patterns are known) or confirmed from official evaluation scripts |

**Overall confidence:** HIGH

### Gaps to Address

- **Exact turn field names in JSON**: `role`, `content`, `has_answer` confirmed from Python source inspection (MEDIUM confidence). Validate against actual downloaded `longmemeval_s_cleaned.json` before finalizing serde types. Not a blocker — discrepancy would surface immediately on first parse.
- **`answer_session_ids` indexing convention**: Confirmed as 0-indexed from Python source, but should add an explicit assertion during implementation that checks a known instance manually.
- **Consolidation mode for full-s profile**: Research recommends `end` mode (matching LoCoMo's best setting), but this should be validated with a smoke run before committing to a full 500-question run. `per-session` may be better for M.
- **Shared code extraction boundary**: Research recommends duplicating small functions (`llm_judge`, `token_f1`, `build_judge_client`) rather than extracting a shared module. This decision should be revisited if the functions diverge significantly during implementation.
- **Connection pool ceiling**: Research recommends `max_connections(20)` but the right number depends on Postgres's `max_connections` setting in the deployment environment. Tune empirically during Phase 2.

## Sources

### Primary (HIGH confidence)
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval) — dataset format (`run_retrieval.py`), evaluation scripts (`evaluate_qa.py`, `print_qa_metrics.py`), README
- [LongMemEval paper (arXiv:2410.10813)](https://arxiv.org/abs/2410.10813) — benchmark methodology, question types, evaluation protocol, baseline results
- `bench/locomo/locomo.rs` (4340 lines, first-party) — existing harness patterns, category mapping, judge, resume, consolidation
- `bench/locomo/internal/hindsight-conv26-comparison-2026-03-10.md` (first-party) — category-label mapping bug, judge failure experience
- `bench/locomo/internal/bench-instrumentation.md` (first-party) — in-process metering design decisions
- Elephant MEMORY.md (first-party) — prompt tuning history, reflect agent behavior, consolidation architecture

### Secondary (MEDIUM confidence)
- [longmemeval-cleaned Dataset (HuggingFace)](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) — cleaned dataset files, mixed-type answer field issue, September 2025 update
- [Backboard LongMemEval Results](https://github.com/Backboard-io/Backboard-longmemEval-results) — 93.4% accuracy, per-question bank isolation approach, multi-judge variance
- [Hindsight benchmarks](https://github.com/vectorize-io/hindsight-benchmarks) — 89.0-91.4% scores, judge protocol

### Tertiary (MEDIUM-LOW confidence)
- [Mastra Observational Memory](https://mastra.ai/research/observational-memory) — 94.9% with GPT-5-mini, alternative architecture
- [EverMemOS paper (arXiv:2601.02163)](https://arxiv.org/pdf/2601.02163.pdf) — 83.0%, stage-level cost breakdown
- [Emergence.ai blog](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag) — RAG parameter sensitivity, implementation experience
- [Zep/Graphiti blog](https://blog.getzep.com/state-of-the-art-agent-memory/) — 71.2%, alternative memory architecture

---
*Research completed: 2026-03-15*
*Ready for roadmap: yes*
