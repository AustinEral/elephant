# Requirements: LongMemEval Benchmark Integration

**Defined:** 2026-03-15
**Core Value:** Produce publication-quality LongMemEval results backing Elephant's claims as a serious competitor to other agentic memory systems

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Dataset & Parsing

- [ ] **DATA-01**: Harness loads `longmemeval_s_cleaned.json` (~50 sessions per instance, ~115k tokens)
- [ ] **DATA-02**: Harness loads `longmemeval_m_cleaned.json` (~500 sessions per instance, ~1.5M tokens)
- [x] **DATA-03**: All 500 questions parsed with correct question_type categorization (7 types)
- [x] **DATA-04**: Mixed-type `answer` field handled via serde_json::Value coercion to string
- [x] **DATA-05**: Dataset fingerprinting (FNV1a-64 hash) stored in manifest for reproducibility
- [x] **DATA-06**: `question_date` parsed and passed to reflect agent as temporal context
- [x] **DATA-07**: `haystack_sessions` and `haystack_dates` zip-validated (equal length assertion)

### Ingestion

- [ ] **INGEST-01**: Per-instance bank isolation — one bank created per question (500 banks for full run)
- [ ] **INGEST-02**: Session-level ingestion with date prefix (e.g., `[Date: 2023-01-15] content`)
- [ ] **INGEST-03**: Consolidation modes supported: end (default), per-session, off
- [ ] **INGEST-04**: Sessions ingested sequentially per instance with timestamps from `haystack_dates`
- [ ] **INGEST-05**: Postgres connection pool explicitly sized for concurrent bank operations

### Evaluation

- [ ] **EVAL-01**: Configurable LLM judge — GPT-4o default, overridable via `--judge-model`
- [ ] **EVAL-02**: Two judge prompt variants: factual (for standard questions) and abstention (for false-premise)
- [ ] **EVAL-03**: All 500 questions scored including 30 false-premise abstention questions
- [ ] **EVAL-04**: Per-category accuracy breakdown across 7 question types (single-session-user, single-session-assistant, single-session-preference, multi-session, knowledge-update, temporal-reasoning, abstention)
- [ ] **EVAL-05**: Overall accuracy metric (correct / total across all 500 questions)
- [ ] **EVAL-06**: Abstention questions identified by `_abs` suffix on question_id and scored with abstention judge

### CLI & Infrastructure

- [ ] **CLI-01**: `longmemeval-bench` binary with `run` subcommand (ingest + consolidate + QA)
- [ ] **CLI-02**: `ingest` subcommand (ingest + consolidate only, no QA)
- [ ] **CLI-03**: `qa` subcommand (score against existing banks from ingest artifact)
- [ ] **CLI-04**: Profile system with `smoke` (small subset), `full-s` (all 500 on S dataset), `full-m` (all 500 on M dataset)
- [ ] **CLI-05**: `--instance-jobs N` concurrency control for parallel instance processing
- [ ] **CLI-06**: `--config` JSON overlay on top of profile for local tuning
- [ ] **CLI-07**: `--instance` flag to run specific question instances (repeatable)
- [ ] **CLI-08**: Three-artifact output: summary JSON, question JSONL sidecar, debug JSONL sidecar
- [ ] **CLI-09**: Manifest with full reproducibility contract (dataset fingerprint, prompt hashes, runtime config, git commit, CLI invocation)
- [ ] **CLI-10**: Resume via `qa` subcommand reusing bank_ids from ingest artifact
- [ ] **CLI-11**: Results default to `bench/longmemeval/results/local/`, promotable via `--out`

### View Tool

- [ ] **VIEW-01**: Separate `longmemeval-view` binary (independent of LoCoMo view)
- [ ] **VIEW-02**: Per-category accuracy display with question counts
- [ ] **VIEW-03**: Single-artifact view mode with config, summary, and question tables

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Evaluation

- **EVAL-07**: Token F1 secondary metric (reuse from LoCoMo)
- **EVAL-08**: Evidence session tracking — compute recall against `answer_session_ids`
- **EVAL-09**: Stage-level cost metrics per stage (retain, consolidate, reflect, judge)
- **EVAL-10**: Multi-judge variance analysis (run with multiple judge models, report range)

### CLI & Infrastructure

- **CLI-12**: `merge` subcommand for combining compatible batch artifacts
- **CLI-13**: Progress reporting during long runs (instance count, elapsed time, ETA)
- **CLI-14**: Consolidation mode ablation profiles (end vs per-session comparison)

### View Tool

- **VIEW-04**: Comparison mode (side-by-side artifact diff with delta highlighting)
- **VIEW-05**: Bank stats summary (aggregate across 500 instances: mean, p50, p95)
- **VIEW-06**: Cross-benchmark comparison with LoCoMo results

### Advanced

- **ADV-01**: Turn-level ingestion mode (alternative to session-level)
- **ADV-02**: Per-instance bank construction stats in artifacts

## Out of Scope

| Feature | Reason |
|---------|--------|
| Auto-download from HuggingFace | Adds HF SDK dependency for a one-time download; manual wget is simpler |
| Custom history compilation | LongMemEval's algorithm for variable-length histories — we use fixed S and M datasets |
| MemoryChat optimizations | Session decomposition, fact-augmented key expansion, time-aware query expansion — benchmarks their system, not ours |
| Retrieval-only evaluation mode | Recall@k/NDCG@k require separate retrieval path; evidence tracking from debug sidecar is sufficient |
| Extending LoCoMo view tool | Coupling viewers creates maintenance burden; separate binary keeps concerns clean |
| Modifying reflect for abstention | If Elephant can't handle false-premise questions naturally, that's an engine change, not a harness feature |
| Shared banks across questions | Violates benchmark protocol; each question has independent evaluation context |
| Python interop for evaluation | Reimplement judge in Rust to match LoCoMo pattern; don't call upstream evaluate_qa.py |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| DATA-06 | Phase 1 | Complete |
| DATA-07 | Phase 1 | Complete |
| INGEST-01 | Phase 2 | Pending |
| INGEST-02 | Phase 2 | Pending |
| INGEST-03 | Phase 2 | Pending |
| INGEST-04 | Phase 2 | Pending |
| INGEST-05 | Phase 2 | Pending |
| EVAL-01 | Phase 4 | Pending |
| EVAL-02 | Phase 4 | Pending |
| EVAL-03 | Phase 4 | Pending |
| EVAL-04 | Phase 4 | Pending |
| EVAL-05 | Phase 4 | Pending |
| EVAL-06 | Phase 4 | Pending |
| CLI-01 | Phase 3 | Pending |
| CLI-02 | Phase 3 | Pending |
| CLI-03 | Phase 3 | Pending |
| CLI-04 | Phase 3 | Pending |
| CLI-05 | Phase 5 | Pending |
| CLI-06 | Phase 3 | Pending |
| CLI-07 | Phase 3 | Pending |
| CLI-08 | Phase 3 | Pending |
| CLI-09 | Phase 3 | Pending |
| CLI-10 | Phase 5 | Pending |
| CLI-11 | Phase 3 | Pending |
| VIEW-01 | Phase 5 | Pending |
| VIEW-02 | Phase 5 | Pending |
| VIEW-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 32 total
- Mapped to phases: 32
- Unmapped: 0

---
*Requirements defined: 2026-03-15*
*Last updated: 2026-03-15 after roadmap creation*
