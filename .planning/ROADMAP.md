# Roadmap: LongMemEval Benchmark Integration

## Overview

Build a LongMemEval benchmark harness for the Elephant memory engine. The harness loads the LongMemEval dataset (500 questions with per-question conversation histories), ingests each into an independent bank, runs the reflect agent, scores with a configurable LLM judge, and outputs publication-quality artifacts. The journey goes: parse the dataset correctly, build the novel per-instance ingestion pipeline, wire up the CLI and artifact infrastructure, add the QA evaluation path, then polish with concurrency, resume, and a view tool.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Dataset Foundation** - Parse and validate LongMemEval dataset with correct types, categories, and fingerprinting (completed 2026-03-15)
- [ ] **Phase 2: Ingestion Pipeline** - Per-instance bank creation, session ingestion with timestamps, and consolidation
- [ ] **Phase 3: CLI and Artifact Infrastructure** - Binary, subcommands, profiles, three-artifact output, manifest, and config overlay
- [ ] **Phase 4: Evaluation Path** - Reflect with temporal context, dual judge prompts, per-category accuracy scoring
- [ ] **Phase 5: Concurrency, Resume, and View Tool** - Parallel execution, resume from ingest artifacts, and standalone view binary

## Phase Details

### Phase 1: Dataset Foundation
**Goal**: User can load and validate any LongMemEval dataset file and get correctly typed, categorized instances ready for ingestion
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07
**Success Criteria** (what must be TRUE):
  1. Running the harness with `longmemeval_s_cleaned.json` parses all 500 instances without error
  2. Running the harness with `longmemeval_m_cleaned.json` parses all instances without error
  3. Each parsed instance has a correct `question_type` matching one of the 7 LongMemEval categories
  4. Mixed-type `answer` fields (string and integer) are coerced to strings without data loss
  5. Dataset fingerprint (FNV1a-64) is deterministic across runs on the same file
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md -- Extract bench/common/ shared infrastructure, create LongMemEval types/loading/validation with unit tests
- [ ] 01-02-PLAN.md -- Integration tests for real dataset file loading (S and M datasets)

### Phase 2: Ingestion Pipeline
**Goal**: User can ingest a LongMemEval instance's full conversation history into an isolated bank with correct timestamps and consolidation
**Depends on**: Phase 1
**Requirements**: INGEST-01, INGEST-02, INGEST-03, INGEST-04, INGEST-05
**Success Criteria** (what must be TRUE):
  1. Each instance gets its own bank (no cross-contamination between questions)
  2. Sessions are ingested sequentially with `[Date: YYYY-MM-DD]` prefix from `haystack_dates`
  3. Consolidation runs in the configured mode (end, per-session, or off) after ingestion completes
  4. Postgres connection pool is explicitly sized to handle concurrent bank operations without exhaustion
**Plans**: TBD

Plans:
- [ ] 02-01: TBD

### Phase 3: CLI and Artifact Infrastructure
**Goal**: User can run the harness via CLI with subcommands, profiles, and get reproducible three-artifact output
**Depends on**: Phase 2
**Requirements**: CLI-01, CLI-02, CLI-03, CLI-04, CLI-06, CLI-07, CLI-08, CLI-09, CLI-11
**Success Criteria** (what must be TRUE):
  1. `longmemeval-bench run` executes the full pipeline (ingest + consolidate + QA) and writes results
  2. `longmemeval-bench ingest` creates banks and writes an ingest artifact (bank mappings) without running QA
  3. `longmemeval-bench qa` scores against existing banks from an ingest artifact
  4. Profile selection (`smoke`, `full-s`, `full-m`) controls dataset, instance subset, and consolidation mode
  5. Output artifacts (summary JSON, question JSONL, debug JSONL) land in `bench/longmemeval/results/local/` by default
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Evaluation Path
**Goal**: User can score all 500 questions including abstention, with per-category accuracy breakdown and configurable judge
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06
**Success Criteria** (what must be TRUE):
  1. Reflect agent receives `question_date` as temporal context for each question
  2. Standard questions are judged with the factual prompt; false-premise questions (identified by `_abs` suffix) are judged with the abstention prompt
  3. Summary artifact shows per-category accuracy across all 7 question types with question counts
  4. Overall accuracy is computed as correct / 500 (all questions in denominator, no exclusions)
  5. Judge model defaults to GPT-4o and is overridable via `--judge-model`
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

### Phase 5: Concurrency, Resume, and View Tool
**Goal**: User can run instances in parallel, resume interrupted runs, and inspect results with a standalone view tool
**Depends on**: Phase 4
**Requirements**: CLI-05, CLI-10, VIEW-01, VIEW-02, VIEW-03
**Success Criteria** (what must be TRUE):
  1. `--instance-jobs N` controls parallel instance processing via semaphore
  2. `qa` subcommand reuses bank_ids from an ingest artifact to skip re-ingestion
  3. `longmemeval-view` binary displays per-category accuracy with question counts from a results artifact
  4. View tool shows config, summary, and per-question table in single-artifact mode
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Dataset Foundation | 2/2 | Complete   | 2026-03-15 |
| 2. Ingestion Pipeline | 0/? | Not started | - |
| 3. CLI and Artifact Infrastructure | 0/? | Not started | - |
| 4. Evaluation Path | 0/? | Not started | - |
| 5. Concurrency, Resume, and View Tool | 0/? | Not started | - |
