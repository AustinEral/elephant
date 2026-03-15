# Architecture Patterns

**Domain:** LongMemEval benchmark harness for Elephant memory engine
**Researched:** 2026-03-15

## Recommended Architecture

The LongMemEval harness reuses Elephant's in-process runtime (same as LoCoMo) but must solve a fundamentally different lifecycle problem: 500 independent bank instances instead of 10 shared ones. This drives every structural decision.

### LoCoMo vs LongMemEval: The Core Structural Difference

```
LoCoMo:
  10 conversations -> 10 banks -> ~154 questions per bank
  Ingestion unit: conversation (shared across all its questions)
  Primary loop: for each conversation { ingest -> consolidate -> ask all questions }
  Concurrency axes: conversation_jobs x question_jobs

LongMemEval:
  500 questions -> 500 banks -> 1 question per bank
  Ingestion unit: per-question conversation history (unique to each question)
  Primary loop: for each question { create bank -> ingest history -> consolidate -> ask -> cleanup }
  Concurrency axes: instance_jobs (banks processed in parallel)
```

The 1:1 mapping of question to bank collapses the two-level LoCoMo loop (conversations x questions) into a single-level loop over "instances" where each instance is an (ingest, consolidate, query) pipeline.

### Component Diagram

```
longmemeval-bench binary
    |
    +-- CLI Parser
    |     parse_args() -> BenchInvocation { command, config }
    |     Subcommands: run, ingest, qa, merge (same as LoCoMo)
    |     Profiles: full-s, full-m, smoke
    |
    +-- Dataset Loader
    |     Load longmemeval_{s,m}_cleaned.json
    |     Parse into Vec<LongMemEvalInstance>
    |     Each instance: question_id, question_type, question, answer,
    |       question_date, haystack_sessions, haystack_dates, answer_session_ids
    |
    +-- Instance Runner (per-question worker)
    |     |
    |     +-- Bank Creator
    |     |     create_bank(name: "lme-{question_id}")
    |     |
    |     +-- Session Ingester
    |     |     for each session in haystack_sessions:
    |     |       format_session(turns, date) -> retain()
    |     |     Track: sessions_ingested, turns_ingested, facts_stored
    |     |
    |     +-- Consolidator
    |     |     consolidate_with_bench_progress()
    |     |     Reuses existing LoCoMo consolidation wrapper
    |     |
    |     +-- Question Asker
    |     |     reflect(question, bank_id) -> hypothesis
    |     |     collect retrieved_context, trace, sources
    |     |
    |     +-- Judge
    |     |     llm_judge(question, gold_answer, hypothesis) -> CORRECT/WRONG
    |     |     Same judge prompt, same binary scoring
    |     |
    |     +-- Bank Cleaner (optional)
    |           Lightweight: just don't keep DB data after run
    |           Full cleanup would be DROP bank, but Postgres handles this
    |
    +-- Shared Results (Arc<Mutex<SharedResults>>)
    |     Incremental artifact flushing (same pattern as LoCoMo)
    |     Per-instance bank tracking (question_id -> bank_id)
    |     Per-category accuracy breakdown
    |
    +-- Artifact Writer
    |     Summary JSON: longmemeval metadata + per_category + per_instance
    |     Question JSONL: per-question results
    |     Debug JSONL: reflect trace + retrieved context
    |
    +-- Merge Validator
          Same compatibility contract as LoCoMo
          Merge on disjoint question_id sets
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| CLI Parser | Parse subcommand + flags, load profile, resolve config | Dataset Loader, main orchestrator |
| Dataset Loader | Parse LongMemEval JSON, validate structure, apply instance selection/limits | Instance Runner |
| Instance Runner | Full lifecycle of one question: bank create -> ingest -> consolidate -> reflect -> judge | ElephantRuntime, Judge, Shared Results |
| Shared Results | Thread-safe accumulation of all results, incremental artifact flushing | Instance Runner (writes), Artifact Writer (reads) |
| Artifact Writer | Serialize summary JSON + JSONL sidecars | Shared Results |
| Judge | Binary correctness evaluation via LLM | Instance Runner (called by) |
| Merge Validator | Ensure merge compatibility, combine disjoint artifacts | Artifact Writer |

### Data Flow

```
Dataset JSON
    |
    v
[Dataset Loader]
    |  Vec<LongMemEvalInstance>
    v
[Semaphore-gated Instance Runner pool] ---- instance_jobs parallelism
    |
    | For each instance (question):
    |
    v
[Bank Creator] --create--> PostgreSQL (new bank per question)
    |
    v
[Session Ingester] --retain()--> ElephantRuntime.retain
    |  Iterates haystack_sessions sequentially
    |  Each session: format with date prefix, call retain()
    |  Track turn provenance (has_answer flag)
    |
    v
[Consolidator] --consolidate()--> ElephantRuntime.consolidator
    |  Mode: end (default), per-session, or off
    |
    v
[Question Asker] --reflect()--> ElephantRuntime.reflect
    |  question_date as temporal context
    |  Collect: hypothesis, retrieved_context, trace, final_done
    |
    v
[Judge] --llm_judge()--> Judge LLM client
    |  Returns: (correct: bool, reasoning: String)
    |
    v
[Shared Results] --push_and_flush()--> Artifact files
    |  Append to question JSONL + debug JSONL
    |  Rewrite summary JSON with updated aggregates
```

## Patterns to Follow

### Pattern 1: Single-Level Instance Loop (not two-level conversation x question)

In LoCoMo, the outer loop iterates conversations and the inner loop iterates questions within each conversation. In LongMemEval, each question IS its own conversation, so the loop is flat.

**What:** One semaphore-gated loop over instances, where each instance does the full ingest-consolidate-reflect-judge cycle.

**When:** Always. This is the fundamental execution model.

**Example:**
```rust
let semaphore = Arc::new(Semaphore::new(config.instance_jobs));
let mut handles = Vec::new();

for (idx, instance) in dataset.into_iter().enumerate() {
    let sem = semaphore.clone();
    let runtime = runtime.clone();
    let judge = judge.clone();
    let shared = shared.clone();
    let tag = format!("q {}/{total}", idx + 1);

    handles.push(tokio::spawn(async move {
        let _permit = sem.acquire().await.expect("semaphore closed");
        run_instance(tag, runtime, instance, judge, options, shared).await
    }));
}
```

### Pattern 2: Reuse ElephantRuntime Directly

**What:** Build `ElephantRuntime` from env exactly as LoCoMo does. Don't abstract or wrap it further.

**Why:** The runtime already provides `retain`, `reflect`, `consolidator`, `store`, `embeddings` as `Arc` fields. The harness calls these directly.

**Example:**
```rust
let runtime = Arc::new(
    build_runtime_from_env(BuildRuntimeOptions {
        metrics: Some(metrics.clone()),
    })
    .await
    .expect("failed to build Elephant runtime"),
);
```

### Pattern 3: Incremental Artifact Flushing via SharedResults

**What:** After each question completes, append to JSONL sidecars and rewrite the summary JSON. This provides crash recovery and real-time progress monitoring.

**Why:** With 500 instances and hours of runtime, losing results on crash is unacceptable.

**How:** Same `Arc<Mutex<SharedResults>>` pattern as LoCoMo, but keyed by `question_id` instead of `sample_id`.

### Pattern 4: Profile-Driven Configuration

**What:** JSON profile files in `bench/longmemeval/profiles/` that set defaults for dataset path, instance limits, consolidation mode, and concurrency.

**Why:** Reproducible benchmark shapes. `smoke` for iteration, `full-s` and `full-m` for publication.

**Example profiles:**
```json
// profiles/smoke.json
{
  "dataset": "data/longmemeval_s_cleaned.json",
  "instance_limit": 5,
  "session_limit": 3,
  "instance_jobs": 1,
  "consolidation": "end"
}

// profiles/full-s.json
{
  "dataset": "data/longmemeval_s_cleaned.json",
  "instance_jobs": 4,
  "consolidation": "end"
}

// profiles/full-m.json
{
  "dataset": "data/longmemeval_m_cleaned.json",
  "instance_jobs": 2,
  "consolidation": "end"
}
```

### Pattern 5: Question Type Categories

**What:** Map LongMemEval's `question_type` strings directly to category names for per-category breakdown.

**Why:** LongMemEval uses descriptive strings, not numeric categories like LoCoMo. The seven types are:
- `single-session-user` (information extraction - user side)
- `single-session-assistant` (information extraction - assistant side)
- `single-session-preference` (preference extraction)
- `multi-session` (multi-session reasoning)
- `knowledge-update` (knowledge updates)
- `temporal-reasoning` (temporal reasoning)
- `*_abs` suffix variants (abstention / false-premise)

The `_abs` suffix on question_id identifies abstention questions. These should be scored (they have answers like "not mentioned" or "unknown") using the same judge prompt, which already handles unanswerable questions.

### Pattern 6: Session Formatting with Dates

**What:** Format each LongMemEval session with its timestamp prefix, matching the LoCoMo session-level ingestion format.

**Example:**
```rust
fn format_lme_session(turns: &[LmeTurn], date: &str) -> String {
    let dialogue = turns
        .iter()
        .map(|t| format!("{}: {}", t.role, t.content))
        .collect::<Vec<_>>()
        .join("\n");
    format!("Date: {date}\n\n{dialogue}")
}
```

**Why:** Elephant's temporal annotation in consolidation (the biggest accuracy gain in LoCoMo benchmarking) depends on the `Date:` prefix being present in the retain input.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Shared Harness Module Between LoCoMo and LongMemEval

**What:** Extracting a `bench/common/` module with shared types and logic.

**Why bad:** The LoCoMo harness is 4340 lines of tightly coupled code. Extracting shared types creates a coupling point that makes both harnesses harder to change independently. The artifact schemas differ (per_conversation vs per_instance, different category names, different bank lifecycle). Shared types would either be overly generic or require constant synchronized changes.

**Instead:** Copy the patterns, not the code. Both harnesses use the same ElephantRuntime, MetricsCollector, and judge approach, but their artifact types, dataset types, CLI options, and instance lifecycle are different enough to justify independence. The judge prompt and judge client builder are small enough to duplicate.

### Anti-Pattern 2: Reusing LoCoMo's `run_conversation` for LongMemEval Instances

**What:** Wrapping each LongMemEval instance as a fake "conversation" and feeding it through LoCoMo's conversation runner.

**Why bad:** LoCoMo's `run_conversation` assumes multi-question-per-bank, turn-level provenance tracking, session-based iteration with speaker metadata, and BLIP caption handling. None of these apply to LongMemEval. The adaptation layer would be more complex than a clean implementation.

**Instead:** Write a clean `run_instance` function that does: create bank -> ingest sessions -> consolidate -> reflect -> judge -> record results. It will be shorter and more readable than adapting `run_conversation`.

### Anti-Pattern 3: Keeping All 500 Banks Alive Simultaneously

**What:** Creating all banks upfront, then querying them all.

**Why bad:** Each bank with ~50 sessions of ingested content creates hundreds of facts, entities, links, and observations. 500 banks simultaneously means ~500K facts in PostgreSQL. This is fine for pgvector but wastes disk and makes the `run` subcommand's ingest phase a massive upfront cost with no incremental results until QA starts.

**Instead:** Process instances sequentially within the parallelism window. Each instance creates its bank, ingests, consolidates, queries, and produces its result before the next instance takes its semaphore slot. Banks persist in Postgres (needed for `qa` resume), but the active working set is bounded by `instance_jobs`.

### Anti-Pattern 4: Per-Instance Question Concurrency

**What:** Adding a `question_jobs` concurrency parameter like LoCoMo has.

**Why bad:** Each LongMemEval instance has exactly one question. There is no inner question loop to parallelize. The only concurrency axis is `instance_jobs` (how many banks are being processed in parallel).

**Instead:** Single concurrency knob: `--instance-jobs N`.

## Key Design Decisions

### Artifact Schema: per_instance Instead of per_conversation

The summary JSON uses `per_instance` instead of `per_conversation`:

```json
{
  "benchmark": "longmemeval",
  "dataset_variant": "s",
  "per_category": {
    "single-session-user": { "accuracy": 0.72, "count": 120 },
    "multi-session": { "accuracy": 0.58, "count": 80 },
    "knowledge-update": { "accuracy": 0.55, "count": 60 },
    "temporal-reasoning": { "accuracy": 0.50, "count": 40 },
    "abstention": { "accuracy": 0.30, "count": 30 }
  },
  "per_instance": {
    "q001": {
      "bank_id": "01KXYZ...",
      "question_type": "single-session-user",
      "accuracy": 1.0,
      "ingest_time_s": 45.2,
      "consolidation_time_s": 12.1,
      "qa_time_s": 8.3,
      "total_time_s": 65.6,
      "bank_stats": { ... },
      "stage_metrics": { ... }
    }
  },
  "bank_ids": {
    "q001": "01KXYZ..."
  }
}
```

This preserves the LoCoMo contract shape (three-artifact pattern, manifest, stage metrics) while reflecting that each "unit" is a question instance rather than a conversation.

### Bank Naming Convention

Banks are named `lme-{question_id}` (e.g., `lme-q001`) to distinguish from LoCoMo banks (`locomo-conv-26`). The `question_id` comes directly from the dataset.

### Session Ingestion: Session-Level Only

LongMemEval sessions are user-assistant turn pairs without speaker metadata or image captions. Session-level ingestion (one retain call per session, with date prefix) is the only sensible mode. There is no turn-level mode because LongMemEval turns lack the provenance markers (dia_id) that make LoCoMo turn-level ingestion valuable for evidence tracking.

However, the `has_answer` field on individual turns can be used for retrieval diagnostics (did we retrieve facts from the evidence turns?), even though ingestion happens at session granularity.

### Evidence Tracking via answer_session_ids

LongMemEval provides `answer_session_ids` (which sessions contain the answer) and per-turn `has_answer` flags. Evidence recall should be computed at session level: did the retrieved context originate from facts that were ingested from an answer session? This maps to LoCoMo's `evidence_recall` metric.

Implementation: during ingestion, track which sessions map to which source records. During QA, check if retrieved facts trace back to answer sessions.

### Abstention Questions

Questions with `_abs` suffix in their question_id are false-premise questions. The gold answer indicates the information was never mentioned. The existing judge prompt already handles this case ("For unanswerable questions: if the gold answer indicates the question cannot be answered and the generated answer also indicates insufficient information or uncertainty, count it as CORRECT").

Score them with the same judge. Track separately in `per_category` under "abstention".

### Dataset Variant in Manifest

The manifest must record `dataset_variant: "s"` or `"m"` to distinguish runs. This is derived from the dataset path or an explicit `--variant` flag.

## Concurrency Model

```
                    instance_jobs semaphore (e.g., 4)
                    |
        +-----------+-----------+-----------+
        |           |           |           |
  [Instance 1] [Instance 2] [Instance 3] [Instance 4]
   ingest        ingest        ingest        ingest
   consolidate   consolidate   consolidate   consolidate
   reflect       reflect       reflect       reflect
   judge         judge         judge         judge
   -> next       -> next       -> next       -> next
```

Each instance runs its full lifecycle within one semaphore permit. There is no inner parallelism axis. This is simpler than LoCoMo's two-axis model because each instance has exactly one question.

**Resource considerations for instance_jobs:**
- Each concurrent instance creates independent LLM calls (retain extraction, consolidation, reflect, judge)
- PostgreSQL handles concurrent bank operations fine
- LLM rate limiting is the practical bottleneck
- `instance_jobs=4` is a reasonable default for S; `instance_jobs=2` for M (larger histories)

**LongMemEval M warning:** Each M instance has ~500 sessions (~1.5M tokens). Ingestion alone takes significant time and LLM calls. Even with `instance_jobs=2`, this means 2 instances ingesting 500 sessions concurrently. Monitor LLM rate limits.

## Scalability Considerations

| Concern | LongMemEval S (50 sessions) | LongMemEval M (500 sessions) |
|---------|----------------------------|------------------------------|
| DB size (500 banks) | ~200K facts, manageable | ~2M facts, needs monitoring |
| Ingestion time per instance | ~2-5 min | ~20-60 min |
| Total ingestion time (serial) | ~17-42 hours | ~170-500 hours |
| Total ingestion time (4x parallel) | ~4-10 hours | ~42-125 hours |
| Consolidation time per instance | ~1-3 min | ~10-30 min |
| Reflect time per instance | ~10-30 sec | ~10-30 sec (same bank size post-consolidation) |
| Full run time estimate | ~6-15 hours | ~60-200 hours |

These estimates come from scaling LoCoMo's observed times (conv-26: 35 sessions, 188s ingest, 7s consolidation). LongMemEval S has comparable session counts; M is 10x larger.

The `ingest` + `qa` split subcommand pattern is essential for M runs where ingestion alone may take days. Ingest in batches, merge, then QA.

## Build Order (Dependencies Between Components)

The components should be built in this order, where each phase produces a testable, usable increment:

### Phase 1: Dataset Types + Loader
- `LongMemEvalInstance` struct with serde deserialization
- Session/turn types
- Dataset loading and validation
- Question type parsing
- **Testable:** Load real dataset file, verify parse, count instances by type
- **No dependencies on other new code**

### Phase 2: Instance Runner (ingest only)
- `run_instance` function (ingest path only)
- Bank creation, session formatting, retain calls
- Session-to-source provenance tracking
- ConversationBankStats equivalent
- **Depends on:** Phase 1 (dataset types)
- **Testable:** Ingest one instance, verify bank contents in Postgres

### Phase 3: CLI + Profiles + Artifact Writer
- Subcommand parsing (run, ingest, qa, merge)
- Profile system (smoke, full-s, full-m)
- Summary JSON, question JSONL, debug JSONL output
- SharedResults with incremental flushing
- Manifest with reproducibility fields
- **Depends on:** Phase 1 (dataset types), Phase 2 (instance runner)
- **Testable:** `longmemeval-bench ingest --profile smoke` produces valid artifacts

### Phase 4: QA Path (reflect + judge)
- Extend `run_instance` with reflect + judge
- Per-category accuracy breakdown (7 categories + abstention)
- Evidence recall via answer_session_ids
- Token F1 scoring
- **Depends on:** Phase 3 (artifacts to store results)
- **Testable:** `longmemeval-bench run --profile smoke` produces scored results

### Phase 5: Resume, Merge, View
- `qa` subcommand (reuse banks from ingest artifact)
- `merge` subcommand (combine disjoint instance sets)
- `longmemeval-view` binary for result inspection
- **Depends on:** Phase 4 (full run produces results)
- **Testable:** Ingest subset, QA subset, merge into canonical artifact

### Phase 6: Concurrency + Polish
- Instance-level parallelism with semaphore
- Progress reporting (running accuracy, ETA)
- Error handling for individual instance failures
- LongMemEval M support (larger session counts)
- **Depends on:** Phase 5 (all subcommands working)
- **Testable:** Full S run with `--instance-jobs 4`

### Why This Order

1. **Dataset types first** because everything depends on parsing the input correctly. Verifiable in isolation without any Elephant infrastructure.
2. **Ingest before QA** because ingestion is the complex new code (session formatting, bank lifecycle). QA reuses existing reflect/judge patterns.
3. **CLI + artifacts before QA** because you need the output infrastructure to capture QA results.
4. **Resume/merge last** because they are refinements on the working pipeline, not prerequisites.
5. **Concurrency last** because getting the serial path correct first avoids debugging race conditions while the core logic is still being developed.

## What to Share vs What to Copy

| Component | Approach | Rationale |
|-----------|----------|-----------|
| `ElephantRuntime` | Share (import) | Single runtime struct, already a clean API |
| `MetricsCollector` / `with_scoped_collector` | Share (import) | Stable, generic, no bench-specific logic |
| `build_runtime_from_env` | Share (import) | Already designed for bench and server use |
| `LlmStage` / `StageUsage` | Share (import) | Type definitions, no behavior |
| `judge_answer.txt` prompt | Copy (or symlink) | Same prompt works for LongMemEval; if judge evolves per-benchmark, they diverge |
| `llm_judge()` function | Duplicate | ~50 lines, tightly coupled to harness error handling |
| `build_judge_client()` function | Duplicate | ~30 lines, references harness-local env var patterns |
| `token_f1()` | Duplicate | ~40 lines, pure function, no dependencies |
| `flush_results()` | Rewrite | Schema is different (per_instance vs per_conversation) |
| `BenchmarkOutput` struct | Rewrite | Different fields, categories, bank tracking |
| `run_conversation()` | Rewrite as `run_instance()` | Fundamentally different lifecycle |
| `SharedResults` | Rewrite | Same pattern, different fields |
| CLI parser | Rewrite | Different flags (no --conversation-jobs, add --instance-jobs, --variant) |
| Profile configs | New | Different profiles for S vs M vs smoke |
| View tool | New binary | Different display needs |

## Sources

- LoCoMo harness: `bench/locomo/locomo.rs` (4340 lines, examined in full)
- LoCoMo results format: `bench/locomo/results-format.md`
- Elephant runtime: `src/runtime.rs`
- Codebase architecture: `.planning/codebase/ARCHITECTURE.md`
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval) - dataset format, evaluation scripts
- [LongMemEval paper](https://arxiv.org/abs/2410.10813) - benchmark methodology, question types, evaluation protocol
- [LongMemEval project page](https://xiaowu0162.github.io/long-mem-eval/) - dataset variants, baseline results

