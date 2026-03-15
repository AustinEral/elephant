# Domain Pitfalls

**Domain:** LongMemEval benchmark integration for Rust memory engine
**Researched:** 2026-03-15

## Critical Pitfalls

Mistakes that cause rewrites, invalid results, or wasted compute.

### Pitfall 1: Category-Label Mapping Bug (Already Happened Once)

**What goes wrong:** The seven LongMemEval `question_type` strings (`single-session-user`, `single-session-assistant`, `single-session-preference`, `multi-session`, `knowledge-update`, `temporal-reasoning`, `false-premise`) get mapped to display names or numeric IDs incorrectly, silently swapping per-category accuracy numbers.

**Why it happens:** This exact bug already occurred in the Hindsight LoCoMo comparison runner (see `bench/locomo/internal/hindsight-conv26-comparison-2026-03-10.md`): the external runner swapped `single-hop` and `multi-hop` category labels for LoCoMo category ids 1 and 2. Total accuracy was fine, but per-category breakdown was wrong. LongMemEval has seven categories instead of LoCoMo's five, making the mapping surface area larger.

**Consequences:** Per-category claims in published results would be wrong. Total accuracy remains correct, so the bug is invisible without manual spot-checking. A paper claiming "92% on temporal-reasoning" when the actual temporal-reasoning number belongs to a different category is a credibility-destroying error.

**Prevention:**
1. Use `question_type` strings directly from the dataset JSON as the category key -- do not create a separate enum or numeric mapping. The dataset already provides human-readable strings.
2. Add a unit test that asserts every `question_type` in the loaded dataset maps to one of the seven known strings.
3. Add a "smoke check" in the view tool that prints one example question per category so a human can eyeball that "temporal-reasoning" questions actually involve time.

**Detection:** Per-category numbers that don't match intuition (e.g., knowledge-update accuracy suspiciously equal to single-session-user). Compare against published baselines per category.

**Phase relevance:** Dataset parsing and metric calculation (early implementation).

---

### Pitfall 2: Abstention Scoring Handled Wrong

**What goes wrong:** The 30 false-premise questions get scored using the same judge prompt as factual questions, producing incorrect accuracy numbers. Or they get excluded entirely from the total, making the denominator 470 instead of 500 and results incomparable with published baselines.

**Why it happens:** LongMemEval's false-premise questions test whether the system correctly says "I don't know" or "that wasn't mentioned." This requires a fundamentally different judge criterion: the system is correct if it *refuses* to answer, not if it matches a gold answer. The upstream `evaluate_qa.py` uses category-specific judge prompts -- the abstention judge prompt checks whether the system declined, not whether it matched an answer string. The existing LoCoMo harness explicitly excludes category 5 (`should_score_question` returns false for unanswerable questions). Carrying this pattern forward would silently drop 30 questions from LongMemEval.

**Consequences:** If abstention questions are dropped: total accuracy computed over 470 questions is incomparable to published results (which use 500). If they're scored with the wrong judge prompt: a system that confidently hallucinates an answer to a false-premise question would be scored as "correct" by a standard factual-matching judge.

**Prevention:**
1. Implement two judge prompts: one for factual questions (does the hypothesis match the gold answer?) and one for abstention questions (does the hypothesis decline to answer?).
2. The `question_type` field tells you which judge to use: `false-premise` gets the abstention judge, everything else gets the factual judge.
3. Always count all 500 questions in the denominator. Publish per-category breakdowns that include the abstention category.
4. Port the upstream `evaluate_qa.py` judge prompts rather than inventing new ones -- this ensures comparability with published results.

**Detection:** Check that total question count is 500, not 470. Verify that a few false-premise questions have judge reasoning about abstention behavior.

**Phase relevance:** Judge implementation (mid implementation). Must be designed correctly before the first scored run.

---

### Pitfall 3: 500 Banks Exhaust Postgres Connections or Disk

**What goes wrong:** LongMemEval requires a separate bank per question (500 banks for the S setting, 500 for M). Each bank's ingestion involves creating entities, storing facts, computing embeddings, and running consolidation. With any concurrency, the connection pool gets exhausted. With the M setting (~1.5M tokens per instance, 500 sessions), the Postgres database can grow to tens of gigabytes.

**Why it happens:** The current Elephant runtime uses `sqlx::PgPool::connect()` with default settings (10 connections). LoCoMo only needs 10 banks total, so this was never an issue. With 500 banks and concurrent ingestion, even modest parallelism (e.g., 5 instance-jobs) can exhaust the pool, especially during consolidation when each job holds a connection for an extended period. The M setting amplifies this: each instance has ~500 sessions with ~1.5M tokens, meaning a single instance's ingestion could take 30+ minutes and produce thousands of facts.

**Consequences:** Connection pool exhaustion causes `sqlx` errors that may or may not be retried cleanly. Disk exhaustion on the Postgres volume causes silent data corruption or crashes. Memory pressure from 500 embedding computations happening concurrently can OOM the process.

**Prevention:**
1. Configure the pool explicitly: `PgPoolOptions::new().max_connections(20)` or higher, tuned to Postgres's `max_connections` setting.
2. Implement instance-level concurrency control (`--instance-jobs N`) separate from question-level concurrency. Start with instance-jobs=1 for M, instance-jobs=3-5 for S.
3. Add a `--cleanup` or `--gc` mode that drops banks after QA is complete for that instance, to bound database size.
4. Monitor disk usage during M runs. A full M run could produce 100GB+ of embeddings and facts.
5. Consider bank reuse detection: if the same `question_id` already has a bank in the artifact, skip re-ingestion (the LoCoMo resume pattern).

**Detection:** Watch for `sqlx::Error` in logs, especially "connection pool timed out" or "too many connections." Monitor `pg_stat_activity` connection count. Track disk usage of the Postgres data directory.

**Phase relevance:** Infrastructure and ingestion (early implementation). Must be addressed before the first full S run.

---

### Pitfall 4: Timestamps Not Propagated to Retain Pipeline

**What goes wrong:** Each session in `haystack_sessions` has a corresponding date in `haystack_dates`, but the harness either ignores these dates or assigns them incorrectly, breaking temporal reasoning questions.

**Why it happens:** The LoCoMo harness extracts session dates via `get_session_date()` and passes them to `parse_session_date()` which parses them into `DateTime<Utc>`. LongMemEval uses a different date format (`haystack_dates` is a top-level array of date strings rather than per-session keys in a HashMap). If the harness doesn't correctly zip `haystack_sessions` with `haystack_dates`, sessions get the wrong timestamps. Additionally, the `question_date` field (when the question is "asked") must be passed to the reflect agent so it can reason about temporal references like "last month."

**Consequences:** Temporal-reasoning questions (one of five core abilities) score near zero. This category is specifically designed to test timestamp awareness, and wrong timestamps make it impossible to answer correctly. Elephant's temporal annotations in consolidation (which yielded +7.7% temporal accuracy on LoCoMo) become useless.

**Prevention:**
1. Parse `haystack_dates` and `haystack_sessions` as parallel arrays. Assert they have the same length during deserialization.
2. Pass `question_date` to the reflect agent as the "current date" context -- this is how temporal references like "two weeks ago" resolve.
3. Add a unit test that checks the first and last session dates for a known question_id match expected values from the dataset.
4. Verify with a temporal-reasoning question that the reflect agent's response references the correct dates.

**Detection:** Temporal-reasoning accuracy significantly below published baselines (GPT-4o scores ~45-55% on temporal questions). If Elephant scores below 30% on temporal, timestamps are likely wrong.

**Phase relevance:** Dataset parsing and ingestion (early implementation).

---

### Pitfall 5: Using the Wrong Dataset Version

**What goes wrong:** Results are computed against `longmemeval_s.json` (original) instead of `longmemeval_s_cleaned.json`, or vice versa, making them incomparable with other published results.

**Why it happens:** The original dataset was updated in September 2025 to remove "noisy history sessions that interfere with answer correctness." The original `xiaowu0162/longmemeval` HuggingFace dataset is deprecated in favor of `xiaowu0162/longmemeval-cleaned`. Most post-2025 published results use the cleaned version. If someone downloads the wrong one, or the code points to a stale local copy, results are against a different dataset.

**Consequences:** Results are incomparable with all recent published baselines. The cleaned version removes confounding sessions, so accuracy numbers will differ between versions. Publishing a number without specifying which version is a credibility issue.

**Prevention:**
1. Hardcode the expected dataset filenames: `longmemeval_s_cleaned.json` and `longmemeval_m_cleaned.json`.
2. Compute and record a dataset fingerprint (hash) in the manifest, like the LoCoMo harness does.
3. Document the expected HuggingFace source in the download instructions: `xiaowu0162/longmemeval-cleaned`.
4. Fail loudly if the dataset file doesn't match the expected filename pattern.

**Detection:** Dataset fingerprint mismatch when comparing runs. Different question count or field names.

**Phase relevance:** Setup and dataset loading (very first step).

## Moderate Pitfalls

### Pitfall 6: Judge Model Mismatch Invalidates Comparisons

**What goes wrong:** Results scored with Claude/Sonnet as judge get compared against published baselines scored with GPT-4o, and the accuracy delta is partly judge variance, not system quality.

**Why it happens:** The LongMemEval paper uses GPT-4o as the judge and reports >97% agreement with human experts. The LoCoMo harness defaults to whatever `JUDGE_MODEL` is set to (usually Sonnet for cost reasons). Different judges have different biases -- the Hindsight comparison already showed judge failures with Anthropic models producing invalid JSON (4 hard failures out of 152 questions).

**Prevention:**
1. Default to GPT-4o for publication runs, matching the upstream evaluation. Make this the "canonical" judge.
2. Support judge override for iteration runs (use cheaper models when tuning).
3. Record the judge model in the manifest and view tool. Flag results as "non-standard judge" when GPT-4o is not used.
4. If using a non-GPT-4o judge for publication, include a calibration note (run N questions with both judges, report agreement rate).

**Detection:** Check the `judge_model` field in the artifact. If it doesn't say `gpt-4o`, flag for comparability review.

**Phase relevance:** Judge implementation and publication preparation.

### Pitfall 7: Per-Question Bank Ingestion Wall-Clock Time Explosion

**What goes wrong:** A full S run takes days instead of hours because each of the 500 instances requires its own ingestion + consolidation cycle, and these are done sequentially.

**Why it happens:** LoCoMo has 10 conversations, so 10 ingestion cycles. LongMemEval S has 500 instances, each with ~50 sessions. That's 500 ingestion cycles. At the LoCoMo conv-26 rate (~81 minutes for ingestion of one conversation), a naive sequential LongMemEval S run would be 500 * 81 minutes = 28 days. Even with smaller per-instance histories (fewer sessions than a full LoCoMo conversation), the sheer count is the problem.

**Prevention:**
1. Instance-level parallelism is mandatory. Design `--instance-jobs N` from the start.
2. Separate the `ingest` subcommand from `qa` so ingestion can run as a batch without blocking on question answering.
3. Profile a single S instance to estimate total run time before committing to a full run.
4. For the M setting (500 sessions per instance), estimate that a single instance ingestion could take 2+ hours. Plan for M runs to take a week.

**Detection:** First full S run taking longer than 24 hours is a sign that concurrency isn't high enough or per-instance ingestion is too slow.

**Phase relevance:** Ingestion implementation and performance tuning.

### Pitfall 8: Resume/Merge Complexity with 500 Banks

**What goes wrong:** A long-running benchmark gets interrupted partway through, and the resume mechanism either re-ingests already-completed instances (wasting hours) or fails to find existing banks.

**Why it happens:** The LoCoMo harness stores `bank_ids` in the artifact keyed by `sample_id`. For 500 instances, this map becomes the critical state for resume. If the artifact file is corrupted, lost, or the harness crashes between writing banks and writing the artifact, all ingestion work is lost. With LoCoMo's 10 conversations, re-ingestion from scratch is tolerable. With 500 LongMemEval instances, it's not.

**Prevention:**
1. Write the bank mapping incrementally (after each instance completes), not just at the end. The LoCoMo harness's `SharedResults` pattern handles this, but verify it flushes to disk regularly.
2. Support `ingest` as a separate subcommand that produces a "banks artifact" file mapping `question_id -> bank_id`. The `qa` subcommand reads this file.
3. Use atomic file writes (write to `.tmp`, rename) to prevent half-written artifacts.
4. The `merge` subcommand must handle combining partial runs correctly, deduplicating by `question_id`.

**Detection:** Re-running after a crash and seeing "Ingesting..." instead of "Using existing bank..." for already-completed instances.

**Phase relevance:** CLI design and artifact management (early-mid implementation).

### Pitfall 9: Consolidation Strategy Inconsistency Between Runs

**What goes wrong:** Different runs use different consolidation strategies (end, per-session, off) without recording it, making accuracy comparisons meaningless.

**Why it happens:** Consolidation is a major lever for accuracy. The LoCoMo harness records `consolidation_strategy` in the manifest. If the LongMemEval harness doesn't enforce consistency or record the strategy, two runs with different consolidation could be compared head-to-head. For LongMemEval M (500 sessions), per-session consolidation vs end-of-conversation consolidation has very different cost and quality implications.

**Prevention:**
1. Record consolidation strategy in the manifest (already a pattern from LoCoMo).
2. The profile system should lock consolidation mode: `full-s` uses `end`, `full-m` uses `per-session` (or whatever is chosen after testing).
3. The view tool should display consolidation strategy prominently.

**Detection:** Compare manifest `consolidation_strategy` fields between runs before drawing accuracy conclusions.

**Phase relevance:** Profile system and manifest design.

### Pitfall 10: Reflect Agent Doesn't Handle "I Don't Know" for False-Premise Questions

**What goes wrong:** The reflect agent always produces a substantive answer (possibly hallucinated) for false-premise questions, instead of saying the information wasn't discussed. The judge then has to decide whether the hallucinated answer counts as an appropriate refusal.

**Why it happens:** Elephant's reflect agent is optimized for synthesis -- "infer from partial evidence rather than declining" (from the reflect prompt tuning history in MEMORY.md). This is great for factual questions but exactly wrong for the 30 false-premise questions. The agent will try to synthesize an answer from tangentially related facts.

**Prevention:**
1. Do NOT modify the reflect agent specifically for LongMemEval false-premise questions. That would be benchmark-fitting, not genuine capability improvement. The PROJECT.md explicitly puts this out of scope.
2. Instead, let the reflect agent handle false-premise questions naturally and measure how it performs. If it scores 0/30 on abstention, that's an honest result and a genuine capability gap to address later.
3. Ensure the abstention judge prompt is lenient enough to detect soft refusals (e.g., "I don't have information about that" even if the agent adds caveats).

**Detection:** Abstention accuracy of 0-10% on the first run. Review a few false-premise answers manually.

**Phase relevance:** QA implementation and analysis phase.

## Minor Pitfalls

### Pitfall 11: `has_answer` Field Misinterpreted

**What goes wrong:** The `has_answer: true` field on turns is used for scoring (checking if the system retrieved the right turn) but confused with being relevant for ingestion (only ingesting turns with `has_answer: true`).

**Why it happens:** In the upstream evaluation, `has_answer` is used for retrieval recall metrics (did the retriever find the evidence-bearing turns?). For Elephant's purposes, ALL turns must be ingested -- `has_answer` is metadata for evaluation, not a filter for ingestion.

**Prevention:** Don't filter on `has_answer` during ingestion. Only use it in evaluation code for evidence-recall metrics (if implementing retrieval metrics later).

**Detection:** If the ingested bank has suspiciously few facts, check whether `has_answer` filtering is active.

**Phase relevance:** Dataset parsing.

### Pitfall 12: `answer_session_ids` Indexing Off-By-One

**What goes wrong:** `answer_session_ids` lists the session indices containing evidence, but it's ambiguous whether they're 0-indexed or 1-indexed. Using the wrong convention means evidence-recall calculations point to the wrong sessions.

**Why it happens:** Python is 0-indexed, but the LoCoMo harness uses 1-indexed session numbering (`1..=ingest_sessions`). The LongMemEval dataset uses 0-indexed `answer_session_ids` (matching Python list indexing of `haystack_sessions`).

**Prevention:** Check the first few instances manually: look at `answer_session_ids`, load those sessions, and verify they contain the evidence mentioned in the `answer` field. Add an assertion in the test suite.

**Detection:** Evidence-recall metrics consistently at 0% or near-random.

**Phase relevance:** Evaluation metric implementation.

### Pitfall 13: M Setting Memory Pressure from Embedding Computation

**What goes wrong:** The M setting has ~1.5M tokens per instance across ~500 sessions. If embedding computation loads all session text into memory simultaneously (for batch embedding), a single instance can consume several GB of RAM. With concurrent instance-jobs, this can OOM.

**Prevention:**
1. Embed session-by-session during ingestion, not all at once.
2. Set `--instance-jobs 1` for M setting initially.
3. Monitor RSS during a single M-instance ingestion before scaling concurrency.

**Detection:** OOM killer or process crash during M ingestion.

**Phase relevance:** M setting support (later phase).

### Pitfall 14: Merging Partial Runs with Overlapping Questions

**What goes wrong:** The merge subcommand combines two partial runs that both scored the same `question_id`, producing duplicate entries and inflated/deflated accuracy.

**Why it happens:** Unlike LoCoMo where each conversation has distinct questions, LongMemEval has a flat list of 500 questions. If two partial runs overlap (e.g., both ran questions 1-100), merge must deduplicate.

**Prevention:**
1. Merge should key by `question_id` and reject duplicates (or take the later run's answer).
2. The LoCoMo merge already validates non-overlap for conversations. Adapt this to question-level deduplication.

**Detection:** Merged artifact has total_questions > 500.

**Phase relevance:** Merge subcommand implementation.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Dataset parsing | Category string mapping, `has_answer` misuse, timestamp array alignment, wrong dataset version | Use `question_type` strings directly; zip dates+sessions with length assertion; require `_cleaned` files |
| Ingestion infrastructure | 500 banks exhaust pool, wall-clock explosion, no bank cleanup | Explicit pool config, instance-level concurrency, incremental bank-map writes |
| Judge implementation | Abstention scoring wrong, judge model mismatch with published baselines | Two judge prompts (factual + abstention), default GPT-4o for publication |
| Metric calculation | Category totals wrong, denominator excludes abstention, off-by-one in evidence indexing | Count all 500 questions, validate per-category against published distributions |
| Resume/merge | Lost ingestion work, duplicate questions in merge | Incremental artifact writes, question-level dedup in merge |
| M setting | Memory pressure, multi-day runtime, disk exhaustion | Low instance concurrency, explicit pool sizing, disk monitoring, bank GC |
| Comparison validity | Wrong dataset version, non-GPT-4o judge, consolidation strategy mismatch | Dataset fingerprint in manifest, judge model in manifest, locked profiles |

## Sources

- [LongMemEval paper (arXiv:2410.10813)](https://arxiv.org/abs/2410.10813)
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval) -- dataset format, evaluation scripts, README
- [LongMemEval cleaned dataset (HuggingFace)](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) -- September 2025 update
- [Emergence.ai SOTA on LongMemEval](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag) -- implementation experience, RAG parameter sensitivity
- [Mastra Observational Memory (95% on LongMemEval)](https://mastra.ai/research/observational-memory) -- alternative architecture achieving high scores
- [Hindsight conv-26 comparison](bench/locomo/internal/hindsight-conv26-comparison-2026-03-10.md) -- category-label mapping bug, judge failure experience, metric source mismatch (HIGH confidence, first-party)
- [LoCoMo harness source](bench/locomo/locomo.rs) -- existing patterns for category mapping, judge retry, resume, consolidation (HIGH confidence, first-party)
- [Bench instrumentation notes](bench/locomo/internal/bench-instrumentation.md) -- in-process metering design decisions (HIGH confidence, first-party)
- Elephant MEMORY.md -- prompt tuning history, reflect agent behavior, consolidation architecture (HIGH confidence, first-party)

# Domain Pitfalls

**Domain:** LongMemEval benchmark integration for Rust memory engine
**Researched:** 2026-03-15

## Critical Pitfalls

Mistakes that cause rewrites, invalid results, or wasted compute.

### Pitfall 1: Category-Label Mapping Bug (Already Happened Once)

**What goes wrong:** The seven LongMemEval `question_type` strings (`single-session-user`, `single-session-assistant`, `single-session-preference`, `multi-session`, `knowledge-update`, `temporal-reasoning`, `false-premise`) get mapped to display names or numeric IDs incorrectly, silently swapping per-category accuracy numbers.

**Why it happens:** This exact bug already occurred in the Hindsight LoCoMo comparison runner (see `bench/locomo/internal/hindsight-conv26-comparison-2026-03-10.md`): the external runner swapped `single-hop` and `multi-hop` category labels for LoCoMo category ids 1 and 2. Total accuracy was fine, but per-category breakdown was wrong. LongMemEval has seven categories instead of LoCoMo's five, making the mapping surface area larger.

**Consequences:** Per-category claims in published results would be wrong. Total accuracy remains correct, so the bug is invisible without manual spot-checking. A paper claiming "92% on temporal-reasoning" when the actual temporal-reasoning number belongs to a different category is a credibility-destroying error.

**Prevention:**
1. Use `question_type` strings directly from the dataset JSON as the category key -- do not create a separate enum or numeric mapping. The dataset already provides human-readable strings.
2. Add a unit test that asserts every `question_type` in the loaded dataset maps to one of the seven known strings.
3. Add a "smoke check" in the view tool that prints one example question per category so a human can eyeball that "temporal-reasoning" questions actually involve time.

**Detection:** Per-category numbers that don't match intuition (e.g., knowledge-update accuracy suspiciously equal to single-session-user). Compare against published baselines per category.

**Phase relevance:** Dataset parsing and metric calculation (early implementation).

---

### Pitfall 2: Abstention Scoring Handled Wrong

**What goes wrong:** The 30 false-premise questions get scored using the same judge prompt as factual questions, producing incorrect accuracy numbers. Or they get excluded entirely from the total, making the denominator 470 instead of 500 and results incomparable with published baselines.

**Why it happens:** LongMemEval's false-premise questions test whether the system correctly says "I don't know" or "that wasn't mentioned." This requires a fundamentally different judge criterion: the system is correct if it *refuses* to answer, not if it matches a gold answer. The upstream `evaluate_qa.py` uses category-specific judge prompts -- the abstention judge prompt checks whether the system declined, not whether it matched an answer string. The existing LoCoMo harness explicitly excludes category 5 (`should_score_question` returns false for unanswerable questions). Carrying this pattern forward would silently drop 30 questions from LongMemEval.

**Consequences:** If abstention questions are dropped: total accuracy computed over 470 questions is incomparable to published results (which use 500). If they're scored with the wrong judge prompt: a system that confidently hallucinates an answer to a false-premise question would be scored as "correct" by a standard factual-matching judge.

**Prevention:**
1. Implement two judge prompts: one for factual questions (does the hypothesis match the gold answer?) and one for abstention questions (does the hypothesis decline to answer?).
2. The `question_type` field tells you which judge to use: `false-premise` gets the abstention judge, everything else gets the factual judge.
3. Always count all 500 questions in the denominator. Publish per-category breakdowns that include the abstention category.
4. Port the upstream `evaluate_qa.py` judge prompts rather than inventing new ones -- this ensures comparability with published results.

**Detection:** Check that total question count is 500, not 470. Verify that a few false-premise questions have judge reasoning about abstention behavior.

**Phase relevance:** Judge implementation (mid implementation). Must be designed correctly before the first scored run.

---

### Pitfall 3: 500 Banks Exhaust Postgres Connections or Disk

**What goes wrong:** LongMemEval requires a separate bank per question (500 banks for the S setting, 500 for M). Each bank's ingestion involves creating entities, storing facts, computing embeddings, and running consolidation. With any concurrency, the connection pool gets exhausted. With the M setting (~1.5M tokens per instance, 500 sessions), the Postgres database can grow to tens of gigabytes.

**Why it happens:** The current Elephant runtime uses `sqlx::PgPool::connect()` with default settings (10 connections). LoCoMo only needs 10 banks total, so this was never an issue. With 500 banks and concurrent ingestion, even modest parallelism (e.g., 5 instance-jobs) can exhaust the pool, especially during consolidation when each job holds a connection for an extended period. The M setting amplifies this: each instance has ~500 sessions with ~1.5M tokens, meaning a single instance's ingestion could take 30+ minutes and produce thousands of facts.

**Consequences:** Connection pool exhaustion causes `sqlx` errors that may or may not be retried cleanly. Disk exhaustion on the Postgres volume causes silent data corruption or crashes. Memory pressure from 500 embedding computations happening concurrently can OOM the process.

**Prevention:**
1. Configure the pool explicitly: `PgPoolOptions::new().max_connections(20)` or higher, tuned to Postgres's `max_connections` setting.
2. Implement instance-level concurrency control (`--instance-jobs N`) separate from question-level concurrency. Start with instance-jobs=1 for M, instance-jobs=3-5 for S.
3. Add a `--cleanup` or `--gc` mode that drops banks after QA is complete for that instance, to bound database size.
4. Monitor disk usage during M runs. A full M run could produce 100GB+ of embeddings and facts.
5. Consider bank reuse detection: if the same `question_id` already has a bank in the artifact, skip re-ingestion (the LoCoMo resume pattern).

**Detection:** Watch for `sqlx::Error` in logs, especially "connection pool timed out" or "too many connections." Monitor `pg_stat_activity` connection count. Track disk usage of the Postgres data directory.

**Phase relevance:** Infrastructure and ingestion (early implementation). Must be addressed before the first full S run.

---

### Pitfall 4: Timestamps Not Propagated to Retain Pipeline

**What goes wrong:** Each session in `haystack_sessions` has a corresponding date in `haystack_dates`, but the harness either ignores these dates or assigns them incorrectly, breaking temporal reasoning questions.

**Why it happens:** The LoCoMo harness extracts session dates via `get_session_date()` and passes them to `parse_session_date()` which parses them into `DateTime<Utc>`. LongMemEval uses a different date format (`haystack_dates` is a top-level array of date strings rather than per-session keys in a HashMap). If the harness doesn't correctly zip `haystack_sessions` with `haystack_dates`, sessions get the wrong timestamps. Additionally, the `question_date` field (when the question is "asked") must be passed to the reflect agent so it can reason about temporal references like "last month."

**Consequences:** Temporal-reasoning questions (one of five core abilities) score near zero. This category is specifically designed to test timestamp awareness, and wrong timestamps make it impossible to answer correctly. Elephant's temporal annotations in consolidation (which yielded +7.7% temporal accuracy on LoCoMo) become useless.

**Prevention:**
1. Parse `haystack_dates` and `haystack_sessions` as parallel arrays. Assert they have the same length during deserialization.
2. Pass `question_date` to the reflect agent as the "current date" context -- this is how temporal references like "two weeks ago" resolve.
3. Add a unit test that checks the first and last session dates for a known question_id match expected values from the dataset.
4. Verify with a temporal-reasoning question that the reflect agent's response references the correct dates.

**Detection:** Temporal-reasoning accuracy significantly below published baselines (GPT-4o scores ~45-55% on temporal questions). If Elephant scores below 30% on temporal, timestamps are likely wrong.

**Phase relevance:** Dataset parsing and ingestion (early implementation).

---

### Pitfall 5: Using the Wrong Dataset Version

**What goes wrong:** Results are computed against `longmemeval_s.json` (original) instead of `longmemeval_s_cleaned.json`, or vice versa, making them incomparable with other published results.

**Why it happens:** The original dataset was updated in September 2025 to remove "noisy history sessions that interfere with answer correctness." The original `xiaowu0162/longmemeval` HuggingFace dataset is deprecated in favor of `xiaowu0162/longmemeval-cleaned`. Most post-2025 published results use the cleaned version. If someone downloads the wrong one, or the code points to a stale local copy, results are against a different dataset.

**Consequences:** Results are incomparable with all recent published baselines. The cleaned version removes confounding sessions, so accuracy numbers will differ between versions. Publishing a number without specifying which version is a credibility issue.

**Prevention:**
1. Hardcode the expected dataset filenames: `longmemeval_s_cleaned.json` and `longmemeval_m_cleaned.json`.
2. Compute and record a dataset fingerprint (hash) in the manifest, like the LoCoMo harness does.
3. Document the expected HuggingFace source in the download instructions: `xiaowu0162/longmemeval-cleaned`.
4. Fail loudly if the dataset file doesn't match the expected filename pattern.

**Detection:** Dataset fingerprint mismatch when comparing runs. Different question count or field names.

**Phase relevance:** Setup and dataset loading (very first step).

## Moderate Pitfalls

### Pitfall 6: Judge Model Mismatch Invalidates Comparisons

**What goes wrong:** Results scored with Claude/Sonnet as judge get compared against published baselines scored with GPT-4o, and the accuracy delta is partly judge variance, not system quality.

**Why it happens:** The LongMemEval paper uses GPT-4o as the judge and reports >97% agreement with human experts. The LoCoMo harness defaults to whatever `JUDGE_MODEL` is set to (usually Sonnet for cost reasons). Different judges have different biases -- the Hindsight comparison already showed judge failures with Anthropic models producing invalid JSON (4 hard failures out of 152 questions).

**Prevention:**
1. Default to GPT-4o for publication runs, matching the upstream evaluation. Make this the "canonical" judge.
2. Support judge override for iteration runs (use cheaper models when tuning).
3. Record the judge model in the manifest and view tool. Flag results as "non-standard judge" when GPT-4o is not used.
4. If using a non-GPT-4o judge for publication, include a calibration note (run N questions with both judges, report agreement rate).

**Detection:** Check the `judge_model` field in the artifact. If it doesn't say `gpt-4o`, flag for comparability review.

**Phase relevance:** Judge implementation and publication preparation.

### Pitfall 7: Per-Question Bank Ingestion Wall-Clock Time Explosion

**What goes wrong:** A full S run takes days instead of hours because each of the 500 instances requires its own ingestion + consolidation cycle, and these are done sequentially.

**Why it happens:** LoCoMo has 10 conversations, so 10 ingestion cycles. LongMemEval S has 500 instances, each with ~50 sessions. That's 500 ingestion cycles. At the LoCoMo conv-26 rate (~81 minutes for ingestion of one conversation), a naive sequential LongMemEval S run would be 500 * 81 minutes = 28 days. Even with smaller per-instance histories (fewer sessions than a full LoCoMo conversation), the sheer count is the problem.

**Prevention:**
1. Instance-level parallelism is mandatory. Design `--instance-jobs N` from the start.
2. Separate the `ingest` subcommand from `qa` so ingestion can run as a batch without blocking on question answering.
3. Profile a single S instance to estimate total run time before committing to a full run.
4. For the M setting (500 sessions per instance), estimate that a single instance ingestion could take 2+ hours. Plan for M runs to take a week.

**Detection:** First full S run taking longer than 24 hours is a sign that concurrency isn't high enough or per-instance ingestion is too slow.

**Phase relevance:** Ingestion implementation and performance tuning.

### Pitfall 8: Resume/Merge Complexity with 500 Banks

**What goes wrong:** A long-running benchmark gets interrupted partway through, and the resume mechanism either re-ingests already-completed instances (wasting hours) or fails to find existing banks.

**Why it happens:** The LoCoMo harness stores `bank_ids` in the artifact keyed by `sample_id`. For 500 instances, this map becomes the critical state for resume. If the artifact file is corrupted, lost, or the harness crashes between writing banks and writing the artifact, all ingestion work is lost. With LoCoMo's 10 conversations, re-ingestion from scratch is tolerable. With 500 LongMemEval instances, it's not.

**Prevention:**
1. Write the bank mapping incrementally (after each instance completes), not just at the end. The LoCoMo harness's `SharedResults` pattern handles this, but verify it flushes to disk regularly.
2. Support `ingest` as a separate subcommand that produces a "banks artifact" file mapping `question_id -> bank_id`. The `qa` subcommand reads this file.
3. Use atomic file writes (write to `.tmp`, rename) to prevent half-written artifacts.
4. The `merge` subcommand must handle combining partial runs correctly, deduplicating by `question_id`.

**Detection:** Re-running after a crash and seeing "Ingesting..." instead of "Using existing bank..." for already-completed instances.

**Phase relevance:** CLI design and artifact management (early-mid implementation).

### Pitfall 9: Consolidation Strategy Inconsistency Between Runs

**What goes wrong:** Different runs use different consolidation strategies (end, per-session, off) without recording it, making accuracy comparisons meaningless.

**Why it happens:** Consolidation is a major lever for accuracy. The LoCoMo harness records `consolidation_strategy` in the manifest. If the LongMemEval harness doesn't enforce consistency or record the strategy, two runs with different consolidation could be compared head-to-head. For LongMemEval M (500 sessions), per-session consolidation vs end-of-conversation consolidation has very different cost and quality implications.

**Prevention:**
1. Record consolidation strategy in the manifest (already a pattern from LoCoMo).
2. The profile system should lock consolidation mode: `full-s` uses `end`, `full-m` uses `per-session` (or whatever is chosen after testing).
3. The view tool should display consolidation strategy prominently.

**Detection:** Compare manifest `consolidation_strategy` fields between runs before drawing accuracy conclusions.

**Phase relevance:** Profile system and manifest design.

### Pitfall 10: Reflect Agent Doesn't Handle "I Don't Know" for False-Premise Questions

**What goes wrong:** The reflect agent always produces a substantive answer (possibly hallucinated) for false-premise questions, instead of saying the information wasn't discussed. The judge then has to decide whether the hallucinated answer counts as an appropriate refusal.

**Why it happens:** Elephant's reflect agent is optimized for synthesis -- "infer from partial evidence rather than declining" (from the reflect prompt tuning history in MEMORY.md). This is great for factual questions but exactly wrong for the 30 false-premise questions. The agent will try to synthesize an answer from tangentially related facts.

**Prevention:**
1. Do NOT modify the reflect agent specifically for LongMemEval false-premise questions. That would be benchmark-fitting, not genuine capability improvement. The PROJECT.md explicitly puts this out of scope.
2. Instead, let the reflect agent handle false-premise questions naturally and measure how it performs. If it scores 0/30 on abstention, that's an honest result and a genuine capability gap to address later.
3. Ensure the abstention judge prompt is lenient enough to detect soft refusals (e.g., "I don't have information about that" even if the agent adds caveats).

**Detection:** Abstention accuracy of 0-10% on the first run. Review a few false-premise answers manually.

**Phase relevance:** QA implementation and analysis phase.

## Minor Pitfalls

### Pitfall 11: `has_answer` Field Misinterpreted

**What goes wrong:** The `has_answer: true` field on turns is used for scoring (checking if the system retrieved the right turn) but confused with being relevant for ingestion (only ingesting turns with `has_answer: true`).

**Why it happens:** In the upstream evaluation, `has_answer` is used for retrieval recall metrics (did the retriever find the evidence-bearing turns?). For Elephant's purposes, ALL turns must be ingested -- `has_answer` is metadata for evaluation, not a filter for ingestion.

**Prevention:** Don't filter on `has_answer` during ingestion. Only use it in evaluation code for evidence-recall metrics (if implementing retrieval metrics later).

**Detection:** If the ingested bank has suspiciously few facts, check whether `has_answer` filtering is active.

**Phase relevance:** Dataset parsing.

### Pitfall 12: `answer_session_ids` Indexing Off-By-One

**What goes wrong:** `answer_session_ids` lists the session indices containing evidence, but it's ambiguous whether they're 0-indexed or 1-indexed. Using the wrong convention means evidence-recall calculations point to the wrong sessions.

**Why it happens:** Python is 0-indexed, but the LoCoMo harness uses 1-indexed session numbering (`1..=ingest_sessions`). The LongMemEval dataset uses 0-indexed `answer_session_ids` (matching Python list indexing of `haystack_sessions`).

**Prevention:** Check the first few instances manually: look at `answer_session_ids`, load those sessions, and verify they contain the evidence mentioned in the `answer` field. Add an assertion in the test suite.

**Detection:** Evidence-recall metrics consistently at 0% or near-random.

**Phase relevance:** Evaluation metric implementation.

### Pitfall 13: M Setting Memory Pressure from Embedding Computation

**What goes wrong:** The M setting has ~1.5M tokens per instance across ~500 sessions. If embedding computation loads all session text into memory simultaneously (for batch embedding), a single instance can consume several GB of RAM. With concurrent instance-jobs, this can OOM.

**Prevention:**
1. Embed session-by-session during ingestion, not all at once.
2. Set `--instance-jobs 1` for M setting initially.
3. Monitor RSS during a single M-instance ingestion before scaling concurrency.

**Detection:** OOM killer or process crash during M ingestion.

**Phase relevance:** M setting support (later phase).

### Pitfall 14: Merging Partial Runs with Overlapping Questions

**What goes wrong:** The merge subcommand combines two partial runs that both scored the same `question_id`, producing duplicate entries and inflated/deflated accuracy.

**Why it happens:** Unlike LoCoMo where each conversation has distinct questions, LongMemEval has a flat list of 500 questions. If two partial runs overlap (e.g., both ran questions 1-100), merge must deduplicate.

**Prevention:**
1. Merge should key by `question_id` and reject duplicates (or take the later run's answer).
2. The LoCoMo merge already validates non-overlap for conversations. Adapt this to question-level deduplication.

**Detection:** Merged artifact has total_questions > 500.

**Phase relevance:** Merge subcommand implementation.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Dataset parsing | Category string mapping, `has_answer` misuse, timestamp array alignment, wrong dataset version | Use `question_type` strings directly; zip dates+sessions with length assertion; require `_cleaned` files |
| Ingestion infrastructure | 500 banks exhaust pool, wall-clock explosion, no bank cleanup | Explicit pool config, instance-level concurrency, incremental bank-map writes |
| Judge implementation | Abstention scoring wrong, judge model mismatch with published baselines | Two judge prompts (factual + abstention), default GPT-4o for publication |
| Metric calculation | Category totals wrong, denominator excludes abstention, off-by-one in evidence indexing | Count all 500 questions, validate per-category against published distributions |
| Resume/merge | Lost ingestion work, duplicate questions in merge | Incremental artifact writes, question-level dedup in merge |
| M setting | Memory pressure, multi-day runtime, disk exhaustion | Low instance concurrency, explicit pool sizing, disk monitoring, bank GC |
| Comparison validity | Wrong dataset version, non-GPT-4o judge, consolidation strategy mismatch | Dataset fingerprint in manifest, judge model in manifest, locked profiles |

## Sources

- [LongMemEval paper (arXiv:2410.10813)](https://arxiv.org/abs/2410.10813)
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval) -- dataset format, evaluation scripts, README
- [LongMemEval cleaned dataset (HuggingFace)](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) -- September 2025 update
- [Emergence.ai SOTA on LongMemEval](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag) -- implementation experience, RAG parameter sensitivity
- [Mastra Observational Memory (95% on LongMemEval)](https://mastra.ai/research/observational-memory) -- alternative architecture achieving high scores
- [Hindsight conv-26 comparison](bench/locomo/internal/hindsight-conv26-comparison-2026-03-10.md) -- category-label mapping bug, judge failure experience, metric source mismatch (HIGH confidence, first-party)
- [LoCoMo harness source](bench/locomo/locomo.rs) -- existing patterns for category mapping, judge retry, resume, consolidation (HIGH confidence, first-party)
- [Bench instrumentation notes](bench/locomo/internal/bench-instrumentation.md) -- in-process metering design decisions (HIGH confidence, first-party)
- Elephant MEMORY.md -- prompt tuning history, reflect agent behavior, consolidation architecture (HIGH confidence, first-party)

