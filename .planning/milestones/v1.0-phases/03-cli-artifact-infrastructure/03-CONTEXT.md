# Phase 3: CLI and Artifact Infrastructure - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Binary entry point, subcommands (run/ingest/qa), profile system (smoke/full-s/full-m), three-artifact output (summary JSON + question JSONL + debug JSONL), manifest with reproducibility contract, config overlay, `--instance` flag. Requirements CLI-01 through CLI-09, CLI-11.

No evaluation logic (Phase 4), no concurrency (Phase 5), no view tool (Phase 5).

</domain>

<decisions>
## Implementation Decisions

### Per-question bank isolation
- Each LongMemEval question has a uniquely compiled conversation history (needle-in-haystack with different distractor sessions per question). Contexts are genuinely unrelated — per-question bank isolation is the benchmark protocol, not an optimization choice.
- Confirmed by Backboard (93.4%, leading published result): "Each question gets its own isolated assistant and thread, so there's no cross-contamination between tests."
- Bank traceability: question_id → bank_id mapping stored in summary artifact for both resume and audit.

### Summary artifact structure
- Follow Backboard's pattern adapted to our artifact format:
  - **Summary JSON**: overall accuracy, per-category breakdown (7 types with counts), `banks` HashMap (question_id → bank_id, ~500 entries ≈ 15KB), manifest, stage metrics, timing
  - **Question JSONL sidecar**: one line per question — question_id, category, score, timing, bank_id, hypothesis, ground_truth, judge reasoning
  - **Debug JSONL sidecar**: one line per question — reflect trace, retrieved context, tool call history
- No per-instance section in summary — that's what the sidecar is for. Summary has aggregates + bank map.
- Per-category breakdown in summary matches LongMemEval's 7 categories: single-session-user, single-session-assistant, single-session-preference, multi-session, knowledge-update, temporal-reasoning, abstention

### Profile system
- Three profiles stored as JSON in `bench/longmemeval/profiles/`:
  - `smoke`: first instance only, S dataset — just verifies the pipeline runs end-to-end without error
  - `full-s`: all 500 instances, S dataset (~50 sessions per instance)
  - `full-m`: all 500 instances, M dataset (~500 sessions per instance)
- Profile provides defaults; `--config` JSON overlay overrides; CLI flags override everything (same layering as LoCoMo)

### Config overlay knobs (mapped from LoCoMo)
- `dataset` — path to dataset file (default from profile)
- `output` — output path
- `tag` — run tag
- `instances` — list of question_ids to run (replaces LoCoMo's `conversations`)
- `session_limit` — truncate haystack sessions per instance (for faster iteration)
- `ingest_format` — text/json (replaces LoCoMo's turn/session/raw-json ingest mode)
- `consolidation` — end/per-session/off
- `instance_jobs` — parallelism for instance processing (single axis, replaces LoCoMo's conversation_jobs + question_jobs)
- `judge_model` — judge model override
- Resolution order: profile defaults → `--config` JSON → CLI flags

### Subcommands
- `run` — full pipeline: ingest + consolidate + QA, writes all three artifacts
- `ingest` — ingest + consolidate only, writes summary with bank mappings (no QA scores)
- `qa <artifact>` — score against existing banks from ingest artifact, writes new summary + sidecars
- No `merge` in Phase 3 (deferred to v2)

### Output paths
- Default: `bench/longmemeval/results/local/{tag}.json` (or `{profile}-{command}.json` if no tag)
- Sidecars: `{stem}.questions.jsonl`, `{stem}.debug.jsonl`
- `--out` overrides the default path

### CLI parsing
- Manual arg parsing (no clap), matching LoCoMo pattern
- `--instance <question_id>` repeatable flag for selecting specific instances
- `qa` takes artifact path as positional arg: `qa <artifact.json>`

### Claude's Discretion
- Exact struct field names and types for LongMemEval-specific artifacts (BenchmarkOutput, QuestionResult, etc.)
- How `run` handles the Phase 4 dependency — stub QA path or only wire ingest+consolidate initially
- Profile JSON file contents (specific default values)
- Manifest struct design (adapted from LoCoMo's BenchmarkManifest)
- Error handling for missing dataset files, invalid instance IDs, etc.

</decisions>

<specifics>
## Specific Ideas

- Backboard organizes results by category folders + consolidated JSON. We achieve the same information via per-category aggregates in summary + category field in JSONL sidecar, without directory overhead.
- LongMemEval paper: "During test time, sessions are provided to the system one by one" — sequential ingestion is correct, parallelism is at the instance level (different questions), not within a question.
- Hindsight evaluates on S setting (500 questions). Our `full-s` profile matches their setup for direct comparison.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bench/locomo/locomo.rs`: CLI parsing pattern (ParsedCli, CliOverrides, RunConfig, FileRunConfig, resolve_fresh_config/resolve_qa_config)
- `bench/locomo/locomo.rs`: BenchmarkOutput, BenchmarkManifest, QuestionResult struct patterns — adapt for LongMemEval
- `bench/locomo/locomo.rs`: Output path resolution (default_output_path) and artifact validation
- `bench/locomo/profiles/`: Profile JSON pattern (smoke.json, full.json)
- `bench/common/io.rs`: `append_jsonl()`, `sidecar_path()` — shared, ready to use
- `bench/common/fingerprint.rs`: `fnv1a64()`, `fnv1a64_hex()` — shared, ready to use

### Established Patterns
- Profile resolution: load profile JSON → apply FileRunConfig → apply CLI overrides
- QA mode: extract config from existing artifact manifest, restrict which flags are allowed
- Manifest: dataset fingerprint, prompt hashes, runtime config, git commit, CLI invocation
- JSONL incremental flush: write question/debug records as they complete, not at end

### Integration Points
- `bench/longmemeval/longmemeval.rs`: Binary entry point (currently stub)
- `bench/longmemeval/ingest.rs`: `ingest_instance()` — already built, returns IngestResult with bank mapping
- `bench/longmemeval/dataset.rs`: `LongMemEvalInstance`, `QuestionType`, dataset loading — already built
- `bench/common/`: Shared fingerprint + I/O utilities

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-cli-artifact-infrastructure*
*Context gathered: 2026-03-15*
