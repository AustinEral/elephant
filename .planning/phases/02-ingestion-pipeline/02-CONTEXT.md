# Phase 2: Ingestion Pipeline - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Per-instance bank creation, session ingestion with timestamps, and consolidation for LongMemEval. Each of the 500 questions gets its own bank with its conversation history ingested sequentially. No CLI, no evaluation, no concurrency — just the ingestion pipeline callable per-instance.

</domain>

<decisions>
## Implementation Decisions

### Session formatting
- Support two ingest formats via configuration: plain text (default) and raw JSON
- Plain text: `[Date: YYYY-MM-DD]\n\nuser: content\nassistant: content` — honest approach, tests Elephant's extraction on realistic dialogue
- Raw JSON: `json.dumps(cleaned_turns)` — matches Hindsight/Vectorize's approach for direct competitive comparison
- Configurable via `IngestFormat` enum (text/json), passed in `IngestConfig`
- Date prefix always `[Date: YYYY-MM-DD]` (strip day-of-week and time from LongMemEval's full format)
- `context` field on `RetainInput`: `None` for session-level ingestion — sessions are self-contained multi-turn dialogues, coreference is resolved within each session

### Consolidation strategy
- Three modes: `end` (default), `per-session`, `off` — configurable via `ConsolidationMode` enum
- Default `end`: ingest all sessions into the bank, then consolidate once
- Per-bank isolation: consolidation only operates within a single instance's bank
- Pool sizing is not a Phase 2 concern — sequential ingestion uses the existing pool. Phase 5 addresses concurrent pool sizing (INGEST-05)

### Return type and metrics
- `ingest_instance()` returns `IngestResult` with `question_id → bank_id` mapping for resume support
- Track per-instance stage metrics matching LoCoMo's `StageUsage`: facts stored, entities resolved, sessions ingested, duration, consolidation stats
- Bank mappings feed Phase 3's ingest artifact for `qa` subcommand resume

### Code organization
- Ingestion logic in `bench/longmemeval/ingest.rs` — separate from the binary entry point
- LongMemEval-specific session formatters (not shared with LoCoMo) — Turn struct differs (role/content vs speaker/text/blip_caption)
- Runtime (store, retain pipeline, consolidator) passed in as parameter, not built internally
- Function signature: `async fn ingest_instance(instance: &LongMemEvalInstance, runtime: &Runtime, config: &IngestConfig) -> Result<IngestResult>`

### Claude's Discretion
- `IngestConfig` and `IngestResult` struct design details
- How to strip `has_answer` field from turns before raw JSON formatting
- Error handling strategy for individual session failures within an instance
- Whether to log per-session progress or only per-instance

</decisions>

<specifics>
## Specific Ideas

- "We should support both [formats] or at least have a way to do what Hindsight does. We want to directly compete with them so it might be important. We should do it the intended way, but where Hindsight deviates, we need a way to compete."
- Hindsight passes `json.dumps(cleaned_turns)` with context string `"Session {id} - you are the assistant in this conversation - happened on {date}"` — our raw JSON mode should produce comparable input
- LoCoMo's `--conversation` flag pattern should inform Phase 3's `--instance` flag for running subsets

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/runtime.rs`: Factory that builds all pipeline components from env vars — `ingest_instance()` receives the built runtime
- `bench/locomo/locomo.rs`: `format_session()` and `format_session_raw()` as reference patterns for text/JSON formatting
- `bench/locomo/locomo.rs`: `StageUsage` metrics collection pattern — extract to `bench/common/` or replicate
- `bench/common/`: Shared FNV fingerprinting, JSONL I/O already extracted from Phase 1

### Established Patterns
- `RetainInput { bank_id, content, timestamp, turn_id: None, context: None, ... }` for session-level ingestion
- `store.create_bank()` for bank creation
- LoCoMo consolidation: `consolidate_with_bench_progress()` called after ingestion completes
- Sequential session ingestion with progress logging per session

### Integration Points
- `bench/longmemeval/longmemeval.rs`: Binary entry imports `ingest.rs` module
- `bench/longmemeval/dataset.rs`: `LongMemEvalInstance` and `Turn` types already defined
- `src/types/pipeline.rs`: `RetainInput`, `RetainOutput` contracts
- `src/consolidation/observation.rs`: `Consolidator` trait for post-ingestion consolidation

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-ingestion-pipeline*
*Context gathered: 2026-03-15*
