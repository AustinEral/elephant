# Phase 1: Dataset Foundation - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Parse and validate LongMemEval dataset files (`longmemeval_s_cleaned.json`, `longmemeval_m_cleaned.json`), producing correctly typed, categorized, and fingerprinted instances ready for ingestion. 500 questions across 7 categories. No ingestion, no evaluation, no CLI — just the data layer.

</domain>

<decisions>
## Implementation Decisions

### Code sharing with LoCoMo
- Extract shared bench infrastructure into `bench/common/` module — both harnesses import from it
- Share where it's natural: FNV fingerprinting, artifact I/O (JSONL helpers, sidecar paths), judge evaluation, stage metrics
- Refactor LoCoMo when straightforward and aligned with multi-bench future (more benchmarks may be integrated later)
- When a LoCoMo refactor would be painful or risky, defer it as tracked tech debt and use a decoupled LongMemEval-specific implementation for now
- Guiding principle: avoid painful tech debt for the future, but don't take on everything right now

### Validation approach
- Two-layer validation: (1) serde JSON parse (fail-on-first, unavoidable), (2) semantic validation pass over all 500 instances collecting all errors before reporting
- All validation runs upfront before any pipeline work begins — no failing an hour into a bench run
- Semantic validation checks: question_type membership, session/date length match (DATA-07), answer coercion success, required fields present

### Dataset file handling
- Dataset files live in `data/` (same convention as LoCoMo's `locomo10.json`), gitignored
- When dataset file is missing, provide a helpful error with download instructions pointing to `xiaowu0162/longmemeval-cleaned` on HuggingFace
- Manual download only — no HF SDK dependency

### Claude's Discretion
- Serde struct design for LongMemEval instances (field types, enum variants)
- Question type enum variant naming (mapped from the 7 LongMemEval categories)
- Answer coercion implementation (`serde_json::Value` to string)
- FNV fingerprinting extraction approach from LoCoMo
- Internal module organization within `bench/longmemeval/`

</decisions>

<specifics>
## Specific Ideas

- LoCoMo harness is a single 148KB file — LongMemEval should follow similar patterns but leverage the new `bench/common/` shared module
- The `_abs` suffix on `question_id` identifies false-premise/abstention questions (relevant for Phase 4, but the type system should accommodate it now)
- Dataset structure per instance: `question_id`, `question_type`, `question`, `question_date`, `answer`, `haystack_sessions` (array of turns), `haystack_dates`, `answer_session_ids`

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bench/locomo/locomo.rs`: FNV1a-64 fingerprinting (`fnv1a64()`, `fnv1a64_hex()`) — extract to `bench/common/`
- `bench/locomo/locomo.rs`: JSONL append helpers, sidecar path generation — extract to `bench/common/`
- `bench/locomo/locomo.rs`: Stage metrics types (`StageUsage`) — extract to `bench/common/`
- `data/locomo10.json`: Establishes `data/` as the dataset directory convention

### Established Patterns
- Dataset loading: `fs::read()` + `serde_json::from_slice()` (no streaming)
- Serde enums: `#[serde(rename_all = "snake_case")]` for category types
- ID types: newtype wrappers (`BankId`, `FactId`) via ULID
- Error handling: `anyhow` in binaries, `thiserror` in library code

### Integration Points
- `Cargo.toml`: New `[[bin]]` target for `longmemeval-bench`
- `bench/common/`: New shared module imported by both harnesses
- `data/`: Dataset file location

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-dataset-foundation*
*Context gathered: 2026-03-15*
