# Benchmark Rebuild Plan

Protocol reference: [../benchmark-protocol.md](../benchmark-protocol.md)

## Goal

Turn Elephant benchmarking from a tuning harness into a publication-grade evaluation system.

That means every serious run must answer:

- Was the answer correct?
- Did retrieval hit the supporting evidence?
- What did each stage cost?

## Required Benchmark Data

### Run manifest

- Dataset path and fingerprint
- CLI invocation
- Git commit and dirty-worktree status
- Category filter and question count
- Ingestion granularity
- Image handling policy
- Concurrency and consolidation mode

### System config

- Retain model
- Reflect/consolidation model
- Judge model
- Embeddings model
- Reranker model

### Stage metrics

- Prompt tokens
- Completion tokens
- LLM calls
- LLM errors
- Aggregate latency

Tracked per stage:

- retain_extract
- retain_resolve
- retain_graph
- retain_opinion
- reflect
- consolidate
- opinion_merge
- judge

### Per-question artifact

- Question text and gold answer
- Category
- Evidence refs from LoCoMo
- Hypothesis
- Judge label and reasoning
- Wall-clock latency
- Retrieved facts
- Retrieved turn refs
- Evidence hit / evidence recall
- Status and error field

## Architecture

### Implemented now

- Shared runtime builder in `src/runtime.rs`
- Stage-aware metering wrapper in `src/metrics.rs`
- In-process LoCoMo runner in `bench/locomo/locomo.rs`
- Turn-level ingest as the default benchmark path
- Turn provenance exported through `retrieved_context.source_turn_id`
- Evidence-aware scoring using LoCoMo `evidence`

### Why this design

The old HTTP harness could measure answer correctness, but it could not observe retain/reflect/consolidate cost cleanly because those LLM calls happened inside the server.

The new design evaluates Elephant directly in process against the same runtime wiring used by the server. That keeps benchmark instrumentation out of the public API while still benchmarking the real product stack.

## Remaining Work

### Phase 1: Artifact polish

- Add final bank construction stats to the public result card/viewer
- Surface prompt hashes and runtime tuning in the viewer/result card
- Add a machine-readable artifact index for multi-run comparisons

### Phase 2: Validation

- Clean conv-26 rerun with the new runner
- Judge-only rerun for variance
- One additional conversation rerun with the same frozen config

### Phase 3: Credibility checks

- Human audit sample for judge disagreements
- Long-context baseline run under the same artifact standard
- README/public result card update after clean reruns

### Phase 4: Full benchmark

Run all 10 conversations only after:

- config freeze
- stage metrics verification
- variance measurement
- clean Cat.1-4 artifact validation

## Publication Standard

Elephant should not publish a headline benchmark number until the run includes:

- full Cat.1-4 artifact set
- stage token disclosure
- evidence-aware retrieval reporting
- reproducible manifest
- at least one rerun or variance note
