# Benchmark Instrumentation Status

Protocol reference: [../benchmark-protocol.md](../benchmark-protocol.md)

## Implemented

The benchmark now instruments LLM-backed stages with a shared collector:

- `retain_extract`
- `retain_resolve`
- `retain_graph`
- `retain_opinion`
- `reflect`
- `consolidate`
- `opinion_merge`
- `judge`

Implementation:

- [src/metrics.rs](/home/austin/elephant/src/metrics.rs)
- [src/runtime.rs](/home/austin/elephant/src/runtime.rs)
- [bench/locomo/locomo.rs](/home/austin/elephant/bench/locomo/locomo.rs)

The metering path wraps `LlmClient.complete` directly, so token counts come from the same provider responses the product already returns in `CompletionResponse`.

## What the benchmark artifact now records

- Prompt tokens
- Completion tokens
- Call count
- Error count
- Aggregate latency

Both per stage and as a run total.

## Why the design changed

The original plan assumed the benchmark harness would keep talking to the server over HTTP and somehow reconstruct server-internal stage cost. That was the wrong abstraction boundary.

The benchmark now runs Elephant in process against the shared runtime builder. This is the cleaner benchmark design because it:

- uses the real pipelines
- avoids adding benchmark-only data to public API responses
- can meter every stage precisely
- keeps the server and benchmark wired from the same construction code

## Still missing

- Prompt/version hashes in the run manifest
- Human-audited judge calibration sample
- Cost normalization helpers for external reporting

## Recommendation

Do not add stage-token fields to the public REST API just for benchmarking unless a real product use case appears. The in-process runner is now the preferred benchmark path.
