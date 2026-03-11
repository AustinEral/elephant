# Benchmark Instrumentation Status

Protocol reference: [../protocol.md](../protocol.md)

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

- [`src/metrics.rs`](../../../src/metrics.rs)
- [`src/runtime.rs`](../../../src/runtime.rs)
- [`bench/locomo/locomo.rs`](../locomo.rs)

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

- Human-audited judge calibration sample
- Cost normalization helpers for external reporting
- Public-facing rendering of prompt hashes/runtime tuning/bank stats in the result card/viewer

## Recommendation

Do not add stage-token fields to the public REST API just for benchmarking unless a real product use case appears. The in-process runner is now the preferred benchmark path.
