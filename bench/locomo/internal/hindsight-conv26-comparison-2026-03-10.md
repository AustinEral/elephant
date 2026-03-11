# Hindsight conv-26 Comparison (2026-03-10)

This note preserves the local Hindsight vs Elephant comparison run for LoCoMo `conv-26`.

It exists because the Hindsight run was expensive enough that we should not need to reconstruct the setup or conclusions from shell history later.

## Scope

- Conversation: `conv-26`
- Questions scored: `152`
- Elephant artifact: [`../results/local/series1-conv-26.json`](../results/local/series1-conv-26.json)
- Hindsight artifact archive: [`../results/archive/external-hindsight/2026-03-10-conv26/`](../results/archive/external-hindsight/2026-03-10-conv26/)

## Run Shapes

### Elephant

- Benchmark runner: `locomo-bench`
- Ingest: `session`
- Consolidation: `end`
- Question concurrency: `5`
- Reflect cap: unset
- Commit in artifact: `5cedc6d`

### Hindsight

- Local API server in Docker
- `HINDSIGHT_API_LLM_PROVIDER=anthropic`
- `HINDSIGHT_API_LLM_MODEL=claude-sonnet-4-6`
- `HINDSIGHT_API_RETAIN_LLM_MODEL=claude-sonnet-4-6`
- `HINDSIGHT_API_REFLECT_LLM_MODEL=claude-sonnet-4-6`
- `HINDSIGHT_API_CONSOLIDATION_LLM_MODEL=claude-sonnet-4-6`
- External one-off runner: [`hindsight_locomo_api_bench.py`](../results/archive/external-hindsight/2026-03-10-conv26/hindsight_locomo_api_bench.py)
- `--wait-consolidation`
- `--question-workers 5`

## Headline Comparison

| Metric | Elephant | Hindsight | Notes |
|---|---:|---:|---|
| Accuracy | `94.7%` | `93.4%` raw | Hindsight accuracy is judge-contaminated; see caveats |
| Accuracy band | `94.7%` | `93.4%-96.1%` | Hindsight had 4 judge failures and 6 definite wrong answers |
| Questions | `152` | `152` | Same conversation slice |
| Total time | `8114.93s` (`135.2m`) | `2210.67s` (`36.8m`) | Hindsight about `3.7x` faster overall |
| Ingest time | `4898.39s` (`81m 38s`) | `57.06s` | Hindsight about `86x` faster on ingest |
| Consolidation time | `1922.56s` (`32m 03s`) | `821.03s` (`13m 41s`) | Hindsight about `2.3x` faster |
| QA total time | `1293.96s` (`21m 34s`) | `1332.55s` (`22m 13s`) | Roughly equal |
| Avg QA / question | `8.51s` | `8.77s` | Hindsight slightly slower in QA |
| Total tokens | `4,998,181` | `17,011,185` | Hindsight number is server-side only, from `/metrics` delta |
| Total tokens excl. judge | `4,899,479` | `17,011,185` | Fairer cost comparison; Hindsight still about `3.5x` higher |
| Total LLM calls | `3276` | `445` | Hindsight uses fewer, much larger calls |

## Interpretation

- Hindsight was much faster at building memory for `conv-26`.
- That speed advantage came almost entirely from ingest and consolidation.
- QA latency was basically the same between systems.
- Hindsight was substantially more token-expensive than Elephant on this run, even before adding the local judge tokens used by the one-off comparison runner.
- Answer quality was in the same band. This was not a decisive accuracy win for either system.

## Important Caveats

### Hindsight judge failures

The Hindsight comparison runner used a local Anthropic judge. That judge produced many invalid-JSON retries and `4` hard failures.

- Raw Hindsight score in the archived JSON: `142/152 = 93.4%`
- Definite wrong answers: `6`
- Judge failures: `4`

So Hindsight's true answer quality for this run is bounded roughly by:

- lower bound: `93.4%`
- upper bound: `96.1%`

This means the run is reliable for speed and token comparison, but not as a perfectly clean final answer-accuracy number.

### Archived run category-label bug

The external runner used for this archived run had a category-label mapping bug for LoCoMo category ids `1` and `2`.

That means the archived Hindsight JSON has the `single-hop` and `multi-hop` names swapped.

Affected per-category rows in the archived Hindsight JSON:

- reported `multi-hop 30/32` should be read as `single-hop 30/32`
- reported `single-hop 32/37` should be read as `multi-hop 32/37`

This does **not** affect:

- total accuracy
- timing
- token totals
- call counts

The helper script has since been corrected for future reruns. The bug only affects the archived `hindsight-conv26.json` category labels from this one comparison run.

### Metric source mismatch

The Hindsight token totals come from server-side Prometheus metrics before/after the run.

That means:

- Hindsight totals include server-side retain / consolidation / reflect
- Hindsight totals do **not** include the external local judge tokens
- Elephant totals do include judge tokens in the main artifact

For the closest cost comparison, use:

- Elephant `4,899,479` tokens excluding judge
- Hindsight `17,011,185` server-side tokens

## Raw Files

Archived under [`../results/archive/external-hindsight/2026-03-10-conv26/`](../results/archive/external-hindsight/2026-03-10-conv26/):

- `hindsight-conv26.json`
- `hindsight-conv26.before.prom`
- `hindsight-conv26.after.prom`
- `hindsight_locomo_api_bench.py`

## Bottom Line

For `conv-26`, Hindsight looked:

- much faster to build memory
- roughly equal on QA latency
- much more token-expensive
- roughly comparable on answer quality

This is useful as a local engineering reference, not a public benchmark claim.
