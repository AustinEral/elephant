# Consolidation Prompt Enrichment — Future Improvements

Potential improvements to the consolidation prompt context, beyond the current approach of full-pipeline recall + simple text formatting.

## 1. Temporal metadata on facts

Append date information to each fact line so the LLM can reason about temporal ordering:

```
[0] Alice ran a charity race on Sunday, May 21, 2023 | date: 2023-05-21
[1] Alice changed jobs to Meta | date: 2024-03-10
[2] Alice's favorite color is blue
```

Facts without a `temporal_range` get no suffix. This helps with:
- Contradiction resolution: "used to work at Google, now works at Meta as of 2024"
- Temporal ordering: knowing which fact is newer when merging

Our facts already have `temporal_range` populated during extraction (the LLM resolves relative dates like "last Sunday" against the input timestamp). Just not surfaced in the consolidation prompt yet.

## 2. Enriched observation context

Show supporting evidence count and date range for each existing observation:

```
[obs-id] Alice enjoys running and participates in charity events.
  Sources: 2 supporting facts | Date range: 2023-05-21 to 2023-05-21
```

Helps the LLM understand how well-established an observation is and its temporal scope, aiding merge/update decisions. An observation backed by 8 facts over 6 months is more established than one with 1 fact.

## 3. Source fact text in observation context

Embed the full text of supporting facts for each observation:

```json
{
  "id": "obs-id",
  "text": "Alice enjoys running and participates in charity events.",
  "proof_count": 2,
  "occurred_start": "2023-05-21",
  "occurred_end": "2023-05-21",
  "source_memories": [
    {"text": "Alice ran a charity race on May 21, 2023", "date": "2023-05-21"},
    {"text": "Alice found the race rewarding", "date": "2023-05-21"}
  ]
}
```

This is the most expensive approach (lots of tokens) but gives the LLM the richest context for merge decisions. It can see exactly what evidence supports each observation and make more informed update/create choices.

Trade-off: significantly increases prompt size. A bank with well-established observations (10+ source facts each) could blow up the context. Would need a token budget or cap on source facts per observation.

## 4. Fact UUIDs instead of indices

Use fact UUIDs as keys instead of integer indices:

```
[a1b2c3d4-...] Alice ran a charity race on May 21, 2023
```

The LLM then references `source_fact_ids: ["a1b2c3d4-..."]` instead of `fact_indices: [0, 1]`. More robust (no off-by-one errors) but costs more tokens per fact line and in the output.

## Implementation priority

1. Temporal metadata on facts — low cost, high value for contradiction handling
2. Enriched observation context (count + date range) — low cost, moderate value
3. Source fact text — high cost, high value for complex merges
4. UUID addressing — moderate cost, marginal value (indices work fine)
