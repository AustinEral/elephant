# Accuracy Evals

`tests/evals/` is Elephant's internal accuracy test area.

The goal is to turn discovered failures into small, durable product tests.

## Principles

- Cases should be **ours**.
- Cases should come from observed failures or known weak points, not invented examples.
- Each case should target the **smallest owning component** possible.
- Cases should be small, explicit, and easy to rerun.
- One case file should map to one real test.
- `guard` cases should be stable.
- `tracking` and `limitation` cases should make current weak points obvious.

## Layout

```text
tests/evals/
  README.md
  extract/
```

Additional component directories can be added later as needed.
Case files should be self-contained by default.

## Case Lifecycle

When a failure is found:

1. Identify the owning stage.
2. Reduce it to the smallest reproducible case.
3. Add it to `tests/evals/...`.
4. Mark it as one of:
   - `guard`
   - `tracking`
   - `limitation`
5. After a real fix, promote `tracking` cases to `guard`.

Do not add placeholder cases just to fill out the structure. Add cases only when we decide they should be part of the suite.
There is currently one trivial smoke guard so the framework has an obvious passing path.

## Status Meanings

- `guard`
  - should pass reliably
- `tracking`
  - known gap; visible but not blocking
- `limitation`
  - known unsolved boundary; documented, not expected to pass right now

## Assertions

Common assertion types should be composable:

- `fact_contains`
- `fact_not_contains`
- `metadata_present`
- `metadata_equals`
- `retrieval_contains`
- `answer_contains`
- `answer_not_blank`
- `score_at_least`

The same top-level case format should work across components. Each component runner decides which assertions are valid.

## Current Shape

The current extract suite uses `datatest-stable`.

- Each JSON file under `tests/evals/extract/` is one test.
- The case contains its own transcript input.
- Two test targets read the same case files:
  - `evals_validate` validates case structure and supported assertion shape.
  - `evals_extract` runs the real extractor against the case input.

Validate case files:

```bash
cargo test --test evals_validate
```

Run live extract checks:

```bash
source .env
cargo test --test evals_extract
```
