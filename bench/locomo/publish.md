# LoCoMo Publish Bundle

`locomo-bench publish` exports a public-facing bundle from one canonical benchmark artifact.

This bundle is meant for:

- GitHub Pages or another static host
- append-only benchmark history
- lightweight public inspection of one run

It is not the same as the native benchmark artifact contract under [`results-format.md`](results-format.md).

## Why a separate bundle exists

The benchmark-native contract is optimized for local reruns, QA-only passes, and strict `merge` compatibility checks.

The publish bundle is optimized for:

- immutable per-run paths
- small default payloads
- gzipped question records for static hosting
- optional large debug traces kept out of the default site payload

## Command

```bash
cargo run --release --bin locomo-bench -- \
  publish \
  bench/locomo/results/canonical/full.json \
  --out bench/locomo/published \
  --run-id 2026-03-10-series1
```

Optional debug export:

```bash
cargo run --release --bin locomo-bench -- \
  publish \
  bench/locomo/results/canonical/full.json \
  --out bench/locomo/published \
  --run-id 2026-03-10-series1 \
  --include-debug
```

## Output layout

```text
bench/locomo/published/
  index.json
  runs/
    2026-03-10-series1/
      summary.json
      questions.jsonl.gz
      debug.jsonl.gz        # optional
```

`index.json` is the root manifest for the published dataset. Each run gets one immutable directory under `runs/`.

## Files

### `index.json`

The top-level publication manifest. It lists every published run with:

- `run_id`
- publication timestamp
- source artifact path and fingerprint
- headline metrics
- relative paths to `summary.json`, `questions.jsonl.gz`, and optional `debug.jsonl.gz`

### `summary.json`

A compact public summary for one run. It includes:

- source artifact provenance
- public artifact paths
- run timestamp, tag, and commit
- model stack
- aggregate metrics
- per-category and per-conversation breakdowns
- manifest and stage-usage metadata

It intentionally does not embed question records inline.

### `questions.jsonl.gz`

The public question-level dataset for the run, compressed for static hosting.

This is the default detailed payload that should be linked from Pages.

### `debug.jsonl.gz`

Optional compressed trace payload. This is much larger than the question sidecar and should usually live in GitHub Releases, not in the default Pages payload.

## Recommended public repo workflow

Use a dedicated public repo, for example `elephant-benchmarks`.

Recommended shape:

1. Keep benchmark execution in the main Elephant repo.
2. Export publish bundles locally with `locomo-bench publish`.
3. Copy only the publish bundle into the public benchmarks repo under `locomo/`.
4. Commit `index.json`, `summary.json`, and `questions.jsonl.gz` there.
5. Upload `debug.jsonl.gz` to a GitHub Release there when needed.

Elephant should own the exporter and the native benchmark contract.

The public `elephant-benchmarks` repo should own:

- GitHub Pages configuration
- landing page or visualizer code
- published benchmark history

This keeps benchmark history append-only and prevents the main code repo from absorbing long-lived benchmark payloads or site scaffolding.
