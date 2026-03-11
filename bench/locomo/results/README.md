# Benchmark Results Layout

`bench/locomo/results/` is split by purpose:

- `local/`: transient local outputs from the current runner
- `canonical/`: merged or otherwise final artifacts worth keeping as the benchmark record
- `archive/legacy-v0/`: deprecated single-file artifacts from the pre-2026-03-10 harness
- `archive/external-hindsight/`: archived one-off comparison artifacts produced outside the Elephant runner

Guidelines:

- Treat `local/` as scratch space.
- Use `merge` plus `--out bench/locomo/results/canonical/<name>.json` when you intentionally want to promote a series to a canonical artifact.
- Do not use anything under `archive/legacy-v0/` for external comparison or publication. Those files predate the current benchmark contract and sidecar schema.
- Treat `archive/external-hindsight/` as reference material only. Those artifacts are useful for internal comparison, but they do not follow Elephant's native artifact schema and may include one-off runner caveats.
