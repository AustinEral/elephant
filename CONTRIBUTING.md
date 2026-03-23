# Contributing

Thanks for working on Elephant.

The most useful contributions are the ones that improve correctness, clarity, reproducibility, and operational trust.

## Before You Start

Read these first:

- [README.md](README.md)
- [docs/getting-started.md](docs/getting-started.md)
- [docs/api.md](docs/api.md)
- [docs/deployment.md](docs/deployment.md)
- [STYLE.md](STYLE.md)

If you are touching benchmark code or benchmark claims, also read:

- [bench/locomo/README.md](bench/locomo/README.md)
- [bench/locomo/protocol.md](bench/locomo/protocol.md)
- [bench/locomo/result-card.md](bench/locomo/result-card.md)

## Contribution Priorities

High-value contributions usually fall into one of these buckets:

- bug fixes in retain, recall, reflect, or storage behavior
- benchmark rigor and reproducibility improvements
- public documentation improvements
- deployment hardening for self-hosted users
- tests that cover real regressions or failure modes

Lower-value contributions are ones that add surface area without improving reliability or clarity.

## Development Workflow

1. Start from a clean branch off `main`.
2. Make the smallest change that solves the problem.
3. Add or update tests when the change affects behavior.
4. Run the relevant checks locally.
5. Update docs if the public behavior changes.

## Local Setup

For the fastest path from zero to a working server, use:

- [docs/getting-started.md](docs/getting-started.md)

For day-to-day work, most contributors will want:

- PostgreSQL with `pgvector`
- a configured `.env`
- valid provider credentials

## Coding Standards

Follow:

- [STYLE.md](STYLE.md)

Practical expectations:

- keep changes focused
- prefer explicitness over cleverness
- preserve benchmark reproducibility
- do not silently change public benchmark framing
- do not mix unrelated refactors into targeted fixes

## Testing

Run the narrowest useful checks for your change.

Common commands:

```bash
cargo test
```

```bash
cargo test --bin locomo-bench --no-run
```

If you change the server API, quickstart, or deployment docs, validate the documented path if practical.

If you change benchmark code:

- keep the benchmark contract stable unless the change is explicitly about methodology
- document any scoring or artifact-format changes
- avoid replacing historical benchmark claims casually

## Benchmarks and Claims

Benchmark numbers are part of the project’s public surface.

If your change affects benchmark behavior or documentation:

- update the relevant benchmark docs
- keep the README and result card consistent
- note whether a number is historical, fresh, same-judge, or otherwise conditioned

Do not introduce a stronger public claim than the repo can support.

## Pull Requests

When opening a PR, make it easy to review:

- explain the problem being solved
- explain the intended behavioral change
- list the validation you ran
- mention any benchmark or docs impact

Small, well-scoped PRs are strongly preferred over broad “cleanup + feature + docs” batches.

## Documentation

If the change affects public usage, update the relevant docs in the same PR:

- [README.md](README.md)
- [docs/getting-started.md](docs/getting-started.md)
- [docs/api.md](docs/api.md)
- [docs/deployment.md](docs/deployment.md)

## Security

If you believe you found a security issue, do not open a public issue with exploit details.

See:

- [SECURITY.md](SECURITY.md)
