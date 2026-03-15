# Benchmarks

Accuracy benchmarks for Elephant memory.

- **[LoCoMo](locomo/README.md)** — Long-term conversational memory (ACL 2024). Elephant benchmark claims are based on LoCoMo Categories 1-4 only, using the in-process runner and evidence-aware artifacts.
- **[LongMemEval](longmemeval/README.md)** — Long-term memory evaluation (Wu et al., 2024). 500 questions testing five core memory abilities across per-instance conversation histories.
- **view** — Inspect or compare LoCoMo artifacts: `cargo run --bin view -- <a.json> [b.json]`
- **longmemeval-view** — Inspect or compare LongMemEval artifacts: `cargo run --bin longmemeval-view -- <a.json> [b.json]`
