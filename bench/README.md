# Benchmarks

Accuracy benchmarks using established memory evaluation datasets.

## Setup

### Download datasets

**LoCoMo** (10 conversations, ~2000 questions):
```bash
mkdir -p data
curl -o data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

### Build

```bash
cargo build --release --bin locomo-bench
```

### Environment

Config is loaded from `.env` (same file as the server). The judge LLM can be configured independently:

```
JUDGE_PROVIDER    — falls back to LLM_PROVIDER
JUDGE_API_KEY     — falls back to LLM_API_KEY
JUDGE_MODEL       — falls back to LLM_MODEL
```

## Running

Start Elephant first:
```bash
docker compose up -d
```

### LoCoMo

```bash
# Full benchmark (all 10 conversations, ~2000 questions)
cargo run --release --bin locomo-bench

# Quick smoke test (1 conversation, 10 questions)
cargo run --release --bin locomo-bench -- --max-conversations 1 --max-questions 10

# Parallel questions (5 at a time per conversation)
cargo run --release --bin locomo-bench -- --question-concurrency 5

# Parallel conversations and questions
cargo run --release --bin locomo-bench -- --conversation-concurrency 2 --question-concurrency 5

# Run with consolidation (synthesizes entity observations before questions)
cargo run --release --bin locomo-bench -- --question-concurrency 5 --consolidate

# Resume from previous results (reuses bank IDs, skips ingestion)
cargo run --release --bin locomo-bench -- --resume bench/results/locomo.json --question-concurrency 5

# Resume with consolidation
cargo run --release --bin locomo-bench -- --resume bench/results/locomo.json --consolidate --question-concurrency 5

# Reuse a single bank (skip ingestion for first conversation)
cargo run --release --bin locomo-bench -- --bank-id <BANK_ID> --max-conversations 1

# Override judge model via CLI
cargo run --release --bin locomo-bench -- --judge-model some-model-name
```

Results are saved to `bench/results/locomo.json`, including a `banks` map of `sample_id → bank_id` for resuming.

## Scoring

Two metrics are computed for each question:

- **LLM-as-judge** (primary): Binary CORRECT/WRONG via the configured LLM. This is the industry standard used by Hindsight, Backboard, and LongMemEval.
- **Token F1** (reference): Token-level overlap between prediction and gold answer. Penalizes verbose answers; included for diagnostics only.

## Benchmarks

### LoCoMo
- **Paper**: [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753) (ACL 2024)
- **What it tests**: Long-term conversational memory across multi-session dialogues
- **Categories**: single-hop, multi-hop, temporal, open-domain, unanswerable
- **Pipeline**: Ingest sessions via retain → (optional) consolidate → ask questions via reflect → score with LLM judge

### LongMemEval (planned)
- **Paper**: [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) (ICLR 2025)
- **What it tests**: 5 memory abilities across 500 questions
