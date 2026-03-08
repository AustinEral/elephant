# LoCoMo Competitive Analysis

Last updated: 2026-03-08

Protocol reference: [benchmark-protocol.md](benchmark-protocol.md)

## Comparison Policy

- We only compare Elephant directly to runs that disclose category scope, judge model, and question count
- Single-conversation results are stress tests, not leaderboard claims
- Full-benchmark results (all 10 conversations, Cat.1-4) are separated from per-conversation slices
- Competitor numbers are tagged by evidence type (self-reported, reproduced, comparison-table)

## Full-Benchmark Leaderboard

Systems ranked by published LoCoMo accuracy (Cat.1-4 unless noted):

| Rank | System | Score | Backbone | Evidence Type | Protocol Notes | Cost Data | Source |
|---:|---|---:|---|---|---|---|---|
| 1 | **Mnemis** | 93.9 | GPT-4.1-mini | Self-reported paper | LLM-as-judge, Cat.5 excluded | Yes (56M tokens) | [arXiv:2602.15313](https://arxiv.org/pdf/2602.15313) |
| 2 | **EverMemOS** | 93.05 | GPT-4.1-mini | Self-reported paper | Stage-by-stage token table | Yes (22M tokens) | [arXiv:2601.02163](https://arxiv.org/pdf/2601.02163) |
| — | **EverMemOS** | 92.3 | GPT-4.1-mini | Reproduced by Mnemis | Different eval setup | — | Mnemis comparison table |
| 3 | **Backboard** | 90.00 | GPT-4.1 judge | Repo benchmark | Cat.5 filtered, per-conv JSONs | No detailed tokens | [GitHub](https://github.com/Backboard-io/Backboard-Locomo-Benchmark) |
| 4 | **Hindsight** | 89.61 | Gemini-3 | Self-reported paper | — | No | [arXiv:2512.12818](https://arxiv.org/html/2512.12818v1) |
| 5 | **HyMem** | 89.55 | GPT-4.1-mini | Self-reported paper | — | Relative claim only | [arXiv:2602.13933](https://arxiv.org/html/2602.13933v1) |
| 6 | **Memvid** | 85.65 | — | Repo benchmark | Cat.1-4 | No | [GitHub](https://github.com/memvid/memvidbench) |
| 7 | **MIRIX** | 85.4 | — | Self-reported paper | — | No | [arXiv:2507.07957](https://arxiv.org/abs/2507.07957) |
| 8 | **EMem-G** | 85.3 | GPT-4.1-mini | Comparison-table result | From Mnemis paper | No | Mnemis comparison table |
| 9 | **MemBuilder** | 84.23 | Qwen3-4B (trained) | Self-reported paper | — | No | [arXiv:2601.05488](https://arxiv.org/html/2601.05488v1) |

Note: EverMemOS appears twice — 93.05 (self-reported) vs 92.3 (Mnemis reproduction). Both are useful but not identical claims.

## Elephant Position

| Metric | Elephant (conv-26) | Mnemis (full) | EverMemOS (full) |
|---|---|---|---|
| Overall accuracy | **94.2%** | 93.9% | 93.05% |
| Question count | 154 | 1,540 | ~1,540 |
| Conversations | 1 | 10 | 10 |
| Answer model | Sonnet 4.6 | GPT-4.1-mini | GPT-4.1-mini |
| Tokens/question | unknown | ~36.4k | ~14.3k |
| Evidence type | **Single-conv stress test** | Full benchmark | Full benchmark |
| Apples-to-apples | **No** | Baseline | Baseline |

**Key caveat**: Our 94.2% is from a single conversation. We need to run all 10 conversations to make a legitimate comparison. Conv-26 may be easier or harder than average.

## Cost Benchmarks

| System | Tokens/Question | Total Tokens | Runtime | Source |
|---|---|---|---|---|
| EverMemOS (GPT-4.1-mini) | ~14.3k (incl judge) | ~22M | not published | arXiv |
| Mnemis | ~36.4k | ~56M | 8,622s (~5.6s/q) | arXiv |
| Elephant | unknown | unknown | ~38.6s/q (reflect only) | internal |

## Long-Context Baseline

Per "Beyond the Context Window" (March 2026, [arXiv:2603.04814](https://arxiv.org/html/2603.04814v1)):

| System | LoCoMo Score | Cost |
|---|---:|---|
| Long-context GPT-5-mini | 92.85 | $14.79 / 504 requests |
| Long-context GPT-OSS-120B | 81.69 | $7.65 / 664 requests |
| Mem0-style fact memory | 57.68 | — |

Takeaway: strong long-context models can score ~93% on LoCoMo without any memory system. Memory systems must justify their existence via cost amortization (cheaper at scale) or accuracy (beating long-context baselines).

## Conv-26 Specific Intel

- **EverMemOS reproduction issue**: A user reported 38.38% on conv-26 ([GitHub issue #73](https://github.com/EverMind-AI/EverMemOS/issues/73))
- **PISA**: Uses conv-26 for hyperparameter analysis ([arXiv:2510.15966](https://arxiv.org/html/2510.15966v1))
- **Backboard**: Supports per-conversation JSON outputs; no published conv-26 numbers
- Most teams only publish aggregate totals — no public conv-26 leaderboard exists

## Systems to Watch

Priority order for competitive tracking:

1. **Mnemis** — best clean public number (93.9), good cost disclosure
2. **EverMemOS** — best cost disclosure (stage-by-stage tokens), but reproduction issues reported
3. **Hindsight** — architectural reference (Elephant implements this), 89.61
4. **Backboard** — good benchmark methodology (per-conv outputs), 90.0
5. **HyMem** — cost-efficiency focus, 89.55
