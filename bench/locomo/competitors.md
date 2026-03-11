# LoCoMo Competitive Analysis

Last audited: 2026-03-09

Protocol reference: [protocol.md](protocol.md)

## Comparison Policy

- Only compare Elephant directly to runs that disclose scope, question count, and judge setup.
- Single-conversation slices are tuning signals, not leaderboard claims.
- Full-benchmark Cat.1-4 runs are the only valid headline comparison.
- Methodology quality matters as much as score quality.

## Methodology References

These are the strongest public examples of benchmark hygiene worth copying.

| System | What they publish | Why it matters | Elephant action | Source |
|---|---|---|---|---|
| **Mnemis** | Judge model, 96% judge-vs-human agreement, 56M total tokens, 8,622s runtime | Makes the reported score auditable and gives readers a quality check on LLM judging | Add judge validation and publish total cost/runtime | [arXiv:2602.15313](https://arxiv.org/pdf/2602.15313.pdf) |
| **EverMemOS** | Full 1,540-question LoCoMo eval plus stage-by-stage token totals (add, search+answer, evaluate) | Lets readers reason about cost amortization instead of accuracy alone | Instrument retain, reflect, and judge by stage | [arXiv:2601.02163](https://arxiv.org/pdf/2601.02163.pdf) |
| **Backboard** | `filtered_answers.json` (1540 answers), per-session result folders, compare mode, explicit note about 2-3% judge variance | Shows clean benchmark scoping and artifact publication | Keep Cat.1-4 filtering explicit, publish per-conversation JSON, measure rerun variance | [GitHub](https://github.com/Backboard-io/Backboard-Locomo-Benchmark) |

## Well-Sourced Published Scores

Only systems with directly inspectable methodology or primary-source papers are listed here.

| System | Score | Scope | Notes | Source |
|---|---:|---|---|---|
| **Mnemis** | 93.9 | Full LoCoMo, Cat.1-4 | Different backbone; strongest disclosure quality | [arXiv:2602.15313](https://arxiv.org/pdf/2602.15313.pdf) |
| **EverMemOS** | 93.05 | Full LoCoMo, 1,540 QA pairs | Strongest stage-cost disclosure | [arXiv:2601.02163](https://arxiv.org/pdf/2601.02163.pdf) |
| **Long-context GPT-5-mini** | 92.85 | LoCoMo long-context baseline | Important non-memory baseline | [arXiv:2603.04814](https://arxiv.org/pdf/2603.04814.pdf) |
| **Backboard** | 90.00 | Filtered repo benchmark | Different judge stack; repo artifacts are strong | [GitHub](https://github.com/Backboard-io/Backboard-Locomo-Benchmark) |

## Elephant Status

Elephant does **not** currently have a benchmark-valid public comparison number.

- Elephant now has a clean current-harness `conv-26` reference run.
- That run is still single-conversation only, so it is a tuning/reference slice rather than a headline benchmark claim.
- A full Cat.1-4 run plus at least one rerun/variance note is still required before citing Elephant competitively.

## Common Standard To Follow

If Elephant wants to be taken seriously, the benchmark workflow should match the strongest competitor practice:

- Use the official LoCoMo preprocessing contract: BLIP-2 captions inline, no raw dataset leakage in the benchmark claim.
- Enforce Cat.1-4 filtering in code, not in prose.
- Fix the answer stack and judge stack before the published run; no mid-run tuning.
- Publish per-question JSON and per-conversation aggregates.
- Publish stage-by-stage token usage and total runtime.
- Record git commit, run timestamp, and exact CLI invocation.
- Measure at least one rerun or judge-only rerun before making a headline claim.
- Compare against both memory competitors and a strong long-context baseline.

## Bottom Line

The right benchmark target is not just "beat Mnemis on score." The real target is:

1. Match Backboard on artifact hygiene.
2. Match EverMemOS on stage-cost disclosure.
3. Match Mnemis on judge validation and run provenance.
4. Then compete on accuracy.
