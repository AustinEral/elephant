# Benchmark Publication Plan

Protocol reference: [benchmark-protocol.md](benchmark-protocol.md)

## Current State

- **94.2%** on conv-26 (154 questions, Cat.1-4)
- Single conversation only — not a full leaderboard claim
- No cost/token tracking, no variance data

## Document Set

### External (public-facing)

| Document | Purpose |
|---|---|
| [benchmark-protocol.md](benchmark-protocol.md) | Single source of truth for methodology |
| [bench-result-card.md](bench-result-card.md) | Current result with all metadata |
| [bench-competitors.md](bench-competitors.md) | Competitive landscape and comparison policy |
| [results-format.md](results-format.md) | Results JSON format documentation |
| `bench/locomo/results/*.json` | Raw results |

### Internal (dev-only)

| Document | Purpose |
|---|---|
| [bench-plan.md](bench-plan.md) | This file — roadmap to publication |
| [bench-instrumentation.md](bench-instrumentation.md) | Token/cost tracking implementation |

## Phases

### Phase 1: Token Instrumentation (required before full run)

Add token counting to the pipeline. This is **required** before the full 10-conversation run — rerunning to add instrumentation is expensive. Mnemis and EverMemOS both publish meaningful cost disclosures; without them our results are incomplete.

Track per-stage: retain, consolidation, reflect, judge. Publish: tokens/question, total tokens, call counts, runtime.

See [bench-instrumentation.md](bench-instrumentation.md) for implementation details.

### Phase 2: Variance Measurement

Before claiming any number, establish variance bounds:
- Rerun judge-only pass on conv-26 (same bank, different judge randomness)
- Rerun full conv-26 pass (fresh bank, fresh extraction/consolidation)
- Report spread across runs

Backboard notes 2-3% variance with GPT-4.1 as judge. We need our own number.

### Phase 3: Second Conversation Gate

Run one more conversation (pick a structurally different one) with the same protocol. If both stay above ~90%, proceed to full run. If not, investigate what's different.

### Phase 4: Full Benchmark Run

Run all 10 conversations (~1,540 questions, Cat.1-4). This is the leaderboard claim. Requires:
- Token instrumentation from Phase 1
- Variance data from Phase 2
- Stable config (no more prompt tuning between conversations)
- Per-conversation breakdown
- Per-category breakdown
- Token/runtime table

### Phase 5: Publication

Format results for README, blog post, or paper. Include:
- Result card with all metadata
- Per-conversation table
- Per-category table
- Token/cost table with competitor comparison
- Comparison to published baselines (with evidence-type tags)
- Variance bounds
- Reproduction instructions

## Target Comparisons

| System | Score | Our target |
|---|---:|---|
| Mnemis | 93.9 | Beat on full run |
| EverMemOS | 93.05 | Beat on full run |
| Long-context GPT-5-mini | 92.85 | Beat (proves memory adds value) |
| Backboard | 90.00 | Above on conv-26 slice; not directly comparable yet |
| Hindsight | 89.61 | Above on conv-26 slice; not directly comparable yet |
