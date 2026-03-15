---
phase: 04-evaluation-path
plan: 01
subsystem: testing
tags: [llm-judge, benchmark, longmemeval, locomo, prompt-templates]

requires:
  - phase: 03-cli-artifact-infrastructure
    provides: "CLI harness structure, benchmark_prompt_hashes, RunConfig"
provides:
  - "Prompt-agnostic judge infrastructure in bench/common/judge.rs"
  - "5 LongMemEval judge prompt files matching upstream evaluate_qa.py"
  - "Prompt selection and rendering functions for LongMemEval QA evaluation"
  - "resolve_judge_model with gpt-4o default per EVAL-01"
  - "Judge prompt hash for manifest tracking"
affects: [04-02-qa-pipeline, evaluation]

tech-stack:
  added: []
  patterns: ["delegating wrapper pattern for extracted common modules", "include_str! constants for prompt templates", "per-type judge prompt selection"]

key-files:
  created:
    - bench/common/judge.rs
    - bench/longmemeval/prompts/judge_factual.txt
    - bench/longmemeval/prompts/judge_temporal.txt
    - bench/longmemeval/prompts/judge_knowledge_update.txt
    - bench/longmemeval/prompts/judge_preference.txt
    - bench/longmemeval/prompts/judge_abstention.txt
  modified:
    - bench/common/mod.rs
    - bench/locomo/locomo.rs
    - bench/longmemeval/longmemeval.rs

key-decisions:
  - "Exported JUDGE_TEMPERATURE/MAX_TOKENS/MAX_ATTEMPTS as pub const from common::judge for locomo BenchmarkRuntimeConfig"
  - "Used string equality (assert_eq!) instead of ptr::eq for prompt selection tests -- include_str! doesn't guarantee pointer identity"
  - "resolve_judge_model returns None when JUDGE_MODEL env var is set, letting common::judge use it directly"

patterns-established:
  - "Delegating wrappers: extracted common functions get thin wrappers in benchmark harness to minimize call-site diff"
  - "Per-type judge prompts: each question type routes to its own prompt template via select_judge_prompt"

requirements-completed: [EVAL-01, EVAL-02, EVAL-06]

duration: 6min
completed: 2026-03-15
---

# Phase 4 Plan 1: Judge Infrastructure and Prompts Summary

**Prompt-agnostic LLM judge module in bench/common/judge.rs with 5 LongMemEval per-type prompt templates and gpt-4o default**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-15T13:45:58Z
- **Completed:** 2026-03-15T13:52:09Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Extracted judge infrastructure (JudgeResponse, llm_judge, build_judge_client, judge_label) to bench/common/judge.rs for shared use
- Created 5 LongMemEval judge prompt files matching upstream evaluate_qa.py per-type variants (factual, temporal, knowledge-update, preference, abstention)
- Added prompt selection routing (select_judge_prompt), rendering (render_judge_prompt), deterministic hashing (judge_prompt_hash), and model resolution (resolve_judge_model) to LongMemEval harness
- LoCoMo updated with delegating wrappers -- all 29 existing tests pass, zero behavioral change

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract common judge module and update LoCoMo** - `07c5c49` (feat)
2. **Task 2: Judge prompts and prompt selection logic** - `a768d21` (feat)

## Files Created/Modified
- `bench/common/judge.rs` - Prompt-agnostic judge infrastructure (JudgeResponse, llm_judge, build_judge_client, judge_label)
- `bench/common/mod.rs` - Added pub mod judge
- `bench/locomo/locomo.rs` - Replaced inline judge code with delegating wrappers, cleaned unused imports
- `bench/longmemeval/prompts/judge_factual.txt` - Judge prompt for SSU, SSA, multi-session questions
- `bench/longmemeval/prompts/judge_temporal.txt` - Judge prompt with off-by-one tolerance for temporal reasoning
- `bench/longmemeval/prompts/judge_knowledge_update.txt` - Judge prompt tolerating prior + updated answers
- `bench/longmemeval/prompts/judge_preference.txt` - Rubric-based judge prompt for preference questions
- `bench/longmemeval/prompts/judge_abstention.txt` - Judge prompt for false-premise/unanswerable questions
- `bench/longmemeval/longmemeval.rs` - include_str! constants, select_judge_prompt, render_judge_prompt, judge_prompt_hash, resolve_judge_model, 11 new tests

## Decisions Made
- Exported judge constants (JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS, JUDGE_MAX_ATTEMPTS) as pub const from common::judge so locomo's BenchmarkRuntimeConfig can reference them
- resolve_judge_model returns None when JUDGE_MODEL env var is set, letting common::judge::build_judge_client read it directly; returns Some("gpt-4o") when neither config nor env var provides a model
- Used assert_eq! string comparison for prompt selection tests instead of ptr::eq since include_str! may not guarantee pointer identity across compilation units

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed unsafe env var operations for Rust 2024 edition**
- **Found during:** Task 2 (judge prompt tests)
- **Issue:** env::set_var and env::remove_var require unsafe blocks in Rust 2024 edition
- **Fix:** Wrapped env var manipulation in test EnvGuard with unsafe blocks
- **Files modified:** bench/longmemeval/longmemeval.rs
- **Verification:** All 134 tests pass
- **Committed in:** a768d21 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor syntactic fix for Rust 2024 edition safety requirements. No scope change.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Judge infrastructure ready for Plan 02 to wire QA evaluation into the pipeline loop
- select_judge_prompt + render_judge_prompt + common::judge::llm_judge form the complete judge call chain
- resolve_judge_model provides gpt-4o default for LongMemEval runs

---
*Phase: 04-evaluation-path*
*Completed: 2026-03-15*
