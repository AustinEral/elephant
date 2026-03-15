# Codebase Concerns

**Analysis Date:** 2025-03-15

## Tech Debt

**Docker Configuration Hardcoding:**
- Issue: Model paths, max_seq_len, and retriever parameters are hardcoded in Dockerfile ENV variables (lines 59-64, 84-87). Changing reranker model requires rebuilding Docker image. RERANKER_MAX_SEQ_LEN fixed at 512 (runtime.rs line 358) even though MiniLM is the only model tested; different models may need different values.
- Files: `Dockerfile`, `src/runtime.rs` (lines 355-362)
- Impact: Poor flexibility for deployment. Can't swap models or tune reranker tokenization without rebuilding. External embedding provider (runtime-external stage) still hardcodes reranker model path, reducing true configurability.
- Fix approach: Move model paths and max_seq_len to runtime env vars with sensible defaults. Document default values per model (MiniLM=512).

**Consolidation Network Filtering Gap:**
- Issue: During consolidation, per-fact recall searches ONLY the Observation network (network_filter at `src/consolidation/observation.rs` line 240). This means facts from different consolidation batches cannot resolve cross-entity references if the linking observation doesn't yet exist.
- Files: `src/consolidation/observation.rs` (lines 229-242)
- Impact: "Becoming Nicole" cross-entity problem: when facts about two entities are in separate batches and the linking observation hasn't been created yet, the LLM cannot synthesize connections even if both facts are in memory. Retrieval pulls correct facts, but synthesis step fails to connect them.
- Fix approach: Two options: (1) search all networks (World+Experience+Observation) during consolidation recall to surface related raw facts as bridges, or (2) implement two-pass consolidation. Hindsight uses `include_source_facts=True` in consolidation recall — investigate whether surface-source-facts-in-context helps without full network expansion.

**Reranker Temporal Context Using Only Start Date:**
- Issue: `format_reranker_input()` at `src/recall/reranker/mod.rs` lines 73-82 only uses temporal_range.start for date formatting, ignores end date. For events with date ranges, this loses information passed to the cross-encoder.
- Files: `src/recall/reranker/mod.rs` (lines 73-82)
- Impact: Reranker sees "[Date: June 15, 2024]" for a 3-month event instead of "June 15 – September 15, 2024". May reduce temporal relevance scoring for range-based queries.
- Fix approach: Format both start and end dates when both present. Use readable format like "June 15 – September 15, 2024" to maximize cross-encoder context.

**Reflect Read-Only Design Removes Opinion Formation:**
- Issue: Reflect pipeline no longer forms new opinions (line 741 in `src/reflect/mod.rs` hardcodes `new_opinions: vec![]`). Read-only design is intentional per CARA architecture, but means opinions are only reinforced via retain path, never enriched via reasoning. Previously polluted banks had reflect-generated opinions that inflated opinion counts.
- Files: `src/reflect/mod.rs` (line 741)
- Impact: Opinion network grows only from retain-extracted facts, not from user interactions during reflect. Reduced signal for preference-conditioned reasoning over longer sessions. Mitigation exists (use disposition/directives instead) but Opinion network underutilized.
- Fix approach: This is architectural. If future work includes agentic opinion formation, ensure separate pool/handling for reflect-generated vs. retain-extracted opinions to avoid mixing signals.

**Opinion Consolidation Skipped:**
- Issue: Consolidation only processes World and Experience facts for observation synthesis (line 204 in `src/consolidation/observation.rs`). Opinion network has no consolidation — observations are never merged or deduplicated.
- Files: `src/consolidation/observation.rs` (lines 202-207), `src/consolidation/opinion_merger.rs`
- Impact: Opinion network grows monotonically without deduplication. Long-running systems accumulate redundant opinions. Opinion merger exists as separate API endpoint (`/consolidate/merge-opinions`) but must be called manually.
- Fix approach: Integrate opinion merger into main consolidation pipeline as automated secondary step, or at minimum auto-trigger opinion merger when observation consolidation completes.

## Known Limitations

**Cross-Entity Resolution in Consolidation:**
- Problem: When facts about related entities are in different consolidation batches and the linking observation hasn't been created yet, the LLM sees isolated fact clusters and cannot synthesize connections.
- Example: Batch 1 creates "Alice" observation, Batch 2 creates "Bob" observation, but the "Alice is Bob's sister" fact is in Batch 3 and Batch 1/2 observations already committed.
- Files: `src/consolidation/observation.rs` (lines 229-242 — network_filter restricted to Observation only)
- Current mitigation: Topic-scoped batching reduces scope of unlinked facts. Higher consolidation batch size (batch_size()) reduces likelihood of mid-consolidation divergence.
- Test coverage: No integration test covering cross-batch entity linking.

**Consolidation Recall Budget Mismatch:**
- Issue: Consolidation uses 512-token recall budget (consolidation.rs line 125, env var CONSOLIDATION_RECALL_BUDGET). This was matched to Hindsight's limit but no evidence in codebase that this is optimal for our multi-network, topic-scoped design.
- Files: `src/consolidation/observation.rs` (lines 120-126)
- Impact: May retrieve too few observations when facts have many entity cross-references. Retrieved context stored in bench results (enables forensic analysis) but no automated heuristic for tuning.
- Test coverage: Bench results track retrieved_context — analysis shows temporal/multi-hop benefited from better consolidation context, but analysis not automated.

## Performance Bottlenecks

**Reflect Agentic Loop Depth:**
- Problem: Reflect tool-calling loop has hardcoded max_iterations (8 by default, configurable via REFLECT_MAX_ITERATIONS). Forced iteration sequence: iter 0→search_observations, 1→recall, 2..last-1→auto. If LLM enters long reasoning chain, hits iteration cap and falls through to forced done-only at iteration last.
- Files: `src/reflect/mod.rs` (lines 41, 393-396), `src/reflect/hierarchy.rs`
- Symptom: "exhaustion" failures in benchmark (4% of failures in reflect-hierarchical run). LLM forced to finalize prematurely when max_iterations exceeded.
- Improvement path: Implement graceful fallback synthesis (use best accumulated context) instead of hard failure. Or: track reasoning depth heuristically and extend max_iterations for complex multi-hop queries.

**Consolidation Batch Processing Single-Threaded:**
- Problem: While consolidation fetches per-fact recall results in parallel (join_all at line 247), the batch loop itself is sequential. Large unconsolidated backlogs process one batch at a time.
- Files: `src/consolidation/observation.rs` (lines 222-247)
- Impact: Consolidation latency scales linearly with batch count. For 1000 unconsolidated facts at batch_size=8, requires 125 batch iterations.
- Improvement path: Process multiple batches in parallel (e.g., rayon or tokio::spawn) while maintaining per-bank consistency. Benchmark consolidation latency at scale.

**Reranker Tokenizer Truncation Every Call:**
- Problem: Local ONNX reranker tokenizes and truncates every fact at RERANKER_MAX_SEQ_LEN (512) on each rerank call. No caching of tokenizations.
- Files: `src/recall/reranker/local.rs`
- Impact: For recall queries returning 100+ candidates before reranking, tokenization cost accumulates. Typical bench run processes 154 questions, each with 50-100 rerank candidates.
- Improvement path: Pre-tokenize facts during retain/consolidation and cache token counts. Lazy-evaluate reranker input formatting.

## Fragile Areas

**Entity Resolution Deduplication Cache:**
- Files: `src/retain/resolver.rs` (lines 200-250 estimated, batch-scoped cache)
- Why fragile: Local cache keyed by canonical name persists only for duration of single resolve() call. If batch contains duplicate entity mentions, deduplication works. If mentions span multiple batches or retain calls, creates duplicate entities. This was a previous bug (fixed per memory context) but relies on retain-level batching to prevent recurrence.
- Safe modification: Document batch boundary assumptions in EntityResolver trait. Add assertion or test confirming batch-scoped caching behavior. Consider persistent dedup store (PostgreSQL unique constraint) as secondary safety layer.
- Test coverage: `src/retain/resolver.rs` contains tests but limited to single-batch scenarios.

**Consolidation LLM Structured Output:**
- Files: `src/consolidation/observation.rs` (lines 86-97, ConsolidateAction struct), lines 150-350+ (LLM call)
- Why fragile: LLM must produce structured JSON with exact action keywords ("CREATE", "UPDATE") and valid indices. If LLM produces invalid JSON or hallucinated observation_ids, deserialization fails and batch aborts.
- Safe modification: Add fallback synthesis (treat as prose observation if struct parse fails). Validate action.observation_id against existing observations before using. Log malformed responses for analysis.
- Test coverage: Unit tests use mocked LLM; no integration test with real LLM failure cases.

**Reflect Tool-Calling Loop State Management:**
- Files: `src/reflect/mod.rs` (lines 400-650 estimated, tool-calling loop implementation)
- Why fragile: Loop maintains mutable state (pending tools, seen facts, trace) across iterations. If LLM returns invalid tool calls (unknown tools, missing required fields), code path may panic or drop state.
- Safe modification: Validate tool names and args against ToolDef schema before processing. Use Result instead of unwrap for serde deserialization. Add comprehensive logging at each state transition.
- Test coverage: Mock LLM in tests always returns well-formed tools; no negative test for malformed ToolCall structs.

**Consolidation Temporal Merging Logic:**
- Files: `src/consolidation/observation.rs` (lines 147-169, merge_temporal function)
- Why fragile: Merges temporal ranges using LEAST(start)/GREATEST(end) logic. If fact has None/None temporal_range, logic passes through correctly, but implicit assumptions about temporal_range validity not documented.
- Safe modification: Document invariant: temporal_range must have start < end if both present. Add test for edge cases (start=end single day, None start with end, inverted ranges).
- Test coverage: Tests at lines 640+ cover basic temporal merge, but no adversarial test for inverted ranges.

## Security Considerations

**No Rate Limiting on API Endpoints:**
- Risk: `/v1/banks/{id}/retain`, `/v1/banks/{id}/recall`, `/v1/banks/{id}/reflect` accept arbitrary request body without size limits or per-user rate limiting. An attacker could exhaust database or LLM API quota.
- Files: `src/server/handlers.rs`
- Current mitigation: Axum stack can add tower middleware, but not currently in use.
- Recommendations: Add request size limits (max 10MB payload). Add per-bank rate limiting (e.g., 100 requests/minute via Redis or in-memory tracker). Log suspicious activity (rapid fire requests, huge payloads).

**LLM API Keys in Logs:**
- Risk: If LLM request fails, error messages may contain API keys from Authorization headers (low risk) or database credentials from DATABASE_URL (high risk).
- Files: `src/llm/mod.rs`, `src/runtime.rs`, error handling throughout
- Current mitigation: Codebase uses thiserror/anyhow; careful with what gets logged in error responses.
- Recommendations: Never log full error context from LLM clients. Redact DATABASE_URL in log output. Use structured logging with field redaction.

**No Input Validation on Bank Names/Directives:**
- Risk: Bank name, mission, directives fields accept arbitrary text. If these are embedded in LLM prompts without escaping, could enable prompt injection.
- Files: `src/server/handlers.rs` (bank creation handler), `src/reflect/mod.rs` (lines 752-764 build_profile_prompt)
- Current mitigation: Profile prompt concatenates directives/disposition verbatim. LLM is unlikely to be "jailbroken" by user-provided directives, but defense-in-depth missing.
- Recommendations: Validate bank metadata fields (reject scripts, URLs). Sanitize directive text before embedding in prompts (quote/escape or use structured format).

## Test Coverage Gaps

**Consolidation with Real LLM:**
- What's not tested: Consolidation behavior with actual LLM responses to malformed facts, ambiguous entity names, temporal conflicts.
- Files: `src/consolidation/observation.rs` (entire module — tests use mocked LLM)
- Risk: Silent failure if consolidation LLM produces unexpected output; failures only surface in benchmark results after hours of running.
- Priority: High — consolidation is critical path and most failure-prone.

**Reflect Tool-Calling Edge Cases:**
- What's not tested: LLM returning invalid tool names, missing required args, hallucinated FactId references, circular tool calls (search_observations → recall → search_observations).
- Files: `src/reflect/mod.rs` (tool loop implementation), `src/reflect/hierarchy.rs`
- Risk: Loop may hang or panic if LLM produces pathological sequences; no timeout per iteration.
- Priority: High — affects user-facing API.

**Reranker Model Swap:**
- What's not tested: Switching from MiniLM to different cross-encoder (e.g., bge-reranker) without code changes.
- Files: `src/recall/reranker/local.rs`, `Dockerfile`
- Risk: max_seq_len hardcoded at 512; new model might need 256 or 1024. Silent truncation/padding errors.
- Priority: Medium — only relevant if scaling to multiple models.

**Temporal Range Invariants:**
- What's not tested: Facts with inverted temporal ranges (end < start), ranges with only end date, temporal merging under edge cases (None + Some, overlaps, gaps).
- Files: `src/consolidation/observation.rs` (merge_temporal), `src/types/` (TemporalRange definition)
- Risk: Silent data corruption if invalid ranges created; affects temporal retrieval and consolidation.
- Priority: Medium — edge case but impacts correctness.

---

*Concerns audit: 2025-03-15*
