# Prompt Caching Research

**Project:** Elephant
**Topic:** Provider prompt caching support for OpenAI and Anthropic
**Researched:** 2026-03-17
**Confidence:** HIGH

## Executive Summary

Prompt caching is implementable in this codebase without a large architectural rewrite, but only if the first cut targets the providers' automatic caching modes. That fits the current abstraction in [`src/types/llm.rs`](/home/austin/elephant/src/types/llm.rs#L88), [`src/llm/openai.rs`](/home/austin/elephant/src/llm/openai.rs#L44), and [`src/llm/anthropic.rs`](/home/austin/elephant/src/llm/anthropic.rs#L66) with localized changes.

The current system cannot yet express Anthropic's explicit block-level `cache_control` placement cleanly. Anthropic allows both automatic and explicit caching, but explicit placement would require richer message/block types because the current request model stores `system` as a single string and message content as plain text plus tool metadata. That is a larger refactor than the change you described.

The right first implementation is:

1. Add prompt-cache request options and prompt-cache usage fields to the shared LLM types.
2. Wire OpenAI Chat Completions request/response fields and Anthropic Messages automatic caching fields.
3. Extend internal metrics to preserve cache-hit/write data.
4. Add benchmark artifact and view support only after the core library behavior and tests are stable.

## Current Code Impact

### Shared request/response seam

The narrowest integration seam is the shared LLM types in [`src/types/llm.rs`](/home/austin/elephant/src/types/llm.rs#L88). Today:

- `CompletionRequest` has no prompt-caching fields.
- `CompletionResponse` only preserves `input_tokens`, `output_tokens`, `stop_reason`, and `tool_calls`.
- There is no place to preserve cached-token counts, cache-read counts, or cache-write counts.

That means prompt caching cannot be measured end-to-end yet even if the providers are already doing work on the backend.

### Provider clients

OpenAI request building in [`src/llm/openai.rs`](/home/austin/elephant/src/llm/openai.rs#L44) currently serializes:

- `model`
- `messages`
- `max_tokens`
- `temperature`
- `tools`
- `tool_choice`

It does not serialize `prompt_cache_key` or `prompt_cache_retention`, and its response parser only reads `prompt_tokens` and `completion_tokens`.

Anthropic request building in [`src/llm/anthropic.rs`](/home/austin/elephant/src/llm/anthropic.rs#L66) currently serializes:

- `model`
- `messages`
- `max_tokens`
- `temperature`
- `system`
- `tools`
- `tool_choice`

It does not serialize top-level `cache_control`, and its response parser only reads `input_tokens` and `output_tokens`.

### Metrics and benchmarks

The metering layer in [`src/metrics.rs`](/home/austin/elephant/src/metrics.rs#L42) only aggregates:

- `prompt_tokens`
- `completion_tokens`
- `calls`
- `errors`
- `latency_ms`

Benchmark artifacts in [`bench/longmemeval/longmemeval.rs`](/home/austin/elephant/bench/longmemeval/longmemeval.rs#L817) already serialize `stage_metrics`, and the viewer in [`bench/longmemeval/view.rs`](/home/austin/elephant/bench/longmemeval/view.rs#L139) is tolerant on deserialization. That makes benchmark support a clean later phase: extend the schema without breaking existing result files.

## Provider Research

### OpenAI

Official prompt caching guidance says it works automatically on API requests. The guide also documents:

- `prompt_cache_key` as a routing/bucketing hint for improving cache-hit rates on shared prefixes.
- `prompt_cache_retention` with `in_memory` and `24h`.
- `usage.prompt_tokens_details.cached_tokens` as the observable cache-hit counter.
- a 1024-token minimum before any prompt tokens can be served from cache.

Implementation implications for this repo:

- OpenAI support is mostly request/response field plumbing, not a message-model rewrite.
- `prompt_cache_key` is optional for correctness. It is a hit-rate optimization.
- `cached_tokens` should be preserved in `CompletionResponse` and aggregated in metrics even if we do not surface benchmark UX immediately.

### Anthropic

Official prompt caching guidance documents two modes:

- automatic caching via a top-level `cache_control` field
- explicit block-level caching via `cache_control` on cacheable blocks

The docs also matter on these details:

- default TTL is 5 minutes
- explicit TTL values are `5m` and `1h`
- automatic caching is specifically positioned for growing multi-turn conversations
- usage accounting is split into `cache_read_input_tokens`, `cache_creation_input_tokens`, and `input_tokens`
- cacheability minimums are model-specific, not universal
- placement rules changed on May 1, 2025 for `tool_result` and `document.source`

Implementation implications for this repo:

- automatic caching fits the current abstraction
- explicit block-level caching does not fit the current abstraction cleanly
- the response accounting is richer than OpenAI's and should be preserved instead of collapsed away

## Chosen Architecture

### Automatic caching only in the first pass

The initial implementation will support:

- OpenAI automatic prompt caching
- Anthropic automatic prompt caching

The initial implementation will not support:

- Anthropic explicit block-level `cache_control`

Reason:

- the current prompt abstraction is not rich enough to expose explicit block tagging cleanly
- automatic caching covers the highest-value path in this repo, especially `reflect`
- this keeps the first implementation narrow enough to ship and measure

### Provider-specific cache config at runtime/client setup

Prompt-cache config will live at the concrete provider client/runtime layer, not on every `CompletionRequest`.

Reason:

- cache settings are transport/provider knobs, not part of the logical prompt payload
- env parsing already happens centrally in runtime/client setup in this project
- the judge currently shares the same settings, so one setup-time config path is simpler

Chosen shape:

- enum-based provider cache config passed into concrete provider clients
- shared prompt-cache usage returned through `CompletionResponse`
- `CompletionRequest` stays provider-agnostic in the first pass

### Cache metrics added in the core phase

The core implementation will preserve returned cache usage metrics immediately.

Chosen additions to internal usage tracking:

- `cached_prompt_tokens`
- `cache_read_input_tokens`
- `cache_creation_input_tokens`

These should default to zero for backward compatibility.

## Highest-Value Call Sites

### 1. Reflect loop

[`src/reflect/mod.rs`](/home/austin/elephant/src/reflect/mod.rs#L255) is the highest-value prompt caching target.

Reason:

- it resends growing conversation state across iterations
- Anthropic automatic caching is designed for this shape
- OpenAI prefix reuse should also be strongest here

### 2. Consolidation and large structured prompts

[`src/consolidation/observation.rs`](/home/austin/elephant/src/consolidation/observation.rs#L288) and [`src/consolidation/opinion_merger.rs`](/home/austin/elephant/src/consolidation/opinion_merger.rs#L130) send large prompt templates plus dynamic content. These are worth instrumenting because they are often long enough to clear provider cacheability thresholds.

### 3. Retain extract / resolve / graph

These paths use repeated prompt templates but more varied user payloads. They are still worth supporting, but expected cache value is lower and should be measured rather than assumed.

### 4. Benchmark judge

[`bench/common/judge.rs`](/home/austin/elephant/bench/common/judge.rs#L34) is likely a lower-yield target because requests may stay under cacheability thresholds. Still worth tracking because the benchmark may generate many near-identical judge prompts.

## Risks and Non-Goals

### Risk 1: Anthropic explicit caching is a larger refactor than it looks

If you try to support explicit `cache_control` now, you will end up redesigning message/block types, tool-result serialization, and possibly prompt construction helpers.

That is not required for a useful first release.

### Risk 2: Cache metrics are not directly comparable across providers

OpenAI reports cache hits as cached prompt tokens.

Anthropic reports:

- read tokens
- creation tokens
- uncached trailing input tokens

Do not flatten these into one ambiguous metric in storage or UI. Preserve raw fields first, derive rollups later.

### Risk 3: Model thresholds differ materially

OpenAI uses a 1024-token floor.

Anthropic has model-specific minimum cacheable lengths. That means:

- a "cache enabled" request is not proof a given stage is benefiting
- benchmark support must show actual cache counters, not just a boolean flag

## Locked Decisions

These decisions were confirmed on 2026-03-17 before implementation starts:

- Anthropic scope for initial rollout: automatic caching only
- OpenAI scope for initial rollout: automatic caching support
- Stage coverage: all stages
- Config surface: env/runtime knobs from the beginning
- Cache enablement default: off by default
- Judge settings: share the same cache settings for now
- OpenAI `prompt_cache_key`: supported but not required

## Config Decisions

The agreed first-pass config shape is:

- `LLM_PROMPT_CACHE_ENABLED=0|1`
- `OPENAI_PROMPT_CACHE_KEY` optional
- `OPENAI_PROMPT_CACHE_RETENTION` optional
- `ANTHROPIC_PROMPT_CACHE_TTL` optional

Current intent:

- one shared config surface for runtime and judge
- no separate judge-specific cache envs in the first pass
- caching stays disabled unless explicitly enabled

## Finalized Implementation Direction

The current implementation direction is now:

- provider-specific prompt-cache config at the client/runtime layer
- enum-based cache config passed into concrete provider clients
- shared prompt-cache usage returned through the generic response type
- env parsing performed once during runtime/client setup, matching the current project style
- provider clients receive already-structured cache settings
- cache enablement off by default
- OpenAI `prompt_cache_key` supported but only used when explicitly configured

Illustrative shape:

```rust
pub enum PromptCacheConfig {
    Disabled,
    OpenAi(OpenAiPromptCacheConfig),
    Anthropic(AnthropicPromptCacheConfig),
}

pub struct OpenAiPromptCacheConfig {
    pub key: Option<String>,
    pub retention: Option<OpenAiPromptCacheRetention>,
}

pub struct AnthropicPromptCacheConfig {
    pub ttl: Option<AnthropicPromptCacheTtl>,
}

pub struct PromptCacheUsage {
    pub cached_tokens: Option<usize>,
    pub cache_read_input_tokens: Option<usize>,
    pub cache_creation_input_tokens: Option<usize>,
}
```

This means:

- runtime reads env once and constructs the provider-specific cache config
- OpenAI and Anthropic clients each receive only the config that applies to them
- `CompletionRequest` stays focused on prompt payloads rather than provider transport knobs
- `CompletionResponse` still exposes shared cache usage data for metrics and benchmarking

## Rejected for Now

- Anthropic explicit block-level cache tagging
- request-level provider-specific cache fields on `CompletionRequest`
- auto-generated fallback `OPENAI_PROMPT_CACHE_KEY`
- separate judge-specific cache envs in the first pass

## Sources

- OpenAI Prompt Caching guide: https://developers.openai.com/api/docs/guides/prompt-caching
- OpenAI Chat Completions API reference: https://platform.openai.com/docs/api-reference/chat/create-chat-completion
- Anthropic Prompt Caching guide: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Anthropic Messages API reference: https://platform.claude.com/docs/en/api/python/messages
- Anthropic release notes: https://platform.claude.com/docs/en/release-notes/overview
