# Prompt Caching Plan

## Goal

Implement automatic prompt caching support for OpenAI and Anthropic across all LLM stages, with env-driven runtime configuration and shared cache usage metrics.

## Scope

- automatic caching only
- all stages
- caching off by default
- judge shares the same cache settings for now
- OpenAI `prompt_cache_key` supported but optional

## Phase 1

### 1. Shared types

- add provider-level `PromptCacheConfig` enum
- add provider-specific config structs:
  - `OpenAiPromptCacheConfig`
  - `AnthropicPromptCacheConfig`
- add shared response usage type:
  - `PromptCacheUsage`
- extend `CompletionResponse` to carry prompt-cache usage

### 2. Runtime/config plumbing

- parse env once during runtime/client setup
- thread structured cache config into:
  - main runtime LLM clients
  - shared `llm::build_client()`
  - benchmark judge client

### 3. Provider wiring

- OpenAI:
  - send `prompt_cache_key` when configured
  - send `prompt_cache_retention` when configured
  - parse `usage.prompt_tokens_details.cached_tokens`
- Anthropic:
  - send top-level automatic cache control when enabled
  - send configured TTL when provided
  - parse `cache_read_input_tokens`
  - parse `cache_creation_input_tokens`

### 4. Metrics

- extend `StageUsage` with cache counters
- update `MetricsCollector`
- keep defaults/backward compatibility for old JSON

## Phase 2

### Tests

- shared type serde tests
- OpenAI request/response unit tests
- Anthropic request/response unit tests
- metrics aggregation tests
- mock client updates for cache-aware responses

## Phase 3

### Benchmark support

- extend benchmark stage metrics output
- extend benchmark view output
- show cache counters by stage
- keep old artifacts readable

## Implementation order

1. shared types
2. runtime/env parsing
3. `llm::build_client()` and provider constructors
4. OpenAI client wiring
5. Anthropic client wiring
6. metrics updates
7. tests
8. benchmark schema/UI later

## Watch items

- avoid putting provider-specific cache knobs on `CompletionRequest` for the first pass
- preserve exact provider usage fields instead of flattening them too early
- do not auto-generate `OPENAI_PROMPT_CACHE_KEY`
- keep uncached behavior byte-for-byte unchanged when caching is disabled

## Env Contract

Lock these names unless implementation forces a change:

- `LLM_PROMPT_CACHE_ENABLED`
- `OPENAI_PROMPT_CACHE_KEY`
- `OPENAI_PROMPT_CACHE_RETENTION`
- `ANTHROPIC_PROMPT_CACHE_TTL`

Initial semantics:

- caching disabled by default
- `OPENAI_PROMPT_CACHE_KEY` is optional and only used when explicitly set
- judge shares the same cache settings in the first pass

## Close-out Checklist

After implementation and review:

1. Update relevant README files with prompt-caching config and behavior.
2. Update `.env.example` with the new cache env vars and comments.
3. Update local `.env` only if needed for manual verification.
4. Add or update any benchmark docs once Phase 3 lands.
5. Remove stale notes or temporary references if the final implementation differs from the plan.
