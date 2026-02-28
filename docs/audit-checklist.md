# Codebase Audit Checklist

Audit performed Feb 2026 against Hindsight paper (arXiv:2512.12818).

## Correctness

- [x] **upsert_entity ID bug** (`src/storage/pg.rs`)
  - Fixed: `ON CONFLICT DO UPDATE SET ... RETURNING id` now returns the actual row ID.

- [x] **Silent disposition failure** (`src/reflect/mod.rs`)
  - Fixed: error now propagated with `?` instead of swallowed.

- [x] **No FTS index** (`migrations/001_init.sql`)
  - Fixed: added `CREATE INDEX idx_facts_fts ON facts USING GIN(to_tsvector('english', content))`.

## Design Gaps

- [ ] **No cross-encoder reranker** (`src/recall/reranker.rs`)
  - `NoOpReranker` (truncation) wired in production. Paper's TEMPR includes reranking step.
  - `Reranker` trait is ready — needs a real implementation (e.g., cross-encoder model or LLM-based).

- [x] **Inline opinion reinforcement prompt** (`src/retain/mod.rs`)
  - Fixed: extracted to `prompts/reinforce_opinion.txt`, loaded via `include_str!`.

## Test Gaps

- [x] **Entity resolver has no unit tests** (`src/retain/resolver.rs`)
  - Fixed: 7 unit tests covering all 4 layers + batch dedup + multi-entity resolution.
  - Added `MemoryStore` and `EmbeddingClient` blanket impls for `Arc<T>`.

## Minor

- [x] **OpenAI LLM client has no base_url** (`src/llm/openai.rs`)
  - Fixed: added optional `base_url` parameter to `OpenAiClient::new()` and `ProviderConfig`.
  - `LLM_BASE_URL` env var in main.rs.

- [x] **Real integration tests hardcode Anthropic** (`tests/real_integration_tests.rs`)
  - Fixed: uses `llm::build_client()` with `LLM_PROVIDER` and `LLM_BASE_URL` env vars.
