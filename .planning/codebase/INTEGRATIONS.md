# External Integrations

**Analysis Date:** 2026-03-15

## APIs & External Services

**LLM Providers:**
- Anthropic Claude API - Core reasoning and synthesis operations (retain, reflect)
  - SDK/Client: Custom HTTP client via `src/llm/anthropic.rs`
  - Auth: API key via `LLM_API_KEY` env var, auto-detects OAuth from `sk-ant-oat-` prefix
  - Endpoint: `https://api.anthropic.com/v1/messages`
  - Models: Configurable via `LLM_MODEL`, `RETAIN_LLM_MODEL`, `REFLECT_LLM_MODEL`
  - Tool calling: Support for tool definitions and tool results via Claude Messages API

- OpenAI API - Alternative LLM provider, OpenAI-compatible endpoints
  - SDK/Client: Custom HTTP client via `src/llm/openai.rs`
  - Auth: API key via `LLM_API_KEY` env var
  - Base URL: Configurable via `LLM_BASE_URL` for OpenAI-compatible providers
  - Feature: Fallback when Anthropic unavailable

**Embedding Providers:**
- OpenAI Embeddings API - Text-to-vector conversion (optional alternative)
  - SDK/Client: Custom HTTP client via `src/embedding/openai.rs`
  - Auth: `EMBEDDING_API_KEY` env var
  - Config: `EMBEDDING_API_MODEL`, `EMBEDDING_API_DIMS` (e.g., text-embedding-3-small, 1536 dims)
  - Used: Vector search foundation when EMBEDDING_PROVIDER=openai

**Reranking Providers:**
- Cohere Rerank API - Alternative reranking backend (optional)
  - SDK/Client: Custom HTTP client via `src/recall/reranker/api.rs`
  - Auth: `RERANKER_API_KEY` env var
  - Config: `RERANKER_API_URL`, `RERANKER_API_MODEL` (e.g., rerank-english-v3.0)
  - Endpoint: Cohere-compatible API format
  - Used: Ranking retrieved facts when RERANKER_PROVIDER=api

**Judge LLM (Optional):**
- Configurable judge model for LoCoMo benchmark evaluation
  - Auth: `JUDGE_API_KEY` env var (separate from main LLM)
  - Config: `JUDGE_PROVIDER` (anthropic/openai), `JUDGE_MODEL`
  - Fallback: Uses `LLM_*` credentials if judge credentials not provided

## Data Storage

**Databases:**
- PostgreSQL 16 with pgvector extension
  - Connection: `DATABASE_URL` env var (format: postgres://user:pass@host:port/dbname)
  - Client: SQLx with compile-time query verification
  - Migrations: Three SQL files in `migrations/` directory
    - `001_init.sql` - Core fact/entity/network schema with pgvector columns
    - `002_consolidated_at.sql` - Add consolidation tracking column
    - `003_sources.sql` - Add source tracking for fact provenance
  - Schema includes: facts, entities, graph_links, sources, fact_sources tables
  - Vector columns: Embedding vectors for similarity search (pgvector `vector` type)
  - Key operations: Vector similarity search, keyword search, graph navigation

**File Storage:**
- Local filesystem only - No cloud storage integration
- Models stored locally: Embedding models, reranker models, tokenizers
- Model paths: `./models/bge-small-en-v1.5/`, `./models/ms-marco-MiniLM-L-6-v2/`
- Docker: Models bundled in container at `/app/models/`

**Caching:**
- In-memory: Fact deduplication cache during retain pipeline (local HashMap, request-scoped)
- No distributed cache (Redis, Memcached) - All state in PostgreSQL

## Authentication & Identity

**Auth Provider:**
- None - No authentication system
- All API credentials passed via environment variables
- Bank ID (UUID) used for logical isolation within single PostgreSQL instance
- MCP: No authentication required (assumes trusted network)

**Access Control:**
- None enforced - Bank IDs are the only logical separation
- All credentials stored in `.env` (never committed, local development only)

## Monitoring & Observability

**Error Tracking:**
- None detected - No Sentry, Rollbar, or similar integration

**Logs:**
- Structured logging via Tracing framework
- Output: stdout with optional JSON formatting
- Levels: Configurable via `RUST_LOG` env var
- Log format: Plain text or JSON via `LOG_FORMAT` env var
- Modules: All components emit debug/info/warn/error logs

**Metrics:**
- Bench harness collects accuracy metrics per conversation
- Results stored in `bench/locomo/results/<tag>.json`
- Metrics: Question-level accuracy, per-question retrieval context, failure classification
- No live monitoring/dashboards - Static file analysis only

## CI/CD & Deployment

**Hosting:**
- Docker containers (self-hosted)
- docker-compose.yml for local development (Postgres + Elephant service)

**CI Pipeline:**
- None detected - No GitHub Actions, GitLab CI, or similar
- Local testing only via `cargo test`, benchmarking via `locomo-bench` binary

**Database Migrations:**
- SQLx migrations embedded in binary via `sqlx::raw_sql(include_str!(...))`
- Runs on startup via PgMemoryStore::new() initialization
- No external migration tool required

## Environment Configuration

**Required env vars:**
- DATABASE_URL: PostgreSQL connection (no default)
- LLM_API_KEY: Anthropic or OpenAI API credentials (no default)
- LLM_PROVIDER: "anthropic" or "openai" (required to select provider)
- EMBEDDING_PROVIDER: "local" or "openai" (required)
- EMBEDDING_MODEL_PATH: Required if EMBEDDING_PROVIDER=local
- RERANKER_PROVIDER: "local", "api", or "none" (optional, default: local)
- RERANKER_MODEL_PATH: Required if RERANKER_PROVIDER=local

**Optional env vars (with defaults):**
- LISTEN_ADDR: Default 0.0.0.0:3001
- RUST_LOG: Default "info"
- LOG_FORMAT: Default "" (plain text), set to "json" for structured logs
- LLM_TIMEOUT_SECS: Default 600 (10 minutes)
- LLM_MODEL: Default model for all LLM operations (overridden by tier-specific vars)
- RETAIN_LLM_MODEL: Override for extraction (default: uses LLM_MODEL)
- REFLECT_LLM_MODEL: Override for synthesis (default: uses LLM_MODEL)
- REFLECT_ENABLE_SOURCE_LOOKUP: Default 1 (enabled); set 0 to disable hierarchical recall in reflect
- DEDUP_THRESHOLD: Default 0.95; set to "none" to disable retain deduplication

**Secrets location:**
- `.env` file (local development, never committed)
- Environment variables (production, injected at runtime)
- Docker Secrets or external secret management recommended (not configured)

## Webhooks & Callbacks

**Incoming:**
- None - No webhook receivers implemented

**Outgoing:**
- None - No external system callbacks

## MCP (Model Context Protocol)

**Server:**
- Exposes elephant pipelines as MCP tools via rmcp 0.16.0
- Endpoint: `http://[LISTEN_ADDR]/mcp/`
- Session management: Local session manager (single-request isolation)
- Transport: Streamable HTTP (JSON-RPC 2.0 framing)

**Tools Exposed:**
- `retain(bank_id, content, context?, timestamp?)` - Store to memory
- `recall(bank_id, query, max_tokens?)` - Retrieve from memory
- `reflect(bank_id, query, context?, budget?)` - Synthesize with agentic search
- `create_bank(bank_id, name?, mission?)` - Create new memory bank
- `list_banks()` - List all banks in instance

**Tool Parameters:**
- All use JSON schema derived from schemars JsonSchema trait
- Types: String, Option<String>, usize primitives
- Validation: Bank ID must be valid UUID

---

*Integration audit: 2026-03-15*
