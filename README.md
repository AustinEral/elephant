# Elephant

A Rust implementation of the [Hindsight](https://arxiv.org/abs/2512.12818) memory architecture for AI agents. Four-network memory (world, experience, observation, opinion), three core operations (retain, recall, reflect), with TEMPR retrieval and RRF fusion.

## Architecture

```
Reflect (CARA)  →  Recall (TEMPR)  →  Storage (pgvector)
     ↑                   ↑
Retain Pipeline    Embedding (local ONNX / API)
     ↓
  LLM Extraction → Entity Resolution → Fact Storage
```

**Memory networks:**
- **World** — objective facts ("PostgreSQL supports JSONB")
- **Experience** — personal events ("Alice joined Acme in 2020")
- **Observation** — synthesized summaries across related facts
- **Opinion** — subjective beliefs with confidence scores

**Operations:**
- **Retain** — extract structured facts from natural language via LLM, resolve entities, store with embeddings
- **Recall** — TEMPR retrieval (temporal, entity, meaning, preference, recency) with reciprocal rank fusion
- **Reflect** — answer questions by reasoning over recalled memory context

**Background consolidation:**
- Observation synthesis (merge related facts about an entity)
- Opinion merging (detect consistent, contradictory, or superseded opinions)
- Mental model generation (cross-cutting patterns from observations)

## Interfaces

### MCP Server

Elephant exposes tools via [MCP](https://modelcontextprotocol.io/) (streamable HTTP):

- `retain` — store new memories
- `recall` — semantic memory search
- `reflect` — reasoned Q&A over memory
- `list_banks` — list memory banks
- `create_bank` — create or get a memory bank

### REST API

Same operations available as HTTP endpoints on port 3000.

## Setup

### Requirements

- Rust (2024 edition)
- PostgreSQL 16 with pgvector extension
- An LLM API key (Anthropic or OpenAI-compatible)
- For local embeddings: ONNX Runtime 1.23.0 + bge-small-en-v1.5 model

### Environment

```bash
cp .env.example .env  # then fill in values
```

Required variables:
```
DATABASE_URL=postgres://user:pass@localhost/elephant
LLM_API_KEY=your-api-key
LLM_MODEL=claude-sonnet-4-5-20250929
```

For local embeddings (recommended):
```
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_PATH=./models/bge-small-en-v1.5
ORT_DYLIB_PATH=./lib/onnxruntime-linux-x64-1.23.0/lib/libonnxruntime.so
```

For API embeddings:
```
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=your-key
EMBEDDING_API_MODEL=text-embedding-3-small
EMBEDDING_API_DIMS=384
```

### Docker

```bash
docker compose up
```

This starts Postgres (pgvector) and Elephant with local embeddings. Set your LLM API key in `.env`.

### From source

```bash
cargo build --release
./target/release/elephant
```

## Testing

```bash
# Unit + mock integration tests (no external deps besides Docker for testcontainers)
cargo test

# Real integration tests (requires .env with API keys)
cargo test --test real_integration_tests -- --ignored

# Prompt evaluation tests (requires LLM_API_KEY)
cargo test --test prompt_eval -- --ignored --nocapture
```

See [tests/README.md](tests/README.md) for details.

## Project Structure

```
src/
  retain/        # Fact extraction, entity resolution, storage
  recall/        # TEMPR retrieval, RRF fusion
  reflect/       # CARA reasoning, disposition system
  consolidation/ # Background observation/opinion/mental model synthesis
  storage/       # PostgreSQL + pgvector
  embedding/     # Local ONNX and OpenAI-compatible embeddings
  llm/           # Anthropic and OpenAI-compatible LLM clients
  mcp/           # MCP server adapter
  server/        # Axum HTTP server
  types/         # Shared types
prompts/         # LLM prompt templates
migrations/      # SQL migrations
```

## References

- [Hindsight: A Memory Architecture for AI Agents](https://arxiv.org/abs/2512.12818)
