# Getting Started

Use this guide if you want a single path from zero to a running Elephant server.

There are two supported paths:

- Docker quickstart
- local source build

If you only want to try Elephant quickly, use Docker.

## Prerequisites

- an LLM API key for one supported provider
- Docker, if using the Docker path
- `jq`, if you want to copy the quickstart commands exactly

Supported LLM providers are documented in [`.env.example`](../.env.example).

## Option 1: Docker Quickstart

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and set at least:

- `LLM_PROVIDER`
- `LLM_API_KEY`
- `RETAIN_LLM_MODEL`
- `REFLECT_LLM_MODEL`

Optional server maintenance defaults are also available:

- `SERVER_AUTO_CONSOLIDATION`
- `SERVER_AUTO_CONSOLIDATION_MIN_FACTS`
- `SERVER_AUTO_CONSOLIDATION_COOLDOWN_SECS`
- `SERVER_AUTO_CONSOLIDATION_MERGE_OPINIONS`

The Docker image bundles:

- ONNX Runtime
- local embedding model
- local reranker model
- PostgreSQL + pgvector through `docker compose`

3. Start the stack:

```bash
docker compose up -d --build
```

4. Verify the server:

```bash
curl localhost:3001/v1/info
```

5. Create a bank:

```bash
export BANK_ID=$(
  curl -s localhost:3001/v1/banks \
    -H 'content-type: application/json' \
    -d '{
      "name": "demo",
      "mission": "Remember user facts, events, and preferences"
    }' | jq -r '.id'
)
```

6. Store a memory:

```bash
curl localhost:3001/v1/banks/$BANK_ID/retain \
  -H 'content-type: application/json' \
  -d "$(
    jq -nc --arg bank_id "$BANK_ID" '{
      bank_id: $bank_id,
      content: "Alice joined Acme Corp in March 2024. She prefers Rust over Go.",
      timestamp: "2024-03-01T00:00:00Z",
      turn_id: null,
      context: null,
      custom_instructions: null,
      speaker: "Alice"
    }'
  )"
```

7. Ask a question:

```bash
curl localhost:3001/v1/banks/$BANK_ID/reflect \
  -H 'content-type: application/json' \
  -d "$(
    jq -nc --arg bank_id "$BANK_ID" '{
      bank_id: $bank_id,
      question: "When did Alice join her company?",
      budget_tokens: 2048,
      temporal_context: null
    }'
  )"
```

The server also exposes MCP at:

```text
http://localhost:3001/mcp/
```

## Option 2: Local Source Build

Use this if you want to run Elephant directly with `cargo run`.

### 1. Start PostgreSQL + pgvector

The simplest path is Docker:

```bash
docker compose up -d postgres
```

The default local database URL is already set in `.env.example`:

```text
postgres://elephant:elephant@localhost:5433/elephant
```

### 2. Copy and edit `.env`

```bash
cp .env.example .env
```

Set at least:

- `DATABASE_URL`
- `LLM_PROVIDER`
- `LLM_API_KEY`
- `RETAIN_LLM_MODEL`
- `REFLECT_LLM_MODEL`

If you want to tune server-side background maintenance, also review:

- `SERVER_AUTO_CONSOLIDATION`
- `SERVER_AUTO_CONSOLIDATION_MIN_FACTS`
- `SERVER_AUTO_CONSOLIDATION_COOLDOWN_SECS`
- `SERVER_AUTO_CONSOLIDATION_MERGE_OPINIONS`

### 3. Install ONNX Runtime

```bash
mkdir -p lib
curl -sL https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-linux-x64-1.23.0.tgz | tar xz -C lib/
```

### 4. Download local models

Embedding model:

```bash
mkdir -p models/bge-small-en-v1.5
curl -Lo models/bge-small-en-v1.5/model.onnx https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx
curl -Lo models/bge-small-en-v1.5/tokenizer.json https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json
```

Reranker model:

```bash
mkdir -p models/ms-marco-MiniLM-L-6-v2
curl -Lo models/ms-marco-MiniLM-L-6-v2/model.onnx https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx
curl -Lo models/ms-marco-MiniLM-L-6-v2/tokenizer.json https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json
```

### 5. Start Elephant

```bash
cargo run --release
```

The server will print:

- REST API URL
- MCP URL

By default, the server also evaluates background observation consolidation after successful `retain` calls. This does not block the retain response. You can tune or disable that behavior in [`.env.example`](../.env.example).

You can then use the same `BANK_ID` example flow shown in the Docker quickstart.

## Next Steps

- Architecture overview: [architecture.md](architecture.md)
- Benchmark workflow: [../bench/locomo/README.md](../bench/locomo/README.md)
- Config reference: [../.env.example](../.env.example)
- Tests and deeper setup: [../tests/README.md](../tests/README.md)
