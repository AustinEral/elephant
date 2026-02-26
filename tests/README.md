# Running Tests

Copy `.env.test.example` to `.env` and fill in your keys.

## Unit + mock integration tests

Needs Docker running (testcontainers spins up Postgres).

```bash
cargo test
```

## Prompt eval tests

Fast iteration on prompts — no Docker, no embeddings. Only needs `LLM_API_KEY` + `LLM_MODEL`.

```bash
cargo test --test prompt_eval -- --ignored --nocapture
```

## Real integration tests

Full retain→recall→reflect pipeline with real LLM and embedding APIs. Needs Docker.

### Setup

**ONNX Runtime** (needed for local embedding tests):

```bash
mkdir -p lib
curl -sL https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-linux-x64-1.23.0.tgz | tar xz -C lib/
```

**Local embedding model** (needed for local embedding tests):

```bash
mkdir -p models/bge-small-en-v1.5
curl -Lo models/bge-small-en-v1.5/model.onnx https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx
curl -Lo models/bge-small-en-v1.5/tokenizer.json https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json
```

### Run

```bash
# Local embeddings
cargo test --test real_integration_tests local -- --ignored --nocapture

# OpenAI embeddings
cargo test --test real_integration_tests openai -- --ignored --nocapture

# All
cargo test --test real_integration_tests -- --ignored --nocapture
```
