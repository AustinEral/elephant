# Running Tests

Copy `.env.example` to `.env` and fill in your keys.

## Unit + mock integration tests

Needs Docker running (testcontainers spins up Postgres).

```bash
cargo test
```

## Prompt eval tests

Fast iteration on prompts — no Docker, no embeddings. Only needs `LLM_PROVIDER`, `LLM_API_KEY`, and `LLM_MODEL`. Valid providers are `anthropic` and `openai`, where `openai` uses the Responses API.

Optional prompt-cache envs from `.env` are honored here too.

```bash
cargo test --test prompt_eval -- --ignored --nocapture
```

## Accuracy eval tests

File-driven internal accuracy cases live under [tests/evals](/home/austin/elephant/tests/evals).

Validate case files:

```bash
cargo test --test evals_validate
```

Run live extract checks:

```bash
source .env
cargo test --test evals_extract
```

## Real integration tests

Full retain→recall→reflect pipeline with real LLM and embedding APIs. Needs Docker.

Optional prompt-cache envs from `.env` are also honored by the LLM-backed tests.

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

**Cross-encoder reranker model** (required):

```bash
mkdir -p models/ms-marco-MiniLM-L-6-v2
curl -Lo models/ms-marco-MiniLM-L-6-v2/model.onnx https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx
curl -Lo models/ms-marco-MiniLM-L-6-v2/tokenizer.json https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json
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
