ARG EMBED_MODE=local

# ---------------------------------------------------------------------------
# Stage 1: Build the Rust binary
# ---------------------------------------------------------------------------
FROM rust:1-trixie AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev cmake clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN cargo build --release

# ---------------------------------------------------------------------------
# Stage 2: Fetch ONNX Runtime + models
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS onnx-fetch

RUN apt-get update && apt-get install -y --no-install-recommends curl tar ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /onnx

# ONNX Runtime shared library
RUN curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-linux-x64-1.23.0.tgz \
    | tar xz --strip-components=1 -C /onnx

# bge-small-en-v1.5 embedding model
RUN mkdir -p /models/bge-small-en-v1.5 \
    && curl -fsSL -o /models/bge-small-en-v1.5/model.onnx \
       "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx" \
    && curl -fsSL -o /models/bge-small-en-v1.5/tokenizer.json \
       "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json"

# ms-marco-MiniLM-L-6-v2 cross-encoder reranker model
RUN mkdir -p /models/ms-marco-MiniLM-L-6-v2 \
    && curl -fsSL -o /models/ms-marco-MiniLM-L-6-v2/model.onnx \
       "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx" \
    && curl -fsSL -o /models/ms-marco-MiniLM-L-6-v2/tokenizer.json \
       "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json"

# ---------------------------------------------------------------------------
# Stage 3a: Runtime with local embeddings
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS runtime-local

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3t64 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/elephant .
COPY --from=onnx-fetch /onnx/lib/libonnxruntime.so* /usr/local/lib/
COPY --from=onnx-fetch /models /app/models

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV EMBEDDING_PROVIDER=local
ENV EMBEDDING_MODEL_PATH=/app/models/bge-small-en-v1.5
ENV RERANKER_PROVIDER=local
ENV RERANKER_MODEL_PATH=/app/models/ms-marco-MiniLM-L-6-v2

EXPOSE 3001
CMD ["./elephant"]

# ---------------------------------------------------------------------------
# Stage 3b: Runtime with external embeddings (still needs ONNX for reranker)
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS runtime-external

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3t64 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/elephant .
COPY --from=onnx-fetch /onnx/lib/libonnxruntime.so* /usr/local/lib/
COPY --from=onnx-fetch /models/ms-marco-MiniLM-L-6-v2 /app/models/ms-marco-MiniLM-L-6-v2

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV RERANKER_PROVIDER=local
ENV RERANKER_MODEL_PATH=/app/models/ms-marco-MiniLM-L-6-v2

EXPOSE 3001
CMD ["./elephant"]

# ---------------------------------------------------------------------------
# Final: Select runtime based on EMBED_MODE
# ---------------------------------------------------------------------------
FROM runtime-${EMBED_MODE} AS runtime
