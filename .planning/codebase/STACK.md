# Technology Stack

**Analysis Date:** 2026-03-15

## Languages

**Primary:**
- Rust 2021 edition - Complete memory engine implementation, API server, benchmarking tools

## Runtime

**Environment:**
- Tokio 1.49.0 (async runtime)
- Linux container deployment (Debian trixie-slim)

**Package Manager:**
- Cargo (Rust package manager)
- Lockfile: `Cargo.lock` present (123,658 bytes)

## Frameworks

**Core:**
- Axum 0.8.8 - HTTP server framework, REST API routes and request handling
- Tower 0.5.3 - Middleware composition, CORS support via tower-http

**Async/Concurrency:**
- Async-trait 0.1.89 - Async trait implementations for all interfaces
- Futures 0.3 - Async combinators and utilities

**LLM/Prompting:**
- No LLM SDK dependency - Custom HTTP clients for Anthropic and OpenAI
- MCP (Model Context Protocol) via rmcp 0.16.0 - Server-side MCP adapter for tool exposure
- Schemars 1 - JSON schema generation from Rust types for MCP tool definitions

**Database:**
- SQLx 0.8.6 - PostgreSQL async driver with compile-time query verification
  - Features: runtime-tokio, tls-rustls, postgres, uuid, chrono, json, migrate
- pgvector 0.4.1 - PostgreSQL vector storage and similarity search
- Chrono 0.4.44 - Date/time handling with serde support

**ML/Embeddings:**
- ONNX Runtime (v1.23.0) via ort crate 2.0.0-rc.11 - Local model inference
  - Features: ndarray, load-dynamic
- Tokenizers 0.22.2 - HuggingFace tokenizers for embedding/reranking models
- ndarray 0.17.2 - Numerical array operations for embedding vectors

**Testing:**
- Testcontainers 0.27.1 - Docker containers for Postgres integration tests
- Testcontainers-modules 0.15.0 - PostgreSQL module with pgvector
- Proptest 1 - Property-based testing
- Datatest-stable 0.3.3 - Data-driven test harness for extraction eval cases

**Logging:**
- Tracing 0.1 - Structured logging framework
- Tracing-subscriber 0.3 - Logging output formatting (fmt, json, env-filter)

**Serialization:**
- Serde 1.0.228 - Serialization framework with derive macros
- Serde-json 1.0.149 - JSON serialization/deserialization
- ULID 1.2.1 - Sortable unique identifiers with serde support
- UUID 1.21.0 - UUID generation and parsing

**Error Handling:**
- Thiserror 2.0.18 - Error type derivation and conversion
- Anyhow 1.0.102 - Flexible error handling (tests only)

**HTTP:**
- Reqwest 0.13.2 - HTTP client for external APIs
  - Features: json, rustls
- TLS via rustls (not OpenSSL)

**Utilities:**
- Regex 1.12.3 - Regular expression support for text processing
- Tabled 0.20.0 - CLI table formatting with ANSI colors
- Dotenvy 0.15 - .env file loading
- Tower-http 0.6.8 - HTTP middleware utilities (CORS)

## Key Dependencies

**Critical:**
- reqwest 0.13.2 - HTTP client for API calls (Anthropic, OpenAI, Cohere)
- pgvector 0.4.1 - Vector similarity search (core to recall)
- ort 2.0.0-rc.11 - ONNX inference for embeddings and reranking
- rmcp 0.16.0 - MCP server protocol, required for tool exposure

**Infrastructure:**
- sqlx 0.8.6 - PostgreSQL driver with migrations support
- tokio 1.49.0 - Async runtime, full feature set
- axum 0.8.8 - REST API routes and middleware

## Configuration

**Environment:**
- Loaded from `.env` file via dotenvy
- LLM_PROVIDER: "anthropic" (default) or "openai"
- LLM_API_KEY: API credentials (required)
- RETAIN_LLM_MODEL / REFLECT_LLM_MODEL: Model overrides for extraction vs synthesis
- EMBEDDING_PROVIDER: "local" (default) or "openai"
- EMBEDDING_MODEL_PATH: Path to local ONNX model directory (required for local)
- RERANKER_PROVIDER: "local" (default), "api", or "none"
- DATABASE_URL: PostgreSQL connection string
- LISTEN_ADDR: HTTP server bind address (default: 0.0.0.0:3001)
- RUST_LOG: Tracing filter level (default: info)
- LOG_FORMAT: "json" for machine-readable logs, plain text otherwise

**Build:**
- `Cargo.toml` - Single crate, modular structure
- Multi-stage Docker build (builder → onnx-fetch → runtime)

## Platform Requirements

**Development:**
- Rust 1.70+ (2021 edition)
- pkg-config, libssl-dev, cmake, clang (for C++ dependencies)

**Production:**
- PostgreSQL 16 with pgvector extension
- ONNX Runtime shared library (libonnxruntime.so.1.23.0) bundled in Docker
- ONNX models (bge-small-en-v1.5, ms-marco-MiniLM-L-6-v2) bundled in Docker
- Memory: varies by model size and batch size
- Storage: PostgreSQL with pgdata volume for fact persistence

**Deployment Target:**
- Docker containers (Debian trixie-slim base)
- Can run with either local ONNX embeddings or external API embeddings
- Configurable via EMBED_MODE build arg (default: local)

---

*Stack analysis: 2026-03-15*
