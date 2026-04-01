# Runtime Configuration

This document is the full runtime env reference for running the Elephant server.

Root [`.env.example`](/home/austin/elephant/.env.example) is intentionally short and onboarding-sized. It shows the common startup path. This file covers the rest of the runtime surface.

Benchmark configuration is separate. Benchmarks do not use the root runtime `.env`; see [bench/README.md](/home/austin/elephant/bench/README.md).

## Server

- `DATABASE_URL`
  Postgres connection string for Elephant storage.
- `LISTEN_ADDR`
  HTTP and MCP bind address.
- `SERVER_AUTO_CONSOLIDATION`
  Enable or disable background consolidation after successful `retain`.
- `SERVER_AUTO_CONSOLIDATION_MIN_FACTS`
  Minimum unconsolidated fact count before scheduling consolidation.
- `SERVER_AUTO_CONSOLIDATION_COOLDOWN_SECS`
  Per-bank cooldown between background consolidation attempts.
- `SERVER_AUTO_CONSOLIDATION_MERGE_OPINIONS`
  Run opinion merge after automatic consolidation.

## LLM Provider

- `LLM_PROVIDER`
  One of `anthropic`, `openai`, `gemini`, or `vertex`.
- `LLM_API_KEY`
  Provider API key for the selected LLM backend.
- `LLM_MODEL`
  Default model for all runtime stages unless split overrides are set.
- `RETAIN_LLM_MODEL`
  Extraction / retain model override.
- `REFLECT_LLM_MODEL`
  Reflect model override.
- `LLM_BASE_URL`
  Optional base URL for OpenAI-compatible endpoints or advanced provider routing.
- `LLM_VERTEX_PROJECT`
  Required when `LLM_PROVIDER=vertex`.
- `LLM_VERTEX_LOCATION`
  Optional Vertex location override.
- `LLM_TIMEOUT_SECS`
  Request timeout for runtime LLM calls.

## Embeddings

- `EMBEDDING_PROVIDER`
  `local` or `openai`.
- `EMBEDDING_MODEL_PATH`
  Local ONNX embedding model directory when `EMBEDDING_PROVIDER=local`.
- `ORT_DYLIB_PATH`
  ONNX Runtime shared library path for local embeddings/reranking.
- `EMBEDDING_API_KEY`
  OpenAI embedding API key when `EMBEDDING_PROVIDER=openai`.
- `EMBEDDING_API_MODEL`
  OpenAI embedding model name.
- `EMBEDDING_API_DIMS`
  Expected embedding dimensionality for API embeddings.

## Reranker

- `RERANKER_PROVIDER`
  `local`, `api`, or `none`.
- `RERANKER_MODEL_PATH`
  Local ONNX reranker model directory when `RERANKER_PROVIDER=local`.
- `RERANKER_API_KEY`
  API key when `RERANKER_PROVIDER=api`.
- `RERANKER_API_URL`
  Cohere-compatible reranker endpoint.
- `RERANKER_API_MODEL`
  API reranker model name.

## Retrieval And Runtime Tuning

- `DEDUP_THRESHOLD`
  Fact insertion dedup threshold. Use `none` to disable.
- `REFLECT_ENABLE_SOURCE_LOOKUP`
  Enable or disable reflect source lookup.
- `REFLECT_MAX_ITERATIONS`
  Maximum reflect tool loop iterations.
- `REFLECT_BUDGET_TOKENS`
  Reflect budget override.
- `REFLECT_SOURCE_LIMIT`
  Maximum number of retrieved facts whose sources may be expanded.
- `REFLECT_SOURCE_MAX_CHARS`
  Maximum characters pulled per source expansion.
- `RETRIEVER_LIMIT`
  Per-channel retrieval limit.
- `MAX_FACTS`
  Final maximum facts returned to reflect.
- `GRAPH_SEMANTIC_THRESHOLD`
  Semantic threshold for graph linking.
- `GRAPH_TEMPORAL_MAX_DAYS`
  Temporal window for graph linking.
- `GRAPH_ENABLE_CAUSAL`
  Enable or disable causal graph edges.
- `CONSOLIDATION_BATCH_SIZE`
  Observation consolidation batch size.
- `CONSOLIDATION_MAX_TOKENS`
  Consolidation token limit.
- `CONSOLIDATION_RECALL_BUDGET`
  Recall budget for consolidation context.
- `EXTRACTION_STRUCTURED_OUTPUT_MAX_ATTEMPTS`
  Maximum structured-output retries for extraction.
- `CONSOLIDATION_STRUCTURED_OUTPUT_MAX_ATTEMPTS`
  Maximum structured-output retries for consolidation.

## Reasoning Effort

Supported values: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`.

- `RETAIN_EXTRACT_REASONING_EFFORT`
- `RETAIN_RESOLVE_REASONING_EFFORT`
- `RETAIN_GRAPH_REASONING_EFFORT`
- `REFLECT_REASONING_EFFORT`
- `CONSOLIDATE_REASONING_EFFORT`
- `OPINION_MERGE_REASONING_EFFORT`

## Temperature Overrides

These are explicit request overrides. If unset, Elephant omits the temperature field and lets the provider/model choose its default.

- `RETAIN_EXTRACT_TEMPERATURE`
- `RETAIN_RESOLVE_TEMPERATURE`
- `RETAIN_GRAPH_TEMPERATURE`
- `REFLECT_TEMPERATURE`
- `CONSOLIDATE_TEMPERATURE`
- `OPINION_MERGE_TEMPERATURE`

Unsupported explicit temperature overrides fail fast during runtime construction.

## Prompt Caching

- `LLM_PROMPT_CACHE_ENABLED`
  Enable prompt caching when supported.
- `OPENAI_PROMPT_CACHE_KEY`
  Optional OpenAI prompt cache namespace key.
- `OPENAI_PROMPT_CACHE_RETENTION`
  OpenAI prompt cache retention, `in_memory` or `24h`.
- `ANTHROPIC_PROMPT_CACHE_TTL`
  Anthropic prompt cache TTL, such as `5m` or `1h`.

Gemini and Vertex use provider-native behavior when cache usage is reported; no extra runtime env is required there.

## Logging

- `RUST_LOG`
  Standard tracing filter.
- `LOG_FORMAT`
  `text` or `json`.
