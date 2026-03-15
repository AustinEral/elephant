# Codebase Structure

**Analysis Date:** 2025-03-15

## Directory Layout

```
elephant/
├── src/
│   ├── main.rs                     # HTTP server entry point
│   ├── lib.rs                      # Library root, pub module declarations
│   ├── runtime.rs                  # Factory for building all pipelines from env
│   ├── error.rs                    # Error enum and Result type
│   ├── util.rs                     # Helper functions (cosine_similarity, etc.)
│   ├── metrics.rs                  # Metrics collection and reporting
│   │
│   ├── server/                     # HTTP API server
│   │   ├── mod.rs                  # Axum router, AppState struct
│   │   ├── handlers.rs             # GET/POST endpoint handlers
│   │   └── error.rs                # HTTP error response types
│   │
│   ├── mcp/                        # MCP server adapter
│   │   └── mod.rs                  # Tool definitions, MCP handler
│   │
│   ├── types/                      # Core type definitions
│   │   ├── mod.rs                  # Exports from all submodules
│   │   ├── id.rs                   # BankId, FactId, EntityId, SourceId, TurnId (ULID-based)
│   │   ├── bank.rs                 # MemoryBank, Disposition
│   │   ├── entity.rs               # Entity, EntityType
│   │   ├── fact.rs                 # Fact, FactType, ScoredFact, RetrievalSource
│   │   ├── network.rs              # NetworkType (World, Experience, Observation, Opinion)
│   │   ├── temporal.rs             # TemporalRange for date annotations
│   │   ├── graph.rs                # GraphLink, LinkType
│   │   ├── source.rs               # Source, FactSourceLookup (provenance tracking)
│   │   ├── llm.rs                  # CompletionRequest, Message, ToolDef, ToolCall
│   │   └── pipeline.rs             # Retain/Recall/Reflect input/output types
│   │
│   ├── storage/                    # Persistence layer
│   │   ├── mod.rs                  # MemoryStore and TransactionHandle traits
│   │   ├── pg.rs                   # PostgreSQL implementation with pgvector
│   │   └── mock.rs                 # In-memory test double
│   │
│   ├── retain/                     # Fact extraction pipeline (2A-2E)
│   │   ├── mod.rs                  # RetainPipeline trait, DefaultRetainPipeline orchestrator
│   │   ├── chunker.rs              # Text chunking (step 2A)
│   │   ├── extractor.rs            # LLM fact extraction (step 2B)
│   │   ├── resolver.rs             # Entity resolution (step 2C)
│   │   └── graph_builder.rs        # Graph link construction (step 2D)
│   │
│   ├── recall/                     # Retrieval pipeline (TEMPR)
│   │   ├── mod.rs                  # RecallPipeline trait, DefaultRecallPipeline orchestrator
│   │   ├── semantic.rs             # Vector similarity retriever
│   │   ├── keyword.rs              # BM25 full-text retriever
│   │   ├── graph.rs                # Graph spreading activation retriever
│   │   ├── temporal.rs             # Temporal/recency retriever
│   │   ├── fusion.rs               # RRF fusion algorithm
│   │   ├── budget.rs               # Token counting and budget enforcement
│   │   └── reranker/
│   │       ├── mod.rs              # Reranker trait and provider factory
│   │       ├── local.rs            # ONNX MiniLM cross-encoder
│   │       └── api.rs              # Cohere-compatible API reranker
│   │
│   ├── reflect/                    # Reasoning pipeline (CARA)
│   │   ├── mod.rs                  # ReflectPipeline trait, DefaultReflectPipeline orchestrator, tool loop
│   │   ├── disposition.rs          # Bank personality verbalization
│   │   ├── hierarchy.rs            # (Deprecated) observation hierarchy
│   │   └── opinion.rs              # Opinion type and handling
│   │
│   ├── consolidation/              # Background synthesis workers
│   │   ├── mod.rs                  # cluster_by_similarity utility
│   │   ├── observation.rs          # Consolidator trait, topic-scoped observation synthesis
│   │   └── opinion_merger.rs       # OpinionMerger trait, opinion deduplication
│   │
│   ├── llm/                        # LLM client abstraction
│   │   ├── mod.rs                  # LlmClient trait, complete_structured helper, JSON extraction
│   │   ├── anthropic.rs            # Anthropic API implementation with tool calling
│   │   ├── openai.rs               # OpenAI API implementation
│   │   ├── retry.rs                # Retry wrapper (exponential backoff, rate limit handling)
│   │   └── mock.rs                 # Test double with configurable responses
│   │
│   ├── embedding/                  # Embedding client abstraction
│   │   ├── mod.rs                  # EmbeddingClient trait, build_client factory
│   │   ├── local.rs                # ONNX BGE-small model (384 dims)
│   │   ├── openai.rs               # OpenAI API implementation
│   │   └── mock.rs                 # Test double with fixed-size vectors
│   │
│   └── bin/
│       └── inspect_retain_source.rs # Utility: inspect source records
│
├── bench/                          # Benchmarking and evaluation
│   ├── locomo/
│   │   ├── locomo.rs               # LoCoMo benchmark harness (binary: locomo-bench)
│   │   └── results/                # Result JSON files tagged by experiment
│   ├── view.rs                     # Result viewing tool (binary: view)
│   └── reset_consolidation.rs      # Utility to clear consolidated_at flags
│
├── tests/                          # Integration and evaluation tests
│   ├── api_integration_tests.rs    # HTTP API roundtrip tests
│   ├── evals_extract.rs            # Fact extraction quality evaluation
│   ├── evals_validate.rs           # Semantic validation of extracted facts
│   ├── prompt_eval.rs              # Prompt template evaluation
│   ├── real_integration_tests.rs   # Full pipeline tests with real LLM (ignored)
│   └── storage_tests.rs            # Storage layer tests
│
├── prompts/                        # Prompt templates (plain text, included via include_str!)
│   ├── extract_facts.txt           # Fact extraction prompt
│   ├── consolidate_topics.txt      # Observation consolidation prompt
│   ├── reflect_agent.txt           # Reflect agent system prompt with tool decomposition
│   ├── synthesize_observation.txt  # Observation synthesis detail prompt
│   ├── merge_opinions.txt          # Opinion merging logic
│   ├── reinforce_opinion.txt       # Opinion reinforcement assessment prompt
│   └── reflect.txt                 # (Unused) legacy reflect prompt
│
├── docs/                           # Documentation
│   ├── architecture.md             # Mermaid diagram and pipeline overview
│   └── audit-checklist.md          # Quality assurance checklist
│
├── Cargo.toml                      # Rust package manifest
├── Cargo.lock                      # Dependency lockfile
├── CLAUDE.md                       # Project instructions for Claude
└── .env (not committed)            # Runtime configuration (secrets, keys, URLs)
```

## Directory Purposes

**src/**: All Rust source code organized by architectural layer
- **main.rs**: Starts HTTP server; loads env, builds runtime, binds listener
- **lib.rs**: Declares all public modules; entry point for library consumers
- **runtime.rs**: Factory function that wires all components from environment variables
- **error.rs**: Centralized error handling with Result<T> alias
- **util.rs**: Shared utilities (vector ops, ID generation helpers)
- **metrics.rs**: Metrics collection wrapper around LLM client

**server/**: HTTP API implementation
- **mod.rs**: Defines AppState (holds all pipeline instances), builds Axum router with routes
- **handlers.rs**: Endpoint functions that extract path/body, call pipelines, return JSON
- **error.rs**: HTTP error response types and status code mapping

**mcp/**: Model Context Protocol server
- **mod.rs**: Defines MCP tool parameter types, tool schemas, and handler callbacks

**types/**: Core domain types
- **id.rs**: ULID-based identifiers with string parsing
- **bank.rs**: Memory bank config (name, mission, directives, personality)
- **entity.rs**: Entity representation (canonical name, aliases, type)
- **fact.rs**: Fact atomic unit with network classification, embedding, temporal range
- **network.rs**: Four-network memory classification enum
- **temporal.rs**: Date range for fact temporal anchoring
- **graph.rs**: Graph link representation with link types
- **llm.rs**: LLM request/response/tool types
- **pipeline.rs**: Input/output contracts for all three pipelines

**storage/**: Persistence abstraction and implementations
- **mod.rs**: MemoryStore trait defining 20+ data access methods
- **pg.rs**: PostgreSQL implementation using sqlx with pgvector extension
- **mock.rs**: In-memory HashMap-based test double

**retain/**: Fact extraction pipeline
- **mod.rs**: RetainPipeline trait and DefaultRetainPipeline orchestrator (wires all subcomponents)
- **chunker.rs**: Chunker trait and SimpleChunker (token-based splitting)
- **extractor.rs**: FactExtractor trait and LlmFactExtractor (LLM-based with tool calling)
- **resolver.rs**: EntityResolver trait and LayeredEntityResolver (fuzzy matching + creation)
- **graph_builder.rs**: GraphBuilder trait and DefaultGraphBuilder (semantic/temporal/causal links)

**recall/**: Retrieval pipeline
- **mod.rs**: RecallPipeline trait and DefaultRecallPipeline orchestrator
- **semantic.rs**: Retriever impl via pgvector cosine distance
- **keyword.rs**: Retriever impl via PostgreSQL tsvector BM25
- **graph.rs**: Retriever impl via spreading activation over entity-fact edges
- **temporal.rs**: Retriever impl via date-range matching and recency
- **fusion.rs**: RRF fusion algorithm combining four ranked lists
- **budget.rs**: Tokenizer trait and token budget enforcement
- **reranker/**: Reranker implementations (local ONNX, API)

**reflect/**: Reasoning pipeline
- **mod.rs**: ReflectPipeline trait, DefaultReflectPipeline, agentic tool loop
- **disposition.rs**: Convert bank personality to verbalized context
- **hierarchy.rs**: (Deprecated) observation aggregation
- **opinion.rs**: Opinion formation (now read-only in reflect)

**consolidation/**: Background synthesis
- **mod.rs**: Similarity clustering utility
- **observation.rs**: Consolidator trait, topic-scoped observation synthesis
- **opinion_merger.rs**: OpinionMerger trait, opinion deduplication and reconciliation

**llm/**: LLM client abstraction
- **mod.rs**: LlmClient trait, complete_structured helper, JSON extraction from markdown
- **anthropic.rs**: Anthropic client with tool-calling support
- **openai.rs**: OpenAI client with tool-calling support
- **retry.rs**: Retry wrapper with exponential backoff
- **mock.rs**: Test double for unit tests

**embedding/**: Embedding client abstraction
- **mod.rs**: EmbeddingClient trait, build_client factory
- **local.rs**: ONNX runtime wrapper for BGE-small (384 dims)
- **openai.rs**: OpenAI API client
- **mock.rs**: Test double with fixed vectors

**bench/**: Benchmarking harness and tools
- **locomo/locomo.rs**: LoCoMo benchmark runner (load conversations, run retain→consolidate→reflect, collect metrics)
- **view.rs**: Results viewing tool (parse JSON, display tables, aggregate by type)
- **reset_consolidation.rs**: Admin utility to clear consolidated_at flags

**tests/**: Integration and evaluation tests
- **api_integration_tests.rs**: Full HTTP roundtrip tests (bank creation, retain, recall, reflect)
- **evals_extract.rs**: Fact extraction quality checks (datatest)
- **evals_validate.rs**: Semantic validation (datatest)
- **prompt_eval.rs**: Prompt effectiveness evaluation
- **real_integration_tests.rs**: Tests calling real LLM APIs (marked #[ignore])
- **storage_tests.rs**: Database layer tests

**prompts/**: Prompt template files (plain text, included at compile time)
- **extract_facts.txt**: Instruct LLM to extract facts with all details, relationships, temporal annotations
- **consolidate_topics.txt**: Instruct LLM to synthesize observations, decide CREATE vs UPDATE
- **reflect_agent.txt**: Agent system prompt with task decomposition, output format
- **synthesize_observation.txt**: Detail instructions for merging facts (counts, specific wording)
- **merge_opinions.txt**: Logic for deduplicating opinions
- **reinforce_opinion.txt**: Classification of new facts vs existing opinions (supports/contradicts/unrelated)

**docs/**: Reference documentation
- **architecture.md**: Mermaid pipeline diagram, network descriptions, retrieval/reasoning explanation
- **audit-checklist.md**: Quality checklist for deployment

## Key File Locations

**Entry Points:**
- `src/main.rs`: HTTP server startup; binds to LISTEN_ADDR (default 0.0.0.0:3001), mounts /v1 REST API and /mcp service
- `bench/locomo/locomo.rs`: Binary entry for benchmarking; loads LoCoMo JSON, runs benchmarks, stores results in `bench/locomo/results/`
- `bench/view.rs`: Binary entry for viewing results; parses JSON files, aggregates metrics

**Configuration:**
- `.env` (not in git): Runtime secrets and env vars — LLM API keys, database URL, reranker settings
- `Cargo.toml`: Dependency list, binary targets
- `src/runtime.rs`: Environment variable reading and component factory

**Core Logic:**
- `src/retain/mod.rs`: Fact extraction pipeline orchestrator (chunking → extraction → resolution → storage → graph → opinion reinforcement)
- `src/recall/mod.rs`: Retrieval pipeline orchestrator (four parallel retrievers → RRF fusion → reranking → token budget)
- `src/reflect/mod.rs`: Reasoning pipeline orchestrator (context assembly → agentic tool loop → response synthesis)
- `src/consolidation/observation.rs`: Observation synthesis (batching facts → LLM consolidation → storage)
- `src/storage/pg.rs`: PostgreSQL schema and queries (facts, entities, banks, graph links, sources)

**Testing:**
- `tests/api_integration_tests.rs`: Full pipeline HTTP tests
- `tests/evals_extract.rs`: Fact extraction quality evaluation
- `tests/real_integration_tests.rs`: Real LLM integration (marked #[ignore])

## Naming Conventions

**Files:**
- Module files match directory names: `src/retain/mod.rs` for the retain module
- Trait implementations: `DefaultRetainPipeline` in `src/retain/mod.rs`, `LlmFactExtractor` in `src/retain/extractor.rs`
- Test files: Append `_tests.rs` (e.g., `api_integration_tests.rs`)
- Utility binaries: Lowercase with underscores (e.g., `locomo.rs`, `view.rs`)

**Directories:**
- Layer modules use lowercase (retain, recall, reflect, storage, llm, embedding)
- Types module organizes domain entities (bank, entity, fact, network, temporal, graph)
- Submodules for provider variations: `reranker/local.rs`, `reranker/api.rs`
- Bench results tagged by experiment name: `bench/locomo/results/{tag}.json`

**Types:**
- ULIDs for IDs: BankId, FactId, EntityId, SourceId, TurnId
- Enums for classification: NetworkType, FactType, LinkType, EntityType
- Input/output pairs: RetainInput/RetainOutput, RecallQuery/RecallResult, ReflectQuery/ReflectResult
- Traits with Default impl: RetainPipeline, RecallPipeline, ReflectPipeline, Consolidator
- Configuration structs: ChunkConfig, DispositionInput, RerankerConfig

**Functions:**
- Pipeline orchestrators: `async fn retain(&self, input: &RetainInput) -> Result<RetainOutput>`
- Trait methods are trait-named: `complete()` for LlmClient, `embed()` for EmbeddingClient, `retrieve()` for Retriever
- Handler functions: kebab-case route → snake_case handler (e.g., `/v1/banks/{id}/retain` → `async fn retain()`)
- Helpers: verb-based (parse_bank_id, create_source_record, dedup_facts)

## Where to Add New Code

**New Feature (e.g., a new retrieval strategy):**
- Primary implementation: `src/recall/` — create `new_strategy.rs` with a struct implementing `Retriever` trait
- Integration: Wire into `DefaultRecallPipeline` in `src/recall/mod.rs` by adding a component field and calling it in the parallel join!
- Tests: Add tests in the `new_strategy.rs` module under `#[cfg(test)]`

**New LLM Provider (e.g., Cohere for completion):**
- Implementation: Create `src/llm/cohere.rs` with struct implementing `LlmClient` trait
- Build factory: Add provider variant to runtime.rs build_runtime_from_env()
- Configuration: Document env vars for provider selection and credentials

**New Network Type (e.g., if adding a fifth network):**
- Add variant to `src/types/network.rs::NetworkType` enum
- Update retrieval filtering: Add handling in `src/storage/pg.rs` and retriever network filters
- Update consolidation: Consider if the new network is subject to consolidation
- Update serialization: NetworkType serde impl automatically handles kebab-case naming

**New Consolidation Worker (e.g., summary generation):**
- Trait: Define in `src/consolidation/mod.rs` or new `consolidation/{worker_name}.rs`
- Implementation: Create struct with store/llm/embeddings dependencies
- Endpoint: Add `/v1/banks/{id}/{worker_name}` handler in `src/server/handlers.rs`
- Factory: Wire into AppState in `src/runtime.rs`

**Utilities:**
- Shared helpers: `src/util.rs` — vector operations, ID generation, string manipulation
- Per-module helpers: Internal functions in each module's impl block
- Math/algorithms: Keep near the user (e.g., cluster_by_similarity in `src/consolidation/mod.rs`)

**Test Infrastructure:**
- Mock clients: `src/{module}/mock.rs` — in-memory doubles
- Test fixtures: In-module `#[cfg(test)]` blocks with example data
- Integration tests: `tests/{feature}_tests.rs` with full roundtrips
- Evaluation tests: datatest harnesses in `tests/evals_*.rs`

## Special Directories

**target/**: Generated binaries and build artifacts
- Generated at compile time, ignored by git
- Contains debug/release builds of main binary and binaries

**bench/locomo/results/**: Benchmark result files
- Generated by `locomo-bench` runs
- Format: JSON with field `results: Vec<ConversationResult>` containing question responses, retrieved context, metrics
- Not committed; use unique tags to avoid overwriting (`--tag` flag)
- Consumed by `view` binary for analysis

**.planning/codebase/**: GSD codebase analysis documents
- Generated by `/gsd:map-codebase` command
- Documents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md, STACK.md, INTEGRATIONS.md
- Read by `/gsd:plan-phase` and `/gsd:execute-phase` for implementation guidance

---

*Structure analysis: 2025-03-15*
