# Architecture

**Analysis Date:** 2025-03-15

## Pattern Overview

**Overall:** Hindsight memory engine with layered pipeline architecture — three parallel, independently testable operation pipelines (Retain/Recall/Reflect) backed by a single PostgreSQL store with shared trait abstractions.

**Key Characteristics:**
- **Trait-driven design**: All persistence, embedding, LLM, and retrieval components are trait interfaces, enabling mock testing and provider swapping
- **Modular pipelines**: Retain, Recall, and Reflect are isolated, composable operations with clear input/output contracts
- **Network classification**: Facts belong to one of four memory networks (World, Experience, Observation, Opinion) with per-network retrieval filtering
- **Temporal annotations**: Facts carry optional temporal ranges; consolidation and retrieval index by date
- **Tool-calling agent**: Reflect pipeline is agentic with bounded iteration and read-only tools (search_observations, recall, done)

## Layers

**HTTP API Server:**
- Purpose: Expose REST endpoints for bank/fact/entity management and pipeline operations
- Location: `src/server/mod.rs`, `src/server/handlers.rs`
- Contains: Axum router, request/response marshalling, HTTP status codes
- Depends on: All pipelines and AppState
- Used by: External clients and MCP bridge

**MCP Adapter:**
- Purpose: Expose memory engine operations as Model Context Protocol (MCP) tools
- Location: `src/mcp/mod.rs`
- Contains: Tool parameter types, tool definitions, MCP server handler
- Depends on: AppState, Retain/Recall/Reflect pipelines
- Used by: MCP clients (Claude, other LLM applications)

**Retain Pipeline (2A-2E):**
- Purpose: Transform raw text into structured facts with entity resolution and opinion reinforcement
- Location: `src/retain/mod.rs` (orchestrator)
- Subcomponents:
  - `chunker.rs` (2A): Split input into overlapping chunks, preserve conversation turns
  - `extractor.rs` (2B): LLM-based fact extraction with structured JSON output
  - `resolver.rs` (2C): Entity resolution, deduplication, and entity creation
  - `graph_builder.rs` (2D): Create semantic/temporal/causal links between facts
- Pipeline flow: Chunk → Extract → Resolve Entities → Embed → Dedup → Store → Build Graph → Reinforce Opinions
- Depends on: LLM client, Embedding client, Storage, Entity resolver
- Used by: Server handlers, MCP tools

**Recall Pipeline (TEMPR Retrieval):**
- Purpose: Execute four parallel retrievers, fuse results, rerank, and enforce token budget
- Location: `src/recall/mod.rs` (orchestrator)
- Subcomponents:
  - `semantic.rs`: Vector similarity search via pgvector cosine distance
  - `keyword.rs`: BM25 full-text search (PostgreSQL tsvector)
  - `graph.rs`: Spreading activation over entity-fact-entity graph
  - `temporal.rs`: Date-range matching and recency boosting
  - `fusion.rs`: Reciprocal Rank Fusion (RRF) with k parameter
  - `reranker/mod.rs`: Cross-encoder reranker (local ONNX or API)
  - `budget.rs`: Token counting and fact trimming to stay within budget
- Pipeline flow: [Parallel: Semantic, Keyword, Graph, Temporal] → RRF Fusion → Rerank Top-N → Token Budget → Return
- Depends on: Storage (vector/keyword search), Reranker, Tokenizer
- Used by: Reflect pipeline, Server handlers, MCP tools

**Reflect Pipeline (CARA Reasoning):**
- Purpose: Agentic reasoning loop that synthesizes answers from retrieved facts and opinions
- Location: `src/reflect/mod.rs` (orchestrator)
- Subcomponents:
  - `disposition.rs`: Bank personality (skepticism, literalism, empathy) verbalized into prompts
  - `hierarchy.rs`: (Deprecated) was for tiered observations
  - `opinion.rs`: Opinion formation during reflection (now read-only)
- Agent tools:
  - `search_observations`: Query observation network only
  - `recall`: Full recall pipeline with World/Experience/Opinion networks
  - `done`: Finalize response (forced on iteration 8)
- Agent flow: Query → Assemble Context → Tool Loop (iter 0→search, 1→recall, 2..7→auto, 8→done) → Return
- Depends on: Recall pipeline, LLM client (with tool-calling support), Storage
- Used by: Server handlers, MCP tools

**Consolidation Workers:**
- Purpose: Background synthesis that produces higher-level memory structures
- Location: `src/consolidation/observation.rs`, `src/consolidation/opinion_merger.rs`
- Observation consolidator:
  - Groups unconsolidated facts by creation order in topic-scoped batches
  - LLM decides whether to CREATE new observation or UPDATE existing one
  - Marks source facts as consolidated_at
- Opinion merger:
  - Clusters similar opinions by embedding cosine similarity
  - Classifies as consistent/contradictory/superseded
  - Updates confidence or merges duplicates
- Depends on: LLM client, Storage, Recall pipeline (for context), Embedding client
- Used by: Server handlers, Background jobs

**Storage Layer:**
- Purpose: Persist facts, entities, banks, sources, and graph links
- Location: `src/storage/mod.rs` (trait), `src/storage/pg.rs` (PostgreSQL implementation)
- Provides: MemoryStore trait with methods for insert/query/update/search
- Key features:
  - Transactions (TransactionHandle) for atomic multi-write operations
  - Vector search with network filtering
  - Keyword search with BM25 scoring
  - Graph link management and neighbor traversal
  - Entity resolution (find_entity, upsert_entity)
  - Source/provenance tracking
- Depends on: PostgreSQL with pgvector extension
- Used by: All pipelines

**Type System:**
- Purpose: Define core entities and contracts
- Location: `src/types/` (9 submodules)
  - `id.rs`: ULID-based identifiers (BankId, FactId, EntityId, SourceId, TurnId)
  - `network.rs`: NetworkType enum (World, Experience, Observation, Opinion)
  - `fact.rs`: Fact (atomic memory unit), ScoredFact, RetrievalSource
  - `entity.rs`: Entity, EntityType (Person, Place, Concept, etc.)
  - `bank.rs`: MemoryBank, Disposition (personality config)
  - `temporal.rs`: TemporalRange for date annotations
  - `graph.rs`: GraphLink, LinkType (semantic, temporal, causal)
  - `pipeline.rs`: Input/output contracts (RetainInput, RecallQuery, etc.)
  - `llm.rs`: CompletionRequest, Message, ToolDef (for tool-calling)
- Used by: All modules

**LLM Abstraction:**
- Purpose: Provide pluggable LLM implementations with retry/metrics
- Location: `src/llm/mod.rs` (trait), `src/llm/anthropic.rs`, `src/llm/openai.rs`, `src/llm/retry.rs`
- Features:
  - LlmClient trait for text completion and JSON parsing
  - Tool-calling support (ToolDef, ToolCall, ToolResult)
  - Automatic JSON extraction from markdown fences
  - Retry logic (exponential backoff, rate limit handling)
  - Metrics collection (completion tokens, latency)
- Depends on: HTTP client (reqwest), serde for JSON
- Used by: Retain (extraction), Consolidation, Reflect

**Embedding Abstraction:**
- Purpose: Provide pluggable embedding implementations
- Location: `src/embedding/mod.rs` (trait), `src/embedding/local.rs`, `src/embedding/openai.rs`
- Features:
  - EmbeddingClient trait for batch embedding
  - Local ONNX model (bge-small-en-v1.5, 384 dims)
  - OpenAI API provider
  - Batch embedding with dimension validation
- Depends on: ONNX runtime (for local), HTTP client (for API)
- Used by: Retain (embedding facts), Consolidation (clustering), Reranker input prep

**Runtime Factory:**
- Purpose: Build and configure all pipelines from environment variables
- Location: `src/runtime.rs`
- Responsibilities:
  - Instantiate LLM client (Anthropic/OpenAI with retry wrapper)
  - Instantiate Embedding client (local/OpenAI)
  - Instantiate Reranker (local ONNX/API)
  - Wire all four recall retrievers and fusion
  - Create Retain orchestrator with all subcomponents
  - Create Reflect agent with tool definitions
  - Create Consolidation workers
  - Capture runtime tuning snapshot and prompt hashes for publication
- Used by: main.rs, bench harness

## Data Flow

**Retain Flow:**

1. User calls `/v1/banks/{id}/retain` with RetainInput (content, bank_id, timestamp, context, speaker)
2. DefaultRetainPipeline.retain_inner():
   - Chunk: Split content into overlapping chunks
   - For each chunk:
     - Extract: LLM parses into ExtractedFact list with entity_mentions
     - Resolve: Match mentions to existing entities or create new ones
     - Embed: Get vector embeddings for each fact
     - Dedup: Filter facts with cosine similarity above threshold
     - Store: Insert facts, create source record, link to source
     - Graph: Build semantic/temporal/causal links
     - Reinforce: Check against opinions, update confidence if similar
3. Return RetainOutput with fact_ids, entities_resolved, links_created, opinions_(reinforced|weakened)

**Recall Flow:**

1. User calls `/v1/banks/{id}/recall` with RecallQuery (query, bank_id, budget_tokens, network_filter)
2. DefaultRecallPipeline.recall():
   - Parallel retrieve:
     - SemanticRetriever: pgvector cosine search
     - KeywordRetriever: PostgreSQL BM25 search
     - GraphRetriever: Spreading activation over graph
     - TemporalRetriever: Date-range matching + recency boost
   - Fusion: Apply RRF (k=60) to merge four ranked lists
   - Rerank: Cross-encoder scores top-N facts (default N=50)
   - Budget: Token count facts in rank order, keep until budget exhausted
3. Return RecallResult with ScoredFact[] and total_tokens

**Reflect Flow:**

1. User calls `/v1/banks/{id}/reflect` with ReflectQuery (question, bank_id, budget_tokens)
2. DefaultReflectPipeline.reflect():
   - Assemble context: Fetch bank disposition, directives
   - Initialize tool loop with max 8 iterations:
     - Iter 0: Force search_observations tool
     - Iter 1: Force recall tool
     - Iter 2..7: LLM auto-selects tool
     - Iter 8: Force done tool only
   - For each tool call:
     - search_observations: Query only Observation network
     - recall: Full recall with World/Experience/Opinion networks
     - done: LLM provides final response text
   - Store retrieved context (fact IDs, content, scores, networks)
   - Return ReflectResult with response, sources, new_opinions, confidence, trace

**Consolidation Flow:**

1. User calls `/v1/banks/{id}/consolidate` with empty body
2. DefaultConsolidator.consolidate_with_progress():
   - Fetch all unconsolidated facts (consolidated_at IS NULL) from World+Experience networks
   - Batch by creation_at in groups of 8 (configurable)
   - For each batch:
     - Recall: Search observations for entity context (512 token budget)
     - LLM consolidate: Decide CREATE new observation vs UPDATE existing
     - Store: Insert/update observations, mark source facts consolidated_at
     - Emit progress event
3. Return ConsolidationReport with facts_consolidated, observations_created/updated

## Key Abstractions

**MemoryStore (Storage Trait):**
- Purpose: Centralized data access with transaction support
- Examples: `src/storage/pg.rs` (PostgreSQL), `src/storage/mock.rs` (test double)
- Pattern: Trait defines 20+ methods (insert_facts, vector_search, keyword_search, get_neighbors, etc.); SQL implementation handles network filtering at query layer
- Transaction support: begin() returns TransactionHandle (implements MemoryStore), commit() makes writes permanent

**Retriever (Recall Trait):**
- Purpose: Pluggable retrieval strategy
- Examples: `src/recall/semantic.rs`, `src/recall/keyword.rs`, `src/recall/graph.rs`, `src/recall/temporal.rs`
- Pattern: Each retriever.retrieve(query) returns Vec<ScoredFact>; four instances are composed and fused in DefaultRecallPipeline

**Chunker (Retain Trait):**
- Purpose: Text splitting strategy with optional turn-preservation
- Examples: `src/retain/chunker.rs` (SimpleChunker with token approximation)
- Pattern: chunk(input, config) returns Vec<Chunk>; chunks carry optional context for coreference

**FactExtractor (Retain Trait):**
- Purpose: LLM-based fact extraction with structured output
- Examples: `src/retain/extractor.rs` (LlmFactExtractor using tool calling)
- Pattern: extract(input) returns Vec<ExtractedFact> with entity_mentions, temporal_range, network classification

**EntityResolver (Retain Trait):**
- Purpose: Match entity mentions to existing entities or create new ones
- Examples: `src/retain/resolver.rs` (LayeredEntityResolver with fuzzy matching)
- Pattern: resolve(mentions, bank_id, store) returns Vec<ResolvedEntity> with (mention, entity_id, is_new)

**GraphBuilder (Retain Trait):**
- Purpose: Infer semantic/temporal/causal links between facts
- Examples: `src/retain/graph_builder.rs` (DefaultGraphBuilder with LLM causal inference)
- Pattern: build_links(facts, bank_id, store) returns Vec<GraphLink> with link_type and weight

**LlmClient (LLM Trait):**
- Purpose: Completion requests and response handling
- Examples: `src/llm/anthropic.rs` (Anthropic API), `src/llm/openai.rs` (OpenAI API), `src/llm/mock.rs` (test double)
- Pattern: complete(request) returns CompletionResponse; handles tool-calling, JSON parsing, retries

**EmbeddingClient (Embedding Trait):**
- Purpose: Text vectorization for semantic search and clustering
- Examples: `src/embedding/local.rs` (ONNX BGE), `src/embedding/openai.rs` (OpenAI API)
- Pattern: embed(texts) returns Vec<Vec<f32>>; dimensions() returns vector size

**Reranker (Reranking Trait):**
- Purpose: Cross-encoder scoring to refine retrieval results
- Examples: `src/recall/reranker/local.rs` (ONNX MiniLM), `src/recall/reranker/api.rs` (Cohere-compatible)
- Pattern: rerank(query, facts, limit) returns Vec<ScoredFact> with updated scores

## Entry Points

**main.rs:**
- Location: `src/main.rs`
- Triggers: Binary startup
- Responsibilities: Parse env (LOG_FORMAT, LISTEN_ADDR), build runtime, start Axum server with Retain/Recall/Reflect/Consolidation pipelines, attach MCP service at /mcp

**Bench Harness:**
- Location: `bench/locomo/locomo.rs` (binary `locomo-bench`)
- Triggers: Manual `cargo run --release --bin locomo-bench -- [flags]`
- Responsibilities: Load LoCoMo conversations, run retain→consolidate→reflect sequence, collect metrics (accuracy, latency, retrieved context)

**Bench View Tool:**
- Location: `bench/view.rs` (binary `view`)
- Triggers: Manual `cargo run --bin view -- [flags]`
- Responsibilities: Parse bench result JSON files, display per-conversation table, compute accuracy by question type

**API HTTP Server:**
- Location: `src/server/mod.rs` → Axum router
- Triggers: requests to /v1/banks, /v1/banks/{id}/retain, /v1/banks/{id}/recall, /v1/banks/{id}/reflect, /v1/banks/{id}/consolidate
- Responsibilities: Route to handler functions, marshal JSON, call pipelines

**MCP Server:**
- Location: `src/mcp/mod.rs`
- Triggers: MCP protocol requests to /mcp endpoint
- Responsibilities: Decode tool calls (retain, recall, reflect, list_banks, create_bank), invoke pipelines, return JSON responses

## Error Handling

**Strategy:** Result<T> alias with typed Error enum; errors propagate up and are converted to HTTP 400/404/500 status codes

**Patterns:**
- **Parsing errors**: parse_bank_id → Error::InvalidId (400)
- **Not found**: store.get_bank → Error::NotFound (404)
- **LLM errors**: complete() → Error::Llm (500, may retry in RetryingLlmClient)
- **Rate limiting**: HTTP 429 captured as Error::RateLimit (retry exponentially)
- **Embedding mismatch**: Error::EmbeddingDimensionMismatch (400)
- **Database errors**: sqlx::Error → Error::Storage (500)

Retry wrapper (src/llm/retry.rs) intercepts RateLimit and ServerError, exponential backoff with jitter.

## Cross-Cutting Concerns

**Logging:** Tracing subscriber with structured JSON optional (LOG_FORMAT env var)
- Entry points use info_span! for context
- Subcomponents use debug!, info!, warn!, error! macros
- Metrics logged at pipeline completion (facts_stored, entities_resolved, links_created)

**Validation:** Input validation at handler layer
- Bank ID parsing (ULID format)
- Disposition field ranges (skepticism 1-5)
- Chunk config reasonable limits
- Embedding dimension match at retain time

**Authentication:** None at API layer; assumes external proxy or client-side trust
- MCP runs in-process, no auth needed
- HTTP API expects trusted network

**Configuration:** Environment variables via dotenvy
- LLM provider/model/API key (RETAIN_LLM_PROVIDER, etc.)
- Embedding provider/model (EMBEDDING_PROVIDER, EMBEDDING_MODEL_PATH)
- Reranker provider (RERANKER_PROVIDER, defaults to local)
- Retriever limits (RETRIEVER_LIMIT, RERANK_TOP_N)
- Database (DATABASE_URL)
- Server port (LISTEN_ADDR)
- Logging (LOG_FORMAT)

---

*Architecture analysis: 2025-03-15*
