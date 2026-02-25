# Build Plan: Rust Memory Engine (Hindsight Architecture)

## Guiding Principle

Every phase produces a crate (or module) with a **trait boundary** at the top and **real tests** you can run. Components communicate through well-defined types, not implementation details. When you wire them together at the end, the integration is mechanical because the contracts are already proven.

---

## The Dependency Graph (What Actually Depends on What)

```
                    ┌─────────────┐
                    │   Reflect   │  Phase 5
                    │   (CARA)    │
                    └──────┬──────┘
                           │ uses
                    ┌──────▼──────┐
                    │   Recall    │  Phase 4
                    │  (TEMPR)    │
                    └──────┬──────┘
                           │ uses
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
   │  Retrieval   │  │   Graph    │  │   Fusion    │  Phase 3
   │  Strategies  │  │ Traversal  │  │ + Reranking │
   └──────┬──────┘  └─────┬──────┘  └─────────────┘
          │                │
   ┌──────▼──────┐  ┌─────▼──────┐
   │   Retain    │  │   Graph    │  Phase 2
   │  Pipeline   │  │   Build    │
   │ + Chunker   │  │            │
   │ + OpReinf.  │  │            │
   └──────┬──────┘  └─────┬──────┘
          │                │
   ┌──────▼────────────────▼──────┐
   │        Storage + Types       │  Phase 1
   │     LLM Client, Embeddings   │
   └──────────────────────────────┘

          ┌──────────────────────────────┐
          │  Phase 6: Background Workers │
          │  6A Observation Consolidator │
          │  6B Opinion Merger           │
          │  6C Mental Model Generator   │
          └──────────────────────────────┘
```

---

## Phase 1: Foundation

Everything else builds on these. No AI logic here — just plumbing that works.

### 1A · Core Types

**What it is:** The shared vocabulary every other component speaks. Fact, Entity, Observation, Opinion, MemoryBank, GraphLink, plus all the enums (FactType, LinkType, NetworkType).

**Why it's first:** If you get these wrong, you refactor everything. Get them right and everything else slots in.

**The types to nail down:**

```rust
// The atomic unit of memory
struct Fact {
    id: FactId,
    bank_id: BankId,
    content: String,           // The natural language fact
    fact_type: FactType,       // World | Experience
    network: NetworkType,      // World | Experience | Observation | Opinion | MentalModel
    entity_ids: Vec<EntityId>, // Resolved entity references
    temporal_range: Option<TemporalRange>,  // When this fact was true
    embedding: Option<Vec<f32>>,
    confidence: Option<f32>,   // For opinions
    evidence_ids: Vec<FactId>, // Facts that support this (for observations, opinions, mental models)
    source_turn_id: Option<TurnId>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

enum FactType { World, Experience }
enum NetworkType { World, Experience, Observation, Opinion, MentalModel }

struct Entity {
    id: EntityId,
    canonical_name: String,
    aliases: Vec<String>,
    entity_type: EntityType,   // Person | System | Concept | Event | ...
    bank_id: BankId,
}

struct GraphLink {
    source_id: FactId,
    target_id: FactId,
    link_type: LinkType,       // Temporal | Semantic | Entity | Causal
    weight: f32,
}

struct MemoryBank {
    id: BankId,
    name: String,
    mission: String,           // What this bank should focus on retaining
    directives: Vec<String>,   // Guardrails/compliance rules that must never be violated
    disposition: Disposition,  // Skepticism, literalism, empathy, bias_strength
}

struct Disposition {
    skepticism: u8,            // 1-5: how much evidence required before accepting claims
    literalism: u8,            // 1-5: interpret literally vs read between lines
    empathy: u8,               // 1-5: weight given to emotional/human factors
    bias_strength: f32,        // 0.0-1.0: how strongly disposition influences responses
}

// What recall returns
struct RecallResult {
    facts: Vec<ScoredFact>,
    total_tokens: usize,
}

struct ScoredFact {
    fact: Fact,
    score: f32,
    sources: Vec<RetrievalSource>,  // Which strategies found this
}

// What reflect returns
struct ReflectResult {
    response: String,
    sources: Vec<FactId>,          // Which facts grounded the response
    new_opinions: Vec<Fact>,       // Opinions formed during reflection
    confidence: f32,
}
```

**How to test:** Pure compile-time correctness plus serialization round-trips. Every type should `Serialize + Deserialize` cleanly. Write property tests (proptest) that generate random Facts and verify `deserialize(serialize(fact)) == fact`.

**Done when:** You can construct every type, serialize to JSON, deserialize back, and everything round-trips. This sounds trivial but catches field type mistakes early.

---

### 1B · Storage Layer

**What it is:** Postgres operations for all the types above. CRUD for facts, entities, observations, opinions, banks, graph links. Vector similarity search via pgvector.

**Interface (trait):**

```rust
#[async_trait]
trait MemoryStore {
    // Facts
    async fn insert_facts(&self, facts: &[Fact]) -> Result<Vec<FactId>>;
    async fn get_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>>;
    async fn get_facts_by_bank(&self, bank: BankId, filter: FactFilter) -> Result<Vec<Fact>>;

    // Entities
    async fn upsert_entity(&self, entity: &Entity) -> Result<EntityId>;
    async fn find_entity(&self, bank: BankId, name: &str) -> Result<Option<Entity>>;
    async fn get_entity_facts(&self, entity: EntityId) -> Result<Vec<Fact>>;

    // Graph
    async fn insert_links(&self, links: &[GraphLink]) -> Result<()>;
    async fn get_neighbors(&self, fact_id: FactId, link_type: Option<LinkType>) -> Result<Vec<(FactId, f32)>>;

    // Vector search
    async fn vector_search(&self, embedding: &[f32], bank: BankId, limit: usize) -> Result<Vec<ScoredFact>>;

    // Banks
    async fn get_bank(&self, id: BankId) -> Result<MemoryBank>;
    async fn create_bank(&self, bank: &MemoryBank) -> Result<BankId>;
}
```

**How to test:** Spin up a Postgres container (use `testcontainers-rs`). Every test gets a fresh database. Real SQL, real pgvector, real assertions:

```
Test: insert 50 facts → vector_search with a known embedding → verify top result is the semantically closest
Test: insert facts with temporal ranges → filter by date range → verify only matching facts returned
Test: insert entities with aliases → find_entity("auth module") matches entity with canonical name "authentication system"
Test: insert graph links → get_neighbors returns correct adjacency with weights
Test: insert into bank A and bank B → queries on bank A never return bank B facts
```

**Done when:** All CRUD works against real Postgres, vector search returns results ordered by cosine similarity, and bank isolation is proven.

**Crate dependencies:** `sqlx`, `pgvector` (or raw SQL with `vector` type), `testcontainers`

---

### 1C · LLM Client

**What it is:** Trait abstraction over LLM providers. Doesn't know about memory — just sends prompts, gets structured responses.

**Interface:**

```rust
#[async_trait]
trait LlmClient: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;
    async fn complete_structured<T: DeserializeOwned>(&self, request: CompletionRequest) -> Result<T>;
}

struct CompletionRequest {
    system: String,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: usize,
}
```

**Implementations:**
- `AnthropicClient` — real API calls to Claude
- `OpenAIClient` — real API calls to GPT
- `MockLlmClient` — returns canned responses for testing

**Per-operation model configuration:** Hindsight uses different models for different operations — strong structured output models (GPT-4o, Claude) for retain/extraction, and cheaper/faster models (GPT-4o-mini, Groq/Llama) for reflect/reasoning. Recall doesn't use an LLM at all. Design the client so each pipeline component receives its own `impl LlmClient`, configured at startup from per-operation config:

```rust
struct LlmConfig {
    default_provider: ProviderConfig,
    retain_provider: Option<ProviderConfig>,   // Override for fact extraction
    reflect_provider: Option<ProviderConfig>,  // Override for reflection
    // Recall doesn't need an LLM
}
```

**How to test:** The mock is the important part. Integration tests with real APIs use a `#[ignore]` attribute so they don't run in CI but you can run them manually:

```
Test (mock): complete_structured returns parsed JSON matching the expected type
Test (mock): handles malformed JSON gracefully (retry or error)
Test (ignored, real): send a simple prompt to Claude, verify response is non-empty
```

**Why this matters:** Every component above this will take `impl LlmClient`. During testing, they all get the mock. This is the seam that makes everything else testable without burning API credits.

**Done when:** Mock client works, at least one real provider works, structured output parsing handles edge cases (markdown fences, extra text around JSON).

---

### 1D · Embedding Client

**What it is:** Same pattern as LLM client but for embeddings.

```rust
#[async_trait]
trait EmbeddingClient: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}
```

**Implementations:**
- `LocalEmbeddings` — ONNX runtime via `ort` crate running BAAI/bge-small-en-v1.5 (384 dims). **This is the primary path** — Hindsight runs local embeddings by default and so should you. No API costs, no latency, deterministic. The `ort` crate has solid Rust bindings.
- `OpenAIEmbeddings` — text-embedding-3-small/large (1536 or 3072 dims). Use when you want higher quality and don't mind API costs.
- `MockEmbeddings` — deterministic embeddings for testing (hash the text to a vector)

**Important:** Once you store facts with a given embedding dimension, you can't change it without losing data. Pick your model early and stick with it. Start with bge-small (384 dims) — it's what Hindsight uses and it's fast enough for local inference.

**How to test:**

```
Test (mock): embed(["hello", "world"]) returns 2 vectors of correct dimension
Test (mock): same text always produces same embedding (deterministic)
Test: cosine_similarity(embed("cat"), embed("dog")) > cosine_similarity(embed("cat"), embed("database"))
      ^ This test uses the real API and validates the embedding space makes sense
```

**Done when:** Mock returns deterministic vectors, real provider returns correct dimensions, batch embedding works.

---

## Phase 2: Retain Pipeline

This is where raw conversation becomes structured memory. Each step is independently testable because inputs and outputs are concrete types.

### 2A · Chunker

**What it is:** Segments input data into extractable chunks before fact extraction. For short conversation turns (a few messages), this is trivial — pass them through. For long documents or multi-turn transcripts, you need to split into overlapping windows that preserve context.

**Interface:**

```rust
trait Chunker {
    fn chunk(&self, input: &str, config: &ChunkConfig) -> Vec<Chunk>;
}

struct ChunkConfig {
    max_tokens: usize,         // Max tokens per chunk (default: ~2000)
    overlap_tokens: usize,     // Overlap between adjacent chunks (default: ~200)
    preserve_turns: bool,      // Don't split in the middle of a conversation turn
}

struct Chunk {
    content: String,
    index: usize,              // Position in original document
    context: Option<String>,   // Summary of preceding chunks for coreference resolution
}
```

**Why this matters:** The paper (Section 4.1.2) has a specific chunking strategy discussion. For conversation turns under the token limit, the chunker is a passthrough. But for document ingestion or very long transcripts, you need this to avoid losing information at chunk boundaries. The overlap and context fields help the fact extractor resolve coreferences like "she" or "the system" that refer to entities mentioned in previous chunks.

**How to test:** Pure function, very testable:

```
Test: Short input (< max_tokens) → single chunk, no splitting
Test: Long input → multiple chunks with correct overlap
Test: preserve_turns=true → never splits mid-turn (respects speaker boundaries)
Test: Chunks have sequential indices
Test: Context field populated from preceding chunk content
```

**Done when:** Short inputs pass through unchanged, long inputs split sensibly, conversation turn boundaries are respected.

---

### 2B · Fact Extractor

**What it is:** Takes a chunk (from 2A), sends it to an LLM with a structured extraction prompt, returns typed facts with temporal ranges and entity mentions.

**Interface:**

```rust
#[async_trait]
trait FactExtractor {
    async fn extract(&self, input: &ExtractionInput) -> Result<Vec<ExtractedFact>>;
}

struct ExtractionInput {
    content: String,           // The chunk text
    bank_id: BankId,
    context: Option<String>,   // Context from chunker (preceding content summary)
    timestamp: DateTime<Utc>,  // When this interaction happened
    custom_instructions: Option<String>,  // Domain-specific extraction guidelines
}

struct ExtractedFact {
    content: String,
    fact_type: FactType,
    entity_mentions: Vec<String>,   // Raw strings, not yet resolved
    temporal_range: Option<TemporalRange>,
}
```

**Custom extraction mode:** The Hindsight implementation supports injecting custom guidelines while keeping the structural parts of the prompt intact (output format, coreference resolution, temporal handling). This is critical for domain-specific use — a dev project bank should emphasize architectural decisions and tradeoffs, while a customer support bank should emphasize user preferences and issue history. The `custom_instructions` field gets injected into the extraction prompt alongside the base structural instructions.

**The real implementation** calls the LLM with a prompt like:

```
Extract factual statements from this conversation. For each fact, provide:
- content: the fact in a standalone sentence
- type: "world" (objective fact) or "experience" (something the agent did/observed)
- entities: list of named things mentioned
- temporal_range: when this fact was true (if inferrable)

Return JSON array. Only include facts worth remembering long-term.
```

**How to test:** This is where golden test fixtures matter. Create 5-10 input conversations and hand-label the expected facts. Then:

```
Test (mock LLM): Given canned LLM response JSON, parser produces correct ExtractedFact structs
Test (mock LLM): Malformed LLM output (missing fields, extra text) handled gracefully
Test (golden, real LLM): "We decided to use Postgres instead of MongoDB because we need 
      transactions" → extracts fact about database choice with entities [Postgres, MongoDB]
Test (golden, real LLM): "I fixed the login bug yesterday" → extracts experience fact 
      with temporal range anchored to yesterday relative to timestamp
```

**The prompt is the product.** You'll iterate on this more than any other code. Keep the prompt in a separate file so you can tweak it without recompiling.

**Done when:** Parser handles all LLM output edge cases. Golden tests pass with at least one real LLM provider. Extraction quality is "good enough" (you'll keep tuning this forever, that's fine).

---

### 2C · Entity Resolver

**What it is:** Takes raw entity mention strings from the extractor and maps them to canonical entities in the database. Creates new entities when needed. Merges duplicates.

**Interface:**

```rust
#[async_trait]
trait EntityResolver {
    async fn resolve(&self, mentions: &[String], bank_id: BankId) -> Result<Vec<ResolvedEntity>>;
}

struct ResolvedEntity {
    entity_id: EntityId,
    canonical_name: String,
    mention: String,           // The original mention that was resolved
    is_new: bool,              // Was this entity just created?
    confidence: f32,           // How confident the resolution is
}
```

**Resolution strategy (layered):**
1. Exact match against existing entity names/aliases → high confidence
2. Embedding similarity against existing entities → medium confidence
3. LLM disambiguation: "Is 'the auth system' the same entity as 'authentication module'?" → used when embedding similarity is borderline
4. Create new entity if no match

**How to test:** This is highly testable with a real database and mock LLM:

```
Test: resolve(["Postgres"]) when DB has entity "PostgreSQL" with alias "Postgres" → exact match
Test: resolve(["the auth module"]) when DB has entity "authentication system" → 
      embedding similarity finds candidate, LLM confirms match
Test: resolve(["React"]) when DB has no frontend entities → creates new entity
Test: resolve(["Postgres", "Postgres", "pg"]) in single call → all resolve to same entity, 
      no duplicates created
Test: resolve across banks → entity "Postgres" in bank A doesn't match bank B
```

**Done when:** Exact matches work, embedding-based fuzzy matching works, new entity creation works, no duplicate entities created for the same concept within a bank.

---

### 2D · Graph Builder

**What it is:** After facts are extracted and entities resolved, this constructs the four types of graph links between facts.

**Interface:**

```rust
#[async_trait]
trait GraphBuilder {
    async fn build_links(&self, new_facts: &[Fact], bank_id: BankId) -> Result<Vec<GraphLink>>;
}
```

**Link construction rules:**
- **Temporal:** Facts with overlapping or adjacent temporal ranges get linked. Weight decreases with temporal distance.
- **Semantic:** Facts with embedding cosine similarity above threshold get linked. Weight = similarity score.
- **Entity:** Facts sharing resolved entities get linked. Weight = number of shared entities / total unique entities.
- **Causal:** LLM call: "Given fact A and fact B, is there a causal relationship?" Binary yes/no + direction. Only checked for facts that share entities or are temporally close (to limit LLM calls).

**How to test:** Synthetic facts with known relationships:

```
Test: Two facts 1 day apart → temporal link created with high weight
Test: Two facts 1 year apart → no temporal link (below threshold)
Test: Two facts about "Postgres" → entity link created
Test: "We chose Postgres" and "Postgres has been reliable" → 
      causal link detected (decision → outcome)
Test: Facts with similar embeddings → semantic link created, weight ≈ cosine similarity
Test: Unrelated facts → no links created (verify sparsity)
```

**Done when:** All four link types constructed correctly. Graph is sparse (not everything connects to everything). Weights are meaningful.

---

### 2E · Retain Orchestrator

**What it is:** Wires 2A + 2B + 2C + 2D together, plus opinion reinforcement. Takes raw input, chunks it, produces stored facts with resolved entities and graph links, and updates existing opinions when new evidence arrives.

**Interface:**

```rust
#[async_trait]
trait RetainPipeline {
    async fn retain(&self, input: &RetainInput) -> Result<RetainOutput>;
}

struct RetainInput {
    content: String,
    bank_id: BankId,
    timestamp: DateTime<Utc>,
    context: Option<String>,
    custom_instructions: Option<String>,  // Domain-specific extraction guidelines
}

struct RetainOutput {
    facts_stored: usize,
    entities_resolved: usize,
    links_created: usize,
    new_entities: Vec<EntityId>,
    opinions_reinforced: usize,    // Existing opinions updated by new evidence
    opinions_weakened: usize,      // Existing opinions contradicted by new evidence
}
```

**Internal flow:**

```
input.content
    → Chunker (2A): split into chunks if needed
    → For each chunk:
        → Fact Extractor (2B): extract facts from chunk
        → Entity Resolver (2C): resolve entity mentions to canonical entities
        → Store facts in DB with resolved entity IDs
        → Graph Builder (2D): construct links between new facts and existing graph
    → Opinion Reinforcement:
        → For each new fact, check existing opinions in the bank
        → If new fact supports an existing opinion → reinforce (increase confidence)
        → If new fact contradicts an existing opinion → weaken (decrease confidence)
    → Return RetainOutput
```

**Opinion reinforcement during retain** is specified in the paper (Section 3.2): "when new evidence arrives, existing beliefs in 𝒪 are also updated through an opinion reinforcement mechanism." This means retain is not just additive — it actively updates the opinion network as a side effect. The reinforcement check uses embedding similarity between new facts and existing opinions, with an LLM call to confirm support/contradiction when similarity is above threshold.

**How to test:** Integration test with real DB, mock LLM:

```
Test: retain("We migrated from MySQL to Postgres last month") → 
      creates facts, resolves entities [MySQL, Postgres], 
      creates entity links between facts mentioning same DB
Test: retain same content twice → no duplicate facts (idempotency)
Test: retain two related messages sequentially → graph links connect them
Test: retain long document → chunker splits, all chunks processed, facts from all chunks stored
Test: retain fact supporting existing opinion → opinion confidence increases
Test: retain fact contradicting existing opinion → opinion confidence decreases
Test: retain with custom_instructions → extraction prompt includes custom guidelines
```

**Done when:** Full pipeline runs end-to-end. Facts, entities, and graph links all land in the database correctly. Opinion reinforcement fires when new evidence matches existing opinions. This is your first "it actually works" moment.

---

## Phase 3: Retrieval Strategies

Each retrieval strategy is independently testable and returns a common `Vec<ScoredFact>`. They don't know about each other.

### 3A · Semantic Retriever

```rust
#[async_trait]
trait Retriever {
    async fn retrieve(&self, query: &RecallQuery) -> Result<Vec<ScoredFact>>;
}
```

Uses the embedding client to embed the query, then does pgvector similarity search.

**Test:** Insert 100 facts with known embeddings. Query with an embedding close to fact #42. Verify fact #42 is top result.

### 3B · Keyword Retriever (BM25)

Same `Retriever` trait. Uses `tantivy` to build a full-text index over fact content.

**Test:** Insert facts containing specific terms. Query "Postgres migration" → facts mentioning those terms ranked highest.

**Implementation note:** Tantivy needs an index directory. On startup, rebuild the index from the database (or maintain it incrementally). This is a design decision — start with full rebuild, optimize later.

### 3C · Graph Retriever

Same `Retriever` trait. Starts from facts that match the query (via keyword or embedding), then walks graph links to discover connected facts.

**Test:** Create a chain: Fact A → (entity link) → Fact B → (causal link) → Fact C. Query matches Fact A. Verify Fact B and C are found with decreasing scores (1-hop vs 2-hop).

**Algorithm:** Spreading activation with decay. Start with seed facts (matched by embedding), propagate activation along edges multiplied by edge weight, decay per hop. Return all facts with activation above threshold.

### 3D · Temporal Retriever

Same `Retriever` trait. Extracts temporal references from the query ("last month", "in November", "recently"), converts to date ranges, filters facts by temporal overlap.

**Test:** Insert facts spanning Jan–Dec. Query "what happened in March" → only March facts returned, ordered by relevance.

**Implementation note:** Temporal extraction from natural language queries is its own sub-problem. Start simple: regex for dates, "last week/month/year", relative references. Use LLM for complex cases. This doesn't need to be perfect on day one.

### 3E · Reciprocal Rank Fusion

**What it is:** Pure function. Takes ranked lists from all retrievers, produces a single merged ranking.

```rust
fn fuse_rankings(
    rankings: &[Vec<ScoredFact>],
    k: f32,  // RRF constant, typically 60.0
) -> Vec<ScoredFact>
```

**Algorithm:** For each fact appearing in any list, compute `score = Σ 1/(k + rank_in_list_i)`. Sort by fused score descending.

**Test:** This is the most testable component in the entire system. Pure function, deterministic, no I/O.

```
Test: Single list → output order matches input
Test: Two lists with same items in different order → fused ranking is reasonable
Test: Item ranked #1 in two lists beats item ranked #1 in one list
Test: Item in all four lists beats item in only one, regardless of position
Test: Empty lists handled gracefully
```

**Done when:** All pure-function tests pass. This never needs to change.

### 3F · Cross-Encoder Reranker

**What it is:** Takes the top-N results from RRF and re-scores them with a cross-encoder model that sees (query, fact) pairs jointly. More accurate than embedding similarity but more expensive.

```rust
#[async_trait]
trait Reranker {
    async fn rerank(&self, query: &str, facts: &[ScoredFact], top_k: usize) -> Result<Vec<ScoredFact>>;
}
```

**Implementations:**
- `CrossEncoderReranker` — runs an ONNX model locally via `ort`, or calls Cohere/Jina rerank API
- `NoOpReranker` — returns input unchanged (for testing, or when you want to skip this step)
- `MockReranker` — reverses the order (to verify the pipeline actually uses reranker output)

**Test with mock:** Verify the pipeline uses reranker output, not pre-rerank order. Use MockReranker that reverses order → if final results are reversed, reranker is wired in correctly.

**Test with real model:** Rerank ["Python is a programming language", "The python snake is large", "Python was created by Guido"] for query "programming language" → programming fact ranked first.

### 3G · Token Budget Enforcer

**What it is:** Pure function. Takes a ranked list of facts and a token budget, returns the subset that fits.

```rust
fn apply_budget(
    facts: &[ScoredFact],
    budget: usize,
    tokenizer: &impl Tokenizer,
) -> Vec<ScoredFact>
```

**Algorithm:** Greedy. Walk the ranked list, accumulate token counts, stop when budget would be exceeded.

**Test:**

```
Test: Budget of 100 tokens, 10 facts of ~20 tokens each → returns first 5
Test: Budget of 0 → returns empty
Test: Single fact exceeding budget → returns empty (don't include partial facts)
Test: Exact budget → includes the fact that fills it exactly
```

---

## Phase 4: Recall Pipeline (TEMPR)

### 4A · Recall Orchestrator

**What it is:** Wires all Phase 3 components together. This IS TEMPR.

```rust
#[async_trait]
trait RecallPipeline {
    async fn recall(&self, query: &RecallQuery) -> Result<RecallResult>;
}

struct RecallQuery {
    query: String,
    bank_id: BankId,
    budget_tokens: usize,          // Default 2048
    types: Option<Vec<NetworkType>>,  // Filter by network type
    temporal_anchor: Option<TemporalRange>,
    tag_filter: Option<Vec<String>>,
}
```

**Internal flow:**

```
query → embed query
     → run 4 retrievers in parallel (tokio::join!)
     → collect Vec<ScoredFact> from each
     → RRF fusion
     → cross-encoder rerank top 50
     → apply token budget
     → return RecallResult
```

**How to test:** Integration test with real DB, mock LLM, mock embeddings:

```
Test: Retain 20 facts about mixed topics. Recall "database decisions" → 
      returns only DB-related facts, within token budget
Test: Recall with types=[Observation] → only observations returned
Test: Recall with budget=100 → result fits in 100 tokens
Test: Recall with temporal_anchor="last week" → only recent facts
Test: Recall empty bank → empty result, no errors
```

**Done when:** Parallel retrieval works, fusion produces sensible rankings, budget is respected, type filtering works.

---

## Phase 5: Reflect Pipeline (CARA)

### 5A · Memory Hierarchy Assembler

**What it is:** Queries memory in priority order: mental models → observations → raw facts. Composes them into a structured context block for the LLM. This priority order matches how Hindsight's reflect operates in practice.

```rust
#[async_trait]
trait HierarchyAssembler {
    async fn assemble(&self, query: &str, bank_id: BankId, budget: usize) -> Result<AssembledContext>;
}

struct AssembledContext {
    mental_models: Vec<Fact>,     // Highest priority — cross-cutting synthesis
    observations: Vec<Fact>,      // Mid priority — entity-level summaries
    raw_facts: Vec<Fact>,         // Lowest priority — individual facts
    opinions: Vec<Fact>,          // Relevant beliefs with confidence scores
    total_tokens: usize,
    formatted: String,            // Ready to inject into LLM prompt
}
```

**Internal flow:**
1. Recall with `types=[MentalModel]`, budget = 15% of total — highest priority, most synthesized
2. Recall with `types=[Observation]`, budget = 30% of total — entity-level summaries
3. Recall with `types=[World, Experience]`, budget = 40% of total — granular facts
4. Recall with `types=[Opinion]`, budget = 15% of total — relevant beliefs
5. Format into structured text with clear section headers and source attribution

**Budget allocation rationale:** Mental models are dense (one model covers what 10+ facts say), so they get less space but highest priority. Raw facts need the most space because they're granular. Observations sit between. Opinions get their own allocation so they don't compete with facts for budget.

**How to test:**

```
Test: Bank with mental models, observations, and facts → mental models appear first in formatted output
Test: Budget splitting → each tier gets approximately its allocated share
Test: Bank with no mental models → budget redistributed to observations and facts (graceful fallback)
Test: Bank with no observations either → all budget goes to raw facts
Test: Formatted output includes fact IDs for source tracking
Test: Opinions section includes confidence scores
```

### 5B · Disposition & Directives Engine

**What it is:** Takes the bank's disposition profile and directives, verbalizes them into system prompt instructions that shape the LLM's response style and enforce compliance rules.

```rust
fn verbalize_bank_profile(bank: &MemoryBank) -> BankPromptContext

struct BankPromptContext {
    disposition_prompt: String,   // Personality/reasoning style instructions
    directives_prompt: String,    // Hard compliance rules (never violate)
    mission_prompt: String,       // What this bank focuses on
}
```

**Disposition** controls reasoning style. **Directives** are hard guardrails — things like "Never recommend competitors," "Always cite sources," "Decline requests outside scope." They get injected as non-negotiable rules in the system prompt, separate from the softer disposition influence.

**Pure function.** Takes disposition parameters (skepticism 1-5, literalism 1-5, empathy 1-5, bias_strength 0-1) plus mission string plus directives list, produces structured prompt components.

Example output for `skepticism=4, literalism=2, empathy=3, bias=0.7`:

```
"You are moderately skeptical — don't accept claims at face value, ask for evidence. 
You interpret things loosely, reading between the lines and inferring intent. 
You balance empathy with objectivity. Your personality traits strongly influence your responses."
```

**How to test:** Deterministic pure function. Snapshot tests:

```
Test: (1,1,1,0.0) → minimal personality influence
Test: (5,5,5,1.0) → maximum personality, very skeptical, very literal, very empathetic
Test: Edge cases → all valid ranges produce valid prompt text
Test: Bank with 3 directives → all appear in directives_prompt as hard rules
Test: Bank with empty directives → directives_prompt is empty/absent
Test: Mission text appears verbatim in mission_prompt
```

### 5C · Opinion Manager

**What it is:** Manages the opinion network. During reflection, the LLM may form new opinions or update existing ones. This component persists those changes.

```rust
#[async_trait]
trait OpinionManager {
    async fn get_opinions(&self, bank_id: BankId, topic: &str) -> Result<Vec<Fact>>;
    async fn form_opinion(&self, bank_id: BankId, opinion: &str, evidence: &[FactId], confidence: f32) -> Result<FactId>;
    async fn reinforce(&self, opinion_id: FactId, new_evidence: &[FactId], delta: f32) -> Result<()>;
    async fn weaken(&self, opinion_id: FactId, contradicting_evidence: &[FactId], delta: f32) -> Result<()>;
}
```

**Test:**

```
Test: form_opinion("Postgres is the right choice", evidence=[fact1, fact2], confidence=0.7) → stored
Test: reinforce with new evidence → confidence increases (capped at 1.0)
Test: weaken with contradicting evidence → confidence decreases (floored at 0.0)
Test: get_opinions("database") → returns opinions related to databases
```

### 5D · Reflect Orchestrator (CARA)

**What it is:** Wires hierarchy assembler + disposition + opinion manager + LLM together.

```rust
#[async_trait]
trait ReflectPipeline {
    async fn reflect(&self, query: &ReflectQuery) -> Result<ReflectResult>;
}

struct ReflectQuery {
    question: String,
    bank_id: BankId,
    budget_tokens: usize,
}
```

**Internal flow:**
1. Assemble memory hierarchy (5A)
2. Verbalize bank profile: disposition + directives + mission (5B)
3. Load existing relevant opinions (5C)
4. Compose prompt: system + directives (hard rules) + mission + disposition + memory context + opinions + question
5. Call LLM
6. Parse response for new/updated opinions
7. Persist opinion changes (5C)
8. Return ReflectResult with response + source attribution

**How to test (mock LLM):**

```
Test: reflect on question about a topic with stored facts → 
      response includes information from those facts
Test: reflect triggers opinion formation → new opinion stored in DB
Test: reflect with contradicting evidence → existing opinion confidence decreases
Test: source attribution → response cites specific fact IDs
```

---

## Phase 6: Consolidation (Background Worker)

### 6A · Observation Consolidator

**What it is:** Background job that runs after retain. Takes recently stored facts, checks if existing observations need updating, creates new observations where patterns emerge.

```rust
#[async_trait]
trait Consolidator {
    async fn consolidate(&self, bank_id: BankId, since: DateTime<Utc>) -> Result<ConsolidationReport>;
}

struct ConsolidationReport {
    observations_created: usize,
    observations_updated: usize,
    observations_unchanged: usize,
}
```

**Internal flow for each entity with new facts:**
1. Get existing observations for this entity
2. Get new facts about this entity since last consolidation
3. Ask LLM: "Given existing observation X and new facts [Y, Z], should the observation be: (a) unchanged, (b) updated with new info, (c) contradicted by new evidence?"
4. Apply the decision

**How to test:**

```
Test: New facts about entity with no observation → new observation created
Test: New fact consistent with observation → observation updated, wording refined
Test: New fact contradicting observation → observation revised, confidence lowered
Test: No new facts since last run → no changes (idempotent)
Test: 5 related facts about same topic → consolidated into single observation
```

**Done when:** Observations grow and evolve as new facts arrive. Contradictions are handled. Consolidation is idempotent.

---

### 6B · Opinion Merger

**What it is:** Background job that consolidates the opinion network. When multiple opinions about overlapping topics accumulate (e.g., several opinions about "database choice" formed at different times), this merges them into coherent, non-redundant beliefs. Described in paper Section 5.6 "Background Merging."

```rust
#[async_trait]
trait OpinionMerger {
    async fn merge(&self, bank_id: BankId) -> Result<OpinionMergeReport>;
}

struct OpinionMergeReport {
    opinions_merged: usize,      // Redundant opinions consolidated
    opinions_superseded: usize,  // Old opinions replaced by newer evidence
    opinions_conflicting: usize, // Contradictory opinions flagged (kept both, lowered confidence)
}
```

**Internal flow:**
1. Get all opinions in the bank, grouped by topic (via embedding similarity clustering)
2. For each cluster with >1 opinion:
   - Ask LLM: "Are these opinions about the same topic? If so, are they consistent, complementary, or contradictory?"
   - Consistent/complementary → merge into single opinion with combined evidence, higher confidence
   - Contradictory → keep the one with stronger evidence, lower confidence on the weaker one
   - Superseded (old opinion invalidated by new info) → mark old as superseded, keep new

**How to test:**

```
Test: Two opinions about same topic with consistent views → merged into one, confidence increases
Test: Two opinions about same topic with contradictory views → both kept, confidences adjusted
Test: Old opinion + newer contradicting opinion → old superseded, new kept
Test: Opinions about unrelated topics → no merging
Test: No opinions in bank → no-op
```

**Done when:** Opinion network stays clean and non-redundant. No topic has 15 opinions saying slightly different versions of the same thing.

---

### 6C · Mental Model Generator

**What it is:** Background job that synthesizes high-level mental models from observations and opinions. Mental models sit at the top of the memory hierarchy — they are the agent's "learned understanding" of cross-cutting topics, synthesized from many individual observations.

The paper describes four networks, but the shipped Hindsight product has evolved to include mental models as a higher layer above observations. During reflect, the agent checks sources in priority order: Mental Models → Observations → Raw Facts.

```rust
#[async_trait]
trait MentalModelGenerator {
    async fn generate(&self, bank_id: BankId) -> Result<MentalModelReport>;
}

struct MentalModelReport {
    models_created: usize,
    models_updated: usize,
    models_unchanged: usize,
}
```

**Internal flow:**
1. Identify topics with sufficient observation density (e.g., 3+ observations about related entities)
2. For each topic cluster:
   - Retrieve all relevant observations and opinions
   - Ask LLM: "Synthesize these observations into a coherent mental model — a high-level understanding of [topic]. Include key relationships, patterns, and current state."
   - Store as `NetworkType::MentalModel` with evidence links to source observations
3. For existing mental models, check if new observations warrant an update

**Example:** If the bank has observations like "Project uses Postgres for persistence," "Auth uses JWT with RS256," and "API layer is GraphQL" — the mental model generator might produce: "The system architecture consists of a GraphQL API layer, JWT-based authentication using RS256, and PostgreSQL for data persistence. These components were chosen for statelessness and type safety."

**How to test:**

```
Test: Bank with 5+ observations about related entities → mental model created
Test: Bank with few scattered observations → no mental model (insufficient density)
Test: New observation added to existing mental model's topic → model updated
Test: Mental model links back to source observations via evidence_ids
Test: Mental models stored with NetworkType::MentalModel
```

**Done when:** Mental models emerge automatically from accumulated observations. They provide the highest-level synthesis in the memory hierarchy. Reflect pipeline (5A) can retrieve them with highest priority.

---

## Phase 7: API Server

### 7A · HTTP API

**What it is:** Axum (or Actix) server exposing retain/recall/reflect as HTTP endpoints. This is the thin shell over everything above.

```
POST /v1/banks                          → create bank
POST /v1/banks/{id}/retain              → retain pipeline
POST /v1/banks/{id}/recall              → recall pipeline  
POST /v1/banks/{id}/reflect             → reflect pipeline
GET  /v1/banks/{id}/entities            → list entities
GET  /v1/banks/{id}/entities/{id}/facts → facts for entity
POST /v1/banks/{id}/consolidate         → trigger observation consolidation
POST /v1/banks/{id}/merge-opinions      → trigger opinion merging
POST /v1/banks/{id}/generate-models     → trigger mental model generation
GET  /v1/banks/{id}/mental-models       → list mental models
PUT  /v1/banks/{id}/directives          → update bank directives
```

**How to test:** `axum::test` or `reqwest` against a running server:

```
Test: POST /retain with conversation text → 200, facts stored
Test: POST /recall with query → 200, ranked facts returned
Test: POST /reflect with question → 200, synthesized response
Test: POST /retain then /recall for same topic → recall finds retained facts
Test: Invalid bank ID → 404
Test: Malformed JSON → 400 with useful error message
```

---

## Build Order Summary

| Order | Component | Depends On | Key Test Strategy | Approx Lines |
|-------|-----------|-----------|-------------------|-------------|
| 1 | Core Types (1A) | nothing | Serde round-trips, proptest | 350-450 |
| 2 | Storage (1B) | Types | testcontainers + real Postgres | 500-600 |
| 3 | LLM Client (1C) | nothing | Mock + ignored real tests | 450-550 |
| 4 | Embedding Client (1D) | nothing | Local ONNX + mock | 250-350 |
| 5 | Chunker (2A) | nothing | Pure function, boundary tests | 100-150 |
| 6 | Fact Extractor (2B) | LLM, Types | Golden fixtures + mock LLM | 250-350 |
| 7 | Entity Resolver (2C) | Storage, LLM, Embeddings | Real DB + mock LLM | 200-300 |
| 8 | Graph Builder (2D) | Storage, LLM, Embeddings | Synthetic facts + assertions | 300-400 |
| 9 | Retain Orchestrator (2E) | 2A+2B+2C+2D+OpinionMgr | Integration, opinion reinforcement | 250-350 |
| 10 | Semantic Retriever (3A) | Storage, Embeddings | Known embeddings in DB | 120-150 |
| 11 | BM25 Retriever (3B) | Tantivy | Known terms in index | 150-200 |
| 12 | Graph Retriever (3C) | Storage | Synthetic graph, known paths | 200-250 |
| 13 | Temporal Retriever (3D) | Storage | Known date ranges | 150-200 |
| 14 | RRF Fusion (3E) | nothing | Pure function, exhaustive | 60-80 |
| 15 | Reranker (3F) | ONNX or API | Mock + real model test | 150-200 |
| 16 | Token Budget (3G) | Tokenizer | Pure function | 60-80 |
| 17 | Recall Orchestrator (4A) | 3A-3G | Integration, parallel exec | 150-200 |
| 18 | Hierarchy Assembler (5A) | Recall | Budget splitting, 4-tier priority | 200-250 |
| 19 | Disposition & Directives (5B) | nothing | Pure function, snapshots | 100-150 |
| 20 | Opinion Manager (5C) | Storage | CRUD + confidence math | 150-200 |
| 21 | Reflect Orchestrator (5D) | 5A+5B+5C+LLM | Mock LLM, end-to-end | 250-300 |
| 22 | Observation Consolidator (6A) | Storage, LLM, Recall | Temporal idempotency | 250-350 |
| 23 | Opinion Merger (6B) | Storage, LLM, Embeddings | Cluster + merge logic | 200-300 |
| 24 | Mental Model Generator (6C) | Storage, LLM, Recall | Density threshold + synthesis | 250-350 |
| 25 | API Server (7A) | Everything | HTTP integration tests | 350-450 |

**Total: ~5,000-6,500 lines** (increased from original estimate due to chunker, opinion reinforcement, opinion merging, mental model generation, and directives)

---

## Practical Advice

### Start with the boring parts
Phases 1A and 1B are unglamorous but they're the foundation everything else sits on. Getting the types and storage right saves you from painful refactors later. Resist the urge to jump to fact extraction.

### Local embeddings first
Don't start with OpenAI embeddings as a crutch. Set up `ort` with bge-small-en-v1.5 early. It's what Hindsight uses, it runs fast on CPU, and you avoid API costs during development. The embedding dimension (384) gets baked into your database schema, so commit to it early.

### The mock LLM is your best friend
Almost every component above uses an LLM, but you don't want to burn API credits on every test run or deal with nondeterministic outputs in CI. The mock LLM returns canned responses. Use `#[ignore]` tests for real LLM validation — run them manually when tuning prompts.

### Use different LLM tiers deliberately
Strong models (Claude, GPT-4o) for retain — fact extraction quality is everything. Cheaper/faster models (GPT-4o-mini, Groq) for reflect — the heavy lifting is already done by the memory hierarchy. Recall uses no LLM at all. This keeps costs manageable as your memory bank grows.

### Keep prompts in files, not in code
Every LLM prompt (fact extraction, entity resolution, causal link detection, observation consolidation, opinion formation, mental model synthesis, reflection) should live in a `prompts/` directory as plain text files with template variables. When you recompile to change a prompt, you've already lost the iteration game. Hindsight supports a `custom` extraction mode for exactly this reason.

### Test contracts, not implementations
The trait boundaries exist so you can swap implementations. Test that `impl Retriever` satisfies the contract (returns scored facts, respects filters) — not that it uses a specific SQL query internally. This lets you change the guts without rewriting tests.

### The golden test fixtures are an investment
Spend real time hand-crafting 10-15 conversation snippets with hand-labeled expected facts, entities, and relationships. These become your regression suite. Every time you change the extraction prompt, run the golden tests. This is the closest thing to ground truth you'll have.

### Feature-flag the expensive parts
Cross-encoder reranking (3F), causal link detection (part of 2D), and mental model generation (6C) all require either a local model or extra API calls. Make them optional behind feature flags. The system should work without them (just slightly less accurate). Add them when you're ready.

### Phase 6 is where the magic happens
The background workers (observation consolidation, opinion merging, mental model generation) are what separate this from "fancy RAG." Without them, you have good retrieval. With them, you have a system that *learns*. But they're also the hardest to get right — the consolidation and synthesis prompts need heavy iteration. Build Phase 1-5 first, prove recall+reflect works, then invest in Phase 6.

### One crate or many?
Start as one crate with modules matching the phases. Split into a workspace with separate crates only if compile times become painful or you want to publish components independently. Premature crate splitting adds ceremony without benefit.

---

## Paper Cross-Reference

Maps build plan components to sections in "Hindsight is 20/20" (arXiv:2512.12818v1).

| Build Plan | Paper Section | Notes |
|-----------|--------------|-------|
| Four networks (1A) | §3.1 Four-Network Memory Organization | Paper defines W, B, O, S. Implementation adds MentalModel as 5th type (evolved beyond paper) |
| Three operations (1A) | §3.2 Three Core Operations | Retain(B,D)→M', Recall(B,Q,k)→{f...}, Reflect(B,Q,Θ)→(r,O') |
| Disposition params (1A) | §5.2.1 Disposition Parameters | skepticism, literalism, empathy ∈ [1,5], bias_strength ∈ [0,1] |
| Directives (1A) | §5.3 Bank Profile Structure | Shipped product feature; not detailed in paper but present in docs |
| Chunker (2A) | §4.1.2 Chunking Strategy | Paper discusses chunking; implementation detail not fully specified |
| Fact Extractor (2B) | §4.1.2 LLM-Based Narrative Fact Extraction | Paper appendix A.1 has the actual extraction prompt |
| Entity Resolver (2C) | §4.1.3 Entity Resolution and Linking | Recognition + disambiguation + entity link structure |
| Graph Builder (2D) | §4.1.4 Link Types and Graph Structure | Four link types: temporal, semantic, entity, causal |
| Opinion Reinforcement (2E) | §3.2 (Retain definition) | "when new evidence arrives, existing beliefs in O are also updated" |
| Observation Consolidation (6A) | §4.1.5 The Observation Paradigm | Background processing, preference-neutral entity summaries |
| Semantic Retrieval (3A) | §4.2.2 Semantic Retrieval (Vector Similarity) | Cosine similarity over embeddings |
| BM25 Retrieval (3B) | §4.2.2 Keyword Retrieval (BM25) | Full-text search |
| Graph Retrieval (3C) | §4.2.2 Graph Retrieval (Spreading Activation) | Multi-hop traversal with decay |
| Temporal Retrieval (3D) | §4.2.2 Temporal Graph Retrieval | Time-anchored retrieval |
| RRF Fusion (3E) | §4.2.3 Reciprocal Rank Fusion (RRF) | Standard RRF with k=60 |
| Reranker (3F) | §4.2.4 Neural Cross-Encoder Reranking | Post-fusion reranking |
| Token Budget (3G) | §4.2.5 Token Budget Filtering | Greedy selection up to budget |
| Memory Hierarchy (5A) | §6.1 Reflect description | "checks sources in priority order: Mental Models → Observations → Raw Facts" (from docs) |
| Disposition Engine (5B) | §5.3.1 Preference Description Generation | Verbalizes numeric params into natural language |
| Opinion Manager (5C) | §5.4 Opinion Network and Opinion Formation | Structure: (text, confidence, timestamp) |
| Opinion Formation (5D) | §5.4.2 Opinion Formation Process | During reflect, LLM may form new opinions |
| Opinion Reinforcement (5C) | §5.5 Opinion Reinforcement | Supporting evidence increases confidence |
| Opinion Merging (6B) | §5.6 Background Merging | Consolidates redundant/conflicting opinions |
| Mental Models (6C) | Shipped product (beyond paper) | Docs: "learned understanding formed by reflecting on raw memories" |
| Custom Extraction (2B) | Shipped product (beyond paper) | Config: HINDSIGHT_API_RETAIN_EXTRACTION_MODE=custom |
| Per-operation LLM (1C) | Shipped product (beyond paper) | Config: HINDSIGHT_API_RETAIN_LLM_MODEL vs REFLECT_LLM_MODEL |

**Paper appendix prompts to study:**
- A.1: Fact Extraction Prompt (TEMPR) — use as starting point for your extraction prompt
- A.2: Opinion Formation Prompt (CARA) — use for opinion detection during reflect
- A.3: Observation Generation Prompt (TEMPR) — use for consolidation (6A)
- A.5: Structured Output Schemas — Fact, Opinion, Observation JSON schemas
