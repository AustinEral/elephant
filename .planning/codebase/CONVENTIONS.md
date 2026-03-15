# Coding Conventions

**Analysis Date:** 2026-03-15

## Naming Patterns

**Files:**
- Modules use `snake_case`: `chunker.rs`, `entity_resolver.rs`, `mock.rs`
- Crates use `snake_case`: `elephant`, `testcontainers-modules`
- Directory names use `snake_case` to match module paths: `src/retain`, `src/recall`, `src/consolidation`

**Functions:**
- Methods and free functions use `snake_case`: `cosine_similarity()`, `estimate_tokens()`, `find_split_point()`
- Constructor: `new()` returns Self
- Conversion methods follow RFC 430:
  - `as_*()` — borrowed view (cheap)
  - `to_*()` — copying/allocating conversion
  - `into_*()` — consuming conversion
- Getters have no `get_` prefix: `dimensions()`, `name()`, `dimensions()`
- Predicates use `is_*()` or `has_*()`: `is_copy()`, `has_children()`

**Variables:**
- Local variables use `snake_case`: `bank_id`, `fact_ids`, `store`, `embeddings`
- Iterator variables typically short: `i`, `j`, `idx`
- Mutable bindings explicitly marked: `let mut facts = Vec::new()`

**Types:**
- Structs, enums, traits use `UpperCamelCase`: `MemoryBank`, `MemoryStore`, `SemanticRetriever`, `Fact`, `FactType`
- Type aliases use `UpperCamelCase`: `Result<T>` defined as `std::result::Result<T, Error>`
- Generic type parameters: `T`, `S`, `E`, `R` (short, uppercase)
- Newtype IDs: `FactId(Ulid)`, `BankId(Ulid)`, `EntityId(Ulid)` — single-field structs wrapping ULID

**Constants:**
- `SCREAMING_SNAKE_CASE`: `DEFAULT_TIMEOUT_SECS`, `MAX_RETRIES`, `EMBED_DIMS`, `OPINION_REINFORCEMENT_TEMPERATURE`, `OPINION_REINFORCEMENT_MAX_TOKENS`
- Prompt templates: `OPINION_REINFORCEMENT_PROMPT_TEMPLATE = include_str!(...)`

## Code Style

**Formatting:**
- Tool: `rustfmt` (no custom config, uses defaults)
- Line width: 100 characters (standard rustfmt)
- Indentation: 4 spaces
- Trailing commas: required in multi-line constructs
- Run before commit: `cargo fmt`

**Linting:**
- Tool: `cargo clippy`
- Fix warnings before committing
- Allow attribute used for compatibility: `#[allow(clippy::too_many_arguments)]` on constructors with many parameters (see `DefaultRetainPipeline::new()`)

**Compiler warnings:**
- Enable `#![warn(missing_docs)]` on library root (`src/lib.rs`)
- Public items require documentation
- Non-public items may skip if obvious from context

## Import Organization

**Order:**
```rust
// 1. Standard library
use std::collections::HashMap;
use std::sync::Arc;

// 2. External crates
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

// 3. Crate root items
use crate::error::Result;
use crate::types::Fact;

// 4. Relative/parent modules
use super::storage;
use super::*;  // In test modules
```

**Path Aliases:**
- No path aliases configured
- Use full qualified paths for clarity when needed
- Avoid `use crate::*` in production code; use in test modules with `use super::*`

**Module structure example:**
- `src/lib.rs` defines public API with `pub use` re-exports
- Internal submodules are private unless explicitly re-exported
- Example from `src/lib.rs`:
  ```rust
  pub use embedding::EmbeddingClient;
  pub use error::{Error, Result};
  pub use llm::LlmClient;
  pub use storage::MemoryStore;
  pub use types::*;
  ```

## Error Handling

**Framework:** `thiserror` for library code, `anyhow` only in tests/binaries

**Pattern:**
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid disposition: {0}")]
    InvalidDisposition(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
```

**Rules:**
- Never use `anyhow` in library crates — type information loss makes errors harder to handle
- Use `#[from]` for transparent error conversion (`sqlx::Error` → `Error::Storage`)
- Provide actionable error messages with context: `"max_tokens {} exceeds model limit {}"` not `"bad value"`
- Don't panic for recoverable errors — use `Result` instead

**When to panic:**
- Exhaustive enum match with `unreachable!()` on impossible arms
- Only when the error is programmer logic (never user input or external API failure)

## Logging

**Framework:** `tracing`

**Pattern:**
```rust
use tracing::{info, debug, info_span, Instrument};

info!("bank created: {bank_id}");
debug!("processing fact: {fact_id}");

let span = info_span!("process_batch", fact_count = facts.len());
async_work().instrument(span).await
```

**When to log:**
- Entry/exit of major operations
- Important state transitions
- Errors and warnings
- In async code: use `.instrument(span)` to attach context

## Comments

**When to comment:**
- Non-obvious algorithm or design choice (explain the "why")
- Important constraints or gotchas
- Links to related code or issues
- Complex regex patterns

**When NOT to comment:**
- Obvious code that explains itself
- Redundant restatement of what the code does
- Type information (types are self-documenting)

**JSDoc/TSDoc:**
- Use `///` doc comments for public items
- Use `//!` module-level doc comments at file start
- Required for all public functions/types/modules
- Keep brief: one-line summary, then explanation if non-obvious
- Examples:
  ```rust
  /// Retrieve facts by embedding the query and running vector similarity search.
  pub struct SemanticRetriever { ... }

  /// Cosine similarity between two vectors.
  ///
  /// Returns 0.0 if either vector has zero magnitude.
  pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32
  ```

## Function Design

**Size:**
- Aim for single, clear responsibility
- Extract helper functions when a block has a nameable purpose (e.g., `find_split_point()`)
- Use early returns to reduce nesting

**Parameters:**
- Prefer owned values over references when practical (cleaner ownership boundaries)
- Use trait objects (`&dyn Trait`) for runtime polymorphism; generic `<T: Trait>` for static dispatch
- No positional arguments past 3-4; use struct with named fields instead

**Return Values:**
- Async: use `Result<T>` for fallible operations
- Builders: return `Self` for chaining
- Iterators: return impl Iterator or named type
- Avoid nested generics in signatures; type aliases help readability

## Module Design

**Exports:**
- `lib.rs` defines public API with `pub use` re-exports
- Submodules marked `pub mod` (path is public) but contents are private unless explicitly re-exported
- Private module content (unmarked items) is crate-private

**Barrel files:**
- Used minimally; `src/lib.rs` re-exports key public types
- Submodule `mod.rs` files re-export traits and main types for that feature

**Example structure:**
```rust
// src/retain/mod.rs
pub mod chunker;
pub mod extractor;
pub mod graph_builder;
pub mod resolver;

pub use observation::{Consolidator, DefaultConsolidator};
pub use opinion_merger::{OpinionMerger, DefaultOpinionMerger};
```

## Trait Design

**Pattern:**
- All dynamic dependencies are traits
- `#[async_trait]` for async trait methods
- `MemoryStore`, `EmbeddingClient`, `LlmClient` — major interfaces
- Trait methods placed at behavior boundaries (storage, embedding, LLM calls)

**Example:**
```rust
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn insert_facts(&self, facts: &[Fact]) -> Result<Vec<FactId>>;
    async fn vector_search(
        &self,
        embedding: &[f32],
        bank: BankId,
        limit: usize,
        network_filter: Option<&[NetworkType]>,
    ) -> Result<Vec<ScoredFact>>;
}
```

## Struct Design (Decoupled Components)

**Dependency Injection:**
- Pass dependencies to constructors; don't create them internally
- Use trait objects `Box<dyn Trait>` for components that can vary
- Example from `DefaultRetainPipeline::new()`:
  ```rust
  pub fn new(
      chunker: Box<dyn Chunker>,
      extractor: Box<dyn FactExtractor>,
      resolver: Box<dyn EntityResolver>,
      graph_builder: Box<dyn GraphBuilder>,
      store: Box<dyn MemoryStore>,
      embeddings: Box<dyn EmbeddingClient>,
      llm: Arc<dyn LlmClient>,
      chunk_config: ChunkConfig,
      dedup_threshold: Option<f32>,
  ) -> Self
  ```

**Ownership model:**
- `Arc<dyn Trait>` for shared, immutable dependencies
- `Box<dyn Trait>` for owned, unique components
- Owned values preferred over references when practical (avoids lifetime complexity)

## Serialization

**Framework:** `serde` with `serde_json`

**Pattern:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact { ... }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactType {
    World,
    Experience,
}
```

**Attributes:**
- `#[serde(rename_all = "snake_case")]` on enums for API consistency
- `#[serde(default)]` on optional fields in responses (handles missing keys gracefully)
- No custom serialization logic unless necessary; lean on derive
- Temporary fields: `#[serde(skip)]` to exclude from serialization

## Type System Patterns

**Newtypes for clarity:**
- All IDs are newtypes: `FactId(Ulid)`, `BankId(Ulid)`, `EntityId(Ulid)`, `SourceId(Ulid)`, `TurnId(Ulid)`
- Prevents accidental ID swapping between types
- Implemented via macro `define_id!()` in `src/types/id.rs`

**Enums for domain models:**
```rust
pub enum NetworkType { World, Experience, Observation, Opinion }
pub enum FactType { World, Experience }
pub enum RetrievalSource { Semantic, Keyword, Graph, Temporal }
```

**Type-safe configuration:**
```rust
pub struct ChunkConfig {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
    pub preserve_turns: bool,
}
```

## Database/SQL

**ORM:** `sqlx` (compile-time checked queries)

**Transaction pattern:**
```rust
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn begin(&self) -> Result<Box<dyn TransactionHandle>>;
}

#[async_trait]
pub trait TransactionHandle: MemoryStore + Send {
    async fn commit(self: Box<Self>) -> Result<()>;
}
```

## Async/Concurrency

**Runtime:** `tokio` with `#[tokio::main]`, `#[tokio::test]`

**Pattern:**
- Functions that do I/O are `async`
- Use `.await` at call site
- Trait methods: `#[async_trait]` for async trait bounds
- Avoid nested async when possible; flat futures are cleaner

**Shared state:**
- `Arc<dyn Trait>` for shared dependencies (immutable)
- `Arc<Mutex<T>>` only when mutation is needed (prefer immutable design)

## Format Strings

**Style:**
- Use inline variable capture: `"{name} v{version}"` not `"{} v{}"` with separate args
- Bind variables first if not in scope: `let code = self.code; format!("[{code}]")`
- Single values may use positional for brevity: `format!("error: {}", code)`

**Examples:**
```rust
let name = "elephant";
let version = "0.1.0";
println!("{name} v{version}");

let code = self.code;
let message = &self.message;
write!(f, "[{code}] {message}")
```

---

*Convention analysis: 2026-03-15*
