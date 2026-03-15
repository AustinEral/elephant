# Testing Patterns

**Analysis Date:** 2026-03-15

## Test Framework

**Runner:**
- `tokio` test harness for async tests (`#[tokio::test]`)
- Standard Rust `#[test]` for synchronous tests
- `datatest-stable` for data-driven tests (see `tests/evals_extract.rs`, `tests/evals_validate.rs`)
- Custom harness disabled on eval tests: `[[test]] harness = false`

**Assertion Library:**
- `assert!()`, `assert_eq!()`, `assert_ne!()` — standard macros
- No external assertion framework (std is sufficient)

**Run Commands:**
```bash
cargo test                    # Run all tests
cargo test -- --test-threads=1  # Serial execution if needed
cargo test -- --nocapture    # Print output from passing tests
cargo test --lib            # Unit tests only
cargo test --test '*'       # Integration tests only
```

**Coverage:**
- Not enforced
- No coverage tool configured
- Manual selection: important business logic and edge cases

## Test File Organization

**Location:**
- Unit tests: co-located with source in same file via `#[cfg(test)] mod tests`
- Integration tests: `tests/` directory with separate binaries
- Both patterns used in this codebase

**Naming:**
- Test module: `mod tests` (standard)
- Test function: `#[test]` or `#[tokio::test]`
- Function names describe scenario: `fn cosine_similarity_identical_vectors()`, `fn bank_crud()`, `fn vector_search_ranking()`
- Pattern: `<function>_<scenario>_<expected>` or `<operation>_<condition>`

**Structure:**
```
src/util.rs
  └── #[cfg(test)]
      └── mod tests
          └── #[test] fn identical_vectors()

tests/storage_tests.rs         // Integration test file
  ├── async fn setup_store()   // Test fixture
  ├── fn make_bank()           // Factory helper
  ├── fn make_fact()           // Factory helper
  └── #[tokio::test]
      async fn bank_crud()     // Test function
```

## Test Structure

**Suite Organization:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Fixtures
    fn default_config() -> ChunkConfig {
        ChunkConfig {
            max_tokens: 1000,
            overlap_tokens: 100,
            preserve_turns: false,
        }
    }

    // Tests
    #[test]
    fn parse_valid_returns_config() {
        let result = parse_config(valid_input);
        assert_eq!(result.max_tokens, 1000);
    }
}
```

**Patterns:**
- Setup fixtures are plain functions: `fn make_bank()`, `fn default_config()`
- Integration test setup is async: `async fn setup_store() -> (Store, Container)`
- No teardown needed in most cases (Postgres container auto-drops)
- Tests in same file as source use full module paths: `use super::*;`
- Integration tests in `/tests/` declare dependencies: `use elephant::*;`

**Async test pattern (integration):**
```rust
#[tokio::test]
async fn bank_crud() {
    let (store, _container) = setup_store().await;  // Fixture
    let bank = make_bank();                         // Factory

    let id = store.create_bank(&bank).await.unwrap();
    assert_eq!(id, bank.id);

    let fetched = store.get_bank(bank.id).await.unwrap();
    assert_eq!(fetched.name, bank.name);
}
```

## Mocking

**Framework:** Custom in-crate implementations

**Mock implementations:**
- `src/embedding/mock.rs` — `MockEmbeddings` with configurable dimensions
- `src/llm/mock.rs` — `MockLlmClient` with response queue
- `src/storage/mock.rs` — `MockMemoryStore` for unit tests (in-memory)

**Pattern for building mocks:**
```rust
// Create a mock client
let embeddings = Arc::new(MockEmbeddings::new(384));  // Set dimensions

// Push responses for the mock to return
llm.push_response(CompletionResponse {
    content: "extracted facts...".into(),
});

// Use as normal
let results = embeddings.embed(&["query"]).await?;
```

**What to Mock:**
- External services: LLM (Anthropic/OpenAI), embedding models
- Storage for unit tests (MockMemoryStore)
- In integration tests: mock LLM/embeddings, real Postgres

**What NOT to Mock:**
- Database queries (use testcontainers Postgres instead)
- Core business logic (test the real thing)
- Trait implementations that are part of the system

## Fixtures and Factories

**Test Data:**
```rust
fn make_bank() -> MemoryBank {
    MemoryBank {
        id: BankId::new(),
        name: "test bank".into(),
        mission: "testing".into(),
        directives: vec!["be accurate".into()],
        disposition: Disposition::new(2, 4, 3, 0.7).unwrap(),
        embedding_model: String::new(),
        embedding_dimensions: 0,
    }
}

fn make_fact(bank_id: BankId) -> Fact {
    Fact {
        id: FactId::new(),
        bank_id,
        content: "Rust is a systems programming language".into(),
        fact_type: FactType::World,
        network: NetworkType::World,
        entity_ids: vec![],
        temporal_range: None,
        embedding: None,
        confidence: None,
        evidence_ids: vec![],
        source_turn_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        consolidated_at: None,
    }
}
```

**Location:**
- Simple factories: defined in test module as plain functions
- Complex setup: `TestHarness` struct with `async fn setup()` method
- See `tests/api_integration_tests.rs::TestHarness` for full harness pattern

**Harness example:**
```rust
struct TestHarness {
    pool: PgPool,
    llm: Arc<MockLlmClient>,
    embeddings: Arc<MockEmbeddings>,
    store: Arc<PgMemoryStore>,
    _container: testcontainers::ContainerAsync<GenericImage>,
}

impl TestHarness {
    async fn setup() -> Self {
        // Start Postgres container
        // Create mock LLM/embeddings
        // Wire full app state
    }

    fn app(&self) -> Router {
        // Wire full Axum router with all pipelines
    }
}
```

## Coverage

**Requirements:** None enforced

**Manual approach:**
- Write tests for business logic and edge cases
- Avoid testing what Rust's type system guarantees:
  - Type correctness (compiler ensures)
  - Exhaustive match arms (compiler ensures)
  - Ownership rules (compiler ensures)
  - Null safety (no null in Rust)

**Do test:**
- Algorithmic correctness: `cosine_similarity()` with known vectors
- Edge cases: empty inputs, boundary conditions
- Error paths: invalid configurations, missing data
- Integration: how components work together

**Anti-pattern (don't test):**
```rust
// Bad: testing what compiler guarantees
#[test]
fn session_id_is_copy() {
    let id = SessionId::new();
    let copy = id;
    assert_eq!(id, copy);
}
```

**Good pattern:**
```rust
#[test]
fn cosine_similarity_identical_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
}

#[test]
fn vector_search_ranking() {
    // Set up facts with known embeddings
    // Query for similar fact
    // Verify correct ordering
}
```

## Test Types

**Unit Tests:**
- Scope: Single function or small module
- Approach: Mock external dependencies, test behavior in isolation
- Location: `#[cfg(test)] mod tests` within source file
- Example: `src/util.rs` tests cosine_similarity with hand-crafted vectors
- Real dependencies: Only when they're part of the behavior being tested

**Integration Tests:**
- Scope: Multiple components working together through public API
- Approach: Real Postgres (testcontainers), mock LLM/embeddings
- Location: `tests/*.rs` files
- Example: `tests/api_integration_tests.rs` tests full retain→recall pipeline
- Docker container setup: Auto-starts and stops with test

**Data-Driven Tests:**
- Framework: `datatest-stable` harness
- Location: `tests/evals_extract.rs`, `tests/evals_validate.rs`
- Pattern: Load test cases from JSON/YAML, run harness over each
- No standard Rust test harness: `[[test]] harness = false` in Cargo.toml

**E2E Tests:**
- Not used — no real LLM API calls in automated tests
- Manual testing with real LLMs uses `#[ignore]` attribute

## Common Patterns

**Async Testing:**
```rust
#[tokio::test]
async fn operation_returns_result() {
    let value = async_operation().await;
    assert_eq!(value, expected);
}

// With error handling
#[tokio::test]
async fn operation_fails_gracefully() {
    let result = async_operation().await;
    assert!(result.is_err());
}
```

**Error Testing:**
```rust
#[test]
fn invalid_input_returns_error() {
    let result = parse_config(invalid_input);
    assert!(result.is_err());

    // Check error type if needed
    match result {
        Err(Error::InvalidDisposition(msg)) => assert!(msg.contains("expected")),
        _ => panic!("wrong error"),
    }
}
```

**Containerized Database (testcontainers):**
```rust
async fn setup_store() -> (PgMemoryStore, testcontainers::ContainerAsync<GenericImage>) {
    let container = GenericImage::new("pgvector/pgvector", "pg16")
        .with_exposed_port(ContainerPort::Tcp(5432))
        .with_wait_for(testcontainers::core::WaitFor::message_on_stderr(
            "database system is ready to accept connections",
        ))
        .start()
        .await
        .expect("failed to start postgres");

    let port = container
        .get_host_port_ipv4(5432)
        .await
        .expect("failed to get port");
    let url = format!("postgres://test:test@127.0.0.1:{port}/test");

    // Retry connection — container may need a moment
    let pool = loop {
        match PgPool::connect(&url).await {
            Ok(p) => break p,
            Err(_) => tokio::time::sleep(Duration::from_millis(200)).await,
        }
    };

    let store = PgMemoryStore::new(pool);
    store.migrate().await.expect("migration failed");
    (store, container)
}
```

**Ignored Tests (require real API keys):**
```rust
#[tokio::test]
#[ignore]
async fn real_llm_api_call() {
    // Calls real Anthropic/OpenAI API
    // Requires LLM_API_KEY in .env
}
```

Run with: `cargo test -- --ignored --test-threads=1`

## Test Isolation

**Principles:**
- Each test is independent (no shared state)
- Database: testcontainers creates fresh container per test suite
- Mock clients: new instance per test or test harness
- No global state or static mocks

**Example from integration tests:**
```rust
struct TestHarness { ... }

#[tokio::test]
async fn test_one() {
    let harness = TestHarness::setup().await;  // Fresh container
    // Test logic
}

#[tokio::test]
async fn test_two() {
    let harness = TestHarness::setup().await;  // Separate container
    // Test logic
}
```

## Real API Tests

**Files:**
- `tests/real_integration_tests.rs` — Calls real LLM/embedding APIs
- `tests/prompt_eval.rs` — Extracts facts from real conversations
- `tests/evals_extract.rs`, `tests/evals_validate.rs` — Data-driven evaluation

**When to use real APIs:**
- Evaluation and benchmark runs (intentional, explicit)
- Development/debugging of prompt changes
- Not in CI by default — mark with `#[ignore]`

**Setup:**
```rust
// Requires env vars:
// LLM_PROVIDER: "anthropic" or "openai"
// LLM_API_KEY: API key for chosen provider
// LLM_MODEL: Model name (e.g., "claude-3-5-sonnet")
// EMBEDDING_MODEL: Model path for embeddings
```

**Run:**
```bash
cargo test --test prompt_eval -- --ignored --nocapture
```

## Test Organization Best Practices

**File layout:**
```
src/
  util.rs
    ├── pub fn cosine_similarity()
    └── #[cfg(test)]
        └── mod tests
            ├── #[test] fn identical_vectors()
            ├── #[test] fn orthogonal_vectors()
            └── #[test] fn zero_vector()

tests/
  storage_tests.rs
    ├── fn setup_store()
    ├── fn make_bank()
    ├── fn make_fact()
    ├── #[tokio::test] async fn bank_crud()
    └── #[tokio::test] async fn fact_insert_get()

  api_integration_tests.rs
    ├── struct TestHarness { ... }
    ├── #[tokio::test] async fn full_pipeline()
    └── ...
```

**Naming clarity:**
- Descriptive test names that explain what they test
- Use scenario-based names: `test_<operation>_<condition>_<expectation>`
- Examples: `vector_search_ranking()`, `bank_crud()`, `fact_insert_get()`

## Common Test Utilities

**Create IDs:**
```rust
let bank_id = BankId::new();      // Random ULID
let fact_id = FactId::new();
let entity_id = EntityId::new();
```

**Create timestamps:**
```rust
use chrono::Utc;
let now = Utc::now();
```

**Mock embeddings (384-dim by default):**
```rust
let embeddings = Arc::new(MockEmbeddings::new(384));
let vecs = embeddings.embed(&["text"]).await?;  // Returns random vectors
```

**Mock LLM:**
```rust
let llm = Arc::new(MockLlmClient::new());
llm.push_response(CompletionResponse {
    content: "structured JSON response".into(),
});
let resp = llm.complete(req).await?;
```

---

*Testing analysis: 2026-03-15*
