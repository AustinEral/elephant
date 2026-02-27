//! Real integration tests exercising retain→recall→reflect with actual LLM and embedding providers.
//!
//! All tests are `#[ignore]` — run with:
//!   cargo test --test real_integration_tests local -- --ignored
//!   cargo test --test real_integration_tests openai -- --ignored
//!
//! Keys are loaded from `.env` automatically.

use std::sync::Arc;

use chrono::Utc;
use sqlx::PgPool;
use testcontainers::core::ContainerPort;
use testcontainers::runners::AsyncRunner;
use testcontainers::GenericImage;
use testcontainers_modules::testcontainers::ImageExt;

use elephant::consolidation::{DefaultConsolidator, DefaultMentalModelGenerator, DefaultOpinionMerger};
use elephant::embedding::{self, EmbeddingClient, EmbeddingConfig, EmbeddingProvider};
use elephant::llm::anthropic::AnthropicClient;
use elephant::llm::LlmClient;
use elephant::recall::budget::EstimateTokenizer;
use elephant::recall::graph::{GraphRetriever, GraphRetrieverConfig};
use elephant::recall::keyword::KeywordRetriever;
use elephant::recall::reranker::NoOpReranker;
use elephant::recall::semantic::SemanticRetriever;
use elephant::recall::temporal::TemporalRetriever;
use elephant::recall::DefaultRecallPipeline;
use elephant::reflect::hierarchy::DefaultHierarchyAssembler;
use elephant::reflect::opinion::DefaultOpinionManager;
use elephant::reflect::DefaultReflectPipeline;
use elephant::retain::chunker::SimpleChunker;
use elephant::retain::extractor::LlmFactExtractor;
use elephant::retain::graph_builder::{DefaultGraphBuilder, GraphConfig};
use elephant::retain::resolver::LayeredEntityResolver;
use elephant::retain::DefaultRetainPipeline;
use elephant::server::{router, AppState};
use elephant::storage::pg::PgMemoryStore;
use elephant::types::*;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use serde_json::{json, Value};
use tower::util::ServiceExt;

// ---------------------------------------------------------------------------
// Config helpers — read from .env with defaults matching main.rs
// ---------------------------------------------------------------------------

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("{key} must be set in .env"))
}

fn llm_api_key() -> String { env("LLM_API_KEY") }
fn llm_model() -> String { env("LLM_MODEL") }
fn embedding_model_path() -> String { env("EMBEDDING_MODEL_PATH") }
fn embedding_api_key() -> String { env("EMBEDDING_API_KEY") }
fn embedding_api_model() -> String { env("EMBEDDING_API_MODEL") }
fn embedding_api_dims() -> usize {
    env("EMBEDDING_API_DIMS").parse().expect("EMBEDDING_API_DIMS must be a number")
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

struct RealTestHarness {
    pool: PgPool,
    store: Arc<PgMemoryStore>,
    embeddings: Arc<dyn EmbeddingClient>,
    llm: Arc<dyn LlmClient>,
    _container: testcontainers::ContainerAsync<GenericImage>,
}

impl RealTestHarness {
    async fn setup(emb_config: EmbeddingConfig) -> Self {
        let _ = dotenvy::dotenv();

        let container = GenericImage::new("pgvector/pgvector", "pg16")
            .with_exposed_port(ContainerPort::Tcp(5432))
            .with_wait_for(testcontainers::core::WaitFor::message_on_stderr(
                "database system is ready to accept connections",
            ))
            .with_env_var("POSTGRES_DB", "test")
            .with_env_var("POSTGRES_USER", "test")
            .with_env_var("POSTGRES_PASSWORD", "test")
            .start()
            .await
            .expect("failed to start postgres");

        let port = container
            .get_host_port_ipv4(5432)
            .await
            .expect("failed to get port");
        let url = format!("postgres://test:test@127.0.0.1:{port}/test");

        let pool = loop {
            match PgPool::connect(&url).await {
                Ok(p) => break p,
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(200)).await,
            }
        };

        let store = Arc::new(PgMemoryStore::new(pool.clone()));
        store.migrate().await.expect("migration failed");

        let llm: Arc<dyn LlmClient> =
            Arc::new(AnthropicClient::new(llm_api_key(), llm_model()));

        let embeddings: Arc<dyn EmbeddingClient> =
            Arc::from(embedding::build_client(&emb_config).expect("failed to build embedding client"));

        Self { pool, store, embeddings, llm, _container: container }
    }

    fn app(&self) -> Router {
        let _ = dotenvy::dotenv();
        let api_key = llm_api_key();
        let model = llm_model();

        let make_llm = || -> Arc<dyn LlmClient> {
            Arc::new(AnthropicClient::new(api_key.clone(), model.clone()))
        };
        let make_emb = || -> Box<dyn EmbeddingClient> {
            let dims = self.embeddings.dimensions();
            let model_name = self.embeddings.model_name().to_string();
            if model_name.contains("bge") {
                Box::new(
                    elephant::embedding::local::LocalEmbeddings::new(
                        std::path::Path::new(&embedding_model_path()),
                    )
                    .expect("local embeddings"),
                )
            } else {
                Box::new(elephant::embedding::openai::OpenAiEmbeddings::new(
                    embedding_api_key(),
                    model_name,
                    dims,
                ))
            }
        };

        let retain = Arc::new(DefaultRetainPipeline::new(
            Box::new(SimpleChunker),
            Box::new(LlmFactExtractor::new(make_llm())),
            Box::new(LayeredEntityResolver::new(
                Box::new(PgMemoryStore::new(self.pool.clone())),
                make_emb(),
                make_llm(),
            )),
            Box::new(DefaultGraphBuilder::new(
                Box::new(PgMemoryStore::new(self.pool.clone())),
                make_llm(),
                GraphConfig::default(),
            )),
            Box::new(PgMemoryStore::new(self.pool.clone())),
            make_emb(),
            make_llm(),
            ChunkConfig {
                max_tokens: 512,
                overlap_tokens: 64,
                preserve_turns: true,
            },
        ));

        let store_arc: Arc<dyn elephant::MemoryStore> =
            Arc::new(PgMemoryStore::new(self.pool.clone()));
        let embed_arc: Arc<dyn EmbeddingClient> = self.embeddings.clone();

        let recall = Arc::new(DefaultRecallPipeline::new(
            Box::new(SemanticRetriever::new(store_arc.clone(), embed_arc.clone(), 20)),
            Box::new(KeywordRetriever::new(store_arc.clone(), 20)),
            Box::new(GraphRetriever::new(store_arc.clone(), embed_arc.clone(), GraphRetrieverConfig::default())),
            Box::new(TemporalRetriever::new(store_arc.clone())),
            Box::new(NoOpReranker),
            Box::new(EstimateTokenizer),
            60.0,
            50,
        ));

        let reflect = Arc::new(DefaultReflectPipeline::new(
            Box::new(DefaultHierarchyAssembler::new(recall.clone())),
            Box::new(DefaultOpinionManager::new(store_arc.clone(), embed_arc.clone())),
            self.llm.clone(),
            store_arc.clone(),
        ));

        let consolidator = Arc::new(DefaultConsolidator::new(store_arc.clone(), self.llm.clone(), self.embeddings.clone()));
        let opinion_merger = Arc::new(DefaultOpinionMerger::new(store_arc.clone(), self.llm.clone(), self.embeddings.clone()));
        let model_generator = Arc::new(DefaultMentalModelGenerator::new(store_arc.clone(), self.llm.clone(), self.embeddings.clone()));

        let state = AppState {
            retain,
            recall,
            reflect,
            consolidator,
            opinion_merger,
            model_generator,
            store: self.store.clone(),
            embeddings: self.embeddings.clone(),
        };

        router(state)
    }
}

// ---------------------------------------------------------------------------
// Embedding configs
// ---------------------------------------------------------------------------

fn local_embedding_config() -> EmbeddingConfig {
    let _ = dotenvy::dotenv();
    EmbeddingConfig {
        provider: EmbeddingProvider::Local,
        model_path: Some(embedding_model_path()),
        api_key: None,
        model: None,
        dimensions: None,
    }
}

fn openai_embedding_config() -> EmbeddingConfig {
    let _ = dotenvy::dotenv();
    EmbeddingConfig {
        provider: EmbeddingProvider::OpenAi,
        model_path: None,
        api_key: Some(embedding_api_key()),
        model: Some(embedding_api_model()),
        dimensions: Some(embedding_api_dims()),
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

fn json_request(method: &str, uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn get_request(uri: &str) -> Request<Body> {
    Request::builder().uri(uri).body(Body::empty()).unwrap()
}

async fn json_body(resp: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn create_bank(app: &Router, name: &str) -> String {
    let req = json_request("POST", "/v1/banks", json!({ "name": name, "mission": "real integration test" }));
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    json_body(resp).await["id"].as_str().unwrap().to_string()
}

async fn do_retain(app: &Router, bank_id: &str, content: &str) -> Value {
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_id}/retain"),
        json!({ "bank_id": "00000000000000000000000000", "content": content, "timestamp": Utc::now().to_rfc3339() }),
    );
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = json_body(resp).await;
    assert_eq!(status, StatusCode::OK, "retain failed: {body}");
    body
}

async fn do_recall(app: &Router, bank_id: &str, query: &str) -> Value {
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_id}/recall"),
        json!({ "bank_id": "00000000000000000000000000", "query": query, "budget_tokens": 4000 }),
    );
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = json_body(resp).await;
    assert_eq!(status, StatusCode::OK, "recall failed: {body}");
    body
}

async fn do_reflect(app: &Router, bank_id: &str, question: &str) -> Value {
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_id}/reflect"),
        json!({ "bank_id": "00000000000000000000000000", "question": question, "budget_tokens": 4000 }),
    );
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = json_body(resp).await;
    assert_eq!(status, StatusCode::OK, "reflect failed: {body}");
    body
}

// ===========================================================================
// Local embedding tests (ANTHROPIC_API_KEY + local ONNX model)
// ===========================================================================

#[tokio::test]
#[ignore = "requires LLM_API_KEY and local ONNX model"]
async fn local_retain_extracts_facts() {
    let h = RealTestHarness::setup(local_embedding_config()).await;
    let app = h.app();
    let bank_id = create_bank(&app, "local-retain").await;

    let body = do_retain(
        &app, &bank_id,
        "Rust was first released in 2015 by Mozilla. It focuses on memory safety without garbage collection.",
    ).await;

    let facts_stored = body["facts_stored"].as_u64().unwrap();
    eprintln!("local_retain_extracts_facts: facts_stored={facts_stored}");
    eprintln!("  response: {body}");
    assert!(facts_stored > 0, "should extract at least one fact");
}

#[tokio::test]
#[ignore = "requires LLM_API_KEY and local ONNX model"]
async fn local_retain_then_recall() {
    let h = RealTestHarness::setup(local_embedding_config()).await;
    let app = h.app();
    let bank_id = create_bank(&app, "local-recall").await;

    let retain_body = do_retain(
        &app, &bank_id,
        "PostgreSQL is an advanced open-source relational database. It supports JSON, full-text search, and extensibility through custom types.",
    ).await;
    eprintln!("local_retain_then_recall: retain={retain_body}");

    let recall_body = do_recall(&app, &bank_id, "database features").await;
    let facts = recall_body["facts"].as_array().unwrap();
    eprintln!("local_retain_then_recall: recalled {} facts", facts.len());
    for f in facts {
        eprintln!("  - {}", f["fact"]["content"].as_str().unwrap_or("?"));
    }
    assert!(!facts.is_empty(), "recall should return at least one fact");
}

#[tokio::test]
#[ignore = "requires LLM_API_KEY and local ONNX model"]
async fn local_retain_then_reflect() {
    let h = RealTestHarness::setup(local_embedding_config()).await;
    let app = h.app();
    let bank_id = create_bank(&app, "local-reflect").await;

    do_retain(&app, &bank_id, "The borrow checker in Rust prevents data races at compile time.").await;
    do_retain(&app, &bank_id, "Rust's ownership model ensures memory safety without a garbage collector.").await;

    let reflect_body = do_reflect(&app, &bank_id, "How does Rust ensure memory safety?").await;
    let response = reflect_body["response"].as_str().unwrap_or("");
    let confidence = reflect_body["confidence"].as_f64().unwrap_or(0.0);
    eprintln!("local_retain_then_reflect: confidence={confidence}");
    eprintln!("  response: {response}");

    assert!(!response.is_empty(), "reflect should produce a response");
    assert!(confidence > 0.0, "confidence should be positive");
}

#[tokio::test]
#[ignore = "requires LLM_API_KEY and local ONNX model"]
async fn local_full_roundtrip() {
    let h = RealTestHarness::setup(local_embedding_config()).await;
    let app = h.app();
    let bank_id = create_bank(&app, "local-roundtrip").await;

    do_retain(&app, &bank_id, "Python was created by Guido van Rossum and first released in 1991.").await;
    do_retain(&app, &bank_id, "Python uses dynamic typing and garbage collection.").await;
    do_retain(&app, &bank_id, "Python's standard library is often described as 'batteries included'.").await;

    let recall_body = do_recall(&app, &bank_id, "Python language features").await;
    let facts = recall_body["facts"].as_array().unwrap();
    eprintln!("local_full_roundtrip: recalled {} facts", facts.len());
    assert!(!facts.is_empty(), "should recall facts about Python");

    let reflect_body = do_reflect(&app, &bank_id, "What are the key characteristics of Python?").await;
    let response = reflect_body["response"].as_str().unwrap_or("");
    eprintln!("local_full_roundtrip reflect: {response}");
    assert!(!response.is_empty(), "reflect should produce a response");

    let req = json_request("POST", &format!("/v1/banks/{bank_id}/consolidate"), json!({ "since": "2020-01-01T00:00:00Z" }));
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let consolidate_body = json_body(resp).await;
    eprintln!("local_full_roundtrip consolidate: {consolidate_body}");
}

#[tokio::test]
#[ignore = "requires LLM_API_KEY and local ONNX model"]
async fn local_entity_resolution() {
    let h = RealTestHarness::setup(local_embedding_config()).await;
    let app = h.app();
    let bank_id = create_bank(&app, "local-entities").await;

    do_retain(&app, &bank_id, "PostgreSQL is a powerful open-source database system.").await;
    do_retain(&app, &bank_id, "Postgres supports advanced indexing including GiST and GIN indexes.").await;

    let req = get_request(&format!("/v1/banks/{bank_id}/entities"));
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let entities = json_body(resp).await;
    let entities = entities.as_array().unwrap();

    eprintln!("local_entity_resolution: {} entities", entities.len());
    for e in entities {
        eprintln!("  - {} ({})", e["canonical_name"].as_str().unwrap_or("?"), e["id"].as_str().unwrap_or("?"));
    }

    let pg_entities: Vec<_> = entities.iter().filter(|e| {
        e["canonical_name"].as_str().unwrap_or("").to_lowercase().contains("postgres")
    }).collect();

    assert_eq!(pg_entities.len(), 1, "Postgres/PostgreSQL should resolve to exactly one entity, got {pg_entities:?}");
}

// ===========================================================================
// OpenAI embedding tests (ANTHROPIC_API_KEY + OPENAI_API_KEY)
// ===========================================================================

#[tokio::test]
#[ignore = "requires LLM_API_KEY and OPENAI_API_KEY"]
async fn openai_retain_extracts_facts() {
    let h = RealTestHarness::setup(openai_embedding_config()).await;
    let app = h.app();
    let bank_id = create_bank(&app, "openai-retain").await;

    let body = do_retain(
        &app, &bank_id,
        "Rust was first released in 2015 by Mozilla. It focuses on memory safety without garbage collection.",
    ).await;

    let facts_stored = body["facts_stored"].as_u64().unwrap();
    eprintln!("openai_retain_extracts_facts: facts_stored={facts_stored}");
    eprintln!("  response: {body}");
    assert!(facts_stored > 0, "should extract at least one fact");
}

#[tokio::test]
#[ignore = "requires LLM_API_KEY and OPENAI_API_KEY"]
async fn openai_retain_then_recall() {
    let h = RealTestHarness::setup(openai_embedding_config()).await;
    let app = h.app();
    let bank_id = create_bank(&app, "openai-recall").await;

    let retain_body = do_retain(
        &app, &bank_id,
        "PostgreSQL is an advanced open-source relational database. It supports JSON, full-text search, and extensibility through custom types.",
    ).await;
    eprintln!("openai_retain_then_recall: retain={retain_body}");

    let recall_body = do_recall(&app, &bank_id, "database features").await;
    let facts = recall_body["facts"].as_array().unwrap();
    eprintln!("openai_retain_then_recall: recalled {} facts", facts.len());
    for f in facts {
        eprintln!("  - {}", f["fact"]["content"].as_str().unwrap_or("?"));
    }
    assert!(!facts.is_empty(), "recall should return at least one fact");
}

#[tokio::test]
#[ignore = "requires LLM_API_KEY and OPENAI_API_KEY"]
async fn openai_retain_then_reflect() {
    let h = RealTestHarness::setup(openai_embedding_config()).await;
    let app = h.app();
    let bank_id = create_bank(&app, "openai-reflect").await;

    do_retain(&app, &bank_id, "The borrow checker in Rust prevents data races at compile time.").await;
    do_retain(&app, &bank_id, "Rust's ownership model ensures memory safety without a garbage collector.").await;

    let reflect_body = do_reflect(&app, &bank_id, "How does Rust ensure memory safety?").await;
    let response = reflect_body["response"].as_str().unwrap_or("");
    let confidence = reflect_body["confidence"].as_f64().unwrap_or(0.0);
    eprintln!("openai_retain_then_reflect: confidence={confidence}");
    eprintln!("  response: {response}");

    assert!(!response.is_empty(), "reflect should produce a response");
    assert!(confidence > 0.0, "confidence should be positive");
}

#[tokio::test]
#[ignore = "requires LLM_API_KEY and OPENAI_API_KEY"]
async fn openai_full_roundtrip() {
    let h = RealTestHarness::setup(openai_embedding_config()).await;
    let app = h.app();
    let bank_id = create_bank(&app, "openai-roundtrip").await;

    do_retain(&app, &bank_id, "Python was created by Guido van Rossum and first released in 1991.").await;
    do_retain(&app, &bank_id, "Python uses dynamic typing and garbage collection.").await;
    do_retain(&app, &bank_id, "Python's standard library is often described as 'batteries included'.").await;

    let recall_body = do_recall(&app, &bank_id, "Python language features").await;
    let facts = recall_body["facts"].as_array().unwrap();
    eprintln!("openai_full_roundtrip: recalled {} facts", facts.len());
    assert!(!facts.is_empty(), "should recall facts about Python");

    let reflect_body = do_reflect(&app, &bank_id, "What are the key characteristics of Python?").await;
    let response = reflect_body["response"].as_str().unwrap_or("");
    eprintln!("openai_full_roundtrip reflect: {response}");
    assert!(!response.is_empty(), "reflect should produce a response");

    let req = json_request("POST", &format!("/v1/banks/{bank_id}/consolidate"), json!({ "since": "2020-01-01T00:00:00Z" }));
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let consolidate_body = json_body(resp).await;
    eprintln!("openai_full_roundtrip consolidate: {consolidate_body}");
}

#[tokio::test]
#[ignore = "requires LLM_API_KEY and OPENAI_API_KEY"]
async fn openai_entity_resolution() {
    let h = RealTestHarness::setup(openai_embedding_config()).await;
    let app = h.app();
    let bank_id = create_bank(&app, "openai-entities").await;

    do_retain(&app, &bank_id, "PostgreSQL is a powerful open-source database system.").await;
    do_retain(&app, &bank_id, "Postgres supports advanced indexing including GiST and GIN indexes.").await;

    let req = get_request(&format!("/v1/banks/{bank_id}/entities"));
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let entities = json_body(resp).await;
    let entities = entities.as_array().unwrap();

    eprintln!("openai_entity_resolution: {} entities", entities.len());
    for e in entities {
        eprintln!("  - {} ({})", e["canonical_name"].as_str().unwrap_or("?"), e["id"].as_str().unwrap_or("?"));
    }

    let pg_entities: Vec<_> = entities.iter().filter(|e| {
        e["canonical_name"].as_str().unwrap_or("").to_lowercase().contains("postgres")
    }).collect();

    assert_eq!(pg_entities.len(), 1, "Postgres/PostgreSQL should resolve to exactly one entity, got {pg_entities:?}");
}
