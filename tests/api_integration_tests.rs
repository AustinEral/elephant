//! Integration tests exercising real pipeline wiring through HTTP endpoints.
//!
//! Uses testcontainers Postgres + mock LLM/embeddings (no real API calls).

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use chrono::Utc;
use serde_json::{Value, json};
use sqlx::PgPool;
use testcontainers::GenericImage;
use testcontainers::core::ContainerPort;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::testcontainers::ImageExt;
use tower::util::ServiceExt;

use elephant::consolidation::{DefaultConsolidator, DefaultOpinionMerger};
use elephant::embedding::mock::MockEmbeddings;
use elephant::llm::LlmClient;
use elephant::llm::mock::MockLlmClient;
use elephant::recall::DefaultRecallPipeline;
use elephant::recall::budget::EstimateTokenizer;
use elephant::recall::graph::{GraphRetriever, GraphRetrieverConfig};
use elephant::recall::keyword::KeywordRetriever;
use elephant::recall::reranker::{self, RerankerConfig, RerankerProvider};
use elephant::recall::semantic::SemanticRetriever;
use elephant::recall::temporal::TemporalRetriever;
use elephant::reflect::DefaultReflectPipeline;
use elephant::retain::DefaultRetainPipeline;
use elephant::retain::chunker::SimpleChunker;
use elephant::retain::extractor::{ExtractionConfig, LlmFactExtractor};
use elephant::retain::graph_builder::{DefaultGraphBuilder, GraphConfig};
use elephant::retain::resolver::LayeredEntityResolver;
use elephant::storage::pg::PgMemoryStore;
use elephant::types::*;
use elephant::{
    AppHandle, ElephantRuntime, RuntimeInfo, RuntimePromptHashes, RuntimeTuning, ServerConfig,
    router,
};

const EMBED_DIMS: usize = 384;

struct TestHarness {
    pool: PgPool,
    llm: Arc<MockLlmClient>,
    embeddings: Arc<MockEmbeddings>,
    store: Arc<PgMemoryStore>,
    _container: testcontainers::ContainerAsync<GenericImage>,
}

impl TestHarness {
    async fn setup() -> Self {
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

        let llm = Arc::new(MockLlmClient::new());
        let embeddings = Arc::new(MockEmbeddings::new(EMBED_DIMS));

        Self {
            pool,
            llm,
            embeddings,
            store,
            _container: container,
        }
    }

    fn app(&self) -> Router {
        // Retain pipeline — each Arc<dyn>/Box<dyn> slot gets a fresh instance from the pool
        let retain_store = Box::new(PgMemoryStore::new(self.pool.clone()));
        let retain_embeddings = Box::new(MockEmbeddings::new(EMBED_DIMS));
        let retain_llm_resolver: Arc<dyn LlmClient> = Arc::new(MockLlmClient::new());
        let retain_llm_graph: Arc<dyn LlmClient> = Arc::new(MockLlmClient::new());

        // Wire the shared MockLlmClient for extraction (where we push responses)
        let extractor = Box::new(LlmFactExtractor::new(
            Arc::new(self.llm.as_ref().clone()),
            ExtractionConfig::default(),
        ));
        let resolver = Box::new(LayeredEntityResolver::new(
            Box::new(MockEmbeddings::new(EMBED_DIMS)),
            retain_llm_resolver,
        ));
        let graph_builder = Box::new(DefaultGraphBuilder::new(
            retain_llm_graph,
            GraphConfig {
                enable_causal: false,
                ..Default::default()
            },
        ));

        let retain = Arc::new(DefaultRetainPipeline::new(
            Box::new(SimpleChunker),
            extractor,
            resolver,
            graph_builder,
            retain_store,
            retain_embeddings,
            Arc::new(self.llm.as_ref().clone()),
            ChunkConfig {
                max_tokens: 512,
                overlap_tokens: 50,
                preserve_turns: false,
            },
            None, // no dedup in tests
        ));

        // Recall pipeline
        let store_arc: Arc<dyn elephant::MemoryStore> =
            Arc::new(PgMemoryStore::new(self.pool.clone()));
        let embed_arc: Arc<dyn elephant::EmbeddingClient> = self.embeddings.clone();

        let recall = Arc::new(DefaultRecallPipeline::new(
            Box::new(SemanticRetriever::new(
                store_arc.clone(),
                embed_arc.clone(),
                20,
            )),
            Box::new(KeywordRetriever::new(store_arc.clone(), 20)),
            Box::new(GraphRetriever::new(
                store_arc.clone(),
                embed_arc.clone(),
                GraphRetrieverConfig::default(),
            )),
            Box::new(TemporalRetriever::new(store_arc.clone())),
            reranker::build_reranker(&RerankerConfig {
                provider: RerankerProvider::None,
                model_path: None,
                max_seq_len: 512,
                api_key: None,
                api_url: None,
                api_model: None,
            })
            .expect("reranker"),
            Box::new(EstimateTokenizer),
            60.0,
            50,
            4_096,
        ));

        // Reflect pipeline
        let reflect = Arc::new(DefaultReflectPipeline::new(
            recall.clone(),
            self.llm.clone() as Arc<dyn elephant::LlmClient>,
            store_arc.clone(),
            8,
        ));

        // Consolidation
        let consolidator = Arc::new(DefaultConsolidator::new(
            store_arc.clone(),
            self.llm.clone() as Arc<dyn elephant::LlmClient>,
            self.embeddings.clone() as Arc<dyn elephant::EmbeddingClient>,
            recall.clone(),
            elephant::consolidation::ConsolidationConfig::default(),
        ));
        let opinion_merger = Arc::new(DefaultOpinionMerger::new(
            store_arc.clone(),
            self.llm.clone() as Arc<dyn elephant::LlmClient>,
            self.embeddings.clone() as Arc<dyn elephant::EmbeddingClient>,
        ));
        let runtime = ElephantRuntime {
            info: RuntimeInfo {
                retain_model: "test".into(),
                reflect_model: "test".into(),
                embedding_model: "test".into(),
                reranker_model: "none".into(),
                tuning: RuntimeTuning::default(),
                prompt_hashes: RuntimePromptHashes::default(),
            },
            retain,
            recall,
            reflect,
            consolidator,
            opinion_merger,
            store: self.store.clone(),
            embeddings: self.embeddings.clone(),
        };
        let app = AppHandle::new(&runtime, &ServerConfig::from_env().expect("server config"))
            .expect("app handle");

        router(app)
    }
}

// --- Helpers ---

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
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn extraction_response(content: &str, entity: &str) -> String {
    serde_json::to_string(&json!([{
        "content": content,
        "fact_type": "world",
        "network": "world",
        "entity_mentions": [entity],
        "temporal_range": null,
        "confidence": 0.95
    }]))
    .unwrap()
}

fn reflect_response(text: &str) -> String {
    serde_json::to_string(&json!({
        "response": text,
        "sources": [],
        "new_opinions": [],
        "confidence": 0.8
    }))
    .unwrap()
}

// --- Tests ---

#[tokio::test]
async fn create_bank_and_get() {
    let h = TestHarness::setup().await;

    let req = json_request(
        "POST",
        "/v1/banks",
        json!({
            "name": "integration test bank",
            "mission": "remember everything"
        }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = json_body(resp).await;
    assert_eq!(body["name"], "integration test bank");
    assert_eq!(body["mission"], "remember everything");
    let bank_id = body["id"].as_str().unwrap().to_string();

    // GET the same bank back
    let req = get_request(&format!("/v1/banks/{bank_id}"));
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = json_body(resp).await;
    assert_eq!(body["id"], bank_id);
    assert_eq!(body["name"], "integration test bank");
    assert_eq!(body["mission"], "remember everything");
}

#[tokio::test]
async fn retain_and_recall_roundtrip() {
    let h = TestHarness::setup().await;

    // Create bank
    let req = json_request(
        "POST",
        "/v1/banks",
        json!({ "name": "recall-test", "mission": "test recall" }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bank_id = json_body(resp).await["id"].as_str().unwrap().to_string();

    // Push extraction LLM response
    h.llm.push_response(extraction_response(
        "Rust is a systems programming language focused on safety",
        "Rust",
    ));

    // Retain
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_id}/retain"),
        json!({
            "bank_id": "00000000000000000000000000",
            "content": "Rust is a systems programming language focused on safety.",
            "timestamp": Utc::now().to_rfc3339()
        }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = json_body(resp).await;
    assert!(body["facts_stored"].as_u64().unwrap() >= 1);

    // Recall with keyword
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_id}/recall"),
        json!({
            "bank_id": "00000000000000000000000000",
            "query": "Rust programming",
            "budget_tokens": 2000
        }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = json_body(resp).await;
    let facts = body["facts"].as_array().unwrap();
    assert!(!facts.is_empty(), "recall should return at least one fact");

    // ScoredFact has nested { fact: { content: ... }, score: ... }
    let any_rust = facts.iter().any(|f| {
        let content = f["fact"]["content"].as_str().unwrap_or("");
        content.contains("Rust") || content.contains("safety")
    });
    assert!(any_rust, "recalled facts should be about Rust");
}

#[tokio::test]
async fn retain_creates_entities() {
    let h = TestHarness::setup().await;

    // Create bank
    let req = json_request(
        "POST",
        "/v1/banks",
        json!({ "name": "entity-test", "mission": "test entities" }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    let bank_id = json_body(resp).await["id"].as_str().unwrap().to_string();

    // Push extraction response mentioning "Rust" entity
    h.llm
        .push_response(extraction_response("Rust has a strong type system", "Rust"));

    // Retain
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_id}/retain"),
        json!({
            "bank_id": "00000000000000000000000000",
            "content": "Rust has a strong type system.",
            "timestamp": Utc::now().to_rfc3339()
        }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = json_body(resp).await;
    assert!(body["entities_resolved"].as_u64().unwrap() >= 1);

    // List entities
    let req = get_request(&format!("/v1/banks/{bank_id}/entities"));
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let entities = json_body(resp).await;
    let entities = entities.as_array().unwrap();
    assert!(!entities.is_empty(), "should have at least one entity");

    let rust_entity = entities.iter().find(|e| {
        e["canonical_name"].as_str().unwrap_or("").contains("Rust")
            || e["canonical_name"].as_str().unwrap_or("").contains("rust")
    });
    assert!(rust_entity.is_some(), "should have a Rust entity");

    // Get facts for the entity
    let entity_id = rust_entity.unwrap()["id"].as_str().unwrap();
    let req = get_request(&format!("/v1/banks/{bank_id}/entities/{entity_id}/facts"));
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let facts = json_body(resp).await;
    let facts = facts.as_array().unwrap();
    assert!(!facts.is_empty(), "entity should have associated facts");
}

#[tokio::test]
async fn reflect_with_context() {
    let h = TestHarness::setup().await;

    // Create bank
    let req = json_request(
        "POST",
        "/v1/banks",
        json!({ "name": "reflect-test", "mission": "test reflect" }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    let bank_id = json_body(resp).await["id"].as_str().unwrap().to_string();

    // Push extraction response
    h.llm.push_response(extraction_response(
        "Rust eliminates data races at compile time",
        "Rust",
    ));

    // Retain first
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_id}/retain"),
        json!({
            "bank_id": "00000000000000000000000000",
            "content": "Rust eliminates data races at compile time.",
            "timestamp": Utc::now().to_rfc3339()
        }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Push reflect LLM response
    h.llm.push_response(reflect_response(
        "Rust prevents data races through its ownership system and borrow checker.",
    ));

    // Reflect
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_id}/reflect"),
        json!({
            "bank_id": "00000000000000000000000000",
            "question": "How does Rust prevent data races?",
            "budget_tokens": 2000
        }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = json_body(resp).await;
    assert!(body["response"].as_str().unwrap().contains("Rust"));
    assert!(body["confidence"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn consolidation_empty_bank() {
    let h = TestHarness::setup().await;

    // Create bank
    let req = json_request(
        "POST",
        "/v1/banks",
        json!({ "name": "consolidation-test", "mission": "test consolidation" }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    let bank_id = json_body(resp).await["id"].as_str().unwrap().to_string();

    // Consolidate on empty bank
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_id}/consolidate"),
        json!({ "since": "2020-01-01T00:00:00Z" }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["observations_created"], 0);
    assert_eq!(body["observations_updated"], 0);

    // Merge opinions on empty bank
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_id}/merge-opinions"),
        json!({}),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["opinions_merged"], 0);
}

#[tokio::test]
async fn invalid_bank_id_returns_400() {
    let h = TestHarness::setup().await;

    let req = get_request("/v1/banks/not-valid");
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn bank_not_found_returns_404() {
    let h = TestHarness::setup().await;

    let fake_id = BankId::new();
    let req = get_request(&format!("/v1/banks/{fake_id}"));
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn bank_isolation() {
    let h = TestHarness::setup().await;

    // Create two banks
    let req = json_request(
        "POST",
        "/v1/banks",
        json!({ "name": "bank-a", "mission": "bank a" }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    let bank_a = json_body(resp).await["id"].as_str().unwrap().to_string();

    let req = json_request(
        "POST",
        "/v1/banks",
        json!({ "name": "bank-b", "mission": "bank b" }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    let bank_b = json_body(resp).await["id"].as_str().unwrap().to_string();

    // Push extraction responses for both retains
    h.llm.push_response(extraction_response(
        "Python is an interpreted language",
        "Python",
    ));
    h.llm.push_response(extraction_response(
        "Go has built-in concurrency with goroutines",
        "Go",
    ));

    // Retain into bank A
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_a}/retain"),
        json!({
            "bank_id": "00000000000000000000000000",
            "content": "Python is an interpreted language.",
            "timestamp": Utc::now().to_rfc3339()
        }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Retain into bank B
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_b}/retain"),
        json!({
            "bank_id": "00000000000000000000000000",
            "content": "Go has built-in concurrency with goroutines.",
            "timestamp": Utc::now().to_rfc3339()
        }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Recall from bank A — should only get Python facts
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_a}/recall"),
        json!({
            "bank_id": "00000000000000000000000000",
            "query": "programming language",
            "budget_tokens": 2000
        }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = json_body(resp).await;
    let facts = body["facts"].as_array().unwrap();
    let has_go = facts
        .iter()
        .any(|f| f["content"].as_str().unwrap_or("").contains("goroutines"));
    assert!(!has_go, "bank A should not contain bank B's facts about Go");

    // Recall from bank B — should only get Go facts
    let req = json_request(
        "POST",
        &format!("/v1/banks/{bank_b}/recall"),
        json!({
            "bank_id": "00000000000000000000000000",
            "query": "programming language",
            "budget_tokens": 2000
        }),
    );
    let resp = h.app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = json_body(resp).await;
    let facts = body["facts"].as_array().unwrap();
    let has_python = facts
        .iter()
        .any(|f| f["content"].as_str().unwrap_or("").contains("Python"));
    assert!(
        !has_python,
        "bank B should not contain bank A's facts about Python"
    );
}
