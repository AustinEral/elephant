//! Axum HTTP server exposing the memory engine API.

pub mod error;
pub mod handlers;

use axum::Router;
use axum::routing::{get, post};
use tower_http::cors::CorsLayer;

pub use crate::app::{
    AppBackgroundConsolidationInfo as ServerBackgroundConsolidationInfo,
    AppConsolidationInfo as ServerConsolidationRuntimeInfo, AppHandle, AppInfo as ServerInfo,
    AppModelsInfo as ServerModelsInfo, AppReflectInfo as ServerReflectInfo,
    AppRetrievalInfo as ServerRetrievalInfo,
};

/// Build the Axum router with all routes.
pub fn router(app: AppHandle) -> Router {
    Router::new()
        .route("/v1/info", get(handlers::server_info))
        .route(
            "/v1/banks",
            get(handlers::list_banks).post(handlers::create_bank),
        )
        .route("/v1/banks/{id}", get(handlers::get_bank))
        .route("/v1/banks/{id}/retain", post(handlers::retain))
        .route("/v1/banks/{id}/recall", post(handlers::recall))
        .route("/v1/banks/{id}/reflect", post(handlers::reflect))
        .route("/v1/banks/{id}/entities", get(handlers::list_entities))
        .route(
            "/v1/banks/{id}/entities/{eid}/facts",
            get(handlers::entity_facts),
        )
        .route("/v1/banks/{id}/consolidate", post(handlers::consolidate))
        .route(
            "/v1/banks/{id}/merge-opinions",
            post(handlers::merge_opinions),
        )
        .layer(CorsLayer::permissive())
        .with_state(app)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidation::{Consolidator, OpinionMerger};
    use crate::embedding::mock::MockEmbeddings;
    use crate::error::Result;
    use crate::recall::RecallPipeline;
    use crate::reflect::ReflectPipeline;
    use crate::retain::RetainPipeline;
    use crate::storage::{MemoryStore, mock::MockMemoryStore};
    use crate::types::*;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use chrono::Utc;
    use serde_json::{Value, json};
    use tower::util::ServiceExt;

    fn test_server_info() -> ServerInfo {
        ServerInfo {
            version: env!("CARGO_PKG_VERSION").into(),
            models: ServerModelsInfo {
                retain: "test".into(),
                reflect: "test".into(),
                embedding: "test".into(),
                reranker: "none".into(),
            },
            retrieval: ServerRetrievalInfo {
                retriever_limit: 20,
                max_facts: 50,
            },
            reflect: ServerReflectInfo {
                max_iterations: 8,
                max_tokens: None,
                source_lookup_enabled: true,
            },
            consolidation: ServerConsolidationRuntimeInfo {
                batch_size: 16,
                max_tokens: 2048,
                recall_budget: 1024,
            },
            server_consolidation: ServerBackgroundConsolidationInfo {
                enabled: true,
                min_facts: 32,
                cooldown_secs: 30,
                merge_opinions_after: false,
            },
        }
    }

    // --- Mock pipelines ---

    struct MockRetainPipeline {
        store: Arc<dyn MemoryStore>,
    }

    #[async_trait]
    impl RetainPipeline for MockRetainPipeline {
        async fn retain(&self, input: &RetainInput) -> Result<RetainOutput> {
            // Store a simple fact for each retain call
            let fact = Fact {
                id: FactId::new(),
                bank_id: input.bank_id,
                content: input.content.clone(),
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
            };
            let ids = self.store.insert_facts(&[fact]).await?;
            Ok(RetainOutput {
                fact_ids: ids,
                facts_stored: 1,
                new_entities: vec![],
                entities_resolved: 0,
                links_created: 0,
                opinions_reinforced: 0,
                opinions_weakened: 0,
            })
        }
    }

    struct MockRecallPipeline {
        captured: Option<Arc<Mutex<Option<RecallQuery>>>>,
    }

    #[async_trait]
    impl RecallPipeline for MockRecallPipeline {
        async fn recall(&self, query: &RecallQuery) -> Result<RecallResult> {
            if let Some(captured) = &self.captured {
                *captured.lock().unwrap() = Some(query.clone());
            }
            Ok(RecallResult {
                facts: vec![],
                total_tokens: 0,
            })
        }
    }

    struct MockReflectPipeline;

    #[async_trait]
    impl ReflectPipeline for MockReflectPipeline {
        async fn reflect(&self, _query: &ReflectQuery) -> Result<ReflectResult> {
            Ok(ReflectResult {
                response: "Mock reflection".into(),
                sources: vec![],
                new_opinions: vec![],
                confidence: 0.5,
                retrieved_context: Default::default(),
                retrieved_sources: Default::default(),
                trace: Default::default(),
                final_done: None,
            })
        }
    }

    struct MockConsolidator;

    #[async_trait]
    impl Consolidator for MockConsolidator {
        async fn consolidate_with_progress(
            &self,
            _bank_id: BankId,
            _progress: Option<
                tokio::sync::mpsc::UnboundedSender<crate::consolidation::ConsolidationProgress>,
            >,
        ) -> Result<ConsolidationReport> {
            Ok(ConsolidationReport::default())
        }
    }

    struct MockOpinionMerger;

    #[async_trait]
    impl OpinionMerger for MockOpinionMerger {
        async fn merge(&self, _bank_id: BankId) -> Result<OpinionMergeReport> {
            Ok(OpinionMergeReport::default())
        }
    }

    fn test_app() -> (Router, Arc<MockMemoryStore>) {
        let store = Arc::new(MockMemoryStore::new());
        let app = AppHandle::from_parts(
            test_server_info(),
            Arc::new(MockRetainPipeline {
                store: store.clone(),
            }),
            Arc::new(MockRecallPipeline { captured: None }),
            Arc::new(MockReflectPipeline),
            Arc::new(MockConsolidator),
            Arc::new(MockOpinionMerger),
            store.clone(),
            Arc::new(MockEmbeddings::new(384)),
        );
        (router(app), store)
    }

    fn test_app_with_recall_capture() -> (
        Router,
        Arc<MockMemoryStore>,
        Arc<Mutex<Option<RecallQuery>>>,
    ) {
        let store = Arc::new(MockMemoryStore::new());
        let captured = Arc::new(Mutex::new(None));
        let app = AppHandle::from_parts(
            test_server_info(),
            Arc::new(MockRetainPipeline {
                store: store.clone(),
            }),
            Arc::new(MockRecallPipeline {
                captured: Some(captured.clone()),
            }),
            Arc::new(MockReflectPipeline),
            Arc::new(MockConsolidator),
            Arc::new(MockOpinionMerger),
            store.clone(),
            Arc::new(MockEmbeddings::new(384)),
        );
        (router(app), store, captured)
    }

    async fn json_body(resp: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

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

    #[tokio::test]
    async fn create_bank_returns_200() {
        let (app, _store) = test_app();

        let req = json_request(
            "POST",
            "/v1/banks",
            json!({
                "name": "test bank",
                "mission": "remember everything"
            }),
        );

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = json_body(resp).await;
        assert_eq!(body["name"], "test bank");
        assert!(body["id"].is_string());
    }

    #[tokio::test]
    async fn server_info_returns_nested_runtime_configuration() {
        let (app, _store) = test_app();

        let resp = app.oneshot(get_request("/v1/info")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = json_body(resp).await;
        assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(body["models"]["retain"], "test");
        assert_eq!(body["models"]["reranker"], "none");
        assert_eq!(body["retrieval"]["retriever_limit"], 20);
        assert_eq!(body["retrieval"]["max_facts"], 50);
        assert_eq!(body["reflect"]["max_iterations"], 8);
        assert_eq!(body["reflect"]["source_lookup_enabled"], true);
        assert_eq!(body["consolidation"]["batch_size"], 16);
        assert_eq!(body["server_consolidation"]["enabled"], true);
        assert_eq!(body["server_consolidation"]["min_facts"], 32);
        assert_eq!(body["server_consolidation"]["cooldown_secs"], 30);
    }

    #[tokio::test]
    async fn retain_and_recall_roundtrip() {
        let (app, store) = test_app();

        // Create a bank first
        let bank = MemoryBank {
            id: BankId::new(),
            name: "test".into(),
            mission: "test".into(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: String::new(),
            embedding_dimensions: 0,
        };
        store.create_bank(&bank).await.unwrap();
        let bank_id = bank.id.to_string();

        // Retain
        let req = json_request(
            "POST",
            &format!("/v1/banks/{bank_id}/retain"),
            json!({
                "bank_id": "00000000000000000000000000",
                "content": "Rust is a systems programming language",
                "timestamp": Utc::now().to_rfc3339()
            }),
        );

        let app2 = router(AppHandle::from_parts(
            test_server_info(),
            Arc::new(MockRetainPipeline {
                store: store.clone(),
            }),
            Arc::new(MockRecallPipeline { captured: None }),
            Arc::new(MockReflectPipeline),
            Arc::new(MockConsolidator),
            Arc::new(MockOpinionMerger),
            store.clone(),
            Arc::new(MockEmbeddings::new(384)),
        ));

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = json_body(resp).await;
        assert_eq!(body["facts_stored"], 1);

        // Recall
        let req = json_request(
            "POST",
            &format!("/v1/banks/{bank_id}/recall"),
            json!({
                "bank_id": "00000000000000000000000000",
                "query": "Rust",
                "budget_tokens": 1000
            }),
        );

        let resp = app2.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn reflect_returns_200() {
        let (app, store) = test_app();

        let bank = MemoryBank {
            id: BankId::new(),
            name: "test".into(),
            mission: "test".into(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: String::new(),
            embedding_dimensions: 0,
        };
        store.create_bank(&bank).await.unwrap();
        let bank_id = bank.id.to_string();

        let req = json_request(
            "POST",
            &format!("/v1/banks/{bank_id}/reflect"),
            json!({
                "bank_id": "00000000000000000000000000",
                "question": "What is Rust?",
                "budget_tokens": 2000
            }),
        );

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = json_body(resp).await;
        assert_eq!(body["response"], "Mock reflection");
    }

    #[tokio::test]
    async fn recall_allows_optional_budget_and_max_facts() {
        let (app, store, captured) = test_app_with_recall_capture();

        let bank = MemoryBank {
            id: BankId::new(),
            name: "test".into(),
            mission: "test".into(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: String::new(),
            embedding_dimensions: 0,
        };
        store.create_bank(&bank).await.unwrap();
        let bank_id = bank.id.to_string();

        let req = json_request(
            "POST",
            &format!("/v1/banks/{bank_id}/recall"),
            json!({
                "bank_id": "00000000000000000000000000",
                "query": "Rust",
                "max_facts": 7
            }),
        );

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let recall_query = captured
            .lock()
            .unwrap()
            .clone()
            .expect("recall query should be captured");
        assert_eq!(recall_query.bank_id.to_string(), bank_id);
        assert_eq!(recall_query.query, "Rust");
        assert_eq!(recall_query.budget_tokens, None);
        assert_eq!(recall_query.max_facts, Some(7));
    }

    #[tokio::test]
    async fn list_entities_after_retain() {
        let (app, store) = test_app();

        let bank = MemoryBank {
            id: BankId::new(),
            name: "test".into(),
            mission: "test".into(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: String::new(),
            embedding_dimensions: 0,
        };
        store.create_bank(&bank).await.unwrap();
        let bank_id = bank.id.to_string();

        // Add an entity directly
        let entity = Entity {
            id: EntityId::new(),
            canonical_name: "Rust".into(),
            aliases: vec!["rust-lang".into()],
            entity_type: EntityType::Concept,
            bank_id: bank.id,
        };
        store.upsert_entity(&entity).await.unwrap();

        let req = get_request(&format!("/v1/banks/{bank_id}/entities"));
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = json_body(resp).await;
        let entities = body.as_array().unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0]["canonical_name"], "Rust");
    }

    #[tokio::test]
    async fn entity_facts_returns_facts() {
        let (app, store) = test_app();

        let bank = MemoryBank {
            id: BankId::new(),
            name: "test".into(),
            mission: "test".into(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: String::new(),
            embedding_dimensions: 0,
        };
        store.create_bank(&bank).await.unwrap();

        let entity_id = EntityId::new();
        let entity = Entity {
            id: entity_id,
            canonical_name: "Rust".into(),
            aliases: vec![],
            entity_type: EntityType::Concept,
            bank_id: bank.id,
        };
        store.upsert_entity(&entity).await.unwrap();

        let fact = Fact {
            id: FactId::new(),
            bank_id: bank.id,
            content: "Rust is fast".into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_ids: vec![entity_id],
            temporal_range: None,
            embedding: None,
            confidence: None,
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            consolidated_at: None,
        };
        store.insert_facts(&[fact]).await.unwrap();

        let req = get_request(&format!(
            "/v1/banks/{}/entities/{}/facts",
            bank.id, entity_id
        ));
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = json_body(resp).await;
        let facts = body.as_array().unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0]["content"], "Rust is fast");
    }

    #[tokio::test]
    async fn invalid_bank_id_returns_400() {
        let (app, _store) = test_app();

        let req = get_request("/v1/banks/not-a-valid-id");
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn bank_not_found_returns_404() {
        let (app, _store) = test_app();

        // Valid ULID format but doesn't exist
        let fake_id = BankId::new();
        let req = get_request(&format!("/v1/banks/{fake_id}"));
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn consolidation_triggers_return_200() {
        let (app, store) = test_app();

        let bank = MemoryBank {
            id: BankId::new(),
            name: "test".into(),
            mission: "test".into(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: String::new(),
            embedding_dimensions: 0,
        };
        store.create_bank(&bank).await.unwrap();
        let bank_id = bank.id.to_string();

        // Build a fresh app since oneshot consumes the router
        let make_app = || {
            router(AppHandle::from_parts(
                test_server_info(),
                Arc::new(MockRetainPipeline {
                    store: store.clone(),
                }),
                Arc::new(MockRecallPipeline { captured: None }),
                Arc::new(MockReflectPipeline),
                Arc::new(MockConsolidator),
                Arc::new(MockOpinionMerger),
                store.clone(),
                Arc::new(MockEmbeddings::new(384)),
            ))
        };

        // Consolidate
        let req = json_request(
            "POST",
            &format!("/v1/banks/{bank_id}/consolidate"),
            json!({}),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Merge opinions
        let req = json_request(
            "POST",
            &format!("/v1/banks/{bank_id}/merge-opinions"),
            json!({}),
        );
        let resp = make_app().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
