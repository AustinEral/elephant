//! Axum HTTP server exposing the memory engine API.

pub mod error;
pub mod handlers;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;

use crate::consolidation::{Consolidator, MentalModelGenerator, OpinionMerger};
use crate::recall::RecallPipeline;
use crate::reflect::ReflectPipeline;
use crate::retain::RetainPipeline;
use crate::storage::MemoryStore;

/// Shared application state holding all pipeline instances.
#[derive(Clone)]
pub struct AppState {
    /// The retain pipeline.
    pub retain: Arc<dyn RetainPipeline>,
    /// The recall pipeline.
    pub recall: Arc<dyn RecallPipeline>,
    /// The reflect pipeline.
    pub reflect: Arc<dyn ReflectPipeline>,
    /// The observation consolidator.
    pub consolidator: Arc<dyn Consolidator>,
    /// The opinion merger.
    pub opinion_merger: Arc<dyn OpinionMerger>,
    /// The mental model generator.
    pub model_generator: Arc<dyn MentalModelGenerator>,
    /// The backing memory store.
    pub store: Arc<dyn MemoryStore>,
}

/// Build the Axum router with all routes.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/banks", post(handlers::create_bank))
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
        .route(
            "/v1/banks/{id}/generate-models",
            post(handlers::generate_models),
        )
        .route(
            "/v1/banks/{id}/mental-models",
            get(handlers::list_mental_models),
        )
        .layer(CorsLayer::permissive())
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidation::{Consolidator, MentalModelGenerator, OpinionMerger};
    use crate::error::Result;
    use crate::recall::RecallPipeline;
    use crate::reflect::ReflectPipeline;
    use crate::retain::RetainPipeline;
    use crate::storage::mock::MockMemoryStore;
    use crate::types::*;

    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use chrono::Utc;
    use serde_json::{json, Value};
    use tower::util::ServiceExt;

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

    struct MockRecallPipeline;

    #[async_trait]
    impl RecallPipeline for MockRecallPipeline {
        async fn recall(&self, _query: &RecallQuery) -> Result<RecallResult> {
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
            })
        }
    }

    struct MockConsolidator;

    #[async_trait]
    impl Consolidator for MockConsolidator {
        async fn consolidate(
            &self,
            _bank_id: BankId,
            _since: chrono::DateTime<Utc>,
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

    struct MockModelGenerator;

    #[async_trait]
    impl MentalModelGenerator for MockModelGenerator {
        async fn generate(&self, _bank_id: BankId) -> Result<MentalModelReport> {
            Ok(MentalModelReport::default())
        }
    }

    fn test_app() -> (Router, Arc<MockMemoryStore>) {
        let store = Arc::new(MockMemoryStore::new());
        let state = AppState {
            retain: Arc::new(MockRetainPipeline {
                store: store.clone(),
            }),
            recall: Arc::new(MockRecallPipeline),
            reflect: Arc::new(MockReflectPipeline),
            consolidator: Arc::new(MockConsolidator),
            opinion_merger: Arc::new(MockOpinionMerger),
            model_generator: Arc::new(MockModelGenerator),
            store: store.clone(),
        };
        (router(state), store)
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
        Request::builder()
            .uri(uri)
            .body(Body::empty())
            .unwrap()
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
    async fn retain_and_recall_roundtrip() {
        let (app, store) = test_app();

        // Create a bank first
        let bank = MemoryBank {
            id: BankId::new(),
            name: "test".into(),
            mission: "test".into(),
            directives: vec![],
            disposition: Disposition::default(),
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

        let app2 = router(AppState {
            retain: Arc::new(MockRetainPipeline {
                store: store.clone(),
            }),
            recall: Arc::new(MockRecallPipeline),
            reflect: Arc::new(MockReflectPipeline),
            consolidator: Arc::new(MockConsolidator),
            opinion_merger: Arc::new(MockOpinionMerger),
            model_generator: Arc::new(MockModelGenerator),
            store: store.clone(),
        });

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
    async fn list_entities_after_retain() {
        let (app, store) = test_app();

        let bank = MemoryBank {
            id: BankId::new(),
            name: "test".into(),
            mission: "test".into(),
            directives: vec![],
            disposition: Disposition::default(),
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
        };
        store.create_bank(&bank).await.unwrap();
        let bank_id = bank.id.to_string();

        // Build a fresh app since oneshot consumes the router
        let make_app = || {
            router(AppState {
                retain: Arc::new(MockRetainPipeline {
                    store: store.clone(),
                }),
                recall: Arc::new(MockRecallPipeline),
                reflect: Arc::new(MockReflectPipeline),
                consolidator: Arc::new(MockConsolidator),
                opinion_merger: Arc::new(MockOpinionMerger),
                model_generator: Arc::new(MockModelGenerator),
                store: store.clone(),
            })
        };

        // Consolidate
        let req = json_request(
            "POST",
            &format!("/v1/banks/{bank_id}/consolidate"),
            json!({ "since": "2020-01-01T00:00:00Z" }),
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

        // Generate models
        let req = json_request(
            "POST",
            &format!("/v1/banks/{bank_id}/generate-models"),
            json!({}),
        );
        let resp = make_app().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
