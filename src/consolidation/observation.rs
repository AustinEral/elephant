//! Observation consolidator — merges raw facts into entity-level observations.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::llm::{complete_structured, LlmClient};
use crate::storage::MemoryStore;
use crate::types::id::{BankId, EntityId, FactId};
use crate::types::llm::{CompletionRequest, Message};
use crate::types::{ConsolidationReport, Fact, FactFilter, FactType, NetworkType};

/// Consolidates raw facts into entity-level observations.
#[async_trait]
pub trait Consolidator: Send + Sync {
    /// Process new facts since `since` and create/update observations.
    async fn consolidate(
        &self,
        bank_id: BankId,
        since: DateTime<Utc>,
    ) -> Result<ConsolidationReport>;
}

/// Default implementation using LLM for synthesis.
pub struct DefaultConsolidator {
    store: Arc<dyn MemoryStore>,
    llm: Arc<dyn LlmClient>,
    embeddings: Arc<dyn EmbeddingClient>,
}

impl DefaultConsolidator {
    /// Create a new consolidator.
    pub fn new(
        store: Arc<dyn MemoryStore>,
        llm: Arc<dyn LlmClient>,
        embeddings: Arc<dyn EmbeddingClient>,
    ) -> Self {
        Self {
            store,
            llm,
            embeddings,
        }
    }
}

#[derive(Deserialize)]
struct SynthesizeResponse {
    observation: String,
    confidence: f32,
}

#[derive(Deserialize)]
struct ConsolidateResponse {
    action: String,
    observation: Option<String>,
    confidence: Option<f32>,
}

const SYNTHESIZE_PROMPT: &str = include_str!("../../prompts/synthesize_observation.txt");
const CONSOLIDATE_PROMPT: &str = include_str!("../../prompts/consolidate_observation.txt");

#[async_trait]
impl Consolidator for DefaultConsolidator {
    async fn consolidate(
        &self,
        bank_id: BankId,
        since: DateTime<Utc>,
    ) -> Result<ConsolidationReport> {
        let mut report = ConsolidationReport::default();

        // 1. Fetch new facts since `since`
        let new_facts = self
            .store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    created_since: Some(since),
                    ..Default::default()
                },
            )
            .await?;

        if new_facts.is_empty() {
            return Ok(report);
        }

        // 2. Collect unique entity IDs from new facts
        let entity_ids: HashSet<EntityId> =
            new_facts.iter().flat_map(|f| &f.entity_ids).copied().collect();

        // 3. For each entity with new facts
        for entity_id in entity_ids {
            // Get facts for this entity from the new batch
            let entity_new_facts: Vec<&Fact> = new_facts
                .iter()
                .filter(|f| f.entity_ids.contains(&entity_id))
                .collect();

            if entity_new_facts.is_empty() {
                continue;
            }

            // Get existing observations for this entity
            let existing_observations = self
                .store
                .get_facts_by_bank(
                    bank_id,
                    FactFilter {
                        network: Some(vec![NetworkType::Observation]),
                        entity_ids: Some(vec![entity_id]),
                        ..Default::default()
                    },
                )
                .await?;

            // Resolve entity name for the prompt
            let entity_name = format!("entity:{entity_id}");

            if existing_observations.is_empty() {
                // No existing observation → synthesize new one
                let facts_text = entity_new_facts
                    .iter()
                    .enumerate()
                    .map(|(i, f)| format!("{}. {}", i + 1, f.content))
                    .collect::<Vec<_>>()
                    .join("\n");

                let prompt = SYNTHESIZE_PROMPT
                    .replace("{entity_name}", &entity_name)
                    .replace("{facts}", &facts_text);

                let request = CompletionRequest {
                    model: String::new(),
                    messages: vec![Message {
                        role: "user".into(),
                        content: prompt,
                    }],
                    max_tokens: Some(1024),
                    temperature: Some(0.3),
                    system: None,
                };

                let resp: SynthesizeResponse =
                    complete_structured(self.llm.as_ref(), request).await?;

                // Embed the observation
                let emb = self.embeddings.embed(&[&resp.observation]).await?;

                let now = Utc::now();
                let observation = Fact {
                    id: FactId::new(),
                    bank_id,
                    content: resp.observation,
                    fact_type: FactType::World,
                    network: NetworkType::Observation,
                    entity_ids: vec![entity_id],
                    temporal_range: None,
                    embedding: emb.into_iter().next(),
                    confidence: Some(resp.confidence),
                    evidence_ids: entity_new_facts.iter().map(|f| f.id).collect(),
                    source_turn_id: None,
                    created_at: now,
                    updated_at: now,
                };

                self.store.insert_facts(&[observation]).await?;
                report.observations_created += 1;
            } else {
                // Existing observation → decide: unchanged / updated / contradicted
                let existing = &existing_observations[0];

                let new_facts_text = entity_new_facts
                    .iter()
                    .enumerate()
                    .map(|(i, f)| format!("{}. {}", i + 1, f.content))
                    .collect::<Vec<_>>()
                    .join("\n");

                let prompt = CONSOLIDATE_PROMPT
                    .replace("{entity_name}", &entity_name)
                    .replace("{existing_observation}", &existing.content)
                    .replace("{new_facts}", &new_facts_text);

                let request = CompletionRequest {
                    model: String::new(),
                    messages: vec![Message {
                        role: "user".into(),
                        content: prompt,
                    }],
                    max_tokens: Some(1024),
                    temperature: Some(0.3),
                    system: None,
                };

                let resp: ConsolidateResponse =
                    complete_structured(self.llm.as_ref(), request).await?;

                match resp.action.as_str() {
                    "unchanged" => {
                        report.observations_unchanged += 1;
                    }
                    "updated" | "contradicted" => {
                        let new_text = resp
                            .observation
                            .unwrap_or_else(|| existing.content.clone());
                        let new_confidence = resp.confidence.unwrap_or_else(|| {
                            if resp.action == "contradicted" {
                                existing.confidence.unwrap_or(0.5) * 0.7
                            } else {
                                existing.confidence.unwrap_or(0.5)
                            }
                        });

                        // Re-embed updated content
                        let emb = self.embeddings.embed(&[&new_text]).await?;

                        // Combine evidence IDs
                        let mut evidence_ids = existing.evidence_ids.clone();
                        for f in &entity_new_facts {
                            if !evidence_ids.contains(&f.id) {
                                evidence_ids.push(f.id);
                            }
                        }

                        let mut updated = existing.clone();
                        updated.content = new_text;
                        updated.confidence = Some(new_confidence);
                        updated.embedding = emb.into_iter().next();
                        updated.evidence_ids = evidence_ids;
                        updated.updated_at = Utc::now();

                        self.store.update_fact(&updated).await?;
                        report.observations_updated += 1;
                    }
                    _ => {
                        report.observations_unchanged += 1;
                    }
                }
            }
        }

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::mock::MockEmbeddings;
    use crate::llm::mock::MockLlmClient;
    use crate::storage::mock::MockMemoryStore;

    fn setup() -> (Arc<MockMemoryStore>, Arc<MockLlmClient>, Arc<MockEmbeddings>) {
        (
            Arc::new(MockMemoryStore::new()),
            Arc::new(MockLlmClient::new()),
            Arc::new(MockEmbeddings::new(384)),
        )
    }

    fn make_fact(bank_id: BankId, entity_id: EntityId, content: &str) -> Fact {
        let now = Utc::now();
        Fact {
            id: FactId::new(),
            bank_id,
            content: content.into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_ids: vec![entity_id],
            temporal_range: None,
            embedding: Some(vec![0.1; 384]),
            confidence: None,
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now,
            updated_at: now,
        }
    }

    #[tokio::test]
    async fn new_facts_create_observation() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();
        let entity_id = EntityId::new();

        // Insert a fact
        let fact = make_fact(bank_id, entity_id, "Rust uses ownership for memory safety");
        store.insert_facts(&[fact]).await.unwrap();

        // Queue LLM response for synthesize
        llm.push_response(
            r#"{"observation": "Rust employs an ownership system for memory safety.", "confidence": 0.9}"#,
        );

        let consolidator = DefaultConsolidator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let since = Utc::now() - chrono::Duration::hours(1);
        let report = consolidator.consolidate(bank_id, since).await.unwrap();

        assert_eq!(report.observations_created, 1);
        assert_eq!(report.observations_updated, 0);
        assert_eq!(report.observations_unchanged, 0);

        // Verify the observation was stored
        let observations = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Observation]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(observations.len(), 1);
        assert_eq!(
            observations[0].content,
            "Rust employs an ownership system for memory safety."
        );
        assert_eq!(observations[0].entity_ids, vec![entity_id]);
        assert!(observations[0].confidence.unwrap() > 0.8);
    }

    #[tokio::test]
    async fn consistent_update_existing_observation() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();
        let entity_id = EntityId::new();

        // Insert existing observation
        let now = Utc::now();
        let existing_obs = Fact {
            id: FactId::new(),
            bank_id,
            content: "Rust uses ownership.".into(),
            fact_type: FactType::World,
            network: NetworkType::Observation,
            entity_ids: vec![entity_id],
            temporal_range: None,
            embedding: Some(vec![0.1; 384]),
            confidence: Some(0.8),
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now - chrono::Duration::hours(2),
            updated_at: now - chrono::Duration::hours(2),
        };
        store.insert_facts(std::slice::from_ref(&existing_obs)).await.unwrap();

        // Insert a new fact
        let fact = make_fact(bank_id, entity_id, "Rust's borrow checker enforces ownership rules");
        store.insert_facts(&[fact]).await.unwrap();

        // Queue LLM response for consolidate
        llm.push_response(
            r#"{"action": "updated", "observation": "Rust uses an ownership system enforced by the borrow checker.", "confidence": 0.85}"#,
        );

        let consolidator = DefaultConsolidator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let since = now - chrono::Duration::hours(1);
        let report = consolidator.consolidate(bank_id, since).await.unwrap();

        assert_eq!(report.observations_updated, 1);

        // Verify the observation was updated
        let observations = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Observation]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(observations.len(), 1);
        assert!(observations[0].content.contains("borrow checker"));
    }

    #[tokio::test]
    async fn contradiction_lowers_confidence() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();
        let entity_id = EntityId::new();

        let now = Utc::now();
        let existing_obs = Fact {
            id: FactId::new(),
            bank_id,
            content: "Python is the fastest language.".into(),
            fact_type: FactType::World,
            network: NetworkType::Observation,
            entity_ids: vec![entity_id],
            temporal_range: None,
            embedding: Some(vec![0.1; 384]),
            confidence: Some(0.8),
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now - chrono::Duration::hours(2),
            updated_at: now - chrono::Duration::hours(2),
        };
        store.insert_facts(&[existing_obs]).await.unwrap();

        let fact = make_fact(bank_id, entity_id, "C++ is significantly faster than Python");
        store.insert_facts(&[fact]).await.unwrap();

        llm.push_response(
            r#"{"action": "contradicted", "observation": "Python is not the fastest language; C++ is significantly faster.", "confidence": 0.5}"#,
        );

        let consolidator = DefaultConsolidator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let since = now - chrono::Duration::hours(1);
        let report = consolidator.consolidate(bank_id, since).await.unwrap();

        assert_eq!(report.observations_updated, 1);

        let observations = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Observation]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(observations[0].confidence.unwrap() <= 0.6);
    }

    #[tokio::test]
    async fn idempotent_no_new_facts() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        let consolidator = DefaultConsolidator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let since = Utc::now();
        let report = consolidator.consolidate(bank_id, since).await.unwrap();

        assert_eq!(report.observations_created, 0);
        assert_eq!(report.observations_updated, 0);
        assert_eq!(report.observations_unchanged, 0);
        assert_eq!(llm.remaining(), 0);
    }

    #[tokio::test]
    async fn multiple_facts_single_observation() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();
        let entity_id = EntityId::new();

        let f1 = make_fact(bank_id, entity_id, "Postgres supports JSON");
        let f2 = make_fact(bank_id, entity_id, "Postgres has good full-text search");
        store.insert_facts(&[f1, f2]).await.unwrap();

        llm.push_response(
            r#"{"observation": "PostgreSQL supports JSON storage and full-text search.", "confidence": 0.85}"#,
        );

        let consolidator = DefaultConsolidator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let since = Utc::now() - chrono::Duration::hours(1);
        let report = consolidator.consolidate(bank_id, since).await.unwrap();

        assert_eq!(report.observations_created, 1);

        let observations = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Observation]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].evidence_ids.len(), 2);
    }
}
