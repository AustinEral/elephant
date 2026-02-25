//! Observation consolidator — synthesizes entity-level observations from raw facts.
//!
//! Matches the paper's approach (§4.1.5): `o_e = SummarizeLLM(F_e)` where F_e is
//! ALL facts mentioning entity e. When new facts arrive, the observation is
//! recomputed from the full fact set — not incrementally updated.

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
///
/// Follows the paper's observation paradigm: for each entity with new facts,
/// re-summarize ALL facts about that entity into a single observation.
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
}

const SYNTHESIZE_PROMPT: &str = include_str!("../../prompts/synthesize_observation.txt");

#[async_trait]
impl Consolidator for DefaultConsolidator {
    async fn consolidate(
        &self,
        bank_id: BankId,
        since: DateTime<Utc>,
    ) -> Result<ConsolidationReport> {
        let mut report = ConsolidationReport::default();

        // 1. Fetch new World/Experience facts since `since`
        //    (only raw fact networks — not Observation/Opinion/MentalModel)
        let new_facts = self
            .store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::World, NetworkType::Experience]),
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

        // 3. Load all entities in the bank for name resolution
        let all_entities = self.store.list_entities(bank_id).await?;

        // 4. For each entity with new facts, re-summarize ALL facts (paper: o_e = SummarizeLLM(F_e))
        for entity_id in entity_ids {
            // Resolve entity canonical name for the prompt
            let entity_name = all_entities
                .iter()
                .find(|e| e.id == entity_id)
                .map(|e| e.canonical_name.as_str())
                .unwrap_or("unknown entity");

            // Get ALL World/Experience facts about this entity (not just new ones)
            let all_entity_facts = self
                .store
                .get_facts_by_bank(
                    bank_id,
                    FactFilter {
                        network: Some(vec![NetworkType::World, NetworkType::Experience]),
                        entity_ids: Some(vec![entity_id]),
                        ..Default::default()
                    },
                )
                .await?;

            if all_entity_facts.is_empty() {
                continue;
            }

            // Get existing observation for this entity
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

            // Synthesize observation from ALL facts
            let facts_text = all_entity_facts
                .iter()
                .enumerate()
                .map(|(i, f)| format!("{}. {}", i + 1, f.content))
                .collect::<Vec<_>>()
                .join("\n");

            let prompt = SYNTHESIZE_PROMPT
                .replace("{entity_name}", entity_name)
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

            let emb = self.embeddings.embed(&[&resp.observation]).await?;
            let evidence_ids: Vec<FactId> = all_entity_facts.iter().map(|f| f.id).collect();

            if let Some(existing) = existing_observations.into_iter().next() {
                // Update existing observation in place
                let mut updated = existing;
                updated.content = resp.observation;
                updated.confidence = None; // Paper: observations lack confidence
                updated.embedding = emb.into_iter().next();
                updated.evidence_ids = evidence_ids;
                updated.updated_at = Utc::now();

                self.store.update_fact(&updated).await?;
                report.observations_updated += 1;
            } else {
                // Create new observation
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
                    confidence: None, // Paper: observations lack confidence
                    evidence_ids,
                    source_turn_id: None,
                    created_at: now,
                    updated_at: now,
                };

                self.store.insert_facts(&[observation]).await?;
                report.observations_created += 1;
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
    use crate::types::Entity;
    use crate::types::entity::EntityType;

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

    async fn insert_entity(store: &MockMemoryStore, bank_id: BankId, entity_id: EntityId, name: &str) {
        store
            .upsert_entity(&Entity {
                id: entity_id,
                canonical_name: name.into(),
                aliases: vec![],
                entity_type: EntityType::Concept,
                bank_id,
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn new_facts_create_observation() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();
        let entity_id = EntityId::new();

        insert_entity(&store, bank_id, entity_id, "Rust").await;

        let fact = make_fact(bank_id, entity_id, "Rust uses ownership for memory safety");
        store.insert_facts(&[fact]).await.unwrap();

        llm.push_response(
            r#"{"observation": "Rust employs an ownership system for memory safety."}"#,
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
        // Paper: observations lack confidence
        assert!(observations[0].confidence.is_none());
    }

    #[tokio::test]
    async fn re_summarizes_all_facts_on_update() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();
        let entity_id = EntityId::new();

        insert_entity(&store, bank_id, entity_id, "Rust").await;

        // Insert old fact + existing observation
        let now = Utc::now();
        let old_fact = Fact {
            id: FactId::new(),
            bank_id,
            content: "Rust uses ownership.".into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_ids: vec![entity_id],
            temporal_range: None,
            embedding: Some(vec![0.1; 384]),
            confidence: None,
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now - chrono::Duration::hours(3),
            updated_at: now - chrono::Duration::hours(3),
        };
        let existing_obs = Fact {
            id: FactId::new(),
            bank_id,
            content: "Rust uses ownership.".into(),
            fact_type: FactType::World,
            network: NetworkType::Observation,
            entity_ids: vec![entity_id],
            temporal_range: None,
            embedding: Some(vec![0.1; 384]),
            confidence: None,
            evidence_ids: vec![old_fact.id],
            source_turn_id: None,
            created_at: now - chrono::Duration::hours(2),
            updated_at: now - chrono::Duration::hours(2),
        };
        store
            .insert_facts(&[old_fact.clone(), existing_obs])
            .await
            .unwrap();

        // Insert a new fact (within the `since` window)
        let new_fact = make_fact(bank_id, entity_id, "Rust's borrow checker enforces ownership rules");
        store.insert_facts(std::slice::from_ref(&new_fact)).await.unwrap();

        // LLM gets ALL facts (old + new) and produces a full re-summary
        llm.push_response(
            r#"{"observation": "Rust uses an ownership system enforced by the borrow checker for memory safety."}"#,
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
        assert_eq!(observations.len(), 1);
        assert!(observations[0].content.contains("borrow checker"));
        // evidence_ids should include BOTH the old and new fact
        assert_eq!(observations[0].evidence_ids.len(), 2);
        assert!(observations[0].evidence_ids.contains(&old_fact.id));
        assert!(observations[0].evidence_ids.contains(&new_fact.id));
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

        insert_entity(&store, bank_id, entity_id, "PostgreSQL").await;

        let f1 = make_fact(bank_id, entity_id, "Postgres supports JSON");
        let f2 = make_fact(bank_id, entity_id, "Postgres has good full-text search");
        store.insert_facts(&[f1, f2]).await.unwrap();

        llm.push_response(
            r#"{"observation": "PostgreSQL supports JSON storage and full-text search."}"#,
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

    #[tokio::test]
    async fn only_processes_world_and_experience_facts() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();
        let entity_id = EntityId::new();

        insert_entity(&store, bank_id, entity_id, "Rust").await;

        // Insert an Opinion fact (should NOT trigger consolidation)
        let now = Utc::now();
        let opinion = Fact {
            id: FactId::new(),
            bank_id,
            content: "Rust is the best language.".into(),
            fact_type: FactType::Experience,
            network: NetworkType::Opinion,
            entity_ids: vec![entity_id],
            temporal_range: None,
            embedding: Some(vec![0.1; 384]),
            confidence: Some(0.8),
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now,
            updated_at: now,
        };
        store.insert_facts(&[opinion]).await.unwrap();

        let consolidator = DefaultConsolidator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let since = now - chrono::Duration::hours(1);
        let report = consolidator.consolidate(bank_id, since).await.unwrap();

        // No observations created — the only new fact was an Opinion
        assert_eq!(report.observations_created, 0);
        assert_eq!(llm.remaining(), 0);
    }
}
