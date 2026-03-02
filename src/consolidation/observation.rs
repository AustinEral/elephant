//! Topic-scoped observation consolidator — produces multiple focused observations
//! from batches of raw facts, Vectorize-style.
//!
//! Instead of one monolithic observation per entity, facts are processed in batches.
//! For each batch the LLM decides whether to CREATE new or UPDATE existing observations,
//! keeping each observation focused on a single topic/facet.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;

use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::llm::{complete_structured, LlmClient};
use crate::storage::MemoryStore;
use crate::types::id::{BankId, FactId};
use crate::types::llm::{CompletionRequest, Message};
use crate::types::{ConsolidationReport, Fact, FactFilter, FactType, NetworkType, TemporalRange};

/// Consolidates raw facts into topic-scoped observations.
#[async_trait]
pub trait Consolidator: Send + Sync {
    /// Process unconsolidated facts and create/update observations.
    async fn consolidate(&self, bank_id: BankId) -> Result<ConsolidationReport>;
}

/// Default implementation using LLM for topic-scoped synthesis.
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
struct ConsolidateResponse {
    actions: Vec<ConsolidateAction>,
}

#[derive(Deserialize)]
struct ConsolidateAction {
    action: String,
    content: String,
    fact_indices: Vec<usize>,
    observation_id: Option<String>,
}

const CONSOLIDATE_PROMPT: &str = include_str!("../../prompts/consolidate_topics.txt");

fn batch_size() -> usize {
    std::env::var("CONSOLIDATION_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
}

fn max_tokens() -> usize {
    std::env::var("CONSOLIDATION_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096)
}

/// Merge temporal ranges from source facts into an existing range using LEAST(start)/GREATEST(end).
fn merge_temporal(existing: Option<&TemporalRange>, facts: &[&Fact]) -> Option<TemporalRange> {
    let mut start = existing.and_then(|r| r.start);
    let mut end = existing.and_then(|r| r.end);

    for f in facts {
        if let Some(ref tr) = f.temporal_range {
            start = match (start, tr.start) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            end = match (end, tr.end) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
        }
    }

    match (start, end) {
        (None, None) => None,
        _ => Some(TemporalRange { start, end }),
    }
}

#[async_trait]
impl Consolidator for DefaultConsolidator {
    async fn consolidate(&self, bank_id: BankId) -> Result<ConsolidationReport> {
        let mut report = ConsolidationReport::default();

        // 1. Fetch unconsolidated World/Experience facts
        let unconsolidated = self
            .store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::World, NetworkType::Experience]),
                    unconsolidated_only: true,
                    ..Default::default()
                },
            )
            .await?;

        if unconsolidated.is_empty() {
            return Ok(report);
        }

        // 2. Process in batches
        let bs = batch_size();
        for batch in unconsolidated.chunks(bs) {
            // 3a. Per-fact recall: search for related observations using each fact's embedding,
            // then union/dedup the results.
            let mut seen_obs_ids = std::collections::HashSet::new();
            let mut related_observations = Vec::new();

            for fact in batch {
                if let Some(ref emb) = fact.embedding {
                    let results = self.store.vector_search(emb, bank_id, 5).await?;
                    for sf in results {
                        if sf.fact.network == NetworkType::Observation
                            && seen_obs_ids.insert(sf.fact.id)
                        {
                            related_observations.push(sf.fact);
                        }
                    }
                }
            }

            // 3b. Format prompt
            let facts_text = batch
                .iter()
                .enumerate()
                .map(|(i, f)| format!("[{i}] {}", f.content))
                .collect::<Vec<_>>()
                .join("\n");

            let obs_text = if related_observations.is_empty() {
                "(none)".to_string()
            } else {
                related_observations
                    .iter()
                    .map(|o| format!("[{}] {}", o.id, o.content))
                    .collect::<Vec<_>>()
                    .join("\n")
            };

            let prompt = CONSOLIDATE_PROMPT
                .replace("{facts}", &facts_text)
                .replace("{observations}", &obs_text);

            let request = CompletionRequest {
                model: String::new(),
                messages: vec![Message {
                    role: "user".into(),
                    content: prompt,
                }],
                max_tokens: Some(max_tokens()),
                temperature: Some(0.3),
                system: None,
                ..Default::default()
            };

            let resp: ConsolidateResponse =
                complete_structured(self.llm.as_ref(), request).await?;

            // 3c. Execute actions
            for action in &resp.actions {
                let emb_vec = self.embeddings.embed(&[&action.content]).await?;
                let embedding = emb_vec.into_iter().next();

                // Collect source facts referenced by this action
                let source_facts: Vec<&Fact> = action
                    .fact_indices
                    .iter()
                    .filter_map(|&i| batch.get(i))
                    .collect();
                let evidence_ids: Vec<FactId> = source_facts.iter().map(|f| f.id).collect();

                // Does the LLM want to update an existing observation?
                let updated_existing = if action.action == "update" {
                    if let Some(ref obs_id_str) = action.observation_id {
                        if let Some(existing) = related_observations
                            .iter()
                            .find(|o| o.id.to_string() == *obs_id_str)
                        {
                            let mut updated = existing.clone();
                            updated.content = action.content.clone();
                            updated.embedding = embedding.clone();
                            for eid in &evidence_ids {
                                if !updated.evidence_ids.contains(eid) {
                                    updated.evidence_ids.push(*eid);
                                }
                            }
                            updated.temporal_range = merge_temporal(
                                updated.temporal_range.as_ref(),
                                &source_facts,
                            );
                            updated.updated_at = Utc::now();
                            self.store.update_fact(&updated).await?;
                            report.observations_updated += 1;
                            true
                        } else {
                            eprintln!(
                                "consolidation: LLM referenced unknown observation ID {obs_id_str}, creating new instead"
                            );
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !updated_existing {
                    let now = Utc::now();
                    let obs = Fact {
                        id: FactId::new(),
                        bank_id,
                        content: action.content.clone(),
                        fact_type: FactType::World,
                        network: NetworkType::Observation,
                        entity_ids: vec![],
                        temporal_range: merge_temporal(None, &source_facts),
                        embedding,
                        confidence: None,
                        evidence_ids,
                        source_turn_id: None,
                        created_at: now,
                        updated_at: now,
                        consolidated_at: None,
                    };
                    self.store.insert_facts(&[obs]).await?;
                    report.observations_created += 1;
                }
            }

            // 3d. Mark batch facts as consolidated
            let batch_ids: Vec<FactId> = batch.iter().map(|f| f.id).collect();
            self.store.mark_consolidated(&batch_ids, Utc::now()).await?;
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

    fn make_fact(bank_id: BankId, content: &str) -> Fact {
        let now = Utc::now();
        Fact {
            id: FactId::new(),
            bank_id,
            content: content.into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(vec![0.1; 384]),
            confidence: None,
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now,
            updated_at: now,
            consolidated_at: None,
        }
    }

    fn make_consolidator(
        store: &Arc<MockMemoryStore>,
        llm: &Arc<MockLlmClient>,
        embeddings: &Arc<MockEmbeddings>,
    ) -> DefaultConsolidator {
        DefaultConsolidator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        )
    }

    #[tokio::test]
    async fn new_facts_create_observations() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        let f1 = make_fact(bank_id, "Caroline works at Google");
        let f2 = make_fact(bank_id, "Caroline has a son named James");
        store.insert_facts(&[f1, f2]).await.unwrap();

        llm.push_response(r#"{"actions": [
            {"action": "create", "content": "Caroline works at Google.", "fact_indices": [0]},
            {"action": "create", "content": "Caroline has a son named James.", "fact_indices": [1]}
        ]}"#);

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let report = consolidator.consolidate(bank_id).await.unwrap();

        assert_eq!(report.observations_created, 2);
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
        assert_eq!(observations.len(), 2);
    }

    #[tokio::test]
    async fn incremental_update() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // First batch: create an observation
        let f1 = make_fact(bank_id, "Caroline works at Google");
        store.insert_facts(&[f1]).await.unwrap();

        llm.push_response(r#"{"actions": [
            {"action": "create", "content": "Caroline works at Google.", "fact_indices": [0]}
        ]}"#);

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let report = consolidator.consolidate(bank_id).await.unwrap();
        assert_eq!(report.observations_created, 1);

        // Get the created observation ID
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
        let obs_id = observations[0].id.to_string();

        // Second batch: update the existing observation
        let f2 = make_fact(bank_id, "Caroline got promoted to senior engineer");
        store.insert_facts(&[f2]).await.unwrap();

        llm.push_response(&format!(r#"{{"actions": [
            {{"action": "update", "observation_id": "{obs_id}", "content": "Caroline works as a senior engineer at Google, having recently been promoted.", "fact_indices": [0]}}
        ]}}"#));

        let report = consolidator.consolidate(bank_id).await.unwrap();
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
        assert!(observations[0].content.contains("senior engineer"));
    }

    #[tokio::test]
    async fn idempotent_no_unconsolidated() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // Insert a fact and mark it consolidated
        let f1 = make_fact(bank_id, "Some fact");
        let f1_id = f1.id;
        store.insert_facts(&[f1]).await.unwrap();
        store
            .mark_consolidated(&[f1_id], Utc::now())
            .await
            .unwrap();

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let report = consolidator.consolidate(bank_id).await.unwrap();

        assert_eq!(report.observations_created, 0);
        assert_eq!(report.observations_updated, 0);
        assert_eq!(llm.remaining(), 0);
    }

    #[tokio::test]
    async fn batch_size_respected() {
        // With batch_size=8 (default), 12 facts should produce 2 LLM calls
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        for i in 0..12 {
            let f = make_fact(bank_id, &format!("Fact number {i}"));
            store.insert_facts(&[f]).await.unwrap();
        }

        // First batch of 8
        llm.push_response(r#"{"actions": [
            {"action": "create", "content": "Batch 1 observation.", "fact_indices": [0,1,2,3,4,5,6,7]}
        ]}"#);
        // Second batch of 4
        llm.push_response(r#"{"actions": [
            {"action": "create", "content": "Batch 2 observation.", "fact_indices": [0,1,2,3]}
        ]}"#);

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let report = consolidator.consolidate(bank_id).await.unwrap();

        assert_eq!(report.observations_created, 2);
        assert_eq!(llm.remaining(), 0);
    }

    #[tokio::test]
    async fn only_world_and_experience() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // Insert an Opinion fact — should NOT be consolidated
        let now = Utc::now();
        let opinion = Fact {
            id: FactId::new(),
            bank_id,
            content: "Rust is the best language.".into(),
            fact_type: FactType::Experience,
            network: NetworkType::Opinion,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(vec![0.1; 384]),
            confidence: Some(0.8),
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now,
            updated_at: now,
            consolidated_at: None,
        };
        store.insert_facts(&[opinion]).await.unwrap();

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let report = consolidator.consolidate(bank_id).await.unwrap();

        assert_eq!(report.observations_created, 0);
        assert_eq!(llm.remaining(), 0);
    }
}
