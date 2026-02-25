//! Mental model generator — synthesizes cross-cutting mental models from observations.

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
use crate::types::{Fact, FactFilter, FactType, MentalModelReport, NetworkType};
use crate::util::cosine_similarity;

use super::cluster_by_similarity;

/// Generates mental models from clusters of observations.
#[async_trait]
pub trait MentalModelGenerator: Send + Sync {
    /// Scan observations and create/update mental models.
    async fn generate(&self, bank_id: BankId) -> Result<MentalModelReport>;
}

/// Default implementation using LLM for synthesis.
pub struct DefaultMentalModelGenerator {
    store: Arc<dyn MemoryStore>,
    llm: Arc<dyn LlmClient>,
    embeddings: Arc<dyn EmbeddingClient>,
}

impl DefaultMentalModelGenerator {
    /// Create a new mental model generator.
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
    mental_model: String,
    confidence: f32,
}

const SYNTHESIZE_PROMPT: &str = include_str!("../../prompts/synthesize_mental_model.txt");

/// Compute the centroid (average) of a set of embeddings.
fn compute_centroid(embeddings: &[&[f32]]) -> Vec<f32> {
    if embeddings.is_empty() {
        return vec![];
    }
    let dims = embeddings[0].len();
    let n = embeddings.len() as f32;
    let mut centroid = vec![0.0f32; dims];
    for emb in embeddings {
        for (i, &val) in emb.iter().enumerate() {
            centroid[i] += val;
        }
    }
    for val in &mut centroid {
        *val /= n;
    }
    centroid
}

#[async_trait]
impl MentalModelGenerator for DefaultMentalModelGenerator {
    async fn generate(&self, bank_id: BankId) -> Result<MentalModelReport> {
        let mut report = MentalModelReport::default();

        // 1. Fetch all observations
        let observations = self
            .store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Observation]),
                    ..Default::default()
                },
            )
            .await?;

        if observations.len() < 3 {
            return Ok(report);
        }

        // Ensure all have embeddings
        let mut obs_embeddings: Vec<Vec<f32>> = Vec::with_capacity(observations.len());
        for obs in &observations {
            if let Some(ref emb) = obs.embedding {
                obs_embeddings.push(emb.clone());
            } else {
                let emb = self.embeddings.embed(&[&obs.content]).await?;
                obs_embeddings.push(emb.into_iter().next().unwrap_or_default());
            }
        }

        // 2. Cluster by embedding similarity (cosine > 0.80)
        let emb_refs: Vec<&[f32]> = obs_embeddings.iter().map(|e| e.as_slice()).collect();
        let clusters = cluster_by_similarity(&emb_refs, 0.80);

        // 3. Filter to clusters with 3+ observations
        let dense_clusters: Vec<Vec<usize>> =
            clusters.into_iter().filter(|c| c.len() >= 3).collect();

        if dense_clusters.is_empty() {
            return Ok(report);
        }

        // 4. Fetch existing mental models
        let existing_models = self
            .store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::MentalModel]),
                    ..Default::default()
                },
            )
            .await?;

        // 5. For each dense cluster
        for cluster in dense_clusters {
            let cluster_obs: Vec<&Fact> = cluster.iter().map(|&i| &observations[i]).collect();
            let cluster_embs: Vec<&[f32]> = cluster.iter().map(|&i| emb_refs[i]).collect();

            // Compute cluster centroid
            let centroid = compute_centroid(&cluster_embs);

            // Check if similar mental model exists
            let similar_model = existing_models.iter().find(|m| {
                if let Some(ref emb) = m.embedding {
                    cosine_similarity(&centroid, emb) > 0.85
                } else {
                    false
                }
            });

            if let Some(existing) = similar_model {
                // Check if it has diverged enough to warrant re-synthesis
                let similarity = existing
                    .embedding
                    .as_ref()
                    .map(|emb| cosine_similarity(&centroid, emb))
                    .unwrap_or(0.0);

                if similarity > 0.95 {
                    // Close enough — unchanged
                    report.models_unchanged += 1;
                } else {
                    // Diverged — re-synthesize
                    let observations_text = cluster_obs
                        .iter()
                        .enumerate()
                        .map(|(i, o)| format!("{}. {}", i + 1, o.content))
                        .collect::<Vec<_>>()
                        .join("\n");

                    let prompt =
                        SYNTHESIZE_PROMPT.replace("{observations}", &observations_text);

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

                    let emb = self.embeddings.embed(&[&resp.mental_model]).await?;

                    let mut updated = existing.clone();
                    updated.content = resp.mental_model;
                    updated.confidence = Some(resp.confidence);
                    updated.embedding = emb.into_iter().next();
                    updated.evidence_ids =
                        cluster_obs.iter().map(|o| o.id).collect();
                    updated.updated_at = Utc::now();

                    self.store.update_fact(&updated).await?;
                    report.models_updated += 1;
                }
            } else {
                // No match — synthesize new mental model
                let observations_text = cluster_obs
                    .iter()
                    .enumerate()
                    .map(|(i, o)| format!("{}. {}", i + 1, o.content))
                    .collect::<Vec<_>>()
                    .join("\n");

                let prompt =
                    SYNTHESIZE_PROMPT.replace("{observations}", &observations_text);

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

                let emb = self.embeddings.embed(&[&resp.mental_model]).await?;

                let now = Utc::now();
                let model = Fact {
                    id: FactId::new(),
                    bank_id,
                    content: resp.mental_model,
                    fact_type: FactType::World,
                    network: NetworkType::MentalModel,
                    entity_ids: vec![],
                    temporal_range: None,
                    embedding: emb.into_iter().next(),
                    confidence: Some(resp.confidence),
                    evidence_ids: cluster_obs.iter().map(|o| o.id).collect(),
                    source_turn_id: None,
                    created_at: now,
                    updated_at: now,
                };

                self.store.insert_facts(&[model]).await?;
                report.models_created += 1;
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
    use crate::types::id::EntityId;

    fn setup() -> (Arc<MockMemoryStore>, Arc<MockLlmClient>, Arc<MockEmbeddings>) {
        (
            Arc::new(MockMemoryStore::new()),
            Arc::new(MockLlmClient::new()),
            Arc::new(MockEmbeddings::new(384)),
        )
    }

    fn make_observation(bank_id: BankId, content: &str, embedding: Vec<f32>) -> Fact {
        let now = Utc::now();
        Fact {
            id: FactId::new(),
            bank_id,
            content: content.into(),
            fact_type: FactType::World,
            network: NetworkType::Observation,
            entity_ids: vec![EntityId::new()],
            temporal_range: None,
            embedding: Some(embedding),
            confidence: Some(0.8),
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now,
            updated_at: now,
        }
    }

    #[tokio::test]
    async fn sufficient_density_creates_model() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // 3 similar observations (same embedding = will cluster)
        let emb = vec![1.0; 384];
        let o1 = make_observation(bank_id, "Rust has zero-cost abstractions", emb.clone());
        let o2 = make_observation(bank_id, "Rust guarantees memory safety", emb.clone());
        let o3 = make_observation(bank_id, "Rust prevents data races", emb);
        store
            .insert_facts(&[o1, o2, o3])
            .await
            .unwrap();

        llm.push_response(
            r#"{"mental_model": "Rust prioritizes safety and performance through zero-cost abstractions and compile-time guarantees.", "confidence": 0.85}"#,
        );

        let generator = DefaultMentalModelGenerator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let report = generator.generate(bank_id).await.unwrap();
        assert_eq!(report.models_created, 1);
        assert_eq!(report.models_updated, 0);

        let models = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::MentalModel]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].evidence_ids.len(), 3);
    }

    #[tokio::test]
    async fn insufficient_density_no_model() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // Only 2 observations — not enough for a cluster of 3
        let emb = vec![1.0; 384];
        let o1 = make_observation(bank_id, "Rust has ownership", emb.clone());
        let o2 = make_observation(bank_id, "Rust has borrowing", emb);
        store.insert_facts(&[o1, o2]).await.unwrap();

        let generator = DefaultMentalModelGenerator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let report = generator.generate(bank_id).await.unwrap();
        assert_eq!(report.models_created, 0);
        assert_eq!(llm.remaining(), 0);
    }

    #[tokio::test]
    async fn evidence_links_populated() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        let emb = vec![1.0; 384];
        let o1 = make_observation(bank_id, "obs 1", emb.clone());
        let o2 = make_observation(bank_id, "obs 2", emb.clone());
        let o3 = make_observation(bank_id, "obs 3", emb);
        let obs_ids = vec![o1.id, o2.id, o3.id];
        store
            .insert_facts(&[o1, o2, o3])
            .await
            .unwrap();

        llm.push_response(
            r#"{"mental_model": "cross-cutting model", "confidence": 0.8}"#,
        );

        let generator = DefaultMentalModelGenerator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        generator.generate(bank_id).await.unwrap();

        let models = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::MentalModel]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(models.len(), 1);
        for id in &obs_ids {
            assert!(models[0].evidence_ids.contains(id));
        }
    }

    #[tokio::test]
    async fn idempotent_unchanged_model() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // Create 3 observations with identical embedding
        let emb = vec![1.0; 384];
        let o1 = make_observation(bank_id, "obs a", emb.clone());
        let o2 = make_observation(bank_id, "obs b", emb.clone());
        let o3 = make_observation(bank_id, "obs c", emb.clone());
        store.insert_facts(&[o1, o2, o3]).await.unwrap();

        // Also create an existing mental model with the same embedding
        let now = Utc::now();
        let existing_model = Fact {
            id: FactId::new(),
            bank_id,
            content: "existing model".into(),
            fact_type: FactType::World,
            network: NetworkType::MentalModel,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(emb),
            confidence: Some(0.8),
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now,
            updated_at: now,
        };
        store.insert_facts(&[existing_model]).await.unwrap();

        let generator = DefaultMentalModelGenerator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let report = generator.generate(bank_id).await.unwrap();

        // The centroid of [1,0,0,...] observations is [1,0,0,...], which is identical
        // to the existing model's embedding → similarity = 1.0 > 0.95 → unchanged
        assert_eq!(report.models_unchanged, 1);
        assert_eq!(report.models_created, 0);
        assert_eq!(llm.remaining(), 0); // No LLM call needed
    }

    #[tokio::test]
    async fn two_distinct_topics_two_models() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // Topic A: all at [1,0,0,...,0]
        let mut emb_a = vec![0.0; 384];
        emb_a[0] = 1.0;
        // Topic B: all at [0,1,0,...,0]
        let mut emb_b = vec![0.0; 384];
        emb_b[1] = 1.0;

        let a1 = make_observation(bank_id, "rust a1", emb_a.clone());
        let a2 = make_observation(bank_id, "rust a2", emb_a.clone());
        let a3 = make_observation(bank_id, "rust a3", emb_a);
        let b1 = make_observation(bank_id, "python b1", emb_b.clone());
        let b2 = make_observation(bank_id, "python b2", emb_b.clone());
        let b3 = make_observation(bank_id, "python b3", emb_b);

        store
            .insert_facts(&[a1, a2, a3, b1, b2, b3])
            .await
            .unwrap();

        // Two LLM calls for two models
        llm.push_response(
            r#"{"mental_model": "Rust model", "confidence": 0.8}"#,
        );
        llm.push_response(
            r#"{"mental_model": "Python model", "confidence": 0.8}"#,
        );

        let generator = DefaultMentalModelGenerator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let report = generator.generate(bank_id).await.unwrap();
        assert_eq!(report.models_created, 2);

        let models = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::MentalModel]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(models.len(), 2);
    }
}
