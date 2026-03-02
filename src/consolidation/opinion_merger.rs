//! Opinion merger — deduplicates and reconciles similar opinions.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;

use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::llm::{complete_structured, LlmClient};
use crate::storage::MemoryStore;
use crate::types::id::BankId;
use crate::types::llm::{CompletionRequest, Message};
use crate::types::{FactFilter, NetworkType, OpinionMergeReport};

use super::cluster_by_similarity;

/// Merges similar opinions within a bank.
#[async_trait]
pub trait OpinionMerger: Send + Sync {
    /// Scan all opinions in a bank and merge similar ones.
    async fn merge(&self, bank_id: BankId) -> Result<OpinionMergeReport>;
}

/// Default implementation using LLM for classification.
pub struct DefaultOpinionMerger {
    store: Arc<dyn MemoryStore>,
    llm: Arc<dyn LlmClient>,
    embeddings: Arc<dyn EmbeddingClient>,
}

impl DefaultOpinionMerger {
    /// Create a new opinion merger.
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
struct MergeResponse {
    classification: String,
    merged_text: Option<String>,
    superseded_index: Option<usize>,
}

const MERGE_PROMPT: &str = include_str!("../../prompts/merge_opinions.txt");

#[async_trait]
impl OpinionMerger for DefaultOpinionMerger {
    async fn merge(&self, bank_id: BankId) -> Result<OpinionMergeReport> {
        let mut report = OpinionMergeReport::default();

        // 1. Fetch all opinions with confidence > 0
        let all_opinions = self
            .store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Opinion]),
                    ..Default::default()
                },
            )
            .await?;

        let active_opinions: Vec<_> = all_opinions
            .into_iter()
            .filter(|f| f.confidence.unwrap_or(0.0) > 0.0)
            .collect();

        if active_opinions.len() < 2 {
            return Ok(report);
        }

        // 2. Cluster by embedding similarity (cosine > 0.85)
        // Ensure all opinions have embeddings (re-embed if needed)
        let mut full_embeddings: Vec<Vec<f32>> = Vec::with_capacity(active_opinions.len());
        for opinion in &active_opinions {
            if let Some(ref emb) = opinion.embedding {
                full_embeddings.push(emb.clone());
            } else {
                let emb = self.embeddings.embed(&[&opinion.content]).await?;
                full_embeddings.push(emb.into_iter().next().unwrap_or_default());
            }
        }

        let emb_refs: Vec<&[f32]> = full_embeddings.iter().map(|e| e.as_slice()).collect();
        let clusters = cluster_by_similarity(&emb_refs, 0.85);

        // 3. For each cluster with >1 opinion
        for cluster in clusters {
            if cluster.len() < 2 {
                continue;
            }

            let cluster_opinions: Vec<_> = cluster.iter().map(|&i| &active_opinions[i]).collect();

            let opinions_text = cluster_opinions
                .iter()
                .enumerate()
                .map(|(i, o)| {
                    format!(
                        "{}. [confidence: {}] {}",
                        i,
                        o.confidence.unwrap_or(0.5),
                        o.content
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            let prompt = MERGE_PROMPT.replace("{opinions}", &opinions_text);

            let request = CompletionRequest {
                model: String::new(),
                messages: vec![Message {
                    role: "user".into(),
                    content: prompt,
                }],
                max_tokens: Some(1024),
                temperature: Some(0.3),
                system: None,
                ..Default::default()
            };

            let resp: MergeResponse = complete_structured(self.llm.as_ref(), request).await?;

            match resp.classification.as_str() {
                "consistent" => {
                    // Pick highest-confidence as winner
                    let winner_idx = cluster_opinions
                        .iter()
                        .enumerate()
                        .max_by(|a, b| {
                            a.1.confidence
                                .unwrap_or(0.0)
                                .partial_cmp(&b.1.confidence.unwrap_or(0.0))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    let merged_text = resp
                        .merged_text
                        .unwrap_or_else(|| cluster_opinions[winner_idx].content.clone());

                    // Compute boosted confidence: min(1.0, avg + 0.1)
                    let avg_confidence: f32 = cluster_opinions
                        .iter()
                        .map(|o| o.confidence.unwrap_or(0.5))
                        .sum::<f32>()
                        / cluster_opinions.len() as f32;
                    let new_confidence = (avg_confidence + 0.1).min(1.0);

                    // Combine evidence IDs
                    let mut combined_evidence: Vec<_> = cluster_opinions
                        .iter()
                        .flat_map(|o| &o.evidence_ids)
                        .copied()
                        .collect();
                    combined_evidence.dedup();

                    // Re-embed merged text
                    let emb = self.embeddings.embed(&[&merged_text]).await?;

                    // Update winner
                    let mut winner = cluster_opinions[winner_idx].clone();
                    winner.content = merged_text;
                    winner.confidence = Some(new_confidence);
                    winner.evidence_ids = combined_evidence;
                    winner.embedding = emb.into_iter().next();
                    winner.updated_at = Utc::now();
                    self.store.update_fact(&winner).await?;

                    // Set losers' confidence to 0.0
                    for (i, opinion) in cluster_opinions.iter().enumerate() {
                        if i != winner_idx {
                            let mut loser = (*opinion).clone();
                            loser.confidence = Some(0.0);
                            loser.updated_at = Utc::now();
                            self.store.update_fact(&loser).await?;
                        }
                    }

                    report.opinions_merged += 1;
                }
                "contradictory" => {
                    // Lower weaker opinion's confidence *= 0.7
                    // Find the weakest
                    let weakest_idx = cluster_opinions
                        .iter()
                        .enumerate()
                        .min_by(|a, b| {
                            a.1.confidence
                                .unwrap_or(0.0)
                                .partial_cmp(&b.1.confidence.unwrap_or(0.0))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    let mut weakened = cluster_opinions[weakest_idx].clone();
                    weakened.confidence =
                        Some(weakened.confidence.unwrap_or(0.5) * 0.7);
                    weakened.updated_at = Utc::now();
                    self.store.update_fact(&weakened).await?;

                    report.opinions_conflicting += 1;
                }
                "superseded" => {
                    // Lower older opinion's confidence *= 0.7
                    let superseded_idx =
                        resp.superseded_index.unwrap_or(0).min(cluster_opinions.len() - 1);

                    let mut superseded = cluster_opinions[superseded_idx].clone();
                    superseded.confidence =
                        Some(superseded.confidence.unwrap_or(0.5) * 0.7);
                    superseded.updated_at = Utc::now();
                    self.store.update_fact(&superseded).await?;

                    report.opinions_superseded += 1;
                }
                _ => {}
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
    use crate::types::id::FactId;
    use crate::types::{Fact, FactType};

    fn setup() -> (Arc<MockMemoryStore>, Arc<MockLlmClient>, Arc<MockEmbeddings>) {
        (
            Arc::new(MockMemoryStore::new()),
            Arc::new(MockLlmClient::new()),
            Arc::new(MockEmbeddings::new(384)),
        )
    }

    fn make_opinion(bank_id: BankId, content: &str, confidence: f32, embedding: Vec<f32>) -> Fact {
        let now = Utc::now();
        Fact {
            id: FactId::new(),
            bank_id,
            content: content.into(),
            fact_type: FactType::Experience,
            network: NetworkType::Opinion,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(embedding),
            confidence: Some(confidence),
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now,
            updated_at: now,
            consolidated_at: None,
        }
    }

    #[tokio::test]
    async fn consistent_opinions_merged() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // Two opinions with identical embeddings (will cluster)
        let emb = vec![1.0; 384];
        let o1 = make_opinion(bank_id, "Rust is great for systems", 0.7, emb.clone());
        let o2 = make_opinion(bank_id, "Rust is excellent for systems programming", 0.8, emb);
        store.insert_facts(&[o1.clone(), o2.clone()]).await.unwrap();

        llm.push_response(
            r#"{"classification": "consistent", "merged_text": "Rust is excellent for systems programming."}"#,
        );

        let merger = DefaultOpinionMerger::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let report = merger.merge(bank_id).await.unwrap();

        assert_eq!(report.opinions_merged, 1);

        // Check that one opinion has confidence 0.0
        let opinions = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Opinion]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let active: Vec<_> = opinions
            .iter()
            .filter(|o| o.confidence.unwrap_or(0.0) > 0.0)
            .collect();
        let retired: Vec<_> = opinions
            .iter()
            .filter(|o| o.confidence.unwrap_or(0.0) == 0.0)
            .collect();

        assert_eq!(active.len(), 1);
        assert_eq!(retired.len(), 1);
        // Winner should have boosted confidence: avg(0.7, 0.8) + 0.1 = 0.85
        assert!((active[0].confidence.unwrap() - 0.85).abs() < 0.01);
    }

    #[tokio::test]
    async fn contradictory_opinions_both_kept() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        let emb = vec![1.0; 384];
        let o1 = make_opinion(bank_id, "Python is the best language", 0.8, emb.clone());
        let o2 = make_opinion(bank_id, "Rust is the best language", 0.6, emb);
        store.insert_facts(&[o1, o2]).await.unwrap();

        llm.push_response(r#"{"classification": "contradictory"}"#);

        let merger = DefaultOpinionMerger::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let report = merger.merge(bank_id).await.unwrap();
        assert_eq!(report.opinions_conflicting, 1);

        let opinions = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Opinion]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Both should still be active, weaker one's confidence reduced
        let active: Vec<_> = opinions
            .iter()
            .filter(|o| o.confidence.unwrap_or(0.0) > 0.0)
            .collect();
        assert_eq!(active.len(), 2);

        // The weaker (0.6) should have been reduced to 0.42
        let weakened = active
            .iter()
            .find(|o| (o.confidence.unwrap() - 0.42).abs() < 0.01);
        assert!(weakened.is_some());
    }

    #[tokio::test]
    async fn superseded_weakens_older() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        let emb = vec![1.0; 384];
        let o1 = make_opinion(bank_id, "Use Python 2", 0.7, emb.clone());
        let o2 = make_opinion(bank_id, "Use Python 3", 0.8, emb);
        store.insert_facts(&[o1, o2]).await.unwrap();

        llm.push_response(r#"{"classification": "superseded", "superseded_index": 0}"#);

        let merger = DefaultOpinionMerger::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let report = merger.merge(bank_id).await.unwrap();
        assert_eq!(report.opinions_superseded, 1);

        let opinions = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Opinion]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // The superseded one (index 0) should have been weakened
        let weakened = opinions
            .iter()
            .find(|o| o.content.contains("Python 2"))
            .unwrap();
        assert!((weakened.confidence.unwrap() - 0.49).abs() < 0.01);
    }

    #[tokio::test]
    async fn unrelated_opinions_no_merge() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // Orthogonal embeddings — won't cluster
        let mut emb1 = vec![0.0; 384];
        emb1[0] = 1.0;
        let mut emb2 = vec![0.0; 384];
        emb2[1] = 1.0;

        let o1 = make_opinion(bank_id, "I like Rust", 0.7, emb1);
        let o2 = make_opinion(bank_id, "I like pizza", 0.8, emb2);
        store.insert_facts(&[o1, o2]).await.unwrap();

        let merger = DefaultOpinionMerger::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let report = merger.merge(bank_id).await.unwrap();
        assert_eq!(report.opinions_merged, 0);
        assert_eq!(report.opinions_superseded, 0);
        assert_eq!(report.opinions_conflicting, 0);
        assert_eq!(llm.remaining(), 0);
    }

    #[tokio::test]
    async fn empty_bank_no_op() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        let merger = DefaultOpinionMerger::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
        );

        let report = merger.merge(bank_id).await.unwrap();
        assert_eq!(report.opinions_merged, 0);
        assert_eq!(report.opinions_superseded, 0);
        assert_eq!(report.opinions_conflicting, 0);
    }
}
