//! Graph retriever — spreading activation over the fact graph.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;

use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::storage::MemoryStore;
use crate::types::{FactId, RecallQuery, RetrievalSource, ScoredFact};

use super::Retriever;

/// Configuration for the graph retriever's spreading activation.
pub struct GraphRetrieverConfig {
    /// Maximum BFS hops from seed nodes.
    pub max_hops: usize,
    /// Activation decay per hop.
    pub decay_factor: f32,
    /// Minimum activation to keep exploring.
    pub activation_threshold: f32,
    /// Maximum seed nodes from initial vector search.
    pub max_seeds: usize,
    /// Maximum facts to return.
    pub max_results: usize,
}

impl Default for GraphRetrieverConfig {
    fn default() -> Self {
        Self {
            max_hops: 3,
            decay_factor: 0.7,
            activation_threshold: 0.1,
            max_seeds: 10,
            max_results: 50,
        }
    }
}

/// Retrieves facts by spreading activation through the knowledge graph.
pub struct GraphRetriever {
    store: Arc<dyn MemoryStore>,
    embeddings: Arc<dyn EmbeddingClient>,
    config: GraphRetrieverConfig,
}

impl GraphRetriever {
    /// Create a new graph retriever.
    pub fn new(
        store: Arc<dyn MemoryStore>,
        embeddings: Arc<dyn EmbeddingClient>,
        config: GraphRetrieverConfig,
    ) -> Self {
        Self {
            store,
            embeddings,
            config,
        }
    }
}

#[async_trait]
impl Retriever for GraphRetriever {
    async fn retrieve(&self, query: &RecallQuery) -> Result<Vec<ScoredFact>> {
        // Step 1: Get seed nodes via vector search
        let vecs = self.embeddings.embed(&[&query.query]).await?;
        let embedding = &vecs[0];
        let seeds = self
            .store
            .vector_search(embedding, query.bank_id, self.config.max_seeds)
            .await?;

        if seeds.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: Initialize activations from seeds
        let mut activations: HashMap<FactId, f32> = HashMap::new();
        for sf in &seeds {
            activations.insert(sf.fact.id, sf.score);
        }

        // Step 3: BFS spreading activation
        let mut frontier: VecDeque<(FactId, f32, usize)> = seeds
            .iter()
            .map(|sf| (sf.fact.id, sf.score, 0))
            .collect();
        let mut visited: HashSet<FactId> = seeds.iter().map(|sf| sf.fact.id).collect();

        while let Some((fact_id, activation, hop)) = frontier.pop_front() {
            if hop >= self.config.max_hops {
                continue;
            }

            let neighbors = self.store.get_neighbors(fact_id, None).await?;
            for (neighbor_id, edge_weight) in neighbors {
                let propagated = activation * edge_weight * self.config.decay_factor;
                if propagated < self.config.activation_threshold {
                    continue;
                }

                let entry = activations.entry(neighbor_id).or_insert(0.0);
                *entry = entry.max(propagated);

                if visited.insert(neighbor_id) {
                    frontier.push_back((neighbor_id, propagated, hop + 1));
                }
            }
        }

        // Step 4: Fetch facts and build scored results
        let fact_ids: Vec<FactId> = activations.keys().copied().collect();
        let facts = self.store.get_facts(&fact_ids).await?;

        let mut scored: Vec<ScoredFact> = facts
            .into_iter()
            .filter_map(|f| {
                activations.get(&f.id).map(|&score| ScoredFact {
                    fact: f,
                    score,
                    sources: vec![RetrievalSource::Graph],
                })
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.max_results);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::mock::MockEmbeddings;
    use crate::storage::mock::MockMemoryStore;
    use crate::types::*;
    use chrono::Utc;

    #[tokio::test]
    async fn spreads_activation_through_links() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = BankId::new();

        // Create facts with embeddings
        let emb = embeddings.embed(&["seed fact"]).await.unwrap();
        let seed_id = FactId::new();
        let neighbor_id = FactId::new();

        let facts = vec![
            Fact {
                id: seed_id,
                bank_id: bank,
                content: "seed fact".into(),
                fact_type: FactType::World,
                network: NetworkType::World,
                entity_ids: vec![],
                temporal_range: None,
                embedding: Some(emb[0].clone()),
                confidence: None,
                evidence_ids: vec![],
                source_turn_id: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                consolidated_at: None,
            },
            Fact {
                id: neighbor_id,
                bank_id: bank,
                content: "neighbor fact".into(),
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
            },
        ];
        store.insert_facts(&facts).await.unwrap();

        // Link seed to neighbor
        let link = GraphLink {
            source_id: seed_id,
            target_id: neighbor_id,
            link_type: LinkType::Semantic,
            weight: 0.8,
        };
        store.insert_links(&[link]).await.unwrap();

        let config = GraphRetrieverConfig {
            max_hops: 2,
            decay_factor: 0.7,
            activation_threshold: 0.01,
            max_seeds: 5,
            max_results: 50,
        };

        let retriever = GraphRetriever::new(store, embeddings, config);
        let query = RecallQuery {
            bank_id: bank,
            query: "seed fact".into(),
            budget_tokens: 1000,
            network_filter: None,
            temporal_anchor: None,
        };

        let results = retriever.retrieve(&query).await.unwrap();
        assert!(results.len() >= 2);
        // Seed should have highest activation
        assert_eq!(results[0].fact.id, seed_id);
        // Neighbor should also be found via graph traversal
        assert!(results.iter().any(|r| r.fact.id == neighbor_id));
    }

    #[tokio::test]
    async fn empty_store_returns_empty() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));

        let retriever = GraphRetriever::new(store, embeddings, GraphRetrieverConfig::default());
        let query = RecallQuery {
            bank_id: BankId::new(),
            query: "anything".into(),
            budget_tokens: 1000,
            network_filter: None,
            temporal_anchor: None,
        };

        let results = retriever.retrieve(&query).await.unwrap();
        assert!(results.is_empty());
    }
}
