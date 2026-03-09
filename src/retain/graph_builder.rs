//! Graph link construction after fact extraction (Phase 2D).

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Duration;

use crate::error::Result;
use crate::llm::LlmClient;
use crate::storage::MemoryStore;
use crate::types::{
    BankId, CompletionRequest, Fact, FactFilter, GraphLink, LinkType, Message, NetworkType,
};
use crate::util::cosine_similarity;

/// Trait for building graph links between facts.
#[async_trait]
pub trait GraphBuilder: Send + Sync {
    /// Build links between new facts and the existing fact graph.
    ///
    /// The `store` parameter controls which store is used for reads/writes,
    /// allowing callers to pass a transaction handle for atomic operations.
    async fn build_links(
        &self,
        new_facts: &[Fact],
        bank_id: BankId,
        store: &dyn MemoryStore,
    ) -> Result<Vec<GraphLink>>;
}

/// Configuration for graph link construction thresholds.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Cosine similarity threshold for semantic links.
    pub semantic_threshold: f32,
    /// Maximum temporal distance (days) for temporal links.
    pub temporal_max_days: i64,
    /// Whether to check for causal links via LLM.
    pub enable_causal: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            semantic_threshold: 0.7,
            temporal_max_days: 30,
            enable_causal: true,
        }
    }
}

/// Graph builder that constructs temporal, semantic, entity, and causal links.
pub struct DefaultGraphBuilder {
    llm: Arc<dyn LlmClient>,
    config: GraphConfig,
}

impl DefaultGraphBuilder {
    /// Create a new graph builder.
    pub fn new(llm: Arc<dyn LlmClient>, config: GraphConfig) -> Self {
        Self { llm, config }
    }

    /// Build temporal links between facts with overlapping/adjacent time ranges.
    fn build_temporal_links(&self, new_facts: &[Fact], existing_facts: &[Fact]) -> Vec<GraphLink> {
        let mut links = Vec::new();
        let max_distance = Duration::days(self.config.temporal_max_days);

        for new_fact in new_facts {
            let Some(ref new_tr) = new_fact.temporal_range else {
                continue;
            };

            for existing in existing_facts {
                if existing.id == new_fact.id {
                    continue;
                }
                let Some(ref existing_tr) = existing.temporal_range else {
                    continue;
                };

                // Check if within max distance
                if let (Some(new_start), Some(ex_start)) = (new_tr.start, existing_tr.start) {
                    let distance = (new_start - ex_start).abs();
                    if distance <= max_distance {
                        // Weight inversely proportional to distance (1.0 for same time, ~0 at max)
                        let weight = 1.0
                            - (distance.num_seconds() as f32 / max_distance.num_seconds() as f32);
                        let weight = weight.max(0.1); // Minimum weight for any temporal link

                        links.push(GraphLink {
                            source_id: new_fact.id,
                            target_id: existing.id,
                            link_type: LinkType::Temporal,
                            weight,
                        });
                    }
                } else if new_tr.overlaps(existing_tr) {
                    // Overlapping but without precise times → moderate weight
                    links.push(GraphLink {
                        source_id: new_fact.id,
                        target_id: existing.id,
                        link_type: LinkType::Temporal,
                        weight: 0.5,
                    });
                }
            }
        }

        links
    }

    /// Build entity links between facts sharing entities.
    fn build_entity_links(&self, new_facts: &[Fact], existing_facts: &[Fact]) -> Vec<GraphLink> {
        let mut links = Vec::new();

        for new_fact in new_facts {
            if new_fact.entity_ids.is_empty() {
                continue;
            }

            for existing in existing_facts {
                if existing.id == new_fact.id || existing.entity_ids.is_empty() {
                    continue;
                }

                let shared: usize = new_fact
                    .entity_ids
                    .iter()
                    .filter(|id| existing.entity_ids.contains(id))
                    .count();

                if shared > 0 {
                    let total_unique = {
                        let mut all = new_fact.entity_ids.clone();
                        for id in &existing.entity_ids {
                            if !all.contains(id) {
                                all.push(*id);
                            }
                        }
                        all.len()
                    };
                    let weight = shared as f32 / total_unique as f32;

                    links.push(GraphLink {
                        source_id: new_fact.id,
                        target_id: existing.id,
                        link_type: LinkType::Entity,
                        weight,
                    });
                }
            }
        }

        links
    }

    /// Build semantic links between facts with high embedding similarity.
    async fn build_semantic_links(
        &self,
        new_facts: &[Fact],
        existing_facts: &[Fact],
    ) -> Result<Vec<GraphLink>> {
        let mut links = Vec::new();

        // Collect facts that have embeddings
        let new_with_emb: Vec<(&Fact, &Vec<f32>)> = new_facts
            .iter()
            .filter_map(|f| f.embedding.as_ref().map(|e| (f, e)))
            .collect();

        let existing_with_emb: Vec<(&Fact, &Vec<f32>)> = existing_facts
            .iter()
            .filter_map(|f| f.embedding.as_ref().map(|e| (f, e)))
            .collect();

        for (new_fact, new_emb) in &new_with_emb {
            for (existing, ex_emb) in &existing_with_emb {
                if new_fact.id == existing.id {
                    continue;
                }
                let sim = cosine_similarity(new_emb, ex_emb);
                if sim >= self.config.semantic_threshold {
                    links.push(GraphLink {
                        source_id: new_fact.id,
                        target_id: existing.id,
                        link_type: LinkType::Semantic,
                        weight: sim,
                    });
                }
            }
        }

        Ok(links)
    }

    /// Check for causal links via LLM between fact pairs that share entities or are temporally close.
    async fn build_causal_links(
        &self,
        new_facts: &[Fact],
        existing_facts: &[Fact],
        entity_links: &[GraphLink],
        temporal_links: &[GraphLink],
    ) -> Result<Vec<GraphLink>> {
        if !self.config.enable_causal {
            return Ok(Vec::new());
        }

        let mut links = Vec::new();

        // Only check pairs that already have entity or temporal connections
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        for link in entity_links.iter().chain(temporal_links.iter()) {
            for (ni, new_fact) in new_facts.iter().enumerate() {
                if link.source_id == new_fact.id {
                    for (ei, existing) in existing_facts.iter().enumerate() {
                        if link.target_id == existing.id {
                            candidates.push((ni, ei));
                        }
                    }
                }
            }
        }

        // Deduplicate candidates
        candidates.sort_unstable();
        candidates.dedup();

        // Limit LLM calls
        let max_checks = 10;
        for (ni, ei) in candidates.into_iter().take(max_checks) {
            let new_fact = &new_facts[ni];
            let existing = &existing_facts[ei];

            let prompt = format!(
                "Given these two facts, is there a causal relationship between them?\n\n\
                 Fact A: {}\n\
                 Fact B: {}\n\n\
                 Answer 'yes' if one fact caused or led to the other, otherwise 'no'.",
                new_fact.content, existing.content
            );

            let request = CompletionRequest {
                model: String::new(),
                system: Some(
                    "You are a causal relationship detector. Answer only 'yes' or 'no'.".into(),
                ),
                messages: vec![Message::text("user", prompt)],
                temperature: Some(0.0),
                max_tokens: Some(10),
                ..Default::default()
            };

            let response = self.llm.complete(request).await?;
            if response.content.trim().to_lowercase().starts_with("yes") {
                links.push(GraphLink {
                    source_id: new_fact.id,
                    target_id: existing.id,
                    link_type: LinkType::Causal,
                    weight: 0.8,
                });
            }
        }

        Ok(links)
    }
}

#[async_trait]
impl GraphBuilder for DefaultGraphBuilder {
    async fn build_links(
        &self,
        new_facts: &[Fact],
        bank_id: BankId,
        store: &dyn MemoryStore,
    ) -> Result<Vec<GraphLink>> {
        if new_facts.is_empty() {
            return Ok(Vec::new());
        }

        // Fetch existing facts in the bank for comparison
        let existing_facts = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![
                        NetworkType::World,
                        NetworkType::Experience,
                        NetworkType::Observation,
                    ]),
                    ..Default::default()
                },
            )
            .await?;

        // Build each link type
        let temporal_links = self.build_temporal_links(new_facts, &existing_facts);
        let entity_links = self.build_entity_links(new_facts, &existing_facts);
        let semantic_links = self
            .build_semantic_links(new_facts, &existing_facts)
            .await?;
        let causal_links = self
            .build_causal_links(new_facts, &existing_facts, &entity_links, &temporal_links)
            .await?;

        let mut all_links = Vec::new();
        all_links.extend(temporal_links);
        all_links.extend(entity_links);
        all_links.extend(semantic_links);
        all_links.extend(causal_links);

        // Store all links
        if !all_links.is_empty() {
            store.insert_links(&all_links).await?;
        }

        Ok(all_links)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::types::{FactId, FactType, TemporalRange};
    use chrono::Utc;

    fn make_fact(content: &str, entity_ids: Vec<crate::types::EntityId>) -> Fact {
        Fact {
            id: FactId::new(),
            bank_id: BankId::new(),
            content: content.into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_ids,
            temporal_range: None,
            embedding: None,
            confidence: None,
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            consolidated_at: None,
        }
    }

    #[test]
    fn entity_links_shared_entities() {
        let builder = DefaultGraphBuilder {
            llm: Arc::new(crate::llm::mock::MockLlmClient::new()),
            config: GraphConfig::default(),
        };

        let entity_a = crate::types::EntityId::new();
        let entity_b = crate::types::EntityId::new();

        let new_facts = vec![make_fact("fact about A and B", vec![entity_a, entity_b])];
        let existing = vec![
            make_fact("existing fact about A", vec![entity_a]),
            make_fact("unrelated fact", vec![]),
        ];

        let links = builder.build_entity_links(&new_facts, &existing);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].link_type, LinkType::Entity);
        // shared=1 (entity_a), total_unique=2 (a,b) → weight = 0.5
        assert!((links[0].weight - 0.5).abs() < 0.01);
    }

    #[test]
    fn temporal_links_close_facts() {
        let builder = DefaultGraphBuilder {
            llm: Arc::new(crate::llm::mock::MockLlmClient::new()),
            config: GraphConfig::default(),
        };

        let now = Utc::now();
        let yesterday = now - Duration::days(1);

        let mut fact_a = make_fact("fact A", vec![]);
        fact_a.temporal_range = Some(TemporalRange {
            start: Some(now),
            end: None,
        });

        let mut fact_b = make_fact("fact B", vec![]);
        fact_b.temporal_range = Some(TemporalRange {
            start: Some(yesterday),
            end: None,
        });

        let links = builder.build_temporal_links(&[fact_a], &[fact_b]);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].link_type, LinkType::Temporal);
        assert!(links[0].weight > 0.9); // 1 day apart out of 30 max → high weight
    }

    #[test]
    fn temporal_links_distant_facts_no_link() {
        let builder = DefaultGraphBuilder {
            llm: Arc::new(crate::llm::mock::MockLlmClient::new()),
            config: GraphConfig {
                temporal_max_days: 30,
                ..Default::default()
            },
        };

        let now = Utc::now();
        let long_ago = now - Duration::days(365);

        let mut fact_a = make_fact("fact A", vec![]);
        fact_a.temporal_range = Some(TemporalRange {
            start: Some(now),
            end: None,
        });

        let mut fact_b = make_fact("fact B", vec![]);
        fact_b.temporal_range = Some(TemporalRange {
            start: Some(long_ago),
            end: None,
        });

        let links = builder.build_temporal_links(&[fact_a], &[fact_b]);
        assert!(links.is_empty());
    }
}
