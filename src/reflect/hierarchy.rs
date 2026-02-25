//! Hierarchy assembler — 4-tier recall with budget allocation and formatting.

use std::fmt::Write;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::recall::RecallPipeline;
use crate::types::{AssembledContext, BankId, Fact, NetworkType, RecallQuery};

/// Assembles memory context from multiple tiers for the reflect pipeline.
#[async_trait]
pub trait HierarchyAssembler: Send + Sync {
    /// Assemble context from the memory hierarchy within the given token budget.
    async fn assemble(
        &self,
        query: &str,
        bank_id: BankId,
        budget: usize,
    ) -> Result<AssembledContext>;
}

/// Tier configuration: network filter and base budget percentage.
struct Tier {
    networks: Vec<NetworkType>,
    base_pct: f32,
}

/// Default hierarchy assembler using the recall pipeline for each tier.
pub struct DefaultHierarchyAssembler {
    recall: Arc<dyn RecallPipeline>,
}

impl DefaultHierarchyAssembler {
    /// Create a new hierarchy assembler.
    pub fn new(recall: Arc<dyn RecallPipeline>) -> Self {
        Self { recall }
    }
}

#[async_trait]
impl HierarchyAssembler for DefaultHierarchyAssembler {
    async fn assemble(
        &self,
        query: &str,
        bank_id: BankId,
        budget: usize,
    ) -> Result<AssembledContext> {
        let tiers = [
            Tier {
                networks: vec![NetworkType::MentalModel],
                base_pct: 0.15,
            },
            Tier {
                networks: vec![NetworkType::Observation],
                base_pct: 0.30,
            },
            Tier {
                networks: vec![NetworkType::World, NetworkType::Experience],
                base_pct: 0.40,
            },
            Tier {
                networks: vec![NetworkType::Opinion],
                base_pct: 0.15,
            },
        ];

        let mut mental_models = Vec::new();
        let mut observations = Vec::new();
        let mut raw_facts = Vec::new();
        let mut opinions = Vec::new();
        let mut total_tokens = 0usize;
        let mut remaining_budget = budget;

        for (i, tier) in tiers.iter().enumerate() {
            // Calculate tier budget: base allocation + any remaining redistribution
            let base_alloc = (budget as f32 * tier.base_pct) as usize;
            // If we're behind on spending, the remaining budget includes unspent from previous tiers
            let expected_spent = (budget as f32
                * tiers[..i].iter().map(|t| t.base_pct).sum::<f32>())
                as usize;
            let actual_remaining = remaining_budget;
            let bonus = expected_spent.saturating_sub(total_tokens);
            let tier_budget = (base_alloc + bonus).min(actual_remaining);

            if tier_budget == 0 {
                continue;
            }

            let result = self
                .recall
                .recall(&RecallQuery {
                    bank_id,
                    query: query.to_string(),
                    budget_tokens: tier_budget,
                    network_filter: Some(tier.networks.clone()),
                    temporal_anchor: None,
                    tag_filter: None,
                })
                .await?;

            let tier_tokens = result.total_tokens;
            total_tokens += tier_tokens;
            remaining_budget = remaining_budget.saturating_sub(tier_tokens);

            let facts: Vec<Fact> = result.facts.into_iter().map(|sf| sf.fact).collect();

            match tier.networks[0] {
                NetworkType::MentalModel => mental_models = facts,
                NetworkType::Observation => observations = facts,
                NetworkType::World => raw_facts = facts,
                NetworkType::Opinion => opinions = facts,
                _ => {}
            }
        }

        let formatted = format_context(&mental_models, &observations, &raw_facts, &opinions);

        Ok(AssembledContext {
            mental_models,
            observations,
            raw_facts,
            opinions,
            total_tokens,
            formatted,
        })
    }
}

/// Format assembled facts into a prompt-ready string with section headers.
fn format_context(
    mental_models: &[Fact],
    observations: &[Fact],
    raw_facts: &[Fact],
    opinions: &[Fact],
) -> String {
    let mut out = String::new();

    if !mental_models.is_empty() {
        writeln!(out, "## Mental Models").unwrap();
        for f in mental_models {
            writeln!(out, "- [{}] {}", f.id, f.content).unwrap();
        }
        writeln!(out).unwrap();
    }

    if !observations.is_empty() {
        writeln!(out, "## Observations").unwrap();
        for f in observations {
            writeln!(out, "- [{}] {}", f.id, f.content).unwrap();
        }
        writeln!(out).unwrap();
    }

    if !raw_facts.is_empty() {
        writeln!(out, "## Facts").unwrap();
        for f in raw_facts {
            writeln!(out, "- [{}] {}", f.id, f.content).unwrap();
        }
        writeln!(out).unwrap();
    }

    if !opinions.is_empty() {
        writeln!(out, "## Opinions").unwrap();
        for f in opinions {
            let conf = f.confidence.unwrap_or(0.5);
            writeln!(out, "- [{}] (confidence: {:.2}) {}", f.id, conf, f.content).unwrap();
        }
        writeln!(out).unwrap();
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::mock::MockEmbeddings;
    use crate::embedding::EmbeddingClient;
    use crate::recall::budget::EstimateTokenizer;
    use crate::recall::graph::{GraphRetriever, GraphRetrieverConfig};
    use crate::recall::keyword::KeywordRetriever;
    use crate::recall::reranker::NoOpReranker;
    use crate::recall::semantic::SemanticRetriever;
    use crate::recall::temporal::TemporalRetriever;
    use crate::recall::DefaultRecallPipeline;
    use crate::storage::mock::MockMemoryStore;
    use crate::storage::MemoryStore;
    use crate::types::*;
    use chrono::Utc;

    fn make_fact(bank: BankId, content: &str, network: NetworkType, embedding: Vec<f32>) -> Fact {
        Fact {
            id: FactId::new(),
            bank_id: bank,
            content: content.into(),
            fact_type: FactType::World,
            network,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(embedding),
            confidence: if network == NetworkType::Opinion {
                Some(0.75)
            } else {
                None
            },
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn build_pipeline(
        store: Arc<MockMemoryStore>,
        embeddings: Arc<MockEmbeddings>,
    ) -> Arc<DefaultRecallPipeline> {
        Arc::new(DefaultRecallPipeline::new(
            Box::new(SemanticRetriever::new(store.clone(), embeddings.clone(), 20)),
            Box::new(KeywordRetriever::new(store.clone(), 20)),
            Box::new(GraphRetriever::new(
                store.clone(),
                embeddings.clone(),
                GraphRetrieverConfig::default(),
            )),
            Box::new(TemporalRetriever::new(store.clone())),
            Box::new(NoOpReranker),
            Box::new(EstimateTokenizer),
            60.0,
            50,
        ))
    }

    #[tokio::test]
    async fn tier_ordering_in_output() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = BankId::new();

        let emb = embeddings.embed(&["test"]).await.unwrap();

        let facts = vec![
            make_fact(bank, "Mental model about testing", NetworkType::MentalModel, emb[0].clone()),
            make_fact(bank, "Observation about tests", NetworkType::Observation, emb[0].clone()),
            make_fact(bank, "World fact about testing", NetworkType::World, emb[0].clone()),
            make_fact(bank, "Opinion on test quality", NetworkType::Opinion, emb[0].clone()),
        ];
        store.insert_facts(&facts).await.unwrap();

        let pipeline = build_pipeline(store, embeddings);
        let assembler = DefaultHierarchyAssembler::new(pipeline);

        let ctx = assembler.assemble("testing", bank, 2000).await.unwrap();

        // Verify section ordering in formatted output
        let mm_pos = ctx.formatted.find("## Mental Models");
        let obs_pos = ctx.formatted.find("## Observations");
        let facts_pos = ctx.formatted.find("## Facts");
        let opinions_pos = ctx.formatted.find("## Opinions");

        assert!(mm_pos.is_some());
        assert!(obs_pos.is_some());
        assert!(facts_pos.is_some());
        assert!(opinions_pos.is_some());

        assert!(mm_pos.unwrap() < obs_pos.unwrap());
        assert!(obs_pos.unwrap() < facts_pos.unwrap());
        assert!(facts_pos.unwrap() < opinions_pos.unwrap());
    }

    #[tokio::test]
    async fn formatted_includes_fact_ids() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = BankId::new();

        let emb = embeddings.embed(&["data"]).await.unwrap();
        let fact = make_fact(bank, "A world fact", NetworkType::World, emb[0].clone());
        let fact_id = fact.id;
        store.insert_facts(&[fact]).await.unwrap();

        let pipeline = build_pipeline(store, embeddings);
        let assembler = DefaultHierarchyAssembler::new(pipeline);

        let ctx = assembler.assemble("data", bank, 2000).await.unwrap();
        assert!(ctx.formatted.contains(&format!("[{}]", fact_id)));
    }

    #[tokio::test]
    async fn opinions_show_confidence() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = BankId::new();

        let emb = embeddings.embed(&["opinion"]).await.unwrap();
        let fact = make_fact(bank, "Strong opinion here", NetworkType::Opinion, emb[0].clone());
        store.insert_facts(&[fact]).await.unwrap();

        let pipeline = build_pipeline(store, embeddings);
        let assembler = DefaultHierarchyAssembler::new(pipeline);

        let ctx = assembler.assemble("opinion", bank, 2000).await.unwrap();
        assert!(ctx.formatted.contains("confidence: 0.75"));
    }

    #[tokio::test]
    async fn empty_tiers_produce_empty_sections() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = BankId::new();

        let pipeline = build_pipeline(store, embeddings);
        let assembler = DefaultHierarchyAssembler::new(pipeline);

        let ctx = assembler.assemble("anything", bank, 2000).await.unwrap();
        assert!(ctx.mental_models.is_empty());
        assert!(ctx.observations.is_empty());
        assert!(ctx.raw_facts.is_empty());
        assert!(ctx.opinions.is_empty());
        assert_eq!(ctx.total_tokens, 0);
        assert!(ctx.formatted.is_empty());
    }

    #[tokio::test]
    async fn budget_redistribution_on_empty_tiers() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = BankId::new();

        // Only insert World facts — MentalModel and Observation tiers will be empty
        let emb = embeddings.embed(&["world data"]).await.unwrap();
        for i in 0..5 {
            let fact = make_fact(
                bank,
                &format!("World fact {i} about interesting data"),
                NetworkType::World,
                emb[0].clone(),
            );
            store.insert_facts(&[fact]).await.unwrap();
        }

        let pipeline = build_pipeline(store, embeddings);
        let assembler = DefaultHierarchyAssembler::new(pipeline);

        // With budget redistribution, World tier should get more than its base 40%
        let ctx = assembler.assemble("world data", bank, 200).await.unwrap();
        assert!(!ctx.raw_facts.is_empty());
        // The World tier should get the unused MentalModel + Observation budget
        assert!(ctx.total_tokens > 0);
    }
}
