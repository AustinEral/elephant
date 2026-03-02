//! Hierarchy assembler — unified recall with post-hoc network grouping and formatting.

use std::fmt::Write;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::recall::RecallPipeline;
use crate::types::{AssembledContext, BankId, Fact, NetworkType, RecallQuery};

/// Assembles memory context from the recall pipeline for the reflect pipeline.
#[async_trait]
pub trait HierarchyAssembler: Send + Sync {
    /// Assemble context from memory within the given token budget.
    async fn assemble(
        &self,
        query: &str,
        bank_id: BankId,
        budget: usize,
    ) -> Result<AssembledContext>;
}

/// Default hierarchy assembler using a single unified recall call.
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
        // Single unified recall — no network filter, full budget.
        let result = self
            .recall
            .recall(&RecallQuery {
                bank_id,
                query: query.to_string(),
                budget_tokens: budget,
                network_filter: None,
                temporal_anchor: None,
            })
            .await?;

        let total_tokens = result.total_tokens;

        // Separate results by network type for formatted output sections.
        let mut observations = Vec::new();
        let mut raw_facts = Vec::new();
        let mut opinions = Vec::new();

        for sf in result.facts {
            match sf.fact.network {
                NetworkType::Observation => observations.push(sf.fact),
                NetworkType::Opinion => opinions.push(sf.fact),
                NetworkType::World | NetworkType::Experience => raw_facts.push(sf.fact),
            }
        }

        let formatted = format_context(&observations, &raw_facts, &opinions);

        Ok(AssembledContext {
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
    observations: &[Fact],
    raw_facts: &[Fact],
    opinions: &[Fact],
) -> String {
    let mut out = String::new();

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

    async fn create_test_bank(store: &MockMemoryStore, dims: u16) -> BankId {
        let id = BankId::new();
        let bank = MemoryBank {
            id,
            name: "test".into(),
            mission: String::new(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: "mock".into(),
            embedding_dimensions: dims,
        };
        store.create_bank(&bank).await.unwrap();
        id
    }

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
            consolidated_at: None,
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
    async fn section_ordering_in_output() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = create_test_bank(&store, 8).await;

        let emb = embeddings.embed(&["test"]).await.unwrap();

        let facts = vec![
            make_fact(bank, "Observation about tests", NetworkType::Observation, emb[0].clone()),
            make_fact(bank, "World fact about testing", NetworkType::World, emb[0].clone()),
            make_fact(bank, "Opinion on test quality", NetworkType::Opinion, emb[0].clone()),
        ];
        store.insert_facts(&facts).await.unwrap();

        let pipeline = build_pipeline(store, embeddings);
        let assembler = DefaultHierarchyAssembler::new(pipeline);

        let ctx = assembler.assemble("testing", bank, 2000).await.unwrap();

        // Verify section ordering in formatted output
        let obs_pos = ctx.formatted.find("## Observations");
        let facts_pos = ctx.formatted.find("## Facts");
        let opinions_pos = ctx.formatted.find("## Opinions");

        assert!(obs_pos.is_some());
        assert!(facts_pos.is_some());
        assert!(opinions_pos.is_some());

        assert!(obs_pos.unwrap() < facts_pos.unwrap());
        assert!(facts_pos.unwrap() < opinions_pos.unwrap());
    }

    #[tokio::test]
    async fn formatted_includes_fact_ids() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = create_test_bank(&store, 8).await;

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
        let bank = create_test_bank(&store, 8).await;

        let emb = embeddings.embed(&["opinion"]).await.unwrap();
        let fact = make_fact(bank, "Strong opinion here", NetworkType::Opinion, emb[0].clone());
        store.insert_facts(&[fact]).await.unwrap();

        let pipeline = build_pipeline(store, embeddings);
        let assembler = DefaultHierarchyAssembler::new(pipeline);

        let ctx = assembler.assemble("opinion", bank, 2000).await.unwrap();
        assert!(ctx.formatted.contains("confidence: 0.75"));
    }

    #[tokio::test]
    async fn empty_bank_produces_empty_context() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = create_test_bank(&store, 8).await;

        let pipeline = build_pipeline(store, embeddings);
        let assembler = DefaultHierarchyAssembler::new(pipeline);

        let ctx = assembler.assemble("anything", bank, 2000).await.unwrap();
        assert!(ctx.observations.is_empty());
        assert!(ctx.raw_facts.is_empty());
        assert!(ctx.opinions.is_empty());
        assert_eq!(ctx.total_tokens, 0);
        assert!(ctx.formatted.is_empty());
    }

    #[tokio::test]
    async fn unified_recall_uses_full_budget() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = create_test_bank(&store, 8).await;

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

        let ctx = assembler.assemble("world data", bank, 200).await.unwrap();
        assert!(!ctx.raw_facts.is_empty());
        assert!(ctx.total_tokens > 0);
    }
}
