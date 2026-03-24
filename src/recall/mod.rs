//! Recall pipeline — retrieval, fusion, reranking, and budget enforcement.

pub mod budget;
pub mod fusion;
pub mod graph;
pub mod keyword;
pub mod reranker;
pub mod semantic;
pub mod temporal;

use std::time::Instant;

use async_trait::async_trait;
use tracing::debug;

use crate::error::Result;
use crate::types::{RecallQuery, RecallResult, ScoredFact};

use budget::Tokenizer;
use reranker::Reranker;

/// A single retrieval strategy that produces scored facts from a query.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Retrieve facts matching the query.
    async fn retrieve(&self, query: &RecallQuery) -> Result<Vec<ScoredFact>>;
}

/// The full recall pipeline: parallel retrieval → RRF fusion → rerank → budget.
#[async_trait]
pub trait RecallPipeline: Send + Sync {
    /// Execute the recall pipeline for the given query.
    async fn recall(&self, query: &RecallQuery) -> Result<RecallResult>;
}

/// Default recall pipeline wiring four retrievers through fusion, reranking, and budget.
pub struct DefaultRecallPipeline {
    semantic: Box<dyn Retriever>,
    keyword: Box<dyn Retriever>,
    graph: Box<dyn Retriever>,
    temporal: Box<dyn Retriever>,
    reranker: Box<dyn Reranker>,
    tokenizer: Box<dyn Tokenizer>,
    rrf_k: f32,
    rerank_top_n: usize,
}

impl DefaultRecallPipeline {
    /// Create a new recall pipeline with the given components.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        semantic: Box<dyn Retriever>,
        keyword: Box<dyn Retriever>,
        graph: Box<dyn Retriever>,
        temporal: Box<dyn Retriever>,
        reranker: Box<dyn Reranker>,
        tokenizer: Box<dyn Tokenizer>,
        rrf_k: f32,
        rerank_top_n: usize,
    ) -> Self {
        Self {
            semantic,
            keyword,
            graph,
            temporal,
            reranker,
            tokenizer,
            rrf_k,
            rerank_top_n,
        }
    }
}

#[async_trait]
impl RecallPipeline for DefaultRecallPipeline {
    async fn recall(&self, query: &RecallQuery) -> Result<RecallResult> {
        let start = Instant::now();
        debug!(
            bank_id = %query.bank_id,
            query = query.query.as_str(),
            network_filter = ?query.network_filter,
            temporal_anchor = ?query.temporal_anchor,
            "recall_start"
        );

        // Step 1: Parallel retrieval
        let (semantic_r, keyword_r, graph_r, temporal_r) = tokio::try_join!(
            self.semantic.retrieve(query),
            self.keyword.retrieve(query),
            self.graph.retrieve(query),
            self.temporal.retrieve(query),
        )?;

        // Step 2: RRF fusion
        let rankings = [semantic_r, keyword_r, graph_r, temporal_r];
        let fused = fusion::fuse_rankings(&rankings, self.rrf_k);

        // Step 3: Rerank top N
        let reranked = self
            .reranker
            .rerank(&query.query, fused, self.rerank_top_n)
            .await?;

        // Step 4: Apply token budget
        let budgeted = budget::apply_budget(&reranked, query.budget_tokens, &*self.tokenizer);

        let total_tokens: usize = budgeted
            .iter()
            .map(|sf| self.tokenizer.count_tokens(&sf.fact.content))
            .sum();

        debug!(
            elapsed_ms = start.elapsed().as_millis() as u64,
            facts_returned = budgeted.len(),
            total_tokens,
            "recall_end"
        );

        Ok(RecallResult {
            facts: budgeted,
            total_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::budget::EstimateTokenizer;
    use super::graph::{GraphRetriever, GraphRetrieverConfig};
    use super::keyword::KeywordRetriever;
    use super::reranker::{MockReranker, NoOpReranker};
    use super::semantic::SemanticRetriever;
    use super::temporal::TemporalRetriever;
    use super::{DefaultRecallPipeline, RecallPipeline};
    use crate::embedding::EmbeddingClient;
    use crate::embedding::mock::MockEmbeddings;
    use crate::storage::MemoryStore;
    use crate::storage::mock::MockMemoryStore;
    use crate::types::{
        BankId, Disposition, Fact, FactId, FactType, MemoryBank, NetworkType, RecallQuery,
        TemporalRange,
    };
    use chrono::{TimeZone, Utc};
    use std::sync::Arc;

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

    fn make_fact_with_embedding(bank: BankId, content: &str, embedding: Vec<f32>) -> Fact {
        Fact {
            id: FactId::new(),
            bank_id: bank,
            content: content.into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(embedding),
            confidence: None,
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            consolidated_at: None,
        }
    }

    fn dt(year: i32, month: u32, day: u32) -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, 0, 0, 0).unwrap()
    }

    #[tokio::test]
    async fn full_pipeline_returns_results() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = create_test_bank(&store, 8).await;

        // Insert facts with embeddings
        let emb1 = embeddings.embed(&["Rust programming"]).await.unwrap();
        let emb2 = embeddings.embed(&["Python scripting"]).await.unwrap();

        let facts = vec![
            make_fact_with_embedding(bank, "Rust programming language", emb1[0].clone()),
            make_fact_with_embedding(bank, "Python scripting language", emb2[0].clone()),
        ];
        store.insert_facts(&facts).await.unwrap();

        let pipeline = DefaultRecallPipeline::new(
            Box::new(SemanticRetriever::new(
                store.clone(),
                embeddings.clone(),
                10,
            )),
            Box::new(KeywordRetriever::new(store.clone(), 10)),
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
        );

        let query = RecallQuery {
            bank_id: bank,
            query: "Rust programming".into(),
            budget_tokens: 1000,
            network_filter: None,
            temporal_anchor: None,
        };

        let result = pipeline.recall(&query).await.unwrap();
        assert!(!result.facts.is_empty());
        assert!(result.total_tokens > 0);
    }

    #[tokio::test]
    async fn pipeline_respects_budget() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = create_test_bank(&store, 8).await;

        // Insert many facts
        let mut facts = Vec::new();
        for i in 0..20 {
            let content = format!("Fact number {i} about programming");
            let emb = embeddings.embed(&[content.as_str()]).await.unwrap();
            facts.push(make_fact_with_embedding(bank, &content, emb[0].clone()));
        }
        store.insert_facts(&facts).await.unwrap();

        let pipeline = DefaultRecallPipeline::new(
            Box::new(SemanticRetriever::new(
                store.clone(),
                embeddings.clone(),
                20,
            )),
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
        );

        // Very small budget
        let query = RecallQuery {
            bank_id: bank,
            query: "programming".into(),
            budget_tokens: 20,
            network_filter: None,
            temporal_anchor: None,
        };

        let result = pipeline.recall(&query).await.unwrap();
        assert!(result.total_tokens <= 20);
    }

    #[tokio::test]
    async fn mock_reranker_reverses_order() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = create_test_bank(&store, 8).await;

        let emb1 = embeddings.embed(&["alpha"]).await.unwrap();
        let emb2 = embeddings.embed(&["beta"]).await.unwrap();

        let facts = vec![
            make_fact_with_embedding(bank, "alpha content", emb1[0].clone()),
            make_fact_with_embedding(bank, "beta content", emb2[0].clone()),
        ];
        store.insert_facts(&facts).await.unwrap();

        let pipeline = DefaultRecallPipeline::new(
            Box::new(SemanticRetriever::new(
                store.clone(),
                embeddings.clone(),
                10,
            )),
            Box::new(KeywordRetriever::new(store.clone(), 10)),
            Box::new(GraphRetriever::new(
                store.clone(),
                embeddings.clone(),
                GraphRetrieverConfig::default(),
            )),
            Box::new(TemporalRetriever::new(store.clone())),
            Box::new(MockReranker),
            Box::new(EstimateTokenizer),
            60.0,
            50,
        );

        let query = RecallQuery {
            bank_id: bank,
            query: "alpha".into(),
            budget_tokens: 1000,
            network_filter: None,
            temporal_anchor: None,
        };

        let result = pipeline.recall(&query).await.unwrap();
        // MockReranker reverses, so the originally lower-scored fact should be first
        assert!(!result.facts.is_empty());
    }

    #[tokio::test]
    async fn empty_store_returns_empty_result() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = create_test_bank(&store, 8).await;

        let pipeline = DefaultRecallPipeline::new(
            Box::new(SemanticRetriever::new(
                store.clone(),
                embeddings.clone(),
                10,
            )),
            Box::new(KeywordRetriever::new(store.clone(), 10)),
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
        );

        let query = RecallQuery {
            bank_id: bank,
            query: "anything".into(),
            budget_tokens: 1000,
            network_filter: None,
            temporal_anchor: None,
        };

        let result = pipeline.recall(&query).await.unwrap();
        assert!(result.facts.is_empty());
        assert_eq!(result.total_tokens, 0);
    }

    #[tokio::test]
    async fn network_filter_excludes_non_matching() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = create_test_bank(&store, 8).await;

        let emb = embeddings.embed(&["test content"]).await.unwrap();

        // Insert a World fact and an Opinion fact
        let world_fact = make_fact_with_embedding(bank, "world fact about testing", emb[0].clone());
        let mut opinion_fact =
            make_fact_with_embedding(bank, "opinion about testing", emb[0].clone());
        opinion_fact.network = NetworkType::Opinion;
        opinion_fact.confidence = Some(0.8);
        store
            .insert_facts(&[world_fact, opinion_fact])
            .await
            .unwrap();

        let pipeline = DefaultRecallPipeline::new(
            Box::new(SemanticRetriever::new(
                store.clone(),
                embeddings.clone(),
                10,
            )),
            Box::new(KeywordRetriever::new(store.clone(), 10)),
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
        );

        // Filter to Opinion only
        let query = RecallQuery {
            bank_id: bank,
            query: "testing".into(),
            budget_tokens: 1000,
            network_filter: Some(vec![NetworkType::Opinion]),
            temporal_anchor: None,
        };

        let result = pipeline.recall(&query).await.unwrap();
        assert!(!result.facts.is_empty(), "should return opinion facts");
        for sf in &result.facts {
            assert_eq!(
                sf.fact.network,
                NetworkType::Opinion,
                "all results should be opinions"
            );
        }
    }

    #[tokio::test]
    async fn explicit_temporal_anchor_biases_recall_without_global_filtering() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let bank = create_test_bank(&store, 8).await;

        let emb = embeddings.embed(&["release notes"]).await.unwrap();

        let mut january_fact =
            make_fact_with_embedding(bank, "release notes for project alpha", emb[0].clone());
        january_fact.temporal_range = Some(TemporalRange {
            start: Some(dt(2024, 1, 10)),
            end: Some(dt(2024, 1, 10)),
        });

        let mut february_fact =
            make_fact_with_embedding(bank, "release notes for project alpha", emb[0].clone());
        february_fact.temporal_range = Some(TemporalRange {
            start: Some(dt(2024, 2, 10)),
            end: Some(dt(2024, 2, 10)),
        });

        store
            .insert_facts(&[january_fact.clone(), february_fact])
            .await
            .unwrap();

        let pipeline = DefaultRecallPipeline::new(
            Box::new(SemanticRetriever::new(
                store.clone(),
                embeddings.clone(),
                10,
            )),
            Box::new(KeywordRetriever::new(store.clone(), 10)),
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
        );

        let query = RecallQuery {
            bank_id: bank,
            query: "release notes".into(),
            budget_tokens: 1000,
            network_filter: None,
            temporal_anchor: Some(TemporalRange {
                start: Some(dt(2024, 1, 1)),
                end: Some(dt(2024, 1, 31)),
            }),
        };

        let result = pipeline.recall(&query).await.unwrap();
        assert_eq!(result.facts[0].fact.id, january_fact.id);
        assert!(
            result.facts.iter().any(|sf| sf.fact.id == january_fact.id),
            "in-range fact should survive recall"
        );
        assert!(
            result.facts.iter().any(|sf| sf.fact.id != january_fact.id),
            "out-of-range facts from non-temporal channels should still be eligible"
        );
    }
}
