//! Reranker trait and implementations for recall pipeline.

use async_trait::async_trait;

use crate::error::Result;
use crate::types::ScoredFact;

/// Reranks a set of scored facts, typically using a cross-encoder or similar model.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank the given facts for the query, returning at most `top_k` results.
    async fn rerank(
        &self,
        query: &str,
        facts: Vec<ScoredFact>,
        top_k: usize,
    ) -> Result<Vec<ScoredFact>>;
}

/// No-op reranker that simply truncates to top_k.
pub struct NoOpReranker;

#[async_trait]
impl Reranker for NoOpReranker {
    async fn rerank(
        &self,
        _query: &str,
        mut facts: Vec<ScoredFact>,
        top_k: usize,
    ) -> Result<Vec<ScoredFact>> {
        facts.truncate(top_k);
        Ok(facts)
    }
}

/// Mock reranker that reverses order (for testing pipeline wiring).
pub struct MockReranker;

#[async_trait]
impl Reranker for MockReranker {
    async fn rerank(
        &self,
        _query: &str,
        mut facts: Vec<ScoredFact>,
        top_k: usize,
    ) -> Result<Vec<ScoredFact>> {
        facts.reverse();
        facts.truncate(top_k);
        Ok(facts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::Utc;

    fn make_scored(content: &str, score: f32) -> ScoredFact {
        ScoredFact {
            fact: Fact {
                id: FactId::new(),
                bank_id: BankId::new(),
                content: content.into(),
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
            },
            score,
            sources: vec![RetrievalSource::Semantic],
        }
    }

    #[tokio::test]
    async fn noop_truncates() {
        let facts = vec![
            make_scored("a", 1.0),
            make_scored("b", 0.9),
            make_scored("c", 0.8),
        ];
        let result = NoOpReranker.rerank("query", facts, 2).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].fact.content, "a");
        assert_eq!(result[1].fact.content, "b");
    }

    #[tokio::test]
    async fn mock_reverses() {
        let facts = vec![
            make_scored("a", 1.0),
            make_scored("b", 0.9),
            make_scored("c", 0.8),
        ];
        let result = MockReranker.rerank("query", facts, 3).await.unwrap();
        assert_eq!(result[0].fact.content, "c");
        assert_eq!(result[1].fact.content, "b");
        assert_eq!(result[2].fact.content, "a");
    }
}
