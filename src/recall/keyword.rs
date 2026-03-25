//! Keyword retriever — wraps MemoryStore::keyword_search.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::storage::MemoryStore;
use crate::types::{RecallQuery, RetrievalSource, ScoredFact};

use super::Retriever;

/// Retrieves facts by keyword/full-text search.
pub struct KeywordRetriever {
    store: Arc<dyn MemoryStore>,
    limit: usize,
}

impl KeywordRetriever {
    /// Create a new keyword retriever.
    pub fn new(store: Arc<dyn MemoryStore>, limit: usize) -> Self {
        Self { store, limit }
    }
}

#[async_trait]
impl Retriever for KeywordRetriever {
    async fn retrieve(&self, query: &RecallQuery) -> Result<Vec<ScoredFact>> {
        let mut results = self
            .store
            .keyword_search(
                &query.query,
                query.bank_id,
                self.limit,
                query.network_filter.as_deref(),
            )
            .await?;
        for sf in &mut results {
            sf.sources = vec![RetrievalSource::Keyword];
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mock::MockMemoryStore;
    use crate::types::*;
    use chrono::Utc;

    #[tokio::test]
    async fn retrieves_by_keyword() {
        let store = Arc::new(MockMemoryStore::new());
        let bank = BankId::new();
        let facts = vec![
            Fact {
                id: FactId::new(),
                bank_id: bank,
                content: "Rust is a systems programming language".into(),
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
            Fact {
                id: FactId::new(),
                bank_id: bank,
                content: "Python is interpreted".into(),
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

        let retriever = KeywordRetriever::new(store, 10);
        let query = RecallQuery::new(bank, "Rust").with_budget_tokens(1000);

        let results = retriever.retrieve(&query).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].fact.content.contains("Rust"));
        assert!(results[0].sources.contains(&RetrievalSource::Keyword));
    }
}
