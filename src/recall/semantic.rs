//! Semantic retriever — embed query then vector search.

use std::sync::Arc;

use async_trait::async_trait;

use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::storage::MemoryStore;
use crate::types::{RecallQuery, RetrievalSource, ScoredFact};

use super::Retriever;

/// Retrieves facts by embedding the query and running vector similarity search.
pub struct SemanticRetriever {
    store: Arc<dyn MemoryStore>,
    embeddings: Arc<dyn EmbeddingClient>,
    limit: usize,
}

impl SemanticRetriever {
    /// Create a new semantic retriever.
    pub fn new(
        store: Arc<dyn MemoryStore>,
        embeddings: Arc<dyn EmbeddingClient>,
        limit: usize,
    ) -> Self {
        Self {
            store,
            embeddings,
            limit,
        }
    }
}

#[async_trait]
impl Retriever for SemanticRetriever {
    async fn retrieve(&self, query: &RecallQuery) -> Result<Vec<ScoredFact>> {
        // Validate embedding dimensions match the bank's config
        let bank = self.store.get_bank(query.bank_id).await?;
        if bank.embedding_dimensions > 0 {
            let client_dims = self.embeddings.dimensions() as u16;
            if client_dims != bank.embedding_dimensions {
                return Err(crate::error::Error::EmbeddingDimensionMismatch {
                    model: bank.embedding_model.clone(),
                    expected: bank.embedding_dimensions,
                    actual: client_dims,
                });
            }
        }

        let vecs = self.embeddings.embed(&[&query.query]).await?;
        let embedding = &vecs[0];
        let mut results = self
            .store
            .vector_search(embedding, query.bank_id, self.limit)
            .await?;
        for sf in &mut results {
            sf.sources = vec![RetrievalSource::Semantic];
        }
        Ok(results)
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
    async fn retrieves_by_embedding_similarity() {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));

        // Create a bank so get_bank() works during dimension validation
        let bank_id = BankId::new();
        let bank_obj = MemoryBank {
            id: bank_id,
            name: "test".into(),
            mission: String::new(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: "mock".into(),
            embedding_dimensions: 8,
        };
        store.create_bank(&bank_obj).await.unwrap();
        let bank = bank_id;
        let emb1 = embeddings.embed(&["rust programming"]).await.unwrap();
        let emb2 = embeddings.embed(&["cooking recipes"]).await.unwrap();

        let facts = vec![
            Fact {
                id: FactId::new(),
                bank_id: bank,
                content: "rust programming".into(),
                fact_type: FactType::World,
                network: NetworkType::World,
                entity_ids: vec![],
                temporal_range: None,
                embedding: Some(emb1[0].clone()),
                confidence: None,
                evidence_ids: vec![],
                source_turn_id: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            Fact {
                id: FactId::new(),
                bank_id: bank,
                content: "cooking recipes".into(),
                fact_type: FactType::World,
                network: NetworkType::World,
                entity_ids: vec![],
                temporal_range: None,
                embedding: Some(emb2[0].clone()),
                confidence: None,
                evidence_ids: vec![],
                source_turn_id: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        ];
        store.insert_facts(&facts).await.unwrap();

        let retriever = SemanticRetriever::new(store, embeddings, 10);
        let query = RecallQuery {
            bank_id: bank,
            query: "rust programming".into(),
            budget_tokens: 1000,
            network_filter: None,
            temporal_anchor: None,
            tag_filter: None,
        };

        let results = retriever.retrieve(&query).await.unwrap();
        assert!(!results.is_empty());
        // The exact query text should match itself perfectly
        assert_eq!(results[0].fact.content, "rust programming");
        assert!(results[0].sources.contains(&RetrievalSource::Semantic));
    }
}
