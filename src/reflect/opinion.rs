//! Opinion manager — CRUD operations for the Opinion memory network.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;

use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::storage::MemoryStore;
use crate::types::{BankId, Fact, FactId, FactType, NetworkType};

/// Manages opinions within a memory bank.
#[async_trait]
pub trait OpinionManager: Send + Sync {
    /// Find existing opinions relevant to a topic.
    async fn get_opinions(&self, bank_id: BankId, topic: &str) -> Result<Vec<Fact>>;

    /// Form a new opinion with initial evidence and confidence.
    async fn form_opinion(
        &self,
        bank_id: BankId,
        opinion: &str,
        evidence: &[FactId],
        confidence: f32,
    ) -> Result<FactId>;

    /// Strengthen an existing opinion by adding supporting evidence.
    async fn reinforce(
        &self,
        opinion_id: FactId,
        new_evidence: &[FactId],
        delta: f32,
    ) -> Result<()>;

    /// Weaken an existing opinion by adding contradicting evidence.
    async fn weaken(
        &self,
        opinion_id: FactId,
        contradicting_evidence: &[FactId],
        delta: f32,
    ) -> Result<()>;
}

/// Default opinion manager backed by the memory store and embedding client.
pub struct DefaultOpinionManager {
    store: Arc<dyn MemoryStore>,
    embeddings: Arc<dyn EmbeddingClient>,
}

impl DefaultOpinionManager {
    /// Create a new opinion manager.
    pub fn new(store: Arc<dyn MemoryStore>, embeddings: Arc<dyn EmbeddingClient>) -> Self {
        Self { store, embeddings }
    }
}

#[async_trait]
impl OpinionManager for DefaultOpinionManager {
    async fn get_opinions(&self, bank_id: BankId, topic: &str) -> Result<Vec<Fact>> {
        let embeddings = self.embeddings.embed(&[topic]).await?;
        let scored = self
            .store
            .vector_search(&embeddings[0], bank_id, 20, None)
            .await?;

        Ok(scored
            .into_iter()
            .filter(|sf| sf.fact.network == NetworkType::Opinion)
            .map(|sf| sf.fact)
            .collect())
    }

    async fn form_opinion(
        &self,
        bank_id: BankId,
        opinion: &str,
        evidence: &[FactId],
        confidence: f32,
    ) -> Result<FactId> {
        let embedding = self.embeddings.embed(&[opinion]).await?;
        let now = Utc::now();
        let fact = Fact {
            id: FactId::new(),
            bank_id,
            content: opinion.to_string(),
            fact_type: FactType::Experience,
            network: NetworkType::Opinion,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(embedding.into_iter().next().unwrap()),
            confidence: Some(confidence.clamp(0.0, 1.0)),
            evidence_ids: evidence.to_vec(),
            source_turn_id: None,
            created_at: now,
            updated_at: now,
            consolidated_at: None,
        };
        let ids = self.store.insert_facts(&[fact]).await?;
        Ok(ids[0])
    }

    async fn reinforce(
        &self,
        opinion_id: FactId,
        new_evidence: &[FactId],
        delta: f32,
    ) -> Result<()> {
        let facts = self.store.get_facts(&[opinion_id]).await?;
        if let Some(mut fact) = facts.into_iter().next() {
            let current = fact.confidence.unwrap_or(0.5);
            fact.confidence = Some((current + delta).min(1.0));
            fact.evidence_ids.extend_from_slice(new_evidence);
            fact.updated_at = Utc::now();
            self.store.update_fact(&fact).await?;
        }
        Ok(())
    }

    async fn weaken(
        &self,
        opinion_id: FactId,
        contradicting_evidence: &[FactId],
        delta: f32,
    ) -> Result<()> {
        let facts = self.store.get_facts(&[opinion_id]).await?;
        if let Some(mut fact) = facts.into_iter().next() {
            let current = fact.confidence.unwrap_or(0.5);
            fact.confidence = Some((current - delta).max(0.0));
            fact.evidence_ids.extend_from_slice(contradicting_evidence);
            fact.updated_at = Utc::now();
            self.store.update_fact(&fact).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::mock::MockEmbeddings;
    use crate::storage::mock::MockMemoryStore;
    use crate::types::BankId;

    fn setup() -> (Arc<MockMemoryStore>, DefaultOpinionManager) {
        let store = Arc::new(MockMemoryStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(8));
        let mgr = DefaultOpinionManager::new(store.clone(), embeddings);
        (store, mgr)
    }

    #[tokio::test]
    async fn form_and_retrieve_opinion() {
        let (_store, mgr) = setup();
        let bank = BankId::new();
        let evidence = vec![FactId::new()];

        let id = mgr
            .form_opinion(bank, "Rust is the best language", &evidence, 0.8)
            .await
            .unwrap();

        let opinions = mgr.get_opinions(bank, "Rust programming").await.unwrap();
        assert!(!opinions.is_empty());
        assert!(opinions.iter().any(|o| o.id == id));
        assert_eq!(opinions[0].network, NetworkType::Opinion);
    }

    #[tokio::test]
    async fn reinforce_caps_at_one() {
        let (_store, mgr) = setup();
        let bank = BankId::new();

        let id = mgr
            .form_opinion(bank, "High confidence opinion", &[], 0.9)
            .await
            .unwrap();

        mgr.reinforce(id, &[FactId::new()], 0.2).await.unwrap();

        let opinions = mgr
            .get_opinions(bank, "High confidence opinion")
            .await
            .unwrap();
        let opinion = opinions.iter().find(|o| o.id == id).unwrap();
        assert!((opinion.confidence.unwrap() - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn weaken_floors_at_zero() {
        let (_store, mgr) = setup();
        let bank = BankId::new();

        let id = mgr
            .form_opinion(bank, "Low confidence opinion", &[], 0.1)
            .await
            .unwrap();

        mgr.weaken(id, &[FactId::new()], 0.2).await.unwrap();

        let opinions = mgr
            .get_opinions(bank, "Low confidence opinion")
            .await
            .unwrap();
        let opinion = opinions.iter().find(|o| o.id == id).unwrap();
        assert!(opinion.confidence.unwrap().abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn get_opinions_filters_to_opinion_network() {
        let (store, mgr) = setup();
        let bank = BankId::new();
        let embeddings = Arc::new(MockEmbeddings::new(8));

        // Insert a World fact alongside an opinion
        let emb = embeddings.embed(&["Rust fact"]).await.unwrap();
        let now = Utc::now();
        let world_fact = Fact {
            id: FactId::new(),
            bank_id: bank,
            content: "Rust fact about memory safety".into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(emb[0].clone()),
            confidence: None,
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now,
            updated_at: now,
            consolidated_at: None,
        };
        store.insert_facts(&[world_fact]).await.unwrap();

        mgr.form_opinion(bank, "Rust is great for safety", &[], 0.7)
            .await
            .unwrap();

        let opinions = mgr.get_opinions(bank, "Rust safety").await.unwrap();
        // All returned results must be Opinion network
        for o in &opinions {
            assert_eq!(o.network, NetworkType::Opinion);
        }
    }

    #[tokio::test]
    async fn reinforce_appends_evidence() {
        let (_store, mgr) = setup();
        let bank = BankId::new();
        let ev1 = FactId::new();
        let ev2 = FactId::new();

        let id = mgr
            .form_opinion(bank, "Evidence test opinion", &[ev1], 0.5)
            .await
            .unwrap();

        mgr.reinforce(id, &[ev2], 0.1).await.unwrap();

        let opinions = mgr
            .get_opinions(bank, "Evidence test opinion")
            .await
            .unwrap();
        let opinion = opinions.iter().find(|o| o.id == id).unwrap();
        assert!(opinion.evidence_ids.contains(&ev1));
        assert!(opinion.evidence_ids.contains(&ev2));
    }
}
