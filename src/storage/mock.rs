//! Mock memory store for unit tests.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::error::Result;
use crate::types::{
    BankId, Entity, EntityId, Fact, FactFilter, FactId, GraphLink, LinkType, MemoryBank,
    RetrievalSource, ScoredFact,
};
use crate::util::cosine_similarity;

use super::MemoryStore;

/// A no-op memory store for unit tests that don't need real persistence.
#[derive(Clone)]
pub struct MockMemoryStore {
    facts: Arc<Mutex<Vec<Fact>>>,
    entities: Arc<Mutex<Vec<Entity>>>,
    links: Arc<Mutex<Vec<GraphLink>>>,
    banks: Arc<Mutex<Vec<MemoryBank>>>,
}

impl MockMemoryStore {
    /// Create a new empty mock store.
    pub fn new() -> Self {
        Self {
            facts: Arc::new(Mutex::new(Vec::new())),
            entities: Arc::new(Mutex::new(Vec::new())),
            links: Arc::new(Mutex::new(Vec::new())),
            banks: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl Default for MockMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryStore for MockMemoryStore {
    async fn insert_facts(&self, facts: &[Fact]) -> Result<Vec<FactId>> {
        let mut store = self.facts.lock().unwrap();
        let ids: Vec<FactId> = facts.iter().map(|f| f.id).collect();
        store.extend_from_slice(facts);
        Ok(ids)
    }

    async fn get_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>> {
        let store = self.facts.lock().unwrap();
        Ok(store.iter().filter(|f| ids.contains(&f.id)).cloned().collect())
    }

    async fn get_facts_by_bank(&self, bank: BankId, filter: FactFilter) -> Result<Vec<Fact>> {
        let store = self.facts.lock().unwrap();
        Ok(store
            .iter()
            .filter(|f| {
                if f.bank_id != bank {
                    return false;
                }
                if let Some(ref networks) = filter.network
                    && !networks.contains(&f.network)
                {
                    return false;
                }
                if let Some(ref ft) = filter.fact_type
                    && f.fact_type != *ft
                {
                    return false;
                }
                if let Some(ref tr) = filter.temporal_range {
                    match &f.temporal_range {
                        Some(fact_tr) if !tr.overlaps(fact_tr) => return false,
                        None => return false,
                        _ => {}
                    }
                }
                if let Some(ref eids) = filter.entity_ids
                    && !eids.iter().any(|eid| f.entity_ids.contains(eid))
                {
                    return false;
                }
                if let Some(since) = filter.created_since
                    && f.created_at < since
                {
                    return false;
                }
                true
            })
            .cloned()
            .collect())
    }

    async fn upsert_entity(&self, entity: &Entity) -> Result<EntityId> {
        let mut store = self.entities.lock().unwrap();
        store.retain(|e| e.id != entity.id);
        store.push(entity.clone());
        Ok(entity.id)
    }

    async fn find_entity(&self, bank: BankId, name: &str) -> Result<Option<Entity>> {
        let store = self.entities.lock().unwrap();
        let lower = name.to_lowercase();
        Ok(store.iter().find(|e| {
            e.bank_id == bank
                && (e.canonical_name.to_lowercase() == lower
                    || e.aliases.iter().any(|a| a.to_lowercase() == lower))
        }).cloned())
    }

    async fn get_entity_facts(&self, entity: EntityId) -> Result<Vec<Fact>> {
        let store = self.facts.lock().unwrap();
        Ok(store.iter().filter(|f| f.entity_ids.contains(&entity)).cloned().collect())
    }

    async fn insert_links(&self, links: &[GraphLink]) -> Result<()> {
        let mut store = self.links.lock().unwrap();
        store.extend_from_slice(links);
        Ok(())
    }

    async fn get_neighbors(
        &self,
        fact_id: FactId,
        link_type: Option<LinkType>,
    ) -> Result<Vec<(FactId, f32)>> {
        let store = self.links.lock().unwrap();
        Ok(store
            .iter()
            .filter(|l| {
                (l.source_id == fact_id || l.target_id == fact_id)
                    && link_type.as_ref().is_none_or(|t| l.link_type == *t)
            })
            .map(|l| {
                let other = if l.source_id == fact_id { l.target_id } else { l.source_id };
                (other, l.weight)
            })
            .collect())
    }

    async fn vector_search(
        &self,
        embedding: &[f32],
        bank: BankId,
        limit: usize,
    ) -> Result<Vec<ScoredFact>> {
        let store = self.facts.lock().unwrap();
        let mut scored: Vec<ScoredFact> = store
            .iter()
            .filter(|f| f.bank_id == bank)
            .filter_map(|f| {
                f.embedding.as_ref().map(|emb| {
                    let score = cosine_similarity(embedding, emb);
                    ScoredFact {
                        fact: f.clone(),
                        score,
                        sources: vec![RetrievalSource::Semantic],
                    }
                })
            })
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }

    async fn update_fact(&self, fact: &Fact) -> Result<()> {
        let mut store = self.facts.lock().unwrap();
        if let Some(existing) = store.iter_mut().find(|f| f.id == fact.id) {
            *existing = fact.clone();
        }
        Ok(())
    }

    async fn keyword_search(
        &self,
        query: &str,
        bank: BankId,
        limit: usize,
    ) -> Result<Vec<ScoredFact>> {
        let lower_query = query.to_lowercase();
        let store = self.facts.lock().unwrap();
        let mut scored: Vec<ScoredFact> = store
            .iter()
            .filter(|f| f.bank_id == bank && f.content.to_lowercase().contains(&lower_query))
            .map(|f| ScoredFact {
                fact: f.clone(),
                score: 1.0,
                sources: vec![RetrievalSource::Keyword],
            })
            .collect();
        scored.truncate(limit);
        Ok(scored)
    }

    async fn list_entities(&self, bank: BankId) -> Result<Vec<Entity>> {
        let store = self.entities.lock().unwrap();
        Ok(store.iter().filter(|e| e.bank_id == bank).cloned().collect())
    }

    async fn get_bank(&self, id: BankId) -> Result<MemoryBank> {
        let store = self.banks.lock().unwrap();
        store
            .iter()
            .find(|b| b.id == id)
            .cloned()
            .ok_or_else(|| crate::error::Error::NotFound(format!("bank {id} not found")))
    }

    async fn create_bank(&self, bank: &MemoryBank) -> Result<BankId> {
        let mut store = self.banks.lock().unwrap();
        store.push(bank.clone());
        Ok(bank.id)
    }

    async fn list_banks(&self) -> Result<Vec<MemoryBank>> {
        let store = self.banks.lock().unwrap();
        Ok(store.clone())
    }
}
