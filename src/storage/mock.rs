//! Mock memory store for unit tests.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::error::Result;
use crate::types::{
    BankId, Entity, EntityId, Fact, FactFilter, FactId, FactSourceLookup, GraphLink, LinkType,
    MemoryBank, NetworkType, RetrievalSource, ScoredFact, Source, SourceId,
};
use crate::util::cosine_similarity;

use super::{MemoryStore, TransactionHandle};

/// A no-op memory store for unit tests that don't need real persistence.
#[derive(Clone)]
pub struct MockMemoryStore {
    facts: Arc<Mutex<Vec<Fact>>>,
    sources: Arc<Mutex<Vec<Source>>>,
    fact_sources: Arc<Mutex<Vec<(FactId, SourceId)>>>,
    entities: Arc<Mutex<Vec<Entity>>>,
    links: Arc<Mutex<Vec<GraphLink>>>,
    banks: Arc<Mutex<Vec<MemoryBank>>>,
}

impl MockMemoryStore {
    /// Create a new empty mock store.
    pub fn new() -> Self {
        Self {
            facts: Arc::new(Mutex::new(Vec::new())),
            sources: Arc::new(Mutex::new(Vec::new())),
            fact_sources: Arc::new(Mutex::new(Vec::new())),
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
    async fn begin(&self) -> Result<Box<dyn TransactionHandle>> {
        Ok(Box::new(MockTransactionHandle {
            inner: self.clone(),
        }))
    }

    async fn insert_facts(&self, facts: &[Fact]) -> Result<Vec<FactId>> {
        let mut store = self.facts.lock().unwrap();
        let ids: Vec<FactId> = facts.iter().map(|f| f.id).collect();
        store.extend_from_slice(facts);
        Ok(ids)
    }

    async fn get_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>> {
        let store = self.facts.lock().unwrap();
        Ok(store
            .iter()
            .filter(|f| ids.contains(&f.id))
            .cloned()
            .collect())
    }

    async fn insert_source(&self, source: &Source) -> Result<SourceId> {
        let mut store = self.sources.lock().unwrap();
        store.push(source.clone());
        Ok(source.id)
    }

    async fn link_facts_to_source(&self, fact_ids: &[FactId], source_id: SourceId) -> Result<()> {
        if fact_ids.is_empty() {
            return Ok(());
        }

        let mut store = self.fact_sources.lock().unwrap();
        for fact_id in fact_ids {
            let link = (*fact_id, source_id);
            if !store.contains(&link) {
                store.push(link);
            }
        }
        Ok(())
    }

    async fn lookup_sources(
        &self,
        fact_ids: &[FactId],
        per_fact_limit: usize,
    ) -> Result<Vec<FactSourceLookup>> {
        if fact_ids.is_empty() || per_fact_limit == 0 {
            return Ok(vec![]);
        }

        let sources = self.sources.lock().unwrap();
        let fact_sources = self.fact_sources.lock().unwrap();
        let mut lookups = Vec::new();

        for fact_id in fact_ids {
            let mut linked_sources = fact_sources
                .iter()
                .filter(|(linked_fact_id, _)| linked_fact_id == fact_id)
                .filter_map(|(_, source_id)| sources.iter().find(|source| source.id == *source_id))
                .cloned()
                .collect::<Vec<_>>();

            linked_sources.sort_by_key(|source| (source.timestamp, source.created_at, source.id));
            linked_sources.truncate(per_fact_limit);

            if !linked_sources.is_empty() {
                lookups.push(FactSourceLookup {
                    fact_id: *fact_id,
                    sources: linked_sources,
                });
            }
        }

        Ok(lookups)
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
                if filter.unconsolidated_only && f.consolidated_at.is_some() {
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
        Ok(store
            .iter()
            .find(|e| {
                e.bank_id == bank
                    && (e.canonical_name.to_lowercase() == lower
                        || e.aliases.iter().any(|a| a.to_lowercase() == lower))
            })
            .cloned())
    }

    async fn get_entity_facts(&self, entity: EntityId) -> Result<Vec<Fact>> {
        let store = self.facts.lock().unwrap();
        Ok(store
            .iter()
            .filter(|f| f.entity_ids.contains(&entity))
            .cloned()
            .collect())
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
    ) -> Result<Vec<(FactId, f32, LinkType)>> {
        let store = self.links.lock().unwrap();
        Ok(store
            .iter()
            .filter(|l| {
                (l.source_id == fact_id || l.target_id == fact_id)
                    && link_type.as_ref().is_none_or(|t| l.link_type == *t)
            })
            .map(|l| {
                let other = if l.source_id == fact_id {
                    l.target_id
                } else {
                    l.source_id
                };
                (other, l.weight, l.link_type)
            })
            .collect())
    }

    async fn vector_search(
        &self,
        embedding: &[f32],
        bank: BankId,
        limit: usize,
        network_filter: Option<&[NetworkType]>,
    ) -> Result<Vec<ScoredFact>> {
        let store = self.facts.lock().unwrap();
        let mut scored: Vec<ScoredFact> = store
            .iter()
            .filter(|f| f.bank_id == bank)
            .filter(|f| network_filter.is_none_or(|nets| nets.contains(&f.network)))
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
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
        network_filter: Option<&[NetworkType]>,
    ) -> Result<Vec<ScoredFact>> {
        let lower_query = query.to_lowercase();
        let store = self.facts.lock().unwrap();
        let mut scored: Vec<ScoredFact> = store
            .iter()
            .filter(|f| f.bank_id == bank && f.content.to_lowercase().contains(&lower_query))
            .filter(|f| network_filter.is_none_or(|nets| nets.contains(&f.network)))
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
        Ok(store
            .iter()
            .filter(|e| e.bank_id == bank)
            .cloned()
            .collect())
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

    async fn mark_consolidated(
        &self,
        ids: &[FactId],
        at: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        let mut store = self.facts.lock().unwrap();
        for fact in store.iter_mut() {
            if ids.contains(&fact.id) {
                fact.consolidated_at = Some(at);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MockTransactionHandle — transparent wrapper (mock writes are immediate)
// ---------------------------------------------------------------------------

/// Mock transaction handle. Since [`MockMemoryStore`] uses `Arc<Mutex<...>>`
/// internally, writes are immediately visible. `commit()` is a no-op and
/// drop without commit does NOT roll back (acceptable for tests).
pub struct MockTransactionHandle {
    inner: MockMemoryStore,
}

#[async_trait]
impl TransactionHandle for MockTransactionHandle {
    async fn commit(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl MemoryStore for MockTransactionHandle {
    async fn begin(&self) -> Result<Box<dyn TransactionHandle>> {
        self.inner.begin().await
    }

    async fn insert_facts(&self, facts: &[Fact]) -> Result<Vec<FactId>> {
        self.inner.insert_facts(facts).await
    }

    async fn get_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>> {
        self.inner.get_facts(ids).await
    }

    async fn insert_source(&self, source: &Source) -> Result<SourceId> {
        self.inner.insert_source(source).await
    }

    async fn link_facts_to_source(&self, fact_ids: &[FactId], source_id: SourceId) -> Result<()> {
        self.inner.link_facts_to_source(fact_ids, source_id).await
    }

    async fn lookup_sources(
        &self,
        fact_ids: &[FactId],
        per_fact_limit: usize,
    ) -> Result<Vec<FactSourceLookup>> {
        self.inner.lookup_sources(fact_ids, per_fact_limit).await
    }

    async fn get_facts_by_bank(&self, bank: BankId, filter: FactFilter) -> Result<Vec<Fact>> {
        self.inner.get_facts_by_bank(bank, filter).await
    }

    async fn upsert_entity(&self, entity: &Entity) -> Result<EntityId> {
        self.inner.upsert_entity(entity).await
    }

    async fn find_entity(&self, bank: BankId, name: &str) -> Result<Option<Entity>> {
        self.inner.find_entity(bank, name).await
    }

    async fn get_entity_facts(&self, entity: EntityId) -> Result<Vec<Fact>> {
        self.inner.get_entity_facts(entity).await
    }

    async fn insert_links(&self, links: &[GraphLink]) -> Result<()> {
        self.inner.insert_links(links).await
    }

    async fn get_neighbors(
        &self,
        fact_id: FactId,
        link_type: Option<LinkType>,
    ) -> Result<Vec<(FactId, f32, LinkType)>> {
        self.inner.get_neighbors(fact_id, link_type).await
    }

    async fn vector_search(
        &self,
        embedding: &[f32],
        bank: BankId,
        limit: usize,
        network_filter: Option<&[NetworkType]>,
    ) -> Result<Vec<ScoredFact>> {
        self.inner
            .vector_search(embedding, bank, limit, network_filter)
            .await
    }

    async fn update_fact(&self, fact: &Fact) -> Result<()> {
        self.inner.update_fact(fact).await
    }

    async fn keyword_search(
        &self,
        query: &str,
        bank: BankId,
        limit: usize,
        network_filter: Option<&[NetworkType]>,
    ) -> Result<Vec<ScoredFact>> {
        self.inner
            .keyword_search(query, bank, limit, network_filter)
            .await
    }

    async fn list_entities(&self, bank: BankId) -> Result<Vec<Entity>> {
        self.inner.list_entities(bank).await
    }

    async fn get_bank(&self, id: BankId) -> Result<MemoryBank> {
        self.inner.get_bank(id).await
    }

    async fn create_bank(&self, bank: &MemoryBank) -> Result<BankId> {
        self.inner.create_bank(bank).await
    }

    async fn list_banks(&self) -> Result<Vec<MemoryBank>> {
        self.inner.list_banks().await
    }

    async fn mark_consolidated(
        &self,
        ids: &[FactId],
        at: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        self.inner.mark_consolidated(ids, at).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[tokio::test]
    async fn lookup_sources_dedups_links_and_orders_sources() {
        let store = MockMemoryStore::new();
        let bank_id = BankId::new();
        let fact_id = FactId::new();
        let early = Source {
            id: SourceId::new(),
            bank_id,
            content: "Earlier source".into(),
            context: Some("Earlier context".into()),
            speaker: Some("Avery".into()),
            rendered_input: Some(
                "Speaker: Avery\n\n## Preceding Context\n\nEarlier context\n\n---\n\n## Content to Extract From\n\nEarlier source\n\nTimestamp: 2024-07-01T12:00:00+00:00".into(),
            ),
            timestamp: Utc.with_ymd_and_hms(2024, 7, 1, 12, 0, 0).unwrap(),
            created_at: Utc.with_ymd_and_hms(2024, 7, 1, 12, 0, 1).unwrap(),
        };
        let late = Source {
            id: SourceId::new(),
            bank_id,
            content: "Later source".into(),
            context: None,
            speaker: None,
            rendered_input: Some(
                "## Content to Extract From\n\nLater source\n\nTimestamp: 2024-07-03T12:00:00+00:00"
                    .into(),
            ),
            timestamp: Utc.with_ymd_and_hms(2024, 7, 3, 12, 0, 0).unwrap(),
            created_at: Utc.with_ymd_and_hms(2024, 7, 3, 12, 0, 1).unwrap(),
        };

        store.insert_source(&late).await.unwrap();
        store.insert_source(&early).await.unwrap();
        store
            .link_facts_to_source(&[fact_id], late.id)
            .await
            .unwrap();
        store
            .link_facts_to_source(&[fact_id], late.id)
            .await
            .unwrap();
        store
            .link_facts_to_source(&[fact_id], early.id)
            .await
            .unwrap();

        let lookups = store.lookup_sources(&[fact_id], 10).await.unwrap();
        assert_eq!(lookups.len(), 1);
        assert_eq!(lookups[0].fact_id, fact_id);
        assert_eq!(lookups[0].sources.len(), 2);
        assert_eq!(lookups[0].sources[0].id, early.id);
        assert_eq!(lookups[0].sources[1].id, late.id);
    }
}
