//! Storage layer for the memory engine.

pub mod mock;
pub mod pg;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::error::Result;
use crate::types::{
    BankId, Entity, EntityId, Fact, FactFilter, FactId, GraphLink, LinkType, MemoryBank,
    NetworkType, ScoredFact,
};

/// Trait for persistent storage of memory engine data.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Insert one or more facts into storage. Returns the IDs of the inserted facts.
    async fn insert_facts(&self, facts: &[Fact]) -> Result<Vec<FactId>>;

    /// Retrieve facts by their IDs.
    async fn get_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>>;

    /// Retrieve facts for a bank, optionally filtered.
    async fn get_facts_by_bank(&self, bank: BankId, filter: FactFilter) -> Result<Vec<Fact>>;

    /// Insert or update an entity. Returns the entity ID.
    async fn upsert_entity(&self, entity: &Entity) -> Result<EntityId>;

    /// Find an entity by canonical name or alias within a bank.
    async fn find_entity(&self, bank: BankId, name: &str) -> Result<Option<Entity>>;

    /// Get all facts that reference a given entity.
    async fn get_entity_facts(&self, entity: EntityId) -> Result<Vec<Fact>>;

    /// Insert graph links between facts. Upserts on conflict.
    async fn insert_links(&self, links: &[GraphLink]) -> Result<()>;

    /// Get neighboring facts connected by graph links.
    async fn get_neighbors(
        &self,
        fact_id: FactId,
        link_type: Option<LinkType>,
    ) -> Result<Vec<(FactId, f32, LinkType)>>;

    /// Find facts by vector similarity. Returns scored results ordered by descending similarity.
    async fn vector_search(
        &self,
        embedding: &[f32],
        bank: BankId,
        limit: usize,
        network_filter: Option<&[NetworkType]>,
    ) -> Result<Vec<ScoredFact>>;

    /// Update a fact's mutable fields (confidence, evidence_ids, updated_at).
    ///
    /// Used by opinion reinforcement/weakening and observation consolidation.
    async fn update_fact(&self, fact: &Fact) -> Result<()>;

    /// Full-text keyword search over fact content. Returns scored results.
    async fn keyword_search(
        &self,
        query: &str,
        bank: BankId,
        limit: usize,
        network_filter: Option<&[NetworkType]>,
    ) -> Result<Vec<ScoredFact>>;

    /// List all entities in a bank.
    async fn list_entities(&self, bank: BankId) -> Result<Vec<Entity>>;

    /// Retrieve a memory bank by ID.
    async fn get_bank(&self, id: BankId) -> Result<MemoryBank>;

    /// Create a new memory bank. Returns the bank ID.
    async fn create_bank(&self, bank: &MemoryBank) -> Result<BankId>;

    /// List all memory banks.
    async fn list_banks(&self) -> Result<Vec<MemoryBank>>;

    /// Mark facts as consolidated.
    async fn mark_consolidated(&self, ids: &[FactId], at: DateTime<Utc>) -> Result<()>;
}
