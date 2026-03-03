//! PostgreSQL implementation of [`MemoryStore`].

use async_trait::async_trait;
use pgvector::Vector;
use sqlx::postgres::{PgConnection, PgRow};
use sqlx::{PgPool, Row};

use super::{MemoryStore, TransactionHandle};
use crate::error::{Error, Result};
use crate::types::{
    BankId, Disposition, Entity, EntityId, EntityType, Fact, FactFilter, FactId, FactType,
    GraphLink, LinkType, MemoryBank, NetworkType, ScoredFact, TemporalRange, TurnId,
};

/// PostgreSQL-backed memory store using pgvector for similarity search.
pub struct PgMemoryStore {
    pool: PgPool,
}

impl PgMemoryStore {
    /// Create a new store from an existing connection pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Run database migrations.
    pub async fn migrate(&self) -> Result<()> {
        sqlx::raw_sql(include_str!("../../migrations/001_init.sql"))
            .execute(&self.pool)
            .await?;
        sqlx::raw_sql(include_str!("../../migrations/002_consolidated_at.sql"))
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

// --- Row conversion helpers ---

fn row_to_fact(row: &PgRow) -> Result<Fact> {
    let id: FactId = row.get("id");
    let bank_id: BankId = row.get("bank_id");
    let content: String = row.get("content");
    let fact_type_str: String = row.get("fact_type");
    let network_str: String = row.get("network");
    let entity_ids_json: serde_json::Value = row.get("entity_ids");
    let temporal_start: Option<chrono::DateTime<chrono::Utc>> = row.get("temporal_start");
    let temporal_end: Option<chrono::DateTime<chrono::Utc>> = row.get("temporal_end");
    let embedding: Option<Vector> = row.get("embedding");
    let confidence: Option<f32> = row.get("confidence");
    let evidence_ids_json: serde_json::Value = row.get("evidence_ids");
    let source_turn_id: Option<uuid::Uuid> = row.get("source_turn_id");
    let created_at: chrono::DateTime<chrono::Utc> = row.get("created_at");
    let updated_at: chrono::DateTime<chrono::Utc> = row.get("updated_at");
    let consolidated_at: Option<chrono::DateTime<chrono::Utc>> = row.get("consolidated_at");

    let fact_type: FactType =
        serde_json::from_value(serde_json::Value::String(fact_type_str))?;
    let network: NetworkType =
        serde_json::from_value(serde_json::Value::String(network_str))?;
    let entity_ids: Vec<EntityId> = serde_json::from_value(entity_ids_json)?;
    let evidence_ids: Vec<FactId> = serde_json::from_value(evidence_ids_json)?;

    let temporal_range = match (temporal_start, temporal_end) {
        (None, None) => None,
        (s, e) => Some(TemporalRange { start: s, end: e }),
    };

    Ok(Fact {
        id,
        bank_id,
        content,
        fact_type,
        network,
        entity_ids,
        temporal_range,
        embedding: embedding.map(|v| v.to_vec()),
        confidence,
        evidence_ids,
        source_turn_id: source_turn_id.map(TurnId::from_uuid),
        created_at,
        updated_at,
        consolidated_at,
    })
}

fn row_to_entity(row: &PgRow) -> Result<Entity> {
    let id: EntityId = row.get("id");
    let bank_id: BankId = row.get("bank_id");
    let canonical_name: String = row.get("canonical_name");
    let aliases_json: serde_json::Value = row.get("aliases");
    let entity_type_str: String = row.get("entity_type");

    let aliases: Vec<String> = serde_json::from_value(aliases_json)?;
    let entity_type: EntityType =
        serde_json::from_value(serde_json::Value::String(entity_type_str))?;

    Ok(Entity {
        id,
        canonical_name,
        aliases,
        entity_type,
        bank_id,
    })
}

fn row_to_bank(row: &PgRow) -> Result<MemoryBank> {
    let id: BankId = row.get("id");
    let name: String = row.get("name");
    let mission: String = row.get("mission");
    let directives_json: serde_json::Value = row.get("directives");
    let skepticism: i16 = row.get("skepticism");
    let literalism: i16 = row.get("literalism");
    let empathy: i16 = row.get("empathy");
    let bias_strength: f32 = row.get("bias_strength");
    let embedding_model: String = row.get("embedding_model");
    let embedding_dims: i16 = row.get("embedding_dims");

    let directives: Vec<String> = serde_json::from_value(directives_json)?;
    let disposition = Disposition::new(
        skepticism as u8,
        literalism as u8,
        empathy as u8,
        bias_strength,
    )?;

    Ok(MemoryBank {
        id,
        name,
        mission,
        directives,
        disposition,
        embedding_model,
        embedding_dimensions: embedding_dims as u16,
    })
}

/// Serialize an enum to its serde string representation.
fn enum_to_sql<T: serde::Serialize>(val: &T) -> Result<String> {
    let json_val = serde_json::to_value(val)?;
    Ok(json_val.as_str().unwrap_or_default().to_string())
}

// ---------------------------------------------------------------------------
// Free functions — all SQL lives here, shared by PgMemoryStore and
// PgTransactionHandle. Both PgPool::acquire() and Transaction deref to
// PgConnection, so these work for either executor.
// ---------------------------------------------------------------------------

async fn create_bank_impl(conn: &mut PgConnection, bank: &MemoryBank) -> Result<BankId> {
    let directives = serde_json::to_value(&bank.directives)?;
    sqlx::query(
        "INSERT INTO memory_banks (id, name, mission, directives, skepticism, literalism, empathy, bias_strength, embedding_model, embedding_dims)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
    )
    .bind(bank.id)
    .bind(&bank.name)
    .bind(&bank.mission)
    .bind(&directives)
    .bind(bank.disposition.skepticism() as i16)
    .bind(bank.disposition.literalism() as i16)
    .bind(bank.disposition.empathy() as i16)
    .bind(bank.disposition.bias_strength())
    .bind(&bank.embedding_model)
    .bind(bank.embedding_dimensions as i16)
    .execute(&mut *conn)
    .await?;
    Ok(bank.id)
}

async fn get_bank_impl(conn: &mut PgConnection, id: BankId) -> Result<MemoryBank> {
    let row = sqlx::query(
        "SELECT id, name, mission, directives, skepticism, literalism, empathy, bias_strength, embedding_model, embedding_dims
         FROM memory_banks WHERE id = $1",
    )
    .bind(id)
    .fetch_optional(&mut *conn)
    .await?
    .ok_or_else(|| Error::NotFound(format!("bank {id}")))?;
    row_to_bank(&row)
}

async fn insert_facts_impl(conn: &mut PgConnection, facts: &[Fact]) -> Result<Vec<FactId>> {
    let mut ids = Vec::with_capacity(facts.len());
    for fact in facts {
        let fact_type = enum_to_sql(&fact.fact_type)?;
        let network = enum_to_sql(&fact.network)?;
        let entity_ids = serde_json::to_value(&fact.entity_ids)?;
        let evidence_ids = serde_json::to_value(&fact.evidence_ids)?;
        let (temporal_start, temporal_end) = match &fact.temporal_range {
            Some(tr) => (tr.start, tr.end),
            None => (None, None),
        };
        let embedding = fact.embedding.as_ref().map(|v| Vector::from(v.clone()));
        let source_turn_uuid = fact.source_turn_id.map(|t| t.to_uuid());

        sqlx::query(
            "INSERT INTO facts (id, bank_id, content, fact_type, network, entity_ids,
             temporal_start, temporal_end, embedding, confidence, evidence_ids,
             source_turn_id, created_at, updated_at, consolidated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)",
        )
        .bind(fact.id)
        .bind(fact.bank_id)
        .bind(&fact.content)
        .bind(&fact_type)
        .bind(&network)
        .bind(&entity_ids)
        .bind(temporal_start)
        .bind(temporal_end)
        .bind(embedding)
        .bind(fact.confidence)
        .bind(&evidence_ids)
        .bind(source_turn_uuid)
        .bind(fact.created_at)
        .bind(fact.updated_at)
        .bind(fact.consolidated_at)
        .execute(&mut *conn)
        .await?;
        ids.push(fact.id);
    }
    Ok(ids)
}

async fn get_facts_impl(conn: &mut PgConnection, ids: &[FactId]) -> Result<Vec<Fact>> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let uuids: Vec<uuid::Uuid> = ids.iter().map(|id| id.to_uuid()).collect();
    let rows = sqlx::query(
        "SELECT id, bank_id, content, fact_type, network, entity_ids,
                temporal_start, temporal_end, embedding, confidence,
                evidence_ids, source_turn_id, created_at, updated_at, consolidated_at
         FROM facts WHERE id = ANY($1)",
    )
    .bind(&uuids)
    .fetch_all(&mut *conn)
    .await?;
    rows.iter().map(row_to_fact).collect()
}

async fn get_facts_by_bank_impl(
    conn: &mut PgConnection,
    bank: BankId,
    filter: FactFilter,
) -> Result<Vec<Fact>> {
    let mut qb = sqlx::QueryBuilder::<sqlx::Postgres>::new(
        "SELECT id, bank_id, content, fact_type, network, entity_ids,
                temporal_start, temporal_end, embedding, confidence,
                evidence_ids, source_turn_id, created_at, updated_at, consolidated_at
         FROM facts WHERE bank_id = ",
    );
    qb.push_bind(bank);

    if let Some(ref networks) = filter.network {
        let network_strs: Vec<String> = networks
            .iter()
            .map(enum_to_sql)
            .collect::<Result<_>>()?;
        qb.push(" AND network = ANY(");
        qb.push_bind(network_strs);
        qb.push(")");
    }

    if let Some(ref ft) = filter.fact_type {
        let ft_str = enum_to_sql(ft)?;
        qb.push(" AND fact_type = ");
        qb.push_bind(ft_str);
    }

    if let Some(ref tr) = filter.temporal_range {
        if let Some(end) = tr.end {
            qb.push(" AND (temporal_start IS NULL OR temporal_start <= ");
            qb.push_bind(end);
            qb.push(")");
        }
        if let Some(start) = tr.start {
            qb.push(" AND (temporal_end IS NULL OR temporal_end >= ");
            qb.push_bind(start);
            qb.push(")");
        }
    }

    if let Some(since) = filter.created_since {
        qb.push(" AND created_at >= ");
        qb.push_bind(since);
    }

    if filter.unconsolidated_only {
        qb.push(" AND consolidated_at IS NULL");
    }

    if let Some(ref eids) = filter.entity_ids
        && !eids.is_empty()
    {
        qb.push(" AND (");
        for (i, eid) in eids.iter().enumerate() {
            if i > 0 {
                qb.push(" OR ");
            }
            let single = serde_json::to_value(vec![eid])?;
            qb.push("entity_ids @> ");
            qb.push_bind(single);
        }
        qb.push(")");
    }

    qb.push(" ORDER BY created_at DESC");

    let rows = qb.build().fetch_all(&mut *conn).await?;
    rows.iter().map(row_to_fact).collect()
}

async fn upsert_entity_impl(conn: &mut PgConnection, entity: &Entity) -> Result<EntityId> {
    let aliases = serde_json::to_value(&entity.aliases)?;
    let entity_type = enum_to_sql(&entity.entity_type)?;

    let row = sqlx::query(
        "INSERT INTO entities (id, bank_id, canonical_name, aliases, entity_type)
         VALUES ($1, $2, $3, $4, $5)
         ON CONFLICT (bank_id, canonical_name)
         DO UPDATE SET aliases = $4, entity_type = $5, updated_at = now()
         RETURNING id",
    )
    .bind(entity.id)
    .bind(entity.bank_id)
    .bind(&entity.canonical_name)
    .bind(&aliases)
    .bind(&entity_type)
    .fetch_one(&mut *conn)
    .await?;
    let id: EntityId = row.get("id");
    Ok(id)
}

async fn find_entity_impl(
    conn: &mut PgConnection,
    bank: BankId,
    name: &str,
) -> Result<Option<Entity>> {
    let alias_json = serde_json::to_value(vec![name])?;
    let row = sqlx::query(
        "SELECT id, bank_id, canonical_name, aliases, entity_type
         FROM entities
         WHERE bank_id = $1 AND (canonical_name = $2 OR aliases @> $3::jsonb)
         LIMIT 1",
    )
    .bind(bank)
    .bind(name)
    .bind(&alias_json)
    .fetch_optional(&mut *conn)
    .await?;
    match row {
        Some(r) => Ok(Some(row_to_entity(&r)?)),
        None => Ok(None),
    }
}

async fn get_entity_facts_impl(conn: &mut PgConnection, entity: EntityId) -> Result<Vec<Fact>> {
    let entity_json = serde_json::to_value(vec![entity])?;
    let rows = sqlx::query(
        "SELECT id, bank_id, content, fact_type, network, entity_ids,
                temporal_start, temporal_end, embedding, confidence,
                evidence_ids, source_turn_id, created_at, updated_at, consolidated_at
         FROM facts WHERE entity_ids @> $1::jsonb",
    )
    .bind(&entity_json)
    .fetch_all(&mut *conn)
    .await?;
    rows.iter().map(row_to_fact).collect()
}

async fn insert_links_impl(conn: &mut PgConnection, links: &[GraphLink]) -> Result<()> {
    for link in links {
        let link_type = enum_to_sql(&link.link_type)?;
        sqlx::query(
            "INSERT INTO graph_links (source_id, target_id, link_type, weight)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (source_id, target_id, link_type)
             DO UPDATE SET weight = $4",
        )
        .bind(link.source_id)
        .bind(link.target_id)
        .bind(&link_type)
        .bind(link.weight)
        .execute(&mut *conn)
        .await?;
    }
    Ok(())
}

async fn get_neighbors_impl(
    conn: &mut PgConnection,
    fact_id: FactId,
    link_type: Option<LinkType>,
) -> Result<Vec<(FactId, f32, LinkType)>> {
    let rows = if let Some(lt) = link_type {
        let lt_str = enum_to_sql(&lt)?;
        sqlx::query(
            "SELECT target_id, weight, link_type FROM graph_links
             WHERE source_id = $1 AND link_type = $2
             UNION ALL
             SELECT source_id, weight, link_type FROM graph_links
             WHERE target_id = $1 AND link_type = $2",
        )
        .bind(fact_id)
        .bind(&lt_str)
        .fetch_all(&mut *conn)
        .await?
    } else {
        sqlx::query(
            "SELECT target_id, weight, link_type FROM graph_links WHERE source_id = $1
             UNION ALL
             SELECT source_id, weight, link_type FROM graph_links WHERE target_id = $1",
        )
        .bind(fact_id)
        .fetch_all(&mut *conn)
        .await?
    };

    let mut neighbors = Vec::with_capacity(rows.len());
    for row in &rows {
        let uuid: uuid::Uuid = row.get(0);
        let weight: f32 = row.get(1);
        let lt_str: String = row.get(2);
        let lt = match lt_str.as_str() {
            "semantic" => LinkType::Semantic,
            "temporal" => LinkType::Temporal,
            "causal" => LinkType::Causal,
            "entity" => LinkType::Entity,
            other => return Err(Error::Llm(format!("unknown link type in graph_links: {other}"))),
        };
        neighbors.push((FactId::from_uuid(uuid), weight, lt));
    }
    Ok(neighbors)
}

async fn vector_search_impl(
    conn: &mut PgConnection,
    embedding: &[f32],
    bank: BankId,
    limit: usize,
    network_filter: Option<&[NetworkType]>,
) -> Result<Vec<ScoredFact>> {
    let vec = Vector::from(embedding.to_vec());
    let rows = if let Some(networks) = network_filter {
        let network_strs: Vec<String> = networks
            .iter()
            .map(enum_to_sql)
            .collect::<Result<_>>()?;
        sqlx::query(
            "SELECT id, bank_id, content, fact_type, network, entity_ids,
                    temporal_start, temporal_end, embedding, confidence,
                    evidence_ids, source_turn_id, created_at, updated_at, consolidated_at,
                    1.0 - (embedding <=> $1::vector) AS score
             FROM facts
             WHERE bank_id = $2 AND embedding IS NOT NULL AND network = ANY($4)
             ORDER BY embedding <=> $1::vector
             LIMIT $3",
        )
        .bind(&vec)
        .bind(bank)
        .bind(limit as i64)
        .bind(&network_strs)
        .fetch_all(&mut *conn)
        .await?
    } else {
        sqlx::query(
            "SELECT id, bank_id, content, fact_type, network, entity_ids,
                    temporal_start, temporal_end, embedding, confidence,
                    evidence_ids, source_turn_id, created_at, updated_at, consolidated_at,
                    1.0 - (embedding <=> $1::vector) AS score
             FROM facts
             WHERE bank_id = $2 AND embedding IS NOT NULL
             ORDER BY embedding <=> $1::vector
             LIMIT $3",
        )
        .bind(&vec)
        .bind(bank)
        .bind(limit as i64)
        .fetch_all(&mut *conn)
        .await?
    };

    let mut results = Vec::with_capacity(rows.len());
    for row in &rows {
        let fact = row_to_fact(row)?;
        let score: f64 = row.get("score");
        results.push(ScoredFact {
            fact,
            score: score as f32,
            sources: vec![crate::types::RetrievalSource::Semantic],
        });
    }
    Ok(results)
}

async fn update_fact_impl(conn: &mut PgConnection, fact: &Fact) -> Result<()> {
    let evidence_ids = serde_json::to_value(&fact.evidence_ids)?;
    let entity_ids = serde_json::to_value(&fact.entity_ids)?;
    let embedding = fact.embedding.as_ref().map(|v| Vector::from(v.clone()));
    let (temporal_start, temporal_end) = match &fact.temporal_range {
        Some(tr) => (tr.start, tr.end),
        None => (None, None),
    };

    sqlx::query(
        "UPDATE facts SET
            content = $2,
            confidence = $3,
            evidence_ids = $4,
            entity_ids = $5,
            embedding = $6,
            temporal_start = $7,
            temporal_end = $8,
            updated_at = $9,
            consolidated_at = $10
         WHERE id = $1",
    )
    .bind(fact.id)
    .bind(&fact.content)
    .bind(fact.confidence)
    .bind(&evidence_ids)
    .bind(&entity_ids)
    .bind(embedding)
    .bind(temporal_start)
    .bind(temporal_end)
    .bind(fact.updated_at)
    .bind(fact.consolidated_at)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn keyword_search_impl(
    conn: &mut PgConnection,
    query: &str,
    bank: BankId,
    limit: usize,
    network_filter: Option<&[NetworkType]>,
) -> Result<Vec<ScoredFact>> {
    let rows = if let Some(networks) = network_filter {
        let network_strs: Vec<String> = networks
            .iter()
            .map(enum_to_sql)
            .collect::<Result<_>>()?;
        sqlx::query(
            "SELECT id, bank_id, content, fact_type, network, entity_ids,
                    temporal_start, temporal_end, embedding, confidence,
                    evidence_ids, source_turn_id, created_at, updated_at, consolidated_at,
                    ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) AS score
             FROM facts
             WHERE bank_id = $2
               AND to_tsvector('english', content) @@ plainto_tsquery('english', $1)
               AND network = ANY($4)
             ORDER BY score DESC
             LIMIT $3",
        )
        .bind(query)
        .bind(bank)
        .bind(limit as i64)
        .bind(&network_strs)
        .fetch_all(&mut *conn)
        .await?
    } else {
        sqlx::query(
            "SELECT id, bank_id, content, fact_type, network, entity_ids,
                    temporal_start, temporal_end, embedding, confidence,
                    evidence_ids, source_turn_id, created_at, updated_at, consolidated_at,
                    ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) AS score
             FROM facts
             WHERE bank_id = $2
               AND to_tsvector('english', content) @@ plainto_tsquery('english', $1)
             ORDER BY score DESC
             LIMIT $3",
        )
        .bind(query)
        .bind(bank)
        .bind(limit as i64)
        .fetch_all(&mut *conn)
        .await?
    };

    let mut results = Vec::with_capacity(rows.len());
    for row in &rows {
        let fact = row_to_fact(row)?;
        let score: f32 = row.get("score");
        results.push(ScoredFact {
            fact,
            score,
            sources: vec![crate::types::RetrievalSource::Keyword],
        });
    }
    Ok(results)
}

async fn list_entities_impl(conn: &mut PgConnection, bank: BankId) -> Result<Vec<Entity>> {
    let rows = sqlx::query(
        "SELECT id, bank_id, canonical_name, aliases, entity_type
         FROM entities WHERE bank_id = $1
         ORDER BY canonical_name",
    )
    .bind(bank)
    .fetch_all(&mut *conn)
    .await?;
    rows.iter().map(row_to_entity).collect()
}

async fn mark_consolidated_impl(
    conn: &mut PgConnection,
    ids: &[FactId],
    at: chrono::DateTime<chrono::Utc>,
) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    let uuids: Vec<uuid::Uuid> = ids.iter().map(|id| id.to_uuid()).collect();
    sqlx::query("UPDATE facts SET consolidated_at = $1 WHERE id = ANY($2)")
        .bind(at)
        .bind(&uuids)
        .execute(&mut *conn)
        .await?;
    Ok(())
}

async fn list_banks_impl(conn: &mut PgConnection) -> Result<Vec<MemoryBank>> {
    let rows = sqlx::query(
        "SELECT id, name, mission, directives, skepticism, literalism, empathy, bias_strength, embedding_model, embedding_dims
         FROM memory_banks ORDER BY name",
    )
    .fetch_all(&mut *conn)
    .await?;
    rows.iter().map(row_to_bank).collect()
}

// ---------------------------------------------------------------------------
// PgMemoryStore — acquires a connection from the pool for each call
// ---------------------------------------------------------------------------

#[async_trait]
impl MemoryStore for PgMemoryStore {
    async fn begin(&self) -> Result<Box<dyn TransactionHandle>> {
        let txn = self.pool.begin().await?;
        Ok(Box::new(PgTransactionHandle {
            txn: tokio::sync::Mutex::new(Some(txn)),
        }))
    }

    async fn create_bank(&self, bank: &MemoryBank) -> Result<BankId> {
        let mut conn = self.pool.acquire().await?;
        create_bank_impl(&mut conn, bank).await
    }

    async fn get_bank(&self, id: BankId) -> Result<MemoryBank> {
        let mut conn = self.pool.acquire().await?;
        get_bank_impl(&mut conn, id).await
    }

    async fn insert_facts(&self, facts: &[Fact]) -> Result<Vec<FactId>> {
        let mut conn = self.pool.acquire().await?;
        insert_facts_impl(&mut conn, facts).await
    }

    async fn get_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>> {
        let mut conn = self.pool.acquire().await?;
        get_facts_impl(&mut conn, ids).await
    }

    async fn get_facts_by_bank(&self, bank: BankId, filter: FactFilter) -> Result<Vec<Fact>> {
        let mut conn = self.pool.acquire().await?;
        get_facts_by_bank_impl(&mut conn, bank, filter).await
    }

    async fn upsert_entity(&self, entity: &Entity) -> Result<EntityId> {
        let mut conn = self.pool.acquire().await?;
        upsert_entity_impl(&mut conn, entity).await
    }

    async fn find_entity(&self, bank: BankId, name: &str) -> Result<Option<Entity>> {
        let mut conn = self.pool.acquire().await?;
        find_entity_impl(&mut conn, bank, name).await
    }

    async fn get_entity_facts(&self, entity: EntityId) -> Result<Vec<Fact>> {
        let mut conn = self.pool.acquire().await?;
        get_entity_facts_impl(&mut conn, entity).await
    }

    async fn insert_links(&self, links: &[GraphLink]) -> Result<()> {
        let mut conn = self.pool.acquire().await?;
        insert_links_impl(&mut conn, links).await
    }

    async fn get_neighbors(
        &self,
        fact_id: FactId,
        link_type: Option<LinkType>,
    ) -> Result<Vec<(FactId, f32, LinkType)>> {
        let mut conn = self.pool.acquire().await?;
        get_neighbors_impl(&mut conn, fact_id, link_type).await
    }

    async fn vector_search(
        &self,
        embedding: &[f32],
        bank: BankId,
        limit: usize,
        network_filter: Option<&[NetworkType]>,
    ) -> Result<Vec<ScoredFact>> {
        let mut conn = self.pool.acquire().await?;
        vector_search_impl(&mut conn, embedding, bank, limit, network_filter).await
    }

    async fn update_fact(&self, fact: &Fact) -> Result<()> {
        let mut conn = self.pool.acquire().await?;
        update_fact_impl(&mut conn, fact).await
    }

    async fn keyword_search(
        &self,
        query: &str,
        bank: BankId,
        limit: usize,
        network_filter: Option<&[NetworkType]>,
    ) -> Result<Vec<ScoredFact>> {
        let mut conn = self.pool.acquire().await?;
        keyword_search_impl(&mut conn, query, bank, limit, network_filter).await
    }

    async fn list_entities(&self, bank: BankId) -> Result<Vec<Entity>> {
        let mut conn = self.pool.acquire().await?;
        list_entities_impl(&mut conn, bank).await
    }

    async fn mark_consolidated(
        &self,
        ids: &[FactId],
        at: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        let mut conn = self.pool.acquire().await?;
        mark_consolidated_impl(&mut conn, ids, at).await
    }

    async fn list_banks(&self) -> Result<Vec<MemoryBank>> {
        let mut conn = self.pool.acquire().await?;
        list_banks_impl(&mut conn).await
    }
}

// ---------------------------------------------------------------------------
// PgTransactionHandle — routes queries through a Postgres transaction
// ---------------------------------------------------------------------------

/// A Postgres transaction that implements [`MemoryStore`].
///
/// Rolls back automatically on drop if [`commit`](TransactionHandle::commit)
/// was not called.
pub struct PgTransactionHandle {
    txn: tokio::sync::Mutex<Option<sqlx::Transaction<'static, sqlx::Postgres>>>,
}

#[async_trait]
impl TransactionHandle for PgTransactionHandle {
    async fn commit(self: Box<Self>) -> Result<()> {
        let mut guard = self.txn.lock().await;
        let txn = guard.take().expect("transaction already consumed");
        txn.commit().await?;
        Ok(())
    }
}

/// Lock the mutex and return a mutable reference to the PgConnection inside
/// the transaction. The guard keeps the lock held for the caller.
macro_rules! txn_conn {
    ($self:expr) => {{
        $self.txn.lock().await
    }};
}

#[async_trait]
impl MemoryStore for PgTransactionHandle {
    async fn begin(&self) -> Result<Box<dyn TransactionHandle>> {
        Err(Error::Internal("nested transactions not supported".into()))
    }

    async fn create_bank(&self, bank: &MemoryBank) -> Result<BankId> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        create_bank_impl(txn, bank).await
    }

    async fn get_bank(&self, id: BankId) -> Result<MemoryBank> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        get_bank_impl(txn, id).await
    }

    async fn insert_facts(&self, facts: &[Fact]) -> Result<Vec<FactId>> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        insert_facts_impl(txn, facts).await
    }

    async fn get_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        get_facts_impl(txn, ids).await
    }

    async fn get_facts_by_bank(&self, bank: BankId, filter: FactFilter) -> Result<Vec<Fact>> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        get_facts_by_bank_impl(txn, bank, filter).await
    }

    async fn upsert_entity(&self, entity: &Entity) -> Result<EntityId> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        upsert_entity_impl(txn, entity).await
    }

    async fn find_entity(&self, bank: BankId, name: &str) -> Result<Option<Entity>> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        find_entity_impl(txn, bank, name).await
    }

    async fn get_entity_facts(&self, entity: EntityId) -> Result<Vec<Fact>> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        get_entity_facts_impl(txn, entity).await
    }

    async fn insert_links(&self, links: &[GraphLink]) -> Result<()> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        insert_links_impl(txn, links).await
    }

    async fn get_neighbors(
        &self,
        fact_id: FactId,
        link_type: Option<LinkType>,
    ) -> Result<Vec<(FactId, f32, LinkType)>> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        get_neighbors_impl(txn, fact_id, link_type).await
    }

    async fn vector_search(
        &self,
        embedding: &[f32],
        bank: BankId,
        limit: usize,
        network_filter: Option<&[NetworkType]>,
    ) -> Result<Vec<ScoredFact>> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        vector_search_impl(txn, embedding, bank, limit, network_filter).await
    }

    async fn update_fact(&self, fact: &Fact) -> Result<()> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        update_fact_impl(txn, fact).await
    }

    async fn keyword_search(
        &self,
        query: &str,
        bank: BankId,
        limit: usize,
        network_filter: Option<&[NetworkType]>,
    ) -> Result<Vec<ScoredFact>> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        keyword_search_impl(txn, query, bank, limit, network_filter).await
    }

    async fn list_entities(&self, bank: BankId) -> Result<Vec<Entity>> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        list_entities_impl(txn, bank).await
    }

    async fn mark_consolidated(
        &self,
        ids: &[FactId],
        at: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        mark_consolidated_impl(txn, ids, at).await
    }

    async fn list_banks(&self) -> Result<Vec<MemoryBank>> {
        let mut guard = txn_conn!(self);
        let txn = guard.as_mut().expect("transaction consumed");
        list_banks_impl(txn).await
    }
}
