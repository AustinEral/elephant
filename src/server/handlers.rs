//! HTTP handler functions for the API server.

use axum::extract::{Path, State};
use axum::Json;
use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::error::{Error, Result};
use crate::types::{
    BankId, ConsolidationReport, Disposition, Entity, Fact, MemoryBank,
    OpinionMergeReport, RecallQuery, RecallResult, ReflectQuery, ReflectResult, RetainInput,
    RetainOutput,
};

use super::AppState;

// --- Request types ---

/// Request body for creating a new memory bank.
#[derive(Debug, Deserialize)]
pub struct CreateBankRequest {
    /// Human-readable name.
    pub name: String,
    /// What this bank should focus on retaining.
    pub mission: String,
    /// Guardrails and compliance rules.
    #[serde(default)]
    pub directives: Vec<String>,
    /// Personality parameters.
    #[serde(default)]
    pub disposition: Option<DispositionInput>,
}

/// Raw disposition fields for deserialization.
#[derive(Debug, Deserialize)]
pub struct DispositionInput {
    /// How much evidence is required before accepting claims (1-5).
    pub skepticism: u8,
    /// How literally to interpret statements (1-5).
    pub literalism: u8,
    /// Weight given to emotional/human factors (1-5).
    pub empathy: u8,
    /// How strongly the disposition influences responses (0.0-1.0).
    pub bias_strength: f32,
}

/// Request body for the consolidate endpoint.
#[derive(Debug, Deserialize)]
pub struct ConsolidateRequest {
    /// Only consolidate facts created since this timestamp.
    pub since: DateTime<Utc>,
}

// --- Helpers ---

fn parse_bank_id(id: &str) -> Result<BankId> {
    id.parse::<BankId>()
        .map_err(|_| Error::InvalidId(format!("invalid bank id: {id}")))
}

fn parse_entity_id(id: &str) -> Result<crate::types::EntityId> {
    id.parse::<crate::types::EntityId>()
        .map_err(|_| Error::InvalidId(format!("invalid entity id: {id}")))
}

// --- Handlers ---

/// GET /v1/info
pub async fn server_info(
    State(state): State<AppState>,
) -> Json<super::ServerInfo> {
    Json(state.info.clone())
}

/// GET /v1/banks
pub async fn list_banks(
    State(state): State<AppState>,
) -> Result<Json<Vec<MemoryBank>>> {
    let banks = state.store.list_banks().await?;
    Ok(Json(banks))
}

/// POST /v1/banks
pub async fn create_bank(
    State(state): State<AppState>,
    Json(body): Json<CreateBankRequest>,
) -> Result<Json<MemoryBank>> {
    let disposition = match body.disposition {
        Some(d) => Disposition::new(d.skepticism, d.literalism, d.empathy, d.bias_strength)?,
        None => Disposition::default(),
    };

    let bank = MemoryBank {
        id: BankId::new(),
        name: body.name,
        mission: body.mission,
        directives: body.directives,
        disposition,
        embedding_model: state.embeddings.model_name().to_string(),
        embedding_dimensions: state.embeddings.dimensions() as u16,
    };

    state.store.create_bank(&bank).await?;
    Ok(Json(bank))
}

/// GET /v1/banks/:id
pub async fn get_bank(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<MemoryBank>> {
    let bank_id = parse_bank_id(&id)?;
    let bank = state.store.get_bank(bank_id).await?;
    Ok(Json(bank))
}

/// POST /v1/banks/:id/retain
pub async fn retain(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(mut body): Json<RetainInput>,
) -> Result<Json<RetainOutput>> {
    body.bank_id = parse_bank_id(&id)?;
    let output = state.retain.retain(&body).await?;
    Ok(Json(output))
}

/// POST /v1/banks/:id/recall
pub async fn recall(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(mut body): Json<RecallQuery>,
) -> Result<Json<RecallResult>> {
    body.bank_id = parse_bank_id(&id)?;
    let result = state.recall.recall(&body).await?;
    Ok(Json(result))
}

/// POST /v1/banks/:id/reflect
pub async fn reflect(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(mut body): Json<ReflectQuery>,
) -> Result<Json<ReflectResult>> {
    body.bank_id = parse_bank_id(&id)?;
    let result = state.reflect.reflect(&body).await?;
    Ok(Json(result))
}

/// GET /v1/banks/:id/entities
pub async fn list_entities(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<Entity>>> {
    let bank_id = parse_bank_id(&id)?;
    let entities = state.store.list_entities(bank_id).await?;
    Ok(Json(entities))
}

/// GET /v1/banks/:id/entities/:eid/facts
pub async fn entity_facts(
    State(state): State<AppState>,
    Path((id, eid)): Path<(String, String)>,
) -> Result<Json<Vec<Fact>>> {
    let _bank_id = parse_bank_id(&id)?;
    let entity_id = parse_entity_id(&eid)?;
    let facts = state.store.get_entity_facts(entity_id).await?;
    Ok(Json(facts))
}

/// POST /v1/banks/:id/consolidate
pub async fn consolidate(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ConsolidateRequest>,
) -> Result<Json<ConsolidationReport>> {
    let bank_id = parse_bank_id(&id)?;
    let report = state.consolidator.consolidate(bank_id, body.since).await?;
    Ok(Json(report))
}

/// POST /v1/banks/:id/merge-opinions
pub async fn merge_opinions(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<OpinionMergeReport>> {
    let bank_id = parse_bank_id(&id)?;
    let report = state.opinion_merger.merge(bank_id).await?;
    Ok(Json(report))
}

