//! HTTP handler functions for the API server.

use axum::Json;
use axum::extract::{Path, State};
use serde::Deserialize;

use crate::error::{Error, Result};
use crate::types::{
    BankId, ConsolidationReport, Disposition, Entity, Fact, MemoryBank, OpinionMergeReport,
    RecallQuery, RecallResult, ReflectQuery, ReflectResult, RetainInput, RetainOutput,
};

use super::AppHandle;

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
pub async fn server_info(State(app): State<AppHandle>) -> Json<super::ServerInfo> {
    Json(app.info().clone())
}

/// GET /v1/banks
pub async fn list_banks(State(app): State<AppHandle>) -> Result<Json<Vec<MemoryBank>>> {
    let banks = app.list_banks().await?;
    Ok(Json(banks))
}

/// POST /v1/banks
pub async fn create_bank(
    State(app): State<AppHandle>,
    Json(body): Json<CreateBankRequest>,
) -> Result<Json<MemoryBank>> {
    let disposition = match body.disposition {
        Some(d) => Disposition::new(d.skepticism, d.literalism, d.empathy, d.bias_strength)?,
        None => Disposition::default(),
    };

    let bank = app.new_bank(body.name, body.mission, body.directives, disposition)?;
    app.create_bank(&bank).await?;
    Ok(Json(bank))
}

/// GET /v1/banks/:id
pub async fn get_bank(
    State(app): State<AppHandle>,
    Path(id): Path<String>,
) -> Result<Json<MemoryBank>> {
    let bank_id = parse_bank_id(&id)?;
    let bank = app.get_bank(bank_id).await?;
    Ok(Json(bank))
}

/// POST /v1/banks/:id/retain
pub async fn retain(
    State(app): State<AppHandle>,
    Path(id): Path<String>,
    Json(mut body): Json<RetainInput>,
) -> Result<Json<RetainOutput>> {
    body.bank_id = parse_bank_id(&id)?;
    let output = app.retain(&body).await?;
    Ok(Json(output))
}

/// POST /v1/banks/:id/recall
pub async fn recall(
    State(app): State<AppHandle>,
    Path(id): Path<String>,
    Json(mut body): Json<RecallQuery>,
) -> Result<Json<RecallResult>> {
    body.bank_id = parse_bank_id(&id)?;
    let result = app.recall(&body).await?;
    Ok(Json(result))
}

/// POST /v1/banks/:id/reflect
pub async fn reflect(
    State(app): State<AppHandle>,
    Path(id): Path<String>,
    Json(mut body): Json<ReflectQuery>,
) -> Result<Json<ReflectResult>> {
    body.bank_id = parse_bank_id(&id)?;
    let result = app.reflect(&body).await?;
    Ok(Json(result))
}

/// GET /v1/banks/:id/entities
pub async fn list_entities(
    State(app): State<AppHandle>,
    Path(id): Path<String>,
) -> Result<Json<Vec<Entity>>> {
    let bank_id = parse_bank_id(&id)?;
    let entities = app.list_entities(bank_id).await?;
    Ok(Json(entities))
}

/// GET /v1/banks/:id/entities/:eid/facts
pub async fn entity_facts(
    State(app): State<AppHandle>,
    Path((id, eid)): Path<(String, String)>,
) -> Result<Json<Vec<Fact>>> {
    let _bank_id = parse_bank_id(&id)?;
    let entity_id = parse_entity_id(&eid)?;
    let facts = app.entity_facts(entity_id).await?;
    Ok(Json(facts))
}

/// POST /v1/banks/:id/consolidate
pub async fn consolidate(
    State(app): State<AppHandle>,
    Path(id): Path<String>,
) -> Result<Json<ConsolidationReport>> {
    let bank_id = parse_bank_id(&id)?;
    let report = app.consolidate(bank_id).await?;
    Ok(Json(report))
}

/// POST /v1/banks/:id/merge-opinions
pub async fn merge_opinions(
    State(app): State<AppHandle>,
    Path(id): Path<String>,
) -> Result<Json<OpinionMergeReport>> {
    let bank_id = parse_bank_id(&id)?;
    let report = app.merge_opinions(bank_id).await?;
    Ok(Json(report))
}
