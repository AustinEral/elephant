//! MCP server adapter — exposes elephant pipelines as MCP tools.

use chrono::DateTime;
use chrono::Utc;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use crate::server::AppState;
use crate::types::{
    BankId, Fact, FactId, MemoryBank, RecallQuery, ReflectQuery, RetainInput, RetrievalSource,
    TemporalRange, TurnId,
};

// ---------------------------------------------------------------------------
// Parameter types
// ---------------------------------------------------------------------------

/// Search budget presets for MCP callers.
#[derive(Debug, Clone, Copy, Default, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchBudget {
    /// Smallest retrieval budget.
    #[default]
    Low,
    /// Medium retrieval budget.
    Mid,
    /// Largest retrieval budget.
    High,
}

impl SearchBudget {
    fn max_tokens(self) -> usize {
        match self {
            Self::Low => 2048,
            Self::Mid => 4096,
            Self::High => 8192,
        }
    }
}

/// Parameters for the retain tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RetainParams {
    /// ID of the memory to use.
    pub bank_id: String,
    /// New information to remember.
    pub content: String,
    /// Optional surrounding context for what should be remembered.
    #[serde(default)]
    pub context: Option<String>,
    /// Optional ISO 8601 timestamp for when the remembered event occurred.
    #[serde(default)]
    pub timestamp: Option<String>,
}

/// Parameters for the recall tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallParams {
    /// ID of the memory to inspect.
    pub bank_id: String,
    /// What remembered details to inspect directly.
    pub query: String,
    /// Optional maximum token budget override for this inspection call.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Optional temporal anchor used when inspecting time-sensitive memory.
    #[serde(default)]
    pub temporal_anchor: Option<crate::types::TemporalRange>,
}

/// Parameters for the reflect tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReflectParams {
    /// ID of the memory to use.
    pub bank_id: String,
    /// What you want to remember and answer.
    pub query: String,
    /// Optional context for this memory task.
    #[serde(default)]
    pub context: Option<String>,
    /// Optional current-time anchor for time-sensitive remembering.
    #[serde(default)]
    pub temporal_context: Option<String>,
    /// Search budget: "low", "mid", or "high" (default: "low").
    #[serde(default)]
    pub budget: SearchBudget,
}

/// Parameters for the create_bank tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateBankParams {
    /// Human-friendly name for the memory.
    pub name: String,
    /// Optional mission describing what this memory is for.
    #[serde(default)]
    pub mission: Option<String>,
}

/// Parameters for the get_bank tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetBankParams {
    /// Memory ID.
    pub bank_id: String,
}

fn parse_bank_id(s: &str) -> Result<BankId, rmcp::ErrorData> {
    s.parse::<BankId>()
        .map_err(|_| rmcp::ErrorData::invalid_params(format!("invalid bank_id: {s}"), None))
}

fn parse_timestamp_or_now(raw: Option<&str>) -> Result<DateTime<Utc>, rmcp::ErrorData> {
    match raw {
        Some(raw) => chrono::DateTime::parse_from_rfc3339(raw)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|_| {
                rmcp::ErrorData::invalid_params(format!("invalid timestamp: {raw}"), None)
            }),
        None => Ok(Utc::now()),
    }
}

fn normalize_optional_text(raw: Option<String>) -> Option<String> {
    raw.and_then(|text| {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn require_text(raw: String, field_name: &str) -> Result<String, rmcp::ErrorData> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        Err(rmcp::ErrorData::invalid_params(
            format!("{field_name} cannot be blank"),
            None,
        ))
    } else {
        Ok(trimmed.to_string())
    }
}

fn json_text<T: serde::Serialize>(val: &T) -> Result<String, rmcp::ErrorData> {
    serde_json::to_string_pretty(val)
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))
}

#[derive(Serialize)]
struct McpBankView {
    #[serde(flatten)]
    bank: MemoryBank,
    active_runtime: crate::server::ServerInfo,
}

#[derive(Serialize)]
struct McpRecallFactView {
    id: FactId,
    content: String,
    score: f32,
    network: crate::types::NetworkType,
    #[serde(skip_serializing_if = "Option::is_none")]
    temporal_range: Option<TemporalRange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_turn_id: Option<TurnId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    retrieval_sources: Vec<RetrievalSource>,
}

#[derive(Serialize)]
struct McpRecallView {
    facts: Vec<McpRecallFactView>,
    total_tokens: usize,
}

#[derive(Serialize)]
struct McpOpinionView {
    id: FactId,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    evidence_ids: Vec<FactId>,
}

#[derive(Serialize)]
struct McpReflectView {
    response: String,
    sources: Vec<FactId>,
    new_opinions: Vec<McpOpinionView>,
    confidence: f32,
}

impl RetainParams {
    fn into_input(self) -> Result<RetainInput, rmcp::ErrorData> {
        Ok(RetainInput {
            bank_id: parse_bank_id(&self.bank_id)?,
            content: self.content,
            timestamp: parse_timestamp_or_now(self.timestamp.as_deref())?,
            turn_id: None,
            context: normalize_optional_text(self.context),
            custom_instructions: None,
            speaker: None,
        })
    }
}

impl RecallParams {
    fn into_query(self) -> Result<RecallQuery, rmcp::ErrorData> {
        let mut query = RecallQuery::new(parse_bank_id(&self.bank_id)?, self.query);
        if let Some(max_tokens) = self.max_tokens {
            query = query.with_budget_tokens(max_tokens);
        }
        if let Some(anchor) = self.temporal_anchor {
            query = query.with_temporal_anchor(anchor);
        }
        Ok(query)
    }
}

impl ReflectParams {
    fn into_query(self) -> Result<ReflectQuery, rmcp::ErrorData> {
        Ok(ReflectQuery {
            bank_id: parse_bank_id(&self.bank_id)?,
            question: self.query,
            context: normalize_optional_text(self.context),
            budget_tokens: self.budget.max_tokens(),
            temporal_context: normalize_optional_text(self.temporal_context),
        })
    }
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

/// MCP server handler that wraps elephant's pipelines and store.
#[derive(Clone)]
pub struct ElephantMcp {
    state: AppState,
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl ElephantMcp {
    /// Create a new MCP server handler from shared application state.
    pub fn new(state: AppState) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    fn bank_view(&self, bank: MemoryBank) -> McpBankView {
        McpBankView {
            bank,
            active_runtime: self.state.info.clone(),
        }
    }

    fn recall_view(&self, result: crate::types::RecallResult) -> McpRecallView {
        McpRecallView {
            facts: result
                .facts
                .into_iter()
                .map(|scored| McpRecallFactView {
                    id: scored.fact.id,
                    content: scored.fact.content,
                    score: scored.score,
                    network: scored.fact.network,
                    temporal_range: scored.fact.temporal_range,
                    source_turn_id: scored.fact.source_turn_id,
                    retrieval_sources: scored.sources,
                })
                .collect(),
            total_tokens: result.total_tokens,
        }
    }

    fn opinion_view(&self, opinion: Fact) -> McpOpinionView {
        McpOpinionView {
            id: opinion.id,
            content: opinion.content,
            confidence: opinion.confidence,
            evidence_ids: opinion.evidence_ids,
        }
    }

    fn reflect_view(&self, result: crate::types::ReflectResult) -> McpReflectView {
        McpReflectView {
            response: result.response,
            sources: result.sources,
            new_opinions: result
                .new_opinions
                .into_iter()
                .map(|opinion| self.opinion_view(opinion))
                .collect(),
            confidence: result.confidence,
        }
    }
}

#[tool_router(router = tool_router)]
impl ElephantMcp {
    // --- Core operations ---

    /// Remember new information.
    #[tool(description = "Remember new information so it becomes part of durable memory.")]
    async fn retain(&self, Parameters(params): Parameters<RetainParams>) -> Result<String, String> {
        let input = params.into_input().map_err(|e| e.message.to_string())?;

        let output = self
            .state
            .retain
            .retain(&input)
            .await
            .map_err(|e| e.to_string())?;
        json_text(&output).map_err(|e| e.message.to_string())
    }

    /// Inspect remembered facts directly.
    #[tool(
        description = "Inspect specific stored memories directly. Use this only when you need to inspect or verify raw remembered details."
    )]
    async fn recall(&self, Parameters(params): Parameters<RecallParams>) -> Result<String, String> {
        let query = params.into_query().map_err(|e| e.message.to_string())?;

        let result = self
            .state
            .recall
            .recall(&query)
            .await
            .map_err(|e| e.to_string())?;
        json_text(&self.recall_view(result)).map_err(|e| e.message.to_string())
    }

    /// Remember and answer from what you know.
    #[tool(
        description = "Use your memory to answer. This is the primary memory interface."
    )]
    async fn reflect(
        &self,
        Parameters(params): Parameters<ReflectParams>,
    ) -> Result<String, String> {
        let query = params.into_query().map_err(|e| e.message.to_string())?;

        let result = self
            .state
            .reflect
            .reflect(&query)
            .await
            .map_err(|e| e.to_string())?;
        json_text(&self.reflect_view(result)).map_err(|e| e.message.to_string())
    }

    // --- Bank management ---

    /// List all available memories.
    #[tool(
        description = "List the available memories you can use, including active runtime configuration."
    )]
    async fn list_banks(&self) -> Result<String, String> {
        let banks = self
            .state
            .store
            .list_banks()
            .await
            .map_err(|e| e.to_string())?;
        let views: Vec<McpBankView> = banks.into_iter().map(|bank| self.bank_view(bank)).collect();
        json_text(&views).map_err(|e| e.message.to_string())
    }

    /// Get one memory by ID.
    #[tool(description = "Get one memory by ID, including active runtime configuration.")]
    async fn get_bank(
        &self,
        Parameters(params): Parameters<GetBankParams>,
    ) -> Result<String, String> {
        let bank = self
            .state
            .store
            .get_bank(parse_bank_id(&params.bank_id).map_err(|e| e.message.to_string())?)
            .await
            .map_err(|e| e.to_string())?;
        json_text(&self.bank_view(bank)).map_err(|e| e.message.to_string())
    }

    /// Create a new memory.
    #[tool(
        description = "Create a new memory and return it with active runtime configuration."
    )]
    async fn create_bank(
        &self,
        Parameters(params): Parameters<CreateBankParams>,
    ) -> Result<String, String> {
        use crate::types::Disposition;

        let id = BankId::new();
        let name = require_text(params.name, "name").map_err(|e| e.message.to_string())?;

        let bank = MemoryBank {
            id,
            name,
            mission: normalize_optional_text(params.mission).unwrap_or_default(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: self.state.embeddings.model_name().to_string(),
            embedding_dimensions: self.state.embeddings.dimensions() as u16,
        };

        self.state
            .store
            .create_bank(&bank)
            .await
            .map_err(|e| e.to_string())?;
        json_text(&self.bank_view(bank)).map_err(|e| e.message.to_string())
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ElephantMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "elephant".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                ..Default::default()
            },
            instructions: Some(
                "Elephant memory engine — store, retrieve, and reason over long-term memories."
                    .into(),
            ),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;

    use crate::consolidation::{ConsolidationProgress, Consolidator, OpinionMerger};
    use crate::embedding::mock::MockEmbeddings;
    use crate::error::Result;
    use crate::recall::RecallPipeline;
    use crate::reflect::ReflectPipeline;
    use crate::retain::RetainPipeline;
    use crate::server::{AppState, ServerInfo};
    use crate::server::{
        ServerBackgroundConsolidationInfo, ServerConsolidationRuntimeInfo, ServerModelsInfo,
        ServerReflectInfo, ServerRetrievalInfo,
    };
    use crate::storage::{MemoryStore, mock::MockMemoryStore};
    use crate::types::{
        ConsolidationReport, Disposition, Fact, FactId, FactType, MemoryBank, NetworkType,
        OpinionMergeReport, RecallResult, ReflectResult, RetainOutput, RetrievalSource,
        ScoredFact,
    };
    use chrono::Utc;

    struct CapturingRetainPipeline {
        last_input: Arc<Mutex<Option<RetainInput>>>,
    }

    #[async_trait]
    impl RetainPipeline for CapturingRetainPipeline {
        async fn retain(&self, input: &RetainInput) -> Result<RetainOutput> {
            *self.last_input.lock().expect("retain mutex poisoned") = Some(input.clone());
            Ok(RetainOutput {
                fact_ids: vec![],
                facts_stored: 1,
                new_entities: vec![],
                entities_resolved: 0,
                links_created: 0,
                opinions_reinforced: 0,
                opinions_weakened: 0,
            })
        }
    }

    struct CapturingRecallPipeline {
        last_query: Arc<Mutex<Option<RecallQuery>>>,
    }

    #[async_trait]
    impl RecallPipeline for CapturingRecallPipeline {
        async fn recall(&self, query: &RecallQuery) -> Result<RecallResult> {
            *self.last_query.lock().expect("recall mutex poisoned") = Some(query.clone());
            Ok(RecallResult {
                facts: vec![ScoredFact {
                    fact: Fact {
                        id: FactId::new(),
                        bank_id: query.bank_id,
                        content: "Austin's name is Austin.".into(),
                        fact_type: FactType::World,
                        network: NetworkType::World,
                        entity_ids: vec![],
                        temporal_range: None,
                        embedding: Some(vec![0.1, 0.2, 0.3]),
                        confidence: None,
                        evidence_ids: vec![],
                        source_turn_id: None,
                        created_at: Utc::now(),
                        updated_at: Utc::now(),
                        consolidated_at: None,
                    },
                    score: 0.95,
                    sources: vec![RetrievalSource::Semantic, RetrievalSource::Graph],
                }],
                total_tokens: 12,
            })
        }
    }

    struct CapturingReflectPipeline {
        last_query: Arc<Mutex<Option<ReflectQuery>>>,
    }

    #[async_trait]
    impl ReflectPipeline for CapturingReflectPipeline {
        async fn reflect(&self, query: &ReflectQuery) -> Result<ReflectResult> {
            *self.last_query.lock().expect("reflect mutex poisoned") = Some(query.clone());
            Ok(ReflectResult {
                response: "ok".into(),
                sources: vec![FactId::new()],
                new_opinions: vec![Fact {
                    id: FactId::new(),
                    bank_id: query.bank_id,
                    content: "Austin likes concise tool output.".into(),
                    fact_type: FactType::Experience,
                    network: NetworkType::Opinion,
                    entity_ids: vec![],
                    temporal_range: None,
                    embedding: Some(vec![0.1, 0.2, 0.3]),
                    confidence: Some(0.9),
                    evidence_ids: vec![FactId::new()],
                    source_turn_id: None,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    consolidated_at: None,
                }],
                confidence: 0.5,
                retrieved_context: vec![crate::types::RetrievedFact {
                    id: FactId::new(),
                    content: "Austin's name is Austin.".into(),
                    score: 0.95,
                    network: NetworkType::World,
                    source_turn_id: None,
                    evidence_ids: vec![],
                    retrieval_sources: vec![RetrievalSource::Semantic],
                    support_turn_ids: vec![],
                }],
                retrieved_sources: vec![],
                trace: vec![crate::types::ReflectTraceStep {
                    iteration: 0,
                    tool_name: "recall".into(),
                    query: "name".into(),
                    returned_fact_ids: vec![],
                    requested_fact_ids: vec![],
                    new_fact_ids: vec![],
                    returned_source_ids: vec![],
                    facts_returned: 1,
                    total_tokens: 12,
                    latency_ms: 10,
                }],
                final_done: Some(crate::types::ReflectDoneTrace {
                    iteration: 1,
                    assistant_content: "done".into(),
                    raw_arguments: serde_json::json!({"response":"ok","source_ids":[]}),
                    used_fallback: false,
                    parse_error: None,
                    stop_reason: Some(crate::types::ReflectStopReason::Completed),
                    response: "ok".into(),
                    source_ids: vec![],
                }),
            })
        }
    }

    struct NoOpConsolidator;

    #[async_trait]
    impl Consolidator for NoOpConsolidator {
        async fn consolidate_with_progress(
            &self,
            _bank_id: BankId,
            _progress: Option<tokio::sync::mpsc::UnboundedSender<ConsolidationProgress>>,
        ) -> Result<ConsolidationReport> {
            Ok(ConsolidationReport::default())
        }
    }

    struct NoOpOpinionMerger;

    #[async_trait]
    impl OpinionMerger for NoOpOpinionMerger {
        async fn merge(&self, _bank_id: BankId) -> Result<OpinionMergeReport> {
            Ok(OpinionMergeReport::default())
        }
    }

    fn test_state(
        retain_input: Arc<Mutex<Option<RetainInput>>>,
        recall_query: Arc<Mutex<Option<RecallQuery>>>,
        reflect_query: Arc<Mutex<Option<ReflectQuery>>>,
        store: Arc<MockMemoryStore>,
    ) -> AppState {
        AppState {
            info: ServerInfo {
                version: env!("CARGO_PKG_VERSION").into(),
                models: ServerModelsInfo {
                    retain: "test".into(),
                    reflect: "test".into(),
                    embedding: "mock".into(),
                    reranker: "none".into(),
                },
                retrieval: ServerRetrievalInfo {
                    retriever_limit: 20,
                    max_facts: 50,
                },
                reflect: ServerReflectInfo {
                    max_iterations: 8,
                    max_tokens: None,
                    source_lookup_enabled: true,
                },
                consolidation: ServerConsolidationRuntimeInfo {
                    batch_size: 16,
                    max_tokens: 2048,
                    recall_budget: 1024,
                },
                server_consolidation: ServerBackgroundConsolidationInfo {
                    enabled: true,
                    min_facts: 32,
                    cooldown_secs: 30,
                    merge_opinions_after: false,
                },
            },
            retain: Arc::new(CapturingRetainPipeline {
                last_input: retain_input,
            }),
            recall: Arc::new(CapturingRecallPipeline {
                last_query: recall_query,
            }),
            reflect: Arc::new(CapturingReflectPipeline {
                last_query: reflect_query,
            }),
            consolidator: Arc::new(NoOpConsolidator),
            opinion_merger: Arc::new(NoOpOpinionMerger),
            store,
            embeddings: Arc::new(MockEmbeddings::new(384)),
        }
    }

    #[tokio::test]
    async fn retain_tool_preserves_public_optional_fields() {
        let retained = Arc::new(Mutex::new(None));
        let recalled = Arc::new(Mutex::new(None));
        let reflected = Arc::new(Mutex::new(None));
        let store = Arc::new(MockMemoryStore::new());
        let mcp = ElephantMcp::new(test_state(retained.clone(), recalled, reflected, store));
        let bank_id = BankId::new();

        let _ = mcp
            .retain(Parameters(RetainParams {
                bank_id: bank_id.to_string(),
                content: "Alice prefers Rust".into(),
                context: Some("chat memory".into()),
                timestamp: Some("2024-03-01T00:00:00Z".into()),
            }))
            .await
            .expect("retain tool should succeed");

        let captured = retained
            .lock()
            .expect("retain mutex poisoned")
            .clone()
            .expect("retain input should be captured");
        assert_eq!(captured.bank_id, bank_id);
        assert_eq!(captured.context.as_deref(), Some("chat memory"));
        assert_eq!(captured.timestamp.to_rfc3339(), "2024-03-01T00:00:00+00:00");
        assert!(captured.turn_id.is_none());
        assert!(captured.custom_instructions.is_none());
        assert!(captured.speaker.is_none());
    }

    #[tokio::test]
    async fn retain_tool_rejects_invalid_timestamp() {
        let retained = Arc::new(Mutex::new(None));
        let recalled = Arc::new(Mutex::new(None));
        let reflected = Arc::new(Mutex::new(None));
        let store = Arc::new(MockMemoryStore::new());
        let mcp = ElephantMcp::new(test_state(retained, recalled, reflected, store));

        let err = mcp
            .retain(Parameters(RetainParams {
                bank_id: BankId::new().to_string(),
                content: "bad timestamp".into(),
                context: None,
                timestamp: Some("not-a-timestamp".into()),
            }))
            .await
            .expect_err("invalid timestamp should be rejected");

        assert!(err.contains("invalid timestamp"));
    }

    #[tokio::test]
    async fn reflect_tool_maps_budget_context_and_temporal_context() {
        let retained = Arc::new(Mutex::new(None));
        let recalled = Arc::new(Mutex::new(None));
        let reflected = Arc::new(Mutex::new(None));
        let store = Arc::new(MockMemoryStore::new());
        let mcp = ElephantMcp::new(test_state(retained, recalled, reflected.clone(), store));

        let raw = mcp
            .reflect(Parameters(ReflectParams {
                bank_id: BankId::new().to_string(),
                query: "What changed?".into(),
                context: Some("release review".into()),
                temporal_context: Some("2026-03-23".into()),
                budget: SearchBudget::High,
            }))
            .await
            .expect("reflect tool should succeed");

        let captured = reflected
            .lock()
            .expect("reflect mutex poisoned")
            .clone()
            .expect("reflect query should be captured");
        assert_eq!(captured.question, "What changed?");
        assert_eq!(captured.context.as_deref(), Some("release review"));
        assert_eq!(captured.temporal_context.as_deref(), Some("2026-03-23"));
        assert_eq!(captured.budget_tokens, 8192);

        let value: serde_json::Value =
            serde_json::from_str(&raw).expect("reflect returns json object");
        assert_eq!(value["response"], "ok");
        assert_eq!(value["confidence"], 0.5);
        assert_eq!(
            value["new_opinions"][0]["content"],
            "Austin likes concise tool output."
        );
        assert_eq!(value["new_opinions"][0]["confidence"], 0.9);
        assert!(value.get("retrieved_context").is_none());
        assert!(value.get("trace").is_none());
        assert!(value.get("final_done").is_none());
    }

    #[tokio::test]
    async fn get_bank_tool_returns_existing_bank() {
        let retained = Arc::new(Mutex::new(None));
        let recalled = Arc::new(Mutex::new(None));
        let reflected = Arc::new(Mutex::new(None));
        let store = Arc::new(MockMemoryStore::new());
        let existing_bank = MemoryBank {
            id: BankId::new(),
            name: "engineering".into(),
            mission: "remember project decisions".into(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: "mock".into(),
            embedding_dimensions: 384,
        };
        store
            .create_bank(&existing_bank)
            .await
            .expect("seed bank should succeed");
        let mcp = ElephantMcp::new(test_state(retained, recalled, reflected, store));

        let raw = mcp
            .get_bank(Parameters(GetBankParams {
                bank_id: existing_bank.id.to_string(),
            }))
            .await
            .expect("get_bank should succeed");

        let value: serde_json::Value =
            serde_json::from_str(&raw).expect("get_bank returns json object");
        let bank: MemoryBank =
            serde_json::from_value(value.clone()).expect("get_bank returns bank fields");
        assert_eq!(bank.id, existing_bank.id);
        assert_eq!(bank.name, "engineering");
        assert_eq!(bank.mission, "remember project decisions");
        assert_eq!(bank.disposition, Disposition::default());
        assert_eq!(value["active_runtime"]["models"]["retain"], "test");
        assert_eq!(value["active_runtime"]["models"]["reflect"], "test");
        assert_eq!(value["active_runtime"]["models"]["embedding"], "mock");
        assert_eq!(value["active_runtime"]["models"]["reranker"], "none");
        assert_eq!(value["active_runtime"]["retrieval"]["retriever_limit"], 20);
    }

    #[tokio::test]
    async fn create_bank_tool_creates_new_bank() {
        let retained = Arc::new(Mutex::new(None));
        let recalled = Arc::new(Mutex::new(None));
        let reflected = Arc::new(Mutex::new(None));
        let store = Arc::new(MockMemoryStore::new());
        let mcp = ElephantMcp::new(test_state(retained, recalled, reflected, store));

        let raw = mcp
            .create_bank(Parameters(CreateBankParams {
                name: "engineering".into(),
                mission: Some("remember project decisions".into()),
            }))
            .await
            .expect("create_bank should succeed");

        let value: serde_json::Value =
            serde_json::from_str(&raw).expect("create_bank returns json object");
        let bank: MemoryBank =
            serde_json::from_value(value.clone()).expect("create_bank returns bank fields");
        assert_eq!(bank.name, "engineering");
        assert_eq!(bank.mission, "remember project decisions");
        assert_eq!(bank.disposition, Disposition::default());
        assert_eq!(value["active_runtime"]["models"]["retain"], "test");
        assert_eq!(value["active_runtime"]["models"]["reflect"], "test");
        assert_eq!(value["active_runtime"]["models"]["embedding"], "mock");
        assert_eq!(value["active_runtime"]["reflect"]["max_iterations"], 8);
    }

    #[tokio::test]
    async fn list_banks_tool_includes_active_runtime_configuration() {
        let retained = Arc::new(Mutex::new(None));
        let recalled = Arc::new(Mutex::new(None));
        let reflected = Arc::new(Mutex::new(None));
        let store = Arc::new(MockMemoryStore::new());
        let bank = MemoryBank {
            id: BankId::new(),
            name: "engineering".into(),
            mission: "remember project decisions".into(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: "mock".into(),
            embedding_dimensions: 384,
        };
        store
            .create_bank(&bank)
            .await
            .expect("seed bank should succeed");
        let mcp = ElephantMcp::new(test_state(retained, recalled, reflected, store));

        let raw = mcp.list_banks().await.expect("list_banks should succeed");

        let value: serde_json::Value =
            serde_json::from_str(&raw).expect("list_banks returns json array");
        let banks = value.as_array().expect("list_banks returns an array");
        assert_eq!(banks.len(), 1);
        assert_eq!(banks[0]["id"], bank.id.to_string());
        assert_eq!(banks[0]["name"], "engineering");
        assert_eq!(banks[0]["active_runtime"]["models"]["retain"], "test");
        assert_eq!(banks[0]["active_runtime"]["models"]["embedding"], "mock");
        assert_eq!(
            banks[0]["active_runtime"]["server_consolidation"]["enabled"],
            true
        );
    }

    #[tokio::test]
    async fn create_bank_tool_rejects_blank_name() {
        let retained = Arc::new(Mutex::new(None));
        let recalled = Arc::new(Mutex::new(None));
        let reflected = Arc::new(Mutex::new(None));
        let store = Arc::new(MockMemoryStore::new());
        let mcp = ElephantMcp::new(test_state(retained, recalled, reflected, store));

        let err = mcp
            .create_bank(Parameters(CreateBankParams {
                name: "   ".into(),
                mission: None,
            }))
            .await
            .expect_err("blank name should fail");

        assert!(err.contains("name cannot be blank"));
    }

    #[tokio::test]
    async fn recall_tool_preserves_optional_budget_and_temporal_anchor() {
        let retained = Arc::new(Mutex::new(None));
        let recalled = Arc::new(Mutex::new(None));
        let reflected = Arc::new(Mutex::new(None));
        let store = Arc::new(MockMemoryStore::new());
        let mcp = ElephantMcp::new(test_state(retained, recalled.clone(), reflected, store));
        let bank_id = BankId::new();

        let raw = mcp
            .recall(Parameters(RecallParams {
                bank_id: bank_id.to_string(),
                query: "release notes".into(),
                max_tokens: Some(1234),
                temporal_anchor: Some(crate::types::TemporalRange {
                    start: Some(
                        chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                            .expect("valid start")
                            .with_timezone(&Utc),
                    ),
                    end: Some(
                        chrono::DateTime::parse_from_rfc3339("2024-01-31T23:59:59Z")
                            .expect("valid end")
                            .with_timezone(&Utc),
                    ),
                }),
            }))
            .await
            .expect("recall tool should succeed");

        let captured = recalled
            .lock()
            .expect("recall mutex poisoned")
            .clone()
            .expect("recall query should be captured");
        assert_eq!(captured.bank_id, bank_id);
        assert_eq!(captured.query, "release notes");
        assert_eq!(captured.budget_tokens, Some(1234));
        assert_eq!(captured.max_facts, None);
        let anchor = captured
            .temporal_anchor
            .expect("temporal anchor should be preserved");
        assert_eq!(
            anchor.start.expect("start").to_rfc3339(),
            "2024-01-01T00:00:00+00:00"
        );
        assert_eq!(
            anchor.end.expect("end").to_rfc3339(),
            "2024-01-31T23:59:59+00:00"
        );

        let value: serde_json::Value =
            serde_json::from_str(&raw).expect("recall returns json object");
        assert_eq!(value["facts"][0]["content"], "Austin's name is Austin.");
        assert_eq!(value["facts"][0]["network"], "world");
        assert_eq!(value["facts"][0]["retrieval_sources"][0], "semantic");
        assert!(value["facts"][0].get("bank_id").is_none());
        assert!(value["facts"][0].get("embedding").is_none());
        assert_eq!(value["total_tokens"], 12);
    }

    #[tokio::test]
    async fn recall_tool_omits_budget_override_when_max_tokens_is_missing() {
        let retained = Arc::new(Mutex::new(None));
        let recalled = Arc::new(Mutex::new(None));
        let reflected = Arc::new(Mutex::new(None));
        let store = Arc::new(MockMemoryStore::new());
        let mcp = ElephantMcp::new(test_state(retained, recalled.clone(), reflected, store));

        let _ = mcp
            .recall(Parameters(RecallParams {
                bank_id: BankId::new().to_string(),
                query: "release notes".into(),
                max_tokens: None,
                temporal_anchor: None,
            }))
            .await
            .expect("recall tool should succeed");

        let captured = recalled
            .lock()
            .expect("recall mutex poisoned")
            .clone()
            .expect("recall query should be captured");
        assert_eq!(captured.budget_tokens, None);
        assert_eq!(captured.max_facts, None);
        assert_eq!(captured.temporal_anchor, None);
    }
}
