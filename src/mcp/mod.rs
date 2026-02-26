//! MCP server adapter — exposes elephant pipelines as MCP tools.

use chrono::Utc;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::server::AppState;
use crate::types::{BankId, RecallQuery, ReflectQuery, RetainInput};

// ---------------------------------------------------------------------------
// Parameter types
// ---------------------------------------------------------------------------

/// Parameters for the retain tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RetainParams {
    /// Memory bank ID.
    pub bank_id: String,
    /// The content to store in memory.
    pub content: String,
    /// Optional context category (default: "general").
    #[serde(default)]
    pub context: Option<String>,
    /// Optional ISO 8601 timestamp for when the event occurred.
    #[serde(default)]
    pub timestamp: Option<String>,
}

/// Parameters for the recall tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallParams {
    /// Memory bank ID.
    pub bank_id: String,
    /// Natural language search query.
    pub query: String,
    /// Maximum tokens to return (default: 4096).
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

/// Parameters for the reflect tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReflectParams {
    /// Memory bank ID.
    pub bank_id: String,
    /// The question or topic to reflect on.
    pub query: String,
    /// Optional context about why this reflection is needed.
    #[serde(default)]
    pub context: Option<String>,
    /// Search budget: "low", "mid", or "high" (default: "low").
    #[serde(default = "default_budget")]
    pub budget: String,
}

/// Parameters for the create_bank tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateBankParams {
    /// Unique identifier for the bank.
    pub bank_id: String,
    /// Optional human-friendly name for the bank.
    #[serde(default)]
    pub name: Option<String>,
    /// Optional mission describing the bank's purpose.
    #[serde(default)]
    pub mission: Option<String>,
}

fn default_max_tokens() -> usize {
    4096
}

fn default_budget() -> String {
    "low".into()
}

fn budget_to_tokens(budget: &str) -> usize {
    match budget {
        "high" => 8192,
        "mid" => 4096,
        _ => 2048,
    }
}

fn parse_bank_id(s: &str) -> Result<BankId, rmcp::ErrorData> {
    s.parse::<BankId>()
        .map_err(|_| rmcp::ErrorData::invalid_params(format!("invalid bank_id: {s}"), None))
}

fn json_text<T: serde::Serialize>(val: &T) -> Result<String, rmcp::ErrorData> {
    serde_json::to_string_pretty(val)
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))
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
}

#[tool_router(router = tool_router)]
impl ElephantMcp {
    // --- Core operations ---

    /// Store information to long-term memory.
    #[tool(description = "Store information to long-term memory.")]
    async fn retain(
        &self,
        Parameters(params): Parameters<RetainParams>,
    ) -> Result<String, String> {
        let bank_id = parse_bank_id(&params.bank_id).map_err(|e| e.message.to_string())?;
        let timestamp = params
            .timestamp
            .as_deref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let input = RetainInput {
            bank_id,
            content: params.content,
            timestamp,
            turn_id: None,
            context: params.context,
            custom_instructions: None,
        };

        let output = self.state.retain.retain(&input).await.map_err(|e| e.to_string())?;
        json_text(&output).map_err(|e| e.message.to_string())
    }

    /// Search memories to provide personalized, context-aware responses.
    #[tool(description = "Search memories to provide personalized, context-aware responses.")]
    async fn recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<String, String> {
        let bank_id = parse_bank_id(&params.bank_id).map_err(|e| e.message.to_string())?;
        let query = RecallQuery {
            bank_id,
            query: params.query,
            budget_tokens: params.max_tokens,
            network_filter: None,
            temporal_anchor: None,
            tag_filter: None,
        };

        let result = self.state.recall.recall(&query).await.map_err(|e| e.to_string())?;
        json_text(&result).map_err(|e| e.message.to_string())
    }

    /// Generate thoughtful analysis by synthesizing stored memories.
    #[tool(
        description = "Generate thoughtful analysis by synthesizing stored memories with the bank's personality."
    )]
    async fn reflect(
        &self,
        Parameters(params): Parameters<ReflectParams>,
    ) -> Result<String, String> {
        let bank_id = parse_bank_id(&params.bank_id).map_err(|e| e.message.to_string())?;
        let budget_tokens = budget_to_tokens(&params.budget);

        let query = ReflectQuery {
            bank_id,
            question: params.query,
            budget_tokens,
        };

        let result = self
            .state
            .reflect
            .reflect(&query)
            .await
            .map_err(|e| e.to_string())?;
        json_text(&result).map_err(|e| e.message.to_string())
    }

    // --- Bank management ---

    /// List all available memory banks.
    #[tool(description = "List all available memory banks.")]
    async fn list_banks(&self) -> Result<String, String> {
        let banks = self.state.store.list_banks().await.map_err(|e| e.to_string())?;
        json_text(&banks).map_err(|e| e.message.to_string())
    }

    /// Create a new memory bank or get an existing one.
    #[tool(description = "Create a new memory bank or get an existing one.")]
    async fn create_bank(
        &self,
        Parameters(params): Parameters<CreateBankParams>,
    ) -> Result<String, String> {
        use crate::types::{Disposition, MemoryBank};

        // Try to parse as existing bank ID first
        if let Ok(existing_id) = params.bank_id.parse::<BankId>()
            && let Ok(bank) = self.state.store.get_bank(existing_id).await
        {
            return json_text(&bank).map_err(|e| e.message.to_string());
        }

        let bank = MemoryBank {
            id: BankId::new(),
            name: params.name.unwrap_or_else(|| params.bank_id.clone()),
            mission: params.mission.unwrap_or_default(),
            directives: vec![],
            disposition: Disposition::default(),
        };

        self.state
            .store
            .create_bank(&bank)
            .await
            .map_err(|e| e.to_string())?;
        json_text(&bank).map_err(|e| e.message.to_string())
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
