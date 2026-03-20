//! LLM request and response types.

use std::env;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Prompt caching configuration for a concrete provider client.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PromptCacheConfig {
    /// Disable prompt caching.
    Disabled,
    /// OpenAI prompt caching configuration.
    OpenAi(OpenAiPromptCacheConfig),
    /// Anthropic prompt caching configuration.
    Anthropic(AnthropicPromptCacheConfig),
}

/// OpenAI prompt caching settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiPromptCacheConfig {
    /// Optional routing hint to improve cache locality.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Optional prompt cache retention policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention: Option<OpenAiPromptCacheRetention>,
}

/// OpenAI prompt cache retention policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiPromptCacheRetention {
    /// Retain in memory.
    #[serde(rename = "in-memory")]
    InMemory,
    /// Retain for 24 hours.
    #[serde(rename = "24h")]
    Hours24,
}

/// Anthropic prompt caching settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicPromptCacheConfig {
    /// Optional Anthropic cache TTL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<AnthropicPromptCacheTtl>,
}

/// Anthropic prompt cache TTL.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnthropicPromptCacheTtl {
    /// Five minute TTL.
    #[serde(rename = "5m")]
    Minutes5,
    /// One hour TTL.
    #[serde(rename = "1h")]
    Hours1,
}

/// Prompt caching usage returned by a provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromptCacheUsage {
    /// OpenAI cached prompt tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
    /// Anthropic cache-hit input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<usize>,
    /// Anthropic cache-write input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<usize>,
}

/// A message in a conversation with an LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    /// The role of the message sender.
    pub role: String,
    /// The content of the message.
    pub content: String,
    /// Tool calls made by the assistant (only for role="assistant").
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Tool results returned to the assistant (only for role="user").
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_results: Vec<ToolResult>,
}

impl Message {
    /// Create a simple text message.
    pub fn text(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: vec![],
            tool_results: vec![],
        }
    }
}

/// A tool definition sent in a completion request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDef {
    /// The name of the tool.
    pub name: String,
    /// A description of what the tool does.
    pub description: String,
    /// The JSON schema describing the tool's input parameters.
    pub input_schema: serde_json::Value,
}

impl ToolDef {
    /// Build a tool definition by deriving the input schema from a `JsonSchema` type.
    pub fn from_schema<T: schemars::JsonSchema>(name: &str, description: &str) -> Self {
        let schema = schemars::schema_for!(T);
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: serde_json::to_value(schema).expect("schema serializes"),
        }
    }
}

/// A tool invocation returned by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// The unique identifier for this tool call.
    pub id: String,
    /// The name of the tool being invoked.
    pub name: String,
    /// The arguments passed to the tool as a JSON value.
    pub arguments: serde_json::Value,
}

/// A tool result to feed back to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    /// The ID of the tool call this result corresponds to.
    pub tool_call_id: String,
    /// The result content returned by the tool.
    pub content: String,
}

/// Controls which tool the LLM should use.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ToolChoice {
    /// Let the LLM choose whether to call a tool.
    Auto,
    /// The LLM must call a tool.
    Required,
    /// The LLM must not call any tool.
    None,
    /// The LLM must call the specified tool by name.
    Specific(String),
}

/// Supported reasoning effort settings for providers that expose them.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// Minimize hidden reasoning work.
    Minimal,
    /// Use a low amount of reasoning work.
    Low,
    /// Use the provider default medium reasoning work.
    Medium,
    /// Use a high amount of reasoning work.
    High,
    /// Use extra-high reasoning work when supported.
    XHigh,
    /// Disable reasoning entirely when supported.
    None,
}

impl ReasoningEffort {
    /// Read an optional reasoning effort override from an environment variable.
    pub fn from_env(name: &str) -> Result<Option<Self>> {
        let Some(raw) = env::var(name).ok() else {
            return Ok(None);
        };

        let value = raw.trim().to_ascii_lowercase();
        let effort = match value.as_str() {
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::XHigh,
            "none" => Self::None,
            _ => {
                return Err(Error::Internal(format!(
                    "{name} must be one of: none, minimal, low, medium, high, xhigh; got: {raw}"
                )));
            }
        };

        Ok(Some(effort))
    }
}

/// Per-stage reasoning-effort overrides loaded from environment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReasoningEffortConfig {
    /// Retain extraction override.
    pub retain_extract: Option<ReasoningEffort>,
    /// Retain entity-resolution override.
    pub retain_resolve: Option<ReasoningEffort>,
    /// Retain graph-builder override.
    pub retain_graph: Option<ReasoningEffort>,
    /// Reflect override.
    pub reflect: Option<ReasoningEffort>,
    /// Consolidation override.
    pub consolidate: Option<ReasoningEffort>,
    /// Opinion-merge override.
    pub opinion_merge: Option<ReasoningEffort>,
}

static REASONING_EFFORT_CONFIG: OnceLock<std::result::Result<ReasoningEffortConfig, String>> =
    OnceLock::new();

impl ReasoningEffortConfig {
    /// Read reasoning-effort overrides from environment.
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            retain_extract: ReasoningEffort::from_env("RETAIN_EXTRACT_REASONING_EFFORT")?,
            retain_resolve: ReasoningEffort::from_env("RETAIN_RESOLVE_REASONING_EFFORT")?,
            retain_graph: ReasoningEffort::from_env("RETAIN_GRAPH_REASONING_EFFORT")?,
            reflect: ReasoningEffort::from_env("REFLECT_REASONING_EFFORT")?,
            consolidate: ReasoningEffort::from_env("CONSOLIDATE_REASONING_EFFORT")?,
            opinion_merge: ReasoningEffort::from_env("OPINION_MERGE_REASONING_EFFORT")?,
        })
    }

    /// Load reasoning-effort overrides once and return the shared config.
    pub fn current() -> Result<&'static Self> {
        match REASONING_EFFORT_CONFIG
            .get_or_init(|| Self::from_env().map_err(|err| err.to_string()))
        {
            Ok(config) => Ok(config),
            Err(message) => Err(Error::Internal(message.clone())),
        }
    }
}

/// A request to an LLM for completion.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CompletionRequest {
    /// The model to use.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<usize>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Optional reasoning effort for providers that support it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Optional system prompt.
    pub system: Option<String>,
    /// Tool definitions for tool-calling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDef>>,
    /// Which tool the LLM should use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Results from previous tool calls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_results: Vec<ToolResult>,
}

/// A response from an LLM completion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionResponse {
    /// The generated text content.
    pub content: String,
    /// Number of input tokens used.
    pub input_tokens: usize,
    /// Number of output tokens generated.
    pub output_tokens: usize,
    /// Provider-specific finish or stop reason, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Tool calls requested by the LLM.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Prompt caching usage, when reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache: Option<PromptCacheUsage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_request_roundtrip() {
        let req = CompletionRequest {
            model: "test-model".into(),
            messages: vec![Message::text("user", "hello")],
            max_tokens: Some(1024),
            temperature: Some(0.7),
            reasoning_effort: None,
            system: Some("You are a helpful assistant.".into()),
            ..Default::default()
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn completion_response_roundtrip() {
        let resp = CompletionResponse {
            content: "Hello! How can I help?".into(),
            input_tokens: 10,
            output_tokens: 6,
            stop_reason: None,
            tool_calls: vec![],
            prompt_cache: Some(PromptCacheUsage {
                cached_tokens: Some(4),
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            }),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: CompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, back);
    }

    #[test]
    fn prompt_cache_config_roundtrip() {
        let config = PromptCacheConfig::OpenAi(OpenAiPromptCacheConfig {
            key: Some("elephant:reflect".into()),
            retention: Some(OpenAiPromptCacheRetention::InMemory),
        });
        let json = serde_json::to_string(&config).unwrap();
        let back: PromptCacheConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }
}
