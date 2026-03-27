//! Provider-agnostic request and response types for LLM clients.

use std::env;
use std::sync::OnceLock;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

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

/// Supported message roles in a completion transcript.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    /// A user-authored message.
    User,
    /// An assistant-authored message.
    Assistant,
}

impl MessageRole {
    /// Return the canonical wire-format role string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// A message in a conversation transcript.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    role: MessageRole,
    content: String,
    /// Tool calls made by the assistant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCall>,
    /// Tool results returned to the assistant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_results: Vec<ToolResult>,
}

impl Message {
    /// Create a plain message with the given role and content.
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }

    /// Create an assistant message that includes tool calls.
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_calls,
            tool_results: Vec::new(),
        }
    }

    /// Create a transcript entry containing one tool result.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::with_tool_results(vec![ToolResult {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }])
    }

    /// Create a transcript entry containing one or more tool results.
    pub fn with_tool_results(tool_results: Vec<ToolResult>) -> Self {
        Self {
            role: MessageRole::User,
            content: String::new(),
            tool_calls: Vec::new(),
            tool_results,
        }
    }

    /// Return the message role.
    pub fn role(&self) -> MessageRole {
        self.role
    }

    /// Return the plain-text message content.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Return tool calls attached to this message.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    /// Return tool results attached to this message.
    pub fn tool_results(&self) -> &[ToolResult] {
        &self.tool_results
    }

    /// Return tool results attached to this message.
    pub fn tool_results_ref(&self) -> &[ToolResult] {
        &self.tool_results
    }
}

/// A tool definition sent in a completion request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

impl ToolDefinition {
    /// Create a tool definition from a name, description, and JSON schema.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }

    /// Build a tool definition by deriving the input schema from a [`JsonSchema`] type.
    pub fn from_schema<T: JsonSchema>(name: &str, description: &str) -> Self {
        let schema = schemars::schema_for!(T);
        Self::new(
            name,
            description,
            serde_json::to_value(schema).expect("schema serializes"),
        )
    }

    /// Return the tool name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the tool description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Return the tool input schema.
    pub fn input_schema(&self) -> &serde_json::Value {
        &self.input_schema
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

impl ToolCall {
    /// Create a new tool call.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

/// A tool result to feed back to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    /// The ID of the tool call this result corresponds to.
    pub tool_call_id: String,
    /// The result content returned by the tool.
    pub content: String,
}

impl ToolResult {
    /// Create a new tool result.
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }
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
                return Err(Error::Configuration(format!(
                    "{name} must be one of: none, minimal, low, medium, high, xhigh; got: {raw}"
                )));
            }
        };

        Ok(Some(effort))
    }
}

/// Read an optional sampling temperature override from an environment variable.
///
/// Temperatures must be finite, non-negative floats. Provider/model-specific
/// upper bounds are validated by the downstream adapter or API.
pub fn temperature_from_env(name: &str) -> Result<Option<f32>> {
    let Some(raw) = env::var(name).ok() else {
        return Ok(None);
    };

    let value = raw
        .trim()
        .parse::<f32>()
        .map_err(|_| Error::Configuration(format!("{name} must be a float, got: {raw}")))?;
    if !value.is_finite() || value < 0.0 {
        return Err(Error::Configuration(format!(
            "{name} must be a finite, non-negative float, got: {raw}"
        )));
    }

    Ok(Some(value))
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
            Err(message) => Err(Error::Configuration(message.clone())),
        }
    }
}

/// A request to an LLM for completion.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CompletionRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<ReasoningEffort>,
    system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
}

impl CompletionRequest {
    /// Start building a completion request.
    pub fn builder() -> CompletionRequestBuilder {
        CompletionRequestBuilder::default()
    }

    /// Return the requested model override, if any.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Return the conversation transcript.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Return the maximum output token cap.
    pub fn max_tokens(&self) -> Option<usize> {
        self.max_tokens
    }

    /// Return the sampling temperature.
    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Return the requested reasoning effort override.
    pub fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.reasoning_effort
    }

    /// Return the optional system prompt.
    pub fn system(&self) -> Option<&str> {
        self.system.as_deref()
    }

    /// Return the available tool definitions.
    pub fn tools(&self) -> Option<&[ToolDefinition]> {
        self.tools.as_deref()
    }

    /// Return the requested tool-choice policy.
    pub fn tool_choice(&self) -> Option<&ToolChoice> {
        self.tool_choice.as_ref()
    }
}

/// Consuming builder for [`CompletionRequest`].
#[derive(Debug, Clone, Default)]
pub struct CompletionRequestBuilder {
    request: CompletionRequest,
}

impl CompletionRequestBuilder {
    /// Set the provider-specific model override.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.request.model = model.into();
        self
    }

    /// Replace the full conversation transcript.
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.request.messages = messages;
        self
    }

    /// Append one message to the transcript.
    pub fn message(mut self, message: Message) -> Self {
        self.request.messages.push(message);
        self
    }

    /// Set the maximum output token cap.
    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.request.max_tokens = Some(max_tokens);
        self
    }

    /// Set or clear the maximum output token cap.
    pub fn max_tokens_opt(mut self, max_tokens: Option<usize>) -> Self {
        self.request.max_tokens = max_tokens;
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.request.temperature = Some(temperature);
        self
    }

    /// Set the reasoning effort override.
    pub fn reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.request.reasoning_effort = Some(reasoning_effort);
        self
    }

    /// Set or clear the optional reasoning effort override.
    pub fn maybe_reasoning_effort(mut self, reasoning_effort: Option<ReasoningEffort>) -> Self {
        self.request.reasoning_effort = reasoning_effort;
        self
    }

    /// Set or clear the optional reasoning effort override.
    pub fn reasoning_effort_opt(self, reasoning_effort: Option<ReasoningEffort>) -> Self {
        self.maybe_reasoning_effort(reasoning_effort)
    }

    /// Set the optional system prompt.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.request.system = Some(system.into());
        self
    }

    /// Set the available tool definitions.
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.request.tools = Some(tools);
        self
    }

    /// Set the requested tool-choice policy.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.request.tool_choice = Some(tool_choice);
        self
    }

    /// Finish building the request.
    pub fn build(self) -> CompletionRequest {
        self.request
    }
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
        let req = CompletionRequest::builder()
            .model("test-model")
            .message(Message::user("hello"))
            .max_tokens(1024)
            .temperature(0.7)
            .system("You are a helpful assistant.")
            .build();
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
    fn assistant_with_tool_calls_preserves_payload() {
        let message = Message::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "call_123".into(),
                name: "lookup".into(),
                arguments: serde_json::json!({"q": "hello"}),
            }],
        );

        assert_eq!(message.role(), MessageRole::Assistant);
        assert_eq!(message.content(), "");
        assert_eq!(message.tool_calls().len(), 1);
        assert!(message.tool_results().is_empty());
    }

    #[test]
    fn temperature_from_env_parses_non_negative_values() {
        const VAR: &str = "TEST_TEMPERATURE_PARSE";
        unsafe {
            env::set_var(VAR, "0");
        }
        assert_eq!(temperature_from_env(VAR).unwrap(), Some(0.0));
        unsafe {
            env::set_var(VAR, "0.3");
        }
        assert_eq!(temperature_from_env(VAR).unwrap(), Some(0.3));
        unsafe {
            env::remove_var(VAR);
        }
    }

    #[test]
    fn temperature_from_env_rejects_negative_values() {
        const VAR: &str = "TEST_TEMPERATURE_NEGATIVE";
        unsafe {
            env::set_var(VAR, "-0.1");
        }
        let err = temperature_from_env(VAR).unwrap_err();
        assert!(err.to_string().contains("non-negative"));
        unsafe {
            env::remove_var(VAR);
        }
    }
}
