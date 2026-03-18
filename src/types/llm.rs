//! LLM request and response types.

use serde::{Deserialize, Serialize};

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
    /// Normalized prompt/completion usage, including cache-aware prompt details.
    #[serde(default)]
    pub usage: CompletionUsage,
    /// Provider-specific finish or stop reason, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Tool calls requested by the LLM.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

/// Explicit cache activity classification for a completion response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CacheStatus {
    /// The provider path does not expose cache-aware usage details.
    Unsupported,
    /// The provider exposed cache-aware usage details but there was no activity.
    NoActivity,
    /// The request wrote cacheable prompt content without any cache hits.
    WriteOnly,
    /// The request hit the cache without writing new cacheable prompt content.
    Hit,
    /// The request both hit and wrote cacheable prompt content.
    HitAndWrite,
}

impl Default for CacheStatus {
    fn default() -> Self {
        Self::Unsupported
    }
}

/// Provider-agnostic cache-aware completion usage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompletionUsage {
    /// Total normalized prompt usage, including uncached, cache-hit, and cache-write tokens.
    pub prompt_tokens: usize,
    /// Prompt tokens that were not served from cache.
    pub uncached_prompt_tokens: usize,
    /// Prompt tokens served from a cache hit.
    pub cache_hit_prompt_tokens: usize,
    /// Prompt tokens newly written to the provider cache.
    pub cache_write_prompt_tokens: usize,
    /// Completion tokens generated by the provider.
    pub completion_tokens: usize,
    /// Explicit cache activity status for this response.
    pub cache_status: CacheStatus,
}

impl CompletionUsage {
    /// Build usage for provider paths that do not expose cache-aware details.
    pub(crate) fn unsupported(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            uncached_prompt_tokens: prompt_tokens,
            cache_hit_prompt_tokens: 0,
            cache_write_prompt_tokens: 0,
            completion_tokens,
            cache_status: CacheStatus::Unsupported,
        }
    }
}

impl Default for CompletionUsage {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            uncached_prompt_tokens: 0,
            cache_hit_prompt_tokens: 0,
            cache_write_prompt_tokens: 0,
            completion_tokens: 0,
            cache_status: CacheStatus::Unsupported,
        }
    }
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
            usage: CompletionUsage {
                prompt_tokens: 10,
                uncached_prompt_tokens: 4,
                cache_hit_prompt_tokens: 3,
                cache_write_prompt_tokens: 3,
                completion_tokens: 6,
                cache_status: CacheStatus::HitAndWrite,
            },
            stop_reason: None,
            tool_calls: vec![],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: CompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, back);
    }

    #[test]
    fn completion_response_legacy_json_defaults_usage() {
        let json = serde_json::json!({
            "content": "Hello! How can I help?",
            "input_tokens": 10,
            "output_tokens": 6,
            "stop_reason": null,
            "tool_calls": [],
        });

        let back: CompletionResponse = serde_json::from_value(json).unwrap();
        assert_eq!(back.usage, CompletionUsage::default());
        assert_eq!(back.usage.cache_status, CacheStatus::Unsupported);
    }
}
