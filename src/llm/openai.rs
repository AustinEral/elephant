//! OpenAI Responses API client.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::llm::{
    CompletionRequest, CompletionResponse, LlmClient, Message, OpenAiConfig,
    OpenAiPromptCacheConfig, OpenAiPromptCacheRetention, PromptCacheUsage, ReasoningEffort,
    ToolCall, ToolChoice, ToolDefinition, ToolResult,
};

const API_URL: &str = "https://api.openai.com/v1";

/// Client for the OpenAI Responses API.
pub struct OpenAiClient {
    client: Client,
    api_key: String,
    default_model: String,
    base_url: String,
    prompt_cache: Option<OpenAiPromptCacheConfig>,
}

impl OpenAiClient {
    /// Create a new OpenAI Responses client with an optional base URL override.
    pub fn new(config: OpenAiConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs()))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;

        Ok(Self {
            client,
            api_key: config.api_key().to_string(),
            default_model: config.model().to_string(),
            base_url: config.base_url().unwrap_or(API_URL).to_string(),
            prompt_cache: config.prompt_cache().cloned(),
        })
    }
}

#[derive(Serialize)]
struct OpenAiResponsesRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    input: Vec<OpenAiResponsesInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenAiResponsesReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAiResponsesToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<OpenAiPromptCacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

impl OpenAiResponsesRequest {
    fn supports_reasoning(model: &str) -> bool {
        model.starts_with("gpt-5")
            || model.starts_with("o1")
            || model.starts_with("o3")
            || model.starts_with("o4")
    }

    fn temperature_for_model(
        model: &str,
        temperature: Option<f32>,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Option<f32> {
        if model.starts_with("gpt-5.1") || model.starts_with("gpt-5.2") {
            return matches!(reasoning_effort, Some(ReasoningEffort::None))
                .then_some(temperature)
                .flatten();
        }

        if model.starts_with("gpt-5") {
            return None;
        }

        temperature
    }

    fn from_completion_request(
        request: &CompletionRequest,
        model: String,
        prompt_cache: Option<&OpenAiPromptCacheConfig>,
    ) -> Self {
        let mut input = Vec::new();
        for message in request.messages() {
            input.extend(OpenAiResponsesInputItem::from_message(message));
        }

        let tools = request
            .tools()
            .map(|defs| defs.iter().cloned().map(OpenAiResponsesTool::from).collect());
        let parallel_tool_calls = tools.as_ref().map(|_| true);
        let supports_reasoning = Self::supports_reasoning(&model);
        let temperature =
            Self::temperature_for_model(&model, request.temperature(), request.reasoning_effort());
        let reasoning = request
            .reasoning_effort()
            .filter(|_| supports_reasoning)
            .map(OpenAiResponsesReasoning::from);

        Self {
            model,
            instructions: request.system().map(str::to_string),
            input,
            max_output_tokens: request.max_tokens(),
            temperature,
            reasoning,
            tools,
            tool_choice: request.tool_choice().cloned().map(OpenAiResponsesToolChoice::from),
            prompt_cache_key: prompt_cache.and_then(|config| config.key().map(str::to_string)),
            prompt_cache_retention: prompt_cache.and_then(|config| config.retention()),
            parallel_tool_calls,
        }
    }
}

#[derive(Serialize)]
struct OpenAiResponsesReasoning {
    effort: ReasoningEffort,
}

impl From<ReasoningEffort> for OpenAiResponsesReasoning {
    fn from(effort: ReasoningEffort) -> Self {
        Self { effort }
    }
}

#[derive(Serialize)]
struct OpenAiResponsesTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    name: String,
    description: String,
    parameters: serde_json::Value,
    strict: bool,
}

impl From<ToolDefinition> for OpenAiResponsesTool {
    fn from(tool: ToolDefinition) -> Self {
        Self {
            tool_type: "function",
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: tool.input_schema().clone(),
            strict: true,
        }
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum OpenAiResponsesToolChoice {
    Mode(String),
    Function {
        #[serde(rename = "type")]
        choice_type: &'static str,
        name: String,
    },
}

impl From<ToolChoice> for OpenAiResponsesToolChoice {
    fn from(choice: ToolChoice) -> Self {
        match choice {
            ToolChoice::Auto => Self::Mode("auto".into()),
            ToolChoice::Required => Self::Mode("required".into()),
            ToolChoice::None => Self::Mode("none".into()),
            ToolChoice::Specific(name) => Self::Function {
                choice_type: "function",
                name,
            },
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum OpenAiResponsesInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: Vec<OpenAiResponsesInputContent>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
}

impl OpenAiResponsesInputItem {
    fn from_message(message: &Message) -> Vec<Self> {
        let mut items = Vec::new();

        if !message.content().is_empty() {
            items.push(Self::Message {
                role: message.role().as_str().to_string(),
                content: vec![OpenAiResponsesInputContent::Text {
                    text: message.content().to_string(),
                }],
            });
        }

        items.extend(
            message
                .tool_calls()
                .iter()
                .map(|tool_call| Self::FunctionCall {
                    call_id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    arguments: tool_call.arguments.to_string(),
                }),
        );

        items.extend(
            message
                .tool_results_ref()
                .iter()
                .cloned()
                .map(Self::from_tool_result),
        );

        items
    }

    fn from_tool_result(tool_result: ToolResult) -> Self {
        Self::FunctionCallOutput {
            call_id: tool_result.tool_call_id,
            output: tool_result.content,
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum OpenAiResponsesInputContent {
    #[serde(rename = "input_text")]
    Text { text: String },
}

#[derive(Deserialize)]
struct OpenAiResponsesResponse {
    #[serde(default)]
    output: Vec<OpenAiResponsesOutputItem>,
    #[serde(default)]
    usage: Option<OpenAiResponsesUsage>,
    #[serde(default)]
    incomplete_details: Option<OpenAiResponsesIncompleteDetails>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum OpenAiResponsesOutputItem {
    #[serde(rename = "message")]
    Message {
        #[serde(default)]
        content: Vec<OpenAiResponsesOutputMessageContent>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum OpenAiResponsesOutputMessageContent {
    #[serde(rename = "output_text")]
    Text { text: String },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct OpenAiResponsesUsage {
    input_tokens: usize,
    output_tokens: usize,
    #[serde(default)]
    input_tokens_details: Option<OpenAiResponsesInputTokenDetails>,
}

#[derive(Deserialize)]
struct OpenAiResponsesInputTokenDetails {
    #[serde(default)]
    cached_tokens: usize,
}

impl From<OpenAiResponsesInputTokenDetails> for PromptCacheUsage {
    fn from(details: OpenAiResponsesInputTokenDetails) -> Self {
        Self {
            cached_tokens: Some(details.cached_tokens),
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        }
    }
}

#[derive(Deserialize)]
struct OpenAiResponsesIncompleteDetails {
    reason: String,
}

impl From<OpenAiResponsesResponse> for CompletionResponse {
    fn from(response: OpenAiResponsesResponse) -> Self {
        let mut content_parts = Vec::new();
        let mut refusal_parts = Vec::new();
        let mut tool_calls = Vec::new();

        for item in response.output {
            match item {
                OpenAiResponsesOutputItem::Message { content } => {
                    for part in content {
                        match part {
                            OpenAiResponsesOutputMessageContent::Text { text }
                                if !text.is_empty() =>
                            {
                                content_parts.push(text);
                            }
                            OpenAiResponsesOutputMessageContent::Refusal { refusal }
                                if !refusal.is_empty() =>
                            {
                                refusal_parts.push(refusal);
                            }
                            OpenAiResponsesOutputMessageContent::Text { .. }
                            | OpenAiResponsesOutputMessageContent::Refusal { .. }
                            | OpenAiResponsesOutputMessageContent::Other => {}
                        }
                    }
                }
                OpenAiResponsesOutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                } => {
                    if let Ok(arguments) = serde_json::from_str(&arguments) {
                        tool_calls.push(ToolCall {
                            id: call_id,
                            name,
                            arguments,
                        });
                    }
                }
                OpenAiResponsesOutputItem::Other => {}
            }
        }

        let content = if content_parts.is_empty() {
            refusal_parts.join("")
        } else {
            content_parts.join("")
        };

        let stop_reason = if refusal_parts.is_empty() {
            response.incomplete_details.map(|details| details.reason)
        } else {
            Some("refusal".into())
        };

        let (input_tokens, output_tokens, prompt_cache) = response
            .usage
            .map(|usage| {
                let prompt_cache = usage.input_tokens_details.map(PromptCacheUsage::from);
                (usage.input_tokens, usage.output_tokens, prompt_cache)
            })
            .unwrap_or((0, 0, None));

        Self {
            content,
            input_tokens,
            output_tokens,
            stop_reason,
            tool_calls,
            prompt_cache,
        }
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = super::resolve_model(request.model(), &self.default_model);
        let requested_max_tokens = request.max_tokens();
        let requested_temperature = request.temperature();
        let requested_reasoning_effort = request.reasoning_effort();

        let body = OpenAiResponsesRequest::from_completion_request(
            &request,
            model.clone(),
            self.prompt_cache.as_ref(),
        );
        let url = format!("{}/responses", self.base_url);

        let resp_text = super::send_and_check(
            "OpenAI",
            self.client
                .post(&url)
                .bearer_auth(&self.api_key)
                .json(&body),
        )
        .await?;

        let parsed: OpenAiResponsesResponse = serde_json::from_str(&resp_text)
            .map_err(|e| Error::Llm(format!("failed to parse OpenAI response: {e}")))?;

        let response = CompletionResponse::from(parsed);
        if response.content.is_empty()
            && response.tool_calls.is_empty()
            && response.stop_reason.as_deref() != Some("refusal")
        {
            tracing::warn!(
                model = %model,
                stop_reason = ?response.stop_reason,
                input_tokens = response.input_tokens,
                output_tokens = response.output_tokens,
                requested_max_tokens = ?requested_max_tokens,
                requested_temperature = ?requested_temperature,
                requested_reasoning_effort = ?requested_reasoning_effort,
                "openai returned empty visible response body"
            );
        }

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ToolDefinition, ToolResult};
    use serde_json::json;

    #[test]
    fn serializes_responses_request_from_completion_request() {
        let request = CompletionRequest::builder()
            .message(Message::user("hello"))
            .message(Message::assistant_with_tool_calls(
                String::new(),
                vec![ToolCall::new("call_123", "lookup", json!({"query": "hi"}))],
            ))
            .message(Message::with_tool_results(vec![ToolResult::new(
                "call_123",
                "tool result",
            )]))
            .message(Message::tool_result("call_456", "legacy tool result"))
            .max_tokens(128)
            .temperature(0.2)
            .reasoning_effort(ReasoningEffort::None)
            .system("be brief")
            .tools(vec![ToolDefinition::new(
                "lookup",
                "Look something up",
                json!({"type": "object"}),
            )])
            .tool_choice(ToolChoice::Specific("lookup".into()))
            .build();

        let body = OpenAiResponsesRequest::from_completion_request(
            &request,
            "gpt-5.1".into(),
            Some(
                &OpenAiPromptCacheConfig::new()
                    .with_key("elephant:reflect")
                    .with_retention(OpenAiPromptCacheRetention::InMemory),
            ),
        );

        let value = serde_json::to_value(&body).unwrap();
        assert_eq!(value["instructions"], "be brief");
        assert_eq!(value["max_output_tokens"], 128);
        assert!((value["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-6);
        assert_eq!(value["reasoning"]["effort"], "none");
        assert_eq!(value["prompt_cache_key"], "elephant:reflect");
        assert_eq!(value["prompt_cache_retention"], "in-memory");
        assert_eq!(value["parallel_tool_calls"], true);
        assert_eq!(value["tool_choice"]["type"], "function");
        assert_eq!(value["tool_choice"]["name"], "lookup");
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][0]["strict"], true);
        assert_eq!(value["input"][0]["type"], "message");
        assert_eq!(value["input"][0]["role"], "user");
        assert_eq!(value["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(value["input"][1]["type"], "function_call");
        assert_eq!(value["input"][1]["call_id"], "call_123");
        assert_eq!(value["input"][2]["type"], "function_call_output");
        assert_eq!(value["input"][2]["call_id"], "call_123");
        assert_eq!(value["input"][3]["type"], "function_call_output");
        assert_eq!(value["input"][3]["call_id"], "call_456");
    }

    #[test]
    fn omits_temperature_for_gpt5_reasoning_requests() {
        let request = CompletionRequest::builder()
            .message(Message::user("hello"))
            .max_tokens(64)
            .temperature(0.2)
            .reasoning_effort(ReasoningEffort::Low)
            .build();

        let body =
            OpenAiResponsesRequest::from_completion_request(&request, "gpt-5".into(), None);
        let value = serde_json::to_value(&body).unwrap();
        assert!(value.get("temperature").is_none());
        assert_eq!(value["reasoning"]["effort"], "low");
    }

    #[test]
    fn parses_response_output_items_and_cached_tokens() {
        let parsed: OpenAiResponsesResponse = serde_json::from_value(json!({
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{\"query\":\"hi\"}"
                },
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "done"}
                    ]
                }
            ],
            "usage": {
                "input_tokens": 1200,
                "output_tokens": 42,
                "input_tokens_details": {
                    "cached_tokens": 1024
                }
            }
        }))
        .unwrap();

        let response = CompletionResponse::from(parsed);
        assert_eq!(response.content, "done");
        assert_eq!(response.input_tokens, 1200);
        assert_eq!(response.output_tokens, 42);
        assert_eq!(response.stop_reason, None);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "call_1");
        assert_eq!(response.tool_calls[0].name, "lookup");
        assert_eq!(response.tool_calls[0].arguments, json!({"query": "hi"}));
        assert_eq!(
            response.prompt_cache,
            Some(PromptCacheUsage {
                cached_tokens: Some(1024),
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            })
        );
    }

    #[test]
    fn parses_refusal_as_stop_reason() {
        let parsed: OpenAiResponsesResponse = serde_json::from_value(json!({
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "refusal", "refusal": "I can't help with that."}
                    ]
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4
            }
        }))
        .unwrap();

        let response = CompletionResponse::from(parsed);
        assert_eq!(response.stop_reason.as_deref(), Some("refusal"));
        assert_eq!(response.content, "I can't help with that.");
    }

    #[test]
    fn surfaces_incomplete_reason() {
        let parsed: OpenAiResponsesResponse = serde_json::from_value(json!({
            "output": [],
            "incomplete_details": {
                "reason": "max_output_tokens"
            }
        }))
        .unwrap();

        let response = CompletionResponse::from(parsed);
        assert_eq!(response.stop_reason.as_deref(), Some("max_output_tokens"));
    }
}
