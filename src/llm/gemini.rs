//! Google Gemini `generateContent` API client.

use std::collections::HashMap;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::error::{Error, Result};
use crate::llm::{
    CompletionRequest, CompletionResponse, GeminiConfig, LlmClient, Message, MessageRole,
    PromptCacheUsage, ReasoningEffort, ToolCall, ToolChoice, ToolDefinition, ToolResult,
};

const API_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const THOUGHT_SIGNATURE_ID_PREFIX: &str = "gemini-thought::";

/// Client for the Gemini `generateContent` API.
pub struct GeminiClient {
    client: Client,
    api_key: String,
    default_model: String,
    base_url: String,
    supports_function_call_ids: bool,
}

impl GeminiClient {
    /// Create a new Gemini client with an optional base URL override.
    pub fn new(config: GeminiConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs()))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;

        Ok(Self {
            client,
            api_key: config.api_key().to_string(),
            default_model: config.model().to_string(),
            base_url: config
                .base_url()
                .unwrap_or(API_URL)
                .trim_end_matches('/')
                .into(),
            supports_function_call_ids: config.supports_function_call_ids(),
        })
    }

    fn build_url(&self, model: &str) -> String {
        let model_path = if model.starts_with("models/") {
            model.to_string()
        } else {
            format!("models/{model}")
        };
        format!("{}/{model_path}:generateContent", self.base_url)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerateContentRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<GeminiToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

impl GeminiGenerateContentRequest {
    fn from_completion_request(
        request: &CompletionRequest,
        model: &str,
        supports_function_call_ids: bool,
    ) -> Self {
        let tool_names = tool_name_lookup(request.messages());
        let contents = request
            .messages()
            .iter()
            .filter_map(|message| {
                GeminiContent::from_message(message, &tool_names, supports_function_call_ids)
            })
            .collect::<Vec<_>>();

        let tools = request.tools().map(|defs| {
            vec![GeminiTool {
                function_declarations: defs
                    .iter()
                    .cloned()
                    .map(GeminiFunctionDeclaration::from)
                    .collect(),
            }]
        });

        let tool_config = request
            .tool_choice()
            .filter(|_| tools.is_some())
            .map(GeminiToolConfig::from);

        let generation_config = GeminiGenerationConfig::from_request(model, request);

        Self {
            contents,
            tools,
            tool_config,
            system_instruction: request
                .system()
                .map(|system| GeminiContent::system(system.to_string())),
            generation_config,
        }
    }
}

fn tool_name_lookup(messages: &[Message]) -> HashMap<String, String> {
    let mut tool_names = HashMap::new();
    for message in messages {
        for tool_call in message.tool_calls() {
            tool_names.insert(tool_call.id.clone(), tool_call.name.clone());
        }
    }
    tool_names
}

#[derive(Serialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiRequestPart>,
}

impl GeminiContent {
    fn system(text: String) -> Self {
        Self {
            role: None,
            parts: vec![GeminiRequestPart::Text(GeminiTextPart { text })],
        }
    }

    fn from_message(
        message: &Message,
        tool_names: &HashMap<String, String>,
        supports_function_call_ids: bool,
    ) -> Option<Self> {
        let mut parts = Vec::new();

        if !message.content().is_empty() {
            parts.push(GeminiRequestPart::Text(GeminiTextPart {
                text: message.content().to_string(),
            }));
        }

        for tool_call in message.tool_calls() {
            let (tool_call_id, thought_signature) = split_gemini_tool_call_id(&tool_call.id);
            parts.push(GeminiRequestPart::FunctionCall {
                function_call: GeminiRequestFunctionCall {
                    id: supports_function_call_ids
                        .then_some(tool_call_id)
                        .filter(|id| !id.is_empty()),
                    name: tool_call.name.clone(),
                    args: tool_call.arguments.clone(),
                },
                thought_signature,
            });
        }

        for tool_result in message.tool_results_ref() {
            let (tool_call_id, _) = split_gemini_tool_call_id(&tool_result.tool_call_id);
            parts.push(GeminiRequestPart::FunctionResponse {
                function_response: GeminiRequestFunctionResponse {
                    id: supports_function_call_ids
                        .then_some(tool_call_id.clone())
                        .filter(|id| !id.is_empty()),
                    name: tool_names
                        .get(&tool_result.tool_call_id)
                        .cloned()
                        .unwrap_or_else(|| "tool".into()),
                    response: json!({
                        "result": tool_result_response_value(tool_result),
                    }),
                },
            });
        }

        if parts.is_empty() {
            return None;
        }

        Some(Self {
            role: Some(gemini_role(message.role()).into()),
            parts,
        })
    }
}

fn gemini_role(role: MessageRole) -> &'static str {
    match role {
        MessageRole::User => "user",
        MessageRole::Assistant => "model",
    }
}

fn tool_result_response_value(tool_result: &ToolResult) -> Value {
    serde_json::from_str(&tool_result.content)
        .unwrap_or_else(|_| Value::String(tool_result.content.clone()))
}

#[derive(Serialize)]
#[serde(untagged)]
enum GeminiRequestPart {
    Text(GeminiTextPart),
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiRequestFunctionCall,
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiRequestFunctionResponse,
    },
}

#[derive(Serialize)]
struct GeminiTextPart {
    text: String,
}

#[derive(Serialize)]
struct GeminiRequestFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    name: String,
    args: Value,
}

#[derive(Serialize)]
struct GeminiRequestFunctionResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    name: String,
    response: Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
    #[serde(
        rename = "parametersJsonSchema",
        skip_serializing_if = "Option::is_none"
    )]
    parameters_json_schema: Option<Value>,
}

impl From<ToolDefinition> for GeminiFunctionDeclaration {
    fn from(tool: ToolDefinition) -> Self {
        Self {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: None,
            parameters_json_schema: Some(normalize_gemini_parameters_json_schema(
                tool.input_schema(),
            )),
        }
    }
}

fn normalize_gemini_parameters_json_schema(schema: &Value) -> Value {
    let mut normalized = schema.clone();
    normalize_gemini_parameters_json_schema_node(&mut normalized);
    normalized
}

fn normalize_gemini_parameters_json_schema_node(node: &mut Value) {
    match node {
        Value::Object(map) => {
            normalize_gemini_type_array(map);
            normalize_gemini_union_key(map, "oneOf");

            // Gemini accepts JSON Schema via `parametersJsonSchema`, but still rejects
            // draft metadata like `$schema` at the wire format layer.
            map.remove("$schema");
            map.remove("$id");
            map.remove("$comment");

            for key in ["properties", "definitions", "$defs"] {
                if let Some(Value::Object(children)) = map.get_mut(key) {
                    for child in children.values_mut() {
                        normalize_gemini_parameters_json_schema_node(child);
                    }
                }
            }

            for key in ["anyOf", "allOf"] {
                if let Some(Value::Array(items)) = map.get_mut(key) {
                    for item in items {
                        normalize_gemini_parameters_json_schema_node(item);
                    }
                }
            }

            if let Some(items) = map.get_mut("items") {
                normalize_gemini_parameters_json_schema_node(items);
            }
        }
        Value::Array(items) => {
            for item in items {
                normalize_gemini_parameters_json_schema_node(item);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn normalize_gemini_type_array(map: &mut Map<String, Value>) {
    let Some(type_value) = map.remove("type") else {
        return;
    };
    let Value::Array(types) = type_value else {
        map.insert("type".into(), type_value);
        return;
    };

    let variants = types
        .into_iter()
        .filter_map(|kind| match kind {
            Value::String(name) => Some(json!({ "type": name })),
            _ => None,
        })
        .collect::<Vec<_>>();

    match variants.as_slice() {
        [] => {}
        [single] => {
            if let Some(kind) = single.get("type").and_then(Value::as_str) {
                map.insert("type".into(), Value::String(kind.to_string()));
            }
        }
        _ => {
            map.insert("anyOf".into(), Value::Array(variants));
        }
    }
}

fn normalize_gemini_union_key(map: &mut Map<String, Value>, key: &str) {
    let Some(value) = map.remove(key) else {
        return;
    };

    if map.contains_key("anyOf") {
        return;
    }

    map.insert("anyOf".into(), value);
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiToolConfig {
    function_calling_config: GeminiFunctionCallingConfig,
}

impl From<&ToolChoice> for GeminiToolConfig {
    fn from(choice: &ToolChoice) -> Self {
        let function_calling_config = match choice {
            ToolChoice::Auto => GeminiFunctionCallingConfig {
                mode: "AUTO",
                allowed_function_names: None,
            },
            ToolChoice::Required => GeminiFunctionCallingConfig {
                mode: "ANY",
                allowed_function_names: None,
            },
            ToolChoice::None => GeminiFunctionCallingConfig {
                mode: "NONE",
                allowed_function_names: None,
            },
            ToolChoice::Specific(name) => GeminiFunctionCallingConfig {
                mode: "ANY",
                allowed_function_names: Some(vec![name.clone()]),
            },
        };

        Self {
            function_calling_config,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFunctionCallingConfig {
    mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<String>>,
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

impl GeminiGenerationConfig {
    fn from_request(model: &str, request: &CompletionRequest) -> Option<Self> {
        let config = Self {
            temperature: request.temperature(),
            max_output_tokens: request.max_tokens(),
            thinking_config: request
                .reasoning_effort()
                .and_then(|effort| GeminiThinkingConfig::for_model(model, effort)),
        };

        if config.temperature.is_none()
            && config.max_output_tokens.is_none()
            && config.thinking_config.is_none()
        {
            None
        } else {
            Some(config)
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_level: Option<&'static str>,
}

impl GeminiThinkingConfig {
    fn for_model(model: &str, effort: ReasoningEffort) -> Option<Self> {
        let model = model.trim().to_ascii_lowercase();

        if model.starts_with("gemini-3") {
            let thinking_level = match effort {
                ReasoningEffort::None | ReasoningEffort::Minimal => "minimal",
                ReasoningEffort::Low => "low",
                ReasoningEffort::Medium => "medium",
                ReasoningEffort::High | ReasoningEffort::XHigh => "high",
            };

            return Some(Self {
                thinking_budget: None,
                thinking_level: Some(thinking_level),
            });
        }

        if model.starts_with("gemini-2.5") {
            let thinking_budget = match effort {
                ReasoningEffort::None => {
                    if model.contains("pro") {
                        128
                    } else {
                        0
                    }
                }
                ReasoningEffort::Minimal | ReasoningEffort::Low => 1024,
                ReasoningEffort::Medium => 8192,
                ReasoningEffort::High => 24_576,
                ReasoningEffort::XHigh => {
                    if model.contains("pro") {
                        32_768
                    } else {
                        24_576
                    }
                }
            };

            return Some(Self {
                thinking_budget: Some(thinking_budget),
                thinking_level: None,
            });
        }

        None
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerateContentResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    prompt_feedback: Option<GeminiPromptFeedback>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiResponseContent>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GeminiResponseContent {
    #[allow(dead_code)]
    role: Option<String>,
    #[serde(default)]
    parts: Vec<GeminiResponsePart>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponsePart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thought: Option<bool>,
    #[serde(default, rename = "thoughtSignature", alias = "thought_signature")]
    thought_signature: Option<String>,
    #[serde(default)]
    function_call: Option<GeminiResponseFunctionCall>,
}

#[derive(Deserialize)]
struct GeminiResponseFunctionCall {
    #[serde(default)]
    id: Option<String>,
    name: String,
    #[serde(default)]
    args: Value,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPromptFeedback {
    #[serde(default)]
    block_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    #[serde(default)]
    prompt_token_count: usize,
    #[serde(default)]
    candidates_token_count: usize,
    #[serde(default)]
    cached_content_token_count: Option<usize>,
}

impl GeminiGenerateContentResponse {
    fn into_completion_response(self) -> Result<CompletionResponse> {
        if self
            .prompt_feedback
            .as_ref()
            .and_then(|feedback| feedback.block_reason.as_deref())
            .is_some()
        {
            return Err(Error::LlmRefusal);
        }

        let usage = self.usage_metadata.unwrap_or_default();
        let Some(candidate) = self.candidates.into_iter().next() else {
            return Err(Error::Llm("Gemini returned no candidates".into()));
        };

        let mut content = Vec::new();
        let mut tool_calls = Vec::new();

        if let Some(candidate_content) = candidate.content {
            for (index, part) in candidate_content.parts.into_iter().enumerate() {
                if part.thought == Some(true) {
                    continue;
                }

                if let Some(text) = part.text
                    && !text.is_empty()
                {
                    content.push(text);
                }

                if let Some(function_call) = part.function_call {
                    tool_calls.push(ToolCall {
                        id: encode_gemini_tool_call_id(
                            &function_call
                                .id
                                .unwrap_or_else(|| format!("gemini-call-{}", index + 1)),
                            part.thought_signature.as_deref(),
                        ),
                        name: function_call.name,
                        arguments: function_call.args,
                    });
                }
            }
        }

        let stop_reason = candidate
            .finish_reason
            .as_deref()
            .map(normalize_finish_reason);

        if stop_reason.as_deref() == Some("refusal") && content.is_empty() && tool_calls.is_empty()
        {
            return Err(Error::LlmRefusal);
        }

        Ok(CompletionResponse {
            content: content.join("\n"),
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count,
            stop_reason,
            tool_calls,
            prompt_cache: usage
                .cached_content_token_count
                .map(|cached_tokens| PromptCacheUsage {
                    cached_tokens: Some(cached_tokens),
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                }),
        })
    }
}

fn encode_gemini_tool_call_id(id: &str, thought_signature: Option<&str>) -> String {
    match thought_signature {
        Some(signature) if !signature.is_empty() => {
            format!("{THOUGHT_SIGNATURE_ID_PREFIX}{signature}::{id}")
        }
        _ => id.to_string(),
    }
}

fn split_gemini_tool_call_id(encoded: &str) -> (String, Option<String>) {
    let Some(payload) = encoded.strip_prefix(THOUGHT_SIGNATURE_ID_PREFIX) else {
        return (encoded.to_string(), None);
    };

    let Some((signature, id)) = payload.rsplit_once("::") else {
        return (encoded.to_string(), None);
    };

    if signature.is_empty() || id.is_empty() {
        return (encoded.to_string(), None);
    }

    (id.to_string(), Some(signature.to_string()))
}

fn normalize_finish_reason(reason: &str) -> String {
    match reason.trim().to_ascii_uppercase().as_str() {
        "STOP" => "stop".into(),
        "MAX_TOKENS" => "max_tokens".into(),
        "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "MODEL_ARMOR" => "refusal".into(),
        other => other.to_ascii_lowercase(),
    }
}

#[async_trait]
impl LlmClient for GeminiClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = super::resolve_model(request.model(), &self.default_model);
        let body = GeminiGenerateContentRequest::from_completion_request(
            &request,
            &model,
            self.supports_function_call_ids,
        );
        let url = self.build_url(&model);

        let resp_text = super::send_and_check(
            "Gemini",
            self.client
                .post(url)
                .header("x-goog-api-key", &self.api_key)
                .json(&body),
        )
        .await?;

        let parsed: GeminiGenerateContentResponse = serde_json::from_str(&resp_text)
            .map_err(|e| Error::Llm(format!("failed to parse Gemini response: {e}")))?;

        parsed.into_completion_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ToolDefinition, ToolResult};
    use schemars::JsonSchema;
    use serde::Deserialize;
    use serde_json::json;

    #[test]
    fn serializes_request_with_tools_and_tool_results() {
        let request = CompletionRequest::builder()
            .system("be brief")
            .message(Message::user("hello"))
            .message(Message::assistant_with_tool_calls(
                String::new(),
                vec![ToolCall::new(
                    encode_gemini_tool_call_id("call_123", Some("sig_123")),
                    "lookup",
                    json!({"query": "hi"}),
                )],
            ))
            .message(Message::with_tool_results(vec![ToolResult::new(
                encode_gemini_tool_call_id("call_123", Some("sig_123")),
                "{\"hits\": 3}",
            )]))
            .temperature(0.2)
            .reasoning_effort(ReasoningEffort::High)
            .max_tokens(128)
            .tools(vec![ToolDefinition::new(
                "lookup",
                "Look something up",
                json!({"type": "object", "properties": {"query": {"type": "string"}}}),
            )])
            .tool_choice(ToolChoice::Specific("lookup".into()))
            .build();

        let body = GeminiGenerateContentRequest::from_completion_request(
            &request,
            "gemini-2.5-flash",
            true,
        );
        let value = serde_json::to_value(&body).unwrap();

        assert_eq!(value["systemInstruction"]["parts"][0]["text"], "be brief");
        assert!(
            (value["generationConfig"]["temperature"]
                .as_f64()
                .expect("temperature should be numeric")
                - 0.2)
                .abs()
                < 1e-6
        );
        assert_eq!(value["generationConfig"]["maxOutputTokens"], 128);
        assert_eq!(
            value["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            24576
        );
        assert_eq!(value["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
        assert_eq!(
            value["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
            "lookup"
        );
        assert_eq!(
            value["contents"][1]["parts"][0]["functionCall"]["name"],
            "lookup"
        );
        assert_eq!(
            value["contents"][1]["parts"][0]["thoughtSignature"],
            "sig_123"
        );
        assert_eq!(
            value["contents"][2]["parts"][0]["functionResponse"]["name"],
            "lookup"
        );
        assert_eq!(
            value["contents"][2]["parts"][0]["functionResponse"]["id"],
            "call_123"
        );
        assert_eq!(
            value["contents"][2]["parts"][0]["functionResponse"]["response"]["result"]["hits"],
            3
        );
    }

    #[test]
    fn parses_response_with_text_tool_calls_and_cache_usage() {
        let parsed: GeminiGenerateContentResponse = serde_json::from_value(json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "done"},
                        {
                            "thoughtSignature": "sig_1",
                            "functionCall": {
                                "id": "call_1",
                                "name": "lookup",
                                "args": { "q": "hi" }
                            }
                        }
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 7,
                "cachedContentTokenCount": 5
            }
        }))
        .unwrap();

        let response = parsed.into_completion_response().unwrap();
        assert_eq!(response.content, "done");
        assert_eq!(response.input_tokens, 12);
        assert_eq!(response.output_tokens, 7);
        assert_eq!(response.stop_reason.as_deref(), Some("stop"));
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(
            split_gemini_tool_call_id(&response.tool_calls[0].id),
            ("call_1".into(), Some("sig_1".into()))
        );
        assert_eq!(
            response.prompt_cache,
            Some(PromptCacheUsage {
                cached_tokens: Some(5),
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            })
        );
    }

    #[test]
    fn prompt_block_maps_to_refusal() {
        let parsed: GeminiGenerateContentResponse = serde_json::from_value(json!({
            "promptFeedback": {
                "blockReason": "SAFETY"
            }
        }))
        .unwrap();

        let err = parsed.into_completion_response().unwrap_err();
        assert!(matches!(err, Error::LlmRefusal));
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct OptionalToolArgs {
        query: String,
        #[serde(default)]
        limit: Option<usize>,
    }

    #[test]
    fn serializes_tool_schema_as_parameters_json_schema() {
        let request = CompletionRequest::builder()
            .message(Message::user("hello"))
            .tools(vec![ToolDefinition::from_schema::<OptionalToolArgs>(
                "lookup",
                "Look something up",
            )])
            .build();

        let body = GeminiGenerateContentRequest::from_completion_request(
            &request,
            "gemini-3.1-pro-preview",
            true,
        );
        let value = serde_json::to_value(&body).unwrap();
        let tool = &value["tools"][0]["functionDeclarations"][0];

        assert!(tool.get("parameters").is_none());
        assert!(tool.get("parametersJsonSchema").is_some());
        assert!(tool["parametersJsonSchema"].get("$schema").is_none());
        assert_eq!(tool["parametersJsonSchema"]["additionalProperties"], false);
        assert_eq!(
            tool["parametersJsonSchema"]["properties"]["query"]["type"],
            "string"
        );
        assert!(
            tool["parametersJsonSchema"]["properties"]["limit"]
                .get("anyOf")
                .is_some()
        );
    }

    #[test]
    fn omits_function_call_ids_when_transport_disallows_them() {
        let request = CompletionRequest::builder()
            .message(Message::user("What happened?"))
            .message(Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    "call_123",
                    "lookup",
                    json!({"query": "october"}),
                )],
            ))
            .message(Message::tool_result(
                "call_123",
                serde_json::to_string(&json!({"hits": 3})).unwrap(),
            ))
            .build();

        let body = GeminiGenerateContentRequest::from_completion_request(
            &request,
            "gemini-3.1-pro-preview",
            false,
        );
        let value = serde_json::to_value(&body).unwrap();

        assert!(
            value["contents"][1]["parts"][0]["functionCall"]
                .get("id")
                .is_none()
        );
        assert!(
            value["contents"][2]["parts"][0]["functionResponse"]
                .get("id")
                .is_none()
        );
    }
}
