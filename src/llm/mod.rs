//! LLM client abstraction and provider implementations.

pub mod anthropic;
pub mod mock;
pub mod openai;
pub mod retry;

use async_trait::async_trait;
use serde::de::DeserializeOwned;

use crate::error::{Error, Result};
use crate::types::llm::{CompletionRequest, CompletionResponse};

/// Trait abstraction over LLM providers.
///
/// Every component that calls an LLM takes `dyn LlmClient`, making the system
/// testable with [`mock::MockLlmClient`] and swappable between providers.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Send a completion request and get a text response.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;
}

/// Send a completion request and parse the response as structured JSON.
///
/// Calls [`LlmClient::complete`], extracts JSON from the response text
/// (handling markdown fences and surrounding prose), then deserializes with serde.
pub async fn complete_structured<T: DeserializeOwned>(
    client: &dyn LlmClient,
    request: CompletionRequest,
) -> Result<T> {
    let response = client.complete(request).await?;
    // Fast path: try parsing the whole response directly
    if let Ok(value) = serde_json::from_str::<T>(&response.content) {
        return Ok(value);
    }
    // Slow path: extract JSON from surrounding text
    let json_str = extract_json(&response.content)?;
    serde_json::from_str::<T>(&json_str).map_err(|e| Error::Llm(format!("JSON parse error: {e}")))
}

/// Extract a JSON value from text that may contain markdown fences or surrounding prose.
///
/// Strategy:
/// 1. Strip markdown fences (```json ... ```)
/// 2. Find first `{` or `[` and match to last `}` or `]`
/// 3. Return that substring
pub fn extract_json(text: &str) -> Result<String> {
    // Strip markdown fences
    let stripped = if let Some(start) = text.find("```") {
        let after_fence = &text[start + 3..];
        // Skip optional language tag on the same line
        let content_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
        let content = &after_fence[content_start..];
        if let Some(end) = content.find("```") {
            content[..end].trim()
        } else {
            content.trim()
        }
    } else {
        text.trim()
    };

    // Try parsing stripped content directly
    if serde_json::from_str::<serde_json::Value>(stripped).is_ok() {
        return Ok(stripped.to_string());
    }

    // Find first { or [ and match to last } or ]
    let obj_start = stripped.find('{');
    let arr_start = stripped.find('[');

    let start = match (obj_start, arr_start) {
        (Some(o), Some(a)) => o.min(a),
        (Some(o), None) => o,
        (None, Some(a)) => a,
        (None, None) => {
            return Err(Error::Llm("no JSON found in response".into()));
        }
    };

    let (open, close) = if stripped.as_bytes()[start] == b'{' {
        (b'{', b'}')
    } else {
        (b'[', b']')
    };

    // Walk forward counting braces to find the matching close
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;
    let mut end_pos = None;

    for (i, &b) in stripped.as_bytes()[start..].iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape_next = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                end_pos = Some(start + i + 1);
                break;
            }
        }
    }

    let end = end_pos.ok_or_else(|| Error::Llm("unbalanced JSON in response".into()))?;
    let candidate = &stripped[start..end];

    // Validate it actually parses
    serde_json::from_str::<serde_json::Value>(candidate)
        .map_err(|e| Error::Llm(format!("extracted JSON is invalid: {e}")))?;

    Ok(candidate.to_string())
}

/// LLM provider selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provider {
    /// Anthropic Claude API.
    Anthropic,
    /// OpenAI API.
    OpenAi,
}

/// Configuration for a single LLM provider.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Which provider to use.
    pub provider: Provider,
    /// API key for authentication.
    pub api_key: String,
    /// Model name/ID to use.
    pub model: String,
    /// Optional base URL override for OpenAI-compatible providers.
    pub base_url: Option<String>,
}

/// Configuration for LLM usage across the system.
///
/// Supports per-operation model selection: a strong model for retain/extraction,
/// a cheaper model for reflect, etc.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Default provider config used for all operations.
    pub default: ProviderConfig,
    /// Optional override for retain/extraction operations.
    pub retain_override: Option<ProviderConfig>,
    /// Optional override for reflect operations.
    pub reflect_override: Option<ProviderConfig>,
}

/// Build an LLM client from a provider configuration.
pub fn build_client(config: &ProviderConfig) -> Box<dyn LlmClient> {
    match config.provider {
        Provider::Anthropic => Box::new(anthropic::AnthropicClient::new(
            config.api_key.clone(),
            config.model.clone(),
        )),
        Provider::OpenAi => Box::new(openai::OpenAiClient::new(
            config.api_key.clone(),
            config.model.clone(),
            config.base_url.clone(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_clean() {
        let input = r#"{"name": "alice", "age": 30}"#;
        let result = extract_json(input).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn extract_json_markdown_fences() {
        let input = "Here is the result:\n```json\n{\"name\": \"bob\"}\n```\nDone.";
        let result = extract_json(input).unwrap();
        assert_eq!(result, r#"{"name": "bob"}"#);
    }

    #[test]
    fn extract_json_surrounding_prose() {
        let input = "Sure! Here is the JSON:\n{\"key\": \"value\"}\nHope that helps!";
        let result = extract_json(input).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }

    #[test]
    fn extract_json_array() {
        let input = "Results: [{\"a\": 1}, {\"a\": 2}] end";
        let result = extract_json(input).unwrap();
        assert_eq!(result, r#"[{"a": 1}, {"a": 2}]"#);
    }

    #[test]
    fn extract_json_nested() {
        let input = r#"{"outer": {"inner": [1, 2, 3]}}"#;
        let result = extract_json(input).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn extract_json_no_json() {
        let input = "This is just plain text with no JSON.";
        let result = extract_json(input);
        assert!(result.is_err());
    }

    #[test]
    fn extract_json_strings_with_braces() {
        let input = r#"{"msg": "hello {world}"}"#;
        let result = extract_json(input).unwrap();
        assert_eq!(result, input);
    }
}
