//! LLM request and response types.

use serde::{Deserialize, Serialize};

/// A message in a conversation with an LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    /// The role of the message sender.
    pub role: String,
    /// The content of the message.
    pub content: String,
}

/// A request to an LLM for completion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
}

/// A response from an LLM completion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompletionResponse {
    /// The generated text content.
    pub content: String,
    /// Number of input tokens used.
    pub input_tokens: usize,
    /// Number of output tokens generated.
    pub output_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_request_roundtrip() {
        let req = CompletionRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![Message {
                role: "user".into(),
                content: "hello".into(),
            }],
            max_tokens: Some(1024),
            temperature: Some(0.7),
            system: Some("You are a helpful assistant.".into()),
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
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: CompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, back);
    }
}
