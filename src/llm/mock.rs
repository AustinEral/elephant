//! Mock LLM client for testing.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::error::Result;
use crate::llm::LlmClient;
use crate::types::llm::{CompletionRequest, CompletionResponse, CompletionUsage};

/// A mock LLM client that returns pre-queued responses.
///
/// Responses are returned in FIFO order. Panics if `complete` is called
/// with an empty queue — tests should set up all expected responses upfront.
#[derive(Debug, Clone)]
pub struct MockLlmClient {
    responses: Arc<Mutex<VecDeque<CompletionResponse>>>,
}

impl MockLlmClient {
    /// Create a new mock client with an empty response queue.
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Queue a text-only response to be returned by the next `complete` call.
    pub fn push_response(&self, text: impl Into<String>) {
        self.responses
            .lock()
            .unwrap()
            .push_back(CompletionResponse {
                content: text.into(),
                input_tokens: 10,
                output_tokens: 20,
                usage: CompletionUsage::unsupported(10, 20),
                stop_reason: None,
                tool_calls: vec![],
            });
    }

    /// Queue a full `CompletionResponse` (e.g. with tool_calls).
    pub fn push_response_full(&self, resp: CompletionResponse) {
        self.responses.lock().unwrap().push_back(resp);
    }

    /// Return the number of responses still in the queue.
    pub fn remaining(&self) -> usize {
        self.responses.lock().unwrap().len()
    }
}

impl Default for MockLlmClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse> {
        let resp = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("MockLlmClient: no responses queued — did you forget push_response()?");
        Ok(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    fn make_request() -> CompletionRequest {
        CompletionRequest {
            model: "mock".into(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
            system: None,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn returns_queued_responses_in_order() {
        let client = MockLlmClient::new();
        client.push_response("first");
        client.push_response("second");

        let r1 = client.complete(make_request()).await.unwrap();
        assert_eq!(r1.content, "first");

        let r2 = client.complete(make_request()).await.unwrap();
        assert_eq!(r2.content, "second");

        assert_eq!(client.remaining(), 0);
    }

    #[tokio::test]
    async fn complete_structured_parses_json() {
        use crate::llm::complete_structured;

        let client = MockLlmClient::new();
        client.push_response(r#"{"name": "alice", "score": 42}"#);

        #[derive(Deserialize, Debug, PartialEq)]
        struct Res {
            name: String,
            score: i32,
        }

        let parsed: Res = complete_structured(&client, make_request()).await.unwrap();
        assert_eq!(
            parsed,
            Res {
                name: "alice".into(),
                score: 42,
            }
        );
    }

    #[tokio::test]
    async fn complete_structured_handles_markdown_fences() {
        use crate::llm::complete_structured;

        let client = MockLlmClient::new();
        client.push_response("Here you go:\n```json\n{\"x\": 1}\n```");

        #[derive(Deserialize, Debug, PartialEq)]
        struct Res {
            x: i32,
        }

        let parsed: Res = complete_structured(&client, make_request()).await.unwrap();
        assert_eq!(parsed, Res { x: 1 });
    }

    #[tokio::test]
    #[should_panic(expected = "no responses queued")]
    async fn panics_when_queue_empty() {
        let client = MockLlmClient::new();
        let _ = client.complete(make_request()).await;
    }
}
