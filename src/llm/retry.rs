//! Retrying decorator for LLM clients.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{Error, Result};
use crate::llm::LlmClient;
use crate::types::llm::{CompletionRequest, CompletionResponse};

/// Configurable retry policy for LLM calls.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (not counting the initial attempt).
    pub max_retries: u32,
    /// Seconds to wait when rate-limited (429).
    pub rate_limit_wait_secs: u64,
    /// Seconds to wait on the first server-error retry; doubles each attempt.
    pub initial_backoff_secs: u64,
    /// Maximum backoff cap in seconds.
    pub max_backoff_secs: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 5,
            rate_limit_wait_secs: 15,
            initial_backoff_secs: 2,
            max_backoff_secs: 60,
        }
    }
}

/// Wraps any [`LlmClient`] with automatic retry on transient errors.
///
/// Retries on [`Error::RateLimit`] and [`Error::ServerError`]; all other errors
/// are returned immediately.
pub struct RetryingLlmClient {
    inner: Arc<dyn LlmClient>,
    policy: RetryPolicy,
}

impl RetryingLlmClient {
    /// Create a new retrying wrapper around an existing client.
    pub fn new(inner: Arc<dyn LlmClient>, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }
}

#[async_trait]
impl LlmClient for RetryingLlmClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let mut last_err: Option<Error> = None;

        for attempt in 0..=self.policy.max_retries {
            let req = request.clone();
            match self.inner.complete(req).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    let wait = match &e {
                        Error::RateLimit(_) => Some(self.policy.rate_limit_wait_secs),
                        Error::ServerError(_) => {
                            let backoff =
                                self.policy.initial_backoff_secs * 2u64.saturating_pow(attempt);
                            Some(backoff.min(self.policy.max_backoff_secs))
                        }
                        _ => None,
                    };

                    if let Some(secs) = wait
                        && attempt < self.policy.max_retries
                    {
                        tracing::warn!(
                            attempt = attempt + 1,
                            max_retries = self.policy.max_retries,
                            wait_secs = secs,
                            error = %e,
                            "llm_retry"
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
                        last_err = Some(e);
                        continue;
                    }

                    return Err(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| Error::Llm("retry exhausted".into())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A mock client that fails N times then succeeds.
    struct FailThenSucceed {
        failures_remaining: AtomicU32,
        error_kind: &'static str, // "rate_limit" or "server"
    }

    impl FailThenSucceed {
        fn new(failures: u32, kind: &'static str) -> Self {
            Self {
                failures_remaining: AtomicU32::new(failures),
                error_kind: kind,
            }
        }
    }

    #[async_trait]
    impl LlmClient for FailThenSucceed {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse> {
            let remaining = self.failures_remaining.load(Ordering::SeqCst);
            if remaining > 0 {
                self.failures_remaining.fetch_sub(1, Ordering::SeqCst);
                match self.error_kind {
                    "rate_limit" => Err(Error::RateLimit("429 too many requests".into())),
                    "server" => Err(Error::ServerError("502 bad gateway".into())),
                    _ => Err(Error::Llm("unknown".into())),
                }
            } else {
                Ok(CompletionResponse {
                    content: "ok".into(),
                    input_tokens: 1,
                    output_tokens: 1,
                    stop_reason: None,
                    tool_calls: vec![],
                })
            }
        }
    }

    fn test_policy() -> RetryPolicy {
        RetryPolicy {
            max_retries: 3,
            rate_limit_wait_secs: 0, // no actual sleep in tests
            initial_backoff_secs: 0,
            max_backoff_secs: 0,
        }
    }

    fn test_request() -> CompletionRequest {
        CompletionRequest {
            model: String::new(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
            system: None,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn retries_on_rate_limit_then_succeeds() {
        let inner = Arc::new(FailThenSucceed::new(2, "rate_limit"));
        let client = RetryingLlmClient::new(inner, test_policy());
        let resp = client.complete(test_request()).await.unwrap();
        assert_eq!(resp.content, "ok");
    }

    #[tokio::test]
    async fn retries_on_server_error_then_succeeds() {
        let inner = Arc::new(FailThenSucceed::new(2, "server"));
        let client = RetryingLlmClient::new(inner, test_policy());
        let resp = client.complete(test_request()).await.unwrap();
        assert_eq!(resp.content, "ok");
    }

    #[tokio::test]
    async fn gives_up_after_max_retries() {
        let inner = Arc::new(FailThenSucceed::new(10, "rate_limit"));
        let client = RetryingLlmClient::new(inner, test_policy());
        let err = client.complete(test_request()).await.unwrap_err();
        assert!(matches!(err, Error::RateLimit(_)));
    }

    #[tokio::test]
    async fn non_retryable_error_returned_immediately() {
        let inner = Arc::new(FailThenSucceed::new(1, "other"));
        let client = RetryingLlmClient::new(inner, test_policy());
        let err = client.complete(test_request()).await.unwrap_err();
        assert!(matches!(err, Error::Llm(_)));
    }
}
