//! Error types for the memory engine.

use crate::config::{ConfigError, ConfigErrorKind};

/// Errors that can occur in the memory engine.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A disposition field value is out of the valid range.
    #[error("invalid disposition: {0}")]
    InvalidDisposition(String),

    /// A serialization or deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// An ID string could not be parsed.
    #[error("invalid id: {0}")]
    InvalidId(String),

    /// A database error from sqlx.
    #[error("storage error: {0}")]
    Storage(#[from] sqlx::Error),

    /// A requested resource was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// An error from an LLM API call or response parsing.
    #[error("llm error: {0}")]
    Llm(String),

    /// The LLM response contained no JSON (model went off-script).
    #[error("llm error: no JSON found in response")]
    LlmNoJson,

    /// The LLM refused to respond (safety filter).
    #[error("llm error: model refused to respond")]
    LlmRefusal,

    /// The LLM provider returned a rate-limit response (HTTP 429).
    #[error("rate limited: {0}")]
    RateLimit(String),

    /// The LLM provider returned a server error (HTTP 5xx).
    #[error("server error: {0}")]
    ServerError(String),

    /// An error from embedding generation.
    #[error("embedding error: {0}")]
    Embedding(String),

    /// An error from the reranker.
    #[error("reranker error: {0}")]
    Reranker(String),

    /// An internal/logic error.
    #[error("internal error: {0}")]
    Internal(String),

    /// Invalid runtime or API configuration.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// The bank's embedding config doesn't match the current embedding client.
    #[error(
        "embedding dimension mismatch: bank expects {expected} dims ({model}), but client produces {actual} dims"
    )]
    EmbeddingDimensionMismatch {
        /// The model name recorded on the bank.
        model: String,
        /// The dimension count the bank expects.
        expected: u16,
        /// The dimension count the current client produces.
        actual: u16,
    },
}

impl From<ConfigError> for Error {
    fn from(error: ConfigError) -> Self {
        let kind = error.kind();
        let message = error.into_message();
        match kind {
            ConfigErrorKind::Configuration => Self::Configuration(message),
            ConfigErrorKind::Internal => Self::Internal(message),
        }
    }
}

/// A `Result` alias using the memory engine [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = Error::InvalidDisposition("skepticism must be 1-5, got 0".into());
        assert!(err.to_string().contains("skepticism"));

        let err = Error::InvalidId("not-a-ulid".into());
        assert!(err.to_string().contains("not-a-ulid"));
    }

    #[test]
    fn from_serde_json_error() {
        let json_err = serde_json::from_str::<String>("not json").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Serialization(_)));
    }
}
