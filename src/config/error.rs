//! Dedicated error type for startup and environment configuration loading.

use thiserror::Error;

/// High-level classification for configuration-loading failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigErrorKind {
    /// Invalid or missing user-facing configuration.
    Configuration,
    /// Legacy internal classification preserved during the seam refactor.
    Internal,
}

/// Error returned while loading typed configuration.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ConfigError {
    kind: ConfigErrorKind,
    message: String,
}

impl ConfigError {
    /// Create a configuration-classified error.
    pub fn configuration(message: impl Into<String>) -> Self {
        Self {
            kind: ConfigErrorKind::Configuration,
            message: message.into(),
        }
    }

    /// Create an internally-classified error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: ConfigErrorKind::Internal,
            message: message.into(),
        }
    }

    /// Return the classification of this error.
    pub fn kind(&self) -> ConfigErrorKind {
        self.kind
    }

    /// Consume the error and return its message.
    pub fn into_message(self) -> String {
        self.message
    }
}

impl From<crate::error::Error> for ConfigError {
    fn from(error: crate::error::Error) -> Self {
        match error {
            crate::error::Error::Configuration(message) => Self::configuration(message),
            crate::error::Error::Internal(message) => Self::internal(message),
            other => Self::configuration(other.to_string()),
        }
    }
}

/// Result alias for typed configuration loading.
pub type Result<T> = std::result::Result<T, ConfigError>;
