//! Validation helpers for programmatic runtime configuration.

use crate::embedding::{EmbeddingConfig, EmbeddingProvider};
use crate::recall::reranker::{RerankerConfig, RerankerProvider};

use super::super::error::{ConfigError, Result};

pub(super) fn validate_nonblank_field(name: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ConfigError::configuration(format!(
            "{name} must not be blank"
        )));
    }
    Ok(())
}

pub(super) fn validate_positive_usize_field(name: &str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(ConfigError::configuration(format!(
            "{name} must be greater than 0"
        )));
    }
    Ok(())
}

pub(super) fn validate_nonnegative_float(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(ConfigError::configuration(format!(
            "{name} must be a finite, non-negative float"
        )));
    }
    Ok(())
}

pub(super) fn validate_optional_nonnegative_float(name: &str, value: Option<f32>) -> Result<()> {
    if let Some(value) = value {
        validate_nonnegative_float(name, value)?;
    }
    Ok(())
}

pub(super) fn validate_embedding_config(config: &EmbeddingConfig) -> Result<()> {
    validate_positive_usize_field("embedding.max_seq_len", config.max_seq_len())?;
    match config.provider() {
        EmbeddingProvider::Local => {
            let model_path = config.model_path().ok_or_else(|| {
                ConfigError::configuration("embedding.model_path must be set for local embeddings")
            })?;
            validate_nonblank_field("embedding.model_path", model_path)?;
        }
        EmbeddingProvider::OpenAi => {
            let api_key = config.api_key().ok_or_else(|| {
                ConfigError::configuration("embedding.api_key must be set for OpenAI embeddings")
            })?;
            validate_nonblank_field("embedding.api_key", api_key)?;
            let model = config.model().ok_or_else(|| {
                ConfigError::configuration("embedding.model must be set for OpenAI embeddings")
            })?;
            validate_nonblank_field("embedding.model", model)?;
            let dimensions = config.dimensions().ok_or_else(|| {
                ConfigError::configuration("embedding.dimensions must be set for OpenAI embeddings")
            })?;
            validate_positive_usize_field("embedding.dimensions", dimensions)?;
        }
    }
    Ok(())
}

pub(super) fn validate_reranker_config(config: &RerankerConfig) -> Result<()> {
    validate_positive_usize_field("reranker.max_seq_len", config.max_seq_len())?;
    match config.provider() {
        RerankerProvider::None => {}
        RerankerProvider::Local => {
            let model_path = config.model_path().ok_or_else(|| {
                ConfigError::configuration("reranker.model_path must be set for local reranker")
            })?;
            validate_nonblank_field("reranker.model_path", model_path)?;
        }
        RerankerProvider::Api => {
            let api_key = config.api_key().ok_or_else(|| {
                ConfigError::configuration("reranker.api_key must be set for API reranker")
            })?;
            validate_nonblank_field("reranker.api_key", api_key)?;
            let api_url = config.api_url().ok_or_else(|| {
                ConfigError::configuration("reranker.api_url must be set for API reranker")
            })?;
            validate_nonblank_field("reranker.api_url", api_url)?;
            let api_model = config.api_model().ok_or_else(|| {
                ConfigError::configuration("reranker.api_model must be set for API reranker")
            })?;
            validate_nonblank_field("reranker.api_model", api_model)?;
        }
    }
    Ok(())
}
