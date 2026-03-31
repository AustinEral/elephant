//! Environment loaders for runtime-specific typed configuration.

use std::env;

use crate::embedding::{EmbeddingConfig, EmbeddingProvider};
use crate::llm::{LlmConfig, Provider, ReasoningEffort, VertexConfig};
use crate::recall::reranker::{RerankerConfig, RerankerProvider};

use super::super::client_env::{
    SharedClientEnvConfig, build_client_config, optional_string_any, required_string_any,
};
use super::super::env as config_env;
use super::super::error::{ConfigError, ConfigErrorKind, Result};

pub(super) fn parse_optional_temperature(name: &'static str) -> Result<Option<f32>> {
    match env::var(name) {
        Ok(raw) => {
            let value = raw.trim().parse::<f32>().map_err(|_| {
                ConfigError::configuration(format!("{name} must be a float, got: {raw}"))
            })?;
            if !value.is_finite() || value < 0.0 {
                return Err(ConfigError::configuration(format!(
                    "{name} must be a finite, non-negative float, got: {raw}"
                )));
            }
            Ok(Some(value))
        }
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(ConfigError::configuration(format!(
            "{name} must be set: {err}"
        ))),
    }
}

pub(super) fn parse_optional_positive_usize(name: &'static str) -> Result<Option<usize>> {
    match config_env::parse_optional_usize(name, ConfigErrorKind::Configuration)? {
        Some(0) => Err(ConfigError::configuration(format!(
            "{name} must be greater than 0"
        ))),
        other => Ok(other),
    }
}

pub(super) fn required_nonblank_string(name: &'static str) -> Result<String> {
    let value = config_env::required_string(name, ConfigErrorKind::Configuration)?;
    if value.trim().is_empty() {
        return Err(ConfigError::configuration(format!(
            "{name} must not be blank"
        )));
    }
    Ok(value)
}

pub(super) fn parse_optional_reasoning_effort(
    name: &'static str,
) -> Result<Option<ReasoningEffort>> {
    match env::var(name) {
        Ok(raw) => {
            let effort = match raw.trim().to_ascii_lowercase().as_str() {
                "minimal" => ReasoningEffort::Minimal,
                "low" => ReasoningEffort::Low,
                "medium" => ReasoningEffort::Medium,
                "high" => ReasoningEffort::High,
                "xhigh" => ReasoningEffort::XHigh,
                "none" => ReasoningEffort::None,
                _ => {
                    return Err(ConfigError::configuration(format!(
                        "{name} must be one of: none, minimal, low, medium, high, xhigh; got: {raw}"
                    )));
                }
            };
            Ok(Some(effort))
        }
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(ConfigError::configuration(format!(
            "{name} must be set: {err}"
        ))),
    }
}

pub(super) fn runtime_llm_config_from_env() -> Result<LlmConfig> {
    let provider = config_env::required_string("LLM_PROVIDER", ConfigErrorKind::Configuration)?
        .parse::<Provider>()
        .map_err(ConfigError::from)?;
    let shared = SharedClientEnvConfig::new(
        provider,
        config_env::required_string("LLM_API_KEY", ConfigErrorKind::Configuration)?,
        if provider == Provider::Vertex {
            Some(required_string_any(&[
                "LLM_VERTEX_PROJECT",
                "GOOGLE_CLOUD_PROJECT",
            ])?)
        } else {
            None
        },
        if provider == Provider::Vertex {
            optional_string_any(&["LLM_VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION"])?
                .or(Some(VertexConfig::DEFAULT_LOCATION.into()))
        } else {
            None
        },
        config_env::optional_string("LLM_BASE_URL", ConfigErrorKind::Configuration)?,
    )?;

    let retain = build_client_config(
        &shared,
        required_string_any(&["RETAIN_LLM_MODEL", "LLM_MODEL"])?,
        "LLM_VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT must be set for provider=vertex",
    )?;
    let reflect = build_client_config(
        &shared,
        required_string_any(&["REFLECT_LLM_MODEL", "LLM_MODEL"])?,
        "LLM_VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT must be set for provider=vertex",
    )?;

    Ok(LlmConfig::new(retain, reflect))
}

pub(super) fn embedding_config_from_env() -> Result<EmbeddingConfig> {
    let provider =
        match config_env::required_string("EMBEDDING_PROVIDER", ConfigErrorKind::Configuration)?
            .as_str()
        {
            "openai" => EmbeddingProvider::OpenAi,
            "local" => EmbeddingProvider::Local,
            other => {
                return Err(ConfigError::configuration(format!(
                    "unknown EMBEDDING_PROVIDER: {other}"
                )));
            }
        };

    match provider {
        EmbeddingProvider::Local => {
            let model_path = required_nonblank_string("EMBEDDING_MODEL_PATH")?;
            let max_seq_len =
                parse_optional_positive_usize("EMBEDDING_MAX_SEQ_LEN")?.unwrap_or(512);
            Ok(EmbeddingConfig::local(model_path).with_max_seq_len(max_seq_len))
        }
        EmbeddingProvider::OpenAi => {
            let api_key = required_nonblank_string("EMBEDDING_API_KEY")?;
            let model = required_nonblank_string("EMBEDDING_API_MODEL")?;
            let dimensions =
                parse_optional_positive_usize("EMBEDDING_API_DIMS")?.ok_or_else(|| {
                    ConfigError::configuration(
                        "EMBEDDING_API_DIMS must be set for openai embeddings",
                    )
                })?;
            Ok(EmbeddingConfig::openai(api_key, model, dimensions))
        }
    }
}

pub(super) fn reranker_config_from_env() -> Result<RerankerConfig> {
    let provider =
        match config_env::required_string("RERANKER_PROVIDER", ConfigErrorKind::Configuration)?
            .as_str()
        {
            "local" => RerankerProvider::Local,
            "api" => RerankerProvider::Api,
            "none" => RerankerProvider::None,
            other => {
                return Err(ConfigError::configuration(format!(
                    "unknown RERANKER_PROVIDER: {other}"
                )));
            }
        };

    match provider {
        RerankerProvider::None => Ok(RerankerConfig::none()),
        RerankerProvider::Local => {
            let model_path = required_nonblank_string("RERANKER_MODEL_PATH")?;
            let max_seq_len = parse_optional_positive_usize("RERANKER_MAX_SEQ_LEN")?.unwrap_or(512);
            Ok(RerankerConfig::local(model_path).with_max_seq_len(max_seq_len))
        }
        RerankerProvider::Api => {
            let api_key = required_nonblank_string("RERANKER_API_KEY")?;
            let api_url = required_nonblank_string("RERANKER_API_URL")?;
            let api_model = required_nonblank_string("RERANKER_API_MODEL")?;
            Ok(RerankerConfig::api(api_key, api_url, api_model))
        }
    }
}
