//! Typed configuration for shared runtime startup.

use std::env;

use crate::consolidation::{ConsolidationConfig, observation, opinion_merger};
use crate::embedding::EmbeddingConfig;
use crate::llm::{self, LlmConfig, ReasoningEffortConfig};
use crate::recall::reranker::RerankerConfig;
use crate::retain::{extractor, graph_builder, resolver};

use super::env as config_env;
use super::error::{ConfigError, Result};

/// Validated startup configuration for the shared Elephant runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    database_url: String,
    llm: LlmConfig,
    embedding: EmbeddingConfig,
    reranker: RerankerConfig,
    dedup_threshold: Option<f32>,
    reasoning_effort: ReasoningEffortConfig,
    extraction: extractor::ExtractionConfig,
    extract_temperature_override: Option<f32>,
    resolve_temperature: f32,
    resolve_temperature_override: Option<f32>,
    graph_causal_temperature: f32,
    graph_temperature_override: Option<f32>,
    consolidation: ConsolidationConfig,
    consolidate_temperature_override: Option<f32>,
    opinion_merge: opinion_merger::OpinionMergeConfig,
    opinion_merge_temperature_override: Option<f32>,
    reflect: ReflectConfig,
    reflect_temperature: f32,
    reflect_temperature_override: Option<f32>,
    retrieval: RetrievalConfig,
}

impl RuntimeConfig {
    /// Load runtime configuration from the process environment.
    pub fn from_env() -> Result<Self> {
        let database_url = config_env::required_string(
            "DATABASE_URL",
            super::error::ConfigErrorKind::Configuration,
        )?;
        let llm = crate::llm::runtime_config_from_env().map_err(ConfigError::from)?;
        let embedding = EmbeddingConfig::from_env()?;
        let reranker = RerankerConfig::from_env()?;
        let dedup_threshold = match env::var("DEDUP_THRESHOLD").as_deref() {
            Ok("none") => None,
            Ok(raw) => Some(raw.parse().map_err(|_| {
                ConfigError::configuration(format!(
                    "DEDUP_THRESHOLD must be a float or 'none', got: {raw}"
                ))
            })?),
            Err(_) => Some(0.95),
        };
        let reasoning_effort = ReasoningEffortConfig::from_env().map_err(ConfigError::from)?;
        let extraction = extractor::config_from_env().map_err(ConfigError::from)?;
        let extract_temperature_override =
            llm::temperature_from_env("RETAIN_EXTRACT_TEMPERATURE").map_err(ConfigError::from)?;
        let resolve_temperature =
            resolver::resolve_temperature_from_env().map_err(ConfigError::from)?;
        let resolve_temperature_override =
            llm::temperature_from_env("RETAIN_RESOLVE_TEMPERATURE").map_err(ConfigError::from)?;
        let graph_causal_temperature =
            graph_builder::causal_temperature_from_env().map_err(ConfigError::from)?;
        let graph_temperature_override =
            llm::temperature_from_env("RETAIN_GRAPH_TEMPERATURE").map_err(ConfigError::from)?;
        let consolidation = observation::config_from_env().map_err(ConfigError::from)?;
        let consolidate_temperature_override =
            llm::temperature_from_env("CONSOLIDATE_TEMPERATURE").map_err(ConfigError::from)?;
        let opinion_merge = opinion_merger::config_from_env().map_err(ConfigError::from)?;
        let opinion_merge_temperature_override =
            llm::temperature_from_env("OPINION_MERGE_TEMPERATURE").map_err(ConfigError::from)?;
        let reflect = ReflectConfig::from_env()?;
        let reflect_temperature =
            crate::reflect::reflect_temperature_from_env().map_err(ConfigError::from)?;
        let reflect_temperature_override =
            llm::temperature_from_env("REFLECT_TEMPERATURE").map_err(ConfigError::from)?;
        let retrieval = RetrievalConfig::from_env();

        Ok(Self {
            database_url,
            llm,
            embedding,
            reranker,
            dedup_threshold,
            reasoning_effort,
            extraction,
            extract_temperature_override,
            resolve_temperature,
            resolve_temperature_override,
            graph_causal_temperature,
            graph_temperature_override,
            consolidation,
            consolidate_temperature_override,
            opinion_merge,
            opinion_merge_temperature_override,
            reflect,
            reflect_temperature,
            reflect_temperature_override,
            retrieval,
        })
    }

    /// Return the configured database URL.
    pub fn database_url(&self) -> &str {
        &self.database_url
    }

    /// Return the validated LLM configuration bundle.
    pub fn llm(&self) -> &LlmConfig {
        &self.llm
    }

    /// Return the validated embedding provider configuration.
    pub fn embedding(&self) -> &EmbeddingConfig {
        &self.embedding
    }

    /// Return the validated reranker provider configuration.
    pub fn reranker(&self) -> &RerankerConfig {
        &self.reranker
    }

    /// Return the configured near-duplicate threshold.
    pub fn dedup_threshold(&self) -> Option<f32> {
        self.dedup_threshold
    }

    /// Return the reasoning-effort overrides.
    pub fn reasoning_effort(&self) -> ReasoningEffortConfig {
        self.reasoning_effort
    }

    /// Return the retain extraction config.
    pub fn extraction(&self) -> extractor::ExtractionConfig {
        self.extraction
    }

    /// Return the explicit retain extraction temperature override, if any.
    pub fn extract_temperature_override(&self) -> Option<f32> {
        self.extract_temperature_override
    }

    /// Return the retain entity-resolution temperature.
    pub fn resolve_temperature(&self) -> f32 {
        self.resolve_temperature
    }

    /// Return the explicit retain entity-resolution temperature override, if any.
    pub fn resolve_temperature_override(&self) -> Option<f32> {
        self.resolve_temperature_override
    }

    /// Return the retain graph causal-link temperature.
    pub fn graph_causal_temperature(&self) -> f32 {
        self.graph_causal_temperature
    }

    /// Return the explicit retain graph temperature override, if any.
    pub fn graph_temperature_override(&self) -> Option<f32> {
        self.graph_temperature_override
    }

    /// Return the observation consolidation config.
    pub fn consolidation(&self) -> ConsolidationConfig {
        self.consolidation
    }

    /// Return the explicit consolidation temperature override, if any.
    pub fn consolidate_temperature_override(&self) -> Option<f32> {
        self.consolidate_temperature_override
    }

    /// Return the opinion-merge config.
    pub fn opinion_merge(&self) -> opinion_merger::OpinionMergeConfig {
        self.opinion_merge
    }

    /// Return the explicit opinion-merge temperature override, if any.
    pub fn opinion_merge_temperature_override(&self) -> Option<f32> {
        self.opinion_merge_temperature_override
    }

    /// Return the reflect-stage runtime config.
    pub fn reflect(&self) -> &ReflectConfig {
        &self.reflect
    }

    /// Return the reflect temperature.
    pub fn reflect_temperature(&self) -> f32 {
        self.reflect_temperature
    }

    /// Return the explicit reflect temperature override, if any.
    pub fn reflect_temperature_override(&self) -> Option<f32> {
        self.reflect_temperature_override
    }

    /// Return the retrieval-stage runtime config.
    pub fn retrieval(&self) -> &RetrievalConfig {
        &self.retrieval
    }
}

/// Validated reflect-stage configuration loaded from environment.
#[derive(Debug, Clone)]
pub struct ReflectConfig {
    max_iterations: usize,
    max_tokens: Option<usize>,
    source_limit: usize,
    source_max_chars: Option<usize>,
    enable_source_lookup: bool,
}

impl ReflectConfig {
    /// Load reflect-stage configuration from the process environment.
    pub fn from_env() -> Result<Self> {
        let max_iterations =
            config_env::optional_usize_lossy("REFLECT_MAX_ITERATIONS").unwrap_or(8);
        let max_tokens = config_env::optional_usize_lossy("REFLECT_MAX_TOKENS");
        let source_limit = config_env::optional_usize_lossy("REFLECT_SOURCE_LIMIT")
            .unwrap_or(crate::reflect::DEFAULT_SOURCE_LOOKUP_LIMIT);
        let source_max_chars = config_env::optional_usize_lossy("REFLECT_SOURCE_MAX_CHARS");
        let enable_source_lookup = match env::var("REFLECT_ENABLE_SOURCE_LOOKUP") {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => true,
                "0" | "false" | "no" | "off" => false,
                other => {
                    return Err(ConfigError::configuration(format!(
                        "REFLECT_ENABLE_SOURCE_LOOKUP must be a boolean, got: {other}"
                    )));
                }
            },
            Err(_) => crate::reflect::DEFAULT_ENABLE_SOURCE_LOOKUP,
        };

        Ok(Self {
            max_iterations,
            max_tokens,
            source_limit,
            source_max_chars,
            enable_source_lookup,
        })
    }

    /// Return the reflect iteration cap.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Return the optional reflect completion cap.
    pub fn max_tokens(&self) -> Option<usize> {
        self.max_tokens
    }

    /// Return the source-lookup result cap.
    pub fn source_limit(&self) -> usize {
        self.source_limit
    }

    /// Return the optional source content truncation cap.
    pub fn source_max_chars(&self) -> Option<usize> {
        self.source_max_chars
    }

    /// Return whether source lookup is enabled.
    pub fn enable_source_lookup(&self) -> bool {
        self.enable_source_lookup
    }
}

/// Validated retrieval-stage configuration loaded from environment.
#[derive(Debug, Clone, Copy)]
pub struct RetrievalConfig {
    retriever_limit: usize,
    max_facts: usize,
}

impl RetrievalConfig {
    /// Load retrieval-stage configuration from the process environment.
    pub fn from_env() -> Self {
        Self {
            retriever_limit: config_env::optional_usize_lossy("RETRIEVER_LIMIT").unwrap_or(40),
            max_facts: config_env::optional_usize_lossy("MAX_FACTS").unwrap_or(50),
        }
    }

    /// Return the candidate limit per retrieval strategy.
    pub fn retriever_limit(&self) -> usize {
        self.retriever_limit
    }

    /// Return the maximum fact count before token budgeting.
    pub fn max_facts(&self) -> usize {
        self.max_facts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn reflect_config_rejects_invalid_source_lookup_flag() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("REFLECT_ENABLE_SOURCE_LOOKUP", "maybe");
        }

        let err = ReflectConfig::from_env().unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("REFLECT_ENABLE_SOURCE_LOOKUP"));

        unsafe {
            env::remove_var("REFLECT_ENABLE_SOURCE_LOOKUP");
        }
    }

    #[test]
    fn retrieval_config_preserves_lossy_defaulting() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("RETRIEVER_LIMIT", "not-a-number");
            env::set_var("MAX_FACTS", "still-not-a-number");
        }

        let config = RetrievalConfig::from_env();
        assert_eq!(config.retriever_limit(), 40);
        assert_eq!(config.max_facts(), 50);

        unsafe {
            env::remove_var("RETRIEVER_LIMIT");
            env::remove_var("MAX_FACTS");
        }
    }

    #[test]
    fn runtime_config_rejects_invalid_dedup_threshold_as_configuration_error() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("DATABASE_URL", "postgres://example");
            env::set_var("LLM_PROVIDER", "openai");
            env::set_var("LLM_API_KEY", "test-key");
            env::set_var("LLM_MODEL", "gpt-4o-mini");
            env::set_var("EMBEDDING_PROVIDER", "local");
            env::set_var("EMBEDDING_MODEL_PATH", "/tmp/model");
            env::set_var("RERANKER_PROVIDER", "none");
            env::set_var("DEDUP_THRESHOLD", "nope");
        }

        let err = RuntimeConfig::from_env().unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("DEDUP_THRESHOLD"));

        unsafe {
            env::remove_var("DATABASE_URL");
            env::remove_var("LLM_PROVIDER");
            env::remove_var("LLM_API_KEY");
            env::remove_var("LLM_MODEL");
            env::remove_var("EMBEDDING_PROVIDER");
            env::remove_var("EMBEDDING_MODEL_PATH");
            env::remove_var("RERANKER_PROVIDER");
            env::remove_var("DEDUP_THRESHOLD");
        }
    }
}
