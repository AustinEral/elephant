//! Typed configuration for shared runtime startup.

use std::env;

use crate::embedding::EmbeddingConfig;
use crate::llm::LlmConfig;
use crate::recall::reranker::RerankerConfig;

use super::env as config_env;
use super::error::{ConfigError, ConfigErrorKind, Result};

/// Validated startup configuration for the shared Elephant runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    database_url: String,
    llm: LlmConfig,
    embedding: EmbeddingConfig,
    reranker: RerankerConfig,
    dedup_threshold: Option<f32>,
    reflect: ReflectConfig,
    retrieval: RetrievalConfig,
}

impl RuntimeConfig {
    /// Load runtime configuration from the process environment.
    pub fn from_env() -> Result<Self> {
        let database_url =
            config_env::required_string("DATABASE_URL", ConfigErrorKind::Configuration)?;
        let llm = crate::llm::runtime_config_from_env().map_err(ConfigError::from)?;
        let embedding = EmbeddingConfig::from_env()?;
        let reranker = RerankerConfig::from_env()?;
        let dedup_threshold = match env::var("DEDUP_THRESHOLD").as_deref() {
            Ok("none") => None,
            Ok(raw) => Some(raw.parse().map_err(|_| {
                ConfigError::internal(format!(
                    "DEDUP_THRESHOLD must be a float or 'none', got: {raw}"
                ))
            })?),
            Err(_) => Some(0.95),
        };
        let reflect = ReflectConfig::from_env()?;
        let retrieval = RetrievalConfig::from_env();

        Ok(Self {
            database_url,
            llm,
            embedding,
            reranker,
            dedup_threshold,
            reflect,
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

    /// Return the reflect-stage runtime config.
    pub fn reflect(&self) -> &ReflectConfig {
        &self.reflect
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
                    return Err(ConfigError::internal(format!(
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
        assert_eq!(err.kind(), ConfigErrorKind::Internal);
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
}
