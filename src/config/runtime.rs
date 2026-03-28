//! Typed configuration for shared runtime startup.

use std::{env, fmt};

use crate::consolidation::{ConsolidationConfig, opinion_merger};
use crate::embedding::{EmbeddingConfig, EmbeddingProvider};
use crate::llm::{
    AnthropicConfig, AnthropicPromptCacheConfig, AnthropicPromptCacheTtl, ClientConfig,
    DEFAULT_TIMEOUT_SECS, GeminiConfig, LlmConfig, OpenAiConfig, OpenAiPromptCacheConfig,
    OpenAiPromptCacheRetention, Provider, ReasoningEffort, ReasoningEffortConfig, VertexConfig,
};
use crate::recall::reranker::{RerankerConfig, RerankerProvider};
use crate::retain::{extractor, graph_builder};

use super::env as config_env;
use super::error::{ConfigError, Result};

/// Validated startup configuration for the shared Elephant runtime.
#[derive(Clone)]
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
    graph: graph_builder::GraphConfig,
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

impl fmt::Debug for RuntimeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeConfig")
            .field("database_url", &"<redacted>")
            .field("llm", &self.llm)
            .field("embedding", &self.embedding)
            .field("reranker", &self.reranker)
            .field("dedup_threshold", &self.dedup_threshold)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("extraction", &self.extraction)
            .field(
                "extract_temperature_override",
                &self.extract_temperature_override,
            )
            .field("resolve_temperature", &self.resolve_temperature)
            .field(
                "resolve_temperature_override",
                &self.resolve_temperature_override,
            )
            .field("graph", &self.graph)
            .field(
                "graph_temperature_override",
                &self.graph_temperature_override,
            )
            .field("consolidation", &self.consolidation)
            .field(
                "consolidate_temperature_override",
                &self.consolidate_temperature_override,
            )
            .field("opinion_merge", &self.opinion_merge)
            .field(
                "opinion_merge_temperature_override",
                &self.opinion_merge_temperature_override,
            )
            .field("reflect", &self.reflect)
            .field("reflect_temperature", &self.reflect_temperature)
            .field(
                "reflect_temperature_override",
                &self.reflect_temperature_override,
            )
            .field("retrieval", &self.retrieval)
            .finish()
    }
}

impl RuntimeConfig {
    /// Load runtime configuration from the process environment.
    pub fn from_env() -> Result<Self> {
        let database_url = config_env::required_string(
            "DATABASE_URL",
            super::error::ConfigErrorKind::Configuration,
        )?;
        let llm = runtime_llm_config_from_env()?;
        let embedding = embedding_config_from_env()?;
        let reranker = reranker_config_from_env()?;
        let dedup_threshold = match env::var("DEDUP_THRESHOLD").as_deref() {
            Ok("none") => None,
            Ok(raw) => Some(raw.parse().map_err(|_| {
                ConfigError::configuration(format!(
                    "DEDUP_THRESHOLD must be a float or 'none', got: {raw}"
                ))
            })?),
            Err(_) => Some(0.95),
        };
        let reasoning_effort = ReasoningEffortConfig {
            retain_extract: parse_optional_reasoning_effort("RETAIN_EXTRACT_REASONING_EFFORT")?,
            retain_resolve: parse_optional_reasoning_effort("RETAIN_RESOLVE_REASONING_EFFORT")?,
            retain_graph: parse_optional_reasoning_effort("RETAIN_GRAPH_REASONING_EFFORT")?,
            reflect: parse_optional_reasoning_effort("REFLECT_REASONING_EFFORT")?,
            consolidate: parse_optional_reasoning_effort("CONSOLIDATE_REASONING_EFFORT")?,
            opinion_merge: parse_optional_reasoning_effort("OPINION_MERGE_REASONING_EFFORT")?,
        };
        let extraction = extractor::ExtractionConfig {
            structured_output_max_attempts: config_env::optional_usize_lossy(
                "RETAIN_EXTRACT_STRUCTURED_OUTPUT_MAX_ATTEMPTS",
            )
            .filter(|&v| v > 0)
            .unwrap_or(extractor::ExtractionConfig::default().structured_output_max_attempts),
            temperature: parse_optional_temperature("RETAIN_EXTRACT_TEMPERATURE")?
                .unwrap_or(extractor::ExtractionConfig::default().temperature),
            reasoning_effort: reasoning_effort.retain_extract,
        };
        let extract_temperature_override =
            parse_optional_temperature("RETAIN_EXTRACT_TEMPERATURE")?;
        let resolve_temperature = parse_optional_temperature("RETAIN_RESOLVE_TEMPERATURE")?
            .unwrap_or(crate::retain::resolver::ENTITY_RESOLUTION_TEMPERATURE);
        let resolve_temperature_override =
            parse_optional_temperature("RETAIN_RESOLVE_TEMPERATURE")?;
        let graph = graph_builder::GraphConfig {
            causal_temperature: parse_optional_temperature("RETAIN_GRAPH_TEMPERATURE")?
                .unwrap_or(graph_builder::CAUSAL_LINK_TEMPERATURE),
            causal_reasoning_effort: reasoning_effort.retain_graph,
            ..graph_builder::GraphConfig::default()
        };
        let graph_temperature_override = parse_optional_temperature("RETAIN_GRAPH_TEMPERATURE")?;
        let consolidation = ConsolidationConfig {
            batch_size: config_env::optional_usize_lossy("CONSOLIDATION_BATCH_SIZE")
                .unwrap_or(ConsolidationConfig::default().batch_size),
            max_tokens: config_env::optional_usize_lossy("CONSOLIDATION_MAX_TOKENS")
                .unwrap_or(ConsolidationConfig::default().max_tokens),
            recall_budget: config_env::optional_usize_lossy("CONSOLIDATION_RECALL_BUDGET")
                .unwrap_or(ConsolidationConfig::default().recall_budget),
            structured_output_max_attempts: config_env::optional_usize_lossy(
                "CONSOLIDATION_STRUCTURED_OUTPUT_MAX_ATTEMPTS",
            )
            .filter(|&v| v > 0)
            .unwrap_or(ConsolidationConfig::default().structured_output_max_attempts),
            temperature: parse_optional_temperature("CONSOLIDATE_TEMPERATURE")?
                .unwrap_or(ConsolidationConfig::default().temperature),
            reasoning_effort: reasoning_effort.consolidate,
        };
        let consolidate_temperature_override =
            parse_optional_temperature("CONSOLIDATE_TEMPERATURE")?;
        let opinion_merge = opinion_merger::OpinionMergeConfig {
            temperature: parse_optional_temperature("OPINION_MERGE_TEMPERATURE")?
                .unwrap_or(opinion_merger::OpinionMergeConfig::default().temperature),
            reasoning_effort: reasoning_effort.opinion_merge,
        };
        let opinion_merge_temperature_override =
            parse_optional_temperature("OPINION_MERGE_TEMPERATURE")?;
        let reflect = ReflectConfig::from_env()?;
        let reflect_temperature = parse_optional_temperature("REFLECT_TEMPERATURE")?
            .unwrap_or(crate::reflect::REFLECT_TEMPERATURE);
        let reflect_temperature_override = parse_optional_temperature("REFLECT_TEMPERATURE")?;
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
            graph,
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

    /// Return the retain graph-building config.
    pub fn graph(&self) -> graph_builder::GraphConfig {
        self.graph.clone()
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

#[derive(Clone)]
struct RuntimeLlmSharedConfig {
    provider: Provider,
    api_key: String,
    vertex_project: Option<String>,
    vertex_location: Option<String>,
    base_url: Option<String>,
    timeout_secs: u64,
    openai_prompt_cache: Option<OpenAiPromptCacheConfig>,
    anthropic_prompt_cache: Option<AnthropicPromptCacheConfig>,
}

fn required_string_any(names: &[&str]) -> Result<String> {
    for name in names {
        if let Some(value) = config_env::optional_string(name) {
            return Ok(value);
        }
    }

    Err(ConfigError::configuration(format!(
        "{} must be set",
        names.join(" or ")
    )))
}

fn optional_string_any(names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| config_env::optional_string(name))
}

fn parse_optional_temperature(name: &'static str) -> Result<Option<f32>> {
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

fn parse_optional_reasoning_effort(name: &'static str) -> Result<Option<ReasoningEffort>> {
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

fn parse_timeout_secs() -> Result<u64> {
    match env::var("LLM_TIMEOUT_SECS") {
        Ok(value) => {
            let timeout_secs = value.parse::<u64>().map_err(|_| {
                ConfigError::configuration(format!(
                    "LLM_TIMEOUT_SECS must be a positive integer, got: {value}"
                ))
            })?;
            if timeout_secs == 0 {
                return Err(ConfigError::configuration(
                    "LLM_TIMEOUT_SECS must be greater than zero",
                ));
            }
            Ok(timeout_secs)
        }
        Err(env::VarError::NotPresent) => Ok(DEFAULT_TIMEOUT_SECS),
        Err(err) => Err(ConfigError::configuration(format!(
            "LLM_TIMEOUT_SECS must be set: {err}"
        ))),
    }
}

fn prompt_cache_enabled() -> Result<bool> {
    match env::var("LLM_PROMPT_CACHE_ENABLED") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            other => Err(ConfigError::configuration(format!(
                "LLM_PROMPT_CACHE_ENABLED must be a boolean, got: {other}"
            ))),
        },
        Err(env::VarError::NotPresent) => Ok(false),
        Err(err) => Err(ConfigError::configuration(format!(
            "LLM_PROMPT_CACHE_ENABLED must be set: {err}"
        ))),
    }
}

fn parse_openai_prompt_cache() -> Result<Option<OpenAiPromptCacheConfig>> {
    if !prompt_cache_enabled()? {
        return Ok(None);
    }

    let retention = match env::var("OPENAI_PROMPT_CACHE_RETENTION") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "in_memory" | "in-memory" => Some(OpenAiPromptCacheRetention::InMemory),
            "24h" => Some(OpenAiPromptCacheRetention::Hours24),
            other => {
                return Err(ConfigError::configuration(format!(
                    "OPENAI_PROMPT_CACHE_RETENTION must be one of: in_memory, in-memory, 24h; got: {other}"
                )));
            }
        },
        Err(env::VarError::NotPresent) => None,
        Err(err) => {
            return Err(ConfigError::configuration(format!(
                "OPENAI_PROMPT_CACHE_RETENTION must be set: {err}"
            )));
        }
    };

    let mut config = OpenAiPromptCacheConfig::new();
    if let Some(key) = config_env::optional_string("OPENAI_PROMPT_CACHE_KEY") {
        config = config.with_key(key);
    }
    if let Some(retention) = retention {
        config = config.with_retention(retention);
    }
    Ok(Some(config))
}

fn parse_anthropic_prompt_cache() -> Result<Option<AnthropicPromptCacheConfig>> {
    if !prompt_cache_enabled()? {
        return Ok(None);
    }

    let ttl = match env::var("ANTHROPIC_PROMPT_CACHE_TTL") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "5m" => Some(AnthropicPromptCacheTtl::Minutes5),
            "1h" => Some(AnthropicPromptCacheTtl::Hours1),
            other => {
                return Err(ConfigError::configuration(format!(
                    "ANTHROPIC_PROMPT_CACHE_TTL must be one of: 5m, 1h; got: {other}"
                )));
            }
        },
        Err(env::VarError::NotPresent) => None,
        Err(err) => {
            return Err(ConfigError::configuration(format!(
                "ANTHROPIC_PROMPT_CACHE_TTL must be set: {err}"
            )));
        }
    };

    let mut config = AnthropicPromptCacheConfig::new();
    if let Some(ttl) = ttl {
        config = config.with_ttl(ttl);
    }
    Ok(Some(config))
}

fn build_runtime_client_config(
    shared: &RuntimeLlmSharedConfig,
    model: String,
) -> Result<ClientConfig> {
    let config = match shared.provider {
        Provider::Anthropic => {
            let mut config = AnthropicConfig::new(shared.api_key.clone(), model)
                .map_err(ConfigError::from)?
                .with_timeout_secs(shared.timeout_secs)
                .map_err(ConfigError::from)?;
            if let Some(prompt_cache) = shared.anthropic_prompt_cache.clone() {
                config = config.with_prompt_cache(prompt_cache);
            }
            ClientConfig::Anthropic(config)
        }
        Provider::OpenAi => {
            let mut config = OpenAiConfig::new(shared.api_key.clone(), model)
                .map_err(ConfigError::from)?
                .with_timeout_secs(shared.timeout_secs)
                .map_err(ConfigError::from)?;
            if let Some(base_url) = shared.base_url.clone() {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            if let Some(prompt_cache) = shared.openai_prompt_cache.clone() {
                config = config.with_prompt_cache(prompt_cache);
            }
            ClientConfig::OpenAi(config)
        }
        Provider::Gemini => {
            let mut config = GeminiConfig::new(shared.api_key.clone(), model)
                .map_err(ConfigError::from)?
                .with_timeout_secs(shared.timeout_secs)
                .map_err(ConfigError::from)?;
            if let Some(base_url) = shared.base_url.clone() {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            ClientConfig::Gemini(config)
        }
        Provider::Vertex => {
            let project = shared.vertex_project.clone().ok_or_else(|| {
                ConfigError::configuration(
                    "LLM_VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT must be set for provider=vertex",
                )
            })?;
            let mut config = VertexConfig::new(shared.api_key.clone(), model, project)
                .map_err(ConfigError::from)?
                .with_timeout_secs(shared.timeout_secs)
                .map_err(ConfigError::from)?;
            if let Some(location) = shared.vertex_location.clone() {
                config = config.with_location(location).map_err(ConfigError::from)?;
            }
            if let Some(base_url) = shared.base_url.clone() {
                config = config.with_base_url(base_url).map_err(ConfigError::from)?;
            }
            ClientConfig::Vertex(config)
        }
    };

    Ok(config)
}

fn runtime_llm_config_from_env() -> Result<LlmConfig> {
    let provider =
        config_env::required_string("LLM_PROVIDER", super::error::ConfigErrorKind::Configuration)?
            .parse::<Provider>()
            .map_err(ConfigError::from)?;
    let shared = RuntimeLlmSharedConfig {
        provider,
        api_key: config_env::required_string(
            "LLM_API_KEY",
            super::error::ConfigErrorKind::Configuration,
        )?,
        vertex_project: if provider == Provider::Vertex {
            Some(required_string_any(&[
                "LLM_VERTEX_PROJECT",
                "GOOGLE_CLOUD_PROJECT",
            ])?)
        } else {
            None
        },
        vertex_location: if provider == Provider::Vertex {
            optional_string_any(&["LLM_VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION"])
                .or(Some(VertexConfig::DEFAULT_LOCATION.into()))
        } else {
            None
        },
        base_url: config_env::optional_string("LLM_BASE_URL"),
        timeout_secs: parse_timeout_secs()?,
        openai_prompt_cache: if provider == Provider::OpenAi {
            parse_openai_prompt_cache()?
        } else {
            None
        },
        anthropic_prompt_cache: if provider == Provider::Anthropic {
            parse_anthropic_prompt_cache()?
        } else {
            None
        },
    };

    let retain = build_runtime_client_config(
        &shared,
        required_string_any(&["RETAIN_LLM_MODEL", "LLM_MODEL"])?,
    )?;
    let reflect = build_runtime_client_config(
        &shared,
        required_string_any(&["REFLECT_LLM_MODEL", "LLM_MODEL"])?,
    )?;

    Ok(LlmConfig::new(retain, reflect))
}

fn embedding_config_from_env() -> Result<EmbeddingConfig> {
    let provider = match config_env::required_string(
        "EMBEDDING_PROVIDER",
        super::error::ConfigErrorKind::Configuration,
    )?
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

    Ok(EmbeddingConfig::from_parts(
        provider,
        config_env::optional_string("EMBEDDING_MODEL_PATH"),
        config_env::optional_usize_lossy("EMBEDDING_MAX_SEQ_LEN").unwrap_or(512),
        config_env::optional_string("EMBEDDING_API_KEY"),
        config_env::optional_string("EMBEDDING_API_MODEL"),
        config_env::optional_usize_lossy("EMBEDDING_API_DIMS"),
    ))
}

fn reranker_config_from_env() -> Result<RerankerConfig> {
    let provider = match config_env::required_string(
        "RERANKER_PROVIDER",
        super::error::ConfigErrorKind::Configuration,
    )?
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

    Ok(RerankerConfig::from_parts(
        provider,
        config_env::optional_string("RERANKER_MODEL_PATH"),
        config_env::optional_usize_lossy("RERANKER_MAX_SEQ_LEN").unwrap_or(512),
        config_env::optional_string("RERANKER_API_KEY"),
        config_env::optional_string("RERANKER_API_URL"),
        config_env::optional_string("RERANKER_API_MODEL"),
    ))
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

    #[test]
    fn runtime_config_debug_redacts_database_url_and_api_keys() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var(
                "DATABASE_URL",
                "postgres://user:db-secret@example.test/elephant",
            );
            env::set_var("LLM_PROVIDER", "openai");
            env::set_var("LLM_API_KEY", "llm-secret");
            env::set_var("LLM_MODEL", "gpt-4o-mini");
            env::set_var("EMBEDDING_PROVIDER", "openai");
            env::set_var("EMBEDDING_API_KEY", "embed-secret");
            env::set_var("EMBEDDING_API_MODEL", "text-embedding-3-small");
            env::set_var("EMBEDDING_API_DIMS", "1536");
            env::set_var("RERANKER_PROVIDER", "api");
            env::set_var("RERANKER_API_KEY", "reranker-secret");
            env::set_var("RERANKER_API_URL", "https://reranker.example.test");
            env::set_var("RERANKER_API_MODEL", "rerank-v1");
        }

        let config = RuntimeConfig::from_env().unwrap();
        let debug = format!("{config:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("db-secret"));
        assert!(!debug.contains("llm-secret"));
        assert!(!debug.contains("embed-secret"));
        assert!(!debug.contains("reranker-secret"));

        unsafe {
            env::remove_var("DATABASE_URL");
            env::remove_var("LLM_PROVIDER");
            env::remove_var("LLM_API_KEY");
            env::remove_var("LLM_MODEL");
            env::remove_var("EMBEDDING_PROVIDER");
            env::remove_var("EMBEDDING_API_KEY");
            env::remove_var("EMBEDDING_API_MODEL");
            env::remove_var("EMBEDDING_API_DIMS");
            env::remove_var("RERANKER_PROVIDER");
            env::remove_var("RERANKER_API_KEY");
            env::remove_var("RERANKER_API_URL");
            env::remove_var("RERANKER_API_MODEL");
        }
    }
}
