//! Typed configuration for shared runtime startup.

mod loaders;
mod validation;

use std::{env, fmt};

use crate::consolidation::{ConsolidationConfig, opinion_merger};
use crate::embedding::EmbeddingConfig;
#[cfg(test)]
use crate::llm::{ClientConfig, OpenAiConfig};
use crate::llm::{LlmConfig, ReasoningEffortConfig};
use crate::recall::reranker::RerankerConfig;
use crate::retain::{extractor, graph_builder};

use super::env as config_env;
use super::error::{ConfigError, ConfigErrorKind, Result};
use loaders::{
    embedding_config_from_env, parse_optional_positive_usize, parse_optional_reasoning_effort,
    parse_optional_temperature, reranker_config_from_env, runtime_llm_config_from_env,
};
use validation::{
    validate_embedding_config, validate_nonblank_field, validate_nonnegative_float,
    validate_optional_nonnegative_float, validate_optional_unit_interval_float,
    validate_positive_usize_field, validate_reranker_config, validate_unit_interval_float,
};

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
    /// Create runtime configuration from validated typed inputs using default stage settings.
    pub fn new(
        database_url: impl Into<String>,
        llm: LlmConfig,
        embedding: EmbeddingConfig,
        reranker: RerankerConfig,
    ) -> Result<Self> {
        let config = Self {
            database_url: database_url.into(),
            llm,
            embedding,
            reranker,
            dedup_threshold: Some(0.95),
            reasoning_effort: ReasoningEffortConfig::default(),
            extraction: extractor::ExtractionConfig::default(),
            extract_temperature_override: None,
            resolve_temperature: crate::retain::resolver::ENTITY_RESOLUTION_TEMPERATURE,
            resolve_temperature_override: None,
            graph: graph_builder::GraphConfig::default(),
            graph_temperature_override: None,
            consolidation: ConsolidationConfig::default(),
            consolidate_temperature_override: None,
            opinion_merge: opinion_merger::OpinionMergeConfig::default(),
            opinion_merge_temperature_override: None,
            reflect: ReflectConfig::default(),
            reflect_temperature: crate::reflect::REFLECT_TEMPERATURE,
            reflect_temperature_override: None,
            retrieval: RetrievalConfig::default(),
        };
        config.validate()?;
        Ok(config)
    }

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
            structured_output_max_attempts: parse_optional_positive_usize(
                "RETAIN_EXTRACT_STRUCTURED_OUTPUT_MAX_ATTEMPTS",
            )?
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
            batch_size: parse_optional_positive_usize("CONSOLIDATION_BATCH_SIZE")?
                .unwrap_or(ConsolidationConfig::default().batch_size),
            max_tokens: parse_optional_positive_usize("CONSOLIDATION_MAX_TOKENS")?
                .unwrap_or(ConsolidationConfig::default().max_tokens),
            recall_budget: parse_optional_positive_usize("CONSOLIDATION_RECALL_BUDGET")?
                .unwrap_or(ConsolidationConfig::default().recall_budget),
            structured_output_max_attempts: parse_optional_positive_usize(
                "CONSOLIDATION_STRUCTURED_OUTPUT_MAX_ATTEMPTS",
            )?
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
        let retrieval = RetrievalConfig::from_env()?;

        Self::new(database_url, llm, embedding, reranker)?
            .with_dedup_threshold(dedup_threshold)?
            .with_reasoning_effort(reasoning_effort)?
            .with_extraction(extraction)?
            .with_extract_temperature_override(extract_temperature_override)?
            .with_resolve_temperature(resolve_temperature)?
            .with_resolve_temperature_override(resolve_temperature_override)?
            .with_graph(graph)?
            .with_graph_temperature_override(graph_temperature_override)?
            .with_consolidation(consolidation)?
            .with_consolidate_temperature_override(consolidate_temperature_override)?
            .with_opinion_merge(opinion_merge)?
            .with_opinion_merge_temperature_override(opinion_merge_temperature_override)?
            .with_reflect(reflect)?
            .with_reflect_temperature(reflect_temperature)?
            .with_reflect_temperature_override(reflect_temperature_override)?
            .with_retrieval(retrieval)
    }

    /// Override the near-duplicate similarity threshold in the range `0.0..=1.0`.
    pub fn with_dedup_threshold(mut self, dedup_threshold: Option<f32>) -> Result<Self> {
        self.dedup_threshold = dedup_threshold;
        self.validate()?;
        Ok(self)
    }

    /// Override per-stage reasoning-effort settings.
    pub fn with_reasoning_effort(
        mut self,
        reasoning_effort: ReasoningEffortConfig,
    ) -> Result<Self> {
        self.reasoning_effort = reasoning_effort;
        self.validate()?;
        Ok(self)
    }

    /// Override retain extraction settings.
    pub fn with_extraction(mut self, extraction: extractor::ExtractionConfig) -> Result<Self> {
        self.extraction = extraction;
        self.validate()?;
        Ok(self)
    }

    /// Set the explicit retain extraction temperature override, if any.
    pub fn with_extract_temperature_override(
        mut self,
        extract_temperature_override: Option<f32>,
    ) -> Result<Self> {
        self.extract_temperature_override = extract_temperature_override;
        self.validate()?;
        Ok(self)
    }

    /// Override the retain entity-resolution temperature.
    pub fn with_resolve_temperature(mut self, resolve_temperature: f32) -> Result<Self> {
        self.resolve_temperature = resolve_temperature;
        self.validate()?;
        Ok(self)
    }

    /// Set the explicit retain entity-resolution temperature override, if any.
    pub fn with_resolve_temperature_override(
        mut self,
        resolve_temperature_override: Option<f32>,
    ) -> Result<Self> {
        self.resolve_temperature_override = resolve_temperature_override;
        self.validate()?;
        Ok(self)
    }

    /// Override retain graph-construction settings.
    pub fn with_graph(mut self, graph: graph_builder::GraphConfig) -> Result<Self> {
        self.graph = graph;
        self.validate()?;
        Ok(self)
    }

    /// Set the explicit retain graph temperature override, if any.
    pub fn with_graph_temperature_override(
        mut self,
        graph_temperature_override: Option<f32>,
    ) -> Result<Self> {
        self.graph_temperature_override = graph_temperature_override;
        self.validate()?;
        Ok(self)
    }

    /// Override observation-consolidation settings.
    pub fn with_consolidation(mut self, consolidation: ConsolidationConfig) -> Result<Self> {
        self.consolidation = consolidation;
        self.validate()?;
        Ok(self)
    }

    /// Set the explicit consolidation temperature override, if any.
    pub fn with_consolidate_temperature_override(
        mut self,
        consolidate_temperature_override: Option<f32>,
    ) -> Result<Self> {
        self.consolidate_temperature_override = consolidate_temperature_override;
        self.validate()?;
        Ok(self)
    }

    /// Override opinion-merge settings.
    pub fn with_opinion_merge(
        mut self,
        opinion_merge: opinion_merger::OpinionMergeConfig,
    ) -> Result<Self> {
        self.opinion_merge = opinion_merge;
        self.validate()?;
        Ok(self)
    }

    /// Set the explicit opinion-merge temperature override, if any.
    pub fn with_opinion_merge_temperature_override(
        mut self,
        opinion_merge_temperature_override: Option<f32>,
    ) -> Result<Self> {
        self.opinion_merge_temperature_override = opinion_merge_temperature_override;
        self.validate()?;
        Ok(self)
    }

    /// Override reflect-stage settings.
    pub fn with_reflect(mut self, reflect: ReflectConfig) -> Result<Self> {
        self.reflect = reflect;
        self.validate()?;
        Ok(self)
    }

    /// Override the reflect temperature.
    pub fn with_reflect_temperature(mut self, reflect_temperature: f32) -> Result<Self> {
        self.reflect_temperature = reflect_temperature;
        self.validate()?;
        Ok(self)
    }

    /// Set the explicit reflect temperature override, if any.
    pub fn with_reflect_temperature_override(
        mut self,
        reflect_temperature_override: Option<f32>,
    ) -> Result<Self> {
        self.reflect_temperature_override = reflect_temperature_override;
        self.validate()?;
        Ok(self)
    }

    /// Override retrieval-stage settings.
    pub fn with_retrieval(mut self, retrieval: RetrievalConfig) -> Result<Self> {
        self.retrieval = retrieval;
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<()> {
        validate_nonblank_field("database_url", &self.database_url)?;
        validate_embedding_config(&self.embedding)?;
        validate_reranker_config(&self.reranker)?;
        validate_optional_unit_interval_float("dedup_threshold", self.dedup_threshold)?;
        validate_positive_usize_field(
            "extraction.structured_output_max_attempts",
            self.extraction.structured_output_max_attempts,
        )?;
        validate_nonnegative_float("extraction.temperature", self.extraction.temperature)?;
        validate_optional_nonnegative_float(
            "extract_temperature_override",
            self.extract_temperature_override,
        )?;
        validate_nonnegative_float("resolve_temperature", self.resolve_temperature)?;
        validate_optional_nonnegative_float(
            "resolve_temperature_override",
            self.resolve_temperature_override,
        )?;
        validate_unit_interval_float("graph.semantic_threshold", self.graph.semantic_threshold)?;
        validate_nonnegative_float("graph.causal_temperature", self.graph.causal_temperature)?;
        validate_optional_nonnegative_float(
            "graph_temperature_override",
            self.graph_temperature_override,
        )?;
        validate_positive_usize_field("consolidation.batch_size", self.consolidation.batch_size)?;
        validate_positive_usize_field("consolidation.max_tokens", self.consolidation.max_tokens)?;
        validate_positive_usize_field(
            "consolidation.recall_budget",
            self.consolidation.recall_budget,
        )?;
        validate_positive_usize_field(
            "consolidation.structured_output_max_attempts",
            self.consolidation.structured_output_max_attempts,
        )?;
        validate_nonnegative_float("consolidation.temperature", self.consolidation.temperature)?;
        validate_optional_nonnegative_float(
            "consolidate_temperature_override",
            self.consolidate_temperature_override,
        )?;
        validate_nonnegative_float("opinion_merge.temperature", self.opinion_merge.temperature)?;
        validate_optional_nonnegative_float(
            "opinion_merge_temperature_override",
            self.opinion_merge_temperature_override,
        )?;
        validate_nonnegative_float("reflect_temperature", self.reflect_temperature)?;
        validate_optional_nonnegative_float(
            "reflect_temperature_override",
            self.reflect_temperature_override,
        )?;
        Ok(())
    }

    /// Return the configured database URL.
    pub(crate) fn database_url(&self) -> &str {
        &self.database_url
    }

    /// Return the validated LLM configuration bundle.
    pub(crate) fn llm(&self) -> &LlmConfig {
        &self.llm
    }

    /// Return the validated embedding provider configuration.
    pub(crate) fn embedding(&self) -> &EmbeddingConfig {
        &self.embedding
    }

    /// Return the validated reranker provider configuration.
    pub(crate) fn reranker(&self) -> &RerankerConfig {
        &self.reranker
    }

    /// Return the configured near-duplicate threshold.
    pub(crate) fn dedup_threshold(&self) -> Option<f32> {
        self.dedup_threshold
    }

    /// Return the reasoning-effort overrides.
    pub(crate) fn reasoning_effort(&self) -> ReasoningEffortConfig {
        self.reasoning_effort
    }

    /// Return the retain extraction config.
    pub(crate) fn extraction(&self) -> extractor::ExtractionConfig {
        self.extraction
    }

    /// Return the explicit retain extraction temperature override, if any.
    pub(crate) fn extract_temperature_override(&self) -> Option<f32> {
        self.extract_temperature_override
    }

    /// Return the retain entity-resolution temperature.
    pub(crate) fn resolve_temperature(&self) -> f32 {
        self.resolve_temperature
    }

    /// Return the explicit retain entity-resolution temperature override, if any.
    pub(crate) fn resolve_temperature_override(&self) -> Option<f32> {
        self.resolve_temperature_override
    }

    /// Return the retain graph-building config.
    pub(crate) fn graph(&self) -> graph_builder::GraphConfig {
        self.graph.clone()
    }

    /// Return the explicit retain graph temperature override, if any.
    pub(crate) fn graph_temperature_override(&self) -> Option<f32> {
        self.graph_temperature_override
    }

    /// Return the observation consolidation config.
    pub(crate) fn consolidation(&self) -> ConsolidationConfig {
        self.consolidation
    }

    /// Return the explicit consolidation temperature override, if any.
    pub(crate) fn consolidate_temperature_override(&self) -> Option<f32> {
        self.consolidate_temperature_override
    }

    /// Return the opinion-merge config.
    pub(crate) fn opinion_merge(&self) -> opinion_merger::OpinionMergeConfig {
        self.opinion_merge
    }

    /// Return the explicit opinion-merge temperature override, if any.
    pub(crate) fn opinion_merge_temperature_override(&self) -> Option<f32> {
        self.opinion_merge_temperature_override
    }

    /// Return the reflect-stage runtime config.
    pub(crate) fn reflect(&self) -> &ReflectConfig {
        &self.reflect
    }

    /// Return the reflect temperature.
    pub(crate) fn reflect_temperature(&self) -> f32 {
        self.reflect_temperature
    }

    /// Return the explicit reflect temperature override, if any.
    pub(crate) fn reflect_temperature_override(&self) -> Option<f32> {
        self.reflect_temperature_override
    }

    /// Return the retrieval-stage runtime config.
    pub(crate) fn retrieval(&self) -> &RetrievalConfig {
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
    /// Create reflect-stage configuration from validated values.
    pub fn new(
        max_iterations: usize,
        max_tokens: Option<usize>,
        source_limit: usize,
        source_max_chars: Option<usize>,
        enable_source_lookup: bool,
    ) -> Result<Self> {
        if max_iterations == 0 {
            return Err(ConfigError::configuration(
                "REFLECT_MAX_ITERATIONS must be greater than 0",
            ));
        }
        if matches!(max_tokens, Some(0)) {
            return Err(ConfigError::configuration(
                "REFLECT_MAX_TOKENS must be greater than 0 if set",
            ));
        }
        if source_limit == 0 {
            return Err(ConfigError::configuration(
                "REFLECT_SOURCE_LIMIT must be greater than 0",
            ));
        }
        if matches!(source_max_chars, Some(0)) {
            return Err(ConfigError::configuration(
                "REFLECT_SOURCE_MAX_CHARS must be greater than 0 if set",
            ));
        }

        Ok(Self {
            max_iterations,
            max_tokens,
            source_limit,
            source_max_chars,
            enable_source_lookup,
        })
    }

    /// Load reflect-stage configuration from the process environment.
    pub fn from_env() -> Result<Self> {
        let max_iterations = config_env::parse_optional_usize(
            "REFLECT_MAX_ITERATIONS",
            ConfigErrorKind::Configuration,
        )?
        .unwrap_or(8);
        let max_tokens =
            config_env::parse_optional_usize("REFLECT_MAX_TOKENS", ConfigErrorKind::Configuration)?;
        let source_limit = config_env::parse_optional_usize(
            "REFLECT_SOURCE_LIMIT",
            ConfigErrorKind::Configuration,
        )?
        .unwrap_or(crate::reflect::DEFAULT_SOURCE_LOOKUP_LIMIT);
        let source_max_chars = config_env::parse_optional_usize(
            "REFLECT_SOURCE_MAX_CHARS",
            ConfigErrorKind::Configuration,
        )?;
        let enable_source_lookup = config_env::parse_optional_bool(
            "REFLECT_ENABLE_SOURCE_LOOKUP",
            ConfigErrorKind::Configuration,
        )?
        .unwrap_or(crate::reflect::DEFAULT_ENABLE_SOURCE_LOOKUP);

        Self::new(
            max_iterations,
            max_tokens,
            source_limit,
            source_max_chars,
            enable_source_lookup,
        )
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

impl Default for ReflectConfig {
    fn default() -> Self {
        Self {
            max_iterations: 8,
            max_tokens: None,
            source_limit: crate::reflect::DEFAULT_SOURCE_LOOKUP_LIMIT,
            source_max_chars: None,
            enable_source_lookup: crate::reflect::DEFAULT_ENABLE_SOURCE_LOOKUP,
        }
    }
}

/// Validated retrieval-stage configuration loaded from environment.
#[derive(Debug, Clone, Copy)]
pub struct RetrievalConfig {
    retriever_limit: usize,
    max_facts: usize,
}

impl RetrievalConfig {
    /// Create retrieval-stage configuration from validated values.
    pub fn new(retriever_limit: usize, max_facts: usize) -> Result<Self> {
        if retriever_limit == 0 {
            return Err(ConfigError::configuration(
                "RETRIEVER_LIMIT must be greater than 0",
            ));
        }
        if max_facts == 0 {
            return Err(ConfigError::configuration(
                "MAX_FACTS must be greater than 0",
            ));
        }

        Ok(Self {
            retriever_limit,
            max_facts,
        })
    }

    /// Load retrieval-stage configuration from the process environment.
    pub fn from_env() -> Result<Self> {
        let retriever_limit =
            config_env::parse_optional_usize("RETRIEVER_LIMIT", ConfigErrorKind::Configuration)?
                .unwrap_or(40);
        let max_facts =
            config_env::parse_optional_usize("MAX_FACTS", ConfigErrorKind::Configuration)?
                .unwrap_or(50);
        Self::new(retriever_limit, max_facts)
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

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            retriever_limit: 40,
            max_facts: 50,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingProvider;
    use crate::recall::reranker::RerankerProvider;
    use std::env;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn set_minimal_runtime_env() {
        unsafe {
            env::set_var("DATABASE_URL", "postgres://example");
            env::set_var("LLM_PROVIDER", "openai");
            env::set_var("LLM_API_KEY", "test-key");
            env::set_var("LLM_MODEL", "gpt-4o-mini");
            env::set_var("EMBEDDING_PROVIDER", "local");
            env::set_var("EMBEDDING_MODEL_PATH", "/tmp/model");
            env::set_var("RERANKER_PROVIDER", "none");
        }
    }

    fn clear_minimal_runtime_env() {
        unsafe {
            env::remove_var("DATABASE_URL");
            env::remove_var("LLM_PROVIDER");
            env::remove_var("LLM_API_KEY");
            env::remove_var("LLM_MODEL");
            env::remove_var("EMBEDDING_PROVIDER");
            env::remove_var("EMBEDDING_MODEL_PATH");
            env::remove_var("RERANKER_PROVIDER");
        }
    }

    fn minimal_programmatic_runtime_config() -> RuntimeConfig {
        let retain = ClientConfig::OpenAi(OpenAiConfig::new("test-key", "gpt-4o-mini").unwrap());
        let reflect = ClientConfig::OpenAi(OpenAiConfig::new("test-key", "gpt-4o-mini").unwrap());
        RuntimeConfig::new(
            "postgres://example",
            LlmConfig::new(retain, reflect),
            EmbeddingConfig::local("/tmp/model"),
            RerankerConfig::none(),
        )
        .unwrap()
    }

    #[test]
    fn runtime_config_new_uses_defaults() {
        let config = minimal_programmatic_runtime_config();

        assert_eq!(config.database_url(), "postgres://example");
        assert_eq!(config.retrieval().retriever_limit(), 40);
        assert_eq!(config.retrieval().max_facts(), 50);
        assert_eq!(config.reflect().max_iterations(), 8);
        assert_eq!(
            config.resolve_temperature(),
            crate::retain::resolver::ENTITY_RESOLUTION_TEMPERATURE
        );
    }

    #[test]
    fn runtime_config_new_rejects_blank_database_url() {
        let retain = ClientConfig::OpenAi(OpenAiConfig::new("test-key", "gpt-4o-mini").unwrap());
        let reflect = ClientConfig::OpenAi(OpenAiConfig::new("test-key", "gpt-4o-mini").unwrap());

        let err = RuntimeConfig::new(
            "   ",
            LlmConfig::new(retain, reflect),
            EmbeddingConfig::local("/tmp/model"),
            RerankerConfig::none(),
        )
        .unwrap_err();

        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("database_url"));
    }

    #[test]
    fn runtime_config_programmatic_path_rejects_invalid_embedding_config() {
        let retain = ClientConfig::OpenAi(OpenAiConfig::new("test-key", "gpt-4o-mini").unwrap());
        let reflect = ClientConfig::OpenAi(OpenAiConfig::new("test-key", "gpt-4o-mini").unwrap());

        let err = RuntimeConfig::new(
            "postgres://example",
            LlmConfig::new(retain, reflect),
            EmbeddingConfig::local(""),
            RerankerConfig::none(),
        )
        .unwrap_err();

        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("embedding.model_path"));
    }

    #[test]
    fn runtime_config_programmatic_path_rejects_invalid_stage_override() {
        let err = minimal_programmatic_runtime_config()
            .with_consolidation(ConsolidationConfig {
                batch_size: 0,
                ..ConsolidationConfig::default()
            })
            .unwrap_err();

        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("consolidation.batch_size"));
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
    fn reflect_config_rejects_zero_source_limit() {
        let err = ReflectConfig::new(8, None, 0, None, true).unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("REFLECT_SOURCE_LIMIT"));
    }

    #[test]
    fn retrieval_config_rejects_invalid_values() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("RETRIEVER_LIMIT", "not-a-number");
            env::set_var("MAX_FACTS", "still-not-a-number");
        }

        let err = RetrievalConfig::from_env().unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("RETRIEVER_LIMIT"));

        unsafe {
            env::remove_var("RETRIEVER_LIMIT");
            env::remove_var("MAX_FACTS");
        }
    }

    #[test]
    fn retrieval_config_rejects_zero_limits() {
        let err = RetrievalConfig::new(0, 50).unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("RETRIEVER_LIMIT"));

        let err = RetrievalConfig::new(40, 0).unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("MAX_FACTS"));
    }

    #[test]
    fn runtime_config_rejects_zero_extraction_retry_count() {
        let _guard = env_lock().lock().unwrap();
        set_minimal_runtime_env();
        unsafe {
            env::set_var("RETAIN_EXTRACT_STRUCTURED_OUTPUT_MAX_ATTEMPTS", "0");
        }

        let err = RuntimeConfig::from_env().unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(
            err.to_string()
                .contains("RETAIN_EXTRACT_STRUCTURED_OUTPUT_MAX_ATTEMPTS")
        );

        unsafe {
            env::remove_var("RETAIN_EXTRACT_STRUCTURED_OUTPUT_MAX_ATTEMPTS");
        }
        clear_minimal_runtime_env();
    }

    #[test]
    fn runtime_config_rejects_invalid_dedup_threshold_as_configuration_error() {
        let _guard = env_lock().lock().unwrap();
        set_minimal_runtime_env();
        unsafe {
            env::set_var("DEDUP_THRESHOLD", "nope");
        }

        let err = RuntimeConfig::from_env().unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("DEDUP_THRESHOLD"));

        unsafe {
            env::remove_var("DEDUP_THRESHOLD");
        }
        clear_minimal_runtime_env();
    }

    #[test]
    fn runtime_config_rejects_dedup_threshold_above_one() {
        let _guard = env_lock().lock().unwrap();
        set_minimal_runtime_env();

        let err = RuntimeConfig::from_env()
            .unwrap()
            .with_dedup_threshold(Some(1.1))
            .unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("dedup_threshold"));

        clear_minimal_runtime_env();
    }

    #[test]
    fn runtime_config_rejects_graph_semantic_threshold_above_one() {
        let _guard = env_lock().lock().unwrap();
        set_minimal_runtime_env();

        let config = RuntimeConfig::from_env().unwrap();
        let graph = graph_builder::GraphConfig {
            semantic_threshold: 1.1,
            ..config.graph()
        };
        let err = config.with_graph(graph).unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("graph.semantic_threshold"));

        clear_minimal_runtime_env();
    }

    #[test]
    fn embedding_config_rejects_missing_openai_dimensions() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("EMBEDDING_PROVIDER", "openai");
            env::set_var("EMBEDDING_API_KEY", "embed-key");
            env::set_var("EMBEDDING_API_MODEL", "text-embedding-3-small");
        }

        let err = embedding_config_from_env().unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("EMBEDDING_API_DIMS"));

        unsafe {
            env::remove_var("EMBEDDING_PROVIDER");
            env::remove_var("EMBEDDING_API_KEY");
            env::remove_var("EMBEDDING_API_MODEL");
        }
    }

    #[test]
    fn embedding_config_ignores_inactive_openai_dimensions_for_local_provider() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("EMBEDDING_PROVIDER", "local");
            env::set_var("EMBEDDING_MODEL_PATH", "/tmp/model");
            env::set_var("EMBEDDING_API_DIMS", "not-a-number");
        }

        let config = embedding_config_from_env().unwrap();
        assert_eq!(config.provider(), EmbeddingProvider::Local);

        unsafe {
            env::remove_var("EMBEDDING_PROVIDER");
            env::remove_var("EMBEDDING_MODEL_PATH");
            env::remove_var("EMBEDDING_API_DIMS");
        }
    }

    #[test]
    fn reranker_config_rejects_missing_api_url() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("RERANKER_PROVIDER", "api");
            env::set_var("RERANKER_API_KEY", "rerank-key");
            env::set_var("RERANKER_API_MODEL", "rerank-v1");
        }

        let err = reranker_config_from_env().unwrap_err();
        assert_eq!(
            err.kind(),
            super::super::error::ConfigErrorKind::Configuration
        );
        assert!(err.to_string().contains("RERANKER_API_URL"));

        unsafe {
            env::remove_var("RERANKER_PROVIDER");
            env::remove_var("RERANKER_API_KEY");
            env::remove_var("RERANKER_API_MODEL");
        }
    }

    #[test]
    fn reranker_config_ignores_inactive_local_seq_len_when_disabled() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("RERANKER_PROVIDER", "none");
            env::set_var("RERANKER_MAX_SEQ_LEN", "not-a-number");
        }

        let config = reranker_config_from_env().unwrap();
        assert_eq!(config.provider(), RerankerProvider::None);

        unsafe {
            env::remove_var("RERANKER_PROVIDER");
            env::remove_var("RERANKER_MAX_SEQ_LEN");
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
