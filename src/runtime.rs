//! Shared runtime builder for the API server and in-process benchmarks.

use std::env;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::consolidation::observation;
use crate::consolidation::opinion_merger;
use crate::consolidation::{DefaultConsolidator, DefaultOpinionMerger};
use crate::embedding::{self, EmbeddingClient, EmbeddingConfig, EmbeddingProvider};
use crate::error::{Error, Result};
use crate::llm::LlmClient;
use crate::llm::anthropic::AnthropicClient;
use crate::llm::openai::OpenAiClient;
use crate::llm::retry::{RetryPolicy, RetryingLlmClient};
use crate::metrics::{LlmStage, MeteredLlmClient, MetricsCollector};
use crate::recall::DefaultRecallPipeline;
use crate::recall::RecallPipeline;
use crate::recall::budget::EstimateTokenizer;
use crate::recall::graph::{GraphRetriever, GraphRetrieverConfig};
use crate::recall::keyword::KeywordRetriever;
use crate::recall::reranker::{self, RerankerConfig, RerankerProvider};
use crate::recall::semantic::SemanticRetriever;
use crate::recall::temporal::TemporalRetriever;
use crate::reflect::{DefaultReflectPipeline, ReflectPipeline};
use crate::retain::chunker::SimpleChunker;
use crate::retain::extractor::{self, LlmFactExtractor};
use crate::retain::graph_builder::{self, DefaultGraphBuilder, GraphConfig};
use crate::retain::resolver::{self, LayeredEntityResolver};
use crate::retain::{self, DefaultRetainPipeline, RetainPipeline};
use crate::storage::MemoryStore;
use crate::storage::pg::PgMemoryStore;
use crate::types::ChunkConfig;

use crate::consolidation::{Consolidator, OpinionMerger};

#[derive(Debug, Clone)]
struct LlmConfig {
    provider: String,
    api_key: String,
    model: String,
    base_url: Option<String>,
    prompt_caching: crate::llm::PromptCachingConfig,
}

/// Stable prompt-template hash bundle for publication provenance.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimePromptHashes {
    /// Fact extraction template hash.
    pub retain_extract: String,
    /// Entity resolution system prompt hash.
    pub retain_resolve_system: String,
    /// Entity resolution user template hash.
    pub retain_resolve_user: String,
    /// Graph causal-link system prompt hash.
    pub retain_graph_system: String,
    /// Graph causal-link user template hash.
    pub retain_graph_user: String,
    /// Opinion reinforcement template hash.
    pub retain_opinion: String,
    /// Reflect agent template hash.
    pub reflect_agent: String,
    /// Consolidation template hash.
    pub consolidate: String,
    /// Opinion merge template hash.
    pub opinion_merge: String,
}

/// Runtime tuning snapshot that affects benchmark behavior.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeTuning {
    /// Text chunk size used during retain.
    pub chunk_max_tokens: usize,
    /// Chunk overlap used during retain.
    pub chunk_overlap_tokens: usize,
    /// Near-duplicate similarity threshold, if enabled.
    pub dedup_threshold: Option<f32>,
    /// Retriever candidate limit per retrieval strategy.
    pub retriever_limit: usize,
    /// RRF fusion constant.
    pub recall_rrf_k: f32,
    /// Top-N passed through reranking.
    pub rerank_top_n: usize,
    /// Reflect tool-loop iteration cap.
    pub reflect_max_iterations: usize,
    /// Reflect completion cap.
    pub reflect_max_tokens: Option<usize>,
    /// Whether reflect exposes the source lookup tool.
    #[serde(default = "default_true")]
    pub reflect_enable_source_lookup: bool,
    /// Graph semantic-link threshold.
    pub graph_semantic_threshold: f32,
    /// Graph temporal-link max distance.
    pub graph_temporal_max_days: i64,
    /// Whether causal links are enabled.
    pub graph_enable_causal: bool,
    /// Max causal link checks per retain batch.
    pub graph_max_causal_checks: usize,
    /// Consolidation batch size.
    pub consolidation_batch_size: usize,
    /// Consolidation completion cap.
    pub consolidation_max_tokens: usize,
    /// Consolidation retrieval budget.
    pub consolidation_recall_budget: usize,
}

/// Human-readable runtime model labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInfo {
    /// Retain/extraction model label.
    pub retain_model: String,
    /// Reflect/consolidation model label.
    pub reflect_model: String,
    /// Embedding model label.
    pub embedding_model: String,
    /// Reranker model label.
    pub reranker_model: String,
    /// Runtime tuning knobs.
    pub tuning: RuntimeTuning,
    /// Prompt template hashes.
    pub prompt_hashes: RuntimePromptHashes,
}

fn default_true() -> bool {
    true
}

/// Fully constructed Elephant runtime.
pub struct ElephantRuntime {
    /// Runtime model labels.
    pub info: RuntimeInfo,
    /// Retain pipeline.
    pub retain: Arc<dyn RetainPipeline>,
    /// Recall pipeline.
    pub recall: Arc<dyn RecallPipeline>,
    /// Reflect pipeline.
    pub reflect: Arc<dyn ReflectPipeline>,
    /// Consolidator.
    pub consolidator: Arc<dyn Consolidator>,
    /// Opinion merger.
    pub opinion_merger: Arc<dyn OpinionMerger>,
    /// Shared storage.
    pub store: Arc<dyn MemoryStore>,
    /// Embedding client.
    pub embeddings: Arc<dyn EmbeddingClient>,
}

/// Build options for runtime construction.
#[derive(Default)]
pub struct BuildRuntimeOptions {
    /// Optional metrics collector for stage-aware metering.
    pub metrics: Option<Arc<MetricsCollector>>,
    /// Maximum Postgres pool connections. Defaults to 10 if not set.
    pub max_pool_connections: Option<u32>,
}

fn make_llm(config: &LlmConfig) -> Result<Box<dyn LlmClient>> {
    match config.provider.as_str() {
        "openai" => Ok(Box::new(OpenAiClient::new(
            config.api_key.clone(),
            config.model.clone(),
            config.base_url.clone(),
        )?)),
        _ => Ok(Box::new(
            AnthropicClient::new(config.api_key.clone(), config.model.clone())?
                .with_prompt_caching(config.prompt_caching.clone()),
        )),
    }
}

fn env_required(name: &str) -> Result<String> {
    env::var(name).map_err(|e| Error::Internal(format!("{name} must be set: {e}")))
}

fn prompt_caching_from_env(var_name: &str) -> Result<crate::llm::PromptCachingConfig> {
    let enabled = match env::var(var_name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => {
                return Err(Error::Internal(format!(
                    "{var_name} must be a boolean, got: {other}"
                )));
            }
        },
        Err(env::VarError::NotPresent) => return Ok(crate::llm::PromptCachingConfig::default()),
        Err(err) => {
            return Err(Error::Internal(format!(
                "{var_name} could not be read: {err}"
            )));
        }
    };

    Ok(crate::llm::PromptCachingConfig { enabled })
}

fn stage_llm(
    config: &LlmConfig,
    stage: LlmStage,
    metrics: Option<&Arc<MetricsCollector>>,
) -> Result<Arc<dyn LlmClient>> {
    let base: Arc<dyn LlmClient> = Arc::from(make_llm(config)?);
    let inner: Arc<dyn LlmClient> = if let Some(collector) = metrics {
        Arc::new(MeteredLlmClient::new(base, collector.clone(), stage))
    } else {
        base
    };
    Ok(Arc::new(RetryingLlmClient::new(
        inner,
        RetryPolicy::default(),
    )))
}

fn embedding_label(config: &EmbeddingConfig) -> String {
    match config.provider {
        EmbeddingProvider::OpenAi => {
            format!("openai/{}", config.model.clone().unwrap_or_default())
        }
        EmbeddingProvider::Local => {
            let name = config
                .model_path
                .as_deref()
                .and_then(|p| std::path::Path::new(p).file_name())
                .and_then(|f| f.to_str())
                .unwrap_or("unknown");
            format!("local/{name}")
        }
    }
}

fn reranker_label(config: &RerankerConfig) -> String {
    match config.provider {
        RerankerProvider::Local => {
            let name = config
                .model_path
                .as_deref()
                .and_then(|p| std::path::Path::new(p).file_name())
                .and_then(|f| f.to_str())
                .unwrap_or("unknown");
            format!("local/{name}")
        }
        RerankerProvider::Api => {
            format!("api/{}", config.api_model.as_deref().unwrap_or("unknown"))
        }
        RerankerProvider::None => "none".into(),
    }
}

fn fnv1a64_hex(data: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in data.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn runtime_prompt_hashes() -> RuntimePromptHashes {
    RuntimePromptHashes {
        retain_extract: fnv1a64_hex(extractor::EXTRACT_PROMPT_TEMPLATE),
        retain_resolve_system: fnv1a64_hex(resolver::ENTITY_RESOLUTION_SYSTEM_PROMPT),
        retain_resolve_user: fnv1a64_hex(resolver::ENTITY_RESOLUTION_USER_PROMPT_TEMPLATE),
        retain_graph_system: fnv1a64_hex(graph_builder::CAUSAL_LINK_SYSTEM_PROMPT),
        retain_graph_user: fnv1a64_hex(graph_builder::CAUSAL_LINK_USER_PROMPT_TEMPLATE),
        retain_opinion: fnv1a64_hex(retain::OPINION_REINFORCEMENT_PROMPT_TEMPLATE),
        reflect_agent: fnv1a64_hex(crate::reflect::REFLECT_AGENT_PROMPT_TEMPLATE),
        consolidate: fnv1a64_hex(observation::CONSOLIDATE_PROMPT),
        opinion_merge: fnv1a64_hex(opinion_merger::MERGE_PROMPT),
    }
}

/// Build the full Elephant runtime from environment variables.
pub async fn build_runtime_from_env(options: BuildRuntimeOptions) -> Result<ElephantRuntime> {
    let database_url = env_required("DATABASE_URL")?;
    let llm_provider = env_required("LLM_PROVIDER")?;
    let llm_api_key = env_required("LLM_API_KEY")?;
    let retain_model = env::var("RETAIN_LLM_MODEL").or_else(|_| env_required("LLM_MODEL"))?;
    let reflect_model = env::var("REFLECT_LLM_MODEL").or_else(|_| env_required("LLM_MODEL"))?;
    let llm_base_url = env::var("LLM_BASE_URL").ok();
    let llm_prompt_caching = prompt_caching_from_env("LLM_PROMPT_CACHING")?;

    let max_conns = options.max_pool_connections.unwrap_or(10);
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(max_conns)
        .connect(&database_url)
        .await?;
    let store = Arc::new(PgMemoryStore::new(pool.clone()));
    store.migrate().await?;

    let retain_config = LlmConfig {
        provider: llm_provider.clone(),
        api_key: llm_api_key.clone(),
        model: retain_model.clone(),
        base_url: llm_base_url.clone(),
        prompt_caching: llm_prompt_caching.clone(),
    };
    let reflect_config = LlmConfig {
        provider: llm_provider.clone(),
        api_key: llm_api_key,
        model: reflect_model.clone(),
        base_url: llm_base_url,
        prompt_caching: llm_prompt_caching.clone(),
    };

    let emb_config = EmbeddingConfig {
        provider: match env_required("EMBEDDING_PROVIDER")?.as_str() {
            "openai" => EmbeddingProvider::OpenAi,
            "local" => EmbeddingProvider::Local,
            other => {
                return Err(Error::Internal(format!(
                    "unknown EMBEDDING_PROVIDER: {other}"
                )));
            }
        },
        model_path: env::var("EMBEDDING_MODEL_PATH").ok(),
        api_key: env::var("EMBEDDING_API_KEY").ok(),
        model: env::var("EMBEDDING_API_MODEL").ok(),
        dimensions: env::var("EMBEDDING_API_DIMS")
            .ok()
            .and_then(|s| s.parse().ok()),
    };
    let embeddings: Arc<dyn EmbeddingClient> = Arc::from(embedding::build_client(&emb_config)?);

    let dedup_threshold: Option<f32> = match env::var("DEDUP_THRESHOLD").as_deref() {
        Ok("none") => None,
        Ok(s) => Some(s.parse().map_err(|_| {
            Error::Internal(format!(
                "DEDUP_THRESHOLD must be a float or 'none', got: {s}"
            ))
        })?),
        Err(_) => Some(0.95),
    };
    let chunk_config = ChunkConfig {
        max_tokens: 512,
        overlap_tokens: 64,
        preserve_turns: true,
    };
    let graph_config = GraphConfig::default();

    let retain = Arc::new(DefaultRetainPipeline::new(
        Box::new(SimpleChunker),
        Box::new(LlmFactExtractor::new(stage_llm(
            &retain_config,
            LlmStage::RetainExtract,
            options.metrics.as_ref(),
        )?)),
        Box::new(LayeredEntityResolver::new(
            embedding::build_client(&emb_config)?,
            stage_llm(
                &retain_config,
                LlmStage::RetainResolve,
                options.metrics.as_ref(),
            )?,
        )),
        Box::new(DefaultGraphBuilder::new(
            stage_llm(
                &retain_config,
                LlmStage::RetainGraph,
                options.metrics.as_ref(),
            )?,
            graph_config.clone(),
        )),
        Box::new(PgMemoryStore::new(pool.clone())),
        embedding::build_client(&emb_config)?,
        stage_llm(
            &retain_config,
            LlmStage::RetainOpinion,
            options.metrics.as_ref(),
        )?,
        chunk_config.clone(),
        dedup_threshold,
    ));

    let reranker_config = RerankerConfig {
        provider: match env_required("RERANKER_PROVIDER")?.as_str() {
            "local" => RerankerProvider::Local,
            "api" => RerankerProvider::Api,
            "none" => RerankerProvider::None,
            other => {
                return Err(Error::Internal(format!(
                    "unknown RERANKER_PROVIDER: {other}"
                )));
            }
        },
        model_path: env::var("RERANKER_MODEL_PATH").ok(),
        max_seq_len: env::var("RERANKER_MAX_SEQ_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512),
        api_key: env::var("RERANKER_API_KEY").ok(),
        api_url: env::var("RERANKER_API_URL").ok(),
        api_model: env::var("RERANKER_API_MODEL").ok(),
    };
    let reranker = reranker::build_reranker(&reranker_config)?;

    let retriever_limit: usize = env::var("RETRIEVER_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let recall_rrf_k = 60.0;
    let rerank_top_n: usize = env::var("RERANK_TOP_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let recall = Arc::new(DefaultRecallPipeline::new(
        Box::new(SemanticRetriever::new(
            store.clone(),
            embeddings.clone(),
            retriever_limit,
        )),
        Box::new(KeywordRetriever::new(store.clone(), retriever_limit)),
        Box::new(GraphRetriever::new(
            store.clone(),
            embeddings.clone(),
            GraphRetrieverConfig::default(),
        )),
        Box::new(TemporalRetriever::new(store.clone())),
        reranker,
        Box::new(EstimateTokenizer),
        recall_rrf_k,
        rerank_top_n,
    ));

    let reflect_max_iter: usize = env::var("REFLECT_MAX_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let reflect_max_tokens = env::var("REFLECT_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok());
    let reflect_source_limit: usize = env::var("REFLECT_SOURCE_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(crate::reflect::DEFAULT_SOURCE_LOOKUP_LIMIT);
    let reflect_source_max_chars = env::var("REFLECT_SOURCE_MAX_CHARS")
        .ok()
        .and_then(|s| s.parse().ok());
    let reflect_enable_source_lookup = match env::var("REFLECT_ENABLE_SOURCE_LOOKUP") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => {
                return Err(Error::Internal(format!(
                    "REFLECT_ENABLE_SOURCE_LOOKUP must be a boolean, got: {other}"
                )));
            }
        },
        Err(_) => crate::reflect::DEFAULT_ENABLE_SOURCE_LOOKUP,
    };
    let reflect = Arc::new(DefaultReflectPipeline::new_with_limits(
        recall.clone(),
        stage_llm(&reflect_config, LlmStage::Reflect, options.metrics.as_ref())?,
        store.clone(),
        reflect_max_iter,
        reflect_max_tokens,
        reflect_source_limit,
        reflect_source_max_chars,
        reflect_enable_source_lookup,
    ));

    let consolidator = Arc::new(DefaultConsolidator::new(
        store.clone(),
        stage_llm(
            &reflect_config,
            LlmStage::Consolidate,
            options.metrics.as_ref(),
        )?,
        embeddings.clone(),
        recall.clone(),
    ));
    let opinion_merger = Arc::new(DefaultOpinionMerger::new(
        store.clone(),
        stage_llm(
            &reflect_config,
            LlmStage::OpinionMerge,
            options.metrics.as_ref(),
        )?,
        embeddings.clone(),
    ));

    Ok(ElephantRuntime {
        info: RuntimeInfo {
            retain_model: format!("{llm_provider}/{retain_model}"),
            reflect_model: format!("{llm_provider}/{reflect_model}"),
            embedding_model: embedding_label(&emb_config),
            reranker_model: reranker_label(&reranker_config),
            tuning: RuntimeTuning {
                chunk_max_tokens: chunk_config.max_tokens,
                chunk_overlap_tokens: chunk_config.overlap_tokens,
                dedup_threshold,
                retriever_limit,
                recall_rrf_k,
                rerank_top_n,
                reflect_max_iterations: reflect_max_iter,
                reflect_max_tokens,
                reflect_enable_source_lookup,
                graph_semantic_threshold: graph_config.semantic_threshold,
                graph_temporal_max_days: graph_config.temporal_max_days,
                graph_enable_causal: graph_config.enable_causal,
                graph_max_causal_checks: graph_builder::MAX_CAUSAL_CHECKS,
                consolidation_batch_size: observation::batch_size(),
                consolidation_max_tokens: observation::max_tokens(),
                consolidation_recall_budget: observation::recall_budget(),
            },
            prompt_hashes: runtime_prompt_hashes(),
        },
        retain,
        recall,
        reflect,
        consolidator,
        opinion_merger,
        store,
        embeddings,
    })
}

#[cfg(test)]
mod tests {
    use super::prompt_caching_from_env;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn clear_env(var_name: &str) {
        unsafe {
            std::env::remove_var(var_name);
        }
    }

    fn set_env(var_name: &str, value: &str) {
        unsafe {
            std::env::set_var(var_name, value);
        }
    }

    #[test]
    fn runtime_prompt_caching_from_env_defaults_disabled() {
        let _guard = env_lock().lock().unwrap();
        clear_env("LLM_PROMPT_CACHING");

        let config = prompt_caching_from_env("LLM_PROMPT_CACHING").unwrap();

        assert_eq!(config, crate::llm::PromptCachingConfig::default());
    }

    #[test]
    fn runtime_prompt_caching_from_env_accepts_truthy_values() {
        let _guard = env_lock().lock().unwrap();
        for value in ["1", "true", "yes", "on"] {
            set_env("LLM_PROMPT_CACHING", value);

            let config = prompt_caching_from_env("LLM_PROMPT_CACHING").unwrap();

            assert!(config.enabled, "expected {value} to enable prompt caching");
        }
        clear_env("LLM_PROMPT_CACHING");
    }

    #[test]
    fn runtime_prompt_caching_from_env_rejects_invalid_value() {
        let _guard = env_lock().lock().unwrap();
        set_env("LLM_PROMPT_CACHING", "maybe");

        let err = prompt_caching_from_env("LLM_PROMPT_CACHING").unwrap_err();

        assert!(matches!(err, crate::error::Error::Internal(_)));
        clear_env("LLM_PROMPT_CACHING");
    }
}
