//! Shared runtime builder for the API server and in-process benchmarks.

use std::env;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::consolidation::observation;
use crate::consolidation::opinion_merger;
use crate::consolidation::{ConsolidationConfig, DefaultConsolidator, DefaultOpinionMerger};
use crate::embedding::{self, EmbeddingClient, EmbeddingConfig, EmbeddingProvider};
use crate::error::{Error, Result};
use crate::llm::retry::{RetryPolicy, RetryingLlmClient};
use crate::llm::{self, ClientConfig, LlmClient, ReasoningEffort, ReasoningEffortConfig};
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
    /// Maximum number of facts kept per recall call before token budgeting.
    #[serde(alias = "rerank_top_n")]
    pub max_facts: usize,
    /// Reflect tool-loop iteration cap.
    pub reflect_max_iterations: usize,
    /// Reflect completion cap.
    pub reflect_max_tokens: Option<usize>,
    /// Retain extraction reasoning effort override.
    pub retain_extract_reasoning_effort: Option<ReasoningEffort>,
    /// Total attempts for malformed extractor structured output.
    pub retain_extract_structured_output_max_attempts: usize,
    /// Retain entity-resolution reasoning effort override.
    pub retain_resolve_reasoning_effort: Option<ReasoningEffort>,
    /// Retain graph-builder reasoning effort override.
    pub retain_graph_reasoning_effort: Option<ReasoningEffort>,
    /// Whether reflect exposes the source lookup tool.
    #[serde(default = "default_true")]
    pub reflect_enable_source_lookup: bool,
    /// Reflect reasoning effort override.
    pub reflect_reasoning_effort: Option<ReasoningEffort>,
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
    /// Total attempts for malformed consolidator structured output.
    pub consolidation_structured_output_max_attempts: usize,
    /// Consolidation reasoning effort override.
    pub consolidate_reasoning_effort: Option<ReasoningEffort>,
    /// Opinion merge reasoning effort override.
    pub opinion_merge_reasoning_effort: Option<ReasoningEffort>,
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

fn env_required(name: &str) -> Result<String> {
    env::var(name).map_err(|e| Error::Configuration(format!("{name} must be set: {e}")))
}

fn stage_llm(
    config: &ClientConfig,
    stage: LlmStage,
    metrics: Option<&Arc<MetricsCollector>>,
) -> Result<Arc<dyn LlmClient>> {
    let base: Arc<dyn LlmClient> = Arc::from(llm::build_client(config)?);
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
    let llm_configs = llm::runtime_config_from_env()?;

    let max_conns = options.max_pool_connections.unwrap_or(10);
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(max_conns)
        .connect(&database_url)
        .await?;
    let store = Arc::new(PgMemoryStore::new(pool.clone()));
    store.migrate().await?;

    let emb_config = embedding::config_from_env()?;
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
    let reasoning_effort = *ReasoningEffortConfig::current()?;
    let extraction_config = extractor::config_from_env();

    let retain = Arc::new(DefaultRetainPipeline::new(
        Box::new(SimpleChunker),
        Box::new(LlmFactExtractor::new(
            stage_llm(
                llm_configs.retain(),
                LlmStage::RetainExtract,
                options.metrics.as_ref(),
            )?,
            extraction_config,
        )),
        Box::new(LayeredEntityResolver::new(
            embedding::build_client(&emb_config)?,
            stage_llm(
                llm_configs.retain(),
                LlmStage::RetainResolve,
                options.metrics.as_ref(),
            )?,
        )),
        Box::new(DefaultGraphBuilder::new(
            stage_llm(
                llm_configs.retain(),
                LlmStage::RetainGraph,
                options.metrics.as_ref(),
            )?,
            graph_config.clone(),
        )),
        Box::new(PgMemoryStore::new(pool.clone())),
        embedding::build_client(&emb_config)?,
        stage_llm(
            llm_configs.retain(),
            LlmStage::RetainOpinion,
            options.metrics.as_ref(),
        )?,
        chunk_config.clone(),
        dedup_threshold,
    ));

    let reranker_config = reranker::config_from_env()?;
    let reranker = reranker::build_reranker(&reranker_config)?;
    let consolidation_config: ConsolidationConfig = observation::config_from_env();

    let retriever_limit: usize = env::var("RETRIEVER_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let recall_rrf_k = 60.0;
    let max_facts: usize = env::var("MAX_FACTS")
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
        max_facts,
        4096,
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
        stage_llm(
            llm_configs.reflect(),
            LlmStage::Reflect,
            options.metrics.as_ref(),
        )?,
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
            llm_configs.reflect(),
            LlmStage::Consolidate,
            options.metrics.as_ref(),
        )?,
        embeddings.clone(),
        recall.clone(),
        consolidation_config,
    ));
    let opinion_merger = Arc::new(DefaultOpinionMerger::new(
        store.clone(),
        stage_llm(
            llm_configs.reflect(),
            LlmStage::OpinionMerge,
            options.metrics.as_ref(),
        )?,
        embeddings.clone(),
    ));

    Ok(ElephantRuntime {
        info: RuntimeInfo {
            retain_model: format!(
                "{}/{}",
                llm_configs.retain().provider().as_str(),
                llm_configs.retain().model()
            ),
            reflect_model: format!(
                "{}/{}",
                llm_configs.reflect().provider().as_str(),
                llm_configs.reflect().model()
            ),
            embedding_model: embedding_label(&emb_config),
            reranker_model: reranker_label(&reranker_config),
            tuning: RuntimeTuning {
                chunk_max_tokens: chunk_config.max_tokens,
                chunk_overlap_tokens: chunk_config.overlap_tokens,
                dedup_threshold,
                retriever_limit,
                recall_rrf_k,
                max_facts,
                reflect_max_iterations: reflect_max_iter,
                reflect_max_tokens,
                retain_extract_reasoning_effort: reasoning_effort.retain_extract,
                retain_extract_structured_output_max_attempts: extraction_config
                    .structured_output_max_attempts,
                retain_resolve_reasoning_effort: reasoning_effort.retain_resolve,
                retain_graph_reasoning_effort: reasoning_effort.retain_graph,
                reflect_enable_source_lookup,
                reflect_reasoning_effort: reasoning_effort.reflect,
                graph_semantic_threshold: graph_config.semantic_threshold,
                graph_temporal_max_days: graph_config.temporal_max_days,
                graph_enable_causal: graph_config.enable_causal,
                graph_max_causal_checks: graph_builder::MAX_CAUSAL_CHECKS,
                consolidation_batch_size: consolidation_config.batch_size,
                consolidation_max_tokens: consolidation_config.max_tokens,
                consolidation_recall_budget: consolidation_config.recall_budget,
                consolidation_structured_output_max_attempts: consolidation_config
                    .structured_output_max_attempts,
                consolidate_reasoning_effort: reasoning_effort.consolidate,
                opinion_merge_reasoning_effort: reasoning_effort.opinion_merge,
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
