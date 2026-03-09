//! Shared runtime builder for the API server and in-process benchmarks.

use std::env;
use std::sync::Arc;

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
use crate::retain::extractor::LlmFactExtractor;
use crate::retain::graph_builder::{DefaultGraphBuilder, GraphConfig};
use crate::retain::resolver::LayeredEntityResolver;
use crate::retain::{DefaultRetainPipeline, RetainPipeline};
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
}

/// Human-readable runtime model labels.
#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    /// Retain/extraction model label.
    pub retain_model: String,
    /// Reflect/consolidation model label.
    pub reflect_model: String,
    /// Embedding model label.
    pub embedding_model: String,
    /// Reranker model label.
    pub reranker_model: String,
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
}

fn make_llm(config: &LlmConfig) -> Result<Box<dyn LlmClient>> {
    match config.provider.as_str() {
        "openai" => Ok(Box::new(OpenAiClient::new(
            config.api_key.clone(),
            config.model.clone(),
            config.base_url.clone(),
        )?)),
        _ => Ok(Box::new(AnthropicClient::new(
            config.api_key.clone(),
            config.model.clone(),
        )?)),
    }
}

fn env_required(name: &str) -> Result<String> {
    env::var(name).map_err(|e| Error::Internal(format!("{name} must be set: {e}")))
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

/// Build the full Elephant runtime from environment variables.
pub async fn build_runtime_from_env(options: BuildRuntimeOptions) -> Result<ElephantRuntime> {
    let database_url = env_required("DATABASE_URL")?;
    let llm_provider = env_required("LLM_PROVIDER")?;
    let llm_api_key = env_required("LLM_API_KEY")?;
    let retain_model = env::var("RETAIN_LLM_MODEL").or_else(|_| env_required("LLM_MODEL"))?;
    let reflect_model = env::var("REFLECT_LLM_MODEL").or_else(|_| env_required("LLM_MODEL"))?;
    let llm_base_url = env::var("LLM_BASE_URL").ok();

    let pool = sqlx::PgPool::connect(&database_url).await?;
    let store = Arc::new(PgMemoryStore::new(pool.clone()));
    store.migrate().await?;

    let retain_config = LlmConfig {
        provider: llm_provider.clone(),
        api_key: llm_api_key.clone(),
        model: retain_model.clone(),
        base_url: llm_base_url.clone(),
    };
    let reflect_config = LlmConfig {
        provider: llm_provider.clone(),
        api_key: llm_api_key,
        model: reflect_model.clone(),
        base_url: llm_base_url,
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
            GraphConfig::default(),
        )),
        Box::new(PgMemoryStore::new(pool.clone())),
        embedding::build_client(&emb_config)?,
        stage_llm(
            &retain_config,
            LlmStage::RetainOpinion,
            options.metrics.as_ref(),
        )?,
        ChunkConfig {
            max_tokens: 512,
            overlap_tokens: 64,
            preserve_turns: true,
        },
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
        60.0,
        rerank_top_n,
    ));

    let reflect_max_iter: usize = env::var("REFLECT_MAX_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let reflect = Arc::new(DefaultReflectPipeline::new(
        recall.clone(),
        stage_llm(&reflect_config, LlmStage::Reflect, options.metrics.as_ref())?,
        store.clone(),
        reflect_max_iter,
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
