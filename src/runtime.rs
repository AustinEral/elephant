//! Shared runtime builder for the API server and in-process benchmarks.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::config::RuntimeConfig;
use crate::consolidation::observation;
use crate::consolidation::opinion_merger;
use crate::consolidation::{DefaultConsolidator, DefaultOpinionMerger};
use crate::embedding::{self, EmbeddingClient, EmbeddingConfig, EmbeddingProvider};
use crate::error::{Error, Result};
use crate::llm::retry::{RetryPolicy, RetryingLlmClient};
use crate::llm::{
    self, ClientConfig, DeterminismAssessment, DeterminismRequirement, LlmClient, ReasoningEffort,
};
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
    /// Requested retain extraction temperature.
    pub retain_extract_temperature: f32,
    /// Effective retain extraction temperature forwarded to the provider, if any.
    pub retain_extract_effective_temperature: Option<f32>,
    /// Total attempts for malformed extractor structured output.
    pub retain_extract_structured_output_max_attempts: usize,
    /// Retain entity-resolution reasoning effort override.
    pub retain_resolve_reasoning_effort: Option<ReasoningEffort>,
    /// Requested retain entity-resolution temperature.
    pub retain_resolve_temperature: f32,
    /// Effective retain entity-resolution temperature forwarded to the provider, if any.
    pub retain_resolve_effective_temperature: Option<f32>,
    /// Retain graph-builder reasoning effort override.
    pub retain_graph_reasoning_effort: Option<ReasoningEffort>,
    /// Requested retain graph-builder temperature.
    pub retain_graph_temperature: f32,
    /// Effective retain graph-builder temperature forwarded to the provider, if any.
    pub retain_graph_effective_temperature: Option<f32>,
    /// Whether reflect exposes the source lookup tool.
    #[serde(default = "default_true")]
    pub reflect_enable_source_lookup: bool,
    /// Reflect reasoning effort override.
    pub reflect_reasoning_effort: Option<ReasoningEffort>,
    /// Requested reflect temperature.
    pub reflect_temperature: f32,
    /// Effective reflect temperature forwarded to the provider, if any.
    pub reflect_effective_temperature: Option<f32>,
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
    /// Requested consolidation temperature.
    pub consolidate_temperature: f32,
    /// Effective consolidation temperature forwarded to the provider, if any.
    pub consolidate_effective_temperature: Option<f32>,
    /// Consolidation reasoning effort override.
    pub consolidate_reasoning_effort: Option<ReasoningEffort>,
    /// Requested opinion-merge temperature.
    pub opinion_merge_temperature: f32,
    /// Effective opinion-merge temperature forwarded to the provider, if any.
    pub opinion_merge_effective_temperature: Option<f32>,
    /// Opinion merge reasoning effort override.
    pub opinion_merge_reasoning_effort: Option<ReasoningEffort>,
    /// Benchmark-facing determinism assessment for each active stage.
    #[serde(default)]
    pub determinism: RuntimeDeterminism,
}

/// Per-stage determinism assessment for benchmark provenance.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct RuntimeDeterminism {
    /// Determinism assessment for retain extraction.
    pub retain_extract: DeterminismAssessment,
    /// Determinism assessment for retain entity resolution.
    pub retain_resolve: DeterminismAssessment,
    /// Determinism assessment for retain graph building.
    pub retain_graph: DeterminismAssessment,
    /// Determinism assessment for reflect.
    pub reflect: DeterminismAssessment,
    /// Determinism assessment for consolidation.
    pub consolidate: DeterminismAssessment,
    /// Determinism assessment for opinion merge.
    pub opinion_merge: DeterminismAssessment,
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

impl fmt::Debug for ElephantRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElephantRuntime")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

/// Builder for a fully constructed Elephant runtime.
pub struct RuntimeBuilder {
    config: RuntimeConfig,
    metrics: Option<Arc<MetricsCollector>>,
    max_pool_connections: Option<u32>,
    determinism_requirement: Option<DeterminismRequirement>,
}

impl fmt::Debug for RuntimeBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeBuilder")
            .field("config", &self.config)
            .field("metrics_installed", &self.metrics.is_some())
            .field("max_pool_connections", &self.max_pool_connections)
            .field("determinism_requirement", &self.determinism_requirement)
            .finish()
    }
}

impl RuntimeBuilder {
    /// Create a new runtime builder from validated config.
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            metrics: None,
            max_pool_connections: None,
            determinism_requirement: None,
        }
    }

    /// Install a stage-aware metrics collector.
    pub fn metrics(mut self, metrics: Arc<MetricsCollector>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Override the maximum Postgres pool connection count.
    pub fn max_pool_connections(mut self, max_pool_connections: u32) -> Self {
        self.max_pool_connections = Some(max_pool_connections);
        self
    }

    /// Require a minimum determinism level for benchmark-sensitive stages.
    pub fn determinism_requirement(mut self, requirement: DeterminismRequirement) -> Self {
        self.determinism_requirement = Some(requirement);
        self
    }

    /// Build the full runtime.
    pub async fn build(self) -> Result<ElephantRuntime> {
        let runtime_config = self.config;
        let llm_configs = runtime_config.llm();
        let emb_config = runtime_config.embedding();
        let reranker_config = runtime_config.reranker();
        let reasoning_effort = runtime_config.reasoning_effort();
        let extraction_config = runtime_config.extraction();
        let extract_temperature_override = runtime_config.extract_temperature_override();
        let resolve_temperature = runtime_config.resolve_temperature();
        let resolve_temperature_override = runtime_config.resolve_temperature_override();
        let graph_temperature_override = runtime_config.graph_temperature_override();
        let consolidation_config = runtime_config.consolidation();
        let consolidate_temperature_override = runtime_config.consolidate_temperature_override();
        let opinion_merge_config = runtime_config.opinion_merge();
        let opinion_merge_temperature_override =
            runtime_config.opinion_merge_temperature_override();
        let reflect_config = runtime_config.reflect();
        let reflect_temperature = runtime_config.reflect_temperature();
        let reflect_temperature_override = runtime_config.reflect_temperature_override();
        let retrieval_config = runtime_config.retrieval();
        let dedup_threshold = runtime_config.dedup_threshold();
        let chunk_config = ChunkConfig {
            max_tokens: 512,
            overlap_tokens: 64,
            preserve_turns: true,
        };
        let mut graph_config = GraphConfig::default();
        graph_config.causal_temperature = runtime_config.graph_causal_temperature();
        let extract_temperature_resolution = llm::resolve_temperature(
            llm_configs.retain(),
            Some(extraction_config.temperature),
            reasoning_effort.retain_extract,
        );
        let extract_determinism = llm::assess_determinism(
            llm_configs.retain(),
            Some(extraction_config.temperature),
            reasoning_effort.retain_extract,
        );
        validate_explicit_temperature_override(
            "retain extraction",
            "RETAIN_EXTRACT_TEMPERATURE",
            extract_temperature_override,
            &extract_temperature_resolution,
        )?;
        let resolve_temperature_resolution = llm::resolve_temperature(
            llm_configs.retain(),
            Some(resolve_temperature),
            reasoning_effort.retain_resolve,
        );
        let resolve_determinism = llm::assess_determinism(
            llm_configs.retain(),
            Some(resolve_temperature),
            reasoning_effort.retain_resolve,
        );
        validate_explicit_temperature_override(
            "retain entity resolution",
            "RETAIN_RESOLVE_TEMPERATURE",
            resolve_temperature_override,
            &resolve_temperature_resolution,
        )?;
        let graph_temperature_resolution = llm::resolve_temperature(
            llm_configs.retain(),
            Some(graph_config.causal_temperature),
            reasoning_effort.retain_graph,
        );
        let graph_determinism = llm::assess_determinism(
            llm_configs.retain(),
            Some(graph_config.causal_temperature),
            reasoning_effort.retain_graph,
        );
        validate_explicit_temperature_override(
            "retain graph building",
            "RETAIN_GRAPH_TEMPERATURE",
            graph_temperature_override,
            &graph_temperature_resolution,
        )?;
        let consolidate_temperature_resolution = llm::resolve_temperature(
            llm_configs.reflect(),
            Some(consolidation_config.temperature),
            reasoning_effort.consolidate,
        );
        let consolidate_determinism = llm::assess_determinism(
            llm_configs.reflect(),
            Some(consolidation_config.temperature),
            reasoning_effort.consolidate,
        );
        validate_explicit_temperature_override(
            "consolidation",
            "CONSOLIDATE_TEMPERATURE",
            consolidate_temperature_override,
            &consolidate_temperature_resolution,
        )?;
        let opinion_merge_temperature_resolution = llm::resolve_temperature(
            llm_configs.reflect(),
            Some(opinion_merge_config.temperature),
            reasoning_effort.opinion_merge,
        );
        let opinion_merge_determinism = llm::assess_determinism(
            llm_configs.reflect(),
            Some(opinion_merge_config.temperature),
            reasoning_effort.opinion_merge,
        );
        validate_explicit_temperature_override(
            "opinion merge",
            "OPINION_MERGE_TEMPERATURE",
            opinion_merge_temperature_override,
            &opinion_merge_temperature_resolution,
        )?;
        let reflect_temperature_resolution = llm::resolve_temperature(
            llm_configs.reflect(),
            Some(reflect_temperature),
            reasoning_effort.reflect,
        );
        let reflect_determinism = llm::assess_determinism(
            llm_configs.reflect(),
            Some(reflect_temperature),
            reasoning_effort.reflect,
        );
        validate_explicit_temperature_override(
            "reflect",
            "REFLECT_TEMPERATURE",
            reflect_temperature_override,
            &reflect_temperature_resolution,
        )?;
        if let Some(requirement) = self.determinism_requirement {
            validate_determinism_requirement(
                "retain extraction",
                requirement,
                &extract_determinism,
            )?;
            validate_determinism_requirement(
                "retain entity resolution",
                requirement,
                &resolve_determinism,
            )?;
            validate_determinism_requirement(
                "retain graph building",
                requirement,
                &graph_determinism,
            )?;
            validate_determinism_requirement("reflect", requirement, &reflect_determinism)?;
            validate_determinism_requirement(
                "consolidation",
                requirement,
                &consolidate_determinism,
            )?;
            validate_determinism_requirement(
                "opinion merge",
                requirement,
                &opinion_merge_determinism,
            )?;
        }

        let max_conns = self.max_pool_connections.unwrap_or(10);
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(max_conns)
            .connect(runtime_config.database_url())
            .await?;
        let store = Arc::new(PgMemoryStore::new(pool.clone()));
        store.migrate().await?;
        let embeddings: Arc<dyn EmbeddingClient> = Arc::from(embedding::build_client(emb_config)?);

        let retain = Arc::new(DefaultRetainPipeline::new(
            Box::new(SimpleChunker),
            Box::new(LlmFactExtractor::new(
                stage_llm(
                    llm_configs.retain(),
                    LlmStage::RetainExtract,
                    self.metrics.as_ref(),
                )?,
                extraction_config,
            )),
            Box::new(LayeredEntityResolver::new_with_temperature(
                embedding::build_client(emb_config)?,
                stage_llm(
                    llm_configs.retain(),
                    LlmStage::RetainResolve,
                    self.metrics.as_ref(),
                )?,
                resolve_temperature,
            )),
            Box::new(DefaultGraphBuilder::new(
                stage_llm(
                    llm_configs.retain(),
                    LlmStage::RetainGraph,
                    self.metrics.as_ref(),
                )?,
                graph_config.clone(),
            )),
            Box::new(PgMemoryStore::new(pool.clone())),
            embedding::build_client(emb_config)?,
            stage_llm(
                llm_configs.retain(),
                LlmStage::RetainOpinion,
                self.metrics.as_ref(),
            )?,
            chunk_config.clone(),
            dedup_threshold,
        ));

        let reranker = reranker::build_reranker(reranker_config)?;
        let retriever_limit = retrieval_config.retriever_limit();
        let recall_rrf_k = 60.0;
        let max_facts = retrieval_config.max_facts();
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
        let reflect = Arc::new(DefaultReflectPipeline::new_with_limits(
            recall.clone(),
            stage_llm(
                llm_configs.reflect(),
                LlmStage::Reflect,
                self.metrics.as_ref(),
            )?,
            store.clone(),
            reflect_config.max_iterations(),
            reflect_config.max_tokens(),
            reflect_config.source_limit(),
            reflect_config.source_max_chars(),
            reflect_config.enable_source_lookup(),
            reflect_temperature,
        ));

        let consolidator = Arc::new(DefaultConsolidator::new(
            store.clone(),
            stage_llm(
                llm_configs.reflect(),
                LlmStage::Consolidate,
                self.metrics.as_ref(),
            )?,
            embeddings.clone(),
            recall.clone(),
            consolidation_config,
        ));
        let opinion_merger = Arc::new(DefaultOpinionMerger::new_with_config(
            store.clone(),
            stage_llm(
                llm_configs.reflect(),
                LlmStage::OpinionMerge,
                self.metrics.as_ref(),
            )?,
            embeddings.clone(),
            opinion_merge_config,
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
                embedding_model: embedding_label(emb_config),
                reranker_model: reranker_label(reranker_config),
                tuning: RuntimeTuning {
                    chunk_max_tokens: chunk_config.max_tokens,
                    chunk_overlap_tokens: chunk_config.overlap_tokens,
                    dedup_threshold,
                    retriever_limit,
                    recall_rrf_k,
                    max_facts,
                    reflect_max_iterations: reflect_config.max_iterations(),
                    reflect_max_tokens: reflect_config.max_tokens(),
                    retain_extract_reasoning_effort: reasoning_effort.retain_extract,
                    retain_extract_temperature: extraction_config.temperature,
                    retain_extract_effective_temperature: extract_temperature_resolution
                        .effective(),
                    retain_extract_structured_output_max_attempts: extraction_config
                        .structured_output_max_attempts,
                    retain_resolve_reasoning_effort: reasoning_effort.retain_resolve,
                    retain_resolve_temperature: resolve_temperature,
                    retain_resolve_effective_temperature: resolve_temperature_resolution
                        .effective(),
                    retain_graph_reasoning_effort: reasoning_effort.retain_graph,
                    retain_graph_temperature: graph_config.causal_temperature,
                    retain_graph_effective_temperature: graph_temperature_resolution.effective(),
                    reflect_enable_source_lookup: reflect_config.enable_source_lookup(),
                    reflect_reasoning_effort: reasoning_effort.reflect,
                    reflect_temperature,
                    reflect_effective_temperature: reflect_temperature_resolution.effective(),
                    graph_semantic_threshold: graph_config.semantic_threshold,
                    graph_temporal_max_days: graph_config.temporal_max_days,
                    graph_enable_causal: graph_config.enable_causal,
                    graph_max_causal_checks: graph_builder::MAX_CAUSAL_CHECKS,
                    consolidation_batch_size: consolidation_config.batch_size,
                    consolidation_max_tokens: consolidation_config.max_tokens,
                    consolidation_recall_budget: consolidation_config.recall_budget,
                    consolidation_structured_output_max_attempts: consolidation_config
                        .structured_output_max_attempts,
                    consolidate_temperature: consolidation_config.temperature,
                    consolidate_effective_temperature: consolidate_temperature_resolution
                        .effective(),
                    consolidate_reasoning_effort: reasoning_effort.consolidate,
                    opinion_merge_temperature: opinion_merge_config.temperature,
                    opinion_merge_effective_temperature: opinion_merge_temperature_resolution
                        .effective(),
                    opinion_merge_reasoning_effort: reasoning_effort.opinion_merge,
                    determinism: RuntimeDeterminism {
                        retain_extract: extract_determinism,
                        retain_resolve: resolve_determinism,
                        retain_graph: graph_determinism,
                        reflect: reflect_determinism,
                        consolidate: consolidate_determinism,
                        opinion_merge: opinion_merge_determinism,
                    },
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
}

fn validate_explicit_temperature_override(
    stage: &str,
    env_name: &str,
    explicit_override: Option<f32>,
    resolution: &llm::TemperatureResolution,
) -> Result<()> {
    if explicit_override.is_some()
        && let Some(reason) = resolution.unsupported_reason()
    {
        return Err(Error::Configuration(format!(
            "{env_name} requested {requested} for {stage}, but {reason}",
            requested = explicit_override.unwrap_or_default()
        )));
    }

    Ok(())
}

fn validate_determinism_requirement(
    stage: &str,
    requirement: DeterminismRequirement,
    assessment: &DeterminismAssessment,
) -> Result<()> {
    if !assessment.satisfies(requirement) {
        return Err(Error::Configuration(format!(
            "benchmark determinism requirement '{}' not met for {stage}: {}",
            requirement.as_str(),
            assessment.reason.as_deref().unwrap_or("no reason provided")
        )));
    }

    Ok(())
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
    match config.provider() {
        EmbeddingProvider::OpenAi => {
            format!("openai/{}", config.model().unwrap_or_default())
        }
        EmbeddingProvider::Local => {
            let name = config
                .model_path()
                .and_then(|p| std::path::Path::new(p).file_name())
                .and_then(|f| f.to_str())
                .unwrap_or("unknown");
            format!("local/{name}")
        }
    }
}

fn reranker_label(config: &RerankerConfig) -> String {
    match config.provider() {
        RerankerProvider::Local => {
            let name = config
                .model_path()
                .and_then(|p| std::path::Path::new(p).file_name())
                .and_then(|f| f.to_str())
                .unwrap_or("unknown");
            format!("local/{name}")
        }
        RerankerProvider::Api => {
            format!("api/{}", config.api_model().unwrap_or("unknown"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::OpenAiConfig;

    fn openai_client(model: &str) -> ClientConfig {
        ClientConfig::OpenAi(OpenAiConfig::new("sk-test", model).unwrap())
    }

    #[test]
    fn explicit_unsupported_temperature_override_is_rejected() {
        let resolution = llm::resolve_temperature(
            &openai_client("gpt-5.4-mini"),
            Some(0.0),
            Some(ReasoningEffort::None),
        );

        let err = validate_explicit_temperature_override(
            "reflect",
            "REFLECT_TEMPERATURE",
            Some(0.0),
            &resolution,
        )
        .unwrap_err();

        assert!(err.to_string().contains("REFLECT_TEMPERATURE"));
        assert!(err.to_string().contains("gpt-5.4-mini"));
    }

    #[test]
    fn default_unsupported_temperature_is_not_rejected() {
        let resolution = llm::resolve_temperature(
            &openai_client("gpt-5.4-mini"),
            Some(crate::reflect::REFLECT_TEMPERATURE),
            Some(ReasoningEffort::None),
        );

        validate_explicit_temperature_override("reflect", "REFLECT_TEMPERATURE", None, &resolution)
            .unwrap();
        assert_eq!(resolution.effective(), None);
    }

    #[test]
    fn best_effort_requirement_rejects_unsupported_assessment() {
        let err = validate_determinism_requirement(
            "reflect",
            DeterminismRequirement::BestEffort,
            &DeterminismAssessment {
                support: llm::DeterminismSupport::Unsupported,
                reason: Some("temperature 0 not effective".into()),
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("best_effort"));
    }

    #[test]
    fn best_effort_requirement_allows_best_effort_assessment() {
        validate_determinism_requirement(
            "reflect",
            DeterminismRequirement::BestEffort,
            &DeterminismAssessment {
                support: llm::DeterminismSupport::BestEffort,
                reason: Some("provider supports best-effort low variance".into()),
            },
        )
        .unwrap();
    }
}
