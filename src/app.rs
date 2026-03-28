//! Shared application facade consumed by transports.

mod consolidation;

use std::sync::Arc;

use crate::config::ServerConfig;
use crate::consolidation::{Consolidator, OpinionMerger};
use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::recall::RecallPipeline;
use crate::reflect::ReflectPipeline;
use crate::retain::RetainPipeline;
use crate::runtime::ElephantRuntime;
use crate::storage::MemoryStore;
use crate::types::{
    BankId, ConsolidationReport, Disposition, Entity, EntityId, Fact, MemoryBank,
    OpinionMergeReport, RecallQuery, RecallResult, ReflectQuery, ReflectResult, RetainInput,
    RetainOutput,
};

/// Shared runtime and policy info exposed to transports.
#[derive(Clone, serde::Serialize)]
pub struct AppInfo {
    /// Server binary version.
    pub version: String,
    /// Active model configuration.
    pub models: AppModelsInfo,
    /// Active retrieval tuning.
    pub retrieval: AppRetrievalInfo,
    /// Active reflect tuning.
    pub reflect: AppReflectInfo,
    /// Active consolidation tuning.
    pub consolidation: AppConsolidationInfo,
    /// Active background consolidation policy.
    pub server_consolidation: AppBackgroundConsolidationInfo,
}

/// Model labels exposed by the app facade.
#[derive(Clone, serde::Serialize)]
pub struct AppModelsInfo {
    /// The LLM model used for retain.
    pub retain: String,
    /// The LLM model used for reflect.
    pub reflect: String,
    /// The embedding model name.
    pub embedding: String,
    /// The reranker model label.
    pub reranker: String,
}

/// Retrieval tuning exposed by the app facade.
#[derive(Clone, serde::Serialize)]
pub struct AppRetrievalInfo {
    /// Retriever candidate limit per retrieval strategy.
    pub retriever_limit: usize,
    /// Maximum number of facts kept before token budgeting.
    pub max_facts: usize,
}

/// Reflect tuning exposed by the app facade.
#[derive(Clone, serde::Serialize)]
pub struct AppReflectInfo {
    /// Reflect iteration cap.
    pub max_iterations: usize,
    /// Optional reflect completion cap.
    pub max_tokens: Option<usize>,
    /// Whether source lookup is enabled.
    pub source_lookup_enabled: bool,
}

/// Consolidation runtime tuning exposed by the app facade.
#[derive(Clone, serde::Serialize)]
pub struct AppConsolidationInfo {
    /// Consolidation batch size.
    pub batch_size: usize,
    /// Consolidation completion cap.
    pub max_tokens: usize,
    /// Consolidation recall budget.
    pub recall_budget: usize,
}

/// Background consolidation policy exposed by the app facade.
#[derive(Clone, serde::Serialize)]
pub struct AppBackgroundConsolidationInfo {
    /// Whether automatic background consolidation is enabled.
    pub enabled: bool,
    /// Minimum unconsolidated world/experience facts before scheduling.
    pub min_facts: usize,
    /// Minimum delay between attempts per bank.
    pub cooldown_secs: u64,
    /// Whether opinion merge runs after consolidation.
    pub merge_opinions_after: bool,
}

/// Shared application facade for HTTP, MCP, and future transports.
#[derive(Clone)]
pub struct AppHandle {
    info: AppInfo,
    retain: Arc<dyn RetainPipeline>,
    recall: Arc<dyn RecallPipeline>,
    reflect: Arc<dyn ReflectPipeline>,
    consolidator: Arc<dyn Consolidator>,
    opinion_merger: Arc<dyn OpinionMerger>,
    store: Arc<dyn MemoryStore>,
    embeddings: Arc<dyn EmbeddingClient>,
}

impl AppHandle {
    /// Build the shared application facade from a fully constructed runtime.
    pub fn new(runtime: &ElephantRuntime, server_config: &ServerConfig) -> Result<Self> {
        let server_consolidation_policy = consolidation::ConsolidationPolicy::from_config(
            server_config.background_consolidation(),
        )?;
        Ok(Self {
            info: AppInfo {
                version: env!("CARGO_PKG_VERSION").into(),
                models: AppModelsInfo {
                    retain: runtime.info.retain_model.clone(),
                    reflect: runtime.info.reflect_model.clone(),
                    embedding: runtime.info.embedding_model.clone(),
                    reranker: runtime.info.reranker_model.clone(),
                },
                retrieval: AppRetrievalInfo {
                    retriever_limit: runtime.info.tuning.retriever_limit,
                    max_facts: runtime.info.tuning.max_facts,
                },
                reflect: AppReflectInfo {
                    max_iterations: runtime.info.tuning.reflect_max_iterations,
                    max_tokens: runtime.info.tuning.reflect_max_tokens,
                    source_lookup_enabled: runtime.info.tuning.reflect_enable_source_lookup,
                },
                consolidation: AppConsolidationInfo {
                    batch_size: runtime.info.tuning.consolidation_batch_size,
                    max_tokens: runtime.info.tuning.consolidation_max_tokens,
                    recall_budget: runtime.info.tuning.consolidation_recall_budget,
                },
                server_consolidation: server_consolidation_policy.to_info(),
            },
            retain: consolidation::wrap_retain_pipeline_with_consolidation(
                runtime.retain.clone(),
                runtime.store.clone(),
                runtime.consolidator.clone(),
                runtime.opinion_merger.clone(),
                server_consolidation_policy,
            )?,
            recall: runtime.recall.clone(),
            reflect: runtime.reflect.clone(),
            consolidator: runtime.consolidator.clone(),
            opinion_merger: runtime.opinion_merger.clone(),
            store: runtime.store.clone(),
            embeddings: runtime.embeddings.clone(),
        })
    }

    #[cfg(test)]
    pub(crate) fn from_parts(
        info: AppInfo,
        retain: Arc<dyn RetainPipeline>,
        recall: Arc<dyn RecallPipeline>,
        reflect: Arc<dyn ReflectPipeline>,
        consolidator: Arc<dyn Consolidator>,
        opinion_merger: Arc<dyn OpinionMerger>,
        store: Arc<dyn MemoryStore>,
        embeddings: Arc<dyn EmbeddingClient>,
    ) -> Self {
        Self {
            info,
            retain,
            recall,
            reflect,
            consolidator,
            opinion_merger,
            store,
            embeddings,
        }
    }

    /// Return shared runtime and policy info for transports.
    pub fn info(&self) -> &AppInfo {
        &self.info
    }

    /// Return the embedding model label used for new banks.
    pub fn embedding_model_name(&self) -> &str {
        self.embeddings.model_name()
    }

    /// Return the embedding dimensionality used for new banks.
    pub fn embedding_dimensions(&self) -> u16 {
        self.embeddings.dimensions() as u16
    }

    /// List all banks.
    pub async fn list_banks(&self) -> Result<Vec<MemoryBank>> {
        self.store.list_banks().await
    }

    /// Create a new bank.
    pub async fn create_bank(&self, bank: &MemoryBank) -> Result<()> {
        self.store.create_bank(bank).await.map(|_| ())
    }

    /// Fetch one bank.
    pub async fn get_bank(&self, bank_id: BankId) -> Result<MemoryBank> {
        self.store.get_bank(bank_id).await
    }

    /// Retain new content.
    pub async fn retain(&self, input: &RetainInput) -> Result<RetainOutput> {
        self.retain.retain(input).await
    }

    /// Recall relevant facts.
    pub async fn recall(&self, query: &RecallQuery) -> Result<RecallResult> {
        self.recall.recall(query).await
    }

    /// Reflect over memory.
    pub async fn reflect(&self, query: &ReflectQuery) -> Result<ReflectResult> {
        self.reflect.reflect(query).await
    }

    /// List entities for a bank.
    pub async fn list_entities(&self, bank_id: BankId) -> Result<Vec<Entity>> {
        self.store.list_entities(bank_id).await
    }

    /// List facts attached to an entity.
    pub async fn entity_facts(&self, entity_id: EntityId) -> Result<Vec<Fact>> {
        self.store.get_entity_facts(entity_id).await
    }

    /// Run consolidation for a bank.
    pub async fn consolidate(&self, bank_id: BankId) -> Result<ConsolidationReport> {
        self.consolidator.consolidate(bank_id).await
    }

    /// Run opinion merge for a bank.
    pub async fn merge_opinions(&self, bank_id: BankId) -> Result<OpinionMergeReport> {
        self.opinion_merger.merge(bank_id).await
    }

    /// Build a new bank value with the active embedding metadata filled in.
    pub fn new_bank(
        &self,
        name: String,
        mission: String,
        directives: Vec<String>,
        disposition: Disposition,
    ) -> MemoryBank {
        MemoryBank {
            id: BankId::new(),
            name,
            mission,
            directives,
            disposition,
            embedding_model: self.embedding_model_name().to_string(),
            embedding_dimensions: self.embedding_dimensions(),
        }
    }
}
