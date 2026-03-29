//! Benchmark runtime startup support.

use std::num::NonZeroU32;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use elephant::consolidation::ConsolidationProgress;
use elephant::llm::DeterminismRequirement;
use elephant::metrics::MetricsCollector;
use elephant::{
    BankId, ConsolidationReport, Disposition, ElephantRuntime, Entity, Fact, FactFilter,
    MemoryBank, ReflectQuery, ReflectResult, Result, RetainInput, RetainOutput, RuntimeBuilder,
    RuntimeConfig, RuntimePromptHashes, RuntimeTuning,
};

use crate::config::BenchConfig;

/// Serialized prompt-hash snapshot used by benchmark artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct BenchRuntimePromptHashes(RuntimePromptHashes);

impl BenchRuntimePromptHashes {
    /// Borrow the underlying Elephant prompt-hash snapshot.
    pub fn as_elephant(&self) -> &RuntimePromptHashes {
        &self.0
    }
}

impl From<RuntimePromptHashes> for BenchRuntimePromptHashes {
    fn from(value: RuntimePromptHashes) -> Self {
        Self(value)
    }
}

/// Serialized runtime-tuning snapshot used by benchmark artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct BenchRuntimeTuning(RuntimeTuning);

impl BenchRuntimeTuning {
    /// Borrow the underlying Elephant runtime tuning snapshot.
    pub fn as_elephant(&self) -> &RuntimeTuning {
        &self.0
    }

    /// Return the consolidation batch size from the wrapped tuning snapshot.
    pub fn consolidation_batch_size(&self) -> usize {
        self.0.consolidation_batch_size
    }
}

impl From<RuntimeTuning> for BenchRuntimeTuning {
    fn from(value: RuntimeTuning) -> Self {
        Self(value)
    }
}

/// Read-only runtime metadata captured for benchmark provenance and reporting.
#[derive(Debug, Clone)]
pub struct BenchRuntimeMetadata {
    retain_model: String,
    reflect_model: String,
    embedding_model: String,
    embedding_dimensions: u16,
    reranker_model: String,
    prompt_hashes: BenchRuntimePromptHashes,
    tuning: BenchRuntimeTuning,
}

impl BenchRuntimeMetadata {
    fn from_runtime(runtime: &ElephantRuntime) -> Self {
        let info = runtime.info();
        Self {
            retain_model: info.retain_model.clone(),
            reflect_model: info.reflect_model.clone(),
            embedding_model: info.embedding_model.clone(),
            embedding_dimensions: runtime.embeddings().dimensions() as u16,
            reranker_model: info.reranker_model.clone(),
            prompt_hashes: info.prompt_hashes.clone().into(),
            tuning: info.tuning.clone().into(),
        }
    }

    /// Return the retain/extraction model label.
    pub fn retain_model(&self) -> &str {
        &self.retain_model
    }

    /// Return the reflect/consolidation model label.
    pub fn reflect_model(&self) -> &str {
        &self.reflect_model
    }

    /// Return the embedding model label.
    pub fn embedding_model(&self) -> &str {
        &self.embedding_model
    }

    /// Return the embedding dimensionality used by the runtime.
    pub fn embedding_dimensions(&self) -> u16 {
        self.embedding_dimensions
    }

    /// Return the reranker model label.
    pub fn reranker_model(&self) -> &str {
        &self.reranker_model
    }

    /// Return the prompt hash snapshot.
    pub fn prompt_hashes(&self) -> &BenchRuntimePromptHashes {
        &self.prompt_hashes
    }

    /// Return the runtime tuning snapshot.
    pub fn tuning(&self) -> &BenchRuntimeTuning {
        &self.tuning
    }

    /// Return the consolidation batch size from the captured runtime tuning.
    pub fn consolidation_batch_size(&self) -> usize {
        self.tuning.consolidation_batch_size()
    }
}

/// Concrete benchmark runtime adapter over the Elephant runtime graph.
#[derive(Debug, Clone)]
pub struct BenchRuntime {
    runtime: Arc<ElephantRuntime>,
    metadata: BenchRuntimeMetadata,
}

impl BenchRuntime {
    /// Return the captured runtime metadata snapshot.
    pub fn metadata(&self) -> &BenchRuntimeMetadata {
        &self.metadata
    }

    /// Create and persist a benchmark memory bank using the runtime embedding configuration.
    pub async fn create_benchmark_bank(
        &self,
        name: impl Into<String>,
        mission: impl Into<String>,
    ) -> Result<MemoryBank> {
        let bank = MemoryBank {
            id: BankId::new(),
            name: name.into(),
            mission: mission.into(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: self.metadata.embedding_model().to_string(),
            embedding_dimensions: self.metadata.embedding_dimensions(),
        };
        self.runtime.store().create_bank(&bank).await?;
        Ok(bank)
    }

    /// Delete a benchmark memory bank and its associated data.
    pub async fn delete_bank(&self, bank_id: BankId) -> Result<()> {
        self.runtime.store().delete_bank(bank_id).await
    }

    /// Retain content through the benchmark runtime.
    pub async fn retain(&self, input: &RetainInput) -> Result<RetainOutput> {
        self.runtime.retain_pipeline().retain(input).await
    }

    /// Reflect through the benchmark runtime.
    pub async fn reflect(&self, query: &ReflectQuery) -> Result<ReflectResult> {
        self.runtime.reflect_pipeline().reflect(query).await
    }

    /// Fetch facts for a bank through the benchmark runtime store.
    pub async fn get_facts_by_bank(
        &self,
        bank_id: BankId,
        filter: FactFilter,
    ) -> Result<Vec<Fact>> {
        self.runtime
            .store()
            .get_facts_by_bank(bank_id, filter)
            .await
    }

    /// List entities for a bank through the benchmark runtime store.
    pub async fn list_entities(&self, bank_id: BankId) -> Result<Vec<Entity>> {
        self.runtime.store().list_entities(bank_id).await
    }

    /// Count unconsolidated world/experience facts for a bank.
    pub async fn count_unconsolidated_facts(&self, bank_id: BankId) -> Result<usize> {
        self.get_facts_by_bank(
            bank_id,
            FactFilter {
                network: Some(vec![
                    elephant::NetworkType::World,
                    elephant::NetworkType::Experience,
                ]),
                unconsolidated_only: true,
                ..Default::default()
            },
        )
        .await
        .map(|facts| facts.len())
    }

    /// Run consolidation without progress reporting.
    pub async fn consolidate(&self, bank_id: BankId) -> Result<ConsolidationReport> {
        self.runtime.consolidator().consolidate(bank_id).await
    }

    /// Run consolidation with optional progress reporting.
    pub async fn consolidate_with_progress(
        &self,
        bank_id: BankId,
        progress: Option<tokio::sync::mpsc::UnboundedSender<ConsolidationProgress>>,
    ) -> Result<ConsolidationReport> {
        self.runtime
            .consolidator()
            .consolidate_with_progress(bank_id, progress)
            .await
    }
}

/// Benchmark-facing startup bundle.
#[derive(Debug)]
pub struct BenchHarness {
    runtime: BenchRuntime,
    determinism_requirement: Option<DeterminismRequirement>,
}

impl BenchHarness {
    /// Return the captured runtime metadata snapshot.
    pub fn metadata(&self) -> &BenchRuntimeMetadata {
        self.runtime.metadata()
    }

    /// Return the configured determinism requirement used for this runtime.
    pub fn determinism_requirement(&self) -> Option<DeterminismRequirement> {
        self.determinism_requirement
    }

    /// Consume the harness and return the constructed benchmark runtime adapter.
    pub fn into_runtime(self) -> BenchRuntime {
        self.runtime
    }
}

/// Builder for benchmark startup policy and runtime construction.
#[derive(Debug)]
pub struct BenchHarnessBuilder {
    runtime_builder: RuntimeBuilder,
    determinism_requirement: Option<DeterminismRequirement>,
}

impl BenchHarnessBuilder {
    /// Load benchmark startup config from the process environment.
    pub fn from_env() -> Result<Self> {
        let runtime_config = RuntimeConfig::from_env()?;
        let bench_config = BenchConfig::from_env()?;
        Ok(Self::new(runtime_config, bench_config))
    }

    /// Create a builder from validated runtime and benchmark config.
    pub(crate) fn new(runtime_config: RuntimeConfig, bench_config: BenchConfig) -> Self {
        let determinism_requirement = bench_config.determinism_requirement();
        let runtime_builder = if let Some(requirement) = determinism_requirement {
            RuntimeBuilder::new(runtime_config).determinism_requirement(requirement)
        } else {
            RuntimeBuilder::new(runtime_config)
        };
        Self {
            runtime_builder,
            determinism_requirement,
        }
    }

    /// Install a stage-aware metrics collector.
    pub fn metrics(mut self, metrics: Arc<MetricsCollector>) -> Self {
        self.runtime_builder = self.runtime_builder.metrics(metrics);
        self
    }

    /// Override the maximum Postgres pool connection count.
    pub fn max_pool_connections(mut self, max_pool_connections: NonZeroU32) -> Self {
        self.runtime_builder = self
            .runtime_builder
            .max_pool_connections(max_pool_connections);
        self
    }

    /// Build the benchmark harness and runtime.
    pub async fn build(self) -> Result<BenchHarness> {
        let runtime = self.runtime_builder.build().await?;
        Ok(BenchHarness {
            runtime: BenchRuntime {
                metadata: BenchRuntimeMetadata::from_runtime(&runtime),
                runtime: Arc::new(runtime),
            },
            determinism_requirement: self.determinism_requirement,
        })
    }
}
