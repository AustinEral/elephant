//! Benchmark runtime startup support.

use std::num::NonZeroU32;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use elephant::llm::DeterminismRequirement;
use elephant::metrics::MetricsCollector;
use elephant::{
    ElephantRuntime, Result, RuntimeBuilder, RuntimeConfig, RuntimeInfo, RuntimePromptHashes,
    RuntimeTuning,
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
    reranker_model: String,
    prompt_hashes: BenchRuntimePromptHashes,
    tuning: BenchRuntimeTuning,
}

impl From<&RuntimeInfo> for BenchRuntimeMetadata {
    fn from(info: &RuntimeInfo) -> Self {
        Self {
            retain_model: info.retain_model.clone(),
            reflect_model: info.reflect_model.clone(),
            embedding_model: info.embedding_model.clone(),
            reranker_model: info.reranker_model.clone(),
            prompt_hashes: info.prompt_hashes.clone().into(),
            tuning: info.tuning.clone().into(),
        }
    }
}

impl BenchRuntimeMetadata {
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

/// Benchmark-facing runtime bundle.
#[derive(Debug)]
pub struct BenchHarness {
    runtime: ElephantRuntime,
    metadata: BenchRuntimeMetadata,
    determinism_requirement: Option<DeterminismRequirement>,
}

impl BenchHarness {
    /// Return the captured runtime metadata snapshot.
    pub fn metadata(&self) -> &BenchRuntimeMetadata {
        &self.metadata
    }

    /// Return the configured determinism requirement used for this runtime.
    pub fn determinism_requirement(&self) -> Option<DeterminismRequirement> {
        self.determinism_requirement
    }

    /// Consume the harness and return the constructed runtime.
    pub fn into_runtime(self) -> ElephantRuntime {
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
            metadata: BenchRuntimeMetadata::from(runtime.info()),
            runtime,
            determinism_requirement: self.determinism_requirement,
        })
    }
}
