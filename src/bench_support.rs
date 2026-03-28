//! Transitional benchmark startup support over the typed config/runtime seam.

use std::num::NonZeroU32;
use std::sync::Arc;

use crate::Result;
use crate::config::{BenchConfig, RuntimeConfig};
use crate::llm::DeterminismRequirement;
use crate::metrics::MetricsCollector;
use crate::runtime::{ElephantRuntime, RuntimeBuilder};

/// Benchmark-facing runtime bundle.
#[derive(Debug)]
pub struct BenchHarness {
    runtime: ElephantRuntime,
    determinism_requirement: Option<DeterminismRequirement>,
}

impl BenchHarness {
    /// Return the configured determinism requirement used for this runtime.
    pub fn determinism_requirement(&self) -> Option<DeterminismRequirement> {
        self.determinism_requirement
    }

    /// Borrow the constructed runtime.
    pub fn runtime(&self) -> &ElephantRuntime {
        &self.runtime
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
    pub fn new(runtime_config: RuntimeConfig, bench_config: BenchConfig) -> Self {
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
        Ok(BenchHarness {
            runtime: self.runtime_builder.build().await?,
            determinism_requirement: self.determinism_requirement,
        })
    }
}
