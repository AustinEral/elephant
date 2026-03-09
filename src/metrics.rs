//! Lightweight stage-aware metrics for benchmark instrumentation.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::llm::LlmClient;
use crate::types::llm::{CompletionRequest, CompletionResponse};

/// Benchmark stage label for LLM-backed operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmStage {
    /// Fact extraction during retain.
    RetainExtract,
    /// Entity verification/resolution during retain.
    RetainResolve,
    /// Link inference during retain graph-building.
    RetainGraph,
    /// Opinion reinforcement during retain.
    RetainOpinion,
    /// Reflect agent loop.
    Reflect,
    /// Observation consolidation.
    Consolidate,
    /// Opinion merge/reconciliation.
    OpinionMerge,
    /// Benchmark judge model.
    Judge,
}

/// Aggregate metrics for one LLM-backed stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageUsage {
    /// Total input/prompt tokens.
    pub prompt_tokens: u64,
    /// Total output/completion tokens.
    pub completion_tokens: u64,
    /// Number of LLM calls attempted.
    pub calls: u64,
    /// Number of calls that returned an error.
    pub errors: u64,
    /// Wall-clock latency across all calls.
    pub latency_ms: u64,
}

impl StageUsage {
    /// Total tokens across prompt and completion.
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// Thread-safe collector for per-stage usage snapshots.
#[derive(Debug, Default)]
pub struct MetricsCollector {
    inner: Mutex<BTreeMap<LlmStage, StageUsage>>,
}

impl MetricsCollector {
    /// Create an empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful call.
    pub fn record_success(&self, stage: LlmStage, response: &CompletionResponse, elapsed_ms: u64) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        let usage = inner.entry(stage).or_default();
        usage.prompt_tokens += response.input_tokens as u64;
        usage.completion_tokens += response.output_tokens as u64;
        usage.calls += 1;
        usage.latency_ms += elapsed_ms;
    }

    /// Record a failed call.
    pub fn record_error(&self, stage: LlmStage, elapsed_ms: u64) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        let usage = inner.entry(stage).or_default();
        usage.calls += 1;
        usage.errors += 1;
        usage.latency_ms += elapsed_ms;
    }

    /// Return a point-in-time snapshot.
    pub fn snapshot(&self) -> BTreeMap<LlmStage, StageUsage> {
        self.inner.lock().expect("metrics mutex poisoned").clone()
    }

    /// Return the summed usage across all stages.
    pub fn total_usage(&self) -> StageUsage {
        self.snapshot()
            .into_values()
            .fold(StageUsage::default(), |mut acc, stage| {
                acc.prompt_tokens += stage.prompt_tokens;
                acc.completion_tokens += stage.completion_tokens;
                acc.calls += stage.calls;
                acc.errors += stage.errors;
                acc.latency_ms += stage.latency_ms;
                acc
            })
    }
}

/// LLM wrapper that attributes calls to a benchmark stage.
pub struct MeteredLlmClient {
    inner: Arc<dyn LlmClient>,
    collector: Arc<MetricsCollector>,
    stage: LlmStage,
}

impl MeteredLlmClient {
    /// Wrap an LLM client with stage-aware metrics.
    pub fn new(
        inner: Arc<dyn LlmClient>,
        collector: Arc<MetricsCollector>,
        stage: LlmStage,
    ) -> Self {
        Self {
            inner,
            collector,
            stage,
        }
    }
}

#[async_trait]
impl LlmClient for MeteredLlmClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let start = Instant::now();
        let result = self.inner.complete(request).await;
        let elapsed_ms = start.elapsed().as_millis() as u64;
        match &result {
            Ok(response) => self
                .collector
                .record_success(self.stage, response, elapsed_ms),
            Err(_) => self.collector.record_error(self.stage, elapsed_ms),
        }
        result
    }
}
