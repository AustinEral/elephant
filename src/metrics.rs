//! Lightweight stage-aware metrics for benchmark instrumentation.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::llm::LlmClient;
use crate::types::llm::{CompletionRequest, CompletionResponse};

tokio::task_local! {
    static ACTIVE_SCOPED_COLLECTORS: RefCell<Vec<Arc<MetricsCollector>>>;
}

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
        self.record_usage(
            stage,
            response.input_tokens as u64,
            response.output_tokens as u64,
            1,
            0,
            elapsed_ms,
        );
    }

    /// Record a failed call.
    pub fn record_error(&self, stage: LlmStage, elapsed_ms: u64) {
        self.record_usage(stage, 0, 0, 1, 1, elapsed_ms);
    }

    fn record_usage(
        &self,
        stage: LlmStage,
        prompt_tokens: u64,
        completion_tokens: u64,
        calls: u64,
        errors: u64,
        latency_ms: u64,
    ) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        let usage = inner.entry(stage).or_default();
        usage.prompt_tokens += prompt_tokens;
        usage.completion_tokens += completion_tokens;
        usage.calls += calls;
        usage.errors += errors;
        usage.latency_ms += latency_ms;
    }

    /// Return a point-in-time snapshot.
    pub fn snapshot(&self) -> BTreeMap<LlmStage, StageUsage> {
        self.inner.lock().expect("metrics mutex poisoned").clone()
    }

    /// Merge a prior snapshot into this collector.
    pub fn extend_snapshot(&self, snapshot: &BTreeMap<LlmStage, StageUsage>) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        for (stage, usage) in snapshot {
            let entry = inner.entry(*stage).or_default();
            entry.prompt_tokens += usage.prompt_tokens;
            entry.completion_tokens += usage.completion_tokens;
            entry.calls += usage.calls;
            entry.errors += usage.errors;
            entry.latency_ms += usage.latency_ms;
        }
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

struct ScopedCollectorGuard;

impl Drop for ScopedCollectorGuard {
    fn drop(&mut self) {
        let _ = ACTIVE_SCOPED_COLLECTORS.try_with(|scopes| {
            let _ = scopes.borrow_mut().pop();
        });
    }
}

/// Run a future while attributing any metered LLM calls to an additional collector.
pub async fn with_scoped_collector<F, T>(collector: Arc<MetricsCollector>, future: F) -> T
where
    F: Future<Output = T>,
{
    if ACTIVE_SCOPED_COLLECTORS
        .try_with(|scopes| scopes.borrow_mut().push(collector.clone()))
        .is_ok()
    {
        let _guard = ScopedCollectorGuard;
        future.await
    } else {
        ACTIVE_SCOPED_COLLECTORS
            .scope(RefCell::new(vec![collector]), future)
            .await
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
            Ok(response) => {
                self.collector
                    .record_success(self.stage, response, elapsed_ms);
                let _ = ACTIVE_SCOPED_COLLECTORS.try_with(|scopes| {
                    let scoped = scopes.borrow().clone();
                    for collector in scoped {
                        collector.record_success(self.stage, response, elapsed_ms);
                    }
                });
            }
            Err(_) => {
                self.collector.record_error(self.stage, elapsed_ms);
                let _ = ACTIVE_SCOPED_COLLECTORS.try_with(|scopes| {
                    let scoped = scopes.borrow().clone();
                    for collector in scoped {
                        collector.record_error(self.stage, elapsed_ms);
                    }
                });
            }
        }
        result
    }
}
