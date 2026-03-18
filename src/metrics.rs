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
    #[serde(default, alias = "prompt_tokens")]
    pub input_tokens: u64,
    /// Total prompt tokens served from cache.
    #[serde(default)]
    pub cached_prompt_tokens: u64,
    /// Total Anthropic cache-hit input tokens.
    #[serde(default)]
    pub cache_read_input_tokens: u64,
    /// Total Anthropic cache-write input tokens.
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    /// Total output/completion tokens.
    #[serde(default, alias = "completion_tokens")]
    pub output_tokens: u64,
    /// Number of LLM calls attempted.
    #[serde(default)]
    pub calls: u64,
    /// Number of calls that returned an error.
    #[serde(default)]
    pub errors: u64,
    /// Wall-clock latency across all calls.
    #[serde(default)]
    pub latency_ms: u64,
}

impl StageUsage {
    /// Total tokens across prompt and completion.
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
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
            response
                .prompt_cache
                .as_ref()
                .and_then(|usage| usage.cached_tokens)
                .unwrap_or(0) as u64,
            response
                .prompt_cache
                .as_ref()
                .and_then(|usage| usage.cache_read_input_tokens)
                .unwrap_or(0) as u64,
            response
                .prompt_cache
                .as_ref()
                .and_then(|usage| usage.cache_creation_input_tokens)
                .unwrap_or(0) as u64,
            response.output_tokens as u64,
            1,
            0,
            elapsed_ms,
        );
    }

    /// Record a failed call.
    pub fn record_error(&self, stage: LlmStage, elapsed_ms: u64) {
        self.record_usage(stage, 0, 0, 0, 0, 0, 1, 1, elapsed_ms);
    }

    fn record_usage(
        &self,
        stage: LlmStage,
        input_tokens: u64,
        cached_prompt_tokens: u64,
        cache_read_input_tokens: u64,
        cache_creation_input_tokens: u64,
        output_tokens: u64,
        calls: u64,
        errors: u64,
        latency_ms: u64,
    ) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        let usage = inner.entry(stage).or_default();
        usage.input_tokens += input_tokens;
        usage.cached_prompt_tokens += cached_prompt_tokens;
        usage.cache_read_input_tokens += cache_read_input_tokens;
        usage.cache_creation_input_tokens += cache_creation_input_tokens;
        usage.output_tokens += output_tokens;
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
            entry.input_tokens += usage.input_tokens;
            entry.cached_prompt_tokens += usage.cached_prompt_tokens;
            entry.cache_read_input_tokens += usage.cache_read_input_tokens;
            entry.cache_creation_input_tokens += usage.cache_creation_input_tokens;
            entry.output_tokens += usage.output_tokens;
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
                acc.input_tokens += stage.input_tokens;
                acc.cached_prompt_tokens += stage.cached_prompt_tokens;
                acc.cache_read_input_tokens += stage.cache_read_input_tokens;
                acc.cache_creation_input_tokens += stage.cache_creation_input_tokens;
                acc.output_tokens += stage.output_tokens;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::llm::PromptCacheUsage;

    #[test]
    fn stage_usage_deserializes_legacy_token_names() {
        let usage: StageUsage = serde_json::from_str(
            r#"{
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "calls": 1,
                "errors": 0,
                "latency_ms": 25
            }"#,
        )
        .unwrap();

        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 4);
        assert_eq!(usage.total_tokens(), 14);
    }

    #[test]
    fn record_success_accumulates_cache_fields() {
        let collector = MetricsCollector::new();
        let response = CompletionResponse {
            content: "ok".into(),
            input_tokens: 100,
            output_tokens: 20,
            stop_reason: None,
            tool_calls: vec![],
            prompt_cache: Some(PromptCacheUsage {
                cached_tokens: Some(80),
                cache_read_input_tokens: Some(60),
                cache_creation_input_tokens: Some(10),
            }),
        };

        collector.record_success(LlmStage::Reflect, &response, 50);
        let snapshot = collector.snapshot();
        let usage = snapshot.get(&LlmStage::Reflect).unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.cached_prompt_tokens, 80);
        assert_eq!(usage.cache_read_input_tokens, 60);
        assert_eq!(usage.cache_creation_input_tokens, 10);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.calls, 1);
        assert_eq!(usage.errors, 0);
        assert_eq!(usage.latency_ms, 50);
    }

    #[test]
    fn extend_snapshot_and_total_usage_include_cache_fields() {
        let collector = MetricsCollector::new();
        let mut prior = BTreeMap::new();
        prior.insert(
            LlmStage::Reflect,
            StageUsage {
                input_tokens: 40,
                cached_prompt_tokens: 30,
                cache_read_input_tokens: 20,
                cache_creation_input_tokens: 5,
                output_tokens: 8,
                calls: 1,
                errors: 0,
                latency_ms: 10,
            },
        );
        prior.insert(
            LlmStage::Judge,
            StageUsage {
                input_tokens: 10,
                cached_prompt_tokens: 0,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                output_tokens: 2,
                calls: 1,
                errors: 0,
                latency_ms: 3,
            },
        );

        collector.extend_snapshot(&prior);
        let total = collector.total_usage();
        assert_eq!(total.input_tokens, 50);
        assert_eq!(total.cached_prompt_tokens, 30);
        assert_eq!(total.cache_read_input_tokens, 20);
        assert_eq!(total.cache_creation_input_tokens, 5);
        assert_eq!(total.output_tokens, 10);
        assert_eq!(total.calls, 2);
        assert_eq!(total.errors, 0);
        assert_eq!(total.latency_ms, 13);
    }
}
