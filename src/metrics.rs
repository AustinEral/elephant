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
use crate::types::llm::{CacheStatus, CompletionRequest, CompletionResponse};

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

/// Stable operator-facing rollup for cache-aware stage reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorStage {
    /// Aggregate retain pipeline stages.
    Retain,
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

/// Aggregate cache-aware metrics for one LLM-backed stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheAwareStageUsage {
    /// Total normalized prompt usage, including uncached, hit, and write tokens.
    pub prompt_tokens: u64,
    /// Prompt tokens that were not served from cache.
    pub uncached_prompt_tokens: u64,
    /// Prompt tokens served from cache hits.
    pub cache_hit_prompt_tokens: u64,
    /// Prompt tokens newly written into the provider cache.
    pub cache_write_prompt_tokens: u64,
    /// Total output/completion tokens.
    pub completion_tokens: u64,
    /// Number of LLM calls attempted.
    pub calls: u64,
    /// Number of calls that returned an error.
    pub errors: u64,
    /// Wall-clock latency across all calls.
    pub latency_ms: u64,
    /// Calls where the provider path exposed cache-aware usage details.
    pub cache_supported_calls: u64,
    /// Calls that reported at least one cache hit.
    pub cache_hit_calls: u64,
    /// Calls that reported at least one cache write.
    pub cache_write_calls: u64,
    /// Calls where cache-aware usage was unsupported or unavailable.
    pub cache_unsupported_calls: u64,
}

impl CacheAwareStageUsage {
    /// Project cache-aware totals back to the legacy stage-usage contract.
    pub fn legacy_totals(&self) -> StageUsage {
        StageUsage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            calls: self.calls,
            errors: self.errors,
            latency_ms: self.latency_ms,
        }
    }

    fn merge(&mut self, other: &Self) {
        self.prompt_tokens += other.prompt_tokens;
        self.uncached_prompt_tokens += other.uncached_prompt_tokens;
        self.cache_hit_prompt_tokens += other.cache_hit_prompt_tokens;
        self.cache_write_prompt_tokens += other.cache_write_prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.calls += other.calls;
        self.errors += other.errors;
        self.latency_ms += other.latency_ms;
        self.cache_supported_calls += other.cache_supported_calls;
        self.cache_hit_calls += other.cache_hit_calls;
        self.cache_write_calls += other.cache_write_calls;
        self.cache_unsupported_calls += other.cache_unsupported_calls;
    }

    fn record_success(&mut self, response: &CompletionResponse, elapsed_ms: u64) {
        let usage = &response.usage;
        self.prompt_tokens += usage.prompt_tokens as u64;
        self.uncached_prompt_tokens += usage.uncached_prompt_tokens as u64;
        self.cache_hit_prompt_tokens += usage.cache_hit_prompt_tokens as u64;
        self.cache_write_prompt_tokens += usage.cache_write_prompt_tokens as u64;
        self.completion_tokens += usage.completion_tokens as u64;
        self.calls += 1;
        self.latency_ms += elapsed_ms;

        match usage.cache_status {
            CacheStatus::Unsupported => {
                self.cache_unsupported_calls += 1;
            }
            CacheStatus::NoActivity => {
                self.cache_supported_calls += 1;
            }
            CacheStatus::WriteOnly => {
                self.cache_supported_calls += 1;
                self.cache_write_calls += 1;
            }
            CacheStatus::Hit => {
                self.cache_supported_calls += 1;
                self.cache_hit_calls += 1;
            }
            CacheStatus::HitAndWrite => {
                self.cache_supported_calls += 1;
                self.cache_hit_calls += 1;
                self.cache_write_calls += 1;
            }
        }
    }

    fn record_error(&mut self, elapsed_ms: u64) {
        self.calls += 1;
        self.errors += 1;
        self.latency_ms += elapsed_ms;
    }

    fn merge_legacy_snapshot(&mut self, usage: &StageUsage) {
        self.prompt_tokens += usage.prompt_tokens;
        self.completion_tokens += usage.completion_tokens;
        self.calls += usage.calls;
        self.errors += usage.errors;
        self.latency_ms += usage.latency_ms;
        self.cache_unsupported_calls += usage.calls;
    }
}

#[derive(Debug, Default)]
struct MetricsState {
    legacy: BTreeMap<LlmStage, StageUsage>,
    cache_aware: BTreeMap<LlmStage, CacheAwareStageUsage>,
}

/// Thread-safe collector for per-stage usage snapshots.
#[derive(Debug, Default)]
pub struct MetricsCollector {
    inner: Mutex<MetricsState>,
}

impl MetricsCollector {
    /// Create an empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful call.
    pub fn record_success(&self, stage: LlmStage, response: &CompletionResponse, elapsed_ms: u64) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        let cache_aware = inner.cache_aware.entry(stage).or_default();
        cache_aware.record_success(response, elapsed_ms);

        let legacy = inner.legacy.entry(stage).or_default();
        legacy.prompt_tokens += response.usage.prompt_tokens as u64;
        legacy.completion_tokens += response.usage.completion_tokens as u64;
        legacy.calls += 1;
        legacy.latency_ms += elapsed_ms;
    }

    /// Record a failed call.
    pub fn record_error(&self, stage: LlmStage, elapsed_ms: u64) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        let legacy = inner.legacy.entry(stage).or_default();
        legacy.calls += 1;
        legacy.errors += 1;
        legacy.latency_ms += elapsed_ms;

        let cache_aware = inner.cache_aware.entry(stage).or_default();
        cache_aware.record_error(elapsed_ms);
    }

    /// Return a point-in-time snapshot.
    pub fn snapshot(&self) -> BTreeMap<LlmStage, StageUsage> {
        self.inner
            .lock()
            .expect("metrics mutex poisoned")
            .legacy
            .clone()
    }

    /// Return a point-in-time cache-aware snapshot.
    pub fn cache_aware_snapshot(&self) -> BTreeMap<LlmStage, CacheAwareStageUsage> {
        self.inner
            .lock()
            .expect("metrics mutex poisoned")
            .cache_aware
            .clone()
    }

    /// Return cache-aware usage rolled up into canonical operator stages.
    pub fn operator_snapshot(&self) -> BTreeMap<OperatorStage, CacheAwareStageUsage> {
        let mut operator_usage = BTreeMap::new();
        for (stage, usage) in self.cache_aware_snapshot() {
            operator_usage
                .entry(OperatorStage::from(stage))
                .or_insert_with(CacheAwareStageUsage::default)
                .merge(&usage);
        }
        operator_usage
    }

    /// Merge a prior snapshot into this collector.
    pub fn extend_snapshot(&self, snapshot: &BTreeMap<LlmStage, StageUsage>) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        for (stage, usage) in snapshot {
            let legacy = inner.legacy.entry(*stage).or_default();
            legacy.prompt_tokens += usage.prompt_tokens;
            legacy.completion_tokens += usage.completion_tokens;
            legacy.calls += usage.calls;
            legacy.errors += usage.errors;
            legacy.latency_ms += usage.latency_ms;

            inner
                .cache_aware
                .entry(*stage)
                .or_default()
                .merge_legacy_snapshot(usage);
        }
    }

    /// Merge a prior cache-aware snapshot into this collector.
    pub fn extend_cache_aware_snapshot(&self, snapshot: &BTreeMap<LlmStage, CacheAwareStageUsage>) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        for (stage, usage) in snapshot {
            inner.cache_aware.entry(*stage).or_default().merge(usage);

            let legacy_usage = usage.legacy_totals();
            let legacy = inner.legacy.entry(*stage).or_default();
            legacy.prompt_tokens += legacy_usage.prompt_tokens;
            legacy.completion_tokens += legacy_usage.completion_tokens;
            legacy.calls += legacy_usage.calls;
            legacy.errors += legacy_usage.errors;
            legacy.latency_ms += legacy_usage.latency_ms;
        }
    }

    /// Merge a cache-aware snapshot when present, otherwise fall back to legacy semantics.
    pub fn extend_cache_aware_or_legacy_snapshot(
        &self,
        cache_aware: &BTreeMap<LlmStage, CacheAwareStageUsage>,
        legacy: &BTreeMap<LlmStage, StageUsage>,
    ) {
        if !cache_aware.is_empty() {
            self.extend_cache_aware_snapshot(cache_aware);
        } else {
            self.extend_snapshot(legacy);
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

    /// Return the summed cache-aware usage across all stages.
    pub fn cache_aware_total_usage(&self) -> CacheAwareStageUsage {
        self.cache_aware_snapshot().into_values().fold(
            CacheAwareStageUsage::default(),
            |mut acc, stage| {
                acc.merge(&stage);
                acc
            },
        )
    }
}

impl From<LlmStage> for OperatorStage {
    fn from(stage: LlmStage) -> Self {
        match stage {
            LlmStage::RetainExtract
            | LlmStage::RetainResolve
            | LlmStage::RetainGraph
            | LlmStage::RetainOpinion => OperatorStage::Retain,
            LlmStage::Reflect => OperatorStage::Reflect,
            LlmStage::Consolidate => OperatorStage::Consolidate,
            LlmStage::OpinionMerge => OperatorStage::OpinionMerge,
            LlmStage::Judge => OperatorStage::Judge,
        }
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
    use crate::types::llm::{CacheStatus, CompletionUsage};

    fn response_with_usage(
        prompt_tokens: usize,
        uncached_prompt_tokens: usize,
        cache_hit_prompt_tokens: usize,
        cache_write_prompt_tokens: usize,
        completion_tokens: usize,
        cache_status: CacheStatus,
    ) -> CompletionResponse {
        CompletionResponse {
            content: "ok".into(),
            input_tokens: prompt_tokens,
            output_tokens: completion_tokens,
            usage: CompletionUsage {
                prompt_tokens,
                uncached_prompt_tokens,
                cache_hit_prompt_tokens,
                cache_write_prompt_tokens,
                completion_tokens,
                cache_status,
            },
            stop_reason: None,
            tool_calls: vec![],
        }
    }

    #[test]
    fn cache_metrics_operator_stage_rollup_retain_substages() {
        let collector = MetricsCollector::new();
        collector.record_success(
            LlmStage::RetainExtract,
            &response_with_usage(12, 6, 4, 2, 3, CacheStatus::HitAndWrite),
            10,
        );
        collector.record_success(
            LlmStage::RetainResolve,
            &response_with_usage(8, 8, 0, 0, 2, CacheStatus::NoActivity),
            20,
        );
        collector.record_success(
            LlmStage::RetainGraph,
            &response_with_usage(4, 4, 0, 0, 1, CacheStatus::Unsupported),
            30,
        );
        collector.record_success(
            LlmStage::RetainOpinion,
            &response_with_usage(6, 5, 1, 0, 4, CacheStatus::Hit),
            40,
        );

        let operator_snapshot = collector.operator_snapshot();
        let retain = operator_snapshot
            .get(&OperatorStage::Retain)
            .expect("retain operator stage present");

        assert_eq!(
            retain,
            &CacheAwareStageUsage {
                prompt_tokens: 30,
                uncached_prompt_tokens: 23,
                cache_hit_prompt_tokens: 5,
                cache_write_prompt_tokens: 2,
                completion_tokens: 10,
                calls: 4,
                errors: 0,
                latency_ms: 100,
                cache_supported_calls: 3,
                cache_hit_calls: 2,
                cache_write_calls: 1,
                cache_unsupported_calls: 1,
            }
        );
        assert_eq!(operator_snapshot.len(), 1);
    }

    #[test]
    fn cache_metrics_status_counters_distinguish_unsupported_from_no_activity() {
        let collector = MetricsCollector::new();
        collector.record_success(
            LlmStage::Reflect,
            &response_with_usage(10, 10, 0, 0, 2, CacheStatus::Unsupported),
            11,
        );
        collector.record_success(
            LlmStage::Reflect,
            &response_with_usage(10, 10, 0, 0, 2, CacheStatus::NoActivity),
            13,
        );

        let snapshot = collector.cache_aware_snapshot();
        let reflect = snapshot
            .get(&LlmStage::Reflect)
            .expect("reflect stage present");

        assert_eq!(reflect.calls, 2);
        assert_eq!(reflect.cache_supported_calls, 1);
        assert_eq!(reflect.cache_hit_calls, 0);
        assert_eq!(reflect.cache_write_calls, 0);
        assert_eq!(reflect.cache_unsupported_calls, 1);
    }

    #[test]
    fn cache_metrics_total_usage_matches_legacy_prompt_and_completion_totals() {
        let collector = MetricsCollector::new();
        collector.record_success(
            LlmStage::Reflect,
            &response_with_usage(18, 10, 8, 0, 7, CacheStatus::Hit),
            25,
        );
        collector.record_success(
            LlmStage::Judge,
            &response_with_usage(12, 7, 0, 5, 4, CacheStatus::WriteOnly),
            35,
        );
        collector.record_error(LlmStage::Judge, 5);

        let cache_total = collector.cache_aware_total_usage();
        let legacy_total = collector.total_usage();

        assert_eq!(cache_total.prompt_tokens, legacy_total.prompt_tokens);
        assert_eq!(
            cache_total.completion_tokens,
            legacy_total.completion_tokens
        );
        assert_eq!(cache_total.calls, legacy_total.calls);
        assert_eq!(cache_total.errors, legacy_total.errors);
        assert_eq!(cache_total.latency_ms, legacy_total.latency_ms);
        assert_eq!(cache_total.legacy_totals(), legacy_total);
    }

    #[test]
    fn cache_aware_snapshot_merge_preserves_hit_and_write_totals() {
        let collector = MetricsCollector::new();
        let mut snapshot = BTreeMap::new();
        snapshot.insert(
            LlmStage::Judge,
            CacheAwareStageUsage {
                prompt_tokens: 25,
                uncached_prompt_tokens: 9,
                cache_hit_prompt_tokens: 10,
                cache_write_prompt_tokens: 6,
                completion_tokens: 7,
                calls: 3,
                errors: 1,
                latency_ms: 40,
                cache_supported_calls: 3,
                cache_hit_calls: 2,
                cache_write_calls: 1,
                cache_unsupported_calls: 0,
            },
        );

        collector.extend_cache_aware_snapshot(&snapshot);

        let cache_snapshot = collector.cache_aware_snapshot();
        let judge = cache_snapshot
            .get(&LlmStage::Judge)
            .expect("judge stage present");

        assert_eq!(
            judge,
            &CacheAwareStageUsage {
                prompt_tokens: 25,
                uncached_prompt_tokens: 9,
                cache_hit_prompt_tokens: 10,
                cache_write_prompt_tokens: 6,
                completion_tokens: 7,
                calls: 3,
                errors: 1,
                latency_ms: 40,
                cache_supported_calls: 3,
                cache_hit_calls: 2,
                cache_write_calls: 1,
                cache_unsupported_calls: 0,
            }
        );
        assert_eq!(
            collector.snapshot().get(&LlmStage::Judge),
            Some(&judge.legacy_totals())
        );
    }

    #[test]
    fn cache_aware_or_legacy_snapshot_falls_back_to_unsupported_calls() {
        let collector = MetricsCollector::new();
        let mut legacy_snapshot = BTreeMap::new();
        legacy_snapshot.insert(
            LlmStage::Reflect,
            StageUsage {
                prompt_tokens: 14,
                completion_tokens: 5,
                calls: 2,
                errors: 1,
                latency_ms: 21,
            },
        );

        collector.extend_cache_aware_or_legacy_snapshot(&BTreeMap::new(), &legacy_snapshot);

        assert_eq!(collector.snapshot(), legacy_snapshot);
        assert_eq!(
            collector.cache_aware_snapshot().get(&LlmStage::Reflect),
            Some(&CacheAwareStageUsage {
                prompt_tokens: 14,
                uncached_prompt_tokens: 0,
                cache_hit_prompt_tokens: 0,
                cache_write_prompt_tokens: 0,
                completion_tokens: 5,
                calls: 2,
                errors: 1,
                latency_ms: 21,
                cache_supported_calls: 0,
                cache_hit_calls: 0,
                cache_write_calls: 0,
                cache_unsupported_calls: 2,
            })
        );
    }
}
