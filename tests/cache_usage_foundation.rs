use elephant::metrics::{LlmStage, MetricsCollector, OperatorStage};
use elephant::types::llm::{CacheStatus, CompletionResponse, CompletionUsage};

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
fn legacy_totals_adapter_preserves_existing_stage_usage() {
    let collector = MetricsCollector::new();
    collector.record_success(
        LlmStage::Reflect,
        &response_with_usage(24, 14, 10, 0, 6, CacheStatus::Hit),
        15,
    );
    collector.record_success(
        LlmStage::Judge,
        &response_with_usage(18, 9, 0, 9, 5, CacheStatus::WriteOnly),
        20,
    );
    collector.record_error(LlmStage::Judge, 5);

    let snapshot = collector.snapshot();
    let cache_snapshot = collector.cache_aware_snapshot();

    for stage in [LlmStage::Reflect, LlmStage::Judge] {
        let legacy = snapshot.get(&stage).expect("legacy stage present");
        let cache_aware = cache_snapshot
            .get(&stage)
            .expect("cache-aware stage present");
        assert_eq!(cache_aware.legacy_totals(), *legacy);
    }

    assert_eq!(
        collector.cache_aware_total_usage().legacy_totals(),
        collector.total_usage()
    );
}

#[test]
fn cache_usage_foundation_rolls_retain_into_operator_stage() {
    let collector = MetricsCollector::new();
    collector.record_success(
        LlmStage::RetainExtract,
        &response_with_usage(7, 4, 2, 1, 2, CacheStatus::HitAndWrite),
        10,
    );
    collector.record_success(
        LlmStage::RetainResolve,
        &response_with_usage(5, 5, 0, 0, 3, CacheStatus::NoActivity),
        11,
    );
    collector.record_success(
        LlmStage::RetainGraph,
        &response_with_usage(9, 6, 3, 0, 4, CacheStatus::Hit),
        12,
    );
    collector.record_success(
        LlmStage::RetainOpinion,
        &response_with_usage(4, 2, 0, 2, 1, CacheStatus::WriteOnly),
        13,
    );

    let operator_snapshot = collector.operator_snapshot();
    assert_eq!(operator_snapshot.len(), 1);
    assert_eq!(
        operator_snapshot.get(&OperatorStage::Retain),
        Some(&elephant::metrics::CacheAwareStageUsage {
            prompt_tokens: 25,
            uncached_prompt_tokens: 17,
            cache_hit_prompt_tokens: 5,
            cache_write_prompt_tokens: 3,
            completion_tokens: 10,
            calls: 4,
            errors: 0,
            latency_ms: 46,
            cache_supported_calls: 4,
            cache_hit_calls: 2,
            cache_write_calls: 2,
            cache_unsupported_calls: 0,
        })
    );
}

#[test]
fn cache_usage_foundation_distinguishes_unsupported_and_zero_hit_calls() {
    let collector = MetricsCollector::new();
    collector.record_success(
        LlmStage::Reflect,
        &response_with_usage(10, 10, 0, 0, 3, CacheStatus::Unsupported),
        8,
    );
    collector.record_success(
        LlmStage::Reflect,
        &response_with_usage(10, 10, 0, 0, 3, CacheStatus::NoActivity),
        8,
    );

    let reflect = collector
        .cache_aware_snapshot()
        .get(&LlmStage::Reflect)
        .cloned()
        .expect("reflect stage present");

    assert_eq!(reflect.prompt_tokens, 20);
    assert_eq!(reflect.completion_tokens, 6);
    assert_eq!(reflect.cache_unsupported_calls, 1);
    assert_eq!(reflect.cache_supported_calls, 1);
    assert_eq!(reflect.cache_hit_calls, 0);
}

#[test]
fn cache_usage_foundation_preserves_legacy_artifact_contracts() {
    let locomo = include_str!("../bench/locomo/locomo.rs");
    let longmemeval = include_str!("../bench/longmemeval/longmemeval.rs");
    let longmemeval_view = include_str!("../bench/longmemeval/view.rs");

    assert!(locomo.contains("stage_metrics: BTreeMap<LlmStage, StageUsage>"));
    assert!(locomo.contains("total_stage_usage: StageUsage"));
    assert!(longmemeval.contains("stage_metrics: BTreeMap<LlmStage, StageUsage>"));
    assert!(longmemeval_view.contains("input_tokens: u64"));
    assert!(longmemeval_view.contains("output_tokens: u64"));
    assert!(longmemeval_view.contains("requests: u64"));
}
