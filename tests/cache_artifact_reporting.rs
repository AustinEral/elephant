use std::collections::BTreeMap;
use std::fs;

use elephant::metrics::{CacheAwareStageUsage, LlmStage, MetricsCollector, StageUsage};

#[path = "../bench/view.rs"]
mod locomo_view;

#[path = "../bench/longmemeval/view.rs"]
mod longmemeval_view;

fn legacy_snapshot_fixture() -> BTreeMap<LlmStage, StageUsage> {
    BTreeMap::from([(
        LlmStage::Judge,
        StageUsage {
            prompt_tokens: 20,
            completion_tokens: 4,
            calls: 2,
            errors: 1,
            latency_ms: 17,
        },
    )])
}

fn cache_aware_snapshot_fixture() -> BTreeMap<LlmStage, CacheAwareStageUsage> {
    BTreeMap::from([(
        LlmStage::Judge,
        CacheAwareStageUsage {
            prompt_tokens: 28,
            uncached_prompt_tokens: 8,
            cache_hit_prompt_tokens: 12,
            cache_write_prompt_tokens: 8,
            completion_tokens: 6,
            calls: 2,
            errors: 0,
            latency_ms: 19,
            cache_supported_calls: 2,
            cache_hit_calls: 1,
            cache_write_calls: 1,
            cache_unsupported_calls: 0,
        },
    )])
}

fn merge_harness(
    cache_aware: &BTreeMap<LlmStage, CacheAwareStageUsage>,
    legacy: &BTreeMap<LlmStage, StageUsage>,
) -> MetricsCollector {
    let collector = MetricsCollector::new();
    collector.extend_cache_aware_or_legacy_snapshot(cache_aware, legacy);
    collector
}

fn locomo_summary(
    dataset_fingerprint: &str,
    prompt_hashes: serde_json::Value,
    cache_total: serde_json::Value,
    cache_stage_metrics: serde_json::Value,
) -> String {
    serde_json::json!({
        "tag": "locomo-run",
        "judge_model": "judge-model",
        "retain_model": "retain-model",
        "reflect_model": "reflect-model",
        "embedding_model": "embed-model",
        "reranker_model": "reranker-model",
        "consolidation_strategy": "end",
        "total_questions": 1,
        "accuracy": 1.0,
        "mean_f1": 1.0,
        "mean_evidence_recall": 1.0,
        "manifest": {
            "dataset_fingerprint": dataset_fingerprint,
            "prompt_hashes": prompt_hashes
        },
        "cache_aware_total_stage_usage": cache_total,
        "cache_aware_stage_metrics": cache_stage_metrics,
        "results": [],
        "total_time_s": 1.0
    })
    .to_string()
}

fn locomo_comparable_pair() -> (String, String) {
    let prompt_hashes = serde_json::json!({
        "judge": "judge-hash",
        "retain_extract": "retain-hash",
        "reflect_agent": "reflect-hash",
        "consolidate": "consolidate-hash"
    });
    let baseline = locomo_summary(
        "dataset-v1",
        prompt_hashes.clone(),
        serde_json::json!({
            "prompt_tokens": 100,
            "uncached_prompt_tokens": 80,
            "cache_hit_prompt_tokens": 10,
            "cache_write_prompt_tokens": 10,
            "completion_tokens": 20,
            "calls": 2,
            "cache_supported_calls": 2,
            "cache_hit_calls": 1,
            "cache_write_calls": 1
        }),
        serde_json::json!({
            "retain_extract": {
                "prompt_tokens": 60,
                "uncached_prompt_tokens": 50,
                "cache_hit_prompt_tokens": 5,
                "cache_write_prompt_tokens": 5,
                "completion_tokens": 12,
                "calls": 1,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 1
            },
            "reflect": {
                "prompt_tokens": 40,
                "uncached_prompt_tokens": 30,
                "cache_hit_prompt_tokens": 5,
                "cache_write_prompt_tokens": 5,
                "completion_tokens": 8,
                "calls": 1,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 1
            }
        }),
    );
    let warm = locomo_summary(
        "dataset-v1",
        prompt_hashes,
        serde_json::json!({
            "prompt_tokens": 100,
            "uncached_prompt_tokens": 50,
            "cache_hit_prompt_tokens": 45,
            "cache_write_prompt_tokens": 5,
            "completion_tokens": 20,
            "calls": 2,
            "cache_supported_calls": 2,
            "cache_hit_calls": 2,
            "cache_write_calls": 1
        }),
        serde_json::json!({
            "retain_extract": {
                "prompt_tokens": 60,
                "uncached_prompt_tokens": 30,
                "cache_hit_prompt_tokens": 25,
                "cache_write_prompt_tokens": 5,
                "completion_tokens": 12,
                "calls": 1,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 1
            },
            "reflect": {
                "prompt_tokens": 40,
                "uncached_prompt_tokens": 20,
                "cache_hit_prompt_tokens": 20,
                "cache_write_prompt_tokens": 0,
                "completion_tokens": 8,
                "calls": 1,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 0
            }
        }),
    );
    (baseline, warm)
}

fn locomo_mismatched_prompt_hash_pair() -> (String, String) {
    let (baseline, _) = locomo_comparable_pair();
    let mismatch = locomo_summary(
        "dataset-v1",
        serde_json::json!({
            "judge": "judge-hash",
            "retain_extract": "retain-hash",
            "reflect_agent": "other-reflect-hash",
            "consolidate": "consolidate-hash"
        }),
        serde_json::json!({
            "prompt_tokens": 100,
            "uncached_prompt_tokens": 50,
            "cache_hit_prompt_tokens": 45,
            "cache_write_prompt_tokens": 5,
            "completion_tokens": 20,
            "calls": 2,
            "cache_supported_calls": 2,
            "cache_hit_calls": 2,
            "cache_write_calls": 1
        }),
        serde_json::json!({
            "reflect": {
                "prompt_tokens": 40,
                "uncached_prompt_tokens": 20,
                "cache_hit_prompt_tokens": 20,
                "cache_write_prompt_tokens": 0,
                "completion_tokens": 8,
                "calls": 1,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 0
            }
        }),
    );
    (baseline, mismatch)
}

fn longmemeval_summary(
    dataset_fingerprint: &str,
    prompt_hashes: serde_json::Value,
    cache_stage_metrics: serde_json::Value,
) -> String {
    serde_json::json!({
        "benchmark": "longmemeval",
        "timestamp": "2026-03-17T00:00:00Z",
        "retain_model": "retain-model",
        "reflect_model": "reflect-model",
        "embedding_model": "embed-model",
        "reranker_model": "reranker-model",
        "judge_model": "judge-model",
        "consolidation_strategy": "end",
        "total_questions": 1,
        "accuracy": 1.0,
        "per_category": {},
        "banks": {},
        "manifest": {
            "dataset_fingerprint": dataset_fingerprint,
            "prompt_hashes": prompt_hashes
        },
        "cache_aware_stage_metrics": cache_stage_metrics,
        "total_time_s": 1.0
    })
    .to_string()
}

fn longmemeval_prompt_hashes() -> serde_json::Value {
    serde_json::json!({
        "judge": "judge-hash",
        "retain_extract": "retain-extract-hash",
        "retain_resolve_system": "retain-resolve-system-hash",
        "retain_resolve_user": "retain-resolve-user-hash",
        "retain_graph_system": "retain-graph-system-hash",
        "retain_graph_user": "retain-graph-user-hash",
        "retain_opinion": "retain-opinion-hash",
        "reflect_agent": "reflect-agent-hash",
        "consolidate": "consolidate-hash",
        "opinion_merge": "opinion-merge-hash"
    })
}

fn longmemeval_comparable_pair() -> (String, String) {
    let prompt_hashes = longmemeval_prompt_hashes();
    let baseline = longmemeval_summary(
        "dataset-v1",
        prompt_hashes.clone(),
        serde_json::json!({
            "retain_extract": {
                "prompt_tokens": 60,
                "uncached_prompt_tokens": 50,
                "cache_hit_prompt_tokens": 5,
                "cache_write_prompt_tokens": 5,
                "completion_tokens": 10,
                "calls": 1,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 1
            },
            "reflect": {
                "prompt_tokens": 40,
                "uncached_prompt_tokens": 30,
                "cache_hit_prompt_tokens": 5,
                "cache_write_prompt_tokens": 5,
                "completion_tokens": 6,
                "calls": 1,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 1
            }
        }),
    );
    let warm = longmemeval_summary(
        "dataset-v1",
        prompt_hashes,
        serde_json::json!({
            "retain_extract": {
                "prompt_tokens": 60,
                "uncached_prompt_tokens": 30,
                "cache_hit_prompt_tokens": 25,
                "cache_write_prompt_tokens": 5,
                "completion_tokens": 10,
                "calls": 1,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 1
            },
            "reflect": {
                "prompt_tokens": 40,
                "uncached_prompt_tokens": 20,
                "cache_hit_prompt_tokens": 20,
                "cache_write_prompt_tokens": 0,
                "completion_tokens": 6,
                "calls": 1,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 0
            }
        }),
    );
    (baseline, warm)
}

fn longmemeval_incomplete_prompt_hash_pair() -> (String, String) {
    let (baseline, _) = longmemeval_comparable_pair();
    let mut prompt_hashes = longmemeval_prompt_hashes();
    prompt_hashes
        .as_object_mut()
        .expect("longmemeval prompt hashes should be an object")
        .remove("retain_graph_user");
    let incomplete = longmemeval_summary(
        "dataset-v1",
        prompt_hashes,
        serde_json::json!({
            "retain_extract": {
                "prompt_tokens": 60,
                "uncached_prompt_tokens": 30,
                "cache_hit_prompt_tokens": 25,
                "cache_write_prompt_tokens": 5,
                "completion_tokens": 10,
                "calls": 1,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 1
            },
            "reflect": {
                "prompt_tokens": 40,
                "uncached_prompt_tokens": 20,
                "cache_hit_prompt_tokens": 20,
                "cache_write_prompt_tokens": 0,
                "completion_tokens": 6,
                "calls": 1,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 0
            }
        }),
    );
    (baseline, incomplete)
}

fn legacy_locomo_pair() -> (String, String) {
    let summary = serde_json::json!({
        "tag": "legacy",
        "judge_model": "judge-model",
        "retain_model": "retain-model",
        "reflect_model": "reflect-model",
        "embedding_model": "embed-model",
        "reranker_model": "reranker-model",
        "consolidation_strategy": "end",
        "total_questions": 1,
        "accuracy": 1.0,
        "mean_f1": 1.0,
        "results": [],
        "total_time_s": 1.0
    })
    .to_string();
    (summary.clone(), summary)
}

#[test]
fn cache_artifact_merge_prefers_cache_aware_snapshot() {
    let collector = merge_harness(&cache_aware_snapshot_fixture(), &legacy_snapshot_fixture());
    let snapshot = collector.cache_aware_snapshot();
    let judge = snapshot.get(&LlmStage::Judge).expect("judge stage present");

    assert_eq!(judge.prompt_tokens, 28);
    assert_eq!(judge.uncached_prompt_tokens, 8);
    assert_eq!(judge.cache_hit_prompt_tokens, 12);
    assert_eq!(judge.cache_write_prompt_tokens, 8);
    assert_eq!(judge.completion_tokens, 6);
    assert_eq!(judge.calls, 2);
    assert_eq!(judge.cache_hit_calls, 1);
    assert_eq!(judge.cache_write_calls, 1);
    assert_eq!(judge.cache_unsupported_calls, 0);
    assert_eq!(
        collector.snapshot().get(&LlmStage::Judge),
        Some(&StageUsage {
            prompt_tokens: 28,
            completion_tokens: 6,
            calls: 2,
            errors: 0,
            latency_ms: 19,
        })
    );
}

#[test]
fn cache_artifact_merge_falls_back_to_legacy_snapshot() {
    let collector = merge_harness(&Default::default(), &legacy_snapshot_fixture());
    let snapshot = collector.snapshot();

    assert_eq!(
        snapshot.get(&LlmStage::Judge),
        Some(&StageUsage {
            prompt_tokens: 20,
            completion_tokens: 4,
            calls: 2,
            errors: 1,
            latency_ms: 17,
        })
    );

    let cache_snapshot = collector.cache_aware_snapshot();
    let judge = cache_snapshot.get(&LlmStage::Judge).expect("judge stage present");
    assert_eq!(judge.prompt_tokens, 20);
    assert_eq!(judge.cache_hit_prompt_tokens, 0);
    assert_eq!(judge.cache_write_prompt_tokens, 0);
    assert_eq!(judge.cache_hit_calls, 0);
    assert_eq!(judge.cache_write_calls, 0);
    assert_eq!(judge.cache_unsupported_calls, 2);
}

#[test]
fn cache_artifact_merge_longmemeval_mixed_inputs() {
    let collector = MetricsCollector::new();
    let cache_aware = cache_aware_snapshot_fixture();
    let legacy = legacy_snapshot_fixture();

    collector.extend_cache_aware_or_legacy_snapshot(&cache_aware, &Default::default());
    collector.extend_cache_aware_or_legacy_snapshot(&Default::default(), &legacy);

    let snapshot = collector.cache_aware_snapshot();
    let judge = snapshot.get(&LlmStage::Judge).expect("judge stage present");

    assert_eq!(judge.cache_hit_prompt_tokens, 12);
    assert_eq!(judge.cache_write_prompt_tokens, 8);
    assert_eq!(judge.cache_hit_calls, 1);
    assert_eq!(judge.cache_write_calls, 1);
    assert_eq!(judge.cache_unsupported_calls, 2);
    assert_eq!(
        collector.snapshot().get(&LlmStage::Judge),
        Some(&StageUsage {
            prompt_tokens: 48,
            completion_tokens: 10,
            calls: 4,
            errors: 1,
            latency_ms: 36,
        })
    );
}

#[test]
fn cache_artifact_reader_compat_longmemeval_view() {
    let dir = std::env::temp_dir().join(format!(
        "cache-artifact-reader-compat-{}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    let artifact = dir.join("longmemeval.json");

    let summary = serde_json::json!({
        "benchmark": "longmemeval",
        "timestamp": "2026-03-16T00:00:00Z",
        "retain_model": "m1",
        "reflect_model": "m2",
        "embedding_model": "m3",
        "reranker_model": "m4",
        "judge_model": "judge",
        "consolidation_strategy": "end",
        "total_questions": 1,
        "accuracy": 1.0,
        "per_category": {},
        "banks": {},
        "stage_metrics": {
            "reflect": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "calls": 1,
                "errors": 0,
                "latency_ms": 5
            }
        },
        "cache_aware_stage_metrics": {
            "reflect": {
                "prompt_tokens": 10,
                "uncached_prompt_tokens": 4,
                "cache_hit_prompt_tokens": 6,
                "cache_write_prompt_tokens": 0,
                "completion_tokens": 2,
                "calls": 1,
                "errors": 0,
                "latency_ms": 5,
                "cache_supported_calls": 1,
                "cache_hit_calls": 1,
                "cache_write_calls": 0,
                "cache_unsupported_calls": 0
            }
        },
        "total_time_s": 3.0
    });
    fs::write(&artifact, serde_json::to_string_pretty(&summary).unwrap()).unwrap();

    longmemeval_view::parse_summary_artifact(
        &fs::read_to_string(&artifact).unwrap(),
    )
    .expect("new-style longmemeval summary should deserialize");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn cache_aware_compare_view_shows_savings_for_same_model_pair() {
    let (baseline, warm) = locomo_comparable_pair();
    let rendered = locomo_view::render_compare_artifacts(
        &baseline,
        &warm,
        "baseline.json",
        "warm.json",
    )
    .expect("locomo compare should render");

    assert!(rendered.contains("cache-aware comparison"));
    assert!(rendered.contains("cache savings verification"));
    assert!(rendered.contains("cache savings visible"));
}

#[test]
fn cache_aware_compare_view_blocks_verification_for_mismatched_prompt_hashes() {
    let (baseline, mismatch) = locomo_mismatched_prompt_hash_pair();
    let rendered = locomo_view::render_compare_artifacts(
        &baseline,
        &mismatch,
        "baseline.json",
        "mismatch.json",
    )
    .expect("locomo compare should render");

    assert!(rendered.contains("cache-aware comparison"));
    assert!(rendered.contains("cache savings verification"));
    assert!(rendered.contains("verification unavailable:"));
    assert!(rendered.contains("verification unavailable: prompt hashes mismatch"));
}

#[test]
fn cache_aware_compare_longmemeval_shows_savings_for_same_model_pair() {
    let (baseline, warm) = longmemeval_comparable_pair();
    let rendered = longmemeval_view::render_compare_artifacts(
        &baseline,
        &warm,
        "baseline.json",
        "warm.json",
    )
    .expect("longmemeval compare should render");

    assert!(rendered.contains("cache-aware comparison"));
    assert!(rendered.contains("cache savings verification"));
    assert!(rendered.contains("cache savings visible"));
}

#[test]
fn cache_aware_compare_longmemeval_blocks_verification_for_incomplete_prompt_hashes() {
    let (baseline, incomplete) = longmemeval_incomplete_prompt_hash_pair();
    let rendered = longmemeval_view::render_compare_artifacts(
        &baseline,
        &incomplete,
        "baseline.json",
        "incomplete.json",
    )
    .expect("longmemeval compare should render");

    assert!(rendered.contains("cache-aware comparison"));
    assert!(rendered.contains("cache savings verification"));
    assert!(rendered.contains("verification unavailable:"));
}

#[test]
fn cache_aware_compare_legacy_artifacts_remain_parseable() {
    let (legacy_a, legacy_b) = legacy_locomo_pair();
    let rendered = locomo_view::render_compare_artifacts(
        &legacy_a,
        &legacy_b,
        "legacy-a.json",
        "legacy-b.json",
    )
    .expect("legacy locomo compare should render");

    assert!(rendered.contains("cache-aware comparison"));
    assert!(rendered.contains("cache savings verification"));
    assert!(rendered.contains("verification unavailable:"));
}
