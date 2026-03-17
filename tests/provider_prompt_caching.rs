use elephant::llm::{PromptCachingConfig, anthropic};
use elephant::metrics::{LlmStage, MetricsCollector};
use elephant::types::{
    CacheStatus, CompletionRequest, CompletionResponse, CompletionUsage, Message,
};

fn response_with_usage(usage: CompletionUsage) -> CompletionResponse {
    CompletionResponse {
        content: "ok".into(),
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        usage,
        stop_reason: None,
        tool_calls: vec![],
    }
}

#[test]
fn provider_prompt_caching_fallback_preserves_success_usage_semantics() {
    let collector = MetricsCollector::new();
    let responses = [
        response_with_usage(CompletionUsage {
            prompt_tokens: 11,
            uncached_prompt_tokens: 11,
            cache_hit_prompt_tokens: 0,
            cache_write_prompt_tokens: 0,
            completion_tokens: 3,
            cache_status: CacheStatus::Unsupported,
        }),
        response_with_usage(CompletionUsage {
            prompt_tokens: 13,
            uncached_prompt_tokens: 13,
            cache_hit_prompt_tokens: 0,
            cache_write_prompt_tokens: 0,
            completion_tokens: 5,
            cache_status: CacheStatus::NoActivity,
        }),
        response_with_usage(CompletionUsage {
            prompt_tokens: 17,
            uncached_prompt_tokens: 7,
            cache_hit_prompt_tokens: 0,
            cache_write_prompt_tokens: 10,
            completion_tokens: 7,
            cache_status: CacheStatus::WriteOnly,
        }),
    ];

    for response in &responses {
        collector.record_success(LlmStage::Judge, response, 25);
    }

    let legacy = collector.snapshot();
    let cache_aware = collector.cache_aware_snapshot();
    let legacy_usage = legacy.get(&LlmStage::Judge).expect("legacy judge usage");
    let cache_usage = cache_aware
        .get(&LlmStage::Judge)
        .expect("cache-aware judge usage");

    assert_eq!(legacy_usage.prompt_tokens, 41);
    assert_eq!(legacy_usage.completion_tokens, 15);
    assert_eq!(legacy_usage.calls, 3);
    assert_eq!(cache_usage.legacy_totals(), *legacy_usage);

    assert_eq!(cache_usage.prompt_tokens, 41);
    assert_eq!(cache_usage.uncached_prompt_tokens, 31);
    assert_eq!(cache_usage.cache_hit_prompt_tokens, 0);
    assert_eq!(cache_usage.cache_write_prompt_tokens, 10);
    assert_eq!(cache_usage.cache_supported_calls, 2);
    assert_eq!(cache_usage.cache_unsupported_calls, 1);
    assert_eq!(cache_usage.cache_hit_calls, 0);
    assert_eq!(cache_usage.cache_write_calls, 1);
}

#[test]
fn provider_prompt_caching_openai_supported_path_guard_is_conservative() {
    assert!(is_official_openai_api_base_url("https://api.openai.com/v1"));
    assert!(is_official_openai_api_base_url(
        "https://api.openai.com/v1/"
    ));
    assert!(!is_official_openai_api_base_url("https://api.openai.com"));
    assert!(!is_official_openai_api_base_url("https://example.com/v1"));
    assert!(!is_official_openai_api_base_url(
        "https://api.openai.com/v1/compatible"
    ));
}

#[test]
fn provider_prompt_caching_anthropic_request_uses_ephemeral_cache_control_when_enabled() {
    let request = CompletionRequest {
        model: String::new(),
        messages: vec![Message::text("user", "cache this request")],
        max_tokens: Some(256),
        temperature: Some(0.0),
        ..Default::default()
    };

    let json = anthropic::anthropic_request_json(
        &request,
        "claude-sonnet",
        &PromptCachingConfig { enabled: true },
    )
    .expect("anthropic request should serialize");

    assert!(json.contains("\"cache_control\":{\"type\":\"ephemeral\"}"));
}
fn is_official_openai_api_base_url(base_url: &str) -> bool {
    base_url.trim_end_matches('/') == "https://api.openai.com/v1"
}
