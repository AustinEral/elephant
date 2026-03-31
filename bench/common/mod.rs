pub mod failure;
pub mod fingerprint;
pub mod io;
pub mod judge;

use elephant::RuntimeTuning as ElephantRuntimeTuning;
use elephant::llm::DeterminismRequirement;
use elephant::llm::ReasoningEffort;
use elephant_bench::BenchConfig;

#[allow(unused_imports)]
pub use fingerprint::{fnv1a64, fnv1a64_hex};
#[allow(unused_imports)]
pub use io::{append_jsonl, resolve_workspace_path, sidecar_path};

#[allow(dead_code)]
fn format_reasoning_effort(effort: Option<ReasoningEffort>) -> &'static str {
    match effort {
        Some(ReasoningEffort::Minimal) => "minimal",
        Some(ReasoningEffort::Low) => "low",
        Some(ReasoningEffort::Medium) => "medium",
        Some(ReasoningEffort::High) => "high",
        Some(ReasoningEffort::XHigh) => "xhigh",
        Some(ReasoningEffort::None) => "none",
        None => "default",
    }
}

#[allow(dead_code)]
pub fn format_reasoning_effort_summary(tuning: &ElephantRuntimeTuning) -> String {
    format!(
        "extract={} resolve={} graph={} reflect={} consolidate={} opinion_merge={}",
        format_reasoning_effort(tuning.retain_extract_reasoning_effort),
        format_reasoning_effort(tuning.retain_resolve_reasoning_effort),
        format_reasoning_effort(tuning.retain_graph_reasoning_effort),
        format_reasoning_effort(tuning.reflect_reasoning_effort),
        format_reasoning_effort(tuning.consolidate_reasoning_effort),
        format_reasoning_effort(tuning.opinion_merge_reasoning_effort),
    )
}

#[allow(dead_code)]
pub fn benchmark_determinism_requirement_from_env() -> Result<Option<DeterminismRequirement>, String>
{
    BenchConfig::from_env()
        .map(|config| config.determinism_requirement())
        .map_err(|err| err.to_string())
}

#[allow(dead_code)]
pub fn format_determinism_requirement(requirement: DeterminismRequirement) -> &'static str {
    requirement.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn parses_best_effort_requirement_from_env() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("BENCH_DETERMINISM_REQUIREMENT", "best_effort");
        }
        let parsed = benchmark_determinism_requirement_from_env().unwrap();
        assert_eq!(parsed, Some(DeterminismRequirement::BestEffort));
        unsafe {
            env::remove_var("BENCH_DETERMINISM_REQUIREMENT");
        }
    }

    #[test]
    fn rejects_invalid_requirement_from_env() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::set_var("BENCH_DETERMINISM_REQUIREMENT", "maybe");
        }
        let err = benchmark_determinism_requirement_from_env().unwrap_err();
        assert!(err.contains("BENCH_DETERMINISM_REQUIREMENT"));
        unsafe {
            env::remove_var("BENCH_DETERMINISM_REQUIREMENT");
        }
    }
}
