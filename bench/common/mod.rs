pub mod failure;
pub mod fingerprint;
pub mod io;
pub mod judge;

use elephant::llm::ReasoningEffort;
use elephant::runtime::RuntimeTuning as ElephantRuntimeTuning;

pub use fingerprint::{fnv1a64, fnv1a64_hex};
pub use io::{append_jsonl, sidecar_path};

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
