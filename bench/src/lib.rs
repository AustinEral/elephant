#![warn(missing_docs)]
//! Internal benchmark support crate for Elephant.

pub mod config;
mod env;
mod harness;

pub use config::{ResolvedBenchConfig, resolve_locomo_bench_config};
pub use env::{BenchConfig, BenchJudgeConfig};
pub use harness::{
    BenchHarness, BenchHarnessBuilder, BenchRuntime, BenchRuntimeMetadata,
    BenchRuntimePromptHashes, BenchRuntimeTuning,
};
