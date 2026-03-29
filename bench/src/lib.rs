#![warn(missing_docs)]
//! Internal benchmark support crate for Elephant.

mod config;
mod harness;

pub use config::{BenchConfig, BenchJudgeConfig};
pub use harness::{
    BenchHarness, BenchHarnessBuilder, BenchRuntime, BenchRuntimeMetadata,
    BenchRuntimePromptHashes, BenchRuntimeTuning,
};
