#![warn(missing_docs)]
//! Elephant — a Rust implementation of the Hindsight memory architecture.

mod app;
mod bench_support;
mod config;
pub mod consolidation;
pub mod embedding;
pub mod error;
pub mod llm;
mod mcp;
pub mod metrics;
pub mod recall;
pub mod reflect;
pub mod retain;
mod runtime;
mod server;
pub mod storage;
pub mod types;
pub mod util;

pub use bench_support::{BenchHarness, BenchHarnessBuilder};
pub use config::{
    BackgroundConsolidationConfig, BenchConfig, BenchJudgeConfig, ConfigError, ConfigErrorKind,
    ConfigResult, LogFormat, ReflectConfig, RetrievalConfig, RuntimeConfig, ServerConfig,
};
pub use embedding::EmbeddingClient;
pub use error::{Error, Result};
pub use llm::LlmClient;
pub use mcp::ElephantMcp;
pub use runtime::{
    ElephantRuntime, RuntimeBuilder, RuntimeDeterminism, RuntimeInfo, RuntimePromptHashes,
    RuntimeTuning,
};
pub use server::{
    AppHandle, ServerBackgroundConsolidationInfo, ServerConsolidationRuntimeInfo, ServerInfo,
    ServerModelsInfo, ServerReflectInfo, ServerRetrievalInfo, router,
};
pub use storage::MemoryStore;
pub use types::*;
