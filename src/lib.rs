#![warn(missing_docs)]
//! Elephant — a Rust implementation of the Hindsight memory architecture.

mod app;
#[doc(hidden)]
pub mod bench_support;
#[doc(hidden)]
pub mod config;
pub mod consolidation;
pub mod embedding;
pub mod error;
pub mod llm;
pub mod mcp;
pub mod metrics;
pub mod recall;
pub mod reflect;
pub mod retain;
pub mod runtime;
pub mod server;
pub mod storage;
pub mod types;
pub mod util;

pub use embedding::EmbeddingClient;
pub use error::{Error, Result};
pub use llm::LlmClient;
pub use storage::MemoryStore;
pub use types::*;
