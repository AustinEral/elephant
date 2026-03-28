//! Transitional typed config API for runtime, server, and benchmark startup.

mod bench;
mod env;
mod error;
mod runtime;
mod server;

pub use bench::BenchConfig;
pub use error::{ConfigError, ConfigErrorKind, Result as ConfigResult};
pub use runtime::{ReflectConfig, RetrievalConfig, RuntimeConfig};
pub use server::{BackgroundConsolidationConfig, LogFormat, ServerConfig};
