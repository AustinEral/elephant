//! Typed config API for runtime and server startup.

mod client_env;
mod env;
mod error;
mod runtime;
mod server;

pub use error::{ConfigError, ConfigErrorKind, Result as ConfigResult};
pub use runtime::{ReflectConfig, RetrievalConfig, RuntimeConfig};
pub use server::{BackgroundConsolidationConfig, LogFormat, ServerConfig};
