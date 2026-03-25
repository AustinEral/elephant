//! Entry point for the elephant memory engine API server.

use std::env;

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

use elephant::mcp::ElephantMcp;
use elephant::runtime::{BuildRuntimeOptions, build_runtime_from_env};
use elephant::server::{self, AppState};
use rmcp::transport::StreamableHttpService;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let log_format = env::var("LOG_FORMAT").unwrap_or_default();
    if log_format == "json" {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer())
            .init();
    }

    let listen_addr = env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:3001".into());
    let runtime = build_runtime_from_env(BuildRuntimeOptions::default()).await?;
    let state = AppState::try_from(&runtime)?;

    let mcp_state = state.clone();
    let mcp_service = StreamableHttpService::new(
        move || Ok(ElephantMcp::new(mcp_state.clone())),
        LocalSessionManager::default().into(),
        Default::default(),
    );

    let app = server::router(state).nest_service("/mcp", mcp_service);
    let listener = tokio::net::TcpListener::bind(&listen_addr).await?;
    println!("Listening on {listen_addr}");
    println!("  REST API: http://{listen_addr}/v1/");
    println!("  MCP:      http://{listen_addr}/mcp/");
    axum::serve(listener, app).await?;
    Ok(())
}
