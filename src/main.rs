//! Entry point for the elephant memory engine API server.

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

use elephant::config::{LogFormat, RuntimeConfig, ServerConfig};
use elephant::mcp::ElephantMcp;
use elephant::runtime::RuntimeBuilder;
use elephant::server::{self, AppHandle};
use rmcp::transport::StreamableHttpService;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let server_config = ServerConfig::from_env()?;

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if matches!(server_config.log_format(), LogFormat::Json) {
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

    let runtime = RuntimeBuilder::new(RuntimeConfig::from_env()?)
        .build()
        .await?;
    let app = AppHandle::new(&runtime, &server_config)?;

    let mcp_app = app.clone();
    let mcp_service = StreamableHttpService::new(
        move || Ok(ElephantMcp::new(mcp_app.clone())),
        LocalSessionManager::default().into(),
        Default::default(),
    );

    let app = server::router(app).nest_service("/mcp", mcp_service);
    let listener = tokio::net::TcpListener::bind(server_config.listen_addr()).await?;
    println!("Listening on {}", server_config.listen_addr());
    println!("  REST API: http://{}/v1/", server_config.listen_addr());
    println!("  MCP:      http://{}/mcp/", server_config.listen_addr());
    axum::serve(listener, app).await?;
    Ok(())
}
