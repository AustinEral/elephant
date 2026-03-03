//! Entry point for the elephant memory engine API server.

use std::env;
use std::sync::Arc;

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

use elephant::consolidation::{DefaultConsolidator, DefaultOpinionMerger};
use elephant::embedding::{self, EmbeddingClient, EmbeddingConfig, EmbeddingProvider};
use elephant::llm::anthropic::AnthropicClient;
use elephant::llm::openai::OpenAiClient;
use elephant::llm::retry::{RetryPolicy, RetryingLlmClient};
use elephant::llm::LlmClient;
use elephant::recall::budget::EstimateTokenizer;
use elephant::recall::graph::{GraphRetriever, GraphRetrieverConfig};
use elephant::recall::keyword::KeywordRetriever;
use elephant::recall::reranker::{self, RerankerConfig, RerankerProvider};
use elephant::recall::semantic::SemanticRetriever;
use elephant::recall::temporal::TemporalRetriever;
use elephant::recall::DefaultRecallPipeline;
use elephant::reflect::DefaultReflectPipeline;
use elephant::retain::chunker::SimpleChunker;
use elephant::retain::extractor::LlmFactExtractor;
use elephant::retain::graph_builder::{DefaultGraphBuilder, GraphConfig};
use elephant::retain::resolver::LayeredEntityResolver;
use elephant::retain::DefaultRetainPipeline;
use elephant::mcp::ElephantMcp;
use elephant::server::{self, AppState};
use elephant::storage::pg::PgMemoryStore;
use elephant::types::ChunkConfig;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::StreamableHttpService;

fn make_llm(provider: &str, api_key: &str, model: &str, base_url: Option<String>) -> Box<dyn LlmClient> {
    match provider {
        "openai" => Box::new(OpenAiClient::new(api_key.into(), model.into(), base_url)),
        _ => Box::new(AnthropicClient::new(api_key.into(), model.into())),
    }
}

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    // Init tracing subscriber: LOG_FORMAT=json for machine-readable, default for human-readable
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn"));
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

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let listen_addr = env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:3001".into());
    let llm_provider = env::var("LLM_PROVIDER").expect("LLM_PROVIDER must be set");
    let llm_api_key = env::var("LLM_API_KEY").expect("LLM_API_KEY must be set");
    let retain_model = env::var("RETAIN_LLM_MODEL")
        .or_else(|_| env::var("LLM_MODEL"))
        .expect("RETAIN_LLM_MODEL or LLM_MODEL must be set");
    let reflect_model = env::var("REFLECT_LLM_MODEL")
        .or_else(|_| env::var("LLM_MODEL"))
        .expect("REFLECT_LLM_MODEL or LLM_MODEL must be set");
    let llm_base_url = env::var("LLM_BASE_URL").ok();

    // 1. Storage — PgPool is internally Arc'd, cheap to clone
    let pool = sqlx::PgPool::connect(&database_url)
        .await
        .expect("failed to connect to database");
    let store = Arc::new(PgMemoryStore::new(pool.clone()));
    store.migrate().await.expect("failed to run migrations");

    // 2. LLM clients — retain (extraction) and reflect/consolidation (synthesis) can use
    //    different models. RETAIN_LLM_MODEL and REFLECT_LLM_MODEL override LLM_MODEL.
    let retry_policy = RetryPolicy::default();
    let retain_llm: Arc<dyn LlmClient> = Arc::new(RetryingLlmClient::new(
        Arc::from(make_llm(&llm_provider, &llm_api_key, &retain_model, llm_base_url.clone())),
        retry_policy.clone(),
    ));
    let reflect_llm: Arc<dyn LlmClient> = Arc::new(RetryingLlmClient::new(
        Arc::from(make_llm(&llm_provider, &llm_api_key, &reflect_model, llm_base_url.clone())),
        retry_policy,
    ));
    let emb_config = EmbeddingConfig {
        provider: match env::var("EMBEDDING_PROVIDER").expect("EMBEDDING_PROVIDER must be set").as_str() {
            "openai" => EmbeddingProvider::OpenAi,
            "local" => EmbeddingProvider::Local,
            other => panic!("unknown EMBEDDING_PROVIDER: {other} (expected 'local' or 'openai')"),
        },
        model_path: env::var("EMBEDDING_MODEL_PATH").ok(),
        api_key: env::var("EMBEDDING_API_KEY").ok(),
        model: env::var("EMBEDDING_API_MODEL").ok(),
        dimensions: env::var("EMBEDDING_API_DIMS").ok().and_then(|s| s.parse().ok()),
    };
    let embeddings: Arc<dyn EmbeddingClient> =
        Arc::from(embedding::build_client(&emb_config).expect("failed to create embedding client"));

    // 3. Retain pipeline — uses retain-tier LLM (extraction, can be a fast/cheap model)
    let dedup_threshold: Option<f32> = match env::var("DEDUP_THRESHOLD").as_deref() {
        Ok("none") => None,
        Ok(s) => Some(s.parse().expect("DEDUP_THRESHOLD must be a float or 'none'")),
        Err(_) => Some(0.95),
    };
    let retain = Arc::new(DefaultRetainPipeline::new(
        Box::new(SimpleChunker),
        Box::new(LlmFactExtractor::new(retain_llm.clone())),
        Box::new(LayeredEntityResolver::new(
            embedding::build_client(&emb_config).expect("embedding client"),
            retain_llm.clone(),
        )),
        Box::new(DefaultGraphBuilder::new(
            retain_llm.clone(),
            GraphConfig::default(),
        )),
        Box::new(PgMemoryStore::new(pool.clone())),
        embedding::build_client(&emb_config).expect("embedding client"),
        retain_llm.clone(),
        ChunkConfig {
            max_tokens: 512,
            overlap_tokens: 64,
            preserve_turns: true,
        },
        dedup_threshold,
    ));

    // 4. Reranker
    let reranker_config = RerankerConfig {
        provider: match env::var("RERANKER_PROVIDER").expect("RERANKER_PROVIDER must be set").as_str() {
            "local" => RerankerProvider::Local,
            "api" => RerankerProvider::Api,
            "none" => RerankerProvider::None,
            other => panic!("unknown RERANKER_PROVIDER: {other} (expected 'local', 'api', or 'none')"),
        },
        model_path: env::var("RERANKER_MODEL_PATH").ok(),
        max_seq_len: env::var("RERANKER_MAX_SEQ_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512),
        api_key: env::var("RERANKER_API_KEY").ok(),
        api_url: env::var("RERANKER_API_URL").ok(),
        api_model: env::var("RERANKER_API_MODEL").ok(),
    };
    let reranker = reranker::build_reranker(&reranker_config)
        .expect("failed to create reranker");

    // 5. Recall pipeline
    let retriever_limit: usize = env::var("RETRIEVER_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let rerank_top_n: usize = env::var("RERANK_TOP_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let recall = Arc::new(DefaultRecallPipeline::new(
        Box::new(SemanticRetriever::new(store.clone(), embeddings.clone(), retriever_limit)),
        Box::new(KeywordRetriever::new(store.clone(), retriever_limit)),
        Box::new(GraphRetriever::new(
            store.clone(),
            embeddings.clone(),
            GraphRetrieverConfig::default(),
        )),
        Box::new(TemporalRetriever::new(store.clone())),
        reranker,
        Box::new(EstimateTokenizer),
        60.0,
        rerank_top_n,
    ));

    // 5. Reflect pipeline — uses reflect-tier LLM (synthesis, quality-sensitive)
    let reflect_max_iter: usize = env::var("REFLECT_MAX_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let reflect = Arc::new(DefaultReflectPipeline::new(
        recall.clone(),
        reflect_llm.clone(),
        store.clone(),
        reflect_max_iter,
    ));

    // 6. Consolidation workers — synthesis like reflect, same tier
    let consolidator = Arc::new(DefaultConsolidator::new(
        store.clone(),
        reflect_llm.clone(),
        embeddings.clone(),
    ));
    let opinion_merger = Arc::new(DefaultOpinionMerger::new(
        store.clone(),
        reflect_llm.clone(),
        embeddings.clone(),
    ));
    // 7. Build app state and serve
    let state = AppState {
        info: server::ServerInfo {
            retain_model: format!("{llm_provider}/{retain_model}"),
            reflect_model: format!("{llm_provider}/{reflect_model}"),
            embedding_model: match emb_config.provider {
                EmbeddingProvider::OpenAi => format!("openai/{}", emb_config.model.clone().unwrap_or_default()),
                EmbeddingProvider::Local => {
                    let name = emb_config.model_path.as_deref()
                        .and_then(|p| std::path::Path::new(p).file_name())
                        .and_then(|f| f.to_str())
                        .unwrap_or("unknown");
                    format!("local/{name}")
                }
            },
            reranker_model: match reranker_config.provider {
                RerankerProvider::Local => {
                    let name = reranker_config.model_path.as_deref()
                        .and_then(|p| std::path::Path::new(p).file_name())
                        .and_then(|f| f.to_str())
                        .unwrap_or("unknown");
                    format!("local/{name}")
                }
                RerankerProvider::Api => format!("api/{}", reranker_config.api_model.as_deref().unwrap_or("unknown")),
                RerankerProvider::None => "none".into(),
            },
        },
        retain,
        recall,
        reflect,
        consolidator,
        opinion_merger,
        store,
        embeddings,
    };

    // 8. MCP server at /mcp
    let mcp_state = state.clone();
    let mcp_service = StreamableHttpService::new(
        move || Ok(ElephantMcp::new(mcp_state.clone())),
        LocalSessionManager::default().into(),
        Default::default(),
    );

    let app = server::router(state).nest_service("/mcp", mcp_service);
    let listener = tokio::net::TcpListener::bind(&listen_addr)
        .await
        .expect("failed to bind listener");
    println!("Listening on {listen_addr}");
    println!("  REST API: http://{listen_addr}/v1/");
    println!("  MCP:      http://{listen_addr}/mcp/");
    axum::serve(listener, app).await.expect("server error");
}
