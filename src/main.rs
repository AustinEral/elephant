//! Entry point for the elephant memory engine API server.

use std::env;
use std::sync::Arc;

use elephant::consolidation::{
    DefaultConsolidator, DefaultMentalModelGenerator, DefaultOpinionMerger,
};
use elephant::embedding::{self, EmbeddingClient, EmbeddingConfig, EmbeddingProvider};
use elephant::llm::anthropic::AnthropicClient;
use elephant::llm::openai::OpenAiClient;
use elephant::llm::LlmClient;
use elephant::recall::budget::EstimateTokenizer;
use elephant::recall::graph::{GraphRetriever, GraphRetrieverConfig};
use elephant::recall::keyword::KeywordRetriever;
use elephant::recall::reranker::NoOpReranker;
use elephant::recall::semantic::SemanticRetriever;
use elephant::recall::temporal::TemporalRetriever;
use elephant::recall::DefaultRecallPipeline;
use elephant::reflect::hierarchy::DefaultHierarchyAssembler;
use elephant::reflect::opinion::DefaultOpinionManager;
use elephant::reflect::DefaultReflectPipeline;
use elephant::retain::chunker::SimpleChunker;
use elephant::retain::extractor::LlmFactExtractor;
use elephant::retain::graph_builder::{DefaultGraphBuilder, GraphConfig};
use elephant::retain::resolver::LayeredEntityResolver;
use elephant::retain::DefaultRetainPipeline;
use elephant::server::{self, AppState};
use elephant::storage::pg::PgMemoryStore;
use elephant::types::ChunkConfig;

fn make_llm(provider: &str, api_key: &str, model: &str) -> Box<dyn LlmClient> {
    match provider {
        "openai" => Box::new(OpenAiClient::new(api_key.into(), model.into(), None)),
        _ => Box::new(AnthropicClient::new(api_key.into(), model.into())),
    }
}

#[tokio::main]
async fn main() {
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let listen_addr = env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".into());
    let llm_provider = env::var("LLM_PROVIDER").unwrap_or_else(|_| "anthropic".into());
    let llm_api_key = env::var("LLM_API_KEY").expect("LLM_API_KEY must be set");
    let llm_model = env::var("LLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());

    // 1. Storage — PgPool is internally Arc'd, cheap to clone
    let pool = sqlx::PgPool::connect(&database_url)
        .await
        .expect("failed to connect to database");
    let store = Arc::new(PgMemoryStore::new(pool.clone()));
    store.migrate().await.expect("failed to run migrations");

    // 2. Shared Arc clients for recall/reflect/consolidation
    let llm: Arc<dyn LlmClient> =
        Arc::from(make_llm(&llm_provider, &llm_api_key, &llm_model));
    let emb_config = EmbeddingConfig {
        provider: if env::var("EMBEDDING_PROVIDER").unwrap_or_default() == "openai" {
            EmbeddingProvider::OpenAi
        } else {
            EmbeddingProvider::Local
        },
        model_path: env::var("EMBEDDING_MODEL_PATH").ok(),
        api_key: env::var("EMBEDDING_API_KEY").ok(),
        model: None,
        base_url: None,
        dimensions: None,
    };
    let embeddings: Arc<dyn EmbeddingClient> =
        Arc::from(embedding::build_client(&emb_config).expect("failed to create embedding client"));

    // 3. Retain pipeline (Box<dyn ...> — create separate lightweight instances sharing the pool)
    let retain = Arc::new(DefaultRetainPipeline::new(
        Box::new(SimpleChunker),
        Box::new(LlmFactExtractor::new(make_llm(
            &llm_provider,
            &llm_api_key,
            &llm_model,
        ))),
        Box::new(LayeredEntityResolver::new(
            Box::new(PgMemoryStore::new(pool.clone())),
            embedding::build_client(&emb_config).expect("embedding client"),
            make_llm(&llm_provider, &llm_api_key, &llm_model),
        )),
        Box::new(DefaultGraphBuilder::new(
            Box::new(PgMemoryStore::new(pool.clone())),
            make_llm(&llm_provider, &llm_api_key, &llm_model),
            GraphConfig::default(),
        )),
        Box::new(PgMemoryStore::new(pool.clone())),
        embedding::build_client(&emb_config).expect("embedding client"),
        make_llm(&llm_provider, &llm_api_key, &llm_model),
        ChunkConfig {
            max_tokens: 512,
            overlap_tokens: 64,
            preserve_turns: true,
        },
    ));

    // 4. Recall pipeline
    let recall = Arc::new(DefaultRecallPipeline::new(
        Box::new(SemanticRetriever::new(store.clone(), embeddings.clone(), 20)),
        Box::new(KeywordRetriever::new(store.clone(), 20)),
        Box::new(GraphRetriever::new(
            store.clone(),
            embeddings.clone(),
            GraphRetrieverConfig::default(),
        )),
        Box::new(TemporalRetriever::new(store.clone())),
        Box::new(NoOpReranker),
        Box::new(EstimateTokenizer),
        60.0,
        50,
    ));

    // 5. Reflect pipeline
    let reflect = Arc::new(DefaultReflectPipeline::new(
        Box::new(DefaultHierarchyAssembler::new(recall.clone())),
        Box::new(DefaultOpinionManager::new(store.clone(), embeddings.clone())),
        llm.clone(),
        store.clone(),
    ));

    // 6. Consolidation workers
    let consolidator = Arc::new(DefaultConsolidator::new(
        store.clone(),
        llm.clone(),
        embeddings.clone(),
    ));
    let opinion_merger = Arc::new(DefaultOpinionMerger::new(
        store.clone(),
        llm.clone(),
        embeddings.clone(),
    ));
    let model_generator = Arc::new(DefaultMentalModelGenerator::new(
        store.clone(),
        llm.clone(),
        embeddings.clone(),
    ));

    // 7. Build app state and serve
    let state = AppState {
        retain,
        recall,
        reflect,
        consolidator,
        opinion_merger,
        model_generator,
        store,
    };

    let app = server::router(state);
    let listener = tokio::net::TcpListener::bind(&listen_addr)
        .await
        .expect("failed to bind listener");
    println!("Listening on {listen_addr}");
    axum::serve(listener, app).await.expect("server error");
}
