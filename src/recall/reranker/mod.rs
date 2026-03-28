//! Reranker trait and implementations for recall pipeline.

pub mod api;
pub mod local;

use async_trait::async_trait;

use crate::config::ConfigError;
use crate::error::{Error, Result};
use crate::types::ScoredFact;

/// Reranker provider selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RerankerProvider {
    /// Local ONNX cross-encoder model.
    Local,
    /// Remote rerank API (Cohere-compatible).
    Api,
    /// No reranking — preserve fused order.
    None,
}

/// Configuration for the reranker provider.
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Which provider to use.
    pub provider: RerankerProvider,
    /// Path to local ONNX model directory (for [`RerankerProvider::Local`]).
    pub model_path: Option<String>,
    /// Max sequence length for local tokenizer truncation (e.g. 512 for MiniLM).
    pub max_seq_len: usize,
    /// API key (for [`RerankerProvider::Api`]).
    pub api_key: Option<String>,
    /// API base URL (for [`RerankerProvider::Api`]).
    pub api_url: Option<String>,
    /// API model name (for [`RerankerProvider::Api`]).
    pub api_model: Option<String>,
}

impl RerankerConfig {
    /// Read reranker configuration from environment.
    pub fn from_env() -> std::result::Result<Self, ConfigError> {
        let provider = match std::env::var("RERANKER_PROVIDER")
            .map_err(|e| ConfigError::configuration(format!("RERANKER_PROVIDER must be set: {e}")))?
            .as_str()
        {
            "local" => RerankerProvider::Local,
            "api" => RerankerProvider::Api,
            "none" => RerankerProvider::None,
            other => {
                return Err(ConfigError::configuration(format!(
                    "unknown RERANKER_PROVIDER: {other}"
                )));
            }
        };

        Ok(Self {
            provider,
            model_path: std::env::var("RERANKER_MODEL_PATH").ok(),
            max_seq_len: std::env::var("RERANKER_MAX_SEQ_LEN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512),
            api_key: std::env::var("RERANKER_API_KEY").ok(),
            api_url: std::env::var("RERANKER_API_URL").ok(),
            api_model: std::env::var("RERANKER_API_MODEL").ok(),
        })
    }
}

/// Read reranker configuration from environment.
pub fn config_from_env() -> Result<RerankerConfig> {
    RerankerConfig::from_env().map_err(Into::into)
}

/// Build a reranker from configuration.
pub fn build_reranker(config: &RerankerConfig) -> Result<Box<dyn Reranker>> {
    match config.provider {
        RerankerProvider::None => Ok(Box::new(NoOpReranker)),
        RerankerProvider::Local => {
            let model_path = config.model_path.as_deref().ok_or_else(|| {
                Error::Reranker("RERANKER_MODEL_PATH must be set for local reranker".into())
            })?;
            let reranker =
                local::LocalReranker::new(std::path::Path::new(model_path), config.max_seq_len)?;
            Ok(Box::new(reranker))
        }
        RerankerProvider::Api => {
            let api_key = config
                .api_key
                .clone()
                .ok_or_else(|| Error::Reranker("RERANKER_API_KEY must be set".into()))?;
            let api_url = config
                .api_url
                .clone()
                .ok_or_else(|| Error::Reranker("RERANKER_API_URL must be set".into()))?;
            let api_model = config
                .api_model
                .clone()
                .ok_or_else(|| Error::Reranker("RERANKER_API_MODEL must be set".into()))?;
            Ok(Box::new(api::ApiReranker::new(api_key, api_url, api_model)))
        }
    }
}

/// Format a fact's content with temporal context for cross-encoder input.
///
/// Prepends date information so the cross-encoder can judge temporal relevance
/// (e.g. "what happened last week?").
pub fn format_reranker_input(fact: &crate::types::ScoredFact) -> String {
    if let Some(ref tr) = fact.fact.temporal_range
        && let Some(start) = tr.start
    {
        let date_iso = start.format("%Y-%m-%d");
        let date_readable = start.format("%B %d, %Y");
        return format!("[Date: {date_readable} ({date_iso})] {}", fact.fact.content);
    }
    fact.fact.content.clone()
}

/// Reranks a set of scored facts, typically using a cross-encoder or similar model.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank the given facts for the query, returning all facts in relevance order.
    async fn rerank(&self, query: &str, facts: Vec<ScoredFact>) -> Result<Vec<ScoredFact>>;
}

/// No-op reranker that preserves the incoming order.
pub struct NoOpReranker;

#[async_trait]
impl Reranker for NoOpReranker {
    async fn rerank(&self, _query: &str, facts: Vec<ScoredFact>) -> Result<Vec<ScoredFact>> {
        Ok(facts)
    }
}

/// Mock reranker that reverses order (for testing pipeline wiring).
pub struct MockReranker;

#[async_trait]
impl Reranker for MockReranker {
    async fn rerank(&self, _query: &str, mut facts: Vec<ScoredFact>) -> Result<Vec<ScoredFact>> {
        facts.reverse();
        Ok(facts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::Utc;

    fn make_scored(content: &str, score: f32) -> ScoredFact {
        ScoredFact {
            fact: Fact {
                id: FactId::new(),
                bank_id: BankId::new(),
                content: content.into(),
                fact_type: FactType::World,
                network: NetworkType::World,
                entity_ids: vec![],
                temporal_range: None,
                embedding: None,
                confidence: None,
                evidence_ids: vec![],
                source_turn_id: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                consolidated_at: None,
            },
            score,
            sources: vec![RetrievalSource::Semantic],
        }
    }

    #[tokio::test]
    async fn noop_empty_input() {
        let result = NoOpReranker.rerank("query", vec![]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn noop_preserves_all_facts() {
        let facts = vec![
            make_scored("a", 1.0),
            make_scored("b", 0.9),
            make_scored("c", 0.8),
        ];
        let result = NoOpReranker.rerank("query", facts).await.unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].fact.content, "a");
        assert_eq!(result[1].fact.content, "b");
        assert_eq!(result[2].fact.content, "c");
    }

    #[tokio::test]
    async fn noop_preserves_order() {
        let facts = vec![
            make_scored("a", 1.0),
            make_scored("b", 0.9),
            make_scored("c", 0.8),
        ];
        let result = NoOpReranker.rerank("query", facts).await.unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].fact.content, "a");
        assert_eq!(result[1].fact.content, "b");
        assert_eq!(result[2].fact.content, "c");
    }

    #[tokio::test]
    async fn mock_empty_input() {
        let result = MockReranker.rerank("query", vec![]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn mock_reverses() {
        let facts = vec![
            make_scored("a", 1.0),
            make_scored("b", 0.9),
            make_scored("c", 0.8),
        ];
        let result = MockReranker.rerank("query", facts).await.unwrap();
        assert_eq!(result[0].fact.content, "c");
        assert_eq!(result[1].fact.content, "b");
        assert_eq!(result[2].fact.content, "a");
    }

    #[test]
    fn build_reranker_none() {
        let config = RerankerConfig {
            provider: RerankerProvider::None,
            model_path: None,
            max_seq_len: 512,
            api_key: None,
            api_url: None,
            api_model: None,
        };
        assert!(build_reranker(&config).is_ok());
    }

    #[test]
    fn build_reranker_local_missing_path() {
        let config = RerankerConfig {
            provider: RerankerProvider::Local,
            model_path: None,
            max_seq_len: 512,
            api_key: None,
            api_url: None,
            api_model: None,
        };
        assert!(build_reranker(&config).is_err());
    }

    #[test]
    fn build_reranker_api_missing_fields() {
        let config = RerankerConfig {
            provider: RerankerProvider::Api,
            model_path: None,
            max_seq_len: 512,
            api_key: None,
            api_url: None,
            api_model: None,
        };
        assert!(build_reranker(&config).is_err());
    }

    #[test]
    fn format_reranker_input_with_temporal() {
        use crate::types::TemporalRange;
        use chrono::TimeZone;
        let mut sf = make_scored("Alice joined Acme Corp", 0.5);
        sf.fact.temporal_range = Some(TemporalRange {
            start: Some(Utc.with_ymd_and_hms(2024, 6, 15, 0, 0, 0).unwrap()),
            end: None,
        });
        let text = format_reranker_input(&sf);
        assert!(text.starts_with("[Date: June 15, 2024 (2024-06-15)]"));
        assert!(text.contains("Alice joined Acme Corp"));
    }

    #[test]
    fn format_reranker_input_without_temporal() {
        let sf = make_scored("plain fact", 0.5);
        assert_eq!(format_reranker_input(&sf), "plain fact");
    }

    #[test]
    fn build_reranker_api_all_fields() {
        let config = RerankerConfig {
            provider: RerankerProvider::Api,
            model_path: None,
            max_seq_len: 512,
            api_key: Some("key".into()),
            api_url: Some("https://api.example.com/v1".into()),
            api_model: Some("rerank-v1".into()),
        };
        assert!(build_reranker(&config).is_ok());
    }
}
