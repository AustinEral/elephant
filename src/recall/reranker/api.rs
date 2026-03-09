//! Cohere-compatible rerank API client.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::Reranker;
use crate::error::{Error, Result};
use crate::types::ScoredFact;

/// A reranker that calls a Cohere-compatible `/rerank` API endpoint.
///
/// Works with Cohere, Jina, Voyage, and other providers that implement
/// the same request/response format.
pub struct ApiReranker {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl ApiReranker {
    /// Create a new API reranker.
    pub fn new(api_key: String, base_url: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url,
            model,
        }
    }
}

#[derive(Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: Vec<&'a str>,
    top_n: usize,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl Reranker for ApiReranker {
    async fn rerank(
        &self,
        query: &str,
        facts: Vec<ScoredFact>,
        top_k: usize,
    ) -> Result<Vec<ScoredFact>> {
        if facts.is_empty() {
            return Ok(facts);
        }

        let doc_texts: Vec<String> = facts.iter().map(super::format_reranker_input).collect();
        let documents: Vec<&str> = doc_texts.iter().map(|s| s.as_str()).collect();

        let request = RerankRequest {
            model: &self.model,
            query,
            documents,
            top_n: top_k,
        };

        let response = self
            .client
            .post(format!("{}/rerank", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::Reranker(format!("request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Reranker(format!("API error {status}: {body}")));
        }

        let result: RerankResponse = response
            .json()
            .await
            .map_err(|e| Error::Reranker(format!("response parse error: {e}")))?;

        // Index into the original vec for O(1) extraction
        let mut indexed: Vec<Option<ScoredFact>> = facts.into_iter().map(Some).collect();

        // Sort by relevance_score descending, then extract in order
        let mut scored: Vec<(usize, f32)> = result
            .results
            .into_iter()
            .map(|r| (r.index, r.relevance_score))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let mut reranked = Vec::with_capacity(scored.len());
        for (idx, score) in scored {
            if let Some(Some(mut sf)) = indexed.get_mut(idx).map(Option::take) {
                sf.score = score;
                reranked.push(sf);
            }
        }

        Ok(reranked)
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
    async fn empty_input_returns_empty() {
        let reranker = ApiReranker::new("key".into(), "http://localhost:1".into(), "model".into());
        let result = reranker.rerank("query", vec![], 10).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn reranks_by_api_scores() {
        // Spin up a minimal mock server
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let app = axum::Router::new().route(
                "/rerank",
                axum::routing::post(|| async {
                    axum::Json(serde_json::json!({
                        "results": [
                            {"index": 2, "relevance_score": 0.95},
                            {"index": 0, "relevance_score": 0.80},
                            {"index": 1, "relevance_score": 0.10}
                        ]
                    }))
                }),
            );
            axum::serve(listener, app).await.unwrap();
        });

        let reranker = ApiReranker::new(
            "test-key".into(),
            format!("http://{addr}"),
            "test-model".into(),
        );

        let facts = vec![
            make_scored("alpha", 0.5),
            make_scored("beta", 0.5),
            make_scored("gamma", 0.5),
        ];

        let result = reranker.rerank("query", facts, 3).await.unwrap();
        assert_eq!(result.len(), 3);
        // Sorted by relevance_score descending
        assert_eq!(result[0].fact.content, "gamma");
        assert!((result[0].score - 0.95).abs() < 1e-5);
        assert_eq!(result[1].fact.content, "alpha");
        assert!((result[1].score - 0.80).abs() < 1e-5);
        assert_eq!(result[2].fact.content, "beta");
        assert!((result[2].score - 0.10).abs() < 1e-5);
    }

    #[tokio::test]
    async fn top_k_limits_results() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let app = axum::Router::new().route(
                "/rerank",
                axum::routing::post(|| async {
                    axum::Json(serde_json::json!({
                        "results": [
                            {"index": 0, "relevance_score": 0.9},
                            {"index": 1, "relevance_score": 0.5}
                        ]
                    }))
                }),
            );
            axum::serve(listener, app).await.unwrap();
        });

        let reranker = ApiReranker::new("key".into(), format!("http://{addr}"), "model".into());

        let facts = vec![make_scored("first", 0.5), make_scored("second", 0.5)];

        // API already returns top_n=2 from server, but we asked for it
        let result = reranker.rerank("query", facts, 2).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn api_error_returns_reranker_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let app = axum::Router::new().route(
                "/rerank",
                axum::routing::post(|| async {
                    (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "bad things")
                }),
            );
            axum::serve(listener, app).await.unwrap();
        });

        let reranker = ApiReranker::new("key".into(), format!("http://{addr}"), "model".into());

        let facts = vec![make_scored("test", 0.5)];
        let err = reranker.rerank("query", facts, 1).await.unwrap_err();
        assert!(matches!(err, Error::Reranker(_)));
        assert!(err.to_string().contains("500"));
    }
}
