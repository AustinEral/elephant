//! Local ONNX cross-encoder reranker (e.g. ms-marco-MiniLM-L-6-v2).

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;

use super::Reranker;
use crate::error::{Error, Result};
use crate::types::ScoredFact;

/// A local cross-encoder reranker running an ONNX model.
///
/// Scores `(query, document)` pairs through a cross-encoder and reorders by
/// relevance. Expects a model that takes `input_ids`, `attention_mask`, and
/// `token_type_ids` and outputs a single logit per pair.
pub struct LocalReranker {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
}

impl LocalReranker {
    /// Load the ONNX cross-encoder model and tokenizer from `model_dir`.
    ///
    /// Expects `model_dir/model.onnx` and `model_dir/tokenizer.json`.
    /// `max_seq_len` truncates inputs to fit the model's context window
    /// (e.g. 512 for MiniLM).
    pub fn new(model_dir: &Path, max_seq_len: usize) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = Session::builder()
            .map_err(|e| Error::Reranker(format!("session builder error: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| Error::Reranker(format!("thread config error: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| Error::Reranker(format!("model load error: {e}")))?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Reranker(format!("tokenizer load error: {e}")))?;

        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: max_seq_len,
                ..Default::default()
            }))
            .map_err(|e| Error::Reranker(format!("truncation config error: {e}")))?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
        })
    }
}

#[async_trait]
impl Reranker for LocalReranker {
    async fn rerank(
        &self,
        query: &str,
        mut facts: Vec<ScoredFact>,
        top_k: usize,
    ) -> Result<Vec<ScoredFact>> {
        if facts.is_empty() {
            return Ok(facts);
        }

        let session = self.session.clone();
        let tokenizer = self.tokenizer.clone();
        let documents: Vec<String> = facts.iter().map(super::format_reranker_input).collect();
        let query = query.to_string();

        let scores = tokio::task::spawn_blocking(move || {
            score_pairs(&session, &tokenizer, &query, &documents)
        })
        .await
        .map_err(|e| Error::Reranker(format!("spawn_blocking error: {e}")))??;

        // Assign cross-encoder scores and sort descending
        for (sf, &score) in facts.iter_mut().zip(scores.iter()) {
            sf.score = score;
        }
        facts.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        facts.truncate(top_k);
        Ok(facts)
    }
}

/// Score all (query, document) pairs through the cross-encoder.
///
/// Returns one sigmoid-normalized score per pair.
fn score_pairs(
    session: &Mutex<Session>,
    tokenizer: &tokenizers::Tokenizer,
    query: &str,
    documents: &[String],
) -> Result<Vec<f32>> {
    // Tokenize as sentence pairs: [CLS] query [SEP] document [SEP]
    let pairs: Vec<_> = documents.iter().map(|doc| (query, doc.as_str())).collect();

    let encodings = tokenizer
        .encode_batch(pairs, true)
        .map_err(|e| Error::Reranker(format!("tokenization error: {e}")))?;

    let batch_size = encodings.len();
    let seq_len = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0);

    let mut input_ids = Array2::<i64>::zeros((batch_size, seq_len));
    let mut attention_mask = Array2::<i64>::zeros((batch_size, seq_len));
    let mut token_type_ids = Array2::<i64>::zeros((batch_size, seq_len));

    for (i, encoding) in encodings.iter().enumerate() {
        for (j, &id) in encoding.get_ids().iter().enumerate() {
            input_ids[[i, j]] = id as i64;
        }
        for (j, &mask) in encoding.get_attention_mask().iter().enumerate() {
            attention_mask[[i, j]] = mask as i64;
        }
        for (j, &type_id) in encoding.get_type_ids().iter().enumerate() {
            token_type_ids[[i, j]] = type_id as i64;
        }
    }

    let input_ids_tensor = TensorRef::from_array_view(input_ids.view())
        .map_err(|e| Error::Reranker(format!("input_ids tensor error: {e}")))?;
    let attention_mask_tensor = TensorRef::from_array_view(attention_mask.view())
        .map_err(|e| Error::Reranker(format!("attention_mask tensor error: {e}")))?;
    let token_type_ids_tensor = TensorRef::from_array_view(token_type_ids.view())
        .map_err(|e| Error::Reranker(format!("token_type_ids tensor error: {e}")))?;

    let mut session = session
        .lock()
        .map_err(|e| Error::Reranker(format!("session lock error: {e}")))?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])
        .map_err(|e| Error::Reranker(format!("inference error: {e}")))?;

    // Cross-encoder output: shape [batch, 1] — single logit per pair
    let logits = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|e| Error::Reranker(format!("output extraction error: {e}")))?;

    // Sigmoid normalize to [0, 1]
    let scores: Vec<f32> = logits
        .iter()
        .map(|&logit| 1.0 / (1.0 + (-logit).exp()))
        .collect();

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_dir() -> PathBuf {
        let _ = dotenvy::dotenv();
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/ms-marco-MiniLM-L-6-v2")
    }

    /// Build-plan test case: disambiguate "Python" across programming vs. snake vs. creator.
    #[tokio::test]
    #[ignore = "requires cross-encoder model files"]
    async fn rerank_orders_by_relevance() {
        let reranker = LocalReranker::new(&model_dir(), 512).unwrap();

        let facts = vec![
            make_scored("The python snake is large"),
            make_scored("Python is a programming language"),
            make_scored("Python was created by Guido"),
        ];

        let result = reranker
            .rerank("programming language", facts, 3)
            .await
            .unwrap();

        assert_eq!(result.len(), 3);
        assert!(
            result[0].fact.content.contains("programming"),
            "expected programming fact first, got: {}",
            result[0].fact.content
        );
    }

    #[tokio::test]
    #[ignore = "requires cross-encoder model files"]
    async fn rerank_truncates() {
        let reranker = LocalReranker::new(&model_dir(), 512).unwrap();

        let facts = vec![
            make_scored("fact one"),
            make_scored("fact two"),
            make_scored("fact three"),
        ];

        let result = reranker.rerank("query", facts, 2).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    #[ignore = "requires cross-encoder model files"]
    async fn rerank_empty_input() {
        let reranker = LocalReranker::new(&model_dir(), 512).unwrap();
        let result = reranker.rerank("query", vec![], 10).await.unwrap();
        assert!(result.is_empty());
    }

    fn make_scored(content: &str) -> ScoredFact {
        use crate::types::*;
        use chrono::Utc;
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
            score: 0.5,
            sources: vec![RetrievalSource::Semantic],
        }
    }
}
