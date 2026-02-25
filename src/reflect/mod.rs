//! Reflect pipeline — CARA reasoning over assembled memory context.

pub mod disposition;
pub mod hierarchy;
pub mod opinion;

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::Result;
use crate::llm::{complete_structured, LlmClient};
use crate::storage::MemoryStore;
use crate::types::llm::{CompletionRequest, Message};
use crate::types::{FactId, ReflectQuery, ReflectResult};

use disposition::verbalize_bank_profile;
use hierarchy::HierarchyAssembler;
use opinion::OpinionManager;

/// The full reflect pipeline: assemble context → reason → persist opinions.
#[async_trait]
pub trait ReflectPipeline: Send + Sync {
    /// Execute the reflect pipeline for the given query.
    async fn reflect(&self, query: &ReflectQuery) -> Result<ReflectResult>;
}

/// Default reflect pipeline wiring assembler, opinion manager, LLM, and store.
pub struct DefaultReflectPipeline {
    assembler: Box<dyn HierarchyAssembler>,
    opinion_mgr: Box<dyn OpinionManager>,
    llm: Arc<dyn LlmClient>,
    store: Arc<dyn MemoryStore>,
}

impl DefaultReflectPipeline {
    /// Create a new reflect pipeline.
    pub fn new(
        assembler: Box<dyn HierarchyAssembler>,
        opinion_mgr: Box<dyn OpinionManager>,
        llm: Arc<dyn LlmClient>,
        store: Arc<dyn MemoryStore>,
    ) -> Self {
        Self {
            assembler,
            opinion_mgr,
            llm,
            store,
        }
    }
}

/// LLM response structure for reflect.
#[derive(Debug, Deserialize)]
struct ReflectLlmResponse {
    response: String,
    sources: Vec<String>,
    #[serde(default)]
    new_opinions: Vec<NewOpinion>,
    confidence: f32,
}

/// A new opinion formed during reflection.
#[derive(Debug, Deserialize)]
struct NewOpinion {
    content: String,
    evidence: Vec<String>,
    confidence: f32,
}

#[async_trait]
impl ReflectPipeline for DefaultReflectPipeline {
    async fn reflect(&self, query: &ReflectQuery) -> Result<ReflectResult> {
        // Step 1: Assemble memory hierarchy
        let context = self
            .assembler
            .assemble(&query.question, query.bank_id, query.budget_tokens)
            .await?;

        // Step 2: Load bank profile and verbalize disposition
        let bank_profile = match self.store.get_bank(query.bank_id).await {
            Ok(bank) => verbalize_bank_profile(&bank),
            Err(_) => crate::types::BankPromptContext {
                disposition_prompt: String::new(),
                directives_prompt: String::new(),
                mission_prompt: String::new(),
            },
        };

        // Step 3: Get existing opinions on this topic
        let existing_opinions = self
            .opinion_mgr
            .get_opinions(query.bank_id, &query.question)
            .await
            .unwrap_or_default();

        let opinions_text = if existing_opinions.is_empty() {
            "No existing opinions on this topic.".to_string()
        } else {
            existing_opinions
                .iter()
                .map(|o| {
                    format!(
                        "- [{}] (confidence: {:.2}) {}",
                        o.id,
                        o.confidence.unwrap_or(0.5),
                        o.content
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        // Step 4: Compose prompt
        let system_prompt = build_system_prompt(&bank_profile);
        let user_prompt = build_user_prompt(&context.formatted, &opinions_text, &query.question);

        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message {
                role: "user".into(),
                content: user_prompt,
            }],
            max_tokens: None,
            temperature: Some(0.3),
            system: if system_prompt.is_empty() {
                None
            } else {
                Some(system_prompt)
            },
        };

        // Step 5: Call LLM with structured output
        let llm_response: ReflectLlmResponse = complete_structured(&*self.llm, request).await?;

        // Step 6: Parse source fact IDs
        let sources: Vec<FactId> = llm_response
            .sources
            .iter()
            .filter_map(|s| s.parse::<FactId>().ok())
            .collect();

        // Step 7: Persist new opinions
        let mut new_opinion_facts = Vec::new();
        for op in &llm_response.new_opinions {
            let evidence: Vec<FactId> = op
                .evidence
                .iter()
                .filter_map(|s| s.parse::<FactId>().ok())
                .collect();
            let id = self
                .opinion_mgr
                .form_opinion(query.bank_id, &op.content, &evidence, op.confidence)
                .await?;

            // Retrieve the persisted fact to include in the result
            let facts = self.store.get_facts(&[id]).await?;
            if let Some(fact) = facts.into_iter().next() {
                new_opinion_facts.push(fact);
            }
        }

        Ok(ReflectResult {
            response: llm_response.response,
            sources,
            new_opinions: new_opinion_facts,
            confidence: llm_response.confidence,
        })
    }
}

/// Build the system prompt from bank profile components.
fn build_system_prompt(profile: &crate::types::BankPromptContext) -> String {
    let mut parts = Vec::new();
    if !profile.disposition_prompt.is_empty() {
        parts.push(profile.disposition_prompt.clone());
    }
    if !profile.directives_prompt.is_empty() {
        parts.push(profile.directives_prompt.clone());
    }
    if !profile.mission_prompt.is_empty() {
        parts.push(format!("Mission: {}", profile.mission_prompt));
    }
    parts.join("\n\n")
}

/// Build the user prompt from the reflect template.
fn build_user_prompt(context: &str, opinions: &str, question: &str) -> String {
    include_str!("../../prompts/reflect.txt")
        .replace("{context}", context)
        .replace("{opinions}", opinions)
        .replace("{question}", question)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::mock::MockEmbeddings;
    use crate::embedding::EmbeddingClient;
    use crate::llm::mock::MockLlmClient;
    use crate::recall::budget::EstimateTokenizer;
    use crate::recall::graph::{GraphRetriever, GraphRetrieverConfig};
    use crate::recall::keyword::KeywordRetriever;
    use crate::recall::reranker::NoOpReranker;
    use crate::recall::semantic::SemanticRetriever;
    use crate::recall::temporal::TemporalRetriever;
    use crate::recall::DefaultRecallPipeline;
    use crate::storage::mock::MockMemoryStore;
    use crate::storage::MemoryStore;
    use crate::types::*;
    use chrono::Utc;

    fn make_fact_with_embedding(
        bank: BankId,
        content: &str,
        network: NetworkType,
        embedding: Vec<f32>,
    ) -> Fact {
        Fact {
            id: FactId::new(),
            bank_id: bank,
            content: content.into(),
            fact_type: FactType::World,
            network,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(embedding),
            confidence: None,
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    struct TestHarness {
        store: Arc<MockMemoryStore>,
        embeddings: Arc<MockEmbeddings>,
        llm: Arc<MockLlmClient>,
        bank_id: BankId,
    }

    impl TestHarness {
        fn new() -> Self {
            Self {
                store: Arc::new(MockMemoryStore::new()),
                embeddings: Arc::new(MockEmbeddings::new(8)),
                llm: Arc::new(MockLlmClient::new()),
                bank_id: BankId::new(),
            }
        }

        fn build_pipeline(&self) -> DefaultReflectPipeline {
            let recall = Arc::new(DefaultRecallPipeline::new(
                Box::new(SemanticRetriever::new(
                    self.store.clone(),
                    self.embeddings.clone(),
                    20,
                )),
                Box::new(KeywordRetriever::new(self.store.clone(), 20)),
                Box::new(GraphRetriever::new(
                    self.store.clone(),
                    self.embeddings.clone(),
                    GraphRetrieverConfig::default(),
                )),
                Box::new(TemporalRetriever::new(self.store.clone())),
                Box::new(NoOpReranker),
                Box::new(EstimateTokenizer),
                60.0,
                50,
            ));

            let assembler = Box::new(hierarchy::DefaultHierarchyAssembler::new(recall));

            let opinion_mgr = Box::new(opinion::DefaultOpinionManager::new(
                self.store.clone(),
                self.embeddings.clone(),
            ));

            DefaultReflectPipeline::new(
                assembler,
                opinion_mgr,
                self.llm.clone(),
                self.store.clone(),
            )
        }
    }

    #[tokio::test]
    async fn reflect_with_stored_facts() {
        let h = TestHarness::new();
        let emb = h.embeddings.embed(&["Rust programming"]).await.unwrap();

        let fact = make_fact_with_embedding(
            h.bank_id,
            "Rust uses ownership for memory safety",
            NetworkType::World,
            emb[0].clone(),
        );
        let fact_id = fact.id;
        h.store.insert_facts(&[fact]).await.unwrap();

        h.llm.push_response(format!(
            r#"{{
                "response": "Based on the context, Rust uses ownership for memory safety [{}].",
                "sources": ["{}"],
                "new_opinions": [],
                "confidence": 0.9
            }}"#,
            fact_id, fact_id
        ));

        let pipeline = h.build_pipeline();
        let result = pipeline
            .reflect(&ReflectQuery {
                bank_id: h.bank_id,
                question: "How does Rust handle memory?".into(),
                budget_tokens: 2000,
            })
            .await
            .unwrap();

        assert!(result.response.contains("ownership"));
        assert!(result.sources.contains(&fact_id));
        assert!((result.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn reflect_triggers_opinion_formation() {
        let h = TestHarness::new();
        let emb = h.embeddings.embed(&["testing"]).await.unwrap();

        let fact = make_fact_with_embedding(
            h.bank_id,
            "Tests improve code quality",
            NetworkType::World,
            emb[0].clone(),
        );
        let fact_id = fact.id;
        h.store.insert_facts(&[fact]).await.unwrap();

        h.llm.push_response(format!(
            r#"{{
                "response": "Testing is valuable [{}].",
                "sources": ["{}"],
                "new_opinions": [
                    {{
                        "content": "Automated testing is essential for quality software",
                        "evidence": ["{}"],
                        "confidence": 0.85
                    }}
                ],
                "confidence": 0.8
            }}"#,
            fact_id, fact_id, fact_id
        ));

        let pipeline = h.build_pipeline();
        let result = pipeline
            .reflect(&ReflectQuery {
                bank_id: h.bank_id,
                question: "Is testing important?".into(),
                budget_tokens: 2000,
            })
            .await
            .unwrap();

        assert_eq!(result.new_opinions.len(), 1);
        assert_eq!(result.new_opinions[0].network, NetworkType::Opinion);
        assert!(result.new_opinions[0]
            .content
            .contains("Automated testing"));
    }

    #[tokio::test]
    async fn source_attribution_extracts_fact_ids() {
        let h = TestHarness::new();
        let emb = h.embeddings.embed(&["data"]).await.unwrap();

        let fact1 = make_fact_with_embedding(
            h.bank_id,
            "First fact about data",
            NetworkType::World,
            emb[0].clone(),
        );
        let fact2 = make_fact_with_embedding(
            h.bank_id,
            "Second fact about data",
            NetworkType::World,
            emb[0].clone(),
        );
        let id1 = fact1.id;
        let id2 = fact2.id;
        h.store.insert_facts(&[fact1, fact2]).await.unwrap();

        h.llm.push_response(format!(
            r#"{{
                "response": "Combining facts [{id1}] and [{id2}] shows data patterns.",
                "sources": ["{id1}", "{id2}"],
                "new_opinions": [],
                "confidence": 0.7
            }}"#,
        ));

        let pipeline = h.build_pipeline();
        let result = pipeline
            .reflect(&ReflectQuery {
                bank_id: h.bank_id,
                question: "What about data?".into(),
                budget_tokens: 2000,
            })
            .await
            .unwrap();

        assert_eq!(result.sources.len(), 2);
        assert!(result.sources.contains(&id1));
        assert!(result.sources.contains(&id2));
    }

    #[tokio::test]
    async fn empty_memory_graceful_response() {
        let h = TestHarness::new();

        h.llm.push_response(
            r#"{
                "response": "I don't have enough information to answer this question.",
                "sources": [],
                "new_opinions": [],
                "confidence": 0.1
            }"#,
        );

        let pipeline = h.build_pipeline();
        let result = pipeline
            .reflect(&ReflectQuery {
                bank_id: h.bank_id,
                question: "What is the meaning of life?".into(),
                budget_tokens: 2000,
            })
            .await
            .unwrap();

        assert!(result.response.contains("don't have enough"));
        assert!(result.sources.is_empty());
        assert!(result.new_opinions.is_empty());
    }

    #[tokio::test]
    async fn disposition_influence_in_system_prompt() {
        let h = TestHarness::new();

        // Create a bank with a strong disposition
        let bank = MemoryBank {
            id: h.bank_id,
            name: "test bank".into(),
            mission: "Remember developer context".into(),
            directives: vec!["Never share secrets".into()],
            disposition: Disposition::new(5, 1, 4, 0.9).unwrap(),
        };
        h.store.create_bank(&bank).await.unwrap();

        h.llm.push_response(
            r#"{
                "response": "Based on available context, here is my analysis.",
                "sources": [],
                "new_opinions": [],
                "confidence": 0.5
            }"#,
        );

        let pipeline = h.build_pipeline();
        let _result = pipeline
            .reflect(&ReflectQuery {
                bank_id: h.bank_id,
                question: "Analyze something".into(),
                budget_tokens: 2000,
            })
            .await
            .unwrap();

        // Verify the LLM was called (response was consumed)
        assert_eq!(h.llm.remaining(), 0);

        // Verify the system prompt would contain personality text
        let profile = verbalize_bank_profile(&bank);
        assert!(profile.disposition_prompt.contains("extremely skeptical"));
        assert!(profile.disposition_prompt.contains("reads between the lines"));
        assert!(profile.directives_prompt.contains("Never share secrets"));
        assert_eq!(profile.mission_prompt, "Remember developer context");
    }
}
