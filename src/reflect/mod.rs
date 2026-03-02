//! Reflect pipeline — agentic CARA reasoning with tool-calling loop.

pub mod disposition;
pub mod hierarchy;
pub mod opinion;

use std::collections::HashSet;
use std::fmt::Write;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::Result;
use crate::llm::{complete_structured, LlmClient};
use crate::storage::MemoryStore;
use crate::types::llm::{CompletionRequest, Message, ToolCall, ToolChoice, ToolDef, ToolResult};
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
    max_iterations: usize,
}

impl DefaultReflectPipeline {
    /// Create a new reflect pipeline.
    pub fn new(
        assembler: Box<dyn HierarchyAssembler>,
        opinion_mgr: Box<dyn OpinionManager>,
        llm: Arc<dyn LlmClient>,
        store: Arc<dyn MemoryStore>,
        max_iterations: usize,
    ) -> Self {
        Self {
            assembler,
            opinion_mgr,
            llm,
            store,
            max_iterations,
        }
    }
}

/// A new opinion formed during reflection.
#[derive(Debug, Deserialize)]
struct NewOpinion {
    content: String,
    evidence: Vec<String>,
    confidence: f32,
}

/// Shared response structure for the `done` tool call and fallback synthesis.
#[derive(Debug, Deserialize)]
struct DoneArgs {
    response: String,
    #[serde(default)]
    sources: Vec<String>,
    #[serde(default)]
    new_opinions: Vec<NewOpinion>,
    confidence: f32,
}

/// Arguments for the `recall` tool call.
#[derive(Debug, Deserialize)]
struct RecallArgs {
    query: String,
}

fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "recall".into(),
            description: "Search stored memories by semantic query. Returns matching facts.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query to find relevant memories"
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDef {
            name: "done".into(),
            description: "Return the final answer. Must include response text and source citations.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The answer with inline [fact-ID] citations"
                    },
                    "sources": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Array of fact IDs referenced"
                    },
                    "new_opinions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string" },
                                "evidence": { "type": "array", "items": { "type": "string" } },
                                "confidence": { "type": "number" }
                            },
                            "required": ["content", "evidence", "confidence"]
                        },
                        "description": "New opinions formed (can be empty)"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Overall confidence 0.0-1.0"
                    }
                },
                "required": ["response", "sources", "confidence"]
            }),
        },
    ]
}

#[async_trait]
impl ReflectPipeline for DefaultReflectPipeline {
    async fn reflect(&self, query: &ReflectQuery) -> Result<ReflectResult> {
        // Load bank profile for system prompt
        let bank = self.store.get_bank(query.bank_id).await?;
        let bank_profile = verbalize_bank_profile(&bank);

        // Build system prompt
        let mut system_parts = Vec::new();
        let agent_prompt = include_str!("../../prompts/reflect_agent.txt");
        system_parts.push(agent_prompt.to_string());
        let profile_prompt = build_profile_prompt(&bank_profile);
        if !profile_prompt.is_empty() {
            system_parts.push(profile_prompt);
        }
        let system_prompt = system_parts.join("\n\n");

        // Conversation messages for the agent loop
        let mut messages: Vec<Message> = vec![Message {
            role: "user".into(),
            content: query.question.clone(),
        }];

        // Track all context accumulated across iterations
        let mut all_context = String::new();
        let mut seen_fact_ids: HashSet<FactId> = HashSet::new();

        let tools = tool_defs();

        for iteration in 0..self.max_iterations {
            // First iteration: force recall. Later: auto.
            let tool_choice = if iteration == 0 {
                Some(ToolChoice::Required)
            } else {
                Some(ToolChoice::Auto)
            };

            let request = CompletionRequest {
                messages: messages.clone(),
                temperature: Some(0.3),
                system: Some(system_prompt.clone()),
                tools: Some(tools.clone()),
                tool_choice,
                ..Default::default()
            };

            let response = self.llm.complete(request).await?;

            // If no tool calls, fall back to synthesis from text content
            if response.tool_calls.is_empty() {
                return self
                    .fallback_synthesis(query, &bank_profile, &response.content, &all_context)
                    .await;
            }

            // Process tool calls
            let mut tool_results = Vec::new();
            for tc in &response.tool_calls {
                match tc.name.as_str() {
                    "recall" => {
                        let args: RecallArgs = serde_json::from_value(tc.arguments.clone())
                            .unwrap_or(RecallArgs { query: query.question.clone() });

                        let context = self
                            .assembler
                            .assemble(&args.query, query.bank_id, query.budget_tokens)
                            .await?;

                        // Deduplicate facts across iterations
                        let mut new_facts = String::new();
                        for f in context.observations.iter()
                            .chain(context.raw_facts.iter())
                            .chain(context.opinions.iter())
                        {
                            if seen_fact_ids.insert(f.id) {
                                writeln!(new_facts, "[FACT {}] {}", f.id, f.content).unwrap();
                            }
                        }

                        if new_facts.is_empty() {
                            new_facts = "No new memories found for this query.".into();
                        }

                        all_context.push_str(&new_facts);
                        tool_results.push(ToolResult {
                            tool_call_id: tc.id.clone(),
                            content: new_facts,
                        });
                    }
                    "done" => {
                        let args: DoneArgs = serde_json::from_value(tc.arguments.clone())
                            .map_err(|e| crate::error::Error::Llm(
                                format!("failed to parse done args: {e}"),
                            ))?;

                        return self
                            .finalize(query, args)
                            .await;
                    }
                    _ => {
                        tool_results.push(ToolResult {
                            tool_call_id: tc.id.clone(),
                            content: format!("Unknown tool: {}", tc.name),
                        });
                    }
                }
            }

            // Append assistant message with tool calls, then user message with results
            messages.push(Message {
                role: "assistant".into(),
                content: format_tool_calls_for_history(&response.tool_calls),
            });
            messages.push(Message {
                role: "user".into(),
                content: format_tool_results_for_history(&tool_results),
            });
        }

        // Max iterations hit — force synthesis
        self.fallback_synthesis(query, &bank_profile, "", &all_context).await
    }
}

impl DefaultReflectPipeline {
    /// Finalize a reflect result from a `done` tool call.
    async fn finalize(
        &self,
        query: &ReflectQuery,
        args: DoneArgs,
    ) -> Result<ReflectResult> {
        let sources: Vec<FactId> = args
            .sources
            .iter()
            .filter_map(|s| s.parse::<FactId>().ok())
            .collect();

        let mut opinion_ids = Vec::new();
        for op in &args.new_opinions {
            let evidence: Vec<FactId> = op
                .evidence
                .iter()
                .filter_map(|s| s.parse::<FactId>().ok())
                .collect();
            let id = self
                .opinion_mgr
                .form_opinion(query.bank_id, &op.content, &evidence, op.confidence)
                .await?;
            opinion_ids.push(id);
        }

        // Batch fetch all opinion facts at once
        let new_opinion_facts = self.store.get_facts(&opinion_ids).await?;

        Ok(ReflectResult {
            response: args.response,
            sources,
            new_opinions: new_opinion_facts,
            confidence: args.confidence,
        })
    }

    /// Fallback: synthesize answer from accumulated context using the original reflect prompt.
    async fn fallback_synthesis(
        &self,
        query: &ReflectQuery,
        bank_profile: &crate::types::BankPromptContext,
        llm_text: &str,
        accumulated_context: &str,
    ) -> Result<ReflectResult> {
        // If the LLM already gave a text response (no tool calls), try parsing it as JSON
        if !llm_text.is_empty() {
            if let Ok(args) = serde_json::from_str::<DoneArgs>(llm_text)
                .or_else(|_| {
                    crate::llm::extract_json(llm_text)
                        .and_then(|j| serde_json::from_str(&j).map_err(|e| crate::error::Error::Llm(e.to_string())))
                })
            {
                return self.finalize(query, args).await;
            }
        }

        // Get existing opinions
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

        // Use accumulated context if available, otherwise do a fresh recall
        let context = if accumulated_context.is_empty() {
            let ctx = self
                .assembler
                .assemble(&query.question, query.bank_id, query.budget_tokens)
                .await?;
            ctx.formatted
        } else {
            accumulated_context.to_string()
        };

        let user_prompt = build_fallback_prompt(&context, &opinions_text, &query.question);
        let system_prompt = build_profile_prompt(bank_profile);

        let request = CompletionRequest {
            messages: vec![Message {
                role: "user".into(),
                content: user_prompt,
            }],
            temperature: Some(0.3),
            system: if system_prompt.is_empty() {
                None
            } else {
                Some(system_prompt)
            },
            ..Default::default()
        };

        let llm_response: DoneArgs = complete_structured(&*self.llm, request).await?;
        self.finalize(query, llm_response).await
    }
}

/// Format tool calls into a text representation for message history.
fn format_tool_calls_for_history(tool_calls: &[ToolCall]) -> String {
    tool_calls
        .iter()
        .map(|tc| format!("[tool_call: {}({})]", tc.name, tc.arguments))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format tool results into a text representation for message history.
fn format_tool_results_for_history(results: &[ToolResult]) -> String {
    results
        .iter()
        .map(|tr| tr.content.clone())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Build system prompt from bank profile components.
fn build_profile_prompt(profile: &crate::types::BankPromptContext) -> String {
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

/// Build the fallback user prompt from the reflect template.
fn build_fallback_prompt(context: &str, opinions: &str, question: &str) -> String {
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
    use crate::types::llm::CompletionResponse;
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
            consolidated_at: None,
        }
    }

    struct TestHarness {
        store: Arc<MockMemoryStore>,
        embeddings: Arc<MockEmbeddings>,
        llm: Arc<MockLlmClient>,
        bank_id: BankId,
    }

    impl TestHarness {
        async fn new() -> Self {
            let store = Arc::new(MockMemoryStore::new());
            let bank_id = BankId::new();
            let bank = MemoryBank {
                id: bank_id,
                name: "test".into(),
                mission: String::new(),
                directives: vec![],
                disposition: Disposition::default(),
                embedding_model: "mock".into(),
                embedding_dimensions: 8,
            };
            store.create_bank(&bank).await.unwrap();
            Self {
                store,
                embeddings: Arc::new(MockEmbeddings::new(8)),
                llm: Arc::new(MockLlmClient::new()),
                bank_id,
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
                5,
            )
        }
    }

    #[tokio::test]
    async fn reflect_with_tool_calls() {
        let h = TestHarness::new().await;
        let emb = h.embeddings.embed(&["Rust programming"]).await.unwrap();

        let fact = make_fact_with_embedding(
            h.bank_id,
            "Rust uses ownership for memory safety",
            NetworkType::World,
            emb[0].clone(),
        );
        let fact_id = fact.id;
        h.store.insert_facts(&[fact]).await.unwrap();

        // Iteration 0: LLM calls recall
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "Rust memory"}),
            }],
        });

        // Iteration 1: LLM calls done
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": format!("Rust uses ownership for memory safety [{}].", fact_id),
                    "sources": [fact_id.to_string()],
                    "new_opinions": [],
                    "confidence": 0.9
                }),
            }],
        });

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
    async fn reflect_fallback_when_no_tool_calls() {
        let h = TestHarness::new().await;
        let emb = h.embeddings.embed(&["Rust programming"]).await.unwrap();

        let fact = make_fact_with_embedding(
            h.bank_id,
            "Rust uses ownership for memory safety",
            NetworkType::World,
            emb[0].clone(),
        );
        let fact_id = fact.id;
        h.store.insert_facts(&[fact]).await.unwrap();

        // LLM returns text (no tool calls) — triggers fallback
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
    }

    #[tokio::test]
    async fn reflect_triggers_opinion_formation() {
        let h = TestHarness::new().await;
        let emb = h.embeddings.embed(&["testing"]).await.unwrap();

        let fact = make_fact_with_embedding(
            h.bank_id,
            "Tests improve code quality",
            NetworkType::World,
            emb[0].clone(),
        );
        let fact_id = fact.id;
        h.store.insert_facts(&[fact]).await.unwrap();

        // recall then done with opinions
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "testing"}),
            }],
        });

        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": format!("Testing is valuable [{}].", fact_id),
                    "sources": [fact_id.to_string()],
                    "new_opinions": [{
                        "content": "Automated testing is essential for quality software",
                        "evidence": [fact_id.to_string()],
                        "confidence": 0.85
                    }],
                    "confidence": 0.8
                }),
            }],
        });

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
    async fn empty_memory_graceful_response() {
        let h = TestHarness::new().await;

        // recall then done with empty sources
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "meaning of life"}),
            }],
        });

        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": "I don't have enough information to answer this question.",
                    "sources": [],
                    "new_opinions": [],
                    "confidence": 0.1
                }),
            }],
        });

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
        let h = TestHarness::new().await;

        // Create a bank with a strong disposition
        let bank = MemoryBank {
            id: h.bank_id,
            name: "test bank".into(),
            mission: "Remember developer context".into(),
            directives: vec!["Never share secrets".into()],
            disposition: Disposition::new(5, 1, 4, 0.9).unwrap(),
            embedding_model: String::new(),
            embedding_dimensions: 0,
        };
        h.store.create_bank(&bank).await.unwrap();

        // recall then done
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "analysis"}),
            }],
        });
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": "Based on available context, here is my analysis.",
                    "sources": [],
                    "new_opinions": [],
                    "confidence": 0.5
                }),
            }],
        });

        let pipeline = h.build_pipeline();
        let _result = pipeline
            .reflect(&ReflectQuery {
                bank_id: h.bank_id,
                question: "Analyze something".into(),
                budget_tokens: 2000,
            })
            .await
            .unwrap();

        // Verify the LLM was called (responses were consumed)
        assert_eq!(h.llm.remaining(), 0);

        // Verify the system prompt would contain personality text
        let profile = verbalize_bank_profile(&bank);
        assert!(profile.disposition_prompt.contains("extremely skeptical"));
        assert!(profile.disposition_prompt.contains("reads between the lines"));
        assert!(profile.directives_prompt.contains("Never share secrets"));
        assert_eq!(profile.mission_prompt, "Remember developer context");
    }

    #[tokio::test]
    async fn reflect_fails_when_bank_not_found() {
        let h = TestHarness::new().await;
        let pipeline = h.build_pipeline();

        // Use a bank_id that doesn't exist in the store
        let bogus_bank = crate::types::BankId::new();
        let result = pipeline
            .reflect(&ReflectQuery {
                bank_id: bogus_bank,
                question: "anything".into(),
                budget_tokens: 2000,
            })
            .await;

        assert!(result.is_err(), "reflect should fail when bank doesn't exist");
    }

    #[tokio::test]
    async fn multi_recall_deduplicates_facts() {
        let h = TestHarness::new().await;
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

        // Two recall calls, then done
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "first data"}),
            }],
        });
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "second data"}),
            }],
        });
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_3".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": format!("Facts [{id1}] and [{id2}] about data."),
                    "sources": [id1.to_string(), id2.to_string()],
                    "new_opinions": [],
                    "confidence": 0.7
                }),
            }],
        });

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
}
