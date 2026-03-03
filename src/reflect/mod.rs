//! Reflect pipeline — agentic CARA reasoning with tool-calling loop.

pub mod disposition;
pub mod hierarchy;
pub mod opinion;

use std::collections::HashSet;
use std::fmt::Write;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::{debug, error, info, trace, info_span, Instrument};

use crate::error::Result;
use crate::llm::LlmClient;
use crate::recall::RecallPipeline;
use crate::storage::MemoryStore;
use crate::types::llm::{CompletionRequest, Message, ToolCall, ToolChoice, ToolDef, ToolResult};
use crate::types::{FactId, NetworkType, RecallQuery, ReflectQuery, ReflectResult};

use disposition::verbalize_bank_profile;

/// The full reflect pipeline: assemble context → reason → persist opinions.
#[async_trait]
pub trait ReflectPipeline: Send + Sync {
    /// Execute the reflect pipeline for the given query.
    async fn reflect(&self, query: &ReflectQuery) -> Result<ReflectResult>;
}

/// Default reflect pipeline wiring recall, LLM, and store.
pub struct DefaultReflectPipeline {
    recall: Arc<dyn RecallPipeline>,
    llm: Arc<dyn LlmClient>,
    store: Arc<dyn MemoryStore>,
    max_iterations: usize,
}

impl DefaultReflectPipeline {
    /// Create a new reflect pipeline.
    pub fn new(
        recall: Arc<dyn RecallPipeline>,
        llm: Arc<dyn LlmClient>,
        store: Arc<dyn MemoryStore>,
        max_iterations: usize,
    ) -> Self {
        Self {
            recall,
            llm,
            store,
            max_iterations,
        }
    }
}

/// Arguments for a search tool call (`search_observations` or `recall`).
#[derive(Debug, Deserialize, JsonSchema)]
struct SearchArgs {
    /// Semantic search query.
    query: String,
    /// Why you are searching for this (guides LLM chain-of-thought).
    #[allow(dead_code)]
    reason: String,
}

/// Arguments for the `done` tool call — also used for fallback synthesis.
#[derive(Debug, Deserialize, JsonSchema)]
struct DoneArgs {
    /// The answer with inline [fact-ID] citations.
    response: String,
    /// Fact IDs referenced in the response.
    #[serde(default)]
    source_ids: Vec<String>,
}

fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef::from_schema::<SearchArgs>(
            "search_observations",
            "Search consolidated observations (high-level summaries). Use first for the big picture.",
        ),
        ToolDef::from_schema::<SearchArgs>(
            "recall",
            "Search raw memories (facts and experiences). Ground truth data. Use to verify or fill gaps.",
        ),
        ToolDef::from_schema::<DoneArgs>(
            "done",
            "Return the final answer with source citations.",
        ),
    ]
}

#[async_trait]
impl ReflectPipeline for DefaultReflectPipeline {
    async fn reflect(&self, query: &ReflectQuery) -> Result<ReflectResult> {
        let reflect_span = info_span!("reflect",
            bank_id = %query.bank_id,
            question = %query.question,
            iterations = tracing::field::Empty,
            source_count = tracing::field::Empty,
        );
        self.reflect_inner(query).instrument(reflect_span).await
    }
}

impl DefaultReflectPipeline {
    async fn reflect_inner(&self, query: &ReflectQuery) -> Result<ReflectResult> {
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

        let mut seen_fact_ids: HashSet<FactId> = HashSet::new();

        let tools = tool_defs();

        let last_iteration = self.max_iterations - 1;
        let done_only = vec![ToolDef::from_schema::<DoneArgs>(
            "done",
            "Return the final answer with source citations.",
        )];

        for iteration in 0..self.max_iterations {
            let forced_tool = match iteration {
                _ if iteration == last_iteration => "done",
                0 => "search_observations",
                1 => "recall",
                _ => "auto",
            };
            debug!(iteration, forced_tool, "reflect_iter_start");

            // Forced tool sequence:
            //   0           → search_observations
            //   1           → recall
            //   2..last-1   → auto (all tools)
            //   last        → done only, required
            let (iter_tools, tool_choice) = if iteration == last_iteration {
                (&done_only, Some(ToolChoice::Required))
            } else {
                let choice = match iteration {
                    0 => Some(ToolChoice::Specific("search_observations".into())),
                    1 => Some(ToolChoice::Specific("recall".into())),
                    _ => Some(ToolChoice::Auto),
                };
                (&tools, choice)
            };

            let request = CompletionRequest {
                messages: messages.clone(),
                temperature: Some(0.3),
                system: Some(system_prompt.clone()),
                tools: Some(iter_tools.clone()),
                tool_choice,
                ..Default::default()
            };

            let llm_start = Instant::now();
            let response = self.llm.complete(request).await?;
            let llm_ms = llm_start.elapsed().as_millis() as u64;
            debug!(
                llm_ms,
                input_tokens = response.input_tokens,
                output_tokens = response.output_tokens,
                tool_calls_count = response.tool_calls.len(),
                content = response.content.as_str(),
                "llm_complete"
            );

            // No tool calls — skip to next iteration (shouldn't happen with forced tool choices)
            if response.tool_calls.is_empty() {
                continue;
            }

            // Process tool calls
            let mut tool_results = Vec::new();
            for tc in &response.tool_calls {
                match tc.name.as_str() {
                    "search_observations" | "recall" => {
                        let network_filter = if tc.name == "search_observations" {
                            Some(vec![NetworkType::Observation])
                        } else {
                            Some(vec![NetworkType::World, NetworkType::Experience, NetworkType::Opinion])
                        };

                        let args: SearchArgs = serde_json::from_value(tc.arguments.clone())
                            .unwrap_or(SearchArgs { query: query.question.clone(), reason: String::new() });

                        let recall_start = Instant::now();
                        let result = self
                            .recall
                            .recall(&RecallQuery {
                                bank_id: query.bank_id,
                                query: args.query.clone(),
                                budget_tokens: query.budget_tokens,
                                network_filter,
                                temporal_anchor: None,
                            })
                            .await?;
                        let recall_ms = recall_start.elapsed().as_millis() as u64;

                        // Deduplicate facts across iterations
                        let mut new_facts = String::new();
                        let mut facts_new = 0usize;
                        for sf in &result.facts {
                            let is_new = seen_fact_ids.insert(sf.fact.id);
                            trace!(
                                fact_id = %sf.fact.id,
                                is_new,
                                score = sf.score,
                                network = ?sf.fact.network,
                                "fact"
                            );
                            if is_new {
                                facts_new += 1;
                                writeln!(new_facts, "[FACT {}] {}", sf.fact.id, sf.fact.content).unwrap();
                            }
                        }

                        debug!(
                            tool_name = tc.name.as_str(),
                            query = args.query.as_str(),
                            recall_ms,
                            facts_returned = result.facts.len(),
                            facts_new,
                            total_tokens = result.total_tokens,
                            "tool_call"
                        );

                        if new_facts.is_empty() {
                            new_facts = "No new memories found for this query.".into();
                        }

                        tool_results.push(ToolResult {
                            tool_call_id: tc.id.clone(),
                            content: new_facts,
                        });
                    }
                    "done" => {
                        let args: DoneArgs = match serde_json::from_value(tc.arguments.clone()) {
                            Ok(a) => a,
                            Err(_) => {
                                // LLM sometimes sends source_ids as a string instead of array.
                                // Fall back to extracting just the response field.
                                let response = tc.arguments.get("response")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                DoneArgs { response, source_ids: vec![] }
                            }
                        };
                        info!(
                            response = args.response.as_str(),
                            source_ids = ?args.source_ids,
                            "done"
                        );
                        let result = self.finalize(args, &seen_fact_ids);
                        if let Ok(ref r) = result {
                            tracing::Span::current().record("iterations", iteration + 1);
                            tracing::Span::current().record("source_count", r.sources.len());
                        }
                        return result;
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

        // Should not be reachable — last iteration forces done() — but handle gracefully.
        error!("reflect agent exhausted all iterations without calling done()");
        Err(crate::error::Error::Llm("reflect agent exhausted all iterations without calling done()".into()))
    }
}

impl DefaultReflectPipeline {
    /// Finalize a reflect result from a `done` tool call.
    fn finalize(
        &self,
        args: DoneArgs,
        seen_fact_ids: &HashSet<FactId>,
    ) -> Result<ReflectResult> {
        // Validate source_ids against seen_fact_ids (drop hallucinated IDs)
        let sources: Vec<FactId> = args
            .source_ids
            .iter()
            .filter_map(|s| s.parse::<FactId>().ok())
            .filter(|id| seen_fact_ids.contains(id))
            .collect();

        Ok(ReflectResult {
            response: args.response,
            sources,
            new_opinions: vec![],
            confidence: 0.85,
        })
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
    use crate::recall::{DefaultRecallPipeline, RecallPipeline};
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
            let recall: Arc<dyn RecallPipeline> = Arc::new(DefaultRecallPipeline::new(
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

            DefaultReflectPipeline::new(
                recall,
                self.llm.clone(),
                self.store.clone(),
                8,
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

        // Iteration 0: forced search_observations
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "search_observations".into(),
                arguments: serde_json::json!({"query": "Rust memory", "reason": "overview"}),
            }],
        });

        // Iteration 1: forced recall
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "Rust memory safety", "reason": "details"}),
            }],
        });

        // Iteration 2: done
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_3".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": format!("Rust uses ownership for memory safety [{}].", fact_id),
                    "source_ids": [fact_id.to_string()]
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
    }

    #[tokio::test]
    async fn last_iteration_forces_done() {
        // Use max_iterations=3 so iteration 2 is the last and forces done.
        let h = TestHarness::new().await;
        let emb = h.embeddings.embed(&["data"]).await.unwrap();

        let fact = make_fact_with_embedding(
            h.bank_id,
            "Important data fact",
            NetworkType::World,
            emb[0].clone(),
        );
        h.store.insert_facts(&[fact]).await.unwrap();

        // Iteration 0: forced search_observations
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "search_observations".into(),
                arguments: serde_json::json!({"query": "data", "reason": "overview"}),
            }],
        });

        // Iteration 1: forced recall
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "data", "reason": "details"}),
            }],
        });

        // Iteration 2 (last): forced done — LLM must synthesize
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_3".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": "Here is the data summary.",
                    "source_ids": []
                }),
            }],
        });

        // Build pipeline with max_iterations=3
        let recall: Arc<dyn RecallPipeline> = Arc::new(DefaultRecallPipeline::new(
            Box::new(SemanticRetriever::new(h.store.clone(), h.embeddings.clone(), 20)),
            Box::new(KeywordRetriever::new(h.store.clone(), 20)),
            Box::new(GraphRetriever::new(h.store.clone(), h.embeddings.clone(), GraphRetrieverConfig::default())),
            Box::new(TemporalRetriever::new(h.store.clone())),
            Box::new(NoOpReranker),
            Box::new(EstimateTokenizer),
            60.0,
            50,
        ));
        let pipeline = DefaultReflectPipeline::new(recall, h.llm.clone(), h.store.clone(), 3);

        let result = pipeline
            .reflect(&ReflectQuery {
                bank_id: h.bank_id,
                question: "Tell me about data".into(),
                budget_tokens: 2000,
            })
            .await
            .unwrap();

        assert!(result.response.contains("data summary"));
        assert_eq!(h.llm.remaining(), 0);
    }

    #[tokio::test]
    async fn reflect_done_validates_source_ids() {
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

        // Iteration 0: search_observations
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "search_observations".into(),
                arguments: serde_json::json!({"query": "testing", "reason": "overview"}),
            }],
        });

        // Iteration 1: recall (finds the world fact)
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "testing", "reason": "details"}),
            }],
        });

        // Iteration 2: done with one real ID and one hallucinated ID
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_3".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": format!("Testing is important [{}].", fact_id),
                    "source_ids": [fact_id.to_string(), FactId::new().to_string()]
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

        // Hallucinated ID should be dropped, only the real one kept
        assert_eq!(result.sources.len(), 1);
        assert!(result.sources.contains(&fact_id));
    }

    #[tokio::test]
    async fn empty_memory_graceful_response() {
        let h = TestHarness::new().await;

        // Iteration 0: forced search_observations (empty bank)
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "search_observations".into(),
                arguments: serde_json::json!({"query": "meaning of life", "reason": "overview"}),
            }],
        });

        // Iteration 1: forced recall (empty bank)
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "meaning of life", "reason": "details"}),
            }],
        });

        // Iteration 2: done (empty bank, nothing found)
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_3".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": "I don't have enough information to answer this question.",
                    "source_ids": []
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

        // Iteration 0: search_observations
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "search_observations".into(),
                arguments: serde_json::json!({"query": "analysis", "reason": "overview"}),
            }],
        });
        // Iteration 1: recall
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "analysis", "reason": "details"}),
            }],
        });
        // Iteration 2: done
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_3".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": "Based on available context, here is my analysis.",
                    "source_ids": []
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

        // Iteration 0: forced search_observations (World facts won't match Observation filter)
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "search_observations".into(),
                arguments: serde_json::json!({"query": "data", "reason": "overview"}),
            }],
        });
        // Iteration 1: forced recall (finds World facts)
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "first data", "reason": "details"}),
            }],
        });
        // Iteration 2: another recall to cover second angle
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_3".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "second data", "reason": "more details"}),
            }],
        });
        // Iteration 3: done
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_4".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": format!("Facts [{id1}] and [{id2}] about data."),
                    "source_ids": [id1.to_string(), id2.to_string()]
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
