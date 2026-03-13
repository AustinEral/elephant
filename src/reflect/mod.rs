//! Reflect pipeline — agentic CARA reasoning with tool-calling loop.

pub mod disposition;
pub mod hierarchy;
pub mod opinion;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::{Instrument, debug, error, info, info_span, trace};

use crate::error::Result;
use crate::llm::LlmClient;
use crate::recall::RecallPipeline;
use crate::storage::MemoryStore;
use crate::types::llm::{CompletionRequest, Message, ToolChoice, ToolDef, ToolResult};
use crate::types::{
    Fact, FactId, NetworkType, RecallQuery, ReflectDoneTrace, ReflectQuery, ReflectResult,
    ReflectTraceStep, RetrievedFact, RetrievedSource, Source, SourceId, TurnId,
};

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
    max_output_tokens: Option<usize>,
    source_lookup_limit: usize,
    source_content_max_chars: Option<usize>,
}

/// Reflect agent prompt template.
pub const REFLECT_AGENT_PROMPT_TEMPLATE: &str = include_str!("../../prompts/reflect_agent.txt");
/// Reflect agent temperature.
pub const REFLECT_TEMPERATURE: f32 = 0.3;
/// Default maximum number of sources returned per fact for source lookups.
pub const DEFAULT_SOURCE_LOOKUP_LIMIT: usize = 3;

impl DefaultReflectPipeline {
    /// Create a new reflect pipeline.
    pub fn new(
        recall: Arc<dyn RecallPipeline>,
        llm: Arc<dyn LlmClient>,
        store: Arc<dyn MemoryStore>,
        max_iterations: usize,
    ) -> Self {
        Self::new_with_limits(
            recall,
            llm,
            store,
            max_iterations,
            None,
            DEFAULT_SOURCE_LOOKUP_LIMIT,
            None,
        )
    }

    /// Create a new reflect pipeline with explicit completion limits.
    pub fn new_with_limits(
        recall: Arc<dyn RecallPipeline>,
        llm: Arc<dyn LlmClient>,
        store: Arc<dyn MemoryStore>,
        max_iterations: usize,
        max_output_tokens: Option<usize>,
        source_lookup_limit: usize,
        source_content_max_chars: Option<usize>,
    ) -> Self {
        Self {
            recall,
            llm,
            store,
            max_iterations,
            max_output_tokens,
            source_lookup_limit: source_lookup_limit.max(1),
            source_content_max_chars,
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

/// Arguments for provenance lookup by fact id.
#[derive(Debug, Deserialize, JsonSchema)]
struct LookupSourcesArgs {
    /// Fact IDs whose linked sources should be returned.
    fact_ids: Vec<String>,
    /// Optional per-fact cap, clamped to the pipeline maximum.
    #[serde(default)]
    limit: Option<usize>,
    /// Why you need the provenance lookup.
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
        ToolDef::from_schema::<LookupSourcesArgs>(
            "lookup_sources",
            "Look up exact extraction inputs for known fact IDs. Use when original source text or source timing would help answer.",
        ),
        ToolDef::from_schema::<DoneArgs>("done", "Return the final answer with source citations."),
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
        system_parts.push(REFLECT_AGENT_PROMPT_TEMPLATE.to_string());
        let profile_prompt = build_profile_prompt(&bank_profile);
        if !profile_prompt.is_empty() {
            system_parts.push(profile_prompt);
        }
        let system_prompt = system_parts.join("\n\n");

        // Conversation messages for the agent loop
        let mut messages: Vec<Message> = vec![Message::text("user", query.question.clone())];

        let mut seen_fact_ids: HashSet<FactId> = HashSet::new();
        let mut seen_source_keys: HashSet<(FactId, SourceId)> = HashSet::new();
        let mut retrieved_context: Vec<RetrievedFact> = Vec::new();
        let mut retrieved_sources: Vec<RetrievedSource> = Vec::new();
        let mut trace: Vec<ReflectTraceStep> = Vec::new();
        let mut support_cache: HashMap<FactId, Fact> = HashMap::new();

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
                (&done_only, Some(ToolChoice::Specific("done".into())))
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
                temperature: Some(REFLECT_TEMPERATURE),
                max_tokens: self.max_output_tokens,
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
                            Some(vec![
                                NetworkType::World,
                                NetworkType::Experience,
                                NetworkType::Opinion,
                            ])
                        };

                        let args: SearchArgs = serde_json::from_value(tc.arguments.clone())
                            .unwrap_or(SearchArgs {
                                query: query.question.clone(),
                                reason: String::new(),
                            });

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
                        let returned_fact_ids =
                            result.facts.iter().map(|sf| sf.fact.id).collect::<Vec<_>>();
                        let mut new_fact_ids = Vec::new();
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
                                new_fact_ids.push(sf.fact.id);
                                writeln!(new_facts, "[FACT {}] {}", sf.fact.id, sf.fact.content)
                                    .unwrap();
                                let support_turn_ids = self
                                    .support_turn_ids_for_fact(&sf.fact, &mut support_cache)
                                    .await?;
                                retrieved_context.push(RetrievedFact {
                                    id: sf.fact.id,
                                    content: sf.fact.content.clone(),
                                    score: sf.score,
                                    network: sf.fact.network,
                                    source_turn_id: sf.fact.source_turn_id,
                                    evidence_ids: sf.fact.evidence_ids.clone(),
                                    retrieval_sources: sf.sources.clone(),
                                    support_turn_ids,
                                });
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

                        trace.push(ReflectTraceStep {
                            iteration,
                            tool_name: tc.name.clone(),
                            query: args.query.clone(),
                            returned_fact_ids,
                            requested_fact_ids: vec![],
                            new_fact_ids,
                            returned_source_ids: vec![],
                            facts_returned: result.facts.len(),
                            total_tokens: result.total_tokens,
                            latency_ms: recall_ms,
                        });

                        if new_facts.is_empty() {
                            new_facts = "No new memories found for this query.".into();
                        }

                        tool_results.push(ToolResult {
                            tool_call_id: tc.id.clone(),
                            content: new_facts,
                        });
                    }
                    "lookup_sources" => {
                        let args: LookupSourcesArgs = serde_json::from_value(tc.arguments.clone())
                            .unwrap_or(LookupSourcesArgs {
                                fact_ids: vec![],
                                limit: None,
                                reason: String::new(),
                            });

                        let requested_fact_ids = Self::parse_requested_fact_ids(&args.fact_ids);
                        let per_fact_limit = self
                            .effective_source_limit(args.limit)
                            .min(self.source_lookup_limit);

                        let lookup_start = Instant::now();
                        let lookups = self
                            .store
                            .lookup_sources(&requested_fact_ids, per_fact_limit)
                            .await?;
                        let lookup_ms = lookup_start.elapsed().as_millis() as u64;
                        let returned_fact_ids = lookups
                            .iter()
                            .map(|lookup| lookup.fact_id)
                            .collect::<Vec<_>>();
                        let sources_returned = lookups
                            .iter()
                            .map(|lookup| lookup.sources.len())
                            .sum::<usize>();
                        let (formatted, surfaced_sources) =
                            self.format_source_lookups(&requested_fact_ids, &lookups);
                        let mut returned_source_ids = Vec::new();
                        for source in surfaced_sources {
                            returned_source_ids.push(source.id);
                            if seen_source_keys.insert((source.fact_id, source.id)) {
                                retrieved_sources.push(source);
                            }
                        }

                        debug!(
                            tool_name = tc.name.as_str(),
                            requested_fact_ids = args.fact_ids.len(),
                            facts_with_sources = returned_fact_ids.len(),
                            sources_returned,
                            lookup_ms,
                            "tool_call"
                        );

                        trace.push(ReflectTraceStep {
                            iteration,
                            tool_name: tc.name.clone(),
                            query: args.fact_ids.join(", "),
                            returned_fact_ids,
                            requested_fact_ids,
                            new_fact_ids: vec![],
                            returned_source_ids,
                            facts_returned: sources_returned,
                            total_tokens: Self::estimate_text_tokens(&formatted),
                            latency_ms: lookup_ms,
                        });

                        tool_results.push(ToolResult {
                            tool_call_id: tc.id.clone(),
                            content: formatted,
                        });
                    }
                    "done" => {
                        let (args, final_done) = Self::parse_done_call(
                            iteration,
                            response.stop_reason.as_deref(),
                            &response.content,
                            &tc.arguments,
                        );
                        info!(
                            response = args.response.as_str(),
                            source_ids = ?args.source_ids,
                            "done"
                        );
                        let result = self.finalize(
                            args,
                            &seen_fact_ids,
                            retrieved_context.clone(),
                            retrieved_sources.clone(),
                            trace,
                            final_done,
                        );
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
                content: response.content.clone(),
                tool_calls: response.tool_calls.clone(),
                tool_results: vec![],
            });
            messages.push(Message {
                role: "user".into(),
                content: String::new(),
                tool_calls: vec![],
                tool_results,
            });
        }

        // Should not be reachable — last iteration forces done() — but handle gracefully.
        error!("reflect agent exhausted all iterations without calling done()");
        Err(crate::error::Error::Llm(
            "reflect agent exhausted all iterations without calling done()".into(),
        ))
    }
}

impl DefaultReflectPipeline {
    fn effective_source_limit(&self, requested: Option<usize>) -> usize {
        requested
            .unwrap_or(self.source_lookup_limit)
            .max(1)
            .min(self.source_lookup_limit)
    }

    fn parse_requested_fact_ids(raw_ids: &[String]) -> Vec<FactId> {
        let mut seen = HashSet::new();
        let mut parsed = Vec::new();
        for raw_id in raw_ids {
            if let Ok(fact_id) = raw_id.parse::<FactId>()
                && seen.insert(fact_id)
            {
                parsed.push(fact_id);
            }
        }
        parsed
    }

    fn format_source_lookups(
        &self,
        requested_fact_ids: &[FactId],
        lookups: &[crate::types::FactSourceLookup],
    ) -> (String, Vec<RetrievedSource>) {
        if requested_fact_ids.is_empty() {
            return (
                "No valid fact IDs were provided for source lookup.".into(),
                vec![],
            );
        }

        let mut grouped = HashMap::new();
        for lookup in lookups {
            grouped.insert(lookup.fact_id, &lookup.sources);
        }

        let mut output = String::new();
        let mut surfaced_sources = Vec::new();
        for fact_id in requested_fact_ids {
            let _ = writeln!(output, "[FACT {fact_id}]");
            match grouped.get(fact_id) {
                Some(sources) if !sources.is_empty() => {
                    for source in *sources {
                        let (content, truncated) = self.render_source_content(source);
                        let _ = writeln!(output, "[SOURCE {}]", source.id);
                        let _ = writeln!(output, "{content}");
                        let _ = writeln!(output, "[END SOURCE]");
                        surfaced_sources.push(RetrievedSource {
                            id: source.id,
                            fact_id: *fact_id,
                            timestamp: source.timestamp,
                            content,
                            truncated,
                        });
                    }
                }
                _ => {
                    let _ = writeln!(output, "No linked sources found.");
                }
            }
        }

        (output.trim_end().to_string(), surfaced_sources)
    }

    fn render_source_content(&self, source: &Source) -> (String, bool) {
        let base = source.rendered_input.as_deref().unwrap_or(&source.content);
        match self.source_content_max_chars {
            Some(limit) => {
                let total_chars = base.chars().count();
                if total_chars > limit {
                    let mut content = base.chars().take(limit).collect::<String>();
                    content.push_str("...");
                    (content, true)
                } else {
                    (base.to_string(), false)
                }
            }
            None => (base.to_string(), false),
        }
    }

    fn estimate_text_tokens(text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn parse_done_call(
        iteration: usize,
        stop_reason: Option<&str>,
        assistant_content: &str,
        arguments: &serde_json::Value,
    ) -> (DoneArgs, ReflectDoneTrace) {
        match serde_json::from_value::<DoneArgs>(arguments.clone()) {
            Ok(args) => {
                let trace = ReflectDoneTrace {
                    iteration,
                    assistant_content: assistant_content.to_string(),
                    raw_arguments: arguments.clone(),
                    used_fallback: false,
                    parse_error: None,
                    stop_reason: stop_reason.map(str::to_string),
                    response: args.response.clone(),
                    source_ids: args.source_ids.clone(),
                };
                (args, trace)
            }
            Err(err) => {
                // LLM sometimes sends source_ids as a string instead of array.
                // Fall back to extracting just the response field.
                let response = arguments
                    .get("response")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let args = DoneArgs {
                    response: response.clone(),
                    source_ids: vec![],
                };
                let trace = ReflectDoneTrace {
                    iteration,
                    assistant_content: assistant_content.to_string(),
                    raw_arguments: arguments.clone(),
                    used_fallback: true,
                    parse_error: Some(err.to_string()),
                    stop_reason: stop_reason.map(str::to_string),
                    response,
                    source_ids: vec![],
                };
                (args, trace)
            }
        }
    }

    async fn support_turn_ids_for_fact(
        &self,
        fact: &Fact,
        cache: &mut HashMap<FactId, Fact>,
    ) -> Result<Vec<TurnId>> {
        let mut turn_ids = BTreeSet::new();
        if let Some(turn_id) = fact.source_turn_id {
            turn_ids.insert(turn_id);
        }

        let mut pending = fact.evidence_ids.clone();
        let mut seen = pending.iter().copied().collect::<HashSet<_>>();

        while !pending.is_empty() {
            let missing = pending
                .iter()
                .copied()
                .filter(|id| !cache.contains_key(id))
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                let fetched = self.store.get_facts(&missing).await?;
                for support in fetched {
                    cache.insert(support.id, support);
                }
            }

            let current = std::mem::take(&mut pending);
            for fact_id in current {
                let Some(support) = cache.get(&fact_id) else {
                    continue;
                };
                if let Some(turn_id) = support.source_turn_id {
                    turn_ids.insert(turn_id);
                }
                for evidence_id in &support.evidence_ids {
                    if seen.insert(*evidence_id) {
                        pending.push(*evidence_id);
                    }
                }
            }
        }

        Ok(turn_ids.into_iter().collect())
    }

    /// Finalize a reflect result from a `done` tool call.
    fn finalize(
        &self,
        args: DoneArgs,
        seen_fact_ids: &HashSet<FactId>,
        retrieved_context: Vec<RetrievedFact>,
        retrieved_sources: Vec<RetrievedSource>,
        trace: Vec<ReflectTraceStep>,
        final_done: ReflectDoneTrace,
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
            retrieved_context,
            retrieved_sources,
            trace,
            final_done: Some(final_done),
        })
    }
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
    use crate::embedding::EmbeddingClient;
    use crate::embedding::mock::MockEmbeddings;
    use crate::llm::mock::MockLlmClient;
    use crate::recall::budget::EstimateTokenizer;
    use crate::recall::graph::{GraphRetriever, GraphRetrieverConfig};
    use crate::recall::keyword::KeywordRetriever;
    use crate::recall::reranker::NoOpReranker;
    use crate::recall::semantic::SemanticRetriever;
    use crate::recall::temporal::TemporalRetriever;
    use crate::recall::{DefaultRecallPipeline, RecallPipeline};
    use crate::storage::MemoryStore;
    use crate::storage::mock::MockMemoryStore;
    use crate::types::llm::CompletionResponse;
    use crate::types::*;
    use chrono::{TimeZone, Utc};

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

            DefaultReflectPipeline::new(recall, self.llm.clone(), self.store.clone(), 8)
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
            stop_reason: Some("tool_use".into()),
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
            stop_reason: Some("tool_use".into()),
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
            stop_reason: Some("tool_use".into()),
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
        assert_eq!(result.trace.len(), 2);
        assert_eq!(result.trace[0].tool_name, "search_observations");
        assert_eq!(result.trace[1].tool_name, "recall");
        assert_eq!(result.trace[1].new_fact_ids, vec![fact_id]);
        let final_done = result.final_done.expect("final done trace");
        assert_eq!(final_done.iteration, 2);
        assert!(!final_done.used_fallback);
        assert_eq!(final_done.stop_reason.as_deref(), Some("tool_use"));
        assert!(final_done.response.contains("ownership"));
        assert_eq!(final_done.source_ids, vec![fact_id.to_string()]);
    }

    #[tokio::test]
    async fn reflect_done_fallback_trace_is_preserved() {
        let h = TestHarness::new().await;

        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            stop_reason: Some("tool_use".into()),
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "search_observations".into(),
                arguments: serde_json::json!({"query": "fallback", "reason": "overview"}),
            }],
        });
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            stop_reason: Some("tool_use".into()),
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "fallback", "reason": "details"}),
            }],
        });
        h.llm.push_response_full(CompletionResponse {
            content: "Synthesizing".into(),
            input_tokens: 10,
            output_tokens: 20,
            stop_reason: Some("max_tokens".into()),
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_3".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": "",
                    "source_ids": "not-an-array"
                }),
            }],
        });

        let pipeline = h.build_pipeline();
        let result = pipeline
            .reflect(&ReflectQuery {
                bank_id: h.bank_id,
                question: "Why did fallback happen?".into(),
                budget_tokens: 2000,
            })
            .await
            .unwrap();

        let final_done = result.final_done.expect("final done trace");
        assert!(final_done.used_fallback);
        assert!(final_done.parse_error.is_some());
        assert_eq!(final_done.stop_reason.as_deref(), Some("max_tokens"));
        assert_eq!(final_done.response, "");
        assert!(final_done.source_ids.is_empty());
        assert_eq!(final_done.assistant_content, "Synthesizing");
        assert_eq!(result.response, "");
    }

    #[tokio::test]
    async fn observation_support_turn_ids_are_expanded_from_evidence() {
        let h = TestHarness::new().await;
        let emb = h.embeddings.embed(&["project timeline"]).await.unwrap();
        let turn_id = TurnId::new();

        let raw_fact = Fact {
            id: FactId::new(),
            bank_id: h.bank_id,
            content: "Alice finalized the project timeline in March.".into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(emb[0].clone()),
            confidence: None,
            evidence_ids: vec![],
            source_turn_id: Some(turn_id),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            consolidated_at: None,
        };
        let raw_fact_id = raw_fact.id;
        let observation = Fact {
            id: FactId::new(),
            bank_id: h.bank_id,
            content: "Alice completed the project timeline in March.".into(),
            fact_type: FactType::World,
            network: NetworkType::Observation,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(emb[0].clone()),
            confidence: None,
            evidence_ids: vec![raw_fact_id],
            source_turn_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            consolidated_at: None,
        };
        let observation_id = observation.id;
        h.store
            .insert_facts(&[raw_fact, observation])
            .await
            .unwrap();

        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            stop_reason: None,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "search_observations".into(),
                arguments: serde_json::json!({"query": "project timeline", "reason": "overview"}),
            }],
        });
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            stop_reason: None,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "project timeline", "reason": "details"}),
            }],
        });
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            stop_reason: None,
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_3".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": format!("Alice finished the timeline [{}].", observation_id),
                    "source_ids": [observation_id.to_string()]
                }),
            }],
        });

        let pipeline = h.build_pipeline();
        let result = pipeline
            .reflect(&ReflectQuery {
                bank_id: h.bank_id,
                question: "When was the project timeline finished?".into(),
                budget_tokens: 2000,
            })
            .await
            .unwrap();

        let observation_entry = result
            .retrieved_context
            .iter()
            .find(|fact| fact.id == observation_id)
            .expect("observation should be in retrieved context");
        assert_eq!(observation_entry.evidence_ids, vec![raw_fact_id]);
        assert_eq!(observation_entry.support_turn_ids, vec![turn_id]);
    }

    #[tokio::test]
    async fn reflect_can_lookup_sources_for_retrieved_facts() {
        let h = TestHarness::new().await;
        let emb = h.embeddings.embed(&["launch plan"]).await.unwrap();

        let fact = make_fact_with_embedding(
            h.bank_id,
            "Avery shared the launch plan with the team",
            NetworkType::World,
            emb[0].clone(),
        );
        let fact_id = fact.id;
        h.store.insert_facts(&[fact]).await.unwrap();

        let source = Source {
            id: SourceId::new(),
            bank_id: h.bank_id,
            content: "Avery shared the launch plan with the team during the July review meeting."
                .into(),
            context: Some("Timelines and launch planning came up in the earlier discussion.".into()),
            speaker: Some("Avery".into()),
            rendered_input: Some(
                "Speaker: Avery\n\n## Preceding Context\n\nTimelines and launch planning came up in the earlier discussion.\n\n---\n\n## Content to Extract From\n\nAvery shared the launch plan with the team during the July review meeting.\n\nTimestamp: 2024-07-23T18:46:00+00:00".into(),
            ),
            timestamp: Utc.with_ymd_and_hms(2024, 7, 23, 18, 46, 0).unwrap(),
            created_at: Utc.with_ymd_and_hms(2024, 7, 23, 18, 46, 1).unwrap(),
        };
        h.store.insert_source(&source).await.unwrap();
        h.store
            .link_facts_to_source(&[fact_id], source.id)
            .await
            .unwrap();

        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            stop_reason: Some("tool_use".into()),
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_1".into(),
                name: "search_observations".into(),
                arguments: serde_json::json!({"query": "launch plan", "reason": "overview"}),
            }],
        });
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            stop_reason: Some("tool_use".into()),
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_2".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "Avery launch plan", "reason": "details"}),
            }],
        });
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            stop_reason: Some("tool_use".into()),
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_3".into(),
                name: "lookup_sources".into(),
                arguments: serde_json::json!({
                    "fact_ids": [fact_id.to_string()],
                    "limit": 1,
                    "reason": "Find when the memory was originally said"
                }),
            }],
        });
        h.llm.push_response_full(CompletionResponse {
            content: String::new(),
            input_tokens: 10,
            output_tokens: 20,
            stop_reason: Some("tool_use".into()),
            tool_calls: vec![crate::types::llm::ToolCall {
                id: "call_4".into(),
                name: "done".into(),
                arguments: serde_json::json!({
                    "response": format!("Avery said it on July 23, 2024 [{}].", fact_id),
                    "source_ids": [fact_id.to_string()]
                }),
            }],
        });

        let pipeline = h.build_pipeline();
        let result = pipeline
            .reflect(&ReflectQuery {
                bank_id: h.bank_id,
                question: "When did Avery say the launch plan was shared?".into(),
                budget_tokens: 2000,
            })
            .await
            .unwrap();

        assert_eq!(result.trace.len(), 3);
        assert_eq!(result.trace[2].tool_name, "lookup_sources");
        assert_eq!(result.trace[2].requested_fact_ids, vec![fact_id]);
        assert_eq!(result.trace[2].returned_fact_ids, vec![fact_id]);
        assert_eq!(result.trace[2].returned_source_ids, vec![source.id]);
        assert_eq!(result.trace[2].facts_returned, 1);
        assert_eq!(result.retrieved_sources.len(), 1);
        assert_eq!(result.retrieved_sources[0].id, source.id);
        assert_eq!(result.retrieved_sources[0].fact_id, fact_id);
        assert_eq!(result.retrieved_sources[0].timestamp, source.timestamp);
        assert_eq!(
            result.retrieved_sources[0].content,
            source.rendered_input.clone().unwrap()
        );
        assert!(!result.retrieved_sources[0].truncated);
        assert!(result.response.contains("July 23, 2024"));
        assert_eq!(result.sources, vec![fact_id]);
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
            stop_reason: None,
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
            stop_reason: None,
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
            stop_reason: None,
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
            Box::new(SemanticRetriever::new(
                h.store.clone(),
                h.embeddings.clone(),
                20,
            )),
            Box::new(KeywordRetriever::new(h.store.clone(), 20)),
            Box::new(GraphRetriever::new(
                h.store.clone(),
                h.embeddings.clone(),
                GraphRetrieverConfig::default(),
            )),
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
            stop_reason: None,
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
            stop_reason: None,
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
            stop_reason: None,
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
            stop_reason: None,
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
            stop_reason: None,
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
            stop_reason: None,
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
            stop_reason: None,
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
            stop_reason: None,
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
            stop_reason: None,
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
        assert!(
            profile
                .disposition_prompt
                .contains("reads between the lines")
        );
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

        assert!(
            result.is_err(),
            "reflect should fail when bank doesn't exist"
        );
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
            stop_reason: None,
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
            stop_reason: None,
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
            stop_reason: None,
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
            stop_reason: None,
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
