//! Topic-scoped observation consolidator — produces multiple focused observations
//! from batches of raw facts, Vectorize-style.
//!
//! Instead of one monolithic observation per entity, facts are processed in batches.
//! For each batch the LLM decides whether to CREATE new or UPDATE existing observations,
//! keeping each observation focused on a single topic/facet.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::Utc;
use futures::future::join_all;
use serde::Deserialize;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{Instrument, debug, info, info_span, warn};

use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::llm::{
    CompletionRequest, LlmClient, Message, ReasoningEffort, StructuredOutputRetryOptions,
    StructuredResponseErrorKind, complete_structured_with_retries,
};
use crate::recall::RecallPipeline;
use crate::storage::MemoryStore;
use crate::types::id::{BankId, FactId};
use crate::types::{
    ConsolidationBreakdown, ConsolidationReport, Fact, FactFilter, FactType, NetworkType,
    RecallQuery, TemporalRange,
};

/// Static configuration for observation consolidation.
#[derive(Debug, Clone, Copy)]
pub struct ConsolidationConfig {
    /// Number of unconsolidated facts per LLM batch.
    pub batch_size: usize,
    /// Completion cap for the consolidation prompt.
    pub max_tokens: usize,
    /// Recall token budget while consolidating observations.
    pub recall_budget: usize,
    /// Total number of attempts for malformed structured output from the consolidator LLM.
    pub structured_output_max_attempts: usize,
    /// Explicit sampling-temperature override for observation consolidation.
    pub temperature: Option<f32>,
    /// Reasoning effort override for observation consolidation, if supported.
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            max_tokens: 4096,
            recall_budget: 512,
            structured_output_max_attempts: 3,
            temperature: None,
            reasoning_effort: None,
        }
    }
}

/// Consolidates raw facts into topic-scoped observations.
#[async_trait]
pub trait Consolidator: Send + Sync {
    /// Process unconsolidated facts and create/update observations.
    async fn consolidate(&self, bank_id: BankId) -> Result<ConsolidationReport> {
        self.consolidate_with_progress(bank_id, None).await
    }

    /// Process unconsolidated facts and optionally emit per-batch progress events.
    async fn consolidate_with_progress(
        &self,
        bank_id: BankId,
        progress: Option<UnboundedSender<ConsolidationProgress>>,
    ) -> Result<ConsolidationReport>;
}

/// Progress update emitted after each consolidation batch commits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsolidationProgress {
    /// Total unconsolidated facts seen at the start of consolidation.
    pub total_facts: usize,
    /// Total number of batches in this consolidation run.
    pub total_batches: usize,
    /// One-based batch index that just completed.
    pub batch_index: usize,
    /// Number of raw facts processed in this batch.
    pub batch_facts: usize,
    /// Cumulative observations created so far.
    pub observations_created: usize,
    /// Cumulative observations updated so far.
    pub observations_updated: usize,
}

/// Default implementation using LLM for topic-scoped synthesis.
pub struct DefaultConsolidator {
    store: Arc<dyn MemoryStore>,
    llm: Arc<dyn LlmClient>,
    embeddings: Arc<dyn EmbeddingClient>,
    recall: Arc<dyn RecallPipeline>,
    config: ConsolidationConfig,
}

impl DefaultConsolidator {
    /// Create a new consolidator.
    pub fn new(
        store: Arc<dyn MemoryStore>,
        llm: Arc<dyn LlmClient>,
        embeddings: Arc<dyn EmbeddingClient>,
        recall: Arc<dyn RecallPipeline>,
        config: ConsolidationConfig,
    ) -> Self {
        Self {
            store,
            llm,
            embeddings,
            recall,
            config,
        }
    }
}

#[derive(Default)]
struct ConsolidationTimingTotals {
    load_unconsolidated: Duration,
    recall: Duration,
    prompt_build: Duration,
    llm_consolidate: Duration,
    begin_txn: Duration,
    action_embed: Duration,
    db_write: Duration,
    mark_consolidated: Duration,
    commit: Duration,
}

impl ConsolidationTimingTotals {
    fn apply_to(self, breakdown: &mut ConsolidationBreakdown) {
        breakdown.load_unconsolidated_ms = duration_to_ms(self.load_unconsolidated);
        breakdown.recall_ms = duration_to_ms(self.recall);
        breakdown.prompt_build_ms = duration_to_ms(self.prompt_build);
        breakdown.llm_consolidate_ms = duration_to_ms(self.llm_consolidate);
        breakdown.begin_txn_ms = duration_to_ms(self.begin_txn);
        breakdown.action_embed_ms = duration_to_ms(self.action_embed);
        breakdown.db_write_ms = duration_to_ms(self.db_write);
        breakdown.mark_consolidated_ms = duration_to_ms(self.mark_consolidated);
        breakdown.commit_ms = duration_to_ms(self.commit);
    }
}

fn duration_to_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[derive(Deserialize)]
struct ConsolidateResponse {
    actions: Vec<ConsolidateAction>,
}

#[derive(Deserialize)]
struct ConsolidateAction {
    action: String,
    content: String,
    fact_indices: Vec<usize>,
    observation_index: Option<usize>,
    #[serde(default)]
    observation_id: Option<String>,
}

/// Observation consolidation prompt template.
pub const CONSOLIDATE_PROMPT: &str = include_str!("../../prompts/consolidate_topics.txt");
/// Format a temporal range as a pipe-separated suffix for consolidation prompts.
/// Returns e.g. `" | occurred: 2022-01-01 to 2022-12-31"` or empty string if no range.
fn format_temporal_suffix(tr: Option<&TemporalRange>) -> String {
    match tr {
        Some(tr) => {
            let start_str = tr.start.map(|d| d.format("%Y-%m-%d").to_string());
            let end_str = tr.end.map(|d| d.format("%Y-%m-%d").to_string());
            match (start_str, end_str) {
                (Some(s), Some(e)) if s == e => format!(" | occurred: {s}"),
                (Some(s), Some(e)) => format!(" | occurred: {s} to {e}"),
                (Some(s), None) => format!(" | occurred: {s}"),
                (None, Some(e)) => format!(" | occurred: until {e}"),
                (None, None) => String::new(),
            }
        }
        None => String::new(),
    }
}

/// Merge temporal ranges from source facts into an existing range using LEAST(start)/GREATEST(end).
fn merge_temporal(existing: Option<&TemporalRange>, facts: &[&Fact]) -> Option<TemporalRange> {
    let mut start = existing.and_then(|r| r.start);
    let mut end = existing.and_then(|r| r.end);

    for f in facts {
        if let Some(ref tr) = f.temporal_range {
            start = match (start, tr.start) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            end = match (end, tr.end) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
        }
    }

    match (start, end) {
        (None, None) => None,
        _ => Some(TemporalRange { start, end }),
    }
}

#[async_trait]
impl Consolidator for DefaultConsolidator {
    async fn consolidate_with_progress(
        &self,
        bank_id: BankId,
        progress: Option<UnboundedSender<ConsolidationProgress>>,
    ) -> Result<ConsolidationReport> {
        let consolidate_span = info_span!("consolidate",
            bank_id = %bank_id,
            unconsolidated = tracing::field::Empty,
            created = tracing::field::Empty,
            updated = tracing::field::Empty,
        );
        self.consolidate_inner(bank_id, progress)
            .instrument(consolidate_span)
            .await
    }
}

impl DefaultConsolidator {
    async fn consolidate_inner(
        &self,
        bank_id: BankId,
        progress: Option<UnboundedSender<ConsolidationProgress>>,
    ) -> Result<ConsolidationReport> {
        let mut report = ConsolidationReport::default();
        let mut timings = ConsolidationTimingTotals::default();

        // 1. Fetch unconsolidated World/Experience facts
        let load_started = Instant::now();
        let unconsolidated = self
            .store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::World, NetworkType::Experience]),
                    unconsolidated_only: true,
                    ..Default::default()
                },
            )
            .await?;
        timings.load_unconsolidated += load_started.elapsed();
        report.breakdown.unconsolidated_fact_count = unconsolidated.len();

        if unconsolidated.is_empty() {
            debug!("no unconsolidated facts");
            return Ok(report);
        }

        tracing::Span::current().record("unconsolidated", unconsolidated.len());
        debug!(facts = unconsolidated.len(), "fetched unconsolidated facts");

        // 2. Process in batches
        let bs = self.config.batch_size;
        let total_batches = unconsolidated.len().div_ceil(bs);
        for (batch_idx, batch) in unconsolidated.chunks(bs).enumerate() {
            report.breakdown.batch_count += 1;
            debug!(
                batch = batch_idx + 1,
                total_batches,
                facts = batch.len(),
                "processing batch"
            );
            // 3a. Per-fact recall: full pipeline (semantic + keyword + graph + temporal)
            // for each fact in parallel, then union/dedup the results.
            let recall_started = Instant::now();
            let budget = self.config.recall_budget;
            let recall_futures: Vec<_> = batch
                .iter()
                .map(|fact| {
                    let recall = self.recall.clone();
                    let mut query = RecallQuery::new(bank_id, fact.content.clone())
                        .with_budget_tokens(budget)
                        .with_network_filter(vec![NetworkType::Observation]);
                    if let Some(anchor) = fact.temporal_range.clone() {
                        query = query.with_temporal_anchor(anchor);
                    }
                    async move { recall.recall(&query).await }
                })
                .collect();

            let recall_results = join_all(recall_futures).await;

            let mut seen_obs_ids = std::collections::HashSet::new();
            let mut related_observations = Vec::new();
            for result in recall_results {
                let result = result?;
                for sf in result.facts {
                    if seen_obs_ids.insert(sf.fact.id) {
                        related_observations.push(sf.fact);
                    }
                }
            }
            timings.recall += recall_started.elapsed();
            report.breakdown.related_observation_count += related_observations.len();

            // 3b. Format prompt with temporal annotations
            let prompt_started = Instant::now();
            let facts_text = batch
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let temporal = format_temporal_suffix(f.temporal_range.as_ref());
                    format!("[{i}] {}{temporal}", f.content)
                })
                .collect::<Vec<_>>()
                .join("\n");

            let obs_text = if related_observations.is_empty() {
                "(none)".to_string()
            } else {
                related_observations
                    .iter()
                    .enumerate()
                    .map(|(i, o)| {
                        let temporal = format_temporal_suffix(o.temporal_range.as_ref());
                        format!("[{i}] {}{temporal}", o.content)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };

            let prompt = CONSOLIDATE_PROMPT
                .replace("{facts}", &facts_text)
                .replace("{observations}", &obs_text);
            timings.prompt_build += prompt_started.elapsed();

            let mut request = CompletionRequest::builder()
                .message(Message::user(prompt))
                .max_tokens(self.config.max_tokens)
                .reasoning_effort_opt(self.config.reasoning_effort);
            if let Some(temperature) = self.config.temperature {
                request = request.temperature(temperature);
            }
            let request = request.build();

            let llm_started = Instant::now();
            let resp: ConsolidateResponse = complete_structured_with_retries(
                self.llm.as_ref(),
                request,
                StructuredOutputRetryOptions {
                    max_attempts: self.config.structured_output_max_attempts,
                    context: "consolidation structured output invalid",
                },
                |kind| {
                    matches!(
                        kind,
                        StructuredResponseErrorKind::Refusal
                            | StructuredResponseErrorKind::Empty
                            | StructuredResponseErrorKind::NoJson
                            | StructuredResponseErrorKind::JsonParse
                            | StructuredResponseErrorKind::JsonStructure
                    )
                },
            )
            .await
            .map_err(|error| {
                warn!(
                    batch = batch_idx + 1,
                    total_batches,
                    batch_facts = batch.len(),
                    error = %error,
                    "consolidation batch failed"
                );
                error
            })?;
            timings.llm_consolidate += llm_started.elapsed();
            report.breakdown.action_count += resp.actions.len();

            // 3c. Execute actions inside a transaction
            let txn_started = Instant::now();
            let txn = self.store.begin().await?;
            timings.begin_txn += txn_started.elapsed();

            for action in &resp.actions {
                let embed_started = Instant::now();
                let emb_vec = self.embeddings.embed(&[&action.content]).await?;
                let embedding = emb_vec.into_iter().next();
                timings.action_embed += embed_started.elapsed();

                // Collect source facts referenced by this action
                let source_facts: Vec<&Fact> = action
                    .fact_indices
                    .iter()
                    .filter_map(|&i| batch.get(i))
                    .collect();
                let evidence_ids: Vec<FactId> = source_facts.iter().map(|f| f.id).collect();

                // Union entity IDs from all source facts
                let mut entity_ids = Vec::new();
                for f in &source_facts {
                    for eid in &f.entity_ids {
                        if !entity_ids.contains(eid) {
                            entity_ids.push(*eid);
                        }
                    }
                }

                // Does the LLM want to update an existing observation?
                let updated_existing = if action.action == "update" {
                    let existing = action
                        .observation_index
                        .and_then(|index| related_observations.get(index))
                        .or_else(|| {
                            action.observation_id.as_deref().and_then(|obs_id_str| {
                                let normalized = obs_id_str.trim().trim_matches(&['[', ']'][..]);
                                related_observations
                                    .iter()
                                    .find(|o| o.id.to_string() == normalized)
                            })
                        });
                    if let Some(existing) = existing {
                        let write_started = Instant::now();
                        let mut updated = existing.clone();
                        updated.content = action.content.clone();
                        updated.embedding = embedding.clone();
                        for eid in &evidence_ids {
                            if !updated.evidence_ids.contains(eid) {
                                updated.evidence_ids.push(*eid);
                            }
                        }
                        for eid in &entity_ids {
                            if !updated.entity_ids.contains(eid) {
                                updated.entity_ids.push(*eid);
                            }
                        }
                        updated.temporal_range =
                            merge_temporal(updated.temporal_range.as_ref(), &source_facts);
                        updated.updated_at = Utc::now();
                        txn.update_fact(&updated).await?;
                        timings.db_write += write_started.elapsed();
                        report.observations_updated += 1;
                        true
                    } else {
                        warn!(
                            observation_index = ?action.observation_index,
                            observation_id = ?action.observation_id,
                            candidate_observations = related_observations.len(),
                            "LLM referenced unknown observation target, creating new instead"
                        );
                        false
                    }
                } else {
                    false
                };

                if !updated_existing {
                    let write_started = Instant::now();
                    let now = Utc::now();
                    let obs = Fact {
                        id: FactId::new(),
                        bank_id,
                        content: action.content.clone(),
                        fact_type: FactType::World,
                        network: NetworkType::Observation,
                        entity_ids,
                        temporal_range: merge_temporal(None, &source_facts),
                        embedding,
                        confidence: None,
                        evidence_ids,
                        source_turn_id: None,
                        created_at: now,
                        updated_at: now,
                        consolidated_at: None,
                    };
                    txn.insert_facts(&[obs]).await?;
                    timings.db_write += write_started.elapsed();
                    report.observations_created += 1;
                }
            }

            // 3d. Mark batch facts as consolidated
            let batch_ids: Vec<FactId> = batch.iter().map(|f| f.id).collect();
            let mark_started = Instant::now();
            txn.mark_consolidated(&batch_ids, Utc::now()).await?;
            timings.mark_consolidated += mark_started.elapsed();

            // Commit the batch transaction
            let commit_started = Instant::now();
            txn.commit().await?;
            timings.commit += commit_started.elapsed();
            debug!(
                batch = batch_idx + 1,
                created = report.observations_created,
                updated = report.observations_updated,
                "batch committed"
            );
            if let Some(progress) = &progress {
                let _ = progress.send(ConsolidationProgress {
                    total_facts: unconsolidated.len(),
                    total_batches,
                    batch_index: batch_idx + 1,
                    batch_facts: batch.len(),
                    observations_created: report.observations_created,
                    observations_updated: report.observations_updated,
                });
            }
        }

        info!(
            created = report.observations_created,
            updated = report.observations_updated,
            "consolidate_complete"
        );
        timings.apply_to(&mut report.breakdown);
        tracing::Span::current().record("created", report.observations_created);
        tracing::Span::current().record("updated", report.observations_updated);

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::mock::MockEmbeddings;
    use crate::llm::mock::MockLlmClient;
    use crate::storage::mock::MockMemoryStore;
    use crate::types::{RecallResult, RetrievalSource, ScoredFact};

    /// Mock recall pipeline that returns Observation facts from the store via vector_search.
    struct MockRecallPipeline {
        store: Arc<MockMemoryStore>,
    }

    #[async_trait]
    impl RecallPipeline for MockRecallPipeline {
        async fn recall(&self, query: &RecallQuery) -> Result<RecallResult> {
            // Return all observations in the bank (simple mock)
            let all = self
                .store
                .get_facts_by_bank(
                    query.bank_id,
                    FactFilter {
                        network: Some(vec![NetworkType::Observation]),
                        ..Default::default()
                    },
                )
                .await?;
            let facts = all
                .into_iter()
                .map(|f| ScoredFact {
                    fact: f,
                    score: 1.0,
                    sources: vec![RetrievalSource::Semantic],
                })
                .collect();
            Ok(RecallResult {
                facts,
                total_tokens: 0,
            })
        }
    }

    fn setup() -> (
        Arc<MockMemoryStore>,
        Arc<MockLlmClient>,
        Arc<MockEmbeddings>,
    ) {
        (
            Arc::new(MockMemoryStore::new()),
            Arc::new(MockLlmClient::new()),
            Arc::new(MockEmbeddings::new(384)),
        )
    }

    fn make_fact(bank_id: BankId, content: &str) -> Fact {
        let now = Utc::now();
        Fact {
            id: FactId::new(),
            bank_id,
            content: content.into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(vec![0.1; 384]),
            confidence: None,
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now,
            updated_at: now,
            consolidated_at: None,
        }
    }

    fn make_consolidator(
        store: &Arc<MockMemoryStore>,
        llm: &Arc<MockLlmClient>,
        embeddings: &Arc<MockEmbeddings>,
    ) -> DefaultConsolidator {
        DefaultConsolidator::new(
            store.clone() as Arc<dyn MemoryStore>,
            llm.clone() as Arc<dyn LlmClient>,
            embeddings.clone() as Arc<dyn EmbeddingClient>,
            Arc::new(MockRecallPipeline {
                store: store.clone(),
            }) as Arc<dyn RecallPipeline>,
            ConsolidationConfig::default(),
        )
    }

    #[test]
    fn timing_totals_preserve_submillisecond_work_before_rounding() {
        let mut timings = ConsolidationTimingTotals::default();
        let mut breakdown = ConsolidationBreakdown::default();

        timings.prompt_build += Duration::from_micros(600);
        timings.prompt_build += Duration::from_micros(700);
        timings.begin_txn += Duration::from_micros(400);
        timings.begin_txn += Duration::from_micros(300);

        timings.apply_to(&mut breakdown);

        assert_eq!(breakdown.prompt_build_ms, 1);
        assert_eq!(breakdown.begin_txn_ms, 0);
    }

    #[tokio::test]
    async fn new_facts_create_observations() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        let f1 = make_fact(bank_id, "Caroline works at Google");
        let f2 = make_fact(bank_id, "Caroline has a son named James");
        store.insert_facts(&[f1, f2]).await.unwrap();

        llm.push_response(
            r#"{"actions": [
            {"action": "create", "content": "Caroline works at Google.", "fact_indices": [0]},
            {"action": "create", "content": "Caroline has a son named James.", "fact_indices": [1]}
        ]}"#,
        );

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let report = consolidator.consolidate(bank_id).await.unwrap();

        assert_eq!(report.observations_created, 2);
        assert_eq!(report.observations_updated, 0);

        let observations = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Observation]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(observations.len(), 2);
    }

    #[tokio::test]
    async fn incremental_update() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // First batch: create an observation
        let f1 = make_fact(bank_id, "Caroline works at Google");
        store.insert_facts(&[f1]).await.unwrap();

        llm.push_response(
            r#"{"actions": [
            {"action": "create", "content": "Caroline works at Google.", "fact_indices": [0]}
        ]}"#,
        );

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let report = consolidator.consolidate(bank_id).await.unwrap();
        assert_eq!(report.observations_created, 1);

        // Verify the created observation exists before the update pass.
        let observations = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Observation]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(observations.len(), 1);

        // Second batch: update the existing observation
        let f2 = make_fact(bank_id, "Caroline got promoted to senior engineer");
        store.insert_facts(&[f2]).await.unwrap();

        llm.push_response(
            r#"{"actions": [
            {"action": "update", "observation_index": 0, "content": "Caroline works as a senior engineer at Google, having recently been promoted.", "fact_indices": [0]}
        ]}"#,
        );

        let report = consolidator.consolidate(bank_id).await.unwrap();
        assert_eq!(report.observations_updated, 1);

        let observations = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Observation]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(observations.len(), 1);
        assert!(observations[0].content.contains("senior engineer"));
    }

    #[tokio::test]
    async fn idempotent_no_unconsolidated() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // Insert a fact and mark it consolidated
        let f1 = make_fact(bank_id, "Some fact");
        let f1_id = f1.id;
        store.insert_facts(&[f1]).await.unwrap();
        store.mark_consolidated(&[f1_id], Utc::now()).await.unwrap();

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let report = consolidator.consolidate(bank_id).await.unwrap();

        assert_eq!(report.observations_created, 0);
        assert_eq!(report.observations_updated, 0);
        assert_eq!(llm.remaining(), 0);
    }

    #[tokio::test]
    async fn batch_size_respected() {
        // With batch_size=8 (default), 12 facts should produce 2 LLM calls
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        for i in 0..12 {
            let f = make_fact(bank_id, &format!("Fact number {i}"));
            store.insert_facts(&[f]).await.unwrap();
        }

        // First batch of 8
        llm.push_response(r#"{"actions": [
            {"action": "create", "content": "Batch 1 observation.", "fact_indices": [0,1,2,3,4,5,6,7]}
        ]}"#);
        // Second batch of 4
        llm.push_response(
            r#"{"actions": [
            {"action": "create", "content": "Batch 2 observation.", "fact_indices": [0,1,2,3]}
        ]}"#,
        );

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let report = consolidator.consolidate(bank_id).await.unwrap();

        assert_eq!(report.observations_created, 2);
        assert_eq!(llm.remaining(), 0);
    }

    #[tokio::test]
    async fn retries_malformed_structured_output_then_succeeds() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        let fact = make_fact(bank_id, "Caroline works at Google");
        store.insert_facts(&[fact]).await.unwrap();

        llm.push_response(
            r#"{"actions": [{"content": "Missing action field.", "fact_indices": [0]}]}"#,
        );
        llm.push_response(
            r#"{"actions": [
            {"action": "create", "content": "Caroline works at Google.", "fact_indices": [0]}
        ]}"#,
        );

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let report = consolidator.consolidate(bank_id).await.unwrap();

        assert_eq!(report.observations_created, 1);
        assert_eq!(llm.remaining(), 0);
    }

    #[tokio::test]
    async fn bracketed_legacy_observation_id_still_updates() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        let f1 = make_fact(bank_id, "Caroline works at Google");
        store.insert_facts(&[f1]).await.unwrap();

        llm.push_response(
            r#"{"actions": [
            {"action": "create", "content": "Caroline works at Google.", "fact_indices": [0]}
        ]}"#,
        );

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        consolidator.consolidate(bank_id).await.unwrap();

        let observations = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Observation]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let obs_id = observations[0].id.to_string();

        let f2 = make_fact(bank_id, "Caroline got promoted to senior engineer");
        store.insert_facts(&[f2]).await.unwrap();

        llm.push_response(format!(
            r#"{{"actions": [
            {{"action": "update", "observation_id": "[{obs_id}]", "content": "Caroline works as a senior engineer at Google, having recently been promoted.", "fact_indices": [0]}}
        ]}}"#
        ));

        let report = consolidator.consolidate(bank_id).await.unwrap();
        assert_eq!(report.observations_updated, 1);

        let observations = store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Observation]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(observations.len(), 1);
        assert!(observations[0].content.contains("senior engineer"));
    }

    #[tokio::test]
    async fn malformed_structured_output_exhausts_retries() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        let fact = make_fact(bank_id, "Caroline works at Google");
        store.insert_facts(&[fact]).await.unwrap();

        for _ in 0..ConsolidationConfig::default().structured_output_max_attempts {
            llm.push_response(
                r#"{"actions": [{"content": "Missing action field.", "fact_indices": [0]}]}"#,
            );
        }

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let error = consolidator.consolidate(bank_id).await.unwrap_err();

        assert!(error.to_string().contains("missing field `action`"));
        assert_eq!(llm.remaining(), 0);
    }

    #[test]
    fn temporal_suffix_formatting() {
        use chrono::TimeZone;

        assert_eq!(format_temporal_suffix(None), "");

        // Both start and end (different dates)
        let tr = TemporalRange {
            start: Some(Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap()),
            end: Some(Utc.with_ymd_and_hms(2022, 12, 31, 0, 0, 0).unwrap()),
        };
        assert_eq!(
            format_temporal_suffix(Some(&tr)),
            " | occurred: 2022-01-01 to 2022-12-31"
        );

        // Same start and end → no range
        let tr = TemporalRange {
            start: Some(Utc.with_ymd_and_hms(2022, 6, 15, 0, 0, 0).unwrap()),
            end: Some(Utc.with_ymd_and_hms(2022, 6, 15, 0, 0, 0).unwrap()),
        };
        assert_eq!(format_temporal_suffix(Some(&tr)), " | occurred: 2022-06-15");

        // Start only
        let tr = TemporalRange {
            start: Some(Utc.with_ymd_and_hms(2024, 3, 1, 0, 0, 0).unwrap()),
            end: None,
        };
        assert_eq!(format_temporal_suffix(Some(&tr)), " | occurred: 2024-03-01");

        // Both None inside TemporalRange
        let tr = TemporalRange {
            start: None,
            end: None,
        };
        assert_eq!(format_temporal_suffix(Some(&tr)), "");
    }

    #[tokio::test]
    async fn only_world_and_experience() {
        let (store, llm, embeddings) = setup();
        let bank_id = BankId::new();

        // Insert an Opinion fact — should NOT be consolidated
        let now = Utc::now();
        let opinion = Fact {
            id: FactId::new(),
            bank_id,
            content: "Rust is the best language.".into(),
            fact_type: FactType::Experience,
            network: NetworkType::Opinion,
            entity_ids: vec![],
            temporal_range: None,
            embedding: Some(vec![0.1; 384]),
            confidence: Some(0.8),
            evidence_ids: vec![],
            source_turn_id: None,
            created_at: now,
            updated_at: now,
            consolidated_at: None,
        };
        store.insert_facts(&[opinion]).await.unwrap();

        let consolidator = make_consolidator(&store, &llm, &embeddings);
        let report = consolidator.consolidate(bank_id).await.unwrap();

        assert_eq!(report.observations_created, 0);
        assert_eq!(llm.remaining(), 0);
    }
}
