//! Retain pipeline: raw text → structured memory.
//!
//! Submodules:
//! - [`chunker`] — Splits input into extractable chunks (2A)
//! - [`extractor`] — LLM-based fact extraction (2B)
//! - [`resolver`] — Entity resolution and deduplication (2C)
//! - [`graph_builder`] — Graph link construction (2D)

pub mod chunker;
pub mod extractor;
pub mod graph_builder;
pub mod resolver;

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tracing::{Instrument, debug, info, info_span};

use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::llm::LlmClient;
use crate::storage::MemoryStore;
use crate::types::{ExtractionInput, Fact, FactId, RetainInput, RetainOutput, Source, SourceId};

use self::chunker::Chunker;
use self::extractor::FactExtractor;
use self::graph_builder::GraphBuilder;
use self::resolver::EntityResolver;

/// Opinion reinforcement prompt template.
pub const OPINION_REINFORCEMENT_PROMPT_TEMPLATE: &str =
    include_str!("../../prompts/reinforce_opinion.txt");
/// Opinion reinforcement system instruction.
pub const OPINION_REINFORCEMENT_SYSTEM_PROMPT: &str = "You are an opinion reinforcement engine. Answer with one word: 'supports', 'contradicts', or 'unrelated'.";
/// Opinion reinforcement temperature.
pub const OPINION_REINFORCEMENT_TEMPERATURE: f32 = 0.0;
/// Opinion reinforcement output cap.
pub const OPINION_REINFORCEMENT_MAX_TOKENS: usize = 10;

/// The top-level retain pipeline trait (2E).
#[async_trait]
pub trait RetainPipeline: Send + Sync {
    /// Process raw input and store structured memory.
    async fn retain(&self, input: &RetainInput) -> Result<RetainOutput>;
}

/// Default retain orchestrator wiring chunker → extractor → resolver → graph builder + opinion reinforcement.
pub struct DefaultRetainPipeline {
    chunker: Box<dyn Chunker>,
    extractor: Box<dyn FactExtractor>,
    resolver: Box<dyn EntityResolver>,
    graph_builder: Box<dyn GraphBuilder>,
    store: Box<dyn MemoryStore>,
    embeddings: Box<dyn EmbeddingClient>,
    llm: Arc<dyn LlmClient>,
    chunk_config: crate::types::ChunkConfig,
    dedup_threshold: Option<f32>,
}

impl DefaultRetainPipeline {
    /// Create a new retain pipeline with all components.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chunker: Box<dyn Chunker>,
        extractor: Box<dyn FactExtractor>,
        resolver: Box<dyn EntityResolver>,
        graph_builder: Box<dyn GraphBuilder>,
        store: Box<dyn MemoryStore>,
        embeddings: Box<dyn EmbeddingClient>,
        llm: Arc<dyn LlmClient>,
        chunk_config: crate::types::ChunkConfig,
        dedup_threshold: Option<f32>,
    ) -> Self {
        Self {
            chunker,
            extractor,
            resolver,
            graph_builder,
            store,
            embeddings,
            llm,
            chunk_config,
            dedup_threshold,
        }
    }

    /// Remove facts that are near-duplicates of existing facts in the bank.
    async fn dedup_facts(
        &self,
        facts: Vec<Fact>,
        bank_id: crate::types::BankId,
        threshold: f32,
        store: &dyn MemoryStore,
    ) -> Result<Vec<Fact>> {
        let mut kept = Vec::with_capacity(facts.len());
        for fact in facts {
            if let Some(ref emb) = fact.embedding {
                let results = store.vector_search(emb, bank_id, 1, None).await?;
                if let Some(top) = results.first()
                    && top.score >= threshold
                {
                    continue;
                }
            }
            kept.push(fact);
        }
        Ok(kept)
    }

    /// Opinion reinforcement — disabled pending redesign (see #12).
    ///
    /// The previous implementation checked every new fact against every existing
    /// opinion via cosine similarity + LLM classification, consuming ~40% of
    /// ingest runtime while producing no measurable accuracy benefit (all
    /// confidences saturated to 1.0). Ablation on conv-26 and conv-30 confirmed
    /// no accuracy loss from disabling it.
    async fn reinforce_opinions(
        &self,
        _new_facts: &[Fact],
        _bank_id: crate::types::BankId,
        _store: &dyn MemoryStore,
    ) -> Result<(usize, usize)> {
        Ok((0, 0))
    }

    async fn retain_inner(&self, input: &RetainInput) -> Result<RetainOutput> {
        // 0. Validate embedding dimensions match the bank's config
        let bank = self.store.get_bank(input.bank_id).await?;
        if bank.embedding_dimensions > 0 {
            let client_dims = self.embeddings.dimensions() as u16;
            if client_dims != bank.embedding_dimensions {
                return Err(crate::error::Error::EmbeddingDimensionMismatch {
                    model: bank.embedding_model.clone(),
                    expected: bank.embedding_dimensions,
                    actual: client_dims,
                });
            }
        }

        // 1. Chunk the input
        let chunks = self.chunker.chunk(&input.content, &self.chunk_config);
        debug!(chunks = chunks.len(), "chunked");

        let mut all_fact_ids = Vec::new();
        let mut all_new_entities = Vec::new();
        let mut total_entities_resolved = 0usize;
        let mut all_stored_facts = Vec::new();
        let mut total_links = 0usize;

        let mut total_reinforced = 0usize;
        let mut total_weakened = 0usize;

        // Single transaction for the entire retain call — either all chunks
        // commit or nothing is written.
        let txn = self.store.begin().await?;

        // 2. Process each chunk
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            // 2a. Extract facts (LLM call)
            let extraction_input = ExtractionInput {
                content: chunk.content.clone(),
                bank_id: input.bank_id,
                context: chunk.context.clone().or_else(|| input.context.clone()),
                timestamp: input.timestamp,
                turn_id: input.turn_id,
                custom_instructions: input.custom_instructions.clone(),
                speaker: input.speaker.clone(),
            };
            let extracted = self.extractor.extract(&extraction_input).await?;
            debug!(chunk = chunk_idx, extracted = extracted.len(), "extracted");

            if extracted.is_empty() {
                continue;
            }

            // 2b. Resolve entities for all extracted facts
            let all_mentions: Vec<String> = extracted
                .iter()
                .flat_map(|f| f.entity_mentions.clone())
                .collect();

            let resolved = if !all_mentions.is_empty() {
                debug!(
                    chunk = chunk_idx,
                    mentions = all_mentions.len(),
                    "resolving_entities"
                );
                self.resolver
                    .resolve(&all_mentions, input.bank_id, &*txn)
                    .await?
            } else {
                Vec::new()
            };
            debug!(
                chunk = chunk_idx,
                resolved = resolved.len(),
                "entities_resolved"
            );

            total_entities_resolved += resolved.len();
            for r in &resolved {
                if r.is_new && !all_new_entities.contains(&r.entity_id) {
                    all_new_entities.push(r.entity_id);
                }
            }

            // Build a mention → entity_id map for quick lookup
            let mention_map: std::collections::HashMap<String, crate::types::EntityId> = resolved
                .iter()
                .map(|r| (r.mention.clone(), r.entity_id))
                .collect();

            // 2c. Convert ExtractedFact → Fact with resolved entities and embeddings
            let fact_texts: Vec<&str> = extracted.iter().map(|f| f.content.as_str()).collect();
            debug!(
                chunk = chunk_idx,
                texts = fact_texts.len(),
                "embedding_facts"
            );
            let embeddings = self.embeddings.embed(&fact_texts).await?;

            let mut facts = Vec::with_capacity(extracted.len());
            for (i, ef) in extracted.iter().enumerate() {
                let entity_ids: Vec<crate::types::EntityId> = ef
                    .entity_mentions
                    .iter()
                    .filter_map(|m| mention_map.get(m))
                    .copied()
                    .collect();

                let fact = Fact {
                    id: FactId::new(),
                    bank_id: input.bank_id,
                    content: ef.content.clone(),
                    fact_type: ef.fact_type.normalize(),
                    network: ef.network.normalize(),
                    entity_ids,
                    temporal_range: ef.temporal_range.clone(),
                    embedding: Some(embeddings[i].clone()),
                    confidence: ef.confidence,
                    evidence_ids: vec![],
                    source_turn_id: input.turn_id,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    consolidated_at: None,
                };
                facts.push(fact);
            }

            // 2d. Dedup against existing facts (reads through txn to see prior chunks)
            let pre_dedup = facts.len();
            let facts = if let Some(threshold) = self.dedup_threshold {
                let deduped = self
                    .dedup_facts(facts, input.bank_id, threshold, &*txn)
                    .await?;
                if deduped.is_empty() {
                    debug!(
                        chunk = chunk_idx,
                        deduped_all = pre_dedup,
                        "all facts deduped, skipping chunk"
                    );
                    continue;
                }
                deduped
            } else {
                facts
            };
            if facts.len() < pre_dedup {
                debug!(
                    chunk = chunk_idx,
                    before = pre_dedup,
                    after = facts.len(),
                    "dedup"
                );
            }

            // 2e. Store facts
            debug!(chunk = chunk_idx, facts = facts.len(), "inserting_facts");
            let source_id = self.create_source_record(&extraction_input, &*txn).await?;
            let ids = txn.insert_facts(&facts).await?;
            txn.link_facts_to_source(&ids, source_id).await?;
            all_fact_ids.extend_from_slice(&ids);

            // 2f. Build graph links
            debug!(chunk = chunk_idx, facts = facts.len(), "building_links");
            let links = self
                .graph_builder
                .build_links(&facts, input.bank_id, &*txn)
                .await?;
            debug!(chunk = chunk_idx, links = links.len(), "links_built");
            total_links += links.len();

            // 2g. Opinion reinforcement
            let (reinforced, weakened) = self
                .reinforce_opinions(&facts, input.bank_id, &*txn)
                .await?;
            total_reinforced += reinforced;
            total_weakened += weakened;

            all_stored_facts.extend(facts);
        }

        // Commit — all writes for the entire retain call become visible
        txn.commit().await?;

        let facts_stored = all_fact_ids.len();
        info!(
            facts_stored,
            entities_resolved = total_entities_resolved,
            links_created = total_links,
            "retain_complete"
        );
        tracing::Span::current().record("facts_stored", facts_stored);
        tracing::Span::current().record("entities_resolved", total_entities_resolved);
        tracing::Span::current().record("links_created", total_links);

        Ok(RetainOutput {
            fact_ids: all_fact_ids.clone(),
            facts_stored,
            new_entities: all_new_entities,
            entities_resolved: total_entities_resolved,
            links_created: total_links,
            opinions_reinforced: total_reinforced,
            opinions_weakened: total_weakened,
        })
    }

    async fn create_source_record(
        &self,
        input: &ExtractionInput,
        store: &dyn MemoryStore,
    ) -> Result<SourceId> {
        let source = Source {
            id: SourceId::new(),
            bank_id: input.bank_id,
            content: input.content.clone(),
            context: input.context.clone(),
            speaker: input.speaker.clone(),
            rendered_input: Some(self.extractor.render_user_message(input)),
            timestamp: input.timestamp,
            created_at: Utc::now(),
        };
        store.insert_source(&source).await
    }
}

#[async_trait]
impl RetainPipeline for DefaultRetainPipeline {
    async fn retain(&self, input: &RetainInput) -> Result<RetainOutput> {
        let retain_span = info_span!("retain",
            bank_id = %input.bank_id,
            content_len = input.content.len(),
            facts_stored = tracing::field::Empty,
            entities_resolved = tracing::field::Empty,
            links_created = tracing::field::Empty,
        );
        self.retain_inner(input).instrument(retain_span).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::embedding::mock::MockEmbeddings;
    use crate::llm::mock::MockLlmClient;
    use crate::storage::mock::MockMemoryStore;
    use crate::types::{Chunk, Disposition, ExtractedFact, FactType, MemoryBank};

    struct StubChunker;

    impl Chunker for StubChunker {
        fn chunk(&self, input: &str, _config: &crate::types::ChunkConfig) -> Vec<Chunk> {
            vec![Chunk {
                content: input.to_string(),
                index: 0,
                context: None,
            }]
        }
    }

    struct StubExtractor;

    #[async_trait]
    impl FactExtractor for StubExtractor {
        async fn extract(&self, _input: &ExtractionInput) -> Result<Vec<ExtractedFact>> {
            Ok(vec![ExtractedFact {
                content: "Avery shared the roadmap".into(),
                fact_type: FactType::Experience,
                network: crate::types::ExtractedNetworkType::Experience,
                entity_mentions: vec![],
                temporal_range: None,
                confidence: None,
            }])
        }
    }

    struct StubResolver;

    #[async_trait]
    impl EntityResolver for StubResolver {
        async fn resolve(
            &self,
            _mentions: &[String],
            _bank_id: crate::types::BankId,
            _store: &dyn MemoryStore,
        ) -> Result<Vec<crate::types::ResolvedEntity>> {
            Ok(vec![])
        }
    }

    struct StubGraphBuilder;

    #[async_trait]
    impl GraphBuilder for StubGraphBuilder {
        async fn build_links(
            &self,
            _new_facts: &[Fact],
            _bank_id: crate::types::BankId,
            _store: &dyn MemoryStore,
        ) -> Result<Vec<crate::types::GraphLink>> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn retain_creates_and_links_a_source_record() {
        let store = MockMemoryStore::new();
        let bank_id = crate::types::BankId::new();
        store
            .create_bank(&MemoryBank {
                id: bank_id,
                name: "test".into(),
                mission: String::new(),
                directives: vec![],
                disposition: Disposition::default(),
                embedding_model: "mock".into(),
                embedding_dimensions: 8,
            })
            .await
            .unwrap();

        let pipeline = DefaultRetainPipeline::new(
            Box::new(StubChunker),
            Box::new(StubExtractor),
            Box::new(StubResolver),
            Box::new(StubGraphBuilder),
            Box::new(store.clone()),
            Box::new(MockEmbeddings::new(8)),
            Arc::new(MockLlmClient::new()),
            crate::types::ChunkConfig {
                max_tokens: 2000,
                overlap_tokens: 0,
                preserve_turns: false,
            },
            None,
        );

        let timestamp = chrono::Utc::now();
        let output = pipeline
            .retain(&RetainInput {
                bank_id,
                content: "Avery shared the roadmap during the weekly sync.".into(),
                timestamp,
                turn_id: None,
                context: None,
                custom_instructions: None,
                speaker: None,
            })
            .await
            .unwrap();

        assert_eq!(output.fact_ids.len(), 1);

        let lookups = store.lookup_sources(&output.fact_ids, 5).await.unwrap();
        assert_eq!(lookups.len(), 1);
        assert_eq!(lookups[0].fact_id, output.fact_ids[0]);
        assert_eq!(lookups[0].sources.len(), 1);
        assert_eq!(
            lookups[0].sources[0].content,
            "Avery shared the roadmap during the weekly sync."
        );
        assert_eq!(lookups[0].sources[0].context, None);
        assert_eq!(lookups[0].sources[0].speaker, None);
        assert_eq!(
            lookups[0].sources[0].rendered_input.as_deref(),
            Some(
                format!(
                    "## Content to Extract From\n\nAvery shared the roadmap during the weekly sync.\n\nTimestamp: {}",
                    timestamp.to_rfc3339()
                )
                .as_str()
            )
        );
        assert_eq!(lookups[0].sources[0].timestamp, timestamp);
    }
}
