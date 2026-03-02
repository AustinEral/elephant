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

use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::llm::LlmClient;
use crate::storage::MemoryStore;
use crate::util::cosine_similarity;
use crate::types::{
    CompletionRequest, ExtractionInput, Fact, FactFilter, FactId, Message, NetworkType,
    RetainInput, RetainOutput,
};

use self::chunker::Chunker;
use self::extractor::FactExtractor;
use self::graph_builder::GraphBuilder;
use self::resolver::EntityResolver;

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
    async fn dedup_facts(&self, facts: Vec<Fact>, bank_id: crate::types::BankId, threshold: f32) -> Result<Vec<Fact>> {
        let mut kept = Vec::with_capacity(facts.len());
        for fact in facts {
            if let Some(ref emb) = fact.embedding {
                let results = self.store.vector_search(emb, bank_id, 1).await?;
                if let Some(top) = results.first() {
                    if top.score >= threshold {
                        continue;
                    }
                }
            }
            kept.push(fact);
        }
        Ok(kept)
    }

    /// Run opinion reinforcement: check new facts against existing opinions.
    async fn reinforce_opinions(
        &self,
        new_facts: &[Fact],
        bank_id: crate::types::BankId,
    ) -> Result<(usize, usize)> {
        let opinions = self
            .store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::Opinion]),
                    ..Default::default()
                },
            )
            .await?;

        if opinions.is_empty() || new_facts.is_empty() {
            return Ok((0, 0));
        }

        let mut reinforced = 0usize;
        let mut weakened = 0usize;

        // Embed new facts for comparison
        let new_texts: Vec<&str> = new_facts.iter().map(|f| f.content.as_str()).collect();
        let new_embeddings = self.embeddings.embed(&new_texts).await?;

        for mut opinion in opinions {
            let Some(ref opinion_emb) = opinion.embedding else {
                continue;
            };

            for (i, new_emb) in new_embeddings.iter().enumerate() {
                let sim = cosine_similarity(opinion_emb, new_emb);
                if sim < 0.7 {
                    continue;
                }

                // High similarity — ask LLM if this supports or contradicts
                let prompt = include_str!("../../prompts/reinforce_opinion.txt")
                    .replace("{opinion}", &opinion.content)
                    .replace("{new_fact}", &new_facts[i].content);

                let request = CompletionRequest {
                    model: String::new(),
                    system: Some("You are an opinion reinforcement engine. Answer with one word: 'supports', 'contradicts', or 'unrelated'.".into()),
                    messages: vec![Message {
                        role: "user".into(),
                        content: prompt,
                    }],
                    temperature: Some(0.0),
                    max_tokens: Some(10),
                };

                let response = self.llm.complete(request).await?;
                let answer = response.content.trim().to_lowercase();

                if answer.starts_with("support") {
                    // Reinforce: increase confidence
                    let current = opinion.confidence.unwrap_or(0.5);
                    opinion.confidence = Some((current + 0.1).min(1.0));
                    opinion.evidence_ids.push(new_facts[i].id);
                    opinion.updated_at = Utc::now();
                    self.store.update_fact(&opinion).await?;
                    reinforced += 1;
                    break; // One reinforcement per opinion per retain
                } else if answer.starts_with("contradict") {
                    // Weaken: decrease confidence
                    let current = opinion.confidence.unwrap_or(0.5);
                    opinion.confidence = Some((current - 0.1).max(0.0));
                    opinion.evidence_ids.push(new_facts[i].id);
                    opinion.updated_at = Utc::now();
                    self.store.update_fact(&opinion).await?;
                    weakened += 1;
                    break;
                }
            }
        }

        Ok((reinforced, weakened))
    }
}

#[async_trait]
impl RetainPipeline for DefaultRetainPipeline {
    async fn retain(&self, input: &RetainInput) -> Result<RetainOutput> {
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

        let mut all_fact_ids = Vec::new();
        let mut all_new_entities = Vec::new();
        let mut total_entities_resolved = 0usize;
        let mut all_stored_facts = Vec::new();
        let mut total_links = 0usize;

        // 2. Process each chunk
        for chunk in &chunks {
            // 2a. Extract facts
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

            if extracted.is_empty() {
                continue;
            }

            // 2b. Resolve entities for all extracted facts
            let all_mentions: Vec<String> = extracted
                .iter()
                .flat_map(|f| f.entity_mentions.clone())
                .collect();

            let resolved = if !all_mentions.is_empty() {
                self.resolver.resolve(&all_mentions, input.bank_id).await?
            } else {
                Vec::new()
            };

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
                    fact_type: ef.fact_type,
                    network: ef.network,
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

            // 2d. Dedup against existing facts
            let facts = if let Some(threshold) = self.dedup_threshold {
                let deduped = self.dedup_facts(facts, input.bank_id, threshold).await?;
                if deduped.is_empty() {
                    continue;
                }
                deduped
            } else {
                facts
            };

            // 2e. Store facts
            let ids = self.store.insert_facts(&facts).await?;
            all_fact_ids.extend_from_slice(&ids);

            // 2f. Build graph links
            let links = self
                .graph_builder
                .build_links(&facts, input.bank_id)
                .await?;
            total_links += links.len();

            all_stored_facts.extend(facts);
        }

        // 3. Opinion reinforcement
        let (opinions_reinforced, opinions_weakened) = self
            .reinforce_opinions(&all_stored_facts, input.bank_id)
            .await?;

        Ok(RetainOutput {
            fact_ids: all_fact_ids.clone(),
            facts_stored: all_fact_ids.len(),
            new_entities: all_new_entities,
            entities_resolved: total_entities_resolved,
            links_created: total_links,
            opinions_reinforced,
            opinions_weakened,
        })
    }
}
