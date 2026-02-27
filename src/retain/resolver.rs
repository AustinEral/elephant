//! Entity resolution and deduplication (Phase 2C).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::util::cosine_similarity;
use crate::llm::LlmClient;
use crate::storage::MemoryStore;
use crate::types::{
    BankId, CompletionRequest, Entity, EntityId, EntityType, Message, ResolvedEntity,
};

/// Trait for resolving raw entity mention strings to canonical entities.
#[async_trait]
pub trait EntityResolver: Send + Sync {
    /// Resolve a batch of entity mentions within a bank.
    ///
    /// Deduplicates within the batch: if the same entity appears multiple times,
    /// all mentions resolve to the same entity ID.
    async fn resolve(
        &self,
        mentions: &[String],
        bank_id: BankId,
    ) -> Result<Vec<ResolvedEntity>>;
}

/// Entity resolver with layered resolution strategy.
///
/// 1. Exact match against existing entity names/aliases
/// 2. Embedding similarity against existing entities
/// 3. LLM disambiguation for borderline cases
/// 4. Create new entity if no match
pub struct LayeredEntityResolver {
    store: Box<dyn MemoryStore>,
    embeddings: Box<dyn EmbeddingClient>,
    llm: Arc<dyn LlmClient>,
    similarity_threshold: f32,
}

impl LayeredEntityResolver {
    /// Create a new resolver.
    pub fn new(
        store: Box<dyn MemoryStore>,
        embeddings: Box<dyn EmbeddingClient>,
        llm: Arc<dyn LlmClient>,
    ) -> Self {
        Self {
            store,
            embeddings,
            llm,
            similarity_threshold: 0.75,
        }
    }

    /// Try exact match against existing entities.
    async fn try_exact_match(
        &self,
        mention: &str,
        bank_id: BankId,
    ) -> Result<Option<ResolvedEntity>> {
        if let Some(entity) = self.store.find_entity(bank_id, mention).await? {
            return Ok(Some(ResolvedEntity {
                mention: mention.to_string(),
                entity_id: entity.id,
                canonical_name: entity.canonical_name,
                entity_type: entity.entity_type,
                is_new: false,
                confidence: 1.0,
            }));
        }
        Ok(None)
    }

    /// Try embedding-based fuzzy match against existing entities.
    async fn try_embedding_match(
        &self,
        mention: &str,
        bank_id: BankId,
    ) -> Result<Option<(Entity, f32)>> {
        let entities = self.store.list_entities(bank_id).await?;
        if entities.is_empty() {
            return Ok(None);
        }

        // Embed the mention
        let mention_embedding = self.embeddings.embed(&[mention]).await?;
        let mention_vec = &mention_embedding[0];

        // Embed all entity canonical names
        let entity_names: Vec<&str> = entities.iter().map(|e| e.canonical_name.as_str()).collect();
        let entity_embeddings = self.embeddings.embed(&entity_names).await?;

        // Find best match
        let mut best_score = 0.0f32;
        let mut best_entity = None;

        for (i, entity_vec) in entity_embeddings.iter().enumerate() {
            let sim = cosine_similarity(mention_vec, entity_vec);
            if sim > best_score {
                best_score = sim;
                best_entity = Some(&entities[i]);
            }
        }

        if best_score >= self.similarity_threshold
            && let Some(entity) = best_entity
        {
            return Ok(Some((entity.clone(), best_score)));
        }

        Ok(None)
    }

    /// Use LLM to confirm whether a mention matches a candidate entity.
    async fn llm_confirm(
        &self,
        mention: &str,
        candidate: &Entity,
    ) -> Result<bool> {
        let prompt = format!(
            "Is the mention \"{}\" referring to the same entity as \"{}\" (aliases: {:?})?\n\
             Answer with just \"yes\" or \"no\".",
            mention, candidate.canonical_name, candidate.aliases
        );

        let request = CompletionRequest {
            model: String::new(),
            system: Some("You are an entity resolution assistant. Answer only 'yes' or 'no'.".into()),
            messages: vec![Message {
                role: "user".into(),
                content: prompt,
            }],
            temperature: Some(0.0),
            max_tokens: Some(10),
        };

        let response = self.llm.complete(request).await?;
        Ok(response.content.trim().to_lowercase().starts_with("yes"))
    }

    /// Create a new entity from a mention.
    async fn create_entity(
        &self,
        mention: &str,
        bank_id: BankId,
    ) -> Result<ResolvedEntity> {
        let entity = Entity {
            id: EntityId::new(),
            canonical_name: mention.to_string(),
            aliases: vec![],
            entity_type: EntityType::Concept, // Default; could use LLM to classify
            bank_id,
        };
        self.store.upsert_entity(&entity).await?;

        Ok(ResolvedEntity {
            mention: mention.to_string(),
            entity_id: entity.id,
            canonical_name: entity.canonical_name,
            entity_type: entity.entity_type,
            is_new: true,
            confidence: 1.0,
        })
    }
}

#[async_trait]
impl EntityResolver for LayeredEntityResolver {
    async fn resolve(
        &self,
        mentions: &[String],
        bank_id: BankId,
    ) -> Result<Vec<ResolvedEntity>> {
        // Local cache for within-batch deduplication
        let mut cache: HashMap<String, ResolvedEntity> = HashMap::new();
        let mut results = Vec::with_capacity(mentions.len());

        for mention in mentions {
            let normalized = mention.trim().to_lowercase();

            // Check batch-local cache first
            if let Some(cached) = cache.get(&normalized) {
                results.push(ResolvedEntity {
                    mention: mention.clone(),
                    ..cached.clone()
                });
                continue;
            }

            // Layer 1: Exact match
            if let Some(resolved) = self.try_exact_match(mention, bank_id).await? {
                cache.insert(normalized, resolved.clone());
                results.push(resolved);
                continue;
            }

            // Layer 2: Embedding similarity
            if let Some((candidate, score)) = self.try_embedding_match(mention, bank_id).await? {
                // High similarity → accept directly
                if score >= 0.9 {
                    let resolved = ResolvedEntity {
                        mention: mention.clone(),
                        entity_id: candidate.id,
                        canonical_name: candidate.canonical_name.clone(),
                        entity_type: candidate.entity_type,
                        is_new: false,
                        confidence: score,
                    };
                    // Add as alias for future exact matches
                    let mut updated = candidate;
                    if !updated.aliases.contains(mention) {
                        updated.aliases.push(mention.clone());
                        self.store.upsert_entity(&updated).await?;
                    }
                    cache.insert(normalized, resolved.clone());
                    results.push(resolved);
                    continue;
                }

                // Borderline similarity → LLM confirmation
                if self.llm_confirm(mention, &candidate).await? {
                    let resolved = ResolvedEntity {
                        mention: mention.clone(),
                        entity_id: candidate.id,
                        canonical_name: candidate.canonical_name.clone(),
                        entity_type: candidate.entity_type,
                        is_new: false,
                        confidence: score,
                    };
                    let mut updated = candidate;
                    if !updated.aliases.contains(mention) {
                        updated.aliases.push(mention.clone());
                        self.store.upsert_entity(&updated).await?;
                    }
                    cache.insert(normalized, resolved.clone());
                    results.push(resolved);
                    continue;
                }
            }

            // Layer 4: Create new entity
            let resolved = self.create_entity(mention, bank_id).await?;
            cache.insert(normalized, resolved.clone());
            results.push(resolved);
        }

        Ok(results)
    }
}

