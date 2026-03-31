//! Entity resolution and deduplication (Phase 2C).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::embedding::EmbeddingClient;
use crate::error::Result;
use crate::llm::{CompletionRequest, LlmClient, Message, ReasoningEffort};
use crate::storage::MemoryStore;
use crate::types::{BankId, Entity, EntityId, EntityType, ResolvedEntity};
use crate::util::cosine_similarity;

/// Trait for resolving raw entity mention strings to canonical entities.
#[async_trait]
pub trait EntityResolver: Send + Sync {
    /// Resolve a batch of entity mentions within a bank.
    ///
    /// Deduplicates within the batch: if the same entity appears multiple times,
    /// all mentions resolve to the same entity ID.
    ///
    /// The `store` parameter controls which store is used for reads/writes,
    /// allowing callers to pass a transaction handle for atomic operations.
    async fn resolve(
        &self,
        mentions: &[String],
        bank_id: BankId,
        store: &dyn MemoryStore,
    ) -> Result<Vec<ResolvedEntity>>;
}

/// Entity resolver with layered resolution strategy.
///
/// 1. Exact match against existing entity names/aliases
/// 2. Embedding similarity against existing entities
/// 3. LLM disambiguation for borderline cases
/// 4. Create new entity if no match
pub struct LayeredEntityResolver {
    embeddings: Box<dyn EmbeddingClient>,
    llm: Arc<dyn LlmClient>,
    similarity_threshold: f32,
    temperature: f32,
    reasoning_effort: Option<ReasoningEffort>,
}

/// Entity resolver system instruction.
pub const ENTITY_RESOLUTION_SYSTEM_PROMPT: &str =
    "You are an entity resolution assistant. Answer only 'yes' or 'no'.";
/// Entity resolver user prompt template.
pub const ENTITY_RESOLUTION_USER_PROMPT_TEMPLATE: &str = "Is the mention \"{mention}\" referring to the same entity as \"{canonical_name}\" (aliases: {aliases})?\nAnswer with just \"yes\" or \"no\".";
/// Entity resolver temperature.
pub const ENTITY_RESOLUTION_TEMPERATURE: f32 = 0.0;
/// Entity resolver output cap.
pub const ENTITY_RESOLUTION_MAX_TOKENS: usize = 32;

impl LayeredEntityResolver {
    /// Create a new resolver.
    pub fn new(embeddings: Box<dyn EmbeddingClient>, llm: Arc<dyn LlmClient>) -> Self {
        Self::new_with_options(embeddings, llm, ENTITY_RESOLUTION_TEMPERATURE, None)
    }

    /// Create a new resolver with an explicit temperature override.
    pub fn new_with_temperature(
        embeddings: Box<dyn EmbeddingClient>,
        llm: Arc<dyn LlmClient>,
        temperature: f32,
    ) -> Self {
        Self::new_with_options(embeddings, llm, temperature, None)
    }

    /// Create a new resolver with explicit runtime options.
    pub fn new_with_options(
        embeddings: Box<dyn EmbeddingClient>,
        llm: Arc<dyn LlmClient>,
        temperature: f32,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Self {
        Self {
            embeddings,
            llm,
            similarity_threshold: 0.75,
            temperature,
            reasoning_effort,
        }
    }

    /// Try exact match against existing entities.
    async fn try_exact_match(
        &self,
        mention: &str,
        bank_id: BankId,
        store: &dyn MemoryStore,
    ) -> Result<Option<ResolvedEntity>> {
        if let Some(entity) = store.find_entity(bank_id, mention).await? {
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
        store: &dyn MemoryStore,
    ) -> Result<Option<(Entity, f32)>> {
        let entities = store.list_entities(bank_id).await?;
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
    async fn llm_confirm(&self, mention: &str, candidate: &Entity) -> Result<bool> {
        let prompt = ENTITY_RESOLUTION_USER_PROMPT_TEMPLATE
            .replace("{mention}", mention)
            .replace("{canonical_name}", &candidate.canonical_name)
            .replace("{aliases}", &format!("{:?}", candidate.aliases));

        let request = CompletionRequest::builder()
            .system(ENTITY_RESOLUTION_SYSTEM_PROMPT)
            .message(Message::user(prompt))
            .temperature(self.temperature)
            .reasoning_effort_opt(self.reasoning_effort)
            .max_tokens(ENTITY_RESOLUTION_MAX_TOKENS)
            .build();

        let response = self.llm.complete(request).await?;
        Ok(response.content.trim().to_lowercase().starts_with("yes"))
    }

    /// Create a new entity from a mention.
    async fn create_entity(
        &self,
        mention: &str,
        bank_id: BankId,
        store: &dyn MemoryStore,
    ) -> Result<ResolvedEntity> {
        let entity = Entity {
            id: EntityId::new(),
            canonical_name: mention.to_string(),
            aliases: vec![],
            entity_type: EntityType::Concept, // Default; could use LLM to classify
            bank_id,
        };
        store.upsert_entity(&entity).await?;

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
        store: &dyn MemoryStore,
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
            if let Some(resolved) = self.try_exact_match(mention, bank_id, store).await? {
                cache.insert(normalized, resolved.clone());
                results.push(resolved);
                continue;
            }

            // Layer 2: Embedding similarity
            if let Some((candidate, score)) =
                self.try_embedding_match(mention, bank_id, store).await?
            {
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
                        store.upsert_entity(&updated).await?;
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
                        store.upsert_entity(&updated).await?;
                    }
                    cache.insert(normalized, resolved.clone());
                    results.push(resolved);
                    continue;
                }
            }

            // Layer 4: Create new entity
            let resolved = self.create_entity(mention, bank_id, store).await?;
            cache.insert(normalized, resolved.clone());
            results.push(resolved);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::mock::MockEmbeddings;
    use crate::llm::mock::MockLlmClient;
    use crate::storage::mock::MockMemoryStore;
    use crate::types::{BankId, Disposition, MemoryBank};

    use crate::storage::MemoryStore;

    async fn setup() -> (MockMemoryStore, MockEmbeddings, Arc<MockLlmClient>, BankId) {
        let store = MockMemoryStore::new();
        let embeddings = MockEmbeddings::new(8);
        let llm = Arc::new(MockLlmClient::new());
        let bank_id = BankId::new();

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

        (store, embeddings, llm, bank_id)
    }

    fn make_resolver(
        embeddings: &MockEmbeddings,
        llm: Arc<MockLlmClient>,
    ) -> LayeredEntityResolver {
        LayeredEntityResolver::new(Box::new(embeddings.clone()), llm)
    }

    #[tokio::test]
    async fn exact_match_resolves_existing_entity() {
        let (store, embeddings, llm, bank_id) = setup().await;

        // Pre-insert an entity
        let entity = Entity {
            id: EntityId::new(),
            canonical_name: "Rust".into(),
            aliases: vec!["rust-lang".into()],
            entity_type: EntityType::Concept,
            bank_id,
        };
        store.upsert_entity(&entity).await.unwrap();

        let resolver = make_resolver(&embeddings, llm);
        let results = resolver
            .resolve(&["Rust".into()], bank_id, &store)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_id, entity.id);
        assert!(!results[0].is_new);
        assert_eq!(results[0].confidence, 1.0);
    }

    #[tokio::test]
    async fn exact_match_by_alias() {
        let (store, embeddings, llm, bank_id) = setup().await;

        let entity = Entity {
            id: EntityId::new(),
            canonical_name: "Rust".into(),
            aliases: vec!["rust-lang".into()],
            entity_type: EntityType::Concept,
            bank_id,
        };
        store.upsert_entity(&entity).await.unwrap();

        let resolver = make_resolver(&embeddings, llm);
        let results = resolver
            .resolve(&["rust-lang".into()], bank_id, &store)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_id, entity.id);
        assert!(!results[0].is_new);
    }

    #[tokio::test]
    async fn no_match_creates_new_entity() {
        let (store, embeddings, llm, bank_id) = setup().await;

        let resolver = make_resolver(&embeddings, llm);
        let results = resolver
            .resolve(&["Rust".into()], bank_id, &store)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].is_new);
        assert_eq!(results[0].canonical_name, "Rust");

        // Entity should now exist in the store
        let found = store.find_entity(bank_id, "Rust").await.unwrap();
        assert!(found.is_some());
    }

    #[tokio::test]
    async fn batch_dedup_same_mention_twice() {
        let (store, embeddings, llm, bank_id) = setup().await;

        let resolver = make_resolver(&embeddings, llm);
        let results = resolver
            .resolve(
                &["Rust".into(), "rust".into(), "RUST".into()],
                bank_id,
                &store,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        // All should resolve to the same entity (batch-local cache normalizes to lowercase)
        assert_eq!(results[0].entity_id, results[1].entity_id);
        assert_eq!(results[1].entity_id, results[2].entity_id);
        // Only the first should be "new", rest come from cache
        assert!(results[0].is_new);
    }

    #[tokio::test]
    async fn llm_confirm_borderline_match() {
        let (_store, embeddings, llm, bank_id) = setup().await;

        // We can't easily control mock embedding similarity, but we can
        // test the LLM confirmation path directly
        let entity = Entity {
            id: EntityId::new(),
            canonical_name: "PostgreSQL".into(),
            aliases: vec!["Postgres".into()],
            entity_type: EntityType::Concept,
            bank_id,
        };

        let resolver = make_resolver(&embeddings, llm.clone());

        // Test llm_confirm directly
        llm.push_response("yes");
        let confirmed = resolver.llm_confirm("pg", &entity).await.unwrap();
        assert!(confirmed);

        llm.push_response("no");
        let rejected = resolver.llm_confirm("MySQL", &entity).await.unwrap();
        assert!(!rejected);
    }

    #[tokio::test]
    async fn multiple_distinct_entities() {
        let (store, embeddings, llm, bank_id) = setup().await;

        let resolver = make_resolver(&embeddings, llm);
        let results = resolver
            .resolve(
                &["Rust".into(), "Python".into(), "Go".into()],
                bank_id,
                &store,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        // All different entities
        assert_ne!(results[0].entity_id, results[1].entity_id);
        assert_ne!(results[1].entity_id, results[2].entity_id);
        assert!(results.iter().all(|r| r.is_new));
    }
}
