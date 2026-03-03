use chrono::Utc;
use sqlx::PgPool;
use testcontainers::core::ContainerPort;
use testcontainers::runners::AsyncRunner;
use testcontainers::GenericImage;
use testcontainers_modules::testcontainers::ImageExt;

use elephant::storage::pg::PgMemoryStore;
use elephant::storage::MemoryStore;
use elephant::types::*;

async fn setup_store(
) -> (PgMemoryStore, testcontainers::ContainerAsync<GenericImage>) {
    let container = GenericImage::new("pgvector/pgvector", "pg16")
        .with_exposed_port(ContainerPort::Tcp(5432))
        .with_wait_for(testcontainers::core::WaitFor::message_on_stderr(
            "database system is ready to accept connections",
        ))
        .with_env_var("POSTGRES_DB", "test")
        .with_env_var("POSTGRES_USER", "test")
        .with_env_var("POSTGRES_PASSWORD", "test")
        .start()
        .await
        .expect("failed to start postgres");

    let port = container
        .get_host_port_ipv4(5432)
        .await
        .expect("failed to get port");
    let url = format!("postgres://test:test@127.0.0.1:{port}/test");

    // Retry connection briefly — container may need a moment
    let pool = loop {
        match PgPool::connect(&url).await {
            Ok(p) => break p,
            Err(_) => tokio::time::sleep(std::time::Duration::from_millis(200)).await,
        }
    };

    let store = PgMemoryStore::new(pool);
    store.migrate().await.expect("migration failed");
    (store, container)
}

fn make_bank() -> MemoryBank {
    MemoryBank {
        id: BankId::new(),
        name: "test bank".into(),
        mission: "testing".into(),
        directives: vec!["be accurate".into()],
        disposition: Disposition::new(2, 4, 3, 0.7).unwrap(),
        embedding_model: String::new(),
        embedding_dimensions: 0,
    }
}

fn make_fact(bank_id: BankId) -> Fact {
    Fact {
        id: FactId::new(),
        bank_id,
        content: "Rust is a systems programming language".into(),
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
    }
}

#[tokio::test]
async fn bank_crud() {
    let (store, _container) = setup_store().await;
    let bank = make_bank();

    let id = store.create_bank(&bank).await.unwrap();
    assert_eq!(id, bank.id);

    let fetched = store.get_bank(bank.id).await.unwrap();
    assert_eq!(fetched.name, bank.name);
    assert_eq!(fetched.mission, bank.mission);
    assert_eq!(fetched.directives, bank.directives);
    assert_eq!(fetched.disposition, bank.disposition);
}

#[tokio::test]
async fn fact_insert_get() {
    let (store, _container) = setup_store().await;
    let bank = make_bank();
    store.create_bank(&bank).await.unwrap();

    let mut fact = make_fact(bank.id);
    fact.confidence = Some(0.9);
    fact.source_turn_id = Some(TurnId::new());

    let ids = store.insert_facts(&[fact.clone()]).await.unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], fact.id);

    let fetched = store.get_facts(&ids).await.unwrap();
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].content, fact.content);
    assert_eq!(fetched[0].fact_type, fact.fact_type);
    assert_eq!(fetched[0].network, fact.network);
    assert_eq!(fetched[0].confidence, fact.confidence);
    assert!(fetched[0].source_turn_id.is_some());
}

#[tokio::test]
async fn vector_search_ranking() {
    let (store, _container) = setup_store().await;
    let bank = make_bank();
    store.create_bank(&bank).await.unwrap();

    // Create facts with known embeddings (384-dim, mostly zeros with distinguishing values)
    let mut close_fact = make_fact(bank.id);
    close_fact.content = "close to query".into();
    let mut close_emb = vec![0.0f32; 384];
    close_emb[0] = 1.0;
    close_emb[1] = 0.9;
    close_fact.embedding = Some(close_emb);

    let mut far_fact = make_fact(bank.id);
    far_fact.id = FactId::new();
    far_fact.content = "far from query".into();
    let mut far_emb = vec![0.0f32; 384];
    far_emb[0] = -1.0;
    far_emb[1] = -0.9;
    far_fact.embedding = Some(far_emb);

    store
        .insert_facts(&[close_fact.clone(), far_fact.clone()])
        .await
        .unwrap();

    // Query embedding similar to close_fact
    let mut query_emb = vec![0.0f32; 384];
    query_emb[0] = 1.0;
    query_emb[1] = 1.0;

    let results = store
        .vector_search(&query_emb, bank.id, 10, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].fact.id, close_fact.id);
    assert!(results[0].score > results[1].score);
}

#[tokio::test]
async fn temporal_filtering() {
    let (store, _container) = setup_store().await;
    let bank = make_bank();
    store.create_bank(&bank).await.unwrap();

    let now = Utc::now();
    let yesterday = now - chrono::Duration::days(1);
    let last_week = now - chrono::Duration::days(7);
    let two_weeks_ago = now - chrono::Duration::days(14);

    let mut recent_fact = make_fact(bank.id);
    recent_fact.content = "recent fact".into();
    recent_fact.temporal_range = Some(TemporalRange {
        start: Some(yesterday),
        end: Some(now),
    });

    let mut old_fact = make_fact(bank.id);
    old_fact.id = FactId::new();
    old_fact.content = "old fact".into();
    old_fact.temporal_range = Some(TemporalRange {
        start: Some(two_weeks_ago),
        end: Some(last_week),
    });

    store
        .insert_facts(&[recent_fact.clone(), old_fact.clone()])
        .await
        .unwrap();

    // Filter for last 3 days — should only get recent_fact
    let filter = FactFilter {
        temporal_range: Some(TemporalRange {
            start: Some(now - chrono::Duration::days(3)),
            end: Some(now),
        }),
        ..Default::default()
    };
    let results = store.get_facts_by_bank(bank.id, filter).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, recent_fact.id);
}

#[tokio::test]
async fn entity_alias_matching() {
    let (store, _container) = setup_store().await;
    let bank = make_bank();
    store.create_bank(&bank).await.unwrap();

    let entity = Entity {
        id: EntityId::new(),
        canonical_name: "Rust".into(),
        aliases: vec!["rust-lang".into(), "Rust language".into()],
        entity_type: EntityType::Concept,
        bank_id: bank.id,
    };
    store.upsert_entity(&entity).await.unwrap();

    // Find by canonical name
    let found = store.find_entity(bank.id, "Rust").await.unwrap();
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, entity.id);

    // Find by alias
    let found = store.find_entity(bank.id, "rust-lang").await.unwrap();
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, entity.id);

    // Not found
    let found = store.find_entity(bank.id, "Python").await.unwrap();
    assert!(found.is_none());
}

#[tokio::test]
async fn graph_neighbors() {
    let (store, _container) = setup_store().await;
    let bank = make_bank();
    store.create_bank(&bank).await.unwrap();

    let fact_a = make_fact(bank.id);
    let mut fact_b = make_fact(bank.id);
    fact_b.id = FactId::new();
    let mut fact_c = make_fact(bank.id);
    fact_c.id = FactId::new();

    store
        .insert_facts(&[fact_a.clone(), fact_b.clone(), fact_c.clone()])
        .await
        .unwrap();

    let links = vec![
        GraphLink {
            source_id: fact_a.id,
            target_id: fact_b.id,
            link_type: LinkType::Semantic,
            weight: 0.8,
        },
        GraphLink {
            source_id: fact_a.id,
            target_id: fact_c.id,
            link_type: LinkType::Causal,
            weight: 0.6,
        },
    ];
    store.insert_links(&links).await.unwrap();

    // All neighbors of A
    let neighbors = store.get_neighbors(fact_a.id, None).await.unwrap();
    assert_eq!(neighbors.len(), 2);

    // Only semantic neighbors of A
    let semantic = store
        .get_neighbors(fact_a.id, Some(LinkType::Semantic))
        .await
        .unwrap();
    assert_eq!(semantic.len(), 1);
    assert_eq!(semantic[0].0, fact_b.id);
    assert!((semantic[0].1 - 0.8).abs() < f32::EPSILON);
    assert_eq!(semantic[0].2, LinkType::Semantic);

    // B should see A as neighbor (bidirectional lookup)
    let b_neighbors = store.get_neighbors(fact_b.id, None).await.unwrap();
    assert_eq!(b_neighbors.len(), 1);
    assert_eq!(b_neighbors[0].0, fact_a.id);
    assert_eq!(b_neighbors[0].2, LinkType::Semantic);
}

#[tokio::test]
async fn bank_isolation() {
    let (store, _container) = setup_store().await;

    let bank_a = MemoryBank {
        id: BankId::new(),
        name: "bank A".into(),
        mission: "A".into(),
        directives: vec![],
        disposition: Disposition::default(),
        embedding_model: String::new(),
        embedding_dimensions: 0,
    };
    let bank_b = MemoryBank {
        id: BankId::new(),
        name: "bank B".into(),
        mission: "B".into(),
        directives: vec![],
        disposition: Disposition::default(),
        embedding_model: String::new(),
        embedding_dimensions: 0,
    };
    store.create_bank(&bank_a).await.unwrap();
    store.create_bank(&bank_b).await.unwrap();

    let mut fact_a = make_fact(bank_a.id);
    fact_a.content = "belongs to A".into();
    let mut fact_b = make_fact(bank_b.id);
    fact_b.id = FactId::new();
    fact_b.content = "belongs to B".into();

    store
        .insert_facts(&[fact_a.clone(), fact_b.clone()])
        .await
        .unwrap();

    let results_a = store
        .get_facts_by_bank(bank_a.id, FactFilter::default())
        .await
        .unwrap();
    assert_eq!(results_a.len(), 1);
    assert_eq!(results_a[0].content, "belongs to A");

    let results_b = store
        .get_facts_by_bank(bank_b.id, FactFilter::default())
        .await
        .unwrap();
    assert_eq!(results_b.len(), 1);
    assert_eq!(results_b[0].content, "belongs to B");
}

#[tokio::test]
async fn upsert_entity_returns_existing_id_on_conflict() {
    let (store, _container) = setup_store().await;
    let bank = make_bank();
    store.create_bank(&bank).await.unwrap();

    let original = Entity {
        id: EntityId::new(),
        canonical_name: "Rust".into(),
        aliases: vec!["rust-lang".into()],
        entity_type: EntityType::Concept,
        bank_id: bank.id,
    };
    let id1 = store.upsert_entity(&original).await.unwrap();
    assert_eq!(id1, original.id);

    // Upsert again with a different UUID but same (bank_id, canonical_name)
    let duplicate = Entity {
        id: EntityId::new(), // different ID
        canonical_name: "Rust".into(),
        aliases: vec!["rust-lang".into(), "Rust programming".into()],
        entity_type: EntityType::Concept,
        bank_id: bank.id,
    };
    let id2 = store.upsert_entity(&duplicate).await.unwrap();

    // Should return the ORIGINAL id, not the new one
    assert_eq!(id2, original.id);
    assert_ne!(id2, duplicate.id);
}

#[tokio::test]
async fn entity_facts_lookup() {
    let (store, _container) = setup_store().await;
    let bank = make_bank();
    store.create_bank(&bank).await.unwrap();

    let entity = Entity {
        id: EntityId::new(),
        canonical_name: "Rust".into(),
        aliases: vec![],
        entity_type: EntityType::Concept,
        bank_id: bank.id,
    };
    store.upsert_entity(&entity).await.unwrap();

    let mut fact = make_fact(bank.id);
    fact.entity_ids = vec![entity.id];
    store.insert_facts(&[fact.clone()]).await.unwrap();

    let entity_facts = store.get_entity_facts(entity.id).await.unwrap();
    assert_eq!(entity_facts.len(), 1);
    assert_eq!(entity_facts[0].id, fact.id);
}
