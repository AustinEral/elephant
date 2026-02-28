//! Lightweight prompt evaluation tests — no Postgres, no embeddings, no containers.
//!
//! Each test builds the exact CompletionRequest the pipeline would build,
//! sends it to the real LLM, parses the response with `complete_structured`,
//! prints the raw output for human review, and asserts structural invariants.
//!
//! Run with:
//!   cargo test --test prompt_eval -- --ignored --nocapture
//!
//! Only requires LLM_API_KEY + LLM_MODEL from .env.

use elephant::llm::anthropic::AnthropicClient;
use elephant::llm::complete_structured;
use elephant::types::llm::{CompletionRequest, Message};
use elephant::types::pipeline::ExtractedFact;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn init() {
    let _ = dotenvy::dotenv();
}

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("{key} must be set in .env"))
}

fn llm_client() -> AnthropicClient {
    let model = std::env::var("LLM_MODEL")
        .or_else(|_| std::env::var("RETAIN_LLM_MODEL"))
        .unwrap_or_else(|_| panic!("LLM_MODEL or RETAIN_LLM_MODEL must be set in .env"));
    AnthropicClient::new(env("LLM_API_KEY"), model)
}

fn extraction_request(content: &str, speaker: Option<&str>) -> CompletionRequest {
    let system = include_str!("../prompts/extract_facts.txt").to_string();
    let mut user_msg = String::new();
    if let Some(name) = speaker {
        user_msg.push_str(&format!("Speaker: {name}\n\n"));
    }
    user_msg.push_str(content);
    CompletionRequest {
        model: String::new(), // AnthropicClient uses its default
        messages: vec![Message {
            role: "user".into(),
            content: user_msg,
        }],
        max_tokens: None,
        temperature: Some(0.0),
        system: Some(system),
    }
}

fn reflect_request(context: &str, opinions: &str, question: &str) -> CompletionRequest {
    let template = include_str!("../prompts/reflect.txt");
    let user_prompt = template
        .replace("{context}", context)
        .replace("{opinions}", opinions)
        .replace("{question}", question);
    CompletionRequest {
        model: String::new(),
        messages: vec![Message {
            role: "user".into(),
            content: user_prompt,
        }],
        max_tokens: None,
        temperature: Some(0.3),
        system: None,
    }
}

fn synthesize_observation_request(entity_name: &str, facts: &str) -> CompletionRequest {
    let template = include_str!("../prompts/synthesize_observation.txt");
    let user_prompt = template
        .replace("{entity_name}", entity_name)
        .replace("{facts}", facts);
    CompletionRequest {
        model: String::new(),
        messages: vec![Message {
            role: "user".into(),
            content: user_prompt,
        }],
        max_tokens: None,
        temperature: Some(0.3),
        system: None,
    }
}

fn merge_opinions_request(opinions: &str) -> CompletionRequest {
    let template = include_str!("../prompts/merge_opinions.txt");
    let user_prompt = template.replace("{opinions}", opinions);
    CompletionRequest {
        model: String::new(),
        messages: vec![Message {
            role: "user".into(),
            content: user_prompt,
        }],
        max_tokens: None,
        temperature: Some(0.0),
        system: None,
    }
}

// ---------------------------------------------------------------------------
// Response types (matching what the pipeline expects)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ReflectResponse {
    response: String,
    sources: Vec<String>,
    #[serde(default)]
    new_opinions: Vec<NewOpinion>,
    confidence: f32,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NewOpinion {
    content: String,
    evidence: Vec<String>,
    confidence: f32,
}

#[derive(Debug, Deserialize)]
struct ObservationResponse {
    observation: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MergeResponse {
    classification: String,
    #[serde(default)]
    merged_text: Option<String>,
    #[serde(default)]
    superseded_index: Option<usize>,
}

// ===========================================================================
// extract_facts
// ===========================================================================

#[tokio::test]
#[ignore]
async fn eval_extract_simple() {
    init();
    let client = llm_client();
    let input = "Alice works at Acme Corp as a senior engineer. She joined in 2020 and uses Rust and Python daily.";

    let request = extraction_request(input, None);
    let facts: Vec<ExtractedFact> = complete_structured(&client, request)
        .await
        .expect("LLM call failed");

    println!("=== eval_extract_simple ===");
    println!("{facts:#?}");

    assert!(!facts.is_empty(), "should extract at least one fact");
    for fact in &facts {
        assert!(!fact.content.is_empty(), "fact content should not be empty");
        assert!(
            fact.fact_type == elephant::types::fact::FactType::World
                || fact.fact_type == elephant::types::fact::FactType::Experience,
            "fact_type should be world or experience, got {:?}",
            fact.fact_type
        );
        if let Some(conf) = fact.confidence {
            assert!(
                (0.0..=1.0).contains(&conf),
                "confidence out of range: {conf}"
            );
        }
    }

    // "Alice joined in 2020" should be experience (something a person did)
    let joined_fact = facts.iter().find(|f| f.content.to_lowercase().contains("joined"));
    if let Some(fact) = joined_fact {
        assert_eq!(
            fact.fact_type,
            elephant::types::fact::FactType::Experience,
            "joining a company is an experience, not world knowledge"
        );
    }

    // Entity coverage
    let all_entities: Vec<&str> = facts
        .iter()
        .flat_map(|f| f.entity_mentions.iter().map(String::as_str))
        .collect();
    let all_entities_lower: Vec<String> = all_entities.iter().map(|e| e.to_lowercase()).collect();
    assert!(
        all_entities_lower.iter().any(|e| e.contains("alice")),
        "should mention Alice, got entities: {all_entities:?}"
    );
    assert!(
        all_entities_lower.iter().any(|e| e.contains("acme")),
        "should mention Acme Corp, got entities: {all_entities:?}"
    );
}

#[tokio::test]
#[ignore]
async fn eval_extract_conversational() {
    init();
    let client = llm_client();
    let input = "So yeah, I talked to Bob yesterday and we decided to switch from MySQL to Postgres \
                 for the new project. He thinks it's better for our use case.";

    let request = extraction_request(input, Some("Austin"));
    let facts: Vec<ExtractedFact> = complete_structured(&client, request)
        .await
        .expect("LLM call failed");

    println!("=== eval_extract_conversational ===");
    println!("{facts:#?}");

    assert!(!facts.is_empty(), "should extract at least one fact");

    let all_content: String = facts
        .iter()
        .map(|f| f.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let all_content_lower = all_content.to_lowercase();

    // Should capture the decision
    assert!(
        all_content_lower.contains("mysql") || all_content_lower.contains("postgres"),
        "should mention the database decision, got: {all_content}"
    );

    // Should resolve "I" to "Austin"
    assert!(
        all_content_lower.contains("austin"),
        "should resolve 'I' to speaker name Austin, got: {all_content}"
    );

    // Entity coverage
    let all_entities: Vec<String> = facts
        .iter()
        .flat_map(|f| f.entity_mentions.iter().map(|e| e.to_lowercase()))
        .collect();
    assert!(
        all_entities.iter().any(|e| e.contains("bob")),
        "should mention Bob, got entities: {all_entities:?}"
    );
}

#[tokio::test]
#[ignore]
async fn eval_extract_selectivity() {
    init();
    let client = llm_client();
    let input = "Hey! How's it going? Yeah so anyway, let me take a look at that. \
                 Hmm one sec... Ok so we decided to use pgvector for similarity search \
                 because it integrates natively with our existing Postgres setup. \
                 Cool, let me know if you have questions.";

    let request = extraction_request(input, Some("Austin"));
    let facts: Vec<ExtractedFact> = complete_structured(&client, request)
        .await
        .expect("LLM call failed");

    println!("=== eval_extract_selectivity ===");
    println!("{facts:#?}");

    // Should extract the decision, skip all the filler
    assert!(
        facts.len() <= 2,
        "should extract at most 2 facts from mostly filler, got {}",
        facts.len()
    );
    assert!(
        !facts.is_empty(),
        "should extract at least the pgvector decision"
    );

    let all_content: String = facts.iter().map(|f| f.content.as_str()).collect::<Vec<_>>().join(" ");
    let lower = all_content.to_lowercase();
    assert!(
        lower.contains("pgvector"),
        "should capture the pgvector decision, got: {all_content}"
    );
    // Should NOT extract greetings or task chatter
    assert!(
        !lower.contains("how's it going") && !lower.contains("one sec") && !lower.contains("let me know"),
        "should skip filler, got: {all_content}"
    );
}

// ===========================================================================
// reflect
// ===========================================================================

#[tokio::test]
#[ignore]
async fn eval_reflect_with_context() {
    init();
    let client = llm_client();

    let context = "\
[FACT 01JEXAMPLE00000000000001] Alice works at Acme Corp as a senior engineer specializing in backend systems.
[FACT 01JEXAMPLE00000000000002] Acme Corp migrated from MySQL to PostgreSQL in Q1 2024 for better JSON support.
[FACT 01JEXAMPLE00000000000003] Alice led the database migration project and wrote the migration tooling in Rust.";

    let opinions = "(none)";
    let question = "What role did Alice play in the database migration?";

    let request = reflect_request(context, opinions, question);
    let resp: ReflectResponse = complete_structured(&client, request)
        .await
        .expect("LLM call failed");

    println!("=== eval_reflect_with_context ===");
    println!("{resp:#?}");

    assert!(
        resp.confidence > 0.5,
        "confidence should be > 0.5 with good context, got {}",
        resp.confidence
    );
    assert!(!resp.sources.is_empty(), "should cite at least one source");
    assert!(!resp.response.is_empty(), "response should not be empty");

    // Should reference the provided fact IDs
    let response_lower = resp.response.to_lowercase();
    assert!(
        response_lower.contains("alice") || response_lower.contains("migration"),
        "response should reference Alice or the migration"
    );
}

#[tokio::test]
#[ignore]
async fn eval_reflect_insufficient() {
    init();
    let client = llm_client();

    let context =
        "[FACT 01JEXAMPLE00000000000001] The office coffee machine was replaced last Tuesday.";
    let opinions = "(none)";
    let question = "What programming language does the team use for the backend?";

    let request = reflect_request(context, opinions, question);
    let resp: ReflectResponse = complete_structured(&client, request)
        .await
        .expect("LLM call failed");

    println!("=== eval_reflect_insufficient ===");
    println!("{resp:#?}");

    assert!(
        resp.confidence < 0.5,
        "confidence should be < 0.5 with insufficient context, got {}",
        resp.confidence
    );
}

// ===========================================================================
// synthesize_observation
// ===========================================================================

#[tokio::test]
#[ignore]
async fn eval_synthesize_observation() {
    init();
    let client = llm_client();

    let entity_name = "PostgreSQL";
    let facts = "\
- The team chose PostgreSQL over MongoDB because they need ACID transactions.
- PostgreSQL 16 was deployed to production in January 2024.
- The team uses PostgreSQL's JSONB columns for semi-structured metadata.
- Database backups run nightly via pg_dump to S3.";

    let request = synthesize_observation_request(entity_name, facts);
    let resp: ObservationResponse = complete_structured(&client, request)
        .await
        .expect("LLM call failed");

    println!("=== eval_synthesize_observation ===");
    println!("{resp:#?}");

    assert!(
        !resp.observation.is_empty(),
        "observation should not be empty"
    );
    assert!(
        resp.observation.to_lowercase().contains("postgres"),
        "observation should mention PostgreSQL, got: {}",
        resp.observation
    );
}

// ===========================================================================
// merge_opinions
// ===========================================================================

#[tokio::test]
#[ignore]
async fn eval_merge_consistent_opinions() {
    init();
    let client = llm_client();

    let opinions = "\
1. Rust's ownership model makes it significantly safer than C++ for systems programming because it eliminates entire classes of memory bugs at compile time.
2. Rust provides stronger memory safety guarantees than C++ through its borrow checker, which catches use-after-free and data race bugs before the program runs.";

    let request = merge_opinions_request(opinions);
    let resp: MergeResponse = complete_structured(&client, request)
        .await
        .expect("LLM call failed");

    println!("=== eval_merge_consistent_opinions ===");
    println!("{resp:#?}");

    let valid_classifications = ["consistent", "contradictory", "superseded"];
    assert!(
        valid_classifications.contains(&resp.classification.as_str()),
        "classification should be one of {valid_classifications:?}, got: {}",
        resp.classification
    );

    if resp.classification == "consistent" {
        assert!(
            resp.merged_text.as_ref().is_some_and(|t| !t.is_empty()),
            "consistent classification should have non-empty merged_text"
        );
    }
}

