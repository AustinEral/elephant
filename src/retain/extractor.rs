//! LLM-based fact extraction from text chunks (Phase 2B).

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::llm::LlmClient;
use crate::types::{CompletionRequest, ExtractedFact, ExtractionInput, Message};

/// Trait for extracting structured facts from raw text.
#[async_trait]
pub trait FactExtractor: Send + Sync {
    /// Extract facts from a single chunk of text.
    async fn extract(&self, input: &ExtractionInput) -> Result<Vec<ExtractedFact>>;

    /// Render the extractor user message for provenance/debugging.
    fn render_user_message(&self, input: &ExtractionInput) -> String {
        render_extraction_user_message(input)
    }
}

/// Fact extractor powered by an LLM.
pub struct LlmFactExtractor {
    llm: Arc<dyn LlmClient>,
}

/// Base extraction prompt template.
pub const EXTRACT_PROMPT_TEMPLATE: &str = include_str!("../../prompts/extract_facts.txt");
/// Extraction temperature.
pub const EXTRACT_TEMPERATURE: f32 = 0.1;
/// Extraction output cap.
pub const EXTRACT_MAX_TOKENS: usize = 4096;

impl LlmFactExtractor {
    /// Create a new extractor with the given LLM client.
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }

    /// Render the exact system prompt and user message sent to the extractor LLM.
    pub fn render_prompt(input: &ExtractionInput) -> (String, String) {
        (
            Self::build_system_prompt(input),
            render_extraction_user_message(input),
        )
    }

    fn build_system_prompt(input: &ExtractionInput) -> String {
        let mut prompt = EXTRACT_PROMPT_TEMPLATE.to_string();

        if let Some(ref custom) = input.custom_instructions {
            prompt.push_str("\n\n## Additional Guidelines\n\n");
            prompt.push_str(custom);
        }

        prompt
    }
}

/// Render the extractor user message from a structured extraction input.
pub fn render_extraction_user_message(input: &ExtractionInput) -> String {
    let mut msg = String::new();

    if let Some(ref speaker) = input.speaker {
        msg.push_str(&format!("Speaker: {speaker}\n\n"));
    }

    if let Some(ref ctx) = input.context {
        msg.push_str("## Preceding Context\n\n");
        msg.push_str(ctx);
        msg.push_str("\n\n---\n\n");
    }

    msg.push_str("## Content to Extract From\n\n");
    msg.push_str(&input.content);
    msg.push_str("\n\nTimestamp: ");
    msg.push_str(&input.timestamp.to_rfc3339());

    msg
}

#[async_trait]
impl FactExtractor for LlmFactExtractor {
    async fn extract(&self, input: &ExtractionInput) -> Result<Vec<ExtractedFact>> {
        let (system, user_msg) = Self::render_prompt(input);

        let request = CompletionRequest {
            model: String::new(),
            system: Some(system),
            messages: vec![Message::text("user", user_msg)],
            temperature: Some(EXTRACT_TEMPERATURE),
            max_tokens: Some(EXTRACT_MAX_TOKENS),
            ..Default::default()
        };

        match crate::llm::complete_structured::<Vec<ExtractedFact>>(&*self.llm, request).await {
            Ok(facts) => Ok(facts),
            Err(crate::error::Error::Llm(msg)) if msg.contains("refused") => {
                tracing::warn!("extraction refused by model — treating as empty");
                Ok(vec![])
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::llm::mock::MockLlmClient;
    use crate::types::{BankId, FactType, NetworkType};

    #[tokio::test]
    async fn extract_parses_valid_json() {
        let response_json = serde_json::to_string(&vec![
            ExtractedFact {
                content: "Rust uses ownership for memory safety".into(),
                fact_type: FactType::World,
                network: NetworkType::World,
                entity_mentions: vec!["Rust".into()],
                temporal_range: None,
                confidence: None,
            },
            ExtractedFact {
                content: "The team chose Postgres over MongoDB".into(),
                fact_type: FactType::Experience,
                network: NetworkType::Experience,
                entity_mentions: vec!["Postgres".into(), "MongoDB".into()],
                temporal_range: None,
                confidence: None,
            },
        ])
        .unwrap();

        let mock = MockLlmClient::new();
        mock.push_response(response_json);

        let extractor = LlmFactExtractor::new(Arc::new(mock));
        let input = ExtractionInput {
            content: "We decided to use Rust with Postgres instead of MongoDB.".into(),
            bank_id: BankId::new(),
            context: None,
            timestamp: chrono::Utc::now(),
            turn_id: None,
            custom_instructions: None,
            speaker: None,
        };

        let facts = extractor.extract(&input).await.unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].entity_mentions, vec!["Rust"]);
        assert_eq!(facts[1].entity_mentions, vec!["Postgres", "MongoDB"]);
    }

    #[tokio::test]
    async fn extract_handles_markdown_fenced_json() {
        let json = serde_json::to_string(&vec![ExtractedFact {
            content: "Python is popular".into(),
            fact_type: FactType::World,
            network: NetworkType::World,
            entity_mentions: vec!["Python".into()],
            temporal_range: None,
            confidence: None,
        }])
        .unwrap();

        let mock = MockLlmClient::new();
        mock.push_response(format!("Here are the facts:\n```json\n{json}\n```"));

        let extractor = LlmFactExtractor::new(Arc::new(mock));
        let input = ExtractionInput {
            content: "Python is a popular language.".into(),
            bank_id: BankId::new(),
            context: None,
            timestamp: chrono::Utc::now(),
            turn_id: None,
            custom_instructions: None,
            speaker: None,
        };

        let facts = extractor.extract(&input).await.unwrap();
        assert_eq!(facts.len(), 1);
    }

    #[tokio::test]
    async fn extract_with_context_and_custom_instructions() {
        let json = serde_json::to_string(&Vec::<ExtractedFact>::new()).unwrap();

        let mock = MockLlmClient::new();
        mock.push_response(json);

        let extractor = LlmFactExtractor::new(Arc::new(mock));
        let input = ExtractionInput {
            content: "Nothing notable.".into(),
            bank_id: BankId::new(),
            context: Some("Previously discussed Rust.".into()),
            timestamp: chrono::Utc::now(),
            turn_id: None,
            custom_instructions: Some("Focus on technical decisions.".into()),
            speaker: None,
        };

        let facts = extractor.extract(&input).await.unwrap();
        assert!(facts.is_empty());
    }
}
