//! Token budget enforcement for recall results.

use crate::types::ScoredFact;

/// Trait for estimating the token count of text.
pub trait Tokenizer: Send + Sync {
    /// Estimate the number of tokens in the given text.
    fn count_tokens(&self, text: &str) -> usize;
}

/// Simple tokenizer that estimates tokens as chars/4.
pub struct EstimateTokenizer;

impl Tokenizer for EstimateTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        text.len().div_ceil(4)
    }
}

/// Greedily select facts that fit within the token budget.
///
/// Assumes facts are pre-sorted by score (descending). Accumulates tokens
/// and stops when the next fact would exceed the budget.
pub fn apply_budget(
    facts: &[ScoredFact],
    budget: usize,
    tokenizer: &dyn Tokenizer,
) -> Vec<ScoredFact> {
    let mut used = 0;
    let mut result = Vec::new();

    for sf in facts {
        let tokens = tokenizer.count_tokens(&sf.fact.content);
        if used + tokens > budget {
            break;
        }
        used += tokens;
        result.push(sf.clone());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::Utc;

    fn make_scored(content: &str, score: f32) -> ScoredFact {
        ScoredFact {
            fact: Fact {
                id: FactId::new(),
                bank_id: BankId::new(),
                content: content.into(),
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
            },
            score,
            sources: vec![RetrievalSource::Semantic],
        }
    }

    #[test]
    fn fills_exactly() {
        let tok = EstimateTokenizer;
        // "abcd" = 4 chars = 1 token (with EstimateTokenizer: (4+3)/4 = 1)
        // Actually (4+3)/4 = 1. Let's use a string that gives exact count.
        // 4 chars = (4+3)/4 = 1 token
        let facts = vec![make_scored("abcd", 1.0), make_scored("efgh", 0.9)];
        // Budget of 2 tokens should fit both (each is ~1 token)
        let result = apply_budget(&facts, 2, &tok);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn budget_zero() {
        let tok = EstimateTokenizer;
        let facts = vec![make_scored("hello", 1.0)];
        let result = apply_budget(&facts, 0, &tok);
        assert!(result.is_empty());
    }

    #[test]
    fn oversized_single_fact() {
        let tok = EstimateTokenizer;
        // "a]" repeated 100 chars = 25+ tokens
        let long = "a".repeat(100);
        let facts = vec![make_scored(&long, 1.0)];
        let result = apply_budget(&facts, 5, &tok);
        assert!(result.is_empty());
    }

    #[test]
    fn normal_budget() {
        let tok = EstimateTokenizer;
        // Each 8-char string = (8+3)/4 = 2 tokens
        let facts = vec![
            make_scored("12345678", 1.0),
            make_scored("abcdefgh", 0.9),
            make_scored("ABCDEFGH", 0.8),
        ];
        // Budget 5: fits 2 facts (4 tokens), third would be 6 > 5
        let result = apply_budget(&facts, 5, &tok);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn estimate_tokenizer_counts() {
        let tok = EstimateTokenizer;
        assert_eq!(tok.count_tokens(""), 0); // (0+3)/4 = 0
        assert_eq!(tok.count_tokens("a"), 1); // (1+3)/4 = 1
        assert_eq!(tok.count_tokens("abcd"), 1); // (4+3)/4 = 1
        assert_eq!(tok.count_tokens("abcde"), 2); // (5+3)/4 = 2
    }
}
