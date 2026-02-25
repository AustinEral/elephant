//! Reciprocal Rank Fusion (RRF) for merging multiple ranked lists.

use std::collections::HashMap;

use crate::types::{FactId, ScoredFact};

/// Fuse multiple ranked lists using Reciprocal Rank Fusion.
///
/// `score = Σ 1/(k + rank + 1)` per list (0-indexed rank).
/// Deduplicates by FactId, merging sources.
pub fn fuse_rankings(rankings: &[Vec<ScoredFact>], k: f32) -> Vec<ScoredFact> {
    // Track: FactId → (accumulated score, best ScoredFact for fact data, merged sources)
    let mut scores: HashMap<FactId, (f32, usize)> = HashMap::new();
    let mut facts: HashMap<FactId, ScoredFact> = HashMap::new();

    for list in rankings {
        for (rank, sf) in list.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f32);
            let entry = scores.entry(sf.fact.id).or_insert((0.0, 0));
            entry.0 += rrf_score;
            entry.1 += 1;

            facts
                .entry(sf.fact.id)
                .and_modify(|existing| {
                    for src in &sf.sources {
                        if !existing.sources.contains(src) {
                            existing.sources.push(*src);
                        }
                    }
                })
                .or_insert_with(|| sf.clone());
        }
    }

    let mut result: Vec<ScoredFact> = facts
        .into_iter()
        .map(|(id, mut sf)| {
            sf.score = scores[&id].0;
            sf
        })
        .collect();

    result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::Utc;

    fn make_fact(id: FactId, content: &str) -> Fact {
        Fact {
            id,
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
        }
    }

    fn scored(fact: Fact, score: f32, source: RetrievalSource) -> ScoredFact {
        ScoredFact {
            fact,
            score,
            sources: vec![source],
        }
    }

    #[test]
    fn single_list() {
        let id1 = FactId::new();
        let id2 = FactId::new();
        let list = vec![
            scored(make_fact(id1, "first"), 0.9, RetrievalSource::Semantic),
            scored(make_fact(id2, "second"), 0.5, RetrievalSource::Semantic),
        ];
        let result = fuse_rankings(&[list], 60.0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].fact.id, id1);
        assert_eq!(result[1].fact.id, id2);
        // rank 0: 1/(60+0) = 1/60
        assert!((result[0].score - 1.0 / 60.0).abs() < 1e-6);
        // rank 1: 1/(60+1) = 1/61
        assert!((result[1].score - 1.0 / 61.0).abs() < 1e-6);
    }

    #[test]
    fn two_lists_shared_fact_ranks_higher() {
        let id1 = FactId::new();
        let id2 = FactId::new();
        let id3 = FactId::new();

        // id1 is #1 in both lists
        let list_a = vec![
            scored(make_fact(id1, "shared"), 0.9, RetrievalSource::Semantic),
            scored(make_fact(id2, "only-a"), 0.5, RetrievalSource::Semantic),
        ];
        let list_b = vec![
            scored(make_fact(id1, "shared"), 0.8, RetrievalSource::Keyword),
            scored(make_fact(id3, "only-b"), 0.3, RetrievalSource::Keyword),
        ];

        let result = fuse_rankings(&[list_a, list_b], 60.0);
        assert_eq!(result[0].fact.id, id1);
        // 1/60 + 1/60 = 2/60
        assert!((result[0].score - 2.0 / 60.0).abs() < 1e-6);
        // Merged sources
        assert!(result[0].sources.contains(&RetrievalSource::Semantic));
        assert!(result[0].sources.contains(&RetrievalSource::Keyword));
    }

    #[test]
    fn disjoint_lists() {
        let id1 = FactId::new();
        let id2 = FactId::new();
        let list_a = vec![scored(make_fact(id1, "a"), 0.9, RetrievalSource::Semantic)];
        let list_b = vec![scored(make_fact(id2, "b"), 0.8, RetrievalSource::Keyword)];

        let result = fuse_rankings(&[list_a, list_b], 60.0);
        assert_eq!(result.len(), 2);
        // Both have same RRF score (1/60), order is stable
        assert!((result[0].score - result[1].score).abs() < 1e-6);
    }

    #[test]
    fn empty_lists() {
        let result = fuse_rankings(&[], 60.0);
        assert!(result.is_empty());

        let result2 = fuse_rankings(&[vec![], vec![]], 60.0);
        assert!(result2.is_empty());
    }

    #[test]
    fn first_in_two_beats_first_in_one() {
        let shared = FactId::new();
        let solo = FactId::new();

        let list_a = vec![
            scored(make_fact(solo, "solo"), 0.99, RetrievalSource::Semantic),
            scored(make_fact(shared, "shared"), 0.5, RetrievalSource::Semantic),
        ];
        let list_b = vec![scored(
            make_fact(shared, "shared"),
            0.8,
            RetrievalSource::Keyword,
        )];

        let result = fuse_rankings(&[list_a, list_b], 60.0);
        // shared: 1/61 + 1/60, solo: 1/60
        let shared_score = 1.0 / 61.0 + 1.0 / 60.0;
        let solo_score = 1.0 / 60.0;
        assert!(shared_score > solo_score);
        assert_eq!(result[0].fact.id, shared);
    }
}
