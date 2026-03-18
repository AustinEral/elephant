//! Background consolidation workers that synthesize higher-level memory structures.
//!
//! Two workers:
//! - **Observation consolidator** — merges raw facts into entity-level observations
//! - **Opinion merger** — deduplicates and reconciles opinions

pub mod observation;
pub mod opinion_merger;

pub use observation::{
    ConsolidationConfig, ConsolidationProgress, Consolidator, DefaultConsolidator,
};
pub use opinion_merger::{DefaultOpinionMerger, OpinionMerger};

use crate::util::cosine_similarity;

/// Single-linkage clustering by pairwise embedding cosine similarity.
///
/// Returns groups of indices into `facts` where all members are within
/// `threshold` similarity of at least one other member in the cluster.
pub fn cluster_by_similarity(embeddings: &[&[f32]], threshold: f32) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(embeddings[i], embeddings[j]);
            if sim >= threshold {
                union(&mut parent, i, j);
            }
        }
    }

    // Group by root
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }

    groups.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_identical_vectors() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let v3 = vec![0.0, 1.0, 0.0]; // orthogonal

        let embeddings: Vec<&[f32]> = vec![&v1, &v2, &v3];
        let clusters = cluster_by_similarity(&embeddings, 0.9);

        // v1 and v2 should cluster together, v3 alone
        assert_eq!(clusters.len(), 2);
        let has_pair = clusters.iter().any(|c| c.len() == 2);
        let has_single = clusters.iter().any(|c| c.len() == 1);
        assert!(has_pair);
        assert!(has_single);
    }

    #[test]
    fn cluster_all_similar() {
        let v1 = vec![1.0, 0.1, 0.0];
        let v2 = vec![1.0, 0.2, 0.0];
        let v3 = vec![1.0, 0.15, 0.0];

        let embeddings: Vec<&[f32]> = vec![&v1, &v2, &v3];
        let clusters = cluster_by_similarity(&embeddings, 0.9);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn cluster_empty() {
        let embeddings: Vec<&[f32]> = vec![];
        let clusters = cluster_by_similarity(&embeddings, 0.9);
        assert!(clusters.is_empty());
    }

    #[test]
    fn cluster_single_linkage_chains() {
        // A-B similar, B-C similar, but A-C not — single linkage should still merge all three
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.8, 0.6]; // sim to v1 ≈ 0.8
        let v3 = vec![0.0, 1.0]; // sim to v2 ≈ 0.6, sim to v1 = 0.0

        let embeddings: Vec<&[f32]> = vec![&v1, &v2, &v3];
        // With threshold 0.7: v1-v2 link, v2-v3 NOT linked, v1-v3 NOT linked
        let clusters = cluster_by_similarity(&embeddings, 0.7);
        assert_eq!(clusters.len(), 2); // v1+v2, v3 alone
    }
}
