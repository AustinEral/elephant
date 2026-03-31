//! App-layer consolidation policy and retain wrapping.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tracing::{debug, info, warn};

use super::AppBackgroundConsolidationInfo;
use crate::config::BackgroundConsolidationConfig;
use crate::consolidation::{Consolidator, OpinionMerger};
use crate::error::{Error, Result};
use crate::retain::RetainPipeline;
use crate::storage::MemoryStore;
use crate::types::{BankId, FactFilter, NetworkType, RetainInput, RetainOutput};

#[derive(Debug, Clone)]
pub(crate) struct ConsolidationPolicy {
    enabled: bool,
    min_unconsolidated_facts: usize,
    cooldown: Duration,
    merge_opinions_after: bool,
}

impl ConsolidationPolicy {
    fn new(
        enabled: bool,
        min_unconsolidated_facts: usize,
        cooldown: Duration,
        merge_opinions_after: bool,
    ) -> Result<Self> {
        if min_unconsolidated_facts == 0 {
            return Err(Error::Configuration(
                "SERVER_AUTO_CONSOLIDATION_MIN_FACTS must be greater than 0".into(),
            ));
        }

        Ok(Self {
            enabled,
            min_unconsolidated_facts,
            cooldown,
            merge_opinions_after,
        })
    }

    pub(crate) fn from_config(config: &BackgroundConsolidationConfig) -> Result<Self> {
        Self::new(
            config.enabled(),
            config.min_unconsolidated_facts(),
            Duration::from_secs(config.cooldown_secs()),
            config.merge_opinions_after(),
        )
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn to_info(&self) -> AppBackgroundConsolidationInfo {
        AppBackgroundConsolidationInfo {
            enabled: self.enabled,
            min_facts: self.min_unconsolidated_facts,
            cooldown_secs: self.cooldown.as_secs(),
            merge_opinions_after: self.merge_opinions_after,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConsolidationDecision {
    Disabled,
    NoNewFacts,
    Cooldown,
    AlreadyRunning,
    Enqueued,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConsolidationOutcome {
    BelowThreshold {
        unconsolidated_facts: usize,
    },
    Consolidated {
        unconsolidated_facts: usize,
        observations_created: usize,
        observations_updated: usize,
        opinions_merged: usize,
    },
}

#[derive(Debug, Default, Clone, Copy)]
struct BankConsolidationState {
    running: bool,
    last_evaluated_at: Option<Instant>,
}

struct ConsolidationCoordinator {
    store: Arc<dyn MemoryStore>,
    consolidator: Arc<dyn Consolidator>,
    opinion_merger: Arc<dyn OpinionMerger>,
    policy: ConsolidationPolicy,
    states: Mutex<HashMap<BankId, BankConsolidationState>>,
}

impl ConsolidationCoordinator {
    fn lock_states(&self) -> std::sync::MutexGuard<'_, HashMap<BankId, BankConsolidationState>> {
        match self.states.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("server consolidation state mutex was poisoned; clearing scheduler state");
                let mut guard = poisoned.into_inner();
                guard.clear();
                guard
            }
        }
    }

    fn new(
        store: Arc<dyn MemoryStore>,
        consolidator: Arc<dyn Consolidator>,
        opinion_merger: Arc<dyn OpinionMerger>,
        policy: ConsolidationPolicy,
    ) -> Self {
        Self {
            store,
            consolidator,
            opinion_merger,
            policy,
            states: Mutex::new(HashMap::new()),
        }
    }

    fn maybe_schedule(
        self: &Arc<Self>,
        bank_id: BankId,
        facts_stored: usize,
    ) -> ConsolidationDecision {
        if !self.policy.enabled() {
            return ConsolidationDecision::Disabled;
        }
        if facts_stored == 0 {
            return ConsolidationDecision::NoNewFacts;
        }

        let now = Instant::now();
        let mut states = self.lock_states();
        let state = states.entry(bank_id).or_default();
        if state.running {
            return ConsolidationDecision::AlreadyRunning;
        }
        if let Some(last_evaluated_at) = state.last_evaluated_at
            && now.duration_since(last_evaluated_at) < self.policy.cooldown
        {
            return ConsolidationDecision::Cooldown;
        }

        state.running = true;
        drop(states);

        let coordinator = Arc::clone(self);
        tokio::spawn(async move {
            coordinator.run(bank_id).await;
        });

        ConsolidationDecision::Enqueued
    }

    async fn run(self: Arc<Self>, bank_id: BankId) {
        let outcome = self.run_once(bank_id).await;
        let evaluated_at = Instant::now();

        let mut states = self.lock_states();
        let state = states.entry(bank_id).or_default();
        state.running = false;
        state.last_evaluated_at = Some(evaluated_at);
        drop(states);

        match outcome {
            Ok(ConsolidationOutcome::BelowThreshold {
                unconsolidated_facts,
            }) => debug!(
                %bank_id,
                unconsolidated_facts,
                min_unconsolidated_facts = self.policy.min_unconsolidated_facts,
                "server consolidation skipped: below threshold"
            ),
            Ok(ConsolidationOutcome::Consolidated {
                unconsolidated_facts,
                observations_created,
                observations_updated,
                opinions_merged,
            }) => info!(
                %bank_id,
                unconsolidated_facts,
                observations_created,
                observations_updated,
                opinions_merged,
                "server consolidation completed"
            ),
            Err(err) => warn!(%bank_id, error = %err, "server consolidation failed"),
        }
    }

    async fn run_once(&self, bank_id: BankId) -> Result<ConsolidationOutcome> {
        let unconsolidated_facts = self
            .store
            .get_facts_by_bank(
                bank_id,
                FactFilter {
                    network: Some(vec![NetworkType::World, NetworkType::Experience]),
                    unconsolidated_only: true,
                    ..Default::default()
                },
            )
            .await?
            .len();

        if unconsolidated_facts < self.policy.min_unconsolidated_facts {
            return Ok(ConsolidationOutcome::BelowThreshold {
                unconsolidated_facts,
            });
        }

        let report = self.consolidator.consolidate(bank_id).await?;
        let opinion_report = if self.policy.merge_opinions_after {
            Some(self.opinion_merger.merge(bank_id).await?)
        } else {
            None
        };

        Ok(ConsolidationOutcome::Consolidated {
            unconsolidated_facts,
            observations_created: report.observations_created,
            observations_updated: report.observations_updated,
            opinions_merged: opinion_report
                .as_ref()
                .map(|report| report.opinions_merged)
                .unwrap_or(0),
        })
    }
}

pub(crate) fn wrap_retain_pipeline_with_consolidation(
    retain: Arc<dyn RetainPipeline>,
    store: Arc<dyn MemoryStore>,
    consolidator: Arc<dyn Consolidator>,
    opinion_merger: Arc<dyn OpinionMerger>,
    policy: ConsolidationPolicy,
) -> Result<Arc<dyn RetainPipeline>> {
    if !policy.enabled() {
        return Ok(retain);
    }

    let coordinator = Arc::new(ConsolidationCoordinator::new(
        store,
        consolidator,
        opinion_merger,
        policy,
    ));
    Ok(Arc::new(ConsolidatingRetainPipeline {
        retain,
        coordinator,
    }))
}

struct ConsolidatingRetainPipeline {
    retain: Arc<dyn RetainPipeline>,
    coordinator: Arc<ConsolidationCoordinator>,
}

#[async_trait]
impl RetainPipeline for ConsolidatingRetainPipeline {
    async fn retain(&self, input: &RetainInput) -> Result<RetainOutput> {
        let output = self.retain.retain(input).await?;
        let decision = self
            .coordinator
            .maybe_schedule(input.bank_id, output.facts_stored);
        debug!(%input.bank_id, ?decision, "evaluated server consolidation after retain");
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::consolidation::ConsolidationProgress;
    use crate::error::Result;
    use crate::storage::mock::MockMemoryStore;
    use crate::types::{
        BankId, ConsolidationReport, Disposition, Fact, FactId, FactType, MemoryBank,
        OpinionMergeReport,
    };
    use async_trait::async_trait;
    use chrono::Utc;
    use tokio::sync::Notify;
    use tokio::time::{Duration as TokioDuration, timeout};

    struct CountingConsolidator {
        calls: Arc<AtomicUsize>,
        started: Arc<Notify>,
        release: Option<Arc<Notify>>,
    }

    #[async_trait]
    impl Consolidator for CountingConsolidator {
        async fn consolidate_with_progress(
            &self,
            _bank_id: BankId,
            _progress: Option<tokio::sync::mpsc::UnboundedSender<ConsolidationProgress>>,
        ) -> Result<ConsolidationReport> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_waiters();
            if let Some(release) = &self.release {
                release.notified().await;
            }
            Ok(ConsolidationReport::default())
        }
    }

    struct CountingOpinionMerger {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl OpinionMerger for CountingOpinionMerger {
        async fn merge(&self, _bank_id: BankId) -> Result<OpinionMergeReport> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(OpinionMergeReport::default())
        }
    }

    struct InsertingRetainPipeline {
        store: Arc<MockMemoryStore>,
    }

    #[async_trait]
    impl RetainPipeline for InsertingRetainPipeline {
        async fn retain(&self, input: &RetainInput) -> Result<RetainOutput> {
            let fact = Fact {
                id: FactId::new(),
                bank_id: input.bank_id,
                content: input.content.clone(),
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
            };
            let ids = self.store.insert_facts(&[fact]).await?;
            Ok(RetainOutput {
                fact_ids: ids,
                facts_stored: 1,
                new_entities: vec![],
                entities_resolved: 0,
                links_created: 0,
                opinions_reinforced: 0,
                opinions_weakened: 0,
            })
        }
    }

    async fn make_bank(store: &Arc<MockMemoryStore>) -> BankId {
        let bank = MemoryBank {
            id: BankId::new(),
            name: "test".into(),
            mission: "test".into(),
            directives: vec![],
            disposition: Disposition::default(),
            embedding_model: String::new(),
            embedding_dimensions: 0,
        };
        store
            .create_bank(&bank)
            .await
            .expect("bank should be created");
        bank.id
    }

    async fn insert_unconsolidated_fact(store: &Arc<MockMemoryStore>, bank_id: BankId) {
        let fact = Fact {
            id: FactId::new(),
            bank_id,
            content: "hello".into(),
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
        };
        store
            .insert_facts(&[fact])
            .await
            .expect("fact should be inserted");
    }

    #[test]
    fn policy_rejects_zero_threshold() {
        let err = ConsolidationPolicy::new(true, 0, Duration::from_secs(30), false)
            .expect_err("zero threshold should be rejected");
        assert!(err.to_string().contains("MIN_FACTS"));
    }

    #[tokio::test]
    async fn coordinator_dedupes_running_bank_and_respects_cooldown() {
        let store = Arc::new(MockMemoryStore::new());
        let bank_id = make_bank(&store).await;
        insert_unconsolidated_fact(&store, bank_id).await;

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let coordinator = Arc::new(ConsolidationCoordinator::new(
            store.clone(),
            Arc::new(CountingConsolidator {
                calls: calls.clone(),
                started: started.clone(),
                release: Some(release.clone()),
            }),
            Arc::new(CountingOpinionMerger {
                calls: Arc::new(AtomicUsize::new(0)),
            }),
            ConsolidationPolicy::new(true, 1, Duration::from_secs(60), false)
                .expect("policy should be valid"),
        ));

        assert_eq!(
            coordinator.maybe_schedule(bank_id, 1),
            ConsolidationDecision::Enqueued
        );
        started.notified().await;
        assert_eq!(
            coordinator.maybe_schedule(bank_id, 1),
            ConsolidationDecision::AlreadyRunning
        );

        release.notify_waiters();
        timeout(TokioDuration::from_secs(1), async {
            loop {
                if calls.load(Ordering::SeqCst) == 1 {
                    let states = coordinator.lock_states();
                    if states
                        .get(&bank_id)
                        .and_then(|state| state.last_evaluated_at)
                        .is_some()
                    {
                        break;
                    }
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("consolidation should finish");

        assert_eq!(
            coordinator.maybe_schedule(bank_id, 1),
            ConsolidationDecision::Cooldown
        );
    }

    #[tokio::test]
    async fn retain_wrapper_schedules_consolidation() {
        let store = Arc::new(MockMemoryStore::new());
        let bank_id = make_bank(&store).await;

        let started = Arc::new(Notify::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let opinion_calls = Arc::new(AtomicUsize::new(0));

        let wrapped = wrap_retain_pipeline_with_consolidation(
            Arc::new(InsertingRetainPipeline {
                store: store.clone(),
            }),
            store.clone(),
            Arc::new(CountingConsolidator {
                calls: calls.clone(),
                started: started.clone(),
                release: None,
            }),
            Arc::new(CountingOpinionMerger {
                calls: opinion_calls.clone(),
            }),
            ConsolidationPolicy::new(true, 1, Duration::from_secs(60), false)
                .expect("policy should be valid"),
        )
        .expect("wrapper should be created");

        let output = wrapped
            .retain(&RetainInput {
                bank_id,
                content: "new fact".into(),
                timestamp: Utc::now(),
                turn_id: None,
                context: None,
                custom_instructions: None,
                speaker: None,
            })
            .await
            .expect("retain should succeed");

        assert_eq!(output.facts_stored, 1);
        timeout(TokioDuration::from_secs(1), started.notified())
            .await
            .expect("consolidation should have been scheduled");
        timeout(TokioDuration::from_secs(1), async {
            loop {
                if calls.load(Ordering::SeqCst) > 0 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("consolidation call should be observed");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(opinion_calls.load(Ordering::SeqCst), 0);
    }
}
