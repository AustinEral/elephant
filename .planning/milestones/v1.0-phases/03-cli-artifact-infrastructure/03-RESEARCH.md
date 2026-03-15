# Phase 3: CLI and Artifact Infrastructure - Research

**Researched:** 2026-03-15
**Domain:** Rust CLI harness, benchmark artifact system, config resolution
**Confidence:** HIGH

## Summary

This phase builds the `longmemeval-bench` binary with subcommands (run/ingest/qa), a profile system, config overlay, and three-artifact output. The project already has a mature reference implementation in `bench/locomo/locomo.rs` (~3500 lines) that establishes every pattern needed. The LongMemEval harness is a structural adaptation of LoCoMo, not a greenfield design.

The key differences from LoCoMo are: (1) per-question bank isolation instead of per-conversation, (2) 7 question categories instead of 5, (3) three profiles (smoke/full-s/full-m) with dataset-path selection, (4) simplified concurrency (single `instance_jobs` axis instead of conversation_jobs + question_jobs), and (5) no merge subcommand in v1.

**Primary recommendation:** Follow the LoCoMo harness structure verbatim -- manual arg parsing, profile JSON files, layered config resolution (profile -> file -> CLI), JSONL incremental flush, output-path-safety checks. Adapt struct names and fields for LongMemEval's domain (instances instead of conversations, question_id instead of sample_id, 7 categories instead of 5).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Per-question bank isolation: each LongMemEval question gets its own bank (benchmark protocol, not optional)
- Summary artifact structure: overall accuracy, per-category breakdown (7 types), banks HashMap (question_id -> bank_id), manifest, stage metrics, timing
- Question JSONL sidecar: one line per question -- question_id, category, score, timing, bank_id, hypothesis, ground_truth, judge reasoning
- Debug JSONL sidecar: one line per question -- reflect trace, retrieved context, tool call history
- Three profiles as JSON in bench/longmemeval/profiles/: smoke (first instance, S dataset), full-s (all 500, S dataset), full-m (all 500, M dataset)
- Config overlay knobs: dataset, output, tag, instances, session_limit, ingest_format, consolidation, instance_jobs, judge_model
- Resolution order: profile defaults -> --config JSON -> CLI flags
- Subcommands: run (full pipeline), ingest (ingest+consolidate only), qa <artifact> (score existing banks)
- Output paths: default bench/longmemeval/results/local/{tag}.json, sidecars {stem}.questions.jsonl and {stem}.debug.jsonl
- Manual arg parsing (no clap), matching LoCoMo pattern
- --instance <question_id> repeatable flag
- qa takes artifact path as positional arg

### Claude's Discretion
- Exact struct field names and types for LongMemEval-specific artifacts (BenchmarkOutput, QuestionResult, etc.)
- How run handles Phase 4 dependency -- stub QA path or only wire ingest+consolidate initially
- Profile JSON file contents (specific default values)
- Manifest struct design (adapted from LoCoMo's BenchmarkManifest)
- Error handling for missing dataset files, invalid instance IDs, etc.

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLI-01 | `longmemeval-bench run` subcommand (ingest + consolidate + QA) | LoCoMo `main()` pattern: parse_args -> resolve_fresh_config -> build_runtime -> run pipeline. QA stub for Phase 3. |
| CLI-02 | `ingest` subcommand (ingest + consolidate only, no QA) | LoCoMo BenchCommand::Ingest pattern: same pipeline but skip QA scoring, write summary with bank mappings only. |
| CLI-03 | `qa` subcommand (score against existing banks from ingest artifact) | LoCoMo resolve_qa_config pattern: load artifact, extract config, validate restricted overrides, check bank_ids exist. |
| CLI-04 | Profile system: smoke, full-s, full-m | LoCoMo RunProfile enum + config_path() -> JSON file loading pattern. Three profile files in bench/longmemeval/profiles/. |
| CLI-06 | --config JSON overlay on top of profile | LoCoMo FileRunConfig.apply() pattern: load JSON -> apply over profile defaults -> CLI flags override. |
| CLI-07 | --instance flag to run specific question instances (repeatable) | LoCoMo --conversation pattern: Vec<String> accumulation in CliOverrides, filter dataset after loading. |
| CLI-08 | Three-artifact output: summary JSON, question JSONL, debug JSONL | LoCoMo pattern: sidecar_path() for JSONL paths, append_jsonl() for incremental writes, final summary JSON write. |
| CLI-09 | Manifest with reproducibility contract | LoCoMo BenchmarkManifest pattern: dataset fingerprint, prompt hashes, runtime config, git commit, CLI invocation string. |
| CLI-11 | Results default to bench/longmemeval/results/local/ | LoCoMo default_output_path() pattern: stem from tag or profile-command, local/ subdirectory. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| std::env::args | stdlib | CLI argument parsing | Project convention: manual parsing, no clap |
| serde / serde_json | 1.x | Config deserialization, artifact serialization | Already in Cargo.toml, used everywhere |
| chrono | 0.4.x | Timestamps for manifest and artifacts | Already in Cargo.toml, used by LoCoMo |
| tokio | 1.x | Async runtime for main() | Already in Cargo.toml, used by LoCoMo |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| bench/common/io | internal | `append_jsonl()`, `sidecar_path()` | All JSONL sidecar writes |
| bench/common/fingerprint | internal | `fnv1a64()`, `fnv1a64_hex()` | Dataset fingerprint, prompt hashes |
| bench/longmemeval/ingest | internal | `ingest_instance()`, `IngestConfig`, `IngestResult` | Run and ingest subcommands |
| bench/longmemeval/dataset | internal | `load_dataset()`, `LongMemEvalInstance`, `QuestionType` | Dataset loading and filtering |
| elephant::runtime | internal | `build_runtime_from_env()`, `ElephantRuntime` | Runtime construction |
| elephant::metrics | internal | `MetricsCollector`, `LlmStage`, `StageUsage` | Stage-level usage tracking |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual arg parsing | clap | Project convention is manual parsing; LoCoMo uses it; adding clap would be inconsistent |

**Installation:**
No new dependencies needed. Everything is already in Cargo.toml.

## Architecture Patterns

### Recommended Project Structure
```
bench/longmemeval/
  longmemeval.rs       # Binary entry point: main(), CLI parsing, pipeline orchestration
  dataset.rs           # Dataset types and loading (ALREADY BUILT)
  ingest.rs            # Ingestion pipeline (ALREADY BUILT)
  profiles/
    smoke.json         # First instance only, S dataset
    full-s.json        # All 500, S dataset
    full-m.json        # All 500, M dataset
bench/longmemeval/results/
  local/               # Default output directory (gitignored)
bench/common/
  mod.rs               # Shared utilities (ALREADY BUILT)
  io.rs                # append_jsonl, sidecar_path (ALREADY BUILT)
  fingerprint.rs       # fnv1a64, fnv1a64_hex (ALREADY BUILT)
```

### Pattern 1: Layered Config Resolution
**What:** Three-tier config: profile JSON defaults -> `--config` JSON overlay -> CLI flags
**When to use:** All subcommands (run/ingest use full resolution; qa uses artifact-based resolution)
**Example:**
```rust
// Follows LoCoMo's resolve_fresh_config pattern exactly
fn resolve_fresh_config(overrides: CliOverrides) -> Result<RunConfig, String> {
    let profile = overrides.profile.unwrap_or_default();
    let mut config = RunConfig {
        profile,
        ..RunConfig::default()
    };
    // Layer 1: profile defaults
    load_json_config(&profile.config_path())?.apply(&mut config);
    // Layer 2: --config JSON overlay
    if let Some(path) = overrides.config_path.clone() {
        load_json_config(&path)?.apply(&mut config);
        config.config_path = Some(path);
    }
    // Layer 3: CLI flags
    overrides.apply(&mut config);
    Ok(config)
}
```

### Pattern 2: QA Config from Artifact
**What:** QA mode reconstructs config from a previous artifact's manifest, restricting which CLI flags are allowed
**When to use:** `qa` subcommand
**Example:**
```rust
fn resolve_qa_config(artifact_path: &Path, overrides: CliOverrides) -> Result<RunConfig, String> {
    validate_qa_overrides(&overrides)?;  // Reject --profile, --config, --dataset, --session-limit, etc.
    let artifact = load_benchmark_output(artifact_path)?;
    let mut config = run_config_from_artifact(&artifact)?;
    // Only allow: --out, --tag, --instance, --instance-jobs, --judge-model, --force
    if let Some(output) = overrides.output { config.output = Some(output); }
    if let Some(tag) = overrides.tag { config.tag = Some(tag); }
    if !overrides.instances.is_empty() { config.instances = overrides.instances; }
    if let Some(jobs) = overrides.instance_jobs { config.instance_jobs = jobs; }
    if let Some(judge_model) = overrides.judge_model { config.judge_model = Some(judge_model); }
    Ok(config)
}
```

### Pattern 3: Incremental JSONL Flush
**What:** Write question/debug sidecar records as each question completes, not all at end
**When to use:** Question JSONL and debug JSONL sidecars
**Example:**
```rust
// From LoCoMo: append_jsonl writes one line per call, creating file if needed
use common::io::{append_jsonl, sidecar_path};

let questions_path = sidecar_path(&output_path, "questions");
let debug_path = sidecar_path(&output_path, "debug");
// Truncate sidecars at start
let _ = fs::write(&questions_path, "");
let _ = fs::write(&debug_path, "");
// After each question completes:
append_jsonl(&questions_path, &question_result);
append_jsonl(&debug_path, &debug_record);
```

### Pattern 4: Output Path Safety
**What:** Refuse to overwrite existing artifacts without `--force`
**When to use:** Before any writes in main()
**Example:**
```rust
// Follows LoCoMo ensure_output_paths_are_safe pattern
fn ensure_output_paths_are_safe(
    command: BenchCommand,
    output_path: &Path,
    artifact_path: Option<&Path>,
    allow_overwrite: bool,
) -> Result<(), String> {
    if allow_overwrite { return Ok(()); }
    // qa: refuse to overwrite source artifact
    // run/ingest: refuse to overwrite any existing output + sidecars
    let questions_path = sidecar_path(output_path, "questions");
    let debug_path = sidecar_path(output_path, "debug");
    let existing: Vec<_> = [output_path, &questions_path, &debug_path]
        .into_iter().filter(|p| p.exists()).collect();
    if !existing.is_empty() {
        return Err(format!("refusing to overwrite: {:?}. Use --tag/--out or --force", existing));
    }
    Ok(())
}
```

### Pattern 5: Profile Enum with Dataset Path Resolution
**What:** Each profile maps to a JSON file and a default dataset path
**When to use:** Profile selection and dataset resolution
```rust
enum RunProfile {
    Smoke,
    FullS,
    FullM,
}

impl RunProfile {
    fn config_path(self) -> PathBuf {
        PathBuf::from(format!("bench/longmemeval/profiles/{}.json", self.as_str()))
    }
    fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::FullS => "full-s",
            Self::FullM => "full-m",
        }
    }
}
```

### Anti-Patterns to Avoid
- **Using clap or structopt:** Project convention is manual arg parsing. Both LoCoMo and LongMemEval binaries must follow the same style.
- **Putting all types in dataset.rs or ingest.rs:** Keep CLI types (BenchCommand, RunConfig, CliOverrides, ParsedCli) and artifact types (BenchmarkOutput, QuestionResult, BenchmarkManifest) in longmemeval.rs alongside the orchestration code, matching LoCoMo's single-file pattern.
- **Making QA work end-to-end in Phase 3:** QA scoring requires the judge LLM (Phase 4). Phase 3 should wire the pipeline skeleton and either stub the QA path or only complete ingest+consolidate.
- **Separate modules for CLI parsing:** LoCoMo keeps everything in one file. Follow that pattern for consistency.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSONL append | Custom file writer | `common::io::append_jsonl()` | Already handles create + append + error swallowing |
| Sidecar paths | String manipulation | `common::io::sidecar_path()` | Consistent {stem}.{suffix}.jsonl naming |
| Dataset fingerprint | Custom hashing | `common::fingerprint::fnv1a64()` | Already used by dataset.rs, deterministic |
| Prompt hashing | Custom hash | `common::fingerprint::fnv1a64_hex()` | Matches LoCoMo's BenchmarkPromptHashes pattern |
| Git commit/dirty state | Custom git ops | Port `git_commit_sha()` / `git_dirty_worktree()` from LoCoMo | Handles missing git, non-zero exit, etc. |
| Runtime construction | Manual wiring | `elephant::runtime::build_runtime_from_env()` | Handles all env-based config, provider setup, etc. |
| Output directory creation | Manual mkdir -p | `fs::create_dir_all(parent).ok()` | LoCoMo's pattern, creates parents, ignores already-exists |

## Common Pitfalls

### Pitfall 1: Forgetting to Truncate JSONL Sidecars at Start
**What goes wrong:** If you don't truncate the sidecar files before writing, a re-run with `--force` appends to old data, producing corrupt artifacts.
**Why it happens:** `append_jsonl()` appends by design. The truncation must happen at the top of main() after resolving output paths.
**How to avoid:** `let _ = fs::write(&questions_path, ""); let _ = fs::write(&debug_path, "");` right after creating parent dirs, matching LoCoMo lines 3448-3449.
**Warning signs:** JSONL files with more lines than expected, duplicate question_ids in sidecar.

### Pitfall 2: QA Override Validation
**What goes wrong:** Allowing `--profile`, `--config`, `--dataset`, `--consolidation` in qa mode lets the user create a config that diverges from the artifact's actual ingestion settings.
**Why it happens:** If you only use the generic `overrides.apply()` without validating first, all flags work in all modes.
**How to avoid:** `validate_qa_overrides()` must reject ingestion-related flags with a clear error message. Only allow --out, --tag, --instance, --instance-jobs, --judge-model, --force.
**Warning signs:** qa runs with different consolidation or dataset than what the banks were built from.

### Pitfall 3: Missing Banks in QA Mode
**What goes wrong:** QA is requested for instances whose bank_id is not in the artifact's banks map.
**Why it happens:** User filters `--instance` to IDs that weren't in the original ingest run.
**How to avoid:** After loading dataset and filtering instances, check that every selected instance has a bank_id in the artifact. Fail early with a clear message listing missing IDs. LoCoMo does this at lines 3390-3403.
**Warning signs:** Runtime errors about bank not found during reflect.

### Pitfall 4: Profile Default Dataset Paths
**What goes wrong:** Profile defaults to a dataset path, but the user hasn't downloaded that dataset file.
**Why it happens:** smoke defaults to S dataset, full-m defaults to M dataset -- different files.
**How to avoid:** Check `config.dataset.exists()` early in main() with a helpful error message including download instructions (matching the pattern in `dataset::load_dataset()`).
**Warning signs:** Confusing file-not-found error deep in the pipeline.

### Pitfall 5: Artifact Serde Compatibility
**What goes wrong:** Adding fields to BenchmarkOutput or BenchmarkManifest breaks deserialization of existing artifacts.
**Why it happens:** New fields without `#[serde(default)]` cause deserialization failures.
**How to avoid:** Use `#[serde(default)]` and `#[serde(skip_serializing_if = "...")]` on all fields that might be absent in older artifacts. Match LoCoMo's pattern of generous `#[serde(default)]`.
**Warning signs:** qa mode failing to load an artifact that was written by an older version.

### Pitfall 6: Default Output Path Collision Between Commands
**What goes wrong:** `run` and `ingest` with the same profile produce the same default output path.
**Why it happens:** If the default filename doesn't include the command name.
**How to avoid:** Default path format: `{profile}-{command}.json` (e.g., `smoke-run.json`, `smoke-ingest.json`), matching LoCoMo's `format!("{}-{}", config.profile.as_str(), command.as_str())`.

## Code Examples

### Example 1: LongMemEval BenchCommand Enum
```rust
// Adapted from LoCoMo -- no Merge in LongMemEval v1
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchCommand {
    Run,
    Ingest,
    Qa,
}

impl BenchCommand {
    fn as_str(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Ingest => "ingest",
            Self::Qa => "qa",
        }
    }
}
```

### Example 2: FileRunConfig (Profile JSON Shape)
```rust
// JSON overlay config -- all fields optional, applied over profile defaults
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct FileRunConfig {
    #[serde(default)]
    dataset: Option<PathBuf>,
    #[serde(default)]
    output: Option<PathBuf>,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    instances: Vec<String>,       // LongMemEval: question_ids (was: conversations)
    #[serde(default)]
    session_limit: Option<usize>,
    #[serde(default)]
    ingest_format: Option<IngestFormat>,  // text/json (was: ingest turn/session/raw-json)
    #[serde(default)]
    consolidation: Option<ConsolidationMode>,
    #[serde(default)]
    instance_jobs: Option<usize>,  // Single axis (was: conversation_jobs + question_jobs)
    #[serde(default)]
    judge_model: Option<String>,
}
```

### Example 3: Profile JSON Files
```json
// bench/longmemeval/profiles/smoke.json
{
  "dataset": "data/longmemeval_s_cleaned.json",
  "instances": [],
  "session_limit": null,
  "ingest_format": "text",
  "consolidation": "end",
  "instance_jobs": 1
}
// Note: smoke.json should select first instance only.
// Options: (a) hardcode in smoke.json instances list,
//          (b) add an instance_limit field.
// Recommend: instance_limit: 1 in smoke profile, like LoCoMo's question_limit.
```

```json
// bench/longmemeval/profiles/full-s.json
{
  "dataset": "data/longmemeval_s_cleaned.json",
  "ingest_format": "text",
  "consolidation": "end",
  "instance_jobs": 1
}
```

```json
// bench/longmemeval/profiles/full-m.json
{
  "dataset": "data/longmemeval_m_cleaned.json",
  "ingest_format": "text",
  "consolidation": "end",
  "instance_jobs": 1
}
```

### Example 4: BenchmarkOutput for LongMemEval
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkOutput {
    benchmark: String,               // "longmemeval"
    timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    tag: Option<String>,
    retain_model: String,
    reflect_model: String,
    embedding_model: String,
    reranker_model: String,
    #[serde(default)]
    judge_model: String,             // Empty until Phase 4
    consolidation_strategy: String,
    total_questions: usize,
    #[serde(default)]
    accuracy: f64,                   // 0.0 until Phase 4
    per_category: HashMap<String, CategoryResult>,  // 7 categories
    /// Maps question_id -> bank_id for resume/audit
    banks: HashMap<String, String>,
    #[serde(default)]
    manifest: BenchmarkManifest,
    #[serde(default)]
    artifacts: BenchmarkArtifacts,
    #[serde(default)]
    stage_metrics: BTreeMap<LlmStage, StageUsage>,
    total_time_s: f64,
}
```

### Example 5: Manifest for LongMemEval
```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BenchmarkManifest {
    protocol_version: String,        // "2026-03-15-longmemeval-v1"
    profile: String,
    mode: String,                    // run/ingest/qa
    #[serde(skip_serializing_if = "Option::is_none", default)]
    config_path: Option<String>,
    dataset_path: String,
    dataset_fingerprint: String,     // FNV1a-64 of raw file bytes
    command: String,                 // Full CLI invocation string
    #[serde(default)]
    selected_instances: Vec<String>, // Which question_ids were selected
    ingest_format: String,           // text/json
    instance_concurrency: usize,
    consolidation_strategy: String,
    session_limit: Option<usize>,
    dirty_worktree: Option<bool>,
    #[serde(default)]
    prompt_hashes: BenchmarkPromptHashes,
    #[serde(default)]
    runtime_config: BenchmarkRuntimeConfig,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    source_artifact: Option<SourceArtifact>,
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| LoCoMo: per-conversation banks | LongMemEval: per-question banks | N/A (different benchmark) | 500 banks instead of 10, bank map is larger |
| LoCoMo: 5 categories (1-5 numeric) | LongMemEval: 7 categories (kebab-case strings) | N/A (different benchmark) | per_category keys are strings matching QuestionType variants |
| LoCoMo: conversation_jobs + question_jobs | LongMemEval: single instance_jobs | Design decision | Simpler concurrency model since each "instance" is one question |
| LoCoMo: resume via --resume flag | LongMemEval: resume via qa subcommand | Design decision | Cleaner separation: ingest writes artifact, qa reads it back |

## Open Questions

1. **Smoke profile instance selection**
   - What we know: smoke should run "first instance only" on S dataset
   - What's unclear: Whether to hardcode a specific question_id in smoke.json `instances` list, or add an `instance_limit: 1` field (like LoCoMo's `question_limit`)
   - Recommendation: Use an `instance_limit` field in profiles/config. More flexible than hardcoding an ID that could change if the dataset changes. Profile JSON sets `"instance_limit": 1`.

2. **QA stub in Phase 3**
   - What we know: Phase 4 builds the judge/evaluation. Phase 3 builds the pipeline skeleton.
   - What's unclear: Should `run` fail when QA would execute, or silently skip it?
   - Recommendation: `run` should execute ingest+consolidate and write the summary with banks but 0.0 accuracy and empty per_category. Print a message "QA scoring not yet implemented (Phase 4)". This way `run` produces a valid artifact that `qa` can consume later.

3. **stage_metrics collection**
   - What we know: IngestResult currently returns empty stage_metrics (noted in STATE.md: "stage_metrics left as empty BTreeMap -- scoped collector wiring deferred to Phase 3")
   - What's unclear: Whether to wire MetricsCollector per-instance in this phase
   - Recommendation: Wire the MetricsCollector in the per-instance pipeline so ingest artifacts include real stage metrics from day one. The runtime already supports it via BuildRuntimeOptions.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (built-in) |
| Config file | Cargo.toml [[bin]] section |
| Quick run command | `cargo test --bin longmemeval-bench` |
| Full suite command | `cargo test --bin longmemeval-bench && cargo test --lib` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLI-01 | `run` subcommand parsed correctly | unit | `cargo test --bin longmemeval-bench -- run_subcommand` | No -- Wave 0 |
| CLI-02 | `ingest` subcommand parsed correctly | unit | `cargo test --bin longmemeval-bench -- ingest_subcommand` | No -- Wave 0 |
| CLI-03 | `qa` subcommand parsed correctly, artifact path required | unit | `cargo test --bin longmemeval-bench -- qa_subcommand` | No -- Wave 0 |
| CLI-04 | Profile selection loads correct defaults | unit | `cargo test --bin longmemeval-bench -- profile_loads` | No -- Wave 0 |
| CLI-06 | Config overlay applies over profile | unit | `cargo test --bin longmemeval-bench -- config_overlay` | No -- Wave 0 |
| CLI-07 | --instance flag accumulates correctly | unit | `cargo test --bin longmemeval-bench -- instance_flag` | No -- Wave 0 |
| CLI-08 | Sidecar paths computed correctly | unit | `cargo test --bin longmemeval-bench -- sidecar_paths` | Partial -- common/io.rs has tests |
| CLI-09 | Manifest populated with all fields | unit | `cargo test --bin longmemeval-bench -- manifest` | No -- Wave 0 |
| CLI-11 | Default output path resolution | unit | `cargo test --bin longmemeval-bench -- default_output` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --bin longmemeval-bench`
- **Per wave merge:** `cargo test --bin longmemeval-bench && cargo test --lib`
- **Phase gate:** Full suite green before verify-work

### Wave 0 Gaps
- [ ] CLI parsing tests for all subcommands (parse_args_from unit tests)
- [ ] Profile loading tests (smoke, full-s, full-m defaults)
- [ ] Config resolution tests (layered apply)
- [ ] Default output path tests
- [ ] QA override validation tests
- [ ] Output path safety tests
- [ ] Profile JSON files: `bench/longmemeval/profiles/smoke.json`, `full-s.json`, `full-m.json`
- [ ] Results directory: `bench/longmemeval/results/local/.gitkeep`

## Sources

### Primary (HIGH confidence)
- `bench/locomo/locomo.rs` -- Full reference implementation: CLI parsing (lines 1506-1666), config resolution (1367-1424), output paths (3216-3248), manifest (3454-3480), safety checks (3165-3214), tests (3658-4160+)
- `bench/locomo/profiles/smoke.json` and `full.json` -- Profile JSON structure
- `bench/longmemeval/ingest.rs` -- Existing ingestion types: IngestResult, IngestStats, IngestTiming, IngestConfig, ConsolidationMode, IngestFormat
- `bench/longmemeval/dataset.rs` -- Existing dataset types: LongMemEvalInstance, QuestionType, load_dataset(), validate_dataset()
- `bench/common/io.rs` -- append_jsonl(), sidecar_path()
- `bench/common/fingerprint.rs` -- fnv1a64(), fnv1a64_hex()
- `03-CONTEXT.md` -- All user decisions and constraints

### Secondary (MEDIUM confidence)
- None needed -- this is an internal pattern replication, not a library research task

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- No new dependencies, all patterns from existing LoCoMo harness
- Architecture: HIGH -- Direct structural adaptation of a working 3500-line reference implementation
- Pitfalls: HIGH -- All pitfalls derived from actual LoCoMo code patterns and known project history

**Research date:** 2026-03-15
**Valid until:** Indefinite (internal codebase patterns, not external libraries)
