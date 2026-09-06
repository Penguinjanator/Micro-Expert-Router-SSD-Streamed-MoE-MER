//! ORACLE-0B-S qualification: perfect-future source scheduling on the current
//! single-queue GPU-native architecture.
//!
//! Source work for position P+1 begins only after position P's ordinary queue
//! submission. Destructive physical replacement is allowed only after the
//! token loop supplies a non-cloneable boundary witness following the normal
//! map, `device.poll(Maintain::Wait)`, report parse, and clean-status checks.
//! Queue writes performed at that boundary intentionally serialize before the
//! next token's commands on the same queue. This module never claims H2D
//! overlap and never uses the production predictor/speculative path.

use crate::backend::gpu_native::{GpuNativePhysicalInstallEvidence, GpuNativeQ4ExpertResidency};
use crate::backend::GpuDeviceIdentity;
use crate::engine::{Engine, RoutedExpertExecutionSnapshot};
use crate::expert_cache::{ExpertResident, GpuDemandSetAdmission, GpuResident};
use crate::gpu_native_oracle_routes::{OracleGeometry, OracleRouteRecord, OracleRouteTrace};
use crate::gpu_native_real_benchmark::{
    Aggregate, BenchmarkFailure, BenchmarkProvenance, PerRunResult, ProductionConfiguration,
    RequestEvidence, RequestSnapshotStart, RunTiming, RuntimeContractEvidence,
    RuntimeContractInput,
};
use crate::gpu_native_residency::{
    GpuNativeDemandExpert, GpuNativePhysicalInstallObserver, GpuNativeTieredResidencyError,
};
use crate::gpu_native_token_loop::{
    GpuNativeOracleScheduleHook, GpuNativeRecoverySnapshot, GpuNativeSafeTokenBoundary,
    GpuNativeTokenLoopSnapshot,
};
use crate::greedy_parity::{BackgroundShutdownEvidence, ModelIdentityEvidence, ModelLoadEvidence};
use clap::ValueEnum;
use futures::{stream, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

pub(crate) const SCHEMA: &str = "mer.gpu-native-oracle-scheduled-residency.v1";
pub(crate) const MODE: &str = "qualify-gpu-native-oracle-scheduled-residency";
const ORACLE_ROUTE_SCHEMA: &str = "mer.gpu-native-oracle-route-trace.v1";
const ORACLE_ROUTE_COMMAND_SHA: &str = "3576cb893586f5dd3f5c5e7355658762db02b534";
const EXPECTED_ORDERED_ROUTE_SHA256: &str =
    "f8834d1ab3b695d615ff43396c3bcb7d347116b19a1e6ae4f547bd07a0764d68";
const EXPECTED_LEGACY_ROUTE_SHA256: &str =
    "27a0d3022744202907d79317a1b0f728e759c94fec2e9f043d650590656a51e8";
const FROZEN_PROMPT: &str =
    "Write a Rust function that adds two i32 values and returns the result.";
const EXPECTED_PROMPT_SHA256: &str =
    "4c8a131a9eaca3a9526da12025dcc3507c8b43c5312840180e0148a013680413";
const EXPECTED_PROMPT_TOKEN_IDS_SHA256: &str =
    "6a4a4d916225833d9c1def33b56cdeb70b2a89b0aba6907f91d79178f3a3863b";
const EXPECTED_ADAPTER_NAME: &str = "NVIDIA L4";
const EXPECTED_POSITIONS: usize = 143;
const EXPECTED_LAYERS: usize = 48;
const EXPECTED_EXPERTS: usize = 128;
const EXPECTED_TOP_K: usize = 8;
const EXPECTED_D_MODEL: usize = 2048;
const EXPECTED_D_FF: usize = 768;
const EXPECTED_LAYER_RECORDS: usize = 6_864;
const EXPECTED_SELECTED_IDS: usize = 54_912;
const FROZEN_OUTPUT_TOKENS: usize = 128;
const FROZEN_WARMUP_RUNS: usize = 1;
const FROZEN_MEASURED_RUNS: usize = 3;
const FROZEN_RAM_CACHE_SLOTS: usize = 384;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum OracleScheduledResidencyMode {
    SourceOnly,
    TokenBoundaryDirect,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum LateSourcePolicy {
    NonblockingDemandFallback,
    AwaitResidualAtSafeBoundary,
}

const fn late_source_policy(mode: OracleScheduledResidencyMode) -> LateSourcePolicy {
    match mode {
        OracleScheduledResidencyMode::SourceOnly => LateSourcePolicy::NonblockingDemandFallback,
        OracleScheduledResidencyMode::TokenBoundaryDirect => {
            LateSourcePolicy::AwaitResidualAtSafeBoundary
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CommandArgs {
    pub(crate) config: PathBuf,
    pub(crate) oracle_trace: PathBuf,
    pub(crate) treatment_mode: OracleScheduledResidencyMode,
    pub(crate) oracle_source_concurrency: usize,
    pub(crate) report_out: PathBuf,
    pub(crate) progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

#[derive(Clone, Debug, Serialize)]
struct SourceOracleEvidence {
    configured_path: String,
    canonical_path: String,
    byte_length: u64,
    artifact_sha256: String,
    schema: String,
    complete: bool,
    diagnostic_only: bool,
    route_command_git_sha: String,
    ordered_route_sequence_sha256: String,
    legacy_selected_route_ids_sha256: String,
    expected_generated_token_ids_sha256: String,
    expected_generated_text_sha256: String,
}

#[derive(Clone)]
struct ValidatedOracleArtifact {
    evidence: SourceOracleEvidence,
    trace: Arc<OracleRouteTrace>,
}

fn artifact_failure(code: &str, detail: impl Into<String>) -> BenchmarkFailure {
    BenchmarkFailure::new("preflight", code, detail)
}

fn required_value<'a>(
    root: &'a serde_json::Value,
    pointer: &str,
) -> Result<&'a serde_json::Value, BenchmarkFailure> {
    root.pointer(pointer).ok_or_else(|| {
        artifact_failure(
            "oracle-artifact-field-missing",
            format!("ORACLE-0A artifact is missing {pointer}"),
        )
    })
}

fn required_string(root: &serde_json::Value, pointer: &str) -> Result<String, BenchmarkFailure> {
    required_value(root, pointer)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| {
            artifact_failure(
                "oracle-artifact-field-type",
                format!("ORACLE-0A artifact field {pointer} is not a string"),
            )
        })
}

fn required_bool(root: &serde_json::Value, pointer: &str) -> Result<bool, BenchmarkFailure> {
    required_value(root, pointer)?.as_bool().ok_or_else(|| {
        artifact_failure(
            "oracle-artifact-field-type",
            format!("ORACLE-0A artifact field {pointer} is not a bool"),
        )
    })
}

fn required_u64(root: &serde_json::Value, pointer: &str) -> Result<u64, BenchmarkFailure> {
    required_value(root, pointer)?.as_u64().ok_or_else(|| {
        artifact_failure(
            "oracle-artifact-field-type",
            format!("ORACLE-0A artifact field {pointer} is not an unsigned integer"),
        )
    })
}

fn load_oracle_artifact(path: &Path) -> Result<ValidatedOracleArtifact, BenchmarkFailure> {
    // One immutable byte snapshot is hashed and deserialized. The qualifier
    // never reopens the source path after this read.
    let bytes = std::fs::read(path).map_err(|error| {
        artifact_failure(
            "oracle-artifact-read-failed",
            format!("failed to read {}: {error}", path.display()),
        )
    })?;
    let artifact_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let root: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| artifact_failure("oracle-artifact-parse-failed", error.to_string()))?;
    let schema = required_string(&root, "/schema")?;
    let complete = required_bool(&root, "/complete")?;
    let diagnostic_only = required_bool(&root, "/diagnostic_only")?;
    let route_command_git_sha = required_string(&root, "/provenance/build/git_sha")?;
    let prompt_sha256 = required_string(&root, "/request/prompt_sha256")?;
    let prompt_token_ids_sha256 = required_string(&root, "/request/prompt_token_ids_sha256")?;
    let requested_output_tokens = required_u64(&root, "/request/requested_output_tokens")?;
    let greedy = required_bool(&root, "/request/greedy")?;
    let expected_generated_token_ids_sha256 =
        required_string(&root, "/measured/generated_token_ids_sha256")?;
    let expected_generated_text_sha256 = required_string(&root, "/measured/generated_text_sha256")?;
    let warmup_token_sha = required_string(&root, "/warmup/generated_token_ids_sha256")?;
    let warmup_text_sha = required_string(&root, "/warmup/generated_text_sha256")?;
    let trace_value = required_value(&root, "/trace")?.clone();
    let trace: OracleRouteTrace = serde_json::from_value(trace_value)
        .map_err(|error| artifact_failure("oracle-trace-parse-failed", error.to_string()))?;
    trace
        .validate()
        .map_err(|error| artifact_failure("oracle-trace-invalid", error.to_string()))?;

    let expected_geometry = OracleGeometry {
        layers: EXPECTED_LAYERS,
        experts: EXPECTED_EXPERTS,
        top_k: EXPECTED_TOP_K,
        d_model: EXPECTED_D_MODEL,
        d_ff: EXPECTED_D_FF,
    };
    let source_model_identity_matches = required_string(&root, "/model_identity/architecture")?
        == "qwen3_moe"
        && required_u64(&root, "/model_identity/num_layers")? == EXPECTED_LAYERS as u64
        && required_u64(&root, "/model_identity/num_experts_per_layer")? == EXPECTED_EXPERTS as u64
        && required_u64(&root, "/model_identity/total_experts")? == 6_144
        && required_u64(&root, "/model_identity/top_k")? == EXPECTED_TOP_K as u64
        && required_u64(&root, "/model_identity/d_model")? == EXPECTED_D_MODEL as u64
        && required_u64(&root, "/model_identity/d_ff")? == EXPECTED_D_FF as u64
        && required_string(&root, "/model_identity/routed_expert_dtype")? == "q4_0";
    if schema != ORACLE_ROUTE_SCHEMA
        || trace.schema != ORACLE_ROUTE_SCHEMA
        || !complete
        || !diagnostic_only
        || !source_model_identity_matches
        || route_command_git_sha != ORACLE_ROUTE_COMMAND_SHA
        || trace.geometry != expected_geometry
        || prompt_sha256 != EXPECTED_PROMPT_SHA256
        || prompt_token_ids_sha256 != EXPECTED_PROMPT_TOKEN_IDS_SHA256
        || requested_output_tokens != FROZEN_OUTPUT_TOKENS as u64
        || !greedy
        || trace.total_positions != EXPECTED_POSITIONS
        || trace.total_layer_records != EXPECTED_LAYER_RECORDS
        || trace.total_selected_expert_ids != EXPECTED_SELECTED_IDS
        || trace.ordered_route_sequence_sha256 != EXPECTED_ORDERED_ROUTE_SHA256
        || trace.legacy_selected_route_ids_sha256 != EXPECTED_LEGACY_ROUTE_SHA256
        || warmup_token_sha != expected_generated_token_ids_sha256
        || warmup_text_sha != expected_generated_text_sha256
    {
        return Err(artifact_failure(
            "oracle-artifact-contract-mismatch",
            "ORACLE-0A artifact does not match the frozen schema, provenance, geometry, workload, totals, hashes, or deterministic output contract",
        ));
    }
    let canonical_path = std::fs::canonicalize(path)
        .map_err(|error| {
            artifact_failure("oracle-artifact-canonicalize-failed", error.to_string())
        })?
        .display()
        .to_string();
    Ok(ValidatedOracleArtifact {
        evidence: SourceOracleEvidence {
            configured_path: path.display().to_string(),
            canonical_path,
            byte_length: bytes.len() as u64,
            artifact_sha256,
            schema,
            complete,
            diagnostic_only,
            route_command_git_sha,
            ordered_route_sequence_sha256: trace.ordered_route_sequence_sha256.clone(),
            legacy_selected_route_ids_sha256: trace.legacy_selected_route_ids_sha256.clone(),
            expected_generated_token_ids_sha256,
            expected_generated_text_sha256,
        },
        trace: Arc::new(trace),
    })
}

#[derive(Clone, Debug)]
struct StrictRouteCursor {
    trace: Arc<OracleRouteTrace>,
    next_position: usize,
    actual_records: Vec<OracleRouteRecord>,
}

impl StrictRouteCursor {
    fn new(trace: Arc<OracleRouteTrace>) -> Self {
        Self {
            actual_records: Vec::with_capacity(trace.total_layer_records),
            trace,
            next_position: 0,
        }
    }

    fn routes_at(&self, position: usize) -> Result<Vec<Vec<u32>>, String> {
        if position >= self.trace.total_positions {
            return Err(format!(
                "route cursor exhausted at position {position}; trace has {} positions",
                self.trace.total_positions
            ));
        }
        let start = position
            .checked_mul(self.trace.geometry.layers)
            .ok_or("route cursor index overflow")?;
        let end = start
            .checked_add(self.trace.geometry.layers)
            .ok_or("route cursor index overflow")?;
        let records = self
            .trace
            .records
            .get(start..end)
            .ok_or("route cursor cannot form a complete position")?;
        if records.len() != self.trace.geometry.layers {
            return Err("route cursor returned a truncated layer set".into());
        }
        records
            .iter()
            .enumerate()
            .map(|(layer_index, record)| {
                if record.position != position || record.layer_index != layer_index {
                    return Err(format!(
                        "route cursor order mismatch at position {position} layer {layer_index}"
                    ));
                }
                Ok(record.ordered_selected_expert_ids.clone())
            })
            .collect()
    }

    fn future_after(&self, position: usize) -> Result<Option<Vec<Vec<u32>>>, String> {
        let future = position
            .checked_add(1)
            .ok_or("future route position overflow")?;
        if future == self.trace.total_positions {
            Ok(None)
        } else if future > self.trace.total_positions {
            Err("route cursor attempted to invent a position after end-of-trace".into())
        } else {
            self.routes_at(future).map(Some)
        }
    }

    fn reconcile(&mut self, position: usize, actual: &[Vec<u32>]) -> Result<(), String> {
        if position != self.next_position {
            return Err(format!(
                "route cursor expected position {}, observed {position}",
                self.next_position
            ));
        }
        let expected = self.routes_at(position)?;
        if actual != expected {
            return Err(format!(
                "actual GPU router route differs from ORACLE-0A at position {position}"
            ));
        }
        for (layer_index, ids) in actual.iter().enumerate() {
            self.actual_records.push(OracleRouteRecord {
                position,
                layer_index,
                ordered_selected_expert_ids: ids.clone(),
            });
        }
        self.next_position += 1;
        Ok(())
    }

    fn finish(self) -> Result<OracleRouteTrace, String> {
        if self.next_position != self.trace.total_positions {
            return Err(format!(
                "route cursor stopped at {} of {} positions",
                self.next_position, self.trace.total_positions
            ));
        }
        OracleRouteTrace::try_new(self.trace.geometry, self.actual_records)
            .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
struct OracleCounters {
    initial_position_source_priming_experts: u64,
    initial_position_source_priming_us: u64,
    future_experts_considered: u64,
    source_skipped_physical_current: u64,
    source_ram_hits: u64,
    source_singleflight_hits: u64,
    source_position_inflight_skips: u64,
    source_reads_started: u64,
    source_reads_completed: u64,
    source_reads_failed: u64,
    source_bytes_read: u64,
    source_prefetch_batches_started: u64,
    source_prefetch_batches_completed: u64,
    source_prefetch_batches_skipped_inflight: u64,
    source_prefetch_peak_inflight: u64,
    source_ready_before_boundary: u64,
    source_late_at_boundary: u64,
    source_end_of_trace: u64,
    future_physical_sets_considered: u64,
    future_physical_experts_needed: u64,
    boundary_replacement_sets_started: u64,
    boundary_replacement_sets_completed: u64,
    boundary_physical_installs: u64,
    boundary_physical_install_bytes: u64,
    boundary_physical_evictions: u64,
    boundary_physical_reinstalls: u64,
    boundary_stale_generation: u64,
    boundary_install_failures: u64,
    boundary_direct_staging_writes: u64,
    boundary_full_slot_vec_materializations: u64,
    oracle_ready_at_demand: u64,
    oracle_not_ready_at_demand: u64,
    oracle_demand_fallback: u64,
    ordinary_residency_miss_attempts: u64,
    ordinary_residency_services: u64,
    source_prefetch_wall_us: u64,
    source_read_aggregate_us: u64,
    host_preparation_overlap_us: u64,
    token_boundary_host_stage_us: u64,
    token_boundary_physical_stage_us: u64,
    token_boundary_commit_us: u64,
    oracle_wait_before_next_token_us: u64,
    qualification_owned_current_slots: u64,
    qualification_owned_current_bytes: u64,
    qualification_owned_peak_slots: u64,
    qualification_owned_peak_bytes: u64,
    source_failures_degraded_to_demand: u64,
    source_late_degraded_to_demand: u64,
    background_tasks_spawned: u64,
    background_tasks_drained: u64,
}

fn saturating_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn add_counter(target: &mut u64, value: u64) {
    *target = target.saturating_add(value);
}

struct LayerBatchGuard {
    counters: Arc<Mutex<OracleCounters>>,
}

impl LayerBatchGuard {
    fn enter(counters: Arc<Mutex<OracleCounters>>) -> Self {
        let mut values = counters.lock();
        values.source_prefetch_batches_started += 1;
        let active = values
            .source_prefetch_batches_started
            .saturating_sub(values.source_prefetch_batches_completed);
        values.source_prefetch_peak_inflight = values.source_prefetch_peak_inflight.max(active);
        drop(values);
        Self { counters }
    }
}

impl Drop for LayerBatchGuard {
    fn drop(&mut self) {
        self.counters.lock().source_prefetch_batches_completed += 1;
    }
}

#[derive(Default)]
struct PrefetchOutcome {
    target_position: usize,
    residents: HashMap<u32, Arc<ExpertResident>>,
    failed_ids: Vec<u32>,
    fatal_error: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SourceReadinessSnapshot {
    required_ready: u64,
    source_residents_ready: u64,
}

#[derive(Default)]
struct SourceReadinessState {
    physical_ready: HashSet<u32>,
    source_residents_ready: HashSet<u32>,
}

#[derive(Default)]
struct SourceReadiness {
    state: Mutex<SourceReadinessState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BoundarySourceReadiness {
    Physical,
    SourceResident,
    Unavailable,
}

impl SourceReadiness {
    fn mark_physical_ready(&self, global_id: u32) {
        self.state.lock().physical_ready.insert(global_id);
    }

    fn mark_source_resident_ready(&self, global_id: u32) {
        let mut state = self.state.lock();
        state.source_residents_ready.insert(global_id);
    }

    fn validate_snapshot(
        snapshot: SourceReadinessSnapshot,
        required_experts: u64,
    ) -> Result<SourceReadinessSnapshot, String> {
        if snapshot.source_residents_ready > snapshot.required_ready
            || snapshot.required_ready > required_experts
        {
            return Err(format!(
                "future source readiness is inconsistent: required_ready={} source_residents_ready={} required_experts={required_experts}",
                snapshot.required_ready, snapshot.source_residents_ready
            ));
        }
        Ok(snapshot)
    }

    fn snapshot_at_boundary<T, F>(
        &self,
        handle: &tokio::task::JoinHandle<T>,
        required_global_ids: &[u32],
        mut current_readiness: F,
    ) -> Result<(SourceReadinessSnapshot, bool), String>
    where
        F: FnMut(u32) -> Result<BoundarySourceReadiness, String>,
    {
        // Source progress updates use this same lock. Holding it across
        // `is_finished` ties the readiness count to the exact task-finished
        // decision used at the boundary instead of rescanning a changing cache.
        let state = self.state.lock();
        let task_finished = handle.is_finished();
        let mut required_ready = 0u64;
        let mut source_residents_ready = 0u64;
        let mut unique = HashSet::with_capacity(required_global_ids.len());
        for &global_id in required_global_ids {
            if !unique.insert(global_id) {
                return Err(format!(
                    "future source readiness contains duplicate global expert {global_id}"
                ));
            }
            let live = current_readiness(global_id)?;
            let physical_ready = state.physical_ready.contains(&global_id)
                || live == BoundarySourceReadiness::Physical;
            let source_ready = state.source_residents_ready.contains(&global_id)
                || live == BoundarySourceReadiness::SourceResident;
            required_ready += u64::from(physical_ready || source_ready);
            source_residents_ready += u64::from(source_ready);
        }
        let snapshot = SourceReadinessSnapshot {
            required_ready,
            source_residents_ready,
        };
        Ok((
            Self::validate_snapshot(snapshot, required_global_ids.len() as u64)?,
            task_finished,
        ))
    }
}

struct PrefetchLease {
    outcome: PrefetchOutcome,
    counters: Arc<Mutex<OracleCounters>>,
    _position_guard: tokio::sync::OwnedMutexGuard<()>,
}

impl PrefetchLease {
    fn new(
        outcome: PrefetchOutcome,
        counters: Arc<Mutex<OracleCounters>>,
        position_guard: tokio::sync::OwnedMutexGuard<()>,
    ) -> Self {
        Self {
            outcome,
            counters,
            _position_guard: position_guard,
        }
    }
}

impl Drop for PrefetchLease {
    fn drop(&mut self) {
        let mut values = self.counters.lock();
        values.qualification_owned_current_slots = 0;
        values.qualification_owned_current_bytes = 0;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceDisposition {
    AlreadyPhysical,
    RamHit,
    SourceInFlight,
    StartSourceRead,
}

const fn source_disposition(
    physical_current: bool,
    ram_hit: bool,
    source_in_flight: bool,
) -> SourceDisposition {
    if physical_current {
        SourceDisposition::AlreadyPhysical
    } else if ram_hit {
        SourceDisposition::RamHit
    } else if source_in_flight {
        SourceDisposition::SourceInFlight
    } else {
        SourceDisposition::StartSourceRead
    }
}

async fn prefetch_layer_batch(
    engine: Arc<Engine>,
    layer_index: usize,
    local_ids: Vec<u32>,
    counters: Arc<Mutex<OracleCounters>>,
    readiness: Arc<SourceReadiness>,
) -> PrefetchOutcome {
    let _guard = LayerBatchGuard::enter(counters.clone());
    let mut outcome = PrefetchOutcome::default();
    let manager = match engine.core.gpu_native_residency.as_ref() {
        Some(manager) => manager.clone(),
        None => {
            outcome.fatal_error = Some("GPU-native residency manager is absent".into());
            return outcome;
        }
    };
    let experts_per_layer = manager.plan().geometry().num_experts() as u32;
    for local_id in local_ids {
        let global_id = match crate::gpu_native_residency::layer_local_to_global(
            layer_index,
            local_id,
            manager.plan().num_layers(),
            experts_per_layer,
        ) {
            Ok(global_id) => global_id,
            Err(error) => {
                outcome.fatal_error = Some(error.to_string());
                return outcome;
            }
        };
        let physical_current = match manager.has_current_for_demand(global_id) {
            Ok(current) => current,
            Err(error) => {
                outcome.fatal_error = Some(error.to_string());
                return outcome;
            }
        };
        let ram_resident = (!physical_current)
            .then(|| engine.core.cache.get(global_id))
            .flatten();
        let singleflight = !physical_current
            && ram_resident.is_none()
            && engine.core.in_flight.contains_key(&global_id);
        match source_disposition(physical_current, ram_resident.is_some(), singleflight) {
            SourceDisposition::AlreadyPhysical => {
                counters.lock().source_skipped_physical_current += 1;
                readiness.mark_physical_ready(global_id);
                continue;
            }
            SourceDisposition::RamHit => {
                counters.lock().source_ram_hits += 1;
                outcome.residents.insert(
                    global_id,
                    ram_resident.expect("RAM-hit disposition has a resident"),
                );
                readiness.mark_source_resident_ready(global_id);
                continue;
            }
            SourceDisposition::SourceInFlight => {
                counters.lock().source_singleflight_hits += 1;
            }
            SourceDisposition::StartSourceRead => {
                counters.lock().source_reads_started += 1;
            }
        }
        let started = Instant::now();
        match engine.fetch_with_retry(global_id).await {
            Ok(resident) => {
                let elapsed = saturating_us(started);
                let mut values = counters.lock();
                add_counter(&mut values.source_read_aggregate_us, elapsed);
                if !singleflight {
                    values.source_reads_completed += 1;
                    add_counter(&mut values.source_bytes_read, resident.data().len() as u64);
                }
                drop(values);
                readiness.mark_source_resident_ready(global_id);
                outcome.residents.insert(global_id, resident);
            }
            Err(error) => {
                if !singleflight {
                    counters.lock().source_reads_failed += 1;
                }
                outcome.failed_ids.push(global_id);
                let _ = error;
            }
        }
    }
    outcome
}

async fn prefetch_position(
    engine: Arc<Engine>,
    target_position: usize,
    routes: Vec<Vec<u32>>,
    concurrency: usize,
    expert_bytes: usize,
    counters: Arc<Mutex<OracleCounters>>,
    readiness: Arc<SourceReadiness>,
) -> PrefetchOutcome {
    let started = Instant::now();
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let batches = stream::iter(routes.into_iter().enumerate().map(|(layer_index, ids)| {
        let engine = engine.clone();
        let counters = counters.clone();
        let readiness = readiness.clone();
        let semaphore = semaphore.clone();
        async move {
            let permit = semaphore
                .acquire_owned()
                .await
                .expect("qualification semaphore remains open");
            let result = prefetch_layer_batch(engine, layer_index, ids, counters, readiness).await;
            drop(permit);
            result
        }
    }))
    .buffer_unordered(concurrency)
    .collect::<Vec<_>>()
    .await;

    let mut outcome = PrefetchOutcome {
        target_position,
        ..PrefetchOutcome::default()
    };
    for batch in batches {
        if outcome.fatal_error.is_none() {
            outcome.fatal_error = batch.fatal_error;
        }
        outcome.failed_ids.extend(batch.failed_ids);
        outcome.residents.extend(batch.residents);
    }
    let slots = outcome.residents.len() as u64;
    let bytes = outcome
        .residents
        .values()
        .map(|resident| resident.data().len() as u64)
        .sum::<u64>();
    let mut values = counters.lock();
    add_counter(&mut values.source_prefetch_wall_us, saturating_us(started));
    values.qualification_owned_current_slots = slots;
    values.qualification_owned_current_bytes = bytes;
    values.qualification_owned_peak_slots = values.qualification_owned_peak_slots.max(slots);
    values.qualification_owned_peak_bytes = values.qualification_owned_peak_bytes.max(bytes);
    if slots > FROZEN_RAM_CACHE_SLOTS as u64
        || bytes > (FROZEN_RAM_CACHE_SLOTS as u64).saturating_mul(expert_bytes as u64)
    {
        outcome.fatal_error = Some(format!(
            "qualification temporary source state exceeded bound: slots={slots} bytes={bytes}"
        ));
    }
    drop(values);
    outcome
}

struct OracleBoundaryObserver {
    counters: Arc<Mutex<OracleCounters>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BoundaryInstallErrorPolicy {
    DegradeToDemand,
    DegradeStaleGeneration,
    FailClosed,
}

fn boundary_install_error_policy(
    error: &GpuNativeTieredResidencyError,
) -> BoundaryInstallErrorPolicy {
    match error {
        GpuNativeTieredResidencyError::LogicalAdmissionStale { .. }
        | GpuNativeTieredResidencyError::StalePhysicalRequester { .. } => {
            BoundaryInstallErrorPolicy::DegradeStaleGeneration
        }
        GpuNativeTieredResidencyError::PhysicalIdentityCorrupt { .. }
        | GpuNativeTieredResidencyError::UnsafeOracleBoundary { .. } => {
            BoundaryInstallErrorPolicy::FailClosed
        }
        _ => BoundaryInstallErrorPolicy::DegradeToDemand,
    }
}

impl GpuNativePhysicalInstallObserver for OracleBoundaryObserver {
    fn record_physical_victim(&self, _global_id: u32) {
        self.counters.lock().boundary_physical_evictions += 1;
    }

    fn record_physical_install_attempt(&self) {}

    fn record_direct_staging_failure(&self) {
        self.counters.lock().boundary_install_failures += 1;
    }

    fn record_physical_install_completion(
        &self,
        _global_id: u32,
        _residency: GpuNativeQ4ExpertResidency,
        evidence: GpuNativePhysicalInstallEvidence,
        _physical_install_total_us: u64,
    ) {
        let mut values = self.counters.lock();
        values.boundary_physical_installs += 1;
        add_counter(
            &mut values.boundary_physical_install_bytes,
            evidence.physical_slot_bytes_staged,
        );
        add_counter(
            &mut values.boundary_direct_staging_writes,
            evidence.direct_staging_writes,
        );
        add_counter(
            &mut values.boundary_full_slot_vec_materializations,
            evidence.full_slot_vec_materializations,
        );
    }

    fn record_physical_install_set(
        &self,
        _width: usize,
        _parallel: bool,
        _caller_in_rayon_worker: bool,
        _rayon_threads: usize,
    ) {
    }

    fn record_reservation_attempt(&self) {}

    fn record_reservation_success(
        &self,
        _global_id: u32,
        _residency: GpuNativeQ4ExpertResidency,
        _install_ticket: u64,
    ) {
    }

    fn record_reservation_failure(&self) {
        self.counters.lock().boundary_install_failures += 1;
    }

    fn record_physical_stage_started(&self) {}

    fn record_physical_stage_completed(
        &self,
        _evidence: GpuNativePhysicalInstallEvidence,
        _individual_stage_us: u64,
    ) {
    }

    fn record_physical_stage_failed(&self) {
        self.counters.lock().boundary_install_failures += 1;
    }

    fn record_parallel_stage_wall(&self, wall_us: u64) {
        add_counter(
            &mut self.counters.lock().token_boundary_physical_stage_us,
            wall_us,
        );
    }

    fn record_ordered_commit_attempt(&self) {}

    fn record_ordered_commit_completed(&self, commit_us: u64) {
        add_counter(
            &mut self.counters.lock().token_boundary_commit_us,
            commit_us,
        );
    }

    fn record_ordered_commit_failed(&self, _violation: bool) {
        self.counters.lock().boundary_install_failures += 1;
    }

    fn record_physical_reservation_wall(&self, _wall_us: u64) {}

    fn record_physical_install_transaction_wall(&self, _wall_us: u64) {}

    fn record_unpublished_physical_writes_after_failure(&self, _count: u64) {}
}

fn future_global_ids(
    layer_index: usize,
    local_ids: &[u32],
    num_layers: usize,
    experts_per_layer: u32,
) -> Result<Vec<u32>, GpuNativeTieredResidencyError> {
    local_ids
        .iter()
        .map(|&local_id| {
            crate::gpu_native_residency::layer_local_to_global(
                layer_index,
                local_id,
                num_layers,
                experts_per_layer,
            )
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoundaryReplacementPlan {
    physical_experts_needed: usize,
    minimum_evictions: usize,
}

fn plan_boundary_replacement(
    slot_capacity: usize,
    resident_slots: usize,
    physical_current: &[bool],
) -> Result<BoundaryReplacementPlan, String> {
    if resident_slots > slot_capacity || physical_current.len() > slot_capacity {
        return Err(format!(
            "future set cannot fit exact physical capacity: residents={resident_slots} future={} capacity={slot_capacity}",
            physical_current.len()
        ));
    }
    let physical_experts_needed = physical_current.iter().filter(|&&current| !current).count();
    let free_slots = slot_capacity - resident_slots;
    Ok(BoundaryReplacementPlan {
        physical_experts_needed,
        minimum_evictions: physical_experts_needed.saturating_sub(free_slots),
    })
}

fn prepare_logical_admissions(
    engine: &Arc<Engine>,
    global_ids: &[u32],
    residents: &HashMap<u32, Arc<ExpertResident>>,
) -> Result<Vec<crate::expert_cache::GpuAdmission>, String> {
    let gpu = engine.execution_context().gpu_expert_cache().clone();
    let mut payloads = HashMap::with_capacity(global_ids.len());
    for attempt in 0..2 {
        match gpu
            .demand_admit_set(global_ids, &payloads)
            .map_err(|error| error.to_string())?
        {
            GpuDemandSetAdmission::Ready { admissions, .. } => return Ok(admissions),
            GpuDemandSetAdmission::PayloadRequired(missing) if attempt == 0 => {
                for global_id in missing {
                    let resident = residents.get(&global_id).ok_or_else(|| {
                        format!("future source for global expert {global_id} is not ready")
                    })?;
                    payloads.insert(
                        global_id,
                        Arc::new(GpuResident::new_with_dtype(
                            global_id,
                            resident.data().to_vec(),
                            engine.core.options.dtype,
                        )),
                    );
                }
            }
            GpuDemandSetAdmission::PayloadRequired(missing) => {
                return Err(format!(
                    "logical admission still requires payloads after complete preparation: {missing:?}"
                ));
            }
        }
    }
    Err("logical admission attempt bound exhausted".into())
}

fn install_future_at_boundary(
    engine: &Arc<Engine>,
    boundary: &mut GpuNativeSafeTokenBoundary,
    routes: &[Vec<u32>],
    residents: &HashMap<u32, Arc<ExpertResident>>,
    counters: Arc<Mutex<OracleCounters>>,
) -> Result<(), String> {
    let manager = engine
        .core
        .gpu_native_residency
        .as_ref()
        .cloned()
        .ok_or("GPU-native residency manager is absent")?;
    if routes.len() != manager.plan().num_layers() {
        return Err(format!(
            "future route has {} layers, expected {}",
            routes.len(),
            manager.plan().num_layers()
        ));
    }
    let before = manager.snapshot();
    let observer = OracleBoundaryObserver {
        counters: counters.clone(),
    };
    let experts_per_layer = manager.plan().geometry().num_experts() as u32;
    for (layer_index, local_ids) in routes.iter().enumerate() {
        counters.lock().future_physical_sets_considered += 1;
        let global_ids = future_global_ids(
            layer_index,
            local_ids,
            manager.plan().num_layers(),
            experts_per_layer,
        )
        .map_err(|error| error.to_string())?;
        let mut physical_current = Vec::with_capacity(global_ids.len());
        for &global_id in &global_ids {
            match manager.has_current_for_demand(global_id) {
                Ok(current) => physical_current.push(current),
                Err(error @ GpuNativeTieredResidencyError::PhysicalIdentityCorrupt { .. }) => {
                    return Err(error.to_string());
                }
                Err(error) => return Err(error.to_string()),
            }
        }
        let missing = global_ids
            .iter()
            .copied()
            .zip(physical_current.iter().copied())
            .filter_map(|(global_id, current)| (!current).then_some(global_id))
            .collect::<Vec<_>>();
        let layer_snapshot = before
            .layers
            .get(layer_index)
            .ok_or_else(|| format!("physical snapshot is missing layer {layer_index}"))?;
        let replacement_plan = plan_boundary_replacement(
            layer_snapshot.slot_capacity,
            layer_snapshot.resident_slots,
            &physical_current,
        )?;
        if replacement_plan.physical_experts_needed != missing.len() {
            return Err("ORACLE boundary replacement plan disagrees with exact currentness".into());
        }
        counters.lock().future_physical_experts_needed +=
            replacement_plan.physical_experts_needed as u64;
        if missing.is_empty() {
            continue;
        }
        if missing
            .iter()
            .any(|global_id| !residents.contains_key(global_id))
        {
            counters.lock().source_failures_degraded_to_demand += 1;
            continue;
        }

        let gpu = engine.execution_context().gpu_expert_cache().clone();
        let _logical_protection = gpu
            .protect_demand_set(&missing)
            .map_err(|error| error.to_string())?;
        let host_started = Instant::now();
        let admissions = match prepare_logical_admissions(engine, &missing, residents) {
            Ok(admissions) => admissions,
            Err(error) => {
                let mut values = counters.lock();
                add_counter(
                    &mut values.token_boundary_host_stage_us,
                    saturating_us(host_started),
                );
                values.boundary_install_failures += 1;
                if error.contains("generation") {
                    values.boundary_stale_generation += 1;
                }
                continue;
            }
        };
        add_counter(
            &mut counters.lock().token_boundary_host_stage_us,
            saturating_us(host_started),
        );
        let admissions_by_id = missing
            .iter()
            .copied()
            .zip(admissions)
            .collect::<HashMap<_, _>>();
        let demands = global_ids
            .iter()
            .copied()
            .zip(physical_current.iter().copied())
            .map(|(global_id, current)| {
                if current {
                    GpuNativeDemandExpert::current(global_id)
                } else {
                    GpuNativeDemandExpert::install(
                        global_id,
                        residents
                            .get(&global_id)
                            .expect("future source presence validated")
                            .clone(),
                        admissions_by_id
                            .get(&global_id)
                            .expect("logical admission returned every missing identity")
                            .clone(),
                    )
                }
            })
            .collect::<Vec<_>>();
        counters.lock().boundary_replacement_sets_started += 1;
        let evictions_before = counters.lock().boundary_physical_evictions;
        match manager.ensure_oracle_future_set_at_safe_boundary(
            boundary,
            layer_index,
            &demands,
            &observer,
        ) {
            Ok(_) => {
                let mut values = counters.lock();
                values.boundary_replacement_sets_completed += 1;
                let observed_evictions = values
                    .boundary_physical_evictions
                    .saturating_sub(evictions_before);
                if observed_evictions != replacement_plan.minimum_evictions as u64 {
                    return Err(format!(
                        "ORACLE boundary replacement evicted {observed_evictions} experts, expected {} for layer {layer_index}",
                        replacement_plan.minimum_evictions
                    ));
                }
            }
            Err(error) => match boundary_install_error_policy(&error) {
                BoundaryInstallErrorPolicy::DegradeStaleGeneration => {
                    let mut values = counters.lock();
                    values.boundary_stale_generation += 1;
                    values.boundary_install_failures += 1;
                }
                BoundaryInstallErrorPolicy::FailClosed => return Err(error.to_string()),
                BoundaryInstallErrorPolicy::DegradeToDemand => {
                    counters.lock().boundary_install_failures += 1;
                }
            },
        }
    }
    let after = manager.snapshot();
    if before.model_slot_capacity != after.model_slot_capacity
        || before.model_arena_allocation_bytes != after.model_arena_allocation_bytes
        || after
            .layers
            .iter()
            .any(|layer| layer.resident_slots > layer.slot_capacity)
    {
        return Err("ORACLE boundary replacement changed or exceeded physical capacity".into());
    }
    counters.lock().boundary_physical_reinstalls += after
        .physical_reinstalls
        .saturating_sub(before.physical_reinstalls);
    Ok(())
}

struct ActivePrefetch {
    target_position: usize,
    handle: Option<tokio::task::JoinHandle<PrefetchLease>>,
    readiness: Arc<SourceReadiness>,
}

enum BoundarySourceResolution {
    DeferredToDemand,
    Completed {
        lease: PrefetchLease,
        was_late: bool,
    },
}

struct SchedulerState {
    cursor: StrictRouteCursor,
    submitted_position: Option<usize>,
    active: Option<ActivePrefetch>,
    background: Vec<tokio::task::JoinHandle<()>>,
    failure: Option<String>,
}

struct OracleScheduler {
    engine: Arc<Engine>,
    mode: OracleScheduledResidencyMode,
    concurrency: usize,
    expert_bytes: usize,
    source_position_gate: Arc<tokio::sync::Mutex<()>>,
    counters: Arc<Mutex<OracleCounters>>,
    state: Arc<Mutex<SchedulerState>>,
}

async fn resolve_source_task_at_boundary<F>(
    mode: OracleScheduledResidencyMode,
    active: ActivePrefetch,
    expected_target_position: usize,
    required_global_ids: Vec<u32>,
    current_readiness: F,
    counters: Arc<Mutex<OracleCounters>>,
    state: Arc<Mutex<SchedulerState>>,
) -> Result<BoundarySourceResolution, String>
where
    F: FnMut(u32) -> Result<BoundarySourceReadiness, String> + Send,
{
    if active.target_position != expected_target_position {
        return Err(format!(
            "future source task targets {}, expected {expected_target_position}",
            active.target_position
        ));
    }
    if mode == OracleScheduledResidencyMode::TokenBoundaryDirect
        && !state.lock().background.is_empty()
    {
        return Err(
            "token-boundary-direct retained a previous-position background source task".into(),
        );
    }

    let Some(handle) = active.handle else {
        if mode == OracleScheduledResidencyMode::TokenBoundaryDirect {
            return Err(
                "token-boundary-direct skipped future source because a prior position lease remained in flight"
                    .into(),
            );
        }
        return Err("source-only position-gate skip was not resolved at the boundary".into());
    };
    let required_experts = required_global_ids.len() as u64;
    let (readiness_at_boundary, task_finished_at_boundary) = active
        .readiness
        .snapshot_at_boundary(&handle, &required_global_ids, current_readiness)?;
    let was_late = !task_finished_at_boundary;
    let late_experts = required_experts.saturating_sub(readiness_at_boundary.required_ready);

    if was_late {
        record_source_late(&mut counters.lock(), late_experts, late_source_policy(mode));
        if mode == OracleScheduledResidencyMode::SourceOnly {
            let state_for_task = state.clone();
            let counters_for_task = counters.clone();
            let discard = tokio::spawn(async move {
                let joined = handle.await;
                counters_for_task.lock().background_tasks_drained += 1;
                match joined {
                    Ok(lease) => {
                        if let Some(error) = lease.outcome.fatal_error.as_ref() {
                            state_for_task.lock().failure = Some(error.clone());
                        }
                    }
                    Err(error) => {
                        state_for_task.lock().failure =
                            Some(format!("future source task failed to join: {error}"));
                    }
                }
            });
            state.lock().background.push(discard);
            return Ok(BoundarySourceResolution::DeferredToDemand);
        }
    }

    let wait_started = Instant::now();
    let joined = handle.await;
    add_counter(
        &mut counters.lock().oracle_wait_before_next_token_us,
        saturating_us(wait_started),
    );
    counters.lock().background_tasks_drained += 1;
    let lease = joined.map_err(|error| format!("future source task failed to join: {error}"))?;
    if lease.outcome.target_position != expected_target_position {
        return Err("completed future source task returned the wrong position".into());
    }
    if let Some(error) = lease.outcome.fatal_error.as_ref() {
        return Err(error.clone());
    }
    record_completed_source_outcome(
        &mut counters.lock(),
        &lease.outcome,
        readiness_at_boundary.source_residents_ready,
    );
    Ok(BoundarySourceResolution::Completed { lease, was_late })
}

fn release_source_position_lease_before_next_token(
    mode: OracleScheduledResidencyMode,
    lease: PrefetchLease,
    source_position_gate: &Arc<tokio::sync::Mutex<()>>,
    state: &Arc<Mutex<SchedulerState>>,
) -> Result<(), String> {
    drop(lease);
    if mode == OracleScheduledResidencyMode::TokenBoundaryDirect {
        let position_guard = source_position_gate.clone().try_lock_owned().map_err(|_| {
            "token-boundary-direct retained the completed position lease before the next token"
                .to_string()
        })?;
        drop(position_guard);
        if !state.lock().background.is_empty() {
            return Err(
                "token-boundary-direct accumulated a previous-position background source task"
                    .into(),
            );
        }
    }
    Ok(())
}

impl OracleScheduler {
    fn new(
        engine: Arc<Engine>,
        trace: Arc<OracleRouteTrace>,
        mode: OracleScheduledResidencyMode,
        concurrency: usize,
        expert_bytes: usize,
    ) -> Self {
        Self {
            engine,
            mode,
            concurrency,
            expert_bytes,
            source_position_gate: Arc::new(tokio::sync::Mutex::new(())),
            counters: Arc::new(Mutex::new(OracleCounters::default())),
            state: Arc::new(Mutex::new(SchedulerState {
                cursor: StrictRouteCursor::new(trace),
                submitted_position: None,
                active: None,
                background: Vec::new(),
                failure: None,
            })),
        }
    }

    async fn prime_initial_position(&self) -> Result<(), String> {
        let routes = self.state.lock().cursor.routes_at(0)?;
        record_source_position_considered(&mut self.counters.lock(), &routes);
        let started = Instant::now();
        let position_guard = self.source_position_gate.clone().lock_owned().await;
        let readiness = Arc::new(SourceReadiness::default());
        let lease = PrefetchLease::new(
            prefetch_position(
                self.engine.clone(),
                0,
                routes,
                self.concurrency,
                self.expert_bytes,
                self.counters.clone(),
                readiness,
            )
            .await,
            self.counters.clone(),
            position_guard,
        );
        if let Some(error) = lease.outcome.fatal_error.as_ref() {
            let error = error.clone();
            return Err(error);
        }
        let mut values = self.counters.lock();
        values.initial_position_source_priming_experts = lease.outcome.residents.len() as u64;
        values.initial_position_source_priming_us = saturating_us(started);
        values.source_failures_degraded_to_demand += lease.outcome.failed_ids.len() as u64;
        drop(values);
        drop(lease);
        Ok(())
    }

    fn observe_demand_readiness(&self, position: usize) -> Result<(), String> {
        let routes = self.state.lock().cursor.routes_at(position)?;
        let manager = self
            .engine
            .core
            .gpu_native_residency
            .as_ref()
            .ok_or("GPU-native residency manager is absent")?;
        let experts_per_layer = manager.plan().geometry().num_experts() as u32;
        let mut current_by_layer = Vec::with_capacity(routes.len());
        for (layer_index, local_ids) in routes.iter().enumerate() {
            let global_ids = future_global_ids(
                layer_index,
                local_ids,
                manager.plan().num_layers(),
                experts_per_layer,
            )
            .map_err(|error| error.to_string())?;
            let mut layer_current = Vec::with_capacity(global_ids.len());
            for global_id in global_ids {
                match manager.has_current_for_demand(global_id) {
                    Ok(current) => layer_current.push(current),
                    Err(error) => return Err(error.to_string()),
                }
            }
            current_by_layer.push(layer_current);
        }
        record_demand_readiness(&mut self.counters.lock(), &current_by_layer);
        Ok(())
    }

    fn future_not_physically_current_count(&self, routes: &[Vec<u32>]) -> Result<u64, String> {
        let manager = self
            .engine
            .core
            .gpu_native_residency
            .as_ref()
            .ok_or("GPU-native residency manager is absent")?;
        let experts_per_layer = manager.plan().geometry().num_experts() as u32;
        let mut unavailable = 0u64;
        for (layer_index, local_ids) in routes.iter().enumerate() {
            for global_id in future_global_ids(
                layer_index,
                local_ids,
                manager.plan().num_layers(),
                experts_per_layer,
            )
            .map_err(|error| error.to_string())?
            {
                if !manager
                    .has_current_for_demand(global_id)
                    .map_err(|error| error.to_string())?
                {
                    unavailable += 1;
                }
            }
        }
        Ok(unavailable)
    }

    fn required_future_global_ids(&self, routes: &[Vec<u32>]) -> Result<Vec<u32>, String> {
        let manager = self
            .engine
            .core
            .gpu_native_residency
            .as_ref()
            .ok_or("GPU-native residency manager is absent")?;
        let experts_per_layer = manager.plan().geometry().num_experts() as u32;
        let mut required = Vec::with_capacity(routes.iter().map(Vec::len).sum());
        for (layer_index, local_ids) in routes.iter().enumerate() {
            required.extend(
                future_global_ids(
                    layer_index,
                    local_ids,
                    manager.plan().num_layers(),
                    experts_per_layer,
                )
                .map_err(|error| error.to_string())?,
            );
        }
        Ok(required)
    }

    fn future_source_not_ready_count(&self, routes: &[Vec<u32>]) -> Result<u64, String> {
        let manager = self
            .engine
            .core
            .gpu_native_residency
            .as_ref()
            .ok_or("GPU-native residency manager is absent")?;
        let experts_per_layer = manager.plan().geometry().num_experts() as u32;
        let mut unavailable = 0u64;
        for (layer_index, local_ids) in routes.iter().enumerate() {
            for global_id in future_global_ids(
                layer_index,
                local_ids,
                manager.plan().num_layers(),
                experts_per_layer,
            )
            .map_err(|error| error.to_string())?
            {
                if !manager
                    .has_current_for_demand(global_id)
                    .map_err(|error| error.to_string())?
                    && !self.engine.core.cache.contains(global_id)
                {
                    unavailable += 1;
                }
            }
        }
        Ok(unavailable)
    }

    fn record_position_gate_skip(&self, routes: &[Vec<u32>]) -> Result<(), String> {
        let manager = self
            .engine
            .core
            .gpu_native_residency
            .as_ref()
            .ok_or("GPU-native residency manager is absent")?;
        let experts_per_layer = manager.plan().geometry().num_experts() as u32;
        let mut physical = 0u64;
        let mut ram = 0u64;
        let mut singleflight = 0u64;
        let mut position_inflight = 0u64;
        for (layer_index, local_ids) in routes.iter().enumerate() {
            for global_id in future_global_ids(
                layer_index,
                local_ids,
                manager.plan().num_layers(),
                experts_per_layer,
            )
            .map_err(|error| error.to_string())?
            {
                let current = manager
                    .has_current_for_demand(global_id)
                    .map_err(|error| error.to_string())?;
                match source_disposition(
                    current,
                    !current && self.engine.core.cache.contains(global_id),
                    !current && self.engine.core.in_flight.contains_key(&global_id),
                ) {
                    SourceDisposition::AlreadyPhysical => physical += 1,
                    SourceDisposition::RamHit => ram += 1,
                    SourceDisposition::SourceInFlight => singleflight += 1,
                    SourceDisposition::StartSourceRead => position_inflight += 1,
                }
            }
        }
        let mut values = self.counters.lock();
        values.source_skipped_physical_current += physical;
        values.source_ram_hits += ram;
        values.source_singleflight_hits += singleflight;
        values.source_position_inflight_skips += position_inflight;
        values.source_prefetch_batches_skipped_inflight += routes.len() as u64;
        Ok(())
    }

    async fn complete_boundary(
        &self,
        mut boundary: GpuNativeSafeTokenBoundary,
        actual_routes: &[Vec<u32>],
    ) -> Result<(), String> {
        let position = boundary.position();
        let (future_routes, active, prior_failure) = {
            let mut state = self.state.lock();
            state.cursor.reconcile(position, actual_routes)?;
            if state.submitted_position != Some(position) {
                return Err(format!(
                    "safe boundary for position {position} has no matching submitted-token record"
                ));
            }
            state.submitted_position = None;
            let future_routes = state.cursor.future_after(position)?;
            let active = state.active.take();
            (future_routes, active, state.failure.clone())
        };
        if let Some(error) = prior_failure {
            return Err(error);
        }

        let Some(routes) = future_routes else {
            if active.is_some() {
                return Err("end-of-trace unexpectedly retained a future source task".into());
            }
            return Ok(());
        };
        let active = active.ok_or_else(|| {
            format!("position {position} completed without a scheduled future source task")
        })?;
        if self.mode == OracleScheduledResidencyMode::SourceOnly && active.handle.is_none() {
            if active.target_position != position + 1 {
                return Err(format!(
                    "future source task targets {}, expected {}",
                    active.target_position,
                    position + 1
                ));
            }
            let late_experts = self.future_source_not_ready_count(&routes)?;
            record_source_late(
                &mut self.counters.lock(),
                late_experts,
                LateSourcePolicy::NonblockingDemandFallback,
            );
            return Ok(());
        }
        let required_global_ids = self.required_future_global_ids(&routes)?;
        let manager = self
            .engine
            .core
            .gpu_native_residency
            .as_ref()
            .ok_or("GPU-native residency manager is absent")?
            .clone();
        let engine = self.engine.clone();
        let BoundarySourceResolution::Completed { lease, was_late } =
            resolve_source_task_at_boundary(
                self.mode,
                active,
                position + 1,
                required_global_ids,
                move |global_id| {
                    if manager
                        .has_current_for_demand(global_id)
                        .map_err(|error| error.to_string())?
                    {
                        Ok(BoundarySourceReadiness::Physical)
                    } else if engine.core.cache.contains(global_id) {
                        Ok(BoundarySourceReadiness::SourceResident)
                    } else {
                        Ok(BoundarySourceReadiness::Unavailable)
                    }
                },
                self.counters.clone(),
                self.state.clone(),
            )
            .await?
        else {
            return Ok(());
        };

        if self.mode == OracleScheduledResidencyMode::TokenBoundaryDirect {
            install_future_at_boundary(
                &self.engine,
                &mut boundary,
                &routes,
                &lease.outcome.residents,
                self.counters.clone(),
            )?;
            let unavailable_after_install = self.future_not_physically_current_count(&routes)?;
            record_direct_late_fallback(
                &mut self.counters.lock(),
                was_late,
                unavailable_after_install,
            );
        }
        release_source_position_lease_before_next_token(
            self.mode,
            lease,
            &self.source_position_gate,
            &self.state,
        )
    }

    async fn finish_request(&self) -> Result<(OracleCounters, OracleRouteTrace), String> {
        let mut background = std::mem::take(&mut self.state.lock().background);
        drain_background_tasks(&mut background).await?;
        let mut state = self.state.lock();
        if state.active.is_some() || state.submitted_position.is_some() {
            return Err("oracle scheduler retained active state after request completion".into());
        }
        if let Some(error) = state.failure.take() {
            return Err(error);
        }
        let trace = state.cursor.clone().finish()?;
        let mut counters = self.counters.lock().clone();
        counters.qualification_owned_current_slots = 0;
        counters.qualification_owned_current_bytes = 0;
        Ok((counters, trace))
    }
}

fn record_completed_source_outcome(
    counters: &mut OracleCounters,
    outcome: &PrefetchOutcome,
    source_residents_ready_at_boundary: u64,
) {
    counters.source_ready_before_boundary += source_residents_ready_at_boundary;
    counters.source_failures_degraded_to_demand += outcome.failed_ids.len() as u64;
}

fn record_source_position_considered(counters: &mut OracleCounters, routes: &[Vec<u32>]) {
    counters.future_experts_considered += routes.iter().map(Vec::len).sum::<usize>() as u64;
}

fn record_source_late(counters: &mut OracleCounters, late_experts: u64, policy: LateSourcePolicy) {
    counters.source_late_at_boundary += late_experts;
    if policy == LateSourcePolicy::NonblockingDemandFallback {
        counters.source_late_degraded_to_demand += 1;
    }
}

fn record_direct_late_fallback(
    counters: &mut OracleCounters,
    was_late: bool,
    unavailable_after_install: u64,
) {
    if was_late && unavailable_after_install != 0 {
        counters.source_late_degraded_to_demand += 1;
    }
}

fn record_demand_readiness(counters: &mut OracleCounters, current_by_layer: &[Vec<bool>]) {
    for layer in current_by_layer {
        let ready = layer.iter().filter(|&&current| current).count() as u64;
        counters.oracle_ready_at_demand += ready;
        counters.oracle_not_ready_at_demand += layer.len() as u64 - ready;
        counters.oracle_demand_fallback += u64::from(ready != layer.len() as u64);
    }
}

async fn drain_background_tasks(
    tasks: &mut Vec<tokio::task::JoinHandle<()>>,
) -> Result<(), String> {
    for task in tasks.drain(..) {
        task.await
            .map_err(|error| format!("background source drain failed: {error}"))?;
    }
    Ok(())
}

impl GpuNativeOracleScheduleHook for OracleScheduler {
    fn on_token_submitted(&self, position: usize) {
        let mut state = self.state.lock();
        if state.failure.is_some() {
            return;
        }
        if state.submitted_position == Some(position) {
            // Recovery may submit more than one segment for the same logical
            // token. Perfect-future source work is scheduled exactly once.
            return;
        }
        if state.submitted_position.is_some() || position != state.cursor.next_position {
            state.failure = Some(format!(
                "submission progression mismatch: next={} submitted={position}",
                state.cursor.next_position
            ));
            return;
        }
        state.submitted_position = Some(position);
        let routes = match state.cursor.future_after(position) {
            Ok(Some(routes)) => routes,
            Ok(None) => {
                self.counters.lock().source_end_of_trace += 1;
                return;
            }
            Err(error) => {
                state.failure = Some(error);
                return;
            }
        };
        if state.active.is_some() {
            state.failure = Some("future source task slot was already occupied".into());
            return;
        }
        record_source_position_considered(&mut self.counters.lock(), &routes);
        let engine = self.engine.clone();
        let counters = self.counters.clone();
        let concurrency = self.concurrency;
        let expert_bytes = self.expert_bytes;
        let target_position = position + 1;
        let readiness = Arc::new(SourceReadiness::default());
        let handle = match self.source_position_gate.clone().try_lock_owned() {
            Ok(position_guard) => {
                let readiness_for_task = readiness.clone();
                let handle = tokio::spawn(async move {
                    let outcome = prefetch_position(
                        engine,
                        target_position,
                        routes,
                        concurrency,
                        expert_bytes,
                        counters.clone(),
                        readiness_for_task,
                    )
                    .await;
                    PrefetchLease::new(outcome, counters, position_guard)
                });
                self.counters.lock().background_tasks_spawned += 1;
                Some(handle)
            }
            Err(_) => {
                if self.mode == OracleScheduledResidencyMode::TokenBoundaryDirect {
                    state.failure = Some(
                        "token-boundary-direct could not acquire the next position lease after the prior safe boundary"
                            .into(),
                    );
                } else if let Err(error) = self.record_position_gate_skip(&routes) {
                    state.failure = Some(error);
                }
                None
            }
        };
        state.active = Some(ActivePrefetch {
            target_position,
            handle,
            readiness,
        });
    }

    fn on_safe_token_boundary<'a>(
        &'a self,
        _engine: &'a Arc<Engine>,
        boundary: GpuNativeSafeTokenBoundary,
        actual_routes: &'a [Vec<u32>],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>> {
        Box::pin(async move { self.complete_boundary(boundary, actual_routes).await })
    }
}

#[derive(Clone, Debug, Serialize)]
struct PhysicalSlotPlanEvidence {
    model_expert_budget_bytes: u64,
    model_arena_allocation_bytes: u64,
    model_slot_capacity: usize,
    per_layer_slot_capacity: Vec<usize>,
    no_extra_physical_vram_capacity: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct TreatmentContract {
    source_overlap: bool,
    late_source_policy: LateSourcePolicy,
    host_preparation_overlap: bool,
    token_boundary_h2d: bool,
    demand_fallback: bool,
    h2d_compute_overlap_claimed: bool,
    same_ordered_queue: bool,
    extra_flush_submit: bool,
    production_predictor_used: bool,
    production_speculative_residency_used: bool,
    production_direct_staging_used: bool,
    legacy_full_slot_vec_used: bool,
}

const fn treatment_contract(mode: OracleScheduledResidencyMode) -> TreatmentContract {
    TreatmentContract {
        source_overlap: true,
        late_source_policy: late_source_policy(mode),
        host_preparation_overlap: false,
        token_boundary_h2d: matches!(mode, OracleScheduledResidencyMode::TokenBoundaryDirect),
        demand_fallback: true,
        h2d_compute_overlap_claimed: false,
        same_ordered_queue: true,
        extra_flush_submit: false,
        production_predictor_used: false,
        production_speculative_residency_used: false,
        production_direct_staging_used: matches!(
            mode,
            OracleScheduledResidencyMode::TokenBoundaryDirect
        ),
        legacy_full_slot_vec_used: false,
    }
}

#[derive(Clone, Debug, Serialize)]
struct FrozenWorkloadEvidence {
    prompt: &'static str,
    prompt_sha256: &'static str,
    prompt_token_ids_sha256: &'static str,
    output_tokens: usize,
    greedy: bool,
    warmup_runs: usize,
    measured_runs: usize,
    repeated_trace_per_request: bool,
    shared_runtime_and_cache: bool,
    initial_position_priming: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct FrozenControlReferences {
    frozen_parent_git_sha: &'static str,
    oracle_route_command_git_sha: &'static str,
    performance_controls_rerun: bool,
    hardware_results_embedded: bool,
}

#[derive(Clone, Debug, Serialize)]
struct RunCorrectness {
    generated_token_ids_sha256: String,
    generated_text_sha256: String,
    actual_route_sequence_sha256: String,
    actual_legacy_route_sha256: String,
    generated_tokens_match_frozen_demand_only: bool,
    generated_text_matches_frozen_demand_only: bool,
    actual_route_matches_oracle: bool,
    actual_legacy_route_matches_oracle: bool,
    no_fatal_failures: bool,
}

fn frozen_output_matches(
    generated_token_sha: &str,
    generated_text_sha: &str,
    expected_token_sha: &str,
    expected_text_sha: &str,
) -> (bool, bool) {
    (
        generated_token_sha == expected_token_sha,
        generated_text_sha == expected_text_sha,
    )
}

#[derive(Clone, Debug, Serialize)]
struct OracleRunResult {
    run_index: usize,
    benchmark: PerRunResult,
    oracle: OracleCounters,
    correctness: RunCorrectness,
}

#[derive(Clone, Debug, Serialize)]
struct AggregateOracleCounters {
    measured_runs: usize,
    totals: OracleCounters,
}

#[derive(Clone, Debug, Serialize)]
struct CorrectnessEvidence {
    warmup_matches_measured: bool,
    all_measured_outputs_identical: bool,
    all_actual_routes_match_oracle: bool,
    generated_token_ids_sha256: String,
    generated_text_sha256: String,
    actual_route_sequence_sha256: String,
    actual_legacy_route_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct OracleScheduledResidencyReport {
    schema: &'static str,
    mode: &'static str,
    complete: bool,
    failure: Option<BenchmarkFailure>,
    diagnostic_only: bool,
    provenance: BenchmarkProvenance,
    source_oracle: SourceOracleEvidence,
    model_identity: ModelIdentityEvidence,
    model_load: ModelLoadEvidence,
    adapter_identity: GpuDeviceIdentity,
    runtime_contract: RuntimeContractEvidence,
    request: RequestEvidence,
    production_configuration: ProductionConfiguration,
    treatment_mode: OracleScheduledResidencyMode,
    oracle_source_concurrency: usize,
    treatment_contract: TreatmentContract,
    physical_slot_plan: PhysicalSlotPlanEvidence,
    frozen_controls: FrozenControlReferences,
    frozen_workload: FrozenWorkloadEvidence,
    warmup: OracleRunResult,
    measured: Vec<OracleRunResult>,
    aggregate_measured_performance: Aggregate,
    aggregate_measured_oracle: AggregateOracleCounters,
    correctness: CorrectnessEvidence,
    shutdown: BackgroundShutdownEvidence,
}

fn accumulate_oracle(total: &mut OracleCounters, value: &OracleCounters) {
    macro_rules! add {
        ($field:ident) => {
            total.$field = total.$field.saturating_add(value.$field)
        };
    }
    add!(initial_position_source_priming_experts);
    add!(initial_position_source_priming_us);
    add!(future_experts_considered);
    add!(source_skipped_physical_current);
    add!(source_ram_hits);
    add!(source_singleflight_hits);
    add!(source_position_inflight_skips);
    add!(source_reads_started);
    add!(source_reads_completed);
    add!(source_reads_failed);
    add!(source_bytes_read);
    add!(source_prefetch_batches_started);
    add!(source_prefetch_batches_completed);
    add!(source_prefetch_batches_skipped_inflight);
    total.source_prefetch_peak_inflight = total
        .source_prefetch_peak_inflight
        .max(value.source_prefetch_peak_inflight);
    add!(source_ready_before_boundary);
    add!(source_late_at_boundary);
    add!(source_end_of_trace);
    add!(future_physical_sets_considered);
    add!(future_physical_experts_needed);
    add!(boundary_replacement_sets_started);
    add!(boundary_replacement_sets_completed);
    add!(boundary_physical_installs);
    add!(boundary_physical_install_bytes);
    add!(boundary_physical_evictions);
    add!(boundary_physical_reinstalls);
    add!(boundary_stale_generation);
    add!(boundary_install_failures);
    add!(boundary_direct_staging_writes);
    add!(boundary_full_slot_vec_materializations);
    add!(oracle_ready_at_demand);
    add!(oracle_not_ready_at_demand);
    add!(oracle_demand_fallback);
    add!(ordinary_residency_miss_attempts);
    add!(ordinary_residency_services);
    add!(source_prefetch_wall_us);
    add!(source_read_aggregate_us);
    add!(host_preparation_overlap_us);
    add!(token_boundary_host_stage_us);
    add!(token_boundary_physical_stage_us);
    add!(token_boundary_commit_us);
    add!(oracle_wait_before_next_token_us);
    total.qualification_owned_current_slots = value.qualification_owned_current_slots;
    total.qualification_owned_current_bytes = value.qualification_owned_current_bytes;
    total.qualification_owned_peak_slots = total
        .qualification_owned_peak_slots
        .max(value.qualification_owned_peak_slots);
    total.qualification_owned_peak_bytes = total
        .qualification_owned_peak_bytes
        .max(value.qualification_owned_peak_bytes);
    add!(source_failures_degraded_to_demand);
    add!(source_late_degraded_to_demand);
    add!(background_tasks_spawned);
    add!(background_tasks_drained);
}

fn validate_oracle_counters(
    counters: &OracleCounters,
    mode: OracleScheduledResidencyMode,
    concurrency: usize,
) -> Result<(), BenchmarkFailure> {
    let classified_source = counters
        .source_skipped_physical_current
        .saturating_add(counters.source_ram_hits)
        .saturating_add(counters.source_singleflight_hits)
        .saturating_add(counters.source_position_inflight_skips)
        .saturating_add(counters.source_reads_started);
    if counters.future_experts_considered != EXPECTED_SELECTED_IDS as u64
        || classified_source != counters.future_experts_considered
        || counters.source_reads_started
            != counters
                .source_reads_completed
                .saturating_add(counters.source_reads_failed)
        || counters
            .source_prefetch_batches_started
            .saturating_add(counters.source_prefetch_batches_skipped_inflight)
            != EXPECTED_LAYER_RECORDS as u64
        || counters.source_prefetch_batches_completed != counters.source_prefetch_batches_started
        || counters.source_prefetch_peak_inflight > concurrency as u64
        || counters
            .oracle_ready_at_demand
            .saturating_add(counters.oracle_not_ready_at_demand)
            != EXPECTED_SELECTED_IDS as u64
        || counters.qualification_owned_current_slots != 0
        || counters.qualification_owned_current_bytes != 0
        || counters.qualification_owned_peak_slots > FROZEN_RAM_CACHE_SLOTS as u64
        || counters.background_tasks_spawned > (EXPECTED_POSITIONS - 1) as u64
        || counters.background_tasks_drained != counters.background_tasks_spawned
        || counters.source_end_of_trace != 1
    {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "oracle-counter-reconciliation",
            format!("ORACLE source/physical/accounting counters do not reconcile: {counters:?}"),
        ));
    }
    if mode == OracleScheduledResidencyMode::SourceOnly
        && (counters.boundary_replacement_sets_started != 0
            || counters.boundary_physical_installs != 0
            || counters.boundary_physical_evictions != 0)
    {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "source-only-physical-mutation",
            "source-only treatment recorded destructive physical work",
        ));
    }
    if counters.boundary_full_slot_vec_materializations != 0 {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "legacy-physical-path-used",
            "ORACLE token-boundary treatment materialized a legacy full-slot Vec",
        ));
    }
    if mode == OracleScheduledResidencyMode::TokenBoundaryDirect
        && (counters.source_position_inflight_skips != 0
            || counters.source_prefetch_batches_skipped_inflight != 0
            || counters.boundary_direct_staging_writes != counters.boundary_physical_installs
            || counters.boundary_physical_evictions > counters.boundary_physical_installs
            || counters.boundary_replacement_sets_completed
                > counters.boundary_replacement_sets_started
            || (counters.boundary_physical_installs == 0)
                != (counters.boundary_physical_install_bytes == 0))
    {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "oracle-physical-counter-reconciliation",
            "ORACLE direct-mode position leases, direct staging, install bytes, eviction, or replacement-set counters do not reconcile",
        ));
    }
    Ok(())
}

struct ValidatedRuntime {
    model_load: ModelLoadEvidence,
    adapter_identity: GpuDeviceIdentity,
    runtime_contract: RuntimeContractEvidence,
    physical_slot_plan: PhysicalSlotPlanEvidence,
}

fn validate_runtime(
    runtime: &crate::BenchRealRuntime,
    resolved_config_sha256: &str,
    trace: &OracleRouteTrace,
) -> Result<ValidatedRuntime, BenchmarkFailure> {
    let observed_config_sha256 = crate::resolved_real_runtime_identity_sha256(
        &runtime.cfg,
        runtime.model.config.architecture,
        runtime.model.config.first_k_dense_replace,
        &runtime.model.config.advanced,
    )
    .map_err(|error| {
        BenchmarkFailure::new(
            "startup",
            "runtime-config-identity-unavailable",
            error.to_string(),
        )
    })?;
    if observed_config_sha256 != resolved_config_sha256 || runtime.cfg.storage.predict_fanout != 0 {
        return Err(BenchmarkFailure::new(
            "startup",
            "runtime-contract-drift",
            "resolved runtime identity drifted or production predictor fanout is nonzero",
        ));
    }
    let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
        BenchmarkFailure::new(
            "startup",
            "missing-gpu-native-token-loop",
            "ORACLE-0B-S requires the authoritative GPU-native token loop",
        )
    })?;
    if OracleGeometry::from(token_loop.model_geometry()) != trace.geometry
        || token_loop.snapshot() != GpuNativeTokenLoopSnapshot::default()
        || token_loop.recovery_snapshot() != GpuNativeRecoverySnapshot::default()
        || runtime.engine.routed_expert_execution_snapshot()
            != RoutedExpertExecutionSnapshot::default()
    {
        return Err(BenchmarkFailure::new(
            "startup",
            "runtime-initial-state-invalid",
            "runtime geometry differs from ORACLE-0A or fresh counters are nonzero",
        ));
    }
    let model_load = crate::greedy_parity_model_load(runtime);
    let contract_input = RuntimeContractInput {
        real_transformer_enabled: runtime.cfg.real_transformer.enabled,
        real_transformer_gpu_native: runtime.cfg.real_transformer.gpu_native,
        compute_offload: runtime.cfg.real_transformer.compute_offload,
        legacy_execution_plan: runtime.engine.execution_context().plan().into(),
        token_loop_geometry: Some(token_loop.model_geometry()),
        authoritative_device: runtime.engine.gpu_device_identity(),
        model_load: model_load.clone(),
        routed_failure_policy: runtime.engine.routed_expert_gpu_failure_policy(),
    };
    let (runtime_contract, adapter_identity) =
        crate::gpu_native_real_benchmark::validate_runtime_contract(
            &contract_input,
            EXPECTED_ADAPTER_NAME,
        )?;
    let snapshot = runtime
        .engine
        .gpu_native_residency_snapshot()
        .ok_or_else(|| {
            BenchmarkFailure::new(
                "startup",
                "missing-physical-slot-plan",
                "GPU-native residency snapshot is unavailable",
            )
        })?;
    Ok(ValidatedRuntime {
        model_load,
        adapter_identity,
        runtime_contract,
        physical_slot_plan: PhysicalSlotPlanEvidence {
            model_expert_budget_bytes: snapshot.model_expert_budget_bytes,
            model_arena_allocation_bytes: snapshot.model_arena_allocation_bytes,
            model_slot_capacity: snapshot.model_slot_capacity,
            per_layer_slot_capacity: snapshot
                .layers
                .iter()
                .map(|layer| layer.slot_capacity)
                .collect(),
            no_extra_physical_vram_capacity: true,
        },
    })
}

async fn execute_oracle_request(
    runtime: &crate::BenchRealRuntime,
    trace: Arc<OracleRouteTrace>,
    expected_token_sha: &str,
    expected_text_sha: &str,
    treatment_mode: OracleScheduledResidencyMode,
    source_concurrency: usize,
    expert_bytes: usize,
    prompt_ids: &[u32],
    run_index: usize,
) -> Result<OracleRunResult, Box<dyn std::error::Error>> {
    let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
        BenchmarkFailure::new(
            "startup",
            "missing-gpu-native-token-loop",
            "ORACLE-0B-S request has no GPU-native token loop",
        )
    })?;
    let scheduler = OracleScheduler::new(
        runtime.engine.clone(),
        trace,
        treatment_mode,
        source_concurrency,
        expert_bytes,
    );
    let mut request = token_loop.create_request_state()?;
    let snapshots = RequestSnapshotStart::capture(runtime)?;
    let request_started = Instant::now();
    scheduler
        .prime_initial_position()
        .await
        .map_err(|error| BenchmarkFailure::new("inference", "initial-source-prime", error))?;

    let mut completed_positions = 0usize;
    let mut generated_ids = Vec::with_capacity(FROZEN_OUTPUT_TOKENS);
    while completed_positions < prompt_ids.len() {
        let token_id = prompt_ids[completed_positions];
        let sample = completed_positions + 1 == prompt_ids.len();
        scheduler
            .observe_demand_readiness(completed_positions)
            .map_err(|error| {
                BenchmarkFailure::new("inference", "oracle-demand-probe-failed", error)
            })?;
        let sampled = token_loop
            .step_token_oracle_scheduled(
                &runtime.engine,
                &mut request,
                token_id,
                completed_positions,
                sample,
                &scheduler,
            )
            .await?;
        completed_positions += 1;
        if sample {
            generated_ids.push(sampled.ok_or_else(|| {
                BenchmarkFailure::new(
                    "inference",
                    "missing-first-generated-token",
                    "final prompt position produced no sampled token",
                )
            })?);
        } else if sampled.is_some() {
            return Err(BenchmarkFailure::new(
                "inference",
                "unexpected-prefix-sample",
                "prompt prefix position unexpectedly produced a sampled token",
            )
            .into());
        }
    }
    let prompt_seconds = request_started.elapsed().as_secs_f64();
    let decode_started = Instant::now();
    let mut decode_latencies = Vec::with_capacity(FROZEN_OUTPUT_TOKENS - 1);
    while generated_ids.len() < FROZEN_OUTPUT_TOKENS {
        let token_id = *generated_ids.last().ok_or_else(|| {
            BenchmarkFailure::new(
                "inference",
                "missing-generated-seed",
                "decode has no preceding generated token",
            )
        })?;
        scheduler
            .observe_demand_readiness(completed_positions)
            .map_err(|error| {
                BenchmarkFailure::new("inference", "oracle-demand-probe-failed", error)
            })?;
        let step_started = Instant::now();
        let sampled = token_loop
            .step_token_oracle_scheduled(
                &runtime.engine,
                &mut request,
                token_id,
                completed_positions,
                true,
                &scheduler,
            )
            .await?
            .ok_or_else(|| {
                BenchmarkFailure::new(
                    "inference",
                    "missing-decode-token",
                    format!("position {completed_positions} produced no sampled token"),
                )
            })?;
        decode_latencies.push(step_started.elapsed().as_secs_f64());
        generated_ids.push(sampled);
        completed_positions += 1;
    }
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    let (mut oracle, actual_trace) = scheduler
        .finish_request()
        .await
        .map_err(|error| BenchmarkFailure::new("postcondition", "oracle-finish-failed", error))?;
    let counters = snapshots.finish(runtime)?;
    crate::gpu_native_real_benchmark::validate_request_postconditions(
        prompt_ids.len(),
        FROZEN_OUTPUT_TOKENS,
        generated_ids.len(),
        counters.token_loop_delta,
        counters.recovery_delta,
        counters.routed_execution_delta,
    )?;
    oracle.ordinary_residency_miss_attempts = counters.token_loop_delta.residency_miss_attempts;
    oracle.ordinary_residency_services = counters.token_loop_delta.residency_services;
    validate_oracle_counters(&oracle, treatment_mode, source_concurrency)?;

    let output_text = runtime.tokenizer.decode(&generated_ids)?;
    let generated_token_ids_sha256 = crate::greedy_parity::token_ids_sha256(&generated_ids);
    let generated_text_sha256 = crate::greedy_parity::sha256_hex(output_text.as_bytes());
    let (generated_tokens_match_frozen_demand_only, generated_text_matches_frozen_demand_only) =
        frozen_output_matches(
            &generated_token_ids_sha256,
            &generated_text_sha256,
            expected_token_sha,
            expected_text_sha,
        );
    let correctness = RunCorrectness {
        generated_tokens_match_frozen_demand_only,
        generated_text_matches_frozen_demand_only,
        actual_route_matches_oracle: actual_trace.ordered_route_sequence_sha256
            == EXPECTED_ORDERED_ROUTE_SHA256,
        actual_legacy_route_matches_oracle: actual_trace.legacy_selected_route_ids_sha256
            == EXPECTED_LEGACY_ROUTE_SHA256,
        no_fatal_failures: counters.token_loop_delta.fatal_failures == 0
            && counters.token_loop_delta.no_progress_failures == 0,
        generated_token_ids_sha256: generated_token_ids_sha256.clone(),
        generated_text_sha256: generated_text_sha256.clone(),
        actual_route_sequence_sha256: actual_trace.ordered_route_sequence_sha256,
        actual_legacy_route_sha256: actual_trace.legacy_selected_route_ids_sha256,
    };
    if !correctness.generated_tokens_match_frozen_demand_only
        || !correctness.generated_text_matches_frozen_demand_only
        || !correctness.actual_route_matches_oracle
        || !correctness.actual_legacy_route_matches_oracle
        || !correctness.no_fatal_failures
    {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "oracle-correctness-mismatch",
            "tokens, text, actual routes, or failure counters differ from frozen ORACLE-0A demand-only evidence",
        )
        .into());
    }
    let benchmark = PerRunResult {
        run_index,
        prompt_tokens: prompt_ids.len(),
        requested_output_tokens: FROZEN_OUTPUT_TOKENS,
        generated_tokens: generated_ids.len(),
        generated_token_ids: generated_ids,
        generated_token_ids_sha256,
        generated_text_sha256,
        timing: RunTiming::from_measurement(
            FROZEN_OUTPUT_TOKENS,
            prompt_seconds,
            prompt_seconds,
            decode_seconds,
            decode_latencies,
        )?,
        counters,
    };
    Ok(OracleRunResult {
        run_index,
        benchmark,
        oracle,
        correctness,
    })
}

fn emit_report(
    report: &OracleScheduledResidencyReport,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = serde_json::to_vec_pretty(report)?;
    bytes.push(b'\n');
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)?;
    eprintln!(
        "GPU-native ORACLE-0B-S report written to {}",
        path.display()
    );
    Ok(())
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.oracle_source_concurrency == 0 || args.oracle_source_concurrency > EXPECTED_LAYERS {
        return Err(artifact_failure(
            "invalid-oracle-source-concurrency",
            format!(
                "--oracle-source-concurrency must be in 1..={EXPECTED_LAYERS}; observed {}",
                args.oracle_source_concurrency
            ),
        )
        .into());
    }
    let oracle = load_oracle_artifact(&args.oracle_trace)?;
    let build = crate::qualification::BuildProvenance::embedded();
    crate::gpu_native_real_benchmark::validate_preflight_provenance(&build)?;
    let cfg = crate::config::Config::from_file(&args.config)?;
    crate::gpu_native_real_benchmark::validate_source_config(&cfg)?;
    if cfg.storage.predict_fanout != 0 {
        return Err(artifact_failure(
            "production-predictor-enabled",
            "ORACLE-0B-S requires storage.predict_fanout=0 and never overrides it",
        )
        .into());
    }
    let (artifacts, artifact_errors) = crate::qualification_artifacts(&args.config, &cfg);
    crate::gpu_native_real_benchmark::validate_artifacts(&artifacts, &artifact_errors)?;
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                BenchmarkFailure::new("preflight", "expert-metadata-unavailable", error)
            })?;
    crate::gpu_native_real_benchmark::validate_expert_metadata(&expert_metadata)?;
    let spec = crate::resolve_real_cli_spec_from_config(
        cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let model_identity = crate::greedy_parity_model_identity(&spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err(artifact_failure(
            "wrong-model-identity",
            format!("requires exact Qwen3-Coder 30B-A3B Q4_0; observed {model_identity:?}"),
        )
        .into());
    }
    let resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&spec)?;
    let tokenizer = crate::load_real_cli_tokenizer(
        &spec.cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let prompt_ids = tokenizer.encode(FROZEN_PROMPT)?;
    let prompt_sha256 = crate::greedy_parity::sha256_hex(FROZEN_PROMPT.as_bytes());
    let prompt_token_ids_sha256 = crate::greedy_parity::token_ids_sha256(&prompt_ids);
    if prompt_sha256 != EXPECTED_PROMPT_SHA256
        || prompt_token_ids_sha256 != EXPECTED_PROMPT_TOKEN_IDS_SHA256
        || prompt_ids
            .len()
            .checked_add(FROZEN_OUTPUT_TOKENS)
            .and_then(|value| value.checked_sub(1))
            != Some(EXPECTED_POSITIONS)
    {
        return Err(artifact_failure(
            "frozen-workload-identity-mismatch",
            "frozen prompt/tokenization/output shape differs from ORACLE-0A",
        )
        .into());
    }
    let production_configuration =
        ProductionConfiguration::from_config(&spec.cfg, &expert_metadata);
    if production_configuration.cache_residency.ram_cache_slots != FROZEN_RAM_CACHE_SLOTS {
        return Err(artifact_failure(
            "ram-cache-slot-count-mismatch",
            format!(
                "ORACLE-0B-S requires the frozen {FROZEN_RAM_CACHE_SLOTS}-slot RAM cache; observed {}",
                production_configuration.cache_residency.ram_cache_slots
            ),
        )
        .into());
    }
    let (executable, executable_sha256) = crate::current_executable_identity()?;
    let executable_canonical_path = std::fs::canonicalize(&executable)?.display().to_string();
    if !crate::gpu_native_real_benchmark::is_hex(&executable_sha256, 64)
        || !crate::gpu_native_real_benchmark::is_hex(&resolved_config_sha256, 64)
    {
        return Err(artifact_failure(
            "provenance-unavailable",
            "executable or resolved-config SHA256 was unavailable",
        )
        .into());
    }
    let provenance = BenchmarkProvenance {
        build,
        executable_canonical_path,
        executable_sha256,
        resolved_config_sha256: resolved_config_sha256.clone(),
        artifacts,
        expert_metadata,
    };
    let request = RequestEvidence {
        prompt_sha256,
        prompt_token_ids_sha256,
        prompt_token_count: prompt_ids.len(),
        requested_output_tokens: FROZEN_OUTPUT_TOKENS,
        greedy: true,
    };

    let runtime = crate::build_isolated_greedy_runtime(
        &spec,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
        tokenizer,
    )
    .await?;
    let execution = async {
        let validated = validate_runtime(&runtime, &resolved_config_sha256, &oracle.trace)?;
        let warmup = crate::with_progress_timeout(
            format!("{MODE} warmup"),
            args.progress_watchdog,
            execute_oracle_request(
                &runtime,
                oracle.trace.clone(),
                &oracle.evidence.expected_generated_token_ids_sha256,
                &oracle.evidence.expected_generated_text_sha256,
                args.treatment_mode,
                args.oracle_source_concurrency,
                spec.cfg.model.expert_size,
                &prompt_ids,
                0,
            ),
        )
        .await?;
        let mut measured = Vec::with_capacity(FROZEN_MEASURED_RUNS);
        for run_index in 0..FROZEN_MEASURED_RUNS {
            measured.push(
                crate::with_progress_timeout(
                    format!("{MODE} measured run {run_index}"),
                    args.progress_watchdog,
                    execute_oracle_request(
                        &runtime,
                        oracle.trace.clone(),
                        &oracle.evidence.expected_generated_token_ids_sha256,
                        &oracle.evidence.expected_generated_text_sha256,
                        args.treatment_mode,
                        args.oracle_source_concurrency,
                        spec.cfg.model.expert_size,
                        &prompt_ids,
                        run_index,
                    ),
                )
                .await?,
            );
        }
        Ok::<_, Box<dyn std::error::Error>>((validated, warmup, measured))
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    let (validated, warmup, measured, shutdown) = match (execution, shutdown) {
        (Ok((validated, warmup, measured)), Ok(shutdown)) => {
            if !shutdown.controlled_shutdown_requested || !shutdown.all_runtime_resources_released {
                return Err(BenchmarkFailure::new(
                    "postcondition",
                    "runtime-shutdown-incomplete",
                    format!("controlled isolated shutdown was incomplete: {shutdown:?}"),
                )
                .into());
            }
            (validated, warmup, measured, shutdown)
        }
        (Err(error), Ok(_)) => return Err(error),
        (Ok(_), Err(error)) => return Err(error.into()),
        (Err(execution), Err(shutdown)) => {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "execution-and-shutdown-failed",
                format!("{execution}; {shutdown}"),
            )
            .into())
        }
    };

    let measured_benchmarks = measured
        .iter()
        .map(|run| run.benchmark.clone())
        .collect::<Vec<_>>();
    let aggregate_measured_performance =
        crate::gpu_native_real_benchmark::aggregate(&measured_benchmarks)?;
    let mut aggregate_oracle = OracleCounters::default();
    for run in &measured {
        accumulate_oracle(&mut aggregate_oracle, &run.oracle);
    }
    let all_measured_outputs_identical = measured.iter().all(|run| {
        run.correctness.generated_token_ids_sha256 == warmup.correctness.generated_token_ids_sha256
            && run.correctness.generated_text_sha256 == warmup.correctness.generated_text_sha256
    });
    let all_actual_routes_match_oracle = measured.iter().all(|run| {
        run.correctness.actual_route_matches_oracle
            && run.correctness.actual_legacy_route_matches_oracle
    });
    if !all_measured_outputs_identical || !all_actual_routes_match_oracle {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "cross-request-correctness-mismatch",
            "warmup and measured output or actual-route hashes differ",
        )
        .into());
    }
    let report = OracleScheduledResidencyReport {
        schema: SCHEMA,
        mode: MODE,
        complete: true,
        failure: None,
        diagnostic_only: true,
        provenance,
        source_oracle: oracle.evidence,
        model_identity,
        model_load: validated.model_load,
        adapter_identity: validated.adapter_identity,
        runtime_contract: validated.runtime_contract,
        request,
        production_configuration,
        treatment_mode: args.treatment_mode,
        oracle_source_concurrency: args.oracle_source_concurrency,
        treatment_contract: treatment_contract(args.treatment_mode),
        physical_slot_plan: validated.physical_slot_plan,
        frozen_controls: FrozenControlReferences {
            frozen_parent_git_sha: ORACLE_ROUTE_COMMAND_SHA,
            oracle_route_command_git_sha: ORACLE_ROUTE_COMMAND_SHA,
            performance_controls_rerun: false,
            hardware_results_embedded: false,
        },
        frozen_workload: FrozenWorkloadEvidence {
            prompt: FROZEN_PROMPT,
            prompt_sha256: EXPECTED_PROMPT_SHA256,
            prompt_token_ids_sha256: EXPECTED_PROMPT_TOKEN_IDS_SHA256,
            output_tokens: FROZEN_OUTPUT_TOKENS,
            greedy: true,
            warmup_runs: FROZEN_WARMUP_RUNS,
            measured_runs: FROZEN_MEASURED_RUNS,
            repeated_trace_per_request: true,
            shared_runtime_and_cache: true,
            initial_position_priming: "synchronous source-to-RAM priming is included in request/TTFT timing; no position-zero physical oracle install is performed",
        },
        correctness: CorrectnessEvidence {
            warmup_matches_measured: all_measured_outputs_identical,
            all_measured_outputs_identical,
            all_actual_routes_match_oracle,
            generated_token_ids_sha256: warmup
                .correctness
                .generated_token_ids_sha256
                .clone(),
            generated_text_sha256: warmup.correctness.generated_text_sha256.clone(),
            actual_route_sequence_sha256: warmup
                .correctness
                .actual_route_sequence_sha256
                .clone(),
            actual_legacy_route_sha256: warmup
                .correctness
                .actual_legacy_route_sha256
                .clone(),
        },
        warmup,
        measured,
        aggregate_measured_performance,
        aggregate_measured_oracle: AggregateOracleCounters {
            measured_runs: FROZEN_MEASURED_RUNS,
            totals: aggregate_oracle,
        },
        shutdown,
    };
    emit_report(&report, &args.report_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    fn test_trace(positions: usize) -> Arc<OracleRouteTrace> {
        let geometry = OracleGeometry {
            layers: 2,
            experts: 4,
            top_k: 2,
            d_model: 8,
            d_ff: 4,
        };
        let mut records = Vec::with_capacity(positions * geometry.layers);
        for position in 0..positions {
            for layer_index in 0..geometry.layers {
                let first = ((position + layer_index) % geometry.experts) as u32;
                records.push(OracleRouteRecord {
                    position,
                    layer_index,
                    ordered_selected_expert_ids: vec![first, (first + 1) % 4],
                });
            }
        }
        Arc::new(OracleRouteTrace::try_new(geometry, records).unwrap())
    }

    fn valid_source_only_counters() -> OracleCounters {
        OracleCounters {
            future_experts_considered: EXPECTED_SELECTED_IDS as u64,
            source_skipped_physical_current: EXPECTED_SELECTED_IDS as u64,
            source_prefetch_batches_started: EXPECTED_LAYER_RECORDS as u64,
            source_prefetch_batches_completed: EXPECTED_LAYER_RECORDS as u64,
            source_prefetch_peak_inflight: 4,
            oracle_ready_at_demand: EXPECTED_SELECTED_IDS as u64,
            source_end_of_trace: 1,
            ..OracleCounters::default()
        }
    }

    fn test_scheduler_state() -> Arc<Mutex<SchedulerState>> {
        Arc::new(Mutex::new(SchedulerState {
            cursor: StrictRouteCursor::new(test_trace(2)),
            submitted_position: None,
            active: None,
            background: Vec::new(),
            failure: None,
        }))
    }

    async fn pending_test_prefetch(
        target_position: usize,
        counters: Arc<Mutex<OracleCounters>>,
    ) -> (
        ActivePrefetch,
        tokio::sync::oneshot::Sender<PrefetchOutcome>,
        Arc<tokio::sync::Mutex<()>>,
        Arc<SourceReadiness>,
    ) {
        let source_position_gate = Arc::new(tokio::sync::Mutex::new(()));
        let position_guard = source_position_gate.clone().lock_owned().await;
        let readiness = Arc::new(SourceReadiness::default());
        let counters_for_task = counters.clone();
        let (send_outcome, receive_outcome) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            let outcome = receive_outcome
                .await
                .expect("test controls source task completion");
            PrefetchLease::new(outcome, counters_for_task, position_guard)
        });
        counters.lock().background_tasks_spawned += 1;
        (
            ActivePrefetch {
                target_position,
                handle: Some(handle),
                readiness: readiness.clone(),
            },
            send_outcome,
            source_position_gate,
            readiness,
        )
    }

    fn successful_test_outcome(target_position: usize) -> PrefetchOutcome {
        PrefetchOutcome {
            target_position,
            ..PrefetchOutcome::default()
        }
    }

    #[test]
    fn exact_route_cursor_progression() {
        let trace = test_trace(3);
        let mut cursor = StrictRouteCursor::new(trace.clone());
        for position in 0..3 {
            let routes = cursor.routes_at(position).unwrap();
            cursor.reconcile(position, &routes).unwrap();
        }
        let actual = cursor.finish().unwrap();
        assert_eq!(actual.records, trace.records);
        assert_eq!(
            actual.ordered_route_sequence_sha256,
            trace.ordered_route_sequence_sha256
        );
    }

    #[test]
    fn route_cursor_fails_closed_at_end_of_trace() {
        let trace = test_trace(2);
        let cursor = StrictRouteCursor::new(trace);
        assert!(cursor.routes_at(2).is_err());
        assert!(cursor.future_after(1).unwrap().is_none());
        assert!(cursor.future_after(2).is_err());
        assert!(cursor.finish().is_err());
    }

    #[test]
    fn identical_trace_repeats_for_warmup_and_three_measurements() {
        let trace = test_trace(3);
        for _request in 0..(FROZEN_WARMUP_RUNS + FROZEN_MEASURED_RUNS) {
            let mut cursor = StrictRouteCursor::new(trace.clone());
            for position in 0..trace.total_positions {
                let routes = cursor.routes_at(position).unwrap();
                cursor.reconcile(position, &routes).unwrap();
            }
            assert_eq!(
                cursor.finish().unwrap().ordered_route_sequence_sha256,
                trace.ordered_route_sequence_sha256
            );
        }
    }

    #[test]
    fn source_only_contract_never_mutates_physical_residency() {
        let contract = treatment_contract(OracleScheduledResidencyMode::SourceOnly);
        assert!(!contract.token_boundary_h2d);
        assert!(!contract.production_direct_staging_used);
        assert!(!contract.h2d_compute_overlap_claimed);
        assert!(contract.demand_fallback);
    }

    #[test]
    fn treatment_mode_cli_values_are_exact() {
        assert_eq!(
            OracleScheduledResidencyMode::from_str("source-only", false).unwrap(),
            OracleScheduledResidencyMode::SourceOnly
        );
        assert_eq!(
            OracleScheduledResidencyMode::from_str("token-boundary-direct", false).unwrap(),
            OracleScheduledResidencyMode::TokenBoundaryDirect
        );
        assert!(OracleScheduledResidencyMode::from_str("overlapped-h2d", false).is_err());
    }

    #[tokio::test]
    async fn source_layer_tasks_obey_bounded_concurrency() {
        let concurrency = 3usize;
        let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let counters = Arc::new(Mutex::new(OracleCounters::default()));
        let mut tasks = Vec::new();
        for _ in 0..12 {
            let semaphore = semaphore.clone();
            let counters = counters.clone();
            tasks.push(tokio::spawn(async move {
                let _permit = semaphore.acquire_owned().await.unwrap();
                let _guard = LayerBatchGuard::enter(counters);
                for _ in 0..4 {
                    tokio::task::yield_now().await;
                }
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        let values = counters.lock();
        assert_eq!(values.source_prefetch_batches_started, 12);
        assert_eq!(values.source_prefetch_batches_completed, 12);
        assert!(values.source_prefetch_peak_inflight <= concurrency as u64);
    }

    #[tokio::test]
    async fn qualification_source_position_state_has_one_bounded_lease() {
        let gate = Arc::new(tokio::sync::Mutex::new(()));
        let first = gate.clone().lock_owned().await;
        assert!(gate.clone().try_lock_owned().is_err());
        drop(first);
        assert!(gate.try_lock_owned().is_ok());
    }

    #[test]
    fn treatment_contract_records_distinct_late_source_policies() {
        let source_only = treatment_contract(OracleScheduledResidencyMode::SourceOnly);
        let direct = treatment_contract(OracleScheduledResidencyMode::TokenBoundaryDirect);
        assert_eq!(
            source_only.late_source_policy,
            LateSourcePolicy::NonblockingDemandFallback
        );
        assert_eq!(
            direct.late_source_policy,
            LateSourcePolicy::AwaitResidualAtSafeBoundary
        );
        assert_eq!(
            serde_json::to_value(source_only).unwrap()["late_source_policy"],
            "nonblocking-demand-fallback"
        );
        assert_eq!(
            serde_json::to_value(direct).unwrap()["late_source_policy"],
            "await-residual-at-safe-boundary"
        );
    }

    #[tokio::test]
    async fn source_only_unfinished_task_does_not_wait_and_retains_demand_fallback() {
        let counters = Arc::new(Mutex::new(OracleCounters::default()));
        let state = test_scheduler_state();
        let (active, send_outcome, source_position_gate, _) =
            pending_test_prefetch(1, counters.clone()).await;

        let resolution = tokio::time::timeout(
            Duration::from_millis(100),
            resolve_source_task_at_boundary(
                OracleScheduledResidencyMode::SourceOnly,
                active,
                1,
                vec![0, 1, 2, 3],
                |_| Ok(BoundarySourceReadiness::Unavailable),
                counters.clone(),
                state.clone(),
            ),
        )
        .await
        .expect("source-only must not wait for an unfinished source task")
        .unwrap();
        assert!(matches!(
            resolution,
            BoundarySourceResolution::DeferredToDemand
        ));
        {
            let values = counters.lock();
            assert_eq!(values.source_late_at_boundary, 4);
            assert_eq!(values.source_late_degraded_to_demand, 1);
            assert_eq!(values.oracle_wait_before_next_token_us, 0);
        }
        record_demand_readiness(&mut counters.lock(), &[vec![false; 4]]);
        assert_eq!(counters.lock().oracle_demand_fallback, 1);

        assert!(send_outcome.send(successful_test_outcome(1)).is_ok());
        let mut background = std::mem::take(&mut state.lock().background);
        drain_background_tasks(&mut background).await.unwrap();
        assert!(source_position_gate.try_lock_owned().is_ok());
    }

    #[tokio::test]
    async fn token_boundary_direct_awaits_late_source_records_wait_and_releases_lease() {
        let counters = Arc::new(Mutex::new(OracleCounters::default()));
        let state = test_scheduler_state();
        let (active, send_outcome, source_position_gate, readiness) =
            pending_test_prefetch(1, counters.clone()).await;
        readiness.mark_physical_ready(0);
        readiness.mark_source_resident_ready(1);
        let mut resolving = Box::pin(resolve_source_task_at_boundary(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            active,
            1,
            vec![0, 1, 2, 3],
            |_| Ok(BoundarySourceReadiness::Unavailable),
            counters.clone(),
            state.clone(),
        ));

        assert!(
            tokio::time::timeout(Duration::from_millis(20), resolving.as_mut())
                .await
                .is_err()
        );
        {
            let values = counters.lock();
            assert_eq!(values.source_late_at_boundary, 2);
            assert_eq!(values.source_late_degraded_to_demand, 0);
            assert_eq!(values.oracle_wait_before_next_token_us, 0);
        }
        assert!(send_outcome.send(successful_test_outcome(1)).is_ok());
        let resolution = tokio::time::timeout(Duration::from_secs(1), resolving)
            .await
            .expect("direct mode must finish after the source task completes")
            .unwrap();
        let BoundarySourceResolution::Completed { lease, was_late } = resolution else {
            panic!("direct mode must return the completed source lease");
        };
        assert!(was_late);
        assert!(counters.lock().oracle_wait_before_next_token_us > 0);
        assert!(source_position_gate.clone().try_lock_owned().is_err());
        record_direct_late_fallback(&mut counters.lock(), was_late, 0);
        assert_eq!(counters.lock().source_late_degraded_to_demand, 0);
        release_source_position_lease_before_next_token(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            lease,
            &source_position_gate,
            &state,
        )
        .unwrap();
        assert!(source_position_gate.try_lock_owned().is_ok());
        assert!(state.lock().background.is_empty());
    }

    #[tokio::test]
    async fn token_boundary_direct_rejects_a_previous_position_lease_skip() {
        let counters = Arc::new(Mutex::new(OracleCounters::default()));
        let state = test_scheduler_state();
        let active = ActivePrefetch {
            target_position: 1,
            handle: None,
            readiness: Arc::new(SourceReadiness::default()),
        };
        let error = resolve_source_task_at_boundary(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            active,
            1,
            vec![0, 1, 2, 3],
            |_| Ok(BoundarySourceReadiness::Unavailable),
            counters,
            state,
        )
        .await
        .err()
        .expect("direct mode must fail closed instead of accumulating late positions");
        assert!(error.contains("prior position lease remained in flight"));
    }

    #[tokio::test]
    async fn token_boundary_direct_rejects_a_previous_position_background_task() {
        let counters = Arc::new(Mutex::new(OracleCounters::default()));
        let state = test_scheduler_state();
        let (active, send_outcome, source_position_gate, _) =
            pending_test_prefetch(1, counters.clone()).await;
        state.lock().background.push(tokio::spawn(async {}));

        let error = match resolve_source_task_at_boundary(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            active,
            1,
            vec![0],
            |_| Ok(BoundarySourceReadiness::Unavailable),
            counters,
            state.clone(),
        )
        .await
        {
            Err(error) => error,
            Ok(_) => panic!("direct mode must fail closed on a previous background task"),
        };
        assert!(error.contains("previous-position background source task"));

        assert!(send_outcome.send(successful_test_outcome(1)).is_ok());
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if source_position_gate.clone().try_lock_owned().is_ok() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("detached test source task must release its lease");
        let mut background = std::mem::take(&mut state.lock().background);
        drain_background_tasks(&mut background).await.unwrap();
    }

    #[tokio::test]
    async fn direct_late_source_failure_preserves_demand_fallback() {
        let counters = Arc::new(Mutex::new(OracleCounters::default()));
        let state = test_scheduler_state();
        let (active, send_outcome, source_position_gate, _) =
            pending_test_prefetch(1, counters.clone()).await;
        let mut resolving = Box::pin(resolve_source_task_at_boundary(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            active,
            1,
            vec![7],
            |_| Ok(BoundarySourceReadiness::Unavailable),
            counters.clone(),
            state.clone(),
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), resolving.as_mut())
                .await
                .is_err()
        );
        let outcome = PrefetchOutcome {
            target_position: 1,
            failed_ids: vec![7],
            ..PrefetchOutcome::default()
        };
        assert!(send_outcome.send(outcome).is_ok());
        let resolution = resolving.await.unwrap();
        let BoundarySourceResolution::Completed { lease, was_late } = resolution else {
            panic!("direct mode must join a source-read failure outcome");
        };
        assert!(was_late);
        record_direct_late_fallback(&mut counters.lock(), was_late, 1);
        record_demand_readiness(&mut counters.lock(), &[vec![false]]);
        {
            let values = counters.lock();
            assert_eq!(values.source_failures_degraded_to_demand, 1);
            assert_eq!(values.source_late_degraded_to_demand, 1);
            assert_eq!(values.oracle_demand_fallback, 1);
        }
        release_source_position_lease_before_next_token(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            lease,
            &source_position_gate,
            &state,
        )
        .unwrap();
    }

    #[tokio::test]
    async fn fatal_source_task_error_still_fails_qualification() {
        let counters = Arc::new(Mutex::new(OracleCounters::default()));
        let state = test_scheduler_state();
        let (active, send_outcome, source_position_gate, _) =
            pending_test_prefetch(1, counters.clone()).await;
        let mut resolving = Box::pin(resolve_source_task_at_boundary(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            active,
            1,
            vec![7],
            |_| Ok(BoundarySourceReadiness::Unavailable),
            counters,
            state,
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), resolving.as_mut())
                .await
                .is_err()
        );
        let outcome = PrefetchOutcome {
            target_position: 1,
            fatal_error: Some("physical identity corrupt".into()),
            ..PrefetchOutcome::default()
        };
        assert!(send_outcome.send(outcome).is_ok());
        let error = match resolving.await {
            Err(error) => error,
            Ok(_) => panic!("fatal source task error must fail qualification"),
        };
        assert_eq!(error, "physical identity corrupt");
        assert!(source_position_gate.try_lock_owned().is_ok());
    }

    #[test]
    fn source_deduplication_uses_existing_singleflight() {
        assert_eq!(
            source_disposition(false, false, true),
            SourceDisposition::SourceInFlight
        );
    }

    #[test]
    fn source_ram_hit_skips_read() {
        assert_eq!(
            source_disposition(false, true, false),
            SourceDisposition::RamHit
        );
    }

    #[test]
    fn physical_current_expert_skips_all_source_work() {
        assert_eq!(
            source_disposition(true, true, true),
            SourceDisposition::AlreadyPhysical
        );
    }

    #[test]
    fn future_source_failure_degrades_to_ordinary_demand() {
        let outcome = PrefetchOutcome {
            target_position: 1,
            failed_ids: vec![7, 8],
            ..PrefetchOutcome::default()
        };
        let mut counters = OracleCounters::default();
        record_completed_source_outcome(&mut counters, &outcome, 0);
        record_demand_readiness(&mut counters, &[vec![false, false]]);
        assert_eq!(counters.source_failures_degraded_to_demand, 2);
        assert_eq!(counters.oracle_demand_fallback, 1);
    }

    #[test]
    fn source_late_at_boundary_degrades_without_fake_readiness() {
        let mut counters = OracleCounters::default();
        record_source_late(
            &mut counters,
            8,
            LateSourcePolicy::NonblockingDemandFallback,
        );
        record_demand_readiness(&mut counters, &[vec![false; 8]]);
        assert_eq!(counters.source_late_at_boundary, 8);
        assert_eq!(counters.source_late_degraded_to_demand, 1);
        assert_eq!(counters.oracle_ready_at_demand, 0);
        assert_eq!(counters.oracle_demand_fallback, 1);
    }

    #[test]
    fn boundary_replacement_preserves_exact_physical_capacity() {
        assert!(plan_boundary_replacement(8, 8, &[false; 9]).is_err());
        assert!(plan_boundary_replacement(8, 9, &[false; 8]).is_err());
        let plan = plan_boundary_replacement(8, 4, &[false; 8]).unwrap();
        assert_eq!(plan.physical_experts_needed, 8);
        assert_eq!(plan.minimum_evictions, 4);
    }

    #[test]
    fn two_gib_top_k_equals_capacity_requires_destructive_replacement() {
        let plan = plan_boundary_replacement(8, 8, &[false; 8]).unwrap();
        assert_eq!(plan.physical_experts_needed, 8);
        assert_eq!(plan.minimum_evictions, 8);
    }

    #[test]
    fn protected_future_members_are_not_counted_as_victims() {
        let plan =
            plan_boundary_replacement(8, 8, &[true, true, true, true, true, true, true, false])
                .unwrap();
        assert_eq!(plan.physical_experts_needed, 1);
        assert_eq!(plan.minimum_evictions, 1);
    }

    #[test]
    fn stale_logical_generation_fails_closed_to_demand() {
        let error = GpuNativeTieredResidencyError::LogicalAdmissionStale {
            global_id: 9,
            generation: 4,
        };
        assert_eq!(
            boundary_install_error_policy(&error),
            BoundaryInstallErrorPolicy::DegradeStaleGeneration
        );
    }

    #[test]
    fn physical_identity_corruption_is_a_fatal_qualification_failure() {
        let error = GpuNativeTieredResidencyError::PhysicalIdentityCorrupt { global_id: 9 };
        assert_eq!(
            boundary_install_error_policy(&error),
            BoundaryInstallErrorPolicy::FailClosed
        );
    }

    #[test]
    fn unsafe_or_reused_boundary_is_a_fatal_qualification_failure() {
        let error = GpuNativeTieredResidencyError::UnsafeOracleBoundary {
            layer_index: 4,
            detail: "reused".into(),
        };
        assert_eq!(
            boundary_install_error_policy(&error),
            BoundaryInstallErrorPolicy::FailClosed
        );
    }

    #[test]
    fn no_fake_physical_hit_is_reported() {
        let mut counters = OracleCounters::default();
        record_demand_readiness(&mut counters, &[vec![false, false], vec![false, false]]);
        assert_eq!(counters.oracle_ready_at_demand, 0);
        assert_eq!(counters.oracle_not_ready_at_demand, 4);
    }

    #[test]
    fn demand_fallback_remains_active_for_any_incomplete_layer_set() {
        let mut counters = OracleCounters::default();
        record_demand_readiness(
            &mut counters,
            &[vec![true, true], vec![true, false], vec![false, true]],
        );
        assert_eq!(counters.oracle_ready_at_demand, 4);
        assert_eq!(counters.oracle_not_ready_at_demand, 2);
        assert_eq!(counters.oracle_demand_fallback, 2);
    }

    #[test]
    fn full_15576_mib_capacity_needs_zero_unnecessary_replacement() {
        let plan = plan_boundary_replacement(128, 128, &[true; EXPECTED_TOP_K]).unwrap();
        assert_eq!(plan.physical_experts_needed, 0);
        assert_eq!(plan.minimum_evictions, 0);
    }

    #[test]
    fn actual_route_mismatch_fails_qualification() {
        let trace = test_trace(1);
        let mut cursor = StrictRouteCursor::new(trace);
        let mut actual = cursor.routes_at(0).unwrap();
        actual[0].swap(0, 1);
        assert!(cursor.reconcile(0, &actual).is_err());
    }

    #[test]
    fn token_or_text_hash_mismatch_fails_qualification() {
        assert_eq!(
            frozen_output_matches("token", "text", "token", "text"),
            (true, true)
        );
        assert_eq!(
            frozen_output_matches("wrong", "text", "token", "text"),
            (false, true)
        );
        assert_eq!(
            frozen_output_matches("token", "wrong", "token", "text"),
            (true, false)
        );
    }

    #[test]
    fn warmup_counters_are_excluded_from_measured_aggregate() {
        let warmup = OracleCounters {
            future_experts_considered: 99,
            ..OracleCounters::default()
        };
        let measured = OracleCounters {
            future_experts_considered: 7,
            ..OracleCounters::default()
        };
        let mut aggregate = OracleCounters::default();
        for _ in 0..FROZEN_MEASURED_RUNS {
            accumulate_oracle(&mut aggregate, &measured);
        }
        assert_eq!(aggregate.future_experts_considered, 21);
        assert_ne!(
            aggregate.future_experts_considered,
            21 + warmup.future_experts_considered
        );
    }

    #[test]
    fn source_physical_and_accounting_counters_reconcile() {
        assert!(validate_oracle_counters(
            &valid_source_only_counters(),
            OracleScheduledResidencyMode::SourceOnly,
            4,
        )
        .is_ok());
    }

    #[test]
    fn source_only_counter_validation_rejects_physical_mutation() {
        let mut counters = valid_source_only_counters();
        counters.boundary_replacement_sets_started = 1;
        counters.boundary_physical_installs = 1;
        assert!(
            validate_oracle_counters(&counters, OracleScheduledResidencyMode::SourceOnly, 4,)
                .is_err()
        );
    }

    #[test]
    fn serialized_h2d_contract_uses_one_ordered_queue_without_flush_submit() {
        let contract = treatment_contract(OracleScheduledResidencyMode::TokenBoundaryDirect);
        assert!(contract.token_boundary_h2d);
        assert!(contract.same_ordered_queue);
        assert!(contract.production_direct_staging_used);
        assert!(!contract.extra_flush_submit);
        assert!(!contract.h2d_compute_overlap_claimed);
    }

    #[test]
    fn initial_position_priming_is_explicitly_counted() {
        let trace = test_trace(1);
        let routes = StrictRouteCursor::new(trace).routes_at(0).unwrap();
        let mut counters = OracleCounters::default();
        record_source_position_considered(&mut counters, &routes);
        counters.initial_position_source_priming_experts = 4;
        assert_eq!(counters.future_experts_considered, 4);
        assert_eq!(counters.initial_position_source_priming_experts, 4);
    }

    #[test]
    fn qualification_temporary_state_bound_is_fail_closed() {
        let mut counters = valid_source_only_counters();
        counters.qualification_owned_peak_slots = FROZEN_RAM_CACHE_SLOTS as u64 + 1;
        assert!(
            validate_oracle_counters(&counters, OracleScheduledResidencyMode::SourceOnly, 4,)
                .is_err()
        );
    }

    #[tokio::test]
    async fn all_background_tasks_drain_before_request_shutdown() {
        let completed = Arc::new(AtomicU64::new(0));
        let mut tasks = Vec::new();
        for _ in 0..4 {
            let completed = completed.clone();
            tasks.push(tokio::spawn(async move {
                tokio::task::yield_now().await;
                completed.fetch_add(1, Ordering::Relaxed);
            }));
        }
        drain_background_tasks(&mut tasks).await.unwrap();
        assert!(tasks.is_empty());
        assert_eq!(completed.load(Ordering::Relaxed), 4);
    }
}
