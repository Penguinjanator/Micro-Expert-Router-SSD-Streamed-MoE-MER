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

use crate::backend::gpu_native::{
    GpuNativePhysicalInstallEvidence, GpuNativePhysicalSlotFillPolicy, GpuNativeQ4ExpertGeometry,
    GpuNativeQ4ExpertKey, GpuNativeQ4ExpertResidency,
};
use crate::backend::GpuDeviceIdentity;
use crate::buffer_pool::{BufferPool, BufferPoolOrigin};
use crate::engine::{Engine, RoutedExpertExecutionSnapshot};
use crate::expert_cache::{ExpertResident, GpuDemandSetAdmission, GpuExpertCache, GpuResident};
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
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub(crate) const SCHEMA: &str = "mer.gpu-native-oracle-scheduled-residency.v5";
pub(crate) const MODE: &str = "qualify-gpu-native-oracle-scheduled-residency";
const FROZEN_PAYLOAD_CONTROL_SHA: &str = "4eb143b34b00826e2031b313f3734bace18a7ecc";
const FROZEN_PAYLOAD_CONTROL_TREE: &str = "37600e48e6dc3552f4a5b33e643351d4243280f5";
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
const ORACLE_FUTURE_SOURCE_POOL_SLOTS: usize = EXPECTED_LAYERS * EXPECTED_TOP_K;
const PREDECESSOR_FIRST_ATTEMPT_CODE_SHA: &str = "d4e34cf302ee7f8d78b45d8e6dd8e7e524e716f3";
const PREDECESSOR_SECOND_ATTEMPT_CODE_SHA: &str = "34b55715452acc355fd6dcf21e9c084d038efcd9";
const PREDECESSOR_SECOND_ATTEMPT_REPORT_SHA256: &str =
    "a80d79457bb0577bc7231d4567db6126fd1b8b0ccf71b0d5ef73af468a04a6a0";
const PREDECESSOR_SECOND_ATTEMPT_RUNNER_SHA256: &str =
    "3e11741e4d7efc58f1ad771d58b810668724968571b881aeb505f1dd2f939c9f";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum OracleScheduledResidencyMode {
    SourceOnly,
    TokenBoundaryDirect,
    TokenBoundaryDirectLogicalOnly,
    TokenBoundaryDirectLogicalOnlyNoZeroFill,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum LogicalPayloadMode {
    Materialized,
    QualificationLogicalOnly,
}

const fn logical_payload_mode(mode: OracleScheduledResidencyMode) -> LogicalPayloadMode {
    match mode {
        OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnly
        | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnlyNoZeroFill => {
            LogicalPayloadMode::QualificationLogicalOnly
        }
        _ => LogicalPayloadMode::Materialized,
    }
}

const fn physical_fill_policy(
    mode: OracleScheduledResidencyMode,
) -> GpuNativePhysicalSlotFillPolicy {
    match mode {
        OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnlyNoZeroFill => {
            GpuNativePhysicalSlotFillPolicy::QualificationNoZeroFill
        }
        _ => GpuNativePhysicalSlotFillPolicy::FullSlotZero,
    }
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
        OracleScheduledResidencyMode::TokenBoundaryDirect
        | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnly
        | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnlyNoZeroFill => {
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
    logical_materializations: u64,
    logical_materialization_bytes: u64,
    logical_only_admissions: u64,
    logical_only_charged_bytes: u64,
    logical_expected_payload_bytes: u64,
    logical_only_data_accesses: u64,
    logical_admission_transactions: u64,
    logical_admissions_returned: u64,
    logical_new_generations: u64,
    logical_current_generations_validated: u64,
    logical_newly_admitted_bytes: u64,
    logical_gpu_used_bytes_before: u64,
    logical_gpu_used_bytes_after: u64,
    logical_gpu_evicted_bytes: u64,
    initial_position_source_priming_experts: u64,
    initial_position_source_priming_us: u64,
    future_experts_considered: u64,
    source_skipped_physical_current: u64,
    source_ram_hits: u64,
    source_prefetch_ram_hits: u64,
    source_singleflight_hits: u64,
    source_position_inflight_skips: u64,
    source_reads_started: u64,
    source_reads_completed: u64,
    source_reads_failed: u64,
    source_bytes_read: u64,
    oracle_source_nvme_reads: u64,
    oracle_source_bytes: u64,
    future_experts_deferred_current_demand_overlap: u64,
    deferred_overlap_resolved_physical: u64,
    deferred_overlap_resolved_production_ram: u64,
    residual_boundary_source_reads: u64,
    residual_boundary_source_failures: u64,
    residual_boundary_source_bytes: u64,
    residual_source_wait_us: u64,
    oracle_source_pool_exhaustion_count: u64,
    source_prefetch_batches_started: u64,
    source_prefetch_batches_completed: u64,
    source_prefetch_batches_skipped_inflight: u64,
    source_prefetch_peak_inflight: u64,
    source_prefetch_read_batches_with_work: u64,
    source_prefetch_positions_with_multiple_read_batches: u64,
    source_prefetch_reads_current_inflight: u64,
    source_prefetch_reads_peak_inflight: u64,
    source_prefetch_read_inflight_accounting_errors: u64,
    source_ready_before_boundary: u64,
    source_late_at_boundary: u64,
    source_end_of_trace: u64,
    future_physical_sets_considered: u64,
    future_physical_experts_needed: u64,
    boundary_replacement_sets_started: u64,
    boundary_replacement_sets_completed: u64,
    boundary_physical_installs: u64,
    boundary_physical_install_order_checks: u64,
    boundary_physical_install_order_errors: u64,
    boundary_physical_install_bytes: u64,
    physical_slot_zero_fill_bytes: u64,
    physical_slot_epoch_write_bytes: u64,
    physical_slot_payload_copy_bytes: u64,
    physical_slot_prepare_us: u64,
    physical_queue_staging_us: u64,
    mapping_publication_us: u64,
    individual_physical_stage_us: u64,
    physical_install_total_us: u64,
    physical_install_evidence_errors: u64,
    physical_install_timing_errors: u64,

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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
struct OracleFutureSourcePoolSnapshot {
    capacity_slots: usize,
    buffer_size_bytes: usize,
    allocated_bytes: usize,
    current_in_use_slots: usize,
    peak_in_use_slots: usize,
    exhaustion_count: u64,
    nvme_reads: u64,
    nvme_bytes: u64,
}

/// One bounded, qualification-owned source plane shared by warmup and all
/// measured requests. It never inserts into the production RAM cache.
pub(crate) struct OracleFutureSourcePool {
    pool: BufferPool,
    peak_in_use_slots: AtomicUsize,
    exhaustion_count: AtomicU64,
    nvme_reads: AtomicU64,
    nvme_bytes: AtomicU64,
}

impl OracleFutureSourcePool {
    pub(crate) fn new(capacity: usize, buffer_size: usize, block_align: usize) -> Self {
        Self {
            pool: BufferPool::new_qualification_oracle_future_source(
                capacity,
                buffer_size,
                block_align,
            ),
            peak_in_use_slots: AtomicUsize::new(0),
            exhaustion_count: AtomicU64::new(0),
            nvme_reads: AtomicU64::new(0),
            nvme_bytes: AtomicU64::new(0),
        }
    }

    fn try_acquire(&self) -> Result<crate::buffer_pool::PooledBuffer, String> {
        let Some(buffer) = self.pool.try_acquire() else {
            self.exhaustion_count.fetch_add(1, Ordering::Relaxed);
            return Err(format!(
                "ORACLE future-source pool exhausted at capacity {}",
                self.pool.capacity()
            ));
        };
        let current = self
            .pool
            .capacity()
            .saturating_sub(self.pool.primary_available());
        self.peak_in_use_slots.fetch_max(current, Ordering::Relaxed);
        Ok(buffer)
    }

    fn snapshot(&self) -> OracleFutureSourcePoolSnapshot {
        OracleFutureSourcePoolSnapshot {
            capacity_slots: self.pool.capacity(),
            buffer_size_bytes: self.pool.buffer_size(),
            allocated_bytes: self.pool.allocated_bytes(),
            current_in_use_slots: self
                .pool
                .capacity()
                .saturating_sub(self.pool.primary_available()),
            peak_in_use_slots: self.peak_in_use_slots.load(Ordering::Relaxed),
            exhaustion_count: self.exhaustion_count.load(Ordering::Relaxed),
            nvme_reads: self.nvme_reads.load(Ordering::Relaxed),
            nvme_bytes: self.nvme_bytes.load(Ordering::Relaxed),
        }
    }

    #[cfg(test)]
    pub(crate) fn current_in_use_slots(&self) -> usize {
        self.snapshot().current_in_use_slots
    }
}

#[derive(Debug)]
enum OracleSourceReadError {
    PoolExhausted(String),
    Io(String),
    Fatal(String),
}

async fn read_oracle_future_source(
    engine: &Arc<Engine>,
    pool: &OracleFutureSourcePool,
    global_id: u32,
) -> Result<(Arc<ExpertResident>, usize), OracleSourceReadError> {
    read_oracle_future_source_from_storage(&engine.core.storage, pool, global_id).await
}

async fn read_oracle_future_source_from_storage(
    storage: &Arc<crate::io_provider::NvmeStorage>,
    pool: &OracleFutureSourcePool,
    global_id: u32,
) -> Result<(Arc<ExpertResident>, usize), OracleSourceReadError> {
    let mut buffer = pool
        .try_acquire()
        .map_err(OracleSourceReadError::PoolExhausted)?;
    let bytes = storage
        .read_expert(global_id, &mut buffer)
        .await
        .map_err(|error| OracleSourceReadError::Io(error.to_string()))?;
    pool.nvme_reads.fetch_add(1, Ordering::Relaxed);
    pool.nvme_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    let resident = Arc::new(ExpertResident::new_with_block_align(
        global_id,
        buffer,
        storage.config().block_align,
    ));
    if resident.buffer_pool_origin() != BufferPoolOrigin::QualificationOracleFutureSource {
        return Err(OracleSourceReadError::Fatal(
            "ORACLE future-source read returned a production-backed resident".into(),
        ));
    }
    Ok((resident, bytes))
}

fn saturating_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

const fn oracle_future_source_allocated_bytes(expert_file_bytes: usize) -> Option<usize> {
    ORACLE_FUTURE_SOURCE_POOL_SLOTS.checked_mul(expert_file_bytes)
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

struct DirectFutureReadGuard {
    counters: Arc<Mutex<OracleCounters>>,
    entered: bool,
}

impl DirectFutureReadGuard {
    fn enter(counters: Arc<Mutex<OracleCounters>>) -> Self {
        let mut values = counters.lock();
        let entered = match values.source_prefetch_reads_current_inflight.checked_add(1) {
            Some(current) => {
                values.source_prefetch_reads_current_inflight = current;
                values.source_prefetch_reads_peak_inflight =
                    values.source_prefetch_reads_peak_inflight.max(current);
                true
            }
            None => {
                values.source_prefetch_read_inflight_accounting_errors = values
                    .source_prefetch_read_inflight_accounting_errors
                    .saturating_add(1);
                false
            }
        };
        drop(values);
        Self { counters, entered }
    }
}

impl Drop for DirectFutureReadGuard {
    fn drop(&mut self) {
        if !self.entered {
            return;
        }
        let mut values = self.counters.lock();
        match values.source_prefetch_reads_current_inflight.checked_sub(1) {
            Some(current) => values.source_prefetch_reads_current_inflight = current,
            None => {
                values.source_prefetch_read_inflight_accounting_errors = values
                    .source_prefetch_read_inflight_accounting_errors
                    .saturating_add(1);
            }
        }
    }
}

async fn run_bounded_layer_tasks<I, F, Fut, T, A, M>(
    work: I,
    concurrency: usize,
    task_factory: F,
    mut aggregate: A,
    mut merge: M,
) -> Result<A, String>
where
    I: IntoIterator,
    I::Item: Send + 'static,
    F: Fn(I::Item) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = T> + Send + 'static,
    T: Send + 'static,
    M: FnMut(&mut A, T),
{
    if concurrency == 0 {
        return Err("qualification layer-task concurrency must be nonzero".into());
    }

    let mut work = work.into_iter();
    let mut tasks = tokio::task::JoinSet::new();
    for item in work.by_ref().take(concurrency) {
        let task_factory = task_factory.clone();
        tasks.spawn(async move { task_factory(item).await });
    }

    while let Some(joined) = tasks.join_next().await {
        match joined {
            Ok(value) => merge(&mut aggregate, value),
            Err(error) => {
                // A panic or cancellation is a qualification failure. Abort
                // and resolve every still-owned task so RAII source leases
                // and inflight guards are released before returning.
                tasks.abort_all();
                while tasks.join_next().await.is_some() {}
                return Err(format!("independent layer source task failed: {error}"));
            }
        }
        if let Some(item) = work.next() {
            let task_factory = task_factory.clone();
            tasks.spawn(async move { task_factory(item).await });
        }
    }
    Ok(aggregate)
}

#[derive(Default)]
struct PrefetchOutcome {
    target_position: usize,
    residents: HashMap<u32, Arc<ExpertResident>>,
    successful_ids: Vec<u32>,
    deferred_overlap_ids: Vec<u32>,
    failed_ids: Vec<u32>,
    fatal_error: Option<String>,
}

fn validate_retained_source_origins(
    mode: OracleScheduledResidencyMode,
    outcome: &PrefetchOutcome,
) -> Result<(), String> {
    match mode {
        OracleScheduledResidencyMode::SourceOnly if !outcome.residents.is_empty() => Err(
            "source-only retained request-local PRIMARY residents across token execution".into(),
        ),
        OracleScheduledResidencyMode::TokenBoundaryDirect
        | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnly
        | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnlyNoZeroFill => {
            for (&global_id, resident) in &outcome.residents {
                if resident.buffer_pool_origin()
                    != BufferPoolOrigin::QualificationOracleFutureSource
                {
                    return Err(format!(
                        "token-boundary-direct retained production-backed future expert {global_id}"
                    ));
                }
            }
            Ok(())
        }
        OracleScheduledResidencyMode::SourceOnly => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SourceReadinessSnapshot {
    required_ready: u64,
    source_residents_ready: u64,
}

#[derive(Default)]
struct SourceReadinessState {
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
            let physical_ready = live == BoundarySourceReadiness::Physical;
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
    _position_guard: tokio::sync::OwnedMutexGuard<()>,
}

impl PrefetchLease {
    fn new(outcome: PrefetchOutcome, position_guard: tokio::sync::OwnedMutexGuard<()>) -> Self {
        Self {
            outcome,
            _position_guard: position_guard,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceDisposition {
    AlreadyPhysical,
    RamHit,
    SourceInFlight,
    StartSourceRead,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectFutureSourceDisposition {
    AlreadyPhysical,
    DeferredCurrentDemandOverlap,
    IsolatedRead,
}

const fn direct_future_source_disposition(
    physical_current: bool,
    overlaps_current_demand: bool,
) -> DirectFutureSourceDisposition {
    if physical_current {
        DirectFutureSourceDisposition::AlreadyPhysical
    } else if overlaps_current_demand {
        DirectFutureSourceDisposition::DeferredCurrentDemandOverlap
    } else {
        DirectFutureSourceDisposition::IsolatedRead
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectBoundarySourceDisposition {
    Physical,
    Isolated,
    ProductionRam,
    ResidualRead,
}

const fn direct_boundary_source_disposition(
    physical_current: bool,
    isolated_ready: bool,
    production_ram_ready: bool,
) -> DirectBoundarySourceDisposition {
    if physical_current {
        DirectBoundarySourceDisposition::Physical
    } else if isolated_ready {
        DirectBoundarySourceDisposition::Isolated
    } else if production_ram_ready {
        DirectBoundarySourceDisposition::ProductionRam
    } else {
        DirectBoundarySourceDisposition::ResidualRead
    }
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
    mode: OracleScheduledResidencyMode,
    current_route_global_ids: Arc<HashSet<u32>>,
    oracle_source_pool: Option<Arc<OracleFutureSourcePool>>,
    counters: Arc<Mutex<OracleCounters>>,
    readiness: Arc<SourceReadiness>,
) -> PrefetchOutcome {
    let _guard = LayerBatchGuard::enter(counters.clone());
    let mut outcome = PrefetchOutcome::default();
    let mut direct_read_started_for_batch = false;
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
        if uses_isolated_oracle_source_pool(mode) {
            match direct_future_source_disposition(
                physical_current,
                current_route_global_ids.contains(&global_id),
            ) {
                DirectFutureSourceDisposition::AlreadyPhysical => {
                    counters.lock().source_skipped_physical_current += 1;
                    outcome.successful_ids.push(global_id);
                    continue;
                }
                DirectFutureSourceDisposition::DeferredCurrentDemandOverlap => {
                    counters
                        .lock()
                        .future_experts_deferred_current_demand_overlap += 1;
                    outcome.deferred_overlap_ids.push(global_id);
                    continue;
                }
                DirectFutureSourceDisposition::IsolatedRead => {}
            }
            let Some(pool) = oracle_source_pool.as_ref() else {
                outcome.fatal_error =
                    Some("token-boundary-direct has no isolated ORACLE source pool".into());
                return outcome;
            };
            if !direct_read_started_for_batch {
                counters.lock().source_prefetch_read_batches_with_work += 1;
                direct_read_started_for_batch = true;
            }
            counters.lock().source_reads_started += 1;
            let started = Instant::now();
            let read_guard = DirectFutureReadGuard::enter(counters.clone());
            let read = read_oracle_future_source(&engine, pool, global_id).await;
            drop(read_guard);
            match read {
                Ok((resident, bytes)) => {
                    let pool_snapshot = pool.snapshot();
                    let mut values = counters.lock();
                    add_counter(&mut values.source_read_aggregate_us, saturating_us(started));
                    values.source_reads_completed += 1;
                    add_counter(&mut values.source_bytes_read, bytes as u64);
                    values.oracle_source_nvme_reads += 1;
                    add_counter(&mut values.oracle_source_bytes, bytes as u64);
                    values.qualification_owned_current_slots =
                        pool_snapshot.current_in_use_slots as u64;
                    values.qualification_owned_current_bytes = (pool_snapshot.current_in_use_slots
                        as u64)
                        .saturating_mul(pool_snapshot.buffer_size_bytes as u64);
                    values.qualification_owned_peak_slots = values
                        .qualification_owned_peak_slots
                        .max(pool_snapshot.current_in_use_slots as u64);
                    values.qualification_owned_peak_bytes =
                        values.qualification_owned_peak_bytes.max(
                            (pool_snapshot.current_in_use_slots as u64)
                                .saturating_mul(pool_snapshot.buffer_size_bytes as u64),
                        );
                    drop(values);
                    readiness.mark_source_resident_ready(global_id);
                    outcome.successful_ids.push(global_id);
                    outcome.residents.insert(global_id, resident);
                }
                Err(OracleSourceReadError::Io(_error)) => {
                    let mut values = counters.lock();
                    add_counter(&mut values.source_read_aggregate_us, saturating_us(started));
                    values.source_reads_failed += 1;
                    outcome.failed_ids.push(global_id);
                }
                Err(OracleSourceReadError::PoolExhausted(error)) => {
                    let mut values = counters.lock();
                    values.source_reads_failed += 1;
                    values.oracle_source_pool_exhaustion_count += 1;
                    outcome.fatal_error = Some(error);
                    return outcome;
                }
                Err(OracleSourceReadError::Fatal(error)) => {
                    counters.lock().source_reads_failed += 1;
                    outcome.fatal_error = Some(error);
                    return outcome;
                }
            }
            continue;
        }

        if physical_current {
            counters.lock().source_skipped_physical_current += 1;
            outcome.successful_ids.push(global_id);
            continue;
        }

        // Source-only intentionally uses the ordinary cache/singleflight
        // machinery, but never parks the returned PRIMARY Arc in the
        // position outcome. Once fetch has installed the resident, the cache
        // is the only long-lived owner and foreground eviction can recycle
        // the buffer immediately.
        let ram_resident = engine.core.cache.get(global_id);
        let singleflight = ram_resident.is_none() && engine.core.in_flight.contains_key(&global_id);
        match source_disposition(false, ram_resident.is_some(), singleflight) {
            SourceDisposition::RamHit => {
                let mut values = counters.lock();
                values.source_ram_hits += 1;
                values.source_prefetch_ram_hits += 1;
                outcome.successful_ids.push(global_id);
                drop(ram_resident);
                continue;
            }
            SourceDisposition::SourceInFlight => {
                counters.lock().source_singleflight_hits += 1;
            }
            SourceDisposition::StartSourceRead => {
                counters.lock().source_reads_started += 1;
            }
            SourceDisposition::AlreadyPhysical => unreachable!("physical current handled above"),
        }
        let started = Instant::now();
        match engine.fetch_with_retry(global_id).await {
            Ok(resident) => {
                let elapsed = saturating_us(started);
                let mut values = counters.lock();
                add_counter(&mut values.source_read_aggregate_us, elapsed);
                if !singleflight {
                    values.source_reads_completed += 1;
                    add_counter(
                        &mut values.source_bytes_read,
                        engine.core.storage.config().expert_size as u64,
                    );
                }
                drop(values);
                outcome.successful_ids.push(global_id);
                drop(resident);
            }
            Err(_error) => {
                if !singleflight {
                    counters.lock().source_reads_failed += 1;
                }
                outcome.failed_ids.push(global_id);
            }
        }
    }
    outcome
}

fn merge_prefetch_outcome(outcome: &mut PrefetchOutcome, mut batch: PrefetchOutcome) {
    if outcome.fatal_error.is_none() {
        outcome.fatal_error = batch.fatal_error.take();
    }
    outcome.failed_ids.append(&mut batch.failed_ids);
    outcome.successful_ids.append(&mut batch.successful_ids);
    outcome
        .deferred_overlap_ids
        .append(&mut batch.deferred_overlap_ids);
    outcome.residents.extend(batch.residents);
}

async fn prefetch_position(
    engine: Arc<Engine>,
    target_position: usize,
    routes: Vec<Vec<u32>>,
    mode: OracleScheduledResidencyMode,
    current_route_global_ids: Arc<HashSet<u32>>,
    oracle_source_pool: Option<Arc<OracleFutureSourcePool>>,
    concurrency: usize,
    expert_bytes: usize,
    counters: Arc<Mutex<OracleCounters>>,
    readiness: Arc<SourceReadiness>,
) -> PrefetchOutcome {
    let started = Instant::now();
    let direct_read_batches_before = if uses_isolated_oracle_source_pool(mode) {
        counters.lock().source_prefetch_read_batches_with_work
    } else {
        0
    };
    let mut outcome = match mode {
        OracleScheduledResidencyMode::SourceOnly => {
            // Preserve the v2 SourceOnly scheduling behavior. Its storage
            // futures do not use the qualification direct-read path.
            let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
            let batches = stream::iter(routes.into_iter().enumerate().map(|(layer_index, ids)| {
                let engine = engine.clone();
                let counters = counters.clone();
                let readiness = readiness.clone();
                let semaphore = semaphore.clone();
                let current_route_global_ids = current_route_global_ids.clone();
                let oracle_source_pool = oracle_source_pool.clone();
                async move {
                    let permit = semaphore
                        .acquire_owned()
                        .await
                        .expect("qualification semaphore remains open");
                    let result = prefetch_layer_batch(
                        engine,
                        layer_index,
                        ids,
                        mode,
                        current_route_global_ids,
                        oracle_source_pool,
                        counters,
                        readiness,
                    )
                    .await;
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
                merge_prefetch_outcome(&mut outcome, batch);
            }
            outcome
        }
        OracleScheduledResidencyMode::TokenBoundaryDirect
        | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnly
        | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnlyNoZeroFill => {
            let task_engine = engine.clone();
            let task_counters = counters.clone();
            let task_readiness = readiness.clone();
            let task_current_route_global_ids = current_route_global_ids.clone();
            let task_oracle_source_pool = oracle_source_pool.clone();
            let task_factory = move |(layer_index, ids)| {
                let engine = task_engine.clone();
                let counters = task_counters.clone();
                let readiness = task_readiness.clone();
                let current_route_global_ids = task_current_route_global_ids.clone();
                let oracle_source_pool = task_oracle_source_pool.clone();
                async move {
                    prefetch_layer_batch(
                        engine,
                        layer_index,
                        ids,
                        OracleScheduledResidencyMode::TokenBoundaryDirect,
                        current_route_global_ids,
                        oracle_source_pool,
                        counters,
                        readiness,
                    )
                    .await
                }
            };
            match run_bounded_layer_tasks(
                routes.into_iter().enumerate(),
                concurrency,
                task_factory,
                PrefetchOutcome {
                    target_position,
                    ..PrefetchOutcome::default()
                },
                merge_prefetch_outcome,
            )
            .await
            {
                Ok(outcome) => outcome,
                Err(error) => PrefetchOutcome {
                    target_position,
                    fatal_error: Some(error),
                    ..PrefetchOutcome::default()
                },
            }
        }
    };
    if uses_isolated_oracle_source_pool(mode) {
        let mut values = counters.lock();
        if values
            .source_prefetch_read_batches_with_work
            .saturating_sub(direct_read_batches_before)
            > 1
        {
            values.source_prefetch_positions_with_multiple_read_batches += 1;
        }
    }
    let slots = outcome.residents.len() as u64;
    let bytes = slots.saturating_mul(expert_bytes as u64);
    let mut values = counters.lock();
    add_counter(&mut values.source_prefetch_wall_us, saturating_us(started));
    values.qualification_owned_current_slots = slots;
    values.qualification_owned_current_bytes = bytes;
    values.qualification_owned_peak_slots = values.qualification_owned_peak_slots.max(slots);
    values.qualification_owned_peak_bytes = values.qualification_owned_peak_bytes.max(bytes);
    if slots > ORACLE_FUTURE_SOURCE_POOL_SLOTS as u64
        || bytes > (ORACLE_FUTURE_SOURCE_POOL_SLOTS as u64).saturating_mul(expert_bytes as u64)
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
    expected_installs: Vec<(u32, u64)>,
    completed_installs: Mutex<usize>,
    layer_index: usize,
    experts_per_layer: u32,
    slot_bytes: u64,
    payload_bytes: u64,
    fill_policy: GpuNativePhysicalSlotFillPolicy,
}

fn record_physical_work_evidence(
    counters: &mut OracleCounters,
    fill_policy: GpuNativePhysicalSlotFillPolicy,
    slot_bytes: u64,
    payload_bytes: u64,
    evidence: GpuNativePhysicalInstallEvidence,
    total_us: u64,
) {
    let expected_zero_bytes = match fill_policy {
        GpuNativePhysicalSlotFillPolicy::FullSlotZero => slot_bytes,
        GpuNativePhysicalSlotFillPolicy::QualificationNoZeroFill => 0,
    };
    let bytes_match = evidence.physical_slot_bytes_staged == slot_bytes
        && evidence.physical_slot_zero_fill_bytes == expected_zero_bytes
        && evidence.physical_slot_epoch_write_bytes == 4
        && evidence.physical_slot_payload_copy_bytes == payload_bytes
        && evidence.direct_staging_writes == 1
        && evidence.full_slot_vec_materializations == 0
        && (fill_policy != GpuNativePhysicalSlotFillPolicy::QualificationNoZeroFill
            || payload_bytes.checked_add(4) == Some(slot_bytes));
    add_counter(
        &mut counters.physical_install_evidence_errors,
        u64::from(!bytes_match),
    );

    // Subphase timers are disjoint, individually rounded-down microseconds.
    // Compare against their enclosing existing timers without tolerances or
    // any cross-arm performance requirement. Zero-duration samples are valid.
    let stage_us = evidence.individual_physical_stage_us;
    let timings_match = evidence
        .physical_slot_prepare_us
        .checked_add(evidence.physical_queue_staging_us)
        .is_some_and(|subphases| subphases <= stage_us)
        && total_us
            .checked_sub(stage_us)
            .is_some_and(|commit_us| evidence.mapping_publication_us <= commit_us)
        && total_us != u64::MAX;
    add_counter(
        &mut counters.physical_install_timing_errors,
        u64::from(!timings_match),
    );
    macro_rules! add_evidence {
        ($field:ident) => {
            add_counter(&mut counters.$field, evidence.$field);
        };
    }
    add_evidence!(physical_slot_zero_fill_bytes);
    add_evidence!(physical_slot_epoch_write_bytes);
    add_evidence!(physical_slot_payload_copy_bytes);
    add_evidence!(physical_slot_prepare_us);
    add_evidence!(physical_queue_staging_us);
    add_evidence!(mapping_publication_us);
    add_evidence!(individual_physical_stage_us);
    add_counter(&mut counters.physical_install_total_us, total_us);
}

impl OracleBoundaryObserver {
    fn record_install_identity(
        &self,
        global_id: u32,
        key: GpuNativeQ4ExpertKey,
        physical_bytes: u64,
    ) {
        let mut completed = self.completed_installs.lock();
        let matches = self.expected_installs.get(*completed)
            == Some(&(global_id, key.logical_generation()))
            && key.layer_index() == self.layer_index
            && key.expert_id() == global_id % self.experts_per_layer
            && physical_bytes == self.slot_bytes;
        *completed += 1;
        drop(completed);
        let mut values = self.counters.lock();
        values.boundary_physical_install_order_checks += 1;
        values.boundary_physical_install_order_errors += u64::from(!matches);
    }
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
        global_id: u32,
        residency: GpuNativeQ4ExpertResidency,
        evidence: GpuNativePhysicalInstallEvidence,
        physical_install_total_us: u64,
    ) {
        self.record_install_identity(
            global_id,
            residency.key(),
            evidence.physical_slot_bytes_staged,
        );
        let mut values = self.counters.lock();
        // Reuse the existing install-completion lock and existing timers.
        record_physical_work_evidence(
            &mut values,
            self.fill_policy,
            self.slot_bytes,
            self.payload_bytes,
            evidence,
            physical_install_total_us,
        );
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

async fn resolve_direct_sources_at_boundary(
    engine: &Arc<Engine>,
    routes: &[Vec<u32>],
    prefetched: &PrefetchOutcome,
    pool: &Arc<OracleFutureSourcePool>,
    counters: &Arc<Mutex<OracleCounters>>,
) -> Result<HashMap<u32, Arc<ExpertResident>>, String> {
    validate_retained_source_origins(
        OracleScheduledResidencyMode::TokenBoundaryDirect,
        prefetched,
    )?;

    let manager = engine
        .core
        .gpu_native_residency
        .as_ref()
        .ok_or("GPU-native residency manager is absent")?;
    let experts_per_layer = manager.plan().geometry().num_experts() as u32;
    let deferred = prefetched
        .deferred_overlap_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut resolved = prefetched.residents.clone();
    for (layer_index, local_ids) in routes.iter().enumerate() {
        for global_id in future_global_ids(
            layer_index,
            local_ids,
            manager.plan().num_layers(),
            experts_per_layer,
        )
        .map_err(|error| error.to_string())?
        {
            let physical_current = manager
                .has_current_for_demand(global_id)
                .map_err(|error| error.to_string())?;
            let isolated_ready = resolved.contains_key(&global_id);
            let production_ram = (!physical_current && !isolated_ready)
                .then(|| engine.core.cache.get(global_id))
                .flatten();
            match direct_boundary_source_disposition(
                physical_current,
                isolated_ready,
                production_ram.is_some(),
            ) {
                DirectBoundarySourceDisposition::Physical => {
                    if deferred.contains(&global_id) {
                        counters.lock().deferred_overlap_resolved_physical += 1;
                    }
                    continue;
                }
                DirectBoundarySourceDisposition::Isolated => continue,
                DirectBoundarySourceDisposition::ProductionRam => {
                    let mut values = counters.lock();
                    values.source_ram_hits += 1;
                    if deferred.contains(&global_id) {
                        values.deferred_overlap_resolved_production_ram += 1;
                    }
                    drop(values);
                    // This PRIMARY Arc is acquired only after current-token
                    // completion and is released before the next token begins.
                    resolved.insert(
                        global_id,
                        production_ram.expect("production-RAM disposition has a resident"),
                    );
                    continue;
                }
                DirectBoundarySourceDisposition::ResidualRead => {}
            }

            let started = Instant::now();
            let read = read_oracle_future_source(engine, pool, global_id).await;
            let elapsed = saturating_us(started);
            let mut values = counters.lock();
            add_counter(&mut values.residual_source_wait_us, elapsed);
            match read {
                Ok((resident, bytes)) => {
                    values.residual_boundary_source_reads += 1;
                    values.oracle_source_nvme_reads += 1;
                    add_counter(&mut values.oracle_source_bytes, bytes as u64);
                    add_counter(&mut values.residual_boundary_source_bytes, bytes as u64);
                    drop(values);
                    resolved.insert(global_id, resident);
                }
                Err(OracleSourceReadError::Io(_error)) => {
                    values.residual_boundary_source_failures += 1;
                    values.source_failures_degraded_to_demand += 1;
                    drop(values);
                    // Correctness remains with the ordinary demand/recovery
                    // path when an allowed residual source read genuinely
                    // fails.
                }
                Err(OracleSourceReadError::PoolExhausted(error)) => {
                    values.oracle_source_pool_exhaustion_count += 1;
                    return Err(error);
                }
                Err(OracleSourceReadError::Fatal(error)) => return Err(error),
            }
        }
    }
    let snapshot = pool.snapshot();
    let mut values = counters.lock();
    values.qualification_owned_current_slots = snapshot.current_in_use_slots as u64;
    values.qualification_owned_current_bytes =
        (snapshot.current_in_use_slots as u64).saturating_mul(snapshot.buffer_size_bytes as u64);
    values.qualification_owned_peak_slots = values
        .qualification_owned_peak_slots
        .max(snapshot.current_in_use_slots as u64);
    values.qualification_owned_peak_bytes = values.qualification_owned_peak_bytes.max(
        (snapshot.current_in_use_slots as u64).saturating_mul(snapshot.buffer_size_bytes as u64),
    );
    drop(values);
    Ok(resolved)
}

fn prepare_logical_admissions(
    gpu: &GpuExpertCache,
    dtype: crate::inference::WeightDtype,
    global_ids: &[u32],
    residents: &HashMap<u32, Arc<ExpertResident>>,
    mode: OracleScheduledResidencyMode,
    counters: &Mutex<OracleCounters>,
    data_accesses: &Arc<AtomicU64>,
) -> Result<Vec<crate::expert_cache::GpuAdmission>, String> {
    let mut payloads = HashMap::with_capacity(global_ids.len());
    for attempt in 0..2 {
        let used_before = gpu.used_bytes();
        match gpu
            .demand_admit_set(global_ids, &payloads)
            .map_err(|error| error.to_string())?
        {
            GpuDemandSetAdmission::Ready {
                admissions,
                newly_admitted,
            } => {
                // Boundary execution is serialized and selected IDs are protected.
                // Validate the returned order, ordinary generation, and exact
                // payload charge without ever reading GpuResident::data().
                if admissions.len() != global_ids.len() || newly_admitted != payloads.len() {
                    return Err(
                        "ORACLE logical admission count changed during boundary preparation".into(),
                    );
                }
                for (&id, admission) in global_ids.iter().zip(&admissions) {
                    let source = residents
                        .get(&id)
                        .ok_or("ORACLE admission source disappeared")?;
                    if admission.resident().id != id
                        || !gpu.contains_generation(id, admission.generation())
                        || admission.generation() == 0
                        || admission.byte_len() != source.data().len()
                        || crate::backend::GpuStorage::byte_len(admission.resident().as_ref())
                            != source.data().len()
                    {
                        return Err(
                            "ORACLE logical admission identity/generation/byte charge mismatch"
                                .into(),
                        );
                    }
                }
                let new_bytes = payloads
                    .values()
                    .map(|payload| payload.byte_len() as u64)
                    .sum::<u64>();
                let used_after = gpu.used_bytes();
                let evicted_bytes = used_before
                    .checked_add(new_bytes)
                    .and_then(|sum| sum.checked_sub(used_after))
                    .ok_or("ORACLE logical GPU byte accounting underflow/overflow")?;
                if used_after > gpu.capacity_bytes() as u64 {
                    return Err("ORACLE logical GPU admission exceeded unchanged capacity".into());
                }
                let mut values = counters.lock();
                values.logical_admission_transactions += 1;
                values.logical_admissions_returned += admissions.len() as u64;
                values.logical_current_generations_validated += admissions.len() as u64;
                values.logical_new_generations += newly_admitted as u64;
                values.logical_newly_admitted_bytes += new_bytes;
                values.logical_gpu_used_bytes_before += used_before;
                values.logical_gpu_used_bytes_after += used_after;
                values.logical_gpu_evicted_bytes += evicted_bytes;
                return Ok(admissions);
            }
            GpuDemandSetAdmission::PayloadRequired(missing) if attempt == 0 => {
                for global_id in missing {
                    let resident = residents.get(&global_id).ok_or_else(|| {
                        format!("future source for global expert {global_id} is not ready")
                    })?;
                    let charged_bytes = resident.data().len();
                    if resident.id != global_id || charged_bytes == 0 {
                        return Err(
                            "ORACLE logical source identity or byte charge is invalid".into()
                        );
                    }
                    let payload = match logical_payload_mode(mode) {
                        LogicalPayloadMode::Materialized => {
                            let payload = GpuResident::new_with_dtype(
                                global_id,
                                resident.data().to_vec(),
                                dtype,
                            );
                            let mut values = counters.lock();
                            values.logical_materializations += 1;
                            values.logical_materialization_bytes += charged_bytes as u64;
                            payload
                        }
                        LogicalPayloadMode::QualificationLogicalOnly => {
                            let payload = GpuResident::new_qualification_logical_only(
                                global_id,
                                charged_bytes,
                                dtype,
                                data_accesses.clone(),
                            );
                            let mut values = counters.lock();
                            values.logical_only_admissions += 1;
                            values.logical_only_charged_bytes += charged_bytes as u64;
                            payload
                        }
                    };
                    counters.lock().logical_expected_payload_bytes += charged_bytes as u64;
                    payloads.insert(global_id, Arc::new(payload));
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
    mode: OracleScheduledResidencyMode,
    data_accesses: &Arc<AtomicU64>,
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
            continue;
        }

        let gpu = engine.execution_context().gpu_expert_cache().clone();
        let _logical_protection = gpu
            .protect_demand_set(&missing)
            .map_err(|error| error.to_string())?;
        let host_started = Instant::now();
        let admissions = match prepare_logical_admissions(
            &gpu,
            engine.core.options.dtype,
            &missing,
            residents,
            mode,
            &counters,
            data_accesses,
        ) {
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
        let observer = OracleBoundaryObserver {
            counters: counters.clone(),
            expected_installs: missing
                .iter()
                .map(|id| (*id, admissions_by_id[id].generation()))
                .collect(),
            completed_installs: Mutex::new(0),
            layer_index,
            experts_per_layer,
            slot_bytes: manager.plan().geometry().slot_stride_bytes() as u64,
            payload_bytes: manager.plan().geometry().logical_expert_bytes() as u64,
            fill_policy: physical_fill_policy(mode),
        };
        counters.lock().boundary_replacement_sets_started += 1;
        let evictions_before = counters.lock().boundary_physical_evictions;
        match manager.ensure_oracle_future_set_at_safe_boundary(
            boundary,
            layer_index,
            &demands,
            physical_fill_policy(mode),
            &observer,
        ) {
            Ok(_) => {
                let mut values = counters.lock();
                values.boundary_replacement_sets_completed += 1;
                if *observer.completed_installs.lock() != missing.len()
                    || values.boundary_physical_install_order_errors != 0
                {
                    return Err(
                        "ORACLE physical install count/order/generation/bytes mismatch".into(),
                    );
                }
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
    oracle_source_pool: Option<Arc<OracleFutureSourcePool>>,
    source_position_gate: Arc<tokio::sync::Mutex<()>>,
    counters: Arc<Mutex<OracleCounters>>,
    logical_data_accesses: Arc<AtomicU64>,
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
    if uses_isolated_oracle_source_pool(mode) && !state.lock().background.is_empty() {
        return Err(
            "token-boundary-direct retained a previous-position background source task".into(),
        );
    }

    let Some(handle) = active.handle else {
        if uses_isolated_oracle_source_pool(mode) {
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
        mode,
    );
    Ok(BoundarySourceResolution::Completed { lease, was_late })
}

fn release_source_position_lease_before_next_token(
    mode: OracleScheduledResidencyMode,
    lease: PrefetchLease,
    source_position_gate: &Arc<tokio::sync::Mutex<()>>,
    state: &Arc<Mutex<SchedulerState>>,
    oracle_source_pool: Option<&Arc<OracleFutureSourcePool>>,
    counters: &Arc<Mutex<OracleCounters>>,
) -> Result<(), String> {
    drop(lease);
    if uses_isolated_oracle_source_pool(mode) {
        let pool = oracle_source_pool
            .ok_or("token-boundary-direct lost its isolated ORACLE source pool")?;
        let snapshot = pool.snapshot();
        {
            let mut values = counters.lock();
            values.qualification_owned_current_slots = snapshot.current_in_use_slots as u64;
            values.qualification_owned_current_bytes = (snapshot.current_in_use_slots as u64)
                .saturating_mul(snapshot.buffer_size_bytes as u64);
        }
        if snapshot.current_in_use_slots != 0 {
            return Err(format!(
                "prior-position ORACLE source lease retained {} buffers before the next token",
                snapshot.current_in_use_slots
            ));
        }
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
        oracle_source_pool: Option<Arc<OracleFutureSourcePool>>,
        logical_data_accesses: Arc<AtomicU64>,
    ) -> Self {
        Self {
            engine,
            mode,
            concurrency,
            expert_bytes,
            oracle_source_pool,
            source_position_gate: Arc::new(tokio::sync::Mutex::new(())),
            counters: Arc::new(Mutex::new(OracleCounters::default())),
            logical_data_accesses,
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
                OracleScheduledResidencyMode::SourceOnly,
                Arc::new(HashSet::new()),
                None,
                self.concurrency,
                self.expert_bytes,
                self.counters.clone(),
                readiness,
            )
            .await,
            position_guard,
        );
        if let Some(error) = lease.outcome.fatal_error.as_ref() {
            let error = error.clone();
            return Err(error);
        }
        let mut values = self.counters.lock();
        values.initial_position_source_priming_experts = lease.outcome.successful_ids.len() as u64;
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
        values.source_prefetch_ram_hits += ram;
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

        if uses_isolated_oracle_source_pool(self.mode) {
            let pool = self
                .oracle_source_pool
                .as_ref()
                .ok_or("token-boundary-direct has no isolated ORACLE source pool")?;
            let resolved = resolve_direct_sources_at_boundary(
                &self.engine,
                &routes,
                &lease.outcome,
                pool,
                &self.counters,
            )
            .await?;
            install_future_at_boundary(
                &self.engine,
                &mut boundary,
                &routes,
                &resolved,
                self.counters.clone(),
                self.mode,
                &self.logical_data_accesses,
            )?;
            let unavailable_after_install = self.future_not_physically_current_count(&routes)?;
            record_direct_late_fallback(
                &mut self.counters.lock(),
                was_late,
                unavailable_after_install,
            );
            drop(resolved);
        } else {
            validate_retained_source_origins(
                OracleScheduledResidencyMode::SourceOnly,
                &lease.outcome,
            )?;
        }
        release_source_position_lease_before_next_token(
            self.mode,
            lease,
            &self.source_position_gate,
            &self.state,
            self.oracle_source_pool.as_ref(),
            &self.counters,
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
        counters.logical_only_data_accesses = self.logical_data_accesses.load(Ordering::Relaxed);
        if let Some(pool) = self.oracle_source_pool.as_ref() {
            let snapshot = pool.snapshot();
            counters.qualification_owned_current_slots = snapshot.current_in_use_slots as u64;
            counters.qualification_owned_current_bytes = (snapshot.current_in_use_slots as u64)
                .saturating_mul(snapshot.buffer_size_bytes as u64);
            if snapshot.current_in_use_slots != 0 {
                return Err(format!(
                    "ORACLE source pool retained {} leases after request completion",
                    snapshot.current_in_use_slots
                ));
            }
        }
        Ok((counters, trace))
    }
}

fn record_completed_source_outcome(
    counters: &mut OracleCounters,
    outcome: &PrefetchOutcome,
    source_residents_ready_at_boundary: u64,
    mode: OracleScheduledResidencyMode,
) {
    counters.source_ready_before_boundary += source_residents_ready_at_boundary;
    if mode == OracleScheduledResidencyMode::SourceOnly {
        counters.source_failures_degraded_to_demand += outcome.failed_ids.len() as u64;
    }
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
        let mode = self.mode;
        let oracle_source_pool = self.oracle_source_pool.clone();
        let target_position = position + 1;
        let current_routes = match state.cursor.routes_at(position) {
            Ok(routes) => routes,
            Err(error) => {
                state.failure = Some(error);
                return;
            }
        };
        let current_route_global_ids = match self.required_future_global_ids(&current_routes) {
            Ok(ids) => Arc::new(ids.into_iter().collect::<HashSet<_>>()),
            Err(error) => {
                state.failure = Some(error);
                return;
            }
        };
        let readiness = Arc::new(SourceReadiness::default());
        let handle = match self.source_position_gate.clone().try_lock_owned() {
            Ok(position_guard) => {
                let readiness_for_task = readiness.clone();
                let handle = tokio::spawn(async move {
                    let outcome = prefetch_position(
                        engine,
                        target_position,
                        routes,
                        mode,
                        current_route_global_ids,
                        oracle_source_pool,
                        concurrency,
                        expert_bytes,
                        counters.clone(),
                        readiness_for_task,
                    )
                    .await;
                    PrefetchLease::new(outcome, position_guard)
                });
                self.counters.lock().background_tasks_spawned += 1;
                Some(handle)
            }
            Err(_) => {
                if uses_isolated_oracle_source_pool(self.mode) {
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
    physical_fill_policy: GpuNativePhysicalSlotFillPolicy,
    physical_zero_fill_accounting: &'static str,
    concurrent_direct_staging_used: bool,
    logical_payload_mode: LogicalPayloadMode,
    logical_capacity_accounting: &'static str,
    physical_byte_source: &'static str,
    logical_only_data_access_policy: &'static str,
    source_overlap: bool,
    late_source_policy: LateSourcePolicy,
    independent_layer_source_tasks: bool,
    per_layer_storage_batch_read_used: bool,
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
    future_source_pool_isolated_from_production_primary: bool,
    production_primary_pool_capacity_unchanged: bool,
}

const fn uses_isolated_oracle_source_pool(mode: OracleScheduledResidencyMode) -> bool {
    matches!(
        mode,
        OracleScheduledResidencyMode::TokenBoundaryDirect
            | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnly
            | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnlyNoZeroFill
    )
}

const fn oracle_source_pool_configured_max_slots(mode: OracleScheduledResidencyMode) -> usize {
    if uses_isolated_oracle_source_pool(mode) {
        ORACLE_FUTURE_SOURCE_POOL_SLOTS
    } else {
        0
    }
}

const fn treatment_contract(mode: OracleScheduledResidencyMode) -> TreatmentContract {
    TreatmentContract {
        physical_fill_policy: physical_fill_policy(mode),
        physical_zero_fill_accounting: "explicit host fill bytes only; epoch and payload writes counted separately; physical staged/H2D bytes are unchanged",
        concurrent_direct_staging_used: uses_isolated_oracle_source_pool(mode),
        logical_payload_mode: logical_payload_mode(mode),
        logical_capacity_accounting: "sum of successful boundary admission transaction snapshots: before + newly charged bytes = after + evicted bytes; intervening ordinary demand is excluded",
        physical_byte_source: "GpuNativeDemandExpert::Install resident: Arc<ExpertResident>; production staging reads ExpertResident::data()",
        logical_only_data_access_policy: "increment run-scoped audit and panic; any nonzero audit rejects qualification including caught panics",
        source_overlap: true,
        late_source_policy: late_source_policy(mode),
        independent_layer_source_tasks: matches!(mode, OracleScheduledResidencyMode::TokenBoundaryDirect | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnly | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnlyNoZeroFill),
        per_layer_storage_batch_read_used: false,
        host_preparation_overlap: false,
        token_boundary_h2d: matches!(mode, OracleScheduledResidencyMode::TokenBoundaryDirect | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnly | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnlyNoZeroFill),
        demand_fallback: true,
        h2d_compute_overlap_claimed: false,
        same_ordered_queue: true,
        extra_flush_submit: false,
        production_predictor_used: false,
        production_speculative_residency_used: false,
        // FullSlotZero now selects the explicit concurrent control; only the
        // historical no-zero arm matches ordinary production's fill policy.
        production_direct_staging_used: matches!(mode, OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnlyNoZeroFill),
        legacy_full_slot_vec_used: false,
        future_source_pool_isolated_from_production_primary: uses_isolated_oracle_source_pool(mode),
        production_primary_pool_capacity_unchanged: true,
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
    logical_payload_control_git_sha: &'static str,
    logical_payload_control_tree_sha: &'static str,
    frozen_parent_git_sha: &'static str,
    oracle_route_command_git_sha: &'static str,
    performance_controls_rerun: bool,
    hardware_results_embedded: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct PredecessorFirstAttemptEvidence {
    code_sha: &'static str,
    result: &'static str,
    failure: &'static str,
    requested: usize,
    acquired: usize,
    interpretation: &'static str,
    first_attempt_rerun: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct PredecessorSecondAttemptEvidence {
    code_sha: &'static str,
    result: &'static str,
    report_sha256: &'static str,
    runner_sha256: &'static str,
    configured_source_concurrency: usize,
    observed_source_prefetch_peak_inflight: u64,
    decode_tps: f64,
    ordinary_residency_misses: u64,
    second_attempt_rerun: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct SourceMemoryPlaneEvidence {
    production_ram_cache_capacity_slots: usize,
    production_ram_cache_max_resident_bytes: usize,
    production_primary_pool_capacity_slots: usize,
    production_primary_headroom_slots: usize,
    production_primary_pool_available_before_requests: usize,
    production_primary_pool_available_after_requests: usize,
    production_primary_pool_buffer_size_bytes: usize,
    production_primary_pool_allocated_bytes: usize,
    oracle_source_pool_configured_max_slots: usize,
    oracle_source_pool_allocated_capacity_slots: usize,
    oracle_source_pool_buffer_size_bytes: usize,
    oracle_source_pool_allocated_bytes: usize,
    oracle_source_pool_current_in_use_slots: usize,
    oracle_source_pool_peak_in_use_slots: usize,
    oracle_source_pool_exhaustion_count: u64,
    oracle_source_nvme_reads: u64,
    oracle_source_bytes: u64,
    oracle_source_pool_reused_across_warmup_and_measured_requests: bool,
    no_oracle_pool_accumulation_across_requests: bool,
    oracle_source_buffers_released_before_runtime_shutdown: bool,
    future_source_pool_isolated_from_production_primary: bool,
    production_primary_pool_capacity_unchanged: bool,
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
    logical_payload_mode: LogicalPayloadMode,
    oracle_source_concurrency: usize,
    treatment_contract: TreatmentContract,
    physical_slot_plan: PhysicalSlotPlanEvidence,
    frozen_controls: FrozenControlReferences,
    frozen_workload: FrozenWorkloadEvidence,
    predecessor_first_attempt: PredecessorFirstAttemptEvidence,
    predecessor_second_attempt: PredecessorSecondAttemptEvidence,
    source_memory_planes: SourceMemoryPlaneEvidence,
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
    add!(logical_materializations);
    add!(logical_materialization_bytes);
    add!(logical_only_admissions);
    add!(logical_only_charged_bytes);
    add!(logical_expected_payload_bytes);
    add!(logical_only_data_accesses);
    add!(logical_admission_transactions);
    add!(logical_admissions_returned);
    add!(logical_new_generations);
    add!(logical_current_generations_validated);
    add!(logical_newly_admitted_bytes);
    add!(logical_gpu_used_bytes_before);
    add!(logical_gpu_used_bytes_after);
    add!(logical_gpu_evicted_bytes);
    add!(initial_position_source_priming_experts);
    add!(initial_position_source_priming_us);
    add!(future_experts_considered);
    add!(source_skipped_physical_current);
    add!(source_ram_hits);
    add!(source_prefetch_ram_hits);
    add!(source_singleflight_hits);
    add!(source_position_inflight_skips);
    add!(source_reads_started);
    add!(source_reads_completed);
    add!(source_reads_failed);
    add!(source_bytes_read);
    add!(oracle_source_nvme_reads);
    add!(oracle_source_bytes);
    add!(future_experts_deferred_current_demand_overlap);
    add!(deferred_overlap_resolved_physical);
    add!(deferred_overlap_resolved_production_ram);
    add!(residual_boundary_source_reads);
    add!(residual_boundary_source_failures);
    add!(residual_boundary_source_bytes);
    add!(residual_source_wait_us);
    add!(oracle_source_pool_exhaustion_count);
    add!(source_prefetch_batches_started);
    add!(source_prefetch_batches_completed);
    add!(source_prefetch_batches_skipped_inflight);
    total.source_prefetch_peak_inflight = total
        .source_prefetch_peak_inflight
        .max(value.source_prefetch_peak_inflight);
    add!(source_prefetch_read_batches_with_work);
    add!(source_prefetch_positions_with_multiple_read_batches);
    total.source_prefetch_reads_current_inflight = value.source_prefetch_reads_current_inflight;
    total.source_prefetch_reads_peak_inflight = total
        .source_prefetch_reads_peak_inflight
        .max(value.source_prefetch_reads_peak_inflight);
    add!(source_prefetch_read_inflight_accounting_errors);
    add!(source_ready_before_boundary);
    add!(source_late_at_boundary);
    add!(source_end_of_trace);
    add!(future_physical_sets_considered);
    add!(future_physical_experts_needed);
    add!(boundary_replacement_sets_started);
    add!(boundary_replacement_sets_completed);
    add!(boundary_physical_installs);
    add!(boundary_physical_install_order_checks);
    add!(boundary_physical_install_order_errors);
    add!(boundary_physical_install_bytes);
    add!(physical_slot_zero_fill_bytes);
    add!(physical_slot_epoch_write_bytes);
    add!(physical_slot_payload_copy_bytes);
    add!(physical_slot_prepare_us);
    add!(physical_queue_staging_us);
    add!(mapping_publication_us);
    add!(individual_physical_stage_us);
    add!(physical_install_total_us);
    add!(physical_install_evidence_errors);
    add!(physical_install_timing_errors);

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

fn validate_source_concurrency_mechanism(
    counters: &OracleCounters,
    mode: OracleScheduledResidencyMode,
    concurrency: usize,
) -> Result<(), BenchmarkFailure> {
    if mode == OracleScheduledResidencyMode::SourceOnly {
        if counters.source_prefetch_read_batches_with_work != 0
            || counters.source_prefetch_positions_with_multiple_read_batches != 0
            || counters.source_prefetch_reads_current_inflight != 0
            || counters.source_prefetch_reads_peak_inflight != 0
            || counters.source_prefetch_read_inflight_accounting_errors != 0
        {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "source-only-direct-read-accounting",
                "SourceOnly recorded TokenBoundaryDirect future-read concurrency evidence",
            ));
        }
        return Ok(());
    }

    if counters.source_prefetch_batches_started != counters.source_prefetch_batches_completed
        || counters.source_prefetch_read_batches_with_work
            > counters.source_prefetch_batches_started
        || (counters.source_prefetch_positions_with_multiple_read_batches != 0
            && counters.source_prefetch_read_batches_with_work < 2)
        || counters.source_prefetch_peak_inflight > concurrency as u64
        || counters.source_prefetch_reads_peak_inflight > concurrency as u64
        || counters.source_prefetch_reads_current_inflight != 0
        || counters.source_prefetch_read_inflight_accounting_errors != 0
    {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "oracle-direct-source-concurrency-accounting",
            format!(
                "direct source task/read concurrency did not reconcile at configured concurrency {concurrency}: {counters:?}"
            ),
        ));
    }

    let enough_independent_read_work =
        counters.source_prefetch_positions_with_multiple_read_batches > 0;
    if concurrency > 1
        && enough_independent_read_work
        && (counters.source_prefetch_peak_inflight <= 1
            || counters.source_prefetch_reads_peak_inflight <= 1)
    {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "oracle-direct-source-concurrency-not-observed",
            format!(
                "configured direct source concurrency {concurrency} had {} layer batches with direct-read work across {} positions with multiple read-bearing batches, but observed layer/read peaks {}/{}",
                counters.source_prefetch_read_batches_with_work,
                counters.source_prefetch_positions_with_multiple_read_batches,
                counters.source_prefetch_peak_inflight,
                counters.source_prefetch_reads_peak_inflight,
            ),
        ));
    }
    Ok(())
}

fn validate_logical_payload_counters(
    counters: &OracleCounters,
    mode: OracleScheduledResidencyMode,
) -> Result<(), BenchmarkFailure> {
    let c = counters;
    let before_plus_new = c
        .logical_gpu_used_bytes_before
        .checked_add(c.logical_newly_admitted_bytes);
    let after_plus_evicted = c
        .logical_gpu_used_bytes_after
        .checked_add(c.logical_gpu_evicted_bytes);
    let fail = || {
        BenchmarkFailure::new("postcondition", "oracle-logical-payload-reconciliation",
        format!("ORACLE logical payload, charge, generation or data-access evidence did not reconcile: {c:?}"))
    };
    if c.logical_only_data_accesses != 0
        || c.logical_admissions_returned != c.logical_current_generations_validated
        || before_plus_new.is_none()
        || before_plus_new != after_plus_evicted
        || c.logical_materialization_bytes
            .checked_add(c.logical_only_charged_bytes)
            != Some(c.logical_expected_payload_bytes)
    {
        return Err(fail());
    }
    match logical_payload_mode(mode) {
        LogicalPayloadMode::Materialized => {
            if c.logical_only_admissions != 0 || c.logical_only_charged_bytes != 0 {
                return Err(fail());
            }
        }
        LogicalPayloadMode::QualificationLogicalOnly => {
            if c.logical_materializations != 0
                || c.logical_materialization_bytes != 0
                || c.logical_only_admissions == 0
                || c.logical_only_charged_bytes == 0
                || c.logical_only_admissions != c.logical_new_generations
                || c.logical_only_charged_bytes != c.logical_newly_admitted_bytes
                || c.logical_admission_transactions == 0
                || c.logical_admissions_returned < c.logical_new_generations
                || c.logical_admissions_returned != c.boundary_physical_installs
                || c.boundary_physical_install_order_checks != c.boundary_physical_installs
                || c.boundary_physical_install_order_errors != 0
                || c.boundary_stale_generation != 0
                || c.boundary_install_failures != 0
            {
                return Err(fail());
            }
        }
    }
    Ok(())
}

fn frozen_physical_geometry() -> GpuNativeQ4ExpertGeometry {
    GpuNativeQ4ExpertGeometry::try_new(
        EXPECTED_D_MODEL,
        EXPECTED_D_FF,
        EXPECTED_EXPERTS,
        EXPECTED_TOP_K,
    )
    .expect("frozen ORACLE Q4 geometry is valid")
}

fn validate_physical_work_counters(
    c: &OracleCounters,
    mode: OracleScheduledResidencyMode,
) -> Result<(), BenchmarkFailure> {
    if logical_payload_mode(mode) != LogicalPayloadMode::QualificationLogicalOnly {
        return Ok(());
    }
    let geometry = frozen_physical_geometry();
    let slot_bytes = geometry.slot_stride_bytes() as u64;
    let payload_bytes = geometry.logical_expert_bytes() as u64;
    let installs = c.boundary_physical_installs;
    let expected_staged = installs.checked_mul(slot_bytes);
    let expected_zero = match physical_fill_policy(mode) {
        GpuNativePhysicalSlotFillPolicy::FullSlotZero => expected_staged,
        GpuNativePhysicalSlotFillPolicy::QualificationNoZeroFill => Some(0),
    };
    let stage_subphases = c
        .physical_slot_prepare_us
        .checked_add(c.physical_queue_staging_us);
    let total_us = c
        .individual_physical_stage_us
        .checked_add(c.token_boundary_commit_us);
    if installs == 0
        || expected_staged.is_none()
        || expected_staged != Some(c.boundary_physical_install_bytes)
        || expected_zero != Some(c.physical_slot_zero_fill_bytes)
        || installs.checked_mul(4) != Some(c.physical_slot_epoch_write_bytes)
        || installs.checked_mul(payload_bytes) != Some(c.physical_slot_payload_copy_bytes)
        || c.boundary_direct_staging_writes != installs
        || c.boundary_full_slot_vec_materializations != 0
        || c.boundary_physical_install_order_checks != installs
        || c.boundary_physical_install_order_errors != 0
        || c.boundary_stale_generation != 0
        || c.boundary_install_failures != 0
        || c.logical_only_data_accesses != 0
        || c.physical_install_evidence_errors != 0
        || c.physical_install_timing_errors != 0
        || !stage_subphases.is_some_and(|us| us <= c.individual_physical_stage_us)
        || c.mapping_publication_us > c.token_boundary_commit_us
        || total_us.is_none()
        || total_us == Some(u64::MAX)
        || total_us != Some(c.physical_install_total_us)
    {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "oracle-physical-work-reconciliation",
            format!(
                "ORACLE physical byte work, install or timing evidence did not reconcile: {c:?}"
            ),
        ));
    }
    Ok(())
}

fn validate_oracle_counters(
    counters: &OracleCounters,
    mode: OracleScheduledResidencyMode,
    concurrency: usize,
    expert_bytes: usize,
) -> Result<(), BenchmarkFailure> {
    validate_logical_only_concurrency(mode, concurrency)?;
    validate_source_concurrency_mechanism(counters, mode, concurrency)?;
    validate_logical_payload_counters(counters, mode)?;
    validate_physical_work_counters(counters, mode)?;
    let classified_source = counters
        .source_skipped_physical_current
        .saturating_add(counters.source_prefetch_ram_hits)
        .saturating_add(counters.source_singleflight_hits)
        .saturating_add(counters.source_position_inflight_skips)
        .saturating_add(counters.future_experts_deferred_current_demand_overlap)
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
        || counters.qualification_owned_peak_slots > ORACLE_FUTURE_SOURCE_POOL_SLOTS as u64
        || counters.qualification_owned_peak_bytes
            > (ORACLE_FUTURE_SOURCE_POOL_SLOTS as u64).saturating_mul(expert_bytes as u64)
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
            || counters.boundary_physical_evictions != 0
            || counters.oracle_source_nvme_reads != 0
            || counters.oracle_source_bytes != 0
            || counters.qualification_owned_peak_slots != 0)
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
    if uses_isolated_oracle_source_pool(mode)
        && (counters.source_position_inflight_skips != 0
            || counters.source_prefetch_batches_skipped_inflight != 0
            || counters.boundary_direct_staging_writes != counters.boundary_physical_installs
            || counters.boundary_physical_evictions > counters.boundary_physical_installs
            || counters.boundary_replacement_sets_completed
                > counters.boundary_replacement_sets_started
            || counters.oracle_source_pool_exhaustion_count != 0
            || counters.oracle_source_bytes
                != counters
                    .oracle_source_nvme_reads
                    .saturating_mul(expert_bytes as u64)
            || counters.residual_boundary_source_bytes
                != counters
                    .residual_boundary_source_reads
                    .saturating_mul(expert_bytes as u64)
            || counters
                .deferred_overlap_resolved_physical
                .saturating_add(counters.deferred_overlap_resolved_production_ram)
                > counters.future_experts_deferred_current_demand_overlap
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
    oracle_source_pool: Option<Arc<OracleFutureSourcePool>>,
    logical_data_accesses: Arc<AtomicU64>,
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
        oracle_source_pool,
        logical_data_accesses,
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
    validate_oracle_counters(&oracle, treatment_mode, source_concurrency, expert_bytes)?;

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

fn validate_logical_only_concurrency(
    mode: OracleScheduledResidencyMode,
    concurrency: usize,
) -> Result<(), BenchmarkFailure> {
    if logical_payload_mode(mode) == LogicalPayloadMode::QualificationLogicalOnly
        && concurrency != 4
    {
        return Err(artifact_failure(
            "logical-only-requires-frozen-c4",
            "logical-only payload treatment requires unchanged --oracle-source-concurrency=4",
        ));
    }
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
    validate_logical_only_concurrency(args.treatment_mode, args.oracle_source_concurrency)?;
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

    let expected_production_primary_capacity = FROZEN_RAM_CACHE_SLOTS + 1;
    let expected_oracle_source_allocated_bytes =
        oracle_future_source_allocated_bytes(spec.cfg.model.expert_size).ok_or_else(|| {
            artifact_failure(
                "oracle-source-pool-allocation-overflow",
                "384-slot ORACLE source allocation exceeds usize",
            )
        })?;

    let runtime = crate::build_isolated_greedy_runtime(
        &spec,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
        tokenizer,
    )
    .await?;
    let production_primary_pool_available_before_requests =
        runtime.engine.core.pool.primary_available();
    let oracle_allocation_before =
        crate::buffer_pool::expert_buffer_pool_qualification_oracle_bytes();
    let oracle_source_pool = match args.treatment_mode {
        OracleScheduledResidencyMode::SourceOnly => None,
        OracleScheduledResidencyMode::TokenBoundaryDirect
        | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnly
        | OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnlyNoZeroFill => {
            Some(Arc::new(OracleFutureSourcePool::new(
                ORACLE_FUTURE_SOURCE_POOL_SLOTS,
                spec.cfg.model.expert_size,
                spec.cfg.storage.block_align,
            )))
        }
    };
    let logical_data_accesses = Arc::new(AtomicU64::new(0));
    let execution = async {
        if runtime.engine.core.cache.capacity() != FROZEN_RAM_CACHE_SLOTS
            || runtime.engine.core.pool.capacity() != expected_production_primary_capacity
            || runtime.engine.core.pool.shadow_capacity() != 0
            || runtime.engine.core.pool.origin() != BufferPoolOrigin::Production
        {
            return Err(BenchmarkFailure::new(
                "startup",
                "production-primary-pool-contract-drift",
                format!(
                    "expected production cache/pool/shadow capacities {FROZEN_RAM_CACHE_SLOTS}/{expected_production_primary_capacity}/0, observed {}/{}/{}",
                    runtime.engine.core.cache.capacity(),
                    runtime.engine.core.pool.capacity(),
                    runtime.engine.core.pool.shadow_capacity(),
                ),
            )
            .into());
        }
        if let Some(pool) = oracle_source_pool.as_ref() {
            let snapshot = pool.snapshot();
            if pool.pool.origin() != BufferPoolOrigin::QualificationOracleFutureSource
                || snapshot.capacity_slots != ORACLE_FUTURE_SOURCE_POOL_SLOTS
                || snapshot.buffer_size_bytes != spec.cfg.model.expert_size
                || snapshot.allocated_bytes != expected_oracle_source_allocated_bytes
                || snapshot.current_in_use_slots != 0
            {
                return Err(BenchmarkFailure::new(
                    "startup",
                    "oracle-source-pool-contract-drift",
                    format!("invalid isolated ORACLE source pool: {snapshot:?}"),
                )
                .into());
            }
        }
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
                oracle_source_pool.clone(),
                logical_data_accesses.clone(),
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
                        oracle_source_pool.clone(),
                        logical_data_accesses.clone(),
                        &prompt_ids,
                        run_index,
                    ),
                )
                .await?,
            );
        }
        let production_primary_pool_available_after_requests =
            runtime.engine.core.pool.primary_available();
        let pool_snapshot = oracle_source_pool
            .as_ref()
            .map(|pool| pool.snapshot())
            .unwrap_or_default();
        if pool_snapshot.current_in_use_slots != 0
            || pool_snapshot.peak_in_use_slots > pool_snapshot.capacity_slots
            || pool_snapshot.exhaustion_count != 0
            || pool_snapshot.nvme_bytes
                != pool_snapshot
                    .nvme_reads
                    .saturating_mul(spec.cfg.model.expert_size as u64)
        {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "oracle-source-pool-reconciliation",
                format!("isolated ORACLE source pool did not reconcile: {pool_snapshot:?}"),
            )
            .into());
        }
        let source_memory_planes = SourceMemoryPlaneEvidence {
            production_ram_cache_capacity_slots: FROZEN_RAM_CACHE_SLOTS,
            production_ram_cache_max_resident_bytes: expected_oracle_source_allocated_bytes,
            production_primary_pool_capacity_slots: runtime.engine.core.pool.capacity(),
            production_primary_headroom_slots: runtime
                .engine
                .core
                .pool
                .capacity()
                .saturating_sub(runtime.engine.core.cache.capacity()),
            production_primary_pool_available_before_requests,
            production_primary_pool_available_after_requests,
            production_primary_pool_buffer_size_bytes: runtime.engine.core.pool.buffer_size(),
            production_primary_pool_allocated_bytes: runtime.engine.core.pool.allocated_bytes(),
            oracle_source_pool_configured_max_slots: oracle_source_pool_configured_max_slots(
                args.treatment_mode,
            ),
            oracle_source_pool_allocated_capacity_slots: pool_snapshot.capacity_slots,
            oracle_source_pool_buffer_size_bytes: pool_snapshot.buffer_size_bytes,
            oracle_source_pool_allocated_bytes: pool_snapshot.allocated_bytes,
            oracle_source_pool_current_in_use_slots: pool_snapshot.current_in_use_slots,
            oracle_source_pool_peak_in_use_slots: pool_snapshot.peak_in_use_slots,
            oracle_source_pool_exhaustion_count: pool_snapshot.exhaustion_count,
            oracle_source_nvme_reads: pool_snapshot.nvme_reads,
            oracle_source_bytes: pool_snapshot.nvme_bytes,
            oracle_source_pool_reused_across_warmup_and_measured_requests: oracle_source_pool
                .is_some(),
            no_oracle_pool_accumulation_across_requests: true,
            oracle_source_buffers_released_before_runtime_shutdown: true,
            future_source_pool_isolated_from_production_primary: uses_isolated_oracle_source_pool(
                args.treatment_mode,
            ),
            production_primary_pool_capacity_unchanged: runtime.engine.core.pool.capacity()
                == expected_production_primary_capacity,
        };
        Ok::<_, Box<dyn std::error::Error>>((validated, warmup, measured, source_memory_planes))
    }
    .await;
    drop(oracle_source_pool);
    let oracle_source_pool_released =
        crate::buffer_pool::expert_buffer_pool_qualification_oracle_bytes()
            == oracle_allocation_before;
    let shutdown = runtime.shutdown_isolated().await;
    let (validated, warmup, measured, source_memory_planes, shutdown) = match (execution, shutdown)
    {
        (Ok((validated, warmup, measured, source_memory_planes)), Ok(shutdown)) => {
            if !oracle_source_pool_released {
                return Err(BenchmarkFailure::new(
                    "postcondition",
                    "oracle-source-pool-shutdown-leak",
                    "qualification-owned ORACLE source allocation survived pre-shutdown release",
                )
                .into());
            }
            if !shutdown.controlled_shutdown_requested || !shutdown.all_runtime_resources_released {
                return Err(BenchmarkFailure::new(
                    "postcondition",
                    "runtime-shutdown-incomplete",
                    format!("controlled isolated shutdown was incomplete: {shutdown:?}"),
                )
                .into());
            }
            (validated, warmup, measured, source_memory_planes, shutdown)
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
    validate_physical_work_counters(&aggregate_oracle, args.treatment_mode)?;
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
    if logical_data_accesses.load(Ordering::Relaxed) != 0 {
        return Err(artifact_failure(
            "logical-only-payload-data-access",
            "qualification logical-only payload data was accessed",
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
        logical_payload_mode: logical_payload_mode(args.treatment_mode),
        oracle_source_concurrency: args.oracle_source_concurrency,
        treatment_contract: treatment_contract(args.treatment_mode),
        physical_slot_plan: validated.physical_slot_plan,
        frozen_controls: FrozenControlReferences {
            logical_payload_control_git_sha: FROZEN_PAYLOAD_CONTROL_SHA,
            logical_payload_control_tree_sha: FROZEN_PAYLOAD_CONTROL_TREE,
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
        predecessor_first_attempt: PredecessorFirstAttemptEvidence {
            code_sha: PREDECESSOR_FIRST_ATTEMPT_CODE_SHA,
            result: "FAIL",
            failure: "ProductionBatchPoolUnavailableAfterReservation",
            requested: 8,
            acquired: 1,
            interpretation: "future source retained production-primary buffers across foreground demand",
            first_attempt_rerun: false,
        },
        predecessor_second_attempt: PredecessorSecondAttemptEvidence {
            code_sha: PREDECESSOR_SECOND_ATTEMPT_CODE_SHA,
            result: "PASS",
            report_sha256: PREDECESSOR_SECOND_ATTEMPT_REPORT_SHA256,
            runner_sha256: PREDECESSOR_SECOND_ATTEMPT_RUNNER_SHA256,
            configured_source_concurrency: 4,
            observed_source_prefetch_peak_inflight: 1,
            decode_tps: 0.8912406378809807,
            ordinary_residency_misses: 132,
            second_attempt_rerun: false,
        },
        source_memory_planes,
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
    use crate::io_provider::{NvmeStorage, StorageConfig};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
    use std::time::Duration;

    struct TempSourceDir {
        path: PathBuf,
    }

    impl TempSourceDir {
        fn with_experts(label: &str, count: u32, expert_size: usize) -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "mer-oracle-source-{label}-{}-{serial}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            for id in 0..count {
                std::fs::write(
                    path.join(format!("expert_{id}.bin")),
                    vec![id as u8; expert_size],
                )
                .unwrap();
            }
            Self { path }
        }

        fn storage(&self, expert_size: usize, block_align: usize) -> Arc<NvmeStorage> {
            Arc::new(
                NvmeStorage::new(StorageConfig {
                    base_path: self.path.clone(),
                    expert_size,
                    block_align,
                    use_direct_io: false,
                    num_experts_per_layer: None,
                })
                .unwrap(),
            )
        }
    }

    impl Drop for TempSourceDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

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

    fn valid_direct_source_mechanism_counters() -> OracleCounters {
        OracleCounters {
            source_prefetch_batches_started: 2,
            source_prefetch_batches_completed: 2,
            source_prefetch_peak_inflight: 2,
            source_prefetch_read_batches_with_work: 2,
            source_prefetch_positions_with_multiple_read_batches: 1,
            source_prefetch_reads_peak_inflight: 2,
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
        let (send_outcome, receive_outcome) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            let outcome = receive_outcome
                .await
                .expect("test controls source task completion");
            PrefetchLease::new(outcome, position_guard)
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

    fn logical_only_mode() -> OracleScheduledResidencyMode {
        OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnly
    }

    fn no_zero_fill_mode() -> OracleScheduledResidencyMode {
        OracleScheduledResidencyMode::TokenBoundaryDirectLogicalOnlyNoZeroFill
    }

    #[test]
    fn no_zero_fill_contract_preserves_logical_only_frozen_c4_source_and_boundary() {
        assert_eq!(
            OracleScheduledResidencyMode::from_str(
                "token-boundary-direct-logical-only-no-zero-fill",
                false,
            )
            .unwrap(),
            no_zero_fill_mode()
        );
        assert_eq!(
            serde_json::to_value(no_zero_fill_mode()).unwrap(),
            "token-boundary-direct-logical-only-no-zero-fill"
        );
        let mut control = serde_json::to_value(treatment_contract(logical_only_mode())).unwrap();
        let mut treatment = serde_json::to_value(treatment_contract(no_zero_fill_mode())).unwrap();
        assert_eq!(
            control
                .as_object_mut()
                .unwrap()
                .remove("physical_fill_policy")
                .unwrap(),
            "full-slot-zero"
        );
        assert_eq!(
            treatment
                .as_object_mut()
                .unwrap()
                .remove("physical_fill_policy")
                .unwrap(),
            "qualification-no-zero-fill"
        );
        assert_eq!(
            control
                .as_object_mut()
                .unwrap()
                .remove("production_direct_staging_used")
                .unwrap(),
            false
        );
        assert_eq!(
            treatment
                .as_object_mut()
                .unwrap()
                .remove("production_direct_staging_used")
                .unwrap(),
            true
        );
        assert_eq!(control, treatment);
        for mode in [logical_only_mode(), no_zero_fill_mode()] {
            assert_eq!(
                logical_payload_mode(mode),
                LogicalPayloadMode::QualificationLogicalOnly
            );
            assert_eq!(
                late_source_policy(mode),
                LateSourcePolicy::AwaitResidualAtSafeBoundary
            );
            assert_eq!(oracle_source_pool_configured_max_slots(mode), 384);
            assert!(treatment_contract(mode).concurrent_direct_staging_used);
            validate_logical_only_concurrency(mode, 4).unwrap();
            for concurrency in [0, 1, 2, 3, 5, 8, 48, usize::MAX] {
                assert!(validate_logical_only_concurrency(mode, concurrency).is_err());
            }
            validate_source_concurrency_mechanism(
                &valid_direct_source_mechanism_counters(),
                mode,
                4,
            )
            .unwrap();
            let mut serialized = valid_direct_source_mechanism_counters();
            serialized.source_prefetch_reads_peak_inflight = 1;
            assert!(validate_source_concurrency_mechanism(&serialized, mode, 4).is_err());
        }
    }

    fn set_valid_physical_work_counters(
        c: &mut OracleCounters,
        mode: OracleScheduledResidencyMode,
    ) {
        let n = c.boundary_physical_installs;
        c.boundary_direct_staging_writes = n;
        c.boundary_physical_install_bytes = n * 2_654_212;
        c.physical_slot_zero_fill_bytes = if mode == no_zero_fill_mode() {
            0
        } else {
            n * 2_654_212
        };
        c.physical_slot_epoch_write_bytes = n * 4;
        c.physical_slot_payload_copy_bytes = n * 2_654_208;
        c.physical_slot_prepare_us = n * 6;
        c.physical_queue_staging_us = n * 3;
        c.mapping_publication_us = n * 2;
        c.individual_physical_stage_us = n * 11;
        c.token_boundary_commit_us = n * 5;
        c.physical_install_total_us = n * 16;
    }

    #[test]
    fn no_zero_fill_physical_work_reconciliation_rejects_corrupted_bytes_timings_and_installs() {
        let corruptions: &[fn(&mut OracleCounters)] = &[
            |c| c.physical_slot_zero_fill_bytes += 1,
            |c| c.physical_slot_payload_copy_bytes += 1,
            |c| c.physical_slot_epoch_write_bytes += 1,
            |c| c.boundary_physical_install_bytes += 1,
            |c| c.boundary_direct_staging_writes -= 1,
            |c| c.boundary_physical_installs = u64::MAX,
            |c| c.boundary_physical_install_order_checks -= 1,
            |c| c.boundary_physical_install_order_errors = 1,
            |c| c.boundary_stale_generation = 1,
            |c| c.boundary_install_failures = 1,
            |c| c.logical_only_data_accesses = 1,
            |c| c.boundary_full_slot_vec_materializations = 1,
            |c| c.physical_install_evidence_errors = 1,
            |c| c.physical_install_timing_errors = 1,
            |c| c.physical_slot_prepare_us = c.individual_physical_stage_us + 1,
            |c| c.physical_queue_staging_us = u64::MAX,
            |c| c.mapping_publication_us = c.token_boundary_commit_us + 1,
            |c| c.individual_physical_stage_us += 1,
            |c| c.token_boundary_commit_us += 1,
            |c| c.physical_install_total_us += 1,
        ];
        for mode in [logical_only_mode(), no_zero_fill_mode()] {
            let mut good = valid_source_only_counters();
            accumulate_oracle(&mut good, &valid_logical_only_counters());
            set_valid_physical_work_counters(&mut good, mode);
            validate_oracle_counters(&good, mode, 4, 8).unwrap();
            for corrupt in corruptions {
                let mut bad = good.clone();
                corrupt(&mut bad);
                assert!(
                    validate_oracle_counters(&bad, mode, 4, 8).is_err(),
                    "accepted {mode:?}: {bad:?}"
                );
            }
            let mut wrong_arm = good;
            wrong_arm.physical_slot_zero_fill_bytes = if mode == no_zero_fill_mode() {
                2 * 2_654_212
            } else {
                0
            };
            assert!(validate_physical_work_counters(&wrong_arm, mode).is_err());
        }
    }

    #[test]
    fn no_zero_fill_completion_evidence_and_aggregation_preserve_exact_host_work_and_h2d() {
        for mode in [logical_only_mode(), no_zero_fill_mode()] {
            let policy = physical_fill_policy(mode);
            let evidence = GpuNativePhysicalInstallEvidence {
                direct_staging_writes: 1,
                physical_slot_bytes_staged: 2_654_212,
                physical_slot_zero_fill_bytes: if mode == no_zero_fill_mode() {
                    0
                } else {
                    2_654_212
                },
                physical_slot_epoch_write_bytes: 4,
                physical_slot_payload_copy_bytes: 2_654_208,
                physical_slot_prepare_us: 6,
                physical_queue_staging_us: 3,
                mapping_publication_us: 2,
                individual_physical_stage_us: 11,
                ..GpuNativePhysicalInstallEvidence::default()
            };
            let mut observed = OracleCounters::default();
            record_physical_work_evidence(
                &mut observed,
                policy,
                2_654_212,
                2_654_208,
                evidence,
                16,
            );
            assert_eq!(observed.physical_install_evidence_errors, 0);
            assert_eq!(observed.physical_install_timing_errors, 0);
            assert_eq!(
                observed.physical_slot_zero_fill_bytes,
                evidence.physical_slot_zero_fill_bytes
            );
            assert_eq!(observed.physical_slot_epoch_write_bytes, 4);
            assert_eq!(observed.physical_slot_payload_copy_bytes, 2_654_208);
            assert_eq!(observed.physical_slot_prepare_us, 6);
            assert_eq!(observed.physical_queue_staging_us, 3);
            assert_eq!(observed.mapping_publication_us, 2);
            assert_eq!(observed.individual_physical_stage_us, 11);
            assert_eq!(observed.physical_install_total_us, 16);
            for corrupt in [
                (|e: &mut GpuNativePhysicalInstallEvidence| e.physical_slot_zero_fill_bytes += 1)
                    as fn(&mut GpuNativePhysicalInstallEvidence),
                |e| e.physical_slot_epoch_write_bytes += 1,
                |e| e.physical_slot_payload_copy_bytes += 1,
                |e| e.physical_slot_bytes_staged += 1,
                |e| e.direct_staging_writes = 0,
                |e| e.full_slot_vec_materializations = 1,
            ] {
                let mut bad_evidence = evidence;
                corrupt(&mut bad_evidence);
                let mut bad = OracleCounters::default();
                record_physical_work_evidence(
                    &mut bad,
                    policy,
                    2_654_212,
                    2_654_208,
                    bad_evidence,
                    16,
                );
                assert_eq!(bad.physical_install_evidence_errors, 1);
            }
            for corrupt in [
                (|e: &mut GpuNativePhysicalInstallEvidence| e.physical_slot_prepare_us = 12)
                    as fn(&mut GpuNativePhysicalInstallEvidence),
                |e| e.physical_queue_staging_us = u64::MAX,
                |e| e.mapping_publication_us = 6,
                |e| e.individual_physical_stage_us = 17,
            ] {
                let mut bad_evidence = evidence;
                corrupt(&mut bad_evidence);
                let mut bad = OracleCounters::default();
                record_physical_work_evidence(
                    &mut bad,
                    policy,
                    2_654_212,
                    2_654_208,
                    bad_evidence,
                    16,
                );
                assert_eq!(bad.physical_install_timing_errors, 1);
            }
            let mut one = valid_logical_only_counters();
            set_valid_physical_work_counters(&mut one, mode);
            let mut total = OracleCounters::default();
            accumulate_oracle(&mut total, &one);
            accumulate_oracle(&mut total, &one);
            validate_physical_work_counters(&total, mode).unwrap();
            let one_json = serde_json::to_value(one).unwrap();
            let total_json = serde_json::to_value(total).unwrap();
            for (name, value) in one_json.as_object().unwrap() {
                if name.starts_with("physical_")
                    || name == "mapping_publication_us"
                    || name == "individual_physical_stage_us"
                    || name == "token_boundary_commit_us"
                    || name.starts_with("boundary_physical_")
                {
                    assert_eq!(
                        total_json[name].as_u64().unwrap(),
                        value.as_u64().unwrap() * 2,
                        "{name}"
                    );
                }
            }
            let mut frozen = valid_logical_only_counters();
            frozen.boundary_physical_installs = 89_676;
            frozen.boundary_physical_install_order_checks = 89_676;
            set_valid_physical_work_counters(&mut frozen, mode);
            validate_physical_work_counters(&frozen, mode).unwrap();
            assert_eq!(frozen.boundary_physical_install_bytes, 238_019_115_312);
            assert_eq!(
                frozen.physical_slot_zero_fill_bytes,
                if mode == no_zero_fill_mode() {
                    0
                } else {
                    238_019_115_312
                }
            );
        }
    }

    #[test]
    fn logical_only_contract_changes_payload_mode_and_preserves_direct_source_and_boundary_contract(
    ) {
        assert_eq!(
            OracleScheduledResidencyMode::from_str("token-boundary-direct-logical-only", false)
                .unwrap(),
            logical_only_mode()
        );
        let control = OracleScheduledResidencyMode::TokenBoundaryDirect;
        assert_eq!(
            logical_payload_mode(control),
            LogicalPayloadMode::Materialized
        );
        let mut control_json = serde_json::to_value(treatment_contract(control)).unwrap();
        let mut treatment_json =
            serde_json::to_value(treatment_contract(logical_only_mode())).unwrap();
        assert_eq!(
            control_json
                .as_object_mut()
                .unwrap()
                .remove("logical_payload_mode")
                .unwrap(),
            "materialized"
        );
        assert_eq!(
            treatment_json
                .as_object_mut()
                .unwrap()
                .remove("logical_payload_mode")
                .unwrap(),
            "qualification-logical-only"
        );
        assert_eq!(control_json, treatment_json);
        assert_eq!(
            oracle_source_pool_configured_max_slots(logical_only_mode()),
            384
        );
        assert!(validate_logical_only_concurrency(logical_only_mode(), 4).is_ok());
        for concurrency in [1, 2, 3, 8] {
            assert!(validate_logical_only_concurrency(logical_only_mode(), concurrency).is_err());
            assert!(validate_logical_only_concurrency(control, concurrency).is_ok());
        }
    }

    #[test]
    fn logical_only_oracle_preparation_matches_control_and_releases_real_source_leases() {
        use crate::inference::WeightDtype;
        let pool = OracleFutureSourcePool::new(2, 72, 8);
        let mut residents = HashMap::new();
        for id in [2, 1] {
            // A real UTH header proves the logical charge is the stripped
            // ExpertResident payload length, not the larger source-pool slot.
            let mut bytes = Vec::new();
            crate::tensor_header::TensorHeader::for_swiglu_expert(WeightDtype::Q4_0, 32, 32)
                .write_padded(8, &mut bytes);
            bytes.extend_from_slice(&[id as u8; 8]);
            assert_eq!(bytes.len(), 72);
            let mut buffer = pool.try_acquire().unwrap();
            buffer.as_mut_slice().copy_from_slice(&bytes);
            let source = Arc::new(ExpertResident::new_with_block_align(id, buffer, 8));
            assert_eq!(source.data().len(), 8);
            residents.insert(id, source);
        }
        let weak = residents.values().map(Arc::downgrade).collect::<Vec<_>>();
        let caches = [
            GpuExpertCache::new(16, 0.0, 0),
            GpuExpertCache::new(16, 0.0, 0),
            GpuExpertCache::new(16, 0.0, 0),
        ];
        let mut signatures = Vec::new();
        let mut retained = Vec::new();
        for (index, mode) in [
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            logical_only_mode(),
            no_zero_fill_mode(),
        ]
        .into_iter()
        .enumerate()
        {
            let audit = Arc::new(AtomicU64::new(0));
            let counters = Mutex::new(OracleCounters::default());
            let admissions = prepare_logical_admissions(
                &caches[index],
                WeightDtype::Q4_0,
                &[2, 1],
                &residents,
                mode,
                &counters,
                &audit,
            )
            .unwrap();
            for admission in &admissions {
                let id = admission.resident().id;
                let source = &residents[&id];
                crate::gpu_native_residency::validate_qualification_physical_source_for_test(
                    &caches[index],
                    id,
                    source,
                    admission,
                )
                .unwrap();
                let demand = GpuNativeDemandExpert::install(id, source.clone(), admission.clone());
                match demand {
                    GpuNativeDemandExpert::Install { resident, .. } => {
                        assert!(Arc::ptr_eq(&resident, source));
                        assert_eq!(resident.data().as_ptr(), source.data().as_ptr());
                    }
                    _ => panic!("expected physical install source"),
                }
                if index == 0 {
                    assert_eq!(admission.resident().data(), source.data());
                    assert_ne!(admission.resident().data().as_ptr(), source.data().as_ptr());
                }
                assert_eq!(Arc::strong_count(source), 1);
            }
            let repeated = prepare_logical_admissions(
                &caches[index],
                WeightDtype::Q4_0,
                &[1, 2],
                &residents,
                mode,
                &counters,
                &audit,
            )
            .unwrap();
            assert_eq!(repeated[0].generation(), admissions[1].generation());
            let c = counters.lock();
            assert_eq!(c.logical_expected_payload_bytes, 16);
            assert_eq!(c.logical_newly_admitted_bytes, 16);
            assert_eq!(c.logical_new_generations, 2);
            assert_eq!(c.logical_current_generations_validated, 4);
            assert_eq!(c.logical_gpu_used_bytes_before, 16);
            assert_eq!(c.logical_gpu_used_bytes_after, 32);
            assert_eq!(c.logical_gpu_evicted_bytes, 0);
            assert_eq!(c.logical_materializations, if index == 0 { 2 } else { 0 });
            assert_eq!(
                c.logical_materialization_bytes,
                if index == 0 { 16 } else { 0 }
            );
            assert_eq!(c.logical_only_admissions, if index != 0 { 2 } else { 0 });
            assert_eq!(
                c.logical_only_charged_bytes,
                if index != 0 { 16 } else { 0 }
            );
            assert_eq!(audit.load(Ordering::Relaxed), 0);
            signatures.push((
                admissions
                    .iter()
                    .map(|a| (a.resident().id, a.generation(), a.byte_len()))
                    .collect::<Vec<_>>(),
                caches[index].used_bytes(),
            ));
            retained.extend(admissions);
        }
        assert_eq!(signatures[0], signatures[1]);
        assert_eq!(signatures[1], signatures[2]);
        drop(residents);
        assert!(weak.iter().all(|source| source.upgrade().is_none()));
        assert_eq!(pool.snapshot().current_in_use_slots, 0);
        assert_eq!(pool.snapshot().exhaustion_count, 0);
        assert_eq!(
            retained.len(),
            6,
            "logical admissions remain alive after all source leases return"
        );
    }

    #[test]
    fn logical_only_cache_cannot_accumulate_isolated_source_pool_leases() {
        use crate::inference::WeightDtype;
        let pool = OracleFutureSourcePool::new(1, 8, 8);
        let cache = GpuExpertCache::new(400 * 8, 0.0, 0);
        let counters = Mutex::new(OracleCounters::default());
        let audit = Arc::new(AtomicU64::new(0));
        for id in 0..400 {
            let mut buffer = pool.try_acquire().unwrap();
            buffer.as_mut_slice().fill(id as u8);
            let source = Arc::new(ExpertResident::new(id, buffer));
            let weak = Arc::downgrade(&source);
            let residents = HashMap::from([(id, source)]);
            prepare_logical_admissions(
                &cache,
                WeightDtype::Q4_0,
                &[id],
                &residents,
                logical_only_mode(),
                &counters,
                &audit,
            )
            .unwrap();
            drop(residents);
            assert!(weak.upgrade().is_none());
            assert_eq!(pool.snapshot().current_in_use_slots, 0);
        }
        assert_eq!(cache.used_bytes(), 400 * 8);
        assert_eq!(pool.snapshot().peak_in_use_slots, 1);
        assert_eq!(pool.snapshot().exhaustion_count, 0);
        assert_eq!(pool.snapshot().nvme_reads, 0);
        assert_eq!(audit.load(Ordering::Relaxed), 0);
    }

    fn valid_logical_only_counters() -> OracleCounters {
        OracleCounters {
            logical_only_admissions: 2,
            logical_only_charged_bytes: 16,
            logical_expected_payload_bytes: 16,
            logical_admission_transactions: 1,
            logical_admissions_returned: 2,
            logical_new_generations: 2,
            logical_current_generations_validated: 2,
            logical_newly_admitted_bytes: 16,
            logical_gpu_used_bytes_before: 8,
            logical_gpu_used_bytes_after: 16,
            logical_gpu_evicted_bytes: 8,
            boundary_physical_installs: 2,
            boundary_physical_install_order_checks: 2,
            ..OracleCounters::default()
        }
    }

    #[test]
    fn logical_only_payload_gates_reject_each_counter_violation_and_caught_data_access() {
        let good = valid_logical_only_counters();
        validate_logical_payload_counters(&good, logical_only_mode()).unwrap();
        assert!(
            validate_logical_payload_counters(&OracleCounters::default(), logical_only_mode())
                .is_err()
        );
        let corruptions: &[fn(&mut OracleCounters)] = &[
            |c| c.logical_materializations = 1,
            |c| c.logical_materialization_bytes = 8,
            |c| c.logical_only_admissions = 0,
            |c| c.logical_only_charged_bytes = 0,
            |c| c.logical_expected_payload_bytes += 1,
            |c| c.logical_only_data_accesses = 1,
            |c| c.logical_admission_transactions = 0,
            |c| c.logical_admissions_returned += 1,
            |c| c.logical_new_generations += 1,
            |c| c.logical_current_generations_validated -= 1,
            |c| c.logical_newly_admitted_bytes += 1,
            |c| c.logical_gpu_used_bytes_before += 1,
            |c| c.logical_gpu_used_bytes_after += 1,
            |c| c.logical_gpu_evicted_bytes += 1,
            |c| {
                c.logical_gpu_used_bytes_before = u64::MAX;
                c.logical_gpu_used_bytes_after = u64::MAX;
            },
            |c| c.boundary_physical_installs -= 1,
            |c| c.boundary_physical_install_order_checks -= 1,
            |c| c.boundary_physical_install_order_errors = 1,
            |c| c.boundary_stale_generation = 1,
            |c| c.boundary_install_failures = 1,
        ];
        for corrupt in corruptions {
            let mut bad = good.clone();
            corrupt(&mut bad);
            assert!(
                validate_logical_payload_counters(&bad, logical_only_mode()).is_err(),
                "accepted {bad:?}"
            );
        }
        let audit = Arc::new(AtomicU64::new(0));
        let resident = GpuResident::new_qualification_logical_only(
            1,
            8,
            crate::inference::WeightDtype::Q4_0,
            audit.clone(),
        );
        assert!(std::panic::catch_unwind(|| resident.data()).is_err());
        drop(resident);
        let mut caught = good;
        caught.logical_only_data_accesses = audit.load(Ordering::Relaxed);
        assert!(validate_logical_payload_counters(&caught, logical_only_mode()).is_err());
    }

    #[test]
    fn logical_only_completion_audit_rejects_install_order_identity_generation_and_byte_drift() {
        let make = || OracleBoundaryObserver {
            counters: Arc::new(Mutex::new(OracleCounters::default())),
            expected_installs: vec![(130, 7), (129, 8)],
            completed_installs: Mutex::new(0),
            layer_index: 1,
            experts_per_layer: 128,
            slot_bytes: 12,
            payload_bytes: 8,
            fill_policy: GpuNativePhysicalSlotFillPolicy::FullSlotZero,
        };
        let good = make();
        good.record_install_identity(130, GpuNativeQ4ExpertKey::new(1, 2, 7), 12);
        good.record_install_identity(129, GpuNativeQ4ExpertKey::new(1, 1, 8), 12);
        assert_eq!(*good.completed_installs.lock(), 2);
        assert_eq!(
            good.counters.lock().boundary_physical_install_order_checks,
            2
        );
        assert_eq!(
            good.counters.lock().boundary_physical_install_order_errors,
            0
        );
        for (id, layer, expert, generation, bytes) in [
            (129, 1, 1, 8, 12), // valid second identity presented out of order
            (130, 0, 2, 7, 12), // wrong layer
            (130, 1, 3, 7, 12), // wrong local expert
            (130, 1, 2, 9, 12), // wrong logical generation
            (130, 1, 2, 7, 8),  // wrong physical slot byte count
        ] {
            let bad = make();
            bad.record_install_identity(
                id,
                GpuNativeQ4ExpertKey::new(layer, expert, generation),
                bytes,
            );
            assert_eq!(
                bad.counters.lock().boundary_physical_install_order_errors,
                1
            );
        }
        good.record_install_identity(129, GpuNativeQ4ExpertKey::new(1, 1, 8), 12);
        assert_eq!(
            good.counters.lock().boundary_physical_install_order_errors,
            1,
            "extra install fails closed"
        );
    }

    #[test]
    fn logical_only_inherits_source_pool_and_c4_fail_closed_gates() {
        let mut good = valid_source_only_counters();
        accumulate_oracle(&mut good, &valid_logical_only_counters());
        set_valid_physical_work_counters(&mut good, logical_only_mode());
        validate_oracle_counters(&good, logical_only_mode(), 4, 8).unwrap();
        for corrupt in [
            (|c: &mut OracleCounters| c.qualification_owned_peak_slots = 385)
                as fn(&mut OracleCounters),
            |c| c.qualification_owned_current_slots = 1,
            |c| c.oracle_source_pool_exhaustion_count = 1,
            |c| c.source_prefetch_peak_inflight = 5,
            |c| c.source_prefetch_reads_peak_inflight = 5,
            |c| c.source_prefetch_reads_current_inflight = 1,
            |c| c.source_reads_started += 1,
            |c| c.oracle_source_nvme_reads += 1,
            |c| c.boundary_full_slot_vec_materializations = 1,
        ] {
            let mut bad = good.clone();
            corrupt(&mut bad);
            assert!(validate_oracle_counters(&bad, logical_only_mode(), 4, 8).is_err());
        }
        let c4 = valid_direct_source_mechanism_counters();
        validate_source_concurrency_mechanism(&c4, logical_only_mode(), 4).unwrap();
        let mut serialized = c4;
        serialized.source_prefetch_reads_peak_inflight = 1;
        assert!(
            validate_source_concurrency_mechanism(&serialized, logical_only_mode(), 4).is_err()
        );
    }

    #[test]
    fn logical_only_counter_aggregation_preserves_all_added_evidence() {
        let one = valid_logical_only_counters();
        let mut total = OracleCounters::default();
        accumulate_oracle(&mut total, &one);
        accumulate_oracle(&mut total, &one);
        let one_json = serde_json::to_value(one).unwrap();
        let total_json = serde_json::to_value(total.clone()).unwrap();
        for (name, value) in one_json.as_object().unwrap() {
            if name.starts_with("logical_") || name.starts_with("boundary_physical_install_order_")
            {
                assert_eq!(
                    total_json[name].as_u64().unwrap(),
                    value.as_u64().unwrap() * 2,
                    "{name}"
                );
            }
        }
        validate_logical_payload_counters(&total, logical_only_mode()).unwrap();
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
        assert!(!contract.independent_layer_source_tasks);
        assert!(!contract.per_layer_storage_batch_read_used);
        assert!(!contract.token_boundary_h2d);
        assert!(!contract.production_direct_staging_used);
        assert!(!contract.h2d_compute_overlap_claimed);
        assert!(!contract.future_source_pool_isolated_from_production_primary);
        assert_eq!(
            oracle_source_pool_configured_max_slots(OracleScheduledResidencyMode::SourceOnly),
            0
        );
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

    #[test]
    fn v5_schema_and_predecessor_attempt_contracts_are_frozen() {
        assert_eq!(SCHEMA, "mer.gpu-native-oracle-scheduled-residency.v5");
        let first = PredecessorFirstAttemptEvidence {
            code_sha: PREDECESSOR_FIRST_ATTEMPT_CODE_SHA,
            result: "FAIL",
            failure: "ProductionBatchPoolUnavailableAfterReservation",
            requested: 8,
            acquired: 1,
            interpretation:
                "future source retained production-primary buffers across foreground demand",
            first_attempt_rerun: false,
        };
        let first = serde_json::to_value(first).unwrap();
        assert_eq!(first["code_sha"], PREDECESSOR_FIRST_ATTEMPT_CODE_SHA);
        assert_eq!(first["result"], "FAIL");
        assert_eq!(first["requested"], 8);
        assert_eq!(first["acquired"], 1);
        assert_eq!(first["first_attempt_rerun"], false);

        let second = serde_json::to_value(PredecessorSecondAttemptEvidence {
            code_sha: PREDECESSOR_SECOND_ATTEMPT_CODE_SHA,
            result: "PASS",
            report_sha256: PREDECESSOR_SECOND_ATTEMPT_REPORT_SHA256,
            runner_sha256: PREDECESSOR_SECOND_ATTEMPT_RUNNER_SHA256,
            configured_source_concurrency: 4,
            observed_source_prefetch_peak_inflight: 1,
            decode_tps: 0.8912406378809807,
            ordinary_residency_misses: 132,
            second_attempt_rerun: false,
        })
        .unwrap();
        assert_eq!(second["code_sha"], PREDECESSOR_SECOND_ATTEMPT_CODE_SHA);
        assert_eq!(second["result"], "PASS");
        assert_eq!(
            second["report_sha256"],
            PREDECESSOR_SECOND_ATTEMPT_REPORT_SHA256
        );
        assert_eq!(
            second["runner_sha256"],
            PREDECESSOR_SECOND_ATTEMPT_RUNNER_SHA256
        );
        assert_eq!(second["configured_source_concurrency"], 4);
        assert_eq!(second["observed_source_prefetch_peak_inflight"], 1);
        assert_eq!(second["decode_tps"], 0.8912406378809807);
        assert_eq!(second["ordinary_residency_misses"], 132);
        assert_eq!(second["second_attempt_rerun"], false);
    }

    #[test]
    fn v3_memory_report_distinguishes_production_and_oracle_planes() {
        let evidence_for = |mode| {
            let isolated = uses_isolated_oracle_source_pool(mode);
            SourceMemoryPlaneEvidence {
                production_ram_cache_capacity_slots: 384,
                production_ram_cache_max_resident_bytes: 384 * 4096,
                production_primary_pool_capacity_slots: 385,
                production_primary_headroom_slots: 1,
                production_primary_pool_available_before_requests: 1,
                production_primary_pool_available_after_requests: 1,
                production_primary_pool_buffer_size_bytes: 4096,
                production_primary_pool_allocated_bytes: 385 * 4096,
                oracle_source_pool_configured_max_slots: oracle_source_pool_configured_max_slots(
                    mode,
                ),
                oracle_source_pool_allocated_capacity_slots: if isolated { 384 } else { 0 },
                oracle_source_pool_buffer_size_bytes: if isolated { 4096 } else { 0 },
                oracle_source_pool_allocated_bytes: if isolated { 384 * 4096 } else { 0 },
                oracle_source_pool_current_in_use_slots: 0,
                oracle_source_pool_peak_in_use_slots: if isolated { 384 } else { 0 },
                oracle_source_pool_exhaustion_count: 0,
                oracle_source_nvme_reads: if isolated { 4 } else { 0 },
                oracle_source_bytes: if isolated { 4 * 4096 } else { 0 },
                oracle_source_pool_reused_across_warmup_and_measured_requests: isolated,
                no_oracle_pool_accumulation_across_requests: true,
                oracle_source_buffers_released_before_runtime_shutdown: true,
                future_source_pool_isolated_from_production_primary: isolated,
                production_primary_pool_capacity_unchanged: true,
            }
        };
        let source_only =
            serde_json::to_value(evidence_for(OracleScheduledResidencyMode::SourceOnly)).unwrap();
        assert_eq!(source_only["oracle_source_pool_configured_max_slots"], 0);
        assert_eq!(
            source_only["oracle_source_pool_allocated_capacity_slots"],
            0
        );
        assert_eq!(
            source_only["future_source_pool_isolated_from_production_primary"],
            false
        );

        let direct = serde_json::to_value(evidence_for(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
        ))
        .unwrap();
        assert_eq!(direct["production_ram_cache_capacity_slots"], 384);
        assert_eq!(direct["production_primary_pool_capacity_slots"], 385);
        assert_eq!(direct["production_primary_headroom_slots"], 1);
        assert_eq!(direct["oracle_source_pool_configured_max_slots"], 384);
        assert_eq!(direct["oracle_source_pool_allocated_capacity_slots"], 384);
        assert_eq!(direct["oracle_source_pool_current_in_use_slots"], 0);
        assert_eq!(
            direct["oracle_source_pool_reused_across_warmup_and_measured_requests"],
            true
        );
        assert_eq!(
            direct["future_source_pool_isolated_from_production_primary"],
            true
        );
    }

    #[test]
    fn exact_oracle_source_bound_uses_runtime_expert_file_bytes() {
        assert_eq!(ORACLE_FUTURE_SOURCE_POOL_SLOTS, 48 * 8);
        assert_eq!(
            oracle_future_source_allocated_bytes(2_658_304),
            Some(1_020_788_736)
        );
    }

    #[test]
    fn retained_source_origin_contract_fails_closed() {
        let production_pool = BufferPool::new(1, 4096, 4096);
        let production_resident = Arc::new(ExpertResident::new_with_block_align(
            7,
            production_pool.try_acquire().unwrap(),
            4096,
        ));
        let mut outcome = PrefetchOutcome::default();
        outcome.residents.insert(7, production_resident);
        assert!(validate_retained_source_origins(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            &outcome,
        )
        .is_err());
        assert!(validate_retained_source_origins(
            OracleScheduledResidencyMode::SourceOnly,
            &outcome,
        )
        .is_err());

        let oracle_pool = OracleFutureSourcePool::new(1, 4096, 4096);
        let oracle_resident = Arc::new(ExpertResident::new_with_block_align(
            8,
            oracle_pool.try_acquire().unwrap(),
            4096,
        ));
        let mut isolated = PrefetchOutcome::default();
        isolated.residents.insert(8, oracle_resident);
        assert!(validate_retained_source_origins(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            &isolated,
        )
        .is_ok());
    }

    #[test]
    fn isolated_oracle_retention_preserves_foreground_primary_capacity() {
        const BUFFER_SIZE: usize = 4096;
        let production_pool = BufferPool::new(3, BUFFER_SIZE, 4096);
        let production_cache = crate::expert_cache::ExpertCache::new(2);
        let insert_primary = |id| {
            production_cache
                .insert(Arc::new(ExpertResident::new_with_block_align(
                    id,
                    production_pool.try_acquire().unwrap(),
                    4096,
                )))
                .unwrap_or_else(|_| panic!("test production-cache insert failed"));
        };

        insert_primary(0);
        insert_primary(1);
        assert_eq!(production_pool.primary_available(), 1);
        let retained_victims = [
            production_cache.get(0).unwrap(),
            production_cache.get(1).unwrap(),
        ];
        drop(production_cache.evict_lru().unwrap());
        drop(production_cache.evict_lru().unwrap());
        assert_eq!(
            production_pool.primary_available(),
            1,
            "request-local victim Arcs reproduce the one-headroom-buffer condition"
        );
        drop(retained_victims);
        assert_eq!(production_pool.primary_available(), 3);

        insert_primary(2);
        insert_primary(3);
        let oracle_pool = OracleFutureSourcePool::new(1, BUFFER_SIZE, 4096);
        let retained_oracle = Arc::new(ExpertResident::new_with_block_align(
            4,
            oracle_pool.try_acquire().unwrap(),
            4096,
        ));
        drop(production_cache.evict_lru().unwrap());
        drop(production_cache.evict_lru().unwrap());
        assert_eq!(production_pool.primary_available(), 3);

        let foreground_a = production_pool.try_acquire().unwrap();
        let foreground_b = production_pool.try_acquire().unwrap();
        assert_eq!(production_pool.primary_available(), 1);
        assert_eq!(oracle_pool.snapshot().current_in_use_slots, 1);
        drop(foreground_a);
        drop(foreground_b);
        drop(retained_oracle);
        assert_eq!(production_pool.primary_available(), 3);
        assert_eq!(oracle_pool.snapshot().current_in_use_slots, 0);
    }

    #[test]
    fn one_source_pool_is_reused_without_request_accumulation() {
        let pool = Arc::new(OracleFutureSourcePool::new(2, 4096, 4096));
        let allocation = Arc::as_ptr(&pool);
        for _ in 0..(FROZEN_WARMUP_RUNS + FROZEN_MEASURED_RUNS) {
            let request_pool = pool.clone();
            assert_eq!(Arc::as_ptr(&request_pool), allocation);
            let first = request_pool.try_acquire().unwrap();
            let second = request_pool.try_acquire().unwrap();
            assert_eq!(request_pool.snapshot().current_in_use_slots, 2);
            drop(first);
            drop(second);
            assert_eq!(request_pool.snapshot().current_in_use_slots, 0);
        }
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.capacity_slots, 2);
        assert_eq!(snapshot.allocated_bytes, 2 * 4096);
        assert_eq!(snapshot.peak_in_use_slots, 2);
        assert_eq!(snapshot.current_in_use_slots, 0);
    }

    #[test]
    fn direct_source_and_boundary_resolution_orders_are_explicit() {
        assert_eq!(
            direct_future_source_disposition(true, true),
            DirectFutureSourceDisposition::AlreadyPhysical
        );
        assert_eq!(
            direct_future_source_disposition(false, true),
            DirectFutureSourceDisposition::DeferredCurrentDemandOverlap
        );
        assert_eq!(
            direct_future_source_disposition(false, false),
            DirectFutureSourceDisposition::IsolatedRead
        );
        assert_eq!(
            direct_boundary_source_disposition(true, true, true),
            DirectBoundarySourceDisposition::Physical
        );
        assert_eq!(
            direct_boundary_source_disposition(false, true, true),
            DirectBoundarySourceDisposition::Isolated
        );
        assert_eq!(
            direct_boundary_source_disposition(false, false, true),
            DirectBoundarySourceDisposition::ProductionRam
        );
        assert_eq!(
            direct_boundary_source_disposition(false, false, false),
            DirectBoundarySourceDisposition::ResidualRead
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_nvme_source_is_isolated_not_cached_and_recycled() {
        const EXPERT_SIZE: usize = 4096;
        let dir = TempSourceDir::with_experts("isolated", 2, EXPERT_SIZE);
        let storage = dir.storage(EXPERT_SIZE, 4096);
        let production_pool = BufferPool::new(1, EXPERT_SIZE, 4096);
        let production_cache = crate::expert_cache::ExpertCache::new(2);
        let production_available = production_pool.primary_available();
        let oracle_pool = OracleFutureSourcePool::new(2, EXPERT_SIZE, 4096);

        let (resident, bytes) = read_oracle_future_source_from_storage(&storage, &oracle_pool, 0)
            .await
            .unwrap();
        assert_eq!(bytes, EXPERT_SIZE);
        assert_eq!(
            resident.buffer_pool_origin(),
            BufferPoolOrigin::QualificationOracleFutureSource
        );
        assert_eq!(production_pool.primary_available(), production_available);
        assert_eq!(production_cache.len(), 0);
        assert_eq!(oracle_pool.current_in_use_slots(), 1);
        let snapshot = oracle_pool.snapshot();
        assert_eq!(snapshot.capacity_slots, 2);
        assert_eq!(snapshot.allocated_bytes, 2 * EXPERT_SIZE);
        assert_eq!(snapshot.peak_in_use_slots, 1);
        assert_eq!(snapshot.nvme_reads, 1);
        assert_eq!(snapshot.nvme_bytes, EXPERT_SIZE as u64);
        drop(resident);
        assert_eq!(oracle_pool.current_in_use_slots(), 0);

        let (reused, _) = read_oracle_future_source_from_storage(&storage, &oracle_pool, 1)
            .await
            .unwrap();
        assert_eq!(oracle_pool.current_in_use_slots(), 1);
        drop(reused);
        assert_eq!(oracle_pool.current_in_use_slots(), 0);
        assert_eq!(oracle_pool.snapshot().nvme_reads, 2);
        assert_eq!(production_cache.len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_source_pool_exhaustion_fails_closed_without_primary_fallback() {
        const EXPERT_SIZE: usize = 4096;
        let dir = TempSourceDir::with_experts("exhaustion", 2, EXPERT_SIZE);
        let storage = dir.storage(EXPERT_SIZE, 4096);
        let production_pool = BufferPool::new(1, EXPERT_SIZE, 4096);
        let oracle_pool = OracleFutureSourcePool::new(1, EXPERT_SIZE, 4096);
        let (held, _) = read_oracle_future_source_from_storage(&storage, &oracle_pool, 0)
            .await
            .unwrap();
        let error = match read_oracle_future_source_from_storage(&storage, &oracle_pool, 1).await {
            Ok(_) => panic!("exhausted ORACLE pool must not fall back to PRIMARY"),
            Err(error) => error,
        };
        assert!(matches!(error, OracleSourceReadError::PoolExhausted(_)));
        assert_eq!(oracle_pool.snapshot().exhaustion_count, 1);
        assert_eq!(production_pool.primary_available(), 1);
        drop(held);
        assert_eq!(oracle_pool.current_in_use_slots(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn direct_layer_task_helper_overlaps_bounds_joins_and_releases_leases() {
        let concurrency = 4usize;
        let counters = Arc::new(Mutex::new(OracleCounters::default()));
        let barrier = Arc::new(tokio::sync::Barrier::new(concurrency));
        let pool = Arc::new(OracleFutureSourcePool::new(concurrency, 4096, 4096));
        let completed = Arc::new(AtomicUsize::new(0));
        let task_factory = {
            let counters = counters.clone();
            let barrier = barrier.clone();
            let pool = pool.clone();
            let completed = completed.clone();
            move |target_position| {
                let counters = counters.clone();
                let barrier = barrier.clone();
                let pool = pool.clone();
                let completed = completed.clone();
                async move {
                    let _guard = LayerBatchGuard::enter(counters);
                    let lease = pool.try_acquire().unwrap();
                    barrier.wait().await;
                    drop(lease);
                    completed.fetch_add(1, Ordering::Relaxed);
                    PrefetchOutcome {
                        target_position,
                        ..PrefetchOutcome::default()
                    }
                }
            }
        };
        let outcomes = run_bounded_layer_tasks(
            0..8,
            concurrency,
            task_factory,
            Vec::new(),
            |outcomes, outcome| outcomes.push(outcome),
        )
        .await
        .unwrap();

        assert_eq!(outcomes.len(), 8);
        assert_eq!(completed.load(Ordering::Relaxed), 8);
        let values = counters.lock();
        assert_eq!(values.source_prefetch_batches_started, 8);
        assert_eq!(values.source_prefetch_batches_completed, 8);
        assert!(values.source_prefetch_peak_inflight > 1);
        assert!(values.source_prefetch_peak_inflight <= concurrency as u64);
        drop(values);
        assert_eq!(pool.current_in_use_slots(), 0);
        assert_eq!(pool.snapshot().exhaustion_count, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn direct_layer_task_helper_propagates_failure_and_drains_owned_tasks() {
        let concurrency = 4usize;
        let counters = Arc::new(Mutex::new(OracleCounters::default()));
        let barrier = Arc::new(tokio::sync::Barrier::new(concurrency));
        let pool = Arc::new(OracleFutureSourcePool::new(concurrency, 4096, 4096));
        let task_factory = {
            let counters = counters.clone();
            let barrier = barrier.clone();
            let pool = pool.clone();
            move |layer_index| {
                let counters = counters.clone();
                let barrier = barrier.clone();
                let pool = pool.clone();
                async move {
                    let _guard = LayerBatchGuard::enter(counters);
                    let lease = pool.try_acquire().unwrap();
                    barrier.wait().await;
                    if layer_index == 0 {
                        panic!("deterministic qualification child failure");
                    }
                    drop(lease);
                    PrefetchOutcome::default()
                }
            }
        };
        let error = run_bounded_layer_tasks(
            0..8,
            concurrency,
            task_factory,
            Vec::new(),
            |outcomes, outcome| outcomes.push(outcome),
        )
        .await
        .err()
        .expect("child panic must fail the scheduling helper");

        assert!(error.contains("independent layer source task failed"));
        let values = counters.lock();
        assert_eq!(
            values.source_prefetch_batches_started,
            values.source_prefetch_batches_completed
        );
        assert!(values.source_prefetch_peak_inflight > 1);
        assert!(values.source_prefetch_peak_inflight <= concurrency as u64);
        drop(values);
        assert_eq!(pool.current_in_use_slots(), 0);
        assert_eq!(pool.snapshot().exhaustion_count, 0);
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
        assert!(!source_only.independent_layer_source_tasks);
        assert!(!source_only.per_layer_storage_batch_read_used);
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
        assert!(!source_only.production_predictor_used);
        assert!(!source_only.production_speculative_residency_used);
        assert!(!source_only.future_source_pool_isolated_from_production_primary);
        assert!(source_only.production_primary_pool_capacity_unchanged);
        assert_eq!(
            oracle_source_pool_configured_max_slots(OracleScheduledResidencyMode::SourceOnly),
            0
        );
        assert!(direct.independent_layer_source_tasks);
        assert!(!direct.per_layer_storage_batch_read_used);
        assert!(!direct.production_predictor_used);
        assert!(!direct.production_speculative_residency_used);
        assert!(direct.future_source_pool_isolated_from_production_primary);
        assert!(direct.production_primary_pool_capacity_unchanged);
        assert_eq!(
            oracle_source_pool_configured_max_slots(
                OracleScheduledResidencyMode::TokenBoundaryDirect
            ),
            ORACLE_FUTURE_SOURCE_POOL_SLOTS
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
        readiness.mark_source_resident_ready(1);
        let mut resolving = Box::pin(resolve_source_task_at_boundary(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            active,
            1,
            vec![0, 1, 2, 3],
            |global_id| {
                Ok(if global_id == 0 {
                    BoundarySourceReadiness::Physical
                } else {
                    BoundarySourceReadiness::Unavailable
                })
            },
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
        let oracle_pool = Arc::new(OracleFutureSourcePool::new(1, 4096, 4096));
        release_source_position_lease_before_next_token(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            lease,
            &source_position_gate,
            &state,
            Some(&oracle_pool),
            &counters,
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
            let mut values = counters.lock();
            assert_eq!(values.source_failures_degraded_to_demand, 0);
            values.source_failures_degraded_to_demand = 1;
            assert_eq!(values.source_late_degraded_to_demand, 1);
            assert_eq!(values.oracle_demand_fallback, 1);
        }
        let oracle_pool = Arc::new(OracleFutureSourcePool::new(1, 4096, 4096));
        release_source_position_lease_before_next_token(
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            lease,
            &source_position_gate,
            &state,
            Some(&oracle_pool),
            &counters,
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
        record_completed_source_outcome(
            &mut counters,
            &outcome,
            0,
            OracleScheduledResidencyMode::SourceOnly,
        );
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
            4096,
        )
        .is_ok());
    }

    #[test]
    fn direct_source_concurrency_mechanism_gates_are_fail_closed() {
        let valid = valid_direct_source_mechanism_counters();
        assert!(validate_source_concurrency_mechanism(
            &valid,
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            4,
        )
        .is_ok());

        let mut layer_peak_one = valid.clone();
        layer_peak_one.source_prefetch_peak_inflight = 1;
        assert!(validate_source_concurrency_mechanism(
            &layer_peak_one,
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            4,
        )
        .is_err());

        let mut read_peak_one = valid.clone();
        read_peak_one.source_prefetch_reads_peak_inflight = 1;
        assert!(validate_source_concurrency_mechanism(
            &read_peak_one,
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            4,
        )
        .is_err());

        for peak in 2..=4 {
            let mut observed = valid.clone();
            observed.source_prefetch_peak_inflight = peak;
            observed.source_prefetch_reads_peak_inflight = peak;
            assert!(validate_source_concurrency_mechanism(
                &observed,
                OracleScheduledResidencyMode::TokenBoundaryDirect,
                4,
            )
            .is_ok());
        }

        let mut layer_over_bound = valid.clone();
        layer_over_bound.source_prefetch_peak_inflight = 5;
        assert!(validate_source_concurrency_mechanism(
            &layer_over_bound,
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            4,
        )
        .is_err());

        let mut read_over_bound = valid.clone();
        read_over_bound.source_prefetch_reads_peak_inflight = 5;
        assert!(validate_source_concurrency_mechanism(
            &read_over_bound,
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            4,
        )
        .is_err());

        let mut current_nonzero = valid.clone();
        current_nonzero.source_prefetch_reads_current_inflight = 1;
        assert!(validate_source_concurrency_mechanism(
            &current_nonzero,
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            4,
        )
        .is_err());

        let mut accounting_error = valid.clone();
        accounting_error.source_prefetch_read_inflight_accounting_errors = 1;
        assert!(validate_source_concurrency_mechanism(
            &accounting_error,
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            4,
        )
        .is_err());

        let mut incomplete_batch = valid;
        incomplete_batch.source_prefetch_batches_completed = 1;
        assert!(validate_source_concurrency_mechanism(
            &incomplete_batch,
            OracleScheduledResidencyMode::TokenBoundaryDirect,
            4,
        )
        .is_err());
    }

    #[test]
    fn source_only_does_not_require_direct_read_concurrency() {
        let counters = OracleCounters {
            source_prefetch_batches_started: 1,
            source_prefetch_batches_completed: 1,
            source_prefetch_peak_inflight: 1,
            ..OracleCounters::default()
        };
        assert!(validate_source_concurrency_mechanism(
            &counters,
            OracleScheduledResidencyMode::SourceOnly,
            4,
        )
        .is_ok());
        assert!(
            !treatment_contract(OracleScheduledResidencyMode::SourceOnly)
                .independent_layer_source_tasks
        );
    }

    #[test]
    fn direct_future_read_guard_records_underflow_as_an_accounting_error() {
        let counters = Arc::new(Mutex::new(OracleCounters::default()));
        let guard = DirectFutureReadGuard::enter(counters.clone());
        counters.lock().source_prefetch_reads_current_inflight = 0;
        drop(guard);
        let counters = counters.lock();
        assert_eq!(counters.source_prefetch_reads_current_inflight, 0);
        assert_eq!(counters.source_prefetch_read_inflight_accounting_errors, 1);
    }

    #[test]
    fn source_only_counter_validation_rejects_physical_mutation() {
        let mut counters = valid_source_only_counters();
        counters.boundary_replacement_sets_started = 1;
        counters.boundary_physical_installs = 1;
        assert!(validate_oracle_counters(
            &counters,
            OracleScheduledResidencyMode::SourceOnly,
            4,
            4096,
        )
        .is_err());
    }

    #[test]
    fn serialized_h2d_contract_uses_one_ordered_queue_without_flush_submit() {
        let contract = treatment_contract(OracleScheduledResidencyMode::TokenBoundaryDirect);
        assert!(contract.independent_layer_source_tasks);
        assert!(!contract.per_layer_storage_batch_read_used);
        assert!(contract.token_boundary_h2d);
        assert!(contract.same_ordered_queue);
        assert!(!contract.production_direct_staging_used);
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
        assert!(validate_oracle_counters(
            &counters,
            OracleScheduledResidencyMode::SourceOnly,
            4,
            4096,
        )
        .is_err());
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
