//! PR3-DIFF0 qualification of ordinary GPU-native out-of-core inference.
//!
//! This command adds only observers around the normal shared runtime and its
//! existing `step_token` request path. It does not provide an alternate
//! loader, residency policy, compute path, or recovery path.

use crate::backend::gpu_native::GpuNativeProductionPhysicalInstallSnapshot;
use crate::backend::{GpuExpertIoSnapshot, GpuExpertMemorySnapshot};
use crate::engine::{
    GpuNativePhysicalInstallConcurrencyQualificationArm,
    GpuNativePhysicalInstallConcurrencyQualificationSnapshot, ProductionDemandSourceSnapshot,
    RoutedExpertExecutionSnapshot,
};
use crate::gpu_native_physical_install_staging::q4_route_parallel::{
    Arm as Q4ObservationArm, Mechanism as Q4Mechanism, Observation as Q4Observation,
};
use crate::gpu_native_real_benchmark::{
    BenchmarkFailure, BenchmarkProvenance, BenchmarkReport, EngineStorageSnapshot,
    GpuNativeResidencyDelta, PerRunResult, ProductionConfiguration, ProductionSemantics,
    RequestEvidence,
};
use crate::gpu_native_residency::GpuNativeTieredResidencySnapshot;
use crate::gpu_native_token_loop::{GpuNativeRecoverySnapshot, GpuNativeTokenLoopSnapshot};
use crate::qualification::{BuildProvenance, QualificationArtifacts};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

pub(crate) const SCHEMA: &str = "mer.gpu-native-out-of-core.v1";
pub(crate) const MODE: &str = "qualify-gpu-native-out-of-core";
pub(crate) const BASE_SHA: &str = "0621f614764663b4faa41a5763a6064be9fe154a";
const FROZEN_CONFIG_PATH: &str = "/home/randyap8/slice11-qwen3-coder-gpu-native.toml";
const FROZEN_CONFIG_SHA256: &str =
    "33d7cf96328d9c68b0ff45448d91d597d2e3a757cb99e6e61c72998ceabdd056";
const FROZEN_PROMPT: &str =
    "Write a Rust function that adds two i32 values and returns the result.";
const FROZEN_OUTPUT_TOKENS: usize = 128;
const FROZEN_WARMUP_RUNS: usize = 1;
const FROZEN_MEASURED_RUNS: usize = 3;
const FROZEN_CACHE_SLOTS: usize = 384;
const FROZEN_NUM_LAYERS: usize = 48;
const FROZEN_EXPERTS_PER_LAYER: u32 = 128;
const FROZEN_EXPERT_SIZE: usize = 2_658_304;
const FROZEN_EXPERT_FOOTPRINT_BYTES: u64 = 16_332_619_776;
const FROZEN_GPU_VRAM_CAPACITY_MB: usize = 2048;
const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const PROC_SELF_CGROUP: &str = "/proc/self/cgroup";
const PROC_SELF_STATUS: &str = "/proc/self/status";
const GPU_NATIVE_H2D_EVIDENCE_SOURCE: &str = "GPU-native physical-arena direct staging via Queue::write_buffer_with; legacy gpu_expert_io counters are non-authoritative";

#[derive(Clone, Debug)]
pub(crate) struct CommandArgs {
    pub(crate) config: PathBuf,
    pub(crate) expected_adapter_name: String,
    pub(crate) report_out: PathBuf,
    pub(crate) progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

#[derive(Clone)]
struct Prepared {
    spec: crate::ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_ids: Vec<u32>,
    resolved_config_sha256: String,
    provenance: BenchmarkProvenance,
    model_identity: crate::greedy_parity::ModelIdentityEvidence,
    request: RequestEvidence,
    production_configuration: ProductionConfiguration,
    canonical_model_data_dir: String,
}

#[derive(Clone, Debug, Serialize)]
struct FrozenWorkload {
    config: &'static str,
    config_sha256: &'static str,
    model: &'static str,
    quantization: &'static str,
    prompt: &'static str,
    output_tokens: usize,
    warmup_runs: usize,
    measured_runs: usize,
    cache_reset: &'static str,
    sampling: &'static str,
    expected_adapter_name: String,
    backend: &'static str,
    storage_cache_slots: usize,
    storage_no_direct: bool,
    storage_predict_fanout: usize,
    storage_pipeline_depth: u32,
    gpu_cache_enabled: bool,
    gpu_vram_capacity_mb: usize,
    gpu_promote_after_hits: u64,
    gpu_cache_dtype: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "bytes", rename_all = "snake_case")]
enum CgroupLimit {
    Max,
    Bytes(u64),
}

impl CgroupLimit {
    fn numeric(self) -> Option<u64> {
        match self {
            Self::Max => None,
            Self::Bytes(bytes) => Some(bytes),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MemoryEvents {
    max: u64,
    oom: u64,
    oom_kill: u64,
    oom_group_kill: Option<u64>,
    all: BTreeMap<String, u64>,
}

impl MemoryEvents {
    fn oom_observed(&self) -> bool {
        self.oom != 0 || self.oom_kill != 0 || self.oom_group_kill.unwrap_or(0) != 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CgroupMemoryState {
    memory_max: CgroupLimit,
    memory_current_bytes: u64,
    memory_peak_bytes: u64,
    memory_swap_max: CgroupLimit,
    memory_swap_current_bytes: u64,
    memory_events: MemoryEvents,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct ProcessMemoryState {
    vm_rss_bytes: u64,
    vm_hwm_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MemoryState {
    cgroup: CgroupMemoryState,
    process: ProcessMemoryState,
}

#[derive(Clone, Debug, Serialize)]
struct MemoryContract {
    cgroup_version: u8,
    cgroup_relative_path: String,
    initial: MemoryState,
    before_execution: Option<MemoryState>,
    after_warmup: Option<MemoryState>,
    after_measured: Option<MemoryState>,
    authoritative_whole_process_bound: &'static str,
}

#[derive(Clone, Debug)]
struct CgroupReader {
    relative_path: String,
    directory: PathBuf,
    proc_status_path: PathBuf,
}

impl CgroupReader {
    fn resolve(
        proc_self_cgroup: &Path,
        cgroup_root: &Path,
        proc_status_path: &Path,
    ) -> Result<Self, BenchmarkFailure> {
        let contents = read_required(proc_self_cgroup, "proc-self-cgroup")?;
        let relative_path = parse_unified_cgroup_path(&contents)
            .map_err(|detail| BenchmarkFailure::new("preflight", "cgroup-v2-unresolved", detail))?;
        let directory = resolve_cgroup_directory(cgroup_root, &relative_path)
            .map_err(|detail| BenchmarkFailure::new("preflight", "cgroup-v2-unresolved", detail))?;
        if !directory.is_dir() {
            return Err(BenchmarkFailure::new(
                "preflight",
                "cgroup-v2-directory-missing",
                format!(
                    "resolved cgroup directory {} does not exist",
                    directory.display()
                ),
            ));
        }
        Ok(Self {
            relative_path,
            directory,
            proc_status_path: proc_status_path.to_path_buf(),
        })
    }

    fn capture(&self) -> Result<MemoryState, BenchmarkFailure> {
        Ok(MemoryState {
            cgroup: self.capture_cgroup()?,
            process: parse_proc_status_memory(&read_required(
                &self.proc_status_path,
                "proc-self-status",
            )?)
            .map_err(|detail| BenchmarkFailure::new("memory", "proc-status-malformed", detail))?,
        })
    }

    fn capture_cgroup(&self) -> Result<CgroupMemoryState, BenchmarkFailure> {
        let read = |name: &str| read_required(&self.directory.join(name), name);
        Ok(CgroupMemoryState {
            memory_max: parse_cgroup_limit(&read("memory.max")?).map_err(|detail| {
                BenchmarkFailure::new("memory", "memory-max-malformed", detail)
            })?,
            memory_current_bytes: parse_cgroup_u64(&read("memory.current")?).map_err(|detail| {
                BenchmarkFailure::new("memory", "memory-current-malformed", detail)
            })?,
            memory_peak_bytes: parse_cgroup_u64(&read("memory.peak")?).map_err(|detail| {
                BenchmarkFailure::new("memory", "memory-peak-malformed", detail)
            })?,
            memory_swap_max: parse_cgroup_limit(&read("memory.swap.max")?).map_err(|detail| {
                BenchmarkFailure::new("memory", "memory-swap-max-malformed", detail)
            })?,
            memory_swap_current_bytes: parse_cgroup_u64(&read("memory.swap.current")?).map_err(
                |detail| BenchmarkFailure::new("memory", "memory-swap-current-malformed", detail),
            )?,
            memory_events: parse_memory_events(&read("memory.events")?).map_err(|detail| {
                BenchmarkFailure::new("memory", "memory-events-malformed", detail)
            })?,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
struct ExpertFootprint {
    num_layers: usize,
    experts_per_layer: u32,
    routed_expert_count: u64,
    expert_size_bytes: usize,
    routed_expert_footprint_bytes: u64,
    expected_frozen_footprint_bytes: u64,
    cgroup_memory_max_bytes: u64,
    excess_bytes: u64,
    footprint_to_limit_ratio: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct RamSnapshot {
    aggregate_cache_capacity_slots: usize,
    buffer_pool_primary_bytes: u64,
    buffer_pool_shadow_bytes: u64,
    buffer_pool_allocated_bytes: u64,
    resident_expert_buffer_bytes: u64,
    prepared_duplicate_expert_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
struct BoundedRamEvidence {
    aggregate_configured_cache_slots: usize,
    num_layers: usize,
    effective_per_layer_capacities: Vec<usize>,
    effective_aggregate_cache_capacity: usize,
    resident_expert_bytes_semantics: &'static str,
    before_execution: RamSnapshot,
    after_warmup: RamSnapshot,
    after_measured: RamSnapshot,
}

#[derive(Clone, Debug, Serialize)]
struct BoundedVramEvidence {
    configured_expert_vram_budget_bytes: u64,
    before_execution: GpuNativeTieredResidencySnapshot,
    after_measured: GpuNativeTieredResidencySnapshot,
    measured_delta: GpuNativeResidencyDelta,
    measured_h2d: GpuNativeH2dEvidence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct GpuNativeH2dEvidence {
    evidence_source: &'static str,
    ram_to_vram_installs: u64,
    direct_staging_writes: u64,
    direct_staging_successes: u64,
    physical_install_completions: u64,
    physical_slot_bytes_staged: u64,
    physical_bytes_staged: u64,
}

impl GpuNativeH2dEvidence {
    fn from_measured(measured: &PhaseWorkEvidence) -> Self {
        Self {
            evidence_source: GPU_NATIVE_H2D_EVIDENCE_SOURCE,
            ram_to_vram_installs: measured.aggregate.gpu_native_residency.ram_to_vram_installs,
            direct_staging_writes: measured.source.direct_staging_writes,
            direct_staging_successes: measured
                .production_physical_install
                .direct_staging_successes,
            physical_install_completions: measured.source.physical_install_completions,
            physical_slot_bytes_staged: measured.source.physical_slot_bytes_staged,
            physical_bytes_staged: measured.source.physical_bytes_staged,
        }
    }

    fn installs_positive_and_reconciled(self) -> bool {
        self.ram_to_vram_installs > 0
            && self.direct_staging_writes > 0
            && self.direct_staging_successes > 0
            && self.physical_install_completions > 0
            && self.ram_to_vram_installs == self.direct_staging_writes
            && self.direct_staging_writes == self.direct_staging_successes
            && self.direct_staging_successes == self.physical_install_completions
    }

    fn bytes_positive_and_reconciled(self) -> bool {
        self.physical_slot_bytes_staged > 0
            && self.physical_bytes_staged > 0
            && self.physical_slot_bytes_staged == self.physical_bytes_staged
    }
}

#[derive(Clone, Debug, Serialize)]
struct WorkEvidence {
    token_loop: GpuNativeTokenLoopSnapshot,
    recovery: GpuNativeRecoverySnapshot,
    routed_execution: RoutedExpertExecutionSnapshot,
    engine_storage: EngineStorageSnapshot,
    gpu_expert_io: GpuExpertIoSnapshot,
    gpu_expert_memory_before: GpuExpertMemorySnapshot,
    gpu_expert_memory_after: GpuExpertMemorySnapshot,
    gpu_native_residency: GpuNativeResidencyDelta,
}

#[derive(Clone)]
struct WorkStart {
    token_loop: GpuNativeTokenLoopSnapshot,
    recovery: GpuNativeRecoverySnapshot,
    routed: RoutedExpertExecutionSnapshot,
    engine_storage: EngineStorageSnapshot,
    gpu_io: GpuExpertIoSnapshot,
    gpu_memory: GpuExpertMemorySnapshot,
    residency: GpuNativeTieredResidencySnapshot,
}

impl WorkStart {
    fn capture(runtime: &crate::BenchRealRuntime) -> Result<Self, BenchmarkFailure> {
        let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
            BenchmarkFailure::new(
                "startup",
                "missing-gpu-native-token-loop",
                "PR3-DIFF0 did not construct the authoritative GPU-native token loop",
            )
        })?;
        Ok(Self {
            token_loop: token_loop.snapshot(),
            recovery: token_loop.recovery_snapshot(),
            routed: runtime.engine.routed_expert_execution_snapshot(),
            engine_storage: EngineStorageSnapshot::from_runtime(runtime),
            gpu_io: runtime.engine.gpu_expert_io_snapshot().ok_or_else(|| {
                BenchmarkFailure::new(
                    "startup",
                    "missing-gpu-io-snapshot",
                    "ordinary runtime did not expose routed-expert GPU I/O counters",
                )
            })?,
            gpu_memory: runtime.engine.gpu_expert_memory_snapshot().ok_or_else(|| {
                BenchmarkFailure::new(
                    "startup",
                    "missing-gpu-memory-snapshot",
                    "ordinary runtime did not expose routed-expert GPU memory counters",
                )
            })?,
            residency: runtime
                .engine
                .gpu_native_residency_snapshot()
                .ok_or_else(|| {
                    BenchmarkFailure::new(
                        "startup",
                        "missing-gpu-native-residency-snapshot",
                        "ordinary runtime did not expose GPU-native residency counters",
                    )
                })?,
        })
    }

    fn finish(self, runtime: &crate::BenchRealRuntime) -> Result<WorkEvidence, BenchmarkFailure> {
        let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "missing-gpu-native-token-loop",
                "GPU-native token loop disappeared after execution",
            )
        })?;
        let storage_after = EngineStorageSnapshot::from_runtime(runtime);
        let gpu_io_after = runtime.engine.gpu_expert_io_snapshot().ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "missing-gpu-io-snapshot",
                "routed-expert GPU I/O counters disappeared after execution",
            )
        })?;
        let gpu_memory_after = runtime.engine.gpu_expert_memory_snapshot().ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "missing-gpu-memory-snapshot",
                "routed-expert GPU memory counters disappeared after execution",
            )
        })?;
        let residency_after = runtime
            .engine
            .gpu_native_residency_snapshot()
            .ok_or_else(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "missing-gpu-native-residency-snapshot",
                    "GPU-native residency counters disappeared after execution",
                )
            })?;
        Ok(WorkEvidence {
            token_loop: crate::gpu_native_real_benchmark::token_loop_delta(
                self.token_loop,
                token_loop.snapshot(),
            )?,
            recovery: crate::gpu_native_real_benchmark::recovery_delta(
                self.recovery,
                token_loop.recovery_snapshot(),
            )?,
            routed_execution: crate::gpu_native_real_benchmark::routed_delta(
                self.routed,
                runtime.engine.routed_expert_execution_snapshot(),
            )?,
            engine_storage: storage_after.checked_delta(self.engine_storage)?,
            gpu_expert_io: crate::gpu_native_real_benchmark::gpu_io_delta(
                self.gpu_io,
                gpu_io_after,
            )?,
            gpu_expert_memory_before: self.gpu_memory,
            gpu_expert_memory_after: gpu_memory_after,
            gpu_native_residency: crate::gpu_native_real_benchmark::gpu_native_residency_delta(
                &self.residency,
                &residency_after,
            )?,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
struct PhaseWorkEvidence {
    aggregate: WorkEvidence,
    source: GpuNativePhysicalInstallConcurrencyQualificationSnapshot,
    production_source: ProductionDemandSourceSnapshot,
    production_physical_install: GpuNativeProductionPhysicalInstallSnapshot,
    q4_route_parallel: Q4Mechanism,
}

#[derive(Clone, Debug, Serialize)]
struct WorkReport {
    canonical_model_data_directory: String,
    direct_io_enabled: bool,
    qualification_observers_only: bool,
    warmup: PhaseWorkEvidence,
    measured: PhaseWorkEvidence,
}

#[derive(Clone, Debug, Serialize)]
struct RunBehavior {
    run_index: usize,
    generated_tokens: usize,
    generated_token_ids_sha256: String,
    generated_text_sha256: String,
}

impl From<&PerRunResult> for RunBehavior {
    fn from(run: &PerRunResult) -> Self {
        Self {
            run_index: run.run_index,
            generated_tokens: run.generated_tokens,
            generated_token_ids_sha256: run.generated_token_ids_sha256.clone(),
            generated_text_sha256: run.generated_text_sha256.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct BehaviorEvidence {
    warmup: Vec<RunBehavior>,
    measured: Vec<RunBehavior>,
}

#[derive(Clone, Copy, Debug, Default)]
struct GateInputs {
    frozen_configuration_exact: bool,
    cgroup_v2_resolved: bool,
    memory_max_numeric: bool,
    expert_footprint_exact: bool,
    footprint_exceeds_memory_max: bool,
    swap_disabled_and_unused: bool,
    no_oom_or_kill: bool,
    cgroup_peak_within_limit: bool,
    cache_geometry_exact: bool,
    ram_pool_accounting_exact: bool,
    resident_ram_within_pool: bool,
    shadow_pool_zero: bool,
    prepared_duplicates_zero: bool,
    vram_budget_bounded: bool,
    direct_io_enabled: bool,
    ordinary_production_path_only: bool,
    benchmark_and_runs_complete: bool,
    generated_hashes_recorded: bool,
    strict_checkpoint_runtime_validated: bool,
    demand_source_requests_gt_zero: bool,
    nvme_reads_gt_zero: bool,
    demand_nvme_bytes_gt_zero: bool,
    h2d_installs_gt_zero: bool,
    h2d_bytes_gt_zero: bool,
    no_degraded_substitution: bool,
    no_full_token_replay: bool,
    no_fatal_or_no_progress: bool,
    no_cache_or_batch_invariant_failures: bool,
    no_unexpected_gpu_status_bits: bool,
    no_mapping_or_install_invariant_failures: bool,
    no_speculative_work: bool,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct QualificationGates {
    frozen_configuration_exact: bool,
    cgroup_v2_resolved: bool,
    memory_max_numeric: bool,
    expert_footprint_exact: bool,
    footprint_exceeds_memory_max: bool,
    swap_disabled_and_unused: bool,
    no_oom_or_kill: bool,
    cgroup_peak_within_limit: bool,
    cache_geometry_exact: bool,
    ram_pool_accounting_exact: bool,
    resident_ram_within_pool: bool,
    shadow_pool_zero: bool,
    prepared_duplicates_zero: bool,
    vram_budget_bounded: bool,
    direct_io_enabled: bool,
    ordinary_production_path_only: bool,
    benchmark_and_runs_complete: bool,
    generated_hashes_recorded: bool,
    strict_checkpoint_runtime_validated: bool,
    demand_source_requests_gt_zero: bool,
    nvme_reads_gt_zero: bool,
    demand_nvme_bytes_gt_zero: bool,
    h2d_installs_gt_zero: bool,
    h2d_bytes_gt_zero: bool,
    no_degraded_substitution: bool,
    no_full_token_replay: bool,
    no_fatal_or_no_progress: bool,
    no_cache_or_batch_invariant_failures: bool,
    no_unexpected_gpu_status_bits: bool,
    no_mapping_or_install_invariant_failures: bool,
    no_speculative_work: bool,
    all_invariants_pass: bool,
}

impl QualificationGates {
    fn from_inputs(inputs: GateInputs) -> Self {
        let all_invariants_pass = inputs.frozen_configuration_exact
            && inputs.cgroup_v2_resolved
            && inputs.memory_max_numeric
            && inputs.expert_footprint_exact
            && inputs.footprint_exceeds_memory_max
            && inputs.swap_disabled_and_unused
            && inputs.no_oom_or_kill
            && inputs.cgroup_peak_within_limit
            && inputs.cache_geometry_exact
            && inputs.ram_pool_accounting_exact
            && inputs.resident_ram_within_pool
            && inputs.shadow_pool_zero
            && inputs.prepared_duplicates_zero
            && inputs.vram_budget_bounded
            && inputs.direct_io_enabled
            && inputs.ordinary_production_path_only
            && inputs.benchmark_and_runs_complete
            && inputs.generated_hashes_recorded
            && inputs.strict_checkpoint_runtime_validated
            && inputs.demand_source_requests_gt_zero
            && inputs.nvme_reads_gt_zero
            && inputs.demand_nvme_bytes_gt_zero
            && inputs.h2d_installs_gt_zero
            && inputs.h2d_bytes_gt_zero
            && inputs.no_degraded_substitution
            && inputs.no_full_token_replay
            && inputs.no_fatal_or_no_progress
            && inputs.no_cache_or_batch_invariant_failures
            && inputs.no_unexpected_gpu_status_bits
            && inputs.no_mapping_or_install_invariant_failures
            && inputs.no_speculative_work;
        Self {
            frozen_configuration_exact: inputs.frozen_configuration_exact,
            cgroup_v2_resolved: inputs.cgroup_v2_resolved,
            memory_max_numeric: inputs.memory_max_numeric,
            expert_footprint_exact: inputs.expert_footprint_exact,
            footprint_exceeds_memory_max: inputs.footprint_exceeds_memory_max,
            swap_disabled_and_unused: inputs.swap_disabled_and_unused,
            no_oom_or_kill: inputs.no_oom_or_kill,
            cgroup_peak_within_limit: inputs.cgroup_peak_within_limit,
            cache_geometry_exact: inputs.cache_geometry_exact,
            ram_pool_accounting_exact: inputs.ram_pool_accounting_exact,
            resident_ram_within_pool: inputs.resident_ram_within_pool,
            shadow_pool_zero: inputs.shadow_pool_zero,
            prepared_duplicates_zero: inputs.prepared_duplicates_zero,
            vram_budget_bounded: inputs.vram_budget_bounded,
            direct_io_enabled: inputs.direct_io_enabled,
            ordinary_production_path_only: inputs.ordinary_production_path_only,
            benchmark_and_runs_complete: inputs.benchmark_and_runs_complete,
            generated_hashes_recorded: inputs.generated_hashes_recorded,
            strict_checkpoint_runtime_validated: inputs.strict_checkpoint_runtime_validated,
            demand_source_requests_gt_zero: inputs.demand_source_requests_gt_zero,
            nvme_reads_gt_zero: inputs.nvme_reads_gt_zero,
            demand_nvme_bytes_gt_zero: inputs.demand_nvme_bytes_gt_zero,
            h2d_installs_gt_zero: inputs.h2d_installs_gt_zero,
            h2d_bytes_gt_zero: inputs.h2d_bytes_gt_zero,
            no_degraded_substitution: inputs.no_degraded_substitution,
            no_full_token_replay: inputs.no_full_token_replay,
            no_fatal_or_no_progress: inputs.no_fatal_or_no_progress,
            no_cache_or_batch_invariant_failures: inputs.no_cache_or_batch_invariant_failures,
            no_unexpected_gpu_status_bits: inputs.no_unexpected_gpu_status_bits,
            no_mapping_or_install_invariant_failures: inputs
                .no_mapping_or_install_invariant_failures,
            no_speculative_work: inputs.no_speculative_work,
            all_invariants_pass,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct OutOfCoreReport {
    schema: &'static str,
    mode: &'static str,
    base_sha: &'static str,
    qualification_only: bool,
    production_inference_behavior_changed: bool,
    benchmark_complete: bool,
    qualification_pass: bool,
    failure: Option<BenchmarkFailure>,
    frozen_workload: FrozenWorkload,
    provenance: BenchmarkProvenance,
    memory_contract: Option<MemoryContract>,
    expert_footprint: Option<ExpertFootprint>,
    bounded_ram: Option<BoundedRamEvidence>,
    bounded_vram: Option<BoundedVramEvidence>,
    work: Option<WorkReport>,
    behavior: Option<BehaviorEvidence>,
    gates: Option<QualificationGates>,
    benchmark: BenchmarkReport,
}

struct ExecutionEvidence {
    memory_before_execution: MemoryState,
    memory_after_warmup: MemoryState,
    memory_after_measured: MemoryState,
    ram_before_execution: RamSnapshot,
    ram_after_warmup: RamSnapshot,
    ram_after_measured: RamSnapshot,
    vram_before_execution: GpuNativeTieredResidencySnapshot,
    vram_after_measured: GpuNativeTieredResidencySnapshot,
    warmup: PhaseWorkEvidence,
    measured: PhaseWorkEvidence,
    behavior: BehaviorEvidence,
}

fn read_required(path: &Path, label: &str) -> Result<String, BenchmarkFailure> {
    std::fs::read_to_string(path).map_err(|error| {
        BenchmarkFailure::new(
            "memory",
            "required-memory-file-unavailable",
            format!("failed to read {label} at {}: {error}", path.display()),
        )
    })
}

fn parse_unified_cgroup_path(contents: &str) -> Result<String, String> {
    let mut unified = None;
    for (index, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields
            .next()
            .ok_or_else(|| format!("line {} has no hierarchy id", index + 1))?;
        let controllers = fields
            .next()
            .ok_or_else(|| format!("line {} has no controller field", index + 1))?;
        let path = fields
            .next()
            .ok_or_else(|| format!("line {} has no cgroup path", index + 1))?;
        if hierarchy == "0" && controllers.is_empty() {
            if unified.replace(path.to_string()).is_some() {
                return Err("/proc/self/cgroup contains multiple unified entries".into());
            }
        }
    }
    let path =
        unified.ok_or_else(|| "/proc/self/cgroup has no unified cgroup-v2 entry".to_string())?;
    if !path.starts_with('/') {
        return Err(format!("unified cgroup path {path:?} is not absolute"));
    }
    if path.contains('\0') {
        return Err("unified cgroup path contains NUL".into());
    }
    Ok(path)
}

fn resolve_cgroup_directory(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let relative = Path::new(relative);
    let mut directory = root.to_path_buf();
    for component in relative.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(part) => directory.push(part),
            _ => {
                return Err(format!(
                    "unified cgroup path {} contains an invalid component",
                    relative.display()
                ));
            }
        }
    }
    Ok(directory)
}

fn parse_cgroup_limit(contents: &str) -> Result<CgroupLimit, String> {
    let value = contents.trim();
    if value == "max" {
        return Ok(CgroupLimit::Max);
    }
    parse_cgroup_u64(value).map(CgroupLimit::Bytes)
}

fn parse_cgroup_u64(contents: &str) -> Result<u64, String> {
    let value = contents.trim();
    if value.is_empty() || value.split_whitespace().count() != 1 {
        return Err(format!("expected one unsigned integer, observed {value:?}"));
    }
    value
        .parse::<u64>()
        .map_err(|error| format!("invalid unsigned integer {value:?}: {error}"))
}

fn parse_memory_events(contents: &str) -> Result<MemoryEvents, String> {
    let mut all = BTreeMap::new();
    for (index, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = line.split_whitespace();
        let key = fields
            .next()
            .ok_or_else(|| format!("memory.events line {} has no key", index + 1))?;
        let value = fields
            .next()
            .ok_or_else(|| format!("memory.events line {} has no value", index + 1))?;
        if fields.next().is_some() {
            return Err(format!(
                "memory.events line {} has unexpected trailing fields",
                index + 1
            ));
        }
        let value = value.parse::<u64>().map_err(|error| {
            format!(
                "memory.events line {} has invalid value {value:?}: {error}",
                index + 1
            )
        })?;
        if all.insert(key.to_string(), value).is_some() {
            return Err(format!("memory.events contains duplicate key {key:?}"));
        }
    }
    let required = |key: &str| {
        all.get(key)
            .copied()
            .ok_or_else(|| format!("memory.events is missing required key {key:?}"))
    };
    Ok(MemoryEvents {
        max: required("max")?,
        oom: required("oom")?,
        oom_kill: required("oom_kill")?,
        oom_group_kill: all.get("oom_group_kill").copied(),
        all,
    })
}

fn parse_proc_status_memory(contents: &str) -> Result<ProcessMemoryState, String> {
    fn value(contents: &str, key: &str) -> Result<u64, String> {
        let mut found = None;
        for line in contents.lines() {
            let Some(rest) = line.strip_prefix(key) else {
                continue;
            };
            if found.is_some() {
                return Err(format!("/proc/self/status contains duplicate {key}"));
            }
            let mut fields = rest.split_whitespace();
            let kib = fields
                .next()
                .ok_or_else(|| format!("{key} has no numeric value"))?
                .parse::<u64>()
                .map_err(|error| format!("{key} has invalid numeric value: {error}"))?;
            if fields.next() != Some("kB") || fields.next().is_some() {
                return Err(format!("{key} must use exactly the kB unit"));
            }
            found = Some(
                kib.checked_mul(1024)
                    .ok_or_else(|| format!("{key} byte conversion overflowed"))?,
            );
        }
        found.ok_or_else(|| format!("/proc/self/status is missing {key}"))
    }
    Ok(ProcessMemoryState {
        vm_rss_bytes: value(contents, "VmRSS:")?,
        vm_hwm_bytes: value(contents, "VmHWM:")?,
    })
}

fn checked_expert_footprint(
    num_layers: usize,
    experts_per_layer: u32,
    expert_size: usize,
) -> Result<(u64, u64), BenchmarkFailure> {
    let layers = u64::try_from(num_layers).map_err(|_| {
        BenchmarkFailure::new(
            "preflight",
            "expert-footprint-overflow",
            "num_layers exceeds u64",
        )
    })?;
    let experts = u64::from(experts_per_layer);
    let size = u64::try_from(expert_size).map_err(|_| {
        BenchmarkFailure::new(
            "preflight",
            "expert-footprint-overflow",
            "expert_size exceeds u64",
        )
    })?;
    let count = layers.checked_mul(experts).ok_or_else(|| {
        BenchmarkFailure::new(
            "preflight",
            "expert-footprint-overflow",
            "num_layers * experts_per_layer overflowed",
        )
    })?;
    let bytes = count.checked_mul(size).ok_or_else(|| {
        BenchmarkFailure::new(
            "preflight",
            "expert-footprint-overflow",
            "routed expert count * expert_size overflowed",
        )
    })?;
    Ok((count, bytes))
}

fn footprint_excess_bytes(footprint: u64, memory_limit: u64) -> Result<u64, String> {
    if footprint <= memory_limit {
        return Err(format!(
            "routed expert footprint {footprint} must exceed memory.max {memory_limit}"
        ));
    }
    footprint
        .checked_sub(memory_limit)
        .ok_or_else(|| "expert footprint subtraction underflowed".to_string())
}

fn distributed_cache_capacities(total: usize, layers: usize) -> Result<Vec<usize>, String> {
    if total == 0 || layers == 0 || total < layers {
        return Err(format!(
            "aggregate cache slots ({total}) must be nonzero and at least num_layers ({layers})"
        ));
    }
    let base = total / layers;
    let extra = total % layers;
    let capacities: Vec<usize> = (0..layers)
        .map(|index| base + usize::from(index < extra))
        .collect();
    let aggregate = capacities
        .iter()
        .try_fold(0usize, |sum, value| sum.checked_add(*value))
        .ok_or_else(|| "per-layer cache capacity sum overflowed".to_string())?;
    if aggregate != total {
        return Err(format!(
            "per-layer cache capacity sum {aggregate} differs from aggregate budget {total}"
        ));
    }
    Ok(capacities)
}

fn ram_snapshot(runtime: &crate::BenchRealRuntime) -> RamSnapshot {
    let report = runtime.engine.report();
    RamSnapshot {
        aggregate_cache_capacity_slots: report.cache_capacity,
        buffer_pool_primary_bytes: report.expert_buffer_pool_primary_bytes,
        buffer_pool_shadow_bytes: report.expert_buffer_pool_shadow_bytes,
        buffer_pool_allocated_bytes: report.expert_buffer_pool_allocated_bytes,
        resident_expert_buffer_bytes: report.resident_expert_buffer_bytes,
        prepared_duplicate_expert_bytes: report.prepared_duplicate_expert_bytes,
    }
}

fn validate_cgroup_contract(state: &MemoryState) -> Result<u64, BenchmarkFailure> {
    let memory_max = state.cgroup.memory_max.numeric().ok_or_else(|| {
        BenchmarkFailure::new(
            "preflight",
            "unlimited-memory-max",
            "memory.max is unlimited; PR3-DIFF0 requires a numeric cgroup limit",
        )
    })?;
    if memory_max == 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "zero-memory-max",
            "memory.max must be greater than zero",
        ));
    }
    match state.cgroup.memory_swap_max {
        CgroupLimit::Max => {
            return Err(BenchmarkFailure::new(
                "preflight",
                "unlimited-swap-max",
                "memory.swap.max is unlimited; PR3-DIFF0 requires swap disabled",
            ));
        }
        CgroupLimit::Bytes(0) => {}
        CgroupLimit::Bytes(bytes) => {
            return Err(BenchmarkFailure::new(
                "preflight",
                "swap-enabled",
                format!("memory.swap.max must be zero; observed {bytes}"),
            ));
        }
    }
    if state.cgroup.memory_swap_current_bytes != 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "swap-in-use",
            format!(
                "memory.swap.current must be zero; observed {}",
                state.cgroup.memory_swap_current_bytes
            ),
        ));
    }
    if state.cgroup.memory_events.oom_observed() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "prior-oom-observed",
            format!(
                "memory.events already reports OOM activity: {:?}",
                state.cgroup.memory_events
            ),
        ));
    }
    if state.cgroup.memory_current_bytes > memory_max || state.cgroup.memory_peak_bytes > memory_max
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "cgroup-memory-bound-violated",
            format!(
                "memory.current={} memory.peak={} memory.max={memory_max}",
                state.cgroup.memory_current_bytes, state.cgroup.memory_peak_bytes
            ),
        ));
    }
    Ok(memory_max)
}

fn validate_frozen_config(cfg: &crate::config::Config) -> Result<(), BenchmarkFailure> {
    let valid = cfg.storage.cache_slots == FROZEN_CACHE_SLOTS
        && !cfg.storage.no_direct
        && cfg.storage.predict_fanout == 0
        && cfg.storage.pipeline_depth == 1
        && cfg.gpu_cache.enabled
        && cfg.gpu_cache.vram_capacity_mb == FROZEN_GPU_VRAM_CAPACITY_MB
        && cfg.gpu_cache.promote_after_hits == 1
        && cfg.gpu_cache.dtype == "q4_0"
        && cfg.model.num_layers == FROZEN_NUM_LAYERS
        && cfg.model.num_experts == FROZEN_EXPERTS_PER_LAYER
        && cfg.model.expert_size == FROZEN_EXPERT_SIZE;
    if !valid {
        return Err(BenchmarkFailure::new(
            "preflight",
            "frozen-configuration-drift",
            format!(
                "requires cache_slots={FROZEN_CACHE_SLOTS}, no_direct=false, predict_fanout=0, pipeline_depth=1, gpu_cache enabled/{FROZEN_GPU_VRAM_CAPACITY_MB} MiB/promote_after_hits=1/q4_0, and model geometry {FROZEN_NUM_LAYERS}x{FROZEN_EXPERTS_PER_LAYER} experts of {FROZEN_EXPERT_SIZE} bytes"
            ),
        ));
    }
    Ok(())
}

fn validate_isolation_config(cfg: &crate::config::Config) -> Result<(), BenchmarkFailure> {
    let predictive = &cfg.predictive;
    if predictive.locality_enabled
        || predictive.speculator_enabled
        || predictive.affinity_enabled
        || predictive.pregate_enabled
        || predictive.static_residency_fraction != 0.0
        || predictive.static_residency_profile.is_some()
        || cfg.storage.pin_after_observations != 0
        || predictive.cost_aware_eviction
        || cfg.storage.packed_blob.is_some()
        || cfg.storage.packed_manifest.is_some()
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "qualification-isolation-drift",
            "PR3-DIFF0 requires the frozen demand-only, strict-LRU, per-file production configuration",
        ));
    }
    Ok(())
}

fn prepare(args: &CommandArgs) -> Result<Prepared, Box<dyn std::error::Error>> {
    if args.config != Path::new(FROZEN_CONFIG_PATH) {
        return Err(BenchmarkFailure::new(
            "preflight",
            "wrong-frozen-config-path",
            format!(
                "PR3-DIFF0 requires config {FROZEN_CONFIG_PATH}; observed {}",
                args.config.display()
            ),
        )
        .into());
    }
    if args.expected_adapter_name.trim().is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "missing-expected-adapter",
            "PR3-DIFF0 requires an exact nonempty adapter name",
        )
        .into());
    }
    let build = BuildProvenance::embedded();
    crate::gpu_native_real_benchmark::validate_preflight_provenance(&build)?;
    let cfg = crate::config::Config::from_file(&args.config)?;
    crate::gpu_native_real_benchmark::validate_source_config(&cfg)?;
    validate_frozen_config(&cfg)?;
    validate_isolation_config(&cfg)?;
    let (artifacts, artifact_errors): (QualificationArtifacts, Vec<String>) =
        crate::qualification_artifacts(&args.config, &cfg);
    crate::gpu_native_real_benchmark::validate_artifacts(&artifacts, &artifact_errors)?;
    let observed_config_sha256 = artifacts
        .config
        .as_ref()
        .map(|digest| digest.sha256.as_str())
        .ok_or_else(|| {
            BenchmarkFailure::new(
                "preflight",
                "config-hash-unavailable",
                "PR3-DIFF0 could not hash the frozen config",
            )
        })?;
    if observed_config_sha256 != FROZEN_CONFIG_SHA256 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "frozen-config-hash-mismatch",
            format!("config SHA256 {observed_config_sha256} did not match {FROZEN_CONFIG_SHA256}"),
        )
        .into());
    }
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                BenchmarkFailure::new("preflight", "expert-metadata-unavailable", error)
            })?;
    crate::gpu_native_real_benchmark::validate_expert_metadata(&expert_metadata)?;
    let canonical_model_data_dir = std::fs::canonicalize(&cfg.model.data_dir)
        .map_err(|error| {
            BenchmarkFailure::new(
                "preflight",
                "model-data-directory-unavailable",
                format!(
                    "failed to canonicalize {}: {error}",
                    cfg.model.data_dir.display()
                ),
            )
        })?
        .display()
        .to_string();
    let spec = crate::resolve_real_cli_spec_from_config(
        cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let model_identity = crate::greedy_parity_model_identity(&spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "wrong-model-identity",
            format!(
                "PR3-DIFF0 requires exact Qwen3-Coder 30B-A3B Q4_0; observed {model_identity:?}"
            ),
        )
        .into());
    }
    let resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&spec)?;
    let tokenizer = crate::load_real_cli_tokenizer(
        &spec.cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let prompt_ids = tokenizer.encode(FROZEN_PROMPT)?;
    if prompt_ids.is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "empty-prompt-tokenization",
            "the frozen PR3-DIFF0 prompt encoded to zero tokens",
        )
        .into());
    }
    let (executable, executable_sha256) = crate::current_executable_identity()?;
    let executable_canonical_path = std::fs::canonicalize(&executable)
        .map_err(|error| {
            BenchmarkFailure::new(
                "preflight",
                "executable-provenance-unavailable",
                format!("failed to canonicalize {}: {error}", executable.display()),
            )
        })?
        .display()
        .to_string();
    if !crate::gpu_native_real_benchmark::is_hex(&executable_sha256, 64)
        || !crate::gpu_native_real_benchmark::is_hex(&resolved_config_sha256, 64)
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "provenance-unavailable",
            "PR3-DIFF0 executable or resolved-config SHA256 was unavailable",
        )
        .into());
    }
    let production_configuration =
        ProductionConfiguration::from_config(&spec.cfg, &expert_metadata);
    Ok(Prepared {
        spec,
        tokenizer,
        prompt_ids: prompt_ids.clone(),
        resolved_config_sha256: resolved_config_sha256.clone(),
        provenance: BenchmarkProvenance {
            build,
            executable_canonical_path,
            executable_sha256,
            resolved_config_sha256,
            artifacts,
            expert_metadata,
        },
        model_identity,
        request: RequestEvidence {
            prompt_sha256: crate::greedy_parity::sha256_hex(FROZEN_PROMPT.as_bytes()),
            prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&prompt_ids),
            prompt_token_count: prompt_ids.len(),
            requested_output_tokens: FROZEN_OUTPUT_TOKENS,
            greedy: true,
        },
        production_configuration,
        canonical_model_data_dir,
    })
}

fn frozen_workload(expected_adapter_name: String) -> FrozenWorkload {
    FrozenWorkload {
        config: FROZEN_CONFIG_PATH,
        config_sha256: FROZEN_CONFIG_SHA256,
        model: "Qwen3-Coder-30B-A3B-Instruct",
        quantization: "pure Q4_0",
        prompt: FROZEN_PROMPT,
        output_tokens: FROZEN_OUTPUT_TOKENS,
        warmup_runs: FROZEN_WARMUP_RUNS,
        measured_runs: FROZEN_MEASURED_RUNS,
        cache_reset: "keep",
        sampling: "greedy",
        expected_adapter_name,
        backend: "WGPU/Vulkan",
        storage_cache_slots: FROZEN_CACHE_SLOTS,
        storage_no_direct: false,
        storage_predict_fanout: 0,
        storage_pipeline_depth: 1,
        gpu_cache_enabled: true,
        gpu_vram_capacity_mb: FROZEN_GPU_VRAM_CAPACITY_MB,
        gpu_promote_after_hits: 1,
        gpu_cache_dtype: "q4_0",
    }
}

fn benchmark_report(prepared: &Prepared) -> BenchmarkReport {
    let mut report = BenchmarkReport::new(
        prepared.provenance.clone(),
        prepared.model_identity.clone(),
        prepared.request.clone(),
        crate::BenchRealCacheReset::Keep,
        FROZEN_WARMUP_RUNS,
        FROZEN_MEASURED_RUNS,
        prepared.production_configuration.clone(),
    );
    report.schema = SCHEMA;
    report.mode = MODE;
    report.optimization = "none-capability-qualification";
    report.immediate_comparison_commit = BASE_SHA;
    report.pr1b_a_experiment_commit = "not-applicable";
    report.pr1b_a_experiment_report_sha256 = "not-applicable";
    report.original_baseline_commit = BASE_SHA;
    report.production_semantics = ProductionSemantics {
        production_inference_math_changed: false,
        production_q4_changed: false,
        production_router_changed: false,
        production_attention_changed: false,
        production_rmsnorm_changed: false,
        production_lm_head_changed: false,
        production_residency_policy_changed: false,
        production_replay_policy_changed: false,
        production_prefetch_policy_changed: false,
        diagnostic_trace_enabled: true,
    };
    report
}

async fn execute_one_request(
    runtime: &crate::BenchRealRuntime,
    prepared: &Prepared,
    args: &CommandArgs,
    phase: &str,
    run_index: usize,
) -> Result<PerRunResult, BenchmarkFailure> {
    crate::with_progress_timeout(
        format!("{MODE} {phase} run {run_index}"),
        args.progress_watchdog,
        crate::gpu_native_real_benchmark::execute_request(
            runtime,
            &prepared.prompt_ids,
            FROZEN_OUTPUT_TOKENS,
            run_index,
        ),
    )
    .await
    .map_err(|error| {
        BenchmarkFailure::new(
            "inference",
            if phase == "warmup" {
                "warmup-request-failed"
            } else {
                "measured-request-failed"
            },
            error.to_string(),
        )
    })
}

fn phase_evidence(
    runtime: &crate::BenchRealRuntime,
    start: WorkStart,
    q4_observation: &Q4Observation,
) -> Result<PhaseWorkEvidence, BenchmarkFailure> {
    Ok(PhaseWorkEvidence {
        aggregate: start.finish(runtime)?,
        source: runtime
            .engine
            .gpu_native_physical_install_concurrency_qualification_snapshot()
            .ok_or_else(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "missing-physical-residency-observer",
                    "production physical-residency observer disappeared",
                )
            })?,
        production_source: runtime.engine.production_demand_source_snapshot(),
        production_physical_install: runtime
            .engine
            .production_physical_install_snapshot()
            .ok_or_else(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "missing-production-physical-install-snapshot",
                    "production physical-install counters disappeared",
                )
            })?,
        q4_route_parallel: q4_observation.take(),
    })
}

async fn execute_qualification(
    prepared: &Prepared,
    args: &CommandArgs,
    cgroup: &CgroupReader,
    benchmark: &mut BenchmarkReport,
) -> Result<ExecutionEvidence, BenchmarkFailure> {
    let runtime = crate::gpu_native_real_benchmark::construct_runtime(
        &prepared.spec,
        prepared.tokenizer.clone(),
        "shared",
        None,
        benchmark,
    )
    .await?;

    let execution = async {
        crate::gpu_native_real_benchmark::validate_and_record_runtime(
            &runtime,
            &prepared.resolved_config_sha256,
            &args.expected_adapter_name,
            benchmark,
        )?;
        runtime
            .engine
            .enable_gpu_native_physical_install_concurrency_qualification(
                GpuNativePhysicalInstallConcurrencyQualificationArm::Treatment,
            )
            .map_err(|detail| {
                BenchmarkFailure::new("startup", "physical-residency-observer-unavailable", detail)
            })?;
        let q4_observation = Arc::new(Q4Observation::new(Q4ObservationArm::Treatment));
        runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or_else(|| {
                BenchmarkFailure::new(
                    "startup",
                    "missing-gpu-native-token-loop",
                    "ordinary runtime has no authoritative token loop",
                )
            })?
            .enable_q4_route_parallel_qualification(q4_observation.clone())
            .map_err(|detail| {
                BenchmarkFailure::new("startup", "q4-production-observer-unavailable", detail)
            })?;

        let memory_before_execution = cgroup.capture()?;
        let ram_before_execution = ram_snapshot(&runtime);
        let vram_before_execution =
            runtime
                .engine
                .gpu_native_residency_snapshot()
                .ok_or_else(|| {
                    BenchmarkFailure::new(
                        "startup",
                        "missing-gpu-native-residency-snapshot",
                        "ordinary runtime did not expose its physical expert arena",
                    )
                })?;

        let warmup_start = WorkStart::capture(&runtime)?;
        let warmup_result = execute_one_request(&runtime, prepared, args, "warmup", 0).await?;
        benchmark.warmup_runs_completed = benchmark
            .warmup_runs_completed
            .checked_add(1)
            .ok_or_else(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "warmup-count-overflow",
                    "warmup run count overflowed",
                )
            })?;
        let warmup = phase_evidence(&runtime, warmup_start, &q4_observation)?;
        let memory_after_warmup = cgroup.capture()?;
        let ram_after_warmup = ram_snapshot(&runtime);

        runtime
            .engine
            .reset_gpu_native_demand_source_qualification()
            .map_err(|detail| {
                BenchmarkFailure::new(
                    "postcondition",
                    "observer-reset-failed",
                    format!("failed to reset qualification counters after warmup: {detail}"),
                )
            })?;

        let measured_start = WorkStart::capture(&runtime)?;
        let mut measured_behavior = Vec::with_capacity(FROZEN_MEASURED_RUNS);
        for index in 0..FROZEN_MEASURED_RUNS {
            let result = execute_one_request(&runtime, prepared, args, "measured", index).await?;
            measured_behavior.push(RunBehavior::from(&result));
            benchmark.per_run_results.push(result);
        }
        let measured = phase_evidence(&runtime, measured_start, &q4_observation)?;
        let memory_after_measured = cgroup.capture()?;
        let ram_after_measured = ram_snapshot(&runtime);
        let vram_after_measured =
            runtime
                .engine
                .gpu_native_residency_snapshot()
                .ok_or_else(|| {
                    BenchmarkFailure::new(
                        "postcondition",
                        "missing-gpu-native-residency-snapshot",
                        "GPU-native residency snapshot disappeared after measurement",
                    )
                })?;

        Ok(ExecutionEvidence {
            memory_before_execution,
            memory_after_warmup,
            memory_after_measured,
            ram_before_execution,
            ram_after_warmup,
            ram_after_measured,
            vram_before_execution,
            vram_after_measured,
            warmup,
            measured,
            behavior: BehaviorEvidence {
                warmup: vec![RunBehavior::from(&warmup_result)],
                measured: measured_behavior,
            },
        })
    }
    .await;

    let shutdown =
        crate::gpu_native_real_benchmark::shutdown_runtime(runtime, "shared", None, benchmark)
            .await;
    match (execution, shutdown) {
        (Ok(evidence), Ok(())) => {
            benchmark.finish()?;
            Ok(evidence)
        }
        (Err(execution_error), Ok(())) => Err(execution_error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error),
        (Err(execution_error), Err(shutdown_error)) => Err(BenchmarkFailure::new(
            "postcondition",
            "execution-and-shutdown-failed",
            format!("{execution_error}; {shutdown_error}"),
        )),
    }
}

fn physical_install_invariants(snapshot: &GpuNativeProductionPhysicalInstallSnapshot) -> bool {
    snapshot.physical_install_sets > 0
        && snapshot.physical_install_experts == snapshot.reservation_attempts
        && snapshot.reservation_attempts == snapshot.reservation_successes
        && snapshot.reservation_failures == 0
        && snapshot.physical_install_attempts == snapshot.physical_stage_attempts
        && snapshot.physical_stage_attempts == snapshot.physical_stage_completions
        && snapshot.physical_stage_failures == 0
        && snapshot.physical_stage_completions == snapshot.direct_staging_successes
        && snapshot.direct_staging_unavailable == 0
        && snapshot.direct_staging_allocation_fallbacks == 0
        && snapshot.ordered_commit_attempts == snapshot.ordered_commit_completions
        && snapshot.ordered_commit_completions == snapshot.physical_stage_completions
        && snapshot.ordered_commit_failures == 0
        && snapshot.ordered_commit_violations == 0
        && snapshot.unpublished_physical_writes_after_failure == 0
        && snapshot.physical_install_failures == 0
}

fn build_gates(
    memory: &MemoryContract,
    footprint: &ExpertFootprint,
    ram: &BoundedRamEvidence,
    vram: &BoundedVramEvidence,
    work: &WorkReport,
    behavior: &BehaviorEvidence,
    benchmark: &BenchmarkReport,
) -> QualificationGates {
    let Some(final_memory) = memory.after_measured.as_ref() else {
        return QualificationGates::default();
    };
    let memory_max = final_memory.cgroup.memory_max.numeric();
    let memory_max_stable = [
        &memory.initial,
        memory.before_execution.as_ref().unwrap_or(&memory.initial),
        memory.after_warmup.as_ref().unwrap_or(&memory.initial),
        final_memory,
    ]
    .iter()
    .all(|state| state.cgroup.memory_max.numeric() == memory_max);
    let swap_disabled_and_unused = [
        &memory.initial,
        memory.before_execution.as_ref().unwrap_or(&memory.initial),
        memory.after_warmup.as_ref().unwrap_or(&memory.initial),
        final_memory,
    ]
    .iter()
    .all(|state| {
        state.cgroup.memory_swap_max == CgroupLimit::Bytes(0)
            && state.cgroup.memory_swap_current_bytes == 0
    });
    let no_oom_or_kill = [
        &memory.initial,
        memory.before_execution.as_ref().unwrap_or(&memory.initial),
        memory.after_warmup.as_ref().unwrap_or(&memory.initial),
        final_memory,
    ]
    .iter()
    .all(|state| !state.cgroup.memory_events.oom_observed());
    let cgroup_peak_within_limit = memory_max
        .is_some_and(|limit| final_memory.cgroup.memory_peak_bytes <= limit)
        && memory_max_stable;

    let ram_snapshots = [ram.before_execution, ram.after_warmup, ram.after_measured];
    let cache_geometry_exact = ram.aggregate_configured_cache_slots == FROZEN_CACHE_SLOTS
        && ram.num_layers == FROZEN_NUM_LAYERS
        && ram.effective_per_layer_capacities.len() == FROZEN_NUM_LAYERS
        && ram
            .effective_per_layer_capacities
            .iter()
            .all(|value| *value == 8)
        && ram.effective_aggregate_cache_capacity == FROZEN_CACHE_SLOTS
        && ram_snapshots
            .iter()
            .all(|snapshot| snapshot.aggregate_cache_capacity_slots == FROZEN_CACHE_SLOTS);
    let ram_pool_accounting_exact = ram_snapshots.iter().all(|snapshot| {
        snapshot.buffer_pool_allocated_bytes
            == snapshot
                .buffer_pool_primary_bytes
                .checked_add(snapshot.buffer_pool_shadow_bytes)
                .unwrap_or(u64::MAX)
    });
    let resident_ram_within_pool = ram_snapshots.iter().all(|snapshot| {
        snapshot.resident_expert_buffer_bytes <= snapshot.buffer_pool_allocated_bytes
    });
    let shadow_pool_zero = ram_snapshots
        .iter()
        .all(|snapshot| snapshot.buffer_pool_shadow_bytes == 0);
    let prepared_duplicates_zero = ram_snapshots
        .iter()
        .all(|snapshot| snapshot.prepared_duplicate_expert_bytes == 0);

    let vram_budget_bounded = vram.configured_expert_vram_budget_bytes
        == vram.after_measured.model_expert_budget_bytes
        && vram.after_measured.model_arena_allocation_bytes
            <= vram.after_measured.model_expert_budget_bytes
        && vram.after_measured.resident_physical_slots <= vram.after_measured.model_slot_capacity
        && vram.after_measured.free_physical_slots <= vram.after_measured.model_slot_capacity;
    let measured = &work.measured;
    let ordinary_production_path_only = measured.source.treatment_uses_ordinary_production_path
        && measured
            .source
            .normal_production_uses_concurrent_physical_staging
        && measured
            .production_source
            .ordinary_production_path_exercised
        && benchmark
            .runtime_contract
            .as_ref()
            .is_some_and(|contract| contract.ordinary_step_token_only)
        && work
            .warmup
            .q4_route_parallel
            .ordinary_production_route_parallel_exercised()
        && measured
            .q4_route_parallel
            .ordinary_production_route_parallel_exercised();
    let benchmark_and_runs_complete = benchmark.benchmark_complete
        && benchmark.warmup_runs_completed == FROZEN_WARMUP_RUNS
        && benchmark.per_run_results.len() == FROZEN_MEASURED_RUNS
        && behavior.warmup.len() == FROZEN_WARMUP_RUNS
        && behavior.measured.len() == FROZEN_MEASURED_RUNS
        && behavior
            .warmup
            .iter()
            .chain(&behavior.measured)
            .all(|run| run.generated_tokens == FROZEN_OUTPUT_TOKENS);
    let strict_checkpoint_runtime_validated = benchmark
        .runtime_contract
        .as_ref()
        .is_some_and(|contract| contract.strict_fail_closed_routed_experts)
        && benchmark.model_load.as_ref().is_some_and(|load| {
            load.strict
                && load.loaded_tensors == load.required_tensors
                && !load.seeded_fallback_remained
                && load.loader != "seeded"
        });
    let phases = [&work.warmup, measured];
    let no_degraded_substitution = phases.iter().all(|phase| {
        phase.aggregate.routed_execution.gpu_dispatch_failures == 0
            && phase
                .aggregate
                .routed_execution
                .cpu_routed_expert_dispatches
                == 0
            && phase.aggregate.routed_execution.gpu_cpu_fallbacks == 0
            && phase
                .aggregate
                .routed_execution
                .degraded_expert_substitutions
                == 0
    });
    let no_full_token_replay = phases.iter().all(|phase| {
        phase.aggregate.token_loop.replay_attempts == 0
            && phase.aggregate.recovery.full_token_replay_attempts == 0
    });
    let no_fatal_or_no_progress = phases.iter().all(|phase| {
        phase.aggregate.token_loop.fatal_failures == 0
            && phase.aggregate.token_loop.no_progress_failures == 0
    });
    let no_cache_or_batch_invariant_failures = phases.iter().all(|phase| {
        phase.production_source.production_cache_reservation_leaks == 0
            && phase.production_source.production_batch_commit_violations == 0
            && phase.production_source.stale_singleflight_entries == 0
    });
    let no_mapping_or_install_invariant_failures = phases.iter().all(|phase| {
        physical_install_invariants(&phase.production_physical_install)
            && phase.source.physical_install_attempts == phase.source.physical_install_completions
            && phase.source.mapping_publications == phase.source.physical_install_completions
            && phase.source.direct_staging_failures == 0
            && phase.source.reservation_failures == 0
            && phase.source.physical_stage_failures == 0
            && phase.source.ordered_commit_failures == 0
            && phase.source.ordered_commit_violations == 0
            && phase.source.unpublished_physical_writes_after_failure == 0
    });
    let no_speculative_work = phases.iter().all(|phase| {
        phase.aggregate.engine_storage.prefetch_completed == 0
            && phase.aggregate.gpu_native_residency.speculative_requests == 0
            && phase.aggregate.gpu_native_residency.speculative_vram_hits == 0
            && phase
                .aggregate
                .gpu_native_residency
                .speculative_ram_to_vram_installs
                == 0
            && phase
                .aggregate
                .gpu_native_residency
                .speculative_dropped_capacity_or_pressure
                == 0
    });

    QualificationGates::from_inputs(GateInputs {
        frozen_configuration_exact: benchmark
            .production_configuration
            .cache_residency
            .ram_cache_slots
            == FROZEN_CACHE_SLOTS
            && benchmark.production_configuration.cache_residency.direct_io
            && benchmark
                .production_configuration
                .cache_residency
                .pipeline_depth
                == 1
            && benchmark
                .production_configuration
                .predictor_prefetch
                .predict_fanout
                == 0
            && benchmark
                .production_configuration
                .cache_residency
                .gpu_cache_enabled
            && benchmark
                .production_configuration
                .cache_residency
                .gpu_vram_capacity_mb
                == FROZEN_GPU_VRAM_CAPACITY_MB
            && benchmark
                .production_configuration
                .cache_residency
                .gpu_promote_after_hits
                == 1
            && benchmark
                .production_configuration
                .cache_residency
                .gpu_cache_dtype
                == "q4_0",
        cgroup_v2_resolved: memory.cgroup_version == 2,
        memory_max_numeric: memory_max.is_some() && memory_max_stable,
        expert_footprint_exact: footprint.routed_expert_count == 6144
            && footprint.expert_size_bytes == FROZEN_EXPERT_SIZE
            && footprint.routed_expert_footprint_bytes == FROZEN_EXPERT_FOOTPRINT_BYTES,
        footprint_exceeds_memory_max: footprint.routed_expert_footprint_bytes
            > footprint.cgroup_memory_max_bytes,
        swap_disabled_and_unused,
        no_oom_or_kill,
        cgroup_peak_within_limit,
        cache_geometry_exact,
        ram_pool_accounting_exact,
        resident_ram_within_pool,
        shadow_pool_zero,
        prepared_duplicates_zero,
        vram_budget_bounded,
        direct_io_enabled: work.direct_io_enabled,
        ordinary_production_path_only,
        benchmark_and_runs_complete,
        generated_hashes_recorded: behavior.warmup.iter().chain(&behavior.measured).all(|run| {
            crate::gpu_native_real_benchmark::is_hex(&run.generated_token_ids_sha256, 64)
                && crate::gpu_native_real_benchmark::is_hex(&run.generated_text_sha256, 64)
        }),
        strict_checkpoint_runtime_validated,
        demand_source_requests_gt_zero: measured.source.demand_source_requests > 0,
        nvme_reads_gt_zero: measured.source.source_nvme_reads > 0,
        demand_nvme_bytes_gt_zero: measured.source.source_nvme_bytes > 0,
        h2d_installs_gt_zero: vram.measured_h2d.installs_positive_and_reconciled(),
        h2d_bytes_gt_zero: vram.measured_h2d.bytes_positive_and_reconciled(),
        no_degraded_substitution,
        no_full_token_replay,
        no_fatal_or_no_progress,
        no_cache_or_batch_invariant_failures,
        no_unexpected_gpu_status_bits: phases
            .iter()
            .all(|phase| phase.q4_route_parallel.unexpected_status_bits() == 0),
        no_mapping_or_install_invariant_failures,
        no_speculative_work,
    })
}

fn emit_report<T: Serialize>(report: &T, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    use std::io::Write;
    file.write_all(&json)?;
    eprintln!(
        "GPU-native out-of-core qualification report written to {}",
        path.display()
    );
    Ok(())
}

fn fail_report(
    report: &mut OutOfCoreReport,
    failure: BenchmarkFailure,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    report.benchmark.fail(failure.clone());
    report.failure = Some(failure.clone());
    emit_report(report, path)?;
    Err(failure.to_string().into())
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<(), Box<dyn std::error::Error>> {
    let prepared = prepare(&args)?;
    let benchmark = benchmark_report(&prepared);
    let mut report = OutOfCoreReport {
        schema: SCHEMA,
        mode: MODE,
        base_sha: BASE_SHA,
        qualification_only: true,
        production_inference_behavior_changed: false,
        benchmark_complete: false,
        qualification_pass: false,
        failure: None,
        frozen_workload: frozen_workload(args.expected_adapter_name.clone()),
        provenance: prepared.provenance.clone(),
        memory_contract: None,
        expert_footprint: None,
        bounded_ram: None,
        bounded_vram: None,
        work: None,
        behavior: None,
        gates: None,
        benchmark,
    };

    let cgroup = match CgroupReader::resolve(
        Path::new(PROC_SELF_CGROUP),
        Path::new(CGROUP_ROOT),
        Path::new(PROC_SELF_STATUS),
    ) {
        Ok(cgroup) => cgroup,
        Err(failure) => return fail_report(&mut report, failure, &args.report_out),
    };
    let initial_memory = match cgroup.capture() {
        Ok(memory) => memory,
        Err(failure) => return fail_report(&mut report, failure, &args.report_out),
    };
    report.memory_contract = Some(MemoryContract {
        cgroup_version: 2,
        cgroup_relative_path: cgroup.relative_path.clone(),
        initial: initial_memory.clone(),
        before_execution: None,
        after_warmup: None,
        after_measured: None,
        authoritative_whole_process_bound: "cgroup-v2 memory.peak <= numeric memory.max",
    });
    let memory_max = match validate_cgroup_contract(&initial_memory) {
        Ok(value) => value,
        Err(failure) => return fail_report(&mut report, failure, &args.report_out),
    };
    let (routed_expert_count, routed_expert_footprint_bytes) = match checked_expert_footprint(
        prepared.spec.cfg.model.num_layers,
        prepared.spec.cfg.model.num_experts,
        prepared.spec.cfg.model.expert_size,
    ) {
        Ok(value) => value,
        Err(failure) => return fail_report(&mut report, failure, &args.report_out),
    };
    if routed_expert_footprint_bytes != FROZEN_EXPERT_FOOTPRINT_BYTES {
        return fail_report(
            &mut report,
            BenchmarkFailure::new(
                "preflight",
                "frozen-expert-footprint-mismatch",
                format!(
                    "computed {routed_expert_footprint_bytes} bytes; expected {FROZEN_EXPERT_FOOTPRINT_BYTES}"
                ),
            ),
            &args.report_out,
        );
    }
    let excess_bytes = match footprint_excess_bytes(routed_expert_footprint_bytes, memory_max) {
        Ok(excess) => excess,
        Err(detail) => {
            report.expert_footprint = Some(ExpertFootprint {
                num_layers: prepared.spec.cfg.model.num_layers,
                experts_per_layer: prepared.spec.cfg.model.num_experts,
                routed_expert_count,
                expert_size_bytes: prepared.spec.cfg.model.expert_size,
                routed_expert_footprint_bytes,
                expected_frozen_footprint_bytes: FROZEN_EXPERT_FOOTPRINT_BYTES,
                cgroup_memory_max_bytes: memory_max,
                excess_bytes: 0,
                footprint_to_limit_ratio: routed_expert_footprint_bytes as f64 / memory_max as f64,
            });
            return fail_report(
                &mut report,
                BenchmarkFailure::new(
                    "preflight",
                    "expert-footprint-does-not-exceed-memory-limit",
                    detail,
                ),
                &args.report_out,
            );
        }
    };
    report.expert_footprint = Some(ExpertFootprint {
        num_layers: prepared.spec.cfg.model.num_layers,
        experts_per_layer: prepared.spec.cfg.model.num_experts,
        routed_expert_count,
        expert_size_bytes: prepared.spec.cfg.model.expert_size,
        routed_expert_footprint_bytes,
        expected_frozen_footprint_bytes: FROZEN_EXPERT_FOOTPRINT_BYTES,
        cgroup_memory_max_bytes: memory_max,
        excess_bytes,
        footprint_to_limit_ratio: routed_expert_footprint_bytes as f64 / memory_max as f64,
    });

    let execution =
        match execute_qualification(&prepared, &args, &cgroup, &mut report.benchmark).await {
            Ok(execution) => execution,
            Err(failure) => return fail_report(&mut report, failure, &args.report_out),
        };
    let capacities = distributed_cache_capacities(
        prepared.spec.cfg.storage.cache_slots,
        prepared.spec.cfg.model.num_layers,
    )
    .map_err(|detail| BenchmarkFailure::new("postcondition", "cache-geometry-invalid", detail))?;
    let aggregate_capacity = capacities.iter().sum();
    let bounded_ram = BoundedRamEvidence {
        aggregate_configured_cache_slots: prepared.spec.cfg.storage.cache_slots,
        num_layers: prepared.spec.cfg.model.num_layers,
        effective_per_layer_capacities: capacities,
        effective_aggregate_cache_capacity: aggregate_capacity,
        resident_expert_bytes_semantics: "occupancy within the preallocated expert buffer pool; do not add to pool allocation bytes",
        before_execution: execution.ram_before_execution,
        after_warmup: execution.ram_after_warmup,
        after_measured: execution.ram_after_measured,
    };
    let configured_expert_vram_budget_bytes =
        u64::try_from(prepared.spec.cfg.gpu_cache.vram_capacity_mb)
            .ok()
            .and_then(|value| value.checked_mul(1024 * 1024))
            .ok_or_else(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "vram-budget-overflow",
                    "configured expert VRAM budget overflowed byte conversion",
                )
            })?;
    let bounded_vram = BoundedVramEvidence {
        configured_expert_vram_budget_bytes,
        before_execution: execution.vram_before_execution.clone(),
        after_measured: execution.vram_after_measured.clone(),
        measured_delta: execution.measured.aggregate.gpu_native_residency,
        measured_h2d: GpuNativeH2dEvidence::from_measured(&execution.measured),
    };
    let memory_contract = report
        .memory_contract
        .as_mut()
        .expect("memory contract stored before execution");
    memory_contract.before_execution = Some(execution.memory_before_execution);
    memory_contract.after_warmup = Some(execution.memory_after_warmup);
    memory_contract.after_measured = Some(execution.memory_after_measured);
    let work = WorkReport {
        canonical_model_data_directory: prepared.canonical_model_data_dir,
        direct_io_enabled: !prepared.spec.cfg.storage.no_direct,
        qualification_observers_only: true,
        warmup: execution.warmup,
        measured: execution.measured,
    };
    let gates = build_gates(
        memory_contract,
        report
            .expert_footprint
            .as_ref()
            .expect("expert footprint stored before execution"),
        &bounded_ram,
        &bounded_vram,
        &work,
        &execution.behavior,
        &report.benchmark,
    );
    let qualification_pass = gates.all_invariants_pass;
    report.benchmark_complete = report.benchmark.benchmark_complete;
    report.qualification_pass = qualification_pass;
    report.benchmark.qualification_pass = qualification_pass;
    report.benchmark.correctness_qualification_pending = false;
    report.bounded_ram = Some(bounded_ram);
    report.bounded_vram = Some(bounded_vram);
    report.work = Some(work);
    report.behavior = Some(execution.behavior);
    report.gates = Some(gates);
    emit_report(&report, &args.report_out)?;
    if qualification_pass {
        Ok(())
    } else {
        Err("PR3-DIFF0 qualification gates did not all pass; see emitted report".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("mer-pr3-diff0-{label}-{}-{id}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write(path: impl AsRef<Path>, contents: &str) {
        std::fs::write(path, contents).unwrap();
    }

    fn valid_events() -> &'static str {
        "low 0\nhigh 0\nmax 3\noom 0\noom_kill 0\noom_group_kill 0\n"
    }

    fn memory_state(memory_max: CgroupLimit, swap_max: CgroupLimit) -> MemoryState {
        MemoryState {
            cgroup: CgroupMemoryState {
                memory_max,
                memory_current_bytes: 1024,
                memory_peak_bytes: 2048,
                memory_swap_max: swap_max,
                memory_swap_current_bytes: 0,
                memory_events: parse_memory_events(valid_events()).unwrap(),
            },
            process: ProcessMemoryState {
                vm_rss_bytes: 1024,
                vm_hwm_bytes: 2048,
            },
        }
    }

    #[test]
    fn unified_cgroup_path_parses_and_rejects_missing_or_malformed_entries() {
        assert_eq!(
            parse_unified_cgroup_path("0::/system.slice/mer.service\n").unwrap(),
            "/system.slice/mer.service"
        );
        assert!(parse_unified_cgroup_path("2:memory:/legacy\n").is_err());
        assert!(parse_unified_cgroup_path("0::relative\n").is_err());
        assert!(parse_unified_cgroup_path("0::/one\n0::/two\n").is_err());
        assert!(parse_unified_cgroup_path("malformed\n").is_err());
        assert!(resolve_cgroup_directory(Path::new("/x"), "/a/../b").is_err());
    }

    #[test]
    fn cgroup_limit_distinguishes_max_from_numeric_values() {
        assert_eq!(parse_cgroup_limit("max\n").unwrap(), CgroupLimit::Max);
        assert_eq!(
            parse_cgroup_limit("12345\n").unwrap(),
            CgroupLimit::Bytes(12_345)
        );
        assert!(parse_cgroup_limit("12 KiB\n").is_err());
        assert!(parse_cgroup_limit("-1\n").is_err());
    }

    #[test]
    fn memory_events_parser_requires_core_keys_and_parses_optional_group_kill() {
        let events = parse_memory_events(valid_events()).unwrap();
        assert_eq!(events.max, 3);
        assert_eq!(events.oom, 0);
        assert_eq!(events.oom_kill, 0);
        assert_eq!(events.oom_group_kill, Some(0));
        assert!(!events.oom_observed());
        assert!(parse_memory_events("max 0\noom 0\n").is_err());
        assert!(parse_memory_events("max 0\noom 0\noom_kill nope\n").is_err());
        assert!(parse_memory_events("max 0\nmax 1\noom 0\noom_kill 0\n").is_err());
    }

    #[test]
    fn authoritative_report_refuses_overwrite_and_preserves_original_bytes() {
        let fixture = TempDir::new("immutable-report");
        let path = fixture.0.join("nested/report.json");
        let mut report = serde_json::json!({ "qualification_pass": false });

        emit_report(&report, &path).unwrap();
        let original = std::fs::read(&path).unwrap();
        assert_eq!(original.last(), Some(&b'\n'));

        report["qualification_pass"] = true.into();
        let error = emit_report(&report, &path).unwrap_err();
        let io_error = error
            .downcast_ref::<std::io::Error>()
            .expect("existing destination must return an I/O error");
        assert_eq!(io_error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&path).unwrap(), original);
    }

    #[test]
    fn proc_status_kib_values_convert_to_bytes_with_checked_arithmetic() {
        let memory = parse_proc_status_memory(
            "Name:\tmer\nVmPeak:\t99 kB\nVmHWM:\t456 kB\nVmRSS:\t123 kB\n",
        )
        .unwrap();
        assert_eq!(memory.vm_rss_bytes, 123 * 1024);
        assert_eq!(memory.vm_hwm_bytes, 456 * 1024);
        assert!(parse_proc_status_memory("VmRSS:\t1 MB\nVmHWM:\t2 kB\n").is_err());
        assert!(parse_proc_status_memory("VmRSS:\t1 kB\n").is_err());
        assert!(
            parse_proc_status_memory(&format!("VmRSS:\t{} kB\nVmHWM:\t1 kB\n", u64::MAX)).is_err()
        );
    }

    #[test]
    fn expert_footprint_uses_checked_arithmetic_and_matches_frozen_artifact() {
        let (count, bytes) = checked_expert_footprint(
            FROZEN_NUM_LAYERS,
            FROZEN_EXPERTS_PER_LAYER,
            FROZEN_EXPERT_SIZE,
        )
        .unwrap();
        assert_eq!(count, 6144);
        assert_eq!(bytes, FROZEN_EXPERT_FOOTPRINT_BYTES);
        assert!(checked_expert_footprint(usize::MAX, u32::MAX, usize::MAX).is_err());
    }

    #[test]
    fn footprint_must_strictly_exceed_memory_limit() {
        assert_eq!(footprint_excess_bytes(101, 100).unwrap(), 1);
        assert!(footprint_excess_bytes(100, 100).is_err());
        assert!(footprint_excess_bytes(99, 100).is_err());
    }

    #[test]
    fn swap_and_unlimited_memory_contracts_fail_closed() {
        assert!(
            validate_cgroup_contract(&memory_state(CgroupLimit::Max, CgroupLimit::Bytes(0)))
                .is_err()
        );
        assert!(validate_cgroup_contract(&memory_state(
            CgroupLimit::Bytes(4096),
            CgroupLimit::Max
        ))
        .is_err());
        assert!(validate_cgroup_contract(&memory_state(
            CgroupLimit::Bytes(4096),
            CgroupLimit::Bytes(1)
        ))
        .is_err());
        let mut current = memory_state(CgroupLimit::Bytes(4096), CgroupLimit::Bytes(0));
        current.cgroup.memory_swap_current_bytes = 1;
        assert!(validate_cgroup_contract(&current).is_err());
    }

    #[test]
    fn cgroup_reader_uses_fixture_files_and_missing_data_fails_closed() {
        let fixture = TempDir::new("cgroup");
        let proc_cgroup = fixture.0.join("proc-self-cgroup");
        let proc_status = fixture.0.join("proc-self-status");
        let root = fixture.0.join("sys-fs-cgroup");
        let group = root.join("slice/mer");
        std::fs::create_dir_all(&group).unwrap();
        write(&proc_cgroup, "0::/slice/mer\n");
        write(&proc_status, "VmRSS:\t12 kB\nVmHWM:\t34 kB\n");
        write(group.join("memory.max"), "4096\n");
        write(group.join("memory.current"), "1024\n");
        write(group.join("memory.peak"), "2048\n");
        write(group.join("memory.swap.max"), "0\n");
        write(group.join("memory.swap.current"), "0\n");
        write(group.join("memory.events"), valid_events());
        let reader = CgroupReader::resolve(&proc_cgroup, &root, &proc_status).unwrap();
        assert_eq!(reader.relative_path, "/slice/mer");
        assert_eq!(reader.capture().unwrap().cgroup.memory_peak_bytes, 2048);
        std::fs::remove_file(group.join("memory.peak")).unwrap();
        assert!(reader.capture().is_err());
        write(group.join("memory.peak"), "not-a-number\n");
        assert!(reader.capture().is_err());
    }

    #[test]
    fn aggregate_cache_slots_are_distributed_without_multiplication() {
        let capacities = distributed_cache_capacities(384, 48).unwrap();
        assert_eq!(capacities, vec![8; 48]);
        assert_eq!(capacities.iter().sum::<usize>(), 384);
        assert_eq!(distributed_cache_capacities(10, 3).unwrap(), vec![4, 3, 3]);
        assert!(distributed_cache_capacities(2, 3).is_err());
    }

    fn first_hardware_artifact_h2d_evidence() -> GpuNativeH2dEvidence {
        GpuNativeH2dEvidence {
            evidence_source: GPU_NATIVE_H2D_EVIDENCE_SOURCE,
            ram_to_vram_installs: 90_717,
            direct_staging_writes: 90_717,
            direct_staging_successes: 90_717,
            physical_install_completions: 90_717,
            physical_slot_bytes_staged: 240_782_150_004,
            physical_bytes_staged: 240_782_150_004,
        }
    }

    #[test]
    fn gpu_native_h2d_gates_accept_direct_staging_when_legacy_gpu_io_is_zero() {
        let legacy_gpu_io = GpuExpertIoSnapshot::default();
        assert_eq!(legacy_gpu_io.expert_weight_uploads, 0);
        assert_eq!(legacy_gpu_io.expert_weight_upload_bytes, 0);

        let evidence = first_hardware_artifact_h2d_evidence();
        assert!(evidence.installs_positive_and_reconciled());
        assert!(evidence.bytes_positive_and_reconciled());
    }

    #[test]
    fn gpu_native_h2d_install_gate_rejects_zero_ram_to_vram_installs() {
        let mut evidence = first_hardware_artifact_h2d_evidence();
        evidence.ram_to_vram_installs = 0;
        assert!(!evidence.installs_positive_and_reconciled());
    }

    #[test]
    fn gpu_native_h2d_install_gate_rejects_zero_direct_staging_writes() {
        let mut evidence = first_hardware_artifact_h2d_evidence();
        evidence.direct_staging_writes = 0;
        assert!(!evidence.installs_positive_and_reconciled());
    }

    #[test]
    fn gpu_native_h2d_install_gate_rejects_zero_direct_staging_successes() {
        let mut evidence = first_hardware_artifact_h2d_evidence();
        evidence.direct_staging_successes = 0;
        assert!(!evidence.installs_positive_and_reconciled());
    }

    #[test]
    fn gpu_native_h2d_install_gate_rejects_disagreeing_install_counts() {
        let mut evidence = first_hardware_artifact_h2d_evidence();
        evidence.physical_install_completions -= 1;
        assert!(!evidence.installs_positive_and_reconciled());
    }

    #[test]
    fn gpu_native_h2d_byte_gate_rejects_zero_physical_slot_bytes() {
        let mut evidence = first_hardware_artifact_h2d_evidence();
        evidence.physical_slot_bytes_staged = 0;
        assert!(!evidence.bytes_positive_and_reconciled());
    }

    #[test]
    fn gpu_native_h2d_byte_gate_rejects_disagreeing_staging_bytes() {
        let mut evidence = first_hardware_artifact_h2d_evidence();
        evidence.physical_bytes_staged -= 1;
        assert!(!evidence.bytes_positive_and_reconciled());
    }

    fn passing_gate_inputs() -> GateInputs {
        GateInputs {
            frozen_configuration_exact: true,
            cgroup_v2_resolved: true,
            memory_max_numeric: true,
            expert_footprint_exact: true,
            footprint_exceeds_memory_max: true,
            swap_disabled_and_unused: true,
            no_oom_or_kill: true,
            cgroup_peak_within_limit: true,
            cache_geometry_exact: true,
            ram_pool_accounting_exact: true,
            resident_ram_within_pool: true,
            shadow_pool_zero: true,
            prepared_duplicates_zero: true,
            vram_budget_bounded: true,
            direct_io_enabled: true,
            ordinary_production_path_only: true,
            benchmark_and_runs_complete: true,
            generated_hashes_recorded: true,
            strict_checkpoint_runtime_validated: true,
            demand_source_requests_gt_zero: true,
            nvme_reads_gt_zero: true,
            demand_nvme_bytes_gt_zero: true,
            h2d_installs_gt_zero: true,
            h2d_bytes_gt_zero: true,
            no_degraded_substitution: true,
            no_full_token_replay: true,
            no_fatal_or_no_progress: true,
            no_cache_or_batch_invariant_failures: true,
            no_unexpected_gpu_status_bits: true,
            no_mapping_or_install_invariant_failures: true,
            no_speculative_work: true,
        }
    }

    #[test]
    fn final_qualification_gate_requires_every_formal_invariant() {
        assert!(QualificationGates::from_inputs(passing_gate_inputs()).all_invariants_pass);
        let setters: &[fn(&mut GateInputs)] = &[
            |g| g.frozen_configuration_exact = false,
            |g| g.cgroup_v2_resolved = false,
            |g| g.memory_max_numeric = false,
            |g| g.expert_footprint_exact = false,
            |g| g.footprint_exceeds_memory_max = false,
            |g| g.swap_disabled_and_unused = false,
            |g| g.no_oom_or_kill = false,
            |g| g.cgroup_peak_within_limit = false,
            |g| g.cache_geometry_exact = false,
            |g| g.ram_pool_accounting_exact = false,
            |g| g.resident_ram_within_pool = false,
            |g| g.shadow_pool_zero = false,
            |g| g.prepared_duplicates_zero = false,
            |g| g.vram_budget_bounded = false,
            |g| g.direct_io_enabled = false,
            |g| g.ordinary_production_path_only = false,
            |g| g.benchmark_and_runs_complete = false,
            |g| g.generated_hashes_recorded = false,
            |g| g.strict_checkpoint_runtime_validated = false,
            |g| g.demand_source_requests_gt_zero = false,
            |g| g.nvme_reads_gt_zero = false,
            |g| g.demand_nvme_bytes_gt_zero = false,
            |g| g.h2d_installs_gt_zero = false,
            |g| g.h2d_bytes_gt_zero = false,
            |g| g.no_degraded_substitution = false,
            |g| g.no_full_token_replay = false,
            |g| g.no_fatal_or_no_progress = false,
            |g| g.no_cache_or_batch_invariant_failures = false,
            |g| g.no_unexpected_gpu_status_bits = false,
            |g| g.no_mapping_or_install_invariant_failures = false,
            |g| g.no_speculative_work = false,
        ];
        for setter in setters {
            let mut input = passing_gate_inputs();
            setter(&mut input);
            assert!(!QualificationGates::from_inputs(input).all_invariants_pass);
        }
    }
}
