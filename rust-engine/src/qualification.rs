//! Strict Hybrid native-Q4_0 execution-integrity qualification.
//!
//! This module intentionally separates hardware-independent validation from
//! command orchestration. Internal memory evidence is limited to MER-owned
//! routed-expert buffers and workspaces; it is not total driver/device memory.

use crate::backend::{
    ComputeOffload, GpuDeviceIdentity, GpuExpertIoSnapshot, GpuExpertMemorySnapshot,
    ResolvedBackend, ResolvedExecutionPlan,
};
use crate::engine::{RoutedExpertExecutionSnapshot, RoutedExpertGpuFailurePolicy};
use crate::inference::{WeightDtype, Q4_0_LAYOUT_STANDARD_V1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: &str = "mer.strict-hybrid-q4.v1";
pub const MODE: &str = "strict-hybrid-q4";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QualificationStatus {
    Pass,
    Fail,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FailureStage {
    Preflight,
    Startup,
    Inference,
    Postcondition,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuFailureEvidence {
    pub layer: u32,
    pub expert_id: u32,
    pub kind: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct QualificationFailure {
    pub stage: FailureStage,
    pub code: String,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_dispatch: Option<GpuFailureEvidence>,
}

impl QualificationFailure {
    pub fn new(stage: FailureStage, code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            stage,
            code: code.to_string(),
            detail: detail.into(),
            gpu_dispatch: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuildProvenance {
    pub git_sha: Option<String>,
    pub dirty: Option<bool>,
    pub package_version: String,
}

impl BuildProvenance {
    pub fn embedded() -> Self {
        let sha = option_env!("MER_BUILD_GIT_SHA")
            .filter(|value| *value != "unavailable")
            .map(str::to_string);
        let dirty = match option_env!("MER_BUILD_GIT_DIRTY") {
            Some("true") => Some(true),
            Some("false") => Some(false),
            _ => None,
        };
        Self {
            git_sha: sha,
            dirty,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ArtifactDigest {
    pub configured_path: String,
    pub canonical_path: String,
    pub byte_length: u64,
    pub sha256: String,
}

pub fn hash_small_file(path: &Path) -> io::Result<ArtifactDigest> {
    let mut file = File::open(path)?;
    let byte_length = file.metadata()?.len();
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(ArtifactDigest {
        configured_path: path.display().to_string(),
        canonical_path: std::fs::canonicalize(path)?.display().to_string(),
        byte_length,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

pub fn hash_optional_small_file(path: Option<&Path>) -> io::Result<Option<ArtifactDigest>> {
    path.filter(|path| path.is_file())
        .map(hash_small_file)
        .transpose()
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct QualificationArtifacts {
    pub config: Option<ArtifactDigest>,
    pub tokenizer: Option<ArtifactDigest>,
    pub expert_metadata: Option<ArtifactDigest>,
    pub packed_manifest: Option<ArtifactDigest>,
    pub weights_config: Option<ArtifactDigest>,
    pub dense_weights_directory: Option<String>,
    pub expert_data_directory: String,
    pub packed_expert_blob: Option<String>,
    /// Metadata hashes identify metadata only, never the large expert blobs.
    pub large_artifacts_recursively_hashed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExpertMetadataEvidence {
    pub dtype: Option<String>,
    pub q4_0_layout: Option<String>,
    pub conversion_mode: Option<String>,
    pub source: Option<String>,
    pub explicitly_synthetic: bool,
}

#[derive(Deserialize)]
struct RawExpertMetadata {
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    q4_0_layout: Option<String>,
    #[serde(default)]
    conversion_mode: Option<String>,
    #[serde(default)]
    source: Option<String>,
}

pub fn read_expert_metadata(path: &Path) -> Result<ExpertMetadataEvidence, String> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read expert metadata {}: {e}", path.display()))?;
    let raw: RawExpertMetadata = serde_json::from_str(&body)
        .map_err(|e| format!("expert metadata {} is invalid JSON: {e}", path.display()))?;
    let explicitly_synthetic = raw
        .conversion_mode
        .as_deref()
        .is_some_and(|value| value.to_ascii_lowercase().contains("synthetic"))
        || raw
            .source
            .as_deref()
            .is_some_and(|value| value.to_ascii_lowercase().contains("synthetic"));
    Ok(ExpertMetadataEvidence {
        dtype: raw.dtype,
        q4_0_layout: raw.q4_0_layout,
        conversion_mode: raw.conversion_mode,
        source: raw.source,
        explicitly_synthetic,
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutionPlanEvidence {
    pub context_id: String,
    pub requested: String,
    pub resolved: String,
    pub embeddings: String,
    pub lm_head: String,
    pub dense_projections: String,
    pub attention: String,
    pub kv: String,
    pub router: String,
    pub routed_experts: String,
    pub routed_expert_dtype: String,
    pub fallback_occurred: bool,
    pub reason: Option<String>,
}

fn requested_name(mode: ComputeOffload) -> &'static str {
    match mode {
        ComputeOffload::Cpu => "cpu",
        ComputeOffload::Gpu => "gpu",
        ComputeOffload::Auto => "auto",
        ComputeOffload::Hybrid => "hybrid",
    }
}

fn resolved_name(mode: ResolvedBackend) -> &'static str {
    match mode {
        ResolvedBackend::Cpu => "cpu",
        ResolvedBackend::Gpu => "gpu",
        ResolvedBackend::HybridCpuAttentionGpuExperts => "hybrid-cpu-attention-gpu-experts",
    }
}

impl From<&ResolvedExecutionPlan> for ExecutionPlanEvidence {
    fn from(plan: &ResolvedExecutionPlan) -> Self {
        Self {
            context_id: plan.context_id().to_string(),
            requested: requested_name(plan.requested()).to_string(),
            resolved: resolved_name(plan.resolved()).to_string(),
            embeddings: plan.embeddings().as_str().to_string(),
            lm_head: plan.lm_head().as_str().to_string(),
            dense_projections: plan.dense_projections().as_str().to_string(),
            attention: plan.attention().as_str().to_string(),
            kv: plan.kv().as_str().to_string(),
            router: plan.router().as_str().to_string(),
            routed_experts: plan.routed_experts().as_str().to_string(),
            routed_expert_dtype: plan.routed_expert_gpu_spec().dtype.as_str().to_string(),
            fallback_occurred: plan.fallback_occurred(),
            reason: plan.reason().map(str::to_string),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RequestEvidence {
    pub input_kind: String,
    pub prompt_sha256: String,
    pub prompt_bytes: usize,
    pub requested_output_tokens: usize,
    pub warmup_runs: usize,
    pub greedy: bool,
}

pub fn request_evidence(
    input_kind: &str,
    prompt: &str,
    requested_output_tokens: usize,
    warmup_runs: usize,
) -> RequestEvidence {
    RequestEvidence {
        input_kind: input_kind.to_string(),
        prompt_sha256: format!("{:x}", Sha256::digest(prompt.as_bytes())),
        prompt_bytes: prompt.len(),
        requested_output_tokens,
        warmup_runs,
        greedy: true,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QualificationTiming {
    pub prompt_tokens: usize,
    pub requested_output_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_seconds: f64,
    pub decode_seconds: f64,
    pub total_seconds: f64,
    /// Actual generated completion tokens after the first token produced at TTFT.
    pub decode_generated_tokens: usize,
    /// Steady autoregressive decode throughput, matching `bench-real` semantics.
    pub decode_tps: f64,
    /// Alias for `decode_tps`, retained as the qualification throughput field.
    pub real_generated_tps: f64,
    /// Actual generated completion tokens divided by total request wall time.
    pub end_to_end_generated_tps: f64,
}

impl QualificationTiming {
    pub fn from_measurement(
        prompt_tokens: usize,
        requested_output_tokens: usize,
        generated_tokens: usize,
        prompt_seconds: f64,
        decode_seconds: f64,
        total_seconds: f64,
    ) -> Self {
        let decode_generated_tokens = generated_tokens.saturating_sub(1);
        let decode_tps = crate::rate_per_second(decode_generated_tokens, decode_seconds);
        Self {
            prompt_tokens,
            requested_output_tokens,
            generated_tokens,
            prompt_seconds,
            decode_seconds,
            total_seconds,
            decode_generated_tokens,
            decode_tps,
            real_generated_tps: decode_tps,
            end_to_end_generated_tps: crate::rate_per_second(generated_tokens, total_seconds),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct QualificationChecks {
    pub clean_build: bool,
    pub strict_real_checkpoint: bool,
    pub requested_hybrid: bool,
    pub resolved_planes_match_contract: bool,
    pub native_q4_0_routed_experts: bool,
    pub canonical_q4_layout: bool,
    pub hardware_gpu_adapter: bool,
    pub strict_gpu_failure_policy: bool,
    pub generated_tokens: bool,
    pub observed_routed_experts: bool,
    pub all_selected_experts_attempted_on_gpu: bool,
    pub all_gpu_attempts_succeeded: bool,
    pub zero_cpu_routed_expert_execution: bool,
    pub zero_gpu_cpu_fallbacks: bool,
    pub zero_degraded_expert_substitutions: bool,
}

impl QualificationChecks {
    pub fn passes(&self) -> bool {
        self.clean_build
            && self.strict_real_checkpoint
            && self.requested_hybrid
            && self.resolved_planes_match_contract
            && self.native_q4_0_routed_experts
            && self.canonical_q4_layout
            && self.hardware_gpu_adapter
            && self.strict_gpu_failure_policy
            && self.generated_tokens
            && self.observed_routed_experts
            && self.all_selected_experts_attempted_on_gpu
            && self.all_gpu_attempts_succeeded
            && self.zero_cpu_routed_expert_execution
            && self.zero_gpu_cpu_fallbacks
            && self.zero_degraded_expert_substitutions
    }
}

#[derive(Clone, Debug)]
pub struct PreflightEvidence {
    pub provenance: BuildProvenance,
    pub real_transformer_enabled: bool,
    pub weights_dir_configured: bool,
    pub strict_weights: bool,
    pub allow_seeded_fallback: bool,
    pub allow_degraded_experts: bool,
    pub allow_attention_fallback: bool,
    pub allow_truncated_expert_payloads: bool,
    pub distributed_enabled: bool,
    pub gpu_cache_enabled: bool,
    pub gpu_expert_capacity_bytes: u64,
    pub requested_mode: ComputeOffload,
    pub routed_expert_dtype: WeightDtype,
    pub metadata: ExpertMetadataEvidence,
}

pub fn validate_preflight(
    evidence: &PreflightEvidence,
    checks: &mut QualificationChecks,
) -> Result<(), QualificationFailure> {
    let sha_available = evidence
        .provenance
        .git_sha
        .as_deref()
        .is_some_and(|sha| sha.len() == 40 && sha.bytes().all(|byte| byte.is_ascii_hexdigit()));
    if !sha_available || evidence.provenance.dirty.is_none() {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "build-provenance-unavailable",
            "build-time full Git SHA or tracked-source dirty state is unavailable",
        ));
    }
    if evidence.provenance.dirty == Some(true) {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "dirty-build",
            "strict qualification requires a clean tracked-source build",
        ));
    }
    checks.clean_build = true;

    if !evidence.real_transformer_enabled {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "not-real-transformer",
            "real_transformer.enabled must be true",
        ));
    }
    if !evidence.weights_dir_configured || !evidence.strict_weights {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "non-strict-weight-policy",
            "a configured weights_dir and strict_weights=true are required",
        ));
    }
    if evidence.allow_seeded_fallback {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "seeded-fallback-enabled",
            "allow_seeded_fallback must be false",
        ));
    }
    if evidence.allow_degraded_experts {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "degraded-experts-enabled",
            "allow_degraded_experts must be false",
        ));
    }
    if evidence.allow_attention_fallback {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "attention-fallback-enabled",
            "allow_nonfinite_attention_fallback must be false",
        ));
    }
    if evidence.allow_truncated_expert_payloads {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "truncated-expert-payloads-enabled",
            "allow_truncated_expert_payloads must be false",
        ));
    }
    if evidence.distributed_enabled {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "distributed-execution-enabled",
            "strict qualification requires the local authoritative expert backend",
        ));
    }
    checks.strict_real_checkpoint = true;

    if evidence.requested_mode != ComputeOffload::Hybrid {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "not-hybrid",
            format!(
                "requested compute mode is {}, expected hybrid",
                requested_name(evidence.requested_mode)
            ),
        ));
    }
    checks.requested_hybrid = true;
    if !evidence.gpu_cache_enabled {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "gpu-cache-disabled",
            "gpu_cache.enabled must be true",
        ));
    }
    if evidence.gpu_expert_capacity_bytes == 0 {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "zero-gpu-expert-capacity",
            "GPU expert capacity must be non-zero",
        ));
    }
    if evidence.routed_expert_dtype != WeightDtype::Q4_0 {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "routed-expert-not-q4-0",
            format!(
                "actual routed-expert runtime dtype is {}, expected q4_0",
                evidence.routed_expert_dtype.as_str()
            ),
        ));
    }
    checks.native_q4_0_routed_experts = true;
    if evidence.metadata.explicitly_synthetic {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "synthetic-artifact",
            "expert metadata explicitly identifies a synthetic dataset",
        ));
    }
    if evidence.metadata.q4_0_layout.as_deref() != Some(Q4_0_LAYOUT_STANDARD_V1) {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "q4-layout-not-standard",
            format!(
                "q4_0_layout is {:?}, expected {:?}",
                evidence.metadata.q4_0_layout, Q4_0_LAYOUT_STANDARD_V1
            ),
        ));
    }
    checks.canonical_q4_layout = true;
    Ok(())
}

pub fn validate_execution_plan(
    plan: &ResolvedExecutionPlan,
    checks: &mut QualificationChecks,
) -> Result<ExecutionPlanEvidence, QualificationFailure> {
    let evidence: ExecutionPlanEvidence = plan.into();
    validate_execution_plan_evidence(&evidence, checks)?;
    Ok(evidence)
}

/// Hardware-independent form of the exact authoritative-plan check. Runtime
/// code always constructs this evidence from `ExecutionContext::plan()`.
pub fn validate_execution_plan_evidence(
    plan: &ExecutionPlanEvidence,
    checks: &mut QualificationChecks,
) -> Result<(), QualificationFailure> {
    if plan.requested != "hybrid" {
        return Err(QualificationFailure::new(
            FailureStage::Startup,
            "not-hybrid",
            "authoritative execution context was not requested as Hybrid",
        ));
    }
    checks.requested_hybrid = true;
    let exact = plan.resolved == "hybrid-cpu-attention-gpu-experts"
        && plan.embeddings == "cpu"
        && plan.lm_head == "cpu"
        && plan.dense_projections == "cpu"
        && plan.attention == "cpu"
        && plan.kv == "cpu"
        && plan.router == "cpu"
        && plan.routed_experts == "gpu"
        && !plan.fallback_occurred;
    if !exact {
        return Err(QualificationFailure::new(
            FailureStage::Startup,
            "invalid-execution-plan",
            format!("resolved component planes do not match strict Hybrid contract: {plan:?}"),
        ));
    }
    checks.resolved_planes_match_contract = true;
    if plan.routed_expert_dtype != "q4_0" {
        return Err(QualificationFailure::new(
            FailureStage::Startup,
            "routed-expert-not-q4-0",
            "authoritative routed-expert GPU spec is not q4_0",
        ));
    }
    checks.native_q4_0_routed_experts = true;
    Ok(())
}

pub fn validate_device(
    device: Option<&GpuDeviceIdentity>,
    checks: &mut QualificationChecks,
) -> Result<(), QualificationFailure> {
    let Some(device) = device else {
        return Err(QualificationFailure::new(
            FailureStage::Startup,
            "gpu-device-unavailable",
            "authoritative execution context has no production GPU identity",
        ));
    };
    if device.software_adapter || device.device_type.eq_ignore_ascii_case("cpu") {
        return Err(QualificationFailure::new(
            FailureStage::Startup,
            "software-adapter",
            format!(
                "selected adapter {:?} is a software/CPU adapter",
                device.name
            ),
        ));
    }
    checks.hardware_gpu_adapter = true;
    Ok(())
}

pub fn validate_gpu_failure_policy(
    policy: RoutedExpertGpuFailurePolicy,
    checks: &mut QualificationChecks,
) -> Result<(), QualificationFailure> {
    if policy != RoutedExpertGpuFailurePolicy::StrictFailClosed {
        return Err(QualificationFailure::new(
            FailureStage::Startup,
            "gpu-failure-policy-not-strict",
            "qualification engine must use StrictFailClosed",
        ));
    }
    checks.strict_gpu_failure_policy = true;
    Ok(())
}

fn delta_field(after: u64, before: u64, name: &str) -> Result<u64, QualificationFailure> {
    after.checked_sub(before).ok_or_else(|| {
        QualificationFailure::new(
            FailureStage::Postcondition,
            "counter-invariant-failed",
            format!("monotonic counter {name} decreased from {before} to {after}"),
        )
    })
}

pub fn routed_execution_delta(
    before: RoutedExpertExecutionSnapshot,
    after: RoutedExpertExecutionSnapshot,
) -> Result<RoutedExpertExecutionSnapshot, QualificationFailure> {
    Ok(RoutedExpertExecutionSnapshot {
        selected_routed_experts: delta_field(
            after.selected_routed_experts,
            before.selected_routed_experts,
            "selected_routed_experts",
        )?,
        gpu_dispatch_attempts: delta_field(
            after.gpu_dispatch_attempts,
            before.gpu_dispatch_attempts,
            "gpu_dispatch_attempts",
        )?,
        gpu_dispatch_successes: delta_field(
            after.gpu_dispatch_successes,
            before.gpu_dispatch_successes,
            "gpu_dispatch_successes",
        )?,
        gpu_dispatch_failures: delta_field(
            after.gpu_dispatch_failures,
            before.gpu_dispatch_failures,
            "gpu_dispatch_failures",
        )?,
        cpu_routed_expert_dispatches: delta_field(
            after.cpu_routed_expert_dispatches,
            before.cpu_routed_expert_dispatches,
            "cpu_routed_expert_dispatches",
        )?,
        gpu_cpu_fallbacks: delta_field(
            after.gpu_cpu_fallbacks,
            before.gpu_cpu_fallbacks,
            "gpu_cpu_fallbacks",
        )?,
        degraded_expert_substitutions: delta_field(
            after.degraded_expert_substitutions,
            before.degraded_expert_substitutions,
            "degraded_expert_substitutions",
        )?,
    })
}

pub fn gpu_io_delta(
    before: GpuExpertIoSnapshot,
    after: GpuExpertIoSnapshot,
) -> Result<GpuExpertIoSnapshot, QualificationFailure> {
    Ok(GpuExpertIoSnapshot {
        expert_weight_uploads: delta_field(
            after.expert_weight_uploads,
            before.expert_weight_uploads,
            "expert_weight_uploads",
        )?,
        expert_weight_upload_bytes: delta_field(
            after.expert_weight_upload_bytes,
            before.expert_weight_upload_bytes,
            "expert_weight_upload_bytes",
        )?,
        hidden_state_uploads: delta_field(
            after.hidden_state_uploads,
            before.hidden_state_uploads,
            "hidden_state_uploads",
        )?,
        hidden_state_upload_bytes: delta_field(
            after.hidden_state_upload_bytes,
            before.hidden_state_upload_bytes,
            "hidden_state_upload_bytes",
        )?,
        queue_submissions: delta_field(
            after.queue_submissions,
            before.queue_submissions,
            "queue_submissions",
        )?,
        map_requests: delta_field(after.map_requests, before.map_requests, "map_requests")?,
        readback_completions: delta_field(
            after.readback_completions,
            before.readback_completions,
            "readback_completions",
        )?,
        readback_bytes: delta_field(
            after.readback_bytes,
            before.readback_bytes,
            "readback_bytes",
        )?,
    })
}

pub fn validate_memory(snapshot: GpuExpertMemorySnapshot) -> Result<(), QualificationFailure> {
    if snapshot
        .expert_live_bytes
        .checked_add(snapshot.workspace_bytes)
        != Some(snapshot.total_tracked_bytes)
        || snapshot.expert_registry_bytes > snapshot.expert_live_bytes
        || snapshot.expert_registry_bytes > snapshot.expert_capacity_bytes
    {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "gpu-memory-invariant-failed",
            format!("invalid PR4 physical memory snapshot: {snapshot:?}"),
        ));
    }
    Ok(())
}

pub fn validate_postconditions(
    generated_tokens: usize,
    routed: RoutedExpertExecutionSnapshot,
    checks: &mut QualificationChecks,
) -> Result<(), QualificationFailure> {
    if generated_tokens == 0 {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "no-generated-tokens",
            "measured request generated no completion tokens",
        ));
    }
    checks.generated_tokens = true;
    if routed.selected_routed_experts == 0 {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "no-routed-experts-observed",
            "measured request selected no routed experts",
        ));
    }
    checks.observed_routed_experts = true;
    if routed.cpu_routed_expert_dispatches != 0 {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "routed-cpu-execution-observed",
            format!(
                "{} selected routed experts executed on CPU",
                routed.cpu_routed_expert_dispatches
            ),
        ));
    }
    checks.zero_cpu_routed_expert_execution = true;
    if routed.gpu_cpu_fallbacks != 0 {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "gpu-cpu-fallback-observed",
            format!("{} GPU-to-CPU fallbacks observed", routed.gpu_cpu_fallbacks),
        ));
    }
    checks.zero_gpu_cpu_fallbacks = true;
    if routed.degraded_expert_substitutions != 0 {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "degraded-substitution-observed",
            format!(
                "{} degraded expert substitutions observed",
                routed.degraded_expert_substitutions
            ),
        ));
    }
    checks.zero_degraded_expert_substitutions = true;
    if routed.gpu_dispatch_failures != 0 {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "routed-gpu-dispatch-failed",
            format!(
                "{} routed GPU dispatch failures observed",
                routed.gpu_dispatch_failures
            ),
        ));
    }
    if routed.gpu_dispatch_attempts != routed.selected_routed_experts {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "counter-invariant-failed",
            format!(
                "GPU attempts {} != selected routed experts {}",
                routed.gpu_dispatch_attempts, routed.selected_routed_experts
            ),
        ));
    }
    checks.all_selected_experts_attempted_on_gpu = true;
    if routed.gpu_dispatch_successes != routed.gpu_dispatch_attempts {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "counter-invariant-failed",
            format!(
                "GPU successes {} != attempts {}",
                routed.gpu_dispatch_successes, routed.gpu_dispatch_attempts
            ),
        ));
    }
    checks.all_gpu_attempts_succeeded = true;
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
pub struct QualificationReport {
    pub schema_version: &'static str,
    pub mode: &'static str,
    pub status: QualificationStatus,
    pub failure: Option<QualificationFailure>,
    pub provenance: BuildProvenance,
    pub artifacts: QualificationArtifacts,
    pub expert_metadata: Option<ExpertMetadataEvidence>,
    pub execution_plan: Option<ExecutionPlanEvidence>,
    pub device: Option<GpuDeviceIdentity>,
    pub request: RequestEvidence,
    pub timing: Option<QualificationTiming>,
    pub routed_experts: Option<RoutedExpertExecutionSnapshot>,
    pub gpu_io: Option<GpuExpertIoSnapshot>,
    pub gpu_memory_before: Option<GpuExpertMemorySnapshot>,
    pub gpu_memory_after: Option<GpuExpertMemorySnapshot>,
    pub external_gpu_memory_artifact: Option<String>,
    pub qualification_checks: QualificationChecks,
}

impl QualificationReport {
    pub fn new(
        provenance: BuildProvenance,
        artifacts: QualificationArtifacts,
        expert_metadata: Option<ExpertMetadataEvidence>,
        request: RequestEvidence,
        external_gpu_memory_artifact: Option<String>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            mode: MODE,
            status: QualificationStatus::Fail,
            failure: None,
            provenance,
            artifacts,
            expert_metadata,
            execution_plan: None,
            device: None,
            request,
            timing: None,
            routed_experts: None,
            gpu_io: None,
            gpu_memory_before: None,
            gpu_memory_after: None,
            external_gpu_memory_artifact,
            qualification_checks: QualificationChecks::default(),
        }
    }

    pub fn fail(&mut self, failure: QualificationFailure) {
        self.status = QualificationStatus::Fail;
        self.failure = Some(failure);
    }

    pub fn finish(&mut self) -> Result<(), QualificationFailure> {
        if !self.qualification_checks.passes() {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "qualification-check-failed",
                "one or more required qualification checks are false",
            ));
        }
        if self.artifacts.config.is_none()
            || self.artifacts.tokenizer.is_none()
            || self.artifacts.expert_metadata.is_none()
            || self.expert_metadata.is_none()
            || self.execution_plan.is_none()
            || self.device.is_none()
            || self.timing.is_none()
            || self.routed_experts.is_none()
            || self.gpu_io.is_none()
            || self.gpu_memory_before.is_none()
            || self.gpu_memory_after.is_none()
        {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "qualification-evidence-incomplete",
                "one or more mandatory qualification evidence sections are absent",
            ));
        }
        self.status = QualificationStatus::Pass;
        self.failure = None;
        Ok(())
    }
}

pub fn canonical_or_configured(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| PathBuf::from(path))
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_provenance() -> BuildProvenance {
        BuildProvenance {
            git_sha: Some("0123456789abcdef0123456789abcdef01234567".to_string()),
            dirty: Some(false),
            package_version: "test".to_string(),
        }
    }

    fn metadata() -> ExpertMetadataEvidence {
        ExpertMetadataEvidence {
            dtype: Some("q4_0".to_string()),
            q4_0_layout: Some(Q4_0_LAYOUT_STANDARD_V1.to_string()),
            conversion_mode: Some("full".to_string()),
            source: None,
            explicitly_synthetic: false,
        }
    }

    fn preflight() -> PreflightEvidence {
        PreflightEvidence {
            provenance: clean_provenance(),
            real_transformer_enabled: true,
            weights_dir_configured: true,
            strict_weights: true,
            allow_seeded_fallback: false,
            allow_degraded_experts: false,
            allow_attention_fallback: false,
            allow_truncated_expert_payloads: false,
            distributed_enabled: false,
            gpu_cache_enabled: true,
            gpu_expert_capacity_bytes: 1,
            requested_mode: ComputeOffload::Hybrid,
            routed_expert_dtype: WeightDtype::Q4_0,
            metadata: metadata(),
        }
    }

    fn routed_success() -> RoutedExpertExecutionSnapshot {
        RoutedExpertExecutionSnapshot {
            selected_routed_experts: 8,
            gpu_dispatch_attempts: 8,
            gpu_dispatch_successes: 8,
            ..Default::default()
        }
    }

    fn exact_plan() -> ExecutionPlanEvidence {
        ExecutionPlanEvidence {
            context_id: "17".to_string(),
            requested: "hybrid".to_string(),
            resolved: "hybrid-cpu-attention-gpu-experts".to_string(),
            embeddings: "cpu".to_string(),
            lm_head: "cpu".to_string(),
            dense_projections: "cpu".to_string(),
            attention: "cpu".to_string(),
            kv: "cpu".to_string(),
            router: "cpu".to_string(),
            routed_experts: "gpu".to_string(),
            routed_expert_dtype: "q4_0".to_string(),
            fallback_occurred: false,
            reason: None,
        }
    }

    #[test]
    fn strict_preflight_accepts_exact_contract() {
        let mut checks = QualificationChecks::default();
        validate_preflight(&preflight(), &mut checks).unwrap();
        assert!(checks.clean_build);
        assert!(checks.strict_real_checkpoint);
        assert!(checks.requested_hybrid);
        assert!(checks.native_q4_0_routed_experts);
        assert!(checks.canonical_q4_layout);
    }

    #[test]
    fn exact_hybrid_plan_passes_and_preserves_context_identity() {
        let plan = exact_plan();
        let mut checks = QualificationChecks::default();
        validate_execution_plan_evidence(&plan, &mut checks).unwrap();
        assert_eq!(plan.context_id, "17");
        assert!(checks.requested_hybrid);
        assert!(checks.resolved_planes_match_contract);
        assert!(checks.native_q4_0_routed_experts);
    }

    #[test]
    fn wrong_expert_attention_and_dense_planes_fail() {
        for field in ["experts", "attention", "dense"] {
            let mut plan = exact_plan();
            match field {
                "experts" => plan.routed_experts = "cpu".to_string(),
                "attention" => plan.attention = "gpu".to_string(),
                "dense" => plan.dense_projections = "gpu".to_string(),
                _ => unreachable!(),
            }
            assert_eq!(
                validate_execution_plan_evidence(&plan, &mut QualificationChecks::default())
                    .unwrap_err()
                    .code,
                "invalid-execution-plan"
            );
        }
    }

    #[test]
    fn cpu_plan_remains_valid_evidence_but_cannot_qualify() {
        let mut plan = exact_plan();
        plan.requested = "cpu".to_string();
        plan.resolved = "cpu".to_string();
        plan.routed_experts = "cpu".to_string();
        assert_eq!(
            validate_execution_plan_evidence(&plan, &mut QualificationChecks::default())
                .unwrap_err()
                .code,
            "not-hybrid"
        );
    }

    #[test]
    fn requested_mode_rejects_cpu_auto_and_gpu() {
        for mode in [
            ComputeOffload::Cpu,
            ComputeOffload::Auto,
            ComputeOffload::Gpu,
        ] {
            let mut evidence = preflight();
            evidence.requested_mode = mode;
            assert_eq!(
                validate_preflight(&evidence, &mut QualificationChecks::default())
                    .unwrap_err()
                    .code,
                "not-hybrid"
            );
        }
    }

    #[test]
    fn each_fail_open_checkpoint_policy_is_rejected() {
        let cases = [
            ("seeded-fallback-enabled", 0),
            ("non-strict-weight-policy", 1),
            ("degraded-experts-enabled", 2),
            ("attention-fallback-enabled", 3),
            ("truncated-expert-payloads-enabled", 4),
        ];
        for (code, which) in cases {
            let mut evidence = preflight();
            match which {
                0 => evidence.allow_seeded_fallback = true,
                1 => evidence.strict_weights = false,
                2 => evidence.allow_degraded_experts = true,
                3 => evidence.allow_attention_fallback = true,
                4 => evidence.allow_truncated_expert_payloads = true,
                _ => unreachable!(),
            }
            assert_eq!(
                validate_preflight(&evidence, &mut QualificationChecks::default())
                    .unwrap_err()
                    .code,
                code
            );
        }
    }

    #[test]
    fn actual_expert_dtype_allows_only_q4_0() {
        for dtype in [WeightDtype::F32, WeightDtype::Q8_0, WeightDtype::Q4K] {
            let mut evidence = preflight();
            evidence.routed_expert_dtype = dtype;
            assert_eq!(
                validate_preflight(&evidence, &mut QualificationChecks::default())
                    .unwrap_err()
                    .code,
                "routed-expert-not-q4-0"
            );
        }
    }

    #[test]
    fn q4_layout_requires_canonical_marker() {
        for marker in [None, Some("legacy-adjacent-nibbles".to_string())] {
            let mut evidence = preflight();
            evidence.metadata.q4_0_layout = marker;
            assert_eq!(
                validate_preflight(&evidence, &mut QualificationChecks::default())
                    .unwrap_err()
                    .code,
                "q4-layout-not-standard"
            );
        }
    }

    #[test]
    fn explicitly_synthetic_metadata_is_rejected() {
        let mut evidence = preflight();
        evidence.metadata.explicitly_synthetic = true;
        assert_eq!(
            validate_preflight(&evidence, &mut QualificationChecks::default())
                .unwrap_err()
                .code,
            "synthetic-artifact"
        );
    }

    #[test]
    fn metadata_parser_marks_explicit_synthetic_sources() {
        let path = std::env::temp_dir().join(format!(
            "mer-pr5-synthetic-metadata-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(
            &path,
            r#"{"dtype":"q4_0","q4_0_layout":"ggml-standard-v1","source":"synthetic generator"}"#,
        )
        .unwrap();
        assert!(read_expert_metadata(&path).unwrap().explicitly_synthetic);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn device_identity_rejects_absent_software_and_cpu() {
        let mut checks = QualificationChecks::default();
        assert_eq!(
            validate_device(None, &mut checks).unwrap_err().code,
            "gpu-device-unavailable"
        );
        for (name, device_type, software) in [
            ("llvmpipe", "Cpu", true),
            ("SwiftShader", "IntegratedGpu", true),
            ("cpu", "Cpu", false),
        ] {
            let identity = GpuDeviceIdentity {
                name: name.to_string(),
                vendor_id: 1,
                device_id: 2,
                device_type: device_type.to_string(),
                wgpu_backend: "vulkan".to_string(),
                driver: "test".to_string(),
                driver_info: String::new(),
                compute_plane: "wgpu-vulkan".to_string(),
                software_adapter: software,
            };
            assert_eq!(
                validate_device(Some(&identity), &mut QualificationChecks::default())
                    .unwrap_err()
                    .code,
                "software-adapter"
            );
        }
    }

    #[test]
    fn hardware_device_and_strict_failure_policy_pass() {
        let mut checks = QualificationChecks::default();
        for device_type in ["DiscreteGpu", "IntegratedGpu"] {
            let identity = GpuDeviceIdentity {
                name: "hardware".to_string(),
                vendor_id: 1,
                device_id: 2,
                device_type: device_type.to_string(),
                wgpu_backend: "vulkan".to_string(),
                driver: "driver".to_string(),
                driver_info: "info".to_string(),
                compute_plane: "wgpu-vulkan".to_string(),
                software_adapter: false,
            };
            validate_device(Some(&identity), &mut checks).unwrap();
        }
        validate_gpu_failure_policy(RoutedExpertGpuFailurePolicy::StrictFailClosed, &mut checks)
            .unwrap();
        assert!(checks.hardware_gpu_adapter && checks.strict_gpu_failure_policy);
        assert_eq!(
            validate_gpu_failure_policy(
                RoutedExpertGpuFailurePolicy::ServingCpuFallback,
                &mut QualificationChecks::default()
            )
            .unwrap_err()
            .code,
            "gpu-failure-policy-not-strict"
        );
    }

    #[test]
    fn routed_counter_success_contract_passes() {
        let mut checks = QualificationChecks::default();
        validate_postconditions(1, routed_success(), &mut checks).unwrap();
        assert!(checks.observed_routed_experts);
        assert!(checks.all_selected_experts_attempted_on_gpu);
        assert!(checks.all_gpu_attempts_succeeded);
    }

    #[test]
    fn routed_counter_failures_are_typed() {
        let mut cases = Vec::new();
        let mut cpu = routed_success();
        cpu.cpu_routed_expert_dispatches = 1;
        cases.push((cpu, "routed-cpu-execution-observed"));
        let mut fallback = routed_success();
        fallback.gpu_cpu_fallbacks = 1;
        cases.push((fallback, "gpu-cpu-fallback-observed"));
        let mut failed = routed_success();
        failed.gpu_dispatch_failures = 1;
        cases.push((failed, "routed-gpu-dispatch-failed"));
        let mut mismatch = routed_success();
        mismatch.gpu_dispatch_attempts = 7;
        cases.push((mismatch, "counter-invariant-failed"));
        let mut no_routes = routed_success();
        no_routes.selected_routed_experts = 0;
        no_routes.gpu_dispatch_attempts = 0;
        no_routes.gpu_dispatch_successes = 0;
        cases.push((no_routes, "no-routed-experts-observed"));
        let mut degraded = routed_success();
        degraded.degraded_expert_substitutions = 1;
        cases.push((degraded, "degraded-substitution-observed"));
        for (snapshot, code) in cases {
            assert_eq!(
                validate_postconditions(1, snapshot, &mut QualificationChecks::default())
                    .unwrap_err()
                    .code,
                code
            );
        }
        assert_eq!(
            validate_postconditions(0, routed_success(), &mut QualificationChecks::default())
                .unwrap_err()
                .code,
            "no-generated-tokens"
        );
    }

    #[test]
    fn monotonic_counter_deltas_are_checked() {
        let before = GpuExpertIoSnapshot {
            expert_weight_uploads: 2,
            expert_weight_upload_bytes: 20,
            hidden_state_uploads: 3,
            hidden_state_upload_bytes: 30,
            queue_submissions: 3,
            map_requests: 3,
            readback_completions: 3,
            readback_bytes: 30,
        };
        let after = GpuExpertIoSnapshot {
            expert_weight_uploads: 3,
            expert_weight_upload_bytes: 24,
            hidden_state_uploads: 5,
            hidden_state_upload_bytes: 38,
            queue_submissions: 5,
            map_requests: 5,
            readback_completions: 5,
            readback_bytes: 38,
        };
        let delta = gpu_io_delta(before, after).unwrap();
        assert_eq!(
            delta,
            GpuExpertIoSnapshot {
                expert_weight_uploads: 1,
                expert_weight_upload_bytes: 4,
                hidden_state_uploads: 2,
                hidden_state_upload_bytes: 8,
                queue_submissions: 2,
                map_requests: 2,
                readback_completions: 2,
                readback_bytes: 8,
            }
        );
        assert!(gpu_io_delta(after, before).is_err());
        assert_eq!(
            routed_execution_delta(RoutedExpertExecutionSnapshot::default(), routed_success())
                .unwrap(),
            routed_success()
        );
        assert!(
            routed_execution_delta(routed_success(), RoutedExpertExecutionSnapshot::default())
                .is_err()
        );
    }

    #[test]
    fn pr4_memory_snapshot_invariants_and_serialization() {
        let snapshot = GpuExpertMemorySnapshot {
            logical_admitted_bytes: 7,
            expert_live_bytes: 10,
            expert_registry_bytes: 8,
            workspace_bytes: 5,
            total_tracked_bytes: 15,
            expert_capacity_bytes: 20,
            physical_entries: 1,
            physical_installs: 2,
            physical_evictions: 3,
            stale_retirements: 4,
        };
        validate_memory(snapshot).unwrap();
        let value = serde_json::to_value(snapshot).unwrap();
        for (field, expected) in [
            ("logical_admitted_bytes", 7),
            ("expert_live_bytes", 10),
            ("expert_registry_bytes", 8),
            ("workspace_bytes", 5),
            ("total_tracked_bytes", 15),
            ("expert_capacity_bytes", 20),
            ("physical_entries", 1),
            ("physical_installs", 2),
            ("physical_evictions", 3),
            ("stale_retirements", 4),
        ] {
            assert_eq!(value[field], expected, "{field}");
        }
        let mut bad = snapshot;
        bad.total_tracked_bytes += 1;
        assert!(validate_memory(bad).is_err());
        bad = snapshot;
        bad.expert_registry_bytes = 11;
        assert!(validate_memory(bad).is_err());
        bad = snapshot;
        bad.expert_capacity_bytes = 7;
        assert!(validate_memory(bad).is_err());
    }

    #[test]
    fn small_artifact_sha256_and_optional_absence_are_exact() {
        let path = std::env::temp_dir().join(format!(
            "mer-pr5-sha-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, b"abc").unwrap();
        let digest = hash_small_file(&path).unwrap();
        assert_eq!(digest.byte_length, 3);
        assert_eq!(
            digest.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_file(&path).unwrap();
        assert_eq!(hash_optional_small_file(Some(&path)).unwrap(), None);
    }

    #[test]
    fn qualification_timing_matches_bench_real_decode_semantics() {
        let timing = QualificationTiming::from_measurement(100, 100, 10, 3.0, 2.0, 5.0);
        assert_eq!(timing.generated_tokens, 10);
        assert_eq!(timing.decode_generated_tokens, 9);
        assert_eq!(timing.decode_tps, 4.5);
        assert_eq!(timing.real_generated_tps, 4.5);
        assert_eq!(timing.end_to_end_generated_tps, 2.0);
    }

    #[test]
    fn qualification_decode_rate_ignores_prompt_and_total_time() {
        let short_prompt = QualificationTiming::from_measurement(10, 10, 10, 1.0, 2.0, 3.0);
        let long_prompt = QualificationTiming::from_measurement(10, 10, 10, 9.0, 2.0, 11.0);
        assert_eq!(short_prompt.decode_tps, long_prompt.decode_tps);
        assert_eq!(
            short_prompt.real_generated_tps,
            long_prompt.real_generated_tps
        );
        assert_ne!(
            short_prompt.end_to_end_generated_tps,
            long_prompt.end_to_end_generated_tps
        );
    }

    #[test]
    fn qualification_decode_rate_is_zero_for_no_timed_decode_work() {
        let one_token = QualificationTiming::from_measurement(10, 10, 1, 1.0, 2.0, 3.0);
        assert_eq!(one_token.decode_generated_tokens, 0);
        assert_eq!(one_token.decode_tps, 0.0);
        assert_eq!(one_token.real_generated_tps, 0.0);

        let zero_duration = QualificationTiming::from_measurement(10, 10, 10, 1.0, 0.0, 1.0);
        assert_eq!(zero_duration.decode_tps, 0.0);
        assert_eq!(zero_duration.real_generated_tps, 0.0);
        assert!(zero_duration.decode_tps.is_finite());
        assert!(zero_duration.real_generated_tps.is_finite());
    }

    #[test]
    fn build_provenance_rejects_dirty_and_unavailable() {
        let mut evidence = preflight();
        evidence.provenance.dirty = Some(true);
        assert_eq!(
            validate_preflight(&evidence, &mut QualificationChecks::default())
                .unwrap_err()
                .code,
            "dirty-build"
        );
        evidence = preflight();
        evidence.provenance.git_sha = None;
        assert_eq!(
            validate_preflight(&evidence, &mut QualificationChecks::default())
                .unwrap_err()
                .code,
            "build-provenance-unavailable"
        );
    }

    fn report_with_all_checks() -> QualificationReport {
        let artifact = ArtifactDigest {
            configured_path: "artifact".to_string(),
            canonical_path: "/artifact".to_string(),
            byte_length: 1,
            sha256: "00".repeat(32),
        };
        let mut report = QualificationReport::new(
            clean_provenance(),
            QualificationArtifacts {
                config: Some(artifact.clone()),
                tokenizer: Some(artifact.clone()),
                expert_metadata: Some(artifact),
                expert_data_directory: "/data".to_string(),
                ..Default::default()
            },
            Some(metadata()),
            request_evidence("prompt", "hello", 1, 0),
            None,
        );
        report.execution_plan = Some(exact_plan());
        report.device = Some(GpuDeviceIdentity {
            name: "hardware".to_string(),
            vendor_id: 1,
            device_id: 2,
            device_type: "DiscreteGpu".to_string(),
            wgpu_backend: "vulkan".to_string(),
            driver: "driver".to_string(),
            driver_info: "info".to_string(),
            compute_plane: "wgpu-vulkan".to_string(),
            software_adapter: false,
        });
        report.timing = Some(QualificationTiming::from_measurement(
            1, 1, 1, 1.0, 0.0, 1.0,
        ));
        report.routed_experts = Some(routed_success());
        report.gpu_io = Some(GpuExpertIoSnapshot::default());
        let memory = GpuExpertMemorySnapshot {
            expert_capacity_bytes: 1,
            ..Default::default()
        };
        report.gpu_memory_before = Some(memory);
        report.gpu_memory_after = Some(memory);
        report.qualification_checks = QualificationChecks {
            clean_build: true,
            strict_real_checkpoint: true,
            requested_hybrid: true,
            resolved_planes_match_contract: true,
            native_q4_0_routed_experts: true,
            canonical_q4_layout: true,
            hardware_gpu_adapter: true,
            strict_gpu_failure_policy: true,
            generated_tokens: true,
            observed_routed_experts: true,
            all_selected_experts_attempted_on_gpu: true,
            all_gpu_attempts_succeeded: true,
            zero_cpu_routed_expert_execution: true,
            zero_gpu_cpu_fallbacks: true,
            zero_degraded_expert_substitutions: true,
        };
        report
    }

    #[test]
    fn pass_and_failure_reports_have_stable_schema() {
        let mut pass = report_with_all_checks();
        pass.finish().unwrap();
        let value = serde_json::to_value(&pass).unwrap();
        assert_eq!(value["schema_version"], SCHEMA_VERSION);
        assert_eq!(value["mode"], MODE);
        assert_eq!(value["status"], "pass");
        for section in [
            "provenance",
            "artifacts",
            "expert_metadata",
            "execution_plan",
            "device",
            "request",
            "timing",
            "routed_experts",
            "gpu_io",
            "qualification_checks",
            "gpu_memory_before",
            "gpu_memory_after",
            "external_gpu_memory_artifact",
        ] {
            assert!(value.get(section).is_some(), "missing section {section}");
        }

        let mut incomplete = report_with_all_checks();
        incomplete.gpu_io = None;
        assert_eq!(
            incomplete.finish().unwrap_err().code,
            "qualification-evidence-incomplete"
        );

        let mut fail = report_with_all_checks();
        fail.fail(QualificationFailure::new(
            FailureStage::Startup,
            "gpu-device-unavailable",
            "missing",
        ));
        let value = serde_json::to_value(&fail).unwrap();
        assert_eq!(value["status"], "fail");
        assert_eq!(value["failure"]["stage"], "startup");
        assert_eq!(value["failure"]["code"], "gpu-device-unavailable");
        assert_eq!(value["failure"]["detail"], "missing");
    }
}
