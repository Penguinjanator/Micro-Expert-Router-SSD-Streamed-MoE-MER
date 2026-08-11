//! Fail-closed numerical qualification for MER's canonical Q4_0 GPU path.
//!
//! The pure fixture/comparison code in this module is hardware-independent.
//! Live dispatch is deliberately delegated to narrow methods on the already
//! selected production [`crate::backend::BackendBox`] and [`crate::engine::Engine`].

use crate::backend::{
    BackendBox, GpuDeviceIdentity, GpuExpertIoSnapshot, GpuExpertMemorySnapshot,
    GpuPhysicalExpertResidency,
};
use crate::engine::RoutedExpertExecutionSnapshot;
use crate::inference::{dequantize_q4_0_block, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS};
use crate::qualification::{
    BuildProvenance, ExecutionPlanEvidence, ExpertMetadataEvidence, FailureStage,
    QualificationArtifacts, QualificationFailure, QualificationStatus,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: &str = "mer.strict-hybrid-q4-parity.v1";
pub const MODE: &str = "strict-hybrid-q4-parity";

/// Raw shader output remains f32, but its block-local operation order differs
/// slightly from materialising every scaled CPU weight before the dot product.
pub const RAW_ABSOLUTE_TOLERANCE: f32 = 1.0e-5;
pub const RAW_RELATIVE_TOLERANCE: f32 = 1.0e-4;

/// The production routed-expert boundary accepts and returns f16. The primary
/// complete-expert oracle is therefore the authoritative CPU f32 result rounded
/// to f16, compared with the returned GPU f16 value.
pub const COMPLETE_ABSOLUTE_TOLERANCE: f32 = 2.0e-3;
pub const COMPLETE_RELATIVE_TOLERANCE: f32 = 5.0e-3;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ErrorTolerance {
    pub absolute: f32,
    pub relative: f32,
    pub formula: &'static str,
}

pub const RAW_TOLERANCE: ErrorTolerance = ErrorTolerance {
    absolute: RAW_ABSOLUTE_TOLERANCE,
    relative: RAW_RELATIVE_TOLERANCE,
    formula: "abs_error <= absolute + relative * abs(cpu_reference)",
};

pub const COMPLETE_TOLERANCE: ErrorTolerance = ErrorTolerance {
    absolute: COMPLETE_ABSOLUTE_TOLERANCE,
    relative: COMPLETE_RELATIVE_TOLERANCE,
    formula: "abs_error <= absolute + relative * abs(cpu_f16_reference)",
};

#[derive(Clone, Debug)]
pub struct RawQ4Case {
    pub name: &'static str,
    pub projection: &'static str,
    pub weights: Vec<u8>,
    pub input: Vec<f32>,
    pub rows: usize,
    pub columns: usize,
    pub w_block_off: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RawQ4CaseReport {
    pub name: String,
    pub projection: String,
    pub rows: usize,
    pub columns: usize,
    pub w_block_off: usize,
    pub byte_offset: usize,
    pub starts_on_unaligned_18_byte_boundary: bool,
    pub weight_bytes: usize,
    pub weights_sha256: String,
    pub input_sha256: String,
    pub cpu_f32: Vec<f32>,
    pub gpu_f32: Vec<f32>,
    pub absolute_errors: Vec<f32>,
    pub relative_errors: Vec<f32>,
    pub allowed_errors: Vec<f32>,
    pub worst_index: usize,
    pub max_absolute_error: f32,
    pub max_relative_error: f32,
    pub passed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RawIsolationEvidence {
    pub memory_before: GpuExpertMemorySnapshot,
    pub memory_after: GpuExpertMemorySnapshot,
    pub gpu_io_before: GpuExpertIoSnapshot,
    pub gpu_io_after: GpuExpertIoSnapshot,
    pub selected_physical_before: Option<GpuPhysicalExpertResidency>,
    pub selected_physical_after: Option<GpuPhysicalExpertResidency>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertIdentityEvidence {
    pub global_expert_id: u32,
    pub layer_index: u32,
    pub layer_local_expert_id: u32,
    pub num_layers: usize,
    pub num_experts_per_layer: u32,
    pub total_global_experts: u32,
}

pub fn expert_identity(
    global_expert_id: u32,
    num_layers: usize,
    num_experts_per_layer: u32,
) -> Result<ExpertIdentityEvidence, QualificationFailure> {
    let layers = u32::try_from(num_layers).map_err(|_| {
        QualificationFailure::new(
            FailureStage::Preflight,
            "expert-namespace-overflow",
            format!("num_layers {num_layers} exceeds u32"),
        )
    })?;
    let total = layers.checked_mul(num_experts_per_layer).ok_or_else(|| {
        QualificationFailure::new(
            FailureStage::Preflight,
            "expert-namespace-overflow",
            format!(
                "num_layers {num_layers} * num_experts_per_layer {num_experts_per_layer} overflows u32"
            ),
        )
    })?;
    if global_expert_id >= total {
        return Err(QualificationFailure::new(
            FailureStage::Preflight,
            "global-expert-id-out-of-range",
            format!(
                "global expert id {global_expert_id} is outside 0..{total} for {num_layers} layers * {num_experts_per_layer} experts/layer"
            ),
        ));
    }
    Ok(ExpertIdentityEvidence {
        global_expert_id,
        layer_index: global_expert_id / num_experts_per_layer,
        layer_local_expert_id: global_expert_id % num_experts_per_layer,
        num_layers,
        num_experts_per_layer,
        total_global_experts: total,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct CompleteDispatchSnapshot {
    pub logical_generation: Option<u64>,
    pub physical: Option<GpuPhysicalExpertResidency>,
    pub memory: GpuExpertMemorySnapshot,
    pub gpu_io: GpuExpertIoSnapshot,
    pub routed: RoutedExpertExecutionSnapshot,
}

#[derive(Clone, Debug)]
pub struct CompleteDispatchExecution {
    pub input: Vec<f32>,
    pub cpu_f32: Vec<f32>,
    /// Values returned through the production f16 boundary, represented as
    /// f32 solely for comparison and JSON serialization.
    pub gpu_f16: Vec<f32>,
    pub before: CompleteDispatchSnapshot,
    pub after: CompleteDispatchSnapshot,
}

#[derive(Clone, Debug)]
pub struct CompleteExpertExecution {
    pub d_model: usize,
    pub d_ff: usize,
    pub checkpoint_block_align: usize,
    /// Canonical unpadded Q4_0 weights consumed by both kernels.
    pub payload_bytes: usize,
    /// Header-stripped checkpoint slot, including zero alignment padding.
    pub checkpoint_payload_bytes: usize,
    pub alignment_padding_bytes: usize,
    pub payload_sha256: String,
    pub checkpoint_payload_sha256: String,
    pub dispatches: Vec<CompleteDispatchExecution>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompleteExpertVectorReport {
    pub vector_index: usize,
    pub input_sha256: String,
    pub cpu_f32: Vec<f32>,
    pub cpu_f16: Vec<f32>,
    pub gpu_f16: Vec<f32>,
    pub absolute_errors: Vec<f32>,
    pub relative_errors: Vec<f32>,
    pub allowed_errors: Vec<f32>,
    pub worst_index: usize,
    pub worst_cpu_f32: f32,
    pub worst_cpu_f16: f32,
    pub worst_gpu_f16: f32,
    pub worst_absolute_error: f32,
    pub worst_relative_error: f32,
    pub max_absolute_error: f32,
    pub max_relative_error: f32,
    pub before: CompleteDispatchSnapshot,
    pub after: CompleteDispatchSnapshot,
    pub gpu_io_delta: GpuExpertIoSnapshot,
    pub routed_delta: RoutedExpertExecutionSnapshot,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompleteExpertReport {
    pub identity: ExpertIdentityEvidence,
    pub d_model: usize,
    pub d_ff: usize,
    pub checkpoint_block_align: usize,
    pub physical_device_bytes: u64,
    /// Canonical unpadded Q4_0 weights consumed by both kernels.
    pub payload_bytes: usize,
    /// Header-stripped checkpoint slot, including zero alignment padding.
    pub checkpoint_payload_bytes: usize,
    pub alignment_padding_bytes: usize,
    pub payload_sha256: String,
    pub checkpoint_payload_sha256: String,
    pub vectors: Vec<CompleteExpertVectorReport>,
}

#[derive(Debug)]
pub struct CompleteExpertValidation {
    pub report: CompleteExpertReport,
    pub tolerance_failure: Option<QualificationFailure>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Q4ParityChecks {
    pub clean_build: bool,
    pub strict_hybrid_preflight: bool,
    pub canonical_q4_0_layout: bool,
    pub exact_execution_plan: bool,
    pub hardware_gpu_adapter: bool,
    pub expected_adapter_exact_match: bool,
    pub strict_gpu_failure_policy: bool,
    pub global_expert_identity_valid: bool,
    pub exact_expert_payload_size: bool,
    pub raw_shader_cases_passed: bool,
    pub raw_dispatch_isolated_from_expert_registry: bool,
    pub initial_physical_install_exactly_once: bool,
    pub subsequent_dispatches_reused_generation: bool,
    pub subsequent_dispatches_uploaded_zero_weight_bytes: bool,
    pub every_dispatch_completed_gpu_io: bool,
    pub zero_evictions_or_stale_retirements: bool,
    pub zero_cpu_fallback_or_degraded_execution: bool,
    pub complete_expert_vectors_passed: bool,
}

impl Q4ParityChecks {
    fn passes(&self) -> bool {
        self.clean_build
            && self.strict_hybrid_preflight
            && self.canonical_q4_0_layout
            && self.exact_execution_plan
            && self.hardware_gpu_adapter
            && self.expected_adapter_exact_match
            && self.strict_gpu_failure_policy
            && self.global_expert_identity_valid
            && self.exact_expert_payload_size
            && self.raw_shader_cases_passed
            && self.raw_dispatch_isolated_from_expert_registry
            && self.initial_physical_install_exactly_once
            && self.subsequent_dispatches_reused_generation
            && self.subsequent_dispatches_uploaded_zero_weight_bytes
            && self.every_dispatch_completed_gpu_io
            && self.zero_evictions_or_stale_retirements
            && self.zero_cpu_fallback_or_degraded_execution
            && self.complete_expert_vectors_passed
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Q4ParityReport {
    pub schema_version: &'static str,
    pub mode: &'static str,
    pub status: QualificationStatus,
    pub failure: Option<QualificationFailure>,
    pub provenance: BuildProvenance,
    pub artifacts: QualificationArtifacts,
    pub expert_metadata: Option<ExpertMetadataEvidence>,
    pub execution_plan: Option<ExecutionPlanEvidence>,
    pub device: Option<GpuDeviceIdentity>,
    pub expected_adapter_name: String,
    pub raw_tolerance: ErrorTolerance,
    pub complete_tolerance: ErrorTolerance,
    pub raw_cases: Vec<RawQ4CaseReport>,
    pub raw_isolation: Option<RawIsolationEvidence>,
    pub complete_expert: Option<CompleteExpertReport>,
    pub checks: Q4ParityChecks,
}

impl Q4ParityReport {
    pub fn new(
        provenance: BuildProvenance,
        artifacts: QualificationArtifacts,
        expert_metadata: Option<ExpertMetadataEvidence>,
        expected_adapter_name: String,
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
            expected_adapter_name,
            raw_tolerance: RAW_TOLERANCE,
            complete_tolerance: COMPLETE_TOLERANCE,
            raw_cases: Vec::new(),
            raw_isolation: None,
            complete_expert: None,
            checks: Q4ParityChecks::default(),
        }
    }

    pub fn fail(&mut self, failure: QualificationFailure) {
        self.status = QualificationStatus::Fail;
        self.failure = Some(failure);
    }

    pub fn finish(&mut self) -> Result<(), QualificationFailure> {
        if !self.checks.passes() {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "q4-parity-check-failed",
                "one or more required Q4_0 parity checks are false",
            ));
        }
        if self.artifacts.config.is_none()
            || self.artifacts.expert_metadata.is_none()
            || self.expert_metadata.is_none()
            || self.execution_plan.is_none()
            || self.device.is_none()
            || self.raw_cases.is_empty()
            || self.raw_isolation.is_none()
            || self.complete_expert.is_none()
        {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "q4-parity-evidence-incomplete",
                "one or more mandatory Q4_0 parity evidence sections are absent",
            ));
        }
        self.status = QualificationStatus::Pass;
        self.failure = None;
        Ok(())
    }
}

fn canonical_block(scale: f32, nibbles: [u8; 16]) -> Vec<u8> {
    let mut block = Vec::with_capacity(Q4_0_BLOCK_BYTES);
    block.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
    block.extend_from_slice(&nibbles);
    block
}

fn deterministic_input(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let signed = ((index * (seed * 2 + 3) + seed) % 31) as i32 - 15;
            // Multiples of 1/32 are exactly representable as f16 and f32.
            half::f16::from_f32(signed as f32 / 32.0).to_f32()
        })
        .collect()
}

/// Literal canonical byte fixtures. They intentionally do not call MER's Q4_0
/// quantizer, so a shared encoder/decoder bug cannot make them self-consistent.
pub fn canonical_raw_cases() -> Vec<RawQ4Case> {
    let alternating = std::array::from_fn(|index| if index % 2 == 0 { 0xf0 } else { 0x0f });
    let ramp = std::array::from_fn(|index| index as u8 | ((15 - index as u8) << 4));
    let inverse_ramp = std::array::from_fn(|index| (15 - index as u8) | ((index as u8) << 4));

    let projection_stream = [
        canonical_block(0.25, ramp),
        canonical_block(-0.5, inverse_ramp),
        canonical_block(0.0, alternating),
    ]
    .concat();

    let multiple_blocks = [
        canonical_block(0.125, alternating),
        canonical_block(-0.25, ramp),
    ]
    .concat();

    vec![
        RawQ4Case {
            name: "zero-scale-extrema",
            projection: "standalone",
            weights: canonical_block(0.0, alternating),
            input: deterministic_input(32, 1),
            rows: 1,
            columns: 32,
            w_block_off: 0,
        },
        RawQ4Case {
            name: "positive-scale-extrema",
            projection: "standalone",
            weights: canonical_block(0.25, alternating),
            input: deterministic_input(32, 2),
            rows: 1,
            columns: 32,
            w_block_off: 0,
        },
        RawQ4Case {
            name: "negative-scale-sign",
            projection: "standalone",
            weights: canonical_block(-0.5, ramp),
            input: deterministic_input(32, 3),
            rows: 1,
            columns: 32,
            w_block_off: 0,
        },
        RawQ4Case {
            name: "multiple-blocks-nontrivial-hidden",
            projection: "standalone",
            weights: multiple_blocks,
            input: deterministic_input(64, 4),
            rows: 1,
            columns: 64,
            w_block_off: 0,
        },
        RawQ4Case {
            name: "gate-projection-offset-zero",
            projection: "gate",
            weights: projection_stream.clone(),
            input: deterministic_input(32, 5),
            rows: 1,
            columns: 32,
            w_block_off: 0,
        },
        RawQ4Case {
            name: "up-projection-offset-one-unaligned",
            projection: "up",
            weights: projection_stream.clone(),
            input: deterministic_input(32, 6),
            rows: 1,
            columns: 32,
            w_block_off: 1,
        },
        RawQ4Case {
            name: "down-projection-offset-two",
            projection: "down",
            weights: projection_stream,
            input: deterministic_input(32, 7),
            rows: 1,
            columns: 32,
            w_block_off: 2,
        },
    ]
}

pub fn deterministic_complete_inputs(d_model: usize) -> Vec<Vec<f32>> {
    vec![
        deterministic_input(d_model, 11),
        deterministic_input(d_model, 19),
        (0..d_model)
            .map(|index| {
                let numerator = match index % 7 {
                    0 => 0,
                    1 | 2 => (index % 13 + 1) as i32,
                    _ => -((index % 11 + 1) as i32),
                };
                half::f16::from_f32(numerator as f32 / 64.0).to_f32()
            })
            .collect(),
    ]
}

pub fn cpu_q4_matvec(case: &RawQ4Case) -> Result<Vec<f32>, String> {
    validate_raw_case(case)?;
    let blocks_per_row = case.columns / Q4_0_BLOCK_ELEMS;
    let mut output = Vec::with_capacity(case.rows);
    for row in 0..case.rows {
        let mut sum = 0.0f32;
        for block_in_row in 0..blocks_per_row {
            let block_index = case
                .w_block_off
                .checked_add(
                    row.checked_mul(blocks_per_row)
                        .ok_or("row block overflow")?,
                )
                .and_then(|value| value.checked_add(block_in_row))
                .ok_or("Q4_0 block index overflow")?;
            let start = block_index
                .checked_mul(Q4_0_BLOCK_BYTES)
                .ok_or("Q4_0 byte offset overflow")?;
            let mut decoded = [0.0f32; Q4_0_BLOCK_ELEMS];
            dequantize_q4_0_block(&case.weights[start..start + Q4_0_BLOCK_BYTES], &mut decoded);
            let x_base = block_in_row * Q4_0_BLOCK_ELEMS;
            for (column, weight) in decoded.iter().enumerate() {
                sum += weight * case.input[x_base + column];
            }
        }
        output.push(sum);
    }
    Ok(output)
}

fn validate_raw_case(case: &RawQ4Case) -> Result<(), String> {
    if case.rows == 0 || case.columns == 0 {
        return Err("Q4_0 raw dispatch requires non-zero rows and columns".to_string());
    }
    if !case.columns.is_multiple_of(Q4_0_BLOCK_ELEMS) {
        return Err(format!(
            "Q4_0 raw dispatch columns {} are not a multiple of {}",
            case.columns, Q4_0_BLOCK_ELEMS
        ));
    }
    if case.input.len() != case.columns {
        return Err(format!(
            "Q4_0 raw input has {} values, expected {}",
            case.input.len(),
            case.columns
        ));
    }
    let blocks_per_row = case.columns / Q4_0_BLOCK_ELEMS;
    let accessed_blocks = case
        .rows
        .checked_mul(blocks_per_row)
        .and_then(|count| case.w_block_off.checked_add(count))
        .ok_or("Q4_0 raw geometry overflow")?;
    let required = accessed_blocks
        .checked_mul(Q4_0_BLOCK_BYTES)
        .ok_or("Q4_0 raw byte length overflow")?;
    if case.weights.len() < required || !case.weights.len().is_multiple_of(Q4_0_BLOCK_BYTES) {
        return Err(format!(
            "Q4_0 raw weights have {} bytes, require at least {required} complete 18-byte blocks",
            case.weights.len()
        ));
    }
    if case.input.iter().any(|value| !value.is_finite()) {
        return Err("Q4_0 raw input contains a nonfinite value".to_string());
    }
    Ok(())
}

fn relative_error(reference: f32, absolute_error: f32) -> f32 {
    if reference == 0.0 {
        if absolute_error == 0.0 {
            0.0
        } else {
            f32::MAX
        }
    } else {
        absolute_error / reference.abs()
    }
}

type ComparisonResult = (Vec<f32>, Vec<f32>, Vec<f32>, usize, f32, f32, bool);

fn compare_vectors(
    reference: &[f32],
    actual: &[f32],
    tolerance: ErrorTolerance,
) -> Result<ComparisonResult, String> {
    if reference.len() != actual.len() || reference.is_empty() {
        return Err(format!(
            "comparison length mismatch or empty output: cpu={} gpu={}",
            reference.len(),
            actual.len()
        ));
    }
    if reference
        .iter()
        .chain(actual)
        .any(|value| !value.is_finite())
    {
        return Err("CPU or GPU output contains a nonfinite value".to_string());
    }
    let mut absolute_errors = Vec::with_capacity(reference.len());
    let mut relative_errors = Vec::with_capacity(reference.len());
    let mut allowed_errors = Vec::with_capacity(reference.len());
    let mut worst_index = 0usize;
    let mut worst_ratio = -1.0f32;
    let mut max_absolute = 0.0f32;
    let mut max_relative = 0.0f32;
    let mut passed = true;
    for (index, (&cpu, &gpu)) in reference.iter().zip(actual).enumerate() {
        let absolute = (gpu - cpu).abs();
        let relative = relative_error(cpu, absolute);
        let allowed = tolerance.absolute + tolerance.relative * cpu.abs();
        let ratio = if allowed == 0.0 {
            if absolute == 0.0 {
                0.0
            } else {
                f32::MAX
            }
        } else {
            absolute / allowed
        };
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_index = index;
        }
        max_absolute = max_absolute.max(absolute);
        max_relative = max_relative.max(relative);
        passed &= absolute <= allowed;
        absolute_errors.push(absolute);
        relative_errors.push(relative);
        allowed_errors.push(allowed);
    }
    Ok((
        absolute_errors,
        relative_errors,
        allowed_errors,
        worst_index,
        max_absolute,
        max_relative,
        passed,
    ))
}

pub fn run_raw_shader_cases(
    backend: &BackendBox,
) -> Result<Vec<RawQ4CaseReport>, QualificationFailure> {
    canonical_raw_cases()
        .into_iter()
        .map(|case| {
            let cpu = cpu_q4_matvec(&case).map_err(|detail| {
                QualificationFailure::new(FailureStage::Preflight, "raw-q4-case-invalid", detail)
            })?;
            let gpu = backend
                .qualification_q4_0_matvec(
                    &case.weights,
                    &case.input,
                    case.rows,
                    case.columns,
                    case.w_block_off,
                )
                .map_err(|detail| {
                    QualificationFailure::new(
                        FailureStage::Inference,
                        "raw-q4-gpu-dispatch-failed",
                        format!("{}: {detail}", case.name),
                    )
                })?;
            let (
                absolute_errors,
                relative_errors,
                allowed_errors,
                worst_index,
                max_absolute_error,
                max_relative_error,
                passed,
            ) = compare_vectors(&cpu, &gpu, RAW_TOLERANCE).map_err(|detail| {
                QualificationFailure::new(
                    FailureStage::Postcondition,
                    "raw-q4-comparison-invalid",
                    format!("{}: {detail}", case.name),
                )
            })?;
            if !passed {
                return Err(QualificationFailure::new(
                    FailureStage::Postcondition,
                    "raw-q4-tolerance-failed",
                    format!(
                        "{} failed at output {worst_index}: cpu={} gpu={} abs_error={} allowed={}",
                        case.name,
                        cpu[worst_index],
                        gpu[worst_index],
                        absolute_errors[worst_index],
                        allowed_errors[worst_index]
                    ),
                ));
            }
            let byte_offset = case.w_block_off * Q4_0_BLOCK_BYTES;
            Ok(RawQ4CaseReport {
                name: case.name.to_string(),
                projection: case.projection.to_string(),
                rows: case.rows,
                columns: case.columns,
                w_block_off: case.w_block_off,
                byte_offset,
                starts_on_unaligned_18_byte_boundary: !byte_offset.is_multiple_of(4),
                weight_bytes: case.weights.len(),
                weights_sha256: format!("{:x}", Sha256::digest(&case.weights)),
                input_sha256: hash_f32(&case.input),
                cpu_f32: cpu,
                gpu_f32: gpu,
                absolute_errors,
                relative_errors,
                allowed_errors,
                worst_index,
                max_absolute_error,
                max_relative_error,
                passed,
            })
        })
        .collect()
}

fn subtract_io(
    before: GpuExpertIoSnapshot,
    after: GpuExpertIoSnapshot,
) -> Result<GpuExpertIoSnapshot, String> {
    let sub = |name: &str, before: u64, after: u64| {
        after
            .checked_sub(before)
            .ok_or_else(|| format!("GPU I/O counter {name} decreased"))
    };
    Ok(GpuExpertIoSnapshot {
        expert_weight_uploads: sub(
            "expert_weight_uploads",
            before.expert_weight_uploads,
            after.expert_weight_uploads,
        )?,
        expert_weight_upload_bytes: sub(
            "expert_weight_upload_bytes",
            before.expert_weight_upload_bytes,
            after.expert_weight_upload_bytes,
        )?,
        hidden_state_uploads: sub(
            "hidden_state_uploads",
            before.hidden_state_uploads,
            after.hidden_state_uploads,
        )?,
        hidden_state_upload_bytes: sub(
            "hidden_state_upload_bytes",
            before.hidden_state_upload_bytes,
            after.hidden_state_upload_bytes,
        )?,
        queue_submissions: sub(
            "queue_submissions",
            before.queue_submissions,
            after.queue_submissions,
        )?,
        map_requests: sub("map_requests", before.map_requests, after.map_requests)?,
        readback_completions: sub(
            "readback_completions",
            before.readback_completions,
            after.readback_completions,
        )?,
        readback_bytes: sub(
            "readback_bytes",
            before.readback_bytes,
            after.readback_bytes,
        )?,
    })
}

fn subtract_routed(
    before: RoutedExpertExecutionSnapshot,
    after: RoutedExpertExecutionSnapshot,
) -> Result<RoutedExpertExecutionSnapshot, String> {
    crate::qualification::routed_execution_delta(before, after).map_err(|failure| failure.detail)
}

pub fn validate_complete_expert(
    identity: ExpertIdentityEvidence,
    execution: CompleteExpertExecution,
    d_model: usize,
) -> Result<CompleteExpertValidation, QualificationFailure> {
    if execution.d_model != d_model
        || execution.d_model == 0
        || execution.d_ff == 0
        || execution.checkpoint_block_align == 0
        || !execution.checkpoint_block_align.is_power_of_two()
    {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "complete-expert-geometry-invalid",
            format!(
                "reported geometry d_model={} d_ff={} block_align={} disagrees with expected d_model={d_model} or is zero",
                execution.d_model, execution.d_ff, execution.checkpoint_block_align
            ),
        ));
    }
    if execution.dispatches.len() < 2 {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "insufficient-complete-expert-vectors",
            "complete-expert qualification requires at least two vectors to prove physical reuse",
        ));
    }
    let payload_bytes = u64::try_from(execution.payload_bytes).map_err(|_| {
        QualificationFailure::new(
            FailureStage::Postcondition,
            "expert-payload-size-overflow",
            "complete expert payload does not fit u64",
        )
    })?;
    let checkpoint_payload_bytes =
        u64::try_from(execution.checkpoint_payload_bytes).map_err(|_| {
            QualificationFailure::new(
                FailureStage::Postcondition,
                "checkpoint-payload-size-overflow",
                "checkpoint payload does not fit u64",
            )
        })?;
    if execution
        .payload_bytes
        .checked_add(execution.alignment_padding_bytes)
        != Some(execution.checkpoint_payload_bytes)
        || execution.payload_bytes == 0
        || execution.alignment_padding_bytes >= execution.checkpoint_block_align
        || !execution
            .checkpoint_payload_bytes
            .is_multiple_of(execution.checkpoint_block_align)
    {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "checkpoint-payload-accounting-invalid",
            "canonical Q4_0 bytes plus alignment padding do not equal checkpoint payload bytes",
        ));
    }
    let expected_payload = crate::inference::expert_weight_bytes_for(
        execution.d_model,
        execution.d_ff,
        crate::inference::WeightDtype::Q4_0,
    );
    if execution.payload_bytes != expected_payload {
        return Err(QualificationFailure::new(
            FailureStage::Postcondition,
            "expert-payload-size-mismatch",
            format!(
                "complete expert reports {} canonical Q4_0 bytes, expected exactly {expected_payload}",
                execution.payload_bytes
            ),
        ));
    }
    let physical_device_bytes = payload_bytes
        .checked_add(3)
        .map(|bytes| bytes / 4 * 4)
        .ok_or_else(|| {
            QualificationFailure::new(
                FailureStage::Postcondition,
                "expert-device-size-overflow",
                "Q4_0 physical device byte size overflowed u64",
            )
        })?;
    let output_bytes = u64::try_from(d_model)
        .ok()
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| {
            QualificationFailure::new(
                FailureStage::Postcondition,
                "expert-output-size-overflow",
                "d_model * sizeof(f32) overflowed u64",
            )
        })?;
    let mut reports = Vec::with_capacity(execution.dispatches.len());
    let mut stable_generation = None;
    let mut stable_physical = None;
    let mut previous_after = None;
    let mut tolerance_failure = None;

    for (vector_index, dispatch) in execution.dispatches.into_iter().enumerate() {
        for snapshot in [dispatch.before.memory, dispatch.after.memory] {
            crate::qualification::validate_memory(snapshot).map_err(|failure| {
                QualificationFailure::new(
                    FailureStage::Postcondition,
                    "complete-expert-capacity-ledger-invalid",
                    format!("vector {vector_index}: {}", failure.detail),
                )
            })?;
        }
        if previous_after.is_some_and(|previous| dispatch.before != previous) {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "inter-vector-evidence-discontinuity",
                format!(
                    "vector {vector_index} before snapshot differs from the prior vector after snapshot"
                ),
            ));
        }
        if dispatch.input.len() != d_model
            || dispatch.cpu_f32.len() != d_model
            || dispatch.gpu_f16.len() != d_model
        {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "complete-expert-output-shape-mismatch",
                format!("vector {vector_index} did not return d_model={d_model} values"),
            ));
        }
        if dispatch
            .input
            .iter()
            .any(|value| !value.is_finite() || half::f16::from_f32(*value).to_f32() != *value)
            || dispatch.cpu_f32.iter().any(|value| !value.is_finite())
            || dispatch.gpu_f16.iter().any(|value| !value.is_finite())
        {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "complete-expert-nonfinite-output",
                format!("vector {vector_index} contains nonfinite CPU or GPU output"),
            ));
        }
        let cpu_f16: Vec<f32> = dispatch
            .cpu_f32
            .iter()
            .map(|value| half::f16::from_f32(*value).to_f32())
            .collect();
        if cpu_f16.iter().any(|value| !value.is_finite()) {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "complete-expert-nonfinite-output",
                format!("vector {vector_index} CPU f32 output overflows f16"),
            ));
        }
        let (
            absolute_errors,
            relative_errors,
            allowed_errors,
            worst_index,
            max_absolute_error,
            max_relative_error,
            passed,
        ) = compare_vectors(&cpu_f16, &dispatch.gpu_f16, COMPLETE_TOLERANCE).map_err(|detail| {
            QualificationFailure::new(
                FailureStage::Postcondition,
                "complete-expert-comparison-invalid",
                format!("vector {vector_index}: {detail}"),
            )
        })?;
        if !passed {
            tolerance_failure.get_or_insert_with(|| QualificationFailure::new(
                FailureStage::Postcondition,
                "complete-expert-tolerance-failed",
                format!(
                    "vector {vector_index} failed at output {worst_index}: cpu_f32={} cpu_f16={} gpu_f16={} abs_error={} allowed={}",
                    dispatch.cpu_f32[worst_index], cpu_f16[worst_index], dispatch.gpu_f16[worst_index], absolute_errors[worst_index], allowed_errors[worst_index]
                ),
            ));
        }

        let io_delta =
            subtract_io(dispatch.before.gpu_io, dispatch.after.gpu_io).map_err(|detail| {
                QualificationFailure::new(
                    FailureStage::Postcondition,
                    "gpu-io-counter-invariant",
                    detail,
                )
            })?;
        let routed_delta =
            subtract_routed(dispatch.before.routed, dispatch.after.routed).map_err(|detail| {
                QualificationFailure::new(
                    FailureStage::Postcondition,
                    "routed-counter-invariant",
                    detail,
                )
            })?;
        let after_physical = dispatch.after.physical.ok_or_else(|| {
            QualificationFailure::new(
                FailureStage::Postcondition,
                "physical-expert-missing-after-dispatch",
                format!("vector {vector_index} completed without a physical registry entry"),
            )
        })?;
        let after_generation = dispatch.after.logical_generation.ok_or_else(|| {
            QualificationFailure::new(
                FailureStage::Postcondition,
                "logical-admission-missing-after-dispatch",
                format!("vector {vector_index} completed without a logical admission"),
            )
        })?;
        if after_physical.expert_id != identity.global_expert_id
            || after_physical.generation != after_generation
            || after_physical.device_bytes != physical_device_bytes
            || dispatch.after.memory.logical_admitted_bytes != checkpoint_payload_bytes
            || dispatch.after.memory.expert_live_bytes != physical_device_bytes
            || dispatch.after.memory.expert_registry_bytes != physical_device_bytes
            || dispatch.after.memory.physical_entries != 1
            || dispatch.after.memory.physical_installs != 1
            || dispatch.after.memory.physical_evictions != 0
            || dispatch.after.memory.stale_retirements != 0
        {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "physical-generation-or-ledger-mismatch",
                format!(
                    "vector {vector_index} physical/logical identity or capacity ledger disagrees"
                ),
            ));
        }

        let memory_install_delta = dispatch
            .after
            .memory
            .physical_installs
            .checked_sub(dispatch.before.memory.physical_installs);
        let eviction_delta = dispatch
            .after
            .memory
            .physical_evictions
            .checked_sub(dispatch.before.memory.physical_evictions);
        let stale_delta = dispatch
            .after
            .memory
            .stale_retirements
            .checked_sub(dispatch.before.memory.stale_retirements);
        if eviction_delta != Some(0) || stale_delta != Some(0) {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "physical-registry-churn",
                format!("vector {vector_index} evicted or retired a physical entry"),
            ));
        }
        if io_delta.hidden_state_uploads != 1
            || io_delta.hidden_state_upload_bytes != output_bytes
            || io_delta.queue_submissions != 1
            || io_delta.map_requests != 1
            || io_delta.readback_completions != 1
            || io_delta.readback_bytes != output_bytes
        {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "incomplete-per-vector-gpu-io",
                format!("vector {vector_index} GPU I/O delta is {io_delta:?}"),
            ));
        }
        if routed_delta.gpu_dispatch_attempts != 1
            || routed_delta.gpu_dispatch_successes != 1
            || routed_delta.gpu_dispatch_failures != 0
            || routed_delta.cpu_routed_expert_dispatches != 0
            || routed_delta.gpu_cpu_fallbacks != 0
            || routed_delta.degraded_expert_substitutions != 0
        {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "invalid-per-vector-routed-execution",
                format!("vector {vector_index} routed delta is {routed_delta:?}"),
            ));
        }

        if vector_index == 0 {
            if dispatch.before.physical.is_some()
                || dispatch.before.logical_generation.is_some()
                || dispatch.before.memory.logical_admitted_bytes != 0
                || dispatch.before.memory.expert_live_bytes != 0
                || dispatch.before.memory.physical_entries != 0
                || dispatch.before.memory.expert_registry_bytes != 0
                || dispatch.before.memory.physical_installs != 0
                || dispatch.before.memory.physical_evictions != 0
                || dispatch.before.memory.stale_retirements != 0
                || dispatch.before.gpu_io != GpuExpertIoSnapshot::default()
                || dispatch.before.routed != RoutedExpertExecutionSnapshot::default()
                || memory_install_delta != Some(1)
                || io_delta.expert_weight_uploads != 1
                || io_delta.expert_weight_upload_bytes != after_physical.device_bytes
            {
                return Err(QualificationFailure::new(
                    FailureStage::Postcondition,
                    "initial-physical-install-not-exactly-once",
                    format!(
                        "initial before={:?} after={:?} io={io_delta:?}",
                        dispatch.before, dispatch.after
                    ),
                ));
            }
            stable_generation = Some(after_generation);
            stable_physical = Some(after_physical);
        } else if dispatch.before.logical_generation != stable_generation
            || dispatch.before.physical != stable_physical
            || dispatch.after.logical_generation != stable_generation
            || dispatch.after.physical != stable_physical
            || dispatch.before.memory.physical_entries != 1
            || dispatch.before.memory.expert_registry_bytes != physical_device_bytes
            || memory_install_delta != Some(0)
            || io_delta.expert_weight_uploads != 0
            || io_delta.expert_weight_upload_bytes != 0
        {
            return Err(QualificationFailure::new(
                FailureStage::Postcondition,
                "physical-expert-not-reused",
                format!(
                    "vector {vector_index} changed generation/residency or re-uploaded weights"
                ),
            ));
        }

        reports.push(CompleteExpertVectorReport {
            vector_index,
            input_sha256: hash_f32(&dispatch.input),
            worst_cpu_f32: dispatch.cpu_f32[worst_index],
            worst_cpu_f16: cpu_f16[worst_index],
            worst_gpu_f16: dispatch.gpu_f16[worst_index],
            worst_absolute_error: absolute_errors[worst_index],
            worst_relative_error: relative_errors[worst_index],
            cpu_f32: dispatch.cpu_f32,
            cpu_f16,
            gpu_f16: dispatch.gpu_f16,
            absolute_errors,
            relative_errors,
            allowed_errors,
            worst_index,
            max_absolute_error,
            max_relative_error,
            before: dispatch.before,
            after: dispatch.after,
            gpu_io_delta: io_delta,
            routed_delta,
            passed,
        });
        previous_after = Some(dispatch.after);
    }

    Ok(CompleteExpertValidation {
        report: CompleteExpertReport {
            identity,
            d_model: execution.d_model,
            d_ff: execution.d_ff,
            checkpoint_block_align: execution.checkpoint_block_align,
            physical_device_bytes,
            payload_bytes: execution.payload_bytes,
            checkpoint_payload_bytes: execution.checkpoint_payload_bytes,
            alignment_padding_bytes: execution.alignment_padding_bytes,
            payload_sha256: execution.payload_sha256,
            checkpoint_payload_sha256: execution.checkpoint_payload_sha256,
            vectors: reports,
        },
        tolerance_failure,
    })
}

fn hash_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory_snapshot(
        logical_admitted_bytes: u64,
        expert_bytes: u64,
        physical_entries: usize,
        physical_installs: u64,
    ) -> GpuExpertMemorySnapshot {
        let workspace_bytes = 64;
        GpuExpertMemorySnapshot {
            logical_admitted_bytes,
            expert_live_bytes: expert_bytes,
            expert_registry_bytes: expert_bytes,
            workspace_bytes,
            total_tracked_bytes: expert_bytes + workspace_bytes,
            expert_capacity_bytes: 256,
            physical_entries,
            physical_installs,
            physical_evictions: 0,
            stale_retirements: 0,
        }
    }

    fn io_snapshot(weight_uploads: u64, weight_bytes: u64, dispatches: u64) -> GpuExpertIoSnapshot {
        GpuExpertIoSnapshot {
            expert_weight_uploads: weight_uploads,
            expert_weight_upload_bytes: weight_bytes,
            hidden_state_uploads: dispatches,
            hidden_state_upload_bytes: dispatches * 8,
            queue_submissions: dispatches,
            map_requests: dispatches,
            readback_completions: dispatches,
            readback_bytes: dispatches * 8,
        }
    }

    fn routed_snapshot(dispatches: u64) -> RoutedExpertExecutionSnapshot {
        RoutedExpertExecutionSnapshot {
            gpu_dispatch_attempts: dispatches,
            gpu_dispatch_successes: dispatches,
            ..RoutedExpertExecutionSnapshot::default()
        }
    }

    fn dispatch_snapshot(
        generation: Option<u64>,
        physical: Option<GpuPhysicalExpertResidency>,
        memory: GpuExpertMemorySnapshot,
        io: GpuExpertIoSnapshot,
        routed: RoutedExpertExecutionSnapshot,
    ) -> CompleteDispatchSnapshot {
        CompleteDispatchSnapshot {
            logical_generation: generation,
            physical,
            memory,
            gpu_io: io,
            routed,
        }
    }

    fn valid_complete_execution() -> (ExpertIdentityEvidence, CompleteExpertExecution) {
        let identity = expert_identity(9, 2, 8).unwrap();
        let physical = GpuPhysicalExpertResidency {
            expert_id: identity.global_expert_id,
            generation: 7,
            device_bytes: 56,
        };
        let mut dispatches = Vec::new();
        for vector_index in 0..3u64 {
            let first = vector_index == 0;
            let before_dispatches = vector_index;
            let after_dispatches = vector_index + 1;
            let before = dispatch_snapshot(
                (!first).then_some(7),
                (!first).then_some(physical),
                if first {
                    memory_snapshot(0, 0, 0, 0)
                } else {
                    memory_snapshot(64, 56, 1, 1)
                },
                if first {
                    io_snapshot(0, 0, before_dispatches)
                } else {
                    io_snapshot(1, 56, before_dispatches)
                },
                routed_snapshot(before_dispatches),
            );
            let after = dispatch_snapshot(
                Some(7),
                Some(physical),
                memory_snapshot(64, 56, 1, 1),
                io_snapshot(1, 56, after_dispatches),
                routed_snapshot(after_dispatches),
            );
            dispatches.push(CompleteDispatchExecution {
                input: vec![vector_index as f32, -0.5],
                cpu_f32: vec![1.0001, -0.5],
                gpu_f16: vec![1.0, -0.5],
                before,
                after,
            });
        }
        (
            identity,
            CompleteExpertExecution {
                d_model: 2,
                d_ff: 2,
                checkpoint_block_align: 64,
                payload_bytes: 54,
                checkpoint_payload_bytes: 64,
                alignment_padding_bytes: 10,
                payload_sha256: "fixture".to_string(),
                checkpoint_payload_sha256: "checkpoint-fixture".to_string(),
                dispatches,
            },
        )
    }

    #[test]
    fn canonical_cases_cover_required_scale_offsets_and_projections() {
        let cases = canonical_raw_cases();
        assert!(cases.iter().any(|case| case.name.contains("zero-scale")));
        assert!(cases
            .iter()
            .any(|case| case.name.contains("positive-scale")));
        assert!(cases
            .iter()
            .any(|case| case.name.contains("negative-scale")));
        assert!(cases.iter().any(|case| case.columns > Q4_0_BLOCK_ELEMS));
        for projection in ["gate", "up", "down"] {
            assert!(cases.iter().any(|case| case.projection == projection));
        }
        let unaligned = cases
            .iter()
            .find(|case| case.name.contains("unaligned"))
            .unwrap();
        assert_ne!((unaligned.w_block_off * Q4_0_BLOCK_BYTES) % 4, 0);
    }

    #[test]
    fn literal_extrema_and_negative_scale_use_authoritative_decoder() {
        let positive = canonical_block(0.25, [0xf0; 16]);
        let mut decoded = [0.0f32; 32];
        dequantize_q4_0_block(&positive, &mut decoded);
        assert_eq!(&decoded[..16], &[-2.0; 16]);
        assert_eq!(&decoded[16..], &[1.75; 16]);

        let negative = canonical_block(-0.5, [0xf0; 16]);
        dequantize_q4_0_block(&negative, &mut decoded);
        assert_eq!(&decoded[..16], &[4.0; 16]);
        assert_eq!(&decoded[16..], &[-3.5; 16]);
    }

    #[test]
    fn q4_parity_complete_cpu_oracle_executes_authoritative_q4_0_forward() {
        let d_model = 32;
        let d_ff = 32;
        let block = canonical_block(
            0.125,
            std::array::from_fn(|index| index as u8 | ((15 - index as u8) << 4)),
        );
        let block_count = crate::inference::expert_weight_bytes_for(
            d_model,
            d_ff,
            crate::inference::WeightDtype::Q4_0,
        ) / Q4_0_BLOCK_BYTES;
        let bytes = block.repeat(block_count);
        let input = deterministic_input(d_model, 23);
        let output =
            crate::inference::q4_0_cpu_reference_forward(&bytes, &input, d_model, d_ff).unwrap();
        assert_eq!(output.len(), d_model);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn zero_scale_is_zero_even_with_extremal_nibbles() {
        let case = &canonical_raw_cases()[0];
        assert_eq!(cpu_q4_matvec(case).unwrap(), vec![0.0]);
    }

    #[test]
    fn fixed_combined_tolerance_accepts_boundary_and_rejects_excess() {
        let cpu = [2.0f32];
        let allowed = RAW_ABSOLUTE_TOLERANCE + RAW_RELATIVE_TOLERANCE * 2.0;
        assert!(
            compare_vectors(&cpu, &[2.0 + allowed * 0.9], RAW_TOLERANCE)
                .unwrap()
                .6
        );
        assert!(
            !compare_vectors(&cpu, &[2.0 + allowed * 1.1], RAW_TOLERANCE)
                .unwrap()
                .6
        );
    }

    #[test]
    fn nonfinite_values_fail_before_tolerance_evaluation() {
        assert!(compare_vectors(&[f32::NAN], &[0.0], RAW_TOLERANCE).is_err());
        assert!(compare_vectors(&[0.0], &[f32::INFINITY], COMPLETE_TOLERANCE).is_err());
    }

    #[test]
    fn deterministic_complete_inputs_are_nontrivial_finite_and_f16_exact() {
        for input in deterministic_complete_inputs(64) {
            assert!(input.iter().any(|value| *value > 0.0));
            assert!(input.iter().any(|value| *value < 0.0));
            assert!(input.iter().all(|value| value.is_finite()));
            assert!(input
                .iter()
                .all(|value| half::f16::from_f32(*value).to_f32() == *value));
        }
    }

    #[test]
    fn global_expert_identity_never_wraps_or_reinterprets() {
        let identity = expert_identity(257, 48, 128).unwrap();
        assert_eq!(identity.layer_index, 2);
        assert_eq!(identity.layer_local_expert_id, 1);
        assert_eq!(identity.total_global_experts, 6144);
        assert_eq!(
            expert_identity(6144, 48, 128).unwrap_err().code,
            "global-expert-id-out-of-range"
        );
        assert_eq!(
            expert_identity(0, usize::MAX, u32::MAX).unwrap_err().code,
            "expert-namespace-overflow"
        );
    }

    #[test]
    fn malformed_raw_geometry_and_payload_fail_loudly() {
        let mut case = canonical_raw_cases().remove(0);
        case.columns = 31;
        assert!(cpu_q4_matvec(&case).unwrap_err().contains("multiple"));
        case.columns = 32;
        case.weights.pop();
        assert!(cpu_q4_matvec(&case)
            .unwrap_err()
            .contains("complete 18-byte blocks"));
    }

    #[test]
    fn complete_report_uses_cpu_f16_as_primary_oracle_and_preserves_f32() {
        let (identity, execution) = valid_complete_execution();
        let validation = validate_complete_expert(identity, execution, 2).unwrap();
        assert!(validation.tolerance_failure.is_none());
        let report = validation.report;
        assert_eq!(report.payload_bytes, 54);
        assert_eq!(report.checkpoint_payload_bytes, 64);
        assert_eq!(report.alignment_padding_bytes, 10);
        assert_eq!(report.physical_device_bytes, 56);
        let vector = &report.vectors[0];
        assert_eq!(vector.cpu_f32, vec![1.0001, -0.5]);
        assert_eq!(vector.cpu_f16, vec![1.0, -0.5]);
        assert_eq!(vector.gpu_f16, vec![1.0, -0.5]);
        assert_eq!(vector.absolute_errors, vec![0.0, 0.0]);
        assert!(vector.passed);
    }

    #[test]
    fn complete_evidence_proves_one_install_then_stable_reuse_and_per_vector_io() {
        let (identity, execution) = valid_complete_execution();
        let validation = validate_complete_expert(identity, execution, 2).unwrap();
        assert!(validation.tolerance_failure.is_none());
        let report = validation.report;
        assert_eq!(report.vectors.len(), 3);
        assert_eq!(report.vectors[0].gpu_io_delta.expert_weight_uploads, 1);
        assert_eq!(
            report.vectors[0].gpu_io_delta.expert_weight_upload_bytes,
            56
        );
        for vector in &report.vectors {
            assert_eq!(vector.gpu_io_delta.hidden_state_uploads, 1);
            assert_eq!(vector.gpu_io_delta.hidden_state_upload_bytes, 8);
            assert_eq!(vector.gpu_io_delta.queue_submissions, 1);
            assert_eq!(vector.gpu_io_delta.map_requests, 1);
            assert_eq!(vector.gpu_io_delta.readback_completions, 1);
            assert_eq!(vector.gpu_io_delta.readback_bytes, 8);
            assert_eq!(vector.routed_delta.gpu_dispatch_attempts, 1);
            assert_eq!(vector.routed_delta.gpu_dispatch_successes, 1);
        }
        for vector in &report.vectors[1..] {
            assert_eq!(vector.before.logical_generation, Some(7));
            assert_eq!(vector.after.logical_generation, Some(7));
            assert_eq!(vector.before.physical, vector.after.physical);
            assert_eq!(vector.gpu_io_delta.expert_weight_uploads, 0);
            assert_eq!(vector.gpu_io_delta.expert_weight_upload_bytes, 0);
        }
    }

    #[test]
    fn complete_evidence_fails_on_reupload_generation_change_or_bad_ledger() {
        let (identity, mut execution) = valid_complete_execution();
        execution.dispatches[1].after.gpu_io.expert_weight_uploads += 1;
        assert_eq!(
            validate_complete_expert(identity.clone(), execution, 2)
                .unwrap_err()
                .code,
            "physical-expert-not-reused"
        );

        let (identity, mut execution) = valid_complete_execution();
        execution.dispatches[1].after.logical_generation = Some(8);
        assert_eq!(
            validate_complete_expert(identity.clone(), execution, 2)
                .unwrap_err()
                .code,
            "physical-generation-or-ledger-mismatch"
        );

        let (identity, mut execution) = valid_complete_execution();
        execution.dispatches[0].after.memory.total_tracked_bytes += 1;
        assert_eq!(
            validate_complete_expert(identity, execution, 2)
                .unwrap_err()
                .code,
            "complete-expert-capacity-ledger-invalid"
        );

        let (identity, mut execution) = valid_complete_execution();
        execution.alignment_padding_bytes = 9;
        assert_eq!(
            validate_complete_expert(identity.clone(), execution, 2)
                .unwrap_err()
                .code,
            "checkpoint-payload-accounting-invalid"
        );

        let (identity, mut execution) = valid_complete_execution();
        execution.dispatches[1].before.routed.gpu_dispatch_attempts += 1;
        assert_eq!(
            validate_complete_expert(identity.clone(), execution, 2)
                .unwrap_err()
                .code,
            "inter-vector-evidence-discontinuity"
        );

        let (identity, mut execution) = valid_complete_execution();
        execution.dispatches[0].gpu_f16[0] = 1.25;
        let validation = validate_complete_expert(identity, execution, 2).unwrap();
        assert_eq!(
            validation.tolerance_failure.unwrap().code,
            "complete-expert-tolerance-failed"
        );
        assert!(!validation.report.vectors[0].passed);
        assert_eq!(validation.report.vectors[0].cpu_f32[0], 1.0001);
        assert_eq!(validation.report.vectors[0].cpu_f16[0], 1.0);
        assert_eq!(validation.report.vectors[0].gpu_f16[0], 1.25);
        assert_eq!(validation.report.vectors[0].worst_index, 0);
    }
}
