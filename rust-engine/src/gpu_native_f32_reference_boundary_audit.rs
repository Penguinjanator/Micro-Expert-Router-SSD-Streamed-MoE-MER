//! Diagnostic-only audit of the CPU-reference boundary used by frozen
//! GPU-native Q4 evidence.
//!
//! This module never constructs a GPU runtime and never contributes to a
//! qualification decision. It consumes immutable report snapshots, replays
//! their exact f32 vectors through existing CPU/diagnostic arithmetic, and
//! generates the frozen v2 corpus through an ordinary strict CPU Q4 runtime.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::gpu_native_expert_permutation_semantic_parity::VectorNumericalEvidence;
use crate::{IsolatedRuntimeShutdownError, RealCliRuntimeMode, ResolvedRealCliSpec};

pub const SCHEMA_VERSION: &str = "mer.gpu-native-f32-reference-boundary-audit.v1";
pub const MODE: &str = "diagnose-gpu-native-f32-reference-boundary";
pub const QUALIFICATION_PASS: bool = false;

pub const FROZEN_V2_REPORT_SHA256: &str =
    "9629de35a18f4457bbe38aeeaeb208fd979bb08e9d2eccb283437d6c16a5c4ad";
pub const FROZEN_STAGE_REPORT_SHA256: &str =
    "f81827b39d2ece62b9c1af432f71ca70045fe41c13e6e6ac15b03d460246365d";
pub const FROZEN_V2_BUILD_SHA: &str = "300448ac3da48ac74ef86fcd2c62ffca27ffa634";
pub const FROZEN_FIRST_ATTRIBUTION_BUILD_SHA: &str = "d826d4578bc7037fb96ad67372f1442b20f5e4d1";
pub const FROZEN_BEA_BUILD_SHA: &str = "bea43722a8fc00fd76d3c45702f70c16e2b63041";
pub const FROZEN_STAGE_BUILD_SHA: &str = "987e7e4b8791dc70f75b62a772e89ad9f1948ec5";
pub const FROZEN_STAGE_OFFLINE_ANALYSIS_SHA256: &str =
    "053ef60e71cbc46f5d035c0f60a90b9fdfc429da6efed44262f58813b2b187cd";
pub const FROZEN_RUST_UPSTREAM_TRACE_SHA256: &str =
    "94d1d00e69a83f4510aece9c3c338f85d57a051619b73e897610d61a3e958920";

const EXPECTED_TARGET_COUNT: usize = 9;
const EXPECTED_EXPERTS_PER_TARGET: usize = 8;
const EXPECTED_EXPERT_RECORD_COUNT: usize = EXPECTED_TARGET_COUNT * EXPECTED_EXPERTS_PER_TARGET;
const EXPECTED_NUM_LAYERS: usize = 48;
const EXPECTED_NUM_EXPERTS: usize = 128;
const EXPECTED_D_MODEL: usize = 2_048;
const AUDIT_RUNTIME_MODE: RealCliRuntimeMode = RealCliRuntimeMode::IsolatedGreedyParityCpu;
const CPU_REFERENCE_IMPLEMENTATION: &str = "existing-production-q4_0_cpu_reference_forward-candle";
const CURRENT_GPU_EMULATOR_IMPLEMENTATION: &str =
    "existing-diagnostic-current-production-gpu-q4-arithmetic-emulator";

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn exact_f64(left: f64, right: f64) -> bool {
    left.to_bits() == right.to_bits()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct FrozenTargetIdentity {
    id: String,
    case: String,
    generated_position: usize,
    layer: usize,
    frozen_worst_local_expert: Option<u32>,
}

fn expected_targets() -> Vec<FrozenTargetIdentity> {
    crate::gpu_native_q4_expert_stage_attribution::FROZEN_TARGETS
        .into_iter()
        .map(|target| FrozenTargetIdentity {
            id: target.id.to_string(),
            case: target.case.to_string(),
            generated_position: target.generated_position,
            layer: target.layer,
            frozen_worst_local_expert: target.frozen_worst_local_expert,
        })
        .collect()
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenBuildEnvelope {
    git_sha: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenProvenanceEnvelope {
    build: FrozenBuildEnvelope,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct FrozenNumericalLimitsEnvelope {
    max_absolute_error_limit: f64,
    rms_error_limit: f64,
    mean_absolute_error_limit: f64,
    nonfinite_mismatch_limit: usize,
    semantic_correctness_not_bit_parity: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenCorpusEnvelope {
    id: String,
    version: u32,
    sha256: String,
    case_count: usize,
    output_token_limit: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenTokenCaseEnvelope {
    case: String,
    reference_generated_token_ids: Vec<u32>,
    gpu_generated_token_ids: Vec<u32>,
    exact_match_count: usize,
    mismatch_count: usize,
    first_mismatch_position: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenV2Envelope {
    schema: String,
    qualification_pass: bool,
    holdout_corpus: FrozenCorpusEnvelope,
    numerical_limits: FrozenNumericalLimitsEnvelope,
    provenance: FrozenProvenanceEnvelope,
    token_cases: Vec<FrozenTokenCaseEnvelope>,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenExactVectorEnvelope {
    source: String,
    vector_length: usize,
    f32_bits_sha256: String,
    f32_bits: Vec<u32>,
    nonfinite_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenInputBoundaryEnvelope {
    effective_gpu_expert_input: FrozenExactVectorEnvelope,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenOutputBoundaryEnvelope {
    cpu_effective_output: FrozenExactVectorEnvelope,
    actual_gpu_effective_output: FrozenExactVectorEnvelope,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenStageExpertEnvelope {
    rank: usize,
    local_expert_id: u32,
    global_expert_id: u32,
    canonical_q4_payload_sha256: String,
    final_output_boundary: FrozenOutputBoundaryEnvelope,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenStageTargetEnvelope {
    target: FrozenTargetIdentity,
    input_boundary: Option<FrozenInputBoundaryEnvelope>,
    experts: Vec<FrozenStageExpertEnvelope>,
    failure: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenStageEnvelope {
    schema: String,
    diagnostic_complete: bool,
    qualification_pass: bool,
    failure: Option<String>,
    provenance: FrozenProvenanceEnvelope,
    frozen_targets: Vec<FrozenTargetIdentity>,
    targets: Vec<FrozenStageTargetEnvelope>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrozenExpertIdentity {
    pub target_id: String,
    pub case: String,
    pub generated_position: usize,
    pub layer: usize,
    pub rank: usize,
    pub local_expert_id: u32,
    pub global_expert_id: u32,
    pub canonical_q4_payload_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrozenHistoricalEvidenceIdentity {
    pub v2_qualification_build_sha: &'static str,
    pub v2_report_sha256: &'static str,
    pub first_attribution_setup_failure_build_sha: &'static str,
    pub successful_v2_failure_attribution_build_sha: &'static str,
    pub successful_q4_stage_attribution_build_sha: &'static str,
    pub stage_report_sha256: &'static str,
    pub stage_offline_analysis_sha256: &'static str,
    pub rust_upstream_trace_sha256: &'static str,
    pub modified_or_reclassified: bool,
}

impl Default for FrozenHistoricalEvidenceIdentity {
    fn default() -> Self {
        Self {
            v2_qualification_build_sha: FROZEN_V2_BUILD_SHA,
            v2_report_sha256: FROZEN_V2_REPORT_SHA256,
            first_attribution_setup_failure_build_sha: FROZEN_FIRST_ATTRIBUTION_BUILD_SHA,
            successful_v2_failure_attribution_build_sha: FROZEN_BEA_BUILD_SHA,
            successful_q4_stage_attribution_build_sha: FROZEN_STAGE_BUILD_SHA,
            stage_report_sha256: FROZEN_STAGE_REPORT_SHA256,
            stage_offline_analysis_sha256: FROZEN_STAGE_OFFLINE_ANALYSIS_SHA256,
            rust_upstream_trace_sha256: FROZEN_RUST_UPSTREAM_TRACE_SHA256,
            modified_or_reclassified: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrozenReportsIdentity {
    pub v2_report: crate::qualification::ArtifactDigest,
    pub expected_v2_report_sha256_argument: String,
    pub stage_report: crate::qualification::ArtifactDigest,
    pub expected_stage_report_sha256_argument: String,
    pub v2_schema: &'static str,
    pub stage_schema: &'static str,
    pub v2_build_sha: &'static str,
    pub stage_build_sha: &'static str,
    pub holdout_corpus_id: &'static str,
    pub holdout_corpus_sha256: &'static str,
    pub holdout_case_count: usize,
    pub output_token_limit: usize,
    pub exact_expert_identity_count: usize,
    pub exact_expert_identities: Vec<FrozenExpertIdentity>,
    pub immutable_inputs_verified_before_runtime: bool,
    pub historical: FrozenHistoricalEvidenceIdentity,
}

struct FrozenAuditInputs {
    v2: FrozenV2Envelope,
    stage: FrozenStageEnvelope,
    identity: FrozenReportsIdentity,
}

fn validate_expected_report_sha(value: &str, frozen: &str, label: &str) -> Result<(), String> {
    if !is_hex(value, 64) || !value.eq_ignore_ascii_case(frozen) {
        return Err(format!("{label} expected SHA must equal frozen {frozen}"));
    }
    Ok(())
}

fn artifact_from_snapshot(
    path: &Path,
    bytes: &[u8],
) -> Result<crate::qualification::ArtifactDigest, Box<dyn std::error::Error>> {
    Ok(crate::qualification::ArtifactDigest {
        configured_path: path.display().to_string(),
        canonical_path: std::fs::canonicalize(path)?.display().to_string(),
        byte_length: u64::try_from(bytes.len()).map_err(|_| "report byte length overflow")?,
        sha256: crate::greedy_parity::sha256_hex(bytes),
    })
}

fn validate_snapshot_hash(
    artifact: &crate::qualification::ArtifactDigest,
    expected_argument: &str,
    frozen: &str,
    label: &str,
) -> Result<(), String> {
    if !artifact.sha256.eq_ignore_ascii_case(expected_argument)
        || !artifact.sha256.eq_ignore_ascii_case(frozen)
    {
        return Err(format!(
            "frozen {label} SHA differs: observed {} expected {frozen}",
            artifact.sha256
        ));
    }
    Ok(())
}

fn validate_frozen_v2(envelope: &FrozenV2Envelope) -> Result<(), String> {
    use crate::gpu_native_semantic_parity_v2 as v2;

    if envelope.schema != v2::SCHEMA_VERSION {
        return Err("frozen v2 report schema differs".into());
    }
    if envelope.qualification_pass {
        return Err("frozen v2 qualification_pass must remain false".into());
    }
    if envelope.provenance.build.git_sha.as_deref() != Some(FROZEN_V2_BUILD_SHA) {
        return Err("frozen v2 report build SHA differs".into());
    }
    let corpus = &envelope.holdout_corpus;
    if corpus.id != v2::HOLDOUT_CORPUS_ID
        || corpus.version != v2::HOLDOUT_CORPUS_VERSION
        || corpus.sha256 != v2::HOLDOUT_CORPUS_SHA256
        || corpus.case_count != v2::HOLDOUT_CORPUS_CASE_COUNT
        || corpus.output_token_limit != v2::OUTPUT_TOKEN_LIMIT
        || v2::holdout_corpus_sha256() != v2::HOLDOUT_CORPUS_SHA256
    {
        return Err("frozen v2 holdout corpus identity differs".into());
    }
    let expected_limits = v2::NumericalLimits::frozen();
    let limits = envelope.numerical_limits;
    if !exact_f64(
        limits.max_absolute_error_limit,
        expected_limits.max_absolute_error_limit,
    ) || !exact_f64(limits.rms_error_limit, expected_limits.rms_error_limit)
        || !exact_f64(
            limits.mean_absolute_error_limit,
            expected_limits.mean_absolute_error_limit,
        )
        || limits.nonfinite_mismatch_limit != expected_limits.nonfinite_mismatch_limit
        || limits.semantic_correctness_not_bit_parity
            != expected_limits.semantic_correctness_not_bit_parity
    {
        return Err("frozen v2 numerical limits differ".into());
    }
    if envelope.token_cases.len() != v2::HOLDOUT_CORPUS_CASE_COUNT {
        return Err("frozen v2 token-case count differs".into());
    }
    for (observed, expected) in envelope.token_cases.iter().zip(v2::HOLDOUT_CORPUS) {
        if observed.case != expected.name
            || observed.reference_generated_token_ids.len() != v2::OUTPUT_TOKEN_LIMIT
            || observed.gpu_generated_token_ids.len() != v2::OUTPUT_TOKEN_LIMIT
        {
            return Err(format!(
                "frozen v2 token case {} is incomplete",
                expected.name
            ));
        }
        let recomputed = crate::gpu_native_semantic_parity_corpus::TokenCaseEvidence::new(
            observed.case.clone(),
            observed.reference_generated_token_ids.clone(),
            observed.gpu_generated_token_ids.clone(),
        );
        if observed.exact_match_count != recomputed.exact_match_count
            || observed.mismatch_count != recomputed.mismatch_count
            || observed.first_mismatch_position != recomputed.first_mismatch_position
        {
            return Err(format!(
                "frozen v2 token comparison fields differ for {}",
                expected.name
            ));
        }
    }
    let rust = envelope
        .token_cases
        .iter()
        .find(|case| case.case == "rust-ownership-holdout")
        .ok_or("frozen v2 rust-ownership case is missing")?;
    if rust.reference_generated_token_ids.get(1) != Some(&785)
        || rust.gpu_generated_token_ids.get(1) != Some(&8822)
    {
        return Err("frozen v2 rust-ownership position-1 evidence differs".into());
    }
    Ok(())
}

fn validate_exact_vector(
    vector: &FrozenExactVectorEnvelope,
    expected_source: &str,
    label: &str,
) -> Result<(), String> {
    if vector.source != expected_source
        || vector.vector_length != EXPECTED_D_MODEL
        || vector.f32_bits.len() != EXPECTED_D_MODEL
        || !is_hex(&vector.f32_bits_sha256, 64)
        || crate::numerical_diagnostics::f32_bits_sha256(&vector.f32_bits) != vector.f32_bits_sha256
    {
        return Err(format!("frozen {label} vector identity differs"));
    }
    let observed_nonfinite = vector
        .f32_bits
        .iter()
        .filter(|bits| !f32::from_bits(**bits).is_finite())
        .count();
    if vector.nonfinite_count != observed_nonfinite || observed_nonfinite != 0 {
        return Err(format!(
            "frozen {label} contains invalid nonfinite evidence"
        ));
    }
    Ok(())
}

fn validate_frozen_stage(
    envelope: &FrozenStageEnvelope,
) -> Result<Vec<FrozenExpertIdentity>, String> {
    if envelope.schema != crate::gpu_native_q4_expert_stage_attribution::SCHEMA_VERSION {
        return Err("frozen stage report schema differs".into());
    }
    if !envelope.diagnostic_complete || envelope.qualification_pass || envelope.failure.is_some() {
        return Err("frozen stage report completion/qualification/failure fields differ".into());
    }
    if envelope.provenance.build.git_sha.as_deref() != Some(FROZEN_STAGE_BUILD_SHA) {
        return Err("frozen stage report build SHA differs".into());
    }
    let expected = expected_targets();
    if envelope.frozen_targets != expected || envelope.targets.len() != expected.len() {
        return Err("frozen Q1-Q4/R1-R5 target identities differ".into());
    }

    let mut identities = Vec::with_capacity(EXPECTED_EXPERT_RECORD_COUNT);
    for (target, expected_target) in envelope.targets.iter().zip(expected) {
        if target.target != expected_target || target.failure.is_some() {
            return Err(format!(
                "frozen stage target {} identity/failure differs",
                expected_target.id
            ));
        }
        let input = target.input_boundary.as_ref().ok_or_else(|| {
            format!(
                "frozen stage target {} has no input boundary",
                target.target.id
            )
        })?;
        validate_exact_vector(
            &input.effective_gpu_expert_input,
            "effective-production-gpu-expert-input",
            &format!("{} effective GPU input", target.target.id),
        )?;
        if target.experts.len() != EXPECTED_EXPERTS_PER_TARGET {
            return Err(format!(
                "frozen stage target {} does not contain exactly eight experts",
                target.target.id
            ));
        }
        let mut local_ids = BTreeSet::new();
        for (index, expert) in target.experts.iter().enumerate() {
            if expert.rank != index + 1
                || expert.local_expert_id as usize >= EXPECTED_NUM_EXPERTS
                || !local_ids.insert(expert.local_expert_id)
                || expert.global_expert_id
                    != (target.target.layer * EXPECTED_NUM_EXPERTS) as u32 + expert.local_expert_id
                || !is_hex(&expert.canonical_q4_payload_sha256, 64)
            {
                return Err(format!(
                    "frozen expert identity differs at target {} rank {}",
                    target.target.id,
                    index + 1
                ));
            }
            validate_exact_vector(
                &expert.final_output_boundary.cpu_effective_output,
                "cpu-diagnostic-f16-boundary-emulated-expert-output",
                &format!(
                    "{} rank {} historical CPU output",
                    target.target.id, expert.rank
                ),
            )?;
            validate_exact_vector(
                &expert.final_output_boundary.actual_gpu_effective_output,
                "actual-production-gpu-expert-output",
                &format!(
                    "{} rank {} actual GPU output",
                    target.target.id, expert.rank
                ),
            )?;
            identities.push(FrozenExpertIdentity {
                target_id: target.target.id.clone(),
                case: target.target.case.clone(),
                generated_position: target.target.generated_position,
                layer: target.target.layer,
                rank: expert.rank,
                local_expert_id: expert.local_expert_id,
                global_expert_id: expert.global_expert_id,
                canonical_q4_payload_sha256: expert.canonical_q4_payload_sha256.clone(),
            });
        }
        if let Some(worst) = target.target.frozen_worst_local_expert {
            if !local_ids.contains(&worst) {
                return Err(format!(
                    "frozen target {} is missing its frozen worst expert {worst}",
                    target.target.id
                ));
            }
        }
    }
    if identities.len() != EXPECTED_EXPERT_RECORD_COUNT {
        return Err("frozen stage report does not contain exactly 72 expert identities".into());
    }
    Ok(identities)
}

impl FrozenAuditInputs {
    fn read_and_validate(
        v2_report: &Path,
        expected_v2_sha256: &str,
        stage_report: &Path,
        expected_stage_sha256: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        validate_expected_report_sha(expected_v2_sha256, FROZEN_V2_REPORT_SHA256, "v2 report")?;
        validate_expected_report_sha(
            expected_stage_sha256,
            FROZEN_STAGE_REPORT_SHA256,
            "stage report",
        )?;

        // Each immutable report is read exactly once. The corresponding hash
        // and typed deserialization both consume this same byte snapshot.
        let v2_bytes = std::fs::read(v2_report)?;
        let stage_bytes = std::fs::read(stage_report)?;
        let v2_artifact = artifact_from_snapshot(v2_report, &v2_bytes)?;
        let stage_artifact = artifact_from_snapshot(stage_report, &stage_bytes)?;
        validate_snapshot_hash(
            &v2_artifact,
            expected_v2_sha256,
            FROZEN_V2_REPORT_SHA256,
            "v2 report",
        )?;
        validate_snapshot_hash(
            &stage_artifact,
            expected_stage_sha256,
            FROZEN_STAGE_REPORT_SHA256,
            "stage report",
        )?;
        let v2: FrozenV2Envelope = serde_json::from_slice(&v2_bytes)
            .map_err(|error| format!("malformed frozen v2 report: {error}"))?;
        let stage: FrozenStageEnvelope = serde_json::from_slice(&stage_bytes)
            .map_err(|error| format!("malformed frozen stage report: {error}"))?;
        validate_frozen_v2(&v2)?;
        let exact_expert_identities = validate_frozen_stage(&stage)?;
        let identity = FrozenReportsIdentity {
            v2_report: v2_artifact,
            expected_v2_report_sha256_argument: expected_v2_sha256.to_ascii_lowercase(),
            stage_report: stage_artifact,
            expected_stage_report_sha256_argument: expected_stage_sha256.to_ascii_lowercase(),
            v2_schema: crate::gpu_native_semantic_parity_v2::SCHEMA_VERSION,
            stage_schema: crate::gpu_native_q4_expert_stage_attribution::SCHEMA_VERSION,
            v2_build_sha: FROZEN_V2_BUILD_SHA,
            stage_build_sha: FROZEN_STAGE_BUILD_SHA,
            holdout_corpus_id: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_ID,
            holdout_corpus_sha256: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_SHA256,
            holdout_case_count: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_CASE_COUNT,
            output_token_limit: crate::gpu_native_semantic_parity_v2::OUTPUT_TOKEN_LIMIT,
            exact_expert_identity_count: exact_expert_identities.len(),
            exact_expert_identities,
            immutable_inputs_verified_before_runtime: true,
            historical: FrozenHistoricalEvidenceIdentity::default(),
        };
        Ok(Self {
            v2,
            stage,
            identity,
        })
    }
}

fn output_path_conflicts_with_frozen_input(
    report_out: &Path,
    inputs: &FrozenReportsIdentity,
) -> Result<bool, Box<dyn std::error::Error>> {
    let output = if report_out.exists() {
        std::fs::canonicalize(report_out)?
    } else {
        let absolute = if report_out.is_absolute() {
            report_out.to_path_buf()
        } else {
            std::env::current_dir()?.join(report_out)
        };
        let mut existing = absolute.as_path();
        let mut missing = Vec::new();
        while !existing.exists() {
            missing.push(
                existing
                    .file_name()
                    .ok_or("report output path has no existing ancestor")?
                    .to_os_string(),
            );
            existing = existing
                .parent()
                .ok_or("report output path has no existing ancestor")?;
        }
        let mut canonical = std::fs::canonicalize(existing)?;
        for component in missing.into_iter().rev() {
            canonical.push(component);
        }
        canonical
    };
    Ok(output == PathBuf::from(&inputs.v2_report.canonical_path)
        || output == PathBuf::from(&inputs.stage_report.canonical_path))
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Q4F32ReferenceReplayBundle {
    pub canonical_payload_sha256: String,
    pub old_boundary_input: Vec<f32>,
    pub old_pre_output: Vec<f32>,
    pub old_effective_output: Vec<f32>,
    pub exact_f32_output: Vec<f32>,
    pub current_gpu_emulator_output: Vec<f32>,
}

fn compute_reference_paths_with<RoundTrip, Production, Emulator>(
    exact_input: &[f32],
    mut round_trip: RoundTrip,
    mut production: Production,
    mut emulator: Emulator,
) -> Result<Q4F32ReferenceReplayBundle, String>
where
    RoundTrip: FnMut(&[f32]) -> Result<Vec<f32>, String>,
    Production: FnMut(&[f32]) -> Result<Vec<f32>, String>,
    Emulator: FnMut(&[f32]) -> Result<Vec<f32>, String>,
{
    let old_boundary_input = round_trip(exact_input)?;
    let old_pre_output = production(&old_boundary_input)?;
    let old_effective_output = round_trip(&old_pre_output)?;
    let exact_f32_output = production(exact_input)?;
    let current_gpu_emulator_output = emulator(exact_input)?;
    Ok(Q4F32ReferenceReplayBundle {
        canonical_payload_sha256: String::new(),
        old_boundary_input,
        old_pre_output,
        old_effective_output,
        exact_f32_output,
        current_gpu_emulator_output,
    })
}

pub(crate) fn audit_q4_reference_paths(
    payload: &[u8],
    exact_input: &[f32],
    d_model: usize,
    d_ff: usize,
    canonical_payload_sha256: String,
) -> Result<Q4F32ReferenceReplayBundle, String> {
    let mut bundle = compute_reference_paths_with(
        exact_input,
        |values| {
            crate::numerical_diagnostics::round_trip_f16_values(values)
                .map_err(|error| error.to_string())
        },
        |values| {
            crate::inference::q4_0_cpu_reference_forward(payload, values, d_model, d_ff)
                .map_err(|error| error.to_string())
        },
        |values| {
            crate::inference::diagnostic_q4_0_expert_arithmetic(
                payload,
                values,
                d_model,
                d_ff,
                crate::inference::DiagnosticQ4DotArithmetic::CurrentGpu,
            )
            .map(|trace| trace.down)
            .map_err(|error| error.to_string())
        },
    )?;
    bundle.canonical_payload_sha256 = canonical_payload_sha256;
    Ok(bundle)
}

fn reconstruct_exact_f32(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VectorIdentityEvidence {
    pub source: &'static str,
    pub vector_length: usize,
    pub f32_bits_sha256: String,
    pub nonfinite_count: usize,
}

impl VectorIdentityEvidence {
    fn new(source: &'static str, values: &[f32]) -> Self {
        let bits = values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        Self {
            source,
            vector_length: values.len(),
            f32_bits_sha256: crate::numerical_diagnostics::f32_bits_sha256(&bits),
            nonfinite_count: values.iter().filter(|value| !value.is_finite()).count(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ErrorImprovementEvidence {
    pub exact_f32_error_smaller_than_old_f16_error_maxabs: bool,
    pub exact_f32_error_smaller_than_old_f16_error_rms: bool,
    pub exact_f32_error_smaller_than_old_f16_error_meanabs: bool,
    pub exact_f32_to_old_f16_maxabs_ratio: Option<f64>,
    pub exact_f32_to_old_f16_rms_ratio: Option<f64>,
    pub exact_f32_to_old_f16_meanabs_ratio: Option<f64>,
}

fn finite_ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    numerator
        .zip(denominator)
        .filter(|(numerator, denominator)| {
            numerator.is_finite() && denominator.is_finite() && *denominator != 0.0
        })
        .map(|(numerator, denominator)| numerator / denominator)
        .filter(|ratio| ratio.is_finite())
}

fn strictly_smaller(left: Option<f64>, right: Option<f64>) -> bool {
    left.zip(right)
        .is_some_and(|(left, right)| left.is_finite() && right.is_finite() && left < right)
}

impl ErrorImprovementEvidence {
    fn new(exact: &VectorNumericalEvidence, old: &VectorNumericalEvidence) -> Self {
        Self {
            exact_f32_error_smaller_than_old_f16_error_maxabs: strictly_smaller(
                exact.max_absolute_error,
                old.max_absolute_error,
            ),
            exact_f32_error_smaller_than_old_f16_error_rms: strictly_smaller(
                exact.rms_error,
                old.rms_error,
            ),
            exact_f32_error_smaller_than_old_f16_error_meanabs: strictly_smaller(
                exact.mean_absolute_error,
                old.mean_absolute_error,
            ),
            exact_f32_to_old_f16_maxabs_ratio: finite_ratio(
                exact.max_absolute_error,
                old.max_absolute_error,
            ),
            exact_f32_to_old_f16_rms_ratio: finite_ratio(exact.rms_error, old.rms_error),
            exact_f32_to_old_f16_meanabs_ratio: finite_ratio(
                exact.mean_absolute_error,
                old.mean_absolute_error,
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExpertBoundaryAuditRecord {
    pub target_id: String,
    pub case: String,
    pub generated_position: usize,
    pub layer: usize,
    pub rank: usize,
    pub local_expert_id: u32,
    pub global_expert_id: u32,
    pub canonical_q4_payload_sha256: String,
    pub exact_gpu_f32_input: VectorIdentityEvidence,
    pub old_f16_boundary_input: VectorIdentityEvidence,
    pub old_f16_reference_reproduces_frozen_historical_output_bit_exactly: bool,
    pub old_f16_reference_reproduction: VectorNumericalEvidence,
    pub old_f16_emulated_cpu_reference_vs_frozen_actual_gpu_output: VectorNumericalEvidence,
    pub exact_f32_cpu_reference_vs_frozen_actual_gpu_output: VectorNumericalEvidence,
    pub current_gpu_arithmetic_emulator_vs_frozen_actual_gpu_output: VectorNumericalEvidence,
    pub exact_f32_cpu_reference_vs_current_gpu_arithmetic_emulator: VectorNumericalEvidence,
    pub improvement: ErrorImprovementEvidence,
    pub exact_f32_cpu_reference_implementation: &'static str,
    pub old_input_f16_round_trip_performed: bool,
    pub old_output_f16_round_trip_performed: bool,
    pub exact_f32_input_f16_round_trip_performed: bool,
    pub exact_f32_output_f16_round_trip_performed: bool,
    pub current_gpu_emulator_implementation: &'static str,
    pub frozen_gpu_output_source: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SelectedWorstExpertAudit {
    pub target_id: String,
    pub selection_basis: &'static str,
    pub expert: ExpertBoundaryAuditRecord,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MetricDistribution {
    pub total_expert_records: usize,
    pub finite_value_count: usize,
    pub minimum: Option<f64>,
    pub median: Option<f64>,
    pub mean: Option<f64>,
    pub p90_nearest_rank: Option<f64>,
    pub p95_nearest_rank: Option<f64>,
    pub p99_nearest_rank: Option<f64>,
    pub maximum: Option<f64>,
    pub sorted_finite_values: Vec<f64>,
}

fn nearest_rank(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let rank = (percentile * values.len() as f64).ceil() as usize;
    values
        .get(rank.saturating_sub(1).min(values.len() - 1))
        .copied()
}

fn metric_distribution(
    total: usize,
    values: impl IntoIterator<Item = Option<f64>>,
) -> MetricDistribution {
    let mut values = values
        .into_iter()
        .flatten()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let count = values.len();
    let mean = (count != 0).then(|| values.iter().sum::<f64>() / count as f64);
    MetricDistribution {
        total_expert_records: total,
        finite_value_count: count,
        minimum: values.first().copied(),
        median: nearest_rank(&values, 0.50),
        mean,
        p90_nearest_rank: nearest_rank(&values, 0.90),
        p95_nearest_rank: nearest_rank(&values, 0.95),
        p99_nearest_rank: nearest_rank(&values, 0.99),
        maximum: values.last().copied(),
        sorted_finite_values: values,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReferenceErrorDistributions {
    pub max_absolute_error: MetricDistribution,
    pub rms_error: MetricDistribution,
    pub mean_absolute_error: MetricDistribution,
}

fn error_distributions<'a>(
    comparisons: impl IntoIterator<Item = &'a VectorNumericalEvidence> + Clone,
) -> ReferenceErrorDistributions {
    let comparisons = comparisons.into_iter().collect::<Vec<_>>();
    let total = comparisons.len();
    ReferenceErrorDistributions {
        max_absolute_error: metric_distribution(
            total,
            comparisons
                .iter()
                .map(|comparison| comparison.max_absolute_error),
        ),
        rms_error: metric_distribution(
            total,
            comparisons.iter().map(|comparison| comparison.rms_error),
        ),
        mean_absolute_error: metric_distribution(
            total,
            comparisons
                .iter()
                .map(|comparison| comparison.mean_absolute_error),
        ),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PartAAggregation {
    pub expert_record_count: usize,
    pub old_f16_reference: ReferenceErrorDistributions,
    pub exact_f32_reference: ReferenceErrorDistributions,
    pub current_gpu_arithmetic_emulator: ReferenceErrorDistributions,
    pub exact_f32_smaller_than_old_f16_maxabs_count: usize,
    pub exact_f32_smaller_than_old_f16_rms_count: usize,
    pub exact_f32_smaller_than_old_f16_meanabs_count: usize,
}

fn aggregate_part_a(records: &[ExpertBoundaryAuditRecord]) -> PartAAggregation {
    PartAAggregation {
        expert_record_count: records.len(),
        old_f16_reference: error_distributions(
            records
                .iter()
                .map(|record| &record.old_f16_emulated_cpu_reference_vs_frozen_actual_gpu_output),
        ),
        exact_f32_reference: error_distributions(
            records
                .iter()
                .map(|record| &record.exact_f32_cpu_reference_vs_frozen_actual_gpu_output),
        ),
        current_gpu_arithmetic_emulator: error_distributions(
            records
                .iter()
                .map(|record| &record.current_gpu_arithmetic_emulator_vs_frozen_actual_gpu_output),
        ),
        exact_f32_smaller_than_old_f16_maxabs_count: records
            .iter()
            .filter(|record| {
                record
                    .improvement
                    .exact_f32_error_smaller_than_old_f16_error_maxabs
            })
            .count(),
        exact_f32_smaller_than_old_f16_rms_count: records
            .iter()
            .filter(|record| {
                record
                    .improvement
                    .exact_f32_error_smaller_than_old_f16_error_rms
            })
            .count(),
        exact_f32_smaller_than_old_f16_meanabs_count: records
            .iter()
            .filter(|record| {
                record
                    .improvement
                    .exact_f32_error_smaller_than_old_f16_error_meanabs
            })
            .count(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CpuTokenCaseAudit {
    pub case: String,
    pub requested_tokens: usize,
    pub completed_tokens: usize,
    pub exact_token_matches: usize,
    pub mismatch_count: usize,
    pub first_mismatch_position: Option<usize>,
    pub corrected_ordinary_f32_cpu_token_ids: Vec<u32>,
    pub frozen_gpu_token_ids: Vec<u32>,
    pub historical_old_boundary_cpu_reference_token_ids: Vec<u32>,
    pub frozen_gpu_tokens_source: &'static str,
}

impl CpuTokenCaseAudit {
    fn new(frozen: &FrozenTokenCaseEnvelope, corrected: Vec<u32>) -> Self {
        let comparison = crate::gpu_native_semantic_parity_corpus::TokenCaseEvidence::new(
            frozen.case.clone(),
            corrected.clone(),
            frozen.gpu_generated_token_ids.clone(),
        );
        Self {
            case: frozen.case.clone(),
            requested_tokens: crate::gpu_native_semantic_parity_v2::OUTPUT_TOKEN_LIMIT,
            completed_tokens: corrected.len(),
            exact_token_matches: comparison.exact_match_count,
            mismatch_count: comparison.mismatch_count,
            first_mismatch_position: comparison.first_mismatch_position,
            corrected_ordinary_f32_cpu_token_ids: corrected,
            frozen_gpu_token_ids: frozen.gpu_generated_token_ids.clone(),
            historical_old_boundary_cpu_reference_token_ids: frozen
                .reference_generated_token_ids
                .clone(),
            frozen_gpu_tokens_source: "immutable-frozen-v2-report-byte-snapshot",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RustOwnershipPositionOneEvidence {
    pub case: &'static str,
    pub generated_position: usize,
    pub historical_old_cpu_token_id: u32,
    pub frozen_gpu_token_id: u32,
    pub new_ordinary_f32_cpu_token_id: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PartBTokenSummary {
    pub requested_tokens: usize,
    pub completed_tokens: usize,
    pub exact_token_matches: usize,
    pub mismatch_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeContractEvidence {
    pub runtime_mode: &'static str,
    pub execution_plan: crate::qualification::ExecutionPlanEvidence,
    pub gpu_runtime_constructed: bool,
    pub gpu_device_identity_exposed: bool,
    pub gpu_native_token_loop_constructed: bool,
    pub boundary_emulation_before_part_a: crate::engine::CpuQ4BoundaryEmulationSnapshot,
    pub boundary_emulation_before_part_b: crate::engine::CpuQ4BoundaryEmulationSnapshot,
    pub boundary_emulation_after_part_b: crate::engine::CpuQ4BoundaryEmulationSnapshot,
    pub routed_execution_after: crate::engine::RoutedExpertExecutionSnapshot,
    pub background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AuditProvenance {
    pub build: crate::qualification::BuildProvenance,
    pub executable_sha256: String,
    pub artifacts: crate::qualification::QualificationArtifacts,
    pub source_config: crate::gpu_native_greedy_parity::SourceConfigEvidence,
    pub cpu_resolved_config_sha256: String,
    pub model_identity: crate::greedy_parity::ModelIdentityEvidence,
    pub model_load: crate::greedy_parity::ModelLoadEvidence,
    pub expert_metadata: crate::qualification::ExpertMetadataEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProductionSemanticsEvidence {
    pub diagnostic_only: bool,
    pub production_inference_changed: bool,
    pub production_q4_changed: bool,
    pub production_q4_wgsl_changed: bool,
    pub production_router_changed: bool,
    pub production_attention_changed: bool,
    pub v1_changed: bool,
    pub v2_changed: bool,
    pub v2_limits_changed: bool,
    pub v2_corpus_or_prompts_changed: bool,
    pub cpu_q4_boundary_emulation_behavior_changed: bool,
    pub numerical_qualification_threshold_introduced: bool,
    pub production_correction_made_or_justified: bool,
    pub frozen_gpu_outputs_read_only_from_stage_report: bool,
    pub frozen_gpu_tokens_read_only_from_v2_report: bool,
}

impl Default for ProductionSemanticsEvidence {
    fn default() -> Self {
        Self {
            diagnostic_only: true,
            production_inference_changed: false,
            production_q4_changed: false,
            production_q4_wgsl_changed: false,
            production_router_changed: false,
            production_attention_changed: false,
            v1_changed: false,
            v2_changed: false,
            v2_limits_changed: false,
            v2_corpus_or_prompts_changed: false,
            cpu_q4_boundary_emulation_behavior_changed: false,
            numerical_qualification_threshold_introduced: false,
            production_correction_made_or_justified: false,
            frozen_gpu_outputs_read_only_from_stage_report: true,
            frozen_gpu_tokens_read_only_from_v2_report: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct F32ReferenceBoundaryAuditReport {
    pub schema: &'static str,
    pub mode: &'static str,
    pub diagnostic_only: bool,
    pub diagnostic_complete: bool,
    qualification_pass: bool,
    pub failure: Option<String>,
    pub frozen_inputs: FrozenReportsIdentity,
    pub provenance: AuditProvenance,
    pub expert_records: Vec<ExpertBoundaryAuditRecord>,
    pub q1_q4_frozen_worst_experts: Vec<SelectedWorstExpertAudit>,
    pub r1_r5_worst_old_boundary_error_experts: Vec<SelectedWorstExpertAudit>,
    pub part_a_aggregation: PartAAggregation,
    pub corrected_cpu_token_cases: Vec<CpuTokenCaseAudit>,
    pub rust_ownership_position_one: RustOwnershipPositionOneEvidence,
    pub part_b_token_summary: PartBTokenSummary,
    pub runtime: RuntimeContractEvidence,
    pub production_semantics: ProductionSemanticsEvidence,
    pub scientific_interpretation: Vec<String>,
}

impl F32ReferenceBoundaryAuditReport {
    pub const fn qualification_pass(&self) -> bool {
        self.qualification_pass
    }
}

struct ExecutionCapture {
    expert_records: Vec<ExpertBoundaryAuditRecord>,
    token_cases: Vec<CpuTokenCaseAudit>,
    rust_ownership_position_one: RustOwnershipPositionOneEvidence,
    model_load: crate::greedy_parity::ModelLoadEvidence,
    runtime: RuntimeContractEvidence,
}

fn model_load_is_strict(load: &crate::greedy_parity::ModelLoadEvidence) -> bool {
    load.strict
        && load.required_tensors > 0
        && load.loaded_tensors == load.required_tensors
        && !load.seeded_fallback_remained
        && load.loader != "seeded"
}

fn boundary_disabled_and_clean(
    snapshot: crate::engine::CpuQ4BoundaryEmulationSnapshot,
) -> Result<(), String> {
    if snapshot.enabled || snapshot.routed_expert_dispatches != 0 {
        return Err(
            "CPU Q4 boundary emulation must remain disabled with zero routed dispatches".into(),
        );
    }
    Ok(())
}

async fn audit_one_expert(
    engine: &Arc<crate::engine::Engine>,
    target: &FrozenStageTargetEnvelope,
    expert: &FrozenStageExpertEnvelope,
) -> Result<ExpertBoundaryAuditRecord, Box<dyn std::error::Error>> {
    let input = target
        .input_boundary
        .as_ref()
        .ok_or("validated target lost its input boundary")?;
    let exact_input = reconstruct_exact_f32(&input.effective_gpu_expert_input.f32_bits);
    if exact_input.iter().map(|value| value.to_bits()).ne(input
        .effective_gpu_expert_input
        .f32_bits
        .iter()
        .copied())
    {
        return Err("exact GPU f32 input reconstruction changed frozen bits".into());
    }
    let frozen_old =
        reconstruct_exact_f32(&expert.final_output_boundary.cpu_effective_output.f32_bits);
    let frozen_gpu = reconstruct_exact_f32(
        &expert
            .final_output_boundary
            .actual_gpu_effective_output
            .f32_bits,
    );
    let bundle = engine
        .audit_q4_0_f32_reference_boundary(
            expert.global_expert_id,
            &exact_input,
            &expert.canonical_q4_payload_sha256,
        )
        .await?;
    let old_reproduction = VectorNumericalEvidence::compare(
        "replayed-old-f16-boundary-cpu-reference",
        "frozen-historical-old-f16-boundary-cpu-reference",
        &bundle.old_effective_output,
        &frozen_old,
    )?;
    if !old_reproduction.exact_bit_equal {
        return Err(format!(
            "old boundary reference did not reproduce frozen output bit-exactly for {} rank {} expert {}",
            target.target.id, expert.rank, expert.local_expert_id
        )
        .into());
    }
    let old_vs_gpu = VectorNumericalEvidence::compare(
        "old-f16-emulated-cpu-reference",
        "frozen-actual-production-gpu-output",
        &bundle.old_effective_output,
        &frozen_gpu,
    )?;
    let exact_vs_gpu = VectorNumericalEvidence::compare(
        "exact-f32-production-cpu-reference",
        "frozen-actual-production-gpu-output",
        &bundle.exact_f32_output,
        &frozen_gpu,
    )?;
    let emulator_vs_gpu = VectorNumericalEvidence::compare(
        "current-gpu-q4-arithmetic-cpu-emulator",
        "frozen-actual-production-gpu-output",
        &bundle.current_gpu_emulator_output,
        &frozen_gpu,
    )?;
    let exact_vs_emulator = VectorNumericalEvidence::compare(
        "exact-f32-production-cpu-reference",
        "current-gpu-q4-arithmetic-cpu-emulator",
        &bundle.exact_f32_output,
        &bundle.current_gpu_emulator_output,
    )?;
    let improvement = ErrorImprovementEvidence::new(&exact_vs_gpu, &old_vs_gpu);
    Ok(ExpertBoundaryAuditRecord {
        target_id: target.target.id.clone(),
        case: target.target.case.clone(),
        generated_position: target.target.generated_position,
        layer: target.target.layer,
        rank: expert.rank,
        local_expert_id: expert.local_expert_id,
        global_expert_id: expert.global_expert_id,
        canonical_q4_payload_sha256: bundle.canonical_payload_sha256,
        exact_gpu_f32_input: VectorIdentityEvidence::new(
            "reconstructed-exact-frozen-actual-gpu-f32-expert-input",
            &exact_input,
        ),
        old_f16_boundary_input: VectorIdentityEvidence::new(
            "explicit-f32-to-f16-to-f32-old-reference-input",
            &bundle.old_boundary_input,
        ),
        old_f16_reference_reproduces_frozen_historical_output_bit_exactly: true,
        old_f16_reference_reproduction: old_reproduction,
        old_f16_emulated_cpu_reference_vs_frozen_actual_gpu_output: old_vs_gpu,
        exact_f32_cpu_reference_vs_frozen_actual_gpu_output: exact_vs_gpu,
        current_gpu_arithmetic_emulator_vs_frozen_actual_gpu_output: emulator_vs_gpu,
        exact_f32_cpu_reference_vs_current_gpu_arithmetic_emulator: exact_vs_emulator,
        improvement,
        exact_f32_cpu_reference_implementation: CPU_REFERENCE_IMPLEMENTATION,
        old_input_f16_round_trip_performed: true,
        old_output_f16_round_trip_performed: true,
        exact_f32_input_f16_round_trip_performed: false,
        exact_f32_output_f16_round_trip_performed: false,
        current_gpu_emulator_implementation: CURRENT_GPU_EMULATOR_IMPLEMENTATION,
        frozen_gpu_output_source: "immutable-frozen-stage-report-byte-snapshot",
    })
}

async fn execute_cpu_audit(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    resolved_config_sha256: &str,
    frozen: &FrozenAuditInputs,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<ExecutionCapture, Box<dyn std::error::Error>> {
    let runtime =
        crate::build_isolated_greedy_runtime(spec, AUDIT_RUNTIME_MODE, tokenizer.clone()).await?;
    let attempt = async {
        let observed_config_sha256 = crate::resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err("CPU audit runtime configuration identity drifted".into());
        }
        let execution_plan: crate::qualification::ExecutionPlanEvidence =
            runtime.engine.execution_context().plan().into();
        let gpu_device_identity_exposed = runtime.engine.gpu_device_identity().is_some();
        let gpu_native_token_loop_constructed = runtime.gpu_native_token_loop.is_some();
        let gpu_runtime_constructed =
            gpu_device_identity_exposed || gpu_native_token_loop_constructed;
        if gpu_runtime_constructed || !crate::greedy_parity::cpu_plan_exact(&execution_plan) {
            return Err("audit did not resolve to an exact CPU-only execution plan".into());
        }
        if runtime.model.config.num_layers != EXPECTED_NUM_LAYERS
            || runtime.model.config.num_experts != EXPECTED_NUM_EXPERTS
            || runtime.model.config.top_k != EXPECTED_EXPERTS_PER_TARGET
            || runtime.model.config.d_model != EXPECTED_D_MODEL
        {
            return Err(
                "CPU audit runtime model geometry differs from frozen Qwen geometry".into(),
            );
        }
        let model_load = crate::greedy_parity_model_load(&runtime);
        if !model_load_is_strict(&model_load) {
            return Err("CPU audit runtime did not load the strict model".into());
        }
        let boundary_before_part_a = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        boundary_disabled_and_clean(boundary_before_part_a)?;

        let mut expert_records = Vec::with_capacity(EXPECTED_EXPERT_RECORD_COUNT);
        for target in &frozen.stage.targets {
            for expert in &target.experts {
                expert_records.push(
                    crate::with_progress_timeout(
                        format!(
                            "f32 boundary audit {} rank {} expert {}",
                            target.target.id, expert.rank, expert.local_expert_id
                        ),
                        watchdog,
                        audit_one_expert(&runtime.engine, target, expert),
                    )
                    .await?,
                );
            }
        }
        if expert_records.len() != EXPECTED_EXPERT_RECORD_COUNT {
            return Err("CPU audit did not replay exactly 72 experts".into());
        }

        let boundary_before_part_b = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        boundary_disabled_and_clean(boundary_before_part_b)?;
        let mut token_cases =
            Vec::with_capacity(crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_CASE_COUNT);
        for (case_index, fixed) in crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS
            .into_iter()
            .enumerate()
        {
            let frozen_case = &frozen.v2.token_cases[case_index];
            if frozen_case.case != fixed.name {
                return Err("validated frozen v2 token-case order drifted".into());
            }
            let prompt_ids = tokenizer.encode(fixed.prompt)?;
            let measured = crate::with_progress_timeout(
                format!("f32 boundary audit ordinary CPU generation {}", fixed.name),
                watchdog,
                crate::run_real_once_from_token_ids(
                    &runtime,
                    &prompt_ids,
                    crate::gpu_native_semantic_parity_v2::OUTPUT_TOKEN_LIMIT,
                    crate::sampling::SamplingParams::greedy(),
                    case_index,
                ),
            )
            .await?;
            let corrected = measured.report.output_token_ids;
            if corrected.len() != crate::gpu_native_semantic_parity_v2::OUTPUT_TOKEN_LIMIT {
                return Err(format!(
                    "ordinary CPU generation {} completed {} tokens, expected {}",
                    fixed.name,
                    corrected.len(),
                    crate::gpu_native_semantic_parity_v2::OUTPUT_TOKEN_LIMIT
                )
                .into());
            }
            token_cases.push(CpuTokenCaseAudit::new(frozen_case, corrected));
        }
        let boundary_after_part_b = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        boundary_disabled_and_clean(boundary_after_part_b)?;
        let rust_case = token_cases
            .iter()
            .find(|case| case.case == "rust-ownership-holdout")
            .ok_or("ordinary CPU audit omitted rust-ownership holdout")?;
        let rust_ownership_position_one = RustOwnershipPositionOneEvidence {
            case: "rust-ownership-holdout",
            generated_position: 1,
            historical_old_cpu_token_id: *rust_case
                .historical_old_boundary_cpu_reference_token_ids
                .get(1)
                .ok_or("historical rust CPU position 1 is missing")?,
            frozen_gpu_token_id: *rust_case
                .frozen_gpu_token_ids
                .get(1)
                .ok_or("frozen rust GPU position 1 is missing")?,
            new_ordinary_f32_cpu_token_id: *rust_case
                .corrected_ordinary_f32_cpu_token_ids
                .get(1)
                .ok_or("ordinary rust CPU position 1 is missing")?,
        };
        if rust_ownership_position_one.historical_old_cpu_token_id != 785
            || rust_ownership_position_one.frozen_gpu_token_id != 8822
        {
            return Err("rust-ownership frozen position-1 evidence changed".into());
        }
        Ok::<_, Box<dyn std::error::Error>>(ExecutionCapture {
            expert_records,
            token_cases,
            rust_ownership_position_one,
            model_load,
            runtime: RuntimeContractEvidence {
                runtime_mode: "RealCliRuntimeMode::IsolatedGreedyParityCpu",
                execution_plan,
                gpu_runtime_constructed,
                gpu_device_identity_exposed,
                gpu_native_token_loop_constructed,
                boundary_emulation_before_part_a: boundary_before_part_a,
                boundary_emulation_before_part_b: boundary_before_part_b,
                boundary_emulation_after_part_b: boundary_after_part_b,
                routed_execution_after: runtime.engine.routed_expert_execution_snapshot(),
                background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(),
            },
        })
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut capture), Ok(shutdown)) => {
            capture.runtime.background_shutdown = shutdown;
            if !shutdown.controlled_shutdown_requested || !shutdown.all_runtime_resources_released {
                return Err("CPU audit runtime failed controlled shutdown".into());
            }
            Ok(capture)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; CPU audit runtime shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

fn select_worst_experts(
    records: &[ExpertBoundaryAuditRecord],
) -> Result<(Vec<SelectedWorstExpertAudit>, Vec<SelectedWorstExpertAudit>), String> {
    let mut q = Vec::with_capacity(4);
    let mut r = Vec::with_capacity(5);
    for target in expected_targets() {
        let candidates = records
            .iter()
            .filter(|record| record.target_id == target.id)
            .collect::<Vec<_>>();
        if candidates.len() != EXPECTED_EXPERTS_PER_TARGET {
            return Err(format!("target {} audit record count differs", target.id));
        }
        if let Some(frozen_worst) = target.frozen_worst_local_expert {
            let record = candidates
                .into_iter()
                .find(|record| record.local_expert_id == frozen_worst)
                .ok_or_else(|| format!("target {} frozen worst expert is missing", target.id))?;
            q.push(SelectedWorstExpertAudit {
                target_id: target.id,
                selection_basis: "frozen-q1-q4-worst-local-expert-identity",
                expert: record.clone(),
            });
        } else {
            let record = candidates
                .into_iter()
                .max_by(|left, right| {
                    left.old_f16_emulated_cpu_reference_vs_frozen_actual_gpu_output
                        .max_absolute_error
                        .unwrap_or(f64::NEG_INFINITY)
                        .total_cmp(
                            &right
                                .old_f16_emulated_cpu_reference_vs_frozen_actual_gpu_output
                                .max_absolute_error
                                .unwrap_or(f64::NEG_INFINITY),
                        )
                        .then_with(|| right.rank.cmp(&left.rank))
                })
                .ok_or_else(|| format!("target {} has no old-boundary comparison", target.id))?;
            r.push(SelectedWorstExpertAudit {
                target_id: target.id,
                selection_basis: "largest-old-f16-reference-vs-frozen-gpu-max-absolute-error",
                expert: record.clone(),
            });
        }
    }
    Ok((q, r))
}

fn emit_report(
    report: &F32ReferenceBoundaryAuditReport,
    report_out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if report.qualification_pass() || !report.diagnostic_only || !report.diagnostic_complete {
        return Err(
            "f32 boundary audit report must be complete, diagnostic-only, and qualification false"
                .into(),
        );
    }
    if let Some(parent) = report_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    std::fs::write(report_out, json)?;
    eprintln!(
        "GPU-native f32 reference-boundary audit report written to {}",
        report_out.display()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn run_diagnostic(
    config: PathBuf,
    cfg: crate::config::Config,
    v2_report: PathBuf,
    expected_v2_report_sha256: String,
    stage_report: PathBuf,
    expected_stage_report_sha256: String,
    report_out: PathBuf,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::BuildProvenance;

    // Immutable report validation is deliberately the first operation in the
    // command handler, before tokenizer, model, engine, or backend creation.
    let frozen = FrozenAuditInputs::read_and_validate(
        &v2_report,
        &expected_v2_report_sha256,
        &stage_report,
        &expected_stage_report_sha256,
    )?;
    if output_path_conflicts_with_frozen_input(&report_out, &frozen.identity)? {
        return Err("report output must not overwrite either frozen historical input".into());
    }
    if progress_watchdog.timeout.is_none() {
        return Err("f32 reference-boundary audit requires a positive progress timeout".into());
    }
    let build = BuildProvenance::embedded();
    if build.dirty != Some(false) || build.git_sha.as_deref().is_none_or(|sha| !is_hex(sha, 40)) {
        return Err("f32 reference-boundary audit requires clean embedded Git provenance".into());
    }
    let (artifacts, artifact_errors) = crate::qualification_artifacts(&config, &cfg);
    if !artifact_errors.is_empty() {
        return Err(format!(
            "f32 reference-boundary artifact preflight failed: {}",
            artifact_errors.join("; ")
        )
        .into());
    }
    let capacity_bytes = cfg
        .gpu_cache
        .vram_capacity_mb
        .checked_mul(1024 * 1024)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(0);
    let source_config = crate::gpu_native_greedy_parity::SourceConfigEvidence {
        real_transformer_enabled: cfg.real_transformer.enabled,
        gpu_native_enabled: cfg.real_transformer.gpu_native,
        weights_dir_configured: cfg.real_transformer.weights_dir.is_some(),
        strict_weights: cfg.real_transformer.strict_weights,
        allow_seeded_fallback: cfg.real_transformer.allow_seeded_fallback,
        allow_degraded_experts: cfg.real_transformer.allow_degraded_experts,
        allow_nonfinite_attention_fallback: cfg.real_transformer.allow_nonfinite_attention_fallback,
        allow_truncated_expert_payloads: cfg.real_transformer.allow_truncated_expert_payloads,
        distributed_enabled: cfg.distributed.enabled,
        gpu_cache_enabled: cfg.gpu_cache.enabled,
        gpu_expert_capacity_bytes: capacity_bytes,
        routed_expert_dtype: cfg.model.dtype.as_str().to_string(),
    };
    if !source_config.is_strict() {
        return Err(
            "f32 reference-boundary audit requires strict GPU-native Q4 source configuration"
                .into(),
        );
    }
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                format!("f32 boundary audit expert metadata preflight failed: {error}")
            })?;
    if expert_metadata.dtype.as_deref() != Some("q4_0")
        || expert_metadata.q4_0_layout.as_deref() != Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
        || expert_metadata.explicitly_synthetic
    {
        return Err(
            "f32 reference-boundary audit requires canonical nonsynthetic Q4_0 metadata".into(),
        );
    }

    let mut cpu_spec =
        crate::resolve_real_cli_spec_from_config(cfg, RealCliRuntimeMode::IsolatedGreedyParityCpu)?;
    let model_identity = crate::greedy_parity_model_identity(&cpu_spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err(
            "f32 reference-boundary audit requires exact Qwen3-Coder 30B-A3B Q4_0 geometry".into(),
        );
    }
    cpu_spec.cfg.real_transformer.gpu_native = false;
    cpu_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Cpu;
    let cpu_resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&cpu_spec)?;
    let tokenizer_path = cpu_spec
        .cfg
        .tokenizer
        .path
        .as_ref()
        .ok_or("f32 reference-boundary audit requires tokenizer.path")?;
    let tokenizer = Arc::new(crate::tokenizer::Tokenizer::from_file(tokenizer_path)?);

    let capture = execute_cpu_audit(
        &cpu_spec,
        tokenizer,
        &cpu_resolved_config_sha256,
        &frozen,
        progress_watchdog,
    )
    .await?;
    let (q1_q4, r1_r5) = select_worst_experts(&capture.expert_records)?;
    let part_a_aggregation = aggregate_part_a(&capture.expert_records);
    let part_b_token_summary = PartBTokenSummary {
        requested_tokens: capture
            .token_cases
            .iter()
            .map(|case| case.requested_tokens)
            .sum(),
        completed_tokens: capture
            .token_cases
            .iter()
            .map(|case| case.completed_tokens)
            .sum(),
        exact_token_matches: capture
            .token_cases
            .iter()
            .map(|case| case.exact_token_matches)
            .sum(),
        mismatch_count: capture
            .token_cases
            .iter()
            .map(|case| case.mismatch_count)
            .sum(),
    };
    let (_, executable_sha256) = crate::current_executable_identity()?;
    let scientific_interpretation = vec![
        format!(
            "exact-F32 CPU reference has smaller max-absolute error than the old f16-emulated reference on {}/72 frozen experts",
            part_a_aggregation.exact_f32_smaller_than_old_f16_maxabs_count
        ),
        format!(
            "corrected ordinary CPU token stream matches the frozen GPU stream at {}/64 positions",
            part_b_token_summary.exact_token_matches
        ),
        "observations are diagnostic-only and do not establish GPU correctness or justify a production correction".to_string(),
    ];
    let report = F32ReferenceBoundaryAuditReport {
        schema: SCHEMA_VERSION,
        mode: MODE,
        diagnostic_only: true,
        diagnostic_complete: true,
        qualification_pass: QUALIFICATION_PASS,
        failure: None,
        frozen_inputs: frozen.identity,
        provenance: AuditProvenance {
            build,
            executable_sha256,
            artifacts,
            source_config,
            cpu_resolved_config_sha256,
            model_identity,
            model_load: capture.model_load,
            expert_metadata,
        },
        expert_records: capture.expert_records,
        q1_q4_frozen_worst_experts: q1_q4,
        r1_r5_worst_old_boundary_error_experts: r1_r5,
        part_a_aggregation,
        corrected_cpu_token_cases: capture.token_cases,
        rust_ownership_position_one: capture.rust_ownership_position_one,
        part_b_token_summary,
        runtime: capture.runtime,
        production_semantics: ProductionSemanticsEvidence::default(),
        scientific_interpretation,
    };
    emit_report(&report, &report_out)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    fn valid_v2_envelope() -> FrozenV2Envelope {
        let limits = crate::gpu_native_semantic_parity_v2::NumericalLimits::frozen();
        let token_cases = crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS
            .into_iter()
            .map(|case| {
                let mut reference =
                    vec![11u32; crate::gpu_native_semantic_parity_v2::OUTPUT_TOKEN_LIMIT];
                let mut gpu = reference.clone();
                if case.name == "rust-ownership-holdout" {
                    reference[1] = 785;
                    gpu[1] = 8822;
                }
                let comparison = crate::gpu_native_semantic_parity_corpus::TokenCaseEvidence::new(
                    case.name,
                    reference.clone(),
                    gpu.clone(),
                );
                FrozenTokenCaseEnvelope {
                    case: case.name.to_string(),
                    reference_generated_token_ids: reference,
                    gpu_generated_token_ids: gpu,
                    exact_match_count: comparison.exact_match_count,
                    mismatch_count: comparison.mismatch_count,
                    first_mismatch_position: comparison.first_mismatch_position,
                }
            })
            .collect();
        FrozenV2Envelope {
            schema: crate::gpu_native_semantic_parity_v2::SCHEMA_VERSION.to_string(),
            qualification_pass: false,
            holdout_corpus: FrozenCorpusEnvelope {
                id: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_ID.to_string(),
                version: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_VERSION,
                sha256: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_SHA256.to_string(),
                case_count: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_CASE_COUNT,
                output_token_limit: crate::gpu_native_semantic_parity_v2::OUTPUT_TOKEN_LIMIT,
            },
            numerical_limits: FrozenNumericalLimitsEnvelope {
                max_absolute_error_limit: limits.max_absolute_error_limit,
                rms_error_limit: limits.rms_error_limit,
                mean_absolute_error_limit: limits.mean_absolute_error_limit,
                nonfinite_mismatch_limit: limits.nonfinite_mismatch_limit,
                semantic_correctness_not_bit_parity: limits.semantic_correctness_not_bit_parity,
            },
            provenance: FrozenProvenanceEnvelope {
                build: FrozenBuildEnvelope {
                    git_sha: Some(FROZEN_V2_BUILD_SHA.to_string()),
                },
            },
            token_cases,
        }
    }

    fn exact_vector(source: &str) -> FrozenExactVectorEnvelope {
        let f32_bits = vec![0u32; EXPECTED_D_MODEL];
        FrozenExactVectorEnvelope {
            source: source.to_string(),
            vector_length: f32_bits.len(),
            f32_bits_sha256: crate::numerical_diagnostics::f32_bits_sha256(&f32_bits),
            f32_bits,
            nonfinite_count: 0,
        }
    }

    fn valid_stage_envelope() -> FrozenStageEnvelope {
        let targets = expected_targets()
            .into_iter()
            .map(|target| {
                let mut local_ids = target
                    .frozen_worst_local_expert
                    .into_iter()
                    .collect::<Vec<_>>();
                for candidate in 0..EXPECTED_NUM_EXPERTS as u32 {
                    if local_ids.len() == EXPECTED_EXPERTS_PER_TARGET {
                        break;
                    }
                    if !local_ids.contains(&candidate) {
                        local_ids.push(candidate);
                    }
                }
                let experts = local_ids
                    .into_iter()
                    .enumerate()
                    .map(|(index, local_expert_id)| FrozenStageExpertEnvelope {
                        rank: index + 1,
                        local_expert_id,
                        global_expert_id: (target.layer * EXPECTED_NUM_EXPERTS) as u32
                            + local_expert_id,
                        canonical_q4_payload_sha256: "a".repeat(64),
                        final_output_boundary: FrozenOutputBoundaryEnvelope {
                            cpu_effective_output: exact_vector(
                                "cpu-diagnostic-f16-boundary-emulated-expert-output",
                            ),
                            actual_gpu_effective_output: exact_vector(
                                "actual-production-gpu-expert-output",
                            ),
                        },
                    })
                    .collect();
                FrozenStageTargetEnvelope {
                    target,
                    input_boundary: Some(FrozenInputBoundaryEnvelope {
                        effective_gpu_expert_input: exact_vector(
                            "effective-production-gpu-expert-input",
                        ),
                    }),
                    experts,
                    failure: None,
                }
            })
            .collect::<Vec<_>>();
        FrozenStageEnvelope {
            schema: crate::gpu_native_q4_expert_stage_attribution::SCHEMA_VERSION.to_string(),
            diagnostic_complete: true,
            qualification_pass: false,
            failure: None,
            provenance: FrozenProvenanceEnvelope {
                build: FrozenBuildEnvelope {
                    git_sha: Some(FROZEN_STAGE_BUILD_SHA.to_string()),
                },
            },
            frozen_targets: expected_targets(),
            targets,
        }
    }

    #[test]
    fn schema_mode_and_qualification_contract_are_versioned_and_false() {
        assert_eq!(
            SCHEMA_VERSION,
            "mer.gpu-native-f32-reference-boundary-audit.v1"
        );
        assert_eq!(MODE, "diagnose-gpu-native-f32-reference-boundary");
        assert!(!QUALIFICATION_PASS);
        let semantics = ProductionSemanticsEvidence::default();
        assert!(!semantics.numerical_qualification_threshold_introduced);
        assert!(!semantics.production_correction_made_or_justified);
    }

    #[test]
    fn exact_f32_reconstruction_preserves_every_u32_bit_pattern() {
        let bits = [0x0000_0000, 0x8000_0000, 0x3f80_0001, 0xbf7f_ffff];
        assert_eq!(
            reconstruct_exact_f32(&bits)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
    }

    #[test]
    fn old_path_round_trips_both_boundaries_and_exact_path_round_trips_neither() {
        let round_trips = RefCell::new(Vec::<Vec<u32>>::new());
        let production_inputs = RefCell::new(Vec::<Vec<u32>>::new());
        let emulator_inputs = RefCell::new(Vec::<Vec<u32>>::new());
        let input = [1.25f32, -2.5];
        let out = compute_reference_paths_with(
            &input,
            |values| {
                round_trips
                    .borrow_mut()
                    .push(values.iter().map(|value| value.to_bits()).collect());
                Ok(values.iter().map(|value| value + 1.0).collect())
            },
            |values| {
                production_inputs
                    .borrow_mut()
                    .push(values.iter().map(|value| value.to_bits()).collect());
                Ok(values.iter().map(|value| value * 2.0).collect())
            },
            |values| {
                emulator_inputs
                    .borrow_mut()
                    .push(values.iter().map(|value| value.to_bits()).collect());
                Ok(values.to_vec())
            },
        )
        .unwrap();
        assert_eq!(round_trips.borrow().len(), 2);
        assert_eq!(production_inputs.borrow().len(), 2);
        assert_eq!(
            production_inputs.borrow()[0],
            out.old_boundary_input
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            production_inputs.borrow()[1],
            input.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            emulator_inputs.borrow()[0],
            input.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            CPU_REFERENCE_IMPLEMENTATION,
            "existing-production-q4_0_cpu_reference_forward-candle"
        );
    }

    #[test]
    fn exact_path_matches_the_existing_production_candle_q4_reference() {
        let d_model = 32;
        let d_ff = 32;
        let payload = vec![
            0u8;
            crate::inference::expert_weight_bytes_for(
                d_model,
                d_ff,
                crate::inference::WeightDtype::Q4_0,
            )
        ];
        let input = (0..d_model)
            .map(|index| index as f32 / d_model as f32)
            .collect::<Vec<_>>();
        let expected =
            crate::inference::q4_0_cpu_reference_forward(&payload, &input, d_model, d_ff).unwrap();
        let observed =
            audit_q4_reference_paths(&payload, &input, d_model, d_ff, "b".repeat(64)).unwrap();
        assert!(observed
            .exact_f32_output
            .iter()
            .map(|value| value.to_bits())
            .eq(expected.iter().map(|value| value.to_bits())));
    }

    #[test]
    fn boundary_snapshot_must_remain_disabled_and_zero() {
        assert!(boundary_disabled_and_clean(
            crate::engine::CpuQ4BoundaryEmulationSnapshot::default()
        )
        .is_ok());
        assert!(
            boundary_disabled_and_clean(crate::engine::CpuQ4BoundaryEmulationSnapshot {
                enabled: true,
                routed_expert_dispatches: 0,
            })
            .is_err()
        );
        assert!(
            boundary_disabled_and_clean(crate::engine::CpuQ4BoundaryEmulationSnapshot {
                enabled: false,
                routed_expert_dispatches: 1,
            })
            .is_err()
        );
    }

    #[test]
    fn exact_72_expert_identity_contract_is_explicit() {
        assert_eq!(EXPECTED_TARGET_COUNT, 9);
        assert_eq!(EXPECTED_EXPERTS_PER_TARGET, 8);
        assert_eq!(EXPECTED_EXPERT_RECORD_COUNT, 72);
        assert_eq!(expected_targets().len(), EXPECTED_TARGET_COUNT);
        assert_eq!(
            expected_targets()
                .iter()
                .map(|target| target.id.as_str())
                .collect::<Vec<_>>(),
            ["Q1", "Q2", "Q3", "Q4", "R1", "R2", "R3", "R4", "R5"]
        );
    }

    #[test]
    fn frozen_sha_arguments_are_exact_and_fail_closed() {
        assert!(validate_expected_report_sha(
            FROZEN_V2_REPORT_SHA256,
            FROZEN_V2_REPORT_SHA256,
            "v2"
        )
        .is_ok());
        assert!(
            validate_expected_report_sha(&"0".repeat(64), FROZEN_V2_REPORT_SHA256, "v2").is_err()
        );
        assert!(validate_expected_report_sha(
            FROZEN_STAGE_REPORT_SHA256,
            FROZEN_STAGE_REPORT_SHA256,
            "stage"
        )
        .is_ok());
    }

    #[test]
    fn frozen_v2_schema_build_corpus_limits_and_tokens_fail_closed() {
        let mut envelope = valid_v2_envelope();
        assert!(validate_frozen_v2(&envelope).is_ok());
        envelope.schema.push_str("-drift");
        assert!(validate_frozen_v2(&envelope).is_err());
        envelope = valid_v2_envelope();
        envelope.provenance.build.git_sha = Some("0".repeat(40));
        assert!(validate_frozen_v2(&envelope).is_err());
        envelope = valid_v2_envelope();
        envelope.holdout_corpus.sha256 = "0".repeat(64);
        assert!(validate_frozen_v2(&envelope).is_err());
        envelope = valid_v2_envelope();
        envelope.numerical_limits.rms_error_limit = 1.0;
        assert!(validate_frozen_v2(&envelope).is_err());
        envelope = valid_v2_envelope();
        envelope.token_cases[0].gpu_generated_token_ids.pop();
        assert!(validate_frozen_v2(&envelope).is_err());
    }

    #[test]
    fn frozen_stage_schema_build_completion_and_all_72_identities_fail_closed() {
        let mut envelope = valid_stage_envelope();
        assert_eq!(validate_frozen_stage(&envelope).unwrap().len(), 72);
        envelope.targets[0].experts.pop();
        assert!(validate_frozen_stage(&envelope).is_err());
        envelope = valid_stage_envelope();
        envelope.provenance.build.git_sha = Some("0".repeat(40));
        assert!(validate_frozen_stage(&envelope).is_err());
        envelope = valid_stage_envelope();
        envelope.diagnostic_complete = false;
        assert!(validate_frozen_stage(&envelope).is_err());
        envelope = valid_stage_envelope();
        envelope.targets[0].target.generated_position += 1;
        assert!(validate_frozen_stage(&envelope).is_err());
        envelope = valid_stage_envelope();
        envelope.targets[0].experts[0]
            .final_output_boundary
            .actual_gpu_effective_output
            .f32_bits_sha256 = "0".repeat(64);
        assert!(validate_frozen_stage(&envelope).is_err());
    }

    #[test]
    fn snapshot_hash_uses_the_same_bytes_and_rejects_drift() {
        let bytes = b"immutable-report-snapshot";
        let sha = crate::greedy_parity::sha256_hex(bytes);
        let artifact = crate::qualification::ArtifactDigest {
            configured_path: "frozen.json".into(),
            canonical_path: "/frozen.json".into(),
            byte_length: bytes.len() as u64,
            sha256: sha.clone(),
        };
        assert!(validate_snapshot_hash(&artifact, &sha, &sha, "test").is_ok());
        assert!(validate_snapshot_hash(&artifact, &"0".repeat(64), &sha, "test").is_err());
    }

    #[test]
    fn runtime_and_historical_contracts_prohibit_gpu_and_overwrite() {
        assert_eq!(
            AUDIT_RUNTIME_MODE,
            RealCliRuntimeMode::IsolatedGreedyParityCpu
        );
        let historical = FrozenHistoricalEvidenceIdentity::default();
        assert!(!historical.modified_or_reclassified);
        let semantics = ProductionSemanticsEvidence::default();
        assert!(semantics.frozen_gpu_outputs_read_only_from_stage_report);
        assert!(semantics.frozen_gpu_tokens_read_only_from_v2_report);
        assert!(!semantics.cpu_q4_boundary_emulation_behavior_changed);
    }

    #[test]
    fn report_destination_cannot_overwrite_frozen_inputs() {
        let directory = std::env::temp_dir().join(format!(
            "mer-f32-boundary-audit-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let v2_path = directory.join("v2.json");
        let stage_path = directory.join("stage.json");
        std::fs::write(&v2_path, b"v2").unwrap();
        std::fs::write(&stage_path, b"stage").unwrap();
        let artifact = |path: &Path| crate::qualification::ArtifactDigest {
            configured_path: path.display().to_string(),
            canonical_path: std::fs::canonicalize(path).unwrap().display().to_string(),
            byte_length: std::fs::metadata(path).unwrap().len(),
            sha256: "a".repeat(64),
        };
        let identity = FrozenReportsIdentity {
            v2_report: artifact(&v2_path),
            expected_v2_report_sha256_argument: FROZEN_V2_REPORT_SHA256.into(),
            stage_report: artifact(&stage_path),
            expected_stage_report_sha256_argument: FROZEN_STAGE_REPORT_SHA256.into(),
            v2_schema: crate::gpu_native_semantic_parity_v2::SCHEMA_VERSION,
            stage_schema: crate::gpu_native_q4_expert_stage_attribution::SCHEMA_VERSION,
            v2_build_sha: FROZEN_V2_BUILD_SHA,
            stage_build_sha: FROZEN_STAGE_BUILD_SHA,
            holdout_corpus_id: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_ID,
            holdout_corpus_sha256: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_SHA256,
            holdout_case_count: 4,
            output_token_limit: 16,
            exact_expert_identity_count: 72,
            exact_expert_identities: Vec::new(),
            immutable_inputs_verified_before_runtime: true,
            historical: FrozenHistoricalEvidenceIdentity::default(),
        };
        assert!(output_path_conflicts_with_frozen_input(&v2_path, &identity).unwrap());
        assert!(output_path_conflicts_with_frozen_input(&stage_path, &identity).unwrap());
        assert!(!output_path_conflicts_with_frozen_input(
            &directory.join("new/nested/audit.json"),
            &identity,
        )
        .unwrap());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn improvement_is_direct_and_has_no_acceptance_threshold() {
        let old = VectorNumericalEvidence::compare("old", "gpu", &[0.0], &[2.0]).unwrap();
        let exact = VectorNumericalEvidence::compare("exact", "gpu", &[1.0], &[2.0]).unwrap();
        let evidence = ErrorImprovementEvidence::new(&exact, &old);
        assert!(evidence.exact_f32_error_smaller_than_old_f16_error_maxabs);
        assert_eq!(evidence.exact_f32_to_old_f16_maxabs_ratio, Some(0.5));
        assert!(
            !ProductionSemanticsEvidence::default().numerical_qualification_threshold_introduced
        );
    }
}
