//! Diagnostic-only three-plane attribution of the frozen Spanish first-token
//! divergence. This module consumes three immutable historical reports, then
//! runs one caller-tokenized prompt through isolated exact-f32 CPU,
//! historical-f16 CPU, and production GPU-native runtimes.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::gpu_native_expert_permutation_semantic_parity::VectorNumericalEvidence;
use crate::gpu_native_router_rank_diagnostics::{
    ActualGpuRouterEvidence, RouterEvaluationEvidence,
};
use crate::numerical_diagnostics::FloatEvidence;
use crate::{IsolatedRuntimeShutdownError, RealCliRuntimeMode, ResolvedRealCliSpec};

pub const SCHEMA_VERSION: &str = "mer.gpu-native-spanish-first-token-attribution.v1";
pub const MODE: &str = "diagnose-gpu-native-spanish-first-token-attribution";
pub const QUALIFICATION_PASS: bool = false;

pub const FROZEN_V2_REPORT_SHA256: &str =
    "9629de35a18f4457bbe38aeeaeb208fd979bb08e9d2eccb283437d6c16a5c4ad";
pub const FROZEN_STAGE_REPORT_SHA256: &str =
    "f81827b39d2ece62b9c1af432f71ca70045fe41c13e6e6ac15b03d460246365d";
pub const FROZEN_BOUNDARY_AUDIT_REPORT_SHA256: &str =
    "8bd568eb41510eae5541bd0829e939e4426cfdf04548677b026a28021c4cf3f1";
pub const FROZEN_BOUNDARY_AUDIT_LOG_SHA256: &str =
    "66bbcb9d3d6ad96de1251545b6fce32a556cf7734d52719854a74d462edea5b1";
pub const FROZEN_BOUNDARY_AUDIT_SUMMARY_SHA256: &str =
    "b88c6c1733185be6034b737355eb402e5c791376603b4b0d525eb66ac2008856";
pub const FROZEN_V2_BUILD_SHA: &str = "300448ac3da48ac74ef86fcd2c62ffca27ffa634";
pub const FROZEN_STAGE_BUILD_SHA: &str = "987e7e4b8791dc70f75b62a772e89ad9f1948ec5";
pub const FROZEN_BOUNDARY_AUDIT_BUILD_SHA: &str = "7d54f6671c71a6c1686a5fccd70059e24314460d";

pub const TARGET_CASE: &str = "spanish-refactor-holdout";
pub const TARGET_GENERATED_POSITION: usize = 0;
pub const HISTORICAL_F16_CPU_TOKEN: u32 = 54_275;
pub const FROZEN_GPU_TOKEN: u32 = 54_275;
pub const EXACT_F32_CPU_TOKEN: u32 = 140_003;
pub const REQUIRED_ADAPTER_NAME: &str = "NVIDIA L4";

const EXPECTED_NUM_LAYERS: usize = 48;
const EXPECTED_NUM_EXPERTS: usize = 128;
const EXPECTED_TOP_K: usize = 8;
const EXPECTED_D_MODEL: usize = 2_048;
const EXPECTED_D_FF: usize = 768;

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenBuildEnvelope {
    git_sha: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenProvenanceEnvelope {
    build: FrozenBuildEnvelope,
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

#[derive(Clone, Copy, Debug, Deserialize)]
struct FrozenNumericalLimitsEnvelope {
    max_absolute_error_limit: f64,
    rms_error_limit: f64,
    mean_absolute_error_limit: f64,
    nonfinite_mismatch_limit: usize,
    semantic_correctness_not_bit_parity: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenV2Envelope {
    schema: String,
    qualification_pass: bool,
    provenance: FrozenProvenanceEnvelope,
    holdout_corpus: FrozenCorpusEnvelope,
    numerical_limits: FrozenNumericalLimitsEnvelope,
    token_cases: Vec<FrozenTokenCaseEnvelope>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize)]
struct FrozenStageTargetIdentity {
    id: String,
    case: String,
    generated_position: usize,
    layer: usize,
    frozen_worst_local_expert: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenStageTargetEnvelope {
    target: FrozenStageTargetIdentity,
    experts: Vec<serde_json::Value>,
    failure: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenStageEnvelope {
    schema: String,
    diagnostic_complete: bool,
    qualification_pass: bool,
    failure: Option<String>,
    provenance: FrozenProvenanceEnvelope,
    frozen_targets: Vec<FrozenStageTargetIdentity>,
    targets: Vec<FrozenStageTargetEnvelope>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct FrozenImprovementEnvelope {
    exact_f32_error_smaller_than_old_f16_error_maxabs: bool,
    exact_f32_error_smaller_than_old_f16_error_rms: bool,
    exact_f32_error_smaller_than_old_f16_error_meanabs: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct FrozenBoundaryExpertEnvelope {
    improvement: FrozenImprovementEnvelope,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct FrozenPartAAggregationEnvelope {
    expert_record_count: usize,
    exact_f32_smaller_than_old_f16_maxabs_count: usize,
    exact_f32_smaller_than_old_f16_rms_count: usize,
    exact_f32_smaller_than_old_f16_meanabs_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenCpuTokenCaseEnvelope {
    case: String,
    requested_tokens: usize,
    completed_tokens: usize,
    exact_token_matches: usize,
    mismatch_count: usize,
    first_mismatch_position: Option<usize>,
    corrected_ordinary_f32_cpu_token_ids: Vec<u32>,
    frozen_gpu_token_ids: Vec<u32>,
    historical_old_boundary_cpu_reference_token_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct FrozenRustPositionOneEnvelope {
    historical_old_cpu_token_id: u32,
    frozen_gpu_token_id: u32,
    new_ordinary_f32_cpu_token_id: u32,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct FrozenPartBEnvelope {
    requested_tokens: usize,
    completed_tokens: usize,
    exact_token_matches: usize,
    mismatch_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenArtifactEnvelope {
    sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenBoundaryInputsEnvelope {
    v2_report: FrozenArtifactEnvelope,
    stage_report: FrozenArtifactEnvelope,
    exact_expert_identity_count: usize,
    immutable_inputs_verified_before_runtime: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct FrozenProductionSemanticsEnvelope {
    diagnostic_only: bool,
    production_inference_changed: bool,
    production_q4_changed: bool,
    production_q4_wgsl_changed: bool,
    production_router_changed: bool,
    production_attention_changed: bool,
    v1_changed: bool,
    v2_changed: bool,
    v2_limits_changed: bool,
    v2_corpus_or_prompts_changed: bool,
    production_correction_made_or_justified: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct FrozenBoundaryEnvelope {
    schema: String,
    diagnostic_only: bool,
    diagnostic_complete: bool,
    qualification_pass: bool,
    failure: Option<String>,
    provenance: FrozenProvenanceEnvelope,
    frozen_inputs: FrozenBoundaryInputsEnvelope,
    expert_records: Vec<FrozenBoundaryExpertEnvelope>,
    part_a_aggregation: FrozenPartAAggregationEnvelope,
    corrected_cpu_token_cases: Vec<FrozenCpuTokenCaseEnvelope>,
    rust_ownership_position_one: FrozenRustPositionOneEnvelope,
    part_b_token_summary: FrozenPartBEnvelope,
    production_semantics: FrozenProductionSemanticsEnvelope,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrozenSpanishTargetIdentity {
    pub case: &'static str,
    pub generated_position: usize,
    pub historical_f16_boundary_cpu_token: u32,
    pub frozen_gpu_token: u32,
    pub corrected_exact_f32_cpu_token: u32,
    pub later_generated_positions_in_scope: bool,
}

impl Default for FrozenSpanishTargetIdentity {
    fn default() -> Self {
        Self {
            case: TARGET_CASE,
            generated_position: TARGET_GENERATED_POSITION,
            historical_f16_boundary_cpu_token: HISTORICAL_F16_CPU_TOKEN,
            frozen_gpu_token: FROZEN_GPU_TOKEN,
            corrected_exact_f32_cpu_token: EXACT_F32_CPU_TOKEN,
            later_generated_positions_in_scope: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrozenEvidencePreflight {
    pub v2_report: crate::qualification::ArtifactDigest,
    pub expected_v2_report_sha256_argument: String,
    pub stage_report: crate::qualification::ArtifactDigest,
    pub expected_stage_report_sha256_argument: String,
    pub boundary_audit_report: crate::qualification::ArtifactDigest,
    pub expected_boundary_audit_report_sha256_argument: String,
    pub boundary_audit_log_sha256: &'static str,
    pub boundary_audit_summary_sha256: &'static str,
    pub reports_read_once: bool,
    pub hash_and_deserialization_used_same_snapshots: bool,
    pub verified_before_runtime_construction: bool,
    pub exact_72_of_72_improvement_all_three_metrics: bool,
    pub rust_corrected_position_one_is_8822: bool,
    pub rust_corrected_stream_matches_gpu_16_of_16: bool,
    pub spanish_corrected_stream_mismatch_count: usize,
    pub spanish_first_mismatch_position: usize,
}

struct FrozenInputs {
    identity: FrozenEvidencePreflight,
}

fn validate_expected_sha(value: &str, frozen: &str, label: &str) -> Result<(), String> {
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

fn validate_artifact_hash(
    artifact: &crate::qualification::ArtifactDigest,
    argument: &str,
    frozen: &str,
    label: &str,
) -> Result<(), String> {
    if !artifact.sha256.eq_ignore_ascii_case(argument)
        || !artifact.sha256.eq_ignore_ascii_case(frozen)
    {
        return Err(format!(
            "frozen {label} SHA differs: observed {} expected {frozen}",
            artifact.sha256
        ));
    }
    Ok(())
}

fn validate_v2(envelope: &FrozenV2Envelope) -> Result<(), String> {
    use crate::gpu_native_semantic_parity_v2 as v2;
    if envelope.schema != v2::SCHEMA_VERSION
        || envelope.qualification_pass
        || envelope.provenance.build.git_sha.as_deref() != Some(FROZEN_V2_BUILD_SHA)
    {
        return Err("frozen v2 schema/build/qualification identity differs".into());
    }
    let corpus = &envelope.holdout_corpus;
    if corpus.id != v2::HOLDOUT_CORPUS_ID
        || corpus.version != v2::HOLDOUT_CORPUS_VERSION
        || corpus.sha256 != v2::HOLDOUT_CORPUS_SHA256
        || corpus.case_count != v2::HOLDOUT_CORPUS_CASE_COUNT
        || corpus.output_token_limit != v2::OUTPUT_TOKEN_LIMIT
        || envelope.token_cases.len() != v2::HOLDOUT_CORPUS_CASE_COUNT
    {
        return Err("frozen v2 corpus identity differs".into());
    }
    let limits = envelope.numerical_limits;
    if limits.max_absolute_error_limit.to_bits() != v2::MAX_ABSOLUTE_ERROR_LIMIT.to_bits()
        || limits.rms_error_limit.to_bits() != v2::RMS_ERROR_LIMIT.to_bits()
        || limits.mean_absolute_error_limit.to_bits() != v2::MEAN_ABSOLUTE_ERROR_LIMIT.to_bits()
        || limits.nonfinite_mismatch_limit != v2::NONFINITE_MISMATCH_LIMIT
        || !limits.semantic_correctness_not_bit_parity
    {
        return Err("frozen v2 numerical limits differ".into());
    }
    for (observed, expected) in envelope.token_cases.iter().zip(v2::HOLDOUT_CORPUS) {
        if observed.case != expected.name
            || observed.reference_generated_token_ids.len() != v2::OUTPUT_TOKEN_LIMIT
            || observed.gpu_generated_token_ids.len() != v2::OUTPUT_TOKEN_LIMIT
        {
            return Err(format!("frozen v2 case {} is incomplete", expected.name));
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
                "frozen v2 comparison drifted for {}",
                expected.name
            ));
        }
    }
    let spanish = envelope
        .token_cases
        .iter()
        .find(|case| case.case == TARGET_CASE)
        .ok_or("frozen v2 Spanish case is missing")?;
    if spanish.reference_generated_token_ids.first() != Some(&HISTORICAL_F16_CPU_TOKEN)
        || spanish.gpu_generated_token_ids.first() != Some(&FROZEN_GPU_TOKEN)
    {
        return Err("frozen v2 Spanish first-token identity differs".into());
    }
    Ok(())
}

fn expected_stage_targets() -> Vec<FrozenStageTargetIdentity> {
    crate::gpu_native_q4_expert_stage_attribution::FROZEN_TARGETS
        .into_iter()
        .map(|target| FrozenStageTargetIdentity {
            id: target.id.to_string(),
            case: target.case.to_string(),
            generated_position: target.generated_position,
            layer: target.layer,
            frozen_worst_local_expert: target.frozen_worst_local_expert,
        })
        .collect()
}

fn validate_stage(envelope: &FrozenStageEnvelope) -> Result<(), String> {
    let expected = expected_stage_targets();
    if envelope.schema != crate::gpu_native_q4_expert_stage_attribution::SCHEMA_VERSION
        || !envelope.diagnostic_complete
        || envelope.qualification_pass
        || envelope.failure.is_some()
        || envelope.provenance.build.git_sha.as_deref() != Some(FROZEN_STAGE_BUILD_SHA)
        || envelope.frozen_targets != expected
        || envelope.targets.len() != expected.len()
    {
        return Err("frozen stage report top-level identity differs".into());
    }
    let mut records = 0usize;
    for (observed, expected) in envelope.targets.iter().zip(expected) {
        if observed.target != expected || observed.failure.is_some() || observed.experts.len() != 8
        {
            return Err(format!("frozen stage target {} differs", expected.id));
        }
        records += observed.experts.len();
    }
    if records != 72 {
        return Err("frozen stage report does not contain exactly 72 expert records".into());
    }
    Ok(())
}

fn validate_boundary(envelope: &FrozenBoundaryEnvelope) -> Result<(), String> {
    if envelope.schema != crate::gpu_native_f32_reference_boundary_audit::SCHEMA_VERSION
        || !envelope.diagnostic_only
        || !envelope.diagnostic_complete
        || envelope.qualification_pass
        || envelope.failure.is_some()
        || envelope.provenance.build.git_sha.as_deref() != Some(FROZEN_BOUNDARY_AUDIT_BUILD_SHA)
    {
        return Err("frozen boundary-audit top-level identity differs".into());
    }
    if !envelope
        .frozen_inputs
        .v2_report
        .sha256
        .eq_ignore_ascii_case(FROZEN_V2_REPORT_SHA256)
        || !envelope
            .frozen_inputs
            .stage_report
            .sha256
            .eq_ignore_ascii_case(FROZEN_STAGE_REPORT_SHA256)
        || envelope.frozen_inputs.exact_expert_identity_count != 72
        || !envelope
            .frozen_inputs
            .immutable_inputs_verified_before_runtime
    {
        return Err("frozen boundary-audit input identity differs".into());
    }
    let aggregation = envelope.part_a_aggregation;
    if envelope.expert_records.len() != 72
        || aggregation.expert_record_count != 72
        || aggregation.exact_f32_smaller_than_old_f16_maxabs_count != 72
        || aggregation.exact_f32_smaller_than_old_f16_rms_count != 72
        || aggregation.exact_f32_smaller_than_old_f16_meanabs_count != 72
        || envelope.expert_records.iter().any(|record| {
            !record
                .improvement
                .exact_f32_error_smaller_than_old_f16_error_maxabs
                || !record
                    .improvement
                    .exact_f32_error_smaller_than_old_f16_error_rms
                || !record
                    .improvement
                    .exact_f32_error_smaller_than_old_f16_error_meanabs
        })
    {
        return Err("frozen boundary-audit 72/72 improvement identity differs".into());
    }
    if envelope.corrected_cpu_token_cases.len() != 4 {
        return Err("frozen boundary-audit token-case count differs".into());
    }
    for (case, expected) in envelope
        .corrected_cpu_token_cases
        .iter()
        .zip(crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS)
    {
        if case.case != expected.name {
            return Err("frozen boundary-audit token-case order/identity differs".into());
        }
        if case.requested_tokens != 16
            || case.completed_tokens != 16
            || case.corrected_ordinary_f32_cpu_token_ids.len() != 16
            || case.frozen_gpu_token_ids.len() != 16
            || case.historical_old_boundary_cpu_reference_token_ids.len() != 16
        {
            return Err(format!(
                "frozen boundary-audit case {} is incomplete",
                case.case
            ));
        }
        let expected_exact = usize::from(case.case != TARGET_CASE) * 16;
        let expected_mismatch = usize::from(case.case == TARGET_CASE) * 16;
        let recomputed_exact = case
            .corrected_ordinary_f32_cpu_token_ids
            .iter()
            .zip(&case.frozen_gpu_token_ids)
            .filter(|(cpu, gpu)| cpu == gpu)
            .count();
        let recomputed_first_mismatch = case
            .corrected_ordinary_f32_cpu_token_ids
            .iter()
            .zip(&case.frozen_gpu_token_ids)
            .position(|(cpu, gpu)| cpu != gpu);
        if case.exact_token_matches != expected_exact
            || case.mismatch_count != expected_mismatch
            || case.exact_token_matches != recomputed_exact
            || case.mismatch_count != 16 - recomputed_exact
            || case.first_mismatch_position != recomputed_first_mismatch
        {
            return Err(format!(
                "frozen boundary-audit case {} counts differ",
                case.case
            ));
        }
    }
    let spanish = envelope
        .corrected_cpu_token_cases
        .iter()
        .find(|case| case.case == TARGET_CASE)
        .ok_or("frozen boundary-audit Spanish case is missing")?;
    if spanish.first_mismatch_position != Some(0)
        || spanish.corrected_ordinary_f32_cpu_token_ids.first() != Some(&EXACT_F32_CPU_TOKEN)
        || spanish.frozen_gpu_token_ids.first() != Some(&FROZEN_GPU_TOKEN)
        || spanish
            .historical_old_boundary_cpu_reference_token_ids
            .first()
            != Some(&HISTORICAL_F16_CPU_TOKEN)
    {
        return Err("frozen boundary-audit Spanish target differs".into());
    }
    let rust = envelope
        .corrected_cpu_token_cases
        .iter()
        .find(|case| case.case == "rust-ownership-holdout")
        .ok_or("frozen boundary-audit Rust case is missing")?;
    if rust.corrected_ordinary_f32_cpu_token_ids != rust.frozen_gpu_token_ids
        || rust.corrected_ordinary_f32_cpu_token_ids.get(1) != Some(&8_822)
        || envelope
            .rust_ownership_position_one
            .historical_old_cpu_token_id
            != 785
        || envelope.rust_ownership_position_one.frozen_gpu_token_id != 8_822
        || envelope
            .rust_ownership_position_one
            .new_ordinary_f32_cpu_token_id
            != 8_822
        || envelope.part_b_token_summary.requested_tokens != 64
        || envelope.part_b_token_summary.completed_tokens != 64
        || envelope.part_b_token_summary.exact_token_matches != 48
        || envelope.part_b_token_summary.mismatch_count != 16
    {
        return Err("frozen boundary-audit corrected Rust/summary identity differs".into());
    }
    let semantics = envelope.production_semantics;
    if !semantics.diagnostic_only
        || semantics.production_inference_changed
        || semantics.production_q4_changed
        || semantics.production_q4_wgsl_changed
        || semantics.production_router_changed
        || semantics.production_attention_changed
        || semantics.v1_changed
        || semantics.v2_changed
        || semantics.v2_limits_changed
        || semantics.v2_corpus_or_prompts_changed
        || semantics.production_correction_made_or_justified
    {
        return Err("frozen boundary-audit production-semantics identity differs".into());
    }
    Ok(())
}

impl FrozenInputs {
    #[allow(clippy::too_many_arguments)]
    fn read_and_validate(
        v2_path: &Path,
        expected_v2: &str,
        stage_path: &Path,
        expected_stage: &str,
        boundary_path: &Path,
        expected_boundary: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        validate_expected_sha(expected_v2, FROZEN_V2_REPORT_SHA256, "v2 report")?;
        validate_expected_sha(expected_stage, FROZEN_STAGE_REPORT_SHA256, "stage report")?;
        validate_expected_sha(
            expected_boundary,
            FROZEN_BOUNDARY_AUDIT_REPORT_SHA256,
            "boundary-audit report",
        )?;

        // Each supplied report is read exactly once. Hashing and typed
        // deserialization both consume these same immutable byte snapshots.
        let v2_bytes = std::fs::read(v2_path)?;
        let stage_bytes = std::fs::read(stage_path)?;
        let boundary_bytes = std::fs::read(boundary_path)?;
        let v2_artifact = artifact_from_snapshot(v2_path, &v2_bytes)?;
        let stage_artifact = artifact_from_snapshot(stage_path, &stage_bytes)?;
        let boundary_artifact = artifact_from_snapshot(boundary_path, &boundary_bytes)?;
        validate_artifact_hash(
            &v2_artifact,
            expected_v2,
            FROZEN_V2_REPORT_SHA256,
            "v2 report",
        )?;
        validate_artifact_hash(
            &stage_artifact,
            expected_stage,
            FROZEN_STAGE_REPORT_SHA256,
            "stage report",
        )?;
        validate_artifact_hash(
            &boundary_artifact,
            expected_boundary,
            FROZEN_BOUNDARY_AUDIT_REPORT_SHA256,
            "boundary-audit report",
        )?;
        let v2: FrozenV2Envelope = serde_json::from_slice(&v2_bytes)
            .map_err(|error| format!("malformed frozen v2 report: {error}"))?;
        let stage: FrozenStageEnvelope = serde_json::from_slice(&stage_bytes)
            .map_err(|error| format!("malformed frozen stage report: {error}"))?;
        let boundary: FrozenBoundaryEnvelope = serde_json::from_slice(&boundary_bytes)
            .map_err(|error| format!("malformed frozen boundary-audit report: {error}"))?;
        validate_v2(&v2)?;
        validate_stage(&stage)?;
        validate_boundary(&boundary)?;
        Ok(Self {
            identity: FrozenEvidencePreflight {
                v2_report: v2_artifact,
                expected_v2_report_sha256_argument: expected_v2.to_ascii_lowercase(),
                stage_report: stage_artifact,
                expected_stage_report_sha256_argument: expected_stage.to_ascii_lowercase(),
                boundary_audit_report: boundary_artifact,
                expected_boundary_audit_report_sha256_argument: expected_boundary
                    .to_ascii_lowercase(),
                boundary_audit_log_sha256: FROZEN_BOUNDARY_AUDIT_LOG_SHA256,
                boundary_audit_summary_sha256: FROZEN_BOUNDARY_AUDIT_SUMMARY_SHA256,
                reports_read_once: true,
                hash_and_deserialization_used_same_snapshots: true,
                verified_before_runtime_construction: true,
                exact_72_of_72_improvement_all_three_metrics: true,
                rust_corrected_position_one_is_8822: true,
                rust_corrected_stream_matches_gpu_16_of_16: true,
                spanish_corrected_stream_mismatch_count: 16,
                spanish_first_mismatch_position: 0,
            },
        })
    }
}

fn canonical_output_path(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if path.exists() {
        return Ok(std::fs::canonicalize(path)?);
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
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
    Ok(canonical)
}

fn output_conflicts_with_input(
    report_out: &Path,
    frozen: &FrozenEvidencePreflight,
) -> Result<bool, Box<dyn std::error::Error>> {
    let output = canonical_output_path(report_out)?;
    Ok([
        &frozen.v2_report.canonical_path,
        &frozen.stage_report.canonical_path,
        &frozen.boundary_audit_report.canonical_path,
    ]
    .into_iter()
    .any(|input| output == PathBuf::from(input)))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SharedTokenizationEvidence {
    pub case: &'static str,
    pub prompt_sha256: String,
    pub prompt_token_ids: Vec<u32>,
    pub prompt_token_ids_sha256: String,
    pub tokenized_exactly_once: bool,
    pub caller_owned_tokenization_shared_by_all_planes: bool,
    pub per_plane_retokenization_performed: bool,
}

fn shared_tokenization(
    tokenizer: &crate::tokenizer::Tokenizer,
) -> Result<(Arc<Vec<u32>>, SharedTokenizationEvidence), Box<dyn std::error::Error>> {
    let fixed = crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS
        .into_iter()
        .find(|case| case.name == TARGET_CASE)
        .ok_or("frozen Spanish holdout prompt is missing")?;
    let ids = Arc::new(tokenizer.encode(fixed.prompt)?);
    if ids.is_empty() {
        return Err("frozen Spanish prompt encoded to zero tokens".into());
    }
    let evidence = SharedTokenizationEvidence {
        case: TARGET_CASE,
        prompt_sha256: crate::greedy_parity::sha256_hex(fixed.prompt.as_bytes()),
        prompt_token_ids: ids.as_ref().clone(),
        prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&ids),
        tokenized_exactly_once: true,
        caller_owned_tokenization_shared_by_all_planes: true,
        per_plane_retokenization_performed: false,
    };
    Ok((ids, evidence))
}

fn shared_plane_token_ids(token_ids: &Arc<Vec<u32>>) -> [Arc<Vec<u32>>; 3] {
    [token_ids.clone(), token_ids.clone(), token_ids.clone()]
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExactVectorTraceEvidence {
    pub source: String,
    pub vector_length: usize,
    pub f32_bits_sha256: String,
    pub f32_bits: Vec<u32>,
    pub nonfinite_count: usize,
}

impl ExactVectorTraceEvidence {
    fn new(source: impl Into<String>, values: &[f32], retain_bits: bool) -> Self {
        let bits = values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        Self {
            source: source.into(),
            vector_length: values.len(),
            f32_bits_sha256: crate::numerical_diagnostics::f32_bits_sha256(&bits),
            f32_bits: retain_bits.then_some(bits).unwrap_or_default(),
            nonfinite_count: values.iter().filter(|value| !value.is_finite()).count(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FullTraceLayerEvidence {
    pub layer: usize,
    pub pre_attention_residual: ExactVectorTraceEvidence,
    pub post_attention: ExactVectorTraceEvidence,
    pub router_input: ExactVectorTraceEvidence,
    pub selected_expert_ids: Vec<u32>,
    pub selected_weights: ExactVectorTraceEvidence,
    pub routed_moe_output: ExactVectorTraceEvidence,
    pub post_moe_residual: ExactVectorTraceEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlaneFullTraceEvidence {
    pub plane: &'static str,
    pub embedding: ExactVectorTraceEvidence,
    pub layers: Vec<FullTraceLayerEvidence>,
    pub final_hidden: ExactVectorTraceEvidence,
    pub final_rmsnorm: ExactVectorTraceEvidence,
    pub full_lm_head_logits: ExactVectorTraceEvidence,
    pub greedy_argmax: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ThreeWayVectorComparison {
    pub exact_f32_cpu_vs_gpu: VectorNumericalEvidence,
    pub historical_f16_cpu_vs_gpu: VectorNumericalEvidence,
    pub exact_f32_cpu_vs_historical_f16_cpu: VectorNumericalEvidence,
}

fn three_way_compare(
    _label: &str,
    exact: &[f32],
    historical: &[f32],
    gpu: &[f32],
) -> Result<ThreeWayVectorComparison, String> {
    let exact_source = "EXACT_F32_CPU";
    let historical_source = "HISTORICAL_F16_CPU";
    let gpu_source = "GPU_NATIVE";
    Ok(ThreeWayVectorComparison {
        exact_f32_cpu_vs_gpu: VectorNumericalEvidence::compare(
            exact_source,
            gpu_source,
            exact,
            gpu,
        )?,
        historical_f16_cpu_vs_gpu: VectorNumericalEvidence::compare(
            historical_source,
            gpu_source,
            historical,
            gpu,
        )?,
        exact_f32_cpu_vs_historical_f16_cpu: VectorNumericalEvidence::compare(
            exact_source,
            historical_source,
            exact,
            historical,
        )?,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PrefillPositionSummary {
    pub position: usize,
    pub token_id: u32,
    pub final_hidden: ThreeWayVectorComparison,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ThreeWaySelectedIds {
    pub exact_f32_cpu: Vec<u32>,
    pub historical_f16_cpu: Vec<u32>,
    pub gpu_native: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ThreeWayLayerTraceComparison {
    pub layer: usize,
    pub pre_attention_residual: ThreeWayVectorComparison,
    pub post_attention: ThreeWayVectorComparison,
    pub router_input: ThreeWayVectorComparison,
    pub selected_ids: ThreeWaySelectedIds,
    pub selected_weights: ThreeWayVectorComparison,
    pub routed_moe_output: ThreeWayVectorComparison,
    pub post_moe_residual: ThreeWayVectorComparison,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FinalPositionTraceComparison {
    pub embedding: ThreeWayVectorComparison,
    pub layers: Vec<ThreeWayLayerTraceComparison>,
    pub final_hidden: ThreeWayVectorComparison,
    pub final_rmsnorm: ThreeWayVectorComparison,
    pub full_lm_head_logits: ThreeWayVectorComparison,
    pub greedy_argmax_exact_f32_cpu: u32,
    pub greedy_argmax_historical_f16_cpu: u32,
    pub greedy_argmax_gpu_native: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DistanceRelation {
    HistoricalF16Closer,
    ExactF32Closer,
    Equal,
    Unavailable,
}

fn distance_relation(historical: Option<f64>, exact: Option<f64>) -> DistanceRelation {
    match historical.zip(exact) {
        Some((historical, exact)) if historical.is_finite() && exact.is_finite() => {
            match historical.total_cmp(&exact) {
                std::cmp::Ordering::Less => DistanceRelation::HistoricalF16Closer,
                std::cmp::Ordering::Greater => DistanceRelation::ExactF32Closer,
                std::cmp::Ordering::Equal => DistanceRelation::Equal,
            }
        }
        _ => DistanceRelation::Unavailable,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompensationStageEvidence {
    pub stage: String,
    pub maxabs: DistanceRelation,
    pub rms: DistanceRelation,
    pub meanabs: DistanceRelation,
}

impl CompensationStageEvidence {
    fn from_comparison(stage: impl Into<String>, comparison: &ThreeWayVectorComparison) -> Self {
        Self {
            stage: stage.into(),
            maxabs: distance_relation(
                comparison.historical_f16_cpu_vs_gpu.max_absolute_error,
                comparison.exact_f32_cpu_vs_gpu.max_absolute_error,
            ),
            rms: distance_relation(
                comparison.historical_f16_cpu_vs_gpu.rms_error,
                comparison.exact_f32_cpu_vs_gpu.rms_error,
            ),
            meanabs: distance_relation(
                comparison.historical_f16_cpu_vs_gpu.mean_absolute_error,
                comparison.exact_f32_cpu_vs_gpu.mean_absolute_error,
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HistoricalF16CompensationAnalysis {
    pub descriptive_only: bool,
    pub numerical_threshold_used: bool,
    pub historical_f16_is_correct_reference_claimed: bool,
    pub stages: Vec<CompensationStageEvidence>,
    pub first_stage_historical_f16_closer_maxabs: Option<String>,
    pub first_stage_historical_f16_farther_maxabs: Option<String>,
}

fn compensation_analysis(
    trace: &FinalPositionTraceComparison,
) -> HistoricalF16CompensationAnalysis {
    let mut stages = vec![CompensationStageEvidence::from_comparison(
        "embedding",
        &trace.embedding,
    )];
    for layer in &trace.layers {
        stages.push(CompensationStageEvidence::from_comparison(
            format!("layer-{}-pre-attention-residual", layer.layer),
            &layer.pre_attention_residual,
        ));
        stages.push(CompensationStageEvidence::from_comparison(
            format!("layer-{}-post-attention", layer.layer),
            &layer.post_attention,
        ));
        stages.push(CompensationStageEvidence::from_comparison(
            format!("layer-{}-router-input", layer.layer),
            &layer.router_input,
        ));
        stages.push(CompensationStageEvidence::from_comparison(
            format!("layer-{}-selected-weights", layer.layer),
            &layer.selected_weights,
        ));
        stages.push(CompensationStageEvidence::from_comparison(
            format!("layer-{}-routed-moe", layer.layer),
            &layer.routed_moe_output,
        ));
        stages.push(CompensationStageEvidence::from_comparison(
            format!("layer-{}-post-moe", layer.layer),
            &layer.post_moe_residual,
        ));
    }
    stages.push(CompensationStageEvidence::from_comparison(
        "final-hidden",
        &trace.final_hidden,
    ));
    stages.push(CompensationStageEvidence::from_comparison(
        "final-rmsnorm",
        &trace.final_rmsnorm,
    ));
    stages.push(CompensationStageEvidence::from_comparison(
        "lm-head-logits",
        &trace.full_lm_head_logits,
    ));
    let first_stage_historical_f16_closer_maxabs = stages
        .iter()
        .find(|stage| stage.maxabs == DistanceRelation::HistoricalF16Closer)
        .map(|stage| stage.stage.clone());
    let first_stage_historical_f16_farther_maxabs = stages
        .iter()
        .find(|stage| stage.maxabs == DistanceRelation::ExactF32Closer)
        .map(|stage| stage.stage.clone());
    HistoricalF16CompensationAnalysis {
        descriptive_only: true,
        numerical_threshold_used: false,
        historical_f16_is_correct_reference_claimed: false,
        stages,
        first_stage_historical_f16_closer_maxabs,
        first_stage_historical_f16_farther_maxabs,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SameInputRouterStage {
    ExactOrderedMatch,
    GpuRouterGemvDrift,
    GpuRouterSoftmaxTopkDrift,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouterLayerAttributionEvidence {
    pub layer: usize,
    pub three_plane_selected_ids: ThreeWaySelectedIds,
    pub exact_f32_cpu_selected_weights: ExactVectorTraceEvidence,
    pub historical_f16_cpu_selected_weights: ExactVectorTraceEvidence,
    pub gpu_native_selected_weights: ExactVectorTraceEvidence,
    pub cpu_production_router_on_exact_gpu_input: RouterEvaluationEvidence,
    pub actual_gpu_production_router: ActualGpuRouterEvidence,
    pub same_input_cpu_vs_gpu_raw_logits: VectorNumericalEvidence,
    pub same_input_ordered_ids_equal: bool,
    pub same_input_stage: SameInputRouterStage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouterAttributionEvidence {
    pub layers: Vec<RouterLayerAttributionEvidence>,
    pub first_same_input_ordered_id_mismatch_layer: Option<usize>,
    pub known_postgres_defect_preserved_as_separate: bool,
    pub known_postgres_defect: &'static str,
}

fn router_layer_evidence(
    layer: usize,
    exact_trace: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    historical_trace: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    gpu_trace: &crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace,
    gpu_semantic: &crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace,
    gate: &crate::gating::LinearGate,
) -> Result<RouterLayerAttributionEvidence, String> {
    let semantic = gpu_semantic
        .layers
        .get(layer)
        .ok_or("GPU semantic router layer is missing")?;
    if gpu_trace.layer_router_input.get(layer) != Some(&semantic.router_input)
        || gpu_trace.layer_selected_ids.get(layer) != Some(&semantic.selected_ids)
        || gpu_trace.layer_selected_weights.get(layer) != Some(&semantic.selected_weights)
    {
        return Err(format!(
            "GPU full and semantic traces disagree at router layer {layer}"
        ));
    }
    let cpu = crate::gpu_native_router_rank_diagnostics::evaluate_cpu_router(
        "cpu-production-router-on-exact-gpu-router-input",
        gate,
        &semantic.router_input,
    )?;
    let actual = crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
        semantic.raw_logits.clone(),
        semantic.selected_ids.clone(),
        semantic.selected_weights.clone(),
    )?;
    let cpu_raw = cpu
        .raw_logits
        .iter()
        .map(|value| f32::from_bits(value.bits))
        .collect::<Vec<_>>();
    let same_input_ordered_ids_equal = cpu.top_8_ids == actual.top_8_ids;
    let same_input_stage = derive_same_input_router_stage(
        &cpu.top_8_ids,
        &actual.cpu_top_8_ids_derived_from_exact_gpu_raw_logits,
        &actual.top_8_ids,
    );
    Ok(RouterLayerAttributionEvidence {
        layer,
        three_plane_selected_ids: ThreeWaySelectedIds {
            exact_f32_cpu: exact_trace.layer_selected_ids[layer].clone(),
            historical_f16_cpu: historical_trace.layer_selected_ids[layer].clone(),
            gpu_native: semantic.selected_ids.clone(),
        },
        exact_f32_cpu_selected_weights: ExactVectorTraceEvidence::new(
            format!("exact-f32-cpu-layer-{layer}-selected-weights"),
            &exact_trace.layer_selected_weights[layer],
            true,
        ),
        historical_f16_cpu_selected_weights: ExactVectorTraceEvidence::new(
            format!("historical-f16-cpu-layer-{layer}-selected-weights"),
            &historical_trace.layer_selected_weights[layer],
            true,
        ),
        gpu_native_selected_weights: ExactVectorTraceEvidence::new(
            format!("production-gpu-native-layer-{layer}-selected-weights"),
            &semantic.selected_weights,
            true,
        ),
        same_input_cpu_vs_gpu_raw_logits: VectorNumericalEvidence::compare(
            "cpu-production-router-raw-logits-on-exact-gpu-input",
            "actual-production-gpu-router-raw-logits",
            &cpu_raw,
            &semantic.raw_logits,
        )?,
        cpu_production_router_on_exact_gpu_input: cpu,
        actual_gpu_production_router: actual,
        same_input_ordered_ids_equal,
        same_input_stage,
    })
}

fn derive_same_input_router_stage(
    cpu_on_gpu_input_ids: &[u32],
    cpu_from_gpu_raw_ids: &[u32],
    actual_gpu_ids: &[u32],
) -> SameInputRouterStage {
    if cpu_on_gpu_input_ids == actual_gpu_ids {
        SameInputRouterStage::ExactOrderedMatch
    } else if cpu_from_gpu_raw_ids == actual_gpu_ids {
        SameInputRouterStage::GpuRouterGemvDrift
    } else {
        SameInputRouterStage::GpuRouterSoftmaxTopkDrift
    }
}

fn router_attribution(
    exact_trace: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    historical_trace: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    gpu_trace: &crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace,
    gpu_semantic: &crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace,
    model: &crate::model::RealModel,
) -> Result<RouterAttributionEvidence, String> {
    let mut layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for layer in 0..EXPECTED_NUM_LAYERS {
        layers.push(router_layer_evidence(
            layer,
            exact_trace,
            historical_trace,
            gpu_trace,
            gpu_semantic,
            &model.layers[layer].gate,
        )?);
    }
    let first_same_input_ordered_id_mismatch_layer = layers
        .iter()
        .find(|layer| !layer.same_input_ordered_ids_equal)
        .map(|layer| layer.layer);
    Ok(RouterAttributionEvidence {
        layers,
        first_same_input_ordered_id_mismatch_layer,
        known_postgres_defect_preserved_as_separate: true,
        known_postgres_defect:
            "postgres-window-holdout generated-position-11 layer-38 CPU-same-input-107 GPU-95 gpu-router-gemv-drift",
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SameInputExpertEvidence {
    pub production_rank: usize,
    pub local_expert_id: u32,
    pub global_expert_id: u32,
    pub three_way_output: ThreeWayVectorComparison,
    pub historical_f16_vs_exact_f32_relation_maxabs: DistanceRelation,
    pub historical_f16_vs_exact_f32_relation_rms: DistanceRelation,
    pub historical_f16_vs_exact_f32_relation_meanabs: DistanceRelation,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SameInputExpertLayerEvidence {
    pub layer: usize,
    pub exact_gpu_f32_input: ExactVectorTraceEvidence,
    pub gpu_selected_expert_ids: Vec<u32>,
    pub gpu_selected_weights: ExactVectorTraceEvidence,
    pub per_expert_in_gpu_production_rank_order: Vec<SameInputExpertEvidence>,
    pub three_way_outputs_combined_with_actual_gpu_weights: ThreeWayVectorComparison,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SameInputExpertBoundaryEvidence {
    pub exact_gpu_f32_input_supplied_to_both_cpu_replays_without_pre_rounding: bool,
    pub exact_f32_cpu_boundary_emulation_enabled: bool,
    pub historical_f16_cpu_boundary_emulation_enabled: bool,
    pub actual_gpu_expert_outputs_observed: bool,
    pub current_gpu_arithmetic_emulator_used: bool,
    pub emulator_not_needed_reason: &'static str,
    pub selected_expert_execution_count: usize,
    pub exact_f32_closer_than_historical_f16_maxabs_count: usize,
    pub exact_f32_closer_than_historical_f16_rms_count: usize,
    pub exact_f32_closer_than_historical_f16_meanabs_count: usize,
    pub historical_f16_closer_than_exact_f32_maxabs_count: usize,
    pub historical_f16_closer_than_exact_f32_rms_count: usize,
    pub historical_f16_closer_than_exact_f32_meanabs_count: usize,
    pub mere_bit_inequality_promoted_to_defect: bool,
    pub layers: Vec<SameInputExpertLayerEvidence>,
}

async fn replay_historical_same_input_experts(
    runtime: &crate::BenchRealRuntime,
    gpu_semantic: &crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace,
) -> Result<Vec<Vec<Vec<f32>>>, Box<dyn std::error::Error>> {
    let mut layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for (layer, gpu) in gpu_semantic.layers.iter().enumerate() {
        if gpu.selected_ids.len() != EXPECTED_TOP_K || gpu.router_input.len() != EXPECTED_D_MODEL {
            return Err(format!(
                "GPU expert trace geometry is incomplete for historical replay at layer {layer}"
            )
            .into());
        }
        let global_ids = gpu
            .selected_ids
            .iter()
            .map(|expert| runtime.model.global_expert_id(layer, *expert))
            .collect::<Vec<_>>();
        let token_index = (layer as u64).wrapping_add(0x4849_5354_5350_4100);
        let outputs = runtime
            .engine
            .moe_step_with_timing(
                token_index,
                layer as u32,
                &gpu.router_input,
                &global_ids,
                None,
            )
            .await?;
        if outputs.len() != EXPECTED_TOP_K
            || outputs
                .iter()
                .any(|output| output.len() != EXPECTED_D_MODEL)
        {
            return Err(format!(
                "historical-f16 same-input expert replay is incomplete at layer {layer}"
            )
            .into());
        }
        layers.push(outputs);
    }
    Ok(layers)
}

async fn same_input_expert_boundary(
    runtime: &crate::BenchRealRuntime,
    gpu_semantic: &crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace,
    historical_outputs: &[Vec<Vec<f32>>],
) -> Result<SameInputExpertBoundaryEvidence, Box<dyn std::error::Error>> {
    if historical_outputs.len() != EXPECTED_NUM_LAYERS {
        return Err("historical same-input expert replay omitted one or more layers".into());
    }
    let mut layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    let mut exact_closer_maxabs = 0usize;
    let mut exact_closer_rms = 0usize;
    let mut exact_closer_meanabs = 0usize;
    let mut historical_closer_maxabs = 0usize;
    let mut historical_closer_rms = 0usize;
    let mut historical_closer_meanabs = 0usize;
    for (layer, gpu) in gpu_semantic.layers.iter().enumerate() {
        if gpu.selected_ids.len() != EXPECTED_TOP_K
            || gpu.selected_weights.len() != EXPECTED_TOP_K
            || gpu.route_outputs.len() != EXPECTED_TOP_K
            || gpu
                .route_outputs
                .iter()
                .any(|output| output.len() != EXPECTED_D_MODEL)
            || historical_outputs[layer].len() != EXPECTED_TOP_K
            || historical_outputs[layer]
                .iter()
                .any(|output| output.len() != EXPECTED_D_MODEL)
        {
            return Err(format!("GPU expert trace geometry is incomplete at layer {layer}").into());
        }
        let global_ids = gpu
            .selected_ids
            .iter()
            .map(|expert| runtime.model.global_expert_id(layer, *expert))
            .collect::<Vec<_>>();
        let token_index = (layer as u64).wrapping_add(0x5350_414e_4953_4800);
        let cpu_outputs = runtime
            .engine
            .moe_step_with_timing(
                token_index,
                layer as u32,
                &gpu.router_input,
                &global_ids,
                None,
            )
            .await?;
        if cpu_outputs.len() != EXPECTED_TOP_K
            || cpu_outputs
                .iter()
                .any(|output| output.len() != EXPECTED_D_MODEL)
        {
            return Err(
                format!("CPU same-input expert replay is incomplete at layer {layer}").into(),
            );
        }
        let mut per_expert = Vec::with_capacity(EXPECTED_TOP_K);
        for rank in 0..EXPECTED_TOP_K {
            let comparison = three_way_compare(
                "same-input-expert-output",
                &cpu_outputs[rank],
                &historical_outputs[layer][rank],
                &gpu.route_outputs[rank],
            )?;
            let maxabs = distance_relation(
                comparison.historical_f16_cpu_vs_gpu.max_absolute_error,
                comparison.exact_f32_cpu_vs_gpu.max_absolute_error,
            );
            let rms = distance_relation(
                comparison.historical_f16_cpu_vs_gpu.rms_error,
                comparison.exact_f32_cpu_vs_gpu.rms_error,
            );
            let meanabs = distance_relation(
                comparison.historical_f16_cpu_vs_gpu.mean_absolute_error,
                comparison.exact_f32_cpu_vs_gpu.mean_absolute_error,
            );
            exact_closer_maxabs += usize::from(maxabs == DistanceRelation::ExactF32Closer);
            exact_closer_rms += usize::from(rms == DistanceRelation::ExactF32Closer);
            exact_closer_meanabs += usize::from(meanabs == DistanceRelation::ExactF32Closer);
            historical_closer_maxabs +=
                usize::from(maxabs == DistanceRelation::HistoricalF16Closer);
            historical_closer_rms += usize::from(rms == DistanceRelation::HistoricalF16Closer);
            historical_closer_meanabs +=
                usize::from(meanabs == DistanceRelation::HistoricalF16Closer);
            per_expert.push(SameInputExpertEvidence {
                production_rank: rank + 1,
                local_expert_id: gpu.selected_ids[rank],
                global_expert_id: global_ids[rank],
                three_way_output: comparison,
                historical_f16_vs_exact_f32_relation_maxabs: maxabs,
                historical_f16_vs_exact_f32_relation_rms: rms,
                historical_f16_vs_exact_f32_relation_meanabs: meanabs,
            });
        }
        let cpu_combined = crate::inference::combine_outputs(&cpu_outputs, &gpu.selected_weights);
        let historical_combined =
            crate::inference::combine_outputs(&historical_outputs[layer], &gpu.selected_weights);
        layers.push(SameInputExpertLayerEvidence {
            layer,
            exact_gpu_f32_input: ExactVectorTraceEvidence::new(
                format!("layer-{layer}-exact-actual-gpu-f32-expert-input"),
                &gpu.router_input,
                true,
            ),
            gpu_selected_expert_ids: gpu.selected_ids.clone(),
            gpu_selected_weights: ExactVectorTraceEvidence::new(
                format!("layer-{layer}-actual-gpu-selected-weights"),
                &gpu.selected_weights,
                true,
            ),
            per_expert_in_gpu_production_rank_order: per_expert,
            three_way_outputs_combined_with_actual_gpu_weights: three_way_compare(
                "same-input-expert-outputs-combined-with-actual-gpu-weights",
                &cpu_combined,
                &historical_combined,
                &gpu.routed_moe_output,
            )?,
        });
    }
    Ok(SameInputExpertBoundaryEvidence {
        exact_gpu_f32_input_supplied_to_both_cpu_replays_without_pre_rounding: true,
        exact_f32_cpu_boundary_emulation_enabled: false,
        historical_f16_cpu_boundary_emulation_enabled: true,
        actual_gpu_expert_outputs_observed: true,
        current_gpu_arithmetic_emulator_used: false,
        emulator_not_needed_reason:
            "actual production GPU per-expert outputs were captured in the same final-prompt traversal",
        selected_expert_execution_count: layers.len() * EXPECTED_TOP_K,
        exact_f32_closer_than_historical_f16_maxabs_count: exact_closer_maxabs,
        exact_f32_closer_than_historical_f16_rms_count: exact_closer_rms,
        exact_f32_closer_than_historical_f16_meanabs_count: exact_closer_meanabs,
        historical_f16_closer_than_exact_f32_maxabs_count: historical_closer_maxabs,
        historical_f16_closer_than_exact_f32_rms_count: historical_closer_rms,
        historical_f16_closer_than_exact_f32_meanabs_count: historical_closer_meanabs,
        mere_bit_inequality_promoted_to_defect: false,
        layers,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankedTokenEvidence {
    pub token_id: u32,
    pub rank: usize,
    pub raw_logit: FloatEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CandidateTokenEvidence {
    pub token_id: u32,
    pub competing_token_id: u32,
    pub rank: usize,
    pub raw_logit: FloatEvidence,
    pub margin_over_competing_candidate: FloatEvidence,
    pub ulp_distance_to_competing_candidate: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LogitViewEvidence {
    pub source: &'static str,
    pub argmax: u32,
    pub top_12: Vec<RankedTokenEvidence>,
    pub candidates_140003_and_54275: Vec<CandidateTokenEvidence>,
}

fn ranked_logit_indices(logits: &[f32]) -> Result<Vec<usize>, String> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err("LM-head logits are empty or nonfinite".into());
    }
    let mut ranked = (0..logits.len()).collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    Ok(ranked)
}

fn deterministic_argmax(logits: &[f32]) -> Result<u32, String> {
    u32::try_from(ranked_logit_indices(logits)?[0])
        .map_err(|_| "argmax token ID does not fit u32".into())
}

fn logit_view(source: &'static str, logits: &[f32]) -> Result<LogitViewEvidence, String> {
    let ranked = ranked_logit_indices(logits)?;
    let candidate = |token_id: u32, competing_token_id: u32| {
        let token = token_id as usize;
        let competing = competing_token_id as usize;
        if token >= logits.len() || competing >= logits.len() {
            return Err("explicit Spanish token candidate is outside vocabulary".to_string());
        }
        let rank = ranked
            .iter()
            .position(|candidate| *candidate == token)
            .ok_or("candidate token missing from LM-head ranking")?
            + 1;
        Ok(CandidateTokenEvidence {
            token_id,
            competing_token_id,
            rank,
            raw_logit: FloatEvidence::new(logits[token]),
            margin_over_competing_candidate: FloatEvidence::new(logits[token] - logits[competing]),
            ulp_distance_to_competing_candidate:
                crate::gpu_native_router_rank_diagnostics::ulp_distance(
                    logits[token],
                    logits[competing],
                ),
        })
    };
    Ok(LogitViewEvidence {
        source,
        argmax: u32::try_from(ranked[0]).map_err(|_| "argmax token ID overflow")?,
        top_12: ranked
            .iter()
            .take(12)
            .enumerate()
            .map(|(rank, token)| RankedTokenEvidence {
                token_id: *token as u32,
                rank: rank + 1,
                raw_logit: FloatEvidence::new(logits[*token]),
            })
            .collect(),
        candidates_140003_and_54275: vec![
            candidate(EXACT_F32_CPU_TOKEN, FROZEN_GPU_TOKEN)?,
            candidate(FROZEN_GPU_TOKEN, EXACT_F32_CPU_TOKEN)?,
        ],
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SameInputFinalRmsnormEvidence {
    pub cpu_production_on_exact_gpu_final_hidden_vs_actual_gpu_final_rmsnorm:
        VectorNumericalEvidence,
    pub cpu_production_final_rmsnorm: ExactVectorTraceEvidence,
    pub actual_gpu_final_rmsnorm: ExactVectorTraceEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SameInputLmHeadEvidence {
    pub cpu_production_on_exact_gpu_final_rmsnorm_vs_actual_gpu_logits: VectorNumericalEvidence,
    pub cpu_same_input_argmax: u32,
    pub gpu_logits_deterministic_argmax: u32,
    pub actual_gpu_greedy_argmax: u32,
    pub views: Vec<LogitViewEvidence>,
}

#[derive(Clone, Debug, PartialEq)]
struct DownstreamCounterfactuals {
    cpu_argmax_on_cpu_norm_of_gpu_hidden: u32,
    cpu_argmax_on_actual_gpu_norm: u32,
    deterministic_argmax_on_actual_gpu_logits: u32,
}

fn same_input_downstream(
    model: &crate::model::RealModel,
    exact_trace: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    historical_trace: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    gpu_trace: &crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace,
) -> Result<
    (
        SameInputFinalRmsnormEvidence,
        SameInputLmHeadEvidence,
        DownstreamCounterfactuals,
    ),
    String,
> {
    let gpu_hidden = gpu_trace
        .layer_post_moe
        .last()
        .ok_or("GPU final hidden is missing")?;
    let cpu_norm_on_gpu_hidden = model.diagnostic_final_rms_norm(gpu_hidden);
    let cpu_logits_on_cpu_norm = model.diagnostic_greedy_logits(&cpu_norm_on_gpu_hidden);
    let cpu_logits_on_gpu_norm = model.diagnostic_greedy_logits(&gpu_trace.final_norm);
    let cpu_argmax_on_cpu_norm_of_gpu_hidden = deterministic_argmax(&cpu_logits_on_cpu_norm)?;
    let cpu_argmax_on_actual_gpu_norm = deterministic_argmax(&cpu_logits_on_gpu_norm)?;
    let deterministic_argmax_on_actual_gpu_logits = deterministic_argmax(&gpu_trace.logits)?;
    Ok((
        SameInputFinalRmsnormEvidence {
            cpu_production_on_exact_gpu_final_hidden_vs_actual_gpu_final_rmsnorm:
                VectorNumericalEvidence::compare(
                    "cpu-production-final-rmsnorm-on-exact-actual-gpu-final-hidden",
                    "actual-production-gpu-final-rmsnorm",
                    &cpu_norm_on_gpu_hidden,
                    &gpu_trace.final_norm,
                )?,
            cpu_production_final_rmsnorm: ExactVectorTraceEvidence::new(
                "cpu-production-final-rmsnorm-on-exact-actual-gpu-final-hidden",
                &cpu_norm_on_gpu_hidden,
                true,
            ),
            actual_gpu_final_rmsnorm: ExactVectorTraceEvidence::new(
                "actual-production-gpu-final-rmsnorm",
                &gpu_trace.final_norm,
                true,
            ),
        },
        SameInputLmHeadEvidence {
            cpu_production_on_exact_gpu_final_rmsnorm_vs_actual_gpu_logits:
                VectorNumericalEvidence::compare(
                    "cpu-production-lm-head-on-exact-actual-gpu-final-rmsnorm",
                    "actual-production-gpu-lm-head-logits",
                    &cpu_logits_on_gpu_norm,
                    &gpu_trace.logits,
                )?,
            cpu_same_input_argmax: cpu_argmax_on_actual_gpu_norm,
            gpu_logits_deterministic_argmax: deterministic_argmax_on_actual_gpu_logits,
            actual_gpu_greedy_argmax: gpu_trace.sampled_token,
            views: vec![
                logit_view("exact-f32-cpu-plane", &exact_trace.logits)?,
                logit_view("historical-f16-cpu-plane", &historical_trace.logits)?,
                logit_view("actual-production-gpu-native-plane", &gpu_trace.logits)?,
                logit_view(
                    "cpu-production-lm-head-on-cpu-rmsnorm-of-exact-gpu-hidden",
                    &cpu_logits_on_cpu_norm,
                )?,
                logit_view(
                    "cpu-production-lm-head-on-exact-actual-gpu-final-rmsnorm",
                    &cpu_logits_on_gpu_norm,
                )?,
            ],
        },
        DownstreamCounterfactuals {
            cpu_argmax_on_cpu_norm_of_gpu_hidden,
            cpu_argmax_on_actual_gpu_norm,
            deterministic_argmax_on_actual_gpu_logits,
        },
    ))
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
#[allow(dead_code)]
pub enum AttributionCategory {
    UpstreamPrefillDrift,
    AttentionDrift,
    RouterGemvDrift,
    RoutedMoeDrift,
    FinalRmsnormDrift,
    LmHeadGemvDrift,
    GreedyCutoffDrift,
    NumericalCompensation,
    MultiStageDrift,
    Unresolved,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConservativeAttributionEvidence {
    pub primary: AttributionCategory,
    pub supporting_classifications: Vec<AttributionCategory>,
    pub classification_uses_hidden_numerical_threshold: bool,
    pub mere_bit_inequality_promoted_to_causal_attribution: bool,
    pub no_production_correction_justified: bool,
}

fn derive_attribution(
    exact_token: u32,
    historical_token: u32,
    gpu_token: u32,
    downstream: &DownstreamCounterfactuals,
    router: &RouterAttributionEvidence,
) -> ConservativeAttributionEvidence {
    let mut causal = BTreeSet::new();
    if downstream.cpu_argmax_on_cpu_norm_of_gpu_hidden != exact_token {
        causal.insert(AttributionCategory::UpstreamPrefillDrift);
    }
    if downstream.cpu_argmax_on_actual_gpu_norm != downstream.cpu_argmax_on_cpu_norm_of_gpu_hidden {
        causal.insert(AttributionCategory::FinalRmsnormDrift);
    }
    if downstream.deterministic_argmax_on_actual_gpu_logits
        != downstream.cpu_argmax_on_actual_gpu_norm
    {
        causal.insert(AttributionCategory::LmHeadGemvDrift);
    }
    if gpu_token != downstream.deterministic_argmax_on_actual_gpu_logits {
        causal.insert(AttributionCategory::GreedyCutoffDrift);
    }
    if router
        .layers
        .iter()
        .any(|layer| layer.same_input_stage == SameInputRouterStage::GpuRouterGemvDrift)
    {
        causal.insert(AttributionCategory::RouterGemvDrift);
    }
    let mut supporting = causal.iter().copied().collect::<Vec<_>>();
    if historical_token == gpu_token && exact_token != gpu_token {
        supporting.push(AttributionCategory::NumericalCompensation);
    }
    let primary = match causal.len() {
        0 if supporting.contains(&AttributionCategory::NumericalCompensation) => {
            AttributionCategory::NumericalCompensation
        }
        0 => AttributionCategory::Unresolved,
        1 => *causal.iter().next().unwrap(),
        _ => AttributionCategory::MultiStageDrift,
    };
    ConservativeAttributionEvidence {
        primary,
        supporting_classifications: supporting,
        classification_uses_hidden_numerical_threshold: false,
        mere_bit_inequality_promoted_to_causal_attribution: false,
        no_production_correction_justified: true,
    }
}

fn cpu_plane_full_trace(
    plane: &'static str,
    trace: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
) -> Result<PlaneFullTraceEvidence, String> {
    if trace.layer_post_attn.len() != EXPECTED_NUM_LAYERS
        || trace.layer_router_input.len() != EXPECTED_NUM_LAYERS
        || trace.layer_selected_ids.len() != EXPECTED_NUM_LAYERS
        || trace.layer_selected_weights.len() != EXPECTED_NUM_LAYERS
        || trace.layer_routed_moe_output.len() != EXPECTED_NUM_LAYERS
        || trace.layer_post_moe.len() != EXPECTED_NUM_LAYERS
    {
        return Err(format!("{plane} full trace layer geometry is incomplete"));
    }
    let mut layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for layer in 0..EXPECTED_NUM_LAYERS {
        layers.push(FullTraceLayerEvidence {
            layer,
            pre_attention_residual: ExactVectorTraceEvidence::new(
                format!("{plane}-layer-{layer}-pre-attention-residual"),
                if layer == 0 {
                    &trace.embedding
                } else {
                    &trace.layer_post_moe[layer - 1]
                },
                true,
            ),
            post_attention: ExactVectorTraceEvidence::new(
                format!("{plane}-layer-{layer}-post-attention"),
                &trace.layer_post_attn[layer],
                true,
            ),
            router_input: ExactVectorTraceEvidence::new(
                format!("{plane}-layer-{layer}-router-input"),
                &trace.layer_router_input[layer],
                true,
            ),
            selected_expert_ids: trace.layer_selected_ids[layer].clone(),
            selected_weights: ExactVectorTraceEvidence::new(
                format!("{plane}-layer-{layer}-selected-weights"),
                &trace.layer_selected_weights[layer],
                true,
            ),
            routed_moe_output: ExactVectorTraceEvidence::new(
                format!("{plane}-layer-{layer}-routed-moe-output"),
                &trace.layer_routed_moe_output[layer],
                true,
            ),
            post_moe_residual: ExactVectorTraceEvidence::new(
                format!("{plane}-layer-{layer}-post-moe-residual"),
                &trace.layer_post_moe[layer],
                true,
            ),
        });
    }
    let final_hidden = trace
        .layer_post_moe
        .last()
        .ok_or_else(|| format!("{plane} final hidden is missing"))?;
    Ok(PlaneFullTraceEvidence {
        plane,
        embedding: ExactVectorTraceEvidence::new(
            format!("{plane}-embedding"),
            &trace.embedding,
            true,
        ),
        layers,
        final_hidden: ExactVectorTraceEvidence::new(
            format!("{plane}-final-hidden"),
            final_hidden,
            true,
        ),
        final_rmsnorm: ExactVectorTraceEvidence::new(
            format!("{plane}-final-rmsnorm"),
            &trace.final_norm,
            true,
        ),
        full_lm_head_logits: ExactVectorTraceEvidence::new(
            format!("{plane}-full-lm-head-logits"),
            &trace.logits,
            true,
        ),
        greedy_argmax: trace.sampled_token,
    })
}

fn gpu_plane_full_trace(
    trace: &crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace,
    semantic: &crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace,
) -> Result<PlaneFullTraceEvidence, String> {
    if trace.layer_post_attn.len() != EXPECTED_NUM_LAYERS
        || trace.layer_router_input.len() != EXPECTED_NUM_LAYERS
        || trace.layer_selected_ids.len() != EXPECTED_NUM_LAYERS
        || trace.layer_selected_weights.len() != EXPECTED_NUM_LAYERS
        || trace.layer_post_moe.len() != EXPECTED_NUM_LAYERS
        || semantic.layers.len() != EXPECTED_NUM_LAYERS
    {
        return Err("GPU-native full trace layer geometry is incomplete".into());
    }
    let mut layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for layer in 0..EXPECTED_NUM_LAYERS {
        let semantic_layer = &semantic.layers[layer];
        if trace.layer_router_input[layer] != semantic_layer.router_input
            || trace.layer_selected_ids[layer] != semantic_layer.selected_ids
            || trace.layer_selected_weights[layer] != semantic_layer.selected_weights
        {
            return Err(format!("GPU diagnostic sinks disagree at layer {layer}"));
        }
        layers.push(FullTraceLayerEvidence {
            layer,
            pre_attention_residual: ExactVectorTraceEvidence::new(
                format!("GPU_NATIVE-layer-{layer}-pre-attention-residual"),
                if layer == 0 {
                    &trace.embedding
                } else {
                    &trace.layer_post_moe[layer - 1]
                },
                true,
            ),
            post_attention: ExactVectorTraceEvidence::new(
                format!("GPU_NATIVE-layer-{layer}-post-attention"),
                &trace.layer_post_attn[layer],
                true,
            ),
            router_input: ExactVectorTraceEvidence::new(
                format!("GPU_NATIVE-layer-{layer}-router-input"),
                &trace.layer_router_input[layer],
                true,
            ),
            selected_expert_ids: trace.layer_selected_ids[layer].clone(),
            selected_weights: ExactVectorTraceEvidence::new(
                format!("GPU_NATIVE-layer-{layer}-selected-weights"),
                &trace.layer_selected_weights[layer],
                true,
            ),
            routed_moe_output: ExactVectorTraceEvidence::new(
                format!("GPU_NATIVE-layer-{layer}-routed-moe-output"),
                &semantic_layer.routed_moe_output,
                true,
            ),
            post_moe_residual: ExactVectorTraceEvidence::new(
                format!("GPU_NATIVE-layer-{layer}-post-moe-residual"),
                &trace.layer_post_moe[layer],
                true,
            ),
        });
    }
    let final_hidden = trace
        .layer_post_moe
        .last()
        .ok_or("GPU-native final hidden is missing")?;
    Ok(PlaneFullTraceEvidence {
        plane: "GPU_NATIVE",
        embedding: ExactVectorTraceEvidence::new("GPU_NATIVE-embedding", &trace.embedding, true),
        layers,
        final_hidden: ExactVectorTraceEvidence::new("GPU_NATIVE-final-hidden", final_hidden, true),
        final_rmsnorm: ExactVectorTraceEvidence::new(
            "GPU_NATIVE-final-rmsnorm",
            &trace.final_norm,
            true,
        ),
        full_lm_head_logits: ExactVectorTraceEvidence::new(
            "GPU_NATIVE-full-lm-head-logits",
            &trace.logits,
            true,
        ),
        greedy_argmax: trace.sampled_token,
    })
}

fn compare_final_traces(
    exact: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    historical: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    gpu: &crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace,
    gpu_semantic: &crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace,
) -> Result<FinalPositionTraceComparison, String> {
    let mut layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for layer in 0..EXPECTED_NUM_LAYERS {
        layers.push(ThreeWayLayerTraceComparison {
            layer,
            pre_attention_residual: three_way_compare(
                "pre-attention-residual",
                if layer == 0 {
                    &exact.embedding
                } else {
                    &exact.layer_post_moe[layer - 1]
                },
                if layer == 0 {
                    &historical.embedding
                } else {
                    &historical.layer_post_moe[layer - 1]
                },
                if layer == 0 {
                    &gpu.embedding
                } else {
                    &gpu.layer_post_moe[layer - 1]
                },
            )?,
            post_attention: three_way_compare(
                "post-attention",
                &exact.layer_post_attn[layer],
                &historical.layer_post_attn[layer],
                &gpu.layer_post_attn[layer],
            )?,
            router_input: three_way_compare(
                "router-input",
                &exact.layer_router_input[layer],
                &historical.layer_router_input[layer],
                &gpu.layer_router_input[layer],
            )?,
            selected_ids: ThreeWaySelectedIds {
                exact_f32_cpu: exact.layer_selected_ids[layer].clone(),
                historical_f16_cpu: historical.layer_selected_ids[layer].clone(),
                gpu_native: gpu.layer_selected_ids[layer].clone(),
            },
            selected_weights: three_way_compare(
                "selected-weights",
                &exact.layer_selected_weights[layer],
                &historical.layer_selected_weights[layer],
                &gpu.layer_selected_weights[layer],
            )?,
            routed_moe_output: three_way_compare(
                "routed-moe-output",
                &exact.layer_routed_moe_output[layer],
                &historical.layer_routed_moe_output[layer],
                &gpu_semantic.layers[layer].routed_moe_output,
            )?,
            post_moe_residual: three_way_compare(
                "post-moe-residual",
                &exact.layer_post_moe[layer],
                &historical.layer_post_moe[layer],
                &gpu.layer_post_moe[layer],
            )?,
        });
    }
    Ok(FinalPositionTraceComparison {
        embedding: three_way_compare(
            "embedding",
            &exact.embedding,
            &historical.embedding,
            &gpu.embedding,
        )?,
        layers,
        final_hidden: three_way_compare(
            "final-hidden",
            exact
                .layer_post_moe
                .last()
                .ok_or("exact final hidden missing")?,
            historical
                .layer_post_moe
                .last()
                .ok_or("historical final hidden missing")?,
            gpu.layer_post_moe
                .last()
                .ok_or("GPU final hidden missing")?,
        )?,
        final_rmsnorm: three_way_compare(
            "final-rmsnorm",
            &exact.final_norm,
            &historical.final_norm,
            &gpu.final_norm,
        )?,
        full_lm_head_logits: three_way_compare(
            "lm-head-logits",
            &exact.logits,
            &historical.logits,
            &gpu.logits,
        )?,
        greedy_argmax_exact_f32_cpu: exact.sampled_token,
        greedy_argmax_historical_f16_cpu: historical.sampled_token,
        greedy_argmax_gpu_native: gpu.sampled_token,
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PlaneRuntimeEvidence {
    pub plane: &'static str,
    pub runtime_mode: &'static str,
    pub isolated_runtime: bool,
    pub kv_or_session_state_shared_with_another_plane: bool,
    pub caller_owned_prompt_token_ids_sha256: String,
    pub per_plane_retokenization_performed: bool,
    pub resolved_config_sha256: String,
    pub execution_plan: crate::qualification::ExecutionPlanEvidence,
    pub model_load: crate::greedy_parity::ModelLoadEvidence,
    pub boundary_before: Option<crate::engine::CpuQ4BoundaryEmulationSnapshot>,
    pub boundary_after: Option<crate::engine::CpuQ4BoundaryEmulationSnapshot>,
    pub routed_execution_after: crate::engine::RoutedExpertExecutionSnapshot,
    pub gpu_token_loop_after: Option<crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot>,
    pub prompt_token_steps_completed: usize,
    pub generated_token_continuation_steps: usize,
    pub background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

struct CpuPlaneCapture {
    final_hidden_by_prompt_position: Vec<Vec<f32>>,
    final_trace: crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    runtime: PlaneRuntimeEvidence,
}

struct HistoricalPlaneCapture {
    plane: CpuPlaneCapture,
    same_input_expert_outputs: Vec<Vec<Vec<f32>>>,
}

struct GpuPlaneCapture {
    final_hidden_by_prompt_position: Vec<Vec<f32>>,
    final_trace: crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace,
    final_semantic_trace: crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace,
    runtime: PlaneRuntimeEvidence,
    device: crate::backend::GpuDeviceIdentity,
}

struct ExactPlaneCapture {
    plane: CpuPlaneCapture,
    router: RouterAttributionEvidence,
    expert_boundary: SameInputExpertBoundaryEvidence,
    same_input_rmsnorm: SameInputFinalRmsnormEvidence,
    same_input_lm_head: SameInputLmHeadEvidence,
    downstream: DownstreamCounterfactuals,
}

fn model_load_is_strict(load: &crate::greedy_parity::ModelLoadEvidence) -> bool {
    load.strict
        && load.required_tensors > 0
        && load.loaded_tensors == load.required_tensors
        && !load.seeded_fallback_remained
        && load.loader != "seeded"
}

fn validate_cpu_geometry(runtime: &crate::BenchRealRuntime) -> Result<(), String> {
    if runtime.model.config.num_layers != EXPECTED_NUM_LAYERS
        || runtime.model.config.num_experts != EXPECTED_NUM_EXPERTS
        || runtime.model.config.top_k != EXPECTED_TOP_K
        || runtime.model.config.d_model != EXPECTED_D_MODEL
        || runtime.model.config.d_ff != EXPECTED_D_FF
    {
        return Err("CPU plane model geometry differs from frozen Qwen geometry".into());
    }
    Ok(())
}

struct GpuNativeAuthoritativeRuntimeEvidence {
    legacy_execution_context: crate::qualification::ExecutionPlanEvidence,
    gpu_native_token_loop_geometry: Option<crate::gpu_native_token_loop::GpuNativeModelGeometry>,
    authoritative_device: Option<crate::backend::GpuDeviceIdentity>,
    model_load: crate::greedy_parity::ModelLoadEvidence,
    real_transformer_gpu_native: bool,
    compute_offload: crate::backend::ComputeOffload,
}

/// Validate the deliberately retained legacy `ExecutionContext` plane map.
///
/// The CPU markers in this evidence do not describe placement in the
/// GPU-native token loop. They prove that the legacy routed-expert registry is
/// disabled while the authoritative context still owns the Vulkan device used
/// by the separate GPU-native full-token path.
fn validate_gpu_native_legacy_execution_context_contract(
    plan: &crate::qualification::ExecutionPlanEvidence,
) -> Result<(), String> {
    if plan.requested != "gpu"
        || plan.resolved != "gpu"
        || plan.embeddings != "cpu"
        || plan.lm_head != "cpu"
        || plan.dense_projections != "cpu"
        || plan.attention != "gpu"
        || plan.kv != "gpu"
        || plan.router != "cpu"
        || plan.routed_experts != "cpu"
        || plan.routed_expert_dtype != "q4_0"
        || plan.fallback_occurred
    {
        return Err(format!(
            "GPU-native authoritative runtime contract rejected legacy ExecutionPlanEvidence: \
             observed={plan:?}; expected requested=\"gpu\", resolved=\"gpu\", \
             embeddings=\"cpu\", lm_head=\"cpu\", dense_projections=\"cpu\", \
             attention=\"gpu\", kv=\"gpu\", router=\"cpu\", \
             routed_experts=\"cpu\", routed_expert_dtype=\"q4_0\", \
             fallback_occurred=false. ExecutionPlanEvidence is legacy \
             execution-context evidence, not GPU-native token-loop placement; \
             gpu_native_token_loop is validated separately as the authoritative \
             GPU-native full-token path"
        ));
    }
    Ok(())
}

fn validate_gpu_native_authoritative_runtime_contract(
    evidence: &GpuNativeAuthoritativeRuntimeEvidence,
    expected_adapter_name: &str,
) -> Result<(), String> {
    validate_gpu_native_legacy_execution_context_contract(&evidence.legacy_execution_context)?;

    if !evidence.real_transformer_gpu_native
        || evidence.compute_offload != crate::backend::ComputeOffload::Gpu
    {
        return Err(format!(
            "GPU-native authoritative runtime contract requires \
             real_transformer.gpu_native=true and compute_offload=Gpu, observed \
             gpu_native={} compute_offload={:?}",
            evidence.real_transformer_gpu_native, evidence.compute_offload
        ));
    }

    let device = evidence.authoritative_device.as_ref().ok_or(
        "GPU-native authoritative runtime contract requires authoritative GPU device identity; \
         gpu_native_token_loop is the separately validated authoritative full-token path",
    )?;
    if expected_adapter_name != REQUIRED_ADAPTER_NAME
        || device.name != expected_adapter_name
        || device.vendor_id != 0x10de
        || device.device_type != "DiscreteGpu"
        || device.wgpu_backend != "vulkan"
        || device.compute_plane != "wgpu-vulkan"
        || device.software_adapter
    {
        return Err(format!(
            "GPU-native authoritative runtime contract rejected authoritative device: \
             observed={device:?}; expected adapter={REQUIRED_ADAPTER_NAME:?}, \
             vendor_id=0x10de, device_type=\"DiscreteGpu\", wgpu_backend=\"vulkan\", \
             compute_plane=\"wgpu-vulkan\", software_adapter=false; \
             expected_adapter_name argument was {expected_adapter_name:?}"
        ));
    }

    let geometry = evidence.gpu_native_token_loop_geometry.ok_or(
        "GPU-native authoritative runtime contract requires gpu_native_token_loop; \
         ExecutionPlanEvidence is only legacy execution-context evidence and is not \
         proof of the authoritative GPU-native full-token path",
    )?;
    if geometry.num_layers != EXPECTED_NUM_LAYERS
        || geometry.num_experts != EXPECTED_NUM_EXPERTS
        || geometry.top_k != EXPECTED_TOP_K
        || geometry.d_model != EXPECTED_D_MODEL
        || geometry.d_ff != EXPECTED_D_FF
    {
        return Err(format!(
            "GPU-native authoritative runtime contract rejected gpu_native_token_loop geometry: \
             observed={geometry:?}; expected num_layers={EXPECTED_NUM_LAYERS}, \
             num_experts={EXPECTED_NUM_EXPERTS}, top_k={EXPECTED_TOP_K}, \
             d_model={EXPECTED_D_MODEL}, d_ff={EXPECTED_D_FF}"
        ));
    }

    if !model_load_is_strict(&evidence.model_load) {
        return Err(format!(
            "GPU-native authoritative runtime contract rejected strict real model load: \
             observed={:?}; expected strict=true, loaded_tensors=required_tensors>0, \
             seeded_fallback_remained=false, loader!=\"seeded\"",
            evidence.model_load
        ));
    }
    Ok(())
}

fn require_exact_boundary_clean(
    snapshot: crate::engine::CpuQ4BoundaryEmulationSnapshot,
) -> Result<(), String> {
    if snapshot.enabled || snapshot.routed_expert_dispatches != 0 {
        return Err(format!(
            "EXACT_F32_CPU boundary must remain disabled/zero, observed enabled={} dispatches={}",
            snapshot.enabled, snapshot.routed_expert_dispatches
        ));
    }
    Ok(())
}

fn enable_historical_boundary_clean(
    engine: &crate::engine::Engine,
) -> Result<crate::engine::CpuQ4BoundaryEmulationSnapshot, String> {
    engine.enable_cpu_q4_boundary_emulation()?;
    let snapshot = engine.cpu_q4_boundary_emulation_snapshot();
    require_historical_boundary_clean(snapshot)?;
    Ok(snapshot)
}

fn require_historical_boundary_clean(
    snapshot: crate::engine::CpuQ4BoundaryEmulationSnapshot,
) -> Result<(), String> {
    if !snapshot.enabled || snapshot.routed_expert_dispatches != 0 {
        return Err(format!(
            "HISTORICAL_F16_CPU boundary must start enabled/zero, observed enabled={} dispatches={}",
            snapshot.enabled, snapshot.routed_expert_dispatches
        ));
    }
    Ok(())
}

async fn capture_cpu_prompt(
    runtime: &crate::BenchRealRuntime,
    token_ids: &[u32],
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
    plane: &'static str,
) -> Result<
    (
        Vec<Vec<f32>>,
        crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    ),
    Box<dyn std::error::Error>,
> {
    let mut kv = runtime.model.fresh_kv_caches();
    let mut final_hidden_by_prompt_position = Vec::with_capacity(token_ids.len());
    let mut final_trace = None;
    for (position, &token_id) in token_ids.iter().enumerate() {
        let trace = crate::with_progress_timeout(
            format!("Spanish first-token {plane} prefill position {position}"),
            watchdog,
            async {
                Ok::<_, Box<dyn std::error::Error>>(
                    runtime
                        .model
                        .forward_token_diagnostic_trace(
                            &runtime.engine,
                            token_id,
                            position,
                            &mut kv,
                            None,
                        )
                        .await?,
                )
            },
        )
        .await?;
        final_hidden_by_prompt_position.push(
            trace
                .layer_post_moe
                .last()
                .ok_or("CPU trace omitted final hidden")?
                .clone(),
        );
        final_trace = Some(trace);
    }
    Ok((
        final_hidden_by_prompt_position,
        final_trace.ok_or("CPU plane did not execute a final prompt position")?,
    ))
}

async fn execute_gpu_plane(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    token_ids: Arc<Vec<u32>>,
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<GpuPlaneCapture, Box<dyn std::error::Error>> {
    let runtime = crate::build_isolated_greedy_runtime(
        spec,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
        tokenizer,
    )
    .await?;
    let attempt = async {
        let observed_config_sha256 = crate::resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err("GPU_NATIVE resolved configuration identity drifted".into());
        }
        let execution_plan: crate::qualification::ExecutionPlanEvidence =
            runtime.engine.execution_context().plan().into();
        let device = runtime.engine.gpu_device_identity();
        let model_load = crate::greedy_parity_model_load(&runtime);
        let authoritative_runtime = GpuNativeAuthoritativeRuntimeEvidence {
            legacy_execution_context: execution_plan.clone(),
            gpu_native_token_loop_geometry: runtime
                .gpu_native_token_loop
                .as_ref()
                .map(|token_loop| token_loop.model_geometry()),
            authoritative_device: device.clone(),
            model_load: model_load.clone(),
            real_transformer_gpu_native: runtime.cfg.real_transformer.gpu_native,
            compute_offload: runtime.cfg.real_transformer.compute_offload,
        };
        validate_gpu_native_authoritative_runtime_contract(
            &authoritative_runtime,
            expected_adapter_name,
        )?;
        let device = device.expect("authoritative runtime contract requires device identity");
        let token_loop = runtime
            .gpu_native_token_loop
            .as_ref()
            .expect("authoritative runtime contract requires gpu_native_token_loop");
        let geometry = token_loop.model_geometry();
        if token_loop.snapshot() != crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot::default()
        {
            return Err("GPU_NATIVE token-loop counters did not start at zero".into());
        }
        let full_layout = crate::gpu_native_diagnostics::GpuNativeDiagnosticTraceLayout::try_new(
            geometry.num_layers,
            geometry.d_model,
            geometry.top_k,
            geometry.vocab_size,
        )?;
        let semantic_layout =
            crate::gpu_native_semantic_parity_corpus::SemanticCorpusTraceLayout::try_new(geometry)?;
        let full_staging = token_loop.create_diagnostic_staging_buffer(&full_layout)?;
        let semantic_staging =
            token_loop.create_semantic_parity_corpus_diagnostic_staging_buffer(&semantic_layout)?;
        let mut request = token_loop.create_semantic_parity_corpus_diagnostic_request_state()?;
        let last_position = token_ids.len() - 1;
        let mut final_hidden_by_prompt_position = Vec::with_capacity(token_ids.len());
        for (position, &token_id) in token_ids[..last_position].iter().enumerate() {
            let (trace, _) = crate::with_progress_timeout(
                format!("Spanish first-token GPU_NATIVE prefill position {position}"),
                watchdog,
                async {
                    Ok::<_, Box<dyn std::error::Error>>(
                        token_loop
                            .step_token_diagnostic(
                                &runtime.engine,
                                &mut request,
                                token_id,
                                position,
                                false,
                                &full_layout,
                                &full_staging,
                            )
                            .await?,
                    )
                },
            )
            .await?;
            if trace.final_status != 0 || trace.layer_statuses.iter().any(|status| *status != 0) {
                return Err(format!(
                    "GPU_NATIVE prefill position {position} reported nonzero diagnostic status"
                )
                .into());
            }
            final_hidden_by_prompt_position.push(
                trace
                    .layer_post_moe
                    .last()
                    .ok_or("GPU prefix trace omitted final hidden")?
                    .clone(),
            );
        }
        let final_token_id = token_ids[last_position];
        let (final_trace, final_semantic_trace, sampled_token, _) =
            crate::with_progress_timeout(
                "Spanish first-token GPU_NATIVE final prompt position".to_string(),
                watchdog,
                async {
                    Ok::<_, Box<dyn std::error::Error>>(
                        token_loop
                            .step_token_full_and_semantic_corpus_diagnostic(
                                &runtime.engine,
                                &mut request,
                                final_token_id,
                                last_position,
                                &full_layout,
                                &full_staging,
                                &semantic_layout,
                                &semantic_staging,
                            )
                            .await?,
                    )
                },
            )
            .await?;
        if final_trace.final_status != 0
            || final_trace
                .layer_statuses
                .iter()
                .any(|status| *status != 0)
        {
            return Err("GPU_NATIVE final prompt position reported nonzero diagnostic status".into());
        }
        if sampled_token != final_trace.sampled_token || sampled_token != FROZEN_GPU_TOKEN {
            return Err(format!(
                "GPU_NATIVE Spanish first token was {sampled_token}, frozen target is {FROZEN_GPU_TOKEN}"
            )
            .into());
        }
        final_hidden_by_prompt_position.push(
            final_trace
                .layer_post_moe
                .last()
                .ok_or("GPU final trace omitted final hidden")?
                .clone(),
        );
        if request.committed_position() != token_ids.len() {
            return Err("GPU_NATIVE executed an incomplete or extra prompt step".into());
        }
        let loop_after = token_loop.snapshot();
        if loop_after.tokens_completed != token_ids.len() as u64
            || loop_after.fatal_failures != 0
            || loop_after.no_progress_failures != 0
        {
            return Err("GPU_NATIVE token loop did not complete the prompt cleanly".into());
        }
        let routed = runtime.engine.routed_expert_execution_snapshot();
        if routed.cpu_routed_expert_dispatches != 0
            || routed.gpu_cpu_fallbacks != 0
            || routed.degraded_expert_substitutions != 0
        {
            return Err("GPU_NATIVE observed CPU expert fallback or degradation".into());
        }
        Ok::<_, Box<dyn std::error::Error>>(GpuPlaneCapture {
            final_hidden_by_prompt_position,
            final_trace,
            final_semantic_trace,
            runtime: PlaneRuntimeEvidence {
                plane: "GPU_NATIVE",
                runtime_mode: "RealCliRuntimeMode::IsolatedGpuNativeDiagnostic",
                isolated_runtime: true,
                kv_or_session_state_shared_with_another_plane: false,
                caller_owned_prompt_token_ids_sha256:
                    crate::greedy_parity::token_ids_sha256(&token_ids),
                per_plane_retokenization_performed: false,
                resolved_config_sha256: observed_config_sha256,
                execution_plan,
                model_load,
                boundary_before: None,
                boundary_after: None,
                routed_execution_after: routed,
                gpu_token_loop_after: Some(loop_after),
                prompt_token_steps_completed: token_ids.len(),
                generated_token_continuation_steps: 0,
                background_shutdown:
                    crate::greedy_parity::BackgroundShutdownEvidence::default(),
            },
            device,
        })
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut capture), Ok(shutdown)) => {
            capture.runtime.background_shutdown = shutdown;
            Ok(capture)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; GPU_NATIVE shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

async fn execute_historical_plane(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    token_ids: Arc<Vec<u32>>,
    resolved_config_sha256: &str,
    gpu_semantic: &crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<HistoricalPlaneCapture, Box<dyn std::error::Error>> {
    let runtime = crate::build_isolated_greedy_runtime(
        spec,
        RealCliRuntimeMode::IsolatedGreedyParityCpu,
        tokenizer,
    )
    .await?;
    let attempt = async {
        validate_cpu_geometry(&runtime)?;
        let observed_config_sha256 = crate::resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err("HISTORICAL_F16_CPU resolved configuration identity drifted".into());
        }
        let execution_plan: crate::qualification::ExecutionPlanEvidence =
            runtime.engine.execution_context().plan().into();
        if !crate::greedy_parity::cpu_plan_exact(&execution_plan)
            || runtime.gpu_native_token_loop.is_some()
            || runtime.engine.gpu_device_identity().is_some()
        {
            return Err("HISTORICAL_F16_CPU did not resolve to strict CPU-only Q4".into());
        }
        let model_load = crate::greedy_parity_model_load(&runtime);
        if !model_load_is_strict(&model_load) {
            return Err("HISTORICAL_F16_CPU did not load the strict real model".into());
        }
        let boundary_before = enable_historical_boundary_clean(&runtime.engine)?;
        let (final_hidden_by_prompt_position, final_trace) =
            capture_cpu_prompt(&runtime, &token_ids, watchdog, "HISTORICAL_F16_CPU").await?;
        if final_trace.sampled_token != HISTORICAL_F16_CPU_TOKEN {
            return Err(format!(
                "HISTORICAL_F16_CPU Spanish first token was {}, frozen target is {HISTORICAL_F16_CPU_TOKEN}",
                final_trace.sampled_token
            )
            .into());
        }
        let same_input_expert_outputs =
            replay_historical_same_input_experts(&runtime, gpu_semantic).await?;
        let boundary_after = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_after.enabled || boundary_after.routed_expert_dispatches == 0 {
            return Err("HISTORICAL_F16_CPU boundary was not exercised".into());
        }
        Ok::<_, Box<dyn std::error::Error>>(HistoricalPlaneCapture {
            plane: CpuPlaneCapture {
                final_hidden_by_prompt_position,
                final_trace,
                runtime: PlaneRuntimeEvidence {
                    plane: "HISTORICAL_F16_CPU",
                    runtime_mode: "RealCliRuntimeMode::IsolatedGreedyParityCpu",
                    isolated_runtime: true,
                    kv_or_session_state_shared_with_another_plane: false,
                    caller_owned_prompt_token_ids_sha256:
                        crate::greedy_parity::token_ids_sha256(&token_ids),
                    per_plane_retokenization_performed: false,
                    resolved_config_sha256: observed_config_sha256,
                    execution_plan,
                    model_load,
                    boundary_before: Some(boundary_before),
                    boundary_after: Some(boundary_after),
                    routed_execution_after: runtime.engine.routed_expert_execution_snapshot(),
                    gpu_token_loop_after: None,
                    prompt_token_steps_completed: token_ids.len(),
                    generated_token_continuation_steps: 0,
                    background_shutdown:
                        crate::greedy_parity::BackgroundShutdownEvidence::default(),
                },
            },
            same_input_expert_outputs,
        })
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut capture), Ok(shutdown)) => {
            capture.plane.runtime.background_shutdown = shutdown;
            Ok(capture)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; HISTORICAL_F16_CPU shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_exact_plane(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    token_ids: Arc<Vec<u32>>,
    resolved_config_sha256: &str,
    historical_trace: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    historical_same_input_expert_outputs: &[Vec<Vec<f32>>],
    gpu: &GpuPlaneCapture,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<ExactPlaneCapture, Box<dyn std::error::Error>> {
    let runtime = crate::build_isolated_greedy_runtime(
        spec,
        RealCliRuntimeMode::IsolatedGreedyParityCpu,
        tokenizer,
    )
    .await?;
    let attempt = async {
        validate_cpu_geometry(&runtime)?;
        let observed_config_sha256 = crate::resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err("EXACT_F32_CPU resolved configuration identity drifted".into());
        }
        let execution_plan: crate::qualification::ExecutionPlanEvidence =
            runtime.engine.execution_context().plan().into();
        if !crate::greedy_parity::cpu_plan_exact(&execution_plan)
            || runtime.gpu_native_token_loop.is_some()
            || runtime.engine.gpu_device_identity().is_some()
        {
            return Err("EXACT_F32_CPU did not resolve to strict CPU-only Q4".into());
        }
        let model_load = crate::greedy_parity_model_load(&runtime);
        if !model_load_is_strict(&model_load) {
            return Err("EXACT_F32_CPU did not load the strict real model".into());
        }
        let boundary_before = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        require_exact_boundary_clean(boundary_before)?;
        let (final_hidden_by_prompt_position, final_trace) =
            capture_cpu_prompt(&runtime, &token_ids, watchdog, "EXACT_F32_CPU").await?;
        if final_trace.sampled_token != EXACT_F32_CPU_TOKEN {
            return Err(format!(
                "EXACT_F32_CPU Spanish first token was {}, frozen target is {EXACT_F32_CPU_TOKEN}",
                final_trace.sampled_token
            )
            .into());
        }
        let router = router_attribution(
            &final_trace,
            historical_trace,
            &gpu.final_trace,
            &gpu.final_semantic_trace,
            &runtime.model,
        )?;
        let expert_boundary = same_input_expert_boundary(
            &runtime,
            &gpu.final_semantic_trace,
            historical_same_input_expert_outputs,
        )
        .await?;
        let (same_input_rmsnorm, same_input_lm_head, downstream) = same_input_downstream(
            &runtime.model,
            &final_trace,
            historical_trace,
            &gpu.final_trace,
        )?;
        let boundary_after = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        require_exact_boundary_clean(boundary_after)?;
        Ok::<_, Box<dyn std::error::Error>>(ExactPlaneCapture {
            plane: CpuPlaneCapture {
                final_hidden_by_prompt_position,
                final_trace,
                runtime: PlaneRuntimeEvidence {
                    plane: "EXACT_F32_CPU",
                    runtime_mode: "RealCliRuntimeMode::IsolatedGreedyParityCpu",
                    isolated_runtime: true,
                    kv_or_session_state_shared_with_another_plane: false,
                    caller_owned_prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(
                        &token_ids,
                    ),
                    per_plane_retokenization_performed: false,
                    resolved_config_sha256: observed_config_sha256,
                    execution_plan,
                    model_load,
                    boundary_before: Some(boundary_before),
                    boundary_after: Some(boundary_after),
                    routed_execution_after: runtime.engine.routed_expert_execution_snapshot(),
                    gpu_token_loop_after: None,
                    prompt_token_steps_completed: token_ids.len(),
                    generated_token_continuation_steps: 0,
                    background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(
                    ),
                },
            },
            router,
            expert_boundary,
            same_input_rmsnorm,
            same_input_lm_head,
            downstream,
        })
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut capture), Ok(shutdown)) => {
            capture.plane.runtime.background_shutdown = shutdown;
            Ok(capture)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; EXACT_F32_CPU shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

fn prefill_summaries(
    token_ids: &[u32],
    exact: &[Vec<f32>],
    historical: &[Vec<f32>],
    gpu: &[Vec<f32>],
) -> Result<Vec<PrefillPositionSummary>, String> {
    if exact.len() != token_ids.len()
        || historical.len() != token_ids.len()
        || gpu.len() != token_ids.len()
    {
        return Err("three-plane prefill summary position counts differ".into());
    }
    token_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(position, token_id)| {
            Ok(PrefillPositionSummary {
                position,
                token_id,
                final_hidden: three_way_compare(
                    "prefill-final-hidden",
                    &exact[position],
                    &historical[position],
                    &gpu[position],
                )?,
            })
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProductionSemanticsEvidence {
    pub diagnostic_only: bool,
    pub production_inference_changed: bool,
    pub production_q4_changed: bool,
    pub production_q4_wgsl_changed: bool,
    pub production_router_gemv_changed: bool,
    pub production_attention_changed: bool,
    pub production_dense_gemv_changed: bool,
    pub production_rmsnorm_changed: bool,
    pub production_lm_head_changed: bool,
    pub production_greedy_argmax_changed: bool,
    pub production_residency_replay_or_prefetch_changed: bool,
    pub v1_changed: bool,
    pub v2_changed: bool,
    pub limits_corpus_or_prompts_changed: bool,
    pub numerical_threshold_introduced: bool,
    pub production_correction_justified: bool,
}

impl Default for ProductionSemanticsEvidence {
    fn default() -> Self {
        Self {
            diagnostic_only: true,
            production_inference_changed: false,
            production_q4_changed: false,
            production_q4_wgsl_changed: false,
            production_router_gemv_changed: false,
            production_attention_changed: false,
            production_dense_gemv_changed: false,
            production_rmsnorm_changed: false,
            production_lm_head_changed: false,
            production_greedy_argmax_changed: false,
            production_residency_replay_or_prefetch_changed: false,
            v1_changed: false,
            v2_changed: false,
            limits_corpus_or_prompts_changed: false,
            numerical_threshold_introduced: false,
            production_correction_justified: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DiagnosticProvenance {
    pub build: crate::qualification::BuildProvenance,
    pub executable_sha256: String,
    pub artifacts: crate::qualification::QualificationArtifacts,
    pub source_config: crate::gpu_native_greedy_parity::SourceConfigEvidence,
    pub model_identity: crate::greedy_parity::ModelIdentityEvidence,
    pub expert_metadata: crate::qualification::ExpertMetadataEvidence,
    pub expected_adapter_name: String,
    pub actual_adapter: crate::backend::GpuDeviceIdentity,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpanishFirstTokenAttributionReport {
    pub schema: &'static str,
    pub mode: &'static str,
    pub diagnostic_only: bool,
    pub diagnostic_complete: bool,
    qualification_pass: bool,
    pub failure: Option<String>,
    pub frozen_evidence_preflight: FrozenEvidencePreflight,
    pub target: FrozenSpanishTargetIdentity,
    pub shared_tokenization: SharedTokenizationEvidence,
    pub provenance: DiagnosticProvenance,
    pub runtimes: Vec<PlaneRuntimeEvidence>,
    pub prefill_position_summaries: Vec<PrefillPositionSummary>,
    pub final_position_full_traces: Vec<PlaneFullTraceEvidence>,
    pub final_position_three_way_comparison: FinalPositionTraceComparison,
    pub same_input_router_analysis: RouterAttributionEvidence,
    pub same_input_expert_boundary: SameInputExpertBoundaryEvidence,
    pub same_input_final_rmsnorm: SameInputFinalRmsnormEvidence,
    pub same_input_lm_head: SameInputLmHeadEvidence,
    pub historical_f16_compensation_analysis: HistoricalF16CompensationAnalysis,
    pub attribution: ConservativeAttributionEvidence,
    pub production_semantics: ProductionSemanticsEvidence,
    pub positions_after_generated_position_zero_inspected: bool,
    pub scientific_interpretation: Vec<String>,
}

impl SpanishFirstTokenAttributionReport {
    pub const fn qualification_pass(&self) -> bool {
        self.qualification_pass
    }
}

fn emit_report(
    report: &SpanishFirstTokenAttributionReport,
    report_out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if report.qualification_pass()
        || !report.diagnostic_only
        || !report.diagnostic_complete
        || report.failure.is_some()
    {
        return Err("Spanish attribution report must remain complete, diagnostic-only, and qualification false".into());
    }
    if let Some(parent) = report_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    std::fs::write(report_out, json)?;
    eprintln!(
        "GPU-native Spanish first-token attribution report written to {}",
        report_out.display()
    );
    Ok(())
}

fn validate_first_token_scope(
    prompt_token_count: usize,
    runtimes: &[PlaneRuntimeEvidence],
) -> Result<(), String> {
    if runtimes.len() != 3
        || runtimes.iter().any(|runtime| {
            runtime.prompt_token_steps_completed != prompt_token_count
                || runtime.generated_token_continuation_steps != 0
        })
    {
        return Err("three-plane execution must stop immediately after first-token argmax".into());
    }
    Ok(())
}

fn validate_three_plane_isolation(
    runtimes: &[PlaneRuntimeEvidence],
    prompt_token_ids_sha256: &str,
) -> Result<(), String> {
    let expected = ["EXACT_F32_CPU", "HISTORICAL_F16_CPU", "GPU_NATIVE"];
    if runtimes.len() != expected.len()
        || runtimes
            .iter()
            .zip(expected)
            .any(|(runtime, plane)| runtime.plane != plane)
        || runtimes.iter().any(|runtime| {
            !runtime.isolated_runtime
                || runtime.kv_or_session_state_shared_with_another_plane
                || runtime.per_plane_retokenization_performed
                || runtime.caller_owned_prompt_token_ids_sha256 != prompt_token_ids_sha256
                || runtime.execution_plan.context_id.is_empty()
        })
        || runtimes
            .iter()
            .map(|runtime| runtime.execution_plan.context_id.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != expected.len()
    {
        return Err("the exact, historical, and GPU planes were not independently isolated".into());
    }
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
    boundary_audit_report: PathBuf,
    expected_boundary_audit_report_sha256: String,
    expected_adapter_name: String,
    report_out: PathBuf,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::BuildProvenance;

    // Immutable historical evidence validation is deliberately the first
    // operation, before tokenizer, model, engine, device, or runtime creation.
    let frozen = FrozenInputs::read_and_validate(
        &v2_report,
        &expected_v2_report_sha256,
        &stage_report,
        &expected_stage_report_sha256,
        &boundary_audit_report,
        &expected_boundary_audit_report_sha256,
    )?;
    if output_conflicts_with_input(&report_out, &frozen.identity)? {
        return Err("report output must not overwrite a frozen historical input".into());
    }
    if progress_watchdog.timeout.is_none() {
        return Err("Spanish first-token attribution requires a positive progress timeout".into());
    }
    if expected_adapter_name != REQUIRED_ADAPTER_NAME {
        return Err(format!(
            "Spanish first-token attribution requires --expected-adapter-name {REQUIRED_ADAPTER_NAME:?}"
        )
        .into());
    }
    let build = BuildProvenance::embedded();
    if build.dirty != Some(false) || build.git_sha.as_deref().is_none_or(|sha| !is_hex(sha, 40)) {
        return Err(
            "Spanish first-token attribution requires clean embedded Git provenance".into(),
        );
    }
    let (artifacts, artifact_errors) = crate::qualification_artifacts(&config, &cfg);
    if !artifact_errors.is_empty() {
        return Err(format!(
            "Spanish first-token attribution artifact preflight failed: {}",
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
            "Spanish first-token attribution requires strict GPU-native Q4 configuration".into(),
        );
    }
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| format!("Spanish attribution expert metadata failed: {error}"))?;
    if expert_metadata.dtype.as_deref() != Some("q4_0")
        || expert_metadata.q4_0_layout.as_deref() != Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
        || expert_metadata.explicitly_synthetic
    {
        return Err(
            "Spanish first-token attribution requires canonical nonsynthetic Q4_0 metadata".into(),
        );
    }

    let mut gpu_spec = crate::resolve_real_cli_spec_from_config(
        cfg,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
    )?;
    gpu_spec.cfg.real_transformer.gpu_native = true;
    gpu_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Gpu;
    let model_identity = crate::greedy_parity_model_identity(&gpu_spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err(
            "Spanish first-token attribution requires exact Qwen3-Coder 30B-A3B Q4_0 geometry"
                .into(),
        );
    }
    let gpu_resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&gpu_spec)?;
    let mut cpu_spec = gpu_spec.clone();
    cpu_spec.cfg.real_transformer.gpu_native = false;
    cpu_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Cpu;
    let cpu_resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&cpu_spec)?;
    let tokenizer_path = gpu_spec
        .cfg
        .tokenizer
        .path
        .as_ref()
        .ok_or("Spanish first-token attribution requires tokenizer.path")?;
    let tokenizer = Arc::new(crate::tokenizer::Tokenizer::from_file(tokenizer_path)?);
    let (token_ids, shared_tokenization) = shared_tokenization(&tokenizer)?;
    let [gpu_token_ids, historical_token_ids, exact_token_ids] = shared_plane_token_ids(&token_ids);

    let gpu = execute_gpu_plane(
        &gpu_spec,
        tokenizer.clone(),
        gpu_token_ids,
        &gpu_resolved_config_sha256,
        &expected_adapter_name,
        progress_watchdog,
    )
    .await?;
    let historical = execute_historical_plane(
        &cpu_spec,
        tokenizer.clone(),
        historical_token_ids,
        &cpu_resolved_config_sha256,
        &gpu.final_semantic_trace,
        progress_watchdog,
    )
    .await?;
    let exact = execute_exact_plane(
        &cpu_spec,
        tokenizer,
        exact_token_ids,
        &cpu_resolved_config_sha256,
        &historical.plane.final_trace,
        &historical.same_input_expert_outputs,
        &gpu,
        progress_watchdog,
    )
    .await?;

    let prefill_position_summaries = prefill_summaries(
        &token_ids,
        &exact.plane.final_hidden_by_prompt_position,
        &historical.plane.final_hidden_by_prompt_position,
        &gpu.final_hidden_by_prompt_position,
    )?;
    let final_position_three_way_comparison = compare_final_traces(
        &exact.plane.final_trace,
        &historical.plane.final_trace,
        &gpu.final_trace,
        &gpu.final_semantic_trace,
    )?;
    let historical_f16_compensation_analysis =
        compensation_analysis(&final_position_three_way_comparison);
    let attribution = derive_attribution(
        exact.plane.final_trace.sampled_token,
        historical.plane.final_trace.sampled_token,
        gpu.final_trace.sampled_token,
        &exact.downstream,
        &exact.router,
    );
    let full_traces = vec![
        cpu_plane_full_trace("EXACT_F32_CPU", &exact.plane.final_trace)?,
        cpu_plane_full_trace("HISTORICAL_F16_CPU", &historical.plane.final_trace)?,
        gpu_plane_full_trace(&gpu.final_trace, &gpu.final_semantic_trace)?,
    ];
    let runtimes = vec![exact.plane.runtime, historical.plane.runtime, gpu.runtime];
    validate_first_token_scope(token_ids.len(), &runtimes)?;
    validate_three_plane_isolation(&runtimes, &shared_tokenization.prompt_token_ids_sha256)?;
    if runtimes.iter().any(|runtime| {
        !runtime.background_shutdown.controlled_shutdown_requested
            || !runtime.background_shutdown.all_runtime_resources_released
            || runtime.caller_owned_prompt_token_ids_sha256
                != shared_tokenization.prompt_token_ids_sha256
    }) {
        return Err(
            "one or more attribution runtimes violated shutdown/tokenization/scope contracts"
                .into(),
        );
    }
    let (_, executable_sha256) = crate::current_executable_identity()?;
    let report = SpanishFirstTokenAttributionReport {
        schema: SCHEMA_VERSION,
        mode: MODE,
        diagnostic_only: true,
        diagnostic_complete: true,
        qualification_pass: QUALIFICATION_PASS,
        failure: None,
        frozen_evidence_preflight: frozen.identity,
        target: FrozenSpanishTargetIdentity::default(),
        shared_tokenization,
        provenance: DiagnosticProvenance {
            build,
            executable_sha256,
            artifacts,
            source_config,
            model_identity,
            expert_metadata,
            expected_adapter_name,
            actual_adapter: gpu.device,
        },
        runtimes,
        prefill_position_summaries,
        final_position_full_traces: full_traces,
        final_position_three_way_comparison,
        same_input_router_analysis: exact.router,
        same_input_expert_boundary: exact.expert_boundary,
        same_input_final_rmsnorm: exact.same_input_rmsnorm,
        same_input_lm_head: exact.same_input_lm_head,
        historical_f16_compensation_analysis,
        attribution,
        production_semantics: ProductionSemanticsEvidence::default(),
        positions_after_generated_position_zero_inspected: false,
        scientific_interpretation: vec![
            "the exact-f32 CPU, historical-f16 CPU, and production GPU-native planes were isolated and consumed one caller-owned frozen tokenization".into(),
            "historical-f16 proximity is descriptive compensation evidence and does not make the old f16 boundary the correct reference".into(),
            "no production correction is justified by this diagnostic branch alone".into(),
        ],
    };
    emit_report(&report, &report_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_v2() -> FrozenV2Envelope {
        let token_cases = crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS
            .into_iter()
            .map(|case| {
                let mut reference = vec![11; 16];
                let mut gpu = reference.clone();
                if case.name == TARGET_CASE {
                    reference[0] = HISTORICAL_F16_CPU_TOKEN;
                    gpu[0] = FROZEN_GPU_TOKEN;
                }
                let comparison = crate::gpu_native_semantic_parity_corpus::TokenCaseEvidence::new(
                    case.name,
                    reference.clone(),
                    gpu.clone(),
                );
                FrozenTokenCaseEnvelope {
                    case: case.name.into(),
                    reference_generated_token_ids: reference,
                    gpu_generated_token_ids: gpu,
                    exact_match_count: comparison.exact_match_count,
                    mismatch_count: comparison.mismatch_count,
                    first_mismatch_position: comparison.first_mismatch_position,
                }
            })
            .collect();
        FrozenV2Envelope {
            schema: crate::gpu_native_semantic_parity_v2::SCHEMA_VERSION.into(),
            qualification_pass: false,
            provenance: FrozenProvenanceEnvelope {
                build: FrozenBuildEnvelope {
                    git_sha: Some(FROZEN_V2_BUILD_SHA.into()),
                },
            },
            holdout_corpus: FrozenCorpusEnvelope {
                id: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_ID.into(),
                version: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_VERSION,
                sha256: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_SHA256.into(),
                case_count: 4,
                output_token_limit: 16,
            },
            numerical_limits: FrozenNumericalLimitsEnvelope {
                max_absolute_error_limit:
                    crate::gpu_native_semantic_parity_v2::MAX_ABSOLUTE_ERROR_LIMIT,
                rms_error_limit: crate::gpu_native_semantic_parity_v2::RMS_ERROR_LIMIT,
                mean_absolute_error_limit:
                    crate::gpu_native_semantic_parity_v2::MEAN_ABSOLUTE_ERROR_LIMIT,
                nonfinite_mismatch_limit:
                    crate::gpu_native_semantic_parity_v2::NONFINITE_MISMATCH_LIMIT,
                semantic_correctness_not_bit_parity: true,
            },
            token_cases,
        }
    }

    fn valid_stage() -> FrozenStageEnvelope {
        let targets = expected_stage_targets();
        FrozenStageEnvelope {
            schema: crate::gpu_native_q4_expert_stage_attribution::SCHEMA_VERSION.into(),
            diagnostic_complete: true,
            qualification_pass: false,
            failure: None,
            provenance: FrozenProvenanceEnvelope {
                build: FrozenBuildEnvelope {
                    git_sha: Some(FROZEN_STAGE_BUILD_SHA.into()),
                },
            },
            frozen_targets: targets.clone(),
            targets: targets
                .into_iter()
                .map(|target| FrozenStageTargetEnvelope {
                    target,
                    experts: vec![serde_json::Value::Null; 8],
                    failure: None,
                })
                .collect(),
        }
    }

    fn token_case(case: &str) -> FrozenCpuTokenCaseEnvelope {
        let spanish = case == TARGET_CASE;
        let corrected = if spanish {
            vec![EXACT_F32_CPU_TOKEN; 16]
        } else if case == "rust-ownership-holdout" {
            let mut values = vec![11; 16];
            values[1] = 8_822;
            values
        } else {
            vec![11; 16]
        };
        let gpu = if spanish {
            vec![FROZEN_GPU_TOKEN; 16]
        } else {
            corrected.clone()
        };
        let historical = if spanish {
            vec![HISTORICAL_F16_CPU_TOKEN; 16]
        } else if case == "rust-ownership-holdout" {
            let mut values = gpu.clone();
            values[1] = 785;
            values
        } else {
            gpu.clone()
        };
        FrozenCpuTokenCaseEnvelope {
            case: case.into(),
            requested_tokens: 16,
            completed_tokens: 16,
            exact_token_matches: if spanish { 0 } else { 16 },
            mismatch_count: if spanish { 16 } else { 0 },
            first_mismatch_position: spanish.then_some(0),
            corrected_ordinary_f32_cpu_token_ids: corrected,
            frozen_gpu_token_ids: gpu,
            historical_old_boundary_cpu_reference_token_ids: historical,
        }
    }

    fn valid_boundary() -> FrozenBoundaryEnvelope {
        FrozenBoundaryEnvelope {
            schema: crate::gpu_native_f32_reference_boundary_audit::SCHEMA_VERSION.into(),
            diagnostic_only: true,
            diagnostic_complete: true,
            qualification_pass: false,
            failure: None,
            provenance: FrozenProvenanceEnvelope {
                build: FrozenBuildEnvelope {
                    git_sha: Some(FROZEN_BOUNDARY_AUDIT_BUILD_SHA.into()),
                },
            },
            frozen_inputs: FrozenBoundaryInputsEnvelope {
                v2_report: FrozenArtifactEnvelope {
                    sha256: FROZEN_V2_REPORT_SHA256.into(),
                },
                stage_report: FrozenArtifactEnvelope {
                    sha256: FROZEN_STAGE_REPORT_SHA256.into(),
                },
                exact_expert_identity_count: 72,
                immutable_inputs_verified_before_runtime: true,
            },
            expert_records: vec![
                FrozenBoundaryExpertEnvelope {
                    improvement: FrozenImprovementEnvelope {
                        exact_f32_error_smaller_than_old_f16_error_maxabs: true,
                        exact_f32_error_smaller_than_old_f16_error_rms: true,
                        exact_f32_error_smaller_than_old_f16_error_meanabs: true,
                    },
                };
                72
            ],
            part_a_aggregation: FrozenPartAAggregationEnvelope {
                expert_record_count: 72,
                exact_f32_smaller_than_old_f16_maxabs_count: 72,
                exact_f32_smaller_than_old_f16_rms_count: 72,
                exact_f32_smaller_than_old_f16_meanabs_count: 72,
            },
            corrected_cpu_token_cases: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS
                .into_iter()
                .map(|case| token_case(case.name))
                .collect(),
            rust_ownership_position_one: FrozenRustPositionOneEnvelope {
                historical_old_cpu_token_id: 785,
                frozen_gpu_token_id: 8_822,
                new_ordinary_f32_cpu_token_id: 8_822,
            },
            part_b_token_summary: FrozenPartBEnvelope {
                requested_tokens: 64,
                completed_tokens: 64,
                exact_token_matches: 48,
                mismatch_count: 16,
            },
            production_semantics: FrozenProductionSemanticsEnvelope {
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
                production_correction_made_or_justified: false,
            },
        }
    }

    fn gpu_native_legacy_plan() -> crate::qualification::ExecutionPlanEvidence {
        crate::qualification::ExecutionPlanEvidence {
            context_id: "gpu-native-authoritative".into(),
            requested: "gpu".into(),
            resolved: "gpu".into(),
            embeddings: "cpu".into(),
            lm_head: "cpu".into(),
            dense_projections: "cpu".into(),
            attention: "gpu".into(),
            kv: "gpu".into(),
            router: "cpu".into(),
            routed_experts: "cpu".into(),
            routed_expert_dtype: "q4_0".into(),
            fallback_occurred: false,
            reason: None,
        }
    }

    fn gpu_native_geometry() -> crate::gpu_native_token_loop::GpuNativeModelGeometry {
        crate::gpu_native_token_loop::GpuNativeModelGeometry {
            num_layers: EXPECTED_NUM_LAYERS,
            d_model: EXPECTED_D_MODEL,
            d_ff: EXPECTED_D_FF,
            num_experts: EXPECTED_NUM_EXPERTS,
            top_k: EXPECTED_TOP_K,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: EXPECTED_D_MODEL,
            rope_dim: EXPECTED_D_MODEL,
            vocab_size: 151_936,
            max_seq_len: 32,
            rms_eps: 1e-6,
            rope_base: 10_000.0,
        }
    }

    fn strict_model_load() -> crate::greedy_parity::ModelLoadEvidence {
        crate::greedy_parity::ModelLoadEvidence {
            strict: true,
            loader: "safetensors".into(),
            loaded_tensors: 435,
            required_tensors: 435,
            optional_probed: 0,
            optional_loaded: 0,
            seeded_fallback_remained: false,
        }
    }

    fn nvidia_l4_device() -> crate::backend::GpuDeviceIdentity {
        crate::backend::GpuDeviceIdentity {
            name: REQUIRED_ADAPTER_NAME.into(),
            vendor_id: 0x10de,
            device_id: 0,
            device_type: "DiscreteGpu".into(),
            wgpu_backend: "vulkan".into(),
            driver: "580.173.02".into(),
            driver_info: "test".into(),
            compute_plane: "wgpu-vulkan".into(),
            software_adapter: false,
        }
    }

    fn gpu_native_authoritative_runtime() -> GpuNativeAuthoritativeRuntimeEvidence {
        GpuNativeAuthoritativeRuntimeEvidence {
            legacy_execution_context: gpu_native_legacy_plan(),
            gpu_native_token_loop_geometry: Some(gpu_native_geometry()),
            authoritative_device: Some(nvidia_l4_device()),
            model_load: strict_model_load(),
            real_transformer_gpu_native: true,
            compute_offload: crate::backend::ComputeOffload::Gpu,
        }
    }

    fn runtime_evidence(plane: &'static str, prompt_steps: usize) -> PlaneRuntimeEvidence {
        PlaneRuntimeEvidence {
            plane,
            runtime_mode: "isolated",
            isolated_runtime: true,
            kv_or_session_state_shared_with_another_plane: false,
            caller_owned_prompt_token_ids_sha256: "a".repeat(64),
            per_plane_retokenization_performed: false,
            resolved_config_sha256: "b".repeat(64),
            execution_plan: crate::qualification::ExecutionPlanEvidence {
                context_id: plane.into(),
                requested: "cpu".into(),
                resolved: "cpu".into(),
                embeddings: "cpu".into(),
                lm_head: "cpu".into(),
                dense_projections: "cpu".into(),
                attention: "cpu".into(),
                kv: "cpu".into(),
                router: "cpu".into(),
                routed_experts: "cpu".into(),
                routed_expert_dtype: "q4_0".into(),
                fallback_occurred: false,
                reason: None,
            },
            model_load: crate::greedy_parity::ModelLoadEvidence {
                strict: true,
                loader: "test".into(),
                loaded_tensors: 1,
                required_tensors: 1,
                optional_probed: 0,
                optional_loaded: 0,
                seeded_fallback_remained: false,
            },
            boundary_before: None,
            boundary_after: None,
            routed_execution_after: crate::engine::RoutedExpertExecutionSnapshot::default(),
            gpu_token_loop_after: None,
            prompt_token_steps_completed: prompt_steps,
            generated_token_continuation_steps: 0,
            background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(),
        }
    }

    #[test]
    fn gpu_native_authoritative_runtime_accepts_factory_legacy_plan_and_separate_token_loop() {
        let runtime = gpu_native_authoritative_runtime();
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &runtime,
            REQUIRED_ADAPTER_NAME
        )
        .is_ok());
        assert_eq!(runtime.legacy_execution_context.embeddings, "cpu");
        assert_eq!(runtime.legacy_execution_context.lm_head, "cpu");
        assert_eq!(runtime.legacy_execution_context.dense_projections, "cpu");
        assert_eq!(runtime.legacy_execution_context.router, "cpu");
        assert_eq!(runtime.legacy_execution_context.routed_experts, "cpu");
        assert!(runtime.gpu_native_token_loop_geometry.is_some());
    }

    #[test]
    fn previous_all_gpu_execution_plan_assumption_is_rejected_with_observed_evidence() {
        let mut runtime = gpu_native_authoritative_runtime();
        runtime.legacy_execution_context.embeddings = "gpu".into();
        runtime.legacy_execution_context.lm_head = "gpu".into();
        runtime.legacy_execution_context.dense_projections = "gpu".into();
        runtime.legacy_execution_context.router = "gpu".into();
        runtime.legacy_execution_context.routed_experts = "gpu".into();
        let error =
            validate_gpu_native_authoritative_runtime_contract(&runtime, REQUIRED_ADAPTER_NAME)
                .unwrap_err();
        assert!(error.contains("observed=ExecutionPlanEvidence"));
        assert!(error.contains("legacy execution-context evidence"));
        assert!(error.contains("gpu_native_token_loop is validated separately"));
    }

    #[test]
    fn gpu_native_legacy_plan_rejects_wrong_mode_and_fallback() {
        for (requested, resolved) in [("cpu", "gpu"), ("gpu", "cpu")] {
            let mut runtime = gpu_native_authoritative_runtime();
            runtime.legacy_execution_context.requested = requested.into();
            runtime.legacy_execution_context.resolved = resolved.into();
            assert!(validate_gpu_native_authoritative_runtime_contract(
                &runtime,
                REQUIRED_ADAPTER_NAME
            )
            .is_err());
        }
        let mut runtime = gpu_native_authoritative_runtime();
        runtime.legacy_execution_context.fallback_occurred = true;
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &runtime,
            REQUIRED_ADAPTER_NAME
        )
        .is_err());
    }

    #[test]
    fn gpu_native_legacy_plan_rejects_cpu_attention_or_kv() {
        let mut attention = gpu_native_authoritative_runtime();
        attention.legacy_execution_context.attention = "cpu".into();
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &attention,
            REQUIRED_ADAPTER_NAME
        )
        .is_err());

        let mut kv = gpu_native_authoritative_runtime();
        kv.legacy_execution_context.kv = "cpu".into();
        assert!(
            validate_gpu_native_authoritative_runtime_contract(&kv, REQUIRED_ADAPTER_NAME).is_err()
        );
    }

    #[test]
    fn gpu_native_legacy_plan_rejects_wrong_dtype_or_active_legacy_expert_registry() {
        let mut dtype = gpu_native_authoritative_runtime();
        dtype.legacy_execution_context.routed_expert_dtype = "f16".into();
        assert!(
            validate_gpu_native_authoritative_runtime_contract(&dtype, REQUIRED_ADAPTER_NAME)
                .is_err()
        );

        let mut legacy_registry = gpu_native_authoritative_runtime();
        legacy_registry.legacy_execution_context.routed_experts = "gpu".into();
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &legacy_registry,
            REQUIRED_ADAPTER_NAME
        )
        .is_err());
    }

    #[test]
    fn gpu_native_authoritative_runtime_requires_token_loop_and_exact_geometry() {
        let mut missing = gpu_native_authoritative_runtime();
        missing.gpu_native_token_loop_geometry = None;
        let error =
            validate_gpu_native_authoritative_runtime_contract(&missing, REQUIRED_ADAPTER_NAME)
                .unwrap_err();
        assert!(error.contains("requires gpu_native_token_loop"));

        let mut wrong = gpu_native_authoritative_runtime();
        wrong.gpu_native_token_loop_geometry.as_mut().unwrap().d_ff += 1;
        assert!(
            validate_gpu_native_authoritative_runtime_contract(&wrong, REQUIRED_ADAPTER_NAME)
                .is_err()
        );
    }

    #[test]
    fn gpu_native_authoritative_runtime_requires_exact_hardware_vulkan_identity() {
        let mut missing = gpu_native_authoritative_runtime();
        missing.authoritative_device = None;
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &missing,
            REQUIRED_ADAPTER_NAME
        )
        .is_err());

        let mut wrong_name = gpu_native_authoritative_runtime();
        wrong_name.authoritative_device.as_mut().unwrap().name = "other".into();
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &wrong_name,
            REQUIRED_ADAPTER_NAME
        )
        .is_err());

        let mut software = gpu_native_authoritative_runtime();
        software
            .authoritative_device
            .as_mut()
            .unwrap()
            .software_adapter = true;
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &software,
            REQUIRED_ADAPTER_NAME
        )
        .is_err());

        let mut wrong_backend = gpu_native_authoritative_runtime();
        wrong_backend
            .authoritative_device
            .as_mut()
            .unwrap()
            .wgpu_backend = "metal".into();
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &wrong_backend,
            REQUIRED_ADAPTER_NAME
        )
        .is_err());
    }

    #[test]
    fn gpu_native_authoritative_runtime_requires_gpu_native_config_and_strict_model() {
        let mut wrong_config = gpu_native_authoritative_runtime();
        wrong_config.real_transformer_gpu_native = false;
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &wrong_config,
            REQUIRED_ADAPTER_NAME
        )
        .is_err());

        let mut wrong_offload = gpu_native_authoritative_runtime();
        wrong_offload.compute_offload = crate::backend::ComputeOffload::Cpu;
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &wrong_offload,
            REQUIRED_ADAPTER_NAME
        )
        .is_err());

        let mut non_strict = gpu_native_authoritative_runtime();
        non_strict.model_load.strict = false;
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &non_strict,
            REQUIRED_ADAPTER_NAME
        )
        .is_err());

        let mut incomplete = gpu_native_authoritative_runtime();
        incomplete.model_load.loaded_tensors -= 1;
        assert!(validate_gpu_native_authoritative_runtime_contract(
            &incomplete,
            REQUIRED_ADAPTER_NAME
        )
        .is_err());

        let mut seeded = gpu_native_authoritative_runtime();
        seeded.model_load.seeded_fallback_remained = true;
        assert!(
            validate_gpu_native_authoritative_runtime_contract(&seeded, REQUIRED_ADAPTER_NAME)
                .is_err()
        );
    }

    #[test]
    fn exact_and_historical_cpu_plane_contracts_remain_cpu_only() {
        for plane in ["EXACT_F32_CPU", "HISTORICAL_F16_CPU"] {
            let runtime = runtime_evidence(plane, 1);
            assert!(crate::greedy_parity::cpu_plan_exact(
                &runtime.execution_plan
            ));
            assert!(runtime.gpu_token_loop_after.is_none());
        }
    }

    #[test]
    fn schema_mode_target_and_diagnostic_contract_are_frozen() {
        assert_eq!(
            SCHEMA_VERSION,
            "mer.gpu-native-spanish-first-token-attribution.v1"
        );
        assert_eq!(MODE, "diagnose-gpu-native-spanish-first-token-attribution");
        assert!(ProductionSemanticsEvidence::default().diagnostic_only);
        assert!(!QUALIFICATION_PASS);
        assert_eq!(FrozenSpanishTargetIdentity::default().case, TARGET_CASE);
        assert_eq!(FrozenSpanishTargetIdentity::default().generated_position, 0);
        assert_eq!(
            FrozenSpanishTargetIdentity::default().historical_f16_boundary_cpu_token,
            HISTORICAL_F16_CPU_TOKEN
        );
        assert_eq!(
            FrozenSpanishTargetIdentity::default().frozen_gpu_token,
            FROZEN_GPU_TOKEN
        );
        assert_eq!(
            FrozenSpanishTargetIdentity::default().corrected_exact_f32_cpu_token,
            140_003
        );
    }

    #[test]
    fn exact_three_report_preflight_envelopes_fail_closed() {
        assert!(validate_v2(&valid_v2()).is_ok());
        assert!(validate_stage(&valid_stage()).is_ok());
        assert!(validate_boundary(&valid_boundary()).is_ok());

        let mut v2 = valid_v2();
        v2.token_cases
            .iter_mut()
            .find(|case| case.case == TARGET_CASE)
            .unwrap()
            .gpu_generated_token_ids[0] = 1;
        assert!(validate_v2(&v2).is_err());
        let mut v2 = valid_v2();
        v2.numerical_limits.max_absolute_error_limit = 1.0;
        assert!(validate_v2(&v2).is_err());
        let mut stage = valid_stage();
        stage.targets[0].experts.pop();
        assert!(validate_stage(&stage).is_err());
        let mut boundary = valid_boundary();
        boundary.expert_records[0]
            .improvement
            .exact_f32_error_smaller_than_old_f16_error_rms = false;
        assert!(validate_boundary(&boundary).is_err());
    }

    #[test]
    fn frozen_sha_arguments_are_literal() {
        assert!(validate_expected_sha(
            FROZEN_BOUNDARY_AUDIT_REPORT_SHA256,
            FROZEN_BOUNDARY_AUDIT_REPORT_SHA256,
            "boundary"
        )
        .is_ok());
        assert!(validate_expected_sha(
            &"0".repeat(64),
            FROZEN_BOUNDARY_AUDIT_REPORT_SHA256,
            "boundary"
        )
        .is_err());
    }

    #[test]
    fn snapshot_hash_validation_uses_one_byte_identity() {
        let bytes = b"one-immutable-snapshot";
        let sha = crate::greedy_parity::sha256_hex(bytes);
        let artifact = crate::qualification::ArtifactDigest {
            configured_path: "report.json".into(),
            canonical_path: "/report.json".into(),
            byte_length: bytes.len() as u64,
            sha256: sha.clone(),
        };
        assert!(validate_artifact_hash(&artifact, &sha, &sha, "report").is_ok());
        assert!(validate_artifact_hash(&artifact, &"0".repeat(64), &sha, "report").is_err());
    }

    #[test]
    fn caller_owned_token_ids_are_shared_by_all_three_planes() {
        let ids = Arc::new(vec![1, 2, 3]);
        let planes = shared_plane_token_ids(&ids);
        assert!(planes.iter().all(|plane| Arc::ptr_eq(&ids, plane)));
        assert_eq!(
            crate::greedy_parity::token_ids_sha256(&planes[0]),
            crate::greedy_parity::token_ids_sha256(&planes[2])
        );
    }

    #[test]
    fn exact_f32_boundary_must_remain_disabled_and_zero() {
        assert!(require_exact_boundary_clean(
            crate::engine::CpuQ4BoundaryEmulationSnapshot::default()
        )
        .is_ok());
        assert!(
            require_exact_boundary_clean(crate::engine::CpuQ4BoundaryEmulationSnapshot {
                enabled: true,
                routed_expert_dispatches: 0,
            })
            .is_err()
        );
    }

    #[test]
    fn historical_f16_boundary_must_start_explicitly_enabled_and_zero() {
        assert!(
            require_historical_boundary_clean(crate::engine::CpuQ4BoundaryEmulationSnapshot {
                enabled: true,
                routed_expert_dispatches: 0,
            })
            .is_ok()
        );
        assert!(require_historical_boundary_clean(
            crate::engine::CpuQ4BoundaryEmulationSnapshot::default()
        )
        .is_err());
    }

    #[test]
    fn three_plane_scope_stops_after_first_argmax() {
        let mut runtimes = vec![
            runtime_evidence("EXACT_F32_CPU", 4),
            runtime_evidence("HISTORICAL_F16_CPU", 4),
            runtime_evidence("GPU_NATIVE", 4),
        ];
        let shared_sha = "a".repeat(64);
        assert!(validate_first_token_scope(4, &runtimes).is_ok());
        assert!(validate_three_plane_isolation(&runtimes, &shared_sha).is_ok());
        runtimes[1].execution_plan.context_id = runtimes[0].execution_plan.context_id.clone();
        assert!(validate_three_plane_isolation(&runtimes, &shared_sha).is_err());
        let mut bad = runtimes;
        bad[2].generated_token_continuation_steps = 1;
        assert!(validate_first_token_scope(4, &bad).is_err());
    }

    #[test]
    fn same_input_router_stage_is_ordered_and_explicit() {
        assert_eq!(
            derive_same_input_router_stage(&[1, 2], &[1, 2], &[1, 2]),
            SameInputRouterStage::ExactOrderedMatch
        );
        assert_eq!(
            derive_same_input_router_stage(&[1, 2], &[2, 1], &[2, 1]),
            SameInputRouterStage::GpuRouterGemvDrift
        );
        assert_eq!(
            derive_same_input_router_stage(&[1, 2], &[1, 2], &[2, 1]),
            SameInputRouterStage::GpuRouterSoftmaxTopkDrift
        );
    }

    #[test]
    fn same_input_rmsnorm_and_lm_head_interventions_drive_discrete_categories() {
        let router = RouterAttributionEvidence {
            layers: Vec::new(),
            first_same_input_ordered_id_mismatch_layer: None,
            known_postgres_defect_preserved_as_separate: true,
            known_postgres_defect: "separate",
        };
        let rms = derive_attribution(
            EXACT_F32_CPU_TOKEN,
            FROZEN_GPU_TOKEN,
            FROZEN_GPU_TOKEN,
            &DownstreamCounterfactuals {
                cpu_argmax_on_cpu_norm_of_gpu_hidden: EXACT_F32_CPU_TOKEN,
                cpu_argmax_on_actual_gpu_norm: FROZEN_GPU_TOKEN,
                deterministic_argmax_on_actual_gpu_logits: FROZEN_GPU_TOKEN,
            },
            &router,
        );
        assert!(rms
            .supporting_classifications
            .contains(&AttributionCategory::FinalRmsnormDrift));
        let lm = derive_attribution(
            EXACT_F32_CPU_TOKEN,
            FROZEN_GPU_TOKEN,
            FROZEN_GPU_TOKEN,
            &DownstreamCounterfactuals {
                cpu_argmax_on_cpu_norm_of_gpu_hidden: EXACT_F32_CPU_TOKEN,
                cpu_argmax_on_actual_gpu_norm: EXACT_F32_CPU_TOKEN,
                deterministic_argmax_on_actual_gpu_logits: FROZEN_GPU_TOKEN,
            },
            &router,
        );
        assert!(lm
            .supporting_classifications
            .contains(&AttributionCategory::LmHeadGemvDrift));
    }

    #[test]
    fn explicit_spanish_candidate_view_contains_140003_and_54275() {
        let mut logits = vec![-10.0; EXACT_F32_CPU_TOKEN as usize + 1];
        logits[EXACT_F32_CPU_TOKEN as usize] = 2.0;
        logits[FROZEN_GPU_TOKEN as usize] = 1.5;
        let view = logit_view("test", &logits).unwrap();
        assert_eq!(view.argmax, EXACT_F32_CPU_TOKEN);
        assert_eq!(view.top_12.len(), 12);
        assert_eq!(
            view.candidates_140003_and_54275
                .iter()
                .map(|candidate| candidate.token_id)
                .collect::<Vec<_>>(),
            [EXACT_F32_CPU_TOKEN, FROZEN_GPU_TOKEN]
        );
    }

    #[test]
    fn compensation_is_direct_descriptive_and_threshold_free() {
        assert_eq!(
            distance_relation(Some(1.0), Some(2.0)),
            DistanceRelation::HistoricalF16Closer
        );
        assert_eq!(
            distance_relation(Some(3.0), Some(2.0)),
            DistanceRelation::ExactF32Closer
        );
        assert!(!ProductionSemanticsEvidence::default().numerical_threshold_introduced);
    }

    #[test]
    fn production_v1_v2_and_numerics_remain_unchanged() {
        let semantics = ProductionSemanticsEvidence::default();
        assert!(semantics.diagnostic_only);
        assert!(!semantics.production_inference_changed);
        assert!(!semantics.production_q4_changed);
        assert!(!semantics.production_q4_wgsl_changed);
        assert!(!semantics.production_router_gemv_changed);
        assert!(!semantics.v1_changed);
        assert!(!semantics.v2_changed);
        assert!(!semantics.limits_corpus_or_prompts_changed);
        assert!(!semantics.production_correction_justified);
    }
}
