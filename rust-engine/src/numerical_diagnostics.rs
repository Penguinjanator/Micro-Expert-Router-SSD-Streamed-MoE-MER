//! Phase-A first-token numerical diagnostics for fixed-corpus greedy parity.
//!
//! This module is deliberately separate from the exact-token qualification
//! report. Workers return the complete first-token logit vector as bounded,
//! typed f32 bits; the final report retains only hashes and comparison
//! evidence. Diagnostic completion can never become qualification PASS.

use crate::backend::GpuExpertIoSnapshot;
use crate::greedy_parity::{
    token_ids_sha256, HybridWorkerProcessEvidence, HybridWorkerRequest,
    HybridWorkerResponse, PlaneRunEvidence,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};

pub const SCHEMA_VERSION: &str = "mer.strict-hybrid-q4-greedy-logit-diagnostic.v1";
pub const MODE: &str = "strict-hybrid-q4-greedy-logit-diagnostic";
pub const WORKER_PROTOCOL_VERSION: &str =
    "mer.strict-hybrid-q4-greedy-logit-worker.v1";
pub const TARGET_CASE: &str = "json-transformation";
pub const REPEATED_RUNS_PER_PLANE: usize = 2;
pub const TOP_LOGIT_COUNT: usize = 16;
pub const SHADOW_EXPERT_COUNT: usize = 8;
pub const SHADOW_D_MODEL: usize = 2048;
pub const MAX_VOCAB_SIZE: usize = 262_144;
pub const MAX_WORKER_STDOUT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticPlane {
    Cpu,
    CpuBoundaryEmulation,
    Hybrid,
}

impl DiagnosticPlane {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::CpuBoundaryEmulation => "cpu-boundary-emulation",
            Self::Hybrid => "hybrid",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticWorkerRequest {
    pub protocol_version: String,
    pub plane: DiagnosticPlane,
    pub run_index: usize,
    /// Reuse the established identity/config/token/executable contract.
    pub base: HybridWorkerRequest,
}

impl DiagnosticWorkerRequest {
    pub fn new(plane: DiagnosticPlane, run_index: usize, base: HybridWorkerRequest) -> Self {
        Self {
            protocol_version: WORKER_PROTOCOL_VERSION.to_string(),
            plane,
            run_index,
            base,
        }
    }

    pub fn validate_static(&self) -> Result<(), String> {
        if self.protocol_version != WORKER_PROTOCOL_VERSION
            || self.run_index >= REPEATED_RUNS_PER_PLANE
            || self.base.case_name != TARGET_CASE
            || self.base.prompt_token_ids.is_empty()
            || self.base.prompt_token_ids_sha256
                != token_ids_sha256(&self.base.prompt_token_ids)
            || self.base.output_token_limit != crate::greedy_parity::OUTPUT_TOKEN_LIMIT
            || self.base.worker_id.is_empty()
            || self.base.expected_adapter_name.is_empty()
        {
            return Err("logit diagnostic worker request is malformed".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticWorkerResponse {
    pub protocol_version: String,
    pub plane: DiagnosticPlane,
    pub run_index: usize,
    /// Reuse the existing typed worker identity and plane evidence.
    pub base: HybridWorkerResponse,
    pub chosen_token_id: Option<u32>,
    pub first_token_logit_bits: Option<Vec<u32>>,
    pub first_token_logit_bits_sha256: Option<String>,
    pub route_capture: Option<crate::engine::RoutedFfnDiagnosticCapture>,
    pub failure: Option<String>,
}

pub fn parse_worker_request_exact(bytes: &[u8]) -> Result<DiagnosticWorkerRequest, String> {
    parse_exact_json(bytes, "logit diagnostic worker request")
}

pub fn parse_worker_response_exact(bytes: &[u8]) -> Result<DiagnosticWorkerResponse, String> {
    parse_exact_json(bytes, "logit diagnostic worker response")
}

fn parse_exact_json<T>(bytes: &[u8], label: &str) -> Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    if bytes.is_empty() {
        return Err(format!("missing {label}"));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = T::deserialize(&mut deserializer)
        .map_err(|error| format!("malformed {label}: {error}"))?;
    deserializer
        .end()
        .map_err(|error| format!("trailing or duplicated {label}: {error}"))?;
    Ok(value)
}

pub fn diagnostic_worker_id(
    build_git_sha: &str,
    executable_sha256: &str,
    plane: DiagnosticPlane,
    run_index: usize,
) -> String {
    let mut hasher = Sha256::new();
    hash_string(&mut hasher, build_git_sha);
    hash_string(&mut hasher, executable_sha256);
    hash_string(&mut hasher, plane.as_str());
    hasher.update((run_index as u64).to_le_bytes());
    format!("greedy-logit-{}-{:x}", plane.as_str(), hasher.finalize())
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

pub fn f32_bits_sha256(bits: &[u32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((bits.len() as u64).to_le_bytes());
    for value in bits {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(crate) fn round_trip_f16_values(
    values: &[f32],
) -> Result<Vec<f32>, crate::inference::ExpertWeightsError> {
    let rounded: Vec<f32> = values
        .iter()
        .map(|value| half::f16::from_f32(*value).to_f32())
        .collect();
    if rounded.iter().any(|value| !value.is_finite()) {
        return Err(crate::inference::ExpertWeightsError::InvalidLayout(
            "diagnostic f16 precision boundary produced a nonfinite value".to_string(),
        ));
    }
    Ok(rounded)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RouteCaptureEvidence {
    pub token_idx: u64,
    pub layer: u32,
    pub cpu_input_sha256: String,
    pub hybrid_input_sha256: String,
    pub input_hashes_match: bool,
    pub cpu_expert_ids: Vec<u32>,
    pub hybrid_expert_ids: Vec<u32>,
    pub expert_ids_match: bool,
    pub cpu_routing_weight_bits: Vec<u32>,
    pub hybrid_routing_weight_bits: Vec<u32>,
    pub routing_weight_bits_match: bool,
    pub all_repeated_captures_match: bool,
    pub exact_capture_match: bool,
}

pub fn reconcile_route_captures(
    captures: &[(DiagnosticPlane, crate::engine::RoutedFfnDiagnosticCapture)],
    expected_token_idx: u64,
) -> Result<RouteCaptureEvidence, String> {
    if captures.len() != REPEATED_RUNS_PER_PLANE * 2 {
        return Err("route capture count is incomplete".to_string());
    }
    for (_, capture) in captures {
        if capture.token_idx != expected_token_idx
            || capture.layer != 0
            || capture.input_bits.len() != SHADOW_D_MODEL
            || capture.expert_ids.len() != SHADOW_EXPERT_COUNT
            || capture.routing_weight_bits.len() != SHADOW_EXPERT_COUNT
            || capture.expert_ids.iter().any(|expert| *expert >= 128)
            || capture.expert_ids.iter().copied().collect::<HashSet<_>>().len()
                != SHADOW_EXPERT_COUNT
            || capture
                .input_bits
                .iter()
                .chain(&capture.routing_weight_bits)
                .any(|bits| !f32::from_bits(*bits).is_finite())
        {
            return Err("layer-0 route capture is malformed".to_string());
        }
    }
    let cpu: Vec<_> = captures
        .iter()
        .filter(|(plane, _)| *plane == DiagnosticPlane::Cpu)
        .map(|(_, capture)| capture)
        .collect();
    let hybrid: Vec<_> = captures
        .iter()
        .filter(|(plane, _)| *plane == DiagnosticPlane::Hybrid)
        .map(|(_, capture)| capture)
        .collect();
    if cpu.len() != REPEATED_RUNS_PER_PLANE || hybrid.len() != REPEATED_RUNS_PER_PLANE {
        return Err("route captures do not cover both repeated planes".to_string());
    }
    let cpu_input_sha256 = f32_bits_sha256(&cpu[0].input_bits);
    let hybrid_input_sha256 = f32_bits_sha256(&hybrid[0].input_bits);
    let input_hashes_match = cpu_input_sha256 == hybrid_input_sha256;
    let expert_ids_match = cpu[0].expert_ids == hybrid[0].expert_ids;
    let routing_weight_bits_match =
        cpu[0].routing_weight_bits == hybrid[0].routing_weight_bits;
    let all_repeated_captures_match = captures
        .iter()
        .all(|(_, capture)| capture == &captures[0].1);
    Ok(RouteCaptureEvidence {
        token_idx: expected_token_idx,
        layer: 0,
        cpu_input_sha256,
        hybrid_input_sha256,
        input_hashes_match,
        cpu_expert_ids: cpu[0].expert_ids.clone(),
        hybrid_expert_ids: hybrid[0].expert_ids.clone(),
        expert_ids_match,
        cpu_routing_weight_bits: cpu[0].routing_weight_bits.clone(),
        hybrid_routing_weight_bits: hybrid[0].routing_weight_bits.clone(),
        routing_weight_bits_match,
        all_repeated_captures_match,
        exact_capture_match: input_hashes_match
            && expert_ids_match
            && routing_weight_bits_match
            && all_repeated_captures_match,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Q4ShadowOutputComparison {
    pub cpu_f32_sha256: String,
    pub cpu_f16_sha256: String,
    pub gpu_f16_sha256: String,
    pub max_absolute_error: f32,
    pub rms_error: f64,
    pub tolerance_pass: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Q4ShadowExpertEvidence {
    pub global_expert_id: u32,
    pub routing_weight: FloatEvidence,
    pub output: Q4ShadowOutputComparison,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActualInputQ4ShadowEvidence {
    pub captured_input_sha256: String,
    pub effective_f16_input_sha256: String,
    pub tolerance: crate::q4_parity::ErrorTolerance,
    pub experts: Vec<Q4ShadowExpertEvidence>,
    pub weighted_aggregate: Q4ShadowOutputComparison,
    pub all_experts_within_tolerance: bool,
    pub weighted_aggregate_within_tolerance: bool,
}

pub struct Q4ShadowExpertOutput {
    pub global_expert_id: u32,
    pub cpu_f32: Vec<f32>,
    pub gpu_f16: Vec<f32>,
}

fn hash_f32(values: &[f32]) -> String {
    f32_bits_sha256(&values.iter().copied().map(f32::to_bits).collect::<Vec<_>>())
}

fn compare_q4_shadow_output(
    cpu_f32: &[f32],
    cpu_f16: &[f32],
    gpu_f16: &[f32],
) -> Result<Q4ShadowOutputComparison, String> {
    if cpu_f32.len() != cpu_f16.len()
        || cpu_f32.len() != gpu_f16.len()
        || cpu_f32.is_empty()
    {
        return Err("Q4 shadow output lengths differ or are empty".to_string());
    }
    if cpu_f32
        .iter()
        .chain(cpu_f16.iter())
        .chain(gpu_f16)
        .any(|value| !value.is_finite())
    {
        return Err("Q4 shadow output contains a nonfinite value".to_string());
    }
    let tolerance = crate::q4_parity::COMPLETE_TOLERANCE;
    let mut max_absolute_error = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut tolerance_pass = true;
    for (&cpu, &gpu) in cpu_f16.iter().zip(gpu_f16) {
        let absolute = (gpu - cpu).abs();
        max_absolute_error = max_absolute_error.max(absolute);
        squared_error += f64::from(absolute) * f64::from(absolute);
        tolerance_pass &= absolute <= tolerance.absolute + tolerance.relative * cpu.abs();
    }
    Ok(Q4ShadowOutputComparison {
        cpu_f32_sha256: hash_f32(cpu_f32),
        cpu_f16_sha256: hash_f32(cpu_f16),
        gpu_f16_sha256: hash_f32(gpu_f16),
        max_absolute_error,
        rms_error: (squared_error / cpu_f16.len() as f64).sqrt(),
        tolerance_pass,
    })
}

pub fn build_actual_input_q4_shadow(
    capture: &crate::engine::RoutedFfnDiagnosticCapture,
    outputs: Vec<Q4ShadowExpertOutput>,
) -> Result<ActualInputQ4ShadowEvidence, String> {
    if outputs.len() != SHADOW_EXPERT_COUNT
        || capture.expert_ids.len() != outputs.len()
        || capture.routing_weight_bits.len() != outputs.len()
    {
        return Err("Q4 shadow expert output count is invalid".to_string());
    }
    let effective_input: Vec<f32> = capture
        .input_bits
        .iter()
        .map(|bits| half::f16::from_f32(f32::from_bits(*bits)).to_f32())
        .collect();
    let mut cpu_aggregate = vec![0.0f32; SHADOW_D_MODEL];
    let mut cpu_f16_aggregate = vec![0.0f32; SHADOW_D_MODEL];
    let mut gpu_aggregate = vec![0.0f32; SHADOW_D_MODEL];
    let mut experts = Vec::with_capacity(outputs.len());
    for (slot, output) in outputs.into_iter().enumerate() {
        if output.global_expert_id != capture.expert_ids[slot]
            || output.cpu_f32.len() != SHADOW_D_MODEL
            || output.gpu_f16.len() != SHADOW_D_MODEL
        {
            return Err("Q4 shadow expert identity or geometry is invalid".to_string());
        }
        let weight = f32::from_bits(capture.routing_weight_bits[slot]);
        let cpu_f16: Vec<f32> = output
            .cpu_f32
            .iter()
            .map(|value| half::f16::from_f32(*value).to_f32())
            .collect();
        for index in 0..SHADOW_D_MODEL {
            cpu_aggregate[index] += weight * output.cpu_f32[index];
            cpu_f16_aggregate[index] += weight * cpu_f16[index];
            gpu_aggregate[index] += weight * output.gpu_f16[index];
        }
        experts.push(Q4ShadowExpertEvidence {
            global_expert_id: output.global_expert_id,
            routing_weight: FloatEvidence::new(weight),
            output: compare_q4_shadow_output(&output.cpu_f32, &cpu_f16, &output.gpu_f16)?,
        });
    }
    let weighted_aggregate =
        compare_q4_shadow_output(&cpu_aggregate, &cpu_f16_aggregate, &gpu_aggregate)?;
    Ok(ActualInputQ4ShadowEvidence {
        captured_input_sha256: f32_bits_sha256(&capture.input_bits),
        effective_f16_input_sha256: hash_f32(&effective_input),
        tolerance: crate::q4_parity::COMPLETE_TOLERANCE,
        all_experts_within_tolerance: experts
            .iter()
            .all(|expert| expert.output.tolerance_pass),
        weighted_aggregate_within_tolerance: weighted_aggregate.tolerance_pass,
        experts,
        weighted_aggregate,
    })
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FloatEvidence {
    /// Nonfinite values are JSON `null`; exact bits and class remain present.
    pub value: Option<f32>,
    pub class: &'static str,
    pub bits: u32,
}

impl FloatEvidence {
    pub fn new(value: f32) -> Self {
        let class = if value.is_nan() {
            "nan"
        } else if value == f32::INFINITY {
            "positive-infinity"
        } else if value == f32::NEG_INFINITY {
            "negative-infinity"
        } else if value == 0.0 && value.is_sign_negative() {
            "negative-zero"
        } else {
            "finite"
        };
        Self {
            value: value.is_finite().then_some(value),
            class,
            bits: value.to_bits(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankedLogitEvidence {
    pub token_id: u32,
    pub rank: usize,
    pub logit: FloatEvidence,
}

fn ranked_logits(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut ranked: Vec<_> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|(left_id, left), (right_id, right)| {
        right
            .total_cmp(left)
            .then_with(|| left_id.cmp(right_id))
    });
    ranked
}

pub fn top_logits(logits: &[f32], count: usize) -> Vec<RankedLogitEvidence> {
    ranked_logits(logits)
        .into_iter()
        .take(count.min(logits.len()))
        .enumerate()
        .map(|(rank, (token_id, value))| RankedLogitEvidence {
            token_id: token_id as u32,
            rank: rank + 1,
            logit: FloatEvidence::new(value),
        })
        .collect()
}

fn ranked_logit_for(logits: &[f32], token_id: u32) -> Option<RankedLogitEvidence> {
    ranked_logits(logits)
        .into_iter()
        .enumerate()
        .find(|(_, (id, _))| *id == token_id as usize)
        .map(|(rank, (_, value))| RankedLogitEvidence {
            token_id,
            rank: rank + 1,
            logit: FloatEvidence::new(value),
        })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CandidateComparisonEvidence {
    pub token_id: u32,
    pub cpu_rank: usize,
    pub hybrid_rank: usize,
    pub cpu: FloatEvidence,
    pub hybrid: FloatEvidence,
    pub absolute_difference: Option<f32>,
    pub relative_difference: Option<f32>,
}

fn finite_differences(reference: f32, actual: f32) -> (Option<f32>, Option<f32>) {
    if !reference.is_finite() || !actual.is_finite() {
        return (None, None);
    }
    let absolute = (actual - reference).abs();
    if !absolute.is_finite() {
        return (None, None);
    }
    let relative = if reference == 0.0 {
        (absolute == 0.0).then_some(0.0)
    } else {
        let relative = absolute / reference.abs();
        relative.is_finite().then_some(relative)
    };
    (Some(absolute), relative)
}

pub fn union_candidate_comparison(
    cpu: &[f32],
    hybrid: &[f32],
    count: usize,
) -> Result<Vec<CandidateComparisonEvidence>, String> {
    if cpu.len() != hybrid.len() || cpu.is_empty() {
        return Err("logit vector lengths differ or are empty".to_string());
    }
    let cpu_ranked = ranked_logits(cpu);
    let hybrid_ranked = ranked_logits(hybrid);
    let mut ids = BTreeSet::new();
    ids.extend(cpu_ranked.iter().take(count).map(|(id, _)| *id));
    ids.extend(hybrid_ranked.iter().take(count).map(|(id, _)| *id));
    let cpu_ranks: BTreeMap<_, _> = cpu_ranked
        .iter()
        .enumerate()
        .map(|(rank, (id, _))| (*id, rank + 1))
        .collect();
    let hybrid_ranks: BTreeMap<_, _> = hybrid_ranked
        .iter()
        .enumerate()
        .map(|(rank, (id, _))| (*id, rank + 1))
        .collect();
    Ok(ids
        .into_iter()
        .map(|token_id| {
            let cpu_value = cpu[token_id];
            let hybrid_value = hybrid[token_id];
            let (absolute_difference, relative_difference) =
                finite_differences(cpu_value, hybrid_value);
            CandidateComparisonEvidence {
                token_id: token_id as u32,
                cpu_rank: cpu_ranks[&token_id],
                hybrid_rank: hybrid_ranks[&token_id],
                cpu: FloatEvidence::new(cpu_value),
                hybrid: FloatEvidence::new(hybrid_value),
                absolute_difference,
                relative_difference,
            }
        })
        .collect())
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MaxErrorEvidence {
    pub index: usize,
    pub cpu: FloatEvidence,
    pub hybrid: FloatEvidence,
    pub absolute_error: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VectorComparisonEvidence {
    pub length: usize,
    pub exact_f32_bits: bool,
    pub max_absolute_error: Option<f32>,
    pub rms_error: Option<f64>,
    pub cpu_nonfinite_count: usize,
    pub hybrid_nonfinite_count: usize,
    pub nonfinite_bit_mismatch_count: usize,
    pub max_error: Option<MaxErrorEvidence>,
}

pub fn compare_vectors(cpu: &[f32], hybrid: &[f32]) -> Result<VectorComparisonEvidence, String> {
    if cpu.len() != hybrid.len() || cpu.is_empty() {
        return Err("logit vector lengths differ or are empty".to_string());
    }
    let mut exact = true;
    let mut cpu_nonfinite = 0usize;
    let mut hybrid_nonfinite = 0usize;
    let mut nonfinite_bit_mismatch = 0usize;
    let mut finite_pairs = 0usize;
    let mut squared_error = 0.0f64;
    let mut max_absolute = -1.0f32;
    let mut max_error = None;
    for (index, (&left, &right)) in cpu.iter().zip(hybrid).enumerate() {
        exact &= left.to_bits() == right.to_bits();
        cpu_nonfinite += usize::from(!left.is_finite());
        hybrid_nonfinite += usize::from(!right.is_finite());
        if !left.is_finite() || !right.is_finite() {
            nonfinite_bit_mismatch += usize::from(left.to_bits() != right.to_bits());
            continue;
        }
        finite_pairs += 1;
        let absolute = (right - left).abs();
        squared_error += f64::from(absolute) * f64::from(absolute);
        if absolute > max_absolute {
            max_absolute = absolute;
            max_error = Some(MaxErrorEvidence {
                index,
                cpu: FloatEvidence::new(left),
                hybrid: FloatEvidence::new(right),
                absolute_error: absolute.is_finite().then_some(absolute),
            });
        }
    }
    Ok(VectorComparisonEvidence {
        length: cpu.len(),
        exact_f32_bits: exact,
        max_absolute_error: (finite_pairs > 0 && max_absolute.is_finite()).then_some(max_absolute),
        rms_error: (finite_pairs > 0)
            .then_some((squared_error / finite_pairs as f64).sqrt())
            .filter(|value| value.is_finite()),
        cpu_nonfinite_count: cpu_nonfinite,
        hybrid_nonfinite_count: hybrid_nonfinite,
        nonfinite_bit_mismatch_count: nonfinite_bit_mismatch,
        max_error,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PairMarginEvidence {
    pub cpu_token_715_minus_5212: FloatEvidence,
    pub hybrid_token_715_minus_5212: FloatEvidence,
    pub cross_plane_margin_change: FloatEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FirstTokenLogitEvidence {
    pub cpu_chosen_token_id: u32,
    pub hybrid_chosen_token_id: u32,
    pub cpu_top_16: Vec<RankedLogitEvidence>,
    pub hybrid_top_16: Vec<RankedLogitEvidence>,
    pub cpu_top_1_top_2_margin: FloatEvidence,
    pub hybrid_top_1_top_2_margin: FloatEvidence,
    pub token_715_vs_5212_margin: PairMarginEvidence,
    pub union_top_16: Vec<CandidateComparisonEvidence>,
    pub full_vector: VectorComparisonEvidence,
    pub token_715_cpu: RankedLogitEvidence,
    pub token_715_hybrid: RankedLogitEvidence,
    pub token_5212_cpu: RankedLogitEvidence,
    pub token_5212_hybrid: RankedLogitEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BoundaryPlaneEvidence {
    pub chosen_token_id: u32,
    pub generated_token_ids_sha256: String,
    pub first_token_logit_bits_sha256: String,
    pub top_16: Vec<RankedLogitEvidence>,
    pub token_715: RankedLogitEvidence,
    pub token_5212: RankedLogitEvidence,
    pub token_715_minus_5212_margin: FloatEvidence,
    pub versus_cpu: VectorComparisonEvidence,
    pub versus_hybrid: VectorComparisonEvidence,
    pub matches_cpu_chosen_token: bool,
    pub matches_hybrid_chosen_token: bool,
    pub matches_cpu_complete_logit_hash: bool,
    pub matches_hybrid_complete_logit_hash: bool,
}

pub fn build_boundary_plane_evidence(
    cpu: &[f32],
    hybrid: &[f32],
    boundary: &[f32],
    cpu_chosen: u32,
    hybrid_chosen: u32,
    boundary_chosen: u32,
    generated_token_ids_sha256: String,
) -> Result<BoundaryPlaneEvidence, String> {
    if cpu.len() != hybrid.len() || cpu.len() != boundary.len() || boundary.len() <= 5212 {
        return Err(
            "boundary-emulation logit vectors differ in length or are incomplete".to_string(),
        );
    }
    let top_16 = top_logits(boundary, TOP_LOGIT_COUNT);
    if top_16.first().map(|entry| entry.token_id) != Some(boundary_chosen) {
        return Err(
            "boundary-emulation logits disagree with production greedy selection".to_string(),
        );
    }
    let cpu_hash = f32_bits_sha256(&cpu.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
    let hybrid_hash =
        f32_bits_sha256(&hybrid.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
    let boundary_hash =
        f32_bits_sha256(&boundary.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
    Ok(BoundaryPlaneEvidence {
        chosen_token_id: boundary_chosen,
        generated_token_ids_sha256,
        first_token_logit_bits_sha256: boundary_hash.clone(),
        token_715: ranked_logit_for(boundary, 715)
            .ok_or("boundary-emulation token 715 is unavailable")?,
        token_5212: ranked_logit_for(boundary, 5212)
            .ok_or("boundary-emulation token 5212 is unavailable")?,
        token_715_minus_5212_margin: FloatEvidence::new(boundary[715] - boundary[5212]),
        versus_cpu: compare_vectors(cpu, boundary)?,
        versus_hybrid: compare_vectors(hybrid, boundary)?,
        matches_cpu_chosen_token: boundary_chosen == cpu_chosen,
        matches_hybrid_chosen_token: boundary_chosen == hybrid_chosen,
        matches_cpu_complete_logit_hash: boundary_hash == cpu_hash,
        matches_hybrid_complete_logit_hash: boundary_hash == hybrid_hash,
        top_16,
    })
}

fn top_margin(top: &[RankedLogitEvidence]) -> Result<f32, String> {
    let first = top
        .first()
        .ok_or("top-logit evidence is empty")?
        .logit
        .bits;
    let second = top
        .get(1)
        .ok_or("top-logit evidence has fewer than two entries")?
        .logit
        .bits;
    Ok(f32::from_bits(first) - f32::from_bits(second))
}

pub fn build_first_token_logit_evidence(
    cpu: &[f32],
    hybrid: &[f32],
    cpu_chosen: u32,
    hybrid_chosen: u32,
) -> Result<FirstTokenLogitEvidence, String> {
    if cpu.len() != hybrid.len() || cpu.len() <= 5212 {
        return Err("complete logit vectors do not cover required tokens".to_string());
    }
    let cpu_top = top_logits(cpu, TOP_LOGIT_COUNT);
    let hybrid_top = top_logits(hybrid, TOP_LOGIT_COUNT);
    if cpu_top.first().map(|item| item.token_id) != Some(cpu_chosen)
        || hybrid_top.first().map(|item| item.token_id) != Some(hybrid_chosen)
    {
        return Err("captured logits disagree with production greedy selection".to_string());
    }
    let cpu_pair_margin = cpu[715] - cpu[5212];
    let hybrid_pair_margin = hybrid[715] - hybrid[5212];
    Ok(FirstTokenLogitEvidence {
        cpu_chosen_token_id: cpu_chosen,
        hybrid_chosen_token_id: hybrid_chosen,
        cpu_top_1_top_2_margin: FloatEvidence::new(top_margin(&cpu_top)?),
        hybrid_top_1_top_2_margin: FloatEvidence::new(top_margin(&hybrid_top)?),
        token_715_vs_5212_margin: PairMarginEvidence {
            cpu_token_715_minus_5212: FloatEvidence::new(cpu_pair_margin),
            hybrid_token_715_minus_5212: FloatEvidence::new(hybrid_pair_margin),
            cross_plane_margin_change: FloatEvidence::new(
                hybrid_pair_margin - cpu_pair_margin,
            ),
        },
        union_top_16: union_candidate_comparison(cpu, hybrid, TOP_LOGIT_COUNT)?,
        full_vector: compare_vectors(cpu, hybrid)?,
        token_715_cpu: ranked_logit_for(cpu, 715).ok_or("CPU token 715 is unavailable")?,
        token_715_hybrid: ranked_logit_for(hybrid, 715)
            .ok_or("Hybrid token 715 is unavailable")?,
        token_5212_cpu: ranked_logit_for(cpu, 5212).ok_or("CPU token 5212 is unavailable")?,
        token_5212_hybrid: ranked_logit_for(hybrid, 5212)
            .ok_or("Hybrid token 5212 is unavailable")?,
        cpu_top_16: cpu_top,
        hybrid_top_16: hybrid_top,
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct RepeatedRunEvidence {
    pub plane: DiagnosticPlane,
    #[serde(rename = "repeat_index")]
    pub run_index: usize,
    pub worker_id: String,
    pub generated_token_ids: Vec<u32>,
    pub generated_token_ids_sha256: String,
    pub first_token_logit_bits_sha256: String,
    pub process: HybridWorkerProcessEvidence,
    pub plane_evidence: PlaneRunEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReproducibilityEvidence {
    pub cpu_bitwise_reproducible: bool,
    pub cpu_boundary_emulation_bitwise_reproducible: bool,
    pub hybrid_bitwise_reproducible: bool,
    pub all_worker_ids_unique: bool,
    pub all_process_ids_unique: bool,
    pub every_worker_exited_zero_and_reaped: bool,
    pub no_retries: bool,
}

pub fn validate_repeated_run_identity(runs: &[RepeatedRunEvidence]) -> ReproducibilityEvidence {
    let cpu: Vec<_> = runs.iter().filter(|run| run.plane == DiagnosticPlane::Cpu).collect();
    let hybrid: Vec<_> = runs
        .iter()
        .filter(|run| run.plane == DiagnosticPlane::Hybrid)
        .collect();
    let boundary: Vec<_> = runs
        .iter()
        .filter(|run| run.plane == DiagnosticPlane::CpuBoundaryEmulation)
        .collect();
    let worker_ids: HashSet<_> = runs.iter().map(|run| run.worker_id.as_str()).collect();
    let process_ids: Vec<_> = runs.iter().filter_map(|run| run.process.process_id).collect();
    let unique_process_ids: HashSet<_> = process_ids.iter().copied().collect();
    let reproducible = |plane: &[&RepeatedRunEvidence]| {
        plane.len() == REPEATED_RUNS_PER_PLANE
            && plane[0].generated_token_ids == plane[1].generated_token_ids
            && plane[0].generated_token_ids_sha256 == plane[1].generated_token_ids_sha256
            && plane[0].first_token_logit_bits_sha256
                == plane[1].first_token_logit_bits_sha256
    };
    ReproducibilityEvidence {
        cpu_bitwise_reproducible: reproducible(&cpu),
        cpu_boundary_emulation_bitwise_reproducible: reproducible(&boundary),
        hybrid_bitwise_reproducible: reproducible(&hybrid),
        all_worker_ids_unique: worker_ids.len() == runs.len(),
        all_process_ids_unique: process_ids.len() == runs.len()
            && unique_process_ids.len() == process_ids.len(),
        every_worker_exited_zero_and_reaped: runs.iter().all(|run| {
            run.worker_id == run.process.worker_id
                && run.plane_evidence.worker_process.as_ref() == Some(&run.process)
                && run.generated_token_ids_sha256 == token_ids_sha256(&run.generated_token_ids)
                && run.plane_evidence.generation.generated_token_ids == run.generated_token_ids
                && run.process.child_process_spawned
                && run.process.executable_identity_verified
                && run.process.build_sha_identity_verified
                && run.process.case_identity_verified
                && run.process.config_identity_verified
                && run.process.expected_adapter_identity_verified
                && run.process.prompt_token_identity_verified
                && run.process.output_token_limit_verified
                && run.process.greedy_sampling_identity_verified
                && run.process.normal_zero_exit
                && run.process.exit_code == Some(0)
                && run.process.signal.is_none()
                && !run.process.timed_out
                && run.process.process_reaped
                && run.process.evidence_emitted
        }),
        no_retries: runs.len() == REPEATED_RUNS_PER_PLANE * 3
            && cpu.iter().map(|run| run.run_index).collect::<BTreeSet<_>>()
                == BTreeSet::from([0, 1])
            && boundary
                .iter()
                .map(|run| run.run_index)
                .collect::<BTreeSet<_>>()
                == BTreeSet::from([0, 1])
            && hybrid
                .iter()
                .map(|run| run.run_index)
                .collect::<BTreeSet<_>>()
                == BTreeSet::from([0, 1]),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DiagnosticFailure {
    pub code: String,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct DiagnosticWorkerFailureEvidence {
    pub worker_id: String,
    pub plane: DiagnosticPlane,
    pub run_index: usize,
    pub detail: String,
    pub process: HybridWorkerProcessEvidence,
    pub stderr: String,
    pub stderr_truncated: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct DiagnosticReport {
    pub schema_version: &'static str,
    pub mode: &'static str,
    pub qualification_pass: bool,
    pub diagnostic_complete: bool,
    pub failure: Option<DiagnosticFailure>,
    pub provenance: crate::qualification::BuildProvenance,
    pub build_git_sha: String,
    pub executable_sha256: String,
    pub resolved_config_sha256: String,
    pub expected_adapter_name: String,
    pub case_name: &'static str,
    pub prompt_sha256: String,
    pub prompt_token_ids_sha256: String,
    pub prompt_token_count: usize,
    pub runs: Vec<RepeatedRunEvidence>,
    pub worker_failures: Vec<DiagnosticWorkerFailureEvidence>,
    pub reproducibility: Option<ReproducibilityEvidence>,
    pub first_token_logits: Option<FirstTokenLogitEvidence>,
    pub cpu_boundary_emulation: Option<BoundaryPlaneEvidence>,
    pub route_capture: Option<RouteCaptureEvidence>,
    pub actual_input_q4_shadow: Option<ActualInputQ4ShadowEvidence>,
}

impl DiagnosticReport {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provenance: crate::qualification::BuildProvenance,
        build_git_sha: String,
        executable_sha256: String,
        resolved_config_sha256: String,
        expected_adapter_name: String,
        prompt_sha256: String,
        prompt_token_ids_sha256: String,
        prompt_token_count: usize,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            mode: MODE,
            qualification_pass: false,
            diagnostic_complete: false,
            failure: None,
            provenance,
            build_git_sha,
            executable_sha256,
            resolved_config_sha256,
            expected_adapter_name,
            case_name: TARGET_CASE,
            prompt_sha256,
            prompt_token_ids_sha256,
            prompt_token_count,
            runs: Vec::new(),
            worker_failures: Vec::new(),
            reproducibility: None,
            first_token_logits: None,
            cpu_boundary_emulation: None,
            route_capture: None,
            actual_input_q4_shadow: None,
        }
    }

    pub fn fail(&mut self, code: impl Into<String>, detail: impl Into<String>) {
        self.qualification_pass = false;
        self.diagnostic_complete = false;
        self.failure = Some(DiagnosticFailure {
            code: code.into(),
            detail: detail.into(),
        });
    }

    pub fn finish(&mut self) -> Result<(), String> {
        self.qualification_pass = false;
        self.diagnostic_complete = false;
        let reproducibility = self
            .reproducibility
            .as_ref()
            .ok_or("diagnostic has no reproducibility evidence")?;
        if self.runs.len() != REPEATED_RUNS_PER_PLANE * 3
            || reproducibility != &validate_repeated_run_identity(&self.runs)
            || !reproducibility.cpu_bitwise_reproducible
            || !reproducibility.cpu_boundary_emulation_bitwise_reproducible
            || !reproducibility.hybrid_bitwise_reproducible
            || !reproducibility.all_worker_ids_unique
            || !reproducibility.all_process_ids_unique
            || !reproducibility.every_worker_exited_zero_and_reaped
            || !reproducibility.no_retries
            || !self.worker_failures.is_empty()
            || self.first_token_logits.is_none()
            || self.cpu_boundary_emulation.is_none()
            || !self
                .route_capture
                .as_ref()
                .is_some_and(|capture| capture.exact_capture_match)
            || !self.actual_input_q4_shadow.as_ref().is_some_and(|shadow| {
                shadow.experts.len() == SHADOW_EXPERT_COUNT
                    && is_sha256_hex(&shadow.captured_input_sha256)
                    && is_sha256_hex(&shadow.effective_f16_input_sha256)
            })
            || !self.first_token_evidence_exact()
            || !self.boundary_evidence_exact()
            || !self.runs.iter().all(|run| self.run_evidence_exact(run))
        {
            return Err("logit diagnostic evidence is partial or mismatched".to_string());
        }
        self.failure = None;
        self.diagnostic_complete = true;
        Ok(())
    }

    fn first_token_evidence_exact(&self) -> bool {
        let Some(evidence) = self.first_token_logits.as_ref() else {
            return false;
        };
        let cpu_token = self
            .runs
            .iter()
            .find(|run| run.plane == DiagnosticPlane::Cpu)
            .and_then(|run| run.generated_token_ids.first())
            .copied();
        let hybrid_token = self
            .runs
            .iter()
            .find(|run| run.plane == DiagnosticPlane::Hybrid)
            .and_then(|run| run.generated_token_ids.first())
            .copied();
        let union_ids: HashSet<_> = evidence
            .union_top_16
            .iter()
            .map(|candidate| candidate.token_id)
            .collect();
        evidence.cpu_top_16.len() == TOP_LOGIT_COUNT
            && evidence.hybrid_top_16.len() == TOP_LOGIT_COUNT
            && evidence.cpu_top_16.first().map(|item| item.token_id) == cpu_token
            && evidence.hybrid_top_16.first().map(|item| item.token_id) == hybrid_token
            && evidence.cpu_chosen_token_id == cpu_token.unwrap_or(u32::MAX)
            && evidence.hybrid_chosen_token_id == hybrid_token.unwrap_or(u32::MAX)
            && evidence.union_top_16.len() >= TOP_LOGIT_COUNT
            && evidence.union_top_16.len() <= TOP_LOGIT_COUNT * 2
            && evidence
                .cpu_top_16
                .iter()
                .chain(&evidence.hybrid_top_16)
                .all(|item| union_ids.contains(&item.token_id))
            && evidence.full_vector.length > 5212
            && evidence.token_715_cpu.token_id == 715
            && evidence.token_715_hybrid.token_id == 715
            && evidence.token_5212_cpu.token_id == 5212
            && evidence.token_5212_hybrid.token_id == 5212
    }

    fn boundary_evidence_exact(&self) -> bool {
        let Some(evidence) = self.cpu_boundary_emulation.as_ref() else {
            return false;
        };
        let find = |plane| self.runs.iter().find(|run| run.plane == plane);
        let (Some(cpu), Some(boundary), Some(hybrid)) = (
            find(DiagnosticPlane::Cpu),
            find(DiagnosticPlane::CpuBoundaryEmulation),
            find(DiagnosticPlane::Hybrid),
        ) else {
            return false;
        };
        let cpu_chosen = cpu.generated_token_ids.first().copied();
        let boundary_chosen = boundary.generated_token_ids.first().copied();
        let hybrid_chosen = hybrid.generated_token_ids.first().copied();
        evidence.chosen_token_id == boundary_chosen.unwrap_or(u32::MAX)
            && evidence.generated_token_ids_sha256 == boundary.generated_token_ids_sha256
            && evidence.first_token_logit_bits_sha256
                == boundary.first_token_logit_bits_sha256
            && evidence.top_16.len() == TOP_LOGIT_COUNT
            && evidence.top_16.first().map(|entry| entry.token_id) == boundary_chosen
            && evidence.token_715.token_id == 715
            && evidence.token_5212.token_id == 5212
            && evidence.versus_cpu.length == evidence.versus_hybrid.length
            && evidence.versus_cpu.length > 5212
            && evidence.matches_cpu_chosen_token == (boundary_chosen == cpu_chosen)
            && evidence.matches_hybrid_chosen_token == (boundary_chosen == hybrid_chosen)
            && evidence.matches_cpu_complete_logit_hash
                == (boundary.first_token_logit_bits_sha256
                    == cpu.first_token_logit_bits_sha256)
            && evidence.matches_hybrid_complete_logit_hash
                == (boundary.first_token_logit_bits_sha256
                    == hybrid.first_token_logit_bits_sha256)
    }

    fn run_evidence_exact(&self, run: &RepeatedRunEvidence) -> bool {
        let plane = &run.plane_evidence;
        let routed = plane.routed_execution_delta;
        let io = plane.gpu_io_delta;
        let initial = &plane.initial_state;
        let common = run.process.executable_sha256 == self.executable_sha256
            && run.process.build_git_sha == self.build_git_sha
            && is_sha256_hex(&run.first_token_logit_bits_sha256)
            && plane.model_load.strict
            && plane.model_load.loaded_tensors == plane.model_load.required_tensors
            && plane.model_load.required_tensors > 0
            && !plane.model_load.seeded_fallback_remained
            && plane.model_load.loader != "seeded"
            && initial.context_id == plane.execution_plan.context_id
            && initial.resolved_config_sha256 == self.resolved_config_sha256
            && initial.kv_cache_count > 0
            && initial.kv_sequence_lengths.len() == initial.kv_cache_count
            && initial.all_kv_empty
            && initial.kv_sequence_lengths.iter().all(|&length| length == 0)
            && initial.cache == crate::greedy_parity::RuntimeCacheSnapshot::default()
            && initial.routed == crate::engine::RoutedExpertExecutionSnapshot::default()
            && initial.gpu_io == GpuExpertIoSnapshot::default()
            && plane.generation.prompt_token_ids_sha256 == self.prompt_token_ids_sha256
            && plane.generation.generated_token_ids_sha256
                == token_ids_sha256(&plane.generation.generated_token_ids)
            && plane.generation.generated_token_ids_sha256
                == run.generated_token_ids_sha256
            && plane.generation.generated_token_count == crate::greedy_parity::OUTPUT_TOKEN_LIMIT
            && plane.generation.termination_reason
                == crate::greedy_parity::TerminationReason::LengthLimit
            && plane.background_shutdown.controlled_shutdown_requested
            && plane.background_shutdown.all_runtime_resources_released
            && plane.attention_softmax_nonfinite_fallbacks == 0
            && routed.gpu_dispatch_failures == 0
            && routed.gpu_cpu_fallbacks == 0
            && routed.degraded_expert_substitutions == 0;
        if !common {
            return false;
        }
        match run.plane {
            DiagnosticPlane::Cpu | DiagnosticPlane::CpuBoundaryEmulation => {
                plane.plane == "cpu"
                    && crate::greedy_parity::cpu_plan_exact(&plane.execution_plan)
                    && plane.device.is_none()
                    && !initial.gpu_io_available
                    && routed.selected_routed_experts > 0
                    && routed.cpu_routed_expert_dispatches == routed.selected_routed_experts
                    && routed.gpu_dispatch_attempts == 0
                    && io == GpuExpertIoSnapshot::default()
                    && plane.gpu_memory_before.is_none()
                    && plane.gpu_memory_after.is_none()
            }
            DiagnosticPlane::Hybrid => {
                plane.plane == "hybrid"
                    && crate::greedy_parity::hybrid_plan_exact(&plane.execution_plan)
                    && plane.device.as_ref().is_some_and(|device| {
                        !device.software_adapter
                            && !device.device_type.eq_ignore_ascii_case("cpu")
                            && device.name == self.expected_adapter_name
                    })
                    && initial.gpu_io_available
                    && plane.routed_expert_gpu_failure_policy == "strict-fail-closed"
                    && routed.selected_routed_experts > 0
                    && routed.gpu_dispatch_attempts == routed.selected_routed_experts
                    && routed.gpu_dispatch_successes == routed.gpu_dispatch_attempts
                    && routed.cpu_routed_expert_dispatches == 0
                    && io.hidden_state_uploads > 0
                    && io.hidden_state_upload_bytes > 0
                    && io.queue_submissions > 0
                    && io.map_requests > 0
                    && io.readback_completions > 0
                    && io.readback_bytes > 0
                    && plane.gpu_memory_before.is_some_and(|snapshot| {
                        crate::qualification::validate_memory(snapshot).is_ok()
                    })
                    && plane.gpu_memory_after.is_some_and(|snapshot| {
                        crate::qualification::validate_memory(snapshot).is_ok()
                    })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeated_run(
        plane: DiagnosticPlane,
        run_index: usize,
        generated_token: u32,
        logit_hash: char,
        process_id: u32,
    ) -> RepeatedRunEvidence {
        let worker_id = format!("{}-{run_index}", plane.as_str());
        let generated_token_ids = vec![generated_token];
        let generated_token_ids_sha256 = token_ids_sha256(&generated_token_ids);
        let process = HybridWorkerProcessEvidence {
            worker_id: worker_id.clone(),
            child_process_spawned: true,
            process_id: Some(process_id),
            executable_sha256: "e".repeat(64),
            build_git_sha: "a".repeat(40),
            executable_identity_verified: true,
            build_sha_identity_verified: true,
            case_identity_verified: true,
            config_identity_verified: true,
            expected_adapter_identity_verified: true,
            prompt_token_identity_verified: true,
            output_token_limit_verified: true,
            greedy_sampling_identity_verified: true,
            normal_zero_exit: true,
            exit_code: Some(0),
            signal: None,
            process_reaped: true,
            timed_out: false,
            evidence_emitted: true,
        };
        let context_id = format!("context-{worker_id}");
        let plane_evidence = PlaneRunEvidence {
            plane: "cpu".to_string(),
            model_load: crate::greedy_parity::ModelLoadEvidence {
                strict: true,
                loader: "safetensors".to_string(),
                loaded_tensors: 1,
                required_tensors: 1,
                optional_probed: 0,
                optional_loaded: 0,
                seeded_fallback_remained: false,
            },
            execution_plan: crate::qualification::ExecutionPlanEvidence {
                context_id: context_id.clone(),
                requested: "cpu".to_string(),
                resolved: "cpu".to_string(),
                embeddings: "cpu".to_string(),
                lm_head: "cpu".to_string(),
                dense_projections: "cpu".to_string(),
                attention: "cpu".to_string(),
                kv: "cpu".to_string(),
                router: "cpu".to_string(),
                routed_experts: "cpu".to_string(),
                routed_expert_dtype: "q4_0".to_string(),
                fallback_occurred: false,
                reason: None,
            },
            routed_expert_gpu_failure_policy: "serving-cpu-fallback".to_string(),
            device: None,
            initial_state: crate::greedy_parity::InitialStateEvidence {
                context_id,
                resolved_config_sha256: "c".repeat(64),
                kv_cache_count: 1,
                kv_sequence_lengths: vec![0],
                all_kv_empty: true,
                cache: crate::greedy_parity::RuntimeCacheSnapshot::default(),
                routed: crate::engine::RoutedExpertExecutionSnapshot::default(),
                gpu_io_available: false,
                gpu_io: GpuExpertIoSnapshot::default(),
            },
            generation: crate::greedy_parity::GenerationEvidence {
                prompt_token_ids_sha256: "p".repeat(64),
                generated_token_ids: generated_token_ids.clone(),
                generated_token_ids_sha256: generated_token_ids_sha256.clone(),
                generated_text_sha256: "t".repeat(64),
                generated_token_count: 1,
                termination_reason: crate::greedy_parity::TerminationReason::LengthLimit,
            },
            routed_execution_delta: crate::engine::RoutedExpertExecutionSnapshot {
                selected_routed_experts: 1,
                cpu_routed_expert_dispatches: 1,
                ..Default::default()
            },
            gpu_io_delta: GpuExpertIoSnapshot::default(),
            attention_softmax_nonfinite_fallbacks: 0,
            gpu_memory_before: None,
            gpu_memory_after: None,
            background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence {
                controlled_shutdown_requested: true,
                all_runtime_resources_released: true,
                poll_iterations: 1,
            },
            worker_process: Some(process.clone()),
        };
        RepeatedRunEvidence {
            plane,
            run_index,
            worker_id,
            generated_token_ids,
            generated_token_ids_sha256,
            first_token_logit_bits_sha256: logit_hash.to_string().repeat(64),
            process,
            plane_evidence,
        }
    }

    #[test]
    fn boundary_conversion_matches_production_f16_round_trip_and_fails_nonfinite() {
        let values = [1.0003, -0.0, 65504.0];
        let rounded = round_trip_f16_values(&values).unwrap();
        assert_eq!(
            rounded[0].to_bits(),
            half::f16::from_f32(values[0]).to_f32().to_bits()
        );
        assert_eq!(rounded[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(rounded[2], 65504.0);
        assert!(round_trip_f16_values(&[70000.0]).is_err());
        assert!(round_trip_f16_values(&[f32::NAN]).is_err());
    }

    #[test]
    fn top_logits_use_total_order_and_lower_token_id_for_ties() {
        let evidence = top_logits(&[1.0, 3.0, 3.0, -0.0, 0.0], 5);
        assert_eq!(
            evidence.iter().map(|item| item.token_id).collect::<Vec<_>>(),
            vec![1, 2, 0, 4, 3]
        );
        assert_eq!(evidence[4].logit.bits, (-0.0f32).to_bits());
    }

    #[test]
    fn float_evidence_preserves_signed_zero_and_nonfinite_bits() {
        let values = [
            FloatEvidence::new(-0.0),
            FloatEvidence::new(f32::from_bits(0x7fc0_1234)),
            FloatEvidence::new(f32::INFINITY),
        ];
        let json = serde_json::to_string(&values).unwrap();
        assert!(json.contains("negative-zero"));
        assert!(json.contains("nan"));
        assert!(json.contains("positive-infinity"));
        assert_eq!(values[0].bits, (-0.0f32).to_bits());
    }

    #[test]
    fn union_comparison_contains_candidates_from_both_planes() {
        let cpu = [5.0, 4.0, 1.0, 0.0];
        let hybrid = [1.0, 2.0, 5.0, 4.0];
        let union = union_candidate_comparison(&cpu, &hybrid, 2).unwrap();
        assert_eq!(union.iter().map(|item| item.token_id).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert_eq!(union.iter().find(|item| item.token_id == 0).unwrap().cpu_rank, 1);
        assert_eq!(union.iter().find(|item| item.token_id == 2).unwrap().hybrid_rank, 1);
    }

    #[test]
    fn vector_comparison_records_max_index_rms_and_nonfinite_counts() {
        let comparison = compare_vectors(&[0.0, 1.0, -2.0], &[0.0, 2.0, 0.0]).unwrap();
        assert_eq!(comparison.max_absolute_error, Some(2.0));
        assert_eq!(comparison.max_error.as_ref().unwrap().index, 2);
        assert!((comparison.rms_error.unwrap() - (5.0f64 / 3.0).sqrt()).abs() < 1e-12);
        let nonfinite = compare_vectors(&[f32::NAN], &[f32::INFINITY]).unwrap();
        assert_eq!(nonfinite.cpu_nonfinite_count, 1);
        assert_eq!(nonfinite.hybrid_nonfinite_count, 1);
        assert_eq!(nonfinite.nonfinite_bit_mismatch_count, 1);
    }

    #[test]
    fn first_token_evidence_preserves_715_5212_margin_change() {
        let mut cpu = vec![-10.0; 6000];
        let mut hybrid = cpu.clone();
        cpu[715] = 4.0;
        cpu[5212] = 3.75;
        hybrid[715] = 3.8;
        hybrid[5212] = 4.0;
        let evidence = build_first_token_logit_evidence(&cpu, &hybrid, 715, 5212).unwrap();
        assert_eq!(evidence.token_715_cpu.rank, 1);
        assert_eq!(evidence.token_5212_hybrid.rank, 1);
        assert_eq!(evidence.token_715_vs_5212_margin.cpu_token_715_minus_5212.value, Some(0.25));
        let hybrid_margin = evidence
            .token_715_vs_5212_margin
            .hybrid_token_715_minus_5212
            .value
            .unwrap();
        let margin_change = evidence
            .token_715_vs_5212_margin
            .cross_plane_margin_change
            .value
            .unwrap();
        assert!((hybrid_margin + 0.2).abs() < 1e-6);
        assert!((margin_change + 0.45).abs() < 1e-6);
    }

    #[test]
    fn boundary_report_compares_complete_logits_and_chosen_token() {
        let mut cpu = vec![-10.0; 6000];
        cpu[715] = 4.0;
        cpu[5212] = 3.75;
        let mut hybrid = cpu.clone();
        hybrid[715] = 3.8;
        hybrid[5212] = 4.0;
        let boundary = hybrid.clone();
        let generated_hash = token_ids_sha256(&[5212]);
        let evidence = build_boundary_plane_evidence(
            &cpu,
            &hybrid,
            &boundary,
            715,
            5212,
            5212,
            generated_hash.clone(),
        )
        .unwrap();
        assert_eq!(evidence.generated_token_ids_sha256, generated_hash);
        assert_eq!(evidence.top_16[0].token_id, 5212);
        assert_eq!(evidence.token_715.rank, 2);
        assert_eq!(evidence.token_5212.rank, 1);
        assert!(evidence.versus_hybrid.exact_f32_bits);
        assert!(!evidence.versus_cpu.exact_f32_bits);
        assert!(!evidence.matches_cpu_chosen_token);
        assert!(evidence.matches_hybrid_chosen_token);
        assert!(!evidence.matches_cpu_complete_logit_hash);
        assert!(evidence.matches_hybrid_complete_logit_hash);
    }

    #[test]
    fn reproducibility_requires_two_exact_unique_runs_for_all_three_planes() {
        let mut runs = Vec::new();
        for (plane, token, hash, pid) in [
            (DiagnosticPlane::Cpu, 715, 'a', 10),
            (DiagnosticPlane::CpuBoundaryEmulation, 5212, 'b', 20),
            (DiagnosticPlane::Hybrid, 5212, 'c', 30),
        ] {
            runs.push(repeated_run(plane, 0, token, hash, pid));
            runs.push(repeated_run(plane, 1, token, hash, pid + 1));
        }
        let exact = validate_repeated_run_identity(&runs);
        assert!(exact.cpu_bitwise_reproducible);
        assert!(exact.cpu_boundary_emulation_bitwise_reproducible);
        assert!(exact.hybrid_bitwise_reproducible);
        assert!(exact.all_worker_ids_unique);
        assert!(exact.all_process_ids_unique);
        assert!(exact.every_worker_exited_zero_and_reaped);
        assert!(exact.no_retries);

        runs[3].first_token_logit_bits_sha256 = "d".repeat(64);
        let mismatch = validate_repeated_run_identity(&runs);
        assert!(mismatch.cpu_bitwise_reproducible);
        assert!(!mismatch.cpu_boundary_emulation_bitwise_reproducible);
        assert!(mismatch.hybrid_bitwise_reproducible);
    }

    #[test]
    fn worker_protocol_rejects_unknown_and_trailing_documents() {
        let base = HybridWorkerRequest::new(
            "worker".to_string(),
            crate::greedy_parity::fixed_case(TARGET_CASE).unwrap(),
            "a".repeat(64),
            "NVIDIA L4".to_string(),
            vec![1, 2],
            "b".repeat(64),
            "c".repeat(40),
        );
        let request = DiagnosticWorkerRequest::new(DiagnosticPlane::Cpu, 0, base);
        let encoded = serde_json::to_vec(&request).unwrap();
        assert_eq!(parse_worker_request_exact(&encoded).unwrap(), request);
        let mut boundary: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        boundary["plane"] = serde_json::json!("cpu_boundary_emulation");
        assert_eq!(
            parse_worker_request_exact(&serde_json::to_vec(&boundary).unwrap())
                .unwrap()
                .plane,
            DiagnosticPlane::CpuBoundaryEmulation
        );
        boundary["plane"] = serde_json::json!("unrecognized_plane");
        assert!(parse_worker_request_exact(&serde_json::to_vec(&boundary).unwrap()).is_err());
        let mut unknown: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        unknown["unknown"] = serde_json::json!(true);
        assert!(parse_worker_request_exact(&serde_json::to_vec(&unknown).unwrap()).is_err());
        let mut trailing = encoded;
        trailing.extend_from_slice(b"{}");
        assert!(parse_worker_request_exact(&trailing).is_err());
    }

    #[test]
    fn diagnostic_report_can_never_claim_qualification_pass() {
        let mut report = DiagnosticReport::new(
            crate::qualification::BuildProvenance {
                git_sha: Some("a".repeat(40)),
                dirty: Some(false),
                package_version: "test".to_string(),
            },
            "a".repeat(40),
            "b".repeat(64),
            "c".repeat(64),
            "NVIDIA L4".to_string(),
            "d".repeat(64),
            "e".repeat(64),
            2,
        );
        report.qualification_pass = true;
        assert!(report.finish().is_err());
        assert!(!report.qualification_pass);
        report.fail("test", "failure");
        assert!(!report.qualification_pass);
        assert!(!report.diagnostic_complete);
    }

    fn route_capture(input_bits: Vec<u32>) -> crate::engine::RoutedFfnDiagnosticCapture {
        crate::engine::RoutedFfnDiagnosticCapture {
            token_idx: 96,
            layer: 0,
            input_bits,
            expert_ids: (0..SHADOW_EXPERT_COUNT as u32).collect(),
            routing_weight_bits: vec![(0.125f32).to_bits(); SHADOW_EXPERT_COUNT],
        }
    }

    #[test]
    fn route_capture_requires_exact_cpu_hybrid_inputs_experts_and_weights() {
        let exact = route_capture(vec![(0.5f32).to_bits(); SHADOW_D_MODEL]);
        let captures = vec![
            (DiagnosticPlane::Cpu, exact.clone()),
            (DiagnosticPlane::Cpu, exact.clone()),
            (DiagnosticPlane::Hybrid, exact.clone()),
            (DiagnosticPlane::Hybrid, exact.clone()),
        ];
        let evidence = reconcile_route_captures(&captures, 96).unwrap();
        assert!(evidence.exact_capture_match);
        assert!(evidence.all_repeated_captures_match);

        let mut mismatched = captures;
        mismatched[2].1.input_bits[17] = (0.25f32).to_bits();
        let evidence = reconcile_route_captures(&mismatched, 96).unwrap();
        assert!(!evidence.input_hashes_match);
        assert!(!evidence.exact_capture_match);
    }

    #[test]
    fn actual_input_shadow_uses_complete_q4_tolerance_without_serializing_vectors() {
        let capture = route_capture(vec![(0.5f32).to_bits(); SHADOW_D_MODEL]);
        let outputs = capture
            .expert_ids
            .iter()
            .map(|&global_expert_id| Q4ShadowExpertOutput {
                global_expert_id,
                cpu_f32: vec![1.0; SHADOW_D_MODEL],
                gpu_f16: vec![1.001; SHADOW_D_MODEL],
            })
            .collect();
        let evidence = build_actual_input_q4_shadow(&capture, outputs).unwrap();
        assert_eq!(evidence.tolerance, crate::q4_parity::COMPLETE_TOLERANCE);
        assert!(evidence.all_experts_within_tolerance);
        assert!(evidence.weighted_aggregate_within_tolerance);
        let json = serde_json::to_string(&evidence).unwrap();
        assert!(!json.contains("input_bits"));
        assert!(!json.contains("cpu_f32\""));
        assert!(!json.contains("gpu_f16\""));
    }
}
