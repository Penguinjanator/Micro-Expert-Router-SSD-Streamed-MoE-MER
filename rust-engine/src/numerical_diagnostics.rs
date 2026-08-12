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
pub const MAX_VOCAB_SIZE: usize = 262_144;
pub const MAX_WORKER_STDOUT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticPlane {
    Cpu,
    Hybrid,
}

impl DiagnosticPlane {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
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
        no_retries: runs.len() == REPEATED_RUNS_PER_PLANE * 2
            && cpu.iter().map(|run| run.run_index).collect::<BTreeSet<_>>()
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
}

impl DiagnosticReport {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
        if self.runs.len() != REPEATED_RUNS_PER_PLANE * 2
            || reproducibility != &validate_repeated_run_identity(&self.runs)
            || !reproducibility.cpu_bitwise_reproducible
            || !reproducibility.hybrid_bitwise_reproducible
            || !reproducibility.all_worker_ids_unique
            || !reproducibility.all_process_ids_unique
            || !reproducibility.every_worker_exited_zero_and_reaped
            || !reproducibility.no_retries
            || !self.worker_failures.is_empty()
            || self.first_token_logits.is_none()
            || !self.first_token_evidence_exact()
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
            DiagnosticPlane::Cpu => {
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
}
