//! Diagnostic-only attribution for a GPU-native router rank permutation.
//!
//! This module owns only typed evidence and hardware-independent analysis.
//! Production routing and qualification PASS semantics do not depend on it.

use std::cmp::Ordering;

use serde::Serialize;

use crate::backend::GpuDeviceIdentity;
use crate::gating::{LinearGate, ScoringFunc};
use crate::gpu_native_token_loop::GpuNativeModelGeometry;
use crate::greedy_parity::{CorpusEvidence, ModelIdentityEvidence, ModelLoadEvidence};
use crate::numerical_diagnostics::{FloatEvidence, VectorComparisonEvidence};
use crate::qualification::{BuildProvenance, ExpertMetadataEvidence, QualificationArtifacts};

pub const SCHEMA_VERSION: &str = "mer.gpu-native-router-rank-divergence.v1";
pub const MODE: &str = "diagnose-gpu-native-router-rank-divergence";
pub const LOWER_EXPERT_ID: u32 = 68;
pub const HIGHER_EXPERT_ID: u32 = 113;
pub const REQUIRED_TOP_K: usize = 8;
pub const RANKED_EXPERT_COUNT: usize = 12;
/// Descriptive evidence only. It is never used by qualification or routing.
pub const NEAR_TIE_ULP_WINDOW: u32 = 16;

/// Target-layer-only diagnostic staging layout. It is allocated exclusively
/// by the router-rank diagnostic command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouterRankTraceLayout {
    pub target_layer: usize,
    pub d_model: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub router_input_offset: u64,
    pub router_input_bytes: u64,
    pub raw_logits_offset: u64,
    pub raw_logits_bytes: u64,
    pub selected_ids_offset: u64,
    pub selected_ids_bytes: u64,
    pub selected_weights_offset: u64,
    pub selected_weights_bytes: u64,
    pub total_bytes: u64,
}

impl RouterRankTraceLayout {
    pub fn try_new(geometry: GpuNativeModelGeometry, target_layer: usize) -> Result<Self, String> {
        if target_layer >= geometry.num_layers
            || geometry.d_model == 0
            || geometry.num_experts == 0
            || geometry.top_k == 0
            || geometry.top_k > geometry.num_experts
        {
            return Err("invalid router-rank trace geometry or target layer".to_string());
        }
        let bytes = |elements: usize, label: &str| {
            elements
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| format!("{label} byte size overflow"))
        };
        let router_input_offset = 0;
        let router_input_bytes = bytes(geometry.d_model, "router input")?;
        let raw_logits_offset = router_input_bytes;
        let raw_logits_bytes = bytes(geometry.num_experts, "raw logits")?;
        let selected_ids_offset = raw_logits_offset
            .checked_add(raw_logits_bytes)
            .ok_or("selected IDs offset overflow")?;
        let selected_ids_bytes = bytes(geometry.top_k, "selected IDs")?;
        let selected_weights_offset = selected_ids_offset
            .checked_add(selected_ids_bytes)
            .ok_or("selected weights offset overflow")?;
        let selected_weights_bytes = bytes(geometry.top_k, "selected weights")?;
        let total_bytes = selected_weights_offset
            .checked_add(selected_weights_bytes)
            .ok_or("router-rank trace total size overflow")?;
        Ok(Self {
            target_layer,
            d_model: geometry.d_model,
            num_experts: geometry.num_experts,
            top_k: geometry.top_k,
            router_input_offset,
            router_input_bytes,
            raw_logits_offset,
            raw_logits_bytes,
            selected_ids_offset,
            selected_ids_bytes,
            selected_weights_offset,
            selected_weights_bytes,
            total_bytes,
        })
    }

    pub fn parse(&self, bytes: &[u8]) -> Result<RouterRankGpuTrace, String> {
        if bytes.len() < self.total_bytes as usize {
            return Err(format!(
                "router-rank staging has {} bytes, expected {}",
                bytes.len(),
                self.total_bytes
            ));
        }
        let parse_f32 = |offset: u64, count: usize| {
            bytes[offset as usize..offset as usize + count * 4]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        };
        let parse_u32 = |offset: u64, count: usize| {
            bytes[offset as usize..offset as usize + count * 4]
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        };
        Ok(RouterRankGpuTrace {
            router_input: parse_f32(self.router_input_offset, self.d_model),
            raw_logits: parse_f32(self.raw_logits_offset, self.num_experts),
            selected_ids: parse_u32(self.selected_ids_offset, self.top_k),
            selected_weights: parse_f32(self.selected_weights_offset, self.top_k),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouterRankGpuTrace {
    pub router_input: Vec<f32>,
    pub raw_logits: Vec<f32>,
    pub selected_ids: Vec<u32>,
    pub selected_weights: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DiagnosticProvenance {
    pub build: BuildProvenance,
    pub executable_sha256: String,
    pub artifacts: QualificationArtifacts,
    pub gpu_resolved_config_sha256: String,
    pub reference_resolved_config_sha256: String,
    pub model_identity: ModelIdentityEvidence,
    pub reference_model_load: ModelLoadEvidence,
    pub gpu_native_model_load: ModelLoadEvidence,
    pub reference_background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
    pub gpu_native_background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
    pub expert_metadata: ExpertMetadataEvidence,
    pub device: GpuDeviceIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TargetEvidence {
    pub case: String,
    pub generated_position: usize,
    pub generated_position_zero_based: bool,
    pub layer: usize,
    pub expected_reference_ids: Vec<u32>,
    pub observed_gpu_ids: Vec<u32>,
    pub top_8_set_match: bool,
    pub rank_order_match: bool,
    pub only_internal_rank_change: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PrefixEvidence {
    pub corpus: CorpusEvidence,
    pub prompt_sha256: String,
    pub prompt_token_ids: Vec<u32>,
    pub prompt_token_ids_sha256: String,
    pub reference_preceding_generated_token_ids: Vec<u32>,
    pub gpu_native_preceding_generated_token_ids: Vec<u32>,
    pub preceding_generated_prefix_match: bool,
    pub reference_target_generated_token_id: u32,
    pub gpu_native_target_generated_token_id: u32,
    pub target_generated_token_match: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouterInputEvidence {
    pub reference_vector_length: usize,
    pub gpu_native_vector_length: usize,
    pub reference_f32_bits_sha256: String,
    pub gpu_native_f32_bits_sha256: String,
    pub reference_f32_bits: Vec<u32>,
    pub gpu_native_f32_bits: Vec<u32>,
    pub bitwise_equal_element_count: usize,
    pub comparison: VectorComparisonEvidence,
    pub worst_index: Option<usize>,
    pub reference_value_at_worst_index: Option<FloatEvidence>,
    pub gpu_native_value_at_worst_index: Option<FloatEvidence>,
}

impl RouterInputEvidence {
    pub fn compare(reference: &[f32], gpu_native: &[f32]) -> Result<Self, String> {
        let comparison = crate::numerical_diagnostics::compare_vectors(reference, gpu_native)?;
        let reference_f32_bits: Vec<u32> = reference.iter().map(|value| value.to_bits()).collect();
        let gpu_native_f32_bits: Vec<u32> =
            gpu_native.iter().map(|value| value.to_bits()).collect();
        let bitwise_equal_element_count = reference_f32_bits
            .iter()
            .zip(&gpu_native_f32_bits)
            .filter(|(reference, gpu)| reference == gpu)
            .count();
        let worst_index = comparison.max_error.as_ref().map(|error| error.index);
        Ok(Self {
            reference_vector_length: reference.len(),
            gpu_native_vector_length: gpu_native.len(),
            reference_f32_bits_sha256: crate::numerical_diagnostics::f32_bits_sha256(
                &reference_f32_bits,
            ),
            gpu_native_f32_bits_sha256: crate::numerical_diagnostics::f32_bits_sha256(
                &gpu_native_f32_bits,
            ),
            reference_f32_bits,
            gpu_native_f32_bits,
            bitwise_equal_element_count,
            worst_index,
            reference_value_at_worst_index: worst_index
                .and_then(|index| reference.get(index).copied())
                .map(FloatEvidence::new),
            gpu_native_value_at_worst_index: worst_index
                .and_then(|index| gpu_native.get(index).copied())
                .map(FloatEvidence::new),
            comparison,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GateTensorIdentity {
    pub canonical_tensor_name: String,
    pub dtype: String,
    pub rows: usize,
    pub cols: usize,
    pub loaded_f32_bits_sha256: String,
}

impl GateTensorIdentity {
    pub fn from_gate(layer: usize, gate: &LinearGate) -> Self {
        let values = gate.weights.to_f32_vec();
        let bits: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
        Self {
            canonical_tensor_name: format!("model.layers.{layer}.mlp.gate.weight"),
            dtype: gate.weights.dtype_name().to_string(),
            rows: gate.weights.rows(),
            cols: gate.weights.cols(),
            loaded_f32_bits_sha256: crate::numerical_diagnostics::f32_bits_sha256(&bits),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GateIdentityEvidence {
    pub reference: GateTensorIdentity,
    pub gpu_native: GateTensorIdentity,
    pub exact_identity_match: bool,
}

impl GateIdentityEvidence {
    pub fn new(reference: GateTensorIdentity, gpu_native: GateTensorIdentity) -> Self {
        let exact_identity_match = reference == gpu_native;
        Self {
            reference,
            gpu_native,
            exact_identity_match,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankedExpertEvidence {
    pub expert_id: u32,
    pub rank: usize,
    pub raw_logit: FloatEvidence,
    pub score: FloatEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExpertPairEvidence {
    pub expert_68_raw_logit: FloatEvidence,
    pub expert_113_raw_logit: FloatEvidence,
    pub raw_logit_68_minus_113: FloatEvidence,
    pub expert_68_score: FloatEvidence,
    pub expert_113_score: FloatEvidence,
    pub score_68_minus_113: FloatEvidence,
    pub expert_68_rank: usize,
    pub expert_113_rank: usize,
    pub raw_logit_ulp_distance: Option<u32>,
    pub score_ulp_distance: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CutoffMarginEvidence {
    pub rank_7_expert_id: u32,
    pub rank_8_expert_id: u32,
    pub rank_9_expert_id: u32,
    pub rank_7_minus_rank_8_score: FloatEvidence,
    pub rank_8_minus_rank_9_score: FloatEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouterEvaluationEvidence {
    pub source: &'static str,
    pub score_origin: &'static str,
    pub raw_logits: Vec<FloatEvidence>,
    pub scored_probabilities: Vec<FloatEvidence>,
    pub top_8_ids: Vec<u32>,
    pub top_8_weights: Vec<FloatEvidence>,
    pub top_12_ranked_experts: Vec<RankedExpertEvidence>,
    pub experts_68_and_113: ExpertPairEvidence,
    pub cutoff_margins: CutoffMarginEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SelectedExpertEvidence {
    pub rank: usize,
    pub expert_id: u32,
    pub weight: FloatEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActualGpuRouterEvidence {
    pub source: &'static str,
    pub score_origin: &'static str,
    pub raw_logits: Vec<FloatEvidence>,
    pub scored_probabilities: Vec<FloatEvidence>,
    pub top_8_ids: Vec<u32>,
    pub top_8_weights: Vec<FloatEvidence>,
    pub production_top_8_selected_experts: Vec<SelectedExpertEvidence>,
    pub top_12_ranked_experts: Vec<RankedExpertEvidence>,
    pub experts_68_and_113: ExpertPairEvidence,
    pub cutoff_margins: CutoffMarginEvidence,
    pub cpu_top_8_ids_derived_from_exact_gpu_raw_logits: Vec<u32>,
    pub cpu_top_8_weights_derived_from_exact_gpu_raw_logits: Vec<FloatEvidence>,
    pub production_top_8_set_matches_cpu_derived_set: bool,
    pub only_internal_rank_changes: bool,
    pub selected_weights_paired_with_expert_ids: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ThreeWayRouterEvidence {
    pub authoritative_reference: RouterEvaluationEvidence,
    pub cpu_router_on_exact_gpu_router_input: RouterEvaluationEvidence,
    pub actual_gpu_router: ActualGpuRouterEvidence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PairOrder {
    Expert68Before113,
    Expert113Before68,
    ExactTie,
    Missing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Attribution {
    UpstreamRouterInputDrift,
    GpuRouterGemvDrift,
    GpuRouterSoftmaxTopkDrift,
    ExactTieOrderingDefect,
    MixedNumericalDrift,
    Unresolved,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AttributionEvidence {
    pub attribution: Attribution,
    pub smallest_stage_where_order_first_changes: &'static str,
    pub reference_cpu_order: PairOrder,
    pub gpu_input_cpu_order: PairOrder,
    pub gpu_raw_logits_cpu_order: PairOrder,
    pub actual_gpu_order: PairOrder,
    pub exact_gpu_raw_and_scored_tie: bool,
    pub near_tie_ulp_window: u32,
    pub reference_raw_logit_ulp_distance: Option<u32>,
    pub cpu_on_gpu_input_raw_logit_ulp_distance: Option<u32>,
    pub gpu_raw_logit_ulp_distance: Option<u32>,
    pub minimum_observed_raw_logit_ulp_distance: Option<u32>,
    pub near_tie_observed: bool,
    pub explanation: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouterRankDivergenceReport {
    pub schema: &'static str,
    pub mode: &'static str,
    pub status: &'static str,
    pub diagnostic_complete: bool,
    pub qualification_pass: bool,
    pub failure: Option<String>,
    pub provenance: DiagnosticProvenance,
    pub target: TargetEvidence,
    pub prefix: PrefixEvidence,
    pub router_input: RouterInputEvidence,
    pub gate_identity: GateIdentityEvidence,
    pub evaluations: ThreeWayRouterEvidence,
    pub attribution: AttributionEvidence,
}

impl RouterRankDivergenceReport {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        provenance: DiagnosticProvenance,
        case: &str,
        generated_position: usize,
        layer: usize,
        prompt_token_ids: Vec<u32>,
        reference_generated: Vec<u32>,
        gpu_generated: Vec<u32>,
        reference_hidden: &[f32],
        gpu_hidden: &[f32],
        gate_identity: GateIdentityEvidence,
        authoritative_reference: RouterEvaluationEvidence,
        cpu_on_gpu_input: RouterEvaluationEvidence,
        actual_gpu_router: ActualGpuRouterEvidence,
    ) -> Result<Self, String> {
        if reference_generated.len() != generated_position + 1
            || gpu_generated.len() != generated_position + 1
        {
            return Err("target generation evidence is incomplete".to_string());
        }
        let reference_preceding = reference_generated[..generated_position].to_vec();
        let gpu_preceding = gpu_generated[..generated_position].to_vec();
        let fixed_case = crate::greedy_parity::fixed_case(case)
            .ok_or_else(|| format!("unknown fixed corpus case {case:?}"))?;
        let preceding_generated_prefix_match = reference_preceding == gpu_preceding;
        let failure = if !preceding_generated_prefix_match {
            Some(
                "preceding generated token prefix differs; router attribution is invalid"
                    .to_string(),
            )
        } else if !gate_identity.exact_identity_match {
            Some("reference and GPU runtimes loaded different gate tensors".to_string())
        } else {
            None
        };
        let target_generated_token_match =
            reference_generated[generated_position] == gpu_generated[generated_position];
        let mut attribution = derive_attribution(
            pair_order_from_ranks(&authoritative_reference.experts_68_and_113),
            pair_order_from_ranks(&cpu_on_gpu_input.experts_68_and_113),
            pair_order_from_ranks(&actual_gpu_router.experts_68_and_113),
            pair_order_from_ids(&actual_gpu_router.top_8_ids),
            actual_gpu_router
                .experts_68_and_113
                .expert_68_raw_logit
                .bits
                == actual_gpu_router
                    .experts_68_and_113
                    .expert_113_raw_logit
                    .bits
                && actual_gpu_router.experts_68_and_113.expert_68_score.bits
                    == actual_gpu_router.experts_68_and_113.expert_113_score.bits,
        );
        let observed_ulp_distances = [
            authoritative_reference
                .experts_68_and_113
                .raw_logit_ulp_distance,
            cpu_on_gpu_input.experts_68_and_113.raw_logit_ulp_distance,
            actual_gpu_router.experts_68_and_113.raw_logit_ulp_distance,
        ];
        attribution.reference_raw_logit_ulp_distance = observed_ulp_distances[0];
        attribution.cpu_on_gpu_input_raw_logit_ulp_distance = observed_ulp_distances[1];
        attribution.gpu_raw_logit_ulp_distance = observed_ulp_distances[2];
        attribution.minimum_observed_raw_logit_ulp_distance =
            observed_ulp_distances.into_iter().flatten().min();
        attribution.near_tie_observed = attribution
            .minimum_observed_raw_logit_ulp_distance
            .is_some_and(|distance| distance <= NEAR_TIE_ULP_WINDOW);
        if failure.is_some() {
            attribution = derive_attribution(
                PairOrder::Missing,
                PairOrder::Missing,
                PairOrder::Missing,
                PairOrder::Missing,
                false,
            );
        }
        Ok(Self {
            schema: SCHEMA_VERSION,
            mode: MODE,
            status: if failure.is_some() {
                "fail"
            } else {
                "diagnostic-only"
            },
            diagnostic_complete: failure.is_none(),
            qualification_pass: false,
            failure,
            target: TargetEvidence {
                case: case.to_string(),
                generated_position,
                generated_position_zero_based: true,
                layer,
                expected_reference_ids: authoritative_reference.top_8_ids.clone(),
                observed_gpu_ids: actual_gpu_router.top_8_ids.clone(),
                top_8_set_match: same_set(
                    &authoritative_reference.top_8_ids,
                    &actual_gpu_router.top_8_ids,
                ),
                rank_order_match: authoritative_reference.top_8_ids == actual_gpu_router.top_8_ids,
                only_internal_rank_change: same_set(
                    &authoritative_reference.top_8_ids,
                    &actual_gpu_router.top_8_ids,
                ) && authoritative_reference.top_8_ids
                    != actual_gpu_router.top_8_ids,
            },
            prefix: PrefixEvidence {
                corpus: CorpusEvidence::fixed(),
                prompt_sha256: crate::greedy_parity::sha256_hex(fixed_case.prompt.as_bytes()),
                prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&prompt_token_ids),
                prompt_token_ids,
                reference_preceding_generated_token_ids: reference_preceding,
                gpu_native_preceding_generated_token_ids: gpu_preceding,
                preceding_generated_prefix_match,
                reference_target_generated_token_id: reference_generated[generated_position],
                gpu_native_target_generated_token_id: gpu_generated[generated_position],
                target_generated_token_match,
            },
            router_input: RouterInputEvidence::compare(reference_hidden, gpu_hidden)?,
            gate_identity,
            evaluations: ThreeWayRouterEvidence {
                authoritative_reference,
                cpu_router_on_exact_gpu_router_input: cpu_on_gpu_input,
                actual_gpu_router,
            },
            attribution,
            provenance,
        })
    }
}

pub fn validate_target(
    case: &str,
    generated_position: usize,
    layer: usize,
    expected_adapter_name: &str,
    geometry: GpuNativeModelGeometry,
) -> Result<(), String> {
    if crate::greedy_parity::fixed_case(case).is_none() {
        return Err(format!("unknown fixed corpus case {case:?}"));
    }
    if generated_position >= crate::greedy_parity::OUTPUT_TOKEN_LIMIT {
        return Err(format!(
            "generated position {generated_position} is outside fixed output range 0..{}",
            crate::greedy_parity::OUTPUT_TOKEN_LIMIT
        ));
    }
    if layer >= geometry.num_layers {
        return Err(format!(
            "layer {layer} is outside model layer range 0..{}",
            geometry.num_layers
        ));
    }
    if geometry.num_experts <= HIGHER_EXPERT_ID as usize
        || geometry.top_k != REQUIRED_TOP_K
        || geometry.d_model == 0
    {
        return Err(format!(
            "unsupported expert geometry: num_experts={} top_k={} d_model={}",
            geometry.num_experts, geometry.top_k, geometry.d_model
        ));
    }
    if expected_adapter_name.trim().is_empty() {
        return Err("expected adapter name must be non-empty".to_string());
    }
    Ok(())
}

pub fn evaluate_cpu_router(
    source: &'static str,
    gate: &LinearGate,
    hidden: &[f32],
) -> Result<RouterEvaluationEvidence, String> {
    validate_gate(gate, hidden.len())?;
    let raw_logits = gate.weights.matvec(hidden);
    let mut scores = raw_logits.clone();
    crate::transformer::softmax_inplace(&mut scores);
    let decision = gate.route(hidden);
    build_router_evaluation(
        source,
        "cpu-production-softmax",
        raw_logits,
        scores,
        decision.experts,
        decision.weights,
    )
}

pub fn evaluate_actual_gpu_router(
    raw_logits: Vec<f32>,
    selected_ids: Vec<u32>,
    selected_weights: Vec<f32>,
) -> Result<ActualGpuRouterEvidence, String> {
    if raw_logits.len() <= HIGHER_EXPERT_ID as usize {
        return Err("GPU raw-logit vector does not include target experts".to_string());
    }
    if selected_ids.len() != REQUIRED_TOP_K || selected_weights.len() != REQUIRED_TOP_K {
        return Err("GPU selected expert result does not have top_k=8".to_string());
    }
    validate_finite("GPU raw logits", &raw_logits)?;
    validate_finite("GPU selected weights", &selected_weights)?;
    let mut scores = raw_logits.clone();
    crate::transformer::softmax_inplace(&mut scores);
    let (derived_ids, derived_weights) = deterministic_top_k(&scores, REQUIRED_TOP_K)?;
    let (ranked, pair, cutoff) = ranked_evidence(&raw_logits, &scores)?;
    let production_top_8_set_matches_cpu_derived_set = same_set(&selected_ids, &derived_ids);
    let only_internal_rank_changes =
        production_top_8_set_matches_cpu_derived_set && selected_ids != derived_ids;
    let selected_weights_paired_with_expert_ids = selected_ids
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len()
        == selected_ids.len()
        && selected_weights
            .windows(2)
            .all(|pair| pair[0].total_cmp(&pair[1]) != Ordering::Less);
    let production_top_8_selected_experts = selected_ids
        .iter()
        .copied()
        .zip(selected_weights.iter().copied())
        .enumerate()
        .map(|(rank, (expert_id, weight))| SelectedExpertEvidence {
            rank: rank + 1,
            expert_id,
            weight: FloatEvidence::new(weight),
        })
        .collect();
    Ok(ActualGpuRouterEvidence {
        source: "gpu-hidden-to-gpu-production-router",
        score_origin: "cpu-softmax-derived-from-exact-gpu-raw-logits",
        raw_logits: raw_logits.into_iter().map(FloatEvidence::new).collect(),
        scored_probabilities: scores.into_iter().map(FloatEvidence::new).collect(),
        top_8_ids: selected_ids,
        top_8_weights: selected_weights
            .into_iter()
            .map(FloatEvidence::new)
            .collect(),
        production_top_8_selected_experts,
        top_12_ranked_experts: ranked,
        experts_68_and_113: pair,
        cutoff_margins: cutoff,
        cpu_top_8_ids_derived_from_exact_gpu_raw_logits: derived_ids,
        cpu_top_8_weights_derived_from_exact_gpu_raw_logits: derived_weights
            .into_iter()
            .map(FloatEvidence::new)
            .collect(),
        production_top_8_set_matches_cpu_derived_set,
        only_internal_rank_changes,
        selected_weights_paired_with_expert_ids,
    })
}

fn validate_gate(gate: &LinearGate, hidden_len: usize) -> Result<(), String> {
    if gate.d_model != hidden_len
        || gate.num_experts <= HIGHER_EXPERT_ID as usize
        || gate.top_k != REQUIRED_TOP_K
        || gate.scoring_func != ScoringFunc::Softmax
        || !gate.normalise_topk
        || gate.correction_bias.is_some()
        || gate.n_group > 1
        || gate.routed_scaling_factor.to_bits() != 1.0f32.to_bits()
    {
        return Err("gate does not match the supported Qwen softmax/top-8 contract".to_string());
    }
    Ok(())
}

fn build_router_evaluation(
    source: &'static str,
    score_origin: &'static str,
    raw_logits: Vec<f32>,
    scores: Vec<f32>,
    top_ids: Vec<u32>,
    top_weights: Vec<f32>,
) -> Result<RouterEvaluationEvidence, String> {
    if raw_logits.len() != scores.len()
        || top_ids.len() != REQUIRED_TOP_K
        || top_weights.len() != REQUIRED_TOP_K
    {
        return Err("router evaluation geometry is invalid".to_string());
    }
    validate_finite("raw logits", &raw_logits)?;
    validate_finite("scores", &scores)?;
    validate_finite("top-k weights", &top_weights)?;
    let (ranked, pair, cutoff) = ranked_evidence(&raw_logits, &scores)?;
    Ok(RouterEvaluationEvidence {
        source,
        score_origin,
        raw_logits: raw_logits.into_iter().map(FloatEvidence::new).collect(),
        scored_probabilities: scores.into_iter().map(FloatEvidence::new).collect(),
        top_8_ids: top_ids,
        top_8_weights: top_weights.into_iter().map(FloatEvidence::new).collect(),
        top_12_ranked_experts: ranked,
        experts_68_and_113: pair,
        cutoff_margins: cutoff,
    })
}

fn ranked_evidence(
    raw_logits: &[f32],
    scores: &[f32],
) -> Result<
    (
        Vec<RankedExpertEvidence>,
        ExpertPairEvidence,
        CutoffMarginEvidence,
    ),
    String,
> {
    if raw_logits.len() != scores.len()
        || raw_logits.len() <= HIGHER_EXPERT_ID as usize
        || raw_logits.len() < RANKED_EXPERT_COUNT
    {
        return Err("router rank evidence geometry is invalid".to_string());
    }
    let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ranked.sort_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    let rank_for = |expert: u32| {
        ranked
            .iter()
            .position(|(id, _)| *id == expert as usize)
            .map(|rank| rank + 1)
            .ok_or_else(|| format!("expert {expert} missing from ranking"))
    };
    let expert_68_rank = rank_for(LOWER_EXPERT_ID)?;
    let expert_113_rank = rank_for(HIGHER_EXPERT_ID)?;
    let logit_68 = raw_logits[LOWER_EXPERT_ID as usize];
    let logit_113 = raw_logits[HIGHER_EXPERT_ID as usize];
    let score_68 = scores[LOWER_EXPERT_ID as usize];
    let score_113 = scores[HIGHER_EXPERT_ID as usize];
    let top_12_ranked_experts = ranked
        .iter()
        .take(RANKED_EXPERT_COUNT)
        .enumerate()
        .map(|(rank, (expert_id, score))| RankedExpertEvidence {
            expert_id: *expert_id as u32,
            rank: rank + 1,
            raw_logit: FloatEvidence::new(raw_logits[*expert_id]),
            score: FloatEvidence::new(*score),
        })
        .collect();
    let rank_7 = ranked[6];
    let rank_8 = ranked[7];
    let rank_9 = ranked[8];
    Ok((
        top_12_ranked_experts,
        ExpertPairEvidence {
            expert_68_raw_logit: FloatEvidence::new(logit_68),
            expert_113_raw_logit: FloatEvidence::new(logit_113),
            raw_logit_68_minus_113: FloatEvidence::new(logit_68 - logit_113),
            expert_68_score: FloatEvidence::new(score_68),
            expert_113_score: FloatEvidence::new(score_113),
            score_68_minus_113: FloatEvidence::new(score_68 - score_113),
            expert_68_rank,
            expert_113_rank,
            raw_logit_ulp_distance: ulp_distance(logit_68, logit_113),
            score_ulp_distance: ulp_distance(score_68, score_113),
        },
        CutoffMarginEvidence {
            rank_7_expert_id: rank_7.0 as u32,
            rank_8_expert_id: rank_8.0 as u32,
            rank_9_expert_id: rank_9.0 as u32,
            rank_7_minus_rank_8_score: FloatEvidence::new(rank_7.1 - rank_8.1),
            rank_8_minus_rank_9_score: FloatEvidence::new(rank_8.1 - rank_9.1),
        },
    ))
}

fn deterministic_top_k(scores: &[f32], top_k: usize) -> Result<(Vec<u32>, Vec<f32>), String> {
    validate_finite("scores", scores)?;
    if top_k == 0 || top_k > scores.len() {
        return Err("invalid deterministic top-k geometry".to_string());
    }
    let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ranked.sort_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    let selected = &ranked[..top_k];
    let sum: f32 = selected.iter().map(|(_, score)| *score).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("selected score sum is nonfinite or nonpositive".to_string());
    }
    Ok((
        selected.iter().map(|(id, _)| *id as u32).collect(),
        selected.iter().map(|(_, score)| *score / sum).collect(),
    ))
}

fn same_set(left: &[u32], right: &[u32]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

fn validate_finite(label: &str, values: &[f32]) -> Result<(), String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} is empty or contains a nonfinite value"));
    }
    Ok(())
}

fn pair_order_from_ranks(pair: &ExpertPairEvidence) -> PairOrder {
    if pair.expert_68_raw_logit.bits == pair.expert_113_raw_logit.bits
        && pair.expert_68_score.bits == pair.expert_113_score.bits
    {
        PairOrder::ExactTie
    } else if pair.expert_68_rank < pair.expert_113_rank {
        PairOrder::Expert68Before113
    } else {
        PairOrder::Expert113Before68
    }
}

fn pair_order_from_ids(ids: &[u32]) -> PairOrder {
    match (
        ids.iter().position(|id| *id == LOWER_EXPERT_ID),
        ids.iter().position(|id| *id == HIGHER_EXPERT_ID),
    ) {
        (Some(left), Some(right)) if left < right => PairOrder::Expert68Before113,
        (Some(_), Some(_)) => PairOrder::Expert113Before68,
        _ => PairOrder::Missing,
    }
}

pub fn derive_attribution(
    reference: PairOrder,
    cpu_on_gpu_input: PairOrder,
    cpu_on_gpu_logits: PairOrder,
    actual_gpu: PairOrder,
    exact_gpu_tie: bool,
) -> AttributionEvidence {
    use PairOrder::{Expert113Before68 as Reversed, Expert68Before113 as Expected};
    let (attribution, stage, explanation) = if exact_gpu_tie && actual_gpu == Reversed {
        (
            Attribution::ExactTieOrderingDefect,
            "gpu-softmax-topk",
            "captured GPU logits and CPU-derived scores tie exactly, but production GPU order violates the lower-ID tie break",
        )
    } else if reference != Expected || actual_gpu != Reversed {
        (
            Attribution::Unresolved,
            "unresolved",
            "the captured endpoint ordering does not match the target 68-before-113 to 113-before-68 divergence",
        )
    } else if cpu_on_gpu_input == Reversed && cpu_on_gpu_logits == Expected {
        (
            Attribution::MixedNumericalDrift,
            "mixed",
            "GPU input flips the pair, GPU GEMV flips it back, and GPU softmax/top-k flips it again",
        )
    } else if cpu_on_gpu_input == Reversed && cpu_on_gpu_logits == Reversed {
        (
            Attribution::UpstreamRouterInputDrift,
            "router-input",
            "CPU production routing on the exact GPU input already reproduces 113 before 68",
        )
    } else if cpu_on_gpu_input == Expected && cpu_on_gpu_logits == Reversed {
        (
            Attribution::GpuRouterGemvDrift,
            "gpu-router-gemv",
            "CPU routing on the GPU input keeps 68 before 113, while captured GPU GEMV logits reverse them",
        )
    } else if cpu_on_gpu_input == Expected && cpu_on_gpu_logits == Expected {
        (
            Attribution::GpuRouterSoftmaxTopkDrift,
            "gpu-softmax-topk",
            "CPU selection from exact GPU logits keeps 68 before 113, while production GPU selection reverses them",
        )
    } else {
        (
            Attribution::Unresolved,
            "unresolved",
            "one or more intermediate pair orders are tied or unavailable",
        )
    };
    AttributionEvidence {
        attribution,
        smallest_stage_where_order_first_changes: stage,
        reference_cpu_order: reference,
        gpu_input_cpu_order: cpu_on_gpu_input,
        gpu_raw_logits_cpu_order: cpu_on_gpu_logits,
        actual_gpu_order: actual_gpu,
        exact_gpu_raw_and_scored_tie: exact_gpu_tie,
        near_tie_ulp_window: NEAR_TIE_ULP_WINDOW,
        reference_raw_logit_ulp_distance: None,
        cpu_on_gpu_input_raw_logit_ulp_distance: None,
        gpu_raw_logit_ulp_distance: None,
        minimum_observed_raw_logit_ulp_distance: None,
        near_tie_observed: false,
        explanation,
    }
}

pub fn ulp_distance(left: f32, right: f32) -> Option<u32> {
    if !left.is_finite() || !right.is_finite() {
        return None;
    }
    let ordered = |value: f32| {
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits | 0x8000_0000
        }
    };
    Some(ordered(left).abs_diff(ordered(right)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> GpuNativeModelGeometry {
        GpuNativeModelGeometry {
            num_layers: 48,
            d_model: 2048,
            d_ff: 768,
            num_experts: 128,
            top_k: 8,
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 64,
            rope_dim: 64,
            vocab_size: 151_936,
            max_seq_len: 4096,
            rms_eps: 1.0e-6,
            rope_base: 1_000_000.0,
        }
    }

    #[test]
    fn deterministic_top_k_tie_prefers_lower_expert_id() {
        let mut scores = vec![0.0; 128];
        for (id, value) in [30, 37, 114, 86, 29, 68, 113, 35]
            .into_iter()
            .zip([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 3.0, 2.0])
        {
            scores[id] = value;
        }
        let (ids, _) = deterministic_top_k(&scores, 8).unwrap();
        assert_eq!(ids, vec![30, 37, 114, 86, 29, 68, 113, 35]);
    }

    #[test]
    fn target_layer_trace_layout_round_trips_exact_bits() {
        let mut small = geometry();
        small.d_model = 4;
        small.num_experts = 128;
        let layout = RouterRankTraceLayout::try_new(small, 44).unwrap();
        assert_eq!(layout.router_input_offset, 0);
        assert_eq!(layout.router_input_bytes, 16);
        assert_eq!(layout.raw_logits_offset, 16);
        assert_eq!(layout.raw_logits_bytes, 512);
        assert_eq!(layout.selected_ids_offset, 528);
        assert_eq!(layout.selected_weights_offset, 560);
        assert_eq!(layout.total_bytes, 592);
        let mut bytes = vec![0u8; layout.total_bytes as usize];
        let write_f32 = |bytes: &mut [u8], offset: u64, values: &[f32]| {
            for (index, value) in values.iter().enumerate() {
                bytes[offset as usize + index * 4..offset as usize + index * 4 + 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
        };
        let hidden = [1.0, -0.0, 3.5, -2.0];
        let logits: Vec<f32> = (0..128).map(|index| index as f32 / 10.0).collect();
        let ids = [30u32, 37, 114, 86, 29, 113, 68, 35];
        let weights = [0.3, 0.2, 0.15, 0.12, 0.1, 0.06, 0.04, 0.03];
        write_f32(&mut bytes, layout.router_input_offset, &hidden);
        write_f32(&mut bytes, layout.raw_logits_offset, &logits);
        for (index, value) in ids.iter().enumerate() {
            bytes[layout.selected_ids_offset as usize + index * 4
                ..layout.selected_ids_offset as usize + index * 4 + 4]
                .copy_from_slice(&value.to_le_bytes());
        }
        write_f32(&mut bytes, layout.selected_weights_offset, &weights);
        let trace = layout.parse(&bytes).unwrap();
        assert_eq!(trace.router_input, hidden);
        assert_eq!(trace.raw_logits, logits);
        assert_eq!(trace.selected_ids, ids);
        assert_eq!(trace.selected_weights, weights);
    }

    #[test]
    fn attribution_identifies_upstream_input_flip() {
        let evidence = derive_attribution(
            PairOrder::Expert68Before113,
            PairOrder::Expert113Before68,
            PairOrder::Expert113Before68,
            PairOrder::Expert113Before68,
            false,
        );
        assert_eq!(evidence.attribution, Attribution::UpstreamRouterInputDrift);
    }

    #[test]
    fn attribution_identifies_gpu_gemv_flip() {
        let evidence = derive_attribution(
            PairOrder::Expert68Before113,
            PairOrder::Expert68Before113,
            PairOrder::Expert113Before68,
            PairOrder::Expert113Before68,
            false,
        );
        assert_eq!(evidence.attribution, Attribution::GpuRouterGemvDrift);
    }

    #[test]
    fn attribution_identifies_gpu_softmax_topk_flip() {
        let evidence = derive_attribution(
            PairOrder::Expert68Before113,
            PairOrder::Expert68Before113,
            PairOrder::Expert68Before113,
            PairOrder::Expert113Before68,
            false,
        );
        assert_eq!(evidence.attribution, Attribution::GpuRouterSoftmaxTopkDrift);
    }

    #[test]
    fn attribution_identifies_exact_tie_ordering_defect() {
        let evidence = derive_attribution(
            PairOrder::Expert68Before113,
            PairOrder::Expert68Before113,
            PairOrder::ExactTie,
            PairOrder::Expert113Before68,
            true,
        );
        assert_eq!(evidence.attribution, Attribution::ExactTieOrderingDefect);
    }

    #[test]
    fn attribution_does_not_hide_multi_stage_flips() {
        let evidence = derive_attribution(
            PairOrder::Expert68Before113,
            PairOrder::Expert113Before68,
            PairOrder::Expert68Before113,
            PairOrder::Expert113Before68,
            false,
        );
        assert_eq!(evidence.attribution, Attribution::MixedNumericalDrift);
    }

    #[test]
    fn target_validation_fails_closed() {
        assert!(validate_target("not-a-case", 5, 44, "NVIDIA L4", geometry()).is_err());
        assert!(validate_target("rust-generation", 16, 44, "NVIDIA L4", geometry()).is_err());
        assert!(validate_target("rust-generation", 5, 48, "NVIDIA L4", geometry()).is_err());
        assert!(validate_target("rust-generation", 5, 44, "", geometry()).is_err());
        let mut bad = geometry();
        bad.num_experts = 64;
        assert!(validate_target("rust-generation", 5, 44, "NVIDIA L4", bad).is_err());
    }

    #[test]
    fn schema_and_required_router_fields_are_stable() {
        let mut logits = vec![-10.0; 128];
        for (id, value) in [30, 37, 114, 86, 29, 68, 113, 35, 12, 13, 14, 15]
            .into_iter()
            .zip([
                12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ])
        {
            logits[id] = value;
        }
        let actual = evaluate_actual_gpu_router(
            logits,
            vec![30, 37, 114, 86, 29, 68, 113, 35],
            vec![0.3, 0.2, 0.15, 0.12, 0.1, 0.06, 0.04, 0.03],
        )
        .unwrap();
        let value = serde_json::to_value(actual).unwrap();
        assert_eq!(value["raw_logits"].as_array().unwrap().len(), 128);
        assert_eq!(value["scored_probabilities"].as_array().unwrap().len(), 128);
        assert_eq!(value["top_8_ids"].as_array().unwrap().len(), 8);
        assert_eq!(
            value["production_top_8_selected_experts"]
                .as_array()
                .unwrap()
                .len(),
            8
        );
        assert_eq!(
            value["production_top_8_selected_experts"][5]["expert_id"],
            68
        );
        assert_eq!(value["top_12_ranked_experts"].as_array().unwrap().len(), 12);
        assert!(value["experts_68_and_113"]["raw_logit_ulp_distance"].is_number());
        assert_eq!(SCHEMA_VERSION, "mer.gpu-native-router-rank-divergence.v1");
    }

    #[test]
    fn complete_report_serialization_keeps_diagnostic_contract() {
        let mut logits = vec![-10.0; 128];
        for (id, value) in [30, 37, 114, 86, 29, 68, 113, 35, 12, 13, 14, 15]
            .into_iter()
            .zip([
                12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ])
        {
            logits[id] = value;
        }
        let mut scores = logits.clone();
        crate::transformer::softmax_inplace(&mut scores);
        let (ids, weights) = deterministic_top_k(&scores, 8).unwrap();
        let reference = build_router_evaluation(
            "reference-hidden-to-cpu-production-router",
            "cpu-production-softmax",
            logits.clone(),
            scores.clone(),
            ids.clone(),
            weights.clone(),
        )
        .unwrap();
        let cpu_on_gpu = build_router_evaluation(
            "gpu-hidden-to-cpu-production-router",
            "cpu-production-softmax",
            logits.clone(),
            scores,
            ids,
            weights,
        )
        .unwrap();
        let actual = evaluate_actual_gpu_router(
            logits,
            vec![30, 37, 114, 86, 29, 113, 68, 35],
            vec![0.3, 0.2, 0.15, 0.12, 0.1, 0.06, 0.04, 0.03],
        )
        .unwrap();
        let gate = GateTensorIdentity {
            canonical_tensor_name: "model.layers.44.mlp.gate.weight".to_string(),
            dtype: "f32".to_string(),
            rows: 128,
            cols: 2048,
            loaded_f32_bits_sha256: "a".repeat(64),
        };
        let model_load = ModelLoadEvidence {
            strict: true,
            loader: "safetensors".to_string(),
            loaded_tensors: 1,
            required_tensors: 1,
            optional_probed: 0,
            optional_loaded: 0,
            seeded_fallback_remained: false,
        };
        let shutdown = crate::greedy_parity::BackgroundShutdownEvidence {
            controlled_shutdown_requested: true,
            all_runtime_resources_released: true,
            poll_iterations: 1,
        };
        let report = RouterRankDivergenceReport::build(
            DiagnosticProvenance {
                build: BuildProvenance {
                    git_sha: Some("b".repeat(40)),
                    dirty: Some(false),
                    package_version: "test".to_string(),
                },
                executable_sha256: "c".repeat(64),
                artifacts: QualificationArtifacts::default(),
                gpu_resolved_config_sha256: "d".repeat(64),
                reference_resolved_config_sha256: "e".repeat(64),
                model_identity: ModelIdentityEvidence {
                    architecture: "qwen3_moe".to_string(),
                    num_layers: 48,
                    num_experts_per_layer: 128,
                    total_experts: 6144,
                    top_k: 8,
                    d_model: 2048,
                    d_ff: 768,
                    routed_expert_dtype: "q4_0".to_string(),
                },
                reference_model_load: model_load.clone(),
                gpu_native_model_load: model_load,
                reference_background_shutdown: shutdown,
                gpu_native_background_shutdown: shutdown,
                expert_metadata: ExpertMetadataEvidence {
                    dtype: Some("q4_0".to_string()),
                    q4_0_layout: Some("standard-v1".to_string()),
                    conversion_mode: Some("real".to_string()),
                    source: Some("test".to_string()),
                    explicitly_synthetic: false,
                },
                device: GpuDeviceIdentity {
                    name: "NVIDIA L4".to_string(),
                    vendor_id: 0x10de,
                    device_id: 0,
                    device_type: "DiscreteGpu".to_string(),
                    wgpu_backend: "vulkan".to_string(),
                    driver: "test".to_string(),
                    driver_info: "test".to_string(),
                    compute_plane: "wgpu-vulkan".to_string(),
                    software_adapter: false,
                },
            },
            "rust-generation",
            5,
            44,
            vec![1, 2, 3],
            vec![10, 11, 12, 13, 14, 15],
            vec![10, 11, 12, 13, 14, 15],
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            GateIdentityEvidence::new(gate.clone(), gate),
            reference,
            cpu_on_gpu,
            actual,
        )
        .unwrap();
        let value = serde_json::to_value(report).unwrap();
        assert_eq!(value["schema"], SCHEMA_VERSION);
        assert_eq!(value["status"], "diagnostic-only");
        assert_eq!(value["qualification_pass"], false);
        assert_eq!(value["target"]["top_8_set_match"], true);
        assert_eq!(value["target"]["only_internal_rank_change"], true);
        assert_eq!(
            value["attribution"]["attribution"],
            "gpu-router-softmax-topk-drift"
        );
        assert_eq!(
            value["router_input"]["reference_f32_bits"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
    }

    #[test]
    fn ulp_distance_handles_sign_and_adjacent_values() {
        assert_eq!(
            ulp_distance(1.0, f32::from_bits(1.0f32.to_bits() + 1)),
            Some(1)
        );
        assert_eq!(ulp_distance(-0.0, 0.0), Some(1));
        assert_eq!(ulp_distance(f32::NAN, 0.0), None);
    }
}
