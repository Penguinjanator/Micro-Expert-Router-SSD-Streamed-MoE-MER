//! Diagnostic-only semantic witness for a GPU-native expert-rank permutation.
//!
//! This module owns typed evidence and host-only recomputation. Production
//! routing, expert execution, accumulation, and qualifier PASS semantics do not
//! depend on it.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};

use serde::Serialize;

use crate::gpu_native_router_rank_diagnostics::{
    ActualGpuRouterEvidence, CutoffMarginEvidence, DiagnosticProvenance, GateIdentityEvidence,
    PrefixEvidence, RankedExpertEvidence, RouterEvaluationEvidence, RouterInputEvidence,
    TargetEvidence,
};
use crate::gpu_native_token_loop::GpuNativeModelGeometry;
use crate::numerical_diagnostics::FloatEvidence;

pub const SCHEMA_VERSION: &str = "mer.gpu-native-expert-permutation-semantic-parity.v1";
pub const MODE: &str = "diagnose-gpu-native-expert-permutation-semantic-parity";

/// Target-layer-only staging layout used solely by the semantic diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticTraceLayout {
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
    pub route_outputs_offset: u64,
    pub route_outputs_bytes: u64,
    pub routed_moe_output_offset: u64,
    pub routed_moe_output_bytes: u64,
    pub total_bytes: u64,
}

impl SemanticTraceLayout {
    pub fn try_new(geometry: GpuNativeModelGeometry, target_layer: usize) -> Result<Self, String> {
        if target_layer >= geometry.num_layers
            || geometry.d_model == 0
            || geometry.num_experts == 0
            || geometry.top_k == 0
            || geometry.top_k > geometry.num_experts
        {
            return Err("invalid semantic trace geometry or target layer".to_string());
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
        let route_outputs_offset = selected_weights_offset
            .checked_add(selected_weights_bytes)
            .ok_or("route outputs offset overflow")?;
        let route_outputs_elements = geometry
            .top_k
            .checked_mul(geometry.d_model)
            .ok_or("route outputs element count overflow")?;
        let route_outputs_bytes = bytes(route_outputs_elements, "route outputs")?;
        let routed_moe_output_offset = route_outputs_offset
            .checked_add(route_outputs_bytes)
            .ok_or("routed MoE output offset overflow")?;
        let routed_moe_output_bytes = bytes(geometry.d_model, "routed MoE output")?;
        let total_bytes = routed_moe_output_offset
            .checked_add(routed_moe_output_bytes)
            .ok_or("semantic trace total size overflow")?;
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
            route_outputs_offset,
            route_outputs_bytes,
            routed_moe_output_offset,
            routed_moe_output_bytes,
            total_bytes,
        })
    }

    pub fn parse(&self, bytes: &[u8]) -> Result<SemanticGpuTrace, String> {
        if bytes.len() < self.total_bytes as usize {
            return Err(format!(
                "semantic staging has {} bytes, expected {}",
                bytes.len(),
                self.total_bytes
            ));
        }
        let parse_f32 = |offset: u64, count: usize| {
            bytes[offset as usize..offset as usize + count * 4]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let parse_u32 = |offset: u64, count: usize| {
            bytes[offset as usize..offset as usize + count * 4]
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let flat_outputs = parse_f32(self.route_outputs_offset, self.top_k * self.d_model);
        Ok(SemanticGpuTrace {
            router_input: parse_f32(self.router_input_offset, self.d_model),
            raw_logits: parse_f32(self.raw_logits_offset, self.num_experts),
            selected_ids: parse_u32(self.selected_ids_offset, self.top_k),
            selected_weights: parse_f32(self.selected_weights_offset, self.top_k),
            route_outputs: flat_outputs
                .chunks_exact(self.d_model)
                .map(<[f32]>::to_vec)
                .collect(),
            routed_moe_output: parse_f32(self.routed_moe_output_offset, self.d_model),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticGpuTrace {
    pub router_input: Vec<f32>,
    pub raw_logits: Vec<f32>,
    pub selected_ids: Vec<u32>,
    pub selected_weights: Vec<f32>,
    pub route_outputs: Vec<Vec<f32>>,
    pub routed_moe_output: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CpuSameInputRoutedMoeCapture {
    pub selected_ids: Vec<u32>,
    pub selected_weights: Vec<f32>,
    pub expert_outputs: Vec<Vec<f32>>,
    pub routed_moe_output: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SelectedExpertSemanticEvidence {
    pub expert_id: u32,
    pub rank: usize,
    pub raw_logit: Option<FloatEvidence>,
    pub pre_normalization_score: Option<FloatEvidence>,
    pub selected_normalized_weight: FloatEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoutingViewEvidence {
    pub source: &'static str,
    pub production_rank_order: Vec<SelectedExpertSemanticEvidence>,
    pub canonical_expert_id_order: Vec<SelectedExpertSemanticEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WeightDeltaEvidence {
    pub expert_id: u32,
    pub left: FloatEvidence,
    pub right: FloatEvidence,
    pub absolute_error: Option<f64>,
    pub relative_error: Option<f64>,
    pub ulp_distance: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CanonicalAssociationWitness {
    pub authoritative_cpu_reference_routing: RoutingViewEvidence,
    pub cpu_production_routing_on_exact_gpu_input: RoutingViewEvidence,
    pub actual_production_gpu_routing: RoutingViewEvidence,
    pub reference_vs_gpu_top_k_membership_equal: bool,
    pub cpu_on_gpu_input_vs_actual_gpu_membership_equal: bool,
    pub actual_gpu_selected_weights_paired_with_expert_ids: bool,
    pub reference_vs_gpu_common_expert_weight_deltas_by_expert_id: Vec<WeightDeltaEvidence>,
    pub cpu_on_gpu_input_vs_actual_gpu_weight_deltas_by_expert_id: Vec<WeightDeltaEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VectorNumericalEvidence {
    pub left_source: &'static str,
    pub right_source: &'static str,
    pub vector_length: usize,
    pub left_f32_bits_sha256: String,
    pub right_f32_bits_sha256: String,
    pub exact_bit_equal: bool,
    pub exact_bit_equal_element_count: usize,
    pub max_absolute_error: Option<f64>,
    pub rms_error: Option<f64>,
    pub mean_absolute_error: Option<f64>,
    pub max_relative_error_where_defined: Option<f64>,
    pub relative_error_defined_element_count: usize,
    pub left_nonfinite_count: usize,
    pub right_nonfinite_count: usize,
    pub nonfinite_bit_mismatch_count: usize,
    pub worst_element_index: Option<usize>,
    pub left_value_at_worst_element: Option<FloatEvidence>,
    pub right_value_at_worst_element: Option<FloatEvidence>,
}

impl VectorNumericalEvidence {
    pub fn compare(
        left_source: &'static str,
        right_source: &'static str,
        left: &[f32],
        right: &[f32],
    ) -> Result<Self, String> {
        if left.is_empty() || left.len() != right.len() {
            return Err("semantic vector lengths differ or are empty".to_string());
        }
        let left_bits: Vec<u32> = left.iter().map(|value| value.to_bits()).collect();
        let right_bits: Vec<u32> = right.iter().map(|value| value.to_bits()).collect();
        let exact_bit_equal_element_count = left_bits
            .iter()
            .zip(&right_bits)
            .filter(|(left, right)| left == right)
            .count();
        let mut finite_pairs = 0usize;
        let mut absolute_sum = 0.0f64;
        let mut squared_sum = 0.0f64;
        let mut max_absolute = None::<f64>;
        let mut max_relative = None::<f64>;
        let mut relative_count = 0usize;
        let mut worst_index = None;
        let mut left_nonfinite_count = 0usize;
        let mut right_nonfinite_count = 0usize;
        let mut nonfinite_bit_mismatch_count = 0usize;
        for (index, (&left_value, &right_value)) in left.iter().zip(right).enumerate() {
            left_nonfinite_count += usize::from(!left_value.is_finite());
            right_nonfinite_count += usize::from(!right_value.is_finite());
            if !left_value.is_finite() || !right_value.is_finite() {
                nonfinite_bit_mismatch_count +=
                    usize::from(left_value.to_bits() != right_value.to_bits());
                continue;
            }
            finite_pairs += 1;
            let absolute = (f64::from(right_value) - f64::from(left_value)).abs();
            absolute_sum += absolute;
            squared_sum += absolute * absolute;
            if max_absolute.is_none_or(|current| absolute > current) {
                max_absolute = Some(absolute);
                worst_index = Some(index);
            }
            if left_value != 0.0 {
                relative_count += 1;
                let relative = absolute / f64::from(left_value).abs();
                if relative.is_finite() && max_relative.is_none_or(|current| relative > current) {
                    max_relative = Some(relative);
                }
            }
        }
        Ok(Self {
            left_source,
            right_source,
            vector_length: left.len(),
            left_f32_bits_sha256: crate::numerical_diagnostics::f32_bits_sha256(&left_bits),
            right_f32_bits_sha256: crate::numerical_diagnostics::f32_bits_sha256(&right_bits),
            exact_bit_equal: exact_bit_equal_element_count == left.len(),
            exact_bit_equal_element_count,
            max_absolute_error: max_absolute,
            rms_error: (finite_pairs > 0).then(|| (squared_sum / finite_pairs as f64).sqrt()),
            mean_absolute_error: (finite_pairs > 0).then(|| absolute_sum / finite_pairs as f64),
            max_relative_error_where_defined: max_relative,
            relative_error_defined_element_count: relative_count,
            left_nonfinite_count,
            right_nonfinite_count,
            nonfinite_bit_mismatch_count,
            worst_element_index: worst_index,
            left_value_at_worst_element: worst_index.map(|index| FloatEvidence::new(left[index])),
            right_value_at_worst_element: worst_index.map(|index| FloatEvidence::new(right[index])),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FrozenExpertOutputEvidence {
    pub expert_id: u32,
    pub associated_weight: FloatEvidence,
    pub expert_output_f32_bits_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PermutationOnlyWitness {
    pub frozen_expert_outputs_in_production_rank_order: Vec<FrozenExpertOutputEvidence>,
    pub production_rank_order_expert_ids: Vec<u32>,
    pub canonical_expert_id_sorted_order: Vec<u32>,
    pub comparison: VectorNumericalEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BoundaryViewEvidence {
    pub source: &'static str,
    pub top_12_experts: Vec<RankedExpertEvidence>,
    pub rank_7_expert_id: u32,
    pub rank_8_expert_id: u32,
    pub rank_9_expert_id: u32,
    pub rank_7_minus_rank_8_score_margin: FloatEvidence,
    pub rank_8_minus_rank_9_score_margin: FloatEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MembershipBoundaryWitness {
    pub authoritative_reference: BoundaryViewEvidence,
    pub cpu_on_exact_gpu_input: BoundaryViewEvidence,
    pub actual_gpu: BoundaryViewEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProductionAccumulationSemanticsEvidence {
    pub cpu: &'static str,
    pub gpu: &'static str,
    pub route_order_can_affect_last_bits: bool,
    pub production_order_was_not_changed: bool,
    pub canonicalization_scope: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExpertPermutationSemanticParityReport {
    pub schema: &'static str,
    pub mode: &'static str,
    pub status: &'static str,
    pub diagnostic_complete: bool,
    qualification_pass: bool,
    pub failure: Option<String>,
    pub provenance: DiagnosticProvenance,
    pub target: TargetEvidence,
    pub prefix: PrefixEvidence,
    pub router_input: RouterInputEvidence,
    pub gate_identity: GateIdentityEvidence,
    pub witness_a_canonical_expert_weight_association: CanonicalAssociationWitness,
    pub witness_b_same_input_cpu_vs_gpu_routed_moe_output: VectorNumericalEvidence,
    pub witness_c_reference_vs_gpu_routed_moe_output: VectorNumericalEvidence,
    pub witness_d_permutation_only_accumulation: PermutationOnlyWitness,
    pub witness_e_membership_boundary_context: MembershipBoundaryWitness,
    pub production_accumulation_semantics: ProductionAccumulationSemanticsEvidence,
}

impl ExpertPermutationSemanticParityReport {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        provenance: DiagnosticProvenance,
        case: &str,
        generated_position: usize,
        layer: usize,
        prompt_token_ids: Vec<u32>,
        reference_generated: Vec<u32>,
        gpu_generated: Vec<u32>,
        reference_router_input: &[f32],
        gpu_router_input: &[f32],
        gate_identity: GateIdentityEvidence,
        authoritative_reference: RouterEvaluationEvidence,
        cpu_on_gpu_input: RouterEvaluationEvidence,
        actual_gpu: ActualGpuRouterEvidence,
        reference_routed_moe_output: &[f32],
        same_input_cpu_routed_moe_output: &[f32],
        actual_gpu_routed_moe_output: &[f32],
        actual_gpu_route_outputs: &[Vec<f32>],
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
                "preceding generated token prefix differs; semantic comparison is invalid"
                    .to_string(),
            )
        } else if !gate_identity.exact_identity_match {
            Some("reference and GPU runtimes loaded different gate tensors".to_string())
        } else {
            None
        };
        let target_generated_token_match =
            reference_generated[generated_position] == gpu_generated[generated_position];
        let witness_a = CanonicalAssociationWitness::build(
            &authoritative_reference,
            &cpu_on_gpu_input,
            &actual_gpu,
        )?;
        let witness_b = VectorNumericalEvidence::compare(
            "cpu-production-routed-moe-on-exact-gpu-input",
            "actual-production-gpu-routed-moe-on-exact-gpu-input",
            same_input_cpu_routed_moe_output,
            actual_gpu_routed_moe_output,
        )?;
        let witness_c = VectorNumericalEvidence::compare(
            "authoritative-cpu-reference-routed-moe",
            "actual-production-gpu-routed-moe",
            reference_routed_moe_output,
            actual_gpu_routed_moe_output,
        )?;
        let witness_d = permutation_only_witness(
            &actual_gpu.top_8_ids,
            &actual_gpu
                .top_8_weights
                .iter()
                .map(|weight| f32::from_bits(weight.bits))
                .collect::<Vec<_>>(),
            actual_gpu_route_outputs,
        )?;
        let witness_e = MembershipBoundaryWitness {
            authoritative_reference: boundary_view(
                authoritative_reference.source,
                &authoritative_reference.top_12_ranked_experts,
                &authoritative_reference.cutoff_margins,
            ),
            cpu_on_exact_gpu_input: boundary_view(
                cpu_on_gpu_input.source,
                &cpu_on_gpu_input.top_12_ranked_experts,
                &cpu_on_gpu_input.cutoff_margins,
            ),
            actual_gpu: boundary_view(
                actual_gpu.source,
                &actual_gpu.top_12_ranked_experts,
                &actual_gpu.cutoff_margins,
            ),
        };
        Ok(Self {
            schema: SCHEMA_VERSION,
            mode: MODE,
            status: if failure.is_some() {
                "incomplete"
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
                observed_gpu_ids: actual_gpu.top_8_ids.clone(),
                top_8_set_match: same_membership(
                    &authoritative_reference.top_8_ids,
                    &actual_gpu.top_8_ids,
                ),
                rank_order_match: authoritative_reference.top_8_ids == actual_gpu.top_8_ids,
                only_internal_rank_change: same_membership(
                    &authoritative_reference.top_8_ids,
                    &actual_gpu.top_8_ids,
                ) && authoritative_reference.top_8_ids != actual_gpu.top_8_ids,
            },
            prefix: PrefixEvidence {
                corpus: crate::greedy_parity::CorpusEvidence::fixed(),
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
            router_input: RouterInputEvidence::compare(
                reference_router_input,
                gpu_router_input,
            )?,
            gate_identity,
            witness_a_canonical_expert_weight_association: witness_a,
            witness_b_same_input_cpu_vs_gpu_routed_moe_output: witness_b,
            witness_c_reference_vs_gpu_routed_moe_output: witness_c,
            witness_d_permutation_only_accumulation: witness_d,
            witness_e_membership_boundary_context: witness_e,
            production_accumulation_semantics: ProductionAccumulationSemanticsEvidence {
                cpu: "sequential f32: accumulator[element] += weight[route] * expert_output[route][element]",
                gpu: "sequential WGSL f32 loop over route slots in expert_combine_main",
                route_order_can_affect_last_bits: true,
                production_order_was_not_changed: true,
                canonicalization_scope: "diagnostic-only host recomputation over frozen GPU expert outputs",
            },
            provenance,
        })
    }

    pub const fn qualification_pass(&self) -> bool {
        self.qualification_pass
    }
}

impl CanonicalAssociationWitness {
    pub fn build(
        reference: &RouterEvaluationEvidence,
        cpu_on_gpu: &RouterEvaluationEvidence,
        actual_gpu: &ActualGpuRouterEvidence,
    ) -> Result<Self, String> {
        let reference_view = cpu_route_view(reference)?;
        let cpu_on_gpu_view = cpu_route_view(cpu_on_gpu)?;
        let actual_gpu_view = gpu_route_view(actual_gpu)?;
        Ok(Self {
            reference_vs_gpu_top_k_membership_equal: same_membership(
                &reference.top_8_ids,
                &actual_gpu.top_8_ids,
            ),
            cpu_on_gpu_input_vs_actual_gpu_membership_equal: same_membership(
                &cpu_on_gpu.top_8_ids,
                &actual_gpu.top_8_ids,
            ),
            actual_gpu_selected_weights_paired_with_expert_ids:
                actual_gpu_pairing_is_structurally_consistent(actual_gpu),
            reference_vs_gpu_common_expert_weight_deltas_by_expert_id: weight_deltas(
                &reference_view.production_rank_order,
                &actual_gpu_view.production_rank_order,
            ),
            cpu_on_gpu_input_vs_actual_gpu_weight_deltas_by_expert_id: weight_deltas(
                &cpu_on_gpu_view.production_rank_order,
                &actual_gpu_view.production_rank_order,
            ),
            authoritative_cpu_reference_routing: reference_view,
            cpu_production_routing_on_exact_gpu_input: cpu_on_gpu_view,
            actual_production_gpu_routing: actual_gpu_view,
        })
    }
}

fn cpu_route_view(evaluation: &RouterEvaluationEvidence) -> Result<RoutingViewEvidence, String> {
    route_view(
        evaluation.source,
        &evaluation.raw_logits,
        &evaluation.scored_probabilities,
        &evaluation.top_8_ids,
        &evaluation.top_8_weights,
    )
}

fn gpu_route_view(evaluation: &ActualGpuRouterEvidence) -> Result<RoutingViewEvidence, String> {
    route_view(
        evaluation.source,
        &evaluation.raw_logits,
        &evaluation.scored_probabilities,
        &evaluation.top_8_ids,
        &evaluation.top_8_weights,
    )
}

fn route_view(
    source: &'static str,
    raw_logits: &[FloatEvidence],
    scores: &[FloatEvidence],
    ids: &[u32],
    weights: &[FloatEvidence],
) -> Result<RoutingViewEvidence, String> {
    if ids.is_empty() || ids.len() != weights.len() {
        return Err("selected expert IDs and weights are empty or misaligned".to_string());
    }
    let mut seen = HashSet::with_capacity(ids.len());
    let production_rank_order = ids
        .iter()
        .copied()
        .zip(weights.iter().cloned())
        .enumerate()
        .map(|(rank, (expert_id, selected_normalized_weight))| {
            if !seen.insert(expert_id) {
                return Err(format!("duplicate selected expert ID {expert_id}"));
            }
            let index = expert_id as usize;
            Ok(SelectedExpertSemanticEvidence {
                expert_id,
                rank: rank + 1,
                raw_logit: raw_logits.get(index).cloned(),
                pre_normalization_score: scores.get(index).cloned(),
                selected_normalized_weight,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut canonical_expert_id_order = production_rank_order.clone();
    canonical_expert_id_order.sort_by_key(|expert| expert.expert_id);
    Ok(RoutingViewEvidence {
        source,
        production_rank_order,
        canonical_expert_id_order,
    })
}

fn weight_deltas(
    left: &[SelectedExpertSemanticEvidence],
    right: &[SelectedExpertSemanticEvidence],
) -> Vec<WeightDeltaEvidence> {
    let left_by_id: BTreeMap<u32, f32> = left
        .iter()
        .map(|item| {
            (
                item.expert_id,
                f32::from_bits(item.selected_normalized_weight.bits),
            )
        })
        .collect();
    let right_by_id: BTreeMap<u32, f32> = right
        .iter()
        .map(|item| {
            (
                item.expert_id,
                f32::from_bits(item.selected_normalized_weight.bits),
            )
        })
        .collect();
    left_by_id
        .into_iter()
        .filter_map(|(expert_id, left)| {
            right_by_id.get(&expert_id).copied().map(|right| {
                let absolute = (f64::from(right) - f64::from(left)).abs();
                WeightDeltaEvidence {
                    expert_id,
                    left: FloatEvidence::new(left),
                    right: FloatEvidence::new(right),
                    absolute_error: (left.is_finite() && right.is_finite()).then_some(absolute),
                    relative_error: (left.is_finite() && right.is_finite() && left != 0.0)
                        .then(|| absolute / f64::from(left).abs())
                        .filter(|value| value.is_finite()),
                    ulp_distance: crate::gpu_native_router_rank_diagnostics::ulp_distance(
                        left, right,
                    ),
                }
            })
        })
        .collect()
}

fn actual_gpu_pairing_is_structurally_consistent(actual: &ActualGpuRouterEvidence) -> bool {
    if actual.top_8_ids.len() != actual.top_8_weights.len()
        || actual
            .top_8_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len()
            != actual.top_8_ids.len()
    {
        return false;
    }
    let mut expected_ids = actual.top_8_ids.clone();
    expected_ids.sort_by(|left, right| {
        let left_score = actual
            .scored_probabilities
            .get(*left as usize)
            .and_then(|value| value.value);
        let right_score = actual
            .scored_probabilities
            .get(*right as usize)
            .and_then(|value| value.value);
        match (left_score, right_score) {
            (Some(left_score), Some(right_score)) => right_score
                .total_cmp(&left_score)
                .then_with(|| left.cmp(right)),
            _ => Ordering::Equal,
        }
    });
    expected_ids == actual.top_8_ids
        && actual
            .top_8_weights
            .windows(2)
            .all(|pair| match (pair[0].value, pair[1].value) {
                (Some(left), Some(right)) => left.total_cmp(&right) != Ordering::Less,
                _ => false,
            })
}

pub fn same_membership(left: &[u32], right: &[u32]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

pub fn permutation_only_witness(
    ids: &[u32],
    weights: &[f32],
    expert_outputs: &[Vec<f32>],
) -> Result<PermutationOnlyWitness, String> {
    if ids.is_empty() || ids.len() != weights.len() || ids.len() != expert_outputs.len() {
        return Err("permutation witness inputs are empty or misaligned".to_string());
    }
    if ids.iter().copied().collect::<HashSet<_>>().len() != ids.len() {
        return Err("permutation witness contains duplicate expert IDs".to_string());
    }
    let width = expert_outputs[0].len();
    if width == 0 || expert_outputs.iter().any(|output| output.len() != width) {
        return Err("permutation witness expert output geometry is invalid".to_string());
    }
    let production = ids
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .zip(expert_outputs.iter())
        .map(|((id, weight), output)| (id, weight, output))
        .collect::<Vec<_>>();
    let mut canonical = production.clone();
    canonical.sort_by_key(|(id, _, _)| *id);
    let accumulate = |ordered: &[(u32, f32, &Vec<f32>)]| {
        let mut output = vec![0.0f32; width];
        for (_, weight, expert_output) in ordered {
            for (destination, value) in output.iter_mut().zip(expert_output.iter()) {
                *destination += *weight * *value;
            }
        }
        output
    };
    let production_sum = accumulate(&production);
    let canonical_sum = accumulate(&canonical);
    Ok(PermutationOnlyWitness {
        frozen_expert_outputs_in_production_rank_order: production
            .iter()
            .map(|(expert_id, weight, output)| FrozenExpertOutputEvidence {
                expert_id: *expert_id,
                associated_weight: FloatEvidence::new(*weight),
                expert_output_f32_bits_sha256: crate::numerical_diagnostics::f32_bits_sha256(
                    &output
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                ),
            })
            .collect(),
        production_rank_order_expert_ids: production.iter().map(|(id, _, _)| *id).collect(),
        canonical_expert_id_sorted_order: canonical.iter().map(|(id, _, _)| *id).collect(),
        comparison: VectorNumericalEvidence::compare(
            "host-accumulation-production-rank-order",
            "host-accumulation-canonical-expert-id-order",
            &production_sum,
            &canonical_sum,
        )?,
    })
}

fn boundary_view(
    source: &'static str,
    top_12: &[RankedExpertEvidence],
    cutoff: &CutoffMarginEvidence,
) -> BoundaryViewEvidence {
    BoundaryViewEvidence {
        source,
        top_12_experts: top_12.to_vec(),
        rank_7_expert_id: cutoff.rank_7_expert_id,
        rank_8_expert_id: cutoff.rank_8_expert_id,
        rank_9_expert_id: cutoff.rank_9_expert_id,
        rank_7_minus_rank_8_score_margin: cutoff.rank_7_minus_rank_8_score.clone(),
        rank_8_minus_rank_9_score_margin: cutoff.rank_8_minus_rank_9_score.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> GpuNativeModelGeometry {
        GpuNativeModelGeometry {
            num_layers: 48,
            d_model: 4,
            d_ff: 8,
            num_experts: 128,
            top_k: 8,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            rope_dim: 2,
            vocab_size: 32,
            max_seq_len: 64,
            rms_eps: 1.0e-6,
            rope_base: 1_000_000.0,
        }
    }

    #[test]
    fn semantic_trace_layout_round_trips_every_observation() {
        let layout = SemanticTraceLayout::try_new(geometry(), 44).unwrap();
        let mut bytes = vec![0u8; layout.total_bytes as usize];
        let write_f32 = |bytes: &mut [u8], offset: u64, values: &[f32]| {
            for (index, value) in values.iter().enumerate() {
                bytes[offset as usize + index * 4..offset as usize + index * 4 + 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
        };
        let ids = [30u32, 37, 114, 86, 29, 113, 68, 35];
        for (index, id) in ids.iter().enumerate() {
            bytes[layout.selected_ids_offset as usize + index * 4
                ..layout.selected_ids_offset as usize + index * 4 + 4]
                .copy_from_slice(&id.to_le_bytes());
        }
        write_f32(
            &mut bytes,
            layout.router_input_offset,
            &[1.0, 2.0, 3.0, 4.0],
        );
        write_f32(
            &mut bytes,
            layout.raw_logits_offset,
            &(0..128).map(|value| value as f32).collect::<Vec<_>>(),
        );
        write_f32(
            &mut bytes,
            layout.selected_weights_offset,
            &[0.3, 0.2, 0.15, 0.12, 0.1, 0.06, 0.04, 0.03],
        );
        write_f32(
            &mut bytes,
            layout.route_outputs_offset,
            &(0..32).map(|value| value as f32).collect::<Vec<_>>(),
        );
        write_f32(
            &mut bytes,
            layout.routed_moe_output_offset,
            &[5.0, 6.0, 7.0, 8.0],
        );
        let trace = layout.parse(&bytes).unwrap();
        assert_eq!(trace.selected_ids, ids);
        assert_eq!(trace.route_outputs.len(), 8);
        assert_eq!(trace.route_outputs[7], [28.0, 29.0, 30.0, 31.0]);
        assert_eq!(trace.routed_moe_output, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn canonicalization_preserves_expert_weight_association() {
        let outputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let witness = permutation_only_witness(&[9, 2, 5], &[0.1, 0.2, 0.7], &outputs).unwrap();
        assert_eq!(witness.production_rank_order_expert_ids, [9, 2, 5]);
        assert_eq!(witness.canonical_expert_id_sorted_order, [2, 5, 9]);
        let by_id: BTreeMap<_, _> = witness
            .frozen_expert_outputs_in_production_rank_order
            .iter()
            .map(|item| (item.expert_id, item.associated_weight.bits))
            .collect();
        assert_eq!(by_id[&9], 0.1f32.to_bits());
        assert_eq!(by_id[&2], 0.2f32.to_bits());
        assert_eq!(by_id[&5], 0.7f32.to_bits());
    }

    #[test]
    fn membership_detects_permutation_and_difference() {
        assert!(same_membership(&[68, 113, 35], &[113, 68, 35]));
        assert!(!same_membership(&[68, 113, 35], &[113, 67, 35]));
    }

    #[test]
    fn permutation_uses_identical_frozen_expert_outputs() {
        let outputs = vec![
            vec![-16_777_216.0, 1.0],
            vec![16_777_216.0, 2.0],
            vec![1.0, 3.0],
        ];
        let witness = permutation_only_witness(&[9, 2, 5], &[1.0; 3], &outputs).unwrap();
        assert_eq!(witness.comparison.vector_length, 2);
        assert_eq!(witness.comparison.exact_bit_equal_element_count, 1);
        assert!(!witness.comparison.exact_bit_equal);
    }

    #[test]
    fn vector_metrics_cover_exact_and_relative_denominators() {
        let exact =
            VectorNumericalEvidence::compare("left", "right", &[0.0, 1.0], &[0.0, 1.0]).unwrap();
        assert!(exact.exact_bit_equal);
        assert_eq!(exact.relative_error_defined_element_count, 1);
        assert_eq!(exact.max_absolute_error, Some(0.0));

        let near_zero = f32::from_bits(1);
        let evidence =
            VectorNumericalEvidence::compare("left", "right", &[0.0, near_zero], &[1.0, 1.0])
                .unwrap();
        assert_eq!(evidence.relative_error_defined_element_count, 1);
        assert!(evidence.max_relative_error_where_defined.unwrap() > 1.0e40);
    }

    #[test]
    fn vector_metrics_report_nonfinite_bits_without_serialization_failure() {
        let evidence = VectorNumericalEvidence::compare(
            "left",
            "right",
            &[f32::NAN, f32::INFINITY, 1.0],
            &[f32::NAN, f32::NEG_INFINITY, 2.0],
        )
        .unwrap();
        assert_eq!(evidence.left_nonfinite_count, 2);
        assert_eq!(evidence.right_nonfinite_count, 2);
        assert_eq!(evidence.nonfinite_bit_mismatch_count, 1);
        serde_json::to_string(&evidence).unwrap();
    }

    #[test]
    fn ulp_metric_is_raw_and_threshold_free() {
        assert_eq!(
            crate::gpu_native_router_rank_diagnostics::ulp_distance(
                1.0,
                f32::from_bits(1.0f32.to_bits() + 1),
            ),
            Some(1)
        );
        assert_eq!(
            crate::gpu_native_router_rank_diagnostics::ulp_distance(f32::NAN, 1.0),
            None
        );
    }

    #[test]
    fn schema_name_and_qualification_contract_are_stable() {
        assert_eq!(
            SCHEMA_VERSION,
            "mer.gpu-native-expert-permutation-semantic-parity.v1"
        );
        #[derive(Serialize)]
        struct Contract<'a> {
            schema: &'a str,
            qualification_pass: bool,
        }
        let value = serde_json::to_value(Contract {
            schema: SCHEMA_VERSION,
            qualification_pass: false,
        })
        .unwrap();
        assert_eq!(value["schema"], SCHEMA_VERSION);
        assert_eq!(value["qualification_pass"], false);
    }

    #[test]
    fn same_membership_with_different_weight_pairing_is_detected() {
        let mut logits = vec![-10.0; 128];
        let ids = vec![30, 37, 114, 86, 29, 113, 68, 35];
        for (rank, id) in ids.iter().copied().enumerate() {
            logits[id as usize] = 10.0 - rank as f32;
        }
        let correctly_paired =
            crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
                logits.clone(),
                ids.clone(),
                vec![0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03],
            )
            .unwrap();
        assert!(actual_gpu_pairing_is_structurally_consistent(
            &correctly_paired
        ));
        let wrongly_paired = crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
            logits,
            ids,
            vec![0.20, 0.30, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03],
        )
        .unwrap();
        assert!(!actual_gpu_pairing_is_structurally_consistent(
            &wrongly_paired
        ));
    }

    #[test]
    fn complete_report_json_serialization_hardcodes_qualification_false() {
        let mut logits = vec![-10.0; 128];
        let actual_ids = vec![30, 37, 114, 86, 29, 113, 68, 35];
        for (rank, id) in actual_ids.iter().copied().enumerate() {
            logits[id as usize] = 10.0 - rank as f32;
        }
        let actual = crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
            logits,
            actual_ids,
            vec![0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03],
        )
        .unwrap();
        let cpu_view = |source: &'static str, ids: Vec<u32>| RouterEvaluationEvidence {
            source,
            score_origin: "cpu-production-softmax",
            raw_logits: actual.raw_logits.clone(),
            scored_probabilities: actual.scored_probabilities.clone(),
            top_8_ids: ids,
            top_8_weights: actual.top_8_weights.clone(),
            top_12_ranked_experts: actual.top_12_ranked_experts.clone(),
            experts_68_and_113: actual.experts_68_and_113.clone(),
            cutoff_margins: actual.cutoff_margins.clone(),
        };
        let reference = cpu_view(
            "reference-hidden-to-cpu-production-router",
            vec![30, 37, 114, 86, 29, 68, 113, 35],
        );
        let cpu_on_gpu = cpu_view(
            "gpu-hidden-to-cpu-production-router",
            vec![30, 37, 114, 86, 29, 113, 68, 35],
        );
        let gate = crate::gpu_native_router_rank_diagnostics::GateTensorIdentity {
            canonical_tensor_name: "model.layers.44.mlp.gate.weight".to_string(),
            dtype: "f32".to_string(),
            rows: 128,
            cols: 4,
            loaded_f32_bits_sha256: "a".repeat(64),
        };
        let model_load = crate::greedy_parity::ModelLoadEvidence {
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
        let report = ExpertPermutationSemanticParityReport::build(
            DiagnosticProvenance {
                build: crate::qualification::BuildProvenance {
                    git_sha: Some("b".repeat(40)),
                    dirty: Some(false),
                    package_version: "test".to_string(),
                },
                executable_sha256: "c".repeat(64),
                artifacts: crate::qualification::QualificationArtifacts::default(),
                gpu_resolved_config_sha256: "d".repeat(64),
                reference_resolved_config_sha256: "e".repeat(64),
                model_identity: crate::greedy_parity::ModelIdentityEvidence {
                    architecture: "qwen3_moe".to_string(),
                    num_layers: 48,
                    num_experts_per_layer: 128,
                    total_experts: 6144,
                    top_k: 8,
                    d_model: 4,
                    d_ff: 8,
                    routed_expert_dtype: "q4_0".to_string(),
                },
                reference_model_load: model_load.clone(),
                gpu_native_model_load: model_load,
                reference_background_shutdown: shutdown,
                gpu_native_background_shutdown: shutdown,
                expert_metadata: crate::qualification::ExpertMetadataEvidence {
                    dtype: Some("q4_0".to_string()),
                    q4_0_layout: Some("standard-v1".to_string()),
                    conversion_mode: Some("real".to_string()),
                    source: Some("test".to_string()),
                    explicitly_synthetic: false,
                },
                device: crate::backend::GpuDeviceIdentity {
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
            0,
            44,
            vec![1, 2, 3],
            vec![10],
            vec![10],
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            GateIdentityEvidence::new(gate.clone(), gate),
            reference,
            cpu_on_gpu,
            actual,
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            &vec![vec![1.0, 2.0, 3.0, 4.0]; 8],
        )
        .unwrap();
        assert!(!report.qualification_pass());
        let value = serde_json::to_value(report).unwrap();
        assert_eq!(value["schema"], SCHEMA_VERSION);
        assert_eq!(value["mode"], MODE);
        assert_eq!(value["qualification_pass"], false);
        assert!(value["witness_b_same_input_cpu_vs_gpu_routed_moe_output"].is_object());
        assert!(value["witness_d_permutation_only_accumulation"].is_object());
    }
}
