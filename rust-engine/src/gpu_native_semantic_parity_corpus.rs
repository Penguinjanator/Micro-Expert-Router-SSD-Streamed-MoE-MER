//! Diagnostic-only full-corpus GPU-native semantic-parity evidence.
//!
//! This module contains only observation layouts, typed report evidence, and
//! host-side deterministic comparison/aggregation. It is not consulted by
//! production inference or by any qualification PASS decision.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;

use crate::gpu_native_expert_permutation_semantic_parity::{
    PermutationOnlyWitness, VectorNumericalEvidence, WeightDeltaEvidence,
};
use crate::gpu_native_router_rank_diagnostics::{
    ActualGpuRouterEvidence, DiagnosticProvenance, RankedExpertEvidence, RouterEvaluationEvidence,
};
use crate::gpu_native_token_loop::GpuNativeModelGeometry;
use crate::greedy_parity::CorpusEvidence;
use crate::numerical_diagnostics::FloatEvidence;

use crate::{IsolatedRuntimeShutdownError, RealCliRuntimeMode, ResolvedRealCliSpec};

pub const SCHEMA_VERSION: &str = "mer.gpu-native-semantic-parity-corpus.v1";
pub const MODE: &str = "diagnose-gpu-native-semantic-parity-corpus";
pub const EXACT_ORDER_SAMPLING_RULE: &str =
    "first-exact-order-event-per-layer-in-frozen-corpus-order";
pub const PERCENTILE_CONVENTION: &str =
    "nearest-rank: sort ascending; rank=max(1,ceil(p*n)); select rank-1";

/// Full-layer staging layout used only by the corpus semantic diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticCorpusTraceLayout {
    pub num_layers: usize,
    pub d_model: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub router_inputs_offset: u64,
    pub raw_logits_offset: u64,
    pub selected_ids_offset: u64,
    pub selected_weights_offset: u64,
    pub route_outputs_offset: u64,
    pub routed_moe_outputs_offset: u64,
    pub router_input_layer_bytes: u64,
    pub raw_logits_layer_bytes: u64,
    pub selected_ids_layer_bytes: u64,
    pub selected_weights_layer_bytes: u64,
    pub route_outputs_layer_bytes: u64,
    pub routed_moe_output_layer_bytes: u64,
    pub total_bytes: u64,
}

impl SemanticCorpusTraceLayout {
    pub fn try_new(geometry: GpuNativeModelGeometry) -> Result<Self, String> {
        if geometry.num_layers == 0
            || geometry.d_model == 0
            || geometry.num_experts == 0
            || geometry.top_k == 0
            || geometry.top_k > geometry.num_experts
        {
            return Err("invalid semantic corpus trace geometry".to_string());
        }
        let bytes = |elements: usize, label: &str| {
            elements
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| format!("{label} byte size overflow"))
        };
        let region = |offset: u64, per_layer: u64, label: &str| {
            per_layer
                .checked_mul(geometry.num_layers as u64)
                .and_then(|size| offset.checked_add(size))
                .ok_or_else(|| format!("{label} region overflow"))
        };

        let router_input_layer_bytes = bytes(geometry.d_model, "router input")?;
        let raw_logits_layer_bytes = bytes(geometry.num_experts, "raw logits")?;
        let selected_ids_layer_bytes = bytes(geometry.top_k, "selected IDs")?;
        let selected_weights_layer_bytes = bytes(geometry.top_k, "selected weights")?;
        let route_outputs_layer_bytes = bytes(
            geometry
                .top_k
                .checked_mul(geometry.d_model)
                .ok_or("route output element count overflow")?,
            "route outputs",
        )?;
        let routed_moe_output_layer_bytes = bytes(geometry.d_model, "routed MoE output")?;

        let router_inputs_offset = 0;
        let raw_logits_offset = region(
            router_inputs_offset,
            router_input_layer_bytes,
            "router inputs",
        )?;
        let selected_ids_offset = region(raw_logits_offset, raw_logits_layer_bytes, "raw logits")?;
        let selected_weights_offset = region(
            selected_ids_offset,
            selected_ids_layer_bytes,
            "selected IDs",
        )?;
        let route_outputs_offset = region(
            selected_weights_offset,
            selected_weights_layer_bytes,
            "selected weights",
        )?;
        let routed_moe_outputs_offset = region(
            route_outputs_offset,
            route_outputs_layer_bytes,
            "route outputs",
        )?;
        let total_bytes = region(
            routed_moe_outputs_offset,
            routed_moe_output_layer_bytes,
            "routed MoE outputs",
        )?;

        Ok(Self {
            num_layers: geometry.num_layers,
            d_model: geometry.d_model,
            num_experts: geometry.num_experts,
            top_k: geometry.top_k,
            router_inputs_offset,
            raw_logits_offset,
            selected_ids_offset,
            selected_weights_offset,
            route_outputs_offset,
            routed_moe_outputs_offset,
            router_input_layer_bytes,
            raw_logits_layer_bytes,
            selected_ids_layer_bytes,
            selected_weights_layer_bytes,
            route_outputs_layer_bytes,
            routed_moe_output_layer_bytes,
            total_bytes,
        })
    }

    fn layer_offset(base: u64, layer_bytes: u64, layer: usize) -> u64 {
        base + layer_bytes * layer as u64
    }

    pub fn router_input_offset(&self, layer: usize) -> u64 {
        Self::layer_offset(
            self.router_inputs_offset,
            self.router_input_layer_bytes,
            layer,
        )
    }

    pub fn raw_logits_offset(&self, layer: usize) -> u64 {
        Self::layer_offset(self.raw_logits_offset, self.raw_logits_layer_bytes, layer)
    }

    pub fn selected_ids_offset(&self, layer: usize) -> u64 {
        Self::layer_offset(
            self.selected_ids_offset,
            self.selected_ids_layer_bytes,
            layer,
        )
    }

    pub fn selected_weights_offset(&self, layer: usize) -> u64 {
        Self::layer_offset(
            self.selected_weights_offset,
            self.selected_weights_layer_bytes,
            layer,
        )
    }

    pub fn route_outputs_offset(&self, layer: usize) -> u64 {
        Self::layer_offset(
            self.route_outputs_offset,
            self.route_outputs_layer_bytes,
            layer,
        )
    }

    pub fn routed_moe_output_offset(&self, layer: usize) -> u64 {
        Self::layer_offset(
            self.routed_moe_outputs_offset,
            self.routed_moe_output_layer_bytes,
            layer,
        )
    }

    pub fn parse(&self, bytes: &[u8]) -> Result<SemanticCorpusGpuTrace, String> {
        if bytes.len() < self.total_bytes as usize {
            return Err(format!(
                "semantic corpus staging has {} bytes, expected {}",
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
        let mut layers = Vec::with_capacity(self.num_layers);
        for layer in 0..self.num_layers {
            let flat_outputs =
                parse_f32(self.route_outputs_offset(layer), self.top_k * self.d_model);
            layers.push(SemanticCorpusGpuLayerTrace {
                router_input: parse_f32(self.router_input_offset(layer), self.d_model),
                raw_logits: parse_f32(self.raw_logits_offset(layer), self.num_experts),
                selected_ids: parse_u32(self.selected_ids_offset(layer), self.top_k),
                selected_weights: parse_f32(self.selected_weights_offset(layer), self.top_k),
                route_outputs: flat_outputs
                    .chunks_exact(self.d_model)
                    .map(<[f32]>::to_vec)
                    .collect(),
                routed_moe_output: parse_f32(self.routed_moe_output_offset(layer), self.d_model),
            });
        }
        Ok(SemanticCorpusGpuTrace { layers })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticCorpusGpuTrace {
    pub layers: Vec<SemanticCorpusGpuLayerTrace>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticCorpusGpuLayerTrace {
    pub router_input: Vec<f32>,
    pub raw_logits: Vec<f32>,
    pub selected_ids: Vec<u32>,
    pub selected_weights: Vec<f32>,
    pub route_outputs: Vec<Vec<f32>>,
    pub routed_moe_output: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RoutingClassification {
    ExactOrderMatch,
    InternalRankPermutation,
    MembershipMismatch,
    InvalidNonfiniteIncomplete,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct EventLocation {
    pub case: String,
    pub generated_position: usize,
    pub layer: usize,
}

impl EventLocation {
    pub fn new(case: impl Into<String>, generated_position: usize, layer: usize) -> Self {
        Self {
            case: case.into(),
            generated_position,
            layer,
        }
    }
}

pub fn classify_routing_event(
    reference_ids: &[u32],
    gpu_ids: &[u32],
    num_experts: usize,
    required_values_finite: bool,
) -> RoutingClassification {
    let structurally_valid = !reference_ids.is_empty()
        && reference_ids.len() == gpu_ids.len()
        && unique_valid_ids(reference_ids, num_experts)
        && unique_valid_ids(gpu_ids, num_experts)
        && required_values_finite;
    if !structurally_valid {
        return RoutingClassification::InvalidNonfiniteIncomplete;
    }
    if reference_ids == gpu_ids {
        RoutingClassification::ExactOrderMatch
    } else if same_membership(reference_ids, gpu_ids) {
        RoutingClassification::InternalRankPermutation
    } else {
        RoutingClassification::MembershipMismatch
    }
}

fn unique_valid_ids(ids: &[u32], num_experts: usize) -> bool {
    ids.iter().all(|&id| (id as usize) < num_experts)
        && ids.iter().copied().collect::<HashSet<_>>().len() == ids.len()
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RankDisplacementEvidence {
    pub displaced_selected_experts: usize,
    pub single_transposition: bool,
    pub multi_rank_change: bool,
}

pub fn rank_displacement(reference_ids: &[u32], gpu_ids: &[u32]) -> RankDisplacementEvidence {
    if reference_ids.len() != gpu_ids.len() || !same_membership(reference_ids, gpu_ids) {
        return RankDisplacementEvidence::default();
    }
    let displaced_selected_experts = reference_ids
        .iter()
        .zip(gpu_ids)
        .filter(|(left, right)| left != right)
        .count();
    let single_transposition = if displaced_selected_experts == 2 {
        let changed = reference_ids
            .iter()
            .zip(gpu_ids)
            .enumerate()
            .filter(|(_, (left, right))| left != right)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        reference_ids[changed[0]] == gpu_ids[changed[1]]
            && reference_ids[changed[1]] == gpu_ids[changed[0]]
    } else {
        false
    };
    RankDisplacementEvidence {
        displaced_selected_experts,
        single_transposition,
        multi_rank_change: displaced_selected_experts > 0 && !single_transposition,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SamplingDecision {
    pub selected: bool,
    pub reason: Option<&'static str>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeterministicEventSampler {
    exact_layer_selected: Vec<bool>,
}

impl DeterministicEventSampler {
    pub fn new(num_layers: usize) -> Self {
        Self {
            exact_layer_selected: vec![false; num_layers],
        }
    }

    pub fn select(
        &mut self,
        layer: usize,
        classification: RoutingClassification,
    ) -> SamplingDecision {
        match classification {
            RoutingClassification::InternalRankPermutation => SamplingDecision {
                selected: true,
                reason: Some("all-internal-rank-permutation-events"),
            },
            RoutingClassification::MembershipMismatch => SamplingDecision {
                selected: true,
                reason: Some("all-membership-mismatch-events"),
            },
            RoutingClassification::ExactOrderMatch
                if self
                    .exact_layer_selected
                    .get(layer)
                    .is_some_and(|selected| !selected) =>
            {
                self.exact_layer_selected[layer] = true;
                SamplingDecision {
                    selected: true,
                    reason: Some(EXACT_ORDER_SAMPLING_RULE),
                }
            }
            RoutingClassification::ExactOrderMatch
            | RoutingClassification::InvalidNonfiniteIncomplete => SamplingDecision {
                selected: false,
                reason: None,
            },
        }
    }

    pub fn layers_with_exact_sample(&self) -> Vec<usize> {
        self.exact_layer_selected
            .iter()
            .enumerate()
            .filter_map(|(layer, &selected)| selected.then_some(layer))
            .collect()
    }

    pub fn layers_without_exact_sample(&self) -> Vec<usize> {
        self.exact_layer_selected
            .iter()
            .enumerate()
            .filter_map(|(layer, &selected)| (!selected).then_some(layer))
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TokenCaseEvidence {
    pub case: String,
    pub reference_generated_token_ids: Vec<u32>,
    pub gpu_generated_token_ids: Vec<u32>,
    pub exact_match_count: usize,
    pub mismatch_count: usize,
    pub first_mismatch_position: Option<usize>,
}

impl TokenCaseEvidence {
    pub fn new(case: impl Into<String>, reference: Vec<u32>, gpu: Vec<u32>) -> Self {
        let compared = reference.len().min(gpu.len());
        let exact_match_count = reference
            .iter()
            .zip(&gpu)
            .filter(|(left, right)| left == right)
            .count();
        let mismatch_count = compared.saturating_sub(exact_match_count)
            + reference.len().max(gpu.len()).saturating_sub(compared);
        let first_mismatch_position = (0..reference.len().max(gpu.len()))
            .find(|&index| reference.get(index) != gpu.get(index));
        Self {
            case: case.into(),
            reference_generated_token_ids: reference,
            gpu_generated_token_ids: gpu,
            exact_match_count,
            mismatch_count,
            first_mismatch_position,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GeneratedTokenMismatchEvidence {
    pub case: String,
    pub generated_position: usize,
    pub reference_token_id: Option<u32>,
    pub gpu_token_id: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BoundaryRouterView {
    pub source: &'static str,
    pub top_12: Vec<RankedExpertEvidence>,
    pub rank_7_expert_id: u32,
    pub rank_8_expert_id: u32,
    pub rank_9_expert_id: u32,
    pub rank_7_minus_rank_8_score_margin: FloatEvidence,
    pub rank_8_minus_rank_9_score_margin: FloatEvidence,
}

impl BoundaryRouterView {
    pub fn from_cpu(router: &RouterEvaluationEvidence) -> Self {
        Self {
            source: router.source,
            top_12: router.top_12_ranked_experts.clone(),
            rank_7_expert_id: router.cutoff_margins.rank_7_expert_id,
            rank_8_expert_id: router.cutoff_margins.rank_8_expert_id,
            rank_9_expert_id: router.cutoff_margins.rank_9_expert_id,
            rank_7_minus_rank_8_score_margin: router
                .cutoff_margins
                .rank_7_minus_rank_8_score
                .clone(),
            rank_8_minus_rank_9_score_margin: router
                .cutoff_margins
                .rank_8_minus_rank_9_score
                .clone(),
        }
    }

    pub fn from_gpu(router: &ActualGpuRouterEvidence) -> Self {
        Self {
            source: router.source,
            top_12: router.top_12_ranked_experts.clone(),
            rank_7_expert_id: router.cutoff_margins.rank_7_expert_id,
            rank_8_expert_id: router.cutoff_margins.rank_8_expert_id,
            rank_9_expert_id: router.cutoff_margins.rank_9_expert_id,
            rank_7_minus_rank_8_score_margin: router
                .cutoff_margins
                .rank_7_minus_rank_8_score
                .clone(),
            rank_8_minus_rank_9_score_margin: router
                .cutoff_margins
                .rank_8_minus_rank_9_score
                .clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MembershipBoundaryEvidence {
    pub selected_membership_equal: bool,
    pub reference_selected_ids: Vec<u32>,
    pub gpu_selected_ids: Vec<u32>,
    pub reference: BoundaryRouterView,
    pub gpu: BoundaryRouterView,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CanonicalSelectedWeightEvidence {
    pub expert_id: u32,
    pub reference_weight: Option<FloatEvidence>,
    pub gpu_weight: Option<FloatEvidence>,
    pub delta: Option<WeightDeltaEvidence>,
}

pub fn canonical_selected_weight_evidence(
    reference_ids: &[u32],
    reference_weights: &[f32],
    gpu_ids: &[u32],
    gpu_weights: &[f32],
    num_experts: usize,
) -> Result<Vec<CanonicalSelectedWeightEvidence>, String> {
    let left = canonical_weight_map(reference_ids, reference_weights, num_experts)?;
    let right = canonical_weight_map(gpu_ids, gpu_weights, num_experts)?;
    let all_ids = left
        .keys()
        .chain(right.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    Ok(all_ids
        .into_iter()
        .map(|expert_id| {
            let left_value = left.get(&expert_id).copied();
            let right_value = right.get(&expert_id).copied();
            let delta = left_value.zip(right_value).map(|(left, right)| {
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
            });
            CanonicalSelectedWeightEvidence {
                expert_id,
                reference_weight: left_value.map(FloatEvidence::new),
                gpu_weight: right_value.map(FloatEvidence::new),
                delta,
            }
        })
        .collect())
}

fn canonical_weight_map(
    ids: &[u32],
    weights: &[f32],
    num_experts: usize,
) -> Result<BTreeMap<u32, f32>, String> {
    if ids.is_empty() || ids.len() != weights.len() {
        return Err("selected expert IDs and weights are empty or incomplete".to_string());
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err("selected expert weights contain a nonfinite value".to_string());
    }
    let mut out = BTreeMap::new();
    for (&id, &weight) in ids.iter().zip(weights) {
        if id as usize >= num_experts {
            return Err(format!("selected expert ID {id} is outside layer geometry"));
        }
        if out.insert(id, weight).is_some() {
            return Err(format!("duplicate selected expert ID {id}"));
        }
    }
    Ok(out)
}

fn scalar_delta(expert_id: u32, left: f32, right: f32) -> WeightDeltaEvidence {
    let absolute = (f64::from(right) - f64::from(left)).abs();
    WeightDeltaEvidence {
        expert_id,
        left: FloatEvidence::new(left),
        right: FloatEvidence::new(right),
        absolute_error: (left.is_finite() && right.is_finite()).then_some(absolute),
        relative_error: (left.is_finite() && right.is_finite() && left != 0.0)
            .then(|| absolute / f64::from(left).abs())
            .filter(|value| value.is_finite()),
        ulp_distance: crate::gpu_native_router_rank_diagnostics::ulp_distance(left, right),
    }
}

fn common_selected_score_deltas(
    cpu: &RouterEvaluationEvidence,
    gpu: &ActualGpuRouterEvidence,
) -> Vec<WeightDeltaEvidence> {
    cpu.top_8_ids
        .iter()
        .copied()
        .filter(|id| gpu.top_8_ids.contains(id))
        .filter_map(|expert_id| {
            let left = cpu.scored_probabilities.get(expert_id as usize)?.value?;
            let right = gpu.scored_probabilities.get(expert_id as usize)?.value?;
            Some(scalar_delta(expert_id, left, right))
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoutingEventEvidence {
    pub location: EventLocation,
    pub classification: RoutingClassification,
    pub selected_membership_equal: Option<bool>,
    pub reference_selected_ids: Vec<u32>,
    pub gpu_selected_ids: Vec<u32>,
    pub displacement: RankDisplacementEvidence,
    pub expert_weight_pairing_defect: bool,
    pub canonical_selected_weights: Option<Vec<CanonicalSelectedWeightEvidence>>,
    pub boundary: Option<MembershipBoundaryEvidence>,
    pub numerically_selected: bool,
    pub numerical_selection_reason: Option<&'static str>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouterSameInputEvidence {
    pub cpu_selected_ids: Vec<u32>,
    pub gpu_selected_ids: Vec<u32>,
    pub membership_equal: bool,
    pub ordered_ids_equal: bool,
    pub common_expert_score_deltas: Vec<WeightDeltaEvidence>,
    pub common_expert_weight_deltas: Vec<WeightDeltaEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SameInputNumericalEventEvidence {
    pub location: EventLocation,
    pub classification: RoutingClassification,
    pub cpu_vs_gpu_routed_moe: VectorNumericalEvidence,
    pub reference_vs_gpu_routed_moe_includes_upstream_drift: VectorNumericalEvidence,
    pub router_same_input: RouterSameInputEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PermutationNumericalEventEvidence {
    pub location: EventLocation,
    pub witness: PermutationOnlyWitness,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MetricDistribution {
    pub count: usize,
    pub minimum: Option<f64>,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub maximum: Option<f64>,
    pub event_producing_maximum: Option<EventLocation>,
}

impl MetricDistribution {
    fn from_points(points: Vec<(f64, EventLocation)>) -> Self {
        let mut finite = points
            .into_iter()
            .filter(|(value, _)| value.is_finite())
            .collect::<Vec<_>>();
        finite.sort_by(|(left, left_location), (right, right_location)| {
            left.total_cmp(right)
                .then_with(|| left_location.cmp(right_location))
        });
        let percentile = |p: f64| {
            if finite.is_empty() {
                None
            } else {
                let rank = (p * finite.len() as f64).ceil().max(1.0) as usize;
                Some(finite[rank - 1].0)
            }
        };
        Self {
            count: finite.len(),
            minimum: finite.first().map(|point| point.0),
            p50: percentile(0.50),
            p95: percentile(0.95),
            p99: percentile(0.99),
            maximum: finite.last().map(|point| point.0),
            event_producing_maximum: finite.last().map(|point| point.1.clone()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NumericalAggregation {
    pub measured_event_count: usize,
    pub percentile_convention: &'static str,
    pub max_absolute_error: MetricDistribution,
    pub rms_error: MetricDistribution,
    pub mean_absolute_error: MetricDistribution,
}

pub fn aggregate_numerics(events: &[SameInputNumericalEventEvidence]) -> NumericalAggregation {
    aggregate_vector_numerics(
        events
            .iter()
            .map(|event| (&event.location, &event.cpu_vs_gpu_routed_moe)),
    )
}

fn aggregate_vector_numerics<'a>(
    events: impl Iterator<Item = (&'a EventLocation, &'a VectorNumericalEvidence)>,
) -> NumericalAggregation {
    let events = events.collect::<Vec<_>>();
    let points = |extract: fn(&VectorNumericalEvidence) -> Option<f64>| {
        events
            .iter()
            .filter_map(|(location, evidence)| {
                extract(evidence).map(|value| (value, (*location).clone()))
            })
            .collect::<Vec<_>>()
    };
    NumericalAggregation {
        measured_event_count: events.len(),
        percentile_convention: PERCENTILE_CONVENTION,
        max_absolute_error: MetricDistribution::from_points(points(|e| e.max_absolute_error)),
        rms_error: MetricDistribution::from_points(points(|e| e.rms_error)),
        mean_absolute_error: MetricDistribution::from_points(points(|e| e.mean_absolute_error)),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClassificationNumericalAggregation {
    pub classification: RoutingClassification,
    pub summary: NumericalAggregation,
}

pub fn aggregate_numerics_by_classification(
    events: &[SameInputNumericalEventEvidence],
) -> Vec<ClassificationNumericalAggregation> {
    [
        RoutingClassification::ExactOrderMatch,
        RoutingClassification::InternalRankPermutation,
        RoutingClassification::MembershipMismatch,
    ]
    .into_iter()
    .map(|classification| {
        let subset = events
            .iter()
            .filter(|event| event.classification == classification)
            .cloned()
            .collect::<Vec<_>>();
        ClassificationNumericalAggregation {
            classification,
            summary: aggregate_numerics(&subset),
        }
    })
    .collect()
}

pub fn aggregate_permutation_numerics(
    events: &[PermutationNumericalEventEvidence],
) -> NumericalAggregation {
    aggregate_vector_numerics(
        events
            .iter()
            .map(|event| (&event.location, &event.witness.comparison)),
    )
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BoundaryMinimumEvidence {
    pub margin: Option<FloatEvidence>,
    pub event: Option<EventLocation>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BoundarySummary {
    pub minimum_observed_reference_rank8_minus_rank9_margin: BoundaryMinimumEvidence,
    pub minimum_observed_gpu_rank8_minus_rank9_margin: BoundaryMinimumEvidence,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RoutingCounters {
    pub total_routing_events: usize,
    pub exact_order_match_events: usize,
    pub internal_rank_permutation_events: usize,
    pub membership_mismatch_events: usize,
    pub invalid_events: usize,
    pub single_transposition_events: usize,
    pub multi_rank_change_events: usize,
    pub maximum_displaced_selected_experts: usize,
    pub event_corresponding_to_maximum_displacement: Option<EventLocation>,
    pub reference_nonfinite_events: usize,
    pub gpu_nonfinite_events: usize,
    pub expert_weight_pairing_defect_events: usize,
}

impl RoutingCounters {
    pub fn record(
        &mut self,
        event: &RoutingEventEvidence,
        reference_nonfinite: bool,
        gpu_nonfinite: bool,
    ) {
        self.total_routing_events += 1;
        match event.classification {
            RoutingClassification::ExactOrderMatch => self.exact_order_match_events += 1,
            RoutingClassification::InternalRankPermutation => {
                self.internal_rank_permutation_events += 1
            }
            RoutingClassification::MembershipMismatch => self.membership_mismatch_events += 1,
            RoutingClassification::InvalidNonfiniteIncomplete => self.invalid_events += 1,
        }
        self.single_transposition_events += usize::from(event.displacement.single_transposition);
        self.multi_rank_change_events += usize::from(event.displacement.multi_rank_change);
        if event.displacement.displaced_selected_experts > self.maximum_displaced_selected_experts {
            self.maximum_displaced_selected_experts = event.displacement.displaced_selected_experts;
            self.event_corresponding_to_maximum_displacement = Some(event.location.clone());
        }
        self.reference_nonfinite_events += usize::from(reference_nonfinite);
        self.gpu_nonfinite_events += usize::from(gpu_nonfinite);
        self.expert_weight_pairing_defect_events += usize::from(event.expert_weight_pairing_defect);
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SamplingSummary {
    pub exact_order_events_total: usize,
    pub exact_order_events_numerically_sampled: usize,
    pub exact_order_sampling_rule: &'static str,
    pub layers_with_exact_order_sample: Vec<usize>,
    pub layers_without_exact_order_sample: Vec<usize>,
    pub internal_permutation_events_numerically_measured: usize,
    pub membership_mismatch_events_numerically_measured: usize,
    pub maximum_exact_order_samples_for_model: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TokenSummary {
    pub case_count: usize,
    pub requested_generated_tokens: usize,
    pub completed_generated_tokens: usize,
    pub exact_generated_token_matches: usize,
    pub generated_token_mismatches: usize,
    pub first_generated_token_mismatch: Option<GeneratedTokenMismatchEvidence>,
}

pub fn summarize_tokens(cases: &[TokenCaseEvidence], requested: usize) -> TokenSummary {
    let completed_generated_tokens = cases
        .iter()
        .map(|case| {
            case.reference_generated_token_ids
                .len()
                .min(case.gpu_generated_token_ids.len())
        })
        .sum();
    let exact_generated_token_matches = cases.iter().map(|case| case.exact_match_count).sum();
    let generated_token_mismatches = cases.iter().map(|case| case.mismatch_count).sum();
    let first_generated_token_mismatch = cases.iter().find_map(|case| {
        case.first_mismatch_position
            .map(|generated_position| GeneratedTokenMismatchEvidence {
                case: case.case.clone(),
                generated_position,
                reference_token_id: case
                    .reference_generated_token_ids
                    .get(generated_position)
                    .copied(),
                gpu_token_id: case
                    .gpu_generated_token_ids
                    .get(generated_position)
                    .copied(),
            })
    });
    TokenSummary {
        case_count: cases.len(),
        requested_generated_tokens: requested,
        completed_generated_tokens,
        exact_generated_token_matches,
        generated_token_mismatches,
        first_generated_token_mismatch,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProductionSemanticsEvidence {
    pub production_inference_math_changed: bool,
    pub production_shader_math_changed: bool,
    pub production_selected_expert_order_changed: bool,
    pub production_accumulation_order_changed: bool,
    pub existing_v1_qualifier_semantics_changed: bool,
    pub numerical_acceptance_thresholds_introduced: bool,
    pub cpu_combiner: &'static str,
    pub gpu_combiner: &'static str,
    pub canonicalization_scope: &'static str,
}

impl Default for ProductionSemanticsEvidence {
    fn default() -> Self {
        Self {
            production_inference_math_changed: false,
            production_shader_math_changed: false,
            production_selected_expert_order_changed: false,
            production_accumulation_order_changed: false,
            existing_v1_qualifier_semantics_changed: false,
            numerical_acceptance_thresholds_introduced: false,
            cpu_combiner: "production sequential f32 accumulation",
            gpu_combiner: "production sequential f32 accumulation",
            canonicalization_scope: "diagnostic host memory only",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GlobalSummary {
    pub routing: RoutingCounters,
    pub tokens: TokenSummary,
    pub pairing_defect_count: usize,
    pub same_input_numerics: NumericalAggregation,
    pub permutation_only_numerics: NumericalAggregation,
    pub boundary: BoundarySummary,
    pub nonfinite_bit_mismatch_count: usize,
    pub sampling: SamplingSummary,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SemanticParityCorpusReport {
    pub schema: &'static str,
    pub mode: &'static str,
    pub status: &'static str,
    pub diagnostic_only: bool,
    pub diagnostic_complete: bool,
    qualification_pass: bool,
    pub failure: Option<String>,
    pub provenance: DiagnosticProvenance,
    pub corpus: CorpusEvidence,
    pub global_summary: GlobalSummary,
    pub token_cases: Vec<TokenCaseEvidence>,
    pub routing_events: Vec<RoutingEventEvidence>,
    pub same_input_numerical_events: Vec<SameInputNumericalEventEvidence>,
    pub same_input_numerics_by_classification: Vec<ClassificationNumericalAggregation>,
    pub permutation_only_events: Vec<PermutationNumericalEventEvidence>,
    pub production_semantics: ProductionSemanticsEvidence,
    pub observation_seams_not_implemented: Vec<String>,
}

impl SemanticParityCorpusReport {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provenance: DiagnosticProvenance,
        diagnostic_complete: bool,
        failure: Option<String>,
        token_cases: Vec<TokenCaseEvidence>,
        routing_events: Vec<RoutingEventEvidence>,
        routing_counters: RoutingCounters,
        sampling: SamplingSummary,
        boundary: BoundarySummary,
        same_input_numerical_events: Vec<SameInputNumericalEventEvidence>,
        permutation_only_events: Vec<PermutationNumericalEventEvidence>,
        observation_seams_not_implemented: Vec<String>,
    ) -> Self {
        let requested =
            crate::greedy_parity::CORPUS_CASE_COUNT * crate::greedy_parity::OUTPUT_TOKEN_LIMIT;
        let tokens = summarize_tokens(&token_cases, requested);
        let same_input_numerics = aggregate_numerics(&same_input_numerical_events);
        let same_input_numerics_by_classification =
            aggregate_numerics_by_classification(&same_input_numerical_events);
        let permutation_only_numerics = aggregate_permutation_numerics(&permutation_only_events);
        let nonfinite_bit_mismatch_count = same_input_numerical_events
            .iter()
            .map(|event| event.cpu_vs_gpu_routed_moe.nonfinite_bit_mismatch_count)
            .sum();
        let global_summary = GlobalSummary {
            pairing_defect_count: routing_counters.expert_weight_pairing_defect_events,
            routing: routing_counters,
            tokens,
            same_input_numerics,
            permutation_only_numerics,
            boundary,
            nonfinite_bit_mismatch_count,
            sampling,
        };
        Self {
            schema: SCHEMA_VERSION,
            mode: MODE,
            status: if diagnostic_complete {
                "diagnostic-only"
            } else {
                "incomplete"
            },
            diagnostic_only: true,
            diagnostic_complete,
            qualification_pass: false,
            failure,
            provenance,
            corpus: CorpusEvidence::fixed(),
            global_summary,
            token_cases,
            routing_events,
            same_input_numerical_events,
            same_input_numerics_by_classification,
            permutation_only_events,
            production_semantics: ProductionSemanticsEvidence::default(),
            observation_seams_not_implemented,
        }
    }

    pub fn qualification_pass(&self) -> bool {
        self.qualification_pass
    }
}

struct GpuCaseCapture {
    case: &'static str,
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    traces: Vec<SemanticCorpusGpuTrace>,
}

struct GpuCapture {
    cases: Vec<GpuCaseCapture>,
    gate_identities: Vec<crate::gpu_native_router_rank_diagnostics::GateTensorIdentity>,
    model_load: crate::greedy_parity::ModelLoadEvidence,
    device: crate::backend::GpuDeviceIdentity,
    model_geometry: GpuNativeModelGeometry,
    background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

struct ReferenceCaseCapture {
    case: &'static str,
    generated_token_ids: Vec<u32>,
    traces: Vec<crate::gpu_native_diagnostics::ModelDiagnosticTrace>,
}

struct EvidenceCapture {
    token_cases: Vec<TokenCaseEvidence>,
    routing_events: Vec<RoutingEventEvidence>,
    routing_counters: RoutingCounters,
    sampling: SamplingSummary,
    boundary: BoundarySummary,
    same_input_events: Vec<SameInputNumericalEventEvidence>,
    permutation_events: Vec<PermutationNumericalEventEvidence>,
    reference_model_load: crate::greedy_parity::ModelLoadEvidence,
    reference_background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

struct PlannedEvent {
    case_index: usize,
    generated_position: usize,
    layer: usize,
    evidence: RoutingEventEvidence,
    reference_router: Option<RouterEvaluationEvidence>,
    actual_gpu_router: Option<ActualGpuRouterEvidence>,
    reference_nonfinite: bool,
    gpu_nonfinite: bool,
}

struct EventPlan {
    events: Vec<PlannedEvent>,
    boundary: BoundarySummary,
    layers_with_exact_sample: Vec<usize>,
    layers_without_exact_sample: Vec<usize>,
}

async fn execute_gpu(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<GpuCapture, Box<dyn std::error::Error>> {
    let runtime = crate::build_isolated_greedy_runtime(
        spec,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
        tokenizer.clone(),
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
            return Err(format!(
                "semantic corpus GPU identity {observed_config_sha256} drifted from {resolved_config_sha256}"
            )
            .into());
        }
        let device = runtime
            .engine
            .gpu_device_identity()
            .ok_or("semantic corpus GPU runtime has no authoritative adapter identity")?;
        if device.name != expected_adapter_name {
            return Err(format!(
                "semantic corpus GPU runtime selected adapter {:?}, expected {:?}",
                device.name, expected_adapter_name
            )
            .into());
        }
        if device.software_adapter || device.device_type.eq_ignore_ascii_case("cpu") {
            return Err(format!(
                "semantic corpus GPU runtime selected software adapter {:?}",
                device.name
            )
            .into());
        }
        let token_loop = runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or("GPU-native token loop was not initialized")?;
        let model_geometry = token_loop.model_geometry();
        if runtime.model.layers.len() != model_geometry.num_layers {
            return Err("semantic corpus GPU layer geometry is incomplete".into());
        }
        let gate_identities = runtime
            .model
            .layers
            .iter()
            .enumerate()
            .map(|(layer, plan)| {
                crate::gpu_native_router_rank_diagnostics::GateTensorIdentity::from_gate(
                    layer, &plan.gate,
                )
            })
            .collect::<Vec<_>>();
        let model_load = crate::greedy_parity_model_load(&runtime);
        if token_loop.snapshot()
            != crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot::default()
        {
            return Err("semantic corpus GPU token-loop counters did not start at zero".into());
        }
        let routed_before = runtime.engine.routed_expert_execution_snapshot();
        let attention_before = crate::transformer::nonfinite_softmax_fallbacks();
        let trace_layout = SemanticCorpusTraceLayout::try_new(model_geometry)?;
        let mut cases = Vec::with_capacity(crate::greedy_parity::CORPUS_CASE_COUNT);
        let mut expected_completed = 0usize;
        for fixed in crate::greedy_parity::FIXED_CORPUS {
            let prompt_token_ids = tokenizer.encode(fixed.prompt)?;
            if prompt_token_ids.is_empty() {
                return Err(format!(
                    "semantic corpus fixed case {:?} encoded to zero tokens",
                    fixed.name
                )
                .into());
            }
            let mut request =
                token_loop.create_semantic_parity_corpus_diagnostic_request_state()?;
            let staging = token_loop
                .create_semantic_parity_corpus_diagnostic_staging_buffer(&trace_layout)?;
            let (generated_token_ids, traces) = crate::with_progress_timeout(
                format!("GPU-native semantic corpus {} candidate", fixed.name),
                watchdog,
                async {
                    let prefix_count = prompt_token_ids.len().saturating_sub(1);
                    for (position, &token_id) in
                        prompt_token_ids[..prefix_count].iter().enumerate()
                    {
                        token_loop
                            .step_token(&runtime.engine, &mut request, token_id, position, false)
                            .await?;
                    }
                    let mut input_token = *prompt_token_ids
                        .last()
                        .ok_or("semantic corpus GPU prompt is empty")?;
                    let mut position = prefix_count;
                    let mut generated_token_ids =
                        Vec::with_capacity(crate::greedy_parity::OUTPUT_TOKEN_LIMIT);
                    let mut traces =
                        Vec::with_capacity(crate::greedy_parity::OUTPUT_TOKEN_LIMIT);
                    while generated_token_ids.len() < crate::greedy_parity::OUTPUT_TOKEN_LIMIT {
                        let (trace, sampled_token, _) = token_loop
                            .step_token_semantic_parity_corpus_diagnostic(
                                &runtime.engine,
                                &mut request,
                                input_token,
                                position,
                                &trace_layout,
                                &staging,
                            )
                            .await?;
                        if trace.layers.len() != model_geometry.num_layers {
                            return Err("semantic corpus GPU trace omitted one or more layers".into());
                        }
                        generated_token_ids.push(sampled_token);
                        traces.push(trace);
                        input_token = sampled_token;
                        position = position
                            .checked_add(1)
                            .ok_or("semantic corpus GPU position overflowed")?;
                    }
                    Ok::<_, Box<dyn std::error::Error>>((generated_token_ids, traces))
                },
            )
            .await?;
            let case_expected = prompt_token_ids
                .len()
                .saturating_sub(1)
                .checked_add(crate::greedy_parity::OUTPUT_TOKEN_LIMIT)
                .ok_or("semantic corpus expected completion count overflowed")?;
            if request.committed_position() != case_expected {
                return Err(format!(
                    "semantic corpus GPU case {} retired at {}, expected {case_expected}",
                    fixed.name,
                    request.committed_position()
                )
                .into());
            }
            expected_completed = expected_completed
                .checked_add(case_expected)
                .ok_or("semantic corpus total completion count overflowed")?;
            drop(request);
            drop(staging);
            cases.push(GpuCaseCapture {
                case: fixed.name,
                prompt_token_ids,
                generated_token_ids,
                traces,
            });
        }
        let counters_after = token_loop.snapshot();
        if counters_after.tokens_completed != expected_completed as u64
            || counters_after.fatal_failures != 0
            || counters_after.no_progress_failures != 0
        {
            return Err(format!(
                "semantic corpus GPU request/counter evidence is invalid: completed={} expected={} fatal={} no_progress={}",
                counters_after.tokens_completed,
                expected_completed,
                counters_after.fatal_failures,
                counters_after.no_progress_failures
            )
            .into());
        }
        let routed_delta = crate::qualification::routed_execution_delta(
            routed_before,
            runtime.engine.routed_expert_execution_snapshot(),
        )
        .map_err(|failure| failure.detail)?;
        if routed_delta.degraded_expert_substitutions != 0
            || routed_delta.gpu_cpu_fallbacks != 0
            || routed_delta.gpu_dispatch_failures != 0
            || crate::transformer::nonfinite_softmax_fallbacks().saturating_sub(attention_before)
                != 0
        {
            return Err("semantic corpus GPU run recorded fallback or dispatch failure".into());
        }
        Ok::<_, Box<dyn std::error::Error>>(GpuCapture {
            cases,
            gate_identities,
            model_load,
            device,
            model_geometry,
            background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(),
        })
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut capture), Ok(shutdown)) => {
            capture.background_shutdown = shutdown;
            Ok(capture)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; semantic corpus GPU shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

fn values_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn update_boundary_minimum(
    current: &mut Option<(f32, FloatEvidence, EventLocation)>,
    margin: &FloatEvidence,
    location: &EventLocation,
) {
    let Some(value) = margin.value else {
        return;
    };
    if current
        .as_ref()
        .is_none_or(|(observed, _, _)| value.total_cmp(observed).is_lt())
    {
        *current = Some((value, margin.clone(), location.clone()));
    }
}

fn plan_events(
    reference_cases: &[ReferenceCaseCapture],
    gpu: &GpuCapture,
    gates: &[crate::gating::LinearGate],
) -> Result<EventPlan, Box<dyn std::error::Error>> {
    if reference_cases.len() != crate::greedy_parity::CORPUS_CASE_COUNT
        || gpu.cases.len() != crate::greedy_parity::CORPUS_CASE_COUNT
        || gates.len() != gpu.model_geometry.num_layers
    {
        return Err("semantic corpus case or gate coverage is incomplete".into());
    }
    let mut sampler = DeterministicEventSampler::new(gpu.model_geometry.num_layers);
    let mut events = Vec::with_capacity(
        crate::greedy_parity::CORPUS_CASE_COUNT
            * crate::greedy_parity::OUTPUT_TOKEN_LIMIT
            * gpu.model_geometry.num_layers,
    );
    let mut minimum_reference = None;
    let mut minimum_gpu = None;

    for (case_index, ((reference_case, gpu_case), fixed)) in reference_cases
        .iter()
        .zip(&gpu.cases)
        .zip(crate::greedy_parity::FIXED_CORPUS)
        .enumerate()
    {
        if reference_case.case != fixed.name
            || gpu_case.case != fixed.name
            || reference_case.traces.len() != crate::greedy_parity::OUTPUT_TOKEN_LIMIT
            || gpu_case.traces.len() != crate::greedy_parity::OUTPUT_TOKEN_LIMIT
        {
            return Err(format!("semantic corpus case {} coverage drifted", fixed.name).into());
        }
        for generated_position in 0..crate::greedy_parity::OUTPUT_TOKEN_LIMIT {
            let reference_trace = &reference_case.traces[generated_position];
            let gpu_trace = &gpu_case.traces[generated_position];
            if reference_trace.layer_selected_ids.len() != gpu.model_geometry.num_layers
                || reference_trace.layer_selected_weights.len() != gpu.model_geometry.num_layers
                || reference_trace.layer_router_input.len() != gpu.model_geometry.num_layers
                || reference_trace.layer_routed_moe_output.len() != gpu.model_geometry.num_layers
                || gpu_trace.layers.len() != gpu.model_geometry.num_layers
            {
                return Err(format!(
                    "semantic corpus case {} position {generated_position} omitted layer evidence",
                    fixed.name
                )
                .into());
            }
            for layer in 0..gpu.model_geometry.num_layers {
                let location = EventLocation::new(fixed.name, generated_position, layer);
                let reference_ids = &reference_trace.layer_selected_ids[layer];
                let reference_weights = &reference_trace.layer_selected_weights[layer];
                let reference_input = &reference_trace.layer_router_input[layer];
                let reference_output = &reference_trace.layer_routed_moe_output[layer];
                let gpu_layer = &gpu_trace.layers[layer];
                let reference_nonfinite = !values_finite(reference_weights)
                    || !values_finite(reference_input)
                    || !values_finite(reference_output);
                let gpu_nonfinite = !values_finite(&gpu_layer.router_input)
                    || !values_finite(&gpu_layer.raw_logits)
                    || !values_finite(&gpu_layer.selected_weights)
                    || !values_finite(&gpu_layer.routed_moe_output)
                    || gpu_layer
                        .route_outputs
                        .iter()
                        .any(|output| !values_finite(output));
                let geometry_complete = reference_ids.len() == gpu.model_geometry.top_k
                    && reference_weights.len() == gpu.model_geometry.top_k
                    && reference_input.len() == gpu.model_geometry.d_model
                    && reference_output.len() == gpu.model_geometry.d_model
                    && gpu_layer.selected_ids.len() == gpu.model_geometry.top_k
                    && gpu_layer.selected_weights.len() == gpu.model_geometry.top_k
                    && gpu_layer.router_input.len() == gpu.model_geometry.d_model
                    && gpu_layer.raw_logits.len() == gpu.model_geometry.num_experts
                    && gpu_layer.route_outputs.len() == gpu.model_geometry.top_k
                    && gpu_layer
                        .route_outputs
                        .iter()
                        .all(|output| output.len() == gpu.model_geometry.d_model)
                    && gpu_layer.routed_moe_output.len() == gpu.model_geometry.d_model;
                let mut classification = classify_routing_event(
                    reference_ids,
                    &gpu_layer.selected_ids,
                    gpu.model_geometry.num_experts,
                    geometry_complete && !reference_nonfinite && !gpu_nonfinite,
                );
                let reference_router =
                    if classification == RoutingClassification::InvalidNonfiniteIncomplete {
                        None
                    } else {
                        crate::gpu_native_router_rank_diagnostics::evaluate_cpu_router(
                            "reference-hidden-to-cpu-production-router",
                            &gates[layer],
                            reference_input,
                        )
                        .ok()
                    };
                let actual_gpu_router =
                    if classification == RoutingClassification::InvalidNonfiniteIncomplete {
                        None
                    } else {
                        crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
                            gpu_layer.raw_logits.clone(),
                            gpu_layer.selected_ids.clone(),
                            gpu_layer.selected_weights.clone(),
                        )
                        .ok()
                    };
                if reference_router.is_none() || actual_gpu_router.is_none() {
                    classification = RoutingClassification::InvalidNonfiniteIncomplete;
                }
                let canonical_selected_weights = if matches!(
                    classification,
                    RoutingClassification::InternalRankPermutation
                        | RoutingClassification::MembershipMismatch
                ) {
                    canonical_selected_weight_evidence(
                        reference_ids,
                        reference_weights,
                        &gpu_layer.selected_ids,
                        &gpu_layer.selected_weights,
                        gpu.model_geometry.num_experts,
                    )
                    .ok()
                } else {
                    None
                };
                let gpu_pairing_structurally_valid = canonical_selected_weight_evidence(
                    &gpu_layer.selected_ids,
                    &gpu_layer.selected_weights,
                    &gpu_layer.selected_ids,
                    &gpu_layer.selected_weights,
                    gpu.model_geometry.num_experts,
                )
                .is_ok();
                let expert_weight_pairing_defect = !gpu_pairing_structurally_valid
                    || actual_gpu_router
                        .as_ref()
                        .is_some_and(|router| !router.selected_weights_paired_with_expert_ids);
                let displacement = rank_displacement(reference_ids, &gpu_layer.selected_ids);
                let boundary = if matches!(
                    classification,
                    RoutingClassification::InternalRankPermutation
                        | RoutingClassification::MembershipMismatch
                ) {
                    reference_router
                        .as_ref()
                        .zip(actual_gpu_router.as_ref())
                        .map(|(reference, actual)| MembershipBoundaryEvidence {
                            selected_membership_equal: same_membership(
                                reference_ids,
                                &gpu_layer.selected_ids,
                            ),
                            reference_selected_ids: reference_ids.clone(),
                            gpu_selected_ids: gpu_layer.selected_ids.clone(),
                            reference: BoundaryRouterView::from_cpu(reference),
                            gpu: BoundaryRouterView::from_gpu(actual),
                        })
                } else {
                    None
                };
                if let Some(router) = &reference_router {
                    update_boundary_minimum(
                        &mut minimum_reference,
                        &router.cutoff_margins.rank_8_minus_rank_9_score,
                        &location,
                    );
                }
                if let Some(router) = &actual_gpu_router {
                    update_boundary_minimum(
                        &mut minimum_gpu,
                        &router.cutoff_margins.rank_8_minus_rank_9_score,
                        &location,
                    );
                }
                let sampling = sampler.select(layer, classification);
                events.push(PlannedEvent {
                    case_index,
                    generated_position,
                    layer,
                    evidence: RoutingEventEvidence {
                        location,
                        classification,
                        selected_membership_equal: (classification
                            != RoutingClassification::InvalidNonfiniteIncomplete)
                            .then(|| same_membership(reference_ids, &gpu_layer.selected_ids)),
                        reference_selected_ids: reference_ids.clone(),
                        gpu_selected_ids: gpu_layer.selected_ids.clone(),
                        displacement,
                        expert_weight_pairing_defect,
                        canonical_selected_weights,
                        boundary,
                        numerically_selected: sampling.selected,
                        numerical_selection_reason: sampling.reason,
                    },
                    reference_router,
                    actual_gpu_router,
                    reference_nonfinite,
                    gpu_nonfinite,
                });
            }
        }
    }

    let boundary_minimum = |minimum: Option<(f32, FloatEvidence, EventLocation)>| match minimum {
        Some((_, margin, event)) => BoundaryMinimumEvidence {
            margin: Some(margin),
            event: Some(event),
        },
        None => BoundaryMinimumEvidence {
            margin: None,
            event: None,
        },
    };
    Ok(EventPlan {
        events,
        boundary: BoundarySummary {
            minimum_observed_reference_rank8_minus_rank9_margin: boundary_minimum(
                minimum_reference,
            ),
            minimum_observed_gpu_rank8_minus_rank9_margin: boundary_minimum(minimum_gpu),
        },
        layers_with_exact_sample: sampler.layers_with_exact_sample(),
        layers_without_exact_sample: sampler.layers_without_exact_sample(),
    })
}

async fn execute_reference_and_evidence(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    resolved_config_sha256: &str,
    gpu: &GpuCapture,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<EvidenceCapture, Box<dyn std::error::Error>> {
    let runtime = crate::build_isolated_greedy_runtime(
        spec,
        RealCliRuntimeMode::IsolatedGreedyParityCpu,
        tokenizer.clone(),
    )
    .await?;
    let attempt = async {
        runtime.engine.enable_cpu_q4_boundary_emulation()?;
        let observed_config_sha256 = crate::resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err(format!(
                "semantic corpus reference identity {observed_config_sha256} drifted from {resolved_config_sha256}"
            )
            .into());
        }
        if runtime.model.layers.len() != gpu.model_geometry.num_layers {
            return Err("semantic corpus reference layer geometry is incomplete".into());
        }
        let gates = runtime
            .model
            .layers
            .iter()
            .map(|layer| layer.gate.clone())
            .collect::<Vec<_>>();
        let reference_gate_identities = gates
            .iter()
            .enumerate()
            .map(|(layer, gate)| {
                crate::gpu_native_router_rank_diagnostics::GateTensorIdentity::from_gate(
                    layer, gate,
                )
            })
            .collect::<Vec<_>>();
        if reference_gate_identities != gpu.gate_identities {
            return Err("semantic corpus reference and GPU gate tensor identities differ".into());
        }
        let model_load = crate::greedy_parity_model_load(&runtime);
        let boundary_before = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_before.enabled || boundary_before.routed_expert_dispatches != 0 {
            return Err("semantic corpus reference boundary emulation did not start clean".into());
        }
        let routed_before = runtime.engine.routed_expert_execution_snapshot();
        let attention_before = crate::transformer::nonfinite_softmax_fallbacks();
        let mut reference_cases = Vec::with_capacity(crate::greedy_parity::CORPUS_CASE_COUNT);
        for (case_index, fixed) in crate::greedy_parity::FIXED_CORPUS.into_iter().enumerate() {
            let prompt_token_ids = tokenizer.encode(fixed.prompt)?;
            if prompt_token_ids != gpu.cases[case_index].prompt_token_ids {
                return Err(format!(
                    "semantic corpus tokenizer identity drifted for case {}",
                    fixed.name
                )
                .into());
            }
            let (generated_token_ids, traces) = crate::with_progress_timeout(
                format!("semantic corpus {} authoritative reference", fixed.name),
                watchdog,
                async {
                    let mut kv = runtime.model.fresh_kv_caches();
                    let prefix_count = prompt_token_ids.len().saturating_sub(1);
                    for (position, &token_id) in
                        prompt_token_ids[..prefix_count].iter().enumerate()
                    {
                        runtime
                            .model
                            .forward_token_hidden(
                                &runtime.engine,
                                token_id,
                                position,
                                &mut kv,
                            )
                            .await?;
                    }
                    let mut input_token = *prompt_token_ids
                        .last()
                        .ok_or("semantic corpus reference prompt is empty")?;
                    let mut position = prefix_count;
                    let mut generated_token_ids =
                        Vec::with_capacity(crate::greedy_parity::OUTPUT_TOKEN_LIMIT);
                    let mut traces =
                        Vec::with_capacity(crate::greedy_parity::OUTPUT_TOKEN_LIMIT);
                    while generated_token_ids.len() < crate::greedy_parity::OUTPUT_TOKEN_LIMIT {
                        let trace = runtime
                            .model
                            .forward_token_diagnostic_trace(
                                &runtime.engine,
                                input_token,
                                position,
                                &mut kv,
                                None,
                            )
                            .await?;
                        input_token = trace.sampled_token;
                        generated_token_ids.push(trace.sampled_token);
                        traces.push(trace);
                        position = position
                            .checked_add(1)
                            .ok_or("semantic corpus reference position overflowed")?;
                    }
                    Ok::<_, Box<dyn std::error::Error>>((generated_token_ids, traces))
                },
            )
            .await?;
            reference_cases.push(ReferenceCaseCapture {
                case: fixed.name,
                generated_token_ids,
                traces,
            });
        }

        let mut plan = plan_events(&reference_cases, gpu, &gates)?;
        let mut same_input_events = Vec::new();
        for planned in plan
            .events
            .iter_mut()
            .filter(|event| event.evidence.numerically_selected)
        {
            let gpu_layer = &gpu.cases[planned.case_index].traces[planned.generated_position]
                .layers[planned.layer];
            let reference_trace =
                &reference_cases[planned.case_index].traces[planned.generated_position];
            let cpu_router = crate::gpu_native_router_rank_diagnostics::evaluate_cpu_router(
                "gpu-hidden-to-cpu-production-router",
                &gates[planned.layer],
                &gpu_layer.router_input,
            )?;
            let routing = gates[planned.layer].route(&gpu_layer.router_input);
            let routing_matches_evaluation = routing.experts == cpu_router.top_8_ids
                && routing
                    .weights
                    .iter()
                    .map(|value| value.to_bits())
                    .eq(cpu_router.top_8_weights.iter().map(|value| value.bits));
            if !routing_matches_evaluation {
                planned.evidence.expert_weight_pairing_defect = true;
            }
            let global_ids = routing
                .experts
                .iter()
                .map(|&expert| runtime.model.global_expert_id(planned.layer, expert))
                .collect::<Vec<_>>();
            let token_index = (planned.case_index as u64)
                .wrapping_mul(crate::greedy_parity::OUTPUT_TOKEN_LIMIT as u64)
                .wrapping_add(planned.generated_position as u64)
                .wrapping_mul(gpu.model_geometry.num_layers as u64)
                .wrapping_add(planned.layer as u64);
            let expert_outputs = runtime
                .engine
                .moe_step_with_timing(
                    token_index,
                    planned.layer as u32,
                    &gpu_layer.router_input,
                    &global_ids,
                    None,
                )
                .await?;
            if expert_outputs.len() != routing.experts.len()
                || expert_outputs
                    .iter()
                    .any(|output| output.len() != gpu.model_geometry.d_model)
            {
                planned.evidence.expert_weight_pairing_defect = true;
                return Err(format!(
                    "same-input CPU expert output coverage is incomplete at {:?}",
                    planned.evidence.location
                )
                .into());
            }
            let cpu_routed_moe_output =
                crate::inference::combine_outputs(&expert_outputs, &routing.weights);
            let canonical = canonical_selected_weight_evidence(
                &routing.experts,
                &routing.weights,
                &gpu_layer.selected_ids,
                &gpu_layer.selected_weights,
                gpu.model_geometry.num_experts,
            )?;
            let common_expert_weight_deltas = canonical
                .into_iter()
                .filter_map(|item| item.delta)
                .collect::<Vec<_>>();
            let actual_gpu_router = planned
                .actual_gpu_router
                .as_ref()
                .ok_or("selected semantic corpus event is missing actual GPU router evidence")?;
            if actual_gpu_router.top_8_ids != gpu_layer.selected_ids
                || !actual_gpu_router.selected_weights_paired_with_expert_ids
            {
                planned.evidence.expert_weight_pairing_defect = true;
            }
            let stored_reference_router = planned
                .reference_router
                .as_ref()
                .ok_or("selected semantic corpus event is missing reference router evidence")?;
            if stored_reference_router.top_8_ids
                != reference_trace.layer_selected_ids[planned.layer]
            {
                planned.evidence.expert_weight_pairing_defect = true;
            }
            same_input_events.push(SameInputNumericalEventEvidence {
                location: planned.evidence.location.clone(),
                classification: planned.evidence.classification,
                cpu_vs_gpu_routed_moe: VectorNumericalEvidence::compare(
                    "cpu-production-routed-moe-on-exact-gpu-input",
                    "actual-production-gpu-routed-moe-on-exact-gpu-input",
                    &cpu_routed_moe_output,
                    &gpu_layer.routed_moe_output,
                )?,
                reference_vs_gpu_routed_moe_includes_upstream_drift:
                    VectorNumericalEvidence::compare(
                        "authoritative-cpu-reference-routed-moe-includes-upstream-drift",
                        "actual-production-gpu-routed-moe",
                        &reference_trace.layer_routed_moe_output[planned.layer],
                        &gpu_layer.routed_moe_output,
                    )?,
                router_same_input: RouterSameInputEvidence {
                    cpu_selected_ids: cpu_router.top_8_ids.clone(),
                    gpu_selected_ids: gpu_layer.selected_ids.clone(),
                    membership_equal: same_membership(
                        &cpu_router.top_8_ids,
                        &gpu_layer.selected_ids,
                    ),
                    ordered_ids_equal: cpu_router.top_8_ids == gpu_layer.selected_ids,
                    common_expert_score_deltas: common_selected_score_deltas(
                        &cpu_router,
                        actual_gpu_router,
                    ),
                    common_expert_weight_deltas,
                },
            });
        }

        let mut permutation_events = Vec::new();
        for planned in plan.events.iter().filter(|event| {
            event.evidence.classification == RoutingClassification::InternalRankPermutation
        }) {
            let gpu_layer = &gpu.cases[planned.case_index].traces[planned.generated_position]
                .layers[planned.layer];
            permutation_events.push(PermutationNumericalEventEvidence {
                location: planned.evidence.location.clone(),
                witness: crate::gpu_native_expert_permutation_semantic_parity::permutation_only_witness(
                    &gpu_layer.selected_ids,
                    &gpu_layer.selected_weights,
                    &gpu_layer.route_outputs,
                )?,
            });
        }

        let boundary_after = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_after.enabled || boundary_after.routed_expert_dispatches == 0 {
            return Err("semantic corpus reference did not exercise the Hybrid F16 boundary".into());
        }
        let routed_delta = crate::qualification::routed_execution_delta(
            routed_before,
            runtime.engine.routed_expert_execution_snapshot(),
        )
        .map_err(|failure| failure.detail)?;
        if routed_delta.degraded_expert_substitutions != 0
            || crate::transformer::nonfinite_softmax_fallbacks().saturating_sub(attention_before)
                != 0
        {
            return Err("semantic corpus reference recorded degraded or nonfinite fallback".into());
        }

        let token_cases = reference_cases
            .iter()
            .zip(&gpu.cases)
            .map(|(reference, gpu_case)| {
                TokenCaseEvidence::new(
                    reference.case,
                    reference.generated_token_ids.clone(),
                    gpu_case.generated_token_ids.clone(),
                )
            })
            .collect::<Vec<_>>();
        let mut routing_counters = RoutingCounters::default();
        for event in &plan.events {
            routing_counters.record(
                &event.evidence,
                event.reference_nonfinite,
                event.gpu_nonfinite,
            );
        }
        let exact_sampled = same_input_events
            .iter()
            .filter(|event| event.classification == RoutingClassification::ExactOrderMatch)
            .count();
        let internal_measured = same_input_events
            .iter()
            .filter(|event| {
                event.classification == RoutingClassification::InternalRankPermutation
            })
            .count();
        let membership_measured = same_input_events
            .iter()
            .filter(|event| event.classification == RoutingClassification::MembershipMismatch)
            .count();
        if internal_measured != routing_counters.internal_rank_permutation_events
            || membership_measured != routing_counters.membership_mismatch_events
            || permutation_events.len() != routing_counters.internal_rank_permutation_events
        {
            return Err("semantic corpus mandatory anomaly sampling coverage is incomplete".into());
        }
        let sampling = SamplingSummary {
            exact_order_events_total: routing_counters.exact_order_match_events,
            exact_order_events_numerically_sampled: exact_sampled,
            exact_order_sampling_rule: EXACT_ORDER_SAMPLING_RULE,
            layers_with_exact_order_sample: plan.layers_with_exact_sample,
            layers_without_exact_order_sample: plan.layers_without_exact_sample,
            internal_permutation_events_numerically_measured: internal_measured,
            membership_mismatch_events_numerically_measured: membership_measured,
            maximum_exact_order_samples_for_model: gpu.model_geometry.num_layers,
        };
        Ok::<_, Box<dyn std::error::Error>>(EvidenceCapture {
            token_cases,
            routing_events: plan
                .events
                .into_iter()
                .map(|event| event.evidence)
                .collect(),
            routing_counters,
            sampling,
            boundary: plan.boundary,
            same_input_events,
            permutation_events,
            reference_model_load: model_load,
            reference_background_shutdown:
                crate::greedy_parity::BackgroundShutdownEvidence::default(),
        })
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut capture), Ok(shutdown)) => {
            capture.reference_background_shutdown = shutdown;
            Ok(capture)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; semantic corpus reference shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

fn emit_report(
    report: &SemanticParityCorpusReport,
    report_out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if report.qualification_pass() {
        return Err("semantic corpus diagnostic cannot claim qualification PASS".into());
    }
    if let Some(parent) = report_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    std::fs::write(report_out, json)?;
    eprintln!(
        "GPU-native semantic-parity corpus diagnostic report written to {}",
        report_out.display()
    );
    Ok(())
}

pub async fn run_diagnostic(
    config: PathBuf,
    cfg: crate::config::Config,
    expected_adapter_name: String,
    report_out: PathBuf,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::BuildProvenance;

    let provenance = BuildProvenance::embedded();
    let (artifacts, artifact_errors) = crate::qualification_artifacts(&config, &cfg);
    if !artifact_errors.is_empty() {
        return Err(format!(
            "semantic corpus diagnostic artifact preflight failed: {}",
            artifact_errors.join("; ")
        )
        .into());
    }
    if progress_watchdog.timeout.is_none() {
        return Err("semantic corpus diagnostic requires a positive progress timeout".into());
    }
    if provenance.dirty != Some(false)
        || provenance
            .git_sha
            .as_deref()
            .is_none_or(|sha| sha.len() != 40 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err(
            "semantic corpus diagnostic requires clean embedded 40-hex Git provenance".into(),
        );
    }
    if CorpusEvidence::fixed().sha256
        != "ea7fdda4f08cde2fe3658165054d80099948f66ae8cba1c904ca41102f0aadc7"
    {
        return Err("semantic corpus identity drifted from the frozen contract".into());
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
            "semantic corpus diagnostic requires the strict GPU-native Q4 qualifier configuration"
                .into(),
        );
    }
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                format!("semantic corpus expert metadata preflight failed: {error}")
            })?;
    if expert_metadata.q4_0_layout.as_deref() != Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
        || expert_metadata.explicitly_synthetic
    {
        return Err(
            "semantic corpus diagnostic requires canonical nonsynthetic Q4_0 metadata".into(),
        );
    }
    if expected_adapter_name.trim().is_empty() {
        return Err("semantic corpus diagnostic requires a nonempty exact adapter name".into());
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
            "semantic corpus diagnostic requires exact Qwen3-Coder 30B-A3B Q4_0 geometry".into(),
        );
    }
    let gpu_resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&gpu_spec)?;
    let mut reference_spec = gpu_spec.clone();
    reference_spec.cfg.real_transformer.gpu_native = false;
    reference_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Cpu;
    let reference_resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&reference_spec)?;
    let tokenizer = crate::load_real_cli_tokenizer(
        &gpu_spec.cfg,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
    )?;

    let gpu = execute_gpu(
        &gpu_spec,
        tokenizer.clone(),
        &gpu_resolved_config_sha256,
        &expected_adapter_name,
        progress_watchdog,
    )
    .await?;
    if gpu.model_geometry.num_layers != 48
        || gpu.model_geometry.num_experts != 128
        || gpu.model_geometry.top_k != 8
        || gpu.model_geometry.d_model != 2048
        || gpu.model_geometry.d_ff != 768
    {
        return Err("semantic corpus GPU runtime geometry drifted after strict startup".into());
    }
    let evidence = execute_reference_and_evidence(
        &reference_spec,
        tokenizer,
        &reference_resolved_config_sha256,
        &gpu,
        progress_watchdog,
    )
    .await?;
    let expected_routing_events = crate::greedy_parity::CORPUS_CASE_COUNT
        * crate::greedy_parity::OUTPUT_TOKEN_LIMIT
        * gpu.model_geometry.num_layers;
    if evidence.routing_events.len() != expected_routing_events
        || evidence.routing_counters.total_routing_events != expected_routing_events
    {
        return Err(format!(
            "semantic corpus routing coverage is incomplete: observed {} expected {expected_routing_events}",
            evidence.routing_events.len()
        )
        .into());
    }
    let (_, executable_sha256) = crate::current_executable_identity()?;
    let report = SemanticParityCorpusReport::new(
        DiagnosticProvenance {
            build: provenance,
            executable_sha256,
            artifacts,
            gpu_resolved_config_sha256,
            reference_resolved_config_sha256,
            model_identity,
            reference_model_load: evidence.reference_model_load,
            gpu_native_model_load: gpu.model_load,
            reference_background_shutdown: evidence.reference_background_shutdown,
            gpu_native_background_shutdown: gpu.background_shutdown,
            expert_metadata,
            device: gpu.device,
        },
        true,
        None,
        evidence.token_cases,
        evidence.routing_events,
        evidence.routing_counters,
        evidence.sampling,
        evidence.boundary,
        evidence.same_input_events,
        evidence.permutation_events,
        Vec::new(),
    );
    emit_report(&report, &report_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> GpuNativeModelGeometry {
        GpuNativeModelGeometry {
            num_layers: 3,
            d_model: 4,
            d_ff: 8,
            num_experts: 12,
            top_k: 3,
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

    fn event(
        case: &str,
        position: usize,
        layer: usize,
        classification: RoutingClassification,
        displacement: RankDisplacementEvidence,
        pairing_defect: bool,
    ) -> RoutingEventEvidence {
        RoutingEventEvidence {
            location: EventLocation::new(case, position, layer),
            classification,
            selected_membership_equal: Some(
                classification != RoutingClassification::MembershipMismatch,
            ),
            reference_selected_ids: vec![1, 2, 3],
            gpu_selected_ids: vec![1, 2, 3],
            displacement,
            expert_weight_pairing_defect: pairing_defect,
            canonical_selected_weights: None,
            boundary: None,
            numerically_selected: false,
            numerical_selection_reason: None,
        }
    }

    #[test]
    fn full_layer_trace_layout_round_trips_every_observation() {
        let layout = SemanticCorpusTraceLayout::try_new(geometry()).unwrap();
        let mut bytes = vec![0u8; layout.total_bytes as usize];
        let write_f32 = |bytes: &mut [u8], offset: u64, values: &[f32]| {
            for (index, value) in values.iter().enumerate() {
                bytes[offset as usize + index * 4..offset as usize + index * 4 + 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
        };
        for layer in 0..layout.num_layers {
            write_f32(
                &mut bytes,
                layout.router_input_offset(layer),
                &[layer as f32, 2.0, 3.0, 4.0],
            );
            write_f32(
                &mut bytes,
                layout.raw_logits_offset(layer),
                &(0..12).map(|id| id as f32).collect::<Vec<_>>(),
            );
            for (rank, id) in [9u32, 2, 5].iter().enumerate() {
                let offset = layout.selected_ids_offset(layer) as usize + rank * 4;
                bytes[offset..offset + 4].copy_from_slice(&id.to_le_bytes());
            }
            write_f32(
                &mut bytes,
                layout.selected_weights_offset(layer),
                &[0.5, 0.3, 0.2],
            );
            write_f32(
                &mut bytes,
                layout.route_outputs_offset(layer),
                &(0..12).map(|index| index as f32).collect::<Vec<_>>(),
            );
            write_f32(
                &mut bytes,
                layout.routed_moe_output_offset(layer),
                &[5.0, 6.0, 7.0, 8.0],
            );
        }
        let trace = layout.parse(&bytes).unwrap();
        assert_eq!(trace.layers.len(), 3);
        assert_eq!(trace.layers[2].router_input[0], 2.0);
        assert_eq!(trace.layers[1].selected_ids, [9, 2, 5]);
        assert_eq!(trace.layers[0].route_outputs[2], [8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn routing_classification_is_exact_and_fail_closed() {
        assert_eq!(
            classify_routing_event(&[1, 2, 3], &[1, 2, 3], 12, true),
            RoutingClassification::ExactOrderMatch
        );
        assert_eq!(
            classify_routing_event(&[1, 2, 3], &[1, 3, 2], 12, true),
            RoutingClassification::InternalRankPermutation
        );
        assert_eq!(
            classify_routing_event(&[1, 2, 3], &[1, 3, 4], 12, true),
            RoutingClassification::MembershipMismatch
        );
        for (left, right, finite) in [
            (&[1, 2][..], &[1, 2, 3][..], true),
            (&[1, 2, 2][..], &[1, 2, 3][..], true),
            (&[1, 2, 12][..], &[1, 2, 3][..], true),
            (&[1, 2, 3][..], &[1, 2, 3][..], false),
        ] {
            assert_eq!(
                classify_routing_event(left, right, 12, finite),
                RoutingClassification::InvalidNonfiniteIncomplete
            );
        }
    }

    #[test]
    fn displacement_distinguishes_transposition_and_multi_rank_change() {
        let transposition = rank_displacement(&[1, 2, 3, 4], &[1, 3, 2, 4]);
        assert_eq!(transposition.displaced_selected_experts, 2);
        assert!(transposition.single_transposition);
        assert!(!transposition.multi_rank_change);
        let multi = rank_displacement(&[1, 2, 3, 4], &[2, 3, 1, 4]);
        assert_eq!(multi.displaced_selected_experts, 3);
        assert!(!multi.single_transposition);
        assert!(multi.multi_rank_change);
    }

    #[test]
    fn canonicalization_preserves_pairing_and_rejects_defects() {
        let evidence = canonical_selected_weight_evidence(
            &[9, 2, 5],
            &[0.1, 0.2, 0.7],
            &[2, 5, 9],
            &[0.2, 0.7, 0.1],
            12,
        )
        .unwrap();
        let by_id = evidence
            .into_iter()
            .map(|item| (item.expert_id, item))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            by_id[&9].gpu_weight.as_ref().unwrap().bits,
            0.1f32.to_bits()
        );
        assert_eq!(
            by_id[&2].gpu_weight.as_ref().unwrap().bits,
            0.2f32.to_bits()
        );
        assert!(canonical_selected_weight_evidence(
            &[1, 1, 2],
            &[0.5, 0.3, 0.2],
            &[1, 2, 3],
            &[0.5, 0.3, 0.2],
            12,
        )
        .unwrap_err()
        .contains("duplicate"));
        assert!(canonical_selected_weight_evidence(
            &[1, 2, 12],
            &[0.5, 0.3, 0.2],
            &[1, 2, 3],
            &[0.5, 0.3, 0.2],
            12,
        )
        .unwrap_err()
        .contains("outside"));
    }

    #[test]
    fn deterministic_sampling_selects_first_exact_per_layer_and_all_anomalies() {
        let classifications = [
            (0, RoutingClassification::ExactOrderMatch),
            (0, RoutingClassification::ExactOrderMatch),
            (1, RoutingClassification::InternalRankPermutation),
            (1, RoutingClassification::ExactOrderMatch),
            (2, RoutingClassification::MembershipMismatch),
            (2, RoutingClassification::InvalidNonfiniteIncomplete),
        ];
        let run = || {
            let mut sampler = DeterministicEventSampler::new(4);
            let decisions = classifications
                .into_iter()
                .map(|(layer, class)| sampler.select(layer, class).selected)
                .collect::<Vec<_>>();
            (decisions, sampler)
        };
        let (first, sampler) = run();
        let (second, _) = run();
        assert_eq!(first, [true, false, true, true, true, false]);
        assert_eq!(first, second);
        assert_eq!(sampler.layers_with_exact_sample(), [0, 1]);
        assert_eq!(sampler.layers_without_exact_sample(), [2, 3]);
    }

    #[test]
    fn routing_counters_retain_maximum_displacement_event() {
        let mut counters = RoutingCounters::default();
        let first = event(
            "a",
            0,
            1,
            RoutingClassification::InternalRankPermutation,
            rank_displacement(&[1, 2, 3], &[1, 3, 2]),
            false,
        );
        let second = event(
            "a",
            0,
            2,
            RoutingClassification::InternalRankPermutation,
            rank_displacement(&[1, 2, 3], &[2, 3, 1]),
            true,
        );
        counters.record(&first, false, false);
        counters.record(&second, true, false);
        assert_eq!(counters.internal_rank_permutation_events, 2);
        assert_eq!(counters.single_transposition_events, 1);
        assert_eq!(counters.multi_rank_change_events, 1);
        assert_eq!(counters.maximum_displaced_selected_experts, 3);
        assert_eq!(
            counters.event_corresponding_to_maximum_displacement,
            Some(second.location)
        );
        assert_eq!(counters.reference_nonfinite_events, 1);
        assert_eq!(counters.expert_weight_pairing_defect_events, 1);
    }

    #[test]
    fn token_parity_retains_first_mismatch_and_incomplete_lengths() {
        let first = TokenCaseEvidence::new("a", vec![1, 2, 3], vec![1, 9, 3]);
        let second = TokenCaseEvidence::new("b", vec![4, 5, 6], vec![4, 5]);
        assert_eq!(first.exact_match_count, 2);
        assert_eq!(first.mismatch_count, 1);
        assert_eq!(first.first_mismatch_position, Some(1));
        assert_eq!(second.first_mismatch_position, Some(2));
        let summary = summarize_tokens(&[first, second], 6);
        assert_eq!(summary.completed_generated_tokens, 5);
        assert_eq!(summary.exact_generated_token_matches, 4);
        assert_eq!(summary.generated_token_mismatches, 2);
        assert_eq!(
            summary
                .first_generated_token_mismatch
                .unwrap()
                .generated_position,
            1
        );
    }

    #[test]
    fn numerical_aggregation_uses_deterministic_nearest_rank_percentiles() {
        let events = (1..=100)
            .map(|index| SameInputNumericalEventEvidence {
                location: EventLocation::new("a", 0, index),
                classification: RoutingClassification::ExactOrderMatch,
                cpu_vs_gpu_routed_moe: VectorNumericalEvidence::compare(
                    "cpu",
                    "gpu",
                    &[0.0],
                    &[index as f32],
                )
                .unwrap(),
                reference_vs_gpu_routed_moe_includes_upstream_drift:
                    VectorNumericalEvidence::compare("reference", "gpu", &[0.0], &[index as f32])
                        .unwrap(),
                router_same_input: RouterSameInputEvidence {
                    cpu_selected_ids: vec![1],
                    gpu_selected_ids: vec![1],
                    membership_equal: true,
                    ordered_ids_equal: true,
                    common_expert_score_deltas: Vec::new(),
                    common_expert_weight_deltas: Vec::new(),
                },
            })
            .collect::<Vec<_>>();
        let summary = aggregate_numerics(&events);
        assert_eq!(summary.max_absolute_error.minimum, Some(1.0));
        assert_eq!(summary.max_absolute_error.p50, Some(50.0));
        assert_eq!(summary.max_absolute_error.p95, Some(95.0));
        assert_eq!(summary.max_absolute_error.p99, Some(99.0));
        assert_eq!(summary.max_absolute_error.maximum, Some(100.0));
        assert_eq!(
            summary
                .max_absolute_error
                .event_producing_maximum
                .unwrap()
                .layer,
            100
        );
    }

    #[test]
    fn permutation_accumulation_uses_frozen_outputs_and_can_differ_in_f32() {
        let outputs = vec![
            vec![-16_777_216.0, 1.0],
            vec![16_777_216.0, 2.0],
            vec![1.0, 3.0],
        ];
        let witness =
            crate::gpu_native_expert_permutation_semantic_parity::permutation_only_witness(
                &[9, 2, 5],
                &[1.0; 3],
                &outputs,
            )
            .unwrap();
        assert_eq!(
            witness.frozen_expert_outputs_in_production_rank_order.len(),
            3
        );
        assert_eq!(witness.comparison.exact_bit_equal_element_count, 1);
        assert!(!witness.comparison.exact_bit_equal);
        let exact = crate::gpu_native_expert_permutation_semantic_parity::permutation_only_witness(
            &[1, 2],
            &[0.5, 0.5],
            &[vec![1.0], vec![1.0]],
        )
        .unwrap();
        assert!(exact.comparison.exact_bit_equal);
    }

    #[test]
    fn schema_corpus_and_qualification_false_are_stable() {
        assert_eq!(SCHEMA_VERSION, "mer.gpu-native-semantic-parity-corpus.v1");
        assert_eq!(MODE, "diagnose-gpu-native-semantic-parity-corpus");
        let corpus = serde_json::to_value(CorpusEvidence::fixed()).unwrap();
        assert_eq!(corpus["id"], crate::greedy_parity::CORPUS_ID);
        assert_eq!(corpus["version"], 1);
        assert_eq!(corpus["case_count"], 4);
        assert_eq!(corpus["output_token_limit"], 16);
        assert_eq!(
            corpus["sha256"],
            "ea7fdda4f08cde2fe3658165054d80099948f66ae8cba1c904ca41102f0aadc7"
        );
        #[derive(Serialize)]
        struct Contract {
            qualification_pass: bool,
        }
        let value = serde_json::to_value(Contract {
            qualification_pass: false,
        })
        .unwrap();
        assert_eq!(value["qualification_pass"], false);
    }

    #[test]
    fn complete_report_serializes_provenance_summary_and_hardcoded_false() {
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
        let report = SemanticParityCorpusReport::new(
            DiagnosticProvenance {
                build: crate::qualification::BuildProvenance {
                    git_sha: Some("a".repeat(40)),
                    dirty: Some(false),
                    package_version: "test".to_string(),
                },
                executable_sha256: "b".repeat(64),
                artifacts: crate::qualification::QualificationArtifacts::default(),
                gpu_resolved_config_sha256: "c".repeat(64),
                reference_resolved_config_sha256: "d".repeat(64),
                model_identity: crate::greedy_parity::ModelIdentityEvidence {
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
                reference_background_shutdown: shutdown.clone(),
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
            true,
            None,
            Vec::new(),
            Vec::new(),
            RoutingCounters::default(),
            SamplingSummary {
                exact_order_events_total: 0,
                exact_order_events_numerically_sampled: 0,
                exact_order_sampling_rule: EXACT_ORDER_SAMPLING_RULE,
                layers_with_exact_order_sample: Vec::new(),
                layers_without_exact_order_sample: (0..48).collect(),
                internal_permutation_events_numerically_measured: 0,
                membership_mismatch_events_numerically_measured: 0,
                maximum_exact_order_samples_for_model: 48,
            },
            BoundarySummary {
                minimum_observed_reference_rank8_minus_rank9_margin: BoundaryMinimumEvidence {
                    margin: None,
                    event: None,
                },
                minimum_observed_gpu_rank8_minus_rank9_margin: BoundaryMinimumEvidence {
                    margin: None,
                    event: None,
                },
            },
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        assert!(!report.qualification_pass());
        let value = serde_json::to_value(report).unwrap();
        assert_eq!(value["schema"], SCHEMA_VERSION);
        assert_eq!(value["mode"], MODE);
        assert_eq!(value["qualification_pass"], false);
        assert_eq!(value["diagnostic_only"], true);
        assert_eq!(value["provenance"]["build"]["dirty"], false);
        assert!(value["global_summary"]["routing"].is_object());
        assert_eq!(
            value["global_summary"]["sampling"]["maximum_exact_order_samples_for_model"],
            48
        );
    }

    #[test]
    fn pairing_defect_is_detected_when_rank_weights_are_repaired_incorrectly() {
        let mut logits = vec![-10.0; 128];
        let ids = vec![30, 37, 114, 86, 29, 113, 68, 35];
        for (rank, id) in ids.iter().copied().enumerate() {
            logits[id as usize] = 10.0 - rank as f32;
        }
        let correct = crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
            logits.clone(),
            ids.clone(),
            vec![0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03],
        )
        .unwrap();
        assert!(correct.selected_weights_paired_with_expert_ids);
        let defect = crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
            logits,
            ids,
            vec![0.20, 0.30, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03],
        )
        .unwrap();
        assert!(!defect.selected_weights_paired_with_expert_ids);
    }

    #[test]
    fn vector_evidence_covers_relative_ulp_and_nonfinite_behavior() {
        let evidence = VectorNumericalEvidence::compare(
            "left",
            "right",
            &[0.0, f32::from_bits(1), f32::NAN],
            &[1.0, 1.0, f32::NAN],
        )
        .unwrap();
        assert_eq!(evidence.relative_error_defined_element_count, 1);
        assert_eq!(evidence.left_nonfinite_count, 1);
        assert_eq!(evidence.right_nonfinite_count, 1);
        assert_eq!(
            crate::gpu_native_router_rank_diagnostics::ulp_distance(
                1.0,
                f32::from_bits(1.0f32.to_bits() + 1),
            ),
            Some(1)
        );
    }
}
