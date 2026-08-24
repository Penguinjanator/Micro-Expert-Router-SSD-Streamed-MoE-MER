//! Diagnostic-only internal-stage attribution for frozen GPU-native Q4_0
//! routed-expert discrepancies.
//!
//! Nothing in this module is consulted by ordinary inference or by any
//! qualification PASS derivation. The report deliberately preserves raw
//! boundary, stage, replay, and arithmetic-emulator evidence without a
//! numerical acceptance threshold.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::gpu_native_expert_permutation_semantic_parity::VectorNumericalEvidence;
use crate::gpu_native_router_rank_diagnostics::{DiagnosticProvenance, GateTensorIdentity};
use crate::gpu_native_token_loop::{GpuNativeModelGeometry, GpuNativeTokenLoopSnapshot};
use crate::{IsolatedRuntimeShutdownError, RealCliRuntimeMode, ResolvedRealCliSpec};

pub const SCHEMA_VERSION: &str = "mer.gpu-native-q4-expert-stage-attribution.v1";
pub const MODE: &str = "diagnose-gpu-native-q4-expert-stages";
pub const QUALIFICATION_PASS: bool = false;
pub const FROZEN_BEA_BUILD_SHA: &str = "bea43722a8fc00fd76d3c45702f70c16e2b63041";
pub const FROZEN_BEA_REPORT_SHA256: &str =
    "804d14db1f521d5353046389d133e90c2cca1a258ea42a7fe33b689fd812025a";
pub const FROZEN_BEA_LOG_SHA256: &str =
    "8d744b343574e8ade1937e6943af1f67f19c1f03e345632f67883ff519e3c7d7";
pub const FROZEN_RUST_UPSTREAM_TRACE_SHA256: &str =
    "94d1d00e69a83f4510aece9c3c338f85d57a051619b73e897610d61a3e958920";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct FrozenStageTarget {
    pub id: &'static str,
    pub case: &'static str,
    pub generated_position: usize,
    pub layer: usize,
    pub frozen_worst_local_expert: Option<u32>,
}

pub const FROZEN_TARGETS: [FrozenStageTarget; 9] = [
    FrozenStageTarget {
        id: "Q1",
        case: "postgres-window-holdout",
        generated_position: 13,
        layer: 47,
        frozen_worst_local_expert: Some(28),
    },
    FrozenStageTarget {
        id: "Q2",
        case: "spanish-refactor-holdout",
        generated_position: 11,
        layer: 47,
        frozen_worst_local_expert: Some(58),
    },
    FrozenStageTarget {
        id: "Q3",
        case: "spanish-refactor-holdout",
        generated_position: 13,
        layer: 47,
        frozen_worst_local_expert: Some(28),
    },
    FrozenStageTarget {
        id: "Q4",
        case: "spanish-refactor-holdout",
        generated_position: 15,
        layer: 47,
        frozen_worst_local_expert: Some(116),
    },
    FrozenStageTarget {
        id: "R1",
        case: "rust-ownership-holdout",
        generated_position: 1,
        layer: 0,
        frozen_worst_local_expert: None,
    },
    FrozenStageTarget {
        id: "R2",
        case: "rust-ownership-holdout",
        generated_position: 1,
        layer: 19,
        frozen_worst_local_expert: None,
    },
    FrozenStageTarget {
        id: "R3",
        case: "rust-ownership-holdout",
        generated_position: 1,
        layer: 33,
        frozen_worst_local_expert: None,
    },
    FrozenStageTarget {
        id: "R4",
        case: "rust-ownership-holdout",
        generated_position: 1,
        layer: 40,
        frozen_worst_local_expert: None,
    },
    FrozenStageTarget {
        id: "R5",
        case: "rust-ownership-holdout",
        generated_position: 1,
        layer: 44,
        frozen_worst_local_expert: None,
    },
];

#[derive(Clone, Debug)]
pub struct Q4ExpertStageTargetLayout {
    pub layer: usize,
    pub router_input_offset: u64,
    pub router_input_bytes: u64,
    pub selected_ids_offset: u64,
    pub selected_ids_bytes: u64,
    pub selected_weights_offset: u64,
    pub selected_weights_bytes: u64,
    pub stages_offset: u64,
    pub stages_bytes: u64,
    pub production_down_offset: u64,
    pub production_down_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct Q4ExpertStageTraceLayout {
    pub d_model: usize,
    pub d_ff: usize,
    pub top_k: usize,
    pub targets: Vec<Q4ExpertStageTargetLayout>,
    pub total_bytes: u64,
}

fn checked_region(cursor: &mut u64, elements: usize, label: &str) -> Result<(u64, u64), String> {
    let bytes = u64::try_from(elements)
        .ok()
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| format!("{label} byte length overflow"))?;
    let offset = *cursor;
    *cursor = cursor
        .checked_add(bytes)
        .ok_or_else(|| format!("{label} offset overflow"))?;
    Ok((offset, bytes))
}

impl Q4ExpertStageTraceLayout {
    pub fn try_new(geometry: GpuNativeModelGeometry, layers: &[usize]) -> Result<Self, String> {
        if geometry.d_model == 0 || geometry.d_ff == 0 || geometry.top_k == 0 || layers.is_empty() {
            return Err("Q4 expert stage trace geometry is empty".to_string());
        }
        let unique = layers.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != layers.len()
            || layers.windows(2).any(|window| window[0] >= window[1])
            || layers.iter().any(|layer| *layer >= geometry.num_layers)
        {
            return Err("Q4 expert stage trace layers must be unique, sorted, and in range".into());
        }
        let stage_elements = geometry
            .top_k
            .checked_mul(
                geometry
                    .d_ff
                    .checked_mul(3)
                    .and_then(|value| value.checked_add(geometry.d_model))
                    .ok_or("Q4 expert stage element overflow")?,
            )
            .ok_or("Q4 expert stage element overflow")?;
        let production_down_elements = geometry
            .top_k
            .checked_mul(geometry.d_model)
            .ok_or("Q4 production down element overflow")?;
        let mut cursor = 0u64;
        let mut targets = Vec::with_capacity(layers.len());
        for &layer in layers {
            let (router_input_offset, router_input_bytes) =
                checked_region(&mut cursor, geometry.d_model, "router input")?;
            let (selected_ids_offset, selected_ids_bytes) =
                checked_region(&mut cursor, geometry.top_k, "selected IDs")?;
            let (selected_weights_offset, selected_weights_bytes) =
                checked_region(&mut cursor, geometry.top_k, "selected weights")?;
            let (stages_offset, stages_bytes) =
                checked_region(&mut cursor, stage_elements, "expert stages")?;
            let (production_down_offset, production_down_bytes) = checked_region(
                &mut cursor,
                production_down_elements,
                "production expert down outputs",
            )?;
            targets.push(Q4ExpertStageTargetLayout {
                layer,
                router_input_offset,
                router_input_bytes,
                selected_ids_offset,
                selected_ids_bytes,
                selected_weights_offset,
                selected_weights_bytes,
                stages_offset,
                stages_bytes,
                production_down_offset,
                production_down_bytes,
            });
        }
        Ok(Self {
            d_model: geometry.d_model,
            d_ff: geometry.d_ff,
            top_k: geometry.top_k,
            targets,
            total_bytes: cursor,
        })
    }

    pub fn target_for_layer(&self, layer: usize) -> Option<&Q4ExpertStageTargetLayout> {
        self.targets.iter().find(|target| target.layer == layer)
    }

    pub fn stage_elements(&self) -> usize {
        self.top_k * (3 * self.d_ff + self.d_model)
    }

    pub fn parse(&self, bytes: &[u8]) -> Result<Q4ExpertStageGpuTrace, String> {
        if bytes.len() != self.total_bytes as usize {
            return Err(format!(
                "Q4 expert stage readback has {} bytes, expected {}",
                bytes.len(),
                self.total_bytes
            ));
        }
        let mut targets = Vec::with_capacity(self.targets.len());
        for target in &self.targets {
            let router_input = read_f32_region(bytes, target.router_input_offset, self.d_model)?;
            let selected_ids = read_u32_region(bytes, target.selected_ids_offset, self.top_k)?;
            let selected_weights =
                read_f32_region(bytes, target.selected_weights_offset, self.top_k)?;
            let stages = read_f32_region(bytes, target.stages_offset, self.stage_elements())?;
            let production_down = read_f32_region(
                bytes,
                target.production_down_offset,
                self.top_k * self.d_model,
            )?;
            let gate_base = 0;
            let up_base = self.top_k * self.d_ff;
            let gated_base = 2 * self.top_k * self.d_ff;
            let down_base = 3 * self.top_k * self.d_ff;
            let routes = (0..self.top_k)
                .map(|rank| Q4ExpertStageGpuRouteTrace {
                    rank,
                    expert_id: selected_ids[rank],
                    gate: stages[gate_base + rank * self.d_ff..gate_base + (rank + 1) * self.d_ff]
                        .to_vec(),
                    up: stages[up_base + rank * self.d_ff..up_base + (rank + 1) * self.d_ff]
                        .to_vec(),
                    gated: stages
                        [gated_base + rank * self.d_ff..gated_base + (rank + 1) * self.d_ff]
                        .to_vec(),
                    diagnostic_down: stages
                        [down_base + rank * self.d_model..down_base + (rank + 1) * self.d_model]
                        .to_vec(),
                    production_down: production_down
                        [rank * self.d_model..(rank + 1) * self.d_model]
                        .to_vec(),
                })
                .collect();
            targets.push(Q4ExpertStageGpuTargetTrace {
                layer: target.layer,
                router_input,
                selected_ids,
                selected_weights,
                routes,
            });
        }
        Ok(Q4ExpertStageGpuTrace { targets })
    }
}

fn read_f32_region(bytes: &[u8], offset: u64, elements: usize) -> Result<Vec<f32>, String> {
    read_u32_region(bytes, offset, elements)
        .map(|bits| bits.into_iter().map(f32::from_bits).collect())
}

fn read_u32_region(bytes: &[u8], offset: u64, elements: usize) -> Result<Vec<u32>, String> {
    let start = usize::try_from(offset).map_err(|_| "readback offset does not fit usize")?;
    let end = start
        .checked_add(elements.checked_mul(4).ok_or("readback size overflow")?)
        .ok_or("readback range overflow")?;
    let region = bytes
        .get(start..end)
        .ok_or("readback region is out of bounds")?;
    Ok(region
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

#[derive(Clone, Debug, PartialEq)]
pub struct Q4ExpertStageGpuRouteTrace {
    pub rank: usize,
    pub expert_id: u32,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub gated: Vec<f32>,
    pub diagnostic_down: Vec<f32>,
    pub production_down: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Q4ExpertStageGpuTargetTrace {
    pub layer: usize,
    pub router_input: Vec<f32>,
    pub selected_ids: Vec<u32>,
    pub selected_weights: Vec<f32>,
    pub routes: Vec<Q4ExpertStageGpuRouteTrace>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Q4ExpertStageGpuTrace {
    pub targets: Vec<Q4ExpertStageGpuTargetTrace>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExactVectorEvidence {
    pub source: &'static str,
    pub vector_length: usize,
    pub f32_bits_sha256: String,
    pub f32_bits: Vec<u32>,
    pub nonfinite_count: usize,
}

impl ExactVectorEvidence {
    fn new(source: &'static str, values: &[f32], retain_bits: bool) -> Self {
        let bits = values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        Self {
            source,
            vector_length: values.len(),
            f32_bits_sha256: crate::numerical_diagnostics::f32_bits_sha256(&bits),
            f32_bits: retain_bits.then_some(bits).unwrap_or_default(),
            nonfinite_count: values.iter().filter(|value| !value.is_finite()).count(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OrderedRouterEvidence {
    pub cpu_production_ids: Vec<u32>,
    pub actual_gpu_ids: Vec<u32>,
    pub exact_ordered_equal: bool,
    pub decomposition_valid: bool,
    pub mismatch_not_repaired_or_reordered: bool,
}

fn pair_routes_by_expert_id(cpu_ids: &[u32], gpu_ids: &[u32]) -> Result<Vec<usize>, String> {
    if cpu_ids != gpu_ids {
        return Err("same-input ordered router IDs differ; decomposition is invalid".to_string());
    }
    let gpu_by_id = gpu_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, expert)| (expert, rank))
        .collect::<BTreeMap<_, _>>();
    if gpu_by_id.len() != gpu_ids.len() {
        return Err("actual GPU selected expert IDs contain duplicates".to_string());
    }
    cpu_ids
        .iter()
        .map(|expert| {
            gpu_by_id
                .get(expert)
                .copied()
                .ok_or_else(|| format!("actual GPU route is missing expert ID {expert}"))
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum StageAttribution {
    InputBoundaryDrift,
    GateProjectionDrift,
    UpProjectionDrift,
    GateAndUpProjectionDrift,
    SwigluDrift,
    DownProjectionDrift,
    MultiStageDrift,
    NoMaterialLocalization,
    Unresolved,
}

fn exact_bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn strictly_positive_aggregate_error(stage: &VectorNumericalEvidence) -> bool {
    stage.max_absolute_error.is_some_and(|value| value > 0.0)
        && stage.rms_error.is_some_and(|value| value > 0.0)
        && stage.mean_absolute_error.is_some_and(|value| value > 0.0)
}

fn derive_attribution(
    boundary_equal: bool,
    gate: &VectorNumericalEvidence,
    up: &VectorNumericalEvidence,
    production_down: &VectorNumericalEvidence,
    projection_implied_final: &VectorNumericalEvidence,
    swiglu_incremental_final: &VectorNumericalEvidence,
    down_incremental_final: &VectorNumericalEvidence,
) -> StageAttribution {
    if !boundary_equal {
        return StageAttribution::InputBoundaryDrift;
    }
    if [
        gate,
        up,
        production_down,
        projection_implied_final,
        swiglu_incremental_final,
        down_incremental_final,
    ]
    .into_iter()
    .any(|stage| stage.left_nonfinite_count != 0 || stage.right_nonfinite_count != 0)
    {
        return StageAttribution::Unresolved;
    }
    // No hidden tolerance: a contribution is observable here only when every
    // aggregate finite-error statistic is strictly non-zero. More
    // importantly, classification uses the comparative A->B, B->C, and
    // C->actual-GPU final-output residuals. A raw non-bit-equal internal stage
    // is therefore never sufficient by itself to name a causal stage.
    let gate_differs = strictly_positive_aggregate_error(gate);
    let up_differs = strictly_positive_aggregate_error(up);
    let projection_contributes = strictly_positive_aggregate_error(projection_implied_final);
    let swiglu_contributes = strictly_positive_aggregate_error(swiglu_incremental_final);
    let down_contributes = strictly_positive_aggregate_error(down_incremental_final);
    let contribution_count = usize::from(projection_contributes)
        + usize::from(swiglu_contributes)
        + usize::from(down_contributes);
    if contribution_count > 1 {
        return StageAttribution::MultiStageDrift;
    }
    if projection_contributes {
        return match (gate_differs, up_differs) {
            (true, false) => StageAttribution::GateProjectionDrift,
            (false, true) => StageAttribution::UpProjectionDrift,
            (true, true) => StageAttribution::GateAndUpProjectionDrift,
            (false, false) => StageAttribution::Unresolved,
        };
    }
    if swiglu_contributes {
        return StageAttribution::SwigluDrift;
    }
    if down_contributes {
        return StageAttribution::DownProjectionDrift;
    }
    StageAttribution::NoMaterialLocalization
}

fn current_gpu_emulator_assessment(stages: &[&VectorNumericalEvidence]) -> &'static str {
    if stages.iter().any(|stage| {
        stage.left_nonfinite_count != 0
            || stage.right_nonfinite_count != 0
            || stage.nonfinite_bit_mismatch_count != 0
    }) {
        "unresolved-nonfinite-review-raw-metrics"
    } else if stages.iter().all(|stage| stage.exact_bit_equal) {
        "bit-exact-at-all-observed-stages"
    } else {
        "not-bit-exact-no-hidden-closeness-threshold-review-raw-metrics"
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StageComparisonEvidence {
    pub gate: VectorNumericalEvidence,
    pub up: VectorNumericalEvidence,
    pub gated_swiglu: VectorNumericalEvidence,
    pub diagnostic_down: VectorNumericalEvidence,
    pub diagnostic_vs_production_gpu_down: VectorNumericalEvidence,
    pub production_down: VectorNumericalEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MixedReplayEvidence {
    pub cpu_baseline_vs_actual_gpu: VectorNumericalEvidence,
    pub gpu_gate_up_cpu_swiglu_down_vs_cpu_baseline: VectorNumericalEvidence,
    pub gpu_gate_up_cpu_swiglu_down_vs_actual_gpu: VectorNumericalEvidence,
    pub gpu_gated_cpu_down_vs_cpu_baseline: VectorNumericalEvidence,
    pub gpu_gated_cpu_down_vs_actual_gpu: VectorNumericalEvidence,
    pub gpu_gate_up_cpu_swiglu_down_vs_gpu_gated_cpu_down: VectorNumericalEvidence,
    pub cpu_gated_current_gpu_down_vs_cpu_baseline: VectorNumericalEvidence,
    pub cpu_gated_current_gpu_down_vs_actual_gpu: VectorNumericalEvidence,
    pub gpu_gated_current_gpu_down_vs_cpu_baseline: VectorNumericalEvidence,
    pub gpu_gated_current_gpu_down_vs_actual_gpu: VectorNumericalEvidence,
    pub diagnostic_only_never_fed_to_production: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ComparativeErrorGrowthEvidence {
    pub contract: &'static str,
    pub projection_implied_final_error_strictly_positive: bool,
    pub swiglu_incremental_final_error_strictly_positive: bool,
    pub down_incremental_final_error_strictly_positive: bool,
    pub arbitrary_numerical_threshold_used: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ArithmeticEmulatorEvidence {
    pub current_gpu_q4_dot_status: &'static str,
    pub current_gpu_q4_dot_hardware_model_assessment: &'static str,
    pub rejected_logical_dequant_q4_dot_status: &'static str,
    pub current_gpu_gate_vs_actual_gpu: VectorNumericalEvidence,
    pub current_gpu_up_vs_actual_gpu: VectorNumericalEvidence,
    pub current_gpu_gated_vs_actual_gpu: VectorNumericalEvidence,
    pub current_gpu_down_vs_actual_gpu: VectorNumericalEvidence,
    pub rejected_logical_gate_vs_cpu_production: VectorNumericalEvidence,
    pub rejected_logical_up_vs_cpu_production: VectorNumericalEvidence,
    pub rejected_logical_gated_vs_cpu_production: VectorNumericalEvidence,
    pub rejected_logical_down_vs_cpu_production: VectorNumericalEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExpertStageAttributionEvidence {
    pub rank: usize,
    pub local_expert_id: u32,
    pub global_expert_id: u32,
    pub canonical_q4_payload_sha256: String,
    pub gpu_route_paired_by_exact_expert_id: bool,
    pub cpu_production_final_bit_identical_to_ordinary_path: bool,
    pub final_output_boundary: OutputBoundaryEvidence,
    pub stage_comparison: StageComparisonEvidence,
    pub mixed_replay: MixedReplayEvidence,
    pub comparative_error_growth: ComparativeErrorGrowthEvidence,
    pub arithmetic_emulators: ArithmeticEmulatorEvidence,
    pub attribution: StageAttribution,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct InputBoundaryEvidence {
    pub exact_pre_boundary_gpu_router_input: ExactVectorEvidence,
    pub effective_gpu_expert_input: ExactVectorEvidence,
    pub cpu_boundary_emulated_input: ExactVectorEvidence,
    pub effective_input_bit_identical: bool,
    pub mismatch_cannot_be_coerced_into_equality: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OutputBoundaryEvidence {
    pub cpu_pre_output: ExactVectorEvidence,
    pub cpu_effective_output: ExactVectorEvidence,
    pub actual_gpu_effective_output: ExactVectorEvidence,
    pub final_boundary_bit_identical: bool,
    pub mismatch_cannot_be_coerced_into_equality: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TargetStageAttributionEvidence {
    pub target: FrozenStageTarget,
    pub actual_gpu_selected_weights: ExactVectorEvidence,
    pub router: OrderedRouterEvidence,
    pub input_boundary: Option<InputBoundaryEvidence>,
    pub experts: Vec<ExpertStageAttributionEvidence>,
    pub failure: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FrozenBuildEnvelope {
    git_sha: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FrozenProvenanceEnvelope {
    build: FrozenBuildEnvelope,
}

#[derive(Debug, Deserialize)]
struct FrozenNumericalLimitsEnvelope {
    max_absolute_error_limit: f64,
    rms_error_limit: f64,
    mean_absolute_error_limit: f64,
    nonfinite_mismatch_limit: usize,
    semantic_correctness_not_bit_parity: bool,
}

#[derive(Debug, Deserialize)]
struct FrozenArtifactEnvelope {
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct FrozenV2IdentityEnvelope {
    report_artifact: FrozenArtifactEnvelope,
    expected_report_sha256_argument: String,
    expected_schema: String,
    frozen_build_sha: String,
    qualification_pass: bool,
    holdout_corpus_id: String,
    holdout_corpus_sha256: String,
    numerical_limits: FrozenNumericalLimitsEnvelope,
    immutable_input_verified: bool,
}

#[derive(Debug, Deserialize)]
struct FrozenBeaEnvelope {
    schema: String,
    diagnostic_complete: bool,
    qualification_pass: bool,
    provenance: FrozenProvenanceEnvelope,
    frozen_v2_result_identity: FrozenV2IdentityEnvelope,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrozenBeaResultIdentity {
    pub report_artifact: crate::qualification::ArtifactDigest,
    pub expected_report_sha256_argument: String,
    pub expected_schema: &'static str,
    pub frozen_build_sha: &'static str,
    pub diagnostic_complete: bool,
    pub qualification_pass: bool,
    pub frozen_v2_report_sha256: &'static str,
    pub frozen_v2_build_sha: &'static str,
    pub frozen_v2_holdout_corpus_id: &'static str,
    pub frozen_v2_holdout_corpus_sha256: &'static str,
    pub immutable_input_verified: bool,
    pub bea_execution_log_sha256: &'static str,
    pub rust_upstream_trace_sha256: &'static str,
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn exact_f64(left: f64, right: f64) -> bool {
    left.to_bits() == right.to_bits()
}

fn validate_expected_bea_report_sha_argument(value: &str) -> Result<(), String> {
    if !is_hex(value, 64) || !value.eq_ignore_ascii_case(FROZEN_BEA_REPORT_SHA256) {
        return Err(format!(
            "expected bea report SHA must equal frozen {FROZEN_BEA_REPORT_SHA256}"
        ));
    }
    Ok(())
}

fn validate_bea_envelope(envelope: &FrozenBeaEnvelope) -> Result<(), String> {
    if envelope.schema != crate::gpu_native_v2_holdout_failure_attribution::SCHEMA_VERSION {
        return Err("frozen bea report schema differs".into());
    }
    if !envelope.diagnostic_complete || envelope.qualification_pass {
        return Err("frozen bea report completion/qualification flags differ".into());
    }
    if envelope.provenance.build.git_sha.as_deref() != Some(FROZEN_BEA_BUILD_SHA) {
        return Err("frozen bea report build SHA differs".into());
    }
    let v2 = &envelope.frozen_v2_result_identity;
    let expected_limits = crate::gpu_native_semantic_parity_v2::NumericalLimits::frozen();
    if v2.report_artifact.sha256
        != crate::gpu_native_v2_holdout_failure_attribution::FROZEN_V2_REPORT_SHA256
        || v2.expected_report_sha256_argument
            != crate::gpu_native_v2_holdout_failure_attribution::FROZEN_V2_REPORT_SHA256
        || v2.expected_schema != crate::gpu_native_semantic_parity_v2::SCHEMA_VERSION
        || v2.frozen_build_sha
            != crate::gpu_native_v2_holdout_failure_attribution::FROZEN_V2_BUILD_SHA
        || v2.qualification_pass
        || v2.holdout_corpus_id != crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_ID
        || v2.holdout_corpus_sha256 != crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_SHA256
        || !v2.immutable_input_verified
        || !exact_f64(
            v2.numerical_limits.max_absolute_error_limit,
            expected_limits.max_absolute_error_limit,
        )
        || !exact_f64(
            v2.numerical_limits.rms_error_limit,
            expected_limits.rms_error_limit,
        )
        || !exact_f64(
            v2.numerical_limits.mean_absolute_error_limit,
            expected_limits.mean_absolute_error_limit,
        )
        || v2.numerical_limits.nonfinite_mismatch_limit != expected_limits.nonfinite_mismatch_limit
        || v2.numerical_limits.semantic_correctness_not_bit_parity
            != expected_limits.semantic_correctness_not_bit_parity
    {
        return Err("frozen V2 identity inside bea report differs".into());
    }
    Ok(())
}

fn validate_frozen_bea_report(
    path: &Path,
    expected_sha256: &str,
) -> Result<FrozenBeaResultIdentity, Box<dyn std::error::Error>> {
    validate_expected_bea_report_sha_argument(expected_sha256)?;
    let bytes = std::fs::read(path)?;
    let artifact = crate::qualification::ArtifactDigest {
        configured_path: path.display().to_string(),
        canonical_path: std::fs::canonicalize(path)?.display().to_string(),
        byte_length: u64::try_from(bytes.len()).map_err(|_| "bea report length overflow")?,
        sha256: crate::greedy_parity::sha256_hex(&bytes),
    };
    if !artifact.sha256.eq_ignore_ascii_case(expected_sha256)
        || !artifact
            .sha256
            .eq_ignore_ascii_case(FROZEN_BEA_REPORT_SHA256)
    {
        return Err(format!(
            "frozen bea report SHA differs: observed {} expected {}",
            artifact.sha256, FROZEN_BEA_REPORT_SHA256
        )
        .into());
    }
    let envelope: FrozenBeaEnvelope = serde_json::from_slice(&bytes)
        .map_err(|error| format!("malformed frozen bea report: {error}"))?;
    validate_bea_envelope(&envelope)?;
    Ok(FrozenBeaResultIdentity {
        report_artifact: artifact,
        expected_report_sha256_argument: expected_sha256.to_ascii_lowercase(),
        expected_schema: crate::gpu_native_v2_holdout_failure_attribution::SCHEMA_VERSION,
        frozen_build_sha: FROZEN_BEA_BUILD_SHA,
        diagnostic_complete: true,
        qualification_pass: false,
        frozen_v2_report_sha256:
            crate::gpu_native_v2_holdout_failure_attribution::FROZEN_V2_REPORT_SHA256,
        frozen_v2_build_sha: crate::gpu_native_v2_holdout_failure_attribution::FROZEN_V2_BUILD_SHA,
        frozen_v2_holdout_corpus_id: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_ID,
        frozen_v2_holdout_corpus_sha256:
            crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_SHA256,
        immutable_input_verified: true,
        bea_execution_log_sha256: FROZEN_BEA_LOG_SHA256,
        rust_upstream_trace_sha256: FROZEN_RUST_UPSTREAM_TRACE_SHA256,
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProductionSemanticsEvidence {
    pub diagnostic_only: bool,
    pub production_q4_dot_arithmetic_changed: bool,
    pub production_q4_gate_up_arithmetic_changed: bool,
    pub production_q4_down_arithmetic_changed: bool,
    pub production_silu_changed: bool,
    pub routed_expert_boundary_changed: bool,
    pub router_changed: bool,
    pub attention_changed: bool,
    pub dense_gemv_changed: bool,
    pub v1_changed: bool,
    pub v2_changed: bool,
    pub limits_changed: bool,
    pub corpus_or_prompts_changed: bool,
    pub diagnostic_shader_contract: &'static str,
    pub qualification_threshold_introduced: bool,
}

impl Default for ProductionSemanticsEvidence {
    fn default() -> Self {
        Self {
            diagnostic_only: true,
            production_q4_dot_arithmetic_changed: false,
            production_q4_gate_up_arithmetic_changed: false,
            production_q4_down_arithmetic_changed: false,
            production_silu_changed: false,
            routed_expert_boundary_changed: false,
            router_changed: false,
            attention_changed: false,
            dense_gemv_changed: false,
            v1_changed: false,
            v2_changed: false,
            limits_changed: false,
            corpus_or_prompts_changed: false,
            diagnostic_shader_contract:
                "dedicated diagnostic-only pipeline reads production arena/input/routes and writes caller-owned scratch; ordinary entrypoints and bind contract unchanged",
            qualification_threshold_introduced: false,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct DiagnosticRuntimeEvidence {
    pub gpu_token_loop: GpuNativeTokenLoopSnapshot,
    pub gpu_expected_completed_token_steps: u64,
    pub reference_background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
    pub gpu_native_background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Q4ExpertStageAttributionReport {
    pub schema: &'static str,
    pub mode: &'static str,
    pub diagnostic_only: bool,
    pub diagnostic_complete: bool,
    qualification_pass: bool,
    pub failure: Option<String>,
    pub provenance: DiagnosticProvenance,
    pub frozen_bea_result_identity: FrozenBeaResultIdentity,
    pub frozen_targets: Vec<FrozenStageTarget>,
    pub targets: Vec<TargetStageAttributionEvidence>,
    pub runtime: DiagnosticRuntimeEvidence,
    pub production_semantics: ProductionSemanticsEvidence,
    pub observation_seams_not_implemented: Vec<String>,
    pub production_numerical_correction_justified: bool,
    pub frozen_historical_evidence_modified: bool,
}

impl Q4ExpertStageAttributionReport {
    fn new(
        provenance: DiagnosticProvenance,
        frozen_bea_result_identity: FrozenBeaResultIdentity,
        failure: Option<String>,
        targets: Vec<TargetStageAttributionEvidence>,
        runtime: DiagnosticRuntimeEvidence,
        observation_seams_not_implemented: Vec<String>,
    ) -> Self {
        let diagnostic_complete = failure.is_none()
            && targets.len() == FROZEN_TARGETS.len()
            && targets.iter().all(|target| target.failure.is_none())
            && observation_seams_not_implemented.is_empty();
        Self {
            schema: SCHEMA_VERSION,
            mode: MODE,
            diagnostic_only: true,
            diagnostic_complete,
            qualification_pass: QUALIFICATION_PASS,
            failure,
            provenance,
            frozen_bea_result_identity,
            frozen_targets: FROZEN_TARGETS.to_vec(),
            targets,
            runtime,
            production_semantics: ProductionSemanticsEvidence::default(),
            observation_seams_not_implemented,
            production_numerical_correction_justified: false,
            frozen_historical_evidence_modified: false,
        }
    }

    pub const fn qualification_pass(&self) -> bool {
        self.qualification_pass
    }
}

struct GpuTargetCapture {
    case: &'static str,
    generated_position: usize,
    trace: Q4ExpertStageGpuTrace,
}

struct GpuCapture {
    targets: Vec<GpuTargetCapture>,
    model_geometry: GpuNativeModelGeometry,
    gate_identities: BTreeMap<usize, GateTensorIdentity>,
    model_load: crate::greedy_parity::ModelLoadEvidence,
    device: crate::backend::GpuDeviceIdentity,
    background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
    token_loop_snapshot: GpuNativeTokenLoopSnapshot,
    expected_completed_token_steps: u64,
}

impl GpuCapture {
    fn target(&self, target: FrozenStageTarget) -> Result<&Q4ExpertStageGpuTargetTrace, String> {
        self.targets
            .iter()
            .find(|capture| {
                capture.case == target.case
                    && capture.generated_position == target.generated_position
            })
            .and_then(|capture| {
                capture
                    .trace
                    .targets
                    .iter()
                    .find(|trace| trace.layer == target.layer)
            })
            .ok_or_else(|| {
                format!(
                    "missing GPU stage target {} {} position {} layer {}",
                    target.id, target.case, target.generated_position, target.layer
                )
            })
    }
}

fn holdout_case(name: &str) -> Result<crate::gpu_native_semantic_parity_v2::HoldoutCase, String> {
    crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS
        .into_iter()
        .find(|case| case.name == name)
        .ok_or_else(|| format!("unknown frozen v2 holdout case {name:?}"))
}

async fn capture_gpu_case(
    runtime: &crate::BenchRealRuntime,
    tokenizer: &crate::tokenizer::Tokenizer,
    case_name: &'static str,
    target_positions: &[(usize, Vec<usize>)],
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<(Vec<GpuTargetCapture>, usize), Box<dyn std::error::Error>> {
    let fixed = holdout_case(case_name)?;
    let prompt_token_ids = tokenizer.encode(fixed.prompt)?;
    if prompt_token_ids.is_empty() || target_positions.is_empty() {
        return Err("stage target prompt or target list is empty".into());
    }
    let maximum_target = target_positions
        .iter()
        .map(|(position, _)| *position)
        .max()
        .ok_or("stage target list is empty")?;
    let token_loop = runtime
        .gpu_native_token_loop
        .as_ref()
        .ok_or("stage diagnostic GPU-native token loop was not initialized")?;
    let mut request = token_loop.create_q4_expert_stage_diagnostic_request_state()?;
    let captures = crate::with_progress_timeout(
        format!("Q4 expert stage diagnostic GPU {case_name}"),
        watchdog,
        async {
            let prefix_count = prompt_token_ids.len().saturating_sub(1);
            for (position, &token_id) in prompt_token_ids[..prefix_count].iter().enumerate() {
                token_loop
                    .step_token(&runtime.engine, &mut request, token_id, position, false)
                    .await?;
            }
            let mut input_token = *prompt_token_ids.last().ok_or("holdout prompt is empty")?;
            let mut model_position = prefix_count;
            let mut captures = Vec::with_capacity(target_positions.len());
            for generated_position in 0..=maximum_target {
                if let Some((_, layers)) = target_positions
                    .iter()
                    .find(|(position, _)| *position == generated_position)
                {
                    let layout =
                        Q4ExpertStageTraceLayout::try_new(token_loop.model_geometry(), layers)?;
                    let staging =
                        token_loop.create_q4_expert_stage_diagnostic_staging_buffer(&layout)?;
                    let (trace, sampled, _) = token_loop
                        .step_token_q4_expert_stage_diagnostic(
                            &runtime.engine,
                            &mut request,
                            input_token,
                            model_position,
                            &layout,
                            &staging,
                        )
                        .await?;
                    captures.push(GpuTargetCapture {
                        case: case_name,
                        generated_position,
                        trace,
                    });
                    input_token = sampled;
                } else {
                    input_token = token_loop
                        .step_token(
                            &runtime.engine,
                            &mut request,
                            input_token,
                            model_position,
                            true,
                        )
                        .await?
                        .ok_or("GPU stage step produced no sampled token")?;
                }
                model_position = model_position
                    .checked_add(1)
                    .ok_or("GPU stage model position overflow")?;
            }
            Ok::<_, Box<dyn std::error::Error>>(captures)
        },
    )
    .await?;
    let expected_completed = prompt_token_ids
        .len()
        .saturating_sub(1)
        .checked_add(maximum_target + 1)
        .ok_or("GPU stage completion count overflow")?;
    if request.committed_position() != expected_completed {
        return Err(format!(
            "GPU stage case {case_name} retired at {}, expected {expected_completed}",
            request.committed_position()
        )
        .into());
    }
    Ok((captures, expected_completed))
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
            return Err("stage diagnostic GPU resolved configuration identity drifted".into());
        }
        let device = runtime
            .engine
            .gpu_device_identity()
            .ok_or("stage diagnostic GPU runtime has no adapter identity")?;
        if device.name != expected_adapter_name
            || device.software_adapter
            || device.device_type.eq_ignore_ascii_case("cpu")
        {
            return Err(format!(
                "stage diagnostic selected adapter {:?}, expected real {:?}",
                device.name, expected_adapter_name
            )
            .into());
        }
        let token_loop = runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or("stage diagnostic GPU-native token loop was not initialized")?;
        let geometry = token_loop.model_geometry();
        if geometry.num_layers != 48
            || geometry.num_experts != 128
            || geometry.top_k != 8
            || geometry.d_model != 2048
            || geometry.d_ff != 768
        {
            return Err("stage diagnostic GPU model geometry differs from frozen V2".into());
        }
        if token_loop.snapshot() != GpuNativeTokenLoopSnapshot::default() {
            return Err("stage diagnostic GPU token-loop counters did not start at zero".into());
        }
        let model_load = crate::greedy_parity_model_load(&runtime);
        let cases: [(&'static str, Vec<(usize, Vec<usize>)>); 3] = [
            ("rust-ownership-holdout", vec![(1, vec![0, 19, 33, 40, 44])]),
            ("postgres-window-holdout", vec![(13, vec![47])]),
            (
                "spanish-refactor-holdout",
                vec![(11, vec![47]), (13, vec![47]), (15, vec![47])],
            ),
        ];
        let mut targets = Vec::new();
        let mut expected_completed = 0usize;
        for (case, positions) in cases {
            let (mut captured, completed) =
                capture_gpu_case(&runtime, &tokenizer, case, &positions, watchdog).await?;
            targets.append(&mut captured);
            expected_completed = expected_completed
                .checked_add(completed)
                .ok_or("GPU stage total completion count overflow")?;
        }
        Ok::<_, Box<dyn std::error::Error>>(GpuCapture {
            targets,
            model_geometry: geometry,
            gate_identities: [0, 19, 33, 40, 44, 47]
                .into_iter()
                .map(|layer| {
                    (
                        layer,
                        GateTensorIdentity::from_gate(layer, &runtime.model.layers[layer].gate),
                    )
                })
                .collect(),
            model_load,
            device,
            background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(),
            token_loop_snapshot: token_loop.snapshot(),
            expected_completed_token_steps: u64::try_from(expected_completed)
                .map_err(|_| "GPU stage completion count does not fit u64")?,
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
            "{error}; stage diagnostic GPU shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

fn compare(
    left_source: &'static str,
    right_source: &'static str,
    left: &[f32],
    right: &[f32],
) -> Result<VectorNumericalEvidence, Box<dyn std::error::Error>> {
    Ok(VectorNumericalEvidence::compare(
        left_source,
        right_source,
        left,
        right,
    )?)
}

async fn analyze_target(
    runtime: &crate::BenchRealRuntime,
    target: FrozenStageTarget,
    gpu: &Q4ExpertStageGpuTargetTrace,
) -> Result<TargetStageAttributionEvidence, Box<dyn std::error::Error>> {
    if gpu.layer != target.layer
        || gpu.router_input.len() != runtime.model.config.d_model
        || gpu.router_input.iter().any(|value| !value.is_finite())
        || gpu.selected_ids.len() != runtime.model.config.top_k
        || gpu.selected_weights.len() != runtime.model.config.top_k
        || gpu.routes.len() != runtime.model.config.top_k
    {
        return Err(format!(
            "target {} GPU stage geometry is incomplete or nonfinite",
            target.id
        )
        .into());
    }
    let cpu_route = runtime.model.layers[target.layer]
        .gate
        .route(&gpu.router_input);
    let ordered_equal = cpu_route.experts == gpu.selected_ids;
    let router = OrderedRouterEvidence {
        cpu_production_ids: cpu_route.experts.clone(),
        actual_gpu_ids: gpu.selected_ids.clone(),
        exact_ordered_equal: ordered_equal,
        decomposition_valid: ordered_equal,
        mismatch_not_repaired_or_reordered: true,
    };
    let selected_weights = ExactVectorEvidence::new(
        "actual-production-gpu-selected-weights",
        &gpu.selected_weights,
        true,
    );
    if !ordered_equal {
        return Ok(TargetStageAttributionEvidence {
            target,
            actual_gpu_selected_weights: selected_weights,
            router,
            input_boundary: None,
            experts: Vec::new(),
            failure: Some(
                "same-input CPU production ordered IDs differ from actual GPU ordered IDs; no repair or rank reordering was applied"
                    .to_string(),
            ),
        });
    }
    if target
        .frozen_worst_local_expert
        .is_some_and(|expert| !gpu.selected_ids.contains(&expert))
    {
        return Ok(TargetStageAttributionEvidence {
            target,
            actual_gpu_selected_weights: selected_weights,
            router,
            input_boundary: None,
            experts: Vec::new(),
            failure: Some("frozen worst local expert was not selected at reproduced target".into()),
        });
    }
    let gpu_ranks = pair_routes_by_expert_id(&cpu_route.experts, &gpu.selected_ids)?;
    let mut experts = Vec::with_capacity(cpu_route.experts.len());
    let mut input_boundary = None;
    for (cpu_rank, local_expert_id) in cpu_route.experts.iter().copied().enumerate() {
        let gpu_rank = gpu_ranks[cpu_rank];
        let gpu_route = &gpu.routes[gpu_rank];
        if gpu_route.expert_id != local_expert_id
            || gpu_route.gate.len() != runtime.model.config.d_ff
            || gpu_route.up.len() != runtime.model.config.d_ff
            || gpu_route.gated.len() != runtime.model.config.d_ff
            || gpu_route.diagnostic_down.len() != runtime.model.config.d_model
            || gpu_route.production_down.len() != runtime.model.config.d_model
        {
            return Err(format!(
                "target {} expert {local_expert_id} GPU stage geometry is incomplete",
                target.id
            )
            .into());
        }
        let global_expert_id = runtime
            .model
            .global_expert_id(target.layer, local_expert_id);
        let bundle = runtime
            .engine
            .diagnose_q4_0_expert_stages(
                global_expert_id,
                &gpu.router_input,
                &gpu_route.gate,
                &gpu_route.up,
                &gpu_route.gated,
            )
            .await?;
        let effective_input_equal = exact_bits_equal(&gpu.router_input, &bundle.cpu_boundary_input);
        let current_input_boundary = InputBoundaryEvidence {
            exact_pre_boundary_gpu_router_input: ExactVectorEvidence::new(
                "exact-pre-boundary-production-gpu-router-and-expert-input",
                &gpu.router_input,
                true,
            ),
            effective_gpu_expert_input: ExactVectorEvidence::new(
                "effective-production-gpu-expert-input",
                &gpu.router_input,
                true,
            ),
            cpu_boundary_emulated_input: ExactVectorEvidence::new(
                "cpu-diagnostic-f16-boundary-emulated-expert-input",
                &bundle.cpu_boundary_input,
                true,
            ),
            effective_input_bit_identical: effective_input_equal,
            mismatch_cannot_be_coerced_into_equality: true,
        };
        if let Some(existing) = &input_boundary {
            if existing != &current_input_boundary {
                return Err(
                    "per-expert CPU input boundary evidence drifted within one target".into(),
                );
            }
        } else {
            input_boundary = Some(current_input_boundary);
        }
        let gate = compare(
            "cpu-production-candle-raw-gate",
            "actual-production-gpu-raw-gate",
            &bundle.cpu_production.gate,
            &gpu_route.gate,
        )?;
        let up = compare(
            "cpu-production-candle-raw-up",
            "actual-production-gpu-raw-up",
            &bundle.cpu_production.up,
            &gpu_route.up,
        )?;
        let gated = compare(
            "cpu-production-candle-post-swiglu-gated",
            "actual-production-gpu-post-swiglu-gated",
            &bundle.cpu_production.gated,
            &gpu_route.gated,
        )?;
        let diagnostic_down = compare(
            "cpu-production-candle-down",
            "diagnostic-gpu-down",
            &bundle.cpu_production.down,
            &gpu_route.diagnostic_down,
        )?;
        let diagnostic_vs_production_gpu_down = compare(
            "diagnostic-gpu-down",
            "actual-production-gpu-down",
            &gpu_route.diagnostic_down,
            &gpu_route.production_down,
        )?;
        let production_down = compare(
            "cpu-production-candle-down",
            "actual-production-gpu-down",
            &bundle.cpu_production.down,
            &gpu_route.production_down,
        )?;
        let final_boundary_equal =
            exact_bits_equal(&bundle.cpu_effective_output, &gpu_route.production_down);
        let final_output_boundary = OutputBoundaryEvidence {
            cpu_pre_output: ExactVectorEvidence::new(
                "cpu-production-candle-pre-boundary-expert-output",
                &bundle.cpu_production.down,
                false,
            ),
            cpu_effective_output: ExactVectorEvidence::new(
                "cpu-diagnostic-f16-boundary-emulated-expert-output",
                &bundle.cpu_effective_output,
                true,
            ),
            actual_gpu_effective_output: ExactVectorEvidence::new(
                "actual-production-gpu-expert-output",
                &gpu_route.production_down,
                true,
            ),
            final_boundary_bit_identical: final_boundary_equal,
            mismatch_cannot_be_coerced_into_equality: true,
        };
        let projection_implied_final = compare(
            "cpu-production-baseline-down",
            "gpu-gate-up-through-cpu-production-swiglu-and-down",
            &bundle.cpu_production.down,
            &bundle.gpu_gate_up_through_cpu_swiglu_down.down,
        )?;
        let swiglu_incremental_final = compare(
            "gpu-gate-up-through-cpu-production-swiglu-and-down",
            "actual-gpu-gated-through-cpu-production-down",
            &bundle.gpu_gate_up_through_cpu_swiglu_down.down,
            &bundle.gpu_gated_through_cpu_down,
        )?;
        let down_incremental_final = compare(
            "actual-gpu-gated-through-cpu-production-down",
            "actual-production-gpu-down",
            &bundle.gpu_gated_through_cpu_down,
            &gpu_route.production_down,
        )?;
        let comparative_error_growth = ComparativeErrorGrowthEvidence {
            contract: "threshold-free comparative final-output residuals: A-to-B projection contribution, B-to-C SwiGLU increment, C-to-actual-GPU down increment",
            projection_implied_final_error_strictly_positive:
                strictly_positive_aggregate_error(&projection_implied_final),
            swiglu_incremental_final_error_strictly_positive:
                strictly_positive_aggregate_error(&swiglu_incremental_final),
            down_incremental_final_error_strictly_positive:
                strictly_positive_aggregate_error(&down_incremental_final),
            arbitrary_numerical_threshold_used: false,
        };
        let attribution = derive_attribution(
            effective_input_equal,
            &gate,
            &up,
            &production_down,
            &projection_implied_final,
            &swiglu_incremental_final,
            &down_incremental_final,
        );
        let mixed_replay = MixedReplayEvidence {
            cpu_baseline_vs_actual_gpu: compare(
                "cpu-production-baseline-down",
                "actual-production-gpu-down",
                &bundle.cpu_production.down,
                &gpu_route.production_down,
            )?,
            gpu_gate_up_cpu_swiglu_down_vs_cpu_baseline: projection_implied_final,
            gpu_gate_up_cpu_swiglu_down_vs_actual_gpu: compare(
                "gpu-gate-up-through-cpu-production-swiglu-and-down",
                "actual-production-gpu-down",
                &bundle.gpu_gate_up_through_cpu_swiglu_down.down,
                &gpu_route.production_down,
            )?,
            gpu_gated_cpu_down_vs_cpu_baseline: compare(
                "cpu-production-baseline-down",
                "actual-gpu-gated-through-cpu-production-down",
                &bundle.cpu_production.down,
                &bundle.gpu_gated_through_cpu_down,
            )?,
            gpu_gated_cpu_down_vs_actual_gpu: down_incremental_final,
            gpu_gate_up_cpu_swiglu_down_vs_gpu_gated_cpu_down: swiglu_incremental_final,
            cpu_gated_current_gpu_down_vs_cpu_baseline: compare(
                "cpu-production-baseline-down",
                "cpu-gated-through-current-gpu-q4-down-emulator",
                &bundle.cpu_production.down,
                &bundle.cpu_gated_through_current_gpu_down,
            )?,
            cpu_gated_current_gpu_down_vs_actual_gpu: compare(
                "cpu-gated-through-current-gpu-q4-down-emulator",
                "actual-production-gpu-down",
                &bundle.cpu_gated_through_current_gpu_down,
                &gpu_route.production_down,
            )?,
            gpu_gated_current_gpu_down_vs_cpu_baseline: compare(
                "cpu-production-baseline-down",
                "actual-gpu-gated-through-current-gpu-q4-down-emulator",
                &bundle.cpu_production.down,
                &bundle.gpu_gated_through_current_gpu_down,
            )?,
            gpu_gated_current_gpu_down_vs_actual_gpu: compare(
                "actual-gpu-gated-through-current-gpu-q4-down-emulator",
                "actual-production-gpu-down",
                &bundle.gpu_gated_through_current_gpu_down,
                &gpu_route.production_down,
            )?,
            diagnostic_only_never_fed_to_production: true,
        };
        let current_gpu_gate_vs_actual_gpu = compare(
            "current-gpu-q4-dot-cpu-emulator-gate",
            "actual-production-gpu-gate",
            &bundle.current_gpu_emulation.gate,
            &gpu_route.gate,
        )?;
        let current_gpu_up_vs_actual_gpu = compare(
            "current-gpu-q4-dot-cpu-emulator-up",
            "actual-production-gpu-up",
            &bundle.current_gpu_emulation.up,
            &gpu_route.up,
        )?;
        let current_gpu_gated_vs_actual_gpu = compare(
            "current-gpu-q4-dot-cpu-emulator-gated",
            "actual-production-gpu-gated",
            &bundle.current_gpu_emulation.gated,
            &gpu_route.gated,
        )?;
        let current_gpu_down_vs_actual_gpu = compare(
            "current-gpu-q4-dot-cpu-emulator-down",
            "actual-production-gpu-down",
            &bundle.current_gpu_emulation.down,
            &gpu_route.production_down,
        )?;
        let current_gpu_q4_dot_hardware_model_assessment = current_gpu_emulator_assessment(&[
            &current_gpu_gate_vs_actual_gpu,
            &current_gpu_up_vs_actual_gpu,
            &current_gpu_gated_vs_actual_gpu,
            &current_gpu_down_vs_actual_gpu,
        ]);
        let arithmetic_emulators = ArithmeticEmulatorEvidence {
            current_gpu_q4_dot_status: crate::inference::DiagnosticQ4DotArithmetic::CurrentGpu
                .status(),
            current_gpu_q4_dot_hardware_model_assessment,
            rejected_logical_dequant_q4_dot_status:
                crate::inference::DiagnosticQ4DotArithmetic::RejectedLogicalDequant.status(),
            current_gpu_gate_vs_actual_gpu,
            current_gpu_up_vs_actual_gpu,
            current_gpu_gated_vs_actual_gpu,
            current_gpu_down_vs_actual_gpu,
            rejected_logical_gate_vs_cpu_production: compare(
                "cpu-production-candle-gate",
                "rejected-logical-dequant-q4-dot-emulator-gate",
                &bundle.cpu_production.gate,
                &bundle.rejected_logical_dequant_emulation.gate,
            )?,
            rejected_logical_up_vs_cpu_production: compare(
                "cpu-production-candle-up",
                "rejected-logical-dequant-q4-dot-emulator-up",
                &bundle.cpu_production.up,
                &bundle.rejected_logical_dequant_emulation.up,
            )?,
            rejected_logical_gated_vs_cpu_production: compare(
                "cpu-production-candle-gated",
                "rejected-logical-dequant-q4-dot-emulator-gated",
                &bundle.cpu_production.gated,
                &bundle.rejected_logical_dequant_emulation.gated,
            )?,
            rejected_logical_down_vs_cpu_production: compare(
                "cpu-production-candle-down",
                "rejected-logical-dequant-q4-dot-emulator-down",
                &bundle.cpu_production.down,
                &bundle.rejected_logical_dequant_emulation.down,
            )?,
        };
        experts.push(ExpertStageAttributionEvidence {
            rank: cpu_rank + 1,
            local_expert_id,
            global_expert_id,
            canonical_q4_payload_sha256: bundle.canonical_payload_sha256,
            gpu_route_paired_by_exact_expert_id: true,
            cpu_production_final_bit_identical_to_ordinary_path: true,
            final_output_boundary,
            stage_comparison: StageComparisonEvidence {
                gate,
                up,
                gated_swiglu: gated,
                diagnostic_down,
                diagnostic_vs_production_gpu_down,
                production_down,
            },
            mixed_replay,
            comparative_error_growth,
            arithmetic_emulators,
            attribution,
        });
    }
    Ok(TargetStageAttributionEvidence {
        target,
        actual_gpu_selected_weights: selected_weights,
        router,
        input_boundary,
        experts,
        failure: None,
    })
}

struct ReferenceCapture {
    targets: Vec<TargetStageAttributionEvidence>,
    failure: Option<String>,
    model_load: crate::greedy_parity::ModelLoadEvidence,
    background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

async fn execute_reference_and_analyze(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    resolved_config_sha256: &str,
    gpu: &GpuCapture,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<ReferenceCapture, Box<dyn std::error::Error>> {
    let runtime = crate::build_isolated_greedy_runtime(
        spec,
        RealCliRuntimeMode::IsolatedGreedyParityCpu,
        tokenizer,
    )
    .await?;
    let attempt = async {
        runtime.engine.enable_cpu_q4_boundary_emulation()?;
        let boundary_before = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_before.enabled || boundary_before.routed_expert_dispatches != 0 {
            return Err("stage diagnostic CPU boundary emulation did not start clean".into());
        }
        let observed_config_sha256 = crate::resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err("stage diagnostic reference configuration identity drifted".into());
        }
        if runtime.model.layers.len() != gpu.model_geometry.num_layers {
            return Err("stage diagnostic reference layer geometry differs from GPU".into());
        }
        for (&layer, gpu_identity) in &gpu.gate_identities {
            let reference_identity =
                GateTensorIdentity::from_gate(layer, &runtime.model.layers[layer].gate);
            if &reference_identity != gpu_identity {
                return Err(
                    format!("stage diagnostic gate identity differs at layer {layer}").into(),
                );
            }
        }
        let model_load = crate::greedy_parity_model_load(&runtime);
        let mut targets = Vec::with_capacity(FROZEN_TARGETS.len());
        for target in FROZEN_TARGETS {
            let gpu_target = gpu.target(target)?;
            targets.push(
                crate::with_progress_timeout(
                    format!("Q4 expert stage CPU analysis {}", target.id),
                    watchdog,
                    analyze_target(&runtime, target, gpu_target),
                )
                .await?,
            );
        }
        let failures = targets
            .iter()
            .filter_map(|target| {
                target
                    .failure
                    .as_ref()
                    .map(|failure| format!("{}: {failure}", target.target.id))
            })
            .collect::<Vec<_>>();
        Ok::<_, Box<dyn std::error::Error>>(ReferenceCapture {
            targets,
            failure: (!failures.is_empty()).then(|| failures.join("; ")),
            model_load,
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
            "{error}; stage diagnostic reference shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

fn model_load_is_strict(load: &crate::greedy_parity::ModelLoadEvidence) -> bool {
    load.strict
        && load.required_tensors > 0
        && load.loaded_tensors == load.required_tensors
        && !load.seeded_fallback_remained
        && load.loader != "seeded"
}

fn emit_report(
    report: &Q4ExpertStageAttributionReport,
    report_out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if report.qualification_pass() || !report.diagnostic_only {
        return Err(
            "Q4 expert stage report must remain diagnostic-only and qualification false".into(),
        );
    }
    if let Some(parent) = report_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    std::fs::write(report_out, json)?;
    eprintln!(
        "GPU-native Q4 expert stage-attribution report written to {}",
        report_out.display()
    );
    Ok(())
}

pub async fn run_diagnostic(
    config: PathBuf,
    cfg: crate::config::Config,
    expected_adapter_name: String,
    bea_report: PathBuf,
    expected_bea_report_sha256: String,
    report_out: PathBuf,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::BuildProvenance;

    // Immutable historical evidence is verified before any runtime or GPU
    // construction. Hashing and deserialization use the same byte snapshot.
    let frozen_bea_result_identity =
        validate_frozen_bea_report(&bea_report, &expected_bea_report_sha256)?;
    if progress_watchdog.timeout.is_none() {
        return Err("Q4 expert stage diagnostic requires a positive progress timeout".into());
    }
    let build = BuildProvenance::embedded();
    if build.dirty != Some(false) || build.git_sha.as_deref().is_none_or(|sha| !is_hex(sha, 40)) {
        return Err("Q4 expert stage diagnostic requires clean embedded Git provenance".into());
    }
    let (artifacts, artifact_errors) = crate::qualification_artifacts(&config, &cfg);
    if !artifact_errors.is_empty() {
        return Err(format!(
            "Q4 expert stage diagnostic artifact preflight failed: {}",
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
            "Q4 expert stage diagnostic requires strict GPU-native Q4 configuration".into(),
        );
    }
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                format!("stage diagnostic expert metadata preflight failed: {error}")
            })?;
    if expert_metadata.dtype.as_deref() != Some("q4_0")
        || expert_metadata.q4_0_layout.as_deref() != Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
        || expert_metadata.explicitly_synthetic
    {
        return Err(
            "Q4 expert stage diagnostic requires canonical nonsynthetic Q4_0 metadata".into(),
        );
    }
    if expected_adapter_name.trim().is_empty() {
        return Err("Q4 expert stage diagnostic requires a nonempty exact adapter name".into());
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
            "Q4 expert stage diagnostic requires exact Qwen3-Coder 30B-A3B Q4_0 geometry".into(),
        );
    }
    let gpu_resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&gpu_spec)?;
    let mut reference_spec = gpu_spec.clone();
    reference_spec.cfg.real_transformer.gpu_native = false;
    reference_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Cpu;
    let reference_resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&reference_spec)?;
    let tokenizer_path = gpu_spec
        .cfg
        .tokenizer
        .path
        .as_ref()
        .ok_or("Q4 expert stage diagnostic requires tokenizer.path")?;
    let tokenizer = Arc::new(crate::tokenizer::Tokenizer::from_file(tokenizer_path)?);

    let gpu = execute_gpu(
        &gpu_spec,
        tokenizer.clone(),
        &gpu_resolved_config_sha256,
        &expected_adapter_name,
        progress_watchdog,
    )
    .await?;
    let reference = execute_reference_and_analyze(
        &reference_spec,
        tokenizer,
        &reference_resolved_config_sha256,
        &gpu,
        progress_watchdog,
    )
    .await?;
    let (_, executable_sha256) = crate::current_executable_identity()?;
    let runtime = DiagnosticRuntimeEvidence {
        gpu_token_loop: gpu.token_loop_snapshot,
        gpu_expected_completed_token_steps: gpu.expected_completed_token_steps,
        reference_background_shutdown: reference.background_shutdown.clone(),
        gpu_native_background_shutdown: gpu.background_shutdown.clone(),
    };
    let mut failures = reference.failure.into_iter().collect::<Vec<_>>();
    if !model_load_is_strict(&reference.model_load) || !model_load_is_strict(&gpu.model_load) {
        failures
            .push("one or both Q4 stage diagnostic runtimes did not load the strict model".into());
    }
    if runtime.gpu_token_loop.tokens_completed != runtime.gpu_expected_completed_token_steps
        || runtime.gpu_token_loop.fatal_failures != 0
        || runtime.gpu_token_loop.no_progress_failures != 0
    {
        failures.push("Q4 stage diagnostic GPU execution did not complete cleanly".into());
    }
    if !runtime
        .reference_background_shutdown
        .controlled_shutdown_requested
        || !runtime
            .reference_background_shutdown
            .all_runtime_resources_released
        || !runtime
            .gpu_native_background_shutdown
            .controlled_shutdown_requested
        || !runtime
            .gpu_native_background_shutdown
            .all_runtime_resources_released
    {
        failures.push("one or both Q4 stage diagnostic runtimes failed controlled shutdown".into());
    }
    let failure = (!failures.is_empty()).then(|| failures.join("; "));
    let report = Q4ExpertStageAttributionReport::new(
        DiagnosticProvenance {
            build,
            executable_sha256,
            artifacts,
            gpu_resolved_config_sha256,
            reference_resolved_config_sha256,
            model_identity,
            reference_model_load: reference.model_load,
            gpu_native_model_load: gpu.model_load,
            reference_background_shutdown: reference.background_shutdown,
            gpu_native_background_shutdown: gpu.background_shutdown,
            expert_metadata,
            device: gpu.device,
        },
        frozen_bea_result_identity,
        failure,
        reference.targets,
        runtime,
        Vec::new(),
    );
    let diagnostic_complete = report.diagnostic_complete;
    let failure = report.failure.clone();
    emit_report(&report, &report_out)?;
    if diagnostic_complete {
        Ok(())
    } else {
        Err(format!(
            "Q4 expert stage attribution incomplete: {}",
            failure.unwrap_or_else(|| "required evidence is incomplete".into())
        )
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> GpuNativeModelGeometry {
        GpuNativeModelGeometry {
            num_layers: 48,
            d_model: 32,
            d_ff: 64,
            num_experts: 128,
            top_k: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            rope_dim: 8,
            vocab_size: 64,
            max_seq_len: 128,
            rms_eps: 1.0e-6,
            rope_base: 1_000_000.0,
        }
    }

    fn valid_bea_envelope() -> FrozenBeaEnvelope {
        let limits = crate::gpu_native_semantic_parity_v2::NumericalLimits::frozen();
        FrozenBeaEnvelope {
            schema: crate::gpu_native_v2_holdout_failure_attribution::SCHEMA_VERSION.into(),
            diagnostic_complete: true,
            qualification_pass: false,
            provenance: FrozenProvenanceEnvelope {
                build: FrozenBuildEnvelope {
                    git_sha: Some(FROZEN_BEA_BUILD_SHA.into()),
                },
            },
            frozen_v2_result_identity: FrozenV2IdentityEnvelope {
                report_artifact: FrozenArtifactEnvelope {
                    sha256:
                        crate::gpu_native_v2_holdout_failure_attribution::FROZEN_V2_REPORT_SHA256
                            .into(),
                },
                expected_report_sha256_argument:
                    crate::gpu_native_v2_holdout_failure_attribution::FROZEN_V2_REPORT_SHA256.into(),
                expected_schema: crate::gpu_native_semantic_parity_v2::SCHEMA_VERSION.into(),
                frozen_build_sha:
                    crate::gpu_native_v2_holdout_failure_attribution::FROZEN_V2_BUILD_SHA.into(),
                qualification_pass: false,
                holdout_corpus_id: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_ID.into(),
                holdout_corpus_sha256: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_SHA256
                    .into(),
                numerical_limits: FrozenNumericalLimitsEnvelope {
                    max_absolute_error_limit: limits.max_absolute_error_limit,
                    rms_error_limit: limits.rms_error_limit,
                    mean_absolute_error_limit: limits.mean_absolute_error_limit,
                    nonfinite_mismatch_limit: limits.nonfinite_mismatch_limit,
                    semantic_correctness_not_bit_parity: limits.semantic_correctness_not_bit_parity,
                },
                immutable_input_verified: true,
            },
        }
    }

    #[test]
    fn schema_and_qualification_contract_are_versioned_and_false() {
        assert_eq!(
            SCHEMA_VERSION,
            "mer.gpu-native-q4-expert-stage-attribution.v1"
        );
        assert_eq!(MODE, "diagnose-gpu-native-q4-expert-stages");
        assert!(!QUALIFICATION_PASS);
        assert!(!ProductionSemanticsEvidence::default().qualification_threshold_introduced);
    }

    #[test]
    fn frozen_target_list_is_exactly_q1_through_q4_and_r1_through_r5() {
        assert_eq!(FROZEN_TARGETS.len(), 9);
        assert_eq!(
            FROZEN_TARGETS.map(|target| (
                target.id,
                target.case,
                target.generated_position,
                target.layer,
                target.frozen_worst_local_expert,
            )),
            [
                ("Q1", "postgres-window-holdout", 13, 47, Some(28)),
                ("Q2", "spanish-refactor-holdout", 11, 47, Some(58)),
                ("Q3", "spanish-refactor-holdout", 13, 47, Some(28)),
                ("Q4", "spanish-refactor-holdout", 15, 47, Some(116)),
                ("R1", "rust-ownership-holdout", 1, 0, None),
                ("R2", "rust-ownership-holdout", 1, 19, None),
                ("R3", "rust-ownership-holdout", 1, 33, None),
                ("R4", "rust-ownership-holdout", 1, 40, None),
                ("R5", "rust-ownership-holdout", 1, 44, None),
            ]
        );
    }

    #[test]
    fn frozen_bea_envelope_validation_is_fail_closed() {
        assert!(validate_expected_bea_report_sha_argument(FROZEN_BEA_REPORT_SHA256).is_ok());
        assert!(validate_expected_bea_report_sha_argument(&"0".repeat(64)).is_err());
        let mut envelope = valid_bea_envelope();
        assert!(validate_bea_envelope(&envelope).is_ok());
        envelope.diagnostic_complete = false;
        assert!(validate_bea_envelope(&envelope).is_err());
        envelope = valid_bea_envelope();
        envelope.schema.push_str("-drift");
        assert!(validate_bea_envelope(&envelope).is_err());
        envelope = valid_bea_envelope();
        envelope.provenance.build.git_sha = Some("0".repeat(40));
        assert!(validate_bea_envelope(&envelope).is_err());
        envelope = valid_bea_envelope();
        envelope.frozen_v2_result_identity.immutable_input_verified = false;
        assert!(validate_bea_envelope(&envelope).is_err());
    }

    #[test]
    fn stage_layout_preserves_route_rank_and_expert_id() {
        let layout = Q4ExpertStageTraceLayout::try_new(geometry(), &[0, 19]).unwrap();
        assert_eq!(layout.targets.len(), 2);
        assert_eq!(layout.stage_elements(), 2 * (3 * 64 + 32));
        let mut bytes = vec![0u8; layout.total_bytes as usize];
        for target in &layout.targets {
            let ids = [17u32, 9u32];
            for (index, id) in ids.into_iter().enumerate() {
                let start = target.selected_ids_offset as usize + index * 4;
                bytes[start..start + 4].copy_from_slice(&id.to_le_bytes());
            }
            for index in 0..layout.stage_elements() {
                let value = (target.layer * 10_000 + index) as f32;
                let start = target.stages_offset as usize + index * 4;
                bytes[start..start + 4].copy_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        let trace = layout.parse(&bytes).unwrap();
        assert_eq!(trace.targets[0].selected_ids, vec![17, 9]);
        assert_eq!(trace.targets[0].routes[0].expert_id, 17);
        assert_eq!(trace.targets[0].routes[1].expert_id, 9);
        assert_eq!(trace.targets[0].routes[0].gate.len(), 64);
        assert_eq!(trace.targets[0].routes[0].down_len_for_test(), 32);
    }

    #[test]
    fn route_pairing_requires_exact_order_and_never_repairs() {
        assert_eq!(
            pair_routes_by_expert_id(&[28, 7], &[28, 7]).unwrap(),
            vec![0, 1]
        );
        assert!(pair_routes_by_expert_id(&[28, 7], &[7, 28]).is_err());
        assert!(pair_routes_by_expert_id(&[28, 7], &[28, 28]).is_err());
    }

    #[test]
    fn boundary_equality_is_exact_and_cannot_coerce_mismatch() {
        assert!(exact_bits_equal(&[1.0, -0.0], &[1.0, -0.0]));
        assert!(!exact_bits_equal(&[1.0, -0.0], &[1.0, 0.0]));
        assert!(!exact_bits_equal(&[1.0], &[1.0, 2.0]));
    }

    #[test]
    fn non_bit_equal_zero_error_does_not_name_a_stage() {
        let zero_sign = VectorNumericalEvidence::compare("left", "right", &[-0.0], &[0.0]).unwrap();
        assert!(!zero_sign.exact_bit_equal);
        assert_eq!(
            derive_attribution(
                true, &zero_sign, &zero_sign, &zero_sign, &zero_sign, &zero_sign, &zero_sign,
            ),
            StageAttribution::NoMaterialLocalization
        );
    }

    #[test]
    fn attribution_uses_comparative_replay_growth_without_a_threshold() {
        let same = VectorNumericalEvidence::compare("left", "right", &[1.0], &[1.0]).unwrap();
        let drift = VectorNumericalEvidence::compare("left", "right", &[1.0], &[2.0]).unwrap();
        assert_eq!(
            derive_attribution(true, &drift, &same, &drift, &drift, &same, &same,),
            StageAttribution::GateProjectionDrift
        );
        assert_eq!(
            derive_attribution(true, &drift, &same, &drift, &drift, &drift, &same,),
            StageAttribution::MultiStageDrift
        );
    }

    #[test]
    fn dot_emulator_statuses_are_descriptive_and_rejected() {
        assert_eq!(
            crate::inference::DiagnosticQ4DotArithmetic::CurrentGpu.status(),
            "descriptive-current-production-gpu-q4-arithmetic"
        );
        assert_eq!(
            crate::inference::DiagnosticQ4DotArithmetic::RejectedLogicalDequant.status(),
            "rejected-software-hypothesis-diagnostic-only-not-production"
        );
    }
}

#[cfg(test)]
impl Q4ExpertStageGpuRouteTrace {
    fn down_len_for_test(&self) -> usize {
        self.diagnostic_down.len()
    }
}
