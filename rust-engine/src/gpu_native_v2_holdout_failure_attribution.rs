//! Diagnostic-only attribution for the frozen first GPU-native semantic
//! parity v2 holdout failure.
//!
//! This module consumes the immutable v2 report as input evidence and reuses
//! existing production observation seams. It is never consulted by ordinary
//! inference or by any qualification PASS derivation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::gpu_native_expert_permutation_semantic_parity::{
    PermutationOnlyWitness, VectorNumericalEvidence, WeightDeltaEvidence,
};
use crate::gpu_native_router_rank_diagnostics::{
    ActualGpuRouterEvidence, DiagnosticProvenance, GateIdentityEvidence, GateTensorIdentity,
    RankedExpertEvidence, RouterEvaluationEvidence,
};
use crate::gpu_native_semantic_parity_corpus::{
    RoutingClassification, SemanticCorpusGpuLayerTrace, SemanticCorpusGpuTrace,
    SemanticCorpusTraceLayout,
};
use crate::gpu_native_token_loop::{GpuNativeModelGeometry, GpuNativeTokenLoopSnapshot};
use crate::numerical_diagnostics::FloatEvidence;
use crate::{IsolatedRuntimeShutdownError, RealCliRuntimeMode, ResolvedRealCliSpec};

pub const SCHEMA_VERSION: &str = "mer.gpu-native-v2-holdout-failure-attribution.v1";
pub const MODE: &str = "diagnose-gpu-native-v2-holdout-failures";
pub const FROZEN_V2_BUILD_SHA: &str = "300448ac3da48ac74ef86fcd2c62ffca27ffa634";
pub const FROZEN_V2_REPORT_SHA256: &str =
    "9629de35a18f4457bbe38aeeaeb208fd979bb08e9d2eccb283437d6c16a5c4ad";
pub const FROZEN_V2_EXECUTION_LOG_SHA256: &str =
    "3122208935d1554e0eb119d8f293da42c1a94b253599af1b454b09b80d317dda";
pub const FROZEN_OFFLINE_ANALYSIS_SHA256: &str =
    "06c0e01fc1eb99b682ff83dc209f39e65e3472d0c70d594a99364b20f85430e8";

pub const RUST_TARGET_CASE: &str = "rust-ownership-holdout";
pub const RUST_TARGET_POSITION: usize = 1;
pub const RUST_PRECEDING_TOKEN: u32 = 4710;
pub const RUST_REFERENCE_TARGET_TOKEN: u32 = 785;
pub const RUST_GPU_TARGET_TOKEN: u32 = 8822;

pub const ROUTER_TARGET_CASE: &str = "postgres-window-holdout";
pub const ROUTER_TARGET_POSITION: usize = 11;
pub const ROUTER_TARGET_LAYER: usize = 38;
pub const ROUTER_CPU_EXPECTED_IDS: [u32; 8] = [102, 73, 30, 87, 71, 36, 115, 107];
pub const ROUTER_GPU_EXPECTED_IDS: [u32; 8] = [102, 73, 30, 87, 71, 36, 115, 95];
pub const ROUTER_EXPLICIT_EXPERTS: [u32; 2] = [95, 107];

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FrozenMoeOutlierTarget {
    pub case: &'static str,
    pub generated_position: usize,
    pub layer: usize,
    pub classification: &'static str,
    pub frozen_max_absolute_error: f64,
    pub frozen_rms_error: f64,
    pub frozen_mean_absolute_error: f64,
}

pub const MOE_OUTLIER_TARGETS: [FrozenMoeOutlierTarget; 4] = [
    FrozenMoeOutlierTarget {
        case: "postgres-window-holdout",
        generated_position: 13,
        layer: 47,
        classification: "internal-rank-permutation",
        frozen_max_absolute_error: 0.030458450317382812,
        frozen_rms_error: 0.0009415900264223419,
        frozen_mean_absolute_error: 0.0002828433401447228,
    },
    FrozenMoeOutlierTarget {
        case: "spanish-refactor-holdout",
        generated_position: 11,
        layer: 47,
        classification: "membership-mismatch",
        frozen_max_absolute_error: 0.025327682495117188,
        frozen_rms_error: 0.0007412235007170731,
        frozen_mean_absolute_error: 0.00021333931897515868,
    },
    FrozenMoeOutlierTarget {
        case: "spanish-refactor-holdout",
        generated_position: 13,
        layer: 47,
        classification: "internal-rank-permutation",
        frozen_max_absolute_error: 0.02174687385559082,
        frozen_rms_error: 0.0006797136324364518,
        frozen_mean_absolute_error: 0.00021244322049795983,
    },
    FrozenMoeOutlierTarget {
        case: "spanish-refactor-holdout",
        generated_position: 15,
        layer: 47,
        classification: "internal-rank-permutation",
        frozen_max_absolute_error: 0.021457672119140625,
        frozen_rms_error: 0.000589083423933804,
        frozen_mean_absolute_error: 0.00016039697858616364,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FrozenRouterCoupledTarget {
    pub case: &'static str,
    pub generated_position: usize,
    pub layer: usize,
    pub classification: &'static str,
    pub frozen_max_absolute_error: f64,
    pub frozen_rms_error: f64,
    pub frozen_mean_absolute_error: f64,
    pub primary_attribution_target: &'static str,
    pub excluded_from_exact_router_moe_decomposition: bool,
}

pub const ROUTER_COUPLED_TARGET: FrozenRouterCoupledTarget = FrozenRouterCoupledTarget {
    case: ROUTER_TARGET_CASE,
    generated_position: ROUTER_TARGET_POSITION,
    layer: ROUTER_TARGET_LAYER,
    classification: "router-coupled-numerical-failure",
    frozen_max_absolute_error: 0.04648581147193909,
    frozen_rms_error: 0.011342482386714535,
    frozen_mean_absolute_error: 0.009020052351949914,
    primary_attribution_target: "target-b-same-input-router-defect",
    excluded_from_exact_router_moe_decomposition: true,
};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FrozenTargetListEvidence {
    pub rust_token_target_case: &'static str,
    pub rust_token_target_position: usize,
    pub rust_preceding_token: u32,
    pub rust_reference_target_token: u32,
    pub rust_gpu_target_token: u32,
    pub router_target_case: &'static str,
    pub router_target_position: usize,
    pub router_target_layer: usize,
    pub router_cpu_expected_ids: [u32; 8],
    pub router_gpu_expected_ids: [u32; 8],
    pub exact_router_moe_outliers: Vec<FrozenMoeOutlierTarget>,
    pub router_coupled_numerical_failure: FrozenRouterCoupledTarget,
    pub target_selection_rule: &'static str,
}

impl FrozenTargetListEvidence {
    pub fn fixed() -> Self {
        Self {
            rust_token_target_case: RUST_TARGET_CASE,
            rust_token_target_position: RUST_TARGET_POSITION,
            rust_preceding_token: RUST_PRECEDING_TOKEN,
            rust_reference_target_token: RUST_REFERENCE_TARGET_TOKEN,
            rust_gpu_target_token: RUST_GPU_TARGET_TOKEN,
            router_target_case: ROUTER_TARGET_CASE,
            router_target_position: ROUTER_TARGET_POSITION,
            router_target_layer: ROUTER_TARGET_LAYER,
            router_cpu_expected_ids: ROUTER_CPU_EXPECTED_IDS,
            router_gpu_expected_ids: ROUTER_GPU_EXPECTED_IDS,
            exact_router_moe_outliers: MOE_OUTLIER_TARGETS.to_vec(),
            router_coupled_numerical_failure: ROUTER_COUPLED_TARGET,
            target_selection_rule: "exactly-one-rust-token-one-router-four-exact-router-moe-targets-frozen-before-diagnostic-hardware",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
struct FrozenNumericalLimits {
    max_absolute_error_limit: f64,
    rms_error_limit: f64,
    mean_absolute_error_limit: f64,
    nonfinite_mismatch_limit: usize,
    semantic_correctness_not_bit_parity: bool,
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
struct FrozenCorpusEnvelope {
    id: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct FrozenV2ReportEnvelope {
    schema: String,
    qualification_pass: bool,
    provenance: FrozenProvenanceEnvelope,
    holdout_corpus: FrozenCorpusEnvelope,
    numerical_limits: FrozenNumericalLimits,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FrozenV2ResultIdentity {
    pub report_artifact: crate::qualification::ArtifactDigest,
    pub expected_report_sha256_argument: String,
    pub expected_schema: &'static str,
    pub frozen_build_sha: &'static str,
    pub qualification_pass: bool,
    pub holdout_corpus_id: &'static str,
    pub holdout_corpus_sha256: &'static str,
    pub numerical_limits: crate::gpu_native_semantic_parity_v2::NumericalLimits,
    pub execution_log_sha256: &'static str,
    pub offline_failure_analysis_sha256: &'static str,
    pub immutable_input_verified: bool,
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn exact_f64(left: f64, right: f64) -> bool {
    left.to_bits() == right.to_bits()
}

fn validate_expected_v2_report_sha_argument(value: &str) -> Result<(), String> {
    if !is_hex(value, 64) || !value.eq_ignore_ascii_case(FROZEN_V2_REPORT_SHA256) {
        return Err(format!(
            "expected V2 report SHA must equal frozen {}",
            FROZEN_V2_REPORT_SHA256
        ));
    }
    Ok(())
}

fn validate_v2_envelope(envelope: &FrozenV2ReportEnvelope) -> Result<(), String> {
    let expected = crate::gpu_native_semantic_parity_v2::NumericalLimits::frozen();
    if envelope.schema != crate::gpu_native_semantic_parity_v2::SCHEMA_VERSION {
        return Err("frozen V2 report schema differs".to_string());
    }
    if envelope.provenance.build.git_sha.as_deref() != Some(FROZEN_V2_BUILD_SHA) {
        return Err("frozen V2 report build SHA differs".to_string());
    }
    if envelope.qualification_pass {
        return Err("frozen V2 report unexpectedly claims qualification PASS".to_string());
    }
    if envelope.holdout_corpus.id != crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_ID
        || envelope.holdout_corpus.sha256
            != crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_SHA256
    {
        return Err("frozen V2 holdout identity differs".to_string());
    }
    let limits = &envelope.numerical_limits;
    if !exact_f64(
        limits.max_absolute_error_limit,
        expected.max_absolute_error_limit,
    ) || !exact_f64(limits.rms_error_limit, expected.rms_error_limit)
        || !exact_f64(
            limits.mean_absolute_error_limit,
            expected.mean_absolute_error_limit,
        )
        || limits.nonfinite_mismatch_limit != expected.nonfinite_mismatch_limit
        || limits.semantic_correctness_not_bit_parity
            != expected.semantic_correctness_not_bit_parity
    {
        return Err("frozen V2 numerical limits differ".to_string());
    }
    Ok(())
}

fn validate_frozen_v2_report(
    path: &Path,
    expected_sha256: &str,
) -> Result<FrozenV2ResultIdentity, Box<dyn std::error::Error>> {
    validate_expected_v2_report_sha_argument(expected_sha256)?;
    // Hash and deserialize the same immutable byte snapshot so a changed file
    // cannot pass the digest check and then supply different JSON.
    let bytes = std::fs::read(path)?;
    let artifact = crate::qualification::ArtifactDigest {
        configured_path: path.display().to_string(),
        canonical_path: std::fs::canonicalize(path)?.display().to_string(),
        byte_length: u64::try_from(bytes.len())
            .map_err(|_| "frozen V2 report length does not fit u64")?,
        sha256: crate::greedy_parity::sha256_hex(&bytes),
    };
    if !artifact.sha256.eq_ignore_ascii_case(expected_sha256)
        || !artifact
            .sha256
            .eq_ignore_ascii_case(FROZEN_V2_REPORT_SHA256)
    {
        return Err(format!(
            "frozen V2 report SHA differs: observed {} expected {}",
            artifact.sha256, FROZEN_V2_REPORT_SHA256
        )
        .into());
    }
    let envelope: FrozenV2ReportEnvelope = serde_json::from_slice(&bytes)
        .map_err(|error| format!("malformed frozen V2 report: {error}"))?;
    validate_v2_envelope(&envelope)?;
    Ok(FrozenV2ResultIdentity {
        report_artifact: artifact,
        expected_report_sha256_argument: expected_sha256.to_ascii_lowercase(),
        expected_schema: crate::gpu_native_semantic_parity_v2::SCHEMA_VERSION,
        frozen_build_sha: FROZEN_V2_BUILD_SHA,
        qualification_pass: false,
        holdout_corpus_id: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_ID,
        holdout_corpus_sha256: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_SHA256,
        numerical_limits: crate::gpu_native_semantic_parity_v2::NumericalLimits::frozen(),
        execution_log_sha256: FROZEN_V2_EXECUTION_LOG_SHA256,
        offline_failure_analysis_sha256: FROZEN_OFFLINE_ANALYSIS_SHA256,
        immutable_input_verified: true,
    })
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
pub struct ProductionSemanticsEvidence {
    pub diagnostic_only: bool,
    pub production_inference_math_changed: bool,
    pub production_router_math_changed: bool,
    pub production_dense_gemv_changed: bool,
    pub shader_or_wgsl_changed: bool,
    pub production_selected_expert_order_changed: bool,
    pub production_accumulation_order_changed: bool,
    pub v1_changed: bool,
    pub v2_changed: bool,
    pub frozen_limits_changed: bool,
    pub frozen_holdout_changed: bool,
    pub cpu_lm_head_contract: &'static str,
    pub gpu_observation_contract: &'static str,
}

impl Default for ProductionSemanticsEvidence {
    fn default() -> Self {
        Self {
            diagnostic_only: true,
            production_inference_math_changed: false,
            production_router_math_changed: false,
            production_dense_gemv_changed: false,
            shader_or_wgsl_changed: false,
            production_selected_expert_order_changed: false,
            production_accumulation_order_changed: false,
            v1_changed: false,
            v2_changed: false,
            frozen_limits_changed: false,
            frozen_holdout_changed: false,
            cpu_lm_head_contract:
                "RealModel::diagnostic_greedy_logits -> DenseWeight::diagnostic_greedy_logits -> exact greedy_argmax per-row row_dot",
            gpu_observation_contract:
                "existing full-token and semantic-corpus diagnostic copy/readback seams only",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DiscreteStageEvidence {
    pub reference: Vec<u32>,
    pub gpu_native: Vec<u32>,
    pub exact_equal: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayerTargetTraceEvidence {
    pub layer: usize,
    pub post_attention: VectorNumericalEvidence,
    pub router_input: VectorNumericalEvidence,
    pub selected_ids: DiscreteStageEvidence,
    pub selected_weights: VectorNumericalEvidence,
    pub post_moe: VectorNumericalEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FullTargetTokenTraceEvidence {
    pub embedding: VectorNumericalEvidence,
    pub layers: Vec<LayerTargetTraceEvidence>,
    pub final_rms_norm: VectorNumericalEvidence,
    pub full_vocabulary_logits: VectorNumericalEvidence,
    pub reference_sampled_token: u32,
    pub gpu_native_sampled_token: u32,
    pub sampled_token_equal: bool,
}

fn compare_target_token_traces(
    reference: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    gpu: &crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace,
) -> Result<FullTargetTokenTraceEvidence, String> {
    let num_layers = reference.layer_post_attn.len();
    if num_layers == 0
        || gpu.layer_post_attn.len() != num_layers
        || reference.layer_router_input.len() != num_layers
        || gpu.layer_router_input.len() != num_layers
        || reference.layer_selected_ids.len() != num_layers
        || gpu.layer_selected_ids.len() != num_layers
        || reference.layer_selected_weights.len() != num_layers
        || gpu.layer_selected_weights.len() != num_layers
        || reference.layer_post_moe.len() != num_layers
        || gpu.layer_post_moe.len() != num_layers
    {
        return Err("target-token trace layer geometry is incomplete".to_string());
    }
    let mut layers = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        layers.push(LayerTargetTraceEvidence {
            layer,
            post_attention: VectorNumericalEvidence::compare(
                "cpu-reference-layer-post-attention",
                "gpu-native-layer-post-attention",
                &reference.layer_post_attn[layer],
                &gpu.layer_post_attn[layer],
            )?,
            router_input: VectorNumericalEvidence::compare(
                "cpu-reference-layer-router-input",
                "gpu-native-layer-router-input",
                &reference.layer_router_input[layer],
                &gpu.layer_router_input[layer],
            )?,
            selected_ids: DiscreteStageEvidence {
                reference: reference.layer_selected_ids[layer].clone(),
                gpu_native: gpu.layer_selected_ids[layer].clone(),
                exact_equal: reference.layer_selected_ids[layer] == gpu.layer_selected_ids[layer],
            },
            selected_weights: VectorNumericalEvidence::compare(
                "cpu-reference-layer-selected-weights",
                "gpu-native-layer-selected-weights",
                &reference.layer_selected_weights[layer],
                &gpu.layer_selected_weights[layer],
            )?,
            post_moe: VectorNumericalEvidence::compare(
                "cpu-reference-layer-post-moe",
                "gpu-native-layer-post-moe",
                &reference.layer_post_moe[layer],
                &gpu.layer_post_moe[layer],
            )?,
        });
    }
    Ok(FullTargetTokenTraceEvidence {
        embedding: VectorNumericalEvidence::compare(
            "cpu-reference-embedding",
            "gpu-native-embedding",
            &reference.embedding,
            &gpu.embedding,
        )?,
        layers,
        final_rms_norm: VectorNumericalEvidence::compare(
            "cpu-reference-final-rmsnorm",
            "gpu-native-final-rmsnorm",
            &reference.final_norm,
            &gpu.final_norm,
        )?,
        full_vocabulary_logits: VectorNumericalEvidence::compare(
            "cpu-reference-full-vocabulary-logits",
            "gpu-native-full-vocabulary-logits",
            &reference.logits,
            &gpu.logits,
        )?,
        reference_sampled_token: reference.sampled_token,
        gpu_native_sampled_token: gpu.sampled_token,
        sampled_token_equal: reference.sampled_token == gpu.sampled_token,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TokenLogitEvidence {
    pub token_id: u32,
    pub rank: usize,
    pub raw_logit: FloatEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GreedyLogitViewEvidence {
    pub source: &'static str,
    pub selected_token_id: u32,
    pub top_10: Vec<TokenLogitEvidence>,
    pub explicit_tokens_785_and_8822: Vec<TokenLogitEvidence>,
    pub top_1_minus_top_2_margin: FloatEvidence,
    pub full_logits: ExactVectorEvidence,
}

fn ranked_logit_indices(logits: &[f32]) -> Result<Vec<usize>, String> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err("logit view is empty or nonfinite".to_string());
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
    let ranked = ranked_logit_indices(logits)?;
    u32::try_from(ranked[0]).map_err(|_| "argmax token ID does not fit u32".to_string())
}

fn greedy_view(
    source: &'static str,
    logits: &[f32],
    selected_token_id: u32,
) -> Result<GreedyLogitViewEvidence, String> {
    let ranked = ranked_logit_indices(logits)?;
    let entry = |token_id: usize, rank: usize| TokenLogitEvidence {
        token_id: token_id as u32,
        rank,
        raw_logit: FloatEvidence::new(logits[token_id]),
    };
    let explicit_tokens_785_and_8822 = [RUST_REFERENCE_TARGET_TOKEN, RUST_GPU_TARGET_TOKEN]
        .into_iter()
        .map(|token_id| {
            let index = usize::try_from(token_id).map_err(|_| "token ID overflow")?;
            if index >= logits.len() {
                return Err(format!("explicit token {token_id} is outside vocabulary"));
            }
            let rank = ranked
                .iter()
                .position(|candidate| *candidate == index)
                .ok_or("explicit token is missing from ranking")?
                + 1;
            Ok(entry(index, rank))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(GreedyLogitViewEvidence {
        source,
        selected_token_id,
        top_10: ranked
            .iter()
            .take(10)
            .enumerate()
            .map(|(rank, token)| entry(*token, rank + 1))
            .collect(),
        explicit_tokens_785_and_8822,
        top_1_minus_top_2_margin: FloatEvidence::new(logits[ranked[0]] - logits[ranked[1]]),
        full_logits: ExactVectorEvidence::new(source, logits, false),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TokenAttributionCategory {
    UpstreamHiddenDrift,
    FinalRmsnormDrift,
    GpuLmHeadGemvDrift,
    GpuGreedyArgmaxDrift,
    MixedNumericalDrift,
    Unresolved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TokenAttributionInputs {
    reference_argmax: u32,
    cpu_argmax_on_cpu_norm_of_gpu_pre_norm: u32,
    cpu_argmax_on_actual_gpu_final_norm: u32,
    cpu_argmax_from_actual_gpu_logits: u32,
    actual_gpu_sampled_token: u32,
}

fn derive_token_attribution(inputs: TokenAttributionInputs) -> TokenAttributionCategory {
    if inputs.reference_argmax != RUST_REFERENCE_TARGET_TOKEN
        || inputs.actual_gpu_sampled_token != RUST_GPU_TARGET_TOKEN
    {
        TokenAttributionCategory::Unresolved
    } else if inputs.cpu_argmax_from_actual_gpu_logits != inputs.actual_gpu_sampled_token {
        TokenAttributionCategory::GpuGreedyArgmaxDrift
    } else if inputs.cpu_argmax_on_actual_gpu_final_norm != inputs.actual_gpu_sampled_token {
        TokenAttributionCategory::GpuLmHeadGemvDrift
    } else if inputs.cpu_argmax_on_cpu_norm_of_gpu_pre_norm == inputs.reference_argmax {
        TokenAttributionCategory::FinalRmsnormDrift
    } else if inputs.cpu_argmax_on_cpu_norm_of_gpu_pre_norm == inputs.actual_gpu_sampled_token {
        TokenAttributionCategory::UpstreamHiddenDrift
    } else {
        TokenAttributionCategory::MixedNumericalDrift
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RustTokenReproducibilityEvidence {
    pub case: &'static str,
    pub target_generated_position: usize,
    pub preceding_reference_token: u32,
    pub preceding_gpu_token: u32,
    pub required_preceding_token: u32,
    pub reference_target_token: u32,
    pub gpu_target_token: u32,
    pub exact_frozen_result_reproduced: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TokenDivergenceAttributionEvidence {
    pub reproducibility: RustTokenReproducibilityEvidence,
    pub full_target_token_trace: FullTargetTokenTraceEvidence,
    pub same_input_final_rmsnorm: VectorNumericalEvidence,
    pub same_input_lm_head_logits: VectorNumericalEvidence,
    pub cpu_finalnorm_replay_lm_head_vs_gpu_logits: VectorNumericalEvidence,
    pub greedy_views: Vec<GreedyLogitViewEvidence>,
    pub attribution: TokenAttributionCategory,
    pub positions_after_target_inspected_for_cpu_gpu_semantic_attribution: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExplicitExpertRawLogitEvidence {
    pub expert_id: u32,
    pub rank: usize,
    pub raw_logit: FloatEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RawLogitViewEvidence {
    pub source: &'static str,
    pub exact_vector: ExactVectorEvidence,
    pub raw_logits: Vec<FloatEvidence>,
    pub top_12: Vec<RankedExpertEvidence>,
    pub experts_95_and_107: Vec<ExplicitExpertRawLogitEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RawLogitComparisonEvidence {
    pub comparison: VectorNumericalEvidence,
    pub worst_expert_ulp_distance: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouterTopKViewEvidence {
    pub source: &'static str,
    pub top_8_ids: Vec<u32>,
    pub top_8_weights: Vec<FloatEvidence>,
    pub top_12: Vec<RankedExpertEvidence>,
}

fn raw_logit_view(source: &'static str, logits: &[f32]) -> Result<RawLogitViewEvidence, String> {
    let ranked = ranked_logit_indices(logits)?;
    let mut scores = logits.to_vec();
    crate::transformer::softmax_inplace(&mut scores);
    let top_12 = ranked
        .iter()
        .take(12)
        .enumerate()
        .map(|(rank, expert)| RankedExpertEvidence {
            expert_id: *expert as u32,
            rank: rank + 1,
            raw_logit: FloatEvidence::new(logits[*expert]),
            score: FloatEvidence::new(scores[*expert]),
        })
        .collect::<Vec<_>>();
    let experts_95_and_107 = ROUTER_EXPLICIT_EXPERTS
        .into_iter()
        .map(|expert_id| {
            let index = expert_id as usize;
            let rank = ranked
                .iter()
                .position(|candidate| *candidate == index)
                .ok_or("explicit router expert missing from rank view")?
                + 1;
            Ok(ExplicitExpertRawLogitEvidence {
                expert_id,
                rank,
                raw_logit: FloatEvidence::new(logits[index]),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(RawLogitViewEvidence {
        source,
        exact_vector: ExactVectorEvidence::new(source, logits, true),
        raw_logits: logits.iter().copied().map(FloatEvidence::new).collect(),
        top_12,
        experts_95_and_107,
    })
}

fn compare_raw_logits(
    left_source: &'static str,
    right_source: &'static str,
    left: &[f32],
    right: &[f32],
) -> Result<RawLogitComparisonEvidence, String> {
    let comparison = VectorNumericalEvidence::compare(left_source, right_source, left, right)?;
    let worst_expert_ulp_distance = comparison.worst_element_index.and_then(|index| {
        crate::gpu_native_router_rank_diagnostics::ulp_distance(left[index], right[index])
    });
    Ok(RawLogitComparisonEvidence {
        comparison,
        worst_expert_ulp_distance,
    })
}

fn top_k_from_raw_logits(
    source: &'static str,
    raw_logits: &[f32],
    top_k: usize,
) -> Result<RouterTopKViewEvidence, String> {
    if top_k == 0 || top_k > raw_logits.len() {
        return Err("invalid router top-k geometry".to_string());
    }
    let mut scores = raw_logits.to_vec();
    crate::transformer::softmax_inplace(&mut scores);
    let mut ranked = (0..scores.len()).collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        scores[*right]
            .total_cmp(&scores[*left])
            .then_with(|| left.cmp(right))
    });
    let selected = &ranked[..top_k];
    let selected_sum = selected.iter().map(|expert| scores[*expert]).sum::<f32>();
    if !selected_sum.is_finite() || selected_sum <= 0.0 {
        return Err("router selected score sum is invalid".to_string());
    }
    Ok(RouterTopKViewEvidence {
        source,
        top_8_ids: selected.iter().map(|expert| *expert as u32).collect(),
        top_8_weights: selected
            .iter()
            .map(|expert| FloatEvidence::new(scores[*expert] / selected_sum))
            .collect(),
        top_12: ranked
            .iter()
            .take(12)
            .enumerate()
            .map(|(rank, expert)| RankedExpertEvidence {
                expert_id: *expert as u32,
                rank: rank + 1,
                raw_logit: FloatEvidence::new(raw_logits[*expert]),
                score: FloatEvidence::new(scores[*expert]),
            })
            .collect(),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RouterAttributionCategory {
    GpuRouterGemvDrift,
    GpuRouterSoftmaxTopkDrift,
    ExactTieOrderingDefect,
    MixedNumericalDrift,
    Unresolved,
}

fn derive_router_attribution(
    cpu_production_ids: &[u32],
    cpu_from_gpu_raw_ids: &[u32],
    actual_gpu_ids: &[u32],
    exact_tie_ordering_violation: bool,
) -> RouterAttributionCategory {
    if cpu_production_ids.is_empty()
        || cpu_production_ids.len() != cpu_from_gpu_raw_ids.len()
        || cpu_production_ids.len() != actual_gpu_ids.len()
    {
        RouterAttributionCategory::MixedNumericalDrift
    } else if exact_tie_ordering_violation {
        RouterAttributionCategory::ExactTieOrderingDefect
    } else if cpu_production_ids != actual_gpu_ids && cpu_from_gpu_raw_ids == actual_gpu_ids {
        RouterAttributionCategory::GpuRouterGemvDrift
    } else if cpu_from_gpu_raw_ids != actual_gpu_ids {
        RouterAttributionCategory::GpuRouterSoftmaxTopkDrift
    } else if cpu_production_ids == actual_gpu_ids {
        RouterAttributionCategory::Unresolved
    } else {
        RouterAttributionCategory::MixedNumericalDrift
    }
}

fn exact_ordered_router_ids_equal(cpu: &[u32], gpu: &[u32]) -> bool {
    !cpu.is_empty() && cpu == gpu
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouterFailureAttributionEvidence {
    pub case: &'static str,
    pub generated_position: usize,
    pub layer: usize,
    pub exact_gpu_router_input: ExactVectorEvidence,
    pub gate_identity: GateIdentityEvidence,
    pub cpu_production_router: RouterEvaluationEvidence,
    pub actual_gpu_router: ActualGpuRouterEvidence,
    pub cpu_production_raw_logits: RawLogitViewEvidence,
    pub scalar_sequential_raw_logits: RawLogitViewEvidence,
    pub actual_gpu_raw_logits: RawLogitViewEvidence,
    pub cpu_production_vs_gpu: RawLogitComparisonEvidence,
    pub scalar_sequential_vs_gpu: RawLogitComparisonEvidence,
    pub cpu_production_vs_scalar_sequential: RawLogitComparisonEvidence,
    pub cpu_production_top_k: RouterTopKViewEvidence,
    pub scalar_sequential_top_k: RouterTopKViewEvidence,
    pub cpu_top_k_from_exact_gpu_raw_logits: RouterTopKViewEvidence,
    pub actual_gpu_production_top_k: RouterTopKViewEvidence,
    pub frozen_cpu_ids_reproduced: bool,
    pub frozen_gpu_ids_reproduced: bool,
    pub exact_gpu_raw_and_scored_tie_95_107: bool,
    pub attribution: RouterAttributionCategory,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PerExpertMoeEvidence {
    pub production_rank: usize,
    pub local_expert_id: u32,
    pub global_expert_id: u32,
    pub weight_delta: WeightDeltaEvidence,
    pub cpu_expert_output_f32_bits_sha256: String,
    pub gpu_expert_output_f32_bits_sha256: String,
    pub expert_output_comparison: VectorNumericalEvidence,
    pub weighted_contribution_comparison: VectorNumericalEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MixedCombinationDecompositionEvidence {
    pub ordered_selected_expert_ids: Vec<u32>,
    pub accumulation_contract: &'static str,
    pub baseline_vs_actual: VectorNumericalEvidence,
    pub baseline_vs_expert_only: VectorNumericalEvidence,
    pub baseline_vs_weight_only: VectorNumericalEvidence,
    pub expert_only_vs_actual: VectorNumericalEvidence,
    pub weight_only_vs_actual: VectorNumericalEvidence,
    pub actual_host_combination_vs_actual_gpu_production: VectorNumericalEvidence,
    pub linear_additivity_claimed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MoeOutlierAttributionEvidence {
    pub target: FrozenMoeOutlierTarget,
    pub same_input_cpu_ordered_selected_ids: Vec<u32>,
    pub actual_gpu_ordered_selected_ids: Vec<u32>,
    pub valid_for_exact_router_decomposition: bool,
    pub invalid_reason: Option<String>,
    pub per_expert_in_production_rank_order: Vec<PerExpertMoeEvidence>,
    pub expert_ids_ranked_by_difference_contribution: Vec<u32>,
    pub worst_expert_id: Option<u32>,
    pub mixed_combination_decomposition: Option<MixedCombinationDecompositionEvidence>,
    pub permutation_only_supporting_evidence: Option<PermutationOnlyWitness>,
}

fn weight_delta(expert_id: u32, left: f32, right: f32) -> WeightDeltaEvidence {
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

fn weighted_output(output: &[f32], weight: f32) -> Vec<f32> {
    output.iter().map(|value| weight * *value).collect()
}

fn gpu_ranks_paired_by_expert_id(cpu_ids: &[u32], gpu_ids: &[u32]) -> Result<Vec<usize>, String> {
    let gpu_by_id = gpu_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, expert)| (expert, rank))
        .collect::<BTreeMap<_, _>>();
    if gpu_by_id.len() != gpu_ids.len() {
        return Err("GPU selected expert IDs contain duplicates".to_string());
    }
    cpu_ids
        .iter()
        .map(|expert| {
            gpu_by_id
                .get(expert)
                .copied()
                .ok_or_else(|| format!("GPU route output is missing selected expert ID {expert}"))
        })
        .collect()
}

fn classify_target(value: &str) -> RoutingClassification {
    match value {
        "internal-rank-permutation" => RoutingClassification::InternalRankPermutation,
        "membership-mismatch" => RoutingClassification::MembershipMismatch,
        _ => RoutingClassification::InvalidNonfiniteIncomplete,
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
pub struct GpuNativeV2HoldoutFailureAttributionReport {
    pub schema: &'static str,
    pub mode: &'static str,
    pub diagnostic_only: bool,
    pub diagnostic_complete: bool,
    qualification_pass: bool,
    pub failure: Option<String>,
    pub provenance: DiagnosticProvenance,
    pub frozen_v2_result_identity: FrozenV2ResultIdentity,
    pub frozen_targets: FrozenTargetListEvidence,
    pub token_divergence_attribution: Option<TokenDivergenceAttributionEvidence>,
    pub router_failure_attribution: Option<RouterFailureAttributionEvidence>,
    pub moe_outlier_attributions: Vec<MoeOutlierAttributionEvidence>,
    pub router_coupled_numerical_failure: FrozenRouterCoupledTarget,
    pub runtime: DiagnosticRuntimeEvidence,
    pub production_semantics: ProductionSemanticsEvidence,
    pub observation_seams_not_implemented: Vec<String>,
}

impl GpuNativeV2HoldoutFailureAttributionReport {
    #[allow(clippy::too_many_arguments)]
    fn new(
        provenance: DiagnosticProvenance,
        frozen_v2_result_identity: FrozenV2ResultIdentity,
        failure: Option<String>,
        token_divergence_attribution: Option<TokenDivergenceAttributionEvidence>,
        router_failure_attribution: Option<RouterFailureAttributionEvidence>,
        moe_outlier_attributions: Vec<MoeOutlierAttributionEvidence>,
        runtime: DiagnosticRuntimeEvidence,
        observation_seams_not_implemented: Vec<String>,
    ) -> Self {
        let all_moe_targets_complete = moe_outlier_attributions.len() == MOE_OUTLIER_TARGETS.len()
            && moe_outlier_attributions
                .iter()
                .all(|target| target.valid_for_exact_router_decomposition);
        let diagnostic_complete = failure.is_none()
            && token_divergence_attribution.is_some()
            && router_failure_attribution.is_some()
            && all_moe_targets_complete
            && observation_seams_not_implemented.is_empty();
        Self {
            schema: SCHEMA_VERSION,
            mode: MODE,
            diagnostic_only: true,
            diagnostic_complete,
            qualification_pass: false,
            failure,
            provenance,
            frozen_v2_result_identity,
            frozen_targets: FrozenTargetListEvidence::fixed(),
            token_divergence_attribution,
            router_failure_attribution,
            moe_outlier_attributions,
            router_coupled_numerical_failure: ROUTER_COUPLED_TARGET,
            runtime,
            production_semantics: ProductionSemanticsEvidence::default(),
            observation_seams_not_implemented,
        }
    }

    pub const fn qualification_pass(&self) -> bool {
        self.qualification_pass
    }
}

struct GpuSemanticCapture {
    case: &'static str,
    generated_position: usize,
    trace: SemanticCorpusGpuTrace,
}

struct GpuCapture {
    rust_preceding_token: u32,
    rust_target_token: u32,
    rust_target_trace: crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace,
    semantic_targets: Vec<GpuSemanticCapture>,
    gate_identities: BTreeMap<usize, GateTensorIdentity>,
    model_geometry: GpuNativeModelGeometry,
    model_load: crate::greedy_parity::ModelLoadEvidence,
    device: crate::backend::GpuDeviceIdentity,
    background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
    token_loop_snapshot: GpuNativeTokenLoopSnapshot,
    expected_completed_token_steps: u64,
}

impl GpuCapture {
    fn semantic_layer(
        &self,
        case: &str,
        generated_position: usize,
        layer: usize,
    ) -> Result<&SemanticCorpusGpuLayerTrace, String> {
        self.semantic_targets
            .iter()
            .find(|target| target.case == case && target.generated_position == generated_position)
            .and_then(|target| target.trace.layers.get(layer))
            .ok_or_else(|| {
                format!(
                    "missing GPU semantic target {case} position {generated_position} layer {layer}"
                )
            })
    }
}

struct ReferenceCapture {
    token_attribution: Option<TokenDivergenceAttributionEvidence>,
    router_attribution: Option<RouterFailureAttributionEvidence>,
    moe_attributions: Vec<MoeOutlierAttributionEvidence>,
    failure: Option<String>,
    model_load: crate::greedy_parity::ModelLoadEvidence,
    background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

fn holdout_case(name: &str) -> Result<crate::gpu_native_semantic_parity_v2::HoldoutCase, String> {
    crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS
        .into_iter()
        .find(|case| case.name == name)
        .ok_or_else(|| format!("unknown frozen v2 holdout case {name:?}"))
}

async fn capture_gpu_semantic_case(
    runtime: &crate::BenchRealRuntime,
    tokenizer: &crate::tokenizer::Tokenizer,
    case_name: &'static str,
    target_positions: &[usize],
    trace_layout: &SemanticCorpusTraceLayout,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<(Vec<GpuSemanticCapture>, usize), Box<dyn std::error::Error>> {
    let fixed = holdout_case(case_name)?;
    let prompt_token_ids = tokenizer.encode(fixed.prompt)?;
    if prompt_token_ids.is_empty() || target_positions.is_empty() {
        return Err("semantic target prompt or target list is empty".into());
    }
    let maximum_target = *target_positions
        .iter()
        .max()
        .ok_or("semantic target list is empty")?;
    let token_loop = runtime
        .gpu_native_token_loop
        .as_ref()
        .ok_or("diagnostic GPU-native token loop was not initialized")?;
    let mut request = token_loop.create_semantic_parity_corpus_diagnostic_request_state()?;
    let staging =
        token_loop.create_semantic_parity_corpus_diagnostic_staging_buffer(trace_layout)?;
    let captures = crate::with_progress_timeout(
        format!("v2 holdout failure attribution GPU {case_name}"),
        watchdog,
        async {
            let prefix_count = prompt_token_ids.len().saturating_sub(1);
            for (position, &token_id) in prompt_token_ids[..prefix_count].iter().enumerate() {
                token_loop
                    .step_token(&runtime.engine, &mut request, token_id, position, false)
                    .await?;
            }
            let mut input_token = *prompt_token_ids.last().ok_or("holdout prompt is empty")?;
            let mut position = prefix_count;
            let mut captures = Vec::with_capacity(target_positions.len());
            for generated_position in 0..=maximum_target {
                let sampled = if target_positions.contains(&generated_position) {
                    let (trace, sampled, _) = token_loop
                        .step_token_semantic_parity_corpus_diagnostic(
                            &runtime.engine,
                            &mut request,
                            input_token,
                            position,
                            trace_layout,
                            &staging,
                        )
                        .await?;
                    captures.push(GpuSemanticCapture {
                        case: case_name,
                        generated_position,
                        trace,
                    });
                    sampled
                } else {
                    token_loop
                        .step_token(&runtime.engine, &mut request, input_token, position, true)
                        .await?
                        .ok_or("GPU semantic target step produced no token")?
                };
                input_token = sampled;
                position = position
                    .checked_add(1)
                    .ok_or("GPU semantic target position overflow")?;
            }
            Ok::<_, Box<dyn std::error::Error>>(captures)
        },
    )
    .await?;
    let expected_completed = prompt_token_ids
        .len()
        .saturating_sub(1)
        .checked_add(maximum_target + 1)
        .ok_or("GPU semantic completion count overflow")?;
    if request.committed_position() != expected_completed {
        return Err(format!(
            "GPU semantic case {case_name} retired at {}, expected {expected_completed}",
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
            return Err("diagnostic GPU resolved configuration identity drifted".into());
        }
        let device = runtime
            .engine
            .gpu_device_identity()
            .ok_or("diagnostic GPU runtime has no authoritative adapter identity")?;
        if device.name != expected_adapter_name
            || device.software_adapter
            || device.device_type.eq_ignore_ascii_case("cpu")
        {
            return Err(format!(
                "diagnostic selected adapter {:?}, expected real {:?}",
                device.name, expected_adapter_name
            )
            .into());
        }
        let token_loop = runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or("diagnostic GPU-native token loop was not initialized")?;
        let geometry = token_loop.model_geometry();
        if geometry.num_layers != 48
            || geometry.num_experts != 128
            || geometry.top_k != 8
            || geometry.d_model != 2048
            || geometry.d_ff != 768
        {
            return Err("diagnostic GPU model geometry differs from frozen V2".into());
        }
        if token_loop.snapshot() != GpuNativeTokenLoopSnapshot::default() {
            return Err("diagnostic GPU token-loop counters did not start at zero".into());
        }
        let gate_identities = [ROUTER_TARGET_LAYER, 47]
            .into_iter()
            .map(|layer| {
                (
                    layer,
                    GateTensorIdentity::from_gate(layer, &runtime.model.layers[layer].gate),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let model_load = crate::greedy_parity_model_load(&runtime);

        let fixed = holdout_case(RUST_TARGET_CASE)?;
        let prompt_token_ids = tokenizer.encode(fixed.prompt)?;
        if prompt_token_ids.is_empty() {
            return Err("rust holdout prompt encoded to zero tokens".into());
        }
        let mut request = token_loop.create_request_state()?;
        let trace_layout = crate::gpu_native_diagnostics::GpuNativeDiagnosticTraceLayout::try_new(
            geometry.num_layers,
            geometry.d_model,
            geometry.top_k,
            geometry.vocab_size,
        )?;
        let staging = token_loop.create_diagnostic_staging_buffer(&trace_layout)?;
        let (rust_preceding_token, rust_target_token, rust_target_trace) =
            crate::with_progress_timeout(
                "v2 holdout failure attribution GPU rust token".to_string(),
                watchdog,
                async {
                    let prefix_count = prompt_token_ids.len().saturating_sub(1);
                    for (position, &token_id) in prompt_token_ids[..prefix_count].iter().enumerate()
                    {
                        token_loop
                            .step_token(&runtime.engine, &mut request, token_id, position, false)
                            .await?;
                    }
                    let final_prompt = *prompt_token_ids
                        .last()
                        .ok_or("rust holdout prompt is empty")?;
                    let preceding = token_loop
                        .step_token(
                            &runtime.engine,
                            &mut request,
                            final_prompt,
                            prefix_count,
                            true,
                        )
                        .await?
                        .ok_or("GPU rust position 0 produced no token")?;
                    let (trace, _) = token_loop
                        .step_token_diagnostic(
                            &runtime.engine,
                            &mut request,
                            preceding,
                            prefix_count + 1,
                            true,
                            &trace_layout,
                            &staging,
                        )
                        .await?;
                    Ok::<_, Box<dyn std::error::Error>>((preceding, trace.sampled_token, trace))
                },
            )
            .await?;
        let rust_expected_completed = prompt_token_ids
            .len()
            .saturating_sub(1)
            .checked_add(2)
            .ok_or("GPU rust completion count overflow")?;
        if request.committed_position() != rust_expected_completed {
            return Err("GPU rust request retirement is incomplete".into());
        }
        drop(request);
        drop(staging);

        let mut semantic_targets = Vec::new();
        let mut expected_completed = rust_expected_completed;
        if rust_preceding_token == RUST_PRECEDING_TOKEN
            && rust_target_token == RUST_GPU_TARGET_TOKEN
        {
            let semantic_layout = SemanticCorpusTraceLayout::try_new(geometry)?;
            let (mut postgres, postgres_completed) = capture_gpu_semantic_case(
                &runtime,
                &tokenizer,
                "postgres-window-holdout",
                &[ROUTER_TARGET_POSITION, 13],
                &semantic_layout,
                watchdog,
            )
            .await?;
            let (mut spanish, spanish_completed) = capture_gpu_semantic_case(
                &runtime,
                &tokenizer,
                "spanish-refactor-holdout",
                &[11, 13, 15],
                &semantic_layout,
                watchdog,
            )
            .await?;
            semantic_targets.append(&mut postgres);
            semantic_targets.append(&mut spanish);
            expected_completed = expected_completed
                .checked_add(postgres_completed)
                .and_then(|value| value.checked_add(spanish_completed))
                .ok_or("GPU diagnostic total completion count overflow")?;
        }
        Ok::<_, Box<dyn std::error::Error>>(GpuCapture {
            rust_preceding_token,
            rust_target_token,
            rust_target_trace,
            semantic_targets,
            gate_identities,
            model_geometry: geometry,
            model_load,
            device,
            background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(),
            token_loop_snapshot: token_loop.snapshot(),
            expected_completed_token_steps: u64::try_from(expected_completed)
                .map_err(|_| "GPU diagnostic completion count does not fit u64")?,
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
            "{error}; diagnostic GPU shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

fn analyze_token_divergence(
    runtime: &crate::BenchRealRuntime,
    reference_trace: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    gpu: &GpuCapture,
    reference_preceding_token: u32,
    reference_target_token: u32,
    model_position: usize,
) -> Result<TokenDivergenceAttributionEvidence, String> {
    let reproducibility = RustTokenReproducibilityEvidence {
        case: RUST_TARGET_CASE,
        target_generated_position: RUST_TARGET_POSITION,
        preceding_reference_token: reference_preceding_token,
        preceding_gpu_token: gpu.rust_preceding_token,
        required_preceding_token: RUST_PRECEDING_TOKEN,
        reference_target_token,
        gpu_target_token: gpu.rust_target_token,
        exact_frozen_result_reproduced: reference_preceding_token == RUST_PRECEDING_TOKEN
            && gpu.rust_preceding_token == RUST_PRECEDING_TOKEN
            && reference_target_token == RUST_REFERENCE_TARGET_TOKEN
            && gpu.rust_target_token == RUST_GPU_TARGET_TOKEN,
    };
    if !reproducibility.exact_frozen_result_reproduced {
        return Err("rust token target did not reproduce the frozen V2 result".to_string());
    }
    let gpu_pre_final = gpu
        .rust_target_trace
        .layer_post_moe
        .get(47)
        .ok_or("GPU full trace omitted layer-47 post-MoE hidden")?;
    let cpu_final_norm_on_gpu_pre = runtime.model.diagnostic_final_rms_norm(gpu_pre_final);
    let cpu_logits_on_gpu_final = runtime
        .model
        .diagnostic_greedy_logits(&gpu.rust_target_trace.final_norm);
    let cpu_logits_on_cpu_norm_of_gpu_pre = runtime
        .model
        .diagnostic_greedy_logits(&cpu_final_norm_on_gpu_pre);
    let reference_argmax = runtime.model.sample_hidden(
        &reference_trace.final_norm,
        &crate::sampling::SamplingParams::greedy(),
        model_position,
    );
    let cpu_argmax_on_actual_gpu_final_norm = runtime.model.sample_hidden(
        &gpu.rust_target_trace.final_norm,
        &crate::sampling::SamplingParams::greedy(),
        model_position,
    );
    let cpu_argmax_on_cpu_norm_of_gpu_pre_norm = runtime.model.sample_hidden(
        &cpu_final_norm_on_gpu_pre,
        &crate::sampling::SamplingParams::greedy(),
        model_position,
    );
    let cpu_argmax_from_actual_gpu_logits = deterministic_argmax(&gpu.rust_target_trace.logits)?;
    let inputs = TokenAttributionInputs {
        reference_argmax,
        cpu_argmax_on_cpu_norm_of_gpu_pre_norm,
        cpu_argmax_on_actual_gpu_final_norm,
        cpu_argmax_from_actual_gpu_logits,
        actual_gpu_sampled_token: gpu.rust_target_trace.sampled_token,
    };
    Ok(TokenDivergenceAttributionEvidence {
        reproducibility,
        full_target_token_trace: compare_target_token_traces(
            reference_trace,
            &gpu.rust_target_trace,
        )?,
        same_input_final_rmsnorm: VectorNumericalEvidence::compare(
            "cpu-production-final-rmsnorm-on-exact-gpu-layer47-post-moe",
            "actual-production-gpu-final-rmsnorm",
            &cpu_final_norm_on_gpu_pre,
            &gpu.rust_target_trace.final_norm,
        )?,
        same_input_lm_head_logits: VectorNumericalEvidence::compare(
            "cpu-production-greedy-row-dot-logits-on-exact-gpu-finalnorm",
            "actual-production-gpu-lm-head-logits",
            &cpu_logits_on_gpu_final,
            &gpu.rust_target_trace.logits,
        )?,
        cpu_finalnorm_replay_lm_head_vs_gpu_logits: VectorNumericalEvidence::compare(
            "cpu-production-lm-head-on-cpu-finalnorm-of-exact-gpu-pre-final-hidden",
            "actual-production-gpu-lm-head-logits",
            &cpu_logits_on_cpu_norm_of_gpu_pre,
            &gpu.rust_target_trace.logits,
        )?,
        greedy_views: vec![
            greedy_view(
                "cpu-production-greedy-argmax-on-reference-finalnorm",
                &reference_trace.logits,
                reference_argmax,
            )?,
            greedy_view(
                "cpu-production-greedy-argmax-on-cpu-finalnorm-of-exact-gpu-pre-final-hidden",
                &cpu_logits_on_cpu_norm_of_gpu_pre,
                cpu_argmax_on_cpu_norm_of_gpu_pre_norm,
            )?,
            greedy_view(
                "cpu-production-greedy-argmax-on-exact-gpu-finalnorm",
                &cpu_logits_on_gpu_final,
                cpu_argmax_on_actual_gpu_final_norm,
            )?,
            greedy_view(
                "cpu-deterministic-argmax-derived-from-exact-gpu-raw-logits",
                &gpu.rust_target_trace.logits,
                cpu_argmax_from_actual_gpu_logits,
            )?,
            greedy_view(
                "actual-gpu-production-sampled-token",
                &gpu.rust_target_trace.logits,
                gpu.rust_target_trace.sampled_token,
            )?,
        ],
        attribution: derive_token_attribution(inputs),
        positions_after_target_inspected_for_cpu_gpu_semantic_attribution: false,
    })
}

fn router_top_k_from_cpu_evaluation(
    source: &'static str,
    evaluation: &RouterEvaluationEvidence,
) -> RouterTopKViewEvidence {
    RouterTopKViewEvidence {
        source,
        top_8_ids: evaluation.top_8_ids.clone(),
        top_8_weights: evaluation.top_8_weights.clone(),
        top_12: evaluation.top_12_ranked_experts.clone(),
    }
}

fn analyze_router_failure(
    gate: &crate::gating::LinearGate,
    gpu_gate_identity: &GateTensorIdentity,
    gpu_layer: &SemanticCorpusGpuLayerTrace,
) -> Result<RouterFailureAttributionEvidence, String> {
    let cpu_production = crate::gpu_native_router_rank_diagnostics::evaluate_cpu_router(
        "cpu-production-router-on-exact-gpu-input",
        gate,
        &gpu_layer.router_input,
    )?;
    let cpu_raw = gate.weights.matvec(&gpu_layer.router_input);
    let scalar_raw = gate
        .weights
        .diagnostic_greedy_logits(&gpu_layer.router_input);
    let actual_gpu = crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
        gpu_layer.raw_logits.clone(),
        gpu_layer.selected_ids.clone(),
        gpu_layer.selected_weights.clone(),
    )?;
    let cpu_top_k = router_top_k_from_cpu_evaluation(
        "cpu-production-router-top-k-on-exact-gpu-input",
        &cpu_production,
    );
    let scalar_top_k = top_k_from_raw_logits(
        "scalar-sequential-row-dot-softmax-top-k",
        &scalar_raw,
        gate.top_k,
    )?;
    let gpu_raw_top_k = RouterTopKViewEvidence {
        source: "cpu-derived-softmax-top-k-from-exact-gpu-raw-logits",
        top_8_ids: actual_gpu
            .cpu_top_8_ids_derived_from_exact_gpu_raw_logits
            .clone(),
        top_8_weights: actual_gpu
            .cpu_top_8_weights_derived_from_exact_gpu_raw_logits
            .clone(),
        top_12: actual_gpu.top_12_ranked_experts.clone(),
    };
    let actual_gpu_top_k = RouterTopKViewEvidence {
        source: "actual-gpu-production-top8-with-cpu-ranked-top12-from-actual-raw",
        top_8_ids: actual_gpu.top_8_ids.clone(),
        top_8_weights: actual_gpu.top_8_weights.clone(),
        top_12: actual_gpu.top_12_ranked_experts.clone(),
    };
    let mut gpu_scores = gpu_layer.raw_logits.clone();
    crate::transformer::softmax_inplace(&mut gpu_scores);
    let exact_tie = gpu_layer.raw_logits[95].to_bits() == gpu_layer.raw_logits[107].to_bits()
        && gpu_scores[95].to_bits() == gpu_scores[107].to_bits();
    let lower_id_tie_break_violated = exact_tie
        && match (
            actual_gpu_top_k.top_8_ids.iter().position(|id| *id == 95),
            actual_gpu_top_k.top_8_ids.iter().position(|id| *id == 107),
        ) {
            (Some(lower_rank), Some(higher_rank)) => higher_rank < lower_rank,
            (None, Some(_)) => true,
            _ => false,
        };
    let gate_identity = GateIdentityEvidence::new(
        GateTensorIdentity::from_gate(ROUTER_TARGET_LAYER, gate),
        gpu_gate_identity.clone(),
    );
    if !gate_identity.exact_identity_match {
        return Err("router target gate tensor identity differs between runtimes".to_string());
    }
    let actual_gpu_ids = actual_gpu_top_k.top_8_ids.clone();
    Ok(RouterFailureAttributionEvidence {
        case: ROUTER_TARGET_CASE,
        generated_position: ROUTER_TARGET_POSITION,
        layer: ROUTER_TARGET_LAYER,
        exact_gpu_router_input: ExactVectorEvidence::new(
            "exact-gpu-layer38-router-input",
            &gpu_layer.router_input,
            true,
        ),
        gate_identity,
        cpu_production_router: cpu_production,
        actual_gpu_router: actual_gpu,
        cpu_production_raw_logits: raw_logit_view("cpu-production-router-raw-logits", &cpu_raw)?,
        scalar_sequential_raw_logits: raw_logit_view(
            "diagnostic-scalar-sequential-f32-row-dot-raw-logits",
            &scalar_raw,
        )?,
        actual_gpu_raw_logits: raw_logit_view(
            "actual-gpu-production-router-raw-logits",
            &gpu_layer.raw_logits,
        )?,
        cpu_production_vs_gpu: compare_raw_logits(
            "cpu-production-router-raw-logits",
            "actual-gpu-production-router-raw-logits",
            &cpu_raw,
            &gpu_layer.raw_logits,
        )?,
        scalar_sequential_vs_gpu: compare_raw_logits(
            "diagnostic-scalar-sequential-f32-row-dot-raw-logits",
            "actual-gpu-production-router-raw-logits",
            &scalar_raw,
            &gpu_layer.raw_logits,
        )?,
        cpu_production_vs_scalar_sequential: compare_raw_logits(
            "cpu-production-router-raw-logits",
            "diagnostic-scalar-sequential-f32-row-dot-raw-logits",
            &cpu_raw,
            &scalar_raw,
        )?,
        frozen_cpu_ids_reproduced: cpu_top_k.top_8_ids == ROUTER_CPU_EXPECTED_IDS,
        frozen_gpu_ids_reproduced: actual_gpu_top_k.top_8_ids == ROUTER_GPU_EXPECTED_IDS,
        attribution: derive_router_attribution(
            &cpu_top_k.top_8_ids,
            &gpu_raw_top_k.top_8_ids,
            &actual_gpu_ids,
            lower_id_tie_break_violated,
        ),
        cpu_production_top_k: cpu_top_k,
        scalar_sequential_top_k: scalar_top_k,
        cpu_top_k_from_exact_gpu_raw_logits: gpu_raw_top_k,
        actual_gpu_production_top_k: actual_gpu_top_k,
        exact_gpu_raw_and_scored_tie_95_107: exact_tie,
    })
}

async fn analyze_moe_outlier(
    runtime: &crate::BenchRealRuntime,
    target: FrozenMoeOutlierTarget,
    gpu_layer: &SemanticCorpusGpuLayerTrace,
) -> Result<MoeOutlierAttributionEvidence, Box<dyn std::error::Error>> {
    let gate = &runtime.model.layers[target.layer].gate;
    let cpu_route = gate.route(&gpu_layer.router_input);
    if !exact_ordered_router_ids_equal(&cpu_route.experts, &gpu_layer.selected_ids) {
        return Ok(MoeOutlierAttributionEvidence {
            target,
            same_input_cpu_ordered_selected_ids: cpu_route.experts,
            actual_gpu_ordered_selected_ids: gpu_layer.selected_ids.clone(),
            valid_for_exact_router_decomposition: false,
            invalid_reason: Some(
                "same-input CPU production ordered IDs differ from actual GPU ordered IDs"
                    .to_string(),
            ),
            per_expert_in_production_rank_order: Vec::new(),
            expert_ids_ranked_by_difference_contribution: Vec::new(),
            worst_expert_id: None,
            mixed_combination_decomposition: None,
            permutation_only_supporting_evidence: None,
        });
    }
    if gpu_layer.route_outputs.len() != cpu_route.experts.len()
        || gpu_layer.selected_weights.len() != cpu_route.experts.len()
    {
        return Err("GPU target route-output geometry is incomplete".into());
    }
    let global_ids = cpu_route
        .experts
        .iter()
        .map(|expert| runtime.model.global_expert_id(target.layer, *expert))
        .collect::<Vec<_>>();
    let case_index = crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS
        .iter()
        .position(|case| case.name == target.case)
        .ok_or("MoE target case is not in frozen holdout")?;
    let token_index = (case_index as u64)
        .wrapping_mul(crate::gpu_native_semantic_parity_v2::OUTPUT_TOKEN_LIMIT as u64)
        .wrapping_add(target.generated_position as u64)
        .wrapping_mul(48)
        .wrapping_add(target.layer as u64);
    let cpu_outputs = runtime
        .engine
        .moe_step_with_timing(
            token_index,
            target.layer as u32,
            &gpu_layer.router_input,
            &global_ids,
            None,
        )
        .await?;
    if cpu_outputs.len() != cpu_route.experts.len()
        || cpu_outputs.iter().any(|output| output.len() != 2048)
    {
        return Err("CPU MoE target expert-output geometry is incomplete".into());
    }
    let paired_gpu_ranks =
        gpu_ranks_paired_by_expert_id(&cpu_route.experts, &gpu_layer.selected_ids)?;
    let mut per_expert = Vec::with_capacity(cpu_route.experts.len());
    for (rank, expert_id) in cpu_route.experts.iter().copied().enumerate() {
        let gpu_rank = paired_gpu_ranks[rank];
        let cpu_output = &cpu_outputs[rank];
        let gpu_output = &gpu_layer.route_outputs[gpu_rank];
        let cpu_weight = cpu_route.weights[rank];
        let gpu_weight = gpu_layer.selected_weights[gpu_rank];
        let expert_output_comparison = VectorNumericalEvidence::compare(
            "cpu-production-expert-output-on-exact-gpu-input",
            "actual-production-gpu-expert-output-on-exact-gpu-input",
            cpu_output,
            gpu_output,
        )?;
        let weighted_contribution_comparison = VectorNumericalEvidence::compare(
            "cpu-expert-output-times-cpu-same-input-weight",
            "gpu-expert-output-times-gpu-actual-weight",
            &weighted_output(cpu_output, cpu_weight),
            &weighted_output(gpu_output, gpu_weight),
        )?;
        per_expert.push(PerExpertMoeEvidence {
            production_rank: rank + 1,
            local_expert_id: expert_id,
            global_expert_id: global_ids[rank],
            weight_delta: weight_delta(expert_id, cpu_weight, gpu_weight),
            cpu_expert_output_f32_bits_sha256: expert_output_comparison
                .left_f32_bits_sha256
                .clone(),
            gpu_expert_output_f32_bits_sha256: expert_output_comparison
                .right_f32_bits_sha256
                .clone(),
            expert_output_comparison,
            weighted_contribution_comparison,
        });
    }
    let mut ranked_by_contribution = per_expert.iter().collect::<Vec<_>>();
    ranked_by_contribution.sort_by(|left, right| {
        right
            .weighted_contribution_comparison
            .max_absolute_error
            .unwrap_or(f64::NEG_INFINITY)
            .total_cmp(
                &left
                    .weighted_contribution_comparison
                    .max_absolute_error
                    .unwrap_or(f64::NEG_INFINITY),
            )
            .then_with(|| left.local_expert_id.cmp(&right.local_expert_id))
    });
    let contribution_ranking = ranked_by_contribution
        .iter()
        .map(|expert| expert.local_expert_id)
        .collect::<Vec<_>>();
    let worst_expert_id = contribution_ranking.first().copied();

    let baseline = crate::inference::combine_outputs(&cpu_outputs, &cpu_route.weights);
    let actual =
        crate::inference::combine_outputs(&gpu_layer.route_outputs, &gpu_layer.selected_weights);
    let expert_only =
        crate::inference::combine_outputs(&gpu_layer.route_outputs, &cpu_route.weights);
    let weight_only = crate::inference::combine_outputs(&cpu_outputs, &gpu_layer.selected_weights);
    let combinations = MixedCombinationDecompositionEvidence {
        ordered_selected_expert_ids: cpu_route.experts.clone(),
        accumulation_contract:
            "production sequential f32 route order: destination += weight[rank] * expert_output[rank]",
        baseline_vs_actual: VectorNumericalEvidence::compare(
            "baseline-cpu-outputs-times-cpu-weights",
            "actual-gpu-outputs-times-gpu-weights-host-replay",
            &baseline,
            &actual,
        )?,
        baseline_vs_expert_only: VectorNumericalEvidence::compare(
            "baseline-cpu-outputs-times-cpu-weights",
            "expert-only-gpu-outputs-times-cpu-weights",
            &baseline,
            &expert_only,
        )?,
        baseline_vs_weight_only: VectorNumericalEvidence::compare(
            "baseline-cpu-outputs-times-cpu-weights",
            "weight-only-cpu-outputs-times-gpu-weights",
            &baseline,
            &weight_only,
        )?,
        expert_only_vs_actual: VectorNumericalEvidence::compare(
            "expert-only-gpu-outputs-times-cpu-weights",
            "actual-gpu-outputs-times-gpu-weights-host-replay",
            &expert_only,
            &actual,
        )?,
        weight_only_vs_actual: VectorNumericalEvidence::compare(
            "weight-only-cpu-outputs-times-gpu-weights",
            "actual-gpu-outputs-times-gpu-weights-host-replay",
            &weight_only,
            &actual,
        )?,
        actual_host_combination_vs_actual_gpu_production: VectorNumericalEvidence::compare(
            "actual-gpu-outputs-times-gpu-weights-host-replay",
            "actual-production-gpu-routed-moe-output",
            &actual,
            &gpu_layer.routed_moe_output,
        )?,
        linear_additivity_claimed: false,
    };
    let permutation_only_supporting_evidence = (classify_target(target.classification)
        == RoutingClassification::InternalRankPermutation)
        .then(|| {
            crate::gpu_native_expert_permutation_semantic_parity::permutation_only_witness(
                &gpu_layer.selected_ids,
                &gpu_layer.selected_weights,
                &gpu_layer.route_outputs,
            )
        })
        .transpose()?;
    Ok(MoeOutlierAttributionEvidence {
        target,
        same_input_cpu_ordered_selected_ids: cpu_route.experts,
        actual_gpu_ordered_selected_ids: gpu_layer.selected_ids.clone(),
        valid_for_exact_router_decomposition: true,
        invalid_reason: None,
        per_expert_in_production_rank_order: per_expert,
        expert_ids_ranked_by_difference_contribution: contribution_ranking,
        worst_expert_id,
        mixed_combination_decomposition: Some(combinations),
        permutation_only_supporting_evidence,
    })
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
            return Err("diagnostic reference resolved configuration identity drifted".into());
        }
        if runtime.model.layers.len() != gpu.model_geometry.num_layers {
            return Err("diagnostic reference layer geometry differs from GPU".into());
        }
        for layer in [ROUTER_TARGET_LAYER, 47] {
            let reference_identity =
                GateTensorIdentity::from_gate(layer, &runtime.model.layers[layer].gate);
            if gpu.gate_identities.get(&layer) != Some(&reference_identity) {
                return Err(format!("diagnostic gate tensor identity differs at layer {layer}").into());
            }
        }
        let boundary_before = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_before.enabled || boundary_before.routed_expert_dispatches != 0 {
            return Err("diagnostic CPU boundary emulation did not start clean".into());
        }
        let model_load = crate::greedy_parity_model_load(&runtime);
        let fixed = holdout_case(RUST_TARGET_CASE)?;
        let prompt_token_ids = tokenizer.encode(fixed.prompt)?;
        if prompt_token_ids.is_empty() {
            return Err("reference rust holdout prompt encoded to zero tokens".into());
        }
        let (reference_preceding_token, reference_trace) = crate::with_progress_timeout(
            "v2 holdout failure attribution CPU rust token".to_string(),
            watchdog,
            async {
                let mut kv = runtime.model.fresh_kv_caches();
                let prefix_count = prompt_token_ids.len().saturating_sub(1);
                for (position, &token_id) in prompt_token_ids[..prefix_count].iter().enumerate() {
                    runtime
                        .model
                        .forward_token_hidden(&runtime.engine, token_id, position, &mut kv)
                        .await?;
                }
                let final_prompt = *prompt_token_ids
                    .last()
                    .ok_or("reference rust holdout prompt is empty")?;
                let hidden = runtime
                    .model
                    .forward_token_hidden(&runtime.engine, final_prompt, prefix_count, &mut kv)
                    .await?;
                let preceding = runtime.model.sample_hidden(
                    &hidden,
                    &crate::sampling::SamplingParams::greedy(),
                    prefix_count,
                );
                let trace = runtime
                    .model
                    .forward_token_diagnostic_trace(
                        &runtime.engine,
                        preceding,
                        prefix_count + 1,
                        &mut kv,
                        None,
                    )
                    .await?;
                Ok::<_, Box<dyn std::error::Error>>((preceding, trace))
            },
        )
        .await?;
        let reference_target_token = reference_trace.sampled_token;
        let frozen_prefix_reproduced = reference_preceding_token == RUST_PRECEDING_TOKEN
            && gpu.rust_preceding_token == RUST_PRECEDING_TOKEN
            && reference_target_token == RUST_REFERENCE_TARGET_TOKEN
            && gpu.rust_target_token == RUST_GPU_TARGET_TOKEN;
        if !frozen_prefix_reproduced {
            return Ok::<_, Box<dyn std::error::Error>>(ReferenceCapture {
                token_attribution: None,
                router_attribution: None,
                moe_attributions: Vec::new(),
                failure: Some(format!(
                    "frozen rust target reproducibility failure: reference=({reference_preceding_token},{reference_target_token}) gpu=({},{}) expected=({RUST_PRECEDING_TOKEN},{RUST_REFERENCE_TARGET_TOKEN})/({RUST_PRECEDING_TOKEN},{RUST_GPU_TARGET_TOKEN})",
                    gpu.rust_preceding_token, gpu.rust_target_token
                )),
                model_load,
                background_shutdown:
                    crate::greedy_parity::BackgroundShutdownEvidence::default(),
            });
        }
        let token_attribution = analyze_token_divergence(
            &runtime,
            &reference_trace,
            gpu,
            reference_preceding_token,
            reference_target_token,
            prompt_token_ids.len(),
        )?;
        let router_layer = gpu.semantic_layer(
            ROUTER_TARGET_CASE,
            ROUTER_TARGET_POSITION,
            ROUTER_TARGET_LAYER,
        )?;
        let router_attribution = analyze_router_failure(
            &runtime.model.layers[ROUTER_TARGET_LAYER].gate,
            gpu.gate_identities
                .get(&ROUTER_TARGET_LAYER)
                .ok_or("GPU router target gate identity is missing")?,
            router_layer,
        )?;

        let mut moe_attributions = Vec::with_capacity(MOE_OUTLIER_TARGETS.len());
        for target in MOE_OUTLIER_TARGETS {
            let gpu_layer = gpu.semantic_layer(
                target.case,
                target.generated_position,
                target.layer,
            )?;
            moe_attributions.push(analyze_moe_outlier(&runtime, target, gpu_layer).await?);
        }
        let mut failures = Vec::new();
        if !router_attribution.frozen_cpu_ids_reproduced {
            failures.push("router target CPU same-input IDs did not reproduce frozen evidence");
        }
        if !router_attribution.frozen_gpu_ids_reproduced {
            failures.push("router target actual GPU IDs did not reproduce frozen evidence");
        }
        if moe_attributions
            .iter()
            .any(|target| !target.valid_for_exact_router_decomposition)
        {
            failures.push("one or more exact-router MoE targets failed same-input ordered-ID validation");
        }
        let boundary_after = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_after.enabled || boundary_after.routed_expert_dispatches == 0 {
            failures.push("diagnostic CPU boundary emulation was not exercised");
        }
        Ok::<_, Box<dyn std::error::Error>>(ReferenceCapture {
            token_attribution: Some(token_attribution),
            router_attribution: Some(router_attribution),
            moe_attributions,
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
            "{error}; diagnostic reference shutdown also failed: {shutdown_error}"
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
    report: &GpuNativeV2HoldoutFailureAttributionReport,
    report_out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if report.qualification_pass() || !report.diagnostic_only {
        return Err("diagnostic report must remain diagnostic-only and qualification false".into());
    }
    if let Some(parent) = report_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    std::fs::write(report_out, json)?;
    eprintln!(
        "GPU-native V2 holdout failure-attribution report written to {}",
        report_out.display()
    );
    Ok(())
}

pub async fn run_diagnostic(
    config: PathBuf,
    cfg: crate::config::Config,
    expected_adapter_name: String,
    v2_report: PathBuf,
    expected_v2_report_sha256: String,
    report_out: PathBuf,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::BuildProvenance;

    // This immutable identity check deliberately occurs before any runtime or
    // hardware construction.
    let frozen_v2_result_identity =
        validate_frozen_v2_report(&v2_report, &expected_v2_report_sha256)?;
    let build = BuildProvenance::embedded();
    if progress_watchdog.timeout.is_none() {
        return Err("V2 holdout diagnostic requires a positive progress timeout".into());
    }
    if build.dirty != Some(false) || build.git_sha.as_deref().is_none_or(|sha| !is_hex(sha, 40)) {
        return Err("V2 holdout diagnostic requires clean embedded Git provenance".into());
    }
    let (artifacts, artifact_errors) = crate::qualification_artifacts(&config, &cfg);
    if !artifact_errors.is_empty() {
        return Err(format!(
            "V2 holdout diagnostic artifact preflight failed: {}",
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
        return Err("V2 holdout diagnostic requires the strict GPU-native Q4 configuration".into());
    }
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| format!("diagnostic expert metadata preflight failed: {error}"))?;
    if expert_metadata.dtype.as_deref() != Some("q4_0")
        || expert_metadata.q4_0_layout.as_deref() != Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
        || expert_metadata.explicitly_synthetic
    {
        return Err("V2 holdout diagnostic requires canonical nonsynthetic Q4_0 metadata".into());
    }
    if expected_adapter_name.trim().is_empty() {
        return Err("V2 holdout diagnostic requires a nonempty exact adapter name".into());
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
            "V2 holdout diagnostic requires exact Qwen3-Coder 30B-A3B Q4_0 geometry".into(),
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
        .ok_or("V2 holdout diagnostic requires tokenizer.path")?;
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
        failures.push("one or both diagnostic runtimes did not load the strict model".to_string());
    }
    if runtime.gpu_token_loop.tokens_completed != runtime.gpu_expected_completed_token_steps
        || runtime.gpu_token_loop.fatal_failures != 0
        || runtime.gpu_token_loop.no_progress_failures != 0
    {
        failures.push("diagnostic GPU execution did not complete cleanly".to_string());
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
        failures.push("one or both diagnostic runtimes failed controlled shutdown".to_string());
    }
    let failure = (!failures.is_empty()).then(|| failures.join("; "));
    let report = GpuNativeV2HoldoutFailureAttributionReport::new(
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
        frozen_v2_result_identity,
        failure,
        reference.token_attribution,
        reference.router_attribution,
        reference.moe_attributions,
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
            "V2 holdout failure attribution incomplete: {}",
            failure.unwrap_or_else(|| "required evidence is incomplete".to_string())
        )
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strict_model_load() -> crate::greedy_parity::ModelLoadEvidence {
        crate::greedy_parity::ModelLoadEvidence {
            strict: true,
            loader: "safetensors".to_string(),
            loaded_tensors: 1,
            required_tensors: 1,
            optional_probed: 0,
            optional_loaded: 0,
            seeded_fallback_remained: false,
        }
    }

    fn released_shutdown() -> crate::greedy_parity::BackgroundShutdownEvidence {
        crate::greedy_parity::BackgroundShutdownEvidence {
            controlled_shutdown_requested: true,
            all_runtime_resources_released: true,
            poll_iterations: 1,
        }
    }

    fn provenance() -> DiagnosticProvenance {
        DiagnosticProvenance {
            build: crate::qualification::BuildProvenance {
                git_sha: Some("a".repeat(40)),
                dirty: Some(false),
                package_version: "0.1.0".to_string(),
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
            reference_model_load: strict_model_load(),
            gpu_native_model_load: strict_model_load(),
            reference_background_shutdown: released_shutdown(),
            gpu_native_background_shutdown: released_shutdown(),
            expert_metadata: crate::qualification::ExpertMetadataEvidence {
                dtype: Some("q4_0".to_string()),
                q4_0_layout: Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1.to_string()),
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
                driver: "580.173.02".to_string(),
                driver_info: "test".to_string(),
                compute_plane: "wgpu-vulkan".to_string(),
                software_adapter: false,
            },
        }
    }

    fn frozen_identity() -> FrozenV2ResultIdentity {
        FrozenV2ResultIdentity {
            report_artifact: crate::qualification::ArtifactDigest {
                configured_path: "/v2.json".to_string(),
                canonical_path: "/v2.json".to_string(),
                byte_length: 1,
                sha256: FROZEN_V2_REPORT_SHA256.to_string(),
            },
            expected_report_sha256_argument: FROZEN_V2_REPORT_SHA256.to_string(),
            expected_schema: crate::gpu_native_semantic_parity_v2::SCHEMA_VERSION,
            frozen_build_sha: FROZEN_V2_BUILD_SHA,
            qualification_pass: false,
            holdout_corpus_id: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_ID,
            holdout_corpus_sha256: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_SHA256,
            numerical_limits: crate::gpu_native_semantic_parity_v2::NumericalLimits::frozen(),
            execution_log_sha256: FROZEN_V2_EXECUTION_LOG_SHA256,
            offline_failure_analysis_sha256: FROZEN_OFFLINE_ANALYSIS_SHA256,
            immutable_input_verified: true,
        }
    }

    #[test]
    fn schema_mode_and_diagnostic_only_contract_are_versioned() {
        assert_eq!(
            SCHEMA_VERSION,
            "mer.gpu-native-v2-holdout-failure-attribution.v1"
        );
        assert_eq!(MODE, "diagnose-gpu-native-v2-holdout-failures");
        let report = GpuNativeV2HoldoutFailureAttributionReport::new(
            provenance(),
            frozen_identity(),
            Some("fixture incomplete".to_string()),
            None,
            None,
            Vec::new(),
            DiagnosticRuntimeEvidence::default(),
            Vec::new(),
        );
        assert!(!report.qualification_pass());
        let value = serde_json::to_value(report).unwrap();
        assert_eq!(value["qualification_pass"], false);
        assert_eq!(value["diagnostic_only"], true);
    }

    #[test]
    fn frozen_v2_report_sha_argument_is_fail_closed() {
        assert!(validate_expected_v2_report_sha_argument(FROZEN_V2_REPORT_SHA256).is_ok());
        assert!(validate_expected_v2_report_sha_argument(&"0".repeat(64)).is_err());
        assert!(validate_expected_v2_report_sha_argument("not-hex").is_err());
    }

    #[test]
    fn exact_v2_schema_build_corpus_and_limits_are_required() {
        let exact = || FrozenV2ReportEnvelope {
            schema: crate::gpu_native_semantic_parity_v2::SCHEMA_VERSION.to_string(),
            qualification_pass: false,
            provenance: FrozenProvenanceEnvelope {
                build: FrozenBuildEnvelope {
                    git_sha: Some(FROZEN_V2_BUILD_SHA.to_string()),
                },
            },
            holdout_corpus: FrozenCorpusEnvelope {
                id: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_ID.to_string(),
                sha256: crate::gpu_native_semantic_parity_v2::HOLDOUT_CORPUS_SHA256.to_string(),
            },
            numerical_limits: FrozenNumericalLimits {
                max_absolute_error_limit: 0.020,
                rms_error_limit: 0.001,
                mean_absolute_error_limit: 0.00075,
                nonfinite_mismatch_limit: 0,
                semantic_correctness_not_bit_parity: true,
            },
        };
        assert!(validate_v2_envelope(&exact()).is_ok());
        let mut wrong = exact();
        wrong.schema.push_str("-drift");
        assert!(validate_v2_envelope(&wrong).is_err());
        let mut wrong = exact();
        wrong.provenance.build.git_sha = Some("0".repeat(40));
        assert!(validate_v2_envelope(&wrong).is_err());
        let mut wrong = exact();
        wrong.holdout_corpus.sha256 = "0".repeat(64);
        assert!(validate_v2_envelope(&wrong).is_err());
        let mut wrong = exact();
        wrong.numerical_limits.max_absolute_error_limit = 0.0200001;
        assert!(validate_v2_envelope(&wrong).is_err());
    }

    #[test]
    fn target_list_is_exactly_one_token_one_router_and_four_moe_outliers() {
        let targets = FrozenTargetListEvidence::fixed();
        assert_eq!(targets.rust_token_target_case, RUST_TARGET_CASE);
        assert_eq!(targets.rust_token_target_position, 1);
        assert_eq!(targets.rust_preceding_token, 4710);
        assert_eq!(targets.rust_reference_target_token, 785);
        assert_eq!(targets.rust_gpu_target_token, 8822);
        assert_eq!(targets.router_target_position, 11);
        assert_eq!(targets.router_target_layer, 38);
        assert_eq!(targets.router_cpu_expected_ids, ROUTER_CPU_EXPECTED_IDS);
        assert_eq!(targets.router_gpu_expected_ids, ROUTER_GPU_EXPECTED_IDS);
        assert_eq!(targets.exact_router_moe_outliers, MOE_OUTLIER_TARGETS);
        assert_eq!(targets.exact_router_moe_outliers.len(), 4);
    }

    #[test]
    fn token_attribution_categories_are_stage_specific_and_conservative() {
        let base = TokenAttributionInputs {
            reference_argmax: 785,
            cpu_argmax_on_cpu_norm_of_gpu_pre_norm: 8822,
            cpu_argmax_on_actual_gpu_final_norm: 8822,
            cpu_argmax_from_actual_gpu_logits: 8822,
            actual_gpu_sampled_token: 8822,
        };
        assert_eq!(
            derive_token_attribution(base),
            TokenAttributionCategory::UpstreamHiddenDrift
        );
        assert_eq!(
            derive_token_attribution(TokenAttributionInputs {
                cpu_argmax_on_cpu_norm_of_gpu_pre_norm: 785,
                ..base
            }),
            TokenAttributionCategory::FinalRmsnormDrift
        );
        assert_eq!(
            derive_token_attribution(TokenAttributionInputs {
                cpu_argmax_on_actual_gpu_final_norm: 785,
                ..base
            }),
            TokenAttributionCategory::GpuLmHeadGemvDrift
        );
        assert_eq!(
            derive_token_attribution(TokenAttributionInputs {
                cpu_argmax_from_actual_gpu_logits: 785,
                ..base
            }),
            TokenAttributionCategory::GpuGreedyArgmaxDrift
        );
        assert_eq!(
            derive_token_attribution(TokenAttributionInputs {
                cpu_argmax_on_cpu_norm_of_gpu_pre_norm: 3333,
                ..base
            }),
            TokenAttributionCategory::MixedNumericalDrift
        );
        assert_eq!(
            derive_token_attribution(TokenAttributionInputs {
                reference_argmax: 999,
                ..base
            }),
            TokenAttributionCategory::Unresolved
        );
    }

    #[test]
    fn cpu_lm_head_contract_names_greedy_compatible_row_dot() {
        let semantics = ProductionSemanticsEvidence::default();
        assert!(semantics
            .cpu_lm_head_contract
            .contains("DenseWeight::diagnostic_greedy_logits"));
        assert!(semantics.cpu_lm_head_contract.contains("row_dot"));
    }

    #[test]
    fn router_attribution_distinguishes_gemv_topk_tie_and_unresolved() {
        let cpu = [1, 2];
        let gpu = [1, 3];
        assert_eq!(
            derive_router_attribution(&cpu, &gpu, &gpu, false),
            RouterAttributionCategory::GpuRouterGemvDrift
        );
        assert_eq!(
            derive_router_attribution(&cpu, &cpu, &gpu, false),
            RouterAttributionCategory::GpuRouterSoftmaxTopkDrift
        );
        assert_eq!(
            derive_router_attribution(&cpu, &cpu, &gpu, true),
            RouterAttributionCategory::ExactTieOrderingDefect
        );
        assert_eq!(
            derive_router_attribution(&[1], &cpu, &cpu, false),
            RouterAttributionCategory::MixedNumericalDrift
        );
        assert_eq!(
            derive_router_attribution(&cpu, &cpu, &cpu, false),
            RouterAttributionCategory::Unresolved
        );
    }

    #[test]
    fn exact_gpu_raw_logit_topk_is_derived_without_reference_hidden() {
        let mut logits = vec![-100.0; 128];
        for (rank, expert) in ROUTER_GPU_EXPECTED_IDS.iter().enumerate() {
            logits[*expert as usize] = 10.0 - rank as f32;
        }
        let actual = crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
            logits,
            ROUTER_GPU_EXPECTED_IDS.to_vec(),
            vec![0.125; 8],
        )
        .unwrap();
        assert_eq!(
            actual.cpu_top_8_ids_derived_from_exact_gpu_raw_logits,
            ROUTER_GPU_EXPECTED_IDS
        );
    }

    #[test]
    fn router_raw_views_retain_experts_95_and_107_and_full_bits() {
        let mut logits = vec![0.0; 128];
        logits[95] = -3.8356452;
        logits[107] = -3.8356473;
        let view = raw_logit_view("gpu", &logits).unwrap();
        assert_eq!(view.exact_vector.f32_bits.len(), 128);
        assert_eq!(
            view.experts_95_and_107
                .iter()
                .map(|expert| expert.expert_id)
                .collect::<Vec<_>>(),
            vec![95, 107]
        );
    }

    #[test]
    fn exact_router_moe_decomposition_refuses_unequal_ordered_ids() {
        assert!(exact_ordered_router_ids_equal(&[1, 2], &[1, 2]));
        assert!(!exact_ordered_router_ids_equal(&[1, 2], &[2, 1]));
        assert!(!exact_ordered_router_ids_equal(&[1, 2], &[1, 3]));
    }

    #[test]
    fn per_expert_pairing_uses_expert_id_not_rank() {
        assert_eq!(
            gpu_ranks_paired_by_expert_id(&[7, 3, 9], &[9, 7, 3]).unwrap(),
            vec![1, 2, 0]
        );
        assert!(gpu_ranks_paired_by_expert_id(&[7, 3], &[7, 9]).is_err());
    }

    #[test]
    fn mixed_combination_uses_production_sequential_f32_order() {
        let outputs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let weights = vec![0.5, 0.25, 0.125];
        let actual = crate::inference::combine_outputs(&outputs, &weights);
        let mut expected = vec![0.0f32; 2];
        for (output, weight) in outputs.iter().zip(&weights) {
            for (destination, value) in expected.iter_mut().zip(output) {
                *destination += *weight * *value;
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn all_four_outliers_and_router_coupled_event_remain_separate() {
        assert_eq!(MOE_OUTLIER_TARGETS.len(), 4);
        assert!(MOE_OUTLIER_TARGETS.iter().all(|target| target.layer == 47));
        assert_eq!(ROUTER_COUPLED_TARGET.layer, 38);
        assert!(ROUTER_COUPLED_TARGET.excluded_from_exact_router_moe_decomposition);
        assert!(MOE_OUTLIER_TARGETS.iter().all(|target| {
            (target.case, target.generated_position, target.layer)
                != (
                    ROUTER_COUPLED_TARGET.case,
                    ROUTER_COUPLED_TARGET.generated_position,
                    ROUTER_COUPLED_TARGET.layer,
                )
        }));
    }

    #[test]
    fn production_and_frozen_contracts_are_declared_unchanged() {
        let evidence = ProductionSemanticsEvidence::default();
        assert!(evidence.diagnostic_only);
        assert!(!evidence.production_inference_math_changed);
        assert!(!evidence.production_router_math_changed);
        assert!(!evidence.production_dense_gemv_changed);
        assert!(!evidence.shader_or_wgsl_changed);
        assert!(!evidence.v1_changed);
        assert!(!evidence.v2_changed);
        assert!(!evidence.frozen_limits_changed);
        assert!(!evidence.frozen_holdout_changed);
    }
}
