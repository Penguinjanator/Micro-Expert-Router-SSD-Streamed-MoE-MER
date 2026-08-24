//! Versioned GPU-native semantic correctness qualification over an independent
//! frozen holdout corpus.
//!
//! This module is qualification-only. It reuses the existing semantic survey's
//! observation types and the production execution paths, but owns an
//! independent corpus, report schema, and PASS derivation. It is never
//! consulted by ordinary inference or by the historical v1 qualifier.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::gpu_native_expert_permutation_semantic_parity::{
    VectorNumericalEvidence, WeightDeltaEvidence,
};
use crate::gpu_native_router_rank_diagnostics::{
    ActualGpuRouterEvidence, DiagnosticProvenance, RouterEvaluationEvidence,
};
use crate::gpu_native_semantic_parity_corpus::{
    aggregate_numerics, aggregate_numerics_by_classification, aggregate_permutation_numerics,
    canonical_selected_weight_evidence, rank_displacement, same_membership, BoundaryRouterView,
    ClassificationNumericalAggregation, DeterministicEventSampler, EventLocation,
    MembershipBoundaryEvidence, NumericalAggregation, PermutationNumericalEventEvidence,
    RankDisplacementEvidence, RouterSameInputEvidence, RoutingClassification, RoutingCounters,
    RoutingEventEvidence, SameInputNumericalEventEvidence, SamplingSummary, SemanticCorpusGpuTrace,
    SemanticCorpusTraceLayout, TokenCaseEvidence, TokenSummary,
};
use crate::gpu_native_token_loop::{GpuNativeModelGeometry, GpuNativeTokenLoopSnapshot};
use crate::qualification::QualificationStatus;
use crate::{IsolatedRuntimeShutdownError, RealCliRuntimeMode, ResolvedRealCliSpec};

pub const SCHEMA_VERSION: &str = "mer.gpu-native-semantic-parity.v2";
pub const MODE: &str = "qualify-gpu-native-semantic-parity-v2";

pub const CALIBRATION_CORPUS_ID: &str = "qwen3-coder-30b-a3b-greedy-v1";
pub const CALIBRATION_CORPUS_VERSION: u32 = 1;
pub const CALIBRATION_CORPUS_SHA256: &str =
    "ea7fdda4f08cde2fe3658165054d80099948f66ae8cba1c904ca41102f0aadc7";

pub const HOLDOUT_CORPUS_ID: &str = "qwen3-coder-30b-a3b-greedy-v2-holdout";
pub const HOLDOUT_CORPUS_VERSION: u32 = 1;
pub const HOLDOUT_CORPUS_CASE_COUNT: usize = 4;
pub const OUTPUT_TOKEN_LIMIT: usize = 16;
pub const HOLDOUT_CORPUS_SHA256: &str =
    "0a680d4e96937782cdb14d48e3609b095368cf69295e1bc794e6354c9eb6513d";

pub const MAX_ABSOLUTE_ERROR_LIMIT: f64 = 0.020;
pub const RMS_ERROR_LIMIT: f64 = 0.001;
pub const MEAN_ABSOLUTE_ERROR_LIMIT: f64 = 0.00075;
pub const NONFINITE_MISMATCH_LIMIT: usize = 0;
pub const EXACT_ORDER_SAMPLING_RULE: &str =
    "first-exact-order-event-per-layer-in-frozen-corpus-order";
const POST_GENERATED_TOKEN_DIVERGENCE_INVALID_REASON: &str =
    "post-generated-token-divergence-noncomparable";
const STRUCTURAL_INVALID_REASON: &str = "routing-structure-nonfinite-or-incomplete";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HoldoutCase {
    pub name: &'static str,
    pub prompt: &'static str,
}

pub const HOLDOUT_CORPUS: [HoldoutCase; HOLDOUT_CORPUS_CASE_COUNT] = [
    HoldoutCase {
        name: "rust-ownership-holdout",
        prompt: "Fix the Rust function below so it compiles without cloning the input.\nReturn only the corrected function followed by one short explanation.\n\nfn first_word(s: String) -> &str {\n    s.split_whitespace().next().unwrap_or(\"\")\n}",
    },
    HoldoutCase {
        name: "python-async-holdout",
        prompt: "Find the concurrency bug in this Python asyncio code and provide the\nminimal corrected version.\n\nasync def load_all(urls):\n    tasks = []\n    for url in urls:\n        tasks.append(asyncio.create_task(fetch(url)))\n    for task in tasks:\n        return await task",
    },
    HoldoutCase {
        name: "postgres-window-holdout",
        prompt: "Write a PostgreSQL query that returns each customer's most recent order,\nincluding customer_id, order_id, ordered_at, and total, with exactly one\nrow per customer. Use a window function.",
    },
    HoldoutCase {
        name: "spanish-refactor-holdout",
        prompt: "En Rust, refactoriza esta función para evitar una asignación innecesaria\nsin cambiar su comportamiento. Devuelve primero el código y luego una\nexplicación breve.\n\nfn normalize(name: &str) -> String {\n    let value = name.to_string();\n    value.trim().to_lowercase()\n}",
    },
];

fn hash_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

pub fn holdout_corpus_sha256() -> String {
    let mut hasher = Sha256::new();
    hash_len_prefixed(&mut hasher, HOLDOUT_CORPUS_ID.as_bytes());
    hasher.update(HOLDOUT_CORPUS_VERSION.to_le_bytes());
    hasher.update((OUTPUT_TOKEN_LIMIT as u64).to_le_bytes());
    for case in HOLDOUT_CORPUS {
        hash_len_prefixed(&mut hasher, case.name.as_bytes());
        hash_len_prefixed(&mut hasher, case.prompt.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CorpusIdentityEvidence {
    pub id: &'static str,
    pub version: u32,
    pub sha256: String,
    pub case_count: usize,
    pub output_token_limit: usize,
    pub role: &'static str,
}

impl CorpusIdentityEvidence {
    pub fn calibration() -> Self {
        Self {
            id: CALIBRATION_CORPUS_ID,
            version: CALIBRATION_CORPUS_VERSION,
            sha256: CALIBRATION_CORPUS_SHA256.to_string(),
            case_count: crate::greedy_parity::CORPUS_CASE_COUNT,
            output_token_limit: crate::greedy_parity::OUTPUT_TOKEN_LIMIT,
            role: "frozen-calibration-evidence-only-not-v2-pass",
        }
    }

    pub fn holdout() -> Self {
        Self {
            id: HOLDOUT_CORPUS_ID,
            version: HOLDOUT_CORPUS_VERSION,
            sha256: holdout_corpus_sha256(),
            case_count: HOLDOUT_CORPUS_CASE_COUNT,
            output_token_limit: OUTPUT_TOKEN_LIMIT,
            role: "independent-v2-qualification-holdout",
        }
    }

    fn is_exact_holdout(&self) -> bool {
        self == &Self::holdout() && self.sha256 == HOLDOUT_CORPUS_SHA256
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct NumericalLimits {
    pub max_absolute_error_limit: f64,
    pub rms_error_limit: f64,
    pub mean_absolute_error_limit: f64,
    pub nonfinite_mismatch_limit: usize,
    pub semantic_correctness_not_bit_parity: bool,
}

impl NumericalLimits {
    pub const fn frozen() -> Self {
        Self {
            max_absolute_error_limit: MAX_ABSOLUTE_ERROR_LIMIT,
            rms_error_limit: RMS_ERROR_LIMIT,
            mean_absolute_error_limit: MEAN_ABSOLUTE_ERROR_LIMIT,
            nonfinite_mismatch_limit: NONFINITE_MISMATCH_LIMIT,
            semantic_correctness_not_bit_parity: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProductionSemanticsEvidence {
    pub production_inference_math_changed: bool,
    pub production_shader_math_changed: bool,
    pub production_router_math_changed: bool,
    pub production_selected_expert_order_changed: bool,
    pub production_accumulation_order_changed: bool,
    pub existing_v1_qualifier_semantics_changed: bool,
    pub existing_semantic_corpus_survey_changed: bool,
    pub frozen_v2_numerical_limits_changed: bool,
    pub frozen_v2_holdout_prompts_changed: bool,
    pub cpu_combiner: &'static str,
    pub gpu_combiner: &'static str,
    pub canonicalization_scope: &'static str,
}

impl Default for ProductionSemanticsEvidence {
    fn default() -> Self {
        Self {
            production_inference_math_changed: false,
            production_shader_math_changed: false,
            production_router_math_changed: false,
            production_selected_expert_order_changed: false,
            production_accumulation_order_changed: false,
            existing_v1_qualifier_semantics_changed: false,
            existing_semantic_corpus_survey_changed: false,
            frozen_v2_numerical_limits_changed: false,
            frozen_v2_holdout_prompts_changed: false,
            cpu_combiner: "production-cpu-sequential-selected-expert-order",
            gpu_combiner: "production-gpu-existing-selected-expert-order",
            canonicalization_scope:
                "qualification-host-evidence-only-no-production-canonicalization",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SameInputRouterEventEvidence {
    pub location: EventLocation,
    pub cpu_production_router_on_exact_gpu_input: Option<RouterSameInputEvidence>,
    pub evaluation_error: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SameInputRouterSummary {
    pub same_input_router_events_total: usize,
    pub same_input_router_ordered_equal: usize,
    pub same_input_router_ordered_mismatch: usize,
    pub first_same_input_mismatch: Option<SameInputRouterEventEvidence>,
}

impl SameInputRouterSummary {
    fn from_events(events: &[SameInputRouterEventEvidence]) -> Self {
        let same_input_router_ordered_equal = events
            .iter()
            .filter(|event| {
                event
                    .cpu_production_router_on_exact_gpu_input
                    .as_ref()
                    .is_some_and(|evidence| evidence.ordered_ids_equal)
                    && event.evaluation_error.is_none()
            })
            .count();
        let first_same_input_mismatch = events.iter().find_map(|event| {
            (event.evaluation_error.is_some()
                || event
                    .cpu_production_router_on_exact_gpu_input
                    .as_ref()
                    .is_none_or(|evidence| !evidence.ordered_ids_equal))
            .then(|| event.clone())
        });
        Self {
            same_input_router_events_total: events.len(),
            same_input_router_ordered_equal,
            same_input_router_ordered_mismatch: events
                .len()
                .saturating_sub(same_input_router_ordered_equal),
            first_same_input_mismatch,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RuntimeCompletionEvidence {
    pub gpu_expected_completed_token_steps: u64,
    pub gpu_token_loop: GpuNativeTokenLoopSnapshot,
    pub reference_routed_execution: crate::engine::RoutedExpertExecutionSnapshot,
    pub gpu_native_routed_execution: crate::engine::RoutedExpertExecutionSnapshot,
    pub reference_nonfinite_attention_fallbacks: u64,
    pub gpu_native_nonfinite_attention_fallbacks: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct QualificationGates {
    pub exact_provenance: bool,
    pub frozen_holdout_corpus_identity: bool,
    pub exact_model_geometry: bool,
    pub strict_model_load: bool,
    pub complete_exact_greedy_token_parity: bool,
    pub routing_structural_validity: bool,
    pub same_input_gpu_router_ordered_ids_exact: bool,
    pub same_input_routed_moe_numerical_limits: bool,
    pub deterministic_numerical_sampling_complete: bool,
    pub no_fallback_or_degradation: bool,
    pub residency_and_execution_completion: bool,
    pub controlled_shutdown: bool,
    pub observation_completeness: bool,
}

impl QualificationGates {
    pub fn qualification_pass(&self) -> bool {
        self.exact_provenance
            && self.frozen_holdout_corpus_identity
            && self.exact_model_geometry
            && self.strict_model_load
            && self.complete_exact_greedy_token_parity
            && self.routing_structural_validity
            && self.same_input_gpu_router_ordered_ids_exact
            && self.same_input_routed_moe_numerical_limits
            && self.deterministic_numerical_sampling_complete
            && self.no_fallback_or_degradation
            && self.residency_and_execution_completion
            && self.controlled_shutdown
            && self.observation_completeness
    }

    fn failed_criteria(&self) -> Vec<&'static str> {
        let values = [
            ("exact-provenance", self.exact_provenance),
            (
                "frozen-holdout-corpus-identity",
                self.frozen_holdout_corpus_identity,
            ),
            ("exact-model-geometry", self.exact_model_geometry),
            ("strict-model-load", self.strict_model_load),
            (
                "complete-exact-greedy-token-parity",
                self.complete_exact_greedy_token_parity,
            ),
            (
                "routing-structural-validity",
                self.routing_structural_validity,
            ),
            (
                "same-input-gpu-router-ordered-ids-exact",
                self.same_input_gpu_router_ordered_ids_exact,
            ),
            (
                "same-input-routed-moe-numerical-limits",
                self.same_input_routed_moe_numerical_limits,
            ),
            (
                "deterministic-numerical-sampling-complete",
                self.deterministic_numerical_sampling_complete,
            ),
            (
                "no-fallback-or-degradation",
                self.no_fallback_or_degradation,
            ),
            (
                "residency-and-execution-completion",
                self.residency_and_execution_completion,
            ),
            ("controlled-shutdown", self.controlled_shutdown),
            ("observation-completeness", self.observation_completeness),
        ];
        values
            .into_iter()
            .filter_map(|(criterion, passed)| (!passed).then_some(criterion))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct V2GlobalSummary {
    pub routing: RoutingCounters,
    pub tokens: TokenSummary,
    pub same_input_router: SameInputRouterSummary,
    pub cross_backend_membership_drift_events: usize,
    pub same_input_numerics: NumericalAggregation,
    pub permutation_only_numerics: NumericalAggregation,
    pub nonfinite_mismatch_count: usize,
    pub sampling: SamplingSummary,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SemanticParityV2Report {
    pub schema: &'static str,
    pub mode: &'static str,
    pub status: QualificationStatus,
    pub qualification_pass: bool,
    pub failed_criteria: Vec<&'static str>,
    pub expected_adapter_name: String,
    pub calibration_corpus: CorpusIdentityEvidence,
    pub holdout_corpus: CorpusIdentityEvidence,
    pub numerical_limits: NumericalLimits,
    pub provenance: DiagnosticProvenance,
    pub gates: QualificationGates,
    pub global_summary: V2GlobalSummary,
    pub token_cases: Vec<TokenCaseEvidence>,
    pub routing_events: Vec<RoutingEventEvidence>,
    pub same_input_router_events: Vec<SameInputRouterEventEvidence>,
    pub same_input_numerical_events: Vec<SameInputNumericalEventEvidence>,
    pub same_input_numerics_by_classification: Vec<ClassificationNumericalAggregation>,
    pub permutation_only_events: Vec<PermutationNumericalEventEvidence>,
    pub runtime_completion: RuntimeCompletionEvidence,
    pub production_semantics: ProductionSemanticsEvidence,
    pub observation_seams_not_implemented: Vec<String>,
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn artifact_is_exact(artifact: Option<&crate::qualification::ArtifactDigest>) -> bool {
    artifact.is_some_and(|artifact| {
        !artifact.canonical_path.is_empty()
            && artifact.byte_length > 0
            && is_hex(&artifact.sha256, 64)
    })
}

fn exact_provenance_pass(provenance: &DiagnosticProvenance, expected_adapter_name: &str) -> bool {
    provenance.build.dirty == Some(false)
        && provenance
            .build
            .git_sha
            .as_deref()
            .is_some_and(|sha| is_hex(sha, 40))
        && !provenance.build.package_version.is_empty()
        && is_hex(&provenance.executable_sha256, 64)
        && artifact_is_exact(provenance.artifacts.config.as_ref())
        && artifact_is_exact(provenance.artifacts.tokenizer.as_ref())
        && artifact_is_exact(provenance.artifacts.expert_metadata.as_ref())
        && is_hex(&provenance.gpu_resolved_config_sha256, 64)
        && is_hex(&provenance.reference_resolved_config_sha256, 64)
        && !expected_adapter_name.trim().is_empty()
        && provenance.device.name == expected_adapter_name
        && !provenance.device.software_adapter
        && !provenance.device.device_type.eq_ignore_ascii_case("cpu")
        && !provenance.device.wgpu_backend.is_empty()
        && !provenance.device.driver.is_empty()
}

fn exact_model_geometry_pass(provenance: &DiagnosticProvenance) -> bool {
    provenance.model_identity.is_qwen3_coder_30b_a3b_q4_0()
        && provenance.expert_metadata.dtype.as_deref() == Some("q4_0")
        && provenance.expert_metadata.q4_0_layout.as_deref()
            == Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
        && !provenance.expert_metadata.explicitly_synthetic
}

fn model_load_is_strict(load: &crate::greedy_parity::ModelLoadEvidence) -> bool {
    load.strict
        && load.loaded_tensors == load.required_tensors
        && load.required_tensors > 0
        && !load.seeded_fallback_remained
        && load.loader != "seeded"
}

fn token_parity_pass(cases: &[TokenCaseEvidence]) -> bool {
    cases.len() == HOLDOUT_CORPUS_CASE_COUNT
        && cases.iter().zip(HOLDOUT_CORPUS).all(|(case, fixed)| {
            case.case == fixed.name
                && case.reference_generated_token_ids.len() == OUTPUT_TOKEN_LIMIT
                && case.gpu_generated_token_ids.len() == OUTPUT_TOKEN_LIMIT
                && case.reference_generated_token_ids == case.gpu_generated_token_ids
                && case.exact_match_count == OUTPUT_TOKEN_LIMIT
                && case.mismatch_count == 0
                && case.first_mismatch_position.is_none()
        })
}

fn routing_structural_pass(counters: &RoutingCounters, expected_events: usize) -> bool {
    counters.total_routing_events == expected_events
        && counters.invalid_events == 0
        && counters.reference_nonfinite_events == 0
        && counters.gpu_nonfinite_events == 0
        && counters.expert_weight_pairing_defect_events == 0
}

fn same_input_router_pass(events: &[SameInputRouterEventEvidence], expected_events: usize) -> bool {
    events.len() == expected_events
        && events.iter().all(|event| {
            event.evaluation_error.is_none()
                && event
                    .cpu_production_router_on_exact_gpu_input
                    .as_ref()
                    .is_some_and(|evidence| evidence.ordered_ids_equal)
        })
}

fn numerical_event_within_limits(event: &SameInputNumericalEventEvidence) -> bool {
    let evidence = &event.cpu_vs_gpu_routed_moe;
    evidence
        .max_absolute_error
        .is_some_and(|value| value <= MAX_ABSOLUTE_ERROR_LIMIT)
        && evidence
            .rms_error
            .is_some_and(|value| value <= RMS_ERROR_LIMIT)
        && evidence
            .mean_absolute_error
            .is_some_and(|value| value <= MEAN_ABSOLUTE_ERROR_LIMIT)
        && evidence.nonfinite_bit_mismatch_count == NONFINITE_MISMATCH_LIMIT
}

fn numerical_sampling_complete(
    routing: &RoutingCounters,
    sampling: &SamplingSummary,
    numerical_events: &[SameInputNumericalEventEvidence],
    permutation_events: &[PermutationNumericalEventEvidence],
    num_layers: usize,
) -> bool {
    sampling.exact_order_sampling_rule == EXACT_ORDER_SAMPLING_RULE
        && sampling.layers_with_exact_order_sample == (0..num_layers).collect::<Vec<_>>()
        && sampling.layers_without_exact_order_sample.is_empty()
        && sampling.exact_order_events_numerically_sampled == num_layers
        && sampling.internal_permutation_events_numerically_measured
            == routing.internal_rank_permutation_events
        && sampling.membership_mismatch_events_numerically_measured
            == routing.membership_mismatch_events
        && numerical_events.len()
            == num_layers
                + routing.internal_rank_permutation_events
                + routing.membership_mismatch_events
        && permutation_events.len() == routing.internal_rank_permutation_events
}

fn no_fallback_or_degradation_pass(runtime: &RuntimeCompletionEvidence) -> bool {
    runtime
        .gpu_native_routed_execution
        .cpu_routed_expert_dispatches
        == 0
        && runtime.gpu_native_routed_execution.gpu_cpu_fallbacks == 0
        && runtime
            .gpu_native_routed_execution
            .degraded_expert_substitutions
            == 0
        && runtime.reference_routed_execution.gpu_cpu_fallbacks == 0
        && runtime
            .reference_routed_execution
            .degraded_expert_substitutions
            == 0
        && runtime.reference_nonfinite_attention_fallbacks == 0
        && runtime.gpu_native_nonfinite_attention_fallbacks == 0
}

fn residency_and_execution_completion_pass(runtime: &RuntimeCompletionEvidence) -> bool {
    runtime.gpu_expected_completed_token_steps > 0
        && runtime.gpu_token_loop.tokens_completed == runtime.gpu_expected_completed_token_steps
        && runtime.gpu_token_loop.fatal_failures == 0
        && runtime.gpu_token_loop.no_progress_failures == 0
        && runtime.gpu_native_routed_execution.gpu_dispatch_failures == 0
}

fn controlled_shutdown_pass(provenance: &DiagnosticProvenance) -> bool {
    provenance
        .reference_background_shutdown
        .controlled_shutdown_requested
        && provenance
            .reference_background_shutdown
            .all_runtime_resources_released
        && provenance
            .gpu_native_background_shutdown
            .controlled_shutdown_requested
        && provenance
            .gpu_native_background_shutdown
            .all_runtime_resources_released
}

#[allow(clippy::too_many_arguments)]
fn derive_gates(
    provenance: &DiagnosticProvenance,
    expected_adapter_name: &str,
    holdout: &CorpusIdentityEvidence,
    token_cases: &[TokenCaseEvidence],
    routing_events: &[RoutingEventEvidence],
    routing: &RoutingCounters,
    sampling: &SamplingSummary,
    same_input_router_events: &[SameInputRouterEventEvidence],
    numerical_events: &[SameInputNumericalEventEvidence],
    permutation_events: &[PermutationNumericalEventEvidence],
    runtime: &RuntimeCompletionEvidence,
    observation_seams_not_implemented: &[String],
) -> QualificationGates {
    let expected_events = HOLDOUT_CORPUS_CASE_COUNT * OUTPUT_TOKEN_LIMIT * 48;
    let observation_completeness = observation_seams_not_implemented.is_empty()
        && routing.total_routing_events == expected_events
        && routing_events.len() == expected_events
        && same_input_router_events.len() == expected_events
        && token_cases.len() == HOLDOUT_CORPUS_CASE_COUNT;
    QualificationGates {
        exact_provenance: exact_provenance_pass(provenance, expected_adapter_name),
        frozen_holdout_corpus_identity: holdout.is_exact_holdout(),
        exact_model_geometry: exact_model_geometry_pass(provenance),
        strict_model_load: model_load_is_strict(&provenance.reference_model_load)
            && model_load_is_strict(&provenance.gpu_native_model_load),
        complete_exact_greedy_token_parity: token_parity_pass(token_cases),
        routing_structural_validity: routing_structural_pass(routing, expected_events),
        same_input_gpu_router_ordered_ids_exact: same_input_router_pass(
            same_input_router_events,
            expected_events,
        ),
        same_input_routed_moe_numerical_limits: !numerical_events.is_empty()
            && numerical_events.iter().all(numerical_event_within_limits),
        deterministic_numerical_sampling_complete: numerical_sampling_complete(
            routing,
            sampling,
            numerical_events,
            permutation_events,
            48,
        ),
        no_fallback_or_degradation: no_fallback_or_degradation_pass(runtime),
        residency_and_execution_completion: residency_and_execution_completion_pass(runtime),
        controlled_shutdown: controlled_shutdown_pass(provenance),
        observation_completeness,
    }
}

impl SemanticParityV2Report {
    #[allow(clippy::too_many_arguments)]
    fn new(
        expected_adapter_name: String,
        provenance: DiagnosticProvenance,
        token_cases: Vec<TokenCaseEvidence>,
        routing_events: Vec<RoutingEventEvidence>,
        routing_counters: RoutingCounters,
        sampling: SamplingSummary,
        same_input_router_events: Vec<SameInputRouterEventEvidence>,
        same_input_numerical_events: Vec<SameInputNumericalEventEvidence>,
        permutation_only_events: Vec<PermutationNumericalEventEvidence>,
        runtime_completion: RuntimeCompletionEvidence,
        observation_seams_not_implemented: Vec<String>,
    ) -> Self {
        let holdout_corpus = CorpusIdentityEvidence::holdout();
        let gates = derive_gates(
            &provenance,
            &expected_adapter_name,
            &holdout_corpus,
            &token_cases,
            &routing_events,
            &routing_counters,
            &sampling,
            &same_input_router_events,
            &same_input_numerical_events,
            &permutation_only_events,
            &runtime_completion,
            &observation_seams_not_implemented,
        );
        let qualification_pass = gates.qualification_pass();
        let tokens = crate::gpu_native_semantic_parity_corpus::summarize_tokens(
            &token_cases,
            HOLDOUT_CORPUS_CASE_COUNT * OUTPUT_TOKEN_LIMIT,
        );
        let same_input_router = SameInputRouterSummary::from_events(&same_input_router_events);
        let same_input_numerics = aggregate_numerics(&same_input_numerical_events);
        let permutation_only_numerics = aggregate_permutation_numerics(&permutation_only_events);
        let nonfinite_mismatch_count = same_input_numerical_events
            .iter()
            .map(|event| event.cpu_vs_gpu_routed_moe.nonfinite_bit_mismatch_count)
            .sum();
        let same_input_numerics_by_classification =
            aggregate_numerics_by_classification(&same_input_numerical_events);
        let failed_criteria = gates.failed_criteria();
        Self {
            schema: SCHEMA_VERSION,
            mode: MODE,
            status: if qualification_pass {
                QualificationStatus::Pass
            } else {
                QualificationStatus::Fail
            },
            qualification_pass,
            failed_criteria,
            expected_adapter_name,
            calibration_corpus: CorpusIdentityEvidence::calibration(),
            holdout_corpus,
            numerical_limits: NumericalLimits::frozen(),
            provenance,
            gates,
            global_summary: V2GlobalSummary {
                cross_backend_membership_drift_events: routing_counters.membership_mismatch_events,
                routing: routing_counters,
                tokens,
                same_input_router,
                same_input_numerics,
                permutation_only_numerics,
                nonfinite_mismatch_count,
                sampling,
            },
            token_cases,
            routing_events,
            same_input_router_events,
            same_input_numerical_events,
            same_input_numerics_by_classification,
            permutation_only_events,
            runtime_completion,
            production_semantics: ProductionSemanticsEvidence::default(),
            observation_seams_not_implemented,
        }
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
    expected_completed_token_steps: u64,
    token_loop_snapshot: GpuNativeTokenLoopSnapshot,
    routed_execution: crate::engine::RoutedExpertExecutionSnapshot,
    nonfinite_attention_fallbacks: u64,
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
    same_input_router_events: Vec<SameInputRouterEventEvidence>,
    same_input_events: Vec<SameInputNumericalEventEvidence>,
    permutation_events: Vec<PermutationNumericalEventEvidence>,
    reference_model_load: crate::greedy_parity::ModelLoadEvidence,
    reference_background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
    reference_routed_execution: crate::engine::RoutedExpertExecutionSnapshot,
    reference_nonfinite_attention_fallbacks: u64,
}

struct PlannedEvent {
    case_index: usize,
    generated_position: usize,
    layer: usize,
    evidence: RoutingEventEvidence,
    actual_gpu_router: Option<ActualGpuRouterEvidence>,
    reference_nonfinite: bool,
    gpu_nonfinite: bool,
}

struct EventPlan {
    events: Vec<PlannedEvent>,
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
                "v2 GPU identity {observed_config_sha256} drifted from {resolved_config_sha256}"
            )
            .into());
        }
        let device = runtime
            .engine
            .gpu_device_identity()
            .ok_or("v2 GPU runtime has no authoritative adapter identity")?;
        if device.name != expected_adapter_name {
            return Err(format!(
                "v2 GPU runtime selected adapter {:?}, expected {:?}",
                device.name, expected_adapter_name
            )
            .into());
        }
        if device.software_adapter || device.device_type.eq_ignore_ascii_case("cpu") {
            return Err(
                format!("v2 GPU runtime selected software adapter {:?}", device.name).into(),
            );
        }
        let token_loop = runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or("v2 GPU-native token loop was not initialized")?;
        let model_geometry = token_loop.model_geometry();
        if runtime.model.layers.len() != model_geometry.num_layers {
            return Err("v2 GPU layer geometry is incomplete".into());
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
        if token_loop.snapshot() != GpuNativeTokenLoopSnapshot::default() {
            return Err("v2 GPU token-loop counters did not start at zero".into());
        }
        let routed_before = runtime.engine.routed_expert_execution_snapshot();
        let attention_before = crate::transformer::nonfinite_softmax_fallbacks();
        let trace_layout = SemanticCorpusTraceLayout::try_new(model_geometry)?;
        let mut cases = Vec::with_capacity(HOLDOUT_CORPUS_CASE_COUNT);
        let mut expected_completed = 0usize;
        for fixed in HOLDOUT_CORPUS {
            let prompt_token_ids = tokenizer.encode(fixed.prompt)?;
            if prompt_token_ids.is_empty() {
                return Err(
                    format!("v2 holdout case {:?} encoded to zero tokens", fixed.name).into(),
                );
            }
            let mut request =
                token_loop.create_semantic_parity_corpus_diagnostic_request_state()?;
            let staging = token_loop
                .create_semantic_parity_corpus_diagnostic_staging_buffer(&trace_layout)?;
            let (generated_token_ids, traces) = crate::with_progress_timeout(
                format!("GPU-native semantic parity v2 {} candidate", fixed.name),
                watchdog,
                async {
                    let prefix_count = prompt_token_ids.len().saturating_sub(1);
                    for (position, &token_id) in prompt_token_ids[..prefix_count].iter().enumerate()
                    {
                        token_loop
                            .step_token(&runtime.engine, &mut request, token_id, position, false)
                            .await?;
                    }
                    let mut input_token = *prompt_token_ids
                        .last()
                        .ok_or("v2 GPU holdout prompt is empty")?;
                    let mut position = prefix_count;
                    let mut generated_token_ids = Vec::with_capacity(OUTPUT_TOKEN_LIMIT);
                    let mut traces = Vec::with_capacity(OUTPUT_TOKEN_LIMIT);
                    while generated_token_ids.len() < OUTPUT_TOKEN_LIMIT {
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
                            return Err("v2 GPU trace omitted one or more layers".into());
                        }
                        generated_token_ids.push(sampled_token);
                        traces.push(trace);
                        input_token = sampled_token;
                        position = position
                            .checked_add(1)
                            .ok_or("v2 GPU position overflowed")?;
                    }
                    Ok::<_, Box<dyn std::error::Error>>((generated_token_ids, traces))
                },
            )
            .await?;
            let case_expected = prompt_token_ids
                .len()
                .saturating_sub(1)
                .checked_add(OUTPUT_TOKEN_LIMIT)
                .ok_or("v2 GPU expected completion count overflowed")?;
            if request.committed_position() != case_expected {
                return Err(format!(
                    "v2 GPU case {} retired at {}, expected {case_expected}",
                    fixed.name,
                    request.committed_position()
                )
                .into());
            }
            expected_completed = expected_completed
                .checked_add(case_expected)
                .ok_or("v2 GPU total completion count overflowed")?;
            drop(request);
            drop(staging);
            cases.push(GpuCaseCapture {
                case: fixed.name,
                prompt_token_ids,
                generated_token_ids,
                traces,
            });
        }
        let token_loop_snapshot = token_loop.snapshot();
        let routed_execution = crate::qualification::routed_execution_delta(
            routed_before,
            runtime.engine.routed_expert_execution_snapshot(),
        )
        .map_err(|failure| failure.detail)?;
        Ok::<_, Box<dyn std::error::Error>>(GpuCapture {
            cases,
            gate_identities,
            model_load,
            device,
            model_geometry,
            background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(),
            expected_completed_token_steps: u64::try_from(expected_completed)
                .map_err(|_| "v2 GPU completion count does not fit u64")?,
            token_loop_snapshot,
            routed_execution,
            nonfinite_attention_fallbacks: crate::transformer::nonfinite_softmax_fallbacks()
                .saturating_sub(attention_before),
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
            "{error}; v2 GPU shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

fn values_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn first_generated_token_mismatch_position(reference: &[u32], gpu: &[u32]) -> Option<usize> {
    (0..reference.len().max(gpu.len())).find(|&index| reference.get(index) != gpu.get(index))
}

fn post_generated_token_divergence_invalid_reason(
    first_mismatch_position: Option<usize>,
    generated_position: usize,
) -> Option<&'static str> {
    first_mismatch_position
        .is_some_and(|mismatch_position| generated_position > mismatch_position)
        .then_some(POST_GENERATED_TOKEN_DIVERGENCE_INVALID_REASON)
}

fn scalar_delta(expert_id: u32, left: f32, right: f32) -> WeightDeltaEvidence {
    let absolute = (f64::from(right) - f64::from(left)).abs();
    WeightDeltaEvidence {
        expert_id,
        left: crate::numerical_diagnostics::FloatEvidence::new(left),
        right: crate::numerical_diagnostics::FloatEvidence::new(right),
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

fn plan_events(
    reference_cases: &[ReferenceCaseCapture],
    gpu: &GpuCapture,
    gates: &[crate::gating::LinearGate],
) -> Result<EventPlan, Box<dyn std::error::Error>> {
    if reference_cases.len() != HOLDOUT_CORPUS_CASE_COUNT
        || gpu.cases.len() != HOLDOUT_CORPUS_CASE_COUNT
        || gates.len() != gpu.model_geometry.num_layers
    {
        return Err("v2 case or gate coverage is incomplete".into());
    }
    let mut sampler = DeterministicEventSampler::new(gpu.model_geometry.num_layers);
    let mut events = Vec::with_capacity(
        HOLDOUT_CORPUS_CASE_COUNT * OUTPUT_TOKEN_LIMIT * gpu.model_geometry.num_layers,
    );
    for (case_index, ((reference_case, gpu_case), fixed)) in reference_cases
        .iter()
        .zip(&gpu.cases)
        .zip(HOLDOUT_CORPUS)
        .enumerate()
    {
        if reference_case.case != fixed.name
            || gpu_case.case != fixed.name
            || reference_case.traces.len() != OUTPUT_TOKEN_LIMIT
            || gpu_case.traces.len() != OUTPUT_TOKEN_LIMIT
        {
            return Err(format!("v2 holdout case {} coverage drifted", fixed.name).into());
        }
        let first_mismatch_position = first_generated_token_mismatch_position(
            &reference_case.generated_token_ids,
            &gpu_case.generated_token_ids,
        );
        for generated_position in 0..OUTPUT_TOKEN_LIMIT {
            let divergence_reason = post_generated_token_divergence_invalid_reason(
                first_mismatch_position,
                generated_position,
            );
            let semantically_comparable = divergence_reason.is_none();
            let reference_trace = &reference_case.traces[generated_position];
            let gpu_trace = &gpu_case.traces[generated_position];
            if reference_trace.layer_selected_ids.len() != gpu.model_geometry.num_layers
                || reference_trace.layer_selected_weights.len() != gpu.model_geometry.num_layers
                || reference_trace.layer_router_input.len() != gpu.model_geometry.num_layers
                || reference_trace.layer_routed_moe_output.len() != gpu.model_geometry.num_layers
                || gpu_trace.layers.len() != gpu.model_geometry.num_layers
            {
                return Err(format!(
                    "v2 case {} position {generated_position} omitted layer evidence",
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
                let reference_router = (semantically_comparable
                    && geometry_complete
                    && !reference_nonfinite
                    && !gpu_nonfinite)
                    .then(|| {
                        crate::gpu_native_router_rank_diagnostics::evaluate_cpu_router(
                            "reference-hidden-to-cpu-production-router",
                            &gates[layer],
                            reference_input,
                        )
                    })
                    .transpose()
                    .ok()
                    .flatten();
                let actual_gpu_router = (geometry_complete && !gpu_nonfinite)
                    .then(|| {
                        crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
                            gpu_layer.raw_logits.clone(),
                            gpu_layer.selected_ids.clone(),
                            gpu_layer.selected_weights.clone(),
                        )
                    })
                    .transpose()
                    .ok()
                    .flatten();
                let mut classification = if semantically_comparable {
                    crate::gpu_native_semantic_parity_corpus::classify_routing_event(
                        reference_ids,
                        &gpu_layer.selected_ids,
                        gpu.model_geometry.num_experts,
                        geometry_complete && !reference_nonfinite && !gpu_nonfinite,
                    )
                } else {
                    RoutingClassification::InvalidNonfiniteIncomplete
                };
                if semantically_comparable
                    && (reference_router.is_none() || actual_gpu_router.is_none())
                {
                    classification = RoutingClassification::InvalidNonfiniteIncomplete;
                }
                let reference_pairing_valid = canonical_selected_weight_evidence(
                    reference_ids,
                    reference_weights,
                    reference_ids,
                    reference_weights,
                    gpu.model_geometry.num_experts,
                )
                .is_ok()
                    && reference_router.as_ref().is_none_or(|router| {
                        router.top_8_ids == *reference_ids
                            && router
                                .top_8_weights
                                .iter()
                                .map(|value| value.bits)
                                .eq(reference_weights.iter().map(|value| value.to_bits()))
                    });
                let gpu_pairing_valid = canonical_selected_weight_evidence(
                    &gpu_layer.selected_ids,
                    &gpu_layer.selected_weights,
                    &gpu_layer.selected_ids,
                    &gpu_layer.selected_weights,
                    gpu.model_geometry.num_experts,
                )
                .is_ok()
                    && actual_gpu_router
                        .as_ref()
                        .is_some_and(|router| router.selected_weights_paired_with_expert_ids);
                let expert_weight_pairing_defect =
                    semantically_comparable && (!reference_pairing_valid || !gpu_pairing_valid);
                let canonical_selected_weights = matches!(
                    classification,
                    RoutingClassification::InternalRankPermutation
                        | RoutingClassification::MembershipMismatch
                )
                .then(|| {
                    canonical_selected_weight_evidence(
                        reference_ids,
                        reference_weights,
                        &gpu_layer.selected_ids,
                        &gpu_layer.selected_weights,
                        gpu.model_geometry.num_experts,
                    )
                })
                .transpose()
                .ok()
                .flatten();
                let displacement = if semantically_comparable {
                    rank_displacement(reference_ids, &gpu_layer.selected_ids)
                } else {
                    RankDisplacementEvidence::default()
                };
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
                let sampling = sampler.select(layer, classification);
                events.push(PlannedEvent {
                    case_index,
                    generated_position,
                    layer,
                    evidence: RoutingEventEvidence {
                        location,
                        classification,
                        invalid_reason: divergence_reason.or_else(|| {
                            (classification == RoutingClassification::InvalidNonfiniteIncomplete)
                                .then_some(STRUCTURAL_INVALID_REASON)
                        }),
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
                    actual_gpu_router,
                    reference_nonfinite,
                    gpu_nonfinite,
                });
            }
        }
    }
    Ok(EventPlan {
        events,
        layers_with_exact_sample: sampler.layers_with_exact_sample(),
        layers_without_exact_sample: sampler.layers_without_exact_sample(),
    })
}

fn evaluate_same_input_router(
    gate: &crate::gating::LinearGate,
    gpu_layer: &crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuLayerTrace,
    actual_gpu_router: Option<&ActualGpuRouterEvidence>,
    num_experts: usize,
) -> Result<RouterSameInputEvidence, String> {
    let cpu_router = crate::gpu_native_router_rank_diagnostics::evaluate_cpu_router(
        "gpu-hidden-to-cpu-production-router",
        gate,
        &gpu_layer.router_input,
    )?;
    let routing = gate.route(&gpu_layer.router_input);
    let routing_matches_evaluation = routing.experts == cpu_router.top_8_ids
        && routing
            .weights
            .iter()
            .map(|value| value.to_bits())
            .eq(cpu_router.top_8_weights.iter().map(|value| value.bits));
    if !routing_matches_evaluation {
        return Err("CPU production route disagrees with same-input router evaluation".to_string());
    }
    let actual_gpu_router = actual_gpu_router
        .ok_or_else(|| "actual GPU production router evidence is unavailable".to_string())?;
    if actual_gpu_router.top_8_ids != gpu_layer.selected_ids
        || !actual_gpu_router.selected_weights_paired_with_expert_ids
    {
        return Err("actual GPU router evidence is invalid or unpaired".to_string());
    }
    let canonical = canonical_selected_weight_evidence(
        &routing.experts,
        &routing.weights,
        &gpu_layer.selected_ids,
        &gpu_layer.selected_weights,
        num_experts,
    )?;
    Ok(RouterSameInputEvidence {
        cpu_selected_ids: cpu_router.top_8_ids.clone(),
        gpu_selected_ids: gpu_layer.selected_ids.clone(),
        membership_equal: same_membership(&cpu_router.top_8_ids, &gpu_layer.selected_ids),
        ordered_ids_equal: cpu_router.top_8_ids == gpu_layer.selected_ids,
        common_expert_score_deltas: common_selected_score_deltas(&cpu_router, actual_gpu_router),
        common_expert_weight_deltas: canonical
            .into_iter()
            .filter_map(|item| item.delta)
            .collect(),
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
                "v2 reference identity {observed_config_sha256} drifted from {resolved_config_sha256}"
            )
            .into());
        }
        if runtime.model.layers.len() != gpu.model_geometry.num_layers {
            return Err("v2 reference layer geometry is incomplete".into());
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
            return Err("v2 reference and GPU gate tensor identities differ".into());
        }
        let model_load = crate::greedy_parity_model_load(&runtime);
        let boundary_before = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_before.enabled || boundary_before.routed_expert_dispatches != 0 {
            return Err("v2 reference boundary emulation did not start clean".into());
        }
        let routed_before = runtime.engine.routed_expert_execution_snapshot();
        let attention_before = crate::transformer::nonfinite_softmax_fallbacks();
        let mut reference_cases = Vec::with_capacity(HOLDOUT_CORPUS_CASE_COUNT);
        for (case_index, fixed) in HOLDOUT_CORPUS.into_iter().enumerate() {
            let prompt_token_ids = tokenizer.encode(fixed.prompt)?;
            if prompt_token_ids != gpu.cases[case_index].prompt_token_ids {
                return Err(
                    format!("v2 tokenizer identity drifted for case {}", fixed.name).into(),
                );
            }
            let (generated_token_ids, traces) = crate::with_progress_timeout(
                format!("semantic parity v2 {} authoritative reference", fixed.name),
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
                        .ok_or("v2 reference holdout prompt is empty")?;
                    let mut position = prefix_count;
                    let mut generated_token_ids = Vec::with_capacity(OUTPUT_TOKEN_LIMIT);
                    let mut traces = Vec::with_capacity(OUTPUT_TOKEN_LIMIT);
                    while generated_token_ids.len() < OUTPUT_TOKEN_LIMIT {
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
                            .ok_or("v2 reference position overflowed")?;
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
        let same_input_router_events = plan
            .events
            .iter()
            .map(|planned| {
                let gpu_layer = &gpu.cases[planned.case_index].traces
                    [planned.generated_position]
                    .layers[planned.layer];
                match evaluate_same_input_router(
                    &gates[planned.layer],
                    gpu_layer,
                    planned.actual_gpu_router.as_ref(),
                    gpu.model_geometry.num_experts,
                ) {
                    Ok(evidence) => SameInputRouterEventEvidence {
                        location: planned.evidence.location.clone(),
                        cpu_production_router_on_exact_gpu_input: Some(evidence),
                        evaluation_error: None,
                    },
                    Err(error) => SameInputRouterEventEvidence {
                        location: planned.evidence.location.clone(),
                        cpu_production_router_on_exact_gpu_input: None,
                        evaluation_error: Some(error),
                    },
                }
            })
            .collect::<Vec<_>>();

        let mut same_input_events = Vec::new();
        for (event_index, planned) in plan
            .events
            .iter_mut()
            .enumerate()
            .filter(|(_, event)| event.evidence.numerically_selected)
        {
            let gpu_layer = &gpu.cases[planned.case_index].traces[planned.generated_position]
                .layers[planned.layer];
            let reference_trace =
                &reference_cases[planned.case_index].traces[planned.generated_position];
            let Some(router_same_input) = same_input_router_events[event_index]
                .cpu_production_router_on_exact_gpu_input
                .clone()
            else {
                continue;
            };
            let routing = gates[planned.layer].route(&gpu_layer.router_input);
            let global_ids = routing
                .experts
                .iter()
                .map(|&expert| runtime.model.global_expert_id(planned.layer, expert))
                .collect::<Vec<_>>();
            let token_index = (planned.case_index as u64)
                .wrapping_mul(OUTPUT_TOKEN_LIMIT as u64)
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
                continue;
            }
            let cpu_routed_moe_output =
                crate::inference::combine_outputs(&expert_outputs, &routing.weights);
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
                router_same_input,
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
                witness:
                    crate::gpu_native_expert_permutation_semantic_parity::permutation_only_witness(
                        &gpu_layer.selected_ids,
                        &gpu_layer.selected_weights,
                        &gpu_layer.route_outputs,
                    )?,
            });
        }

        let boundary_after = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_after.enabled || boundary_after.routed_expert_dispatches == 0 {
            return Err("v2 reference did not exercise the Hybrid F16 boundary".into());
        }
        let reference_routed_execution = crate::qualification::routed_execution_delta(
            routed_before,
            runtime.engine.routed_expert_execution_snapshot(),
        )
        .map_err(|failure| failure.detail)?;
        let reference_nonfinite_attention_fallbacks =
            crate::transformer::nonfinite_softmax_fallbacks().saturating_sub(attention_before);
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
            same_input_router_events,
            same_input_events,
            permutation_events,
            reference_model_load: model_load,
            reference_background_shutdown:
                crate::greedy_parity::BackgroundShutdownEvidence::default(),
            reference_routed_execution,
            reference_nonfinite_attention_fallbacks,
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
            "{error}; v2 reference shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

fn emit_report(
    report: &SemanticParityV2Report,
    report_out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if report.qualification_pass != report.gates.qualification_pass() {
        return Err("v2 serialized qualification PASS disagrees with frozen gates".into());
    }
    if let Some(parent) = report_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    std::fs::write(report_out, json)?;
    eprintln!(
        "GPU-native semantic parity v2 qualification report written to {}",
        report_out.display()
    );
    Ok(())
}

pub async fn run_qualification(
    config: PathBuf,
    cfg: crate::config::Config,
    expected_adapter_name: String,
    report_out: PathBuf,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::BuildProvenance;

    let build = BuildProvenance::embedded();
    let (artifacts, artifact_errors) = crate::qualification_artifacts(&config, &cfg);
    if !artifact_errors.is_empty() {
        return Err(format!(
            "v2 qualification artifact preflight failed: {}",
            artifact_errors.join("; ")
        )
        .into());
    }
    if progress_watchdog.timeout.is_none() {
        return Err("v2 qualification requires a positive progress timeout".into());
    }
    if build.dirty != Some(false) || build.git_sha.as_deref().is_none_or(|sha| !is_hex(sha, 40)) {
        return Err("v2 qualification requires clean embedded 40-hex Git provenance".into());
    }
    if crate::greedy_parity::CorpusEvidence::fixed().sha256 != CALIBRATION_CORPUS_SHA256 {
        return Err("v2 calibration corpus identity drifted from the frozen contract".into());
    }
    if holdout_corpus_sha256() != HOLDOUT_CORPUS_SHA256 {
        return Err("v2 holdout corpus identity drifted from the frozen contract".into());
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
        return Err("v2 qualification requires the strict GPU-native Q4 configuration".into());
    }
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| format!("v2 expert metadata preflight failed: {error}"))?;
    if expert_metadata.q4_0_layout.as_deref() != Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
        || expert_metadata.dtype.as_deref() != Some("q4_0")
        || expert_metadata.explicitly_synthetic
    {
        return Err("v2 qualification requires canonical nonsynthetic Q4_0 metadata".into());
    }
    if expected_adapter_name.trim().is_empty() {
        return Err("v2 qualification requires a nonempty exact adapter name".into());
    }

    let mut gpu_spec = crate::resolve_real_cli_spec_from_config(
        cfg,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
    )?;
    gpu_spec.cfg.real_transformer.gpu_native = true;
    gpu_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Gpu;
    let model_identity = crate::greedy_parity_model_identity(&gpu_spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err("v2 qualification requires exact Qwen3-Coder 30B-A3B Q4_0 geometry".into());
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
        .ok_or("v2 qualification requires tokenizer.path")?;
    let tokenizer = crate::tokenizer::Tokenizer::from_file(tokenizer_path)
        .map(Arc::new)
        .map_err(|error| {
            format!(
                "v2 qualification failed to load tokenizer {}: {error}",
                tokenizer_path.display()
            )
        })?;

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
        return Err("v2 GPU runtime geometry drifted after strict startup".into());
    }
    let evidence = execute_reference_and_evidence(
        &reference_spec,
        tokenizer,
        &reference_resolved_config_sha256,
        &gpu,
        progress_watchdog,
    )
    .await?;
    let expected_routing_events =
        HOLDOUT_CORPUS_CASE_COUNT * OUTPUT_TOKEN_LIMIT * gpu.model_geometry.num_layers;
    if evidence.routing_events.len() != expected_routing_events
        || evidence.routing_counters.total_routing_events != expected_routing_events
        || evidence.same_input_router_events.len() != expected_routing_events
    {
        return Err(format!(
            "v2 observation coverage is incomplete: routing={} same-input={} expected={expected_routing_events}",
            evidence.routing_events.len(),
            evidence.same_input_router_events.len()
        )
        .into());
    }
    let (_, executable_sha256) = crate::current_executable_identity()?;
    let runtime_completion = RuntimeCompletionEvidence {
        gpu_expected_completed_token_steps: gpu.expected_completed_token_steps,
        gpu_token_loop: gpu.token_loop_snapshot,
        reference_routed_execution: evidence.reference_routed_execution,
        gpu_native_routed_execution: gpu.routed_execution,
        reference_nonfinite_attention_fallbacks: evidence.reference_nonfinite_attention_fallbacks,
        gpu_native_nonfinite_attention_fallbacks: gpu.nonfinite_attention_fallbacks,
    };
    let report = SemanticParityV2Report::new(
        expected_adapter_name.clone(),
        DiagnosticProvenance {
            build,
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
        evidence.token_cases,
        evidence.routing_events,
        evidence.routing_counters,
        evidence.sampling,
        evidence.same_input_router_events,
        evidence.same_input_events,
        evidence.permutation_events,
        runtime_completion,
        Vec::new(),
    );
    let qualification_pass = report.qualification_pass;
    let failed_criteria = report.failed_criteria.join(", ");
    emit_report(&report, &report_out)?;
    if qualification_pass {
        Ok(())
    } else {
        Err(format!("v2 qualification FAIL: {failed_criteria}").into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(label: &str) -> crate::qualification::ArtifactDigest {
        crate::qualification::ArtifactDigest {
            configured_path: format!("/{label}"),
            canonical_path: format!("/canonical/{label}"),
            byte_length: 1,
            sha256: "a".repeat(64),
        }
    }

    fn strict_model_load() -> crate::greedy_parity::ModelLoadEvidence {
        crate::greedy_parity::ModelLoadEvidence {
            strict: true,
            loader: "safetensors".to_string(),
            loaded_tensors: 435,
            required_tensors: 435,
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

    fn passing_provenance() -> DiagnosticProvenance {
        DiagnosticProvenance {
            build: crate::qualification::BuildProvenance {
                git_sha: Some("b".repeat(40)),
                dirty: Some(false),
                package_version: "0.1.0".to_string(),
            },
            executable_sha256: "c".repeat(64),
            artifacts: crate::qualification::QualificationArtifacts {
                config: Some(artifact("config.toml")),
                tokenizer: Some(artifact("tokenizer.json")),
                expert_metadata: Some(artifact("metadata.json")),
                ..crate::qualification::QualificationArtifacts::default()
            },
            gpu_resolved_config_sha256: "d".repeat(64),
            reference_resolved_config_sha256: "e".repeat(64),
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
                device_id: 0x27b8,
                device_type: "DiscreteGpu".to_string(),
                wgpu_backend: "vulkan".to_string(),
                driver: "580.173.02".to_string(),
                driver_info: "test".to_string(),
                compute_plane: "wgpu-vulkan".to_string(),
                software_adapter: false,
            },
        }
    }

    fn passing_token_cases() -> Vec<TokenCaseEvidence> {
        HOLDOUT_CORPUS
            .iter()
            .enumerate()
            .map(|(case_index, case)| {
                let ids = (0..OUTPUT_TOKEN_LIMIT)
                    .map(|position| (case_index * OUTPUT_TOKEN_LIMIT + position) as u32)
                    .collect::<Vec<_>>();
                TokenCaseEvidence::new(case.name, ids.clone(), ids)
            })
            .collect()
    }

    fn router_same_input(cpu: &[u32], gpu: &[u32]) -> RouterSameInputEvidence {
        RouterSameInputEvidence {
            cpu_selected_ids: cpu.to_vec(),
            gpu_selected_ids: gpu.to_vec(),
            membership_equal: same_membership(cpu, gpu),
            ordered_ids_equal: cpu == gpu,
            common_expert_score_deltas: Vec::new(),
            common_expert_weight_deltas: Vec::new(),
        }
    }

    fn same_input_router_event(index: usize) -> SameInputRouterEventEvidence {
        let ids = [0, 1, 2, 3, 4, 5, 6, 7];
        SameInputRouterEventEvidence {
            location: EventLocation::new("rust-ownership-holdout", index / 48, index % 48),
            cpu_production_router_on_exact_gpu_input: Some(router_same_input(&ids, &ids)),
            evaluation_error: None,
        }
    }

    fn routing_event(index: usize) -> RoutingEventEvidence {
        RoutingEventEvidence {
            location: EventLocation::new("rust-ownership-holdout", index / 48, index % 48),
            classification: RoutingClassification::ExactOrderMatch,
            invalid_reason: None,
            selected_membership_equal: Some(true),
            reference_selected_ids: vec![0, 1, 2, 3, 4, 5, 6, 7],
            gpu_selected_ids: vec![0, 1, 2, 3, 4, 5, 6, 7],
            displacement: RankDisplacementEvidence::default(),
            expert_weight_pairing_defect: false,
            canonical_selected_weights: None,
            boundary: None,
            numerically_selected: index < 48,
            numerical_selection_reason: (index < 48).then_some(EXACT_ORDER_SAMPLING_RULE),
        }
    }

    fn numerical_event(
        layer: usize,
        classification: RoutingClassification,
    ) -> SameInputNumericalEventEvidence {
        SameInputNumericalEventEvidence {
            location: EventLocation::new("rust-ownership-holdout", 0, layer),
            classification,
            cpu_vs_gpu_routed_moe: VectorNumericalEvidence::compare("cpu", "gpu", &[0.0], &[0.0])
                .unwrap(),
            reference_vs_gpu_routed_moe_includes_upstream_drift: VectorNumericalEvidence::compare(
                "reference",
                "gpu",
                &[0.0],
                &[0.0],
            )
            .unwrap(),
            router_same_input: router_same_input(&[0, 1], &[0, 1]),
        }
    }

    fn numerical_with_metrics(
        max_absolute: f64,
        rms: f64,
        mean_absolute: f64,
        nonfinite_mismatches: usize,
    ) -> SameInputNumericalEventEvidence {
        let mut event = numerical_event(0, RoutingClassification::ExactOrderMatch);
        event.cpu_vs_gpu_routed_moe.max_absolute_error = Some(max_absolute);
        event.cpu_vs_gpu_routed_moe.rms_error = Some(rms);
        event.cpu_vs_gpu_routed_moe.mean_absolute_error = Some(mean_absolute);
        event.cpu_vs_gpu_routed_moe.nonfinite_bit_mismatch_count = nonfinite_mismatches;
        event
    }

    fn passing_routing_counters() -> RoutingCounters {
        RoutingCounters {
            total_routing_events: HOLDOUT_CORPUS_CASE_COUNT * OUTPUT_TOKEN_LIMIT * 48,
            exact_order_match_events: HOLDOUT_CORPUS_CASE_COUNT * OUTPUT_TOKEN_LIMIT * 48,
            ..RoutingCounters::default()
        }
    }

    fn passing_sampling() -> SamplingSummary {
        SamplingSummary {
            exact_order_events_total: HOLDOUT_CORPUS_CASE_COUNT * OUTPUT_TOKEN_LIMIT * 48,
            exact_order_events_numerically_sampled: 48,
            exact_order_sampling_rule: EXACT_ORDER_SAMPLING_RULE,
            layers_with_exact_order_sample: (0..48).collect(),
            layers_without_exact_order_sample: Vec::new(),
            internal_permutation_events_numerically_measured: 0,
            membership_mismatch_events_numerically_measured: 0,
            maximum_exact_order_samples_for_model: 48,
        }
    }

    fn passing_runtime() -> RuntimeCompletionEvidence {
        RuntimeCompletionEvidence {
            gpu_expected_completed_token_steps: 64,
            gpu_token_loop: GpuNativeTokenLoopSnapshot {
                tokens_completed: 64,
                ..GpuNativeTokenLoopSnapshot::default()
            },
            ..RuntimeCompletionEvidence::default()
        }
    }

    fn all_gates_pass() -> QualificationGates {
        QualificationGates {
            exact_provenance: true,
            frozen_holdout_corpus_identity: true,
            exact_model_geometry: true,
            strict_model_load: true,
            complete_exact_greedy_token_parity: true,
            routing_structural_validity: true,
            same_input_gpu_router_ordered_ids_exact: true,
            same_input_routed_moe_numerical_limits: true,
            deterministic_numerical_sampling_complete: true,
            no_fallback_or_degradation: true,
            residency_and_execution_completion: true,
            controlled_shutdown: true,
            observation_completeness: true,
        }
    }

    #[test]
    fn schema_mode_and_frozen_limits_are_exact() {
        assert_eq!(SCHEMA_VERSION, "mer.gpu-native-semantic-parity.v2");
        assert_eq!(MODE, "qualify-gpu-native-semantic-parity-v2");
        assert_eq!(MAX_ABSOLUTE_ERROR_LIMIT, 0.020);
        assert_eq!(RMS_ERROR_LIMIT, 0.001);
        assert_eq!(MEAN_ABSOLUTE_ERROR_LIMIT, 0.00075);
        assert_eq!(NONFINITE_MISMATCH_LIMIT, 0);
    }

    #[test]
    fn calibration_and_holdout_identities_are_distinct() {
        let calibration = CorpusIdentityEvidence::calibration();
        let holdout = CorpusIdentityEvidence::holdout();
        assert_ne!(calibration.id, holdout.id);
        assert_ne!(calibration.sha256, holdout.sha256);
        assert_eq!(calibration.sha256, CALIBRATION_CORPUS_SHA256);
    }

    #[test]
    fn holdout_case_order_prompts_and_token_limit_are_frozen() {
        assert_eq!(HOLDOUT_CORPUS.len(), 4);
        assert_eq!(OUTPUT_TOKEN_LIMIT, 16);
        assert_eq!(
            HOLDOUT_CORPUS.map(|case| case.name),
            [
                "rust-ownership-holdout",
                "python-async-holdout",
                "postgres-window-holdout",
                "spanish-refactor-holdout",
            ]
        );
        assert_eq!(
            HOLDOUT_CORPUS[0].prompt,
            "Fix the Rust function below so it compiles without cloning the input.\nReturn only the corrected function followed by one short explanation.\n\nfn first_word(s: String) -> &str {\n    s.split_whitespace().next().unwrap_or(\"\")\n}"
        );
        assert_eq!(
            HOLDOUT_CORPUS[1].prompt,
            "Find the concurrency bug in this Python asyncio code and provide the\nminimal corrected version.\n\nasync def load_all(urls):\n    tasks = []\n    for url in urls:\n        tasks.append(asyncio.create_task(fetch(url)))\n    for task in tasks:\n        return await task"
        );
        assert_eq!(
            HOLDOUT_CORPUS[2].prompt,
            "Write a PostgreSQL query that returns each customer's most recent order,\nincluding customer_id, order_id, ordered_at, and total, with exactly one\nrow per customer. Use a window function."
        );
        assert_eq!(
            HOLDOUT_CORPUS[3].prompt,
            "En Rust, refactoriza esta función para evitar una asignación innecesaria\nsin cambiar su comportamiento. Devuelve primero el código y luego una\nexplicación breve.\n\nfn normalize(name: &str) -> String {\n    let value = name.to_string();\n    value.trim().to_lowercase()\n}"
        );
    }

    #[test]
    fn holdout_sha_is_deterministic_and_frozen() {
        assert_eq!(holdout_corpus_sha256(), HOLDOUT_CORPUS_SHA256);
        assert_eq!(
            CorpusIdentityEvidence::holdout().sha256,
            HOLDOUT_CORPUS_SHA256
        );
    }

    #[test]
    fn token_mismatch_and_incomplete_generation_fail() {
        let mut mismatch = passing_token_cases();
        mismatch[0].gpu_generated_token_ids[3] ^= 1;
        mismatch[0] = TokenCaseEvidence::new(
            mismatch[0].case.clone(),
            mismatch[0].reference_generated_token_ids.clone(),
            mismatch[0].gpu_generated_token_ids.clone(),
        );
        assert!(!token_parity_pass(&mismatch));

        let mut incomplete = passing_token_cases();
        incomplete[1].gpu_generated_token_ids.pop();
        incomplete[1] = TokenCaseEvidence::new(
            incomplete[1].case.clone(),
            incomplete[1].reference_generated_token_ids.clone(),
            incomplete[1].gpu_generated_token_ids.clone(),
        );
        assert!(!token_parity_pass(&incomplete));
    }

    #[test]
    fn invalid_duplicate_out_of_range_and_nonfinite_routing_fail() {
        use crate::gpu_native_semantic_parity_corpus::classify_routing_event;
        assert_eq!(
            classify_routing_event(&[1, 1], &[1, 2], 128, true),
            RoutingClassification::InvalidNonfiniteIncomplete
        );
        assert_eq!(
            classify_routing_event(&[1, 128], &[1, 2], 128, true),
            RoutingClassification::InvalidNonfiniteIncomplete
        );
        assert_eq!(
            classify_routing_event(&[1, 2], &[1, 2], 128, false),
            RoutingClassification::InvalidNonfiniteIncomplete
        );
    }

    #[test]
    fn pairing_defect_fails_structural_gate() {
        let mut counters = passing_routing_counters();
        counters.expert_weight_pairing_defect_events = 1;
        assert!(!routing_structural_pass(
            &counters,
            HOLDOUT_CORPUS_CASE_COUNT * OUTPUT_TOKEN_LIMIT * 48
        ));
    }

    #[test]
    fn same_input_ordered_match_passes_router_gate() {
        let event = SameInputRouterEventEvidence {
            location: EventLocation::new("case", 0, 0),
            cpu_production_router_on_exact_gpu_input: Some(router_same_input(
                &[1, 2, 3],
                &[1, 2, 3],
            )),
            evaluation_error: None,
        };
        assert!(same_input_router_pass(&[event], 1));
    }

    #[test]
    fn same_input_membership_same_but_order_different_fails_router_gate() {
        let event = SameInputRouterEventEvidence {
            location: EventLocation::new("case", 0, 0),
            cpu_production_router_on_exact_gpu_input: Some(router_same_input(
                &[1, 2, 3],
                &[1, 3, 2],
            )),
            evaluation_error: None,
        };
        assert!(
            event
                .cpu_production_router_on_exact_gpu_input
                .as_ref()
                .unwrap()
                .membership_equal
        );
        assert!(!same_input_router_pass(&[event], 1));
    }

    #[test]
    fn same_input_membership_difference_fails_router_gate() {
        let event = SameInputRouterEventEvidence {
            location: EventLocation::new("case", 0, 0),
            cpu_production_router_on_exact_gpu_input: Some(router_same_input(
                &[1, 2, 3],
                &[1, 2, 4],
            )),
            evaluation_error: None,
        };
        assert!(!same_input_router_pass(&[event], 1));
    }

    #[test]
    fn cross_backend_membership_drift_is_not_an_independent_fail_gate() {
        let mut counters = passing_routing_counters();
        counters.exact_order_match_events -= 1;
        counters.membership_mismatch_events = 1;
        assert!(routing_structural_pass(
            &counters,
            HOLDOUT_CORPUS_CASE_COUNT * OUTPUT_TOKEN_LIMIT * 48
        ));
        let same_input = SameInputRouterEventEvidence {
            location: EventLocation::new("case", 0, 0),
            cpu_production_router_on_exact_gpu_input: Some(router_same_input(
                &[1, 2, 3],
                &[1, 2, 3],
            )),
            evaluation_error: None,
        };
        assert!(same_input_router_pass(&[same_input], 1));
    }

    #[test]
    fn numerical_max_absolute_boundary_is_inclusive_and_excess_fails() {
        assert!(numerical_event_within_limits(&numerical_with_metrics(
            0.020, 0.0, 0.0, 0
        )));
        assert!(!numerical_event_within_limits(&numerical_with_metrics(
            0.020_000_1,
            0.0,
            0.0,
            0
        )));
    }

    #[test]
    fn numerical_rms_boundary_is_inclusive_and_excess_fails() {
        assert!(numerical_event_within_limits(&numerical_with_metrics(
            0.0, 0.001, 0.0, 0
        )));
        assert!(!numerical_event_within_limits(&numerical_with_metrics(
            0.0,
            0.001_000_1,
            0.0,
            0
        )));
    }

    #[test]
    fn numerical_mean_absolute_boundary_is_inclusive_and_excess_fails() {
        assert!(numerical_event_within_limits(&numerical_with_metrics(
            0.0, 0.0, 0.00075, 0
        )));
        assert!(!numerical_event_within_limits(&numerical_with_metrics(
            0.0,
            0.0,
            0.000_750_1,
            0
        )));
    }

    #[test]
    fn numerical_nonfinite_mismatch_fails() {
        assert!(!numerical_event_within_limits(&numerical_with_metrics(
            0.0, 0.0, 0.0, 1
        )));
    }

    #[test]
    fn deterministic_sampler_selects_first_exact_per_layer_and_all_anomalies() {
        let mut sampler = DeterministicEventSampler::new(2);
        assert!(
            sampler
                .select(0, RoutingClassification::ExactOrderMatch)
                .selected
        );
        assert!(
            !sampler
                .select(0, RoutingClassification::ExactOrderMatch)
                .selected
        );
        assert!(
            sampler
                .select(0, RoutingClassification::MembershipMismatch)
                .selected
        );
        assert!(
            sampler
                .select(1, RoutingClassification::InternalRankPermutation)
                .selected
        );
        assert!(
            sampler
                .select(1, RoutingClassification::ExactOrderMatch)
                .selected
        );
        assert_eq!(sampler.layers_with_exact_sample(), vec![0, 1]);
    }

    #[test]
    fn every_membership_drift_and_internal_permutation_must_be_measured() {
        let routing = RoutingCounters {
            total_routing_events: 5,
            exact_order_match_events: 3,
            internal_rank_permutation_events: 1,
            membership_mismatch_events: 1,
            ..RoutingCounters::default()
        };
        let mut sampling = SamplingSummary {
            exact_order_events_total: 3,
            exact_order_events_numerically_sampled: 3,
            exact_order_sampling_rule: EXACT_ORDER_SAMPLING_RULE,
            layers_with_exact_order_sample: vec![0, 1, 2],
            layers_without_exact_order_sample: Vec::new(),
            internal_permutation_events_numerically_measured: 1,
            membership_mismatch_events_numerically_measured: 1,
            maximum_exact_order_samples_for_model: 3,
        };
        let numerical = vec![
            numerical_event(0, RoutingClassification::ExactOrderMatch),
            numerical_event(1, RoutingClassification::ExactOrderMatch),
            numerical_event(2, RoutingClassification::ExactOrderMatch),
            numerical_event(0, RoutingClassification::InternalRankPermutation),
            numerical_event(0, RoutingClassification::MembershipMismatch),
        ];
        let permutation = vec![PermutationNumericalEventEvidence {
            location: EventLocation::new("case", 0, 0),
            witness:
                crate::gpu_native_expert_permutation_semantic_parity::permutation_only_witness(
                    &[2, 1],
                    &[0.5, 0.5],
                    &[vec![1.0], vec![2.0]],
                )
                .unwrap(),
        }];
        assert!(numerical_sampling_complete(
            &routing,
            &sampling,
            &numerical,
            &permutation,
            3
        ));
        sampling.membership_mismatch_events_numerically_measured = 0;
        assert!(!numerical_sampling_complete(
            &routing,
            &sampling,
            &numerical,
            &permutation,
            3
        ));
        sampling.membership_mismatch_events_numerically_measured = 1;
        sampling.internal_permutation_events_numerically_measured = 0;
        assert!(!numerical_sampling_complete(
            &routing,
            &sampling,
            &numerical,
            &permutation,
            3
        ));
    }

    #[test]
    fn cpu_fallback_and_degraded_substitution_fail() {
        let mut runtime = passing_runtime();
        runtime
            .gpu_native_routed_execution
            .cpu_routed_expert_dispatches = 1;
        assert!(!no_fallback_or_degradation_pass(&runtime));
        runtime
            .gpu_native_routed_execution
            .cpu_routed_expert_dispatches = 0;
        runtime
            .gpu_native_routed_execution
            .degraded_expert_substitutions = 1;
        assert!(!no_fallback_or_degradation_pass(&runtime));
    }

    #[test]
    fn residency_failure_or_incomplete_execution_fails() {
        let mut runtime = passing_runtime();
        assert!(residency_and_execution_completion_pass(&runtime));
        runtime.gpu_token_loop.no_progress_failures = 1;
        assert!(!residency_and_execution_completion_pass(&runtime));
        runtime.gpu_token_loop.no_progress_failures = 0;
        runtime.gpu_native_routed_execution.gpu_dispatch_failures = 1;
        assert!(!residency_and_execution_completion_pass(&runtime));
        runtime.gpu_native_routed_execution.gpu_dispatch_failures = 0;
        runtime.gpu_token_loop.tokens_completed = 63;
        assert!(!residency_and_execution_completion_pass(&runtime));
    }

    #[test]
    fn both_controlled_shutdown_witnesses_are_required() {
        let mut provenance = passing_provenance();
        assert!(controlled_shutdown_pass(&provenance));
        provenance
            .reference_background_shutdown
            .all_runtime_resources_released = false;
        assert!(!controlled_shutdown_pass(&provenance));
        provenance.reference_background_shutdown = released_shutdown();
        provenance
            .gpu_native_background_shutdown
            .controlled_shutdown_requested = false;
        assert!(!controlled_shutdown_pass(&provenance));
    }

    #[test]
    fn qualification_pass_cannot_survive_any_failed_criterion() {
        let gates = all_gates_pass();
        assert!(gates.qualification_pass());
        macro_rules! assert_field_fails {
            ($field:ident) => {{
                let mut failed = gates.clone();
                failed.$field = false;
                assert!(!failed.qualification_pass(), stringify!($field));
            }};
        }
        assert_field_fails!(exact_provenance);
        assert_field_fails!(frozen_holdout_corpus_identity);
        assert_field_fails!(exact_model_geometry);
        assert_field_fails!(strict_model_load);
        assert_field_fails!(complete_exact_greedy_token_parity);
        assert_field_fails!(routing_structural_validity);
        assert_field_fails!(same_input_gpu_router_ordered_ids_exact);
        assert_field_fails!(same_input_routed_moe_numerical_limits);
        assert_field_fails!(deterministic_numerical_sampling_complete);
        assert_field_fails!(no_fallback_or_degradation);
        assert_field_fails!(residency_and_execution_completion);
        assert_field_fails!(controlled_shutdown);
        assert_field_fails!(observation_completeness);
    }

    #[test]
    fn complete_report_can_pass_only_from_frozen_v2_evidence() {
        let expected_events = HOLDOUT_CORPUS_CASE_COUNT * OUTPUT_TOKEN_LIMIT * 48;
        let report = SemanticParityV2Report::new(
            "NVIDIA L4".to_string(),
            passing_provenance(),
            passing_token_cases(),
            (0..expected_events).map(routing_event).collect(),
            passing_routing_counters(),
            passing_sampling(),
            (0..expected_events).map(same_input_router_event).collect(),
            (0..48)
                .map(|layer| numerical_event(layer, RoutingClassification::ExactOrderMatch))
                .collect(),
            Vec::new(),
            passing_runtime(),
            Vec::new(),
        );
        assert!(report.qualification_pass);
        assert_eq!(report.status, QualificationStatus::Pass);
        assert_eq!(report.schema, SCHEMA_VERSION);
        assert_eq!(
            report
                .global_summary
                .same_input_router
                .same_input_router_events_total,
            expected_events
        );
        assert_eq!(
            report.global_summary.cross_backend_membership_drift_events,
            0
        );
        assert!(report.failed_criteria.is_empty());
        assert!(
            !report
                .production_semantics
                .production_inference_math_changed
        );
        assert!(!report.production_semantics.production_shader_math_changed);
        assert!(
            !report
                .production_semantics
                .existing_v1_qualifier_semantics_changed
        );
    }
}
