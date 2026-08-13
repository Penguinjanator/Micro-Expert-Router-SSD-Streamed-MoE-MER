//! Fail-closed fixed-corpus CPU-versus-Hybrid greedy-token qualification.
//!
//! The corpus, comparison, evidence, and PASS derivation live here so they are
//! hardware-independent. `main.rs` owns the deliberately private runtime
//! factory that executes the existing real-model path on the selected device.

use crate::backend::{GpuDeviceIdentity, GpuExpertIoSnapshot, GpuExpertMemorySnapshot};
use crate::engine::RoutedExpertExecutionSnapshot;
use crate::qualification::{
    BuildProvenance, ExecutionPlanEvidence, ExpertMetadataEvidence, QualificationArtifacts,
    QualificationFailure, QualificationStatus,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

#[cfg(test)]
pub const LEGACY_SCHEMA_VERSION: &str = "mer.strict-hybrid-q4-greedy-parity.v1";
pub const SCHEMA_VERSION: &str = "mer.strict-hybrid-q4-greedy-parity.v2";
pub const MODE: &str = "strict-hybrid-q4-greedy-parity";
pub const CORPUS_ID: &str = "qwen3-coder-30b-a3b-greedy-v1";
pub const CORPUS_VERSION: u32 = 1;
pub const OUTPUT_TOKEN_LIMIT: usize = 16;
pub const CORPUS_CASE_COUNT: usize = 4;
pub const MAX_DECODED_PREFIX_BYTES: usize = 512;
pub const WORKER_PROTOCOL_VERSION: &str = "mer.strict-hybrid-q4-greedy-worker.v1";
pub const MAX_WORKER_STDOUT_BYTES: usize = 1024 * 1024;
pub const MAX_WORKER_STDERR_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedCorpusCase {
    pub name: &'static str,
    pub behavior: &'static str,
    pub prompt: &'static str,
}

pub const FIXED_CORPUS: [FixedCorpusCase; CORPUS_CASE_COUNT] = [
    FixedCorpusCase {
        name: "rust-generation",
        behavior: "Rust code generation",
        prompt: "Write a Rust function named is_even that accepts an i64 and returns a bool. Return only the function.",
    },
    FixedCorpusCase {
        name: "rust-debugging",
        behavior: "Rust code correction",
        prompt: "Fix this Rust code and return only the corrected code:\nfn add(a: i32, b: i32) -> i32 { a - b }",
    },
    FixedCorpusCase {
        name: "json-transformation",
        behavior: "Structured JSON transformation",
        prompt: "Transform this JSON into one compact JSON object with keys in alphabetical order: {\"z\":3,\"a\":1,\"m\":2}",
    },
    FixedCorpusCase {
        name: "multilingual-spanish",
        behavior: "Short multilingual instruction",
        prompt: "Responde en español con una frase breve: ¿Qué hace un compilador?",
    },
];

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

pub fn corpus_sha256() -> String {
    let mut hasher = Sha256::new();
    hash_len_prefixed(&mut hasher, CORPUS_ID.as_bytes());
    hasher.update(CORPUS_VERSION.to_le_bytes());
    hasher.update((OUTPUT_TOKEN_LIMIT as u64).to_le_bytes());
    for case in FIXED_CORPUS {
        hash_len_prefixed(&mut hasher, case.name.as_bytes());
        hash_len_prefixed(&mut hasher, case.behavior.as_bytes());
        hash_len_prefixed(&mut hasher, case.prompt.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub fn token_ids_sha256(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((ids.len() as u64).to_le_bytes());
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub fn fixed_case(name: &str) -> Option<FixedCorpusCase> {
    FIXED_CORPUS
        .iter()
        .copied()
        .find(|candidate| candidate.name == name)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CorpusEvidence {
    pub id: &'static str,
    pub version: u32,
    pub sha256: String,
    pub case_count: usize,
    pub output_token_limit: usize,
}

impl CorpusEvidence {
    pub fn fixed() -> Self {
        Self {
            id: CORPUS_ID,
            version: CORPUS_VERSION,
            sha256: corpus_sha256(),
            case_count: CORPUS_CASE_COUNT,
            output_token_limit: OUTPUT_TOKEN_LIMIT,
        }
    }

    fn is_exact(&self) -> bool {
        self.id == CORPUS_ID
            && self.version == CORPUS_VERSION
            && self.sha256 == corpus_sha256()
            && self.case_count == CORPUS_CASE_COUNT
            && self.output_token_limit == OUTPUT_TOKEN_LIMIT
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GreedySamplingEvidence {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub deterministic_greedy: bool,
}

/// Exact typed stdin contract between the aggregate orchestrator and one
/// short-lived Hybrid worker. It intentionally contains token IDs, not prompt
/// text, so the worker has no encoding surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridWorkerRequest {
    pub protocol_version: String,
    pub worker_id: String,
    pub case_name: String,
    pub prompt_sha256: String,
    pub resolved_config_sha256: String,
    pub expected_adapter_name: String,
    pub prompt_token_ids: Vec<u32>,
    pub prompt_token_ids_sha256: String,
    pub output_token_limit: usize,
    pub sampling: GreedySamplingEvidence,
    pub executable_sha256: String,
    pub build_git_sha: String,
}

impl HybridWorkerRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        worker_id: String,
        case: FixedCorpusCase,
        resolved_config_sha256: String,
        expected_adapter_name: String,
        prompt_token_ids: Vec<u32>,
        executable_sha256: String,
        build_git_sha: String,
    ) -> Self {
        Self {
            protocol_version: WORKER_PROTOCOL_VERSION.to_string(),
            worker_id,
            case_name: case.name.to_string(),
            prompt_sha256: sha256_hex(case.prompt.as_bytes()),
            resolved_config_sha256,
            expected_adapter_name,
            prompt_token_ids_sha256: token_ids_sha256(&prompt_token_ids),
            prompt_token_ids,
            output_token_limit: OUTPUT_TOKEN_LIMIT,
            sampling: GreedySamplingEvidence::fixed(),
            executable_sha256,
            build_git_sha,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridWorkerIdentityEvidence {
    pub protocol_version_verified: bool,
    pub case_identity_verified: bool,
    pub config_identity_verified: bool,
    pub expected_adapter_verified: bool,
    pub prompt_token_identity_verified: bool,
    pub output_token_limit_verified: bool,
    pub greedy_sampling_identity_verified: bool,
    pub executable_identity_verified: bool,
    pub build_sha_identity_verified: bool,
}

impl HybridWorkerIdentityEvidence {
    pub fn all_verified(&self) -> bool {
        self.protocol_version_verified
            && self.case_identity_verified
            && self.config_identity_verified
            && self.expected_adapter_verified
            && self.prompt_token_identity_verified
            && self.output_token_limit_verified
            && self.greedy_sampling_identity_verified
            && self.executable_identity_verified
            && self.build_sha_identity_verified
    }
}

pub fn validate_hybrid_worker_request(
    request: &HybridWorkerRequest,
    observed_config_sha256: &str,
    observed_executable_sha256: &str,
    observed_build_git_sha: Option<&str>,
) -> HybridWorkerIdentityEvidence {
    let fixed = fixed_case(&request.case_name);
    HybridWorkerIdentityEvidence {
        protocol_version_verified: request.protocol_version == WORKER_PROTOCOL_VERSION,
        case_identity_verified: fixed.is_some_and(|case| {
            request.prompt_sha256 == sha256_hex(case.prompt.as_bytes())
        }),
        config_identity_verified: is_sha256_hex(&request.resolved_config_sha256)
            && request.resolved_config_sha256 == observed_config_sha256,
        expected_adapter_verified: !request.expected_adapter_name.is_empty(),
        prompt_token_identity_verified: !request.prompt_token_ids.is_empty()
            && request.prompt_token_ids_sha256 == token_ids_sha256(&request.prompt_token_ids),
        output_token_limit_verified: request.output_token_limit == OUTPUT_TOKEN_LIMIT,
        greedy_sampling_identity_verified: request.sampling.is_exact(),
        executable_identity_verified: is_sha256_hex(&request.executable_sha256)
            && request.executable_sha256 == observed_executable_sha256,
        build_sha_identity_verified: observed_build_git_sha.is_some_and(|sha| {
            sha.len() == 40
                && sha.bytes().all(|byte| byte.is_ascii_hexdigit())
                && request.build_git_sha == sha
        }),
    }
}

impl GreedySamplingEvidence {
    pub const fn fixed() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            deterministic_greedy: true,
        }
    }

    fn is_exact(&self) -> bool {
        self.temperature == 0.0
            && self.top_p == 1.0
            && self.top_k == 0
            && self.seed == 0
            && self.deterministic_greedy
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelIdentityEvidence {
    pub architecture: String,
    pub num_layers: usize,
    pub num_experts_per_layer: u32,
    pub total_experts: u64,
    pub top_k: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub routed_expert_dtype: String,
}

impl ModelIdentityEvidence {
    pub(crate) fn is_qwen3_coder_30b_a3b_q4_0(&self) -> bool {
        self.architecture == "qwen3_moe"
            && self.num_layers == 48
            && self.num_experts_per_layer == 128
            && self.total_experts == 6_144
            && self.top_k == 8
            && self.d_model == 2_048
            && self.d_ff == 768
            && self.routed_expert_dtype == "q4_0"
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StrictHybridPreflightEvidence {
    pub real_transformer_enabled: bool,
    pub weights_dir_configured: bool,
    pub strict_weights: bool,
    pub allow_seeded_fallback: bool,
    pub allow_degraded_experts: bool,
    pub allow_attention_fallback: bool,
    pub allow_truncated_expert_payloads: bool,
    pub distributed_enabled: bool,
    pub gpu_cache_enabled: bool,
    pub gpu_expert_capacity_bytes: u64,
    pub requested_mode: String,
    pub routed_expert_dtype: String,
}

impl StrictHybridPreflightEvidence {
    fn is_exact(&self) -> bool {
        self.real_transformer_enabled
            && self.weights_dir_configured
            && self.strict_weights
            && !self.allow_seeded_fallback
            && !self.allow_degraded_experts
            && !self.allow_attention_fallback
            && !self.allow_truncated_expert_payloads
            && !self.distributed_enabled
            && self.gpu_cache_enabled
            && self.gpu_expert_capacity_bytes > 0
            && self.requested_mode == "hybrid"
            && self.routed_expert_dtype == "q4_0"
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelLoadEvidence {
    pub strict: bool,
    pub loader: String,
    pub loaded_tensors: usize,
    pub required_tensors: usize,
    pub optional_probed: usize,
    pub optional_loaded: usize,
    pub seeded_fallback_remained: bool,
}

impl ModelLoadEvidence {
    fn is_strict_complete(&self) -> bool {
        self.strict
            && self.loaded_tensors == self.required_tensors
            && self.required_tensors > 0
            && !self.seeded_fallback_remained
            && self.loader != "seeded"
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeCacheSnapshot {
    pub ram_entries: usize,
    pub ram_hits: u64,
    pub ram_misses: u64,
    pub bytes_read: u64,
    pub prefetch_completed: u64,
    pub predictor_observations: u64,
    pub logical_gpu_hits: u64,
    pub logical_gpu_misses: u64,
    pub logical_gpu_promotions: u64,
    pub logical_admitted_bytes: u64,
    pub logical_anchor_entries: usize,
    pub logical_lru_entries: usize,
    pub physical_entries: usize,
    pub physical_installs: u64,
    pub physical_evictions: u64,
    pub stale_retirements: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitialStateEvidence {
    pub context_id: String,
    pub resolved_config_sha256: String,
    pub kv_cache_count: usize,
    pub kv_sequence_lengths: Vec<usize>,
    pub all_kv_empty: bool,
    pub cache: RuntimeCacheSnapshot,
    pub routed: RoutedExpertExecutionSnapshot,
    pub gpu_io_available: bool,
    pub gpu_io: GpuExpertIoSnapshot,
}

impl InitialStateEvidence {
    fn is_clean(&self) -> bool {
        self.kv_cache_count > 0
            && self.kv_sequence_lengths.len() == self.kv_cache_count
            && self.all_kv_empty
            && self.kv_sequence_lengths.iter().all(|&len| len == 0)
            && self.cache == RuntimeCacheSnapshot::default()
            && self.routed == RoutedExpertExecutionSnapshot::default()
            && self.gpu_io == GpuExpertIoSnapshot::default()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TerminationReason {
    LengthLimit,
    /// Typed failure evidence can represent a stop-reason mismatch even
    /// though the fixed-corpus command currently always runs to its limit.
    #[allow(dead_code)]
    EndOfSequence,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationEvidence {
    pub prompt_token_ids_sha256: String,
    pub generated_token_ids: Vec<u32>,
    pub generated_token_ids_sha256: String,
    pub generated_text_sha256: String,
    pub generated_token_count: usize,
    pub termination_reason: TerminationReason,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackgroundShutdownEvidence {
    pub controlled_shutdown_requested: bool,
    pub all_runtime_resources_released: bool,
    pub poll_iterations: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlaneRunEvidence {
    pub plane: String,
    pub model_load: ModelLoadEvidence,
    pub execution_plan: ExecutionPlanEvidence,
    pub routed_expert_gpu_failure_policy: String,
    pub device: Option<GpuDeviceIdentity>,
    pub initial_state: InitialStateEvidence,
    pub generation: GenerationEvidence,
    pub routed_execution_delta: RoutedExpertExecutionSnapshot,
    pub gpu_io_delta: GpuExpertIoSnapshot,
    pub attention_softmax_nonfinite_fallbacks: u64,
    pub gpu_memory_before: Option<GpuExpertMemorySnapshot>,
    pub gpu_memory_after: Option<GpuExpertMemorySnapshot>,
    pub background_shutdown: BackgroundShutdownEvidence,
    /// Populated by the parent only after the worker has exited and been
    /// reaped. A worker cannot attest to its own process termination.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_process: Option<HybridWorkerProcessEvidence>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridWorkerProcessEvidence {
    pub worker_id: String,
    pub child_process_spawned: bool,
    pub process_id: Option<u32>,
    pub executable_sha256: String,
    pub build_git_sha: String,
    pub executable_identity_verified: bool,
    pub build_sha_identity_verified: bool,
    pub case_identity_verified: bool,
    pub config_identity_verified: bool,
    pub expected_adapter_identity_verified: bool,
    pub prompt_token_identity_verified: bool,
    pub output_token_limit_verified: bool,
    pub greedy_sampling_identity_verified: bool,
    pub normal_zero_exit: bool,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub process_reaped: bool,
    pub timed_out: bool,
    pub evidence_emitted: bool,
}

impl HybridWorkerProcessEvidence {
    fn is_exact(&self) -> bool {
        !self.worker_id.is_empty()
            && self.child_process_spawned
            && self.process_id.is_some()
            && is_sha256_hex(&self.executable_sha256)
            && self.build_git_sha.len() == 40
            && self
                .build_git_sha
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            && self.executable_identity_verified
            && self.build_sha_identity_verified
            && self.case_identity_verified
            && self.config_identity_verified
            && self.expected_adapter_identity_verified
            && self.prompt_token_identity_verified
            && self.output_token_limit_verified
            && self.greedy_sampling_identity_verified
            && self.normal_zero_exit
            && self.exit_code == Some(0)
            && self.signal.is_none()
            && self.process_reaped
            && !self.timed_out
            && self.evidence_emitted
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HybridWorkerFailureEvidence {
    pub worker_id: String,
    pub case_name: String,
    pub child_process_spawned: bool,
    pub process_id: Option<u32>,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub timed_out: bool,
    pub process_reaped: bool,
    pub evidence_emitted: bool,
    pub identity_validation_succeeded: bool,
    pub identity_validation: Option<HybridWorkerParentValidation>,
    pub stderr: String,
    pub stderr_truncated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridWorkerResponse {
    pub protocol_version: String,
    pub worker_id: String,
    pub case_name: String,
    pub prompt_sha256: String,
    pub resolved_config_sha256: String,
    pub expected_adapter_name: String,
    pub prompt_token_ids_sha256: String,
    pub output_token_limit: usize,
    pub sampling: GreedySamplingEvidence,
    pub executable_sha256: String,
    pub build_git_sha: String,
    pub identity: HybridWorkerIdentityEvidence,
    pub plane: Option<PlaneRunEvidence>,
    pub failure: Option<String>,
}

impl HybridWorkerResponse {
    pub fn from_request(
        request: &HybridWorkerRequest,
        observed_config_sha256: &str,
        observed_executable_sha256: &str,
        observed_build_git_sha: Option<&str>,
        identity: HybridWorkerIdentityEvidence,
        plane: Option<PlaneRunEvidence>,
        failure: Option<String>,
    ) -> Self {
        Self {
            protocol_version: WORKER_PROTOCOL_VERSION.to_string(),
            worker_id: request.worker_id.clone(),
            case_name: request.case_name.clone(),
            prompt_sha256: request.prompt_sha256.clone(),
            resolved_config_sha256: observed_config_sha256.to_string(),
            expected_adapter_name: request.expected_adapter_name.clone(),
            prompt_token_ids_sha256: request.prompt_token_ids_sha256.clone(),
            output_token_limit: request.output_token_limit,
            sampling: request.sampling,
            executable_sha256: observed_executable_sha256.to_string(),
            build_git_sha: observed_build_git_sha.unwrap_or_default().to_string(),
            identity,
            plane,
            failure,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct HybridWorkerParentValidation {
    pub executable_identity_verified: bool,
    pub build_sha_identity_verified: bool,
    pub case_identity_verified: bool,
    pub config_identity_verified: bool,
    pub expected_adapter_identity_verified: bool,
    pub prompt_token_identity_verified: bool,
    pub output_token_limit_verified: bool,
    pub greedy_sampling_identity_verified: bool,
}

impl HybridWorkerParentValidation {
    pub fn all_verified(self) -> bool {
        self.executable_identity_verified
            && self.build_sha_identity_verified
            && self.case_identity_verified
            && self.config_identity_verified
            && self.expected_adapter_identity_verified
            && self.prompt_token_identity_verified
            && self.output_token_limit_verified
            && self.greedy_sampling_identity_verified
    }
}

pub fn validate_hybrid_worker_response(
    request: &HybridWorkerRequest,
    response: &HybridWorkerResponse,
) -> HybridWorkerParentValidation {
    HybridWorkerParentValidation {
        executable_identity_verified: response.identity.executable_identity_verified
            && response.executable_sha256 == request.executable_sha256,
        build_sha_identity_verified: response.identity.build_sha_identity_verified
            && response.build_git_sha == request.build_git_sha,
        case_identity_verified: response.identity.protocol_version_verified
            && response.protocol_version == WORKER_PROTOCOL_VERSION
            && response.worker_id == request.worker_id
            && response.case_name == request.case_name
            && response.prompt_sha256 == request.prompt_sha256,
        config_identity_verified: response.identity.config_identity_verified
            && response.resolved_config_sha256 == request.resolved_config_sha256,
        expected_adapter_identity_verified: response.identity.expected_adapter_verified
            && response.expected_adapter_name == request.expected_adapter_name,
        prompt_token_identity_verified: response.identity.prompt_token_identity_verified
            && response.prompt_token_ids_sha256 == request.prompt_token_ids_sha256,
        output_token_limit_verified: response.identity.output_token_limit_verified
            && response.output_token_limit == request.output_token_limit,
        greedy_sampling_identity_verified: response.identity.greedy_sampling_identity_verified
            && response.sampling == request.sampling,
    }
}

pub fn parse_hybrid_worker_request_exact(bytes: &[u8]) -> Result<HybridWorkerRequest, String> {
    parse_exact_json(bytes, "worker request")
}

pub fn parse_hybrid_worker_response_exact(bytes: &[u8]) -> Result<HybridWorkerResponse, String> {
    parse_exact_json(bytes, "worker response")
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
        .map_err(|error| format!("trailing or duplicated {label} output: {error}"))?;
    Ok(value)
}

pub fn hybrid_worker_id(
    build_git_sha: &str,
    executable_sha256: &str,
    case_index: usize,
    case_name: &str,
) -> String {
    let mut hasher = Sha256::new();
    hash_len_prefixed(&mut hasher, build_git_sha.as_bytes());
    hash_len_prefixed(&mut hasher, executable_sha256.as_bytes());
    hasher.update((case_index as u64).to_le_bytes());
    hash_len_prefixed(&mut hasher, case_name.as_bytes());
    format!("greedy-hybrid-{:x}", hasher.finalize())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FirstDivergenceEvidence {
    pub position: usize,
    pub cpu_token_id: Option<u32>,
    pub hybrid_token_id: Option<u32>,
    pub cpu_decoded_prefix: String,
    pub hybrid_decoded_prefix: String,
    pub cpu_prefix_truncated: bool,
    pub hybrid_prefix_truncated: bool,
    pub cpu_generated_length: usize,
    pub hybrid_generated_length: usize,
    pub cpu_termination_reason: TerminationReason,
    pub hybrid_termination_reason: TerminationReason,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CaseComparisonEvidence {
    pub exact_token_ids: bool,
    pub equal_generated_count: bool,
    pub equal_termination_reason: bool,
    pub equal_generated_text_hash: bool,
    pub first_divergence: Option<FirstDivergenceEvidence>,
}

fn bounded_utf8_prefix(value: String) -> (String, bool) {
    if value.len() <= MAX_DECODED_PREFIX_BYTES {
        return (value, false);
    }
    let mut end = MAX_DECODED_PREFIX_BYTES;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    (value[..end].to_string(), true)
}

pub fn compare_generations<F>(
    cpu: &GenerationEvidence,
    hybrid: &GenerationEvidence,
    mut decode: F,
) -> Result<CaseComparisonEvidence, String>
where
    F: FnMut(&[u32]) -> Result<String, String>,
{
    let equal_generated_count = cpu.generated_token_count == hybrid.generated_token_count;
    let equal_termination_reason = cpu.termination_reason == hybrid.termination_reason;
    let equal_generated_text_hash = cpu.generated_text_sha256 == hybrid.generated_text_sha256;
    let exact_token_ids = cpu.generated_token_ids == hybrid.generated_token_ids;

    let any_divergence = !exact_token_ids
        || !equal_generated_count
        || !equal_termination_reason
        || !equal_generated_text_hash;
    let first_divergence = if !any_divergence {
        None
    } else {
        let common = cpu
            .generated_token_ids
            .len()
            .min(hybrid.generated_token_ids.len());
        let position = (0..common)
            .find(|&index| cpu.generated_token_ids[index] != hybrid.generated_token_ids[index])
            .unwrap_or(common);
        let cpu_end = (position + 1).min(cpu.generated_token_ids.len());
        let hybrid_end = (position + 1).min(hybrid.generated_token_ids.len());
        let (cpu_decoded_prefix, cpu_prefix_truncated) =
            bounded_utf8_prefix(decode(&cpu.generated_token_ids[..cpu_end])?);
        let (hybrid_decoded_prefix, hybrid_prefix_truncated) =
            bounded_utf8_prefix(decode(&hybrid.generated_token_ids[..hybrid_end])?);
        Some(FirstDivergenceEvidence {
            position,
            cpu_token_id: cpu.generated_token_ids.get(position).copied(),
            hybrid_token_id: hybrid.generated_token_ids.get(position).copied(),
            cpu_decoded_prefix,
            hybrid_decoded_prefix,
            cpu_prefix_truncated,
            hybrid_prefix_truncated,
            cpu_generated_length: cpu.generated_token_ids.len(),
            hybrid_generated_length: hybrid.generated_token_ids.len(),
            cpu_termination_reason: cpu.termination_reason,
            hybrid_termination_reason: hybrid.termination_reason,
        })
    };

    Ok(CaseComparisonEvidence {
        exact_token_ids,
        equal_generated_count,
        equal_termination_reason,
        equal_generated_text_hash,
        first_divergence,
    })
}

fn comparison_matches_generations(
    reference: &GenerationEvidence,
    hybrid: &GenerationEvidence,
    comparison: &CaseComparisonEvidence,
) -> bool {
    let exact_token_ids = reference.generated_token_ids == hybrid.generated_token_ids;
    let equal_generated_count =
        reference.generated_token_count == hybrid.generated_token_count;
    let equal_termination_reason =
        reference.termination_reason == hybrid.termination_reason;
    let equal_generated_text_hash =
        reference.generated_text_sha256 == hybrid.generated_text_sha256;
    if comparison.exact_token_ids != exact_token_ids
        || comparison.equal_generated_count != equal_generated_count
        || comparison.equal_termination_reason != equal_termination_reason
        || comparison.equal_generated_text_hash != equal_generated_text_hash
    {
        return false;
    }
    let diverged = !exact_token_ids
        || !equal_generated_count
        || !equal_termination_reason
        || !equal_generated_text_hash;
    match (&comparison.first_divergence, diverged) {
        (None, false) => true,
        (Some(divergence), true) => {
            let common = reference
                .generated_token_ids
                .len()
                .min(hybrid.generated_token_ids.len());
            let position = (0..common)
                .find(|&index| {
                    reference.generated_token_ids[index] != hybrid.generated_token_ids[index]
                })
                .unwrap_or(common);
            divergence.position == position
                && divergence.cpu_token_id
                    == reference.generated_token_ids.get(position).copied()
                && divergence.hybrid_token_id
                    == hybrid.generated_token_ids.get(position).copied()
                && divergence.cpu_generated_length == reference.generated_token_ids.len()
                && divergence.hybrid_generated_length == hybrid.generated_token_ids.len()
                && divergence.cpu_termination_reason == reference.termination_reason
                && divergence.hybrid_termination_reason == hybrid.termination_reason
                && divergence.cpu_decoded_prefix.len() <= MAX_DECODED_PREFIX_BYTES
                && divergence.hybrid_decoded_prefix.len() <= MAX_DECODED_PREFIX_BYTES
        }
        _ => false,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CaseReport {
    pub name: &'static str,
    pub behavior: &'static str,
    pub prompt: &'static str,
    pub prompt_sha256: String,
    pub prompt_token_ids: Vec<u32>,
    pub prompt_token_ids_sha256: String,
    pub tokenization_calls: u32,
    pub output_token_limit: usize,
    pub cpu: Option<PlaneRunEvidence>,
    pub boundary_reference: Option<PlaneRunEvidence>,
    pub hybrid: Option<PlaneRunEvidence>,
    pub ordinary_cpu_vs_hybrid: Option<CaseComparisonEvidence>,
    pub boundary_reference_vs_hybrid: Option<CaseComparisonEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_failure: Option<HybridWorkerFailureEvidence>,
    pub failure: Option<QualificationFailure>,
}

impl CaseReport {
    pub fn new(case: FixedCorpusCase, prompt_token_ids: Vec<u32>) -> Self {
        let prompt_token_ids_sha256 = token_ids_sha256(&prompt_token_ids);
        Self {
            name: case.name,
            behavior: case.behavior,
            prompt: case.prompt,
            prompt_sha256: sha256_hex(case.prompt.as_bytes()),
            prompt_token_ids,
            prompt_token_ids_sha256,
            tokenization_calls: 1,
            output_token_limit: OUTPUT_TOKEN_LIMIT,
            cpu: None,
            boundary_reference: None,
            hybrid: None,
            ordinary_cpu_vs_hybrid: None,
            boundary_reference_vs_hybrid: None,
            worker_failure: None,
            failure: None,
        }
    }
}

pub(crate) fn cpu_plan_exact(plan: &ExecutionPlanEvidence) -> bool {
    plan.requested == "cpu"
        && plan.resolved == "cpu"
        && plan.embeddings == "cpu"
        && plan.lm_head == "cpu"
        && plan.dense_projections == "cpu"
        && plan.attention == "cpu"
        && plan.kv == "cpu"
        && plan.router == "cpu"
        && plan.routed_experts == "cpu"
        && plan.routed_expert_dtype == "q4_0"
        && !plan.fallback_occurred
}

pub(crate) fn hybrid_plan_exact(plan: &ExecutionPlanEvidence) -> bool {
    plan.requested == "hybrid"
        && plan.resolved == "hybrid-cpu-attention-gpu-experts"
        && plan.embeddings == "cpu"
        && plan.lm_head == "cpu"
        && plan.dense_projections == "cpu"
        && plan.attention == "cpu"
        && plan.kv == "cpu"
        && plan.router == "cpu"
        && plan.routed_experts == "gpu"
        && plan.routed_expert_dtype == "q4_0"
        && !plan.fallback_occurred
}

fn cpu_counters_exact(routed: RoutedExpertExecutionSnapshot) -> bool {
    routed.selected_routed_experts > 0
        && routed.cpu_routed_expert_dispatches == routed.selected_routed_experts
        && routed.gpu_dispatch_attempts == 0
        && routed.gpu_dispatch_successes == 0
        && routed.gpu_dispatch_failures == 0
        && routed.gpu_cpu_fallbacks == 0
        && routed.degraded_expert_substitutions == 0
}

fn hybrid_counters_exact(routed: RoutedExpertExecutionSnapshot) -> bool {
    routed.selected_routed_experts > 0
        && routed.gpu_dispatch_attempts == routed.selected_routed_experts
        && routed.gpu_dispatch_successes == routed.gpu_dispatch_attempts
        && routed.gpu_dispatch_failures == 0
        && routed.cpu_routed_expert_dispatches == 0
        && routed.gpu_cpu_fallbacks == 0
        && routed.degraded_expert_substitutions == 0
}

fn gpu_io_observed(io: GpuExpertIoSnapshot) -> bool {
    io.hidden_state_uploads > 0
        && io.hidden_state_upload_bytes > 0
        && io.queue_submissions > 0
        && io.map_requests > 0
        && io.readback_completions > 0
        && io.readback_bytes > 0
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct GreedyParityChecks {
    pub clean_build: bool,
    pub strict_real_checkpoint: bool,
    pub qwen3_coder_model_identity: bool,
    pub canonical_q4_0_layout: bool,
    pub fixed_corpus_identity: bool,
    pub fixed_greedy_sampling: bool,
    pub resolved_config_identity: bool,
    pub tokenized_once_and_shared: bool,
    pub fresh_state_every_run: bool,
    pub unique_execution_context_every_run: bool,
    pub controlled_background_shutdown_every_run: bool,
    pub hybrid_worker_processes_exact: bool,
    pub unique_hybrid_worker_identity: bool,
    pub unique_hybrid_process_identity: bool,
    pub cpu_plans_exact: bool,
    pub cpu_counter_invariants: bool,
    pub cpu_gpu_io_zero: bool,
    pub hybrid_plans_exact: bool,
    pub hardware_adapter_exact_match: bool,
    pub software_adapter_false: bool,
    pub strict_gpu_failure_policy: bool,
    pub hybrid_counter_invariants: bool,
    pub hybrid_gpu_io_observed: bool,
    pub hybrid_memory_ledger_valid: bool,
    pub zero_gpu_failures_fallbacks_or_degraded: bool,
    pub ordinary_cpu_comparison_recorded: bool,
    pub boundary_reference_exact_generated_token_ids: bool,
    pub boundary_reference_identical_generated_token_count: bool,
    pub boundary_reference_identical_termination_reason: bool,
    pub boundary_reference_identical_generated_text_hash: bool,
}

impl GreedyParityChecks {
    fn passes(&self) -> bool {
        self.clean_build
            && self.strict_real_checkpoint
            && self.qwen3_coder_model_identity
            && self.canonical_q4_0_layout
            && self.fixed_corpus_identity
            && self.fixed_greedy_sampling
            && self.resolved_config_identity
            && self.tokenized_once_and_shared
            && self.fresh_state_every_run
            && self.unique_execution_context_every_run
            && self.controlled_background_shutdown_every_run
            && self.hybrid_worker_processes_exact
            && self.unique_hybrid_worker_identity
            && self.unique_hybrid_process_identity
            && self.cpu_plans_exact
            && self.cpu_counter_invariants
            && self.cpu_gpu_io_zero
            && self.hybrid_plans_exact
            && self.hardware_adapter_exact_match
            && self.software_adapter_false
            && self.strict_gpu_failure_policy
            && self.hybrid_counter_invariants
            && self.hybrid_gpu_io_observed
            && self.hybrid_memory_ledger_valid
            && self.zero_gpu_failures_fallbacks_or_degraded
            && self.ordinary_cpu_comparison_recorded
            && self.boundary_reference_exact_generated_token_ids
            && self.boundary_reference_identical_generated_token_count
            && self.boundary_reference_identical_termination_reason
            && self.boundary_reference_identical_generated_text_hash
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GreedyParityReport {
    pub schema_version: &'static str,
    pub mode: &'static str,
    pub status: QualificationStatus,
    pub failure: Option<QualificationFailure>,
    pub provenance: BuildProvenance,
    pub artifacts: QualificationArtifacts,
    pub expert_metadata: Option<ExpertMetadataEvidence>,
    pub model_identity: Option<ModelIdentityEvidence>,
    pub source_preflight: Option<StrictHybridPreflightEvidence>,
    pub resolved_config_sha256: Option<String>,
    pub orchestrator_executable_sha256: Option<String>,
    pub corpus: CorpusEvidence,
    pub sampling: GreedySamplingEvidence,
    pub expected_adapter_name: String,
    pub cases: Vec<CaseReport>,
    pub checks: GreedyParityChecks,
}

impl GreedyParityReport {
    pub fn new(
        provenance: BuildProvenance,
        artifacts: QualificationArtifacts,
        expert_metadata: Option<ExpertMetadataEvidence>,
        expected_adapter_name: String,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            mode: MODE,
            status: QualificationStatus::Fail,
            failure: None,
            provenance,
            artifacts,
            expert_metadata,
            model_identity: None,
            source_preflight: None,
            resolved_config_sha256: None,
            orchestrator_executable_sha256: None,
            corpus: CorpusEvidence::fixed(),
            sampling: GreedySamplingEvidence::fixed(),
            expected_adapter_name,
            cases: Vec::new(),
            checks: GreedyParityChecks::default(),
        }
    }

    pub fn fail(&mut self, failure: QualificationFailure) {
        self.status = QualificationStatus::Fail;
        self.failure = Some(failure);
    }

    fn derive_checks(&self) -> GreedyParityChecks {
        let cases_complete = self.cases.len() == CORPUS_CASE_COUNT
            && self.cases.iter().zip(FIXED_CORPUS).all(|(report, fixed)| {
                report.name == fixed.name
                    && report.behavior == fixed.behavior
                    && report.prompt == fixed.prompt
                    && report.prompt_sha256 == sha256_hex(fixed.prompt.as_bytes())
                    && report.output_token_limit == OUTPUT_TOKEN_LIMIT
                    && report.worker_failure.is_none()
                    && report.failure.is_none()
                    && report.cpu.is_some()
                    && report.boundary_reference.is_some()
                    && report.hybrid.is_some()
                    && report.ordinary_cpu_vs_hybrid.is_some()
                    && report.boundary_reference_vs_hybrid.is_some()
            });
        let runs = self.cases.iter().filter_map(|case| {
            Some((
                case,
                case.cpu.as_ref()?,
                case.boundary_reference.as_ref()?,
                case.hybrid.as_ref()?,
                case.ordinary_cpu_vs_hybrid.as_ref()?,
                case.boundary_reference_vs_hybrid.as_ref()?,
            ))
        });
        let collected: Vec<_> = runs.collect();
        let exact_run_count = cases_complete && collected.len() == CORPUS_CASE_COUNT;
        let config_hash = self.resolved_config_sha256.as_deref();

        let strict_real_checkpoint = exact_run_count
            && self
                .source_preflight
                .as_ref()
                .is_some_and(StrictHybridPreflightEvidence::is_exact)
            && collected.iter().all(|(_, cpu, boundary, hybrid, _, _)| {
                cpu.model_load.is_strict_complete()
                    && boundary.model_load.is_strict_complete()
                    && hybrid.model_load.is_strict_complete()
            });
        let resolved_config_identity = exact_run_count
            && config_hash.is_some_and(is_sha256_hex)
            && collected.iter().all(|(_, cpu, boundary, hybrid, _, _)| {
                Some(cpu.initial_state.resolved_config_sha256.as_str()) == config_hash
                    && Some(boundary.initial_state.resolved_config_sha256.as_str()) == config_hash
                    && Some(hybrid.initial_state.resolved_config_sha256.as_str()) == config_hash
            });
        let tokenized_once_and_shared = exact_run_count
            && collected.iter().all(|(case, cpu, boundary, hybrid, _, _)| {
                !case.prompt_token_ids.is_empty()
                    && case.tokenization_calls == 1
                    && case.prompt_token_ids_sha256 == token_ids_sha256(&case.prompt_token_ids)
                    && cpu.generation.prompt_token_ids_sha256 == case.prompt_token_ids_sha256
                    && boundary.generation.prompt_token_ids_sha256
                        == case.prompt_token_ids_sha256
                    && hybrid.generation.prompt_token_ids_sha256 == case.prompt_token_ids_sha256
                    && cpu.generation.generated_token_count == OUTPUT_TOKEN_LIMIT
                    && boundary.generation.generated_token_count == OUTPUT_TOKEN_LIMIT
                    && hybrid.generation.generated_token_count == OUTPUT_TOKEN_LIMIT
                    && cpu.generation.generated_token_count
                        == cpu.generation.generated_token_ids.len()
                    && boundary.generation.generated_token_count
                        == boundary.generation.generated_token_ids.len()
                    && hybrid.generation.generated_token_count
                        == hybrid.generation.generated_token_ids.len()
                    && cpu.generation.generated_token_ids_sha256
                        == token_ids_sha256(&cpu.generation.generated_token_ids)
                    && boundary.generation.generated_token_ids_sha256
                        == token_ids_sha256(&boundary.generation.generated_token_ids)
                    && hybrid.generation.generated_token_ids_sha256
                        == token_ids_sha256(&hybrid.generation.generated_token_ids)
                    && is_sha256_hex(&cpu.generation.generated_text_sha256)
                    && is_sha256_hex(&boundary.generation.generated_text_sha256)
                    && is_sha256_hex(&hybrid.generation.generated_text_sha256)
                    && cpu.generation.termination_reason == TerminationReason::LengthLimit
                    && boundary.generation.termination_reason == TerminationReason::LengthLimit
                    && hybrid.generation.termination_reason == TerminationReason::LengthLimit
            });
        let fresh_state_every_run = exact_run_count
            && collected.iter().all(|(_, cpu, boundary, hybrid, _, _)| {
                cpu.initial_state.is_clean()
                    && boundary.initial_state.is_clean()
                    && hybrid.initial_state.is_clean()
                    && cpu.initial_state.context_id == cpu.execution_plan.context_id
                    && boundary.initial_state.context_id == boundary.execution_plan.context_id
                    && hybrid.initial_state.context_id == hybrid.execution_plan.context_id
            });
        // CPU and boundary-reference contexts are created in this parent
        // process, so their process-local IDs must be unique as bare IDs.
        let parent_context_ids: HashSet<&str> = collected
            .iter()
            .flat_map(|(_, cpu, boundary, _, _, _)| {
                [
                    cpu.execution_plan.context_id.as_str(),
                    boundary.execution_plan.context_id.as_str(),
                ]
            })
            .collect();
        // Hybrid context IDs are process-local. Qualify them with the worker
        // PID so a fresh worker may validly allocate context "1", while reuse
        // of the same process/context identity remains a collision.
        let hybrid_process_context_ids: HashSet<(u32, &str)> = collected
            .iter()
            .filter_map(|(_, _, _, hybrid, _, _)| {
                let process_id = hybrid.worker_process.as_ref()?.process_id?;
                Some((process_id, hybrid.execution_plan.context_id.as_str()))
            })
            .collect();
        let unique_execution_context_every_run = exact_run_count
            && parent_context_ids.len() == CORPUS_CASE_COUNT * 2
            && parent_context_ids
                .iter()
                .all(|context_id| !context_id.is_empty())
            && hybrid_process_context_ids.len() == CORPUS_CASE_COUNT
            && hybrid_process_context_ids
                .iter()
                .all(|(_, context_id)| !context_id.is_empty());
        let controlled_background_shutdown_every_run = exact_run_count
            && collected.iter().all(|(_, cpu, boundary, hybrid, _, _)| {
                cpu.background_shutdown.controlled_shutdown_requested
                    && cpu.background_shutdown.all_runtime_resources_released
                    && boundary.background_shutdown.controlled_shutdown_requested
                    && boundary.background_shutdown.all_runtime_resources_released
                    && hybrid.background_shutdown.controlled_shutdown_requested
                    && hybrid.background_shutdown.all_runtime_resources_released
            });
        let hybrid_worker_processes_exact = exact_run_count
            && self
                .orchestrator_executable_sha256
                .as_deref()
                .is_some_and(is_sha256_hex)
            && collected.iter().all(|(_, cpu, boundary, hybrid, _, _)| {
                cpu.worker_process.is_none()
                    && boundary.worker_process.is_none()
                    && hybrid.worker_process.as_ref().is_some_and(|process| {
                        process.is_exact()
                            && Some(process.executable_sha256.as_str())
                                == self.orchestrator_executable_sha256.as_deref()
                            && Some(process.build_git_sha.as_str())
                                == self.provenance.git_sha.as_deref()
                    })
            });
        let worker_ids: HashSet<&str> = collected
            .iter()
            .filter_map(|(_, _, _, hybrid, _, _)| {
                hybrid
                    .worker_process
                    .as_ref()
                    .map(|process| process.worker_id.as_str())
            })
            .collect();
        let unique_hybrid_worker_identity =
            hybrid_worker_processes_exact && worker_ids.len() == CORPUS_CASE_COUNT;
        let process_ids: HashSet<u32> = collected
            .iter()
            .filter_map(|(_, _, _, hybrid, _, _)| {
                hybrid
                    .worker_process
                    .as_ref()
                    .and_then(|process| process.process_id)
            })
            .collect();
        let unique_hybrid_process_identity =
            hybrid_worker_processes_exact && process_ids.len() == CORPUS_CASE_COUNT;
        let cpu_plans_exact = exact_run_count
            && collected.iter().all(|(_, cpu, boundary, _, _, _)| {
                cpu.plane == "cpu"
                    && boundary.plane == "cpu"
                    && cpu_plan_exact(&cpu.execution_plan)
                    && cpu_plan_exact(&boundary.execution_plan)
            });
        let cpu_counter_invariants = exact_run_count
            && collected.iter().all(|(_, cpu, boundary, _, _, _)| {
                cpu_counters_exact(cpu.routed_execution_delta)
                    && cpu_counters_exact(boundary.routed_execution_delta)
            });
        let cpu_gpu_io_zero = exact_run_count
            && collected.iter().all(|(_, cpu, boundary, _, _, _)| {
                cpu.device.is_none()
                    && !cpu.initial_state.gpu_io_available
                    && cpu.initial_state.gpu_io == GpuExpertIoSnapshot::default()
                    && cpu.gpu_io_delta == GpuExpertIoSnapshot::default()
                    && cpu.gpu_memory_before.is_none()
                    && cpu.gpu_memory_after.is_none()
                    && boundary.device.is_none()
                    && !boundary.initial_state.gpu_io_available
                    && boundary.initial_state.gpu_io == GpuExpertIoSnapshot::default()
                    && boundary.gpu_io_delta == GpuExpertIoSnapshot::default()
                    && boundary.gpu_memory_before.is_none()
                    && boundary.gpu_memory_after.is_none()
            });
        let hybrid_plans_exact = exact_run_count
            && collected.iter().all(|(_, _, _, hybrid, _, _)| {
                hybrid.plane == "hybrid" && hybrid_plan_exact(&hybrid.execution_plan)
            });
        let hardware_adapter_exact_match = exact_run_count
            && !self.expected_adapter_name.is_empty()
            && collected.iter().all(|(_, _, _, hybrid, _, _)| {
                hybrid.device.as_ref().is_some_and(|device| {
                    device.name == self.expected_adapter_name
                        && !device.device_type.eq_ignore_ascii_case("cpu")
                })
            });
        let software_adapter_false = exact_run_count
            && collected.iter().all(|(_, _, _, hybrid, _, _)| {
                hybrid
                    .device
                    .as_ref()
                    .is_some_and(|device| !device.software_adapter)
            });
        let hybrid_counter_invariants = exact_run_count
            && collected
                .iter()
                .all(|(_, _, _, hybrid, _, _)| hybrid_counters_exact(hybrid.routed_execution_delta));
        let hybrid_gpu_io_observed = exact_run_count
            && collected.iter().all(|(_, _, _, hybrid, _, _)| {
                hybrid.initial_state.gpu_io_available && gpu_io_observed(hybrid.gpu_io_delta)
            });
        let hybrid_memory_ledger_valid = exact_run_count
            && collected.iter().all(|(_, _, _, hybrid, _, _)| {
                hybrid
                    .gpu_memory_before
                    .is_some_and(|snapshot| crate::qualification::validate_memory(snapshot).is_ok())
                    && hybrid.gpu_memory_after.is_some_and(|snapshot| {
                        crate::qualification::validate_memory(snapshot).is_ok()
                    })
            });
        let zero_gpu_failures_fallbacks_or_degraded = exact_run_count
            && collected.iter().all(|(_, cpu, boundary, hybrid, _, _)| {
                cpu.routed_execution_delta.gpu_dispatch_failures == 0
                    && cpu.routed_execution_delta.gpu_cpu_fallbacks == 0
                    && cpu.routed_execution_delta.degraded_expert_substitutions == 0
                    && cpu.attention_softmax_nonfinite_fallbacks == 0
                    && boundary.routed_execution_delta.gpu_dispatch_failures == 0
                    && boundary.routed_execution_delta.gpu_cpu_fallbacks == 0
                    && boundary.routed_execution_delta.degraded_expert_substitutions == 0
                    && boundary.attention_softmax_nonfinite_fallbacks == 0
                    && hybrid.routed_execution_delta.gpu_dispatch_failures == 0
                    && hybrid.routed_execution_delta.cpu_routed_expert_dispatches == 0
                    && hybrid.routed_execution_delta.gpu_cpu_fallbacks == 0
                    && hybrid.routed_execution_delta.degraded_expert_substitutions == 0
                    && hybrid.attention_softmax_nonfinite_fallbacks == 0
            });

        GreedyParityChecks {
            clean_build: self.provenance.git_sha.as_deref().is_some_and(|sha| {
                sha.len() == 40 && sha.bytes().all(|byte| byte.is_ascii_hexdigit())
            }) && self.provenance.dirty == Some(false),
            strict_real_checkpoint,
            qwen3_coder_model_identity: self
                .model_identity
                .as_ref()
                .is_some_and(ModelIdentityEvidence::is_qwen3_coder_30b_a3b_q4_0),
            canonical_q4_0_layout: self.expert_metadata.as_ref().is_some_and(|metadata| {
                !metadata.explicitly_synthetic
                    && metadata.q4_0_layout.as_deref()
                        == Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
            }),
            fixed_corpus_identity: self.corpus.is_exact() && cases_complete,
            fixed_greedy_sampling: self.sampling.is_exact(),
            resolved_config_identity,
            tokenized_once_and_shared,
            fresh_state_every_run,
            unique_execution_context_every_run,
            controlled_background_shutdown_every_run,
            hybrid_worker_processes_exact,
            unique_hybrid_worker_identity,
            unique_hybrid_process_identity,
            cpu_plans_exact,
            cpu_counter_invariants,
            cpu_gpu_io_zero,
            hybrid_plans_exact,
            hardware_adapter_exact_match,
            software_adapter_false,
            strict_gpu_failure_policy: exact_run_count
                && collected.iter().all(|(_, _, _, hybrid, _, _)| {
                    hybrid.routed_expert_gpu_failure_policy == "strict-fail-closed"
                }),
            hybrid_counter_invariants,
            hybrid_gpu_io_observed,
            hybrid_memory_ledger_valid,
            zero_gpu_failures_fallbacks_or_degraded,
            ordinary_cpu_comparison_recorded: exact_run_count
                && collected.iter().all(|(_, cpu, _, hybrid, comparison, _)| {
                    comparison_matches_generations(
                        &cpu.generation,
                        &hybrid.generation,
                        comparison,
                    )
                }),
            boundary_reference_exact_generated_token_ids: exact_run_count
                && collected.iter().all(|(_, _, boundary, hybrid, _, comparison)| {
                    comparison.exact_token_ids
                        && comparison.first_divergence.is_none()
                        && boundary.generation.generated_token_ids
                            == hybrid.generation.generated_token_ids
                        && comparison_matches_generations(
                            &boundary.generation,
                            &hybrid.generation,
                            comparison,
                        )
                }),
            boundary_reference_identical_generated_token_count: exact_run_count
                && collected.iter().all(|(_, _, boundary, hybrid, _, comparison)| {
                    comparison.equal_generated_count
                        && boundary.generation.generated_token_count
                            == hybrid.generation.generated_token_count
                }),
            boundary_reference_identical_termination_reason: exact_run_count
                && collected.iter().all(|(_, _, boundary, hybrid, _, comparison)| {
                    comparison.equal_termination_reason
                        && boundary.generation.termination_reason
                            == hybrid.generation.termination_reason
                }),
            boundary_reference_identical_generated_text_hash: exact_run_count
                && collected.iter().all(|(_, _, boundary, hybrid, _, comparison)| {
                    comparison.equal_generated_text_hash
                        && boundary.generation.generated_text_sha256
                            == hybrid.generation.generated_text_sha256
                }),
        }
    }

    pub fn finish(&mut self) -> Result<(), QualificationFailure> {
        self.checks = self.derive_checks();
        if self.artifacts.config.is_none()
            || self.artifacts.tokenizer.is_none()
            || self.artifacts.expert_metadata.is_none()
            || self.artifacts.weights_config.is_none()
            || self.model_identity.is_none()
            || self.source_preflight.is_none()
            || self.resolved_config_sha256.is_none()
            || self.orchestrator_executable_sha256.is_none()
        {
            return Err(QualificationFailure::new(
                crate::qualification::FailureStage::Postcondition,
                "greedy-parity-evidence-incomplete",
                "one or more mandatory fixed-corpus parity evidence sections are absent",
            ));
        }
        if !self.checks.passes() {
            return Err(QualificationFailure::new(
                crate::qualification::FailureStage::Postcondition,
                "greedy-parity-check-failed",
                "one or more required fixed-corpus greedy parity checks are false",
            ));
        }
        self.status = QualificationStatus::Pass;
        self.failure = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> crate::qualification::ArtifactDigest {
        crate::qualification::ArtifactDigest {
            configured_path: "artifact".to_string(),
            canonical_path: "/artifact".to_string(),
            byte_length: 1,
            sha256: "00".repeat(32),
        }
    }

    fn model_load() -> ModelLoadEvidence {
        ModelLoadEvidence {
            strict: true,
            loader: "safetensors".to_string(),
            loaded_tensors: 10,
            required_tensors: 10,
            optional_probed: 0,
            optional_loaded: 0,
            seeded_fallback_remained: false,
        }
    }

    fn plan(context_id: String, hybrid: bool) -> ExecutionPlanEvidence {
        ExecutionPlanEvidence {
            context_id,
            requested: if hybrid { "hybrid" } else { "cpu" }.to_string(),
            resolved: if hybrid {
                "hybrid-cpu-attention-gpu-experts"
            } else {
                "cpu"
            }
            .to_string(),
            embeddings: "cpu".to_string(),
            lm_head: "cpu".to_string(),
            dense_projections: "cpu".to_string(),
            attention: "cpu".to_string(),
            kv: "cpu".to_string(),
            router: "cpu".to_string(),
            routed_experts: if hybrid { "gpu" } else { "cpu" }.to_string(),
            routed_expert_dtype: "q4_0".to_string(),
            fallback_occurred: false,
            reason: None,
        }
    }

    fn initial_state(context_id: String, hybrid: bool) -> InitialStateEvidence {
        InitialStateEvidence {
            context_id,
            resolved_config_sha256: "11".repeat(32),
            kv_cache_count: 48,
            kv_sequence_lengths: vec![0; 48],
            all_kv_empty: true,
            cache: RuntimeCacheSnapshot::default(),
            routed: RoutedExpertExecutionSnapshot::default(),
            gpu_io_available: hybrid,
            gpu_io: GpuExpertIoSnapshot::default(),
        }
    }

    fn plane_evidence(case_index: usize, hybrid: bool, ids: &[u32]) -> PlaneRunEvidence {
        let context_id = format!("{}-{case_index}", if hybrid { "hybrid" } else { "cpu" });
        PlaneRunEvidence {
            plane: if hybrid { "hybrid" } else { "cpu" }.to_string(),
            model_load: model_load(),
            execution_plan: plan(context_id.clone(), hybrid),
            routed_expert_gpu_failure_policy: if hybrid {
                "strict-fail-closed"
            } else {
                "serving-cpu-fallback"
            }
            .to_string(),
            device: hybrid.then(|| GpuDeviceIdentity {
                name: "NVIDIA L4".to_string(),
                vendor_id: 0x10de,
                device_id: 1,
                device_type: "DiscreteGpu".to_string(),
                wgpu_backend: "vulkan".to_string(),
                driver: "driver".to_string(),
                driver_info: "info".to_string(),
                compute_plane: "wgpu-vulkan".to_string(),
                software_adapter: false,
            }),
            initial_state: initial_state(context_id, hybrid),
            generation: generation(ids, "same"),
            routed_execution_delta: if hybrid {
                RoutedExpertExecutionSnapshot {
                    selected_routed_experts: 8,
                    gpu_dispatch_attempts: 8,
                    gpu_dispatch_successes: 8,
                    ..Default::default()
                }
            } else {
                RoutedExpertExecutionSnapshot {
                    selected_routed_experts: 8,
                    cpu_routed_expert_dispatches: 8,
                    ..Default::default()
                }
            },
            gpu_io_delta: if hybrid {
                GpuExpertIoSnapshot {
                    hidden_state_uploads: 1,
                    hidden_state_upload_bytes: 4,
                    queue_submissions: 1,
                    map_requests: 1,
                    readback_completions: 1,
                    readback_bytes: 4,
                    ..Default::default()
                }
            } else {
                GpuExpertIoSnapshot::default()
            },
            attention_softmax_nonfinite_fallbacks: 0,
            gpu_memory_before: hybrid.then(GpuExpertMemorySnapshot::default),
            gpu_memory_after: hybrid.then(GpuExpertMemorySnapshot::default),
            background_shutdown: BackgroundShutdownEvidence {
                controlled_shutdown_requested: true,
                all_runtime_resources_released: true,
                poll_iterations: 1,
            },
            worker_process: None,
        }
    }

    fn passing_report() -> GreedyParityReport {
        let artifact = artifact();
        let mut report = GreedyParityReport::new(
            BuildProvenance {
                git_sha: Some("0123456789abcdef0123456789abcdef01234567".to_string()),
                dirty: Some(false),
                package_version: "test".to_string(),
            },
            QualificationArtifacts {
                config: Some(artifact.clone()),
                tokenizer: Some(artifact.clone()),
                expert_metadata: Some(artifact.clone()),
                weights_config: Some(artifact),
                expert_data_directory: "/data".to_string(),
                ..Default::default()
            },
            Some(ExpertMetadataEvidence {
                dtype: Some("q4_0".to_string()),
                q4_0_layout: Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1.to_string()),
                conversion_mode: Some("full".to_string()),
                source: Some("Qwen3-Coder-30B-A3B".to_string()),
                explicitly_synthetic: false,
            }),
            "NVIDIA L4".to_string(),
        );
        report.model_identity = Some(ModelIdentityEvidence {
            architecture: "qwen3_moe".to_string(),
            num_layers: 48,
            num_experts_per_layer: 128,
            total_experts: 6_144,
            top_k: 8,
            d_model: 2_048,
            d_ff: 768,
            routed_expert_dtype: "q4_0".to_string(),
        });
        report.source_preflight = Some(StrictHybridPreflightEvidence {
            real_transformer_enabled: true,
            weights_dir_configured: true,
            strict_weights: true,
            allow_seeded_fallback: false,
            allow_degraded_experts: false,
            allow_attention_fallback: false,
            allow_truncated_expert_payloads: false,
            distributed_enabled: false,
            gpu_cache_enabled: true,
            gpu_expert_capacity_bytes: 1,
            requested_mode: "hybrid".to_string(),
            routed_expert_dtype: "q4_0".to_string(),
        });
        report.resolved_config_sha256 = Some("11".repeat(32));
        report.orchestrator_executable_sha256 = Some("22".repeat(32));
        for (index, fixed) in FIXED_CORPUS.into_iter().enumerate() {
            let prompt_ids = vec![10 + index as u32, 20 + index as u32];
            let ids: Vec<u32> = (0..OUTPUT_TOKEN_LIMIT)
                .map(|offset| 100 + index as u32 + offset as u32)
                .collect();
            let mut case = CaseReport::new(fixed, prompt_ids);
            let prompt_hash = case.prompt_token_ids_sha256.clone();
            let mut cpu = plane_evidence(index, false, &ids);
            let mut boundary = plane_evidence(index, false, &ids);
            let boundary_context_id = format!("boundary-{index}");
            boundary.execution_plan.context_id = boundary_context_id.clone();
            boundary.initial_state.context_id = boundary_context_id;
            let mut hybrid = plane_evidence(index, true, &ids);
            hybrid.worker_process = Some(HybridWorkerProcessEvidence {
                worker_id: hybrid_worker_id(
                    report.provenance.git_sha.as_deref().unwrap(),
                    report
                        .orchestrator_executable_sha256
                        .as_deref()
                        .unwrap(),
                    index,
                    fixed.name,
                ),
                child_process_spawned: true,
                process_id: Some(1000 + index as u32),
                executable_sha256: report
                    .orchestrator_executable_sha256
                    .clone()
                    .unwrap(),
                build_git_sha: report.provenance.git_sha.clone().unwrap(),
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
            });
            cpu.generation.prompt_token_ids_sha256 = prompt_hash.clone();
            boundary.generation.prompt_token_ids_sha256 = prompt_hash.clone();
            hybrid.generation.prompt_token_ids_sha256 = prompt_hash;
            case.ordinary_cpu_vs_hybrid = Some(
                compare_generations(&cpu.generation, &hybrid.generation, |_| {
                    Ok("same".to_string())
                })
                .unwrap(),
            );
            case.boundary_reference_vs_hybrid = Some(
                compare_generations(&boundary.generation, &hybrid.generation, |_| {
                    Ok("same".to_string())
                })
                .unwrap(),
            );
            case.cpu = Some(cpu);
            case.boundary_reference = Some(boundary);
            case.hybrid = Some(hybrid);
            report.cases.push(case);
        }
        report
    }

    fn generation(ids: &[u32], text: &str) -> GenerationEvidence {
        GenerationEvidence {
            prompt_token_ids_sha256: token_ids_sha256(&[7, 8]),
            generated_token_ids: ids.to_vec(),
            generated_token_ids_sha256: token_ids_sha256(ids),
            generated_text_sha256: sha256_hex(text.as_bytes()),
            generated_token_count: ids.len(),
            termination_reason: TerminationReason::LengthLimit,
        }
    }

    #[test]
    fn greedy_parity_fixed_corpus_contract_is_versioned_and_nontrivial() {
        assert_eq!(SCHEMA_VERSION, "mer.strict-hybrid-q4-greedy-parity.v2");
        assert_eq!(LEGACY_SCHEMA_VERSION, "mer.strict-hybrid-q4-greedy-parity.v1");
        assert_ne!(SCHEMA_VERSION, LEGACY_SCHEMA_VERSION);
        assert_eq!(FIXED_CORPUS.len(), 4);
        assert_eq!(OUTPUT_TOKEN_LIMIT, 16);
        assert_eq!(FIXED_CORPUS[0].name, "rust-generation");
        assert_eq!(FIXED_CORPUS[1].name, "rust-debugging");
        assert_eq!(FIXED_CORPUS[2].name, "json-transformation");
        assert_eq!(FIXED_CORPUS[3].name, "multilingual-spanish");
        assert!(FIXED_CORPUS.iter().all(|case| !case.prompt.is_empty()));
        assert!(FIXED_CORPUS[3].prompt.contains('¿'));
        assert_eq!(corpus_sha256().len(), 64);
        assert!(CorpusEvidence::fixed().is_exact());
    }

    #[test]
    fn greedy_parity_sampling_contract_is_exact() {
        assert!(GreedySamplingEvidence::fixed().is_exact());
        let mut drift = GreedySamplingEvidence::fixed();
        drift.seed = 1;
        assert!(!drift.is_exact());
        drift = GreedySamplingEvidence::fixed();
        drift.temperature = f32::NAN;
        assert!(!drift.is_exact());
    }

    #[test]
    fn greedy_parity_exact_match_has_no_divergence() {
        let cpu = generation(&[1, 2, 3], "same");
        let hybrid = generation(&[1, 2, 3], "same");
        let result = compare_generations(&cpu, &hybrid, |ids| Ok(format!("{ids:?}"))).unwrap();
        assert!(result.exact_token_ids);
        assert!(result.equal_generated_count);
        assert!(result.equal_termination_reason);
        assert!(result.equal_generated_text_hash);
        assert_eq!(result.first_divergence, None);
    }

    #[test]
    fn greedy_parity_first_middle_and_final_token_divergences_are_exact() {
        for (position, hybrid_ids) in [(0, vec![9, 2, 3]), (1, vec![1, 9, 3]), (2, vec![1, 2, 9])] {
            let cpu = generation(&[1, 2, 3], "cpu");
            let hybrid = generation(&hybrid_ids, "hybrid");
            let result = compare_generations(&cpu, &hybrid, |ids| Ok(format!("{ids:?}"))).unwrap();
            let divergence = result.first_divergence.unwrap();
            assert_eq!(divergence.position, position);
            assert_eq!(
                divergence.cpu_token_id,
                Some(cpu.generated_token_ids[position])
            );
            assert_eq!(divergence.hybrid_token_id, Some(hybrid_ids[position]));
        }
    }

    #[test]
    fn greedy_parity_length_divergence_preserves_nullable_token_id() {
        let cpu = generation(&[1, 2], "cpu");
        let hybrid = generation(&[1, 2, 3], "hybrid");
        let result = compare_generations(&cpu, &hybrid, |ids| Ok(format!("{ids:?}"))).unwrap();
        let divergence = result.first_divergence.unwrap();
        assert_eq!(divergence.position, 2);
        assert_eq!(divergence.cpu_token_id, None);
        assert_eq!(divergence.hybrid_token_id, Some(3));
        assert!(!result.equal_generated_count);
    }

    #[test]
    fn greedy_parity_count_stop_and_text_hash_divergences_are_independent() {
        let cpu = generation(&[1, 2], "cpu");

        let mut count = cpu.clone();
        count.generated_token_count = 3;
        let result = compare_generations(&cpu, &count, |_| Ok("same-prefix".to_string())).unwrap();
        assert!(result.exact_token_ids);
        assert!(!result.equal_generated_count);
        assert!(result.equal_termination_reason);
        assert!(result.equal_generated_text_hash);
        assert!(result.first_divergence.is_some());

        let mut stop = cpu.clone();
        stop.termination_reason = TerminationReason::EndOfSequence;
        let result = compare_generations(&cpu, &stop, |_| Ok("same-prefix".to_string())).unwrap();
        assert!(result.exact_token_ids);
        assert!(result.equal_generated_count);
        assert!(!result.equal_termination_reason);
        assert!(result.equal_generated_text_hash);
        assert!(result.first_divergence.is_some());

        let mut text_hash = cpu.clone();
        text_hash.generated_text_sha256 = sha256_hex(b"hybrid");
        let result =
            compare_generations(&cpu, &text_hash, |_| Ok("same-prefix".to_string())).unwrap();
        assert!(result.exact_token_ids);
        assert!(result.equal_generated_count);
        assert!(result.equal_termination_reason);
        assert!(!result.equal_generated_text_hash);
        assert!(result.first_divergence.is_some());
    }

    #[test]
    fn greedy_parity_decoded_failure_prefixes_are_utf8_bounded() {
        let cpu = generation(&[1], "cpu");
        let hybrid = generation(&[2], "hybrid");
        let value = "é".repeat(400);
        let result = compare_generations(&cpu, &hybrid, |_| Ok(value.clone())).unwrap();
        let divergence = result.first_divergence.unwrap();
        assert!(divergence.cpu_prefix_truncated);
        assert!(divergence.hybrid_prefix_truncated);
        assert!(divergence.cpu_decoded_prefix.len() <= MAX_DECODED_PREFIX_BYTES);
        assert!(divergence
            .cpu_decoded_prefix
            .is_char_boundary(divergence.cpu_decoded_prefix.len()));
    }

    #[test]
    fn greedy_parity_cpu_and_hybrid_counter_contracts_reject_false_confidence() {
        let cpu = RoutedExpertExecutionSnapshot {
            selected_routed_experts: 8,
            cpu_routed_expert_dispatches: 8,
            ..Default::default()
        };
        assert!(cpu_counters_exact(cpu));
        assert!(!cpu_counters_exact(RoutedExpertExecutionSnapshot {
            gpu_dispatch_attempts: 1,
            ..cpu
        }));

        let hybrid = RoutedExpertExecutionSnapshot {
            selected_routed_experts: 8,
            gpu_dispatch_attempts: 8,
            gpu_dispatch_successes: 8,
            ..Default::default()
        };
        assert!(hybrid_counters_exact(hybrid));
        assert!(!hybrid_counters_exact(RoutedExpertExecutionSnapshot {
            gpu_dispatch_failures: 1,
            ..hybrid
        }));
        assert!(!hybrid_counters_exact(RoutedExpertExecutionSnapshot {
            cpu_routed_expert_dispatches: 1,
            ..hybrid
        }));
    }

    #[test]
    fn greedy_parity_gpu_io_requires_every_observed_operation() {
        let complete = GpuExpertIoSnapshot {
            hidden_state_uploads: 1,
            hidden_state_upload_bytes: 4,
            queue_submissions: 1,
            map_requests: 1,
            readback_completions: 1,
            readback_bytes: 4,
            ..Default::default()
        };
        assert!(gpu_io_observed(complete));
        for missing in 0..6 {
            let mut candidate = complete;
            match missing {
                0 => candidate.hidden_state_uploads = 0,
                1 => candidate.hidden_state_upload_bytes = 0,
                2 => candidate.queue_submissions = 0,
                3 => candidate.map_requests = 0,
                4 => candidate.readback_completions = 0,
                _ => candidate.readback_bytes = 0,
            }
            assert!(!gpu_io_observed(candidate));
        }
    }

    #[test]
    fn greedy_parity_cpu_and_hybrid_plan_validators_are_plane_exact() {
        let cpu = ExecutionPlanEvidence {
            context_id: "ctx-cpu".to_string(),
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
        };
        assert!(cpu_plan_exact(&cpu));
        let mut cpu_with_reason = cpu.clone();
        cpu_with_reason.reason = Some("CPU control selected for qualification".to_string());
        assert!(cpu_plan_exact(&cpu_with_reason));

        let mut hybrid = cpu.clone();
        hybrid.context_id = "ctx-hybrid".to_string();
        hybrid.requested = "hybrid".to_string();
        hybrid.resolved = "hybrid-cpu-attention-gpu-experts".to_string();
        hybrid.routed_experts = "gpu".to_string();
        hybrid.reason = Some(
            "hybrid: attention pinned to the checked CPU path; routed-expert FFN compute offloaded to the GPU"
                .to_string(),
        );
        assert!(hybrid_plan_exact(&hybrid));

        let mut hybrid_without_reason = hybrid.clone();
        hybrid_without_reason.reason = None;
        assert!(hybrid_plan_exact(&hybrid_without_reason));

        let mut bad_attention = hybrid.clone();
        bad_attention.attention = "gpu".to_string();
        assert!(!hybrid_plan_exact(&bad_attention));

        let mut bad_experts = hybrid.clone();
        bad_experts.routed_experts = "cpu".to_string();
        assert!(!hybrid_plan_exact(&bad_experts));

        let mut bad_dtype = hybrid.clone();
        bad_dtype.routed_expert_dtype = "f16".to_string();
        assert!(!hybrid_plan_exact(&bad_dtype));

        let mut bad_resolution = hybrid.clone();
        bad_resolution.resolved = "gpu".to_string();
        assert!(!hybrid_plan_exact(&bad_resolution));

        let mut bad_fallback = hybrid;
        bad_fallback.fallback_occurred = true;
        assert!(!hybrid_plan_exact(&bad_fallback));
    }

    #[test]
    fn greedy_parity_clean_initial_state_rejects_any_residue() {
        let clean = InitialStateEvidence {
            context_id: "ctx".to_string(),
            resolved_config_sha256: "hash".to_string(),
            kv_cache_count: 2,
            kv_sequence_lengths: vec![0, 0],
            all_kv_empty: true,
            cache: RuntimeCacheSnapshot::default(),
            routed: RoutedExpertExecutionSnapshot::default(),
            gpu_io_available: false,
            gpu_io: GpuExpertIoSnapshot::default(),
        };
        assert!(clean.is_clean());
        let mut dirty = clean.clone();
        dirty.cache.logical_lru_entries = 1;
        assert!(!dirty.is_clean());
        let mut dirty = clean.clone();
        dirty.kv_sequence_lengths[0] = 1;
        assert!(!dirty.is_clean());
    }

    #[test]
    fn greedy_parity_missing_evidence_and_nullable_provenance_never_pass() {
        let mut report = GreedyParityReport::new(
            BuildProvenance {
                git_sha: Some("a".repeat(40)),
                dirty: None,
                package_version: "0.1.0".to_string(),
            },
            QualificationArtifacts::default(),
            None,
            "NVIDIA L4".to_string(),
        );
        let failure = report.finish().unwrap_err();
        assert_eq!(failure.code, "greedy-parity-evidence-incomplete");
        assert_eq!(report.status, QualificationStatus::Fail);
        assert!(!report.checks.clean_build);
        assert!(!report.checks.hardware_adapter_exact_match);
        assert!(!report.checks.software_adapter_false);
    }

    #[test]
    fn greedy_parity_failure_state_is_explicit_and_serializable() {
        let mut report = GreedyParityReport::new(
            BuildProvenance {
                git_sha: Some("a".repeat(40)),
                dirty: Some(false),
                package_version: "0.1.0".to_string(),
            },
            QualificationArtifacts::default(),
            None,
            "NVIDIA L4".to_string(),
        );
        let failure = QualificationFailure::new(
            crate::qualification::FailureStage::Inference,
            "token-divergence",
            "case rust-generation diverged at position 0",
        );
        report.fail(failure.clone());
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["status"], "fail");
        assert_eq!(json["failure"]["code"], "token-divergence");
        assert_eq!(report.failure, Some(failure));
    }

    #[test]
    fn greedy_parity_complete_observed_contract_passes() {
        let mut report = passing_report();
        report.finish().unwrap();
        assert_eq!(report.status, QualificationStatus::Pass);
        assert!(report.checks.passes());
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["schema_version"], SCHEMA_VERSION);
        assert_ne!(json["schema_version"], LEGACY_SCHEMA_VERSION);
        assert_eq!(
            json["cases"][0]["hybrid"]["device"]["software_adapter"],
            false
        );
    }

    #[test]
    fn greedy_parity_ordinary_cpu_divergence_is_informational() {
        let mut report = passing_report();
        let case = &mut report.cases[0];
        let cpu = case.cpu.as_mut().unwrap();
        cpu.generation.generated_token_ids[0] ^= 1;
        cpu.generation.generated_token_ids_sha256 =
            token_ids_sha256(&cpu.generation.generated_token_ids);
        cpu.generation.generated_text_sha256 = sha256_hex(b"ordinary-cpu-diverged");
        case.ordinary_cpu_vs_hybrid = Some(
            compare_generations(
                &cpu.generation,
                &case.hybrid.as_ref().unwrap().generation,
                |ids| Ok(format!("{ids:?}")),
            )
            .unwrap(),
        );

        report.finish().unwrap();
        assert_eq!(report.status, QualificationStatus::Pass);
        assert!(report.checks.ordinary_cpu_comparison_recorded);
        assert!(!report.cases[0]
            .ordinary_cpu_vs_hybrid
            .as_ref()
            .unwrap()
            .exact_token_ids);
        assert!(report.cases[0]
            .boundary_reference_vs_hybrid
            .as_ref()
            .unwrap()
            .exact_token_ids);
    }

    #[test]
    fn greedy_parity_boundary_reference_divergence_fails() {
        let mut report = passing_report();
        let case = &mut report.cases[1];
        let boundary = case.boundary_reference.as_mut().unwrap();
        boundary.generation.generated_token_ids[3] ^= 1;
        boundary.generation.generated_token_ids_sha256 =
            token_ids_sha256(&boundary.generation.generated_token_ids);
        boundary.generation.generated_text_sha256 = sha256_hex(b"boundary-diverged");
        case.boundary_reference_vs_hybrid = Some(
            compare_generations(
                &boundary.generation,
                &case.hybrid.as_ref().unwrap().generation,
                |ids| Ok(format!("{ids:?}")),
            )
            .unwrap(),
        );

        assert!(report.finish().is_err());
        assert!(!report
            .checks
            .boundary_reference_exact_generated_token_ids);
    }

    #[test]
    fn greedy_parity_partial_boundary_or_process_evidence_fails() {
        let mut missing_boundary = passing_report();
        missing_boundary.cases[0].boundary_reference = None;
        assert!(missing_boundary.finish().is_err());

        let mut partial_corpus = passing_report();
        partial_corpus.cases.pop();
        assert!(partial_corpus.finish().is_err());

        let mut reused_process = passing_report();
        let process_id = reused_process.cases[0]
            .hybrid
            .as_ref()
            .unwrap()
            .worker_process
            .as_ref()
            .unwrap()
            .process_id;
        reused_process.cases[1]
            .hybrid
            .as_mut()
            .unwrap()
            .worker_process
            .as_mut()
            .unwrap()
            .process_id = process_id;
        assert!(reused_process.finish().is_err());
        assert!(!reused_process.checks.unique_hybrid_process_identity);
    }

    #[test]
    fn greedy_parity_cannot_pass_with_configuration_or_context_drift() {
        let mut preflight_drift = passing_report();
        preflight_drift
            .source_preflight
            .as_mut()
            .unwrap()
            .allow_degraded_experts = true;
        assert!(preflight_drift.finish().is_err());
        assert!(!preflight_drift.checks.strict_real_checkpoint);

        let mut config_drift = passing_report();
        config_drift.cases[2]
            .hybrid
            .as_mut()
            .unwrap()
            .initial_state
            .resolved_config_sha256 = "drift".to_string();
        assert!(config_drift.finish().is_err());
        assert!(!config_drift.checks.resolved_config_identity);

        let mut context_reuse = passing_report();
        let reused = context_reuse.cases[0]
            .cpu
            .as_ref()
            .unwrap()
            .execution_plan
            .context_id
            .clone();
        context_reuse.cases[1]
            .cpu
            .as_mut()
            .unwrap()
            .execution_plan
            .context_id = reused.clone();
        context_reuse.cases[1]
            .cpu
            .as_mut()
            .unwrap()
            .initial_state
            .context_id = reused;
        assert!(context_reuse.finish().is_err());
        assert!(!context_reuse.checks.unique_execution_context_every_run);
    }

    #[test]
    fn greedy_parity_worker_context_identity_is_process_aware_and_fail_closed() {
        let mut process_local_ids = passing_report();
        for case in &mut process_local_ids.cases {
            let hybrid = case.hybrid.as_mut().unwrap();
            hybrid.execution_plan.context_id = "1".to_string();
            hybrid.initial_state.context_id = "1".to_string();
        }
        assert!(process_local_ids.finish().is_ok());
        assert!(process_local_ids.checks.unique_execution_context_every_run);
        assert!(process_local_ids.checks.unique_hybrid_process_identity);

        let mut reused_process_context = passing_report();
        for case in &mut reused_process_context.cases {
            let hybrid = case.hybrid.as_mut().unwrap();
            hybrid.execution_plan.context_id = "1".to_string();
            hybrid.initial_state.context_id = "1".to_string();
        }
        let reused_process_id = reused_process_context.cases[0]
            .hybrid
            .as_ref()
            .unwrap()
            .worker_process
            .as_ref()
            .unwrap()
            .process_id;
        reused_process_context.cases[1]
            .hybrid
            .as_mut()
            .unwrap()
            .worker_process
            .as_mut()
            .unwrap()
            .process_id = reused_process_id;
        assert!(reused_process_context.finish().is_err());
        assert!(!reused_process_context
            .checks
            .unique_execution_context_every_run);
        assert!(!reused_process_context
            .checks
            .unique_hybrid_process_identity);

        let mut missing_process = passing_report();
        missing_process.cases[2]
            .hybrid
            .as_mut()
            .unwrap()
            .worker_process
            .as_mut()
            .unwrap()
            .process_id = None;
        assert!(missing_process.finish().is_err());
        assert!(!missing_process
            .checks
            .unique_execution_context_every_run);
        assert!(!missing_process.checks.hybrid_worker_processes_exact);

        let mut missing_context = passing_report();
        let hybrid = missing_context.cases[3].hybrid.as_mut().unwrap();
        hybrid.execution_plan.context_id.clear();
        hybrid.initial_state.context_id.clear();
        assert!(missing_context.finish().is_err());
        assert!(!missing_context
            .checks
            .unique_execution_context_every_run);
    }

    #[test]
    fn greedy_parity_cannot_pass_with_dirty_state_or_background_leak() {
        let mut dirty = passing_report();
        dirty.cases[0]
            .cpu
            .as_mut()
            .unwrap()
            .initial_state
            .cache
            .ram_entries = 1;
        assert!(dirty.finish().is_err());
        assert!(!dirty.checks.fresh_state_every_run);

        let mut leaked = passing_report();
        leaked.cases[3]
            .hybrid
            .as_mut()
            .unwrap()
            .background_shutdown
            .all_runtime_resources_released = false;
        assert!(leaked.finish().is_err());
        assert!(!leaked.checks.controlled_background_shutdown_every_run);
    }

    #[test]
    fn greedy_parity_cannot_pass_with_counter_or_gpu_io_gaps() {
        let mut cpu = passing_report();
        cpu.cases[0]
            .cpu
            .as_mut()
            .unwrap()
            .routed_execution_delta
            .cpu_routed_expert_dispatches = 7;
        assert!(cpu.finish().is_err());
        assert!(!cpu.checks.cpu_counter_invariants);

        let mut hybrid = passing_report();
        hybrid.cases[1]
            .hybrid
            .as_mut()
            .unwrap()
            .gpu_io_delta
            .map_requests = 0;
        assert!(hybrid.finish().is_err());
        assert!(!hybrid.checks.hybrid_gpu_io_observed);

        let mut attention = passing_report();
        attention.cases[1]
            .cpu
            .as_mut()
            .unwrap()
            .attention_softmax_nonfinite_fallbacks = 1;
        assert!(attention.finish().is_err());
        assert!(!attention.checks.zero_gpu_failures_fallbacks_or_degraded);

        let mut invalid_memory = passing_report();
        invalid_memory.cases[2]
            .hybrid
            .as_mut()
            .unwrap()
            .gpu_memory_after
            .as_mut()
            .unwrap()
            .total_tracked_bytes = 1;
        assert!(invalid_memory.finish().is_err());
        assert!(!invalid_memory.checks.hybrid_memory_ledger_valid);
    }

    #[test]
    fn greedy_parity_cannot_pass_with_nullable_or_software_device_evidence() {
        let mut missing = passing_report();
        missing.cases[0].hybrid.as_mut().unwrap().device = None;
        assert!(missing.finish().is_err());
        assert!(!missing.checks.software_adapter_false);

        let mut software = passing_report();
        software.cases[0]
            .hybrid
            .as_mut()
            .unwrap()
            .device
            .as_mut()
            .unwrap()
            .software_adapter = true;
        assert!(software.finish().is_err());
        assert!(!software.checks.software_adapter_false);
    }

    #[test]
    fn greedy_parity_rederives_token_checks_instead_of_trusting_flags() {
        let mut report = passing_report();
        let hybrid = report.cases[0].hybrid.as_mut().unwrap();
        hybrid.generation.generated_token_ids[4] ^= 1;
        // Leave the stored comparison flags untouched to simulate copied or
        // stale intended evidence. PASS must still be impossible.
        assert!(report.finish().is_err());
        assert!(!report
            .checks
            .boundary_reference_exact_generated_token_ids);

        let mut bad_hash = passing_report();
        bad_hash.cases[0]
            .cpu
            .as_mut()
            .unwrap()
            .generation
            .generated_token_ids_sha256 = "copied".to_string();
        assert!(bad_hash.finish().is_err());
        assert!(!bad_hash.checks.tokenized_once_and_shared);
    }

    fn worker_request() -> HybridWorkerRequest {
        HybridWorkerRequest::new(
            hybrid_worker_id(&"a".repeat(40), &"b".repeat(64), 0, FIXED_CORPUS[0].name),
            FIXED_CORPUS[0],
            "c".repeat(64),
            "NVIDIA L4".to_string(),
            vec![17, 29, 41, 53],
            "b".repeat(64),
            "a".repeat(40),
        )
    }

    fn verified_worker_identity() -> HybridWorkerIdentityEvidence {
        HybridWorkerIdentityEvidence {
            protocol_version_verified: true,
            case_identity_verified: true,
            config_identity_verified: true,
            expected_adapter_verified: true,
            prompt_token_identity_verified: true,
            output_token_limit_verified: true,
            greedy_sampling_identity_verified: true,
            executable_identity_verified: true,
            build_sha_identity_verified: true,
        }
    }

    fn worker_response(request: &HybridWorkerRequest) -> HybridWorkerResponse {
        HybridWorkerResponse::from_request(
            request,
            &request.resolved_config_sha256,
            &request.executable_sha256,
            Some(&request.build_git_sha),
            verified_worker_identity(),
            Some(plane_evidence(0, true, &[1, 2, 3])),
            None,
        )
    }

    #[test]
    fn greedy_parity_worker_transport_preserves_exact_token_ids_without_prompt_text() {
        let request = worker_request();
        let json = serde_json::to_vec(&request).unwrap();
        assert!(!json
            .windows(FIXED_CORPUS[0].prompt.len())
            .any(|window| window == FIXED_CORPUS[0].prompt.as_bytes()));
        let parsed = parse_hybrid_worker_request_exact(&json).unwrap();
        assert_eq!(parsed, request);
        assert_eq!(parsed.prompt_token_ids, vec![17, 29, 41, 53]);
        assert_eq!(
            parsed.prompt_token_ids_sha256,
            token_ids_sha256(&[17, 29, 41, 53])
        );
    }

    #[test]
    fn greedy_parity_worker_request_validates_every_frozen_identity() {
        let request = worker_request();
        let valid = validate_hybrid_worker_request(
            &request,
            &request.resolved_config_sha256,
            &request.executable_sha256,
            Some(&request.build_git_sha),
        );
        assert!(valid.all_verified());

        let mut drift = request.clone();
        drift.prompt_token_ids[0] ^= 1;
        assert!(!validate_hybrid_worker_request(
            &drift,
            &drift.resolved_config_sha256,
            &drift.executable_sha256,
            Some(&drift.build_git_sha),
        )
        .prompt_token_identity_verified);
        let mut drift = request.clone();
        drift.prompt_sha256 = "d".repeat(64);
        assert!(!validate_hybrid_worker_request(
            &drift,
            &drift.resolved_config_sha256,
            &drift.executable_sha256,
            Some(&drift.build_git_sha),
        )
        .case_identity_verified);
        assert!(!validate_hybrid_worker_request(
            &request,
            &"d".repeat(64),
            &request.executable_sha256,
            Some(&request.build_git_sha),
        )
        .config_identity_verified);
    }

    #[test]
    fn greedy_parity_worker_response_requires_exact_parent_identities() {
        let request = worker_request();
        let response = worker_response(&request);
        assert!(validate_hybrid_worker_response(&request, &response).all_verified());

        let mut wrong_case = worker_response(&request);
        wrong_case.case_name = FIXED_CORPUS[1].name.to_string();
        assert!(!validate_hybrid_worker_response(&request, &wrong_case)
            .case_identity_verified);
        let mut wrong_config = worker_response(&request);
        wrong_config.resolved_config_sha256 = "d".repeat(64);
        assert!(!validate_hybrid_worker_response(&request, &wrong_config)
            .config_identity_verified);
        let mut wrong_token = worker_response(&request);
        wrong_token.prompt_token_ids_sha256 = "e".repeat(64);
        assert!(!validate_hybrid_worker_response(&request, &wrong_token)
            .prompt_token_identity_verified);
    }

    #[test]
    fn greedy_parity_worker_output_rejects_missing_malformed_duplicate_and_trailing_data() {
        let request = worker_request();
        let valid = serde_json::to_vec(&worker_response(&request)).unwrap();
        assert!(parse_hybrid_worker_response_exact(&valid).is_ok());
        assert!(parse_hybrid_worker_response_exact(b"").is_err());
        assert!(parse_hybrid_worker_response_exact(b"{broken").is_err());

        let mut duplicated = valid.clone();
        duplicated.extend_from_slice(&valid);
        assert!(parse_hybrid_worker_response_exact(&duplicated).is_err());
        let mut trailing = valid;
        trailing.extend_from_slice(b"unexpected");
        assert!(parse_hybrid_worker_response_exact(&trailing).is_err());
    }

    #[test]
    fn greedy_parity_process_evidence_is_fail_closed_for_absence_or_false_fields() {
        let mut missing = passing_report();
        missing.cases[0]
            .hybrid
            .as_mut()
            .unwrap()
            .worker_process = None;
        assert!(missing.finish().is_err());
        assert!(!missing.checks.hybrid_worker_processes_exact);

        let original = passing_report().cases[0]
            .hybrid
            .as_ref()
            .unwrap()
            .worker_process
            .clone()
            .unwrap();
        let mutations: [fn(&mut HybridWorkerProcessEvidence); 14] = [
            |value| value.child_process_spawned = false,
            |value| value.process_id = None,
            |value| value.executable_identity_verified = false,
            |value| value.build_sha_identity_verified = false,
            |value| value.case_identity_verified = false,
            |value| value.config_identity_verified = false,
            |value| value.expected_adapter_identity_verified = false,
            |value| value.prompt_token_identity_verified = false,
            |value| value.output_token_limit_verified = false,
            |value| value.greedy_sampling_identity_verified = false,
            |value| value.normal_zero_exit = false,
            |value| value.process_reaped = false,
            |value| value.timed_out = true,
            |value| value.evidence_emitted = false,
        ];
        for mutate in mutations {
            let mut evidence = original.clone();
            mutate(&mut evidence);
            assert!(!evidence.is_exact());
        }
    }

    #[test]
    fn greedy_parity_requires_unique_worker_identity_for_all_four_cases() {
        let mut report = passing_report();
        report.finish().unwrap();
        assert!(report.checks.unique_hybrid_worker_identity);

        let reused = report.cases[0]
            .hybrid
            .as_ref()
            .unwrap()
            .worker_process
            .as_ref()
            .unwrap()
            .worker_id
            .clone();
        report.cases[1]
            .hybrid
            .as_mut()
            .unwrap()
            .worker_process
            .as_mut()
            .unwrap()
            .worker_id = reused;
        assert!(report.finish().is_err());
        assert!(!report.checks.unique_hybrid_worker_identity);
    }

    #[test]
    fn greedy_parity_failure_report_preserves_completed_cases_and_worker_diagnostics() {
        let mut report = passing_report();
        report.cases.truncate(2);
        let mut failed = CaseReport::new(FIXED_CORPUS[2], vec![1, 2]);
        failed.worker_failure = Some(HybridWorkerFailureEvidence {
            worker_id: "worker-3".to_string(),
            case_name: FIXED_CORPUS[2].name.to_string(),
            child_process_spawned: true,
            process_id: Some(123),
            exit_code: None,
            signal: Some(15),
            timed_out: false,
            process_reaped: true,
            evidence_emitted: false,
            identity_validation_succeeded: false,
            identity_validation: None,
            stderr: "bounded diagnostic".to_string(),
            stderr_truncated: false,
        });
        failed.failure = Some(QualificationFailure::new(
            crate::qualification::FailureStage::Inference,
            "hybrid-worker-failed",
            "worker terminated",
        ));
        report.cases.push(failed);
        report.fail(QualificationFailure::new(
            crate::qualification::FailureStage::Inference,
            "hybrid-worker-failed",
            "worker terminated",
        ));
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["cases"].as_array().unwrap().len(), 3);
        assert!(json["cases"][0]["boundary_reference_vs_hybrid"].is_object());
        assert!(json["cases"][1]["ordinary_cpu_vs_hybrid"].is_object());
        assert_eq!(json["cases"][2]["worker_failure"]["signal"], 15);
    }
}
